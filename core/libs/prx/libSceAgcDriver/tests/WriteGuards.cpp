#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
namespace {

using namespace AgcDriver::GuestMemory;

constexpr std::size_t Page = 65536;
constexpr std::size_t Granule = 16384;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::mutex reportMutex;
std::set<std::uint64_t> resident;
std::vector<std::uint64_t> written;
std::vector<std::pair<std::uint64_t, std::size_t>> dropped;
std::vector<std::string> order;

bool anyResident(std::uint64_t address, std::size_t bytes) {
    for (auto page = address >> ResidencyPageShift; page <= (address + bytes - 1) >> ResidencyPageShift; ++page) {
        if (resident.contains(page)) return true;
    }
    return false;
}

void cpuWritten(std::span<const std::uint64_t> pages) {
    std::lock_guard lock(reportMutex);
    written.insert(written.end(), pages.begin(), pages.end());
}

void cpuWrittenTakingTheTracker(std::span<const std::uint64_t> pages) {
    static_cast<void>(TrackerGeneration());
    cpuWritten(pages);
}

void dropRange(std::uint64_t address, std::size_t bytes) {
    dropped.emplace_back(address, bytes);
}

void writeBack(std::uint64_t, std::size_t) {
    order.push_back("write-back");
}

void flushHook(std::uint64_t, std::size_t) {
    order.push_back("flush");
}

struct RefusalClear {
    std::uint64_t address = 0;
    std::size_t bytes = 0;
    std::uint32_t cleared = 0;
};
std::vector<RefusalClear> refusalClears;

void refusalCleared(std::uint64_t address, std::size_t bytes, std::uint32_t cleared) {
    std::lock_guard lock(reportMutex);
    refusalClears.push_back({address, bytes, cleared});
}

bool clearedFor(std::uint64_t address, GuestArena::GuardRefusal refusal) {
    std::lock_guard lock(reportMutex);
    return std::any_of(refusalClears.begin(), refusalClears.end(), [&](const RefusalClear& clear) { return clear.address <= address && address < clear.address + clear.bytes && (clear.cleared & GuestArena::RefusalBit(refusal)) != 0; });
}

void clearRefusalClears() {
    std::lock_guard lock(reportMutex);
    refusalClears.clear();
}

const ResidencyHooks hooks{&anyResident, &cpuWritten, &dropRange, &writeBack, nullptr, &refusalCleared};
const ResidencyHooks lockCheckingHooks{&anyResident, &cpuWrittenTakingTheTracker, &dropRange, &writeBack};

std::size_t reported(std::uint64_t page) {
    std::lock_guard lock(reportMutex);
    return static_cast<std::size_t>(std::count(written.begin(), written.end(), page));
}

void clearReports() {
    std::lock_guard lock(reportMutex);
    written.clear();
}

DWORD protectionAt(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    Require(VirtualQuery(address, &info, sizeof(info)) == sizeof(info), "cannot query a guarded page");
    return info.Protect;
}

std::uint8_t* commitPrivate(std::size_t bytes, std::size_t granule) {
    auto* memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, Page));
    GuestArena::GuestArenaCommit_nid_postfix(memory, bytes, PAGE_READWRITE, granule);
    return memory;
}

std::uint64_t pageOf(const void* address) {
    return reinterpret_cast<std::uint64_t>(address) >> ResidencyPageShift;
}

std::uint8_t* mapShared(std::size_t bytes, HANDLE section = nullptr) {
    if (section == nullptr) section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
    Require(section != nullptr, "cannot create the shared backing");
    auto* memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, Page));
    GuestArena::GuestArenaMap_nid_postfix(memory, bytes, section, 0, PAGE_READWRITE);
    return memory;
}

void CheckPrivateGuards() {
    auto* fresh = commitPrivate(Page, Granule);
    clearReports();
    Require(GuardWrites(reinterpret_cast<std::uint64_t>(fresh), Page) == GuestArena::GuardRefusal::None, "an unimported private page could not be guarded");
    Require(WriteGuarded(pageOf(fresh)) && protectionAt(fresh) == PAGE_READONLY && protectionAt(fresh + 3 * Granule) == PAGE_READONLY, "the guard did not make the whole private page read-only");
    fresh[100] = 1;
    Require(reported(pageOf(fresh)) == 1 && fresh[100] == 1, "the first store to a guarded private page was not reported");
    Require(!WriteGuarded(pageOf(fresh)) && protectionAt(fresh) == PAGE_READWRITE, "the fault did not restore the private page's protection");

    auto* memory = commitPrivate(Page, Granule);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    NoteHostImport(base, Page);
    const auto stamp = HostImportStamp();
    Require(!HostImportSettled(base, Page), "a fresh host import was reported settled");
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::UnsettledImport, "a private page was guarded before its host import settled, or refused for another reason: the GPU would lose the device on its read-only pages");
    Require(GuardAccess(base, Granule) == GuestArena::GuardRefusal::UnsettledImport, "a private page was access guarded before its host import settled, or refused for another reason");
    Require(!WriteGuarded(pageOf(memory)) && !AccessGuarded(pageOf(memory)) && protectionAt(memory) == PAGE_READWRITE, "a refused guard changed the page");
    SettleHostImports(base, Page, stamp - 1);
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::UnsettledImport, "a settle mark older than the import settled it");
    clearRefusalClears();
    SettleHostImports(base, Page, stamp);
    Require(HostImportSettled(base, Page), "the settle mark did not settle the import");
    Require(clearedFor(base, GuestArena::GuardRefusal::UnsettledImport), "settling the import did not report that an unsettled-import refusal can clear");
    clearReports();
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::None, "a settled private page could not be guarded");
    Require(WriteGuarded(pageOf(memory)) && protectionAt(memory) == PAGE_READONLY, "the guard did not make the settled private page read-only");
    memory[200] = 2;
    Require(reported(pageOf(memory)) == 1 && memory[200] == 2, "the first store to a guarded settled page was not reported");
    Require(!WriteGuarded(pageOf(memory)) && protectionAt(memory) == PAGE_READWRITE, "the fault did not restore the settled page's protection");
    NoteHostImport(base, Page);
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::UnsettledImport, "a new host import of a settled page did not need its own settle");
    SettleHostImports(base, Page, HostImportStamp());
    GuestArena::GuestArenaPinWritable_nid_postfix(memory, Page);
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::PrivatePinned, "a pinned private page was guarded, or refused for another reason");
    clearRefusalClears();
    GuestArena::GuestArenaUnpinWritable_nid_postfix(memory, Page);
    Require(clearedFor(base, GuestArena::GuardRefusal::PrivatePinned) && !clearedFor(base, GuestArena::GuardRefusal::UnsettledImport), "the pin release did not report that exactly a pin refusal can clear");
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::None, "the unpinned private page could not be guarded");
    std::cout << "private pages guarded only after their import settled: ok\n";
}

void CheckSharedGuard() {
    auto* memory = mapShared(2 * Page);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    clearReports();
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::None, "a mapped shared page could not be guarded");
    Require(WriteGuarded(pageOf(memory)) && protectionAt(memory) == PAGE_READONLY && protectionAt(memory + 3 * Granule) == PAGE_READONLY, "the guard did not make the whole page read-only");
    Require(protectionAt(memory + Page) == PAGE_READWRITE, "the guard reached the next page");
    const auto before = CollectWritesUncached(base, 2 * Page);
    memory[100] = 1;
    Require(reported(pageOf(memory)) == 1, "the first store to a guarded page was not reported");
    Require(!WriteGuarded(pageOf(memory)) && protectionAt(memory) == PAGE_READWRITE, "the fault did not restore the guest's protection");
    memory[200] = 2;
    memory[2 * Granule] = 2;
    Require(reported(pageOf(memory)) == 1, "a second store in the same epoch was reported again");
    memory[Page + 8] = 3;
    Require(reported(pageOf(memory + Page)) == 0, "a store to an unguarded page was reported");
    CollectWritesUncached(base, 2 * Page);
    Require(!UnchangedSince(base, Page, before), "the write tracking missed the store a guard let through");
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::None, "the page could not be guarded again");
    memory[300] = 4;
    Require(reported(pageOf(memory)) == 2 && memory[300] == 4 && memory[100] == 1, "the re-armed guard did not report the next store");
    std::cout << "shared write guard: ok\n";
}

void CheckGuestProtectionPreserved() {
    auto* memory = mapShared(Page);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    clearReports();
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::None, "the page could not be guarded");
    GuestArena::GuestArenaSetProtection_nid_postfix(base + Granule, Granule, PAGE_READONLY);
    DWORD previous = 0;
    Require(VirtualProtect(memory + Granule, Granule, PAGE_READONLY, &previous) != 0, "the guest's protection could not be applied");
    Require(reported(pageOf(memory)) == 1, "a guest protection change over a guard was not reported as a possible write");
    Require(!WriteGuarded(pageOf(memory)), "the guard outlived the guest's protection change");
    memory[0] = 5;
    memory[2 * Granule] = 6;
    Require(protectionAt(memory + Granule) == PAGE_READONLY, "a store elsewhere in the page undid the guest's read-only protection");
    Require(protectionAt(memory) == PAGE_READWRITE && protectionAt(memory + 2 * Granule) == PAGE_READWRITE, "the stores did not get the guest's read-write protection back");
    Require(reported(pageOf(memory)) == 1, "a store after the guard was lifted was reported");
    std::cout << "guest protection over a guard: ok\n";
}

void CheckSharedGuardAliases() {
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(Page), nullptr);
    Require(section != nullptr, "cannot create the shared backing");
    auto* first = mapShared(Page, section);
    auto* second = mapShared(Page, section);
    const auto firstBase = reinterpret_cast<std::uint64_t>(first);
    const auto secondBase = reinterpret_cast<std::uint64_t>(second);
    const auto aliases = Aliases(firstBase, Page);
    Require(aliases.size() == 1 && aliases.front().first == secondBase && aliases.front().second == Page, "the second view was not reported as the first's alias");
    Require(Aliases(firstBase + Granule, 2 * Granule) == std::vector<std::pair<std::uint64_t, std::size_t>>{{secondBase + Granule, 2 * Granule}}, "a partial alias query did not keep its offsets");
    const auto backing = Backing(pageOf(first));
    const auto aliasBacking = Backing(pageOf(second));
    Require(backing.shared && backing.uniform && backing.aliased && backing.section == aliasBacking.section && backing.offset == aliasBacking.offset && backing.offset == 0, "the two views do not share one backing identity");
    auto* privateMemory = commitPrivate(Page, Page);
    const auto privateBacking = Backing(pageOf(privateMemory));
    Require(!privateBacking.shared && !privateBacking.aliased && privateBacking.offset == reinterpret_cast<std::uint64_t>(privateMemory), "private memory has a shared backing identity");
    Require(Aliases(reinterpret_cast<std::uint64_t>(privateMemory), Page).empty(), "private memory reported an alias");
    clearReports();
    Require(GuardWrites(firstBase, Page) == GuestArena::GuardRefusal::None, "the shared page could not be guarded");
    second[300] = 7;
    Require(reported(pageOf(first)) == 1 && reported(pageOf(second)) == 1, "a store through the alias did not report both views of the guarded backing");
    Require(first[300] == 7, "the aliased store did not land");
    first[400] = 8;
    Require(reported(pageOf(first)) == 1, "a second store in the same epoch was reported again");
    std::cout << "shared write guard and aliases: ok\n";
}

void CheckHostWriteOpensGuard() {
    auto* memory = mapShared(Page);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    clearReports();
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::None, "the page could not be guarded");
    {
        const GuestArena::HostWrite open(memory, 4096);
        Require(open.Open() && reported(pageOf(memory)) == 1, "a host write did not open the guard as a write");
        Require(protectionAt(memory) == PAGE_READWRITE, "a host write left the guarded page read-only");
        Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::AliasHostWrite, "a page was guarded while a host write was open, or refused for another reason");
        std::memset(memory, 9, 4096);
        clearRefusalClears();
    }
    Require(clearedFor(base, GuestArena::GuardRefusal::AliasHostWrite), "the end of the host write did not report that a host-write refusal can clear");
    Require(GuardWrites(base, Page) == GuestArena::GuardRefusal::None, "the page could not be guarded after the host write");
    std::cout << "host write over a guard: ok\n";
}

void CheckResidencyNotes() {
    auto* memory = commitPrivate(2 * Page, Page);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    resident = {pageOf(memory)};
    SetResidencyHooks(&lockCheckingHooks);
    clearReports();
    CollectWritesUncached(base, 2 * Page);
    memory[16] = 1;
    memory[Page + 16] = 1;
    CollectWritesUncached(base, 2 * Page);
    Require(reported(pageOf(memory)) == 1 && reported(pageOf(memory + Page)) == 0, "a collect did not report exactly the resident page's CPU store");
    const std::array<std::byte, 4> bytes{};
    StoreOwnBytes(base + 64, bytes);
    Require(reported(pageOf(memory)) == 2, "a driver store to a resident page was not reported");
    Require(MarkWritten(base + 128, 4) != 0 && reported(pageOf(memory)) == 2, "a GPU write mark on a resident page was reported to residency as a CPU store");
    SetResidencyHooks(&hooks);
    dropped.clear();
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(memory + Page, 256, true, true, true);
    }
    Require(std::find(dropped.begin(), dropped.end(), std::pair<std::uint64_t, std::size_t>{base + Page, 256}) != dropped.end(), "a registry change did not reach dropRange");
    order.clear();
    SetFlushHook(&flushHook);
    std::array<std::byte, 8> read{};
    Read(base, read);
    Require(order == std::vector<std::string>{"flush", "write-back"}, "the write-back hook did not run after the flush hook");
    SetFlushHook(nullptr);
    resident.clear();
    std::cout << "residency notes: ok\n";
}

void CheckConsumedImportMarks() {
    auto* memory = commitPrivate(Page, Page);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto committed = CollectWritesUncached(base, Page);
    std::memset(memory, 1, Page);
    ConsumeImportMarks(base, Page);
    CollectWritesUncached(base, Page);
    Require(UnchangedSinceCollected(base, Page, committed), "marks consumed after an import were collected as CPU stores");
    memory[4096] = 2;
    Require(!UnchangedSinceCollected(base, Page, committed), "a CPU store after the consumed marks was missed");
    std::cout << "consumed import marks: ok\n";
}

}
#endif

int main() {
#ifdef _WIN32
    try {
        if (!AgcDriver::GuestMemory::WriteWatched()) {
            std::cout << "no write watching: skipped\n";
            return 77;
        }
        AgcDriver::GuestMemory::SetResidencyHooks(&hooks);
        CheckPrivateGuards();
        CheckSharedGuard();
        CheckGuestProtectionPreserved();
        CheckSharedGuardAliases();
        CheckHostWriteOpensGuard();
        CheckResidencyNotes();
        CheckConsumedImportMarks();
        AgcDriver::GuestMemory::SetResidencyHooks(nullptr);
    } catch (const std::exception& error) {
        std::cerr << "write guard test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "write guard tests passed\n";
    return 0;
#else
    std::cout << "write guards are Windows-only: skipped\n";
    return 77;
#endif
}
