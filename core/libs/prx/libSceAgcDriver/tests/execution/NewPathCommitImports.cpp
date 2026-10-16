#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CommittedImports.hpp"
#include "VulkanTestDevice.hpp"
#include <exception>
#include <memory>

#ifdef _WIN32

namespace {

using namespace NewPath;
using AgcDriver::Graphics::ImportOf;
using AgcDriver::Graphics::ResidencyAccess;

constexpr std::uint32_t Lanes = MirrorReader::Lanes;
constexpr std::uint32_t MirrorMarker = 0x5b000001u;
constexpr std::uint32_t DriverStore = 0x5b000002u;
constexpr std::uint32_t LabelValue = 0x5b000003u;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::uint32_t Pattern(std::uint32_t tag, std::uint64_t page, std::uint32_t word) {
    return (tag << 28u) | (static_cast<std::uint32_t>(page) << 14u) | word;
}

void FillPattern(volatile std::uint32_t* words, std::uint64_t pages, std::uint32_t tag) {
    for (std::uint64_t page = 0; page < pages; ++page) {
        for (std::uint32_t word = 0; word < WordsPerPage; ++word) words[page * WordsPerPage + word] = Pattern(tag, page, word);
    }
}

VkDevice Device() {
    VkDevice device = VK_NULL_HANDLE;
    Locked([&] {
        auto* recorder = AgcDriver::Graphics::Recorder::Active();
        Require(recorder != nullptr, "commit imports: the driver opened no device");
        device = recorder->Device();
    });
    return device;
}

void ExpectRead(const std::vector<std::uint32_t>& lanes, std::uint32_t pages, const std::function<std::uint32_t(std::uint64_t page, std::uint32_t subpage)>& expected, const std::string& what) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const auto page = lane / SubpagesPerPage;
        const auto subpage = lane % SubpagesPerPage;
        const auto wanted = page < pages ? expected(page, subpage) : 0u;
        if (lanes[lane] == wanted) continue;
        Expect(false, what + ": page " + std::to_string(page) + " subpage " + std::to_string(subpage) + " read " + Hex(lanes[lane]) + ", expected " + Hex(wanted));
        return;
    }
}

std::uint64_t PageNumber(std::uint64_t address) {
    return address >> AgcDriver::GuestMemory::ResidencyPageShift;
}

void ImportedAtCommit(MirrorReader& reader) {
    PrivatePages pages(2);
    const auto vk = Device();
    const auto first = ImportOf(vk, PageNumber(pages.Address()));
    const auto second = ImportOf(vk, PageNumber(pages.Address(PageBytes)));
    Expect(first && second && first->bytes == PageBytes && second->bytes == PageBytes, "eager import: a commit was not imported whole before its first GPU use");
    FillPattern(pages.Words(), 2, 1);
    ExpectRead(reader.Read(pages.Address(), 2), 2, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(1, page, subpage * WordsPerSubpage); }, "eager import: a dispatch right after a commit");
    const auto after = ImportOf(vk, PageNumber(pages.Address()));
    Expect(first && after && after->serial == first->serial && after->buffer == first->buffer, "eager import: the first GPU use of a committed page imported it again");
}

void ReleaseAndRecommit(MirrorReader& reader) {
    const auto bytes = 2 * PageBytes;
    auto* base = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
    Require(base != nullptr, "release and recommit: cannot reserve two pages");
    GuestArena::GuestArenaCommit_nid_postfix(base, bytes, PAGE_READWRITE, 16384);
    GuestAllocations::Mutation().Add(base, bytes, true, true, true);
    auto* words = reinterpret_cast<volatile std::uint32_t*>(base);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base));
    FillPattern(words, 2, 2);
    ExpectRead(reader.Read(address, 2), 2, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(2, page, subpage * WordsPerSubpage); }, "release and recommit: the read before the release");
    const auto vk = Device();
    const auto before = ImportOf(vk, PageNumber(address + PageBytes));
    GuestAllocations::Mutation().Remove(base);
    GuestArena::GuestArenaReset_nid_postfix(base + PageBytes, PageBytes);
    GuestArena::GuestArenaCommit_nid_postfix(base + PageBytes, PageBytes, PAGE_READWRITE, 16384);
    GuestAllocations::Mutation().Add(base, bytes, true, true, true);
    FillPattern(words + WordsPerPage, 1, 3);
    const auto again = ImportOf(vk, PageNumber(address + PageBytes));
    Expect(before && again && again->serial != before->serial, "release and recommit: a page committed again kept its old import");
    ExpectRead(reader.Read(address, 2), 2, [](std::uint64_t page, std::uint32_t subpage) { return page == 0 ? Pattern(2, 0, subpage * WordsPerSubpage) : Pattern(3, 0, subpage * WordsPerSubpage); }, "release and recommit: a dispatch after a page was released and committed again did not read the new page");
    GuestAllocations::Mutation().Remove(base);
    GuestArena::GuestArenaReset_nid_postfix(base, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(base, bytes);
}

void SharedAliases(MirrorReader& reader) {
    const auto bytes = 2 * PageBytes;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
    Require(section != nullptr, "shared aliases: cannot create a section");
    auto* first = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
    auto* second = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
    GuestArena::GuestArenaMap_nid_postfix(first, bytes, section, 0, PAGE_READWRITE);
    GuestArena::GuestArenaMap_nid_postfix(second, bytes, section, 0, PAGE_READWRITE);
    GuestAllocations::Mutation().Add(first, bytes, true, true, true);
    GuestAllocations::Mutation().Add(second, bytes, true, true, true);
    FillPattern(reinterpret_cast<volatile std::uint32_t*>(first), 2, 4);
    const auto through = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(second));
    ExpectRead(reader.Read(through, 2), 2, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(4, page, subpage * WordsPerSubpage); }, "shared aliases: a dispatch through one view of a section did not read what the CPU stored through the other");
    GuestAllocations::Mutation().Remove(first);
    GuestAllocations::Mutation().Remove(second);
    GuestArena::GuestArenaReset_nid_postfix(first, bytes);
    GuestArena::GuestArenaReset_nid_postfix(second, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(first, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(second, bytes);
    CloseHandle(section);
}

bool Resident(std::uint64_t address) {
    bool resident = false;
    Locked([&] { resident = Residency().Manager().Slot(PageOf(address)).has_value(); });
    return resident;
}

void DriverStoreToUnguardedResidentPage(MirrorReader& reader) {
    PrivatePages page(1, false);
    FillPattern(page.Words(), 1, 5);
    AgcDriver::GuestMemory::CollectWritesUncached(page.Address(), PageBytes);
    Locked([&] { Residency().Manager().Use(page.Address(), PageBytes, ResidencyAccess::Read); });
    Expect(Resident(page.Address()), "write watch: a page the guest cannot write was not admitted");
    Expect(!AgcDriver::GuestMemory::WriteGuarded(PageOf(page.Address())), "write watch: a page the guest cannot write was guarded");
    FillMirror(page.Address(SubpageBytes), MirrorMarker);
    ExpectRead(reader.Read(page.Address(), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return subpage == 1 ? MirrorMarker : Pattern(5, 0, subpage * WordsPerSubpage); }, "write watch: the read through the mirror");
    const std::uint32_t value = DriverStore;
    AgcDriver::GuestMemory::StoreOwnBytes(page.Address(2 * SubpageBytes), std::as_bytes(std::span(&value, 1)));
    ExpectRead(reader.Read(page.Address(), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return subpage == 2 ? DriverStore : Pattern(5, 0, subpage * WordsPerSubpage); }, "write watch: the dispatch after a driver CPU store into an unguarded resident page read a stale mirror");
    Settle();
}

enum class Recording { Idle, Busy };

constexpr std::uint32_t WrittenSubpage = 3;

void DriverWrite(MirrorReader& reader, MirrorReader& other, SharedPages& pages, Recording recording, const std::string& what, std::uint32_t value, const std::function<void(Commands&, const SharedPages&)>& store) {
    PrivatePages scratch(1);
    FillPattern(pages.Words(), 2, 6);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    bool stored = false;
    const auto expected = [&](std::uint64_t page, std::uint32_t subpage) { return stored && page == 0 && subpage == WrittenSubpage ? value : Pattern(6, page, subpage * WordsPerSubpage); };
    Locked([&] { Residency().Manager().Use(pages.Address(), PageBytes, ResidencyAccess::Read); });
    Expect(Resident(pages.Address()), "driver writes: page 0 was not admitted");
    ExpectRead(reader.Read(pages.Address(), 2), 2, expected, "driver writes: the read after admission");
    Settle();
    Commands commands;
    if (recording == Recording::Busy) other.Record(commands, scratch.Address(), 1);
    store(commands, pages);
    reader.Record(commands, pages.Address(), 2);
    commands.Submit(0);
    stored = true;
    const std::string where = recording == Recording::Idle ? "with the recorder idle" : "behind a recorded dispatch";
    ExpectRead(reader.Collect(), 2, expected, "driver writes: a dispatch after " + what + " into a resident page in the same submission, " + where + ", read a stale mirror");
    Expect(Resident(pages.Address()), "driver writes: " + what + " into the import " + where + " made the page leave residency");
    Settle();
}

void DriverWritesBeforeDispatch(MirrorReader& reader) {
    MirrorReader other;
    std::vector<std::unique_ptr<SharedPages>> regions;
    const auto fresh = [&]() -> SharedPages& { return *regions.emplace_back(std::make_unique<SharedPages>(2)); };
    for (const auto recording : {Recording::Idle, Recording::Busy}) {
        DriverWrite(reader, other, fresh(), recording, "a WRITE_DATA", LabelValue, [](Commands& commands, const SharedPages& pages) { commands.WriteData(pages.Address(0, WrittenSubpage), {LabelValue}); });
        DriverWrite(reader, other, fresh(), recording, "a 4 KiB DMA copy", Pattern(6, 1, 0), [](Commands& commands, const SharedPages& pages) { commands.DmaCopy(pages.Address(0, WrittenSubpage), pages.Address(1, 0), static_cast<std::uint32_t>(SubpageBytes)); });
        DriverWrite(reader, other, fresh(), recording, "a 6-byte DMA copy", Pattern(6, 1, 2 * WordsPerSubpage), [](Commands& commands, const SharedPages& pages) { commands.DmaCopy(pages.Address(0, WrittenSubpage), pages.Address(1, 2), 6u); });
    }
}

}

#endif

int main() {
#ifdef _WIN32
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    try {
        MirrorReader reader;
        {
            PrivatePages warm(1);
            reader.Read(warm.Address(), 1);
            reader.Read(warm.Address(), 1);
        }
        if (!ImportOf(Device(), PageNumber(PrivatePages(1).Address()))) {
            std::printf("skipped, the device has no host imports\n");
            AgcDriverShutdown_nid_postfix();
            return VulkanTestSkipped;
        }
        ImportedAtCommit(reader);
        ReleaseAndRecommit(reader);
        SharedAliases(reader);
        Locked([] { Residency().Manager().SetBudget(std::uint64_t{1} << 30); });
        DriverStoreToUnguardedResidentPage(reader);
        DriverWritesBeforeDispatch(reader);
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d new path commit import checks failed\n", failures);
        return 1;
    }
    std::printf("new path commit import tests passed\n");
    return 0;
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
