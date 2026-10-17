#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CommittedImports.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "VulkanTestDevice.hpp"
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <psapi.h>
#endif

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

#ifdef _WIN32

namespace {

namespace Graphics = AgcDriver::Graphics;
using namespace NewPath;

constexpr std::uint64_t MiB = 1ull << 20u;
constexpr std::uint64_t RunBytes = 512 * MiB;
constexpr std::uint64_t PieceBytes = 2 * MiB;
constexpr std::uint64_t SectionBytes = 256 * MiB;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

struct Sample {
    std::uint64_t privateBytes = 0;
    std::uint64_t workingSet = 0;
    std::uint64_t committedImports = 0;
    std::uint64_t committedViews = 0;
    std::uint64_t hostImports = 0;
};

Sample Take() {
    Sample sample;
    PROCESS_MEMORY_COUNTERS_EX process{};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process), sizeof(process));
    sample.privateBytes = process.PrivateUsage;
    sample.workingSet = process.WorkingSetSize;
    const auto committed = Graphics::CommittedImportCounters();
    sample.committedImports = committed.pinnedBytes;
    sample.committedViews = committed.views;
    sample.hostImports = Graphics::HostImportLiveBytes();
    return sample;
}

double Delta(std::uint64_t from, std::uint64_t to) {
    return (static_cast<double>(to) - static_cast<double>(from)) / static_cast<double>(MiB);
}

void Print(const char* step, const Sample& from, const Sample& to) {
    std::printf("%-46s private %+7.1f  working set %+7.1f  committed imports %+7.1f (%+lld views)  host imports %+7.1f MiB\n", step, Delta(from.privateBytes, to.privateBytes), Delta(from.workingSet, to.workingSet), Delta(from.committedImports, to.committedImports), static_cast<long long>(to.committedViews) - static_cast<long long>(from.committedViews), Delta(from.hostImports, to.hostImports));
    std::fflush(stdout);
}

void Touch(std::uint8_t* base, std::uint64_t bytes) {
    for (std::uint64_t offset = 0; offset < bytes; offset += 4096) base[offset] = static_cast<std::uint8_t>(offset >> 12u);
}

std::uint8_t* MapSection(HANDLE section, std::uint64_t bytes) {
    auto* address = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
    Require(address != nullptr, "host import memory: cannot reserve guest pages for a view");
    GuestArena::GuestArenaMap_nid_postfix(address, bytes, section, 0, PAGE_READWRITE);
    return address;
}

void Unmap(std::uint8_t* address, std::uint64_t bytes) {
    GuestAllocations::Mutation().Remove(address);
    GuestArena::GuestArenaReset_nid_postfix(address, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(address, bytes);
}

void ImportAcross(const Graphics::Context& context, const std::uint8_t* address, std::uint64_t bytes, const std::string& what) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const auto begin = reinterpret_cast<std::uint64_t>(address);
    const auto* import = Graphics::HostImportFor(context, begin, static_cast<std::size_t>(bytes));
    Expect(import != nullptr && import->base <= begin && begin + bytes <= import->base + import->bytes, what + ": no host import serves the range");
    std::printf("  %s is served by %s\n", what.c_str(), import == nullptr ? "nothing" : Graphics::CommittedImportCovering(context.device, begin, begin + bytes) == import ? "a committed view" : "a second import of the same memory");
}

void Submitted() {
    Locked([] {
        auto* recorder = Graphics::Recorder::Active();
        Require(recorder != nullptr, "host import memory: the device has no recorder");
        Graphics::ReleaseRetiredCommittedImports();
        recorder->Submit();
        recorder->Sync();
    });
}

void Run(const Graphics::Context& context) {
    const auto baseline = Take();

    auto* run = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(RunBytes, PageBytes));
    Require(run != nullptr, "host import memory: cannot reserve guest memory");
    for (std::uint64_t offset = 0; offset < RunBytes; offset += PieceBytes) GuestArena::GuestArenaCommit_nid_postfix(run + offset, PieceBytes, PAGE_READWRITE, 16384);
    GuestAllocations::Mutation().Add(run, RunBytes, true, true, true);
    Touch(run, RunBytes);
    const auto committed = Take();
    Print("512 MiB committed in 2 MiB pieces", baseline, committed);
    Expect(committed.committedImports - baseline.committedImports == RunBytes, "committed private memory is not imported exactly once");
    Expect(committed.committedViews - baseline.committedViews == RunBytes / PieceBytes, "committed private memory is not one view per commit");

    ImportAcross(context, run, RunBytes, "an import over the whole run");
    const auto runImported = Take();
    Print("an import over the whole run", committed, runImported);
    Expect(runImported.hostImports - committed.hostImports <= RunBytes, "an import over committed private memory imported more than the run");
    Expect(Delta(committed.privateBytes, runImported.privateBytes) < 0.02 * Delta(0, RunBytes), "a second import of committed private memory cost 2% or more of its bytes in private memory");

    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(SectionBytes), nullptr);
    Require(section != nullptr, "host import memory: cannot create a section");
    std::vector<std::uint8_t*> views{MapSection(section, SectionBytes)};
    GuestAllocations::Mutation().Add(views[0], SectionBytes, true, true, true);
    Touch(views[0], SectionBytes);
    const auto mapped = Take();
    Print("a 256 MiB section mapped once", runImported, mapped);
    Expect(mapped.committedImports - runImported.committedImports == SectionBytes, "a mapped section view is not imported exactly once");
    views.push_back(MapSection(section, SectionBytes));
    GuestAllocations::Mutation().Add(views[1], SectionBytes, true, true, true);
    const auto aliased = Take();
    Print("the same section mapped at a second address", mapped, aliased);
    Expect(aliased.committedImports - mapped.committedImports == SectionBytes, "a second view of a section is not imported exactly once");
    Expect(Delta(mapped.privateBytes, aliased.privateBytes) < 0.03 * Delta(0, SectionBytes), "a second view of a section cost 3% or more of its bytes in private memory");

    HANDLE low = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(SectionBytes / 2), nullptr);
    HANDLE high = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(SectionBytes / 2), nullptr);
    Require(low != nullptr && high != nullptr, "host import memory: cannot create the adjacent sections");
    auto* pair = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(SectionBytes, PageBytes));
    Require(pair != nullptr, "host import memory: cannot reserve guest pages for adjacent views");
    GuestArena::GuestArenaMap_nid_postfix(pair, SectionBytes / 2, low, 0, PAGE_READWRITE);
    GuestArena::GuestArenaMap_nid_postfix(pair + SectionBytes / 2, SectionBytes / 2, high, 0, PAGE_READWRITE);
    GuestAllocations::Mutation().Add(pair, SectionBytes, true, true, true);
    Touch(pair, SectionBytes);
    const auto paired = Take();
    Print("two 128 MiB sections mapped side by side", aliased, paired);
    ImportAcross(context, pair, SectionBytes, "an import across both sections");
    const auto pairImported = Take();
    Print("an import across both sections", paired, pairImported);
    Expect(pairImported.hostImports - paired.hostImports <= SectionBytes, "an import across mapped sections imported more than the range");
    Expect(Delta(paired.privateBytes, pairImported.privateBytes) < 0.02 * Delta(0, SectionBytes), "a second import of mapped sections cost 2% or more of its bytes in private memory");

    Unmap(pair, SectionBytes);
    CloseHandle(low);
    CloseHandle(high);
    for (auto* view : views) Unmap(view, SectionBytes);
    CloseHandle(section);
    Unmap(run, RunBytes);
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        static_cast<void>(Graphics::HostImportFor(context, reinterpret_cast<std::uint64_t>(run), 4096));
    }
    Submitted();
    Submitted();
    const auto submittedAt = std::chrono::steady_clock::now();
    auto released = Take();
    for (const auto deadline = submittedAt + std::chrono::seconds(10); std::chrono::steady_clock::now() < deadline && released.committedImports != baseline.committedImports;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        released = Take();
    }
    std::printf("  imports of unmapped memory released %lld ms after the submissions\n", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - submittedAt).count()));
    Print("everything unmapped and submitted", pairImported, released);
    Print("the end against the start", baseline, released);
    Expect(released.committedImports == baseline.committedImports && released.committedViews == baseline.committedViews, "committed imports of unmapped memory are still pinned after the next submission");
    Expect(released.hostImports == baseline.hostImports, "host imports of unmapped memory are still live after the next submission");
}

}

#endif

int main() {
#ifdef _WIN32
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        if (context.hostImportAlignment == 0) {
            std::puts("host import memory: the device imports no host memory");
            return VulkanTestSkipped;
        }
        Run(context);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d host import memory checks failed\n", failures);
        return 1;
    }
    std::puts("host import memory tests passed");
    return 0;
#else
    return VulkanTestSkipped;
#endif
}
