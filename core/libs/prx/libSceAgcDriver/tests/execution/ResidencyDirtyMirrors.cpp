#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <chrono>
#include <exception>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using AgcDriver::Graphics::ResidencyAccess;
using AgcDriver::Graphics::ResidencyCounts;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t ReadLanes = MirrorReader::Lanes;
constexpr std::uint32_t ReadPages = MirrorReader::Pages;
constexpr std::uint32_t WriterQueue = 0x20;
constexpr std::uint32_t ReaderQueue = 0x28;
constexpr std::uint32_t Equal = 3;
constexpr std::uint32_t CpuStored = 0x5d000001u;
constexpr std::uint32_t CpuAfterGpu = 0x5d000002u;
constexpr std::uint32_t CpuBeforeGpu = 0x5d000003u;
constexpr std::uint32_t UnboundedValue = 0x5d000004u;
constexpr std::uint32_t LabelValue = 0x5d000005u;
constexpr std::uint32_t LabelWord = 100;
constexpr std::uint32_t BufferLoadDwordX2 = 0x0du;
constexpr auto LabelLimit = std::chrono::seconds(5);

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

const Program& WriteProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(12), 0)
              << Vop2(VAddNcU32, 2, 4, 0)
              << Mubuf(BufferStoreDword, 2, 1, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& UnboundedProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop1(VMovB32, 1, InlineInteger(0))
              << Mubuf(BufferLoadDwordX2, 2, 1, 0)
              << Vop1(VMovB32, 4, Literal) << UnboundedValue
              << WaitAll
              << Global(GlobalStoreDword, 0, 2, 4, NullSgpr)
              << EndProgram;
        return built;
    }();
    return program;
}

std::uint32_t Pattern(std::uint32_t tag, std::uint64_t page, std::uint32_t word) {
    return (tag << 28u) | (static_cast<std::uint32_t>(page) << 14u) | word;
}

std::uint32_t Written(std::uint32_t tag, std::uint64_t page, std::uint32_t subpage) {
    return 0x7c000000u | (tag << 20u) | (static_cast<std::uint32_t>(page) << 8u) | subpage;
}

class Model {
public:
    Model(std::uint64_t base, std::uint64_t bytes, std::uint32_t tag) : base(base), bytes(bytes), tag(tag) {
        for (std::uint64_t page = 0; page < bytes / PageBytes; ++page) {
            for (std::uint32_t word = 0; word < WordsPerPage; ++word) Words(page)[word] = Pattern(tag, page, word);
        }
        AgcDriver::GuestMemory::CollectWritesUncached(base, bytes);
    }

    volatile std::uint32_t* Words(std::uint64_t page) const { return reinterpret_cast<volatile std::uint32_t*>(static_cast<std::uintptr_t>(base + page * PageBytes)); }
    std::uint64_t Address(std::uint64_t page, std::uint32_t subpage = 0, std::uint32_t word = 0) const { return base + page * PageBytes + std::uint64_t{subpage} * SubpageBytes + std::uint64_t{word} * 4u; }
    std::uint64_t Pages() const { return bytes / PageBytes; }

    void RecordWrite(Commands& commands, std::uint64_t page, std::uint32_t firstSubpage = 0, std::uint32_t subpages = SubpagesPerPage) {
        const auto buffer = BufferResource(Address(page, firstSubpage), static_cast<std::uint32_t>(subpages * SubpageBytes));
        const std::array<std::uint32_t, 5> userData{buffer[0], buffer[1], buffer[2], buffer[3], Written(tag, page, firstSubpage)};
        commands.Dispatch(WriteProgram(), userData, subpages);
        for (std::uint32_t subpage = firstSubpage; subpage < firstSubpage + subpages; ++subpage) values[{page, subpage * WordsPerSubpage}] = Written(tag, page, subpage);
    }

    void Store(std::uint64_t page, std::uint32_t word, std::uint32_t value) {
        Words(page)[word] = value;
        values[{page, word}] = value;
    }

    void Set(std::uint64_t page, std::uint32_t word, std::uint32_t value) { values[{page, word}] = value; }

    std::uint32_t Value(std::uint64_t page, std::uint32_t word) const {
        const auto found = values.find({page, word});
        return found != values.end() ? found->second : Pattern(tag, page, word);
    }

private:
    std::uint64_t base;
    std::uint64_t bytes;
    std::uint32_t tag;
    std::map<std::pair<std::uint64_t, std::uint32_t>, std::uint32_t> values;
};

ResidencyCounts Counts() {
    ResidencyCounts counts;
    Locked([&] { counts = Residency().Manager().Counts(); });
    return counts;
}

bool Resident(std::uint64_t address) {
    bool resident = false;
    Locked([&] { resident = Residency().Manager().Slot(PageOf(address)).has_value(); });
    return resident;
}

std::uint32_t DirtyMask(std::uint64_t address) {
    std::uint32_t mask = 0;
    Locked([&] { mask = Residency().Manager().Dirty(PageOf(address)); });
    return mask;
}

void Use(std::uint64_t address, std::uint64_t pages) {
    Locked([&] { Residency().Manager().Use(address, pages * PageBytes, ResidencyAccess::Read); });
}

DWORD Protection(std::uint64_t address) {
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) return 0;
    return info.Protect & 0xffu;
}

bool ReadableProtection(DWORD protection) {
    return protection == PAGE_READWRITE || protection == PAGE_READONLY;
}

void ExpectResident(const Model& model, std::uint64_t first, std::uint64_t count, const std::string& what) {
    std::size_t missing = 0;
    std::string pages;
    for (std::uint64_t page = first; page < first + count; ++page) {
        if (Resident(model.Address(page))) continue;
        if (missing++ < 8u) pages += " " + std::to_string(page);
    }
    Expect(missing == 0u, what + ": " + std::to_string(missing) + " of " + std::to_string(count) + " pages of a shared view have no mirror (pages" + pages + ")");
}

void ExpectDirtySubpages(const Model& model, std::uint64_t page, std::uint32_t mask, const std::string& what) {
    const auto dirty = DirtyMask(model.Address(page));
    Expect(dirty == mask, what + ": page " + std::to_string(page) + " has dirty subpages " + Hex(dirty) + ", expected " + Hex(mask));
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) {
        const auto protection = Protection(model.Address(page, subpage));
        if ((mask & (1u << subpage)) != 0) {
            if (protection == PAGE_NOACCESS) continue;
            Expect(false, what + ": dirty subpage " + std::to_string(subpage) + " of page " + std::to_string(page) + " is not guarded against CPU access (protection " + Hex(protection) + ")");
            return;
        }
        if (ReadableProtection(protection)) continue;
        Expect(false, what + ": clean subpage " + std::to_string(subpage) + " of page " + std::to_string(page) + " is not readable (protection " + Hex(protection) + ")");
        return;
    }
}

void ExpectClean(const Model& model, std::uint64_t page, const std::string& what) {
    ExpectDirtySubpages(model, page, 0u, what);
}

void ExpectBytes(const Model& model, std::uint64_t page, const std::string& what) {
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) {
        const auto actual = model.Words(page)[word];
        const auto expected = model.Value(page, word);
        if (actual == expected || wrong++ != 0u) continue;
        first = "word " + std::to_string(word) + " holds " + Hex(actual) + ", expected " + Hex(expected);
    }
    Expect(wrong == 0u, what + ": the CPU read " + std::to_string(wrong) + " wrong words of page " + std::to_string(page) + " directly, first " + first);
}

void ExpectLanes(const std::vector<std::uint32_t>& lanes, const Model& model, std::uint64_t firstPage, const std::string& what) {
    for (std::uint32_t lane = 0; lane < ReadLanes; ++lane) {
        const auto page = firstPage + lane / SubpagesPerPage;
        const auto subpage = lane % SubpagesPerPage;
        const auto expected = model.Value(page, subpage * WordsPerSubpage);
        if (lanes[lane] == expected) continue;
        Expect(false, what + ": page " + std::to_string(page) + " subpage " + std::to_string(subpage) + " read " + Hex(lanes[lane]) + ", expected " + Hex(expected));
        return;
    }
}

std::string Submitted(Commands& commands, std::uint32_t queue = 0) {
    std::string failure;
    StderrCapture capture;
    try {
        commands.Submit(queue);
        WaitIdle();
        Settle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    const auto log = capture.Finish();
    if (log.find("failed") != std::string::npos) failure += "\n" + log;
    return failure;
}

bool AwaitLabel(volatile std::uint32_t* label, std::uint32_t value) {
    const auto until = Clock::now() + LabelLimit;
    while (*label != value) {
        if (Clock::now() > until) return false;
        std::this_thread::yield();
    }
    return true;
}

void Store(Model& model, std::uint64_t page, const std::string& what, std::uint32_t firstSubpage = 0, std::uint32_t subpages = SubpagesPerPage) {
    Commands commands;
    model.RecordWrite(commands, page, firstSubpage, subpages);
    const auto failure = Submitted(commands);
    Expect(failure.empty(), what + ": the storing submission failed: " + failure);
}

std::uint32_t Mask(std::uint32_t firstSubpage, std::uint32_t subpages) {
    return ((1u << subpages) - 1u) << firstSubpage;
}

constexpr std::uint32_t WholePage = 0xffffu;

void CpuWritesAfterMirroring(MirrorReader& reader) {
    SharedPages pages(ReadPages);
    Model model(pages.Address(), pages.Bytes(), 1);
    Settle();
    const std::string what = "CPU writes after mirroring";
    ExpectResident(model, 0, ReadPages, what + ": a shared view before any GPU use");
    model.Store(3, 5 * WordsPerSubpage, CpuStored);
    model.Store(9, 0, CpuStored + 1u);
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch after the CPU stored into mirrored pages 3 and 9");
    ExpectResident(model, 0, ReadPages, what + ": after the dispatch");
}

void CpuReadsGpuStore(MirrorReader& reader) {
    SharedPages pages(ReadPages);
    Model model(pages.Address(), pages.Bytes(), 2);
    Settle();
    const std::string what = "CPU read of a GPU store";
    const auto before = Counts();
    Store(model, 2, what);
    ExpectResident(model, 2, 1, what + ": after a dispatch stored into page 2");
    ExpectDirtySubpages(model, 2, WholePage, what + ": before the CPU read");
    ExpectClean(model, 3, what + ": a page the GPU did not store into");
    const auto stored = Counts();
    Expect(stored.writtenBackBytes == before.writtenBackBytes && stored.cpuWriteBacks == before.cpuWriteBacks, what + ": " + std::to_string(stored.writtenBackBytes - before.writtenBackBytes) + " bytes were written back before any CPU read");
    ExpectBytes(model, 2, what + ": the test's own read of page 2 after the store");
    const auto read = Counts();
    Expect(read.cpuWriteBacks > stored.cpuWriteBacks && read.writtenBackBytes == stored.writtenBackBytes + PageBytes, what + ": the CPU read wrote back " + std::to_string(read.writtenBackBytes - stored.writtenBackBytes) + " bytes in " + std::to_string(read.cpuWriteBacks - stored.cpuWriteBacks) + " CPU write-backs, expected the page's 16 dirty subpages");
    ExpectClean(model, 2, what + ": after the CPU read wrote page 2 back");
    ExpectResident(model, 2, 1, what + ": after the CPU read");
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch after the write-back");
}

void GpuReadsGpuStore(MirrorReader& reader) {
    SharedPages pages(ReadPages);
    Model model(pages.Address(), pages.Bytes(), 3);
    Settle();
    const std::string what = "GPU read of a GPU store";
    Store(model, 4, what);
    const auto stored = Counts();
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch in the submission after a store into page 4");
    Settle();
    const auto read = Counts();
    Expect(read.writtenBackBytes == stored.writtenBackBytes, what + ": the later GPU read wrote back " + std::to_string(read.writtenBackBytes - stored.writtenBackBytes) + " bytes");
    ExpectResident(model, 4, 1, what + ": after the read");
    ExpectDirtySubpages(model, 4, WholePage, what + ": after a later GPU read, with no CPU read between");
    ExpectBytes(model, 4, what + ": the test's own read of page 4 at the end");
}

void UnboundedWriter(MirrorReader& reader) {
    SharedPages pages(ReadPages);
    PrivatePages addresses(1);
    Model model(pages.Address(), pages.Bytes(), 4);
    const auto target = model.Address(6) + 8u;
    addresses.Words()[0] = Low(target);
    addresses.Words()[1] = High(target);
    AgcDriver::GuestMemory::CollectWritesUncached(addresses.Address(), addresses.Bytes());
    Settle();
    const std::string what = "unbounded writer";
    ExpectResident(model, 0, ReadPages, what + ": before the dispatch");
    const auto before = Counts();
    Commands commands;
    commands.Dispatch(UnboundedProgram(), BufferResource(addresses.Address(), 16u), 1);
    model.Set(6, 2, UnboundedValue);
    const auto failure = Submitted(commands);
    Expect(failure.empty(), what + ": the submission of a dispatch storing through an address it loads from memory failed: " + failure);
    const auto after = Counts();
    Expect(after.demoted >= before.demoted + ReadPages, what + ": a dispatch storing through an address it loads from memory demoted " + std::to_string(after.demoted - before.demoted) + " of the view's " + std::to_string(ReadPages) + " GPU-writable mirrors before it ran");
    Expect(after.dirty == 0u, what + ": " + std::to_string(after.dirty) + " pages are dirty in their mirrors after a store with no record-time grant");
    ExpectClean(model, 6, what + ": the page the store reached");
    ExpectBytes(model, 6, what + ": the test's own read of the stored page");
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch after the unbounded store");
}

void SubpageGranularity(MirrorReader& reader) {
    SharedPages pages(ReadPages);
    Model model(pages.Address(), pages.Bytes(), 5);
    Settle();
    const std::string what = "4 KiB granularity";
    Store(model, 8, what, 2, 1);
    ExpectDirtySubpages(model, 8, Mask(2, 1), what + ": a dispatch stored into subpage 2 of page 8 only");
    const auto stored = Counts();
    model.Store(8, 5 * WordsPerSubpage + 7, CpuAfterGpu);
    const auto written = Counts();
    Expect(written.writtenBackBytes == stored.writtenBackBytes && written.cpuWriteBacks == stored.cpuWriteBacks, what + ": a CPU store into clean subpage 5 of the dirty page wrote back " + std::to_string(written.writtenBackBytes - stored.writtenBackBytes) + " bytes");
    ExpectDirtySubpages(model, 8, Mask(2, 1), what + ": after a CPU store into a clean subpage");
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch after the CPU store into the clean subpage");
    Settle();
    const auto first = model.Words(8)[2 * WordsPerSubpage];
    const auto read = Counts();
    Expect(first == model.Value(8, 2 * WordsPerSubpage), what + ": a CPU read of dirty subpage 2 read " + Hex(first) + ", expected the GPU's " + Hex(model.Value(8, 2 * WordsPerSubpage)));
    Expect(read.cpuWriteBacks > written.cpuWriteBacks && read.writtenBackBytes == written.writtenBackBytes + SubpageBytes, what + ": the CPU read of the dirty subpage wrote back " + std::to_string(read.writtenBackBytes - written.writtenBackBytes) + " bytes, expected the one dirty subpage");
    ExpectClean(model, 8, what + ": after the CPU read");
    ExpectBytes(model, 8, what + ": the test's own read of page 8 at the end");
}

void CpuAndGpuInOnePage(MirrorReader& reader) {
    SharedPages pages(ReadPages);
    PrivatePages labels(1);
    Model model(pages.Address(), pages.Bytes(), 6);
    labels.Words()[0] = 0u;
    labels.Words()[16] = 0u;
    AgcDriver::GuestMemory::CollectWritesUncached(labels.Address(), labels.Bytes());
    Settle();
    const std::string what = "CPU and GPU stores into different 4 KiB parts of page 7, ordered by labels";
    Commands gpuFirst;
    model.RecordWrite(gpuFirst, 7, 2, 1);
    gpuFirst.Barrier();
    gpuFirst.WriteData(labels.Address(), {1u});
    gpuFirst.Submit(0);
    Expect(AwaitLabel(labels.Words(), 1u), what + ": the label after the GPU store never reached the CPU");
    model.Store(7, 5 * WordsPerSubpage, CpuAfterGpu);
    model.Store(7, 9 * WordsPerSubpage, CpuBeforeGpu);
    labels.Words()[16] = 1u;
    Commands cpuFirst;
    cpuFirst.Wait(labels.Address(64), Equal, 1u, 0xffffffffu);
    model.RecordWrite(cpuFirst, 7, 11, 1);
    const auto failure = Submitted(cpuFirst);
    Expect(failure.empty(), what + ": the submission waiting for the CPU's label failed: " + failure);
    ExpectResident(model, 7, 1, what);
    ExpectDirtySubpages(model, 7, Mask(2, 1) | Mask(11, 1), what + ": after both GPU stores");
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch reading every 4 KiB part afterwards");
    ExpectBytes(model, 7, what + ": the test's own read of page 7 at the end");
}

void CrossStream(MirrorReader& reader) {
    SharedPages pages(ReadPages);
    PrivatePages labels(1);
    Model model(pages.Address(), pages.Bytes(), 7);
    labels.Words()[0] = 0u;
    AgcDriver::GuestMemory::CollectWritesUncached(labels.Address(), labels.Bytes());
    Settle();
    const std::string what = "cross-stream";
    Commands waiter;
    waiter.Wait(labels.Address(), Equal, 1u, 0xffffffffu);
    reader.Record(waiter, model.Address(0), ReadPages);
    waiter.Submit(ReaderQueue);
    Commands writer;
    model.RecordWrite(writer, 5);
    model.RecordWrite(writer, 6);
    writer.Barrier();
    writer.WriteData(labels.Address(), {1u});
    writer.Submit(WriterQueue);
    ExpectLanes(reader.Collect(), model, 0, what + ": a dispatch on one queue waiting for the label another queue writes after storing into pages 5 and 6");
    Settle();
    ExpectResident(model, 0, ReadPages, what + ": after the read on the other queue");
    ExpectDirtySubpages(model, 5, WholePage, what + ": after the read on the other queue");
    ExpectDirtySubpages(model, 6, WholePage, what + ": after the read on the other queue");
    ExpectBytes(model, 5, what + ": the test's own read");
    ExpectBytes(model, 6, what + ": the test's own read");
}

void LabelInDirtySubpage() {
    SharedPages pages(ReadPages);
    Model model(pages.Address(), pages.Bytes(), 8);
    Settle();
    const std::string what = "a label stored into the 4 KiB subpage a dispatch stored into";
    Commands commands;
    model.RecordWrite(commands, 10, 3, 1);
    commands.Barrier();
    commands.WriteData(model.Address(10, 3, LabelWord), {LabelValue});
    model.Set(10, 3 * WordsPerSubpage + LabelWord, LabelValue);
    const auto failure = Submitted(commands);
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    ExpectBytes(model, 10, what + ": the test's own read after the batch, which must see the label and the dispatch's store");
}

void ProtectDirtyPage(MirrorReader& reader) {
    SharedPages pages(ReadPages);
    Model model(pages.Address(), pages.Bytes(), 9);
    Settle();
    const std::string what = "guest protection of a dirty page";
    Store(model, 3, what);
    ExpectDirtySubpages(model, 3, WholePage, what + ": before the protection");
    const auto before = Counts();
    std::string protection;
    try {
        GuestAllocations::Mutation().Protect(reinterpret_cast<void*>(model.Address(3)), PageBytes, true, false, true, [&] { GuestArena::GuestArenaSetProtection_nid_postfix(static_cast<std::uintptr_t>(model.Address(3)), PageBytes, PAGE_READONLY); });
    } catch (const std::exception& error) {
        protection = error.what();
    }
    Expect(protection.empty(), what + ": making page 3 read-only failed: " + protection);
    const auto after = Counts();
    Expect(after.writtenBackBytes == before.writtenBackBytes && DirtyMask(model.Address(3)) == WholePage, what + ": the protection wrote back " + std::to_string(after.writtenBackBytes - before.writtenBackBytes) + " bytes or cleaned the page; a protection change keeps it dirty and guarded until a CPU read");
    ExpectBytes(model, 3, what + ": the test's own read of the page made read-only after the GPU stored into it");
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch reading the read-only page");
    GuestAllocations::Mutation().Protect(reinterpret_cast<void*>(model.Address(3)), PageBytes, true, true, true, [&] { GuestArena::GuestArenaSetProtection_nid_postfix(static_cast<std::uintptr_t>(model.Address(3)), PageBytes, PAGE_READWRITE); });
}

void UnmapDirtyView(MirrorReader& reader) {
    const std::string what = "guest unmap of a dirty view";
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(PageBytes), nullptr);
    Require(section != nullptr, "unmap of a dirty view: cannot create a section");
    auto* first = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(PageBytes, PageBytes));
    auto* second = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(PageBytes, PageBytes));
    Require(first != nullptr && second != nullptr, "unmap of a dirty view: cannot reserve two pages");
    GuestArena::GuestArenaMap_nid_postfix(first, PageBytes, section, 0, PAGE_READWRITE);
    GuestAllocations::Mutation().Add(first, PageBytes, true, true, true);
    Model model(reinterpret_cast<std::uint64_t>(first), PageBytes, 10);
    Settle();
    Store(model, 0, what);
    ExpectDirtySubpages(model, 0, WholePage, what + ": before the unmap");
    std::string unmap;
    try {
        GuestAllocations::Mutation().Remove(first);
        GuestArena::GuestArenaReset_nid_postfix(first, PageBytes);
    } catch (const std::exception& error) {
        unmap = error.what();
    }
    Expect(unmap.empty(), what + ": the unmap failed: " + unmap);
    GuestArena::GuestArenaMap_nid_postfix(second, PageBytes, section, 0, PAGE_READWRITE);
    GuestAllocations::Mutation().Add(second, PageBytes, true, true, true);
    std::size_t wrong = 0;
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) wrong += reinterpret_cast<volatile std::uint32_t*>(second)[word] != model.Value(0, word) ? 1u : 0u;
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " words of another view of the section, mapped after the dirty view was unmapped, miss the GPU's store");
    const auto lanes = reader.Read(reinterpret_cast<std::uint64_t>(second), 1);
    std::size_t gpuWrong = 0;
    for (std::uint32_t lane = 0; lane < SubpagesPerPage; ++lane) gpuWrong += lanes[lane] != model.Value(0, lane * WordsPerSubpage) ? 1u : 0u;
    Expect(gpuWrong == 0u, what + ": a dispatch read " + std::to_string(gpuWrong) + " subpages of the other view without the GPU's store");
    GuestAllocations::Mutation().Remove(second);
    GuestArena::GuestArenaReset_nid_postfix(second, PageBytes);
    GuestArena::GuestArenaRelease_nid_postfix(first, PageBytes);
    GuestArena::GuestArenaRelease_nid_postfix(second, PageBytes);
    CloseHandle(section);
}

void UnmapDirtyPage(MirrorReader& reader) {
    SharedPages pages(1);
    Model model(pages.Address(), pages.Bytes(), 11);
    Settle();
    const std::string what = "guest unmap of a dirty page and a new section at the same address";
    Store(model, 0, what);
    ExpectDirtySubpages(model, 0, WholePage, what + ": before the unmap");
    pages.Unmap();
    pages.Map();
    std::size_t stale = 0;
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) stale += pages.Words(0)[word] != 0u ? 1u : 0u;
    Expect(stale == 0u, what + ": " + std::to_string(stale) + " words of the new section are not zero to the CPU");
    const auto lanes = reader.Read(pages.Address(), 1);
    std::size_t gpuStale = 0;
    for (std::uint32_t lane = 0; lane < SubpagesPerPage; ++lane) gpuStale += lanes[lane] != 0u ? 1u : 0u;
    Expect(gpuStale == 0u, what + ": a dispatch read " + std::to_string(gpuStale) + " non-zero words from the new section");
}

void PrivatePagesDemote(MirrorReader& reader) {
    PrivatePages pages(ReadPages);
    Model model(pages.Address(), pages.Bytes(), 12);
    Settle();
    const std::string what = "private pages";
    Expect(!Resident(model.Address(2)), what + ": a private page has a mirror before any use");
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": the read that admits the pages");
    Expect(Resident(model.Address(2)), what + ": page 2 was not admitted by the read");
    Expect(!AgcDriver::GuestMemory::WriteGuarded(PageOf(model.Address(2))) && !AgcDriver::GuestMemory::AccessGuarded(PageOf(model.Address(2))), what + ": an admitted private page is guarded");
    const auto before = Counts();
    Store(model, 2, what);
    const auto after = Counts();
    Expect(!Resident(model.Address(2)) && after.demoted > before.demoted, what + ": a write grant into a private page did not demote it (" + std::to_string(after.demoted - before.demoted) + " demotions)");
    Expect(DirtyMask(model.Address(2)) == 0u && after.dirty == before.dirty, what + ": a private page is dirty in its mirror");
    Expect(!AgcDriver::GuestMemory::WriteGuarded(PageOf(model.Address(2))) && !AgcDriver::GuestMemory::AccessGuarded(PageOf(model.Address(2))), what + ": the stored private page is guarded");
    Expect(ReadableProtection(Protection(model.Address(2))), what + ": the stored private page is not readable (protection " + Hex(Protection(model.Address(2))) + ")");
    ExpectBytes(model, 2, what + ": the test's own read of the stored page");
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch after the store");
}

void EvictionWritesBack(MirrorReader& reader) {
    Settle();
    std::uint32_t chunkSlots = 0;
    Locked([&] {
        auto& manager = Residency().Manager();
        chunkSlots = manager.ChunkSlots();
        manager.SetBudget(std::uint64_t{chunkSlots} * PageBytes);
    });
    Settle();
    SharedPages pages(chunkSlots + ReadPages);
    Model model(pages.Address(), pages.Bytes(), 13);
    const std::string what = "eviction under a one-chunk budget";
    Use(model.Address(0), ReadPages);
    Settle();
    Store(model, 1, what);
    ExpectResident(model, 1, 1, what + ": after a dispatch stored into page 1");
    ExpectDirtySubpages(model, 1, WholePage, what + ": before the eviction");
    Use(model.Address(ReadPages), chunkSlots - ReadPages);
    Settle();
    const auto before = Counts();
    Use(model.Address(chunkSlots), ReadPages);
    Settle();
    const auto after = Counts();
    Expect(after.evicted >= before.evicted + ReadPages, what + ": admitting " + std::to_string(ReadPages) + " pages evicted " + std::to_string(after.evicted - before.evicted) + " of the " + std::to_string(ReadPages) + " least recently used");
    Expect(!Resident(model.Address(1)), what + ": page 1, among the least recently used, stayed resident");
    Expect(after.writtenBackBytes >= before.writtenBackBytes + PageBytes, what + ": evicting the dirty page wrote back " + std::to_string(after.writtenBackBytes - before.writtenBackBytes) + " bytes");
    Expect(ReadableProtection(Protection(model.Address(1))), what + ": page 1 is not readable after its eviction (protection " + Hex(Protection(model.Address(1))) + ")");
    ExpectBytes(model, 1, what + ": the test's own read of page 1, written back when it was evicted");
    ExpectLanes(reader.Read(model.Address(chunkSlots), ReadPages), model, chunkSlots, what + ": a dispatch reading the pages admitted into the evicted slots");
    ExpectLanes(reader.Read(model.Address(0), ReadPages), model, 0, what + ": a dispatch reading the evicted pages");
    Settle();
    Locked([&] { Residency().Manager().SetBudget(0); });
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
        }
        Locked([] { Residency().Manager().SetBudget(std::uint64_t{1} << 30); });
        for (const auto& [name, run] : std::initializer_list<std::pair<const char*, void (*)(MirrorReader&)>>{
                 {"CPU writes after mirroring", CpuWritesAfterMirroring},
                 {"CPU read of a GPU store", CpuReadsGpuStore},
                 {"GPU read of a GPU store", GpuReadsGpuStore},
                 {"unbounded writer", UnboundedWriter},
                 {"4 KiB granularity", SubpageGranularity},
                 {"CPU and GPU in one page", CpuAndGpuInOnePage},
                 {"cross-stream", CrossStream},
                 {"label in a dirty subpage", [](MirrorReader&) { LabelInDirtySubpage(); }},
                 {"protection of a dirty page", ProtectDirtyPage},
                 {"unmap of a dirty view", UnmapDirtyView},
                 {"unmap of a dirty page", UnmapDirtyPage},
                 {"private pages", PrivatePagesDemote},
                 {"eviction", EvictionWritesBack},
             }) {
            try {
                run(reader);
            } catch (const std::exception& error) {
                Expect(false, std::string(name) + ": exception: " + error.what());
            }
        }
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d residency dirty mirror checks failed\n", failures);
        return 1;
    }
    std::printf("residency dirty mirror tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
