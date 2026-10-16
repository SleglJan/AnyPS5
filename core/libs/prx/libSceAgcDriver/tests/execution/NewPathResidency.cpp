#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <cstdlib>
#include <exception>
#include <map>
#include <utility>

#ifdef _WIN32

namespace {

using namespace NewPath;
using AgcDriver::Graphics::ResidencyAccess;
using AgcDriver::Graphics::ResidencyCounts;

constexpr std::uint32_t ReadLanes = MirrorReader::Lanes;
constexpr std::uint32_t ReadPages = MirrorReader::Pages;
constexpr std::uint32_t SectionPages = 32;
constexpr std::uint32_t WriteLanes = SubpagesPerPage;
constexpr std::uint32_t WrittenBase = 0x7c000000u;
constexpr std::uint32_t MirrorMarker = 0x5a000001u;
constexpr std::uint32_t StoredByCpu = 0x5a000002u;
constexpr std::uint32_t HotStore = 0x5a000100u;
constexpr std::uint32_t WrittenBack = 0x5a000003u;
constexpr std::uint32_t DroppedMarker = 0x5a000004u;
constexpr std::uint32_t EvictedMarker = 0x5a000005u;
constexpr std::uint32_t AdmittedMarker = 0x5a000006u;
constexpr std::uint64_t TestBudget = std::uint64_t{1} << 30;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

Program WriteProgram;

void BuildPrograms() {
    using namespace Isa;
    WriteProgram << Vop2(VLshlRevB32, 1, InlineInteger(12), 0)
                 << Vop2(VAddNcU32, 2, Literal, 0) << WrittenBase
                 << Mubuf(BufferStoreDword, 2, 1, 0)
                 << EndProgram;
}

std::uint32_t Pattern(std::uint32_t tag, std::uint64_t page, std::uint32_t word) {
    return (tag << 28u) | (static_cast<std::uint32_t>(page) << 14u) | word;
}

class Model {
public:
    Model(SharedPages& pages, std::uint32_t tag) : pages(pages), tag(tag) {}

    void Fill() {
        const auto count = pages.Bytes() / PageBytes;
        for (std::uint64_t page = 0; page < count; ++page) {
            auto* words = pages.Words(page);
            for (std::uint32_t word = 0; word < WordsPerPage; ++word) words[word] = Pattern(tag, page, word);
        }
        AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
        values.clear();
    }

    void Retag(std::uint32_t next) {
        tag = next;
        Fill();
    }

    void Store(std::uint64_t page, std::uint32_t word, std::uint32_t value) {
        pages.Words(page)[word] = value;
        values[{page, word}] = value;
    }

    void Set(std::uint64_t page, std::uint32_t word, std::uint32_t value) { values[{page, word}] = value; }

    std::uint32_t Value(std::uint64_t page, std::uint32_t word) const {
        const auto found = values.find({page, word});
        return found != values.end() ? found->second : Pattern(tag, page, word);
    }

    SharedPages& Pages() { return pages; }

private:
    SharedPages& pages;
    std::uint32_t tag;
    std::map<std::pair<std::uint64_t, std::uint32_t>, std::uint32_t> values;
};

class Harness {
public:
    void RecordRead(Commands& commands, const SharedPages& source, std::uint64_t firstPage, std::uint32_t pages = ReadPages) { reader.Record(commands, source.Address(firstPage), pages); }
    std::vector<std::uint32_t> Collect() { return reader.Collect(); }
    std::vector<std::uint32_t> Read(const SharedPages& source, std::uint64_t firstPage, std::uint32_t pages = ReadPages) { return reader.Read(source.Address(firstPage), pages); }

    void RecordWrite(Commands& commands, const SharedPages& target, std::uint64_t page) {
        const auto buffer = BufferResource(target.Address(page), static_cast<std::uint32_t>(PageBytes));
        commands.Dispatch(WriteProgram, buffer, WriteLanes);
    }

private:
    MirrorReader reader;
};

std::string Where(const char* what, std::uint64_t page, std::uint32_t subpage) {
    return std::string(what) + ": page " + std::to_string(page) + " subpage " + std::to_string(subpage);
}

void ExpectLanes(const std::vector<std::uint32_t>& lanes, const Model& model, std::uint64_t firstPage, std::uint32_t pages, const std::map<std::uint32_t, std::uint32_t>& overrides, const char* what) {
    for (std::uint32_t lane = 0; lane < ReadLanes; ++lane) {
        const auto page = firstPage + lane / SubpagesPerPage;
        const auto subpage = lane % SubpagesPerPage;
        auto expected = lane / SubpagesPerPage < pages ? model.Value(page, subpage * WordsPerSubpage) : 0u;
        if (const auto found = overrides.find(lane); found != overrides.end()) expected = found->second;
        if (lanes[lane] == expected) continue;
        Expect(false, Where(what, page, subpage) + " read " + Hex(lanes[lane]) + ", expected " + Hex(expected));
        return;
    }
}

ResidencyCounts Counts() {
    ResidencyCounts counts;
    Locked([&] { counts = Residency().Manager().Counts(); });
    return counts;
}

bool Resident(const SharedPages& pages, std::uint64_t page) {
    bool resident = false;
    Locked([&] { resident = Residency().Manager().Slot(PageOf(pages.Address(page))).has_value(); });
    return resident;
}

bool Guarded(const SharedPages& pages, std::uint64_t page) {
    return AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address(page)));
}

bool HotResident(const SharedPages& pages, std::uint64_t page) {
    bool hot = false;
    Locked([&] { hot = Residency().Manager().HotResident(PageOf(pages.Address(page))); });
    return hot;
}

std::uint16_t Dirty(const SharedPages& pages, std::uint64_t page) {
    std::uint16_t dirty = 0;
    Locked([&] { dirty = Residency().Manager().Dirty(PageOf(pages.Address(page))); });
    return dirty;
}

void Use(const SharedPages& pages, std::uint64_t first, std::uint64_t count) {
    Locked([&] { Residency().Manager().Use(pages.Address(first), count * PageBytes, ResidencyAccess::Read); });
}

void WarmUp(Harness& harness) {
    SharedPages pages(ReadPages);
    Model model(pages, 9);
    model.Fill();
    harness.Read(pages, 0);
    const auto lanes = harness.Read(pages, 0);
    ExpectLanes(lanes, model, 0, ReadPages, {}, "warm-up read of guest memory");
}

void Admission(Harness& harness, Model& model) {
    auto& pages = model.Pages();
    bool full = false;
    Locked([&] { full = Residency().Manager().FullResidency(); });
    Expect(full, "admission: a budget above the guest size is not full residency");
    harness.Read(pages, 0);
    Settle();
    for (std::uint64_t page = 0; page < SectionPages; ++page) {
        Expect(Resident(pages, page), "admission: page " + std::to_string(page) + " of a shared view mapped before a dispatch has no mirror under full residency");
        Expect(Guarded(pages, page), "admission: CPU-writable page " + std::to_string(page) + " is resident without a write guard");
    }
    FillMirror(pages.Address(1, 2), MirrorMarker);
    const auto read = Counts();
    const auto lanes = harness.Read(pages, 0);
    ExpectLanes(lanes, model, 0, ReadPages, {{1 * SubpagesPerPage + 2, MirrorMarker}}, "a dispatch reading a resident page through the page table did not read its mirror");
    Expect(pages.Words(1)[2 * WordsPerSubpage] == model.Value(1, 2 * WordsPerSubpage), "admission: a change made only to the mirror reached guest memory");
    Expect(Counts().admitted == read.admitted && Counts().writtenBackBytes == read.writtenBackBytes, "admission: a read of resident pages admitted or wrote back pages");
}

void DemotionAndVersions(Harness& harness, Model& model) {
    auto& pages = model.Pages();
    const auto before = Counts();
    Commands commands;
    harness.RecordRead(commands, pages, 0);
    harness.RecordWrite(commands, pages, 1);
    harness.RecordWrite(commands, pages, 2);
    commands.Submit(0);
    const auto lanes = harness.Collect();
    ExpectLanes(lanes, model, 0, ReadPages, {{1 * SubpagesPerPage + 2, MirrorMarker}}, "a read recorded before a demotion in the same submission did not keep the page table version with the mirror");
    for (const std::uint64_t page : {1u, 2u}) {
        for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) {
            const auto stored = pages.Words(page)[subpage * WordsPerSubpage];
            Expect(stored == WrittenBase + subpage, Where("a store to a page that was resident before its dispatch", page, subpage) + " left " + Hex(stored) + ", expected " + Hex(WrittenBase + subpage));
            model.Set(page, subpage * WordsPerSubpage, WrittenBase + subpage);
        }
    }
    const auto after = Counts();
    Expect(after.demoted == before.demoted, "bounded stores: two dispatches that store through bounded ranges into two shared pages demoted " + std::to_string(after.demoted - before.demoted) + " mirrors");
    for (const std::uint64_t page : {1u, 2u}) {
        Expect(Resident(pages, page) && Dirty(pages, page) == 0, "bounded stores: page " + std::to_string(page) + " left residency or stayed dirty after the CPU read its stores");
    }
    for (const std::uint64_t page : {0u, 3u}) {
        Expect(Resident(pages, page) && Guarded(pages, page), "bounded stores: page " + std::to_string(page) + ", outside the bounded store ranges, left residency or its write guard");
    }
    const auto again = harness.Read(pages, 0);
    ExpectLanes(again, model, 0, ReadPages, {}, "bounded stores: a dispatch after the stores did not read them");
}

void GuardFaultAndRefresh(Harness& harness, Model& model) {
    auto& pages = model.Pages();
    Use(pages, 5, 1);
    Expect(Resident(pages, 5) && Guarded(pages, 5), "write guard: page 5 was not admitted with a guard");
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "write guard: the read before the store");
    const auto before = Counts();
    Locked([&] {
        model.Store(5, 3 * WordsPerSubpage, StoredByCpu);
        AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(5), PageBytes);
    });
    Expect(!Guarded(pages, 5), "write guard: a CPU store to a guarded resident page did not take the guard's fault");
    const auto lanes = harness.Read(pages, 0);
    ExpectLanes(lanes, model, 0, ReadPages, {}, "write guard: the dispatch after a CPU store to a resident page read a stale mirror");
    const auto after = Counts();
    Expect(after.faults == before.faults + 1 && after.refreshed >= before.refreshed + 1, "write guard: the ordering point applied " + std::to_string(after.faults - before.faults) + " faults and " + std::to_string(after.refreshed - before.refreshed) + " refreshes, expected one of each");
    Expect(Resident(pages, 5) && Guarded(pages, 5), "write guard: the refreshed page did not stay resident with its guard re-armed");
}

void HotRule(Harness& harness, Model& model) {
    auto& pages = model.Pages();
    Use(pages, 6, 1);
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: the read after admission");
    Expect(Resident(pages, 6), "hot rule: page 6 was not admitted");
    std::uint32_t stores = 0;
    for (; stores < 8 && !HotResident(pages, 6); ++stores) {
        model.Store(6, 4 * WordsPerSubpage, HotStore + stores);
        ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: a dispatch after a CPU store to page 6");
    }
    Expect(Resident(pages, 6) && HotResident(pages, 6), "hot rule: a page the CPU stored to before each of " + std::to_string(stores) + " submissions left residency or did not turn hot");
    Expect(!Guarded(pages, 6), "hot rule: a hot page kept its guard");
    const auto faults = Counts().faults;
    model.Store(6, 5 * WordsPerSubpage, HotStore + 0x80u);
    Expect(Counts().faults == faults, "hot rule: a store to a hot page faulted");
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: the read of a hot page after an unguarded CPU store");
    const auto cooled = Counts().cooled;
    bool cool = false;
    for (std::uint32_t quiet = 0; quiet < 6 && !cool; ++quiet) {
        harness.Read(pages, 0);
        cool = !HotResident(pages, 6);
    }
    Expect(cool && Counts().cooled > cooled && Resident(pages, 6) && Guarded(pages, 6), "hot rule: a hot page the CPU stopped writing was not guarded again within six quiet submissions");
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: the read after the page cooled");
}

void WriteBackOnCpuRead(Harness& harness, Model& model) {
    auto& pages = model.Pages();
    Use(pages, 7, 1);
    harness.Read(pages, 0);
    FillMirror(pages.Address(7, 3), WrittenBack);
    Locked([&] { Residency().Manager().NoteGpuWritten(pages.Address(7, 3), SubpageBytes); });
    Settle();
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(reinterpret_cast<const void*>(pages.Address(7, 3)), &memory, sizeof(memory)) == sizeof(memory), "write-back: VirtualQuery failed");
    Expect((memory.Protect & 0xffu) == PAGE_NOACCESS && Dirty(pages, 7) == 0x0008, "write-back: the dirty subpage is not guarded against CPU access before a CPU read (protection " + Hex(memory.Protect) + ", dirty " + Hex(Dirty(pages, 7)) + ")");
    const auto before = Counts();
    std::uint32_t value = 0;
    AgcDriver::GuestMemory::Read(pages.Address(7, 3, 5), std::as_writable_bytes(std::span(&value, 1)), 4);
    Expect(value == WrittenBack, "write-back: a driver CPU read of a dirty mirror page read " + Hex(value) + ", expected the GPU's " + Hex(WrittenBack));
    for (std::uint32_t word = 0; word < WordsPerSubpage; ++word) {
        if (pages.Words(7)[3 * WordsPerSubpage + word] == WrittenBack) continue;
        Expect(false, "write-back: word " + std::to_string(word) + " of the dirty subpage was not written back");
        break;
    }
    Expect(pages.Words(7)[2 * WordsPerSubpage] == model.Value(7, 2 * WordsPerSubpage) && pages.Words(7)[4 * WordsPerSubpage] == model.Value(7, 4 * WordsPerSubpage), "write-back: a clean subpage of the page was overwritten");
    const auto after = Counts();
    Expect(after.writtenBackBytes == before.writtenBackBytes + SubpageBytes, "write-back: " + std::to_string(after.writtenBackBytes - before.writtenBackBytes) + " bytes written back, expected one subpage");
    std::uint16_t dirty = 1;
    Locked([&] { dirty = Residency().Manager().Dirty(PageOf(pages.Address(7))); });
    Expect(dirty == 0, "write-back: the page stayed dirty after its write-back");
    for (std::uint32_t word = 0; word < WordsPerSubpage; ++word) model.Set(7, 3 * WordsPerSubpage + word, WrittenBack);
}

void DropOnUnmap(Harness& harness) {
    SharedPages pages(2);
    Model model(pages, 2);
    model.Fill();
    Use(pages, 0, 1);
    Expect(Resident(pages, 0), "drop: the page was not admitted");
    FillMirror(pages.Address(0, 4), DroppedMarker);
    ExpectLanes(harness.Read(pages, 0, 2), model, 0, 2, {{4, DroppedMarker}}, "drop: the read through the mirror before the unmap");
    const auto before = Counts();
    pages.Unmap();
    pages.Map();
    Locked([] { Residency().Manager().Apply(); });
    Expect(Counts().dropped >= before.dropped + 1, "drop: the unmap dropped no resident page");
    model.Retag(3);
    ExpectLanes(harness.Read(pages, 0, 2), model, 0, 2, {}, "drop: a dispatch after an unmap and a new mapping at the same address did not read the new memory");
}

void EvictionUnderBudget(Harness& harness) {
    Settle();
    std::uint32_t chunkSlots = 0;
    Locked([&] {
        auto& manager = Residency().Manager();
        manager.SetBudget(0);
        chunkSlots = manager.ChunkSlots();
    });
    Settle();
    const auto cleared = Counts();
    Expect(cleared.resident == 0 && cleared.chunks == 0, "eviction: a zero budget left " + std::to_string(cleared.resident) + " pages resident in " + std::to_string(cleared.chunks) + " chunks");
    SharedPages pages(chunkSlots + ReadPages);
    Model model(pages, 4);
    model.Fill();
    Locked([&] { Residency().Manager().SetBudget(std::uint64_t{chunkSlots} * PageBytes); });
    Use(pages, 0, ReadPages);
    FillMirror(pages.Address(1, 2), EvictedMarker);
    Settle();
    Use(pages, ReadPages, chunkSlots - ReadPages);
    Settle();
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {{1 * SubpagesPerPage + 2, EvictedMarker}}, "eviction: the read through the mirror before the budget was reached");
    Settle();
    const auto before = Counts();
    Use(pages, chunkSlots, ReadPages);
    const auto after = Counts();
    Expect(after.evicted == before.evicted + ReadPages, "eviction: admitting " + std::to_string(ReadPages) + " pages past a one-chunk budget evicted " + std::to_string(after.evicted - before.evicted));
    for (std::uint64_t page = 0; page < ReadPages; ++page) {
        Expect(Resident(pages, page), "eviction: page " + std::to_string(page) + ", read by the most recent dispatch, was evicted before less recently used pages");
        Expect(!Resident(pages, ReadPages + page), "eviction: least recent page " + std::to_string(ReadPages + page) + " stayed resident");
        Expect(Resident(pages, chunkSlots + page), "eviction: newly used page " + std::to_string(chunkSlots + page) + " was not admitted");
    }
    Expect(Resident(pages, 2 * ReadPages), "eviction: a more recently used page was evicted");
    ExpectLanes(harness.Read(pages, ReadPages), model, ReadPages, ReadPages, {}, "eviction: a dispatch did not read the guest memory of evicted pages");
    FillMirror(pages.Address(chunkSlots + 1, 2), AdmittedMarker);
    ExpectLanes(harness.Read(pages, chunkSlots), model, chunkSlots, ReadPages, {{1 * SubpagesPerPage + 2, AdmittedMarker}}, "eviction: a dispatch did not read the mirror of a page admitted in an evicted page's slot");
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
        BuildPrograms();
        SharedPages section(SectionPages);
        Model model(section, 1);
        model.Fill();
        Harness harness;
        WarmUp(harness);
        Locked([] { Residency().Manager().SetBudget(TestBudget); });
        Admission(harness, model);
        DemotionAndVersions(harness, model);
        GuardFaultAndRefresh(harness, model);
        HotRule(harness, model);
        WriteBackOnCpuRead(harness, model);
        DropOnUnmap(harness);
        EvictionUnderBudget(harness);
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d new path residency checks failed\n", failures);
        return 1;
    }
    std::printf("new path residency tests passed\n");
    return 0;
#else
    std::printf("skipped, commit imports and write guards are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
