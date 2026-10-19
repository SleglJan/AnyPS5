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
using AgcDriver::Graphics::ResidencyHotQuietFrames;
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

bool Hot(const SharedPages& pages, std::uint64_t page) {
    bool hot = false;
    Locked([&] { hot = Residency().Manager().Hot(PageOf(pages.Address(page))); });
    return hot;
}

std::uint16_t Dirty(const SharedPages& pages, std::uint64_t page) {
    std::uint16_t dirty = 0;
    Locked([&] { dirty = Residency().Manager().Dirty(PageOf(pages.Address(page))); });
    return dirty;
}

bool BoundAtGuestOffset(const SharedPages& pages, std::uint64_t page) {
    bool bound = false;
    Locked([&] {
        auto& residency = Residency();
        const auto at = PageOf(pages.Address(page));
        const auto mirror = residency.Manager().Translate(at);
        bound = mirror.has_value() && residency.Bound(at) && residency.ResidentAddress(at, *mirror) == residency.SparseAddress() + (at << 16u);
    });
    return bound;
}

bool Unbound(const SharedPages& pages, std::uint64_t page) {
    bool unbound = false;
    Locked([&] {
        Residency().SyncBindings();
        unbound = !Residency().Bound(PageOf(pages.Address(page)));
    });
    return unbound;
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
    for (std::uint64_t page = 0; page < ReadPages; ++page) Expect(BoundAtGuestOffset(pages, page), "sparse residency: resident page " + std::to_string(page) + " read by a dispatch is not bound at its guest offset in the sparse residency buffer");
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
    model.Store(6, 3 * WordsPerSubpage, HotStore + 0x40u);
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: a dispatch after one CPU store to page 6");
    model.Store(6, 3 * WordsPerSubpage, HotStore + 0x41u);
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: a dispatch after a second CPU store to page 6 in the same frame");
    Expect(Resident(pages, 6) && Guarded(pages, 6) && !Hot(pages, 6), "hot rule: CPU stores inside one frame moved page 6 out of its mirror or left it unguarded");
    std::uint32_t frames = 0;
    for (; frames < 8 && !Hot(pages, 6); ++frames) {
        Residency().Flip();
        model.Store(6, 4 * WordsPerSubpage, HotStore + frames);
        ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: a dispatch after a CPU store to page 6 in a new frame");
    }
    Expect(Hot(pages, 6) && frames <= 2, "hot rule: a page the CPU stored to in each of " + std::to_string(frames) + " frames did not turn hot within two frames");
    for (std::uint32_t read = 0; read < 2; ++read) ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: a read while page 6 is hot");
    Expect(!Resident(pages, 6) && !Guarded(pages, 6), "hot rule: a hot page kept its mirror or its guard");
    const auto faults = Counts().faults;
    const auto polled = Counts();
    model.Store(6, 5 * WordsPerSubpage, HotStore + 0x80u);
    Expect(Counts().faults == faults, "hot rule: a store to a hot page faulted");
    for (std::uint32_t read = 0; read < 3; ++read) ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "hot rule: the read of a hot page after an unguarded CPU store");
    const auto afterPolled = Counts();
    Expect(afterPolled.hotCollects == polled.hotCollects && afterPolled.copiedInBytes == polled.copiedInBytes && afterPolled.writtenBackBytes == polled.writtenBackBytes, "hot rule: the CPU store reached three reads of the hot page with " + std::to_string(afterPolled.hotCollects - polled.hotCollects) + " hot collects, " + std::to_string(afterPolled.copiedInBytes - polled.copiedInBytes) + " bytes copied in and " + std::to_string(afterPolled.writtenBackBytes - polled.writtenBackBytes) + " written back, expected none");
    const auto cooled = Counts().cooled;
    bool cool = false;
    std::uint32_t quiet = 0;
    for (; quiet < ResidencyHotQuietFrames + 2u && !cool; ++quiet) {
        Residency().Flip();
        harness.Read(pages, 0);
        Use(pages, 6, 1);
        cool = !Hot(pages, 6);
    }
    Expect(cool && quiet >= ResidencyHotQuietFrames && Counts().cooled > cooled && Resident(pages, 6) && Guarded(pages, 6), "hot rule: a hot page the CPU stopped writing did not return to a guarded mirror after " + std::to_string(ResidencyHotQuietFrames) + " quiet frames (" + std::to_string(quiet) + " frames, cool " + std::to_string(cool) + ")");
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
    Locked([&] { Residency().Manager().SetBudget(std::uint64_t{chunkSlots} * PageBytes); });
    SharedPages filler(chunkSlots);
    AdmitEagerly();
    const auto filled = Counts();
    Expect(filled.resident == chunkSlots, "eviction: eager admission filled " + std::to_string(filled.resident) + " of the " + std::to_string(chunkSlots) + " slots of a one-chunk budget");
    SharedPages pages(chunkSlots + ReadPages);
    Model model(pages, 4);
    model.Fill();
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
    for (std::uint64_t page = 0; page < ReadPages; ++page) Expect(Unbound(pages, ReadPages + page), "sparse residency: evicted page " + std::to_string(ReadPages + page) + " is still bound in the sparse residency buffer after a dispatch");
    FillMirror(pages.Address(chunkSlots + 1, 2), AdmittedMarker);
    ExpectLanes(harness.Read(pages, chunkSlots), model, chunkSlots, ReadPages, {{1 * SubpagesPerPage + 2, AdmittedMarker}}, "eviction: a dispatch did not read the mirror of a page admitted in an evicted page's slot");
    for (std::uint64_t page = 0; page < ReadPages; ++page) Expect(BoundAtGuestOffset(pages, chunkSlots + page), "sparse residency: page " + std::to_string(chunkSlots + page) + ", admitted in an evicted page's slot, is not bound at its guest offset");
    Settle();
    Locked([&] { Residency().Manager().SetBudget(0); });
}

}

void CopyThroughSparse(const SharedPages& pages, std::uint64_t from, std::uint64_t to, std::uint32_t subpage) {
    using namespace AgcDriver::Graphics;
    Locked([&] {
        auto& residency = Residency();
        const auto slot = residency.Manager().Slot(PageOf(pages.Address(to)));
        Require(slot.has_value(), "unbind order: the copy target page is not resident");
        const auto chunk = residency.Mirrors().Chunk(slot->chunk);
        auto* recorder = Recorder::Active();
        const std::array<Recorder::SectionObject, 1> objects{Recorder::SectionObject{chunk.get(), true}};
        recorder->DeclareSection(objects);
        const auto commands = recorder->Commands();
        const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        const auto barrier = chunk->context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        const VkBufferCopy region{pages.Address(from, subpage), (VkDeviceSize{slot->slot} << 16u) + subpage * SubpageBytes, SubpageBytes};
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        chunk->context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, residency.SparseBuffer(), chunk->buffer, 1, &region);
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        recorder->MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        recorder->Keep(chunk);
    });
}

void UnbindAfterHolders(Harness& harness) {
    Locked([&] { Residency().Manager().SetBudget(TestBudget); });
    SharedPages pages(ReadPages);
    Model model(pages, 31);
    model.Fill();
    harness.Read(pages, 0);
    Settle();
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {}, "unbind order: the read after admission");
    Expect(BoundAtGuestOffset(pages, 1) && Resident(pages, 3), "unbind order: page 1 is not bound at its guest offset or page 3 is not resident before the demotion");
    FillMirror(pages.Address(1, 2), MirrorMarker);
    AgcDriver::Graphics::ResidencyDriverCounts before{};
    Locked([&] { before = Residency().Counts(); });
    CopyThroughSparse(pages, 1, 3, 2);
    Locked([&] {
        Residency().Manager().Demote(pages.Address(1), PageBytes);
        Residency().SyncBindings();
    });
    AgcDriver::Graphics::ResidencyDriverCounts after{};
    Locked([&] { after = Residency().Counts(); });
    Expect(after.bindWaits == before.bindWaits, "unbind order: unbinding a demoted page waited on the CPU " + std::to_string(after.bindWaits - before.bindWaits) + " times");
    Expect(after.deferredUnbinds == before.deferredUnbinds + 1, "unbind order: the unbind of a page mapped by a pending batch was not deferred");
    Settle();
    Expect(Unbound(pages, 1), "unbind order: the demoted page is still bound after the batch that held its mapping finished");
    ExpectLanes(harness.Read(pages, 0), model, 0, ReadPages, {{3 * SubpagesPerPage + 2, MirrorMarker}}, "unbind order: a copy recorded through the sparse buffer before a demotion and submitted after it did not read the demoted page's mirror");
    Settle();
    Locked([&] { Residency().Manager().SetBudget(0); });
}

std::uint64_t MixedAtRecord() {
    std::uint64_t mixed = 0;
    Locked([&] { mixed = Residency().Counts().mixedAtRecord; });
    return mixed;
}

std::uint64_t ResidentIn(const SharedPages& pages, std::uint64_t count) {
    std::uint64_t resident = 0;
    for (std::uint64_t page = 0; page < count; ++page) resident += Resident(pages, page) ? 1u : 0u;
    return resident;
}

void MixedRangeCompleted(Harness& harness) {
    constexpr std::uint32_t WidePages = 160;
    Settle();
    Locked([] { Residency().SyncBindings(); });
    Locked([&] { Residency().Manager().SetBudget(TestBudget); });
    SharedPages wide(WidePages);
    Model model(wide, 6);
    model.Fill();
    Use(wide, 0, WidePages);
    Settle();
    Locked([&] { Residency().Manager().Demote(wide.Address(4), (WidePages - 4) * PageBytes); });
    Settle();
    Expect(ResidentIn(wide, WidePages) == 4, "mixed range: the setup left " + std::to_string(ResidentIn(wide, WidePages)) + " of " + std::to_string(WidePages) + " pages resident, expected 4");
    FillMirror(wide.Address(1, 2), MirrorMarker);
    const auto before = Counts();
    const auto mixed = MixedAtRecord();
    ExpectLanes(harness.Read(wide, 0, WidePages), model, 0, WidePages, {{1 * SubpagesPerPage + 2, MirrorMarker}}, "mixed range: a dispatch over a range that fits the budget did not read the mirror of its resident page");
    const auto after = Counts();
    Expect(MixedAtRecord() == mixed, "mixed range: a dispatch was recorded with " + std::to_string(MixedAtRecord() - mixed) + " V# ranges part mirrored and part import-served");
    Expect(ResidentIn(wide, WidePages) == WidePages, "mixed range: a range that fits the budget was not admitted whole: " + std::to_string(ResidentIn(wide, WidePages)) + " of " + std::to_string(WidePages) + " pages resident");
    Expect(after.mixedRanges == before.mixedRanges + 1 && after.completedRanges == before.completedRanges + 1, "mixed range: the range was not counted as mixed and completed");
    for (std::uint64_t page = 0; page < WidePages; ++page) {
        if (!BoundAtGuestOffset(wide, page)) {
            Expect(false, "mixed range: completed page " + std::to_string(page) + " is not bound at its guest offset");
            break;
        }
    }
    Settle();
    Locked([&] { Residency().Manager().SetBudget(0); });
    Settle();
}

void RangeAboveBudgetFromImport(Harness& harness) {
    Settle();
    std::uint32_t chunkSlots = 0;
    Locked([&] {
        Residency().Manager().SetBudget(0);
        chunkSlots = Residency().Manager().ChunkSlots();
    });
    Settle();
    Locked([&] { Residency().Manager().SetBudget(std::uint64_t{chunkSlots} * PageBytes); });
    const auto rangePages = chunkSlots + ReadPages;
    SharedPages big(rangePages);
    Model model(big, 7);
    model.Fill();
    Use(big, 0, 4);
    Settle();
    Expect(Resident(big, 1), "range above the budget: the setup did not admit page 1");
    FillMirror(big.Address(1, 2), MirrorMarker);
    const auto before = Counts();
    const auto mixed = MixedAtRecord();
    ExpectLanes(harness.Read(big, 0, rangePages), model, 0, rangePages, {}, "range above the budget: a dispatch did not read the range's guest memory through the import");
    const auto after = Counts();
    Expect(MixedAtRecord() == mixed, "range above the budget: a dispatch was recorded with " + std::to_string(MixedAtRecord() - mixed) + " V# ranges part mirrored and part import-served");
    Expect(ResidentIn(big, rangePages) == 0, "range above the budget: " + std::to_string(ResidentIn(big, rangePages)) + " of its pages stayed resident, expected it import-served whole");
    Expect(after.mixedRanges == before.mixedRanges + 1 && after.demotedRanges == before.demotedRanges + 1 && after.rangeConflicts == before.rangeConflicts, "range above the budget: the range was not counted as mixed and demoted without conflict");
    Settle();
    Locked([&] { Residency().Manager().SetBudget(0); });
    Settle();
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
        UnbindAfterHolders(harness);
        MixedRangeCompleted(harness);
        RangeAboveBudgetFromImport(harness);
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
