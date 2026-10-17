#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t SpinLanes = 64;
constexpr std::uint32_t SpinLabelWord = 1024;
constexpr std::uint32_t RecordedLabelWord = 2048;
constexpr std::uint32_t CalibrationRounds = 1u << 20u;
constexpr std::uint64_t MaxRounds = std::uint64_t{1} << 31u;
constexpr auto Busy = std::chrono::milliseconds(600);
constexpr auto CalibrationTime = std::chrono::milliseconds(50);

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

const Program& SpinProgram() {
    static const Program program = [] {
        using namespace Isa;
        constexpr std::uint32_t VLshrRevB32 = 0x16u;
        constexpr std::uint32_t VXorB32 = 0x1du;
        constexpr std::uint32_t MoveS4ToS9 = 0xbe890304u;
        constexpr std::uint32_t DecrementS9 = 0x80898109u;
        constexpr std::uint32_t CompareS9NotZero = 0xbf078009u;
        constexpr std::uint32_t BranchBackWhileNotZero = 0xbf85fff7u;
        Program built;
        built << Vop2(VAddNcU32, 1, InlineInteger(1), 0) << MoveS4ToS9
              << Vop2(VLshlRevB32, 2, InlineInteger(13), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << Vop2(VLshrRevB32, 2, InlineInteger(17), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << Vop2(VLshlRevB32, 2, InlineInteger(5), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << DecrementS9 << CompareS9NotZero << BranchBackWhileNotZero
              << Vop2(VLshlRevB32, 3, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 1, 3, 0) << EndProgram;
        return built;
    }();
    return program;
}

void Spin(Commands& commands, const PrivatePages& scratch, std::uint32_t rounds) {
    const auto buffer = BufferResource(scratch.Address(), SpinLanes * 4u);
    const std::array<std::uint32_t, 5> userData{buffer[0], buffer[1], buffer[2], buffer[3], rounds};
    commands.Dispatch(SpinProgram(), userData, SpinLanes);
}

std::uint32_t Calibrate() {
    PrivatePages scratch(1);
    for (std::uint64_t measured = 0;; measured = measured == 0u ? CalibrationRounds : measured * 4u) {
        Commands commands;
        Spin(commands, scratch, measured == 0u ? 1u : static_cast<std::uint32_t>(measured));
        const auto start = Clock::now();
        commands.Submit(0);
        WaitIdle();
        Settle();
        const auto took = Clock::now() - start;
        if (measured == 0u || (took < CalibrationTime && measured < MaxRounds / 4u)) continue;
        const auto microseconds = std::max<long long>(std::chrono::duration_cast<std::chrono::microseconds>(took).count(), 1);
        const auto rounds = measured * static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Busy).count()) / static_cast<std::uint64_t>(microseconds);
        return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(rounds, CalibrationRounds, MaxRounds));
    }
}

bool LabelPending(std::uint64_t address) {
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr && recorder->LabelPendingAt(address);
}

AgcDriver::Graphics::ResidencyCounts Counts() {
    AgcDriver::Graphics::ResidencyCounts counts;
    Locked([&] { counts = Residency().Manager().Counts(); });
    return counts;
}

std::optional<AgcDriver::Graphics::MirrorSlot> SlotOf(const SharedPages& pages, std::uint32_t page) {
    std::optional<AgcDriver::Graphics::MirrorSlot> slot;
    Locked([&] { slot = Residency().Manager().Slot(PageOf(pages.Address(page))); });
    return slot;
}

bool UsedOf(const SharedPages& pages, std::uint32_t page) {
    bool used = false;
    Locked([&] { used = Residency().Manager().Used(PageOf(pages.Address(page))); });
    return used;
}

bool SameSlot(const std::optional<AgcDriver::Graphics::MirrorSlot>& left, const std::optional<AgcDriver::Graphics::MirrorSlot>& right) {
    return left.has_value() && right.has_value() && left->chunk == right->chunk && left->slot == right->slot;
}

void ApplyWrites() {
    Locked([] { Residency().Manager().Apply(); });
}

std::uint32_t Before(std::uint32_t page, std::uint32_t subpage) {
    return 0x3c000000u | (page << 8u) | subpage;
}

std::uint32_t After(std::uint32_t page, std::uint32_t subpage) {
    return 0xc5000000u | (page << 8u) | subpage;
}

void WriteSubpages(const SharedPages& pages, std::uint32_t page, std::uint32_t (*value)(std::uint32_t, std::uint32_t)) {
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) pages.Words(page)[subpage * WordsPerSubpage] = value(page, subpage);
}

void ExpectRead(const std::vector<std::uint32_t>& lanes, std::uint32_t pageCount, const std::string& what) {
    for (std::uint32_t page = 0; page < pageCount; ++page) {
        for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) {
            const auto lane = lanes[page * SubpagesPerPage + subpage];
            if (lane == After(page, subpage)) continue;
            Expect(false, what + ": the dispatch reads " + Hex(lane) + " from page " + std::to_string(page) + " subpage " + std::to_string(subpage) + ", the CPU wrote " + Hex(After(page, subpage)));
            return;
        }
    }
}

void UsedAndUnused() {
    const std::string what = "two eagerly admitted pages, page 0 read by a dispatch and page 1 untouched, both written by the CPU";
    SharedPages pages(2);
    for (std::uint32_t page = 0; page < 2u; ++page) WriteSubpages(pages, page, Before);
    AdmitEagerly();
    Expect(SlotOf(pages, 0).has_value() && SlotOf(pages, 1).has_value(), what + ": eager admission left a page without a mirror");
    Expect(!UsedOf(pages, 0) && !UsedOf(pages, 1), what + ": a page counts as used by the GPU right after its eager admission");
    MirrorReader reader;
    const auto first = reader.Read(pages.Address(), 1);
    Settle();
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) {
        if (first[subpage] == Before(0, subpage)) continue;
        Expect(false, what + ": the first dispatch reads " + Hex(first[subpage]) + " from page 0 subpage " + std::to_string(subpage) + ", expected " + Hex(Before(0, subpage)));
        break;
    }
    Expect(UsedOf(pages, 0), what + ": page 0 does not count as used after a dispatch read it");
    Expect(!UsedOf(pages, 1), what + ": page 1 counts as used though no GPU work touched it");
    const auto usedSlot = SlotOf(pages, 0);
    const auto unusedSlot = SlotOf(pages, 1);
    Expect(usedSlot.has_value() && unusedSlot.has_value(), what + ": a page lost its mirror before the CPU wrote it");
    const auto before = Counts();
    for (std::uint32_t page = 0; page < 2u; ++page) WriteSubpages(pages, page, After);
    ApplyWrites();
    const auto after = Counts();
    Expect(SameSlot(SlotOf(pages, 0), usedSlot), what + ": the CPU write took the used page's mirror away; a page the GPU used keeps its mirror and is refreshed");
    Expect(after.refreshed > before.refreshed, what + ": the used page's mirror was not refreshed after the CPU write");
    Expect(!SlotOf(pages, 1).has_value(), what + ": the unused page keeps its mirror after the CPU's first write; it is dropped instead of refreshed");
    Expect(after.unusedDemoted - before.unusedDemoted == 1u, what + ": " + std::to_string(after.unusedDemoted - before.unusedDemoted) + " unused pages were dropped at the CPU write, expected 1");
    ExpectRead(reader.Read(pages.Address(), 2), 2, what + ", a later dispatch reading both");
}

void PendingBatch(std::uint32_t rounds) {
    const std::string what = "an eagerly admitted page no GPU work used, written by the CPU while a pending batch's page table maps its mirror";
    SharedPages pages(1);
    WriteSubpages(pages, 0, Before);
    AdmitEagerly();
    const auto slot = SlotOf(pages, 0);
    Expect(slot.has_value(), what + ": eager admission left the page without a mirror");
    Expect(!UsedOf(pages, 0), what + ": the page counts as used before any GPU work");
    PrivatePages scratch(1);
    scratch.Words()[SpinLabelWord] = 0u;
    scratch.Words()[RecordedLabelWord] = 0u;
    const auto spinLabel = scratch.Address(SpinLabelWord * 4u);
    const auto recordedLabel = scratch.Address(RecordedLabelWord * 4u);
    Commands commands;
    Spin(commands, scratch, rounds);
    commands.WriteData(spinLabel, {1u});
    commands.WriteData(recordedLabel, {1u});
    commands.Submit(0);
    bool recorded = false;
    for (const auto deadline = Clock::now() + std::chrono::seconds(10); !recorded && Clock::now() < deadline;) {
        recorded = LabelPending(recordedLabel) || scratch.Words()[RecordedLabelWord] == 1u;
        if (!recorded) std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    Expect(recorded, what + ": the pending batch was never recorded");
    ApplyWrites();
    const auto before = Counts();
    Expect(SameSlot(SlotOf(pages, 0), slot), what + ": the page lost its mirror before the CPU wrote it");
    WriteSubpages(pages, 0, After);
    ApplyWrites();
    const auto dropped = Counts();
    const bool running = scratch.Words()[SpinLabelWord] == 0u;
    Expect(running, what + ": the spin had finished when the CPU write was applied, so the drop did not meet a pending batch");
    Expect(!SlotOf(pages, 0).has_value(), what + ": the page keeps its mirror after the CPU's first write");
    Expect(dropped.unusedDemoted - before.unusedDemoted == 1u, what + ": " + std::to_string(dropped.unusedDemoted - before.unusedDemoted) + " unused pages were dropped at the CPU write, expected 1");
    Expect(dropped.retiring == before.retiring + 1u, what + ": " + std::to_string(dropped.retiring) + " mirror slots wait to retire after the drop, " + std::to_string(before.retiring) + " before it; the dropped mirror was freed while the batch that maps it is pending");
    Expect(dropped.free == before.free, what + ": the free mirror slots went from " + std::to_string(before.free) + " to " + std::to_string(dropped.free) + " while the batch that maps the dropped mirror is pending");
    WaitIdle();
    Settle();
    const auto retired = Counts();
    Expect(retired.retiring == before.retiring, what + ": " + std::to_string(retired.retiring) + " mirror slots still wait to retire after the batch completed, " + std::to_string(before.retiring) + " before the drop");
    MirrorReader reader;
    const auto lanes = reader.Read(pages.Address(), 1);
    ExpectRead(lanes, 1, what + ", a later dispatch reading it");
}

}

#endif

int main() {
#ifdef _WIN32
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    const std::array<std::pair<const char*, void (*)()>, 2> cases{{
        {"used and unused eager pages written by the CPU", [] { UsedAndUnused(); }},
        {"unused eager page dropped under a pending batch", [] { PendingBatch(Calibrate()); }},
    }};
    try {
        MirrorReader reader;
        PrivatePages warm(1);
        static_cast<void>(reader.Read(warm.Address(), 1));
        Settle();
        Locked([] { Residency().Manager().SetBudget(std::uint64_t{1} << 30u); });
    } catch (const std::exception& error) {
        Expect(false, std::string("warm-up: exception: ") + error.what());
    }
    for (const auto& [name, run] : cases) {
        try {
            run();
        } catch (const std::exception& error) {
            Expect(false, std::string(name) + ": exception: " + error.what());
        }
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d residency eager first write checks failed\n", failures);
        return 1;
    }
    std::printf("residency eager first write tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
