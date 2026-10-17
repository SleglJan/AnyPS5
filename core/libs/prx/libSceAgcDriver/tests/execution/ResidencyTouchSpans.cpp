#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t OutputHandle = 12;
constexpr std::uint32_t Tag = 17;
constexpr std::uint32_t ReadLanes = 4;
constexpr std::uint32_t ReadPages = ReadLanes + 1;
constexpr std::uint32_t ResultWords = 2 * ReadLanes;
constexpr std::uint32_t AllocationPages = 12;
constexpr std::uint32_t GuestChunks = 3;
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t BufferStoreDwordX2 = 0x1du;
constexpr std::uint32_t PageEndOffset = static_cast<std::uint32_t>(PageBytes) - 4u;
constexpr std::uint32_t SourceBytes = static_cast<std::uint32_t>(AllocationPages * PageBytes);
constexpr std::uint32_t SMovB32 = 0x03u;

class Request final : public AgcDriver::IFlipRequest {
public:
    explicit Request(std::atomic<int>& ready) : ready(ready) {}
    void GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>&) override { ++ready; }
    void Fail(std::exception_ptr) noexcept override {}

private:
    std::atomic<int>& ready;
};

class Output final : public AgcDriver::IVideoOutput {
public:
    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo&) override { return std::make_shared<Request>(ready); }
    void Fail(std::exception_ptr) noexcept override {}
    std::atomic<int> ready = 0;
};

constexpr std::uint32_t Sop1(std::uint32_t opcode, std::uint32_t sdst, std::uint32_t ssrc0) {
    return 0xbe800000u | (sdst << 16u) | (opcode << 8u) | ssrc0;
}

const Program& SpanReadProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop1(VMovB32, 12, InlineInteger(0))
              << Mubuf(BufferLoadDwordX2, 13, 12, 4)
              << WaitAll
              << Vop1(VReadFirstLaneB32, 8, Vgpr(13))
              << Vop1(VReadFirstLaneB32, 9, Vgpr(14))
              << Sop1(SMovB32, 10, Literal) << SourceBytes
              << Sop1(SMovB32, 11, Literal) << RawBuffer
              << Vop2(VLshlRevB32, 1, InlineInteger(16), 0)
              << Vop2(VAddNcU32, 1, Literal, 1) << PageEndOffset
              << WaitAll
              << Mubuf(BufferLoadDwordX2, 2, 1, 8)
              << WaitAll
              << Vop2(VLshlRevB32, 6, InlineInteger(3), 0)
              << Mubuf(BufferStoreDwordX2, 2, 6, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

void Flip() {
    std::array<std::uint32_t, 6> words{AgcDriver::FlipPacketHeader, OutputHandle, 0xfffffffeu, 1u, 0u, 0u};
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
}

bool AwaitFlips(const Output& output, int flips) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (output.ready.load() < flips && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return output.ready.load() >= flips;
}

struct Frame {
    PrivatePages descriptors{1};
    PrivatePages results{1};
};

void RecordSpanRead(Commands& commands, Frame& frame, std::uint64_t target) {
    const auto source = BufferResource(target, SourceBytes);
    for (std::uint32_t word = 0; word < 2u; ++word) frame.descriptors.Words()[word] = source[word];
    AgcDriver::GuestMemory::CollectWritesUncached(frame.descriptors.Address(), frame.descriptors.Bytes());
    for (std::uint32_t word = 0; word < ResultWords; ++word) frame.results.Words()[word] = Untouched;
    AgcDriver::GuestMemory::CollectWritesUncached(frame.results.Address(), frame.results.Bytes());
    const auto results = BufferResource(frame.results.Address(), ResultWords * 4u);
    const auto descriptors = BufferResource(frame.descriptors.Address(), 8u);
    const std::array<std::uint32_t, 8> user{results[0], results[1], results[2], results[3], descriptors[0], descriptors[1], descriptors[2], descriptors[3]};
    commands.Dispatch(SpanReadProgram(), user, ReadLanes);
}

void ExpectSpans(Frame& frame, const std::string& what) {
    WaitIdle();
    AgcDriver::GuestMemory::FlushGpuWrites(frame.results.Address(), ResultWords * 4u);
    for (std::uint32_t lane = 0; lane < ReadLanes; ++lane) {
        const std::array<std::uint32_t, 2> wanted{Pattern(Tag, lane, WordsPerPage - 1u), Pattern(Tag, lane + 1u, 0)};
        for (std::uint32_t half = 0; half < 2u; ++half) {
            const auto value = frame.results.Words()[2u * lane + half];
            if (value == wanted[half]) continue;
            Expect(false, what + ": lane " + std::to_string(lane) + " dword " + std::to_string(half) + " loaded " + Hex(value) + ", expected " + Hex(wanted[half]));
            return;
        }
    }
}

std::uint32_t ResidentOf(const SharedPages& pages, std::uint32_t first, std::uint32_t count) {
    std::uint32_t resident = 0;
    for (std::uint32_t page = first; page < first + count; ++page) resident += Resident(pages.Address(page)) ? 1u : 0u;
    return resident;
}

std::string ResidentList(const SharedPages& pages, std::uint32_t count) {
    std::string list;
    for (std::uint32_t page = 0; page < count; ++page) list += Resident(pages.Address(page)) ? "R" : "-";
    return list;
}

void SpanReadPagesAdmittedForTheNextFrame(MirrorReader&) {
    const std::string what = "pages reads through one run-time descriptor touched across page boundaries";
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(OutputHandle, output);
    {
        std::uint64_t chunkBytes = 0;
        Locked([&] { chunkBytes = std::uint64_t{Residency().Manager().ChunkSlots()} * PageBytes; });
        Locked([&] { Residency().Manager().SetBudget(chunkBytes); });
        SharedPages filler(static_cast<std::uint32_t>(chunkBytes / PageBytes));
        AdmitEagerly();
        SharedPages pages(static_cast<std::uint32_t>(GuestChunks * chunkBytes / PageBytes));
        Fill(pages.Words(), AllocationPages, Tag);
        AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), std::uint64_t{AllocationPages} * PageBytes);
        bool full = true;
        Locked([&] { full = Residency().Manager().FullResidency(); });
        Expect(!full, what + ": a budget of one mirror chunk is full residency, so read discovery is not what admits");
        const auto start = Residency().Counts();
        const auto startCounts = Counts();
        Frame first;
        Commands frame;
        RecordSpanRead(frame, first, pages.Address());
        frame.Submit(0);
        Flip();
        ExpectSpans(first, what + ": the reads in frame N");
        Expect(AwaitFlips(*output, 1), what + ": the flip ending frame N never completed");
        Expect(ResidentOf(pages, 0, AllocationPages) == 0u, what + ": " + std::to_string(ResidentOf(pages, 0, AllocationPages)) + " pages became resident in frame N, before any touch record was read back");
        PrivatePages other(ReadPages);
        Fill(other.Words(), ReadPages, Tag);
        AgcDriver::GuestMemory::CollectWritesUncached(other.Address(), other.Bytes());
        Frame second;
        Commands next;
        RecordSpanRead(next, second, other.Address());
        next.Submit(0);
        Flip();
        ExpectSpans(second, what + ": the reads of other memory in frame N+1");
        Expect(AwaitFlips(*output, 2), what + ": the flip ending frame N+1 never completed");
        Frame third;
        Commands again;
        RecordSpanRead(again, third, pages.Address());
        again.Submit(0);
        ExpectSpans(third, what + ": the reads in frame N+2");
        Settle();
        Locked([] { Residency().Manager().Prepare(); });
        const auto end = Residency().Counts();
        const auto endCounts = Counts();
        Expect(end.touchHarvests > start.touchHarvests && end.touchedPages >= start.touchedPages + ReadPages, what + ": " + std::to_string(end.touchHarvests - start.touchHarvests) + " touch tables read back with " + std::to_string(end.touchedPages - start.touchedPages) + " touched pages, expected at least the " + std::to_string(ReadPages) + " pages frame N read");
        Expect(ResidentOf(pages, 0, ReadPages) == ReadPages && endCounts.touchAdmitted >= startCounts.touchAdmitted + ReadPages, what + ": pages " + ResidentList(pages, AllocationPages) + " resident (R), expected the first " + std::to_string(ReadPages) + "; " + std::to_string(endCounts.touchAdmitted - startCounts.touchAdmitted) + " admitted from the touch records");
        Expect(ResidentOf(pages, ReadPages, AllocationPages - ReadPages) == 0u, what + ": " + std::to_string(ResidentOf(pages, ReadPages, AllocationPages - ReadPages)) + " pages no read touched became resident");
        Settle();
        Locked([] { Residency().Manager().SetBudget(AgcDriver::Graphics::ResidencyMeasuredBudget); });
    }
    AgcDriverUnregisterVideoOutput_nid_postfix(OutputHandle, output);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency touch spans", SpanReadPagesAdmittedForTheNextFrame);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
