#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t OutputHandle = 11;
constexpr std::uint32_t Tag = 13;
constexpr std::uint32_t ReadPages = 4;
constexpr std::uint32_t AllocationPages = 12;
constexpr std::uint32_t GuestChunks = 3;
constexpr std::uint32_t Untouched = 0xdeadbeefu;

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

const Program& LoadedAddressProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop1(VMovB32, 1, InlineInteger(0))
              << Mubuf(BufferLoadDwordX2, 2, 1, 0)
              << WaitAll
              << Vop2(VLshlRevB32, 4, InlineInteger(16), 0)
              << Vop2(VAddNcU32, 2, Vgpr(4), 2)
              << Global(GlobalLoadDword, 5, 2, 0, NullSgpr)
              << WaitAll
              << Vop2(VLshlRevB32, 6, InlineInteger(2), 0)
              << Mubuf(BufferStoreDword, 5, 6, 4)
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
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (output.ready.load() < flips && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return output.ready.load() >= flips;
}

struct Frame {
    PrivatePages addresses{1};
    PrivatePages results{1};
};

void RecordLoadedRead(Commands& commands, Frame& frame, std::uint64_t target) {
    frame.addresses.Words()[0] = Low(target);
    frame.addresses.Words()[1] = High(target);
    AgcDriver::GuestMemory::CollectWritesUncached(frame.addresses.Address(), frame.addresses.Bytes());
    for (std::uint32_t lane = 0; lane < ReadPages; ++lane) frame.results.Words()[lane] = Untouched;
    AgcDriver::GuestMemory::CollectWritesUncached(frame.results.Address(), frame.results.Bytes());
    const auto addresses = BufferResource(frame.addresses.Address(), 16u);
    const auto results = BufferResource(frame.results.Address(), ReadPages * 4u);
    const std::array<std::uint32_t, 8> user{addresses[0], addresses[1], addresses[2], addresses[3], results[0], results[1], results[2], results[3]};
    commands.Dispatch(LoadedAddressProgram(), user, ReadPages);
}

void ExpectLoaded(Frame& frame, const std::string& what) {
    WaitIdle();
    AgcDriver::GuestMemory::FlushGpuWrites(frame.results.Address(), ReadPages * 4u);
    for (std::uint32_t lane = 0; lane < ReadPages; ++lane) {
        const auto value = frame.results.Words()[lane];
        const auto wanted = Pattern(Tag, lane, 0);
        if (value == wanted) continue;
        Expect(false, what + ": lane " + std::to_string(lane) + " loaded " + Hex(value) + ", expected " + Hex(wanted));
        return;
    }
}

std::uint32_t ResidentOf(const SharedPages& pages, std::uint32_t first, std::uint32_t count) {
    std::uint32_t resident = 0;
    for (std::uint32_t page = first; page < first + count; ++page) resident += Resident(pages.Address(page)) ? 1u : 0u;
    return resident;
}

void TouchedPagesAdmittedForTheNextFrame(MirrorReader&) {
    const std::string what = "pages an undeclared read touched in one frame";
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(OutputHandle, output);
    {
        std::uint64_t chunkBytes = 0;
        Locked([&] { chunkBytes = std::uint64_t{Residency().Manager().ChunkSlots()} * PageBytes; });
        SharedPages pages(static_cast<std::uint32_t>(GuestChunks * chunkBytes / PageBytes));
        Fill(pages.Words(), AllocationPages, Tag);
        AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), std::uint64_t{AllocationPages} * PageBytes);
        bool full = true;
        Locked([&] {
            Residency().Manager().SetBudget(chunkBytes);
            full = Residency().Manager().FullResidency();
        });
        Expect(!full, what + ": a budget of one mirror chunk is full residency, so read discovery is not what admits");
        const auto start = Residency().Counts();
        const auto startCounts = Counts();
        Frame first;
        Commands frame;
        RecordLoadedRead(frame, first, pages.Address());
        frame.Submit(0);
        Flip();
        ExpectLoaded(first, what + ": the read in frame N");
        Expect(AwaitFlips(*output, 1), what + ": the flip ending frame N never completed");
        Expect(ResidentOf(pages, 0, AllocationPages) == 0u, what + ": " + std::to_string(ResidentOf(pages, 0, AllocationPages)) + " pages became resident in frame N, before any touch record was read back");
        PrivatePages other(ReadPages);
        Fill(other.Words(), ReadPages, Tag);
        AgcDriver::GuestMemory::CollectWritesUncached(other.Address(), other.Bytes());
        Frame second;
        Commands next;
        RecordLoadedRead(next, second, other.Address());
        next.Submit(0);
        Flip();
        ExpectLoaded(second, what + ": the read of other memory in frame N+1");
        Expect(AwaitFlips(*output, 2), what + ": the flip ending frame N+1 never completed");
        Frame third;
        Commands again;
        RecordLoadedRead(again, third, pages.Address());
        again.Submit(0);
        ExpectLoaded(third, what + ": the read in frame N+2");
        const auto end = Residency().Counts();
        const auto endCounts = Counts();
        Expect(end.touchHarvests > start.touchHarvests && end.touchedPages >= start.touchedPages + ReadPages, what + ": " + std::to_string(end.touchHarvests - start.touchHarvests) + " touch tables read back with " + std::to_string(end.touchedPages - start.touchedPages) + " touched pages, expected the " + std::to_string(ReadPages) + " pages frame N read");
        Expect(ResidentOf(pages, 0, ReadPages) == ReadPages && endCounts.touchAdmitted >= startCounts.touchAdmitted + ReadPages, what + ": " + std::to_string(ResidentOf(pages, 0, ReadPages)) + " of the " + std::to_string(ReadPages) + " touched pages are resident, " + std::to_string(endCounts.touchAdmitted - startCounts.touchAdmitted) + " admitted from the touch records");
        Expect(ResidentOf(pages, ReadPages, AllocationPages - ReadPages) == 0u, what + ": " + std::to_string(ResidentOf(pages, ReadPages, AllocationPages - ReadPages)) + " pages no dispatch touched became resident");
        Settle();
        Locked([] { Residency().Manager().SetBudget(AgcDriver::Graphics::ResidencyMeasuredBudget); });
    }
    AgcDriverUnregisterVideoOutput_nid_postfix(OutputHandle, output);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency touch admission", TouchedPagesAdmittedForTheNextFrame);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
