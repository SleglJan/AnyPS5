#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t OutputHandle = 9;
constexpr std::uint32_t AllocationPages = 12;
constexpr std::uint32_t ReadPages = 4;
constexpr std::uint32_t GuestChunks = 3;
constexpr std::uint32_t BudgetChunks = 1;

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

void ReadInOneFrameAdmittedForTheNext(MirrorReader& reader) {
    const std::string what = "pages a dispatch read in one frame";
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(OutputHandle, output);
    {
        std::uint64_t chunkBytes = 0;
        Locked([&] { chunkBytes = std::uint64_t{Residency().Manager().ChunkSlots()} * PageBytes; });
        SharedPages pages(static_cast<std::uint32_t>(GuestChunks * chunkBytes / PageBytes));
        Fill(pages.Words(), AllocationPages, 9);
        AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), std::uint64_t{AllocationPages} * PageBytes);
        const auto budget = BudgetChunks * chunkBytes;
        std::uint64_t guestBytes = 0;
        bool full = true;
        Locked([&] {
            auto& residency = Residency();
            residency.Manager().SetBudget(budget);
            guestBytes = residency.GuestBytes();
            full = residency.Manager().FullResidency();
        });
        Expect(guestBytes > budget && !full, what + ": the budget of " + std::to_string(BudgetChunks) + " mirror chunks (" + std::to_string(budget) + " bytes) is not below the process's " + std::to_string(guestBytes) + " guest bytes, so read discovery is not what admits");
        PrivatePages other(1);
        Commands frame;
        reader.Record(frame, pages.Address(), ReadPages);
        frame.Submit(0);
        Flip();
        ExpectRead(reader.Collect(), ReadPages, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(9, page, subpage * WordsPerSubpage); }, what + ": the read in frame N");
        Expect(AwaitFlips(*output, 1), what + ": the flip ending frame N never completed");
        Commands next;
        reader.Record(next, other.Address(), 1);
        next.Submit(0);
        Flip();
        reader.Collect();
        Expect(AwaitFlips(*output, 2), what + ": the flip ending frame N+1 never completed");
        std::uint32_t admitted = 0;
        for (std::uint32_t page = 0; page < ReadPages; ++page) admitted += Resident(pages.Address(page)) ? 1u : 0u;
        Expect(admitted == ReadPages, what + ": " + std::to_string(admitted) + " of the " + std::to_string(ReadPages) + " pages read in frame N were admitted for frame N+1 under a budget of " + std::to_string(BudgetChunks) + " mirror chunk(s)");
        std::uint32_t unread = 0;
        for (std::uint32_t page = ReadPages; page < AllocationPages; ++page) unread += Resident(pages.Address(page)) ? 1u : 0u;
        Expect(unread == 0u, what + ": " + std::to_string(unread) + " pages of the allocation no dispatch read became resident under a budget below the guest size");
        ExpectRead(reader.Read(pages.Address(), ReadPages), ReadPages, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(9, page, subpage * WordsPerSubpage); }, what + ": the read from the admitted pages");
        Settle();
        Locked([] { Residency().Manager().SetBudget(AgcDriver::Graphics::ResidencyMeasuredBudget); });
    }
    AgcDriverUnregisterVideoOutput_nid_postfix(OutputHandle, output);
}

void Run(MirrorReader& reader) {
    ReadInOneFrameAdmittedForTheNext(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency read discovery", Run);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
