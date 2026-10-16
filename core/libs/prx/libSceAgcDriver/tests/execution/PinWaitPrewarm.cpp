#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "VulkanTestDevice.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <thread>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

#ifdef _WIN32

namespace {

namespace Graphics = AgcDriver::Graphics;
using namespace NewPath;
using Clock = std::chrono::steady_clock;

constexpr std::size_t RangeBytes = 0x10000;
constexpr const char* PinWaitMs = "1500";
constexpr auto Blocked = std::chrono::milliseconds(200);
constexpr auto Prompt = std::chrono::milliseconds(1000);

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

class ArenaRange {
public:
    ArenaRange() {
        base = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(RangeBytes, RangeBytes));
        Require(base != nullptr, "pin wait prewarm: cannot reserve guest arena memory");
        GuestArena::GuestArenaCommit_nid_postfix(base, RangeBytes, PAGE_READWRITE, 16384);
        GuestAllocations::Mutation().Add(base, RangeBytes, true, true, true);
        registered = true;
    }

    ~ArenaRange() {
        try {
            if (registered) GuestAllocations::Mutation().Remove(base);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: pin wait prewarm: a range could not be removed at the end: %s\n", error.what());
            ++failures;
        }
        GuestArena::GuestArenaReset_nid_postfix(base, RangeBytes);
        GuestArena::GuestArenaRelease_nid_postfix(base, RangeBytes);
    }

    ArenaRange(const ArenaRange&) = delete;
    ArenaRange& operator=(const ArenaRange&) = delete;

    std::string Remove() {
        try {
            GuestAllocations::Mutation().Remove(base);
            registered = false;
        } catch (const std::exception& error) {
            return error.what();
        }
        return {};
    }

private:
    std::uint8_t* base = nullptr;
    bool registered = false;
};

void BuildLocked(const Graphics::Context& context) {
    Graphics::GuestBufferMemory memory(context);
    memory.AcquireRegistered();
    memory.Upload();
    static_cast<void>(memory.PageTable(context));
    memory.Release();
}

void PrewarmDuringPinWait(const Graphics::Context& context) {
    std::optional<ArenaRange> held;
    held.emplace();
    auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
    std::string heldRemoval;
    std::thread waiter([&] { heldRemoval = held->Remove(); });
    std::this_thread::sleep_for(Blocked);
    ArenaRange cached;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        BuildLocked(context);
    }
    ArenaRange stale;
    std::atomic<bool> prewarmed{false};
    std::thread prewarm([&] {
        Graphics::GuestBufferMemory::Prewarm(context);
        prewarmed.store(true);
    });
    std::this_thread::sleep_for(Blocked);
    const auto start = Clock::now();
    const auto failure = cached.Remove();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
    lease = {};
    waiter.join();
    prewarm.join();
    const std::string what = "removing a range held only by the cached address space, while a prewarm waits for another range's pin wait to end";
    Expect(failure.empty(), what + ": the removal failed after " + std::to_string(elapsed.count()) + " ms: " + failure);
    Expect(elapsed < Prompt, what + ": the removal took " + std::to_string(elapsed.count()) + " ms");
    Expect(heldRemoval.empty(), "the range the lease held was not removed once the lease was dropped: " + heldRemoval);
    Expect(prewarmed.load(), "the prewarm did not finish after the pin waits ended");
}

}

#endif

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIN_WAIT_MS", PinWaitMs);
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        if (context.hostImportAlignment == 0 || !context.bufferDeviceAddress) {
            std::puts("pin wait prewarm: host imports or buffer device addresses unavailable");
            return VulkanTestSkipped;
        }
        PrewarmDuringPinWait(context);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d pin wait prewarm checks failed\n", failures);
        return 1;
    }
    std::puts("pin wait prewarm tests passed");
    return 0;
#else
    return VulkanTestSkipped;
#endif
}
