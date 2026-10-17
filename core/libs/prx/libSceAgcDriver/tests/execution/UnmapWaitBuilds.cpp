#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
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
using Graphics::Require;
using Clock = std::chrono::steady_clock;

constexpr std::size_t RangeBytes = 65536;
constexpr auto Hold = std::chrono::milliseconds(500);
constexpr auto Idle = std::chrono::milliseconds(300);
constexpr auto CpuLimit = std::chrono::milliseconds(50);
constexpr auto WakeLimit = std::chrono::milliseconds(20);
constexpr auto IdleLimit = std::chrono::milliseconds(100);

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::chrono::microseconds ThreadCpu() {
    FILETIME creation{};
    FILETIME exit{};
    FILETIME kernel{};
    FILETIME user{};
    Require(GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user) != 0, "unmap wait: GetThreadTimes failed");
    const auto ticks = [](const FILETIME& time) { return (static_cast<std::uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime; };
    return std::chrono::microseconds((ticks(kernel) + ticks(user)) / 10u);
}

long long Milliseconds(Clock::duration duration) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

class HostRange {
public:
    HostRange() {
        memory = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, RangeBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        Require(memory != nullptr, "unmap wait: cannot allocate a host range");
        std::memset(memory, 0x3c, RangeBytes);
        GuestAllocations::Mutation().Add(memory, RangeBytes, true, true, true);
    }

    ~HostRange() {
        if (registered) {
            try {
                GuestAllocations::Mutation().Remove(memory);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "FAIL: unmap wait: a range could not be removed at the end: %s\n", error.what());
            }
        }
        VirtualFree(memory, 0, MEM_RELEASE);
    }

    HostRange(const HostRange&) = delete;
    HostRange& operator=(const HostRange&) = delete;

    void* Data() const { return memory; }
    void Unregistered() { registered = false; }

private:
    std::uint8_t* memory = nullptr;
    bool registered = true;
};

struct Removal {
    std::string failure;
    Clock::time_point start;
    Clock::time_point end;
    std::chrono::microseconds cpu{0};
};

Removal Remove(HostRange& range) {
    Removal removal;
    const auto cpu = ThreadCpu();
    removal.start = Clock::now();
    try {
        GuestAllocations::Mutation().Remove(range.Data());
        range.Unregistered();
    } catch (const std::exception& error) {
        removal.failure = error.what();
    }
    removal.end = Clock::now();
    removal.cpu = ThreadCpu() - cpu;
    return removal;
}

void BuilderHoldsLease(const Graphics::Context& context) {
    HostRange range;
    std::atomic<bool> held{false};
    std::atomic<Clock::rep> releasing{0};
    std::atomic<Clock::rep> released{0};
    std::string buildFailure;
    std::thread builder([&] {
        try {
            {
                Graphics::GuestBufferMemory memory(context);
                memory.AcquireRegistered();
                held.store(true);
                std::this_thread::sleep_for(Hold);
                releasing.store(Clock::now().time_since_epoch().count());
            }
            released.store(Clock::now().time_since_epoch().count());
        } catch (const std::exception& error) {
            buildFailure = error.what();
            held.store(true);
        }
    });
    while (!held.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto removal = Remove(range);
    builder.join();
    const std::string what = "a guest removal of a range a build outside the device lock leases for " + std::to_string(Hold.count()) + " ms";
    Expect(buildFailure.empty(), what + ": the build failed: " + buildFailure);
    Expect(removal.failure.empty(), what + ": the removal failed: " + removal.failure);
    const Clock::time_point releaseStart{Clock::duration(releasing.load())};
    const Clock::time_point releaseEnd{Clock::duration(released.load())};
    Expect(removal.end >= releaseStart, what + ": the removal returned " + std::to_string(Milliseconds(releaseStart - removal.end)) + " ms before the build began to release its lease");
    Expect(removal.end < releaseEnd + WakeLimit, what + ": the removal returned " + std::to_string(Milliseconds(removal.end - releaseEnd)) + " ms after the build had released its lease, expected within " + std::to_string(WakeLimit.count()) + " ms");
    Expect(removal.cpu < CpuLimit, what + ": the removing thread used " + std::to_string(removal.cpu.count() / 1000) + " ms of CPU time over a wait of " + std::to_string(Milliseconds(removal.end - removal.start)) + " ms, expected under " + std::to_string(CpuLimit.count()) + " ms");
}

void IdleCachedSpace(const Graphics::Context& context) {
    HostRange range;
    {
        Graphics::GuestBufferMemory memory(context);
        memory.AcquireRegistered();
    }
    std::this_thread::sleep_for(Idle);
    const auto removal = Remove(range);
    const std::string what = "a guest removal of a range only the cached address space pins, after " + std::to_string(Idle.count()) + " ms without driver work";
    Expect(removal.failure.empty(), what + ": the removal failed: " + removal.failure);
    Expect(removal.end - removal.start < IdleLimit, what + ": the removal took " + std::to_string(Milliseconds(removal.end - removal.start)) + " ms, expected under " + std::to_string(IdleLimit.count()) + " ms");
    Expect(removal.cpu < CpuLimit, what + ": the removing thread used " + std::to_string(removal.cpu.count() / 1000) + " ms of CPU time");
}

}

#endif

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIN_WAIT_MS", "5000");
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        if (context.hostImportAlignment == 0) {
            std::puts("unmap wait builds: host imports unavailable");
            return VulkanTestSkipped;
        }
        BuilderHoldsLease(context);
        IdleCachedSpace(context);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d unmap wait build checks failed\n", failures);
        return 1;
    }
    std::puts("unmap wait build tests passed");
    return 0;
#else
    return VulkanTestSkipped;
#endif
}
