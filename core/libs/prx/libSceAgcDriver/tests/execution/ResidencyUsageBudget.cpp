#ifdef _WIN32

#include "ResidencyBudgetHarness.hpp"
#include <atomic>
#include <chrono>

namespace {

using namespace ResidencyBudgetTest;

constexpr std::uint64_t Injected = 3 * ChunkBytes;
constexpr std::uint64_t AboveGuest = 9 * ChunkBytes;

PFN_vkGetPhysicalDeviceMemoryProperties2 realProperties = nullptr;
std::atomic<std::uint64_t> injected{0};
std::atomic<std::uint32_t> injectedHeap{0};
std::atomic<std::uint64_t> reads{0};

VKAPI_ATTR void VKAPI_CALL InjectedProperties(VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2* properties) {
    reads.fetch_add(1);
    realProperties(physical, properties);
    for (auto* next = static_cast<VkBaseOutStructure*>(properties->pNext); next != nullptr; next = next->pNext) {
        if (next->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) continue;
        reinterpret_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(next)->heapUsage[injectedHeap.load()] += injected.load();
    }
}

void UsageBeyondAllocations(Harness& harness, const Context& context) {
    harness.Start();
    const auto base = harness.Others();
    harness.ExpectFilled(base, "before any usage outside vkAllocateMemory");

    const auto readsBefore = reads.load();
    for (int round = 0; round < 8; ++round) harness.Prepare();
    const auto idleReads = reads.load() - readsBefore;
    Expect(idleReads == 0, "admission points with no allocation change and no frame boundary read the heap usage " + std::to_string(idleReads) + " times");

    const auto query = std::chrono::steady_clock::now();
    constexpr int Queries = 1000;
    for (int round = 0; round < Queries; ++round) static_cast<void>(HeapUsage(context));
    const auto micros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - query).count() / Queries;
    std::printf("one heap usage read costs %.2f us\n", micros);

    injected.store(Injected);
    const auto evictedBefore = harness.driver.Manager().Counts().evicted;
    harness.FrameBoundary();
    const auto grown = harness.Others();
    Expect(grown >= base + Injected, "usage that grew by " + MiB(Injected) + " outside vkAllocateMemory left the other device-local bytes at " + MiB(grown) + ", from " + MiB(base));
    Expect(harness.driver.Manager().Counts().evicted > evictedBefore, "usage that grew by " + MiB(Injected) + " outside vkAllocateMemory evicted no mirror at the next admission point after a frame boundary");
    harness.ExpectWithin(grown, "after usage grew by " + MiB(Injected) + " outside vkAllocateMemory");
    harness.Fill();
    harness.ExpectFilled(harness.Others(), "with usage grown by " + MiB(Injected));

    injected.store(AboveGuest);
    harness.FrameBoundary();
    harness.Fill();
    Expect(harness.Mirrors() == 0, "usage above the guest size outside vkAllocateMemory left " + MiB(harness.Mirrors()) + " of mirrors");

    injected.store(0);
    harness.FrameBoundary();
    harness.Fill();
    harness.ExpectFilled(harness.Others(), "after the usage outside vkAllocateMemory went back");
    Expect(harness.Others() < base + ChunkBytes, "the other device-local bytes stayed at " + MiB(harness.Others()) + " after the usage went back to " + MiB(base));
}

}

int main() {
    return Run("residency usage budget", [](AgcDriver::VulkanDevice& device, const Context& context, Recorder& recorder, const GuestRegion& region) {
        realProperties = context.memoryProperties2;
        injectedHeap.store(DeviceHeap(context));
        auto inject = context;
        inject.memoryProperties2 = InjectedProperties;
        Harness harness(device, inject, recorder, region);
        UsageBeyondAllocations(harness, context);
    });
}

#else

#include <cstdio>

int main() {
    std::printf("skipped, residency is Windows-only\n");
    return 77;
}

#endif
