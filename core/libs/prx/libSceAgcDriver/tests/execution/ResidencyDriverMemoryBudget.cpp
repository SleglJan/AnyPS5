#ifdef _WIN32

#include "ResidencyBudgetHarness.hpp"

namespace {

using namespace ResidencyBudgetTest;

constexpr std::uint64_t Growth = 3 * ChunkBytes;
constexpr int MostDevices = 16;

class DriverDevices {
public:
    explicit DriverDevices(const Context& context) : physical(context.physical) {
        auto* loader = GetModuleHandleA("vulkan-1.dll");
        Require(loader != nullptr, "driver memory budget: the Vulkan loader is not loaded");
        create = reinterpret_cast<PFN_vkCreateDevice>(reinterpret_cast<void*>(GetProcAddress(loader, "vkCreateDevice")));
        destroy = reinterpret_cast<PFN_vkDestroyDevice>(reinterpret_cast<void*>(GetProcAddress(loader, "vkDestroyDevice")));
        Require(create != nullptr && destroy != nullptr, "driver memory budget: the Vulkan loader exports no vkCreateDevice or vkDestroyDevice");
    }

    ~DriverDevices() {
        for (auto device : devices) destroy(device, nullptr);
    }

    DriverDevices(const DriverDevices&) = delete;
    DriverDevices& operator=(const DriverDevices&) = delete;

    void Add() {
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = 0;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        info.queueCreateInfoCount = 1;
        info.pQueueCreateInfos = &queue;
        VkDevice device = VK_NULL_HANDLE;
        Check(create(physical, &info, nullptr, &device), "vkCreateDevice");
        devices.push_back(device);
    }

    std::size_t Count() const { return devices.size(); }

private:
    VkPhysicalDevice physical;
    PFN_vkCreateDevice create = nullptr;
    PFN_vkDestroyDevice destroy = nullptr;
    std::vector<VkDevice> devices;
};

void DriverMemoryOutsideAllocations(Harness& harness, const Context& context) {
    harness.Start();
    const auto base = harness.Others();
    harness.ExpectFilled(base, "before the other devices");
    const auto mirrors = harness.Mirrors();

    DriverDevices devices(context);
    const auto start = HeapUsage(context);
    while (devices.Count() < MostDevices && HeapUsage(context) < start + Growth) devices.Add();
    const auto grown = HeapUsage(context) - start;
    std::printf("%zu more Vulkan devices moved the device-local heap usage by %s with no vkAllocateMemory\n", devices.Count(), MiB(grown).c_str());
    Require(grown >= Growth, "more Vulkan devices did not move the device-local heap usage by " + MiB(Growth));
    Expect(harness.Mirrors() == mirrors, "creating Vulkan devices changed the mirrors");

    harness.FrameBoundary();
    harness.Prepare();
    const auto usage = HeapUsage(context);
    Expect(usage <= harness.guest.guestBytes, "device-local heap usage " + MiB(usage) + " (mirrors " + MiB(harness.Mirrors()) + ") exceeds the guest's " + MiB(harness.guest.guestBytes) + " after the other devices grew it by " + MiB(grown));
    const auto others = harness.Others();
    Expect(others + ChunkBytes >= base + grown, "the other device-local bytes are " + MiB(others) + ", short of the " + MiB(base + grown) + " the heap usage shows");
    harness.ExpectWithin(others, "with the other devices");
    harness.Fill();
    harness.ExpectFilled(harness.Others(), "refilled with the other devices");
}

}

int main() {
    return Run("residency driver memory budget", [](AgcDriver::VulkanDevice& device, const Context& context, Recorder& recorder, const GuestRegion& region) {
        Harness harness(device, context, recorder, region);
        DriverMemoryOutsideAllocations(harness, context);
    });
}

#else

#include <cstdio>

int main() {
    std::printf("skipped, residency is Windows-only\n");
    return 77;
}

#endif
