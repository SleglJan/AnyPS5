#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using namespace NewPath;
using AgcDriver::Graphics::Recorder;

constexpr std::uint32_t Batches = 8;

int failures = 0;
PFN_vkGetDeviceProcAddr resolver = nullptr;
PFN_vkGetFenceStatus fenceStatus = nullptr;
std::atomic<std::uint64_t> fenceStatusCalls{0};

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

VKAPI_ATTR VkResult VKAPI_CALL CountedFenceStatus(VkDevice device, VkFence fence) {
    fenceStatusCalls.fetch_add(1, std::memory_order_relaxed);
    return fenceStatus(device, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL CountingProc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkGetFenceStatus") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&CountedFenceStatus);
    return resolver(device, name);
}

void Run(AgcDriver::VulkanDevice& device) {
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    resolver = context.deviceProc;
    fenceStatus = reinterpret_cast<PFN_vkGetFenceStatus>(resolver(context.device, "vkGetFenceStatus"));
    Require(fenceStatus != nullptr, "reap timeline: vkGetFenceStatus");
    context.deviceProc = CountingProc;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = 256;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    Require(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &bufferInfo, nullptr, &buffer) == VK_SUCCESS, "reap timeline: vkCreateBuffer");
    VkMemoryRequirements needs{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &needs);
    VkMemoryAllocateInfo memoryInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryInfo.allocationSize = needs.size;
    memoryInfo.memoryTypeIndex = context.MemoryType(needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Require(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &memoryInfo, nullptr, &memory) == VK_SUCCESS, "reap timeline: vkAllocateMemory");
    Require(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, 0) == VK_SUCCESS, "reap timeline: vkBindBufferMemory");
    const auto update = context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
    const auto waitIdle = context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
    const std::uint32_t word = 0x5a5a5a5au;

    {
        Recorder recorder(context, true);
        Require(recorder.HasTimeline(), "reap timeline: the recorder has no timeline semaphore");
        bool drained = false;
        std::uint64_t calls = 0;
        Locked([&] {
            for (std::uint32_t index = 0; index < Batches; ++index) {
                update(recorder.Commands(), buffer, 0, sizeof(word), &word);
                recorder.Submit();
            }
            Require(waitIdle(context.device) == VK_SUCCESS, "reap timeline: vkDeviceWaitIdle");
            fenceStatusCalls.store(0, std::memory_order_relaxed);
            drained = recorder.Reap();
            calls = fenceStatusCalls.load(std::memory_order_relaxed);
        });
        std::printf("reap timeline: reaping %u completed batches called vkGetFenceStatus %llu times\n", Batches, static_cast<unsigned long long>(calls));
        Expect(drained, "a reap after the device went idle left batches in flight");
        Expect(calls == 0, "a reap queried each batch's fence (" + std::to_string(calls) + " vkGetFenceStatus calls for " + std::to_string(Batches) + " batches) instead of reading the stream's timeline once");

        Locked([&] {
            update(recorder.Commands(), buffer, 0, sizeof(word), &word);
            recorder.Submit();
            Require(waitIdle(context.device) == VK_SUCCESS, "reap timeline: vkDeviceWaitIdle");
            Expect(recorder.Reap(), "a batch submitted after a reap was not retired once the device went idle");
        });
    }

    context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d reap timeline checks failed\n", failures);
        return 1;
    }
    std::puts("reap timeline tests passed");
    return 0;
}
