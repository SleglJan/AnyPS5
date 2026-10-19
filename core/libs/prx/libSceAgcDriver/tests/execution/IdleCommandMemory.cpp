#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <cstdio>
#include <exception>
#include <string>
#include <vector>
#ifdef _WIN32
#include <psapi.h>
#endif

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

#ifdef _WIN32

namespace {

using namespace NewPath;
using AgcDriver::Graphics::Recorder;

constexpr std::uint64_t MiB = 1ull << 20u;
constexpr VkDeviceSize UpdateBytes = 64 * 1024;
constexpr std::uint32_t Updates = 8192;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::uint64_t PrivateBytes() {
    PROCESS_MEMORY_COUNTERS_EX process{};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process), sizeof(process));
    return process.PrivateUsage;
}

void Run(AgcDriver::VulkanDevice& device) {
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    Recorder* recorder = nullptr;
    Locked([&] { recorder = Recorder::Active(); });
    Require(recorder != nullptr, "idle command memory: the device has no recorder");

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = UpdateBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    Require(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &bufferInfo, nullptr, &buffer) == VK_SUCCESS, "idle command memory: vkCreateBuffer");
    VkMemoryRequirements needs{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &needs);
    VkMemoryAllocateInfo memoryInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryInfo.allocationSize = needs.size;
    memoryInfo.memoryTypeIndex = context.MemoryType(needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Require(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &memoryInfo, nullptr, &memory) == VK_SUCCESS, "idle command memory: vkAllocateMemory");
    Require(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, 0) == VK_SUCCESS, "idle command memory: vkBindBufferMemory");
    const std::vector<std::uint8_t> data(UpdateBytes, 0x5a);
    const auto update = context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
    const auto fill = [&](VkCommandBuffer commands) {
        for (std::uint32_t index = 0; index < Updates; ++index) update(commands, buffer, 0, UpdateBytes, data.data());
    };

    Locked([&] {
        fill(recorder->Commands());
        device.SubmitRecorded(true);
        update(recorder->Commands(), buffer, 0, 4, data.data());
        device.SubmitRecorded(true);
        device.WaitIdle();
        update(recorder->Commands(), buffer, 0, 4, data.data());
        device.SubmitRecorded(true);
        device.WaitIdle();
    });

    double grown = 0;
    Locked([&] {
        const auto before = PrivateBytes();
        fill(recorder->Commands());
        grown = static_cast<double>(PrivateBytes()) - static_cast<double>(before);
        device.SubmitRecorded(true);
        device.WaitIdle();
    });
    std::printf("idle command memory: %u MiB recorded after a completed batch of the same size grew private bytes by %.1f MiB\n", static_cast<unsigned>(UpdateBytes * Updates / MiB), grown / MiB);
    Expect(grown < static_cast<double>(UpdateBytes * Updates / 2), "a completed batch's command buffer kept its recording's memory: recording the same amount in a later batch grew private bytes by " + std::to_string(static_cast<long long>(grown / MiB)) + " MiB");

    context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

}

#endif

int main() {
#ifdef _WIN32
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d idle command memory checks failed\n", failures);
        return 1;
    }
    std::puts("idle command memory tests passed");
    return 0;
#else
    return VulkanTestSkipped;
#endif
}
