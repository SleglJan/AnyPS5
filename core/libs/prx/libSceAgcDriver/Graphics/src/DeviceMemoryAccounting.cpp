#include "prx/libSceAgcDriver/Graphics/include/DeviceMemoryAccounting.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace AgcDriver::Graphics::DeviceMemoryAccounting {

namespace {

struct Allocation {
    std::uint32_t heap = 0;
    VkDeviceSize bytes = 0;
    bool mirror = false;
};

struct Device {
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    std::array<std::int32_t, VK_MAX_MEMORY_TYPES> heapOfType{};
    std::array<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps{};
    std::unordered_map<VkDeviceMemory, Allocation> live;
};

std::mutex mutex;
std::unordered_map<VkDevice, Device> devices;
std::atomic<PFN_vkGetDeviceProcAddr> realProc{nullptr};

bool HostImport(const VkMemoryAllocateInfo& info) {
    for (auto* next = static_cast<const VkBaseInStructure*>(info.pNext); next != nullptr; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT) return true;
    }
    return false;
}

VKAPI_ATTR VkResult VKAPI_CALL Allocate(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
    PFN_vkAllocateMemory allocate = nullptr;
    {
        const std::lock_guard lock(mutex);
        const auto found = devices.find(device);
        Require(found != devices.end(), "device memory accounting: vkAllocateMemory on a device that is not tracked");
        allocate = found->second.allocate;
    }
    const auto result = allocate(device, info, callbacks, memory);
    if (result != VK_SUCCESS || HostImport(*info)) return result;
    const std::lock_guard lock(mutex);
    auto& tracked = devices.at(device);
    const auto heap = info->memoryTypeIndex < VK_MAX_MEMORY_TYPES ? tracked.heapOfType[info->memoryTypeIndex] : -1;
    if (heap < 0) return result;
    tracked.live[*memory] = {static_cast<std::uint32_t>(heap), info->allocationSize, false};
    tracked.heaps[static_cast<std::uint32_t>(heap)].allocated += info->allocationSize;
    return result;
}

VKAPI_ATTR void VKAPI_CALL Free(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
    PFN_vkFreeMemory free = nullptr;
    {
        const std::lock_guard lock(mutex);
        const auto found = devices.find(device);
        Require(found != devices.end(), "device memory accounting: vkFreeMemory on a device that is not tracked");
        auto& tracked = found->second;
        free = tracked.free;
        if (const auto allocation = tracked.live.find(memory); allocation != tracked.live.end()) {
            auto& heap = tracked.heaps[allocation->second.heap];
            heap.allocated -= allocation->second.bytes;
            if (allocation->second.mirror) heap.mirrors -= allocation->second.bytes;
            tracked.live.erase(allocation);
        }
    }
    free(device, memory, callbacks);
}

PFN_vkVoidFunction VKAPI_CALL Proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkAllocateMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(Allocate);
    if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(Free);
    return realProc.load()(device, name);
}

}

PFN_vkGetDeviceProcAddr Wrap(PFN_vkGetDeviceProcAddr real) {
    Require(real != nullptr && real != Proc, "device memory accounting: no device function resolver to wrap");
    realProc.store(real);
    return Proc;
}

void Track(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties) {
    const auto real = realProc.load();
    Require(real != nullptr, "device memory accounting: a device is tracked before the resolver is wrapped");
    Device tracked;
    tracked.allocate = reinterpret_cast<PFN_vkAllocateMemory>(real(device, "vkAllocateMemory"));
    tracked.free = reinterpret_cast<PFN_vkFreeMemory>(real(device, "vkFreeMemory"));
    Require(tracked.allocate != nullptr && tracked.free != nullptr, "device memory accounting: the device has no vkAllocateMemory or vkFreeMemory");
    tracked.heapOfType.fill(-1);
    for (std::uint32_t type = 0; type < properties.memoryTypeCount && type < VK_MAX_MEMORY_TYPES; ++type) {
        const auto heap = properties.memoryTypes[type].heapIndex;
        if ((properties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0 || heap >= VK_MAX_MEMORY_HEAPS) continue;
        if ((properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        tracked.heapOfType[type] = static_cast<std::int32_t>(heap);
    }
    const std::lock_guard lock(mutex);
    devices[device] = std::move(tracked);
}

void Forget(VkDevice device) {
    const std::lock_guard lock(mutex);
    devices.erase(device);
}

void MarkMirror(VkDevice device, VkDeviceMemory memory) {
    const std::lock_guard lock(mutex);
    const auto found = devices.find(device);
    Require(found != devices.end(), "device memory accounting: a mirror on a device that is not tracked");
    const auto allocation = found->second.live.find(memory);
    Require(allocation != found->second.live.end(), "device memory accounting: a mirror outside the device-local allocations");
    if (allocation->second.mirror) return;
    allocation->second.mirror = true;
    found->second.heaps[allocation->second.heap].mirrors += allocation->second.bytes;
}

HeapUsage Usage(VkDevice device, std::uint32_t heap) {
    const std::lock_guard lock(mutex);
    const auto found = devices.find(device);
    Require(found != devices.end() && heap < VK_MAX_MEMORY_HEAPS, "device memory accounting: usage of a device that is not tracked");
    return found->second.heaps[heap];
}

}
