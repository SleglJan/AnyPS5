#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEMEMORYACCOUNTING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEVICEMEMORYACCOUNTING_HPP

#include <vulkan/vulkan.h>
#include <cstdint>

namespace AgcDriver::Graphics::DeviceMemoryAccounting {

struct HeapUsage {
    std::uint64_t allocated = 0;
    std::uint64_t mirrors = 0;
};

PFN_vkGetDeviceProcAddr Wrap(PFN_vkGetDeviceProcAddr real);
void Track(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties);
void Forget(VkDevice device);
void MarkMirror(VkDevice device, VkDeviceMemory memory);
HeapUsage Usage(VkDevice device, std::uint32_t heap);

}

#endif
