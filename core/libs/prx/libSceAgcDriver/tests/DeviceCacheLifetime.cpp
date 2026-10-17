#include "GraphicsTests.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using namespace AgcDriver::Graphics;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

VKAPI_ATTR void VKAPI_CALL withD16S8(VkPhysicalDevice, VkFormat, VkFormatProperties* properties) {
    *properties = {};
    properties->optimalTilingFeatures = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
}

VKAPI_ATTR void VKAPI_CALL onlyD32S8(VkPhysicalDevice, VkFormat format, VkFormatProperties* properties) {
    *properties = {};
    if (format == VK_FORMAT_D32_SFLOAT_S8_UINT) properties->optimalTilingFeatures = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
}

constexpr std::uint64_t HeapBytes = 24ull << 30u;
constexpr std::uint64_t HeapUsage = 1ull << 30u;
constexpr std::uint64_t SmallBudget = 8ull << 30u;
constexpr std::uint64_t LargeBudget = 16ull << 30u;

void report(VkPhysicalDeviceMemoryProperties2* properties, std::uint64_t budget) {
    auto* reported = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
    for (std::uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        reported->heapBudget[heap] = budget;
        reported->heapUsage[heap] = HeapUsage;
    }
}

VKAPI_ATTR void VKAPI_CALL reportSmall(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2* properties) {
    report(properties, SmallBudget);
}

VKAPI_ATTR void VKAPI_CALL reportLarge(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2* properties) {
    report(properties, LargeBudget);
}

int deviceTag = 0;
int physicalTag = 0;

Context fakeDevice(PFN_vkGetPhysicalDeviceFormatProperties formats, PFN_vkGetPhysicalDeviceMemoryProperties2 budget) {
    Context context{};
    context.device = reinterpret_cast<VkDevice>(&deviceTag);
    context.physical = reinterpret_cast<VkPhysicalDevice>(&physicalTag);
    context.formatProperties = formats;
    context.memoryProperties2 = budget;
    context.memory.memoryTypeCount = 1;
    context.memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    context.memory.memoryTypes[0].heapIndex = 0;
    context.memory.memoryHeapCount = 1;
    context.memory.memoryHeaps[0].size = HeapBytes;
    context.memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    return context;
}

std::uint64_t expectedSampled(const Context& context, std::uint64_t budget) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &reported};
    report(&properties, budget);
    return SampledTextureBudget(context.memory, &reported, SampledTextureMemory() + TextureCacheUsage().storageBytes);
}

std::uint64_t expectedStorage(const Context& context, std::uint64_t budget) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &reported};
    report(&properties, budget);
    return StorageTextureBudget(context.memory, &reported);
}

void depthFormatTests() {
    const auto first = fakeDevice(onlyD32S8, reportSmall);
    Require(HostDepthFormat(first, VK_FORMAT_D16_UNORM_S8_UINT) == VK_FORMAT_D32_SFLOAT_S8_UINT, "a device with only D32_SFLOAT_S8_UINT must store D16_UNORM_S8_UINT as it");
    ClearDepthSurfaces(first.device);
    const auto second = fakeDevice(withD16S8, reportLarge);
    Require(HostDepthFormat(second, VK_FORMAT_D16_UNORM_S8_UINT) == VK_FORMAT_D16_UNORM_S8_UINT, "a device created with the handles of a destroyed device was given the destroyed device's depth format for D16_UNORM_S8_UINT");
    ClearDepthSurfaces(second.device);
}

void budgetTests() {
    const auto first = fakeDevice(onlyD32S8, reportSmall);
    Require(expectedSampled(first, SmallBudget) != expectedSampled(first, LargeBudget) && expectedStorage(first, SmallBudget) != expectedStorage(first, LargeBudget), "the two reported budgets must give different cache budgets");
    Require(SampledTextureCacheBudget(first) == expectedSampled(first, SmallBudget), "the sampled texture cache budget does not follow the device's reported heap budget");
    Require(StorageTextureCacheBudget(first) == expectedStorage(first, SmallBudget), "the storage image cache budget does not follow the device's reported heap budget");
    ClearCachedTextures(first.device);
    const auto second = fakeDevice(withD16S8, reportLarge);
    Require(StorageTextureCacheBudget(second) == expectedStorage(second, LargeBudget), "a device created with the handles of a destroyed device was given the destroyed device's storage image cache budget");
    Require(SampledTextureCacheBudget(second) == expectedSampled(second, LargeBudget), "a device created with the handles of a destroyed device was given the destroyed device's sampled texture cache budget");
    ClearCachedTextures(second.device);
}

}

void RunDeviceCacheLifetimeTests() {
    depthFormatTests();
    budgetTests();
}
