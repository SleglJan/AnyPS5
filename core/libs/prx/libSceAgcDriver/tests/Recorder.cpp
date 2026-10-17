#include "prx/libSceAgcDriver/Execution/include/PipelineLinkOptimization.hpp"
#include "Triangle_frag_spv.h"
#include "Triangle_vert_spv.h"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DescriptorHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/BdaFeatures.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "ResidentPresent.hpp"
#include "SampleLod_spv.h"
#include "BufferUpdate_spv.h"
#include "SampleArray_spv.h"
#include "Cover_vert_spv.h"
#include "Cover_frag_spv.h"
#include "VertexRange_vert_spv.h"
#include "VertexRangeProducer_frag_spv.h"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include <SDL_loadso.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <functional>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
}

using namespace AgcDriver::Graphics;
using AgcDriver::GuestMemory::GpuMutex;

void* AllocateWatched(std::size_t bytes, std::size_t alignment) {
    if (!AgcDriver::GuestMemory::WriteWatched()) return nullptr;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, alignment);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    if (!AgcDriver::GuestMemory::Watched(reinterpret_cast<std::uint64_t>(block), bytes)) throw std::runtime_error("the watched block is not watched");
    return block;
}

void ReleaseWatched(void* block, std::size_t bytes) {
#ifdef _WIN32
    GuestArena::GuestArenaReset_nid_postfix(block, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(block, bytes);
#else
    munmap(block, bytes);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
#endif
}

// A compute-capable device with host imports (VK_EXT_external_memory_host) when the host offers
// them, as the driver creates its own; the recorder's batches need a real queue and real fences.
class Device {
public:
    Device() {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        Require(library != nullptr, "cannot load Vulkan");
        try {
            instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
            Require(instanceProc != nullptr, "missing Vulkan instance resolver");
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &application;
            Check(function<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr, &instance), "vkCreateInstance");
            std::uint32_t count = 0;
            const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
            Check(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            Require(count != 0, "no Vulkan device");
            std::vector<VkPhysicalDevice> devices(count);
            Check(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            const auto rankDeviceType = [](VkPhysicalDeviceType type) {
                switch (type) {
                    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
                    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
                    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
                    default: return 0;
                }
            };
            const auto physicalProperties = function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
            int selectedRank = -1;
            for (const auto physical : devices) {
                VkPhysicalDeviceProperties candidate{};
                physicalProperties(physical, &candidate);
                if (candidate.apiVersion < VK_API_VERSION_1_1) continue;
                const int rank = rankDeviceType(candidate.deviceType);
                if (rank <= selectedRank) continue;
                context.physical = physical;
                selectedRank = rank;
            }
            Require(context.physical != VK_NULL_HANDLE, "no Vulkan 1.1 device");
            VkPhysicalDeviceProperties selected{};
            physicalProperties(context.physical, &selected);
            std::cout << "Vulkan device: " << selected.deviceName << '\n';
            const auto extensions = function<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
            Check(extensions(context.physical, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
            std::vector<VkExtensionProperties> available(count);
            Check(extensions(context.physical, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
            const auto hasExtension = [&](const char* name) {
                for (const auto& extension : available) {
                    if (std::strcmp(extension.extensionName, name) == 0) return true;
                }
                return false;
            };
            auto bytes = AgcDriver::QueryBdaByteFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            auto address = AgcDriver::QueryBdaFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
            queues(context.physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            queues(context.physical, &count, families.data());
            std::uint32_t family = 0;
            while (family < count && (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) ++family;
            Require(family < count, "no Vulkan compute queue");
            const float priority = 1;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            VkPhysicalDeviceFeatures enabled{};
            enabled.shaderInt64 = VK_TRUE;
            graphics = (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            VkPhysicalDeviceFeatures supported{};
            function<PFN_vkGetPhysicalDeviceFeatures>("vkGetPhysicalDeviceFeatures")(context.physical, &supported);
            enabled.occlusionQueryPrecise = supported.occlusionQueryPrecise;
            context.occlusionQueryPrecise = supported.occlusionQueryPrecise == VK_TRUE;
            enabled.shaderStorageImageReadWithoutFormat = supported.shaderStorageImageReadWithoutFormat;
            enabled.shaderStorageImageWriteWithoutFormat = supported.shaderStorageImageWriteWithoutFormat;
            context.singlePassStorage = supported.shaderStorageImageReadWithoutFormat == VK_TRUE && supported.shaderStorageImageWriteWithoutFormat == VK_TRUE;
            address.pNext = &bytes;
            std::vector<const char*> extensionsEnabled{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_8BIT_STORAGE_EXTENSION_NAME};
            VkPhysicalDeviceDriverProperties driverProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
            if (hasExtension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
                extensionsEnabled.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
                VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT, &driverProperties};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &hostProperties};
                function<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(context.physical, &properties);
                context.hostImportAlignment = hostProperties.minImportedHostPointerAlignment;
            }
            if (driverProperties.driverID != VK_DRIVER_ID_NVIDIA_PROPRIETARY && hasExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) && hasExtension(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)) {
                extensionsEnabled.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
                extensionsEnabled.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
                context.dmaBufImport = true;
            }
            VkPhysicalDeviceImageViewMinLodFeaturesEXT minLod{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
            if (hasExtension(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &minLod};
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
                context.imageViewMinLod = minLod.minLod == VK_TRUE;
            }
            minLod = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
            minLod.minLod = VK_TRUE;
            if (context.imageViewMinLod) {
                extensionsEnabled.push_back(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
                minLod.pNext = address.pNext;
                address.pNext = &minLod;
            }
            VkPhysicalDeviceDynamicRenderingFeaturesKHR rendering{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
            VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamicState{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, &rendering};
            VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT library{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, &dynamicState};
            const std::array<const char*, 6> libraryExtensions{VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME};
            if (std::getenv("APS5_NO_GPL") == nullptr && std::all_of(libraryExtensions.begin(), libraryExtensions.end(), hasExtension)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &library};
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
                VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT libraryProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &libraryProperties};
                function<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(context.physical, &properties);
                context.graphicsPipelineLibrary = library.graphicsPipelineLibrary == VK_TRUE && dynamicState.extendedDynamicState == VK_TRUE && rendering.dynamicRendering == VK_TRUE && libraryProperties.graphicsPipelineLibraryFastLinking == VK_TRUE && properties.properties.limits.maxPushConstantsSize >= PipelinePushConstantBytes;
                if (context.graphicsPipelineLibrary && hasExtension(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)) {
                    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
                    VkPhysicalDeviceProperties2 driverProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
                    function<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(context.physical, &driverProperties);
                    context.pipelineLinkTimeOptimization = AgcDriver::PipelineLinkTimeOptimizationUsable(driver.driverID);
                }
            }
            rendering = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR, address.pNext, VK_TRUE};
            library = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, &dynamicState, VK_TRUE};
            dynamicState = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, &rendering, VK_TRUE};
            if (context.graphicsPipelineLibrary) {
                extensionsEnabled.insert(extensionsEnabled.end(), libraryExtensions.begin(), libraryExtensions.end());
                address.pNext = &library;
            }
            if (hasExtension(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME)) {
                extensionsEnabled.push_back(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME);
                context.pipelineCreationFeedback = true;
            }
            VkPhysicalDeviceRobustness2FeaturesEXT robustness{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
            if (hasExtension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &robustness};
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
                nullDescriptors = robustness.nullDescriptor == VK_TRUE;
            }
            robustness = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
            robustness.nullDescriptor = VK_TRUE;
            if (nullDescriptors) {
                extensionsEnabled.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
                robustness.pNext = address.pNext;
                address.pNext = &robustness;
            }
            VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT, address.pNext};
            indexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
            indexing.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
            indexing.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
            indexing.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
            indexing.descriptorBindingPartiallyBound = VK_TRUE;
            indexing.runtimeDescriptorArray = VK_TRUE;
            extensionsEnabled.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
            address.pNext = &indexing;
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &address};
            device.queueCreateInfoCount = 1;
            device.pQueueCreateInfos = &queue;
            device.enabledExtensionCount = static_cast<std::uint32_t>(extensionsEnabled.size());
            device.ppEnabledExtensionNames = extensionsEnabled.data();
            device.pEnabledFeatures = &enabled;
            Check(function<PFN_vkCreateDevice>("vkCreateDevice")(context.physical, &device, nullptr, &context.device), "vkCreateDevice");
            context.deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(context.physical, &context.memory);
            VkPhysicalDeviceProperties properties{};
            function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &properties);
            std::cout << "Vulkan device: " << properties.deviceName << '\n';
            context.limits = properties.limits;
            context.bufferDeviceAddress = true;
            context.formatProperties = function<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            context.imageFormatProperties = function<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
            context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, family, 0, &context.queue);
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = family;
            Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.pool), "vkCreateCommandPool");
            auto heapContext = context;
            heapContext.nullDescriptors = nullDescriptors;
            heap = std::make_unique<DescriptorHeap>(heapContext, DescriptorHeap::Capacities{1024u, 4096u, 1024u});
            context.descriptorHeap = heap.get();
        } catch (...) {
            release();
            throw;
        }
    }

    ~Device() { release(); }
    bool NullDescriptors() const { return nullDescriptors; }
    const Context& GetContext() const { return context; }
    bool Graphics() const { return graphics; }
    VkPhysicalDeviceProperties Properties() const {
        VkPhysicalDeviceProperties properties{};
        function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &properties);
        return properties;
    }
    void WaitQueue() const { Check(context.Function<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(context.queue), "vkQueueWaitIdle"); }

private:
    template<typename TFunction>
    TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(instanceProc(instance, name));
        Require(result != nullptr, name);
        return result;
    }

    void release() noexcept {
        heap.reset();
        if (context.pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, context.pool, nullptr);
        context.bufferPool.reset();
        if (context.device != VK_NULL_HANDLE) function<PFN_vkDestroyDevice>("vkDestroyDevice")(context.device, nullptr);
        if (instance != VK_NULL_HANDLE) function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        if (library != nullptr) SDL_UnloadObject(library);
    }

    void* library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    Context context{};
    std::unique_ptr<DescriptorHeap> heap;
    bool graphics = false;
    bool nullDescriptors = false;
};

using Kind = Recorder::ReadKind;

void readTrackingTests(const Device& device, Recorder& recorder) {
    Require(Recorder::ReadTracking(), "read tracking is off (APS5_COPY_READ_TRACKING=0 set?)");
    Require(recorder.Idle() && !recorder.PendingReadOverlaps(0x10000, 16), "a fresh recorder reports a read");
    recorder.NotePendingRead(0x10000, 0x100, Kind::DispatchElement);
    Require(recorder.Recording(), "a read note did not open a batch");
    Require(recorder.PendingReadOverlaps(0x10080, 4) && recorder.PendingReadOverlaps(0xff00, 0x101) && recorder.PendingReadOverlaps(0x100ff, 1), "an open batch's read is not seen");
    Require(!recorder.PendingReadOverlaps(0x10100, 4) && !recorder.PendingReadOverlaps(0xff00, 0x100) && !recorder.PendingReadOverlaps(0x10000, 0), "a disjoint range is reported as read");
    const auto open = recorder.DescribePendingRead(0x10000, 4);
    Require(open.has_value() && open->open && !open->signaled && open->kind == Kind::DispatchElement && open->serial == recorder.Submissions() + 1, "the open batch's read is described wrongly");
    recorder.Submit();
    Require(!recorder.Recording() && !recorder.Idle(), "the batch is not in flight after Submit");
    Require(recorder.PendingReadOverlaps(0x10000, 4, false), "the raw scan does not see the in-flight batch's read");
    // An empty batch completes at once: once its fence signaled, its read is no reader any more,
    // although the batch stays in flight until it is reaped.
    device.WaitQueue();
    Require(!recorder.PendingReadOverlaps(0x10000, 4), "a signaled batch's read still refuses");
    Require(!recorder.Idle(), "an overlap query reaped the batch");
    const auto signaled = recorder.DescribePendingRead(0x10000, 4);
    Require(signaled.has_value() && !signaled->open && signaled->signaled && signaled->serial == recorder.Submissions(), "the signaled batch's read is described wrongly");
    const auto counts = Recorder::ReadCounts();
    Require(counts.noted == 1 && counts.staleIgnored >= 1 && counts.hits[static_cast<std::size_t>(Kind::DispatchElement)] >= 3 && counts.queries >= 6, "read counters are off");
    recorder.Sync();
    Require(recorder.Idle() && !recorder.PendingReadOverlaps(0x10000, 4, false) && !recorder.DescribePendingRead(0x10000, 4).has_value(), "a read outlived its batch");
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{{0x20000, 0x21000}, {0x30000, 0x30010}, {0x40000, 0x40000}};
    recorder.NotePendingReads(ranges, Kind::AddressBased);
    Require(recorder.PendingReadOverlaps(0x20fff, 1) && recorder.PendingReadOverlaps(0x30000, 16) && !recorder.PendingReadOverlaps(0x21000, 16) && !recorder.PendingReadOverlaps(0x40000, 16), "noted ranges are not seen as reads");
    const auto batch = recorder.DescribePendingRead(0x30008, 4);
    Require(batch.has_value() && batch->kind == Kind::AddressBased && batch->open, "a noted range has the wrong reader kind");
    Require(Recorder::ReadCounts().hits[static_cast<std::size_t>(Kind::AddressBased)] >= 2, "address-based hits are not counted");
    recorder.Sync();
    Require(recorder.Idle() && !recorder.PendingReadOverlaps(0x20000, 0x1000, false), "noted ranges outlived their batch");
}

Recorder::WriteRanges expectedWriteUnion(Recorder::WriteRanges ranges) {
    std::sort(ranges.begin(), ranges.end());
    Recorder::WriteRanges result;
    for (const auto& [begin, end] : ranges) {
        if (!result.empty() && begin <= result.back().second) result.back().second = std::max(result.back().second, end);
        else result.emplace_back(begin, end);
    }
    return result;
}

void pendingWriteSnapshotTests(const Device& device, Recorder& recorder) {
    recorder.Sync();
    Recorder::WriteRanges expected;
    const auto check = [&] {
        const auto snapshot = Recorder::PendingWriteSnapshot();
        Require(snapshot != nullptr && *snapshot == expectedWriteUnion(expected), "pending write snapshot differs from the full union");
    };
    const auto note = [&](std::uint64_t address, std::size_t bytes) {
        const auto previous = Recorder::PendingWriteSnapshot();
        const auto saved = previous == nullptr ? Recorder::WriteRanges{} : *previous;
        recorder.NotePendingWrite(address, bytes);
        if (bytes != 0) expected.emplace_back(address, address + bytes);
        check();
        Require(previous == nullptr || *previous == saved, "publishing a new write modified an older reader's snapshot");
    };
    note(0x10000, 16);
    note(0x10020, 16);
    note(0x10010, 16);
    note(0x10004, 4);
    note(0x10000, 0);
    note(0, 1);
    note(UINT64_MAX - 16, 16);
    const Recorder::WriteRanges additions{{0x20020, 0x20040}, {0x20000, 0x20020}, {0x20010, 0x20030}, {0x20040, 0x20040}, {0x20050, 0x20040}};
    recorder.NotePendingWrites(additions);
    for (const auto& range : additions) if (range.second > range.first) expected.push_back(range);
    check();
    recorder.NotePendingFill(0x30000, 16, 0xA5);
    expected.emplace_back(0x30000, 0x30010);
    check();
    alignas(64) static std::array<std::byte, 64> labels{};
    const auto labelAddress = reinterpret_cast<std::uint64_t>(labels.data());
    const std::array value{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    recorder.NoteLabel(labelAddress, value, 1, 0);
    expected.emplace_back(labelAddress, labelAddress + value.size());
    check();
    recorder.Submit();
    device.WaitQueue();
    recorder.AfterCompletions(labelAddress + 8, value, 2, 0, true);
    expected.emplace_back(labelAddress + 8, labelAddress + 8 + value.size());
    check();
    note(0x40000, 16);
    bool completing = false;
    recorder.OnComplete([&] {
        const auto snapshot = Recorder::PendingWriteSnapshot();
        Require(Recorder::SnapshotOverlaps(snapshot.get(), 0x40000, 16), "a finishing batch left the snapshot before its completion");
        completing = true;
    });
    recorder.Sync();
    Require(completing && Recorder::PendingWriteSnapshot()->empty(), "completed writes remained published");
    std::mt19937_64 random(0x41505335u);
    for (int trial = 0; trial < 16; ++trial) {
        expected.clear();
        for (int step = 0; step < 64; ++step) note(0x80000 + random() % 4096, 1 + random() % 128);
        recorder.Sync();
        Require(Recorder::PendingWriteSnapshot()->empty(), "random writes survived their GPU batch");
    }
    expected.clear();
    note(0x90000, 16);
    {
        std::atomic<bool> invalid{false};
        std::atomic<unsigned> ready{0};
        std::atomic<std::uint64_t> reads{0};
        std::array<std::jthread, 2> readers;
        for (auto& reader : readers) reader = std::jthread([&](std::stop_token stop) {
            ready.fetch_add(1, std::memory_order_release);
            while (!stop.stop_requested()) {
                const auto snapshot = Recorder::PendingWriteSnapshot();
                if (snapshot == nullptr || !Recorder::SnapshotOverlaps(snapshot.get(), 0x90000, 16)) invalid.store(true, std::memory_order_relaxed);
                if (snapshot != nullptr) {
                    for (std::size_t i = 0; i < snapshot->size(); ++i) {
                        const auto [begin, end] = (*snapshot)[i];
                        if (end <= begin || (i != 0 && (*snapshot)[i - 1].second >= begin)) invalid.store(true, std::memory_order_relaxed);
                    }
                }
                reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
        while (ready.load(std::memory_order_acquire) != readers.size()) std::this_thread::yield();
        for (std::uint64_t i = 0; i < 256; ++i) note(0x91000 + i * 128, 32);
        for (auto& reader : readers) reader.request_stop();
        for (auto& reader : readers) reader.join();
        Require(reads.load() != 0 && !invalid.load(), "concurrent snapshot readers observed an invalid or incomplete union");
    }
    recorder.Sync();
    expected.clear();
    note(UINT64_MAX - 3, 8);
    note(0x50000, 16);
    recorder.Sync();
    recorder.NotePendingWrite(0x58000, 16);
    {
        Recorder other(device.GetContext());
        other.Activate();
        recorder.NotePendingWrite(0x59000, 16);
        recorder.Activate();
        recorder.NotePendingWrite(0x5A000, 16);
        Require(*Recorder::PendingWriteSnapshot() == Recorder::WriteRanges{{0x58000, 0x58010}, {0x59000, 0x59010}, {0x5A000, 0x5A010}}, "reactivation missed writes recorded while inactive");
        other.Activate();
        recorder.Sync();
        recorder.Activate();
        recorder.NotePendingWrite(0x5B000, 16);
        Require(*Recorder::PendingWriteSnapshot() == Recorder::WriteRanges{{0x5B000, 0x5B010}}, "reactivation retained writes completed while inactive");
        recorder.Sync();
        other.Activate();
        other.NotePendingWrite(0x60000, 16);
        const auto otherSnapshot = Recorder::PendingWriteSnapshot();
        recorder.Activate();
        recorder.NotePendingWrite(0x70000, 16);
        const auto ownSnapshot = Recorder::PendingWriteSnapshot();
        Require(ownSnapshot != nullptr && *ownSnapshot == Recorder::WriteRanges{{0x70000, 0x70010}}, "a recorder extended another recorder's snapshot");
        Require(*otherSnapshot == Recorder::WriteRanges{{0x60000, 0x60010}}, "a replaced owner's retained snapshot changed");
        other.Sync();
        Require(Recorder::PendingWriteSnapshot() == ownSnapshot, "an inactive recorder replaced the active snapshot");
        recorder.Sync();
    }
    Require(Recorder::PendingWriteSnapshot()->empty(), "snapshot ownership tests left pending writes");
    std::cout << "Pending write snapshot differential, lifetime, completion and ownership tests passed\n";
}

void pendingWriteSnapshotBenchmark(Recorder& recorder) {
    for (std::uint64_t distinct : {8u, 32u, 512u}) {
        for (int duplicates : {0, 8, 64}) {
            std::array<double, 5> samples{};
            for (auto& sample : samples) {
                recorder.Sync();
                static_cast<void>(recorder.Commands());
                const auto start = std::chrono::steady_clock::now();
                for (std::uint64_t i = 0; i < distinct; ++i) {
                    const auto address = 0x100000 + i * 4096;
                    recorder.NotePendingWrite(address, 256);
                    for (int repeat = 0; repeat < duplicates; ++repeat) recorder.NotePendingWrite(address, 256);
                }
                sample = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                const auto snapshot = Recorder::PendingWriteSnapshot();
                Require(snapshot != nullptr && snapshot->size() == distinct, "snapshot benchmark produced another union");
            }
            std::sort(samples.begin(), samples.end());
            std::cout << "snapshot-note benchmark: " << distinct << " distinct, " << duplicates << " repeats, median " << samples[2] << " ms\n";
        }
    }
    recorder.Sync();
}

struct TestRange {
    std::uint64_t begin;
    std::uint64_t end;
};

void writeSettledTests(const Device& device, Recorder& recorder) {
    Require(recorder.PendingWriteSettled(0x50000, 0x100), "an unwritten range is not settled");
    recorder.NotePendingWrite(0x50000, 0x100);
    Require(recorder.PendingWriteOverlaps(0x50000, 0x100) && !recorder.PendingWriteSettled(0x50000, 0x100) && recorder.PendingWriteSettled(0x50100, 0x100), "an open batch's write counts as settled");
    recorder.Submit();
    device.WaitQueue();
    Require(recorder.PendingWriteOverlaps(0x50000, 0x100) && recorder.PendingWriteSettled(0x50000, 0x100), "a signaled batch's write is not settled");
    recorder.NotePendingWrite(0x50080, 0x10);
    Require(!recorder.PendingWriteSettled(0x50000, 0x100) && recorder.PendingWriteSettled(0x50000, 0x80), "a later open write over the range still counts as settled");
    recorder.Sync();
    Require(recorder.Idle() && !recorder.PendingWriteOverlaps(0x50000, 0x100), "writes outlived their batches");
}

// Completion counting: a write-back completion (OnComplete) is pending until its batch finished;
// a completion label the GPU also stored (AfterCompletions storedOnGpu) is pending only once a CPU
// write-back overlapped it (NoteWrittenBack, once per label), one the GPU has no view of from its
// registration; both counts return to 0 when the batch finishes.
void completionCountTests(const Device& device, Recorder& recorder) {
    alignas(64) static std::uint32_t memory[64];
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const std::array<std::byte, 4> value{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    Require(recorder.Idle() && Recorder::PendingCompletionLabels() == 0 && Recorder::PendingWriteBackCompletions() == 0, "a fresh recorder reports pending completions");
    int ran = 0;
    recorder.OnComplete([&] { ++ran; });
    Require(Recorder::PendingWriteBackCompletions() == 1 && recorder.HasCompletions(), "a write-back completion is not pending");
    // APS5_COUNT_ALL_COMPLETION_LABELS=1 (or APS5_LABEL_STORE_ALWAYS=1) counts the GPU-stored
    // labels at registration, as before; the write-back then changes nothing.
    const bool countAll = std::getenv("APS5_COUNT_ALL_COMPLETION_LABELS") != nullptr || std::getenv("APS5_LABEL_STORE_ALWAYS") != nullptr;
    const std::uint64_t registered = countAll ? 2 : 0;
    recorder.AfterCompletions(base, value, 3, 0, true);
    recorder.AfterCompletions(base + 8, value, 4, 0, true);
    Require(Recorder::PendingCompletionLabels() == registered, "a GPU-stored completion label counts before any write-back");
    recorder.AfterCompletions(base + 0x40, value, 5, 0, false);
    Require(Recorder::PendingCompletionLabels() == registered + 1, "a label the GPU has no view of is not pending");
    const Recorder::LabelAccepts any = [](std::uint64_t) { return true; };
    bool behind = false;
    Require(!Recorder::LookupLabel(base + 0x40, 4, 5, any, &behind).has_value() && behind, "a not-imported completion label is not reported behind a completion");
    Require(!Recorder::LookupLabel(base, 4, 3, any, &behind).has_value() && behind == countAll, "a GPU-stored label is reported behind a completion before any write-back");
    Require(Recorder::LookupLabel(base, 4, 2, any, &behind).has_value(), "an early lookup of a completion label is refused");
    Recorder::NoteWrittenBack(base + 0x80, 0x40);
    Require(Recorder::PendingCompletionLabels() == registered + 1, "a disjoint write-back counted a label");
    Recorder::NoteWrittenBack(base, 4);
    Recorder::NoteWrittenBack(base + 2, 4);
    const std::uint64_t counted = countAll ? 3 : 2;
    Require(Recorder::PendingCompletionLabels() == counted, "a write-back over a GPU-stored label did not count it exactly once");
    Require(!Recorder::LookupLabel(base, 4, 3, any, &behind).has_value() && behind, "a written-back GPU-stored label is not reported behind a completion");
    Require(!Recorder::LookupLabel(base + 8, 4, 4, any, &behind).has_value() && behind == countAll, "an untouched GPU-stored label became behind a completion");
    recorder.Submit();
    device.WaitQueue();
    Require(Recorder::PendingCompletionLabels() == counted && Recorder::PendingWriteBackCompletions() == 1, "counts dropped before the batch finished");
    memory[0] = 7;
    memory[2] = 7;
    memory[16] = 7;
    recorder.Sync();
    Require(ran == 1 && Recorder::PendingCompletionLabels() == 0 && Recorder::PendingWriteBackCompletions() == 0 && recorder.Idle(), "counts did not return to 0 at finish");
    const bool storeAlways = std::getenv("APS5_LABEL_STORE_ALWAYS") != nullptr;
    Require(memory[0] == 1 && memory[2] == (storeAlways ? 1u : 7u) && memory[16] == 1, "completion stores ran for the wrong labels (overlapped and not-imported ones store, the untouched GPU-stored one skips)");
}

void directCompletionLabelTests(Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t size = 65536;
    struct Release {
        Recorder& recorder;
        void* mapping = nullptr;
        std::int64_t physical = -1;
        ~Release() {
            recorder.Sync();
            if (mapping != nullptr) sceKernelMunmap(mapping, size);
            if (physical >= 0) sceKernelReleaseDirectMemory(physical, size);
        }
    } release{recorder};
    Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, size, size, 0, &release.physical) == 0, "allocate completion label backing");
    Require(sceKernelMapDirectMemory(&release.mapping, size, 3, 0, release.physical, size) == 0, "map completion label backing");
    auto* memory = static_cast<std::byte*>(release.mapping);
    std::memset(memory, 7, size);
    const auto address = reinterpret_cast<std::uint64_t>(memory) + 4092;
    std::array<std::byte, 16> value;
    for (const auto bytes : {4u, 8u, 16u}) {
        value.fill(static_cast<std::byte>(bytes));
        const auto before = CollectWritesUncached(address, bytes);
        recorder.NotePendingWrite(address, bytes);
        recorder.AfterCompletions(address, std::span(value).first(bytes), 100 + bytes, 0, false);
        value.fill(std::byte{99});
        bool observed = false;
        Require(recorder.AfterRecordedWork([&] {
            observed = std::all_of(memory + 4092, memory + 4092 + bytes, [bytes](auto b) { return b == static_cast<std::byte>(bytes); });
        }), "completion observer was not recorded");
        recorder.Sync();
        Require(observed, "a following completion observed incorrect label bytes");
        Require(memory[4091] == std::byte{7} && memory[4108] == std::byte{7}, "completion label changed neighboring bytes");
        Require(recorder.Idle() && Recorder::PendingCompletionLabels() == 0, "completion label remained pending after sync");
        if (before != 0) {
            Require(StoredOver(address, bytes, before), "completion label was not stamped");
            Require(UnchangedSinceCollected(address, bytes, before), "completion label dirtied the guest page");
            const auto after = CollectWritesUncached(address, bytes);
            memory[4092] = std::byte{81};
            Require(!UnchangedSinceCollected(address, bytes, after), "completion store disabled subsequent guest tracking");
        }
    }
}

void afterRecordedWorkTests(const Device& device, Recorder& recorder) {
    alignas(64) static std::uint32_t memory[16];
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const std::array<std::byte, 4> value{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    recorder.Sync();
    std::vector<int> ran;
    std::uint32_t seen = 0;
    Require(recorder.Idle() && !recorder.AfterRecordedWork([&] { ran.push_back(0); }) && ran.empty() && Recorder::PendingCompletionLabels() == 0, "an idle recorder kept an action for recorded work");
    recorder.NotePendingWrite(0x52000, 0x100);
    Require(!Recorder::PendingLabelSince().has_value(), "a write started the label flush deadline");
    Require(recorder.AfterRecordedWork([&] { ran.push_back(1); }) && Recorder::PendingCompletionLabels() == 1 && Recorder::PendingLabelSince().has_value(), "an action behind the open batch is not pending under the label flush deadline");
    recorder.AfterCompletions(base, value, 6, 0, false);
    Require(recorder.AfterRecordedWork([&] { seen = memory[0]; ran.push_back(2); }) && Recorder::PendingCompletionLabels() == 3, "an action behind a completion label is not pending");
    recorder.Submit();
    Require(recorder.AfterRecordedWork([&] { ran.push_back(3); }) && Recorder::PendingCompletionLabels() == 4, "an action behind an in-flight batch is not pending");
    device.WaitQueue();
    Require(ran.empty(), "an action ran before its batch was reaped");
    recorder.Sync();
    Require(ran == std::vector<int>{1, 2, 3} && Recorder::PendingCompletionLabels() == 0 && recorder.Idle(), "actions behind recorded work did not run once each, in order");
    Require(seen == 1, "an action ran before the completion label recorded ahead of it");
}

void completionWaitTests(const Device& device, Recorder& recorder) {
    alignas(64) static std::uint32_t memory[16];
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    recorder.Sync();
    const std::array<std::byte, 4> gpuValue{std::byte{5}, std::byte{0}, std::byte{0}, std::byte{0}};
    bool ran = false;
    recorder.NotePendingWrite(base, sizeof(memory));
    recorder.AfterCompletions(base, gpuValue, 7, 0, true);
    Require(recorder.AfterRecordedWork([&] { ran = true; }) && Recorder::PendingCompletionLabels() != 0, "completion wait: the action behind the open batch is not pending");
    CompletePendingCpuWrites(device.GetContext(), base, sizeof(memory));
    Require(!ran && !recorder.Idle(), "completion wait: a reader of a GPU-stored label waited for a batch whose only completion action writes no memory");
    recorder.Sync();
    Require(ran && recorder.Idle(), "completion wait: the action did not run at the sync");
    const std::array<std::byte, 4> cpuValue{std::byte{9}, std::byte{0}, std::byte{0}, std::byte{0}};
    recorder.NotePendingWrite(base, sizeof(memory));
    recorder.AfterCompletions(base + 4u, cpuValue, 8, 0, false);
    CompletePendingCpuWrites(device.GetContext(), base, sizeof(memory));
    Require(memory[1] == 9u && Recorder::PendingCompletionLabels() == 0, "completion wait: a reader of a label stored behind a completion did not wait for that store");
}

void batchStampTests(Recorder& recorder) {
    Require(recorder.Idle(), "batch stamps: the recorder is busy");
    double previousEnd = 0;
    for (int round = 0; round < 3; ++round) {
        recorder.NotePendingWrite(0x60000, 0x100);
        const auto serial = recorder.Submissions() + 1;
        recorder.Submit();
        recorder.Sync();
        std::size_t missing = 0;
        const auto batches = recorder.CompletedBatches(serial - 1, serial, missing);
        Require(missing == 0 && batches.size() == 1 && batches.front().serial == serial, "a finished batch has no completion record");
        const auto& batch = batches.front();
        if (Recorder::BatchStampsEnabled()) {
            Require(batch.gpuStartNs > 0 && batch.gpuEndNs >= batch.gpuStartNs && batch.gpuStartNs >= previousEnd, "batch stamps are missing or out of order");
            previousEnd = batch.gpuEndNs;
        } else {
            Require(batch.gpuStartNs == 0 && batch.gpuEndNs == 0, "batch stamps were written with profiling off");
        }
    }
    std::cout << "batch stamps " << (Recorder::BatchStampsEnabled() ? "checked" : "off") << '\n';
}

void labelTests(Recorder& recorder) {
    const std::array<std::byte, 4> value{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    Require(!recorder.PendingLabelIn(0x60000, 0x100), "an empty table reports a label");
    // A label this worker queued but has not recorded yet is inside the range for a CPU store decision.
    Recorder::NoteQueuedLabel(0x60010, value, 1, AgcDriver::GuestMemory::GpuLockThreadTag());
    Require(recorder.PendingLabelIn(0x60010, 4) && recorder.PendingLabelIn(0x60000, 0x100) && !recorder.PendingLabelIn(0x60014, 4) && !recorder.PendingLabelIn(0x60000, 0x10), "a queued label is not seen by PendingLabelIn");
    Recorder::ForgetQueuedLabels();
    Require(!recorder.PendingLabelIn(0x60000, 0x100), "a forgotten queued label is still pending");
    recorder.NoteLabel(0x60020, value, 2, 0);
    Require(recorder.PendingLabelIn(0x60020, 4) && recorder.PendingLabelIn(0x60000, 0x100) && !recorder.PendingLabelIn(0x60000, 0x20), "a recorded label is not seen by PendingLabelIn");
    recorder.Sync();
    Require(!recorder.PendingLabelIn(0x60000, 0x100), "a recorded label outlived its batch");
}

void labelRegistryTests(Recorder& recorder) {
    using Writer = Recorder::LabelWriter;
    const Recorder::LabelAccepts any = [](std::uint64_t) { return true; };
    const auto equals = [](std::uint64_t expected) { return Recorder::LabelAccepts([expected](std::uint64_t value) { return value == expected; }); };
    {
        const std::array<std::vector<Writer>, 1> dwords{std::vector<Writer>{{1, 10, 0x20, 50, 1}, {2, 4, 0x28, 60, 2}}};
        auto hit = Recorder::SelectLabelWriter(dwords, 0, 40, equals(1));
        Require(hit.has_value() && hit->stream == 1 && hit->batch == 10 && hit->value == 1 && hit->queue == 0x20 && hit->stamp == 50, "(1) the older satisfying writer of another stream is not chosen when a newer one fails the compare");
        hit = Recorder::SelectLabelWriter(dwords, 0, 40, any);
        Require(hit.has_value() && hit->stream == 1, "(2) the earliest satisfying writer is not chosen");
        hit = Recorder::SelectLabelWriter(dwords, 2, 40, any);
        Require(hit.has_value() && hit->stream == 2 && hit->batch == 4, "(3) the caller's own stream is not preferred");
        hit = Recorder::SelectLabelWriter(dwords, 0, 50, any);
        Require(hit.has_value() && hit->stream == 2, "(4) a writer recorded before the wait was received was taken");
        Require(!Recorder::SelectLabelWriter(dwords, 0, 60, any).has_value(), "(4) writers recorded before the wait was received satisfied it");
        Require(!Recorder::SelectLabelWriter(dwords, 0, 0, equals(3)).has_value(), "(5) a writer whose value fails the compare was taken");
    }
    {
        const std::array<std::vector<Writer>, 2> dwords{std::vector<Writer>{{1, 12, 0x20, 70, 5}, {2, 3, 0x28, 65, 9}}, std::vector<Writer>{{1, 11, 0x20, 66, 7}}};
        const auto hit = Recorder::SelectLabelWriter(dwords, 0, 0, any);
        Require(hit.has_value() && hit->stream == 1 && hit->value == 0x700000005ull && hit->batch == 12 && hit->stamp == 70, "(6) a 64-bit value is not composed from one stream at its newest batch");
        Require(!Recorder::SelectLabelWriter(dwords, 0, 0, equals(0x700000009ull)).has_value(), "(6) dwords of two streams were composed into one value");
    }
    const std::array<std::byte, 4> one{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    const std::array<std::byte, 4> two{std::byte{2}, std::byte{0}, std::byte{0}, std::byte{0}};
    const std::array<std::byte, 8> pair{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}, std::byte{2}, std::byte{0}, std::byte{0}, std::byte{0}};
    const auto tag = AgcDriver::GuestMemory::GpuLockThreadTag();
    constexpr std::uint64_t a = 0x70000, b = 0x70100, c = 0x70200;
    recorder.NoteLabel(a, one, 10, 0);
    auto hit = Recorder::LookupLabel(a, 4, 5, any);
    Require(hit.has_value() && hit->value == 1 && hit->stamp == 10 && hit->batch != 0, "(7) a recorded label is not served");
    Require(!Recorder::LookupLabel(a, 4, 10, any).has_value(), "(7) a label recorded before the wait was received is served");
    Require(!Recorder::LookupLabel(a, 4, 5, equals(2)).has_value(), "(7) a label whose value fails the compare is served");
    recorder.NoteLabel(a, two, 20, 0);
    hit = Recorder::LookupLabel(a, 4, 5, any);
    Require(hit.has_value() && hit->value == 2 && hit->stamp == 20, "(7) a later label of the same stream does not replace its entry");
    Require(!Recorder::LookupLabel(a, 4, 5, equals(1)).has_value(), "(7) a replaced value of the same stream is still served");
    const std::uint32_t other = tag == 7 ? 8 : 7;
    Recorder::NoteQueuedLabel(b, one, 30, other);
    Require(!Recorder::LookupLabel(b, 4, 0, any).has_value(), "(8) another queue's queued label is served");
    Recorder::NoteQueuedLabel(b, one, 31, tag);
    Recorder::ForgetQueuedLabels();
    Recorder::NoteQueuedLabel(c, two, 40, tag);
    hit = Recorder::LookupLabel(c, 4, 35, any);
    Require(hit.has_value() && hit->value == 2 && hit->batch == 0, "(8) this queue's queued label is not served");
    Require(!Recorder::LookupLabel(c, 4, 40, any).has_value(), "(8) a queued label noted before the wait was received is served");
    Recorder::ForgetQueuedLabels();
    Require(!Recorder::LookupLabel(c, 4, 0, any).has_value(), "(8) a forgotten queued label is still served");
    Recorder::NoteQueuedLabel(a, one, 50, tag);
    hit = Recorder::LookupLabel(a, 4, 5, any);
    Require(hit.has_value() && hit->value == 1 && hit->batch == 0, "(8) this queue's queued label does not take precedence over its recorded one");
    Recorder::ForgetQueuedLabels();
    recorder.NoteLabel(b, pair, 60, 0);
    hit = Recorder::LookupLabel(b, 8, 0, equals(0x200000001ull));
    Require(hit.has_value() && hit->value == 0x200000001ull && hit->stamp == 60, "(9) a 64-bit label is not served");
    recorder.Sync();
    Require(!Recorder::LookupLabel(a, 4, 0, any).has_value() && !Recorder::LookupLabel(b, 8, 0, any).has_value() && !recorder.PendingLabelIn(0x70000, 0x400), "(10) entries outlived their batch");
}

void labelMatchTests(Recorder& recorder) {
    alignas(16) static std::array<std::uint32_t, 128> memory{};
    memory.fill(0x11111111u);
    const auto base = reinterpret_cast<std::uint64_t>(memory.data());
    std::array<std::byte, 256> wide{};
    std::memcpy(wide.data(), memory.data(), wide.size());
    const std::array<std::byte, 4> narrow{std::byte{0x11}, std::byte{0x11}, std::byte{0x11}, std::byte{0x11}};
    recorder.Sync();
    Require(Recorder::PendingLabelsMatchMemory(base, 16), "a range without labels does not match memory");
    recorder.NoteLabel(base, wide, 300, 0);
    recorder.NoteLabel(base + 0x100, narrow, 301, 0);
    Require(Recorder::WideLabelIn(base + 0x18, 8) && Recorder::LookupLabel(base + 0x100, 4, 0, [](std::uint64_t) { return true; }).has_value(), "the labels were not noted");
    Require(Recorder::PendingLabelsMatchMemory(base + 0x18, 8) && Recorder::PendingLabelsMatchMemory(base + 0xf8, 0x10), "labels that store the bytes memory holds were taken as changing them");
    memory[7] = 0x22222222u;
    Require(!Recorder::PendingLabelsMatchMemory(base + 0x18, 8) && Recorder::PendingLabelsMatchMemory(base + 0x20, 8), "a wide label over other bytes than memory holds was not found, or one beside them was");
    memory[7] = 0x11111111u;
    memory[64] = 0x33333333u;
    Require(!Recorder::PendingLabelsMatchMemory(base + 0xfc, 8) && Recorder::PendingLabelsMatchMemory(base + 0x104, 4), "a narrow label over other bytes than memory holds was not found, or one beside it was");
    recorder.Sync();
    Require(Recorder::PendingLabelsMatchMemory(base + 0xfc, 8), "labels outlived their batch");
}

void largeLabelTests(Recorder& recorder) {
    constexpr std::uint64_t slot = 0x71000, small = slot + 0x10, outside = slot - 4, wide = 0x72000;
    const std::array<std::byte, 4> one{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    std::array<std::byte, 256> clear{};
    std::array<std::byte, Recorder::LabelTableBytes> limit{};
    limit.fill(std::byte{3});
    recorder.Sync();
    Require(recorder.PendingLabels() == 0 && !Recorder::PendingLabelSince().has_value() && !Recorder::WideLabelIn(0, ~0ull), "the table is not empty before the large label tests");
    recorder.NoteLabel(small, one, 100, 0);
    recorder.NoteLabel(outside, one, 100, 0);
    Require(recorder.PendingLabels() == 2 && Recorder::LookupLabel(small, 4, 50, [](std::uint64_t) { return true; }).has_value(), "4-byte labels did not enter the table");
    recorder.NoteLabel(slot, clear, 101, 0);
    Require(recorder.PendingLabels() == 1 && Recorder::LookupLabel(outside, 4, 50, [](std::uint64_t) { return true; }).has_value(), "a label larger than the table bound entered dwords or removed one outside its range");
    Require(!Recorder::LookupLabel(small, 4, 50, [](std::uint64_t) { return true; }).has_value() && !Recorder::LookupLabel(slot, 8, 0, [](std::uint64_t) { return true; }).has_value(), "an older label under a larger one still composes the value");
    Require(Recorder::WideLabelIn(slot, 4) && Recorder::WideLabelIn(slot + clear.size() - 4, 4) && Recorder::WideLabelIn(slot - 0x100, 0x104) && !Recorder::WideLabelIn(slot - 0x100, 0x100) && !Recorder::WideLabelIn(slot + clear.size(), 4), "a large label's range is wrong");
    Require(recorder.PendingLabelIn(slot + 0x20, 0x20) && recorder.PendingWriteOverlaps(slot, clear.size()), "a large label is not pending");
    Require(Recorder::PendingLabelSince().has_value(), "a large label did not start the label flush deadline");
    recorder.NoteLabel(small, one, 102, 0);
    const auto later = Recorder::LookupLabel(small, 4, 101, [](std::uint64_t) { return true; });
    Require(later.has_value() && later->value == 1 && later->stamp == 102, "a label after a large one is not served");
    recorder.NoteLabel(wide, limit, 103, 0);
    Require(recorder.PendingLabels() == 2 + Recorder::LabelTableBytes / 4 && !Recorder::WideLabelIn(wide, limit.size()), "a label at the table bound did not enter every dword");
    const auto bound = Recorder::LookupLabel(wide + Recorder::LabelTableBytes - 8, 8, 0, [](std::uint64_t) { return true; });
    Require(bound.has_value() && bound->value == 0x0303030303030303ull && bound->stamp == 103, "a label at the table bound is not served");
    recorder.Sync();
    Require(recorder.PendingLabels() == 0 && !Recorder::WideLabelIn(0, ~0ull) && !recorder.PendingLabelIn(slot, clear.size()) && !recorder.PendingWriteOverlaps(slot, clear.size()), "large label tests left entries, ranges or writes behind");
}

// (6) GuestMemory::UnchangedSinceCollected: false outside the watched arena, and around a
// MarkWritten or a CPU write of the block inside it.
void unchangedSinceTests() {
    using namespace AgcDriver::GuestMemory;
    static std::uint32_t outside[16];
    Require(!UnchangedSinceCollected(reinterpret_cast<std::uint64_t>(outside), 4, TrackerGeneration()), "(6) a range outside the watched memory is unchanged");
    Require(!Watched(reinterpret_cast<std::uint64_t>(outside), 4) && CollectWrites(reinterpret_cast<std::uint64_t>(outside), 4) == 0, "(6) a range outside the watched memory is watched");
    constexpr std::size_t bytes = 65536;
    void* block = AllocateWatched(bytes, bytes);
    if (block == nullptr) {
        std::cout << "no write watching: UnchangedSinceCollected is always false\n";
        return;
    }
    auto* words = static_cast<volatile std::uint32_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    words[0] = 0;
    CollectWritesUncached(address, 4);
    const auto generation = TrackerGeneration();
    Require(UnchangedSinceCollected(address, 4, generation), "(6) an untouched range is not unchanged");
    Require(UnchangedSinceCollected(address, 4, generation), "(6) the check itself dirtied the range");
    Require(!UnchangedSinceCollected(address, 4, 0), "(6) generation 0 is unchanged");
    MarkWritten(address, 4);
    Require(UnchangedSinceCollected(address, 4, generation), "(6) a MarkWritten of the block (a GPU label record) refuses");
    Require(!UnchangedSince(address, 4, generation), "(6) a MarkWritten of the block is not seen by UnchangedSince");
    const auto generation2 = TrackerGeneration();
    Require(UnchangedSinceCollected(address, 4, generation2), "(6) unchanged after the stamp fails");
    words[1] = 1;
    Require(!UnchangedSinceCollected(address, 4, generation2), "(6) a CPU write to the page is not seen");
    Require(UnchangedSinceCollected(address, 4, TrackerGeneration()), "(6) the collect did not reset the page");
    const auto generation3 = TrackerGeneration();
    words[1] = 2;
    CollectWritesUncached(address, 4);
    Require(!UnchangedSinceCollected(address, 4, generation3), "(6) a CPU write collected by another caller is not seen");
    ReleaseWatched(block, bytes);
}

// (9) ProvedClearKeys: a surface's DCC keys are scanned once and answered from the proof after,
// until the key range is stamped (a GPU key store's MarkWritten) or the CPU writes it; a scan made
// while recorded work still writes the keys is not kept. With APS5_NO_KEY_FAST_PATH=1 every call
// scans and no proof is stored.
void keyProofTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t bytes = 65536;
    void* block = AllocateWatched(bytes, bytes);
    if (block == nullptr) {
        std::cout << "no write watching: key proofs not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    constexpr std::size_t keyCount = 1024;
    constexpr std::uint64_t surfaceBytes = keyCount * 256;
    auto* keys = static_cast<std::uint8_t*>(block);
    std::memset(keys, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    GuestTextureResource resource{};
    resource.baseAddress = address + 8192;
    resource.width = 256;
    resource.height = 256;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dccAddress = address;
    DccKeyProof proof;
    auto before = KeyProofCounts();
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "(9) 0x00 keys are not a 0000 clear");
    auto after = KeyProofCounts();
    Require(after.scanned == before.scanned + 1 && after.proved == before.proved, "(9) the first call did not scan");
    if (!KeyFastPath()) {
        Require(proof.generation == 0, "(9) APS5_NO_KEY_FAST_PATH stored a proof");
        Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000 && proof.generation == 0, "(9) APS5_NO_KEY_FAST_PATH proved keys");
        std::cout << "key fast path off: every call scans\n";
        return;
    }
    Require(proof.generation != 0 && proof.keys == DccKeys::Clear0000 && after.unstable == before.unstable, "(9) a stable scan left no proof");
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "(9) the proof answers other keys");
    before = KeyProofCounts();
    Require(before.proved == after.proved + 1 && before.scanned == after.scanned, "(9) the second call scanned");
    const auto generation = proof.generation;
    MarkWritten(address, keyCount);
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "(9) unchanged bytes read differently");
    after = KeyProofCounts();
    Require(after.scanned == before.scanned + 1 && after.proved == before.proved && proof.generation > generation, "(9) a stamped key range was proved");
    std::memset(keys, 0x40, keyCount);
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001, "(9) a CPU write of the keys was not seen");
    before = KeyProofCounts();
    Require(before.scanned == after.scanned + 1 && proof.keys == DccKeys::Clear0001, "(9) the CPU write did not make a scan");
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && KeyProofCounts().proved == before.proved + 1, "(9) the new keys were not proved");
    recorder.NotePendingWrite(address, keyCount);
    MarkWritten(address, keyCount);
    before = KeyProofCounts();
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && proof.generation != 0, "(9) an unknown pending write was not synchronized before proving keys");
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && KeyProofCounts().proved == before.proved + 1, "(9) the synchronized proof did not hold");
    after = KeyProofCounts();
    Require(after.scanned == before.scanned + 1 && after.unstable == before.unstable && after.proved == before.proved + 1, "(9) a synchronized key scan was miscounted");
    recorder.Submit();
    device.WaitQueue();
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && proof.generation != 0, "(9) a signaled write kept the scan unstable");
    recorder.Sync();
    before = KeyProofCounts();
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && KeyProofCounts().proved == before.proved + 1, "(9) the proof did not hold after the batch finished");
}

void targetKeyProofTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t bytes = 65536;
    void* block = AllocateWatched(bytes, bytes);
    if (block == nullptr) {
        std::cout << "no write watching: target key proofs not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    constexpr std::size_t keyCount = 1024;
    constexpr std::uint64_t surfaceBytes = keyCount * 256;
    auto* keys = static_cast<std::uint8_t*>(block);
    std::memset(keys, 0x20, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto scans = [] { return KeyProofCounts().rangeScanned; };
    const auto proofs = [] { return KeyProofCounts().rangeProved; };
    DccRangeProof proof;
    auto scanned = scans();
    auto proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::ClearRegister, "register clear keys read as another code");
    if (!KeyFastPath()) {
        Require(proof.generation == 0 && scans() == scanned && proofs() == proved, "target key proofs were kept while disabled");
        std::cout << "key fast path off: every target key read scans\n";
        return;
    }
    Require(scans() == scanned + 1 && proofs() == proved && proof.generation != 0 && proof.keys == DccKeys::ClearRegister, "the first read of a settled key range left no proof");
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::ClearRegister && scans() == scanned + 1 && proofs() == proved + 1, "an unchanged key range was scanned again");
    MarkWritten(address, keyCount);
    scanned = scans();
    proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::ClearRegister && scans() == scanned + 1 && proofs() == proved, "a key store over the range was answered from the proof");
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::ClearRegister && proofs() == proved + 1, "the rescan after a key store left no proof");
    std::memset(keys, 0xff, keyCount);
    scanned = scans();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Uncompressed && scans() == scanned + 1, "a CPU write of the keys was not seen");
    scanned = scans();
    proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes / 2, proof) == DccKeys::Uncompressed && scans() == scanned + 1 && proofs() == proved, "the proof of another key range answered");
    Require(ProvedCurrentDccKeys(address + 256, surfaceBytes / 2, proof) == DccKeys::Uncompressed && scans() == scanned + 2 && proofs() == proved, "the proof of a key range at another address answered");
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Uncompressed && scans() == scanned + 3, "a proof taken over a shorter range answered the whole range");
    recorder.NotePendingWrite(address, keyCount);
    MarkWritten(address, keyCount);
    NoteKeysFillOnGpu(address, keyCount, DccKeys::Clear0000);
    scanned = scans();
    proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Clear0000 && proof.generation == 0, "a pending key fill was not the answer, or its answer was kept");
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Clear0000 && scans() == scanned + 2 && proofs() == proved, "a pending key fill was answered from a proof");
    std::memset(keys, 0x00, keyCount);
    recorder.Submit();
    device.WaitQueue();
    recorder.Sync();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Clear0000 && proof.generation != 0, "the landed fill was not proved");
    proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Clear0000 && proofs() == proved + 1, "the proof after the fill landed did not hold");
}

// A resource build over host-imported guest memory notes its in-place reads when its writes are
// marked, so a CPU copy into that memory is refused until the batch ran.
// Per-batch store runs (Recorder::RecordStore): the stores of a batch queue until Submit, a
// later writer or reader of a queued store's bytes has the run recorded in place first, and the
// stores land in the import's memory once the batch ran. With APS5_LABEL_RUNS_INLINE=1 or
// APS5_NO_LABEL_RUNS=1 (run the binary with either set) nothing queues and the stores land alike.
void storeRunTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: store runs not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the store test block");
    std::memset(block, 0, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the store test block refused: store runs not tested\n";
        return;
    }
    const bool perBatch = std::getenv("APS5_LABEL_RUNS_INLINE") == nullptr && std::getenv("APS5_NO_LABEL_RUNS") == nullptr && std::getenv("APS5_NO_LABEL_BATCHING") == nullptr;
    auto* words = static_cast<volatile std::uint32_t*>(block);
    const auto store = [&](std::size_t word, std::uint32_t value) {
        const std::array<std::byte, 4> value4{std::byte{static_cast<unsigned char>(value)}, std::byte{static_cast<unsigned char>(value >> 8u)}, std::byte{static_cast<unsigned char>(value >> 16u)}, std::byte{static_cast<unsigned char>(value >> 24u)}};
        recorder.RecordStore(import->buffer, address + word * 4 - import->base, value4, address + word * 4);
    };
    Require(recorder.Idle() && !recorder.HasQueuedStores(), "a fresh recorder has queued stores");
    store(0, 1);
    store(1, 2);
    store(4, 3);
    store(4, 4);
    Require(recorder.Recording(), "a store did not open a batch");
    Require(recorder.HasQueuedStores() == perBatch, "per-batch runs do not queue (or inline runs do)");
    Require(recorder.QueuedStoreOverlaps(address, 8) == perBatch && recorder.QueuedStoreOverlaps(address + 16, 4) == perBatch && !recorder.QueuedStoreOverlaps(address + 8, 8), "queued store ranges are wrong");
    const auto before = Recorder::StoreCounts();
    // A reader or writer of untouched bytes forces nothing; one over a queued store records the run.
    recorder.FlushStoresOverlapping(address + 8, 8);
    Require(recorder.HasQueuedStores() == perBatch, "a disjoint range flushed the run");
    recorder.FlushStoresOverlapping(address + 16, 4);
    Require(!recorder.HasQueuedStores() && Recorder::StoreCounts().runsForced == before.runsForced + (perBatch ? 1 : 0), "an overlapping writer did not record the run in place");
    store(2, 5);
    Require(recorder.HasQueuedStores() == perBatch, "a store after a forced run did not start a new run");
    recorder.Submit();
    Require(!recorder.HasQueuedStores() && Recorder::StoreCounts().runsAtSubmit == before.runsAtSubmit + (perBatch ? 1 : 0), "Submit did not record the run");
    recorder.Sync();
    Require(words[0] == 1 && words[1] == 2 && words[2] == 5 && words[4] == 4 && words[3] == 0, "the stores did not land (or the later store of a dword lost)");
    const auto after = Recorder::StoreCounts();
    Require(after.stores == before.stores + 1 && after.replaced == before.replaced + 0 && after.joined == before.joined + 0, "store counters are off");
    // The covered-access mask: reported once by the next Commands() call, then cleared.
    VkAccessFlags covered = 0xffffffffu;
    recorder.Commands(&covered);
    Require(covered == 0, "a new batch starts covered");
    recorder.MarkCovered(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    recorder.Commands(&covered);
    Require(covered == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT), "the covered mask is not reported");
    recorder.Commands(&covered);
    Require(covered == 0, "the covered mask survived a Commands() call");
    recorder.Sync();
    // The hazard tracker (counting mode only): accesses are noted without effect on the batch.
    const std::pair<std::uint64_t, std::uint64_t> range{address, address + 64};
    recorder.NoteAccess(Recorder::CommandClass::Fill, Recorder::Access{{}, std::span(&range, 1), {}, VK_PIPELINE_STAGE_TRANSFER_BIT});
    recorder.NoteAccess(Recorder::CommandClass::DispatchLeading, Recorder::Access{std::span(&range, 1), {}, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});
    Require(recorder.Recording() == Recorder::BarrierValidate(), "a noted access opened a batch with the tracker off (or none with it on)");
    recorder.Sync();
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    HostImportFor(context, address, bytes);
}

void remappedImportTests(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: remapped imports not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the remap test block");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the remap test block refused: remapped imports not tested\n";
        return;
    }
    const auto first = HostImportSerial(context, address, bytes, true);
    Require(first != 0 && HostImportSerial(context, address, bytes, true) == first, "an unchanged range lost its import");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        mutation.Add(block, bytes, true, true, true);
    }
    Require(HostImportFor(context, address, bytes) != nullptr, "the remapped range was not imported again");
    const auto second = HostImportSerial(context, address, bytes, true);
    Require(second != 0 && second != first, "the import of a range mapped again was kept");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    HostImportFor(context, address, bytes);
}

void hostImportRetirementTests(const Device& device) {
    const auto& context = device.GetContext();
    constexpr std::size_t UnitBytes = 65536;
    constexpr std::size_t AllocationBytes = 4 * UnitBytes;
    constexpr std::size_t CopyBytes = 256;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, AllocationBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(UnitBytes, AllocationBytes);
#endif
    Require(block != nullptr, "cannot allocate the host import retirement block");
    const auto release = [](void* pointer) {
#ifdef _WIN32
        VirtualFree(pointer, 0, MEM_RELEASE);
#else
        std::free(pointer);
#endif
    };
    std::unique_ptr<void, decltype(release)> allocation(block, release);
    auto* memory = static_cast<std::byte*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    Recorder recorder(context);
    recorder.Activate();
    Buffer readback(context, CopyBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::array<bool, 4> registered{};
    const auto cleanup = [&] {
        if (!gpu.owns_lock()) gpu.lock();
        recorder.Sync();
        recorder.FlushDeferredReleases();
        ClearHostImports(context.device);
        GuestAllocations::Mutation mutation;
        for (std::size_t index = 0; index < registered.size(); ++index) {
            if (!registered[index]) continue;
            mutation.Remove(memory + index * UnitBytes);
            registered[index] = false;
        }
        SetHostImportWindow(0);
    };
    const auto add = [&](std::size_t index) {
        GuestAllocations::Mutation mutation;
        mutation.Add(memory + index * UnitBytes, UnitBytes, true, true, true);
        registered[index] = true;
    };
    const auto mapped = [&](std::size_t index) {
        GuestAllocations::Mapped result;
        for (const auto& range : GuestAllocations::GuestAllocationsAcquire_nid_postfix()) {
            if (range->address == address + index * UnitBytes) result.push_back(range);
        }
        Require(result.size() == 1, "host import retirement lost its registered range");
        return result;
    };
    bool deferredEveryRetirement = true;
    bool preservedEveryRecorder = true;
    try {
        const std::array<const char*, 3> shapes{"same-base growth", "earlier-base overlap", "predecessor split"};
        for (std::size_t shape = 0; shape < shapes.size(); ++shape) {
            SetHostImportWindow(0);
            Require(ImportMappedRanges(context, {}, GuestAllocations::GuestAllocationsGeneration_nid_postfix(), true), "host import retirement could not adopt the device");
            const std::size_t first = shape == 1 ? 1 : 0;
            add(first);
            if (shape == 2) add(1);
            for (std::size_t index = 0; index < AllocationBytes; ++index) memory[index] = static_cast<std::byte>((index * 37u + shape * 19u + 11u) & 0xffu);
            const auto initial = mapped(first);
            const auto initialGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
            gpu.unlock();
            const bool fresh = ImportMappedRanges(context, initial, initialGeneration, false);
            const bool unchanged = ImportMappedRanges(context, initial, initialGeneration, false);
            gpu.lock();
            Require(fresh && unchanged, "host import retirement blocked a fresh or unchanged fast import");
            const auto sourceAddress = address + (shape == 2 ? UnitBytes : first * UnitBytes);
            const auto* imported = HostImportFor(context, sourceAddress, CopyBytes);
            Require(imported != nullptr, "host import retirement could not import its source");
            const auto source = imported->buffer;
            const auto sourceOffset = sourceAddress - imported->base;
            std::array<std::byte, CopyBytes> expected{};
            std::memcpy(expected.data(), reinterpret_cast<const void*>(sourceAddress), CopyBytes);
            recorder.Commands();
            recorder.Submit();
            device.WaitQueue();
            Require(!recorder.Recording() && recorder.InFlightBatches() == 1, "host import retirement requires a submitted batch without an open batch");
            std::size_t changed = 0;
            if (shape == 0) {
                changed = 1;
                add(changed);
            } else if (shape == 1) {
                add(changed);
            } else {
                SetHostImportWindow(UnitBytes);
                add(3);
                changed = 1;
            }
            const auto ranges = mapped(changed);
            const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
            gpu.unlock();
            const bool deferred = !ImportMappedRanges(context, ranges, generation, false);
            gpu.lock();
            const bool preserved = !recorder.Recording() && recorder.InFlightBatches() == 1;
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, source, sourceOffset, readback.Handle(), 0, CopyBytes);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            Require(ImportMappedRanges(context, ranges, generation, true), "host import retirement refused its retry under the GPU lock");
            Require(HostImportFor(context, sourceAddress, CopyBytes) != nullptr, "host import retirement lost the replacement import");
            recorder.Sync();
            readback.Invalidate();
            Require(std::equal(expected.begin(), expected.end(), readback.Bytes().begin()), "host import retirement changed bytes copied from the retired import");
            std::cout << "host import retirement: " << shapes[shape] << " GPU readback verified; deferred=" << deferred << "; recorder preserved=" << preserved << '\n';
            deferredEveryRetirement = deferredEveryRetirement && deferred;
            preservedEveryRecorder = preservedEveryRecorder && preserved;
            cleanup();
        }
    } catch (...) {
        cleanup();
        throw;
    }
    Require(deferredEveryRetirement, "unlocked mapping retired an import instead of requesting the GPU lock");
    Require(preservedEveryRecorder, "unlocked mapping changed recorder batches");
    std::cout << "Host import retirement locking and GPU readback tests passed\n";
}
#ifdef _WIN32
void secondMappingImportTests(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: imports after a second mapping not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
    constexpr std::size_t page = 16384;
    struct Release {
        void* mapping = nullptr;
        void* alias = nullptr;
        std::int64_t physical = -1;
        ~Release() {
            if (alias != nullptr) sceKernelMunmap(alias, page);
            if (mapping != nullptr) sceKernelMunmap(mapping, bytes);
            if (physical >= 0) sceKernelReleaseDirectMemory(physical, bytes);
        }
    } release;
    Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, bytes, bytes, 0, &release.physical) == 0, "allocate the second mapping test backing");
    Require(sceKernelMapDirectMemory(&release.mapping, bytes, 0x33, 0, release.physical, bytes) == 0, "map the second mapping test backing");
    const auto address = reinterpret_cast<std::uint64_t>(release.mapping);
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of singly mapped direct memory refused: imports after a second mapping not tested\n";
        return;
    }
    const auto first = HostImportSerial(context, address, bytes, true);
    Require(first != 0 && HostImportSerial(context, address, bytes, true) == first, "singly mapped direct memory lost its import");
    Require(sceKernelMapDirectMemory(&release.alias, page, 0x33, 0, release.physical, page) == 0, "map the first page a second time");
    Require(HostImportFor(context, address, bytes) != nullptr, "direct memory was not imported again after a second mapping of its first page");
    const auto second = HostImportSerial(context, address, bytes, true);
    Require(second != 0 && second != first, "the import of direct memory whose pages were replaced was kept");
    HostImportFor(context, address, bytes);
}
#endif

void movedMetadataTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: moved DCC metadata not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the moved metadata block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* firstKeys = texels + surfaceBytes;
    auto* secondKeys = firstKeys + 4096;
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the moved metadata block refused: moved DCC metadata not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    Require(DescribeSurface(resource).guestBytes == surfaceBytes, "the moved metadata surface has an unexpected size");
    const auto first = address + surfaceBytes;
    const auto second = first + 4096;
    const auto withKeys = [&](std::uint64_t keys) {
        auto described = resource;
        described.dccAddress = keys;
        return described;
    };
    const auto holds = [&](const StorageTexture& image, std::array<std::uint8_t, 4> texel) {
        Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {side, side, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        const auto pixels = readback.Bytes();
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            if (std::to_integer<std::uint8_t>(pixels[i]) != texel[i % 4]) return false;
        }
        return true;
    };
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(firstKeys, 0xff, keyCount);
    std::memset(secondKeys, 0x00, keyCount);
    const auto original = CachedStorageSurface(context, withKeys(first));
    Require(original->Descriptor().dccAddress == first && holds(*original, {0x55, 0x55, 0x55, 0x55}), "the first image does not hold the stored texels");
    const auto moved = CachedStorageSurface(context, withKeys(second));
    Require(moved->Descriptor().dccAddress == second, "the storage image kept the keys the surface no longer names");
    Require(!StorageImageCached(context, original.get()), "the image of the old keys is still the surface's");
    Require(holds(*moved, {0, 0, 0, 0}), "a fast clear of the moved keys did not reach the image");
    Require(CachedStorageSurface(context, resource) == moved && moved->Descriptor().dccAddress == second, "a descriptor without metadata replaced the image");
    const auto back = CachedStorageSurface(context, withKeys(first));
    Require(back != moved && back->Descriptor().dccAddress == first, "the keys moving back did not remake the image");
    Require(holds(*back, {0x55, 0x55, 0x55, 0x55}), "the image remade under the first keys does not hold the stored texels");
}

void viewPastLastMipTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: views past the last mip not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t bytes = 0x60000;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the past-last-mip block");
    auto* texels = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the past-last-mip block refused: views past the last mip not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    const auto view = [&](std::uint32_t level) {
        const std::array<std::uint32_t, 8> words{
            static_cast<std::uint32_t>(address >> 8u),
            static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((side - 1u) & 3u) << 30u),
            ((side - 1u) >> 2u) | ((side - 1u) << 14u),
            0xfacu | (level << 12u) | (level << 16u) | (0x1bu << 20u) | (9u << 28u),
            0u,
            2u << 4u,
            0u,
            0u,
        };
        return DecodeTextureResource(words);
    };
    const auto texel = [&](const StorageTexture& image, std::uint32_t level) {
        Buffer readback(context, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
        copy.imageExtent = {1, 1, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        std::uint32_t value = 0;
        std::memcpy(&value, readback.Bytes().data(), sizeof(value));
        return value;
    };
    std::memset(texels, 0x55, bytes);
    std::memset(texels + 0x8400, 0x66, 4);
    std::memset(texels + 0x4800, 0x77, 4);
    const auto allocatedView = view(2);
    const auto pastView = view(3);
    Require(allocatedView.mipCount == 3 && pastView.mipCount == 4 && DescribeSurface(pastView).guestBytes == bytes, "a 256x256 32 bpp SW_64KB_R_X surface viewed past MAX_MIP 2 decoded wrongly");
    const auto allocated = CachedStorageSurface(context, allocatedView);
    Require(allocated->Descriptor().mipCount == 3, "the allocated levels' storage image has an unexpected level count");
    const auto past = CachedStorageSurface(context, pastView);
    Require(past->Descriptor().mipCount == 4, "the storage image of a view past the last mip lacks the level it names");
    Require(!StorageImageCached(context, allocated.get()), "the allocated levels' image is still the surface's after a view past its last mip");
    Require(CachedStorageSurface(context, allocatedView) == past, "a view of the allocated levels took another image than the view past the last mip");
    Require(texel(*past, 2) == 0x66666666u, "level 2 was not read from its addrlib tail slot");
    Require(texel(*past, 3) == 0x77777777u, "the level past the last mip was not read from its addrlib tail slot");
}

void resourceReadTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: resource read notes not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the test block");
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the test block refused: resource read notes not tested\n";
        return;
    }
    const auto element = address + 4096;
    constexpr std::size_t elementBytes = 1024;
    {
        GuestBufferMemory memory(context);
        memory.AddInPlace(element, elementBytes, false);
        memory.Upload();
        const auto reads = memory.InPlaceReads();
        Require(reads.size() == 1 && reads[0].first == element && reads[0].second == element + elementBytes, "an imported region is not an in-place read");
        Require(memory.Writes().empty(), "a readable element counts as written");
    }
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    HostImportFor(context, address, bytes);
}

class DescriptorUpdates {
public:
    struct Contents {
        std::map<std::pair<std::uint32_t, std::uint32_t>, VkDescriptorBufferInfo> buffers;
        std::map<std::pair<std::uint32_t, std::uint32_t>, VkDescriptorImageInfo> images;
    };
    explicit DescriptorUpdates(Context& context) : context(context), previous(context.functions),
        update(context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")) {
        Require(active == nullptr, "nested descriptor update capture");
        if (previous != nullptr) functions = *previous;
        functions.updateDescriptorSets = captureUpdate;
        context.functions = &functions;
        active = this;
    }
    ~DescriptorUpdates() { context.functions = previous; active = nullptr; }
    std::vector<Contents> calls;
    std::uint32_t copies = 0;

private:
    static VKAPI_ATTR void VKAPI_CALL captureUpdate(VkDevice device, std::uint32_t count, const VkWriteDescriptorSet* writes, std::uint32_t copyCount, const VkCopyDescriptorSet* copies) {
        Contents contents;
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& write = writes[i];
            for (std::uint32_t element = 0; element < write.descriptorCount; ++element) {
                const auto key = std::pair{write.dstBinding, write.dstArrayElement + element};
                if (write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) contents.buffers[key] = write.pBufferInfo[element];
                else contents.images[key] = write.pImageInfo[element];
            }
        }
        active->calls.push_back(std::move(contents));
        active->copies += copyCount;
        active->update(device, count, writes, copyCount, copies);
    }
    Context& context;
    const DeviceFunctions* previous;
    DeviceFunctions functions{};
    PFN_vkUpdateDescriptorSets update;
    static inline DescriptorUpdates* active = nullptr;
};

void drawDescriptorWriteTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    DescriptorUpdates updates(context);
    {
        DescriptorCache cache(context);
        context.descriptorCache = &cache;
        Recorder snapshots(context);
        snapshots.Activate();
        ShaderRecompiler::RecompileResult program;
        ShaderRecompiler::DescriptorBinding data{};
        data.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
        data.role = ShaderRecompiler::DescriptorRole::FlattenedSrt;
        data.binding = 2;
        data.count = 1;
        data.guestDescriptor = {1, 2, 3, 4};
        program.bindings.push_back(data);
        data.binding = 7;
        program.bindings.push_back(data);
        const CompiledShader shader{ShaderRecompiler::ShaderStage::Compute, &program, 0};
        ShaderResources resources(context, shader);
        Require(updates.calls.size() == 1, "building resources did not write one descriptor set");
        const auto original = updates.calls.front();
        Require(original.buffers.size() == 2 && original.images.empty(), "descriptor test lacks mixed bindings, or an image array took descriptors in the set");
        snapshots.Sync();
    }
    recorder.Activate();
}

void drawInputInPlaceTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    constexpr std::size_t bytes = 65536;
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "host imports or write watching unavailable: in-place draw inputs not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the watched block refused: in-place draw inputs not tested\n";
        return;
    }
    context.recorder = &recorder;
    recorder.Sync();
    const auto vertices = address + 4096;
    const auto* import = HostImportFor(context, vertices, 256);
    const auto inPlace = InPlaceDrawInput(context, vertices, 256, 4, false);
    Require(import != nullptr && inPlace && inPlace.gathered == nullptr && inPlace.buffer == import->buffer && inPlace.offset == vertices - import->base, "an imported draw input was not bound in place");
    const auto read = recorder.DescribePendingRead(vertices, 256);
    Require(read.has_value() && read->kind == Recorder::ReadKind::DrawInput && read->open, "an in-place draw input's read was not noted on the open batch");
    const auto misaligned = InPlaceDrawInput(context, vertices + 2, 256, 4, false);
    Require(misaligned && misaligned.gathered != nullptr && misaligned.buffer == misaligned.gathered->Handle() && misaligned.offset == 0, "a draw input misaligned for its format was not gathered into a buffer of its own");
    const auto aligned = InPlaceDrawInput(context, vertices + 2, 256, 2, false);
    Require(aligned && aligned.gathered == nullptr, "an aligned 16-bit draw input was not bound in place");
    recorder.NotePendingWrite(address + 8192, 16);
    Require(InPlaceDrawInput(context, address + 8192, 256, 4, true) && !recorder.PendingWriteOverlaps(address + 8192, 256), "a draw input the CPU scans under a pending GPU write was bound before the write completed");
    const std::array<std::byte, 4> label{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    recorder.RecordStore(import->buffer, address + 12288 - import->base, label, address + 12288);
    Require(recorder.QueuedStoreOverlaps(address + 12288, 4), "the label store was not queued");
    Require(InPlaceDrawInput(context, address + 12288, 256, 4, false) && !recorder.QueuedStoreOverlaps(address + 12288, 4), "a draw input under a queued label store was bound before the store was recorded");
    Require(static_cast<bool>(InPlaceDrawInput(context, address + 16384, 256, 4, false)), "a disjoint draw input was not bound in place");
    alignas(64) static std::array<std::byte, 256> unregistered{};
    const auto refused = [](const auto& read) {
        try {
            static_cast<void>(read());
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    Require(refused([&] { return InPlaceDrawInput(context, reinterpret_cast<std::uint64_t>(unregistered.data()), unregistered.size(), 4, false); }), "memory no import or mirror serves was bound in place");
    auto unrecorded = context;
    unrecorded.recorder = nullptr;
    Require(refused([&] { return InPlaceDrawInput(unrecorded, vertices, 256, 4, false); }), "a draw without a recorder bound its input in place");
    recorder.Sync();
    Require(!recorder.PendingReadOverlaps(vertices, 256), "an in-place draw input's read outlived its batch");
    if (!AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "host imports are compared, not watched: kept highest indices not tested\n";
        return;
    }
    const auto indices = address + 20480;
    constexpr std::size_t kept = Recorder::KeptIndexRangeIndices * 2;
    std::vector<std::uint16_t> first(Recorder::KeptIndexRangeIndices * 2, 1);
    first[0] = 3;
    first[1] = 9;
    first[2] = 0xffff;
    std::memcpy(reinterpret_cast<void*>(indices), first.data(), kept * 2);
    auto range = InPlaceIndexRange(recorder, indices, kept, 2, false);
    Require(!range.reused && range.highest == 0xffffu, "an in-place index range was not scanned at its first use");
    range = InPlaceIndexRange(recorder, indices, kept, 2, false);
    Require(range.reused && range.highest == 0xffffu, "an unchanged in-place index range was scanned again");
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(!range.reused && range.highest == 9u, "an in-place index range served a scan made under the other restart rule");
    Require(!InPlaceIndexRange(recorder, indices, kept + 2, 2, true).reused, "an in-place index range served a scan of another size");
    const std::uint16_t raised = 12;
    std::memcpy(reinterpret_cast<void*>(indices + 2), &raised, sizeof(raised));
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range kept its highest index over a guest write");
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(range.reused && range.highest == 12u, "a rescanned in-place index range was not kept");
    range = InPlaceIndexRange(recorder, indices, kept - 2, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range below the kept size was not scanned");
    range = InPlaceIndexRange(recorder, indices, kept - 2, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range below the kept size was kept");
    range = InPlaceIndexRange(recorder, indices, kept * 2 - 4, 4, false);
    Require(!range.reused && range.highest == 0x000c0003u, "a 32-bit in-place index range below the kept size was not scanned");
    Require(!InPlaceIndexRange(recorder, indices, kept * 2 - 4, 4, false).reused, "a 32-bit in-place index range below the kept size was kept");
    Require(!InPlaceIndexRange(recorder, indices, kept * 2, 4, false).reused, "a 32-bit in-place index range was not scanned at its first use");
    range = InPlaceIndexRange(recorder, indices, kept * 2, 4, false);
    Require(range.reused && range.highest == 0x000c0003u, "an unchanged 32-bit in-place index range was scanned again");
    const std::vector<std::uint16_t> restarts(Recorder::KeptIndexRangeIndices, 0xffff);
    std::memcpy(reinterpret_cast<void*>(indices + 20480), restarts.data(), kept);
    Require(!InPlaceIndexRange(recorder, indices + 20480, kept, 2, true).highest.has_value(), "an index range of restart indices has a highest index");
    range = InPlaceIndexRange(recorder, indices + 20480, kept, 2, true);
    Require(range.reused && !range.highest.has_value(), "an empty in-place index range was scanned again");
}

void storageRefreshTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: storage refresh not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = nullptr;
    if (watched) {
        block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: storage refresh in watched memory not tested\n";
            return;
        }
    } else {
#ifdef _WIN32
        block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        block = std::aligned_alloc(65536, bytes);
#endif
    }
    Require(block != nullptr, "cannot allocate the storage refresh block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    const auto* import = HostImportFor(base, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the storage refresh block refused: storage refresh not tested\n";
        return;
    }
    if (watched && !AgcDriver::GuestMemory::Watched(address, bytes)) std::cout << "host imports are compared, not watched: storage refresh in watched memory runs as unwatched\n";
    watched = watched && AgcDriver::GuestMemory::Watched(address, bytes);
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    Require(DescribeSurface(resource).guestBytes == surfaceBytes, "the test surface has an unexpected size");
    DccKeyProof proof;
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "0x00 keys are not a 0000 clear");
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto draw = [&](VkClearColorValue value) {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        };
        const auto holds = [&](std::array<std::uint8_t, 4> texel) {
            Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                if (std::to_integer<std::uint8_t>(pixels[i]) != texel[i % 4]) return false;
            }
            return true;
        };
        const auto memoryHolds = [&](std::array<std::uint8_t, 4> texel, std::size_t first = 0, std::uint8_t other = 0) {
            std::vector<std::byte> read(surfaceBytes);
            if (watched) AgcDriver::GuestMemory::Read(address, read);
            else std::memcpy(read.data(), texels, surfaceBytes);
            for (std::size_t i = 0; i < surfaceBytes; ++i) {
                if (std::to_integer<std::uint8_t>(read[i]) != (i < first ? other : texel[i % 4])) return false;
            }
            return true;
        };
        const auto holdsMixed = [&](std::array<std::uint8_t, 4> texel, std::size_t count, std::uint8_t other) {
            Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            std::size_t others = 0;
            for (std::size_t i = 0; i < pixels.size(); i += 4) {
                bool isOther = true, isTexel = true;
                for (std::size_t c = 0; c < 4; ++c) {
                    const auto value = std::to_integer<std::uint8_t>(pixels[i + c]);
                    isOther = isOther && value == other;
                    isTexel = isTexel && value == texel[c];
                }
                if (isOther) ++others;
                else if (!isTexel) return false;
            }
            return others == count;
        };
        Require(holds({0, 0, 0, 0}), "a surface under 0000 clear keys was not cleared");
        draw({{1.0f, 0.0f, 0.0f, 1.0f}});
        image->Refresh();
        Require(holds({255, 0, 0, 255}), "a refresh under the image's own clear keys dropped the draw's results");
        if (watched) {
            Require(memoryHolds({255, 0, 0, 255}), "the refresh's store of the results did not reach watched guest memory");
        } else {
            Require(StorageTexture::FindPending(address, surfaceBytes) == image, "an unchanged untracked refresh discarded the pending image");
            Require(StorageTexture::FlushPending(address, surfaceBytes, nullptr, "storage refresh visibility test"), "the pending untracked image was not flushed");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(memoryHolds({255, 0, 0, 255}), "an explicit flush did not make pending results visible in untracked guest memory");
        }
        if (watched) {
            draw({{0.0f, 0.0f, 1.0f, 1.0f}});
            std::memset(texels, 0x33, 65536);
            image->Refresh();
            Require(holdsMixed({0, 0, 255, 255}, 65536 / 4, 0x33), "a CPU store into a unit with results pending was not seen by the refresh");
            Require(memoryHolds({0, 0, 255, 255}, 65536, 0x33), "guest memory lost the CPU store or the other units' results");
            image->Refresh();
            Require(holdsMixed({0, 0, 255, 255}, 65536 / 4, 0x33), "an unchanged surface changed at a refresh");
        }
        draw({{0.0f, 1.0f, 0.0f, 1.0f}});
        std::memset(keys, 0x40, keyCount);
        image->Refresh();
        Require(holds({0, 0, 0, 255}), "new 0001 clear keys did not clear the image");
        Require(watched ? memoryHolds({0, 0, 255, 255}, 65536, 0x33) : memoryHolds({255, 0, 0, 255}), "results dead under new clear keys were stored");
    }
    recorder.Sync();
}

bool writtenBufferOrderTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: written buffer order not tested\n";
        return false;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t bytes = side * side * 4;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: written buffer order not tested\n";
        return false;
    }
    std::memset(block, 0x55, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the written buffer order block refused: written buffer order not tested\n";
        return false;
    }
    std::cout << "written buffer order: host imports are " << (AgcDriver::GuestMemory::Watched(address, bytes) ? "watched" : "compared") << "\n";
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    Require(DescribeSurface(resource).guestBytes == bytes, "the written buffer order surface has an unexpected size");
    const auto settle = [&] {
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    };
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
        {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        }
        constexpr std::size_t written = 40800;
        GuestBufferMemory memory(context);
        memory.AcquireRegistered();
        memory.AddInPlace(address, written, true);
        memory.Upload();
        std::uint32_t adjustment = 0;
        const auto view = memory.Address(address, written, adjustment);
        Require(adjustment == 0, "the written buffer view is not aligned to its element");
        const auto* import = HostImportFor(context, address, written);
        Require(import != nullptr && view.first == import->address + (address - import->base), "the written buffer view is not in place in its host import");
        {
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, import->buffer, address - import->base, written, 0xa5a5a5a5u);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        memory.MarkDirectWrites();
        settle();
        StorageTexture::FlushAllPending("written buffer order test");
        settle();
        std::vector<std::byte> read(bytes);
        AgcDriver::GuestMemory::Read(address, read);
        Require(std::all_of(read.begin(), read.begin() + written, [](std::byte value) { return value == std::byte{0xa5}; }), "storage results pending under a buffer a later use wrote were stored over the buffer's contents");
        for (std::size_t i = written; i < bytes; ++i) Require(std::to_integer<std::uint8_t>(read[i]) == (i % 4 == 1 || i % 4 == 2 ? 0 : 255), "the storage results outside the written buffer did not reach guest memory");
    }
    recorder.Sync();
    return true;
}

void hostImportUnmapTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: import unmap lifetime not tested\n";
        return;
    }
    const auto alignment = static_cast<std::size_t>(std::max<VkDeviceSize>(65536u, context.hostImportAlignment));
    constexpr std::size_t RequestedBytes = 17u << 20u;
    const auto bytes = ((RequestedBytes + alignment - 1u) / alignment) * alignment;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(alignment, bytes);
#endif
    Require(block != nullptr, "cannot allocate the host import unmap block");
    struct Block {
        void* address;
        std::size_t bytes;
        bool registered = true;
        bool released = false;
        void Release() {
#ifdef _WIN32
            Require(VirtualFree(address, 0, MEM_RELEASE) != 0, "cannot release the host import unmap block");
#else
            std::free(address);
#endif
            released = true;
        }
        ~Block() {
            if (registered) {
                try {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(address);
                } catch (...) {
                    return;
                }
            }
            if (!released) {
#ifdef _WIN32
                VirtualFree(address, 0, MEM_RELEASE);
#else
                std::free(address);
#endif
            }
        }
    } allocation{block, bytes};
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    const HostImport* imported = nullptr;
    {
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(context, address, bytes);
    }
    if (imported == nullptr) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        allocation.registered = false;
        allocation.Release();
        std::cout << "host import refused: import unmap lifetime not tested\n";
        return;
    }
    bool retiredBeforeRelease = false;
    {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(block, bytes, [&](const void*, std::size_t, const void*, bool) {
            retiredBeforeRelease = !HostImportCovers(context, address, bytes);
            if (retiredBeforeRelease) allocation.Release();
        });
        allocation.registered = false;
    }
    if (!retiredBeforeRelease) {
        {
            std::lock_guard gpu(GpuMutex());
            static_cast<void>(HostImportFor(context, address, bytes));
            recorder.Sync();
        }
        allocation.Release();
    }
    Require(retiredBeforeRelease, "guest unmap released pages before retiring their host import");
    std::cout << "host import unmap lifetime: ok\n";
}

void parkedImportUnmapTests(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: parked import unmap not tested\n";
        return;
    }
    const auto alignment = static_cast<std::size_t>(std::max<VkDeviceSize>(65536u, context.hostImportAlignment));
    const auto bytes = alignment * 4u;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(alignment, bytes);
#endif
    Require(block != nullptr, "cannot allocate the parked import unmap block");
    const auto release = [&] {
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    };
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    std::unique_ptr<Device> replacement;
    bool imported = false;
    {
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(context, address, bytes) != nullptr;
    }
    if (imported) {
        replacement = std::make_unique<Device>();
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(replacement->GetContext(), address, bytes) != nullptr;
    }
    if (!imported) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        if (replacement) ClearHostImports(replacement->GetContext().device);
        release();
        std::cout << "host import refused: parked import unmap not tested\n";
        return;
    }
    bool unmapped = false;
    try {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(block, bytes, [&](const void*, std::size_t, const void*, bool) { unmapped = true; });
    } catch (const std::exception& error) {
        ClearHostImports(replacement->GetContext().device);
        ClearHostImports(context.device);
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        release();
        throw std::runtime_error(std::string("guest unmap after a device replacement: ") + error.what());
    }
    ClearHostImports(replacement->GetContext().device);
    ClearImageMirrors(replacement->GetContext().device);
    replacement.reset();
    release();
    Require(unmapped, "guest unmap of a range a replaced device imported did not run");
    std::cout << "parked import unmap: ok\n";
}

void importWatchTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
#ifdef _WIN32
    if (context.hostImportAlignment == 0 || !WriteWatched()) {
        std::cout << "host imports or write watching unavailable: import watch decisions not tested\n";
        return;
    }
    const auto probe = ProbeImportWriteProtection(context);
    Require(probe.failure == nullptr, std::string("(w) the import probe failed at ") + (probe.failure != nullptr ? probe.failure : "") + " (" + std::to_string(static_cast<int>(probe.result)) + ")");
    std::cout << "import probe: " << probe.writtenAtImport << " of " << probe.pages << " scratch pages marked inside the import, " << probe.writtenAfterSubmit << " after a GPU read, " << probe.writtenAfterFill << " after a GPU write, " << probe.writtenByCpu << " after a CPU store\n";
    Require(probe.writtenByCpu == 1, "(w) a CPU store into an imported scratch page was not collected exactly");
    const bool late = probe.writtenAfterSubmit != 0 || probe.writtenAfterFill != 0;
    Require(PrepareImportWatch(context) == (late ? ImportWatch::Unwatch : ImportWatch::Watch), "(w) the Windows import watch decision does not follow the probe");
    std::cout << "import watch decisions: Windows host imports " << (late ? "use comparisons" : "stay watched") << "\n";
#else
    if (context.hostImportAlignment == 0 || !WriteWatched()) {
        std::cout << "host imports or write watching unavailable: import watch decisions not tested\n";
        return;
    }
    const auto probe = ProbeImportWriteProtection(context);
    Require(probe.failure == nullptr, std::string("(u) the import probe failed at ") + (probe.failure != nullptr ? probe.failure : "") + " (" + std::to_string(static_cast<int>(probe.result)) + ")");
    std::cout << "import probe: " << probe.writtenAfterSubmit << " of " << probe.pages << " scratch pages written after a GPU read, " << probe.writtenAfterFill << " after a GPU write, " << probe.writtenAtImport << " after the import\n";
    Require(probe.writtenByCpu != 0, "(u) a CPU store after the import probe's GPU read was not collected");
    const int udmabuf = context.dmaBufImport ? open("/dev/udmabuf", O_RDWR | O_CLOEXEC) : -1;
    if (udmabuf >= 0) close(udmabuf);
    if (context.dmaBufImport && udmabuf < 0) {
        std::cout << "/dev/udmabuf is not accessible: the dma-buf import probe not tested\n";
    } else if (context.dmaBufImport) {
        const auto dmaBuf = ProbeDmaBufImportWriteProtection(context);
        Require(dmaBuf.failure == nullptr, std::string("(u) the dma-buf import probe failed at ") + (dmaBuf.failure != nullptr ? dmaBuf.failure : "") + " (" + std::to_string(static_cast<int>(dmaBuf.result)) + ")");
        Require(dmaBuf.writtenByCpu != 0, "(u) a CPU store into a dma-buf imported scratch range was not collected");
        std::cout << "dma-buf import probe: " << dmaBuf.writtenAfterSubmit << " of " << dmaBuf.pages << " scratch pages written after a GPU read, " << dmaBuf.writtenAtImport << " after the import, " << dmaBuf.writtenByCpu << " after a CPU store\n";
    } else {
        std::cout << "dma-buf imports unavailable: the dma-buf import probe not tested\n";
    }
    const auto decided = PrepareImportWatch(context);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{context, decided};
    constexpr std::size_t unit = 65536;
    const auto remap = [](void* block, std::size_t bytes) {
        munmap(block, bytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
        Require(mmap(block, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == block, "(u) cannot map the test block again");
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
    };
    const auto registerRange = [](void* block, std::size_t bytes) {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    };
    const auto unregisterRange = [](void* block) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    };
    SetImportWatch(context, ImportWatch::Unwatch);
    {
        constexpr std::size_t bytes = 3 * unit;
        void* block = AllocateWatched(bytes, unit);
        const auto address = reinterpret_cast<std::uint64_t>(block);
        std::memset(block, 0x11, bytes);
        const auto before = CollectWrites(address, bytes);
        Require(before != 0, "(u) a watched block is not collected");
        BumpCollectEpoch();
        Require(CollectWrites(address, 2 * unit) != 0, "(u) the memoized collect of a watched block failed");
        registerRange(block, 2 * unit);
        const auto* import = HostImportFor(context, address, 2 * unit);
        Require(import != nullptr && import->unwatched, "(u) an import made under the unwatch decision is not marked unwatched");
        Require(!Watched(address, 2 * unit) && !Watched(address + unit, 4096), "(u) an imported range stays watched");
        Require(CollectWrites(address, 2 * unit) == 0 && CollectWrites(address + 4096, 4096) == 0, "(u) a collect memoized before the import answers for the unwatched range");
        Require(CollectWrites(address + 2 * unit - 4096, 8192) == 0, "(u) a range partly imported is collected");
        Require(Watched(address + 2 * unit, unit), "(u) the unimported neighbour left the watch");
        const auto neighbour = CollectWrites(address + 2 * unit, unit);
        Require(neighbour != 0, "(u) the unimported neighbour is not collected");
        static_cast<volatile std::uint8_t*>(block)[4096] = 0x22;
        Require(!UnchangedSince(address, 2 * unit, before) && !UnchangedSince(address, 4096, TrackerGeneration()) && !UnchangedSinceCollected(address, 4096, TrackerGeneration()), "(u) an unwatched range reports unchanged");
        const std::array<std::uint64_t, 2> generations{before, TrackerGeneration()};
        std::array<std::uint8_t, 2> changed{};
        std::array<std::uint8_t, 2> cpu{};
        Require(!ChangedBlocks(address, 2 * unit, generations, changed, cpu) && changed[0] != 0 && changed[1] != 0, "(u) the stamps of an unwatched range are offered as tracked");
        Require(MarkWritten(address, unit) == 0, "(u) a GPU write into an unwatched range claims a generation");
        Require(MarkWritten(address + 2 * unit, 4096) != 0 && !UnchangedSince(address + 2 * unit, unit, neighbour), "(u) a GPU write into the watched neighbour is not stamped");
        unregisterRange(block);
        remap(block, 2 * unit);
        Require(Watched(address, 2 * unit), "(u) a remapped range is not watched");
        registerRange(block, 2 * unit);
        const auto* kept = HostImportFor(context, address, 2 * unit);
        Require(kept != nullptr && kept->unwatched && !Watched(address, 2 * unit) && CollectWrites(address, 2 * unit) == 0, "(u) an import kept over a remap left the range watched");
        unregisterRange(block);
        remap(block, 2 * unit);
        registerRange(block, unit);
        const auto* again = HostImportFor(context, address, unit);
        Require(again != nullptr && again->unwatched && again->bytes == unit && !Watched(address, unit), "(u) a re-import after an unmap did not follow the decision");
        Require(Watched(address + unit, unit) && CollectWrites(address + unit, unit) != 0, "(u) the unregistered rest of a remapped range is not watched");
        unregisterRange(block);
        Require(HostImportFor(context, address, unit) == nullptr, "(u) an unregistered range still imports");
        ReleaseWatched(block, bytes);
    }
    SetImportWatch(context, ImportWatch::Watch);
    {
        constexpr std::size_t bytes = unit;
        void* block = AllocateWatched(bytes, unit);
        const auto address = reinterpret_cast<std::uint64_t>(block);
        registerRange(block, bytes);
        const auto* import = HostImportFor(context, address, bytes);
        Require(import != nullptr && !import->unwatched && Watched(address, bytes), "(u) an import made under the watch decision left the watch");
        const auto generation = CollectWrites(address, bytes);
        Require(generation != 0 && UnchangedSince(address, bytes, generation), "(u) a watched import is not collected");
        Require(MarkWritten(address, 4096) != 0 && !UnchangedSince(address, bytes, generation), "(u) a GPU write into a watched import is not stamped");
        unregisterRange(block);
        HostImportFor(context, address, bytes);
        ReleaseWatched(block, bytes);
    }
#endif
}

void staleGenerationTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
#ifdef _WIN32
    static_cast<void>(recorder);
    static_cast<void>(base);
    std::cout << "a range leaving the watch: Linux write watch only\n";
#else
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: a range leaving the watch not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: a range leaving the watch not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    const auto decided = PrepareImportWatch(base);
    SetImportWatch(base, ImportWatch::Unwatch);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{base, decided};
    auto* texels = static_cast<std::uint8_t*>(block);
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(texels + surfaceBytes, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    {
        Require(AgcDriver::GuestMemory::Watched(address, bytes), "(s) the block is not watched before the storage image takes its generation");
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto commands = recorder.Commands();
        recorder.Keep(image);
        Require(image->Generation() != 0, "(s) the storage image took no generation while its memory was watched");
        const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        image->MarkDirty();
        const auto* import = HostImportFor(base, address, bytes);
        if (import == nullptr) {
            std::cout << "host import of the stale generation block refused: a range leaving the watch not tested\n";
            recorder.Sync();
            return;
        }
        Require(!AgcDriver::GuestMemory::Watched(address, bytes), "(s) the imported range stayed watched");
        image->Refresh();
        Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto copyCommands = recorder.Commands();
        RecordMemoryBarrier(context, copyCommands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {side, side, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(copyCommands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, copyCommands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        Require(StorageTexture::FindPending(address, surfaceBytes) == image, "(s) the image is not pending after the refresh");
        Require(StorageTexture::FlushPending(address, surfaceBytes, nullptr, "stale generation test"), "(s) the pending image was not flushed");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        const auto pixels = readback.Bytes();
        constexpr std::array<std::uint8_t, 4> expected{255, 0, 0, 255};
        for (std::size_t i = 0; i < pixels.size(); ++i) Require(std::to_integer<std::uint8_t>(pixels[i]) == expected[i % 4], "(s) results pending when their memory left the watch were dropped");
        for (std::size_t i = 0; i < surfaceBytes; ++i) Require(texels[i] == expected[i % 4], "(s) results pending when their memory left the watch did not reach it");
    }
    recorder.Sync();
#endif
}

// A write-back stores the texels of an image and leaves every other byte of the surface's memory
// as it was: the padding of the tile blocks the surface covers only partly (its last block column
// and row), the tail block of a mip chain, the bytes between the mips and the pitch padding of a
// linear surface. Another surface over the memory, or the CPU, reads those bytes. Each case clears
// every mip of an image over memory holding a pattern no texel equals, stores it and reads the
// memory back (watched memory through the flush hook, after a store into the import's unit shadow):
// each 4-byte word is the clear texel or the pattern, and the texels are exactly the surface's.
// The 64 KiB-aligned tiled cases store through block windows, the others whole layers.
void writeBackPaddingTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: write-back padding not tested\n";
        return;
    }
    struct Case {
        const char* name;
        TextureTileMode tileMode;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t mipCount;
        std::uint64_t offset;
    };
    constexpr Case cases[] = {
        {"64 KiB R_X 200x150", TextureTileMode::kR64KBX, 200, 150, 1, 0},
        {"64 KiB R_X 200x150, 3 mips", TextureTileMode::kR64KBX, 200, 150, 3, 0},
        {"64 KiB R_X 520x260, 6 mips", TextureTileMode::kR64KBX, 520, 260, 6, 0},
        {"64 KiB R_X 200x150 off a 64 KiB boundary, 3 mips", TextureTileMode::kR64KBX, 200, 150, 3, 4096},
        {"4 KiB S 100x70, 2 mips", TextureTileMode::kStandard4KB, 100, 70, 2, 0},
        {"linear 200x150", TextureTileMode::kLinear, 200, 150, 1, 0},
    };
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    std::string failures;
    for (const auto& test : cases) {
        const std::string what = std::string(watched ? "(watched) " : "") + test.name;
        GuestTextureResource resource{};
        resource.width = test.width;
        resource.height = test.height;
        resource.mipCount = test.mipCount;
        resource.tileMode = test.tileMode;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        const auto surfaceBytes = static_cast<std::size_t>(geometry.guestBytes);
        std::size_t texelCount = 0;
        for (const auto& mip : geometry.mips) texelCount += static_cast<std::size_t>(mip.width) * mip.height;
        // The surface, then a 64 KiB unit past it that no store may reach.
        const auto bytes = (static_cast<std::size_t>(test.offset) + surfaceBytes + 65535) / 65536 * 65536 + 65536;
        void* block = nullptr;
        if (watched) {
            block = AllocateWatched(bytes, 65536);
            if (block == nullptr) {
                std::cout << "no write watching: write-back padding in watched memory not tested\n";
                return;
            }
        } else {
#ifdef _WIN32
            block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
            block = std::aligned_alloc(65536, bytes);
#endif
        }
        Require(block != nullptr, "cannot allocate the write-back padding block");
        auto* memory = static_cast<std::uint8_t*>(block);
        for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + test.offset;
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(block, bytes, true, true, true);
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            bool watched;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                if (watched) ReleaseWatched(block, bytes);
            }
        } unregister{base, block, blockAddress, bytes, watched};
        if (HostImportFor(base, blockAddress, bytes) == nullptr) {
            std::cout << "host import of the write-back padding block refused: write-back padding not tested\n";
            return;
        }
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        resource.baseAddress = address;
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, test.mipCount, 0, 1};
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
            image->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            std::vector<std::byte> read(bytes);
            if (watched) AgcDriver::GuestMemory::Read(blockAddress, read);
            else std::memcpy(read.data(), memory, bytes);
            std::size_t texels = 0, garbage = 0, outside = 0;
            for (std::size_t i = 0; i < bytes; i += 4) {
                const bool inside = i >= test.offset && i < test.offset + surfaceBytes;
                bool isTexel = inside, isPattern = true;
                for (std::size_t c = 0; c < 4; ++c) {
                    const auto byte = std::to_integer<std::uint8_t>(read[i + c]);
                    isTexel = isTexel && byte == red[c];
                    isPattern = isPattern && byte == pattern(i + c);
                }
                if (isTexel) ++texels;
                else if (!isPattern) ++(inside ? garbage : outside);
            }
            if (garbage != 0 || outside != 0 || texels != texelCount) {
                failures += what + ": the write-back stored " + std::to_string(texels) + " texels of " + std::to_string(texelCount) + " and changed " + std::to_string(garbage) + " words no texel holds (" + std::to_string(outside) + " past the surface)\n";
            }
        }
        recorder.Sync();
    }
    if (!failures.empty()) throw std::runtime_error("write-back padding:\n" + failures);
    std::cout << "write-back padding" << (watched ? " (watched)" : "") << ": ok\n";
}

void splitWriteBackTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0 || 4096 % base.hostImportAlignment != 0) {
        std::cout << "host imports unavailable: the split write-back not tested\n";
        return;
    }
    struct Case {
        const char* name;
        TextureTileMode tileMode;
        std::uint32_t mipCount;
        std::uint64_t offset;
        std::uint64_t split;
        bool gpu;
        bool direct;
        bool twoAllocations;
        bool secondStore;
    };
    const Case cases[] = {
        {"64 KiB R_X 200x150, 3 mips, ranges split at 128 KiB", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, false, false},
        {"64 KiB R_X 200x150, 3 mips, off a 64 KiB boundary, ranges split at 132 KiB", TextureTileMode::kR64KBX, 3, 4096, 135168, true, true, false, false},
        {"linear 200x150, ranges split at 64 KiB", TextureTileMode::kLinear, 1, 0, 65536, true, true, false, false},
        {"64 KiB R_X 200x150, 3 mips, second range of another access", TextureTileMode::kR64KBX, 3, 0, 131072, false, false, false, false},
        {"64 KiB R_X 200x150, 3 mips, ranges split inside a 64 KiB unit", TextureTileMode::kR64KBX, 3, 0, 135168, true, true, false, false},
        {"64 KiB R_X 200x150, 3 mips, two adjacent allocations split at 128 KiB", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, true, false},
        {"64 KiB R_X 200x150, 3 mips, ranges split at 128 KiB, a second store outside the write watch", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, false, true},
        {"linear 200x150, ranges split at 64 KiB, a second store outside the write watch", TextureTileMode::kLinear, 1, 0, 65536, true, true, false, true},
    };
    SetHostImportWindow(64ull << 20u);
    struct Window {
        ~Window() { SetHostImportWindow(0); }
    } window;
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    std::string failures;
    for (const auto& test : cases) {
        GuestTextureResource resource{};
        resource.width = 200;
        resource.height = 150;
        resource.mipCount = test.mipCount;
        resource.tileMode = test.tileMode;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        const auto surfaceBytes = static_cast<std::size_t>(geometry.guestBytes);
        Require(test.offset < test.split && test.split < test.offset + surfaceBytes, "the split test's ranges do not divide its surface");
        std::size_t texelCount = 0;
        for (const auto& mip : geometry.mips) texelCount += static_cast<std::size_t>(mip.width) * mip.height;
        const auto bytes = (static_cast<std::size_t>(test.offset) + surfaceBytes + 65535) / 65536 * 65536 + 65536;
#ifdef _WIN32
        void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* block = std::aligned_alloc(65536, bytes);
#endif
        Require(block != nullptr, "cannot allocate the split write-back block");
        auto* memory = static_cast<std::uint8_t*>(block);
        for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + test.offset;
        {
            GuestAllocations::Mutation mutation;
            if (test.twoAllocations) {
                mutation.Add(block, static_cast<std::size_t>(test.split), true, true, true);
                mutation.Add(memory + test.split, static_cast<std::size_t>(bytes - test.split), true, true, test.gpu);
            } else {
                mutation.Add(block, bytes, true, true, true);
                mutation.Protect(memory + test.split, static_cast<std::size_t>(bytes - test.split), true, true, test.gpu, [] {});
            }
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            void* second;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                    if (second != nullptr) mutation.Remove(second);
                }
                HostImportFor(context, address, bytes);
#ifdef _WIN32
                VirtualFree(block, 0, MEM_RELEASE);
#else
                std::free(block);
#endif
            }
        } unregister{base, block, blockAddress, bytes, test.twoAllocations ? static_cast<void*>(memory + test.split) : nullptr};
        Require(HostImportFor(base, address, surfaceBytes) == nullptr, std::string(test.name) + ": one import covers both ranges");
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        resource.baseAddress = address;
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, test.mipCount, 0, 1};
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
            image->WriteBack();
            const bool direct = !image->GuestSnapshotValid();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            std::size_t texels = 0, garbage = 0, outside = 0;
            for (std::size_t i = 0; i < bytes; i += 4) {
                const bool inside = i >= test.offset && i < test.offset + surfaceBytes;
                bool isTexel = inside, isPattern = true;
                for (std::size_t c = 0; c < 4; ++c) {
                    isTexel = isTexel && memory[i + c] == red[c];
                    isPattern = isPattern && memory[i + c] == pattern(i + c);
                }
                if (isTexel) ++texels;
                else if (!isPattern) ++(inside ? garbage : outside);
            }
            if (direct != test.direct) failures += std::string(test.name) + ": the write-back took the " + (direct ? "GPU-direct" : "CPU") + " path\n";
            if (direct && !(HostImportCovers(base, blockAddress, static_cast<std::size_t>(test.split)) && HostImportCovers(base, blockAddress + test.split, static_cast<std::size_t>(bytes - test.split)))) failures += std::string(test.name) + ": a range has no import after the GPU-direct write-back\n";
            if (garbage != 0 || outside != 0 || texels != texelCount) failures += std::string(test.name) + ": the write-back stored " + std::to_string(texels) + " texels of " + std::to_string(texelCount) + " and changed " + std::to_string(garbage) + " words no texel holds (" + std::to_string(outside) + " past the surface)\n";
            else if (test.secondStore) {
                const auto cpuPage = static_cast<std::size_t>(test.split) + 16384;
                Require(cpuPage + 4096 <= test.offset + surfaceBytes, std::string(test.name) + ": the CPU's page lies past the surface");
                if (AgcDriver::GuestMemory::Watched(address, surfaceBytes)) failures += std::string(test.name) + ": the surface is write-watched\n";
                if (!image->GuestSnapshotValid()) failures += std::string(test.name) + ": the first write-back left no saved bytes\n";
                for (std::size_t i = cpuPage; i < cpuPage + 4096; ++i) memory[i] = static_cast<std::uint8_t>(~pattern(i));
                const auto again = recorder.Commands();
                const VkClearColorValue greenValue{{0.0f, 1.0f, 0.0f, 1.0f}};
                RecordMemoryBarrier(context, again, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(again, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &greenValue, 1, &range);
                RecordMemoryBarrier(context, again, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                constexpr std::array<std::uint8_t, 4> green{0, 255, 0, 255};
                std::size_t kept = 0, greenBelow = 0, greenAbove = 0;
                for (std::size_t i = cpuPage; i < cpuPage + 4096; ++i) kept += memory[i] == static_cast<std::uint8_t>(~pattern(i)) ? 1 : 0;
                for (std::size_t i = test.offset; i < test.offset + surfaceBytes; i += 4) {
                    if (i >= cpuPage && i < cpuPage + 4096) continue;
                    if (std::equal(green.begin(), green.end(), memory + i)) ++(i < test.split ? greenBelow : greenAbove);
                }
                if (kept != 4096) failures += std::string(test.name) + ": the second write-back overwrote " + std::to_string(4096 - kept) + " of the 4096 bytes the CPU wrote after the first\n";
                if (greenBelow == 0 || greenAbove == 0) failures += std::string(test.name) + ": the second write-back stored " + std::to_string(greenBelow) + " texels below the split and " + std::to_string(greenAbove) + " above it\n";
            }
        }
        recorder.Sync();
    }
    if (!failures.empty()) throw std::runtime_error("split write-back:\n" + failures);
    std::cout << "split write-back: ok\n";
}

void singlePassTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0 || !base.singlePassStorage) {
        std::cout << "host imports or storage access without a format unavailable: single-pass storage moves not tested\n";
        return;
    }
    struct Case {
        const char* name;
        std::uint32_t format;
        TextureTileMode tileMode;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t mipCount;
        std::uint32_t pipeBankXor = 0;
    };
    constexpr Case cases[] = {
        {"R8 64 KiB R_X 200x150", 1, TextureTileMode::kR64KBX, 200, 150, 1},
        {"R16 64 KiB R_X 200x150, 3 mips", 7, TextureTileMode::kR64KBX, 200, 150, 3},
        {"RGBA8 64 KiB R_X 520x260, 6 mips", 56, TextureTileMode::kR64KBX, 520, 260, 6},
        {"RGBA16F 64 KiB R_X 384x256", 71, TextureTileMode::kR64KBX, 384, 256, 1},
        {"RGBA16F 64 KiB R_X 256x200, 4 mips", 71, TextureTileMode::kR64KBX, 256, 200, 4},
        {"RGBA32F 64 KiB R_X 136x72, 2 mips", 77, TextureTileMode::kR64KBX, 136, 72, 2},
        {"RGBA8 64 KiB S 300x170", 56, TextureTileMode::kStandard64KB, 300, 170, 1},
        {"RGBA16F 4 KiB S 100x70, 2 mips", 71, TextureTileMode::kStandard4KB, 100, 70, 2},
        {"R8 256 B S 90x40", 1, TextureTileMode::kStandard256B, 90, 40, 1},
        {"RGBA8 64 KiB R_X 520x260, 2 mips, pipe/bank XOR 5", 56, TextureTileMode::kR64KBX, 520, 260, 2, 5},
    };
    std::string failures;
    std::size_t tested = 0;
    for (const auto& test : cases) {
        if (!StorageFormatAvailable(base, test.format)) continue;
        const std::string what = std::string(watched ? "(watched) " : "") + test.name;
        GuestTextureResource resource{};
        resource.width = test.width;
        resource.height = test.height;
        resource.mipCount = test.mipCount;
        resource.tileMode = test.tileMode;
        resource.dimension = TextureDimension::k2D;
        resource.format = test.format;
        resource.pipeBankXor = test.pipeBankXor;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        const auto elementBytes = BytesPerElement(test.format);
        const auto bytes = (static_cast<std::size_t>(geometry.guestBytes) + 65535) / 65536 * 65536 + 65536;
        struct Result {
            std::vector<std::uint8_t> uploaded, memory, reuploaded;
            std::uint64_t moves = 0;
        };
        std::array<Result, 2> results;
        bool skipped = false;
        for (int mode = 0; mode < 2 && !skipped; ++mode) {
            auto context = base;
            context.singlePassStorage = mode == 1;
            void* block = nullptr;
            if (watched) {
                block = AllocateWatched(bytes, 65536);
            } else {
#ifdef _WIN32
                block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
                block = std::aligned_alloc(65536, bytes);
#endif
            }
            if (block == nullptr) {
                std::cout << "no write watching: single-pass storage moves in watched memory not tested\n";
                return;
            }
            auto* memory = static_cast<std::uint8_t*>(block);
            std::uint32_t seed = 0x13579bdfu;
            const auto random = [&] {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return static_cast<std::uint8_t>(seed >> 9);
            };
            for (std::size_t i = 0; i < bytes; ++i) memory[i] = random();
            const auto address = reinterpret_cast<std::uint64_t>(block);
            {
                GuestAllocations::Mutation mutation;
                mutation.Add(block, bytes, true, true, true);
            }
            struct Unregister {
                const Context& context;
                void* block;
                std::uint64_t address;
                std::size_t bytes;
                bool watched;
                ~Unregister() {
                    {
                        GuestAllocations::Mutation mutation;
                        mutation.Remove(block);
                    }
                    HostImportFor(context, address, bytes);
                    if (watched) ReleaseWatched(block, bytes);
#ifdef _WIN32
                    else VirtualFree(block, 0, MEM_RELEASE);
#else
                    else std::free(block);
#endif
                }
            } unregister{base, block, address, bytes, watched};
            if (HostImportFor(base, address, bytes) == nullptr) {
                std::cout << "host import refused: single-pass storage moves not tested\n";
                return;
            }
            TextureDetiler detiler(base);
            context.detiler = &detiler;
            resource.baseAddress = address;
            std::size_t linearBytes = 0;
            for (const auto& mip : geometry.mips) linearBytes += static_cast<std::size_t>(mip.width) * mip.height * elementBytes;
            const auto readImage = [&](const StorageTexture& image) {
                Buffer readback(context, linearBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                const auto commands = recorder.Commands();
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                std::vector<VkBufferImageCopy> copies;
                std::size_t at = 0;
                for (std::uint32_t level = 0; level < geometry.mips.size(); ++level) {
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = at;
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                    copy.imageExtent = {geometry.mips[level].width, geometry.mips[level].height, 1};
                    copies.push_back(copy);
                    at += static_cast<std::size_t>(geometry.mips[level].width) * geometry.mips[level].height * elementBytes;
                }
                context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), static_cast<std::uint32_t>(copies.size()), copies.data());
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                const auto read = readback.Bytes();
                return std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(read.data()), reinterpret_cast<const std::uint8_t*>(read.data()) + linearBytes);
            };
            std::vector<std::uint8_t> texels(linearBytes);
            for (auto& value : texels) value = random();
            const auto movesBefore = StorageTexture::SinglePassMoves();
            {
                auto image = StorageTexture::Create(context, detiler, resource, 0);
                recorder.Keep(image);
                results[mode].uploaded = readImage(*image);
                Buffer staging(context, linearBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                std::memcpy(staging.Bytes().data(), texels.data(), linearBytes);
                const auto commands = recorder.Commands();
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                std::vector<VkBufferImageCopy> copies;
                std::size_t at = 0;
                for (std::uint32_t level = 0; level < geometry.mips.size(); ++level) {
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = at;
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                    copy.imageExtent = {geometry.mips[level].width, geometry.mips[level].height, 1};
                    copies.push_back(copy);
                    at += static_cast<std::size_t>(geometry.mips[level].width) * geometry.mips[level].height * elementBytes;
                }
                context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, staging.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(copies.size()), copies.data());
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                auto again = StorageTexture::Create(context, detiler, resource, 0);
                recorder.Keep(again);
                results[mode].reuploaded = readImage(*again);
                std::vector<std::byte> read(bytes);
                if (watched) AgcDriver::GuestMemory::Read(address, read);
                else {
                    StorageTexture::FlushPending(address, bytes, nullptr, "test", PublishScope::Whole);
                    recorder.Submit();
                    device.WaitQueue();
                    recorder.Sync();
                    std::memcpy(read.data(), memory, bytes);
                }
                results[mode].memory.assign(reinterpret_cast<const std::uint8_t*>(read.data()), reinterpret_cast<const std::uint8_t*>(read.data()) + bytes);
            }
            recorder.Sync();
            results[mode].moves = StorageTexture::SinglePassMoves() - movesBefore;
            if (results[mode].reuploaded != texels) failures += what + (mode == 1 ? " (single pass)" : " (copies)") + ": the second image does not hold the texels written back\n";
        }
        if (results[0].moves != 0 || results[1].moves < 3) failures += what + ": single-pass moves " + std::to_string(results[0].moves) + " with the copies and " + std::to_string(results[1].moves) + " in single pass (want 0 and at least 3)\n";
        if (results[0].uploaded != results[1].uploaded) failures += what + ": the first upload differs between the copies and single pass\n";
        if (results[0].memory != results[1].memory) {
            std::size_t differ = 0;
            for (std::size_t i = 0; i < results[0].memory.size(); ++i) differ += results[0].memory[i] != results[1].memory[i];
            failures += what + ": guest memory after the write-back differs in " + std::to_string(differ) + " bytes between the copies and single pass\n";
        }
        ++tested;
    }
    if (!failures.empty()) throw std::runtime_error("single-pass storage moves:\n" + failures);
    std::cout << "single-pass storage moves" << (watched ? " (watched)" : "") << ": ok (" << tested << " surfaces)\n";
}

class StorageUploadProbe {
public:
    StorageUploadProbe(Context& context, bool failView) : context(context), resolver(context.deviceProc), failView(failView) {
        Require(active == nullptr, "nested storage upload probe");
        active = this;
        context.deviceProc = resolve;
    }
    ~StorageUploadProbe() {
        for (const auto view : destroyedViews) function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
        if (imageDestroyed) function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
        if (memoryFreed) function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        context.deviceProc = resolver;
        active = nullptr;
    }
    VkImage Image() const { return image; }
    bool Destroyed() const { return imageDestroyed || memoryFreed; }
    bool Released() const { return imageDestroyed && memoryFreed; }
    bool FailedView() const { return failedView; }
    std::uint32_t Clears() const { return clears; }

private:
    template<typename TFunction>
    TFunction function(const char* name) const { return reinterpret_cast<TFunction>(resolver(context.device, name)); }
    static VKAPI_ATTR VkResult VKAPI_CALL createImage(VkDevice device, const VkImageCreateInfo* info, const VkAllocationCallbacks* callbacks, VkImage* created) {
        const auto result = active->function<PFN_vkCreateImage>("vkCreateImage")(device, info, callbacks, created);
        if (result == VK_SUCCESS) active->image = *created;
        return result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL bindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset) {
        const auto result = active->function<PFN_vkBindImageMemory>("vkBindImageMemory")(device, image, memory, offset);
        if (result == VK_SUCCESS && image == active->image) active->memory = memory;
        return result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL createView(VkDevice device, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* callbacks, VkImageView* created) {
        if (info->image == active->image && active->failView && active->clears != 0) {
            active->failedView = true;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        const auto result = active->function<PFN_vkCreateImageView>("vkCreateImageView")(device, info, callbacks, created);
        if (result == VK_SUCCESS && info->image == active->image) active->views.push_back(*created);
        return result;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyView(VkDevice device, VkImageView view, const VkAllocationCallbacks* callbacks) {
        if (std::find(active->views.begin(), active->views.end(), view) != active->views.end()) active->destroyedViews.push_back(view);
        else active->function<PFN_vkDestroyImageView>("vkDestroyImageView")(device, view, callbacks);
    }
    static VKAPI_ATTR void VKAPI_CALL destroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* callbacks) {
        if (image == active->image) active->imageDestroyed = true;
        else active->function<PFN_vkDestroyImage>("vkDestroyImage")(device, image, callbacks);
    }
    static VKAPI_ATTR void VKAPI_CALL freeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
        if (memory == active->memory) active->memoryFreed = true;
        else active->function<PFN_vkFreeMemory>("vkFreeMemory")(device, memory, callbacks);
    }
    static VKAPI_ATTR void VKAPI_CALL clearImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearColorValue* value, std::uint32_t count, const VkImageSubresourceRange* ranges) {
        if (image == active->image) ++active->clears;
        active->function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image, layout, value, count, ranges);
    }
    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL resolve(VkDevice device, const char* name) {
        if (std::strcmp(name, "vkCreateImage") == 0) return reinterpret_cast<PFN_vkVoidFunction>(createImage);
        if (std::strcmp(name, "vkBindImageMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(bindImageMemory);
        if (std::strcmp(name, "vkCreateImageView") == 0) return reinterpret_cast<PFN_vkVoidFunction>(createView);
        if (std::strcmp(name, "vkDestroyImageView") == 0) return reinterpret_cast<PFN_vkVoidFunction>(destroyView);
        if (std::strcmp(name, "vkDestroyImage") == 0) return reinterpret_cast<PFN_vkVoidFunction>(destroyImage);
        if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(freeMemory);
        if (std::strcmp(name, "vkCmdClearColorImage") == 0) return reinterpret_cast<PFN_vkVoidFunction>(clearImage);
        return active->resolver(device, name);
    }
    Context& context;
    PFN_vkGetDeviceProcAddr resolver;
    bool failView;
    bool failedView = false;
    bool imageDestroyed = false;
    bool memoryFreed = false;
    std::uint32_t clears = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::vector<VkImageView> views;
    std::vector<VkImageView> destroyedViews;
    static inline StorageUploadProbe* active = nullptr;
};

void storageUploadLifetimeTests(const Device& device, Recorder& previous, bool failView) {
    auto context = device.GetContext();
    context.hostImportAlignment = 0;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.width = 64;
    resource.height = 64;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    const auto guestBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
    std::vector<std::byte> texels(guestBytes + 256, std::byte{0x55});
    std::vector<std::byte> keys(guestBytes / 256, std::byte{0});
    resource.baseAddress = (reinterpret_cast<std::uintptr_t>(texels.data()) + 255) & ~std::uintptr_t{255};
    resource.dccAddress = reinterpret_cast<std::uintptr_t>(keys.data());
    DccKeyProof proof;
    Require(ProvedClearKeys(resource, guestBytes, proof) == DccKeys::Clear0000, "storage lifetime fixture has no DCC clear");
    StorageUploadProbe probe(context, failView);
    Buffer readback(device.GetContext(), resource.width * resource.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(readback.Bytes().data(), 0x55, readback.Bytes().size());
    std::weak_ptr<StorageTexture> lifetime;
    bool premature = false;
    bool retained = true;
    bool threw = false;
    {
        Recorder recorder(context);
        recorder.Activate();
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            lifetime = image;
            image.reset();
            retained = !lifetime.expired();
        }
        premature = probe.Destroyed();
        Require(probe.Clears() == 1 && recorder.Recording(), "storage creation did not record its initial clear");
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {resource.width, resource.height, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, probe.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        try {
            recorder.Sync();
        } catch (const std::exception& error) {
            if (!probe.FailedView() || std::string_view(error.what()).find("vkCreateImageView storage") == std::string_view::npos) throw;
            threw = true;
        }
    }
    previous.Activate();
    Require(threw == failView && probe.FailedView() == failView, "storage view failure injection did not fail the submission of the batch that recorded the image");
    const auto expected = failView ? std::byte{0x55} : std::byte{0};
    Require(std::all_of(readback.Bytes().begin(), readback.Bytes().end(), [&](std::byte value) { return value == expected; }), failView ? "a batch ran although a view it was submitted after failed on the view pool" : "recorded storage clear did not reach every texel");
    Require(lifetime.expired() && probe.Released(), "storage upload resources outlived recorder teardown");
    Require(!premature && retained, failView ? "failed storage creation released an image still referenced by its recorded clear" : "storage creation did not retain its image through the recorded clear");
    std::cout << "Storage upload lifetime" << (failView ? " after view failure" : " after caller release") << ": passed\n";
}

void refreshProofTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: refresh proofs not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: refresh proofs not tested\n";
        return;
    }
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0xff, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr || !AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "no watched host import: refresh proofs not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    const auto proved = [] { return StorageTexture::RefreshesProved(); };
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto draw = [&](VkClearColorValue value) {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        };
        const auto settle = [&] {
            image->Refresh();
            image->Refresh();
        };
        const auto expectProved = [&](const char* what) {
            const auto before = proved();
            Require(image->Refresh(), what);
            Require(proved() == before + 1, what);
        };
        const auto expectFull = [&](const char* what) {
            const auto before = proved();
            image->Refresh();
            Require(proved() == before, what);
        };
        settle();
        expectProved("an unchanged surface was not proved current");
        expectProved("a proved surface was not proved again");
        draw({{1.0f, 0.0f, 0.0f, 1.0f}});
        settle();
        expectProved("a target holding its own pending results was not proved current");
        draw({{0.0f, 1.0f, 0.0f, 1.0f}});
        expectProved("a draw into an already pending target broke its proof");
        texels[0] = 0x11;
        expectFull("a CPU write of the surface was answered from the proof");
        settle();
        expectProved("the proof was not taken again after a CPU write");
        AgcDriver::GuestMemory::MarkWritten(address + 65536, 4096);
        expectFull("a driver store into the surface was answered from the proof");
        settle();
        AgcDriver::GuestMemory::MarkWritten(resource.dccAddress, keyCount);
        expectFull("a key store over the surface's keys was answered from the proof");
        settle();
        expectProved("the proof was not taken again after a key store");
        auto other = resource;
        other.dccAddress = 0;
        auto pending = StorageTexture::Create(context, detiler, other, 0);
        recorder.Keep(pending);
        pending->MarkDirty();
        expectFull("another image's pending results over the surface were answered from the proof");
        settle();
        expectProved("the proof was not taken again after another image's results were stored");
        pending->MarkDirty();
        expectFull("another image's pending results were answered from the proof");
        Require(!AgcDriver::Graphics::PendingStorageOverlaps(address, surfaceBytes, image.get()), "the full refresh left another image's results pending");
        auto fresh = StorageTexture::Create(context, detiler, resource, 0);
        const auto before = proved();
        fresh->Refresh();
        Require(proved() == before, "a new image of the surface was answered from another image's proof");
        std::memset(keys, 0x00, keyCount);
        expectFull("a fast clear of the keys was answered from the proof");
        recorder.Sync();
    }
    recorder.Sync();
}


void unchangedCpuStampTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::uint64_t offset = 0x2000;
    constexpr std::size_t bytes = 6 * 65536;
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    for (const bool imported : {false, true}) {
#ifdef _WIN32
        if (imported) {
            std::cout << "unchanged CPU stamps over a watched host import: Linux write watch only\n";
            continue;
        }
#endif
        if (imported && base.hostImportAlignment == 0) {
            std::cout << "host imports unavailable: unchanged CPU stamps over a watched host import not tested\n";
            continue;
        }
        const auto decided = PrepareImportWatch(base);
        if (imported) SetImportWatch(base, ImportWatch::Watch);
        struct Restore {
            const Context& context;
            ImportWatch decided;
            ~Restore() { SetImportWatch(context, decided); }
        } restore{base, decided};
        for (const bool changed : {false, true}) {
            const std::string what = std::string(changed ? "a CPU store that changed a byte" : "a CPU store that left the bytes as they were") + (imported ? " over a watched host import" : " over watched memory");
            void* block = AllocateWatched(bytes, 65536);
            if (block == nullptr) {
                std::cout << "no write watching: unchanged CPU stamps not tested\n";
                return;
            }
            auto* memory = static_cast<std::uint8_t*>(block);
            for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
            const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
            const auto address = blockAddress + offset;
            {
                GuestAllocations::Mutation mutation;
                mutation.Add(block, bytes, true, true, true);
            }
            struct Unregister {
                const Context& context;
                void* block;
                std::uint64_t address;
                std::size_t bytes;
                ~Unregister() {
                    {
                        GuestAllocations::Mutation mutation;
                        mutation.Remove(block);
                    }
                    HostImportFor(context, address, bytes);
                    ReleaseWatched(block, bytes);
                }
            } unregister{base, block, blockAddress, bytes};
            AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
            TextureDetiler detiler(base);
            auto context = base;
            if (!imported) context.hostImportAlignment = 0;
            context.detiler = &detiler;
            GuestTextureResource resource{};
            resource.baseAddress = address;
            resource.width = side;
            resource.height = side;
            resource.mipCount = 1;
            resource.tileMode = TextureTileMode::kLinear;
            resource.dimension = TextureDimension::k2D;
            resource.format = 56;
            resource.dstSelX = 4;
            resource.dstSelY = 5;
            resource.dstSelZ = 6;
            resource.dstSelW = 7;
            Require(DescribeSurface(resource).guestBytes == surfaceBytes, "(c) unexpected surface size");
            {
                auto image = StorageTexture::Create(context, detiler, resource, 0);
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                const auto commands = recorder.Commands();
                recorder.Keep(image);
                const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                for (std::size_t i = 0; i < 4096; ++i) memory[offset + i] = pattern(offset + i);
                if (changed) memory[offset + 16] = static_cast<std::uint8_t>(~pattern(offset + 16));
                AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
                Require(AgcDriver::GuestMemory::Watched(address, surfaceBytes) && (HostImportFor(context, address, surfaceBytes) != nullptr) == imported, "(c) " + what + ": the image is not on the path under test");
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                std::vector<std::byte> read(bytes);
                AgcDriver::GuestMemory::Read(blockAddress, read);
                const auto texel = [&](std::size_t at) {
                    for (std::size_t c = 0; c < 4; ++c) {
                        if (std::to_integer<std::uint8_t>(read[at + c]) != red[c]) return false;
                    }
                    return true;
                };
                const auto headEnd = static_cast<std::size_t>(65536);
                std::size_t stored = 0, total = 0;
                for (std::size_t at = offset; at < offset + surfaceBytes; at += 4) {
                    ++total;
                    if (texel(at)) ++stored;
                }
                if (!changed) {
                    if (stored != total) throw std::runtime_error(std::string("unchanged CPU stamps: ") + what + " dropped " + std::to_string(total - stored) + " of " + std::to_string(total) + " texels the GPU wrote");
                } else {
                    Require(std::to_integer<std::uint8_t>(read[offset + 16]) == static_cast<std::uint8_t>(~pattern(offset + 16)), "(c) " + what + ": the changed byte was overwritten by the GPU results");
                    for (std::size_t at = headEnd; at < offset + surfaceBytes; at += 4) Require(texel(at), "(c) " + what + ": GPU results outside the block the CPU changed were dropped");
                }
            }
            recorder.Sync();
        }
    }
    std::cout << "unchanged CPU stamps: ok\n";
}

void keptUnitSettleTests(const Device& device, Recorder& recorder) {
    Require(AgcDriver::GuestMemory::WriteWatched(), "(k) kept-unit settle tests require host write watching");
    const auto& base = device.GetContext();
    constexpr std::uint32_t width = 1024;
    constexpr std::uint32_t height = 512;
    constexpr std::size_t surfaceBytes = std::size_t{width} * height * 4;
    constexpr std::size_t unitBytes = 65536;
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    constexpr std::array<std::uint8_t, 4> blue{0, 0, 255, 255};
    for (const bool direct : {false, true}) {
        const std::string what = direct ? "GPU-mapped direct memory" : "watched memory";
        if (direct && base.hostImportAlignment == 0) {
            std::cout << "host imports unavailable: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        struct Backing {
            Recorder& recorder;
            bool direct;
            void* block = nullptr;
            std::int64_t physical = -1;
            ~Backing() {
                try {
                    recorder.Sync();
                } catch (const std::exception& error) {
                    std::cerr << "(k) the recorded work did not finish, the test memory stays mapped: " << error.what() << '\n';
                    return;
                }
                if (direct) {
                    if (block != nullptr) sceKernelMunmap(block, surfaceBytes);
                    if (physical >= 0) sceKernelReleaseDirectMemory(physical, surfaceBytes);
                    return;
                }
                if (block == nullptr) return;
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                ReleaseWatched(block, surfaceBytes);
            }
        } backing{recorder, direct};
        if (direct) {
            Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, surfaceBytes, unitBytes, 0, &backing.physical) == 0, "(k) allocate the direct memory");
            Require(sceKernelMapDirectMemory(&backing.block, surfaceBytes, 0x33, 0, backing.physical, unitBytes) == 0, "(k) map the direct memory for the GPU");
        } else {
            backing.block = AllocateWatched(surfaceBytes, unitBytes);
            Require(backing.block != nullptr, "(k) no watched block although write watching is on");
            GuestAllocations::Mutation mutation;
            mutation.Add(backing.block, surfaceBytes, true, true, true);
        }
        auto* memory = static_cast<std::uint8_t*>(backing.block);
        std::memset(memory, 0x5a, surfaceBytes);
        const auto address = reinterpret_cast<std::uint64_t>(backing.block);
        AgcDriver::GuestMemory::CollectWritesUncached(address, surfaceBytes);
        TextureDetiler detiler(base);
        auto context = base;
        if (!direct) context.hostImportAlignment = 0;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = width;
        resource.height = height;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kR64KBX;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        Require(DescribeSurface(resource).guestBytes == surfaceBytes, "(k) unexpected surface size");
        if (direct && !AgcDriver::GuestMemory::Watched(address, surfaceBytes)) {
            std::cout << "GPU-mapped direct memory is not write-watched here: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        Require(AgcDriver::GuestMemory::Watched(address, surfaceBytes), "(k) " + what + " is not watched");
        if (direct && HostImportFor(context, address, surfaceBytes) == nullptr) {
            std::cout << "host import of the direct memory refused: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        if (direct && !AgcDriver::GuestMemory::Watched(address, surfaceBytes)) {
            std::cout << "the host import of the direct memory took it out of the write watch (its imports are compared): units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        Require(direct || HostImportFor(context, address, surfaceBytes) == nullptr, "(k) watched memory was imported although host imports are off for it");
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const auto clear = [&](VkClearColorValue value) {
                const auto commands = recorder.Commands();
                recorder.Keep(image);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
            };
            const auto store = [&] {
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
            };
            const auto unitHolds = [&](std::size_t unit, const std::array<std::uint8_t, 4>& texel) {
                std::vector<std::byte> read(unitBytes);
                AgcDriver::GuestMemory::Read(address + unit * unitBytes, read);
                for (std::size_t i = 0; i < unitBytes; ++i) {
                    if (std::to_integer<std::uint8_t>(read[i]) != texel[i % 4]) return false;
                }
                return true;
            };
            clear({{1.0f, 0.0f, 0.0f, 1.0f}});
            memory[16] = 0x77;
            AgcDriver::GuestMemory::CollectWritesUncached(address, surfaceBytes);
            store();
            std::vector<std::byte> head(unitBytes);
            AgcDriver::GuestMemory::Read(address, head);
            Require(std::to_integer<std::uint8_t>(head[16]) == 0x77, "(k) " + what + ": the CPU store into unit 0 was overwritten by results it was newer than");
            for (std::size_t unit = 1; unit < surfaceBytes / unitBytes; ++unit) Require(unitHolds(unit, red), "(k) " + what + ": the first write-back did not store unit " + std::to_string(unit));
            clear({{0.0f, 0.0f, 1.0f, 1.0f}});
            store();
            for (std::size_t unit = 1; unit < surfaceBytes / unitBytes; ++unit) {
                Require(unitHolds(unit, blue), "(k) " + what + ": unit " + std::to_string(unit) + " kept the results of the earlier write-back; that write-back's own store was taken for a CPU write because a neighbouring unit was kept for the CPU");
            }
            AgcDriver::GuestMemory::Read(address, head);
            Require(std::to_integer<std::uint8_t>(head[16]) == 0x77, "(k) " + what + ": the second write-back stored over unit 0, which the CPU wrote after that unit's generation");
        }
        std::cout << "units stored next to a unit kept for the CPU in " << what << ": ok\n";
    }
}

void settledBaselineTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
#ifdef _WIN32
    std::cout << "generation baseline over a pending store: Linux write watch only\n";
    return;
#endif
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: generation baseline over a pending store not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::uint64_t offset = 0x2000;
    constexpr std::size_t bytes = 6 * 65536;
    const auto decided = PrepareImportWatch(base);
    SetImportWatch(base, ImportWatch::Watch);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{base, decided};
    for (const bool pending : {false, true}) {
        void* block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: generation baseline over a pending store not tested\n";
            return;
        }
        std::memset(block, 0x5a, bytes);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + offset;
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(block, bytes, true, true, true);
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                ReleaseWatched(block, bytes);
            }
        } unregister{base, block, blockAddress, bytes};
        AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = side;
        resource.height = side;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kLinear;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto surfaceBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
        Require(HostImportFor(context, address, surfaceBytes) != nullptr && AgcDriver::GuestMemory::Watched(address, surfaceBytes), "(b) the surface is not a watched host import");
        static_cast<void>(recorder.Commands());
        if (pending) recorder.NotePendingWrite(address + 4096, 4096);
        const auto before = recorder.Submissions();
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            recorder.Keep(image);
            if (pending) Require(recorder.Submissions() == before, "(b) an upload over a pending recorded store submitted the open batch to read its generation baseline");
        }
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::cout << "generation baseline over a pending store: ok\n";
}

void importWindowTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& base = device.GetContext();
#ifdef _WIN32
    static_cast<void>(recorder);
    static_cast<void>(base);
    std::cout << "the import window: Linux write watch only\n";
#else
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: the import window not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: the import window not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    Require(ProbeImportWriteProtection(base).failure == nullptr, "(w) the import probe failed");
    const auto decided = PrepareImportWatch(base);
    if (decided == ImportWatch::Unwatch) {
        std::cout << "host imports are compared, not watched: the import window not tested\n";
        return;
    }
    SetImportWatch(base, ImportWatch::Watch);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{base, decided};
    auto* texels = static_cast<std::uint8_t*>(block);
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(texels + surfaceBytes, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    {
        Require(!HostImportCovers(base, address, bytes), "(w) the block is imported before the cache takes its generation");
        const auto cached = CollectWritesUncached(address, surfaceBytes);
        Require(cached != 0 && UnchangedSince(address, surfaceBytes, cached), "(w) the cache's generation is not current before the import");
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto commands = recorder.Commands();
        recorder.Keep(image);
        const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        image->MarkDirty();
        const auto* import = HostImportFor(base, address, bytes);
        if (import == nullptr) {
            std::cout << "host import of the import window block refused: the import window not tested\n";
            recorder.Sync();
            return;
        }
        Require(!import->unwatched && Watched(address, bytes), "(w) an import under the watch decision left the watch");
        CollectWritesUncached(address, surfaceBytes);
        Require(UnchangedSince(address, surfaceBytes, cached), "(w) an import that stored nothing reads as a change to a cache taken before it");
        Require(image->Generation() != 0 && UnchangedSince(address, surfaceBytes, image->Generation()), "(w) an import that stored nothing reads as a change to the pending storage image");
        std::array<std::uint8_t, surfaceBytes / 65536> changed{};
        const std::array<std::uint64_t, surfaceBytes / 65536> generations{cached, cached, cached, cached};
        Require(ChangedBlocks(address, surfaceBytes, generations, changed) && std::all_of(changed.begin(), changed.end(), [](std::uint8_t value) { return value == BlockUnchanged; }), "(w) the pages the import resolved read as written");
        const auto after = CollectWritesUncached(address, surfaceBytes);
        Require(after != 0 && UnchangedSince(address, surfaceBytes, after), "(w) the imported range is not watched again after the import window");
        image->Refresh();
        Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto copyCommands = recorder.Commands();
        RecordMemoryBarrier(context, copyCommands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {side, side, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(copyCommands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, copyCommands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        const auto pixels = readback.Bytes();
        constexpr std::array<std::uint8_t, 4> expected{255, 0, 0, 255};
        for (std::size_t i = 0; i < pixels.size(); ++i) Require(std::to_integer<std::uint8_t>(pixels[i]) == expected[i % 4], "(w) results pending over the import window were dropped");
        std::vector<std::byte> memory(surfaceBytes);
        Read(address, memory);
        for (std::size_t i = 0; i < surfaceBytes; ++i) Require(std::to_integer<std::uint8_t>(memory[i]) == expected[i % 4], "(w) results pending over the import window did not reach guest memory");
        const auto stored = CollectWritesUncached(address, surfaceBytes);
        texels[65536 + 8] = 0x77;
        CollectWritesUncached(address, surfaceBytes);
        Require(WrittenSince(address + 65536, 65536, stored) && !WrittenSince(address, 65536, stored), "(w) a CPU store after the import window is not reported as one");
    }
    recorder.Sync();
#endif
}

void queuedWriteGroupTests() {
    using Write = Recorder::QueuedWrite;
    using Groups = std::vector<std::size_t>;
    const auto paint = [](const std::vector<Write>& writes, const Groups& groups, std::size_t bytes) {
        std::vector<std::uint8_t> memory(bytes, 0);
        std::size_t first = 0;
        for (const auto end : groups) {
            for (const bool computed : {false, true}) {
                for (auto i = first; i < end; ++i) {
                    if (writes[i].computed != computed) continue;
                    for (auto at = writes[i].begin; at < writes[i].end; ++at) memory[at] = static_cast<std::uint8_t>(i + 1);
                }
            }
            first = end;
        }
        return memory;
    };
    const auto programOrder = [](const std::vector<Write>& writes, std::size_t bytes) {
        std::vector<std::uint8_t> memory(bytes, 0);
        for (std::size_t i = 0; i < writes.size(); ++i) {
            for (auto at = writes[i].begin; at < writes[i].end; ++at) memory[at] = static_cast<std::uint8_t>(i + 1);
        }
        return memory;
    };
    Require(Recorder::QueuedWriteGroups({}).empty(), "no queued writes made a group");
    const std::vector<Write> query{{0x100, 0x200, false}, {0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x200, 0x300, false}, {0x200, 0x2f8, true}, {0x208, 0x300, true}};
    Require(Recorder::QueuedWriteGroups(query) == Groups{3, 6}, "the clear of the next query slot did not wait for the dumps before it");
    const std::vector<Write> reused{{0x100, 0x1f8, true}, {0x100, 0x200, false}, {0x100, 0x1f8, true}};
    Require(Recorder::QueuedWriteGroups(reused) == Groups{1, 3}, "a store over a computed write of its group did not start a new group");
    const std::vector<Write> apart{{0x100, 0x1f8, true}, {0x1f8, 0x200, false}, {0x300, 0x310, false}};
    Require(Recorder::QueuedWriteGroups(apart) == Groups{1, 3}, "stores after the computed writes joined their group");
    const std::vector<Write> label{{0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x400, 0x408, false}};
    Require(Recorder::QueuedWriteGroups(label) == Groups{2, 3}, "a label after the dumps landed before them");
    const std::vector<Write> stores{{0x100, 0x110, false}, {0x100, 0x110, false}};
    Require(Recorder::QueuedWriteGroups(stores) == Groups{2}, "stores alone were split into groups");
    std::uint64_t seed = 0x9e3779b97f4a7c15ull;
    const auto next = [&](std::uint64_t bound) {
        seed ^= seed << 13u;
        seed ^= seed >> 7u;
        seed ^= seed << 17u;
        return seed % bound;
    };
    constexpr std::size_t bytes = 64;
    for (int round = 0; round < 20000; ++round) {
        std::vector<Write> writes;
        const auto count = 1 + next(12);
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto begin = next(bytes - 1);
            const auto end = begin + 1 + next(std::min<std::uint64_t>(16, bytes - begin));
            writes.push_back({begin, std::min<std::uint64_t>(end, bytes), next(2) == 0});
        }
        const auto groups = Recorder::QueuedWriteGroups(writes);
        Require(!groups.empty() && groups.back() == writes.size() && std::is_sorted(groups.begin(), groups.end()), "queued write groups do not end at the last write in order");
        Require(paint(writes, groups, bytes) == programOrder(writes, bytes), "recording each group's stores before its computed writes changed the final bytes of the program order");
    }
}

constexpr std::uint64_t SampleReady = 1ull << 63u;

void queuedSampleDumpTests(Recorder& recorder, const HostImport& import, std::uint64_t* words, std::uint64_t address, VkDeviceAddress target) {
    constexpr std::uint64_t untouched = 0xaaaaaaaaaaaaaaaaull;
    const auto storeZero = [&](std::uint64_t at, std::size_t bytes) {
        const std::vector<std::byte> zero(bytes);
        recorder.RecordStore(import.buffer, at - import.base, zero, at);
    };
    recorder.Sync();
    const auto base = recorder.SamplesTotal();
    std::fill(words, words + 256, untouched);
    Require(recorder.DumpSamples(target, address), "a dump was not queued");
    Require(recorder.HasQueuedStores() && recorder.QueuedStoreOverlaps(address + 240, 8) && !recorder.QueuedStoreOverlaps(address + 248, 8), "a queued dump does not cover its 16 counters");
    recorder.FlushStoresOverlapping(address + 256, 64);
    Require(recorder.HasQueuedStores(), "a reader of other bytes recorded the queued dump");
    Require(words[0] == untouched, "a queued dump landed before anything recorded it");
    recorder.FlushStoresOverlapping(address + 16, 8);
    Require(!recorder.HasQueuedStores(), "a reader of the dumped bytes left the dump queued");
    recorder.Sync();
    Require(words[0] == (base | SampleReady) && words[2] == SampleReady && words[1] == untouched, "the dump a reader recorded stored the wrong counters");
    std::fill(words, words + 256, untouched);
    storeZero(address, 256);
    Require(recorder.DumpSamples(target, address), "a dump after a clear was not queued");
    storeZero(address + 512, 16);
    Require(recorder.DumpSamples(target + 8, address + 8), "the end dump was not queued");
    Require(recorder.DumpSamples(target + 256, address + 256), "a dump of another slot was not queued");
    storeZero(address + 256, 16);
    Require(recorder.DumpSamples(target + 256 + 8, address + 256 + 8), "a dump after a store over an earlier dump was not queued");
    recorder.Sync();
    Require(words[0] == (base | SampleReady) && words[1] == (base | SampleReady) && words[2] == SampleReady && words[3] == SampleReady, "a clear and its dumps landed out of order");
    Require(words[32] == 0 && words[34] == SampleReady, "a store after a dump of its bytes did not land last");
    Require(words[33] == (base | SampleReady) && words[35] == SampleReady, "a dump after a store over an earlier dump landed before it");
    Require(words[64] == 0 && words[65] == 0 && words[66] == untouched, "a store between dumps was lost");
}

class CoverPass {
public:
    explicit CoverPass(const Context& context) : context(context), target(context, ColorTarget{0, {8, 8}, VK_FORMAT_R8G8B8A8_UNORM, 8 * 8 * 4, 0}, false) {
        try {
            VkAttachmentDescription attachment{};
            attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
            attachment.samples = VK_SAMPLE_COUNT_1_BIT;
            attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
            const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &color;
            VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            passInfo.attachmentCount = 1;
            passInfo.pAttachments = &attachment;
            passInfo.subpassCount = 1;
            passInfo.pSubpasses = &subpass;
            Check(context.Function<PFN_vkCreateRenderPass>("vkCreateRenderPass")(context.device, &passInfo, nullptr, &renderPass), "vkCreateRenderPass cover");
            const auto view = target.View();
            VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &view;
            framebufferInfo.width = 8;
            framebufferInfo.height = 8;
            framebufferInfo.layers = 1;
            Check(context.Function<PFN_vkCreateFramebuffer>("vkCreateFramebuffer")(context.device, &framebufferInfo, nullptr, &framebuffer), "vkCreateFramebuffer cover");
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout cover");
            const auto module = [&](const std::uint32_t* code, std::size_t bytes) {
                VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                info.codeSize = bytes;
                info.pCode = code;
                VkShaderModule result = VK_NULL_HANDLE;
                Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &info, nullptr, &result), "vkCreateShaderModule cover");
                return result;
            };
            vertex = module(COVER_VERT_SPV, sizeof(COVER_VERT_SPV));
            fragment = module(COVER_FRAG_SPV, sizeof(COVER_FRAG_SPV));
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr};
            stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr};
            VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            const VkViewport viewport{0, 0, 8, 8, 0, 1};
            const VkRect2D scissor{{0, 0}, {8, 8}};
            VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;
            VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.lineWidth = 1;
            VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo blendState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blendState.attachmentCount = 1;
            blendState.pAttachments = &blend;
            VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
            pipelineInfo.pStages = stages.data();
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &assembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &raster;
            pipelineInfo.pMultisampleState = &multisample;
            pipelineInfo.pColorBlendState = &blendState;
            pipelineInfo.layout = layout;
            pipelineInfo.renderPass = renderPass;
            Check(context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateGraphicsPipelines cover");
        } catch (...) {
            release();
            throw;
        }
    }
    ~CoverPass() { release(); }
    CoverPass(const CoverPass&) = delete;
    CoverPass& operator=(const CoverPass&) = delete;

    bool Draw(Recorder& recorder) const {
        const auto start = recorder.StartDrawPass(PassKey, false, false, PassAccess{});
        const bool continued = start.continued;
        const auto commands = start.commands;
        if (!continued) {
            recorder.PrepareSampleSlot();
            VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            begin.renderPass = renderPass;
            begin.framebuffer = framebuffer;
            begin.renderArea = {{0, 0}, {8, 8}};
            context.Function<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass")(commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
        }
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        recorder.NoteSampledDraw(commands);
        context.Function<PFN_vkCmdDraw>("vkCmdDraw")(commands, 3, 1, 0, 0);
        recorder.LeaveRenderPassOpen(PassKey, Recorder::NoTiming, false, PassAccess{});
        return continued;
    }

    static constexpr std::uint64_t Samples = 64;

private:
    static constexpr std::uint64_t PassKey = 0xc0de;
    void release() noexcept {
        const auto device = context.device;
        if (pipeline != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device, pipeline, nullptr);
        if (vertex != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device, vertex, nullptr);
        if (fragment != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device, fragment, nullptr);
        if (layout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device, layout, nullptr);
        if (framebuffer != VK_NULL_HANDLE) context.Function<PFN_vkDestroyFramebuffer>("vkDestroyFramebuffer")(device, framebuffer, nullptr);
        if (renderPass != VK_NULL_HANDLE) context.Function<PFN_vkDestroyRenderPass>("vkDestroyRenderPass")(device, renderPass, nullptr);
        pipeline = VK_NULL_HANDLE;
        vertex = fragment = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
        framebuffer = VK_NULL_HANDLE;
        renderPass = VK_NULL_HANDLE;
    }
    const Context& context;
    RenderTarget target;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule vertex = VK_NULL_HANDLE;
    VkShaderModule fragment = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

void inPassSampleDumpTests(Recorder& recorder, const Context& context, const CoverPass& cover, std::uint64_t* words, std::uint64_t address, VkDeviceAddress target) {
    const auto samples = [&](std::uint64_t word) { return word & ~SampleReady; };
    recorder.Sync();
    auto base = recorder.SamplesTotal();
    std::fill(words, words + 64, 0);
    Require(!cover.Draw(recorder), "the first draw of a batch continued a pass");
    Require(recorder.DumpSamples(target, address), "a dump inside a render pass was not made on the GPU");
    Require(cover.Draw(recorder), "an occlusion dump ended the render pass");
    Require(recorder.DumpSamples(target + 8, address + 8), "the second dump inside the pass was not made on the GPU");
    Require(cover.Draw(recorder), "the second occlusion dump ended the render pass");
    Require(words[0] == 0 && words[1] == 0, "a dump inside the pass landed before its batch ran");
    recorder.Sync();
    const auto first = samples(words[0]) - base, second = samples(words[1]) - base, total = recorder.SamplesTotal() - base;
    Require((words[0] & SampleReady) != 0 && (words[1] & SampleReady) != 0, "a dump inside a pass lost its ready bit");
    Require(first > 0 && second > first && total > second, "dumps inside one pass do not split its draws' samples in order");
    if (context.occlusionQueryPrecise) Require(first == CoverPass::Samples && second == 2 * CoverPass::Samples && total == 3 * CoverPass::Samples, "dumps inside one pass counted the wrong samples");
    base = recorder.SamplesTotal();
    Require(!cover.Draw(recorder), "the first draw of a new batch continued a pass");
    Require(recorder.DumpSamples(target, address), "a dump before a reader was not made on the GPU");
    recorder.FlushStoresOverlapping(address, 8);
    Require(!cover.Draw(recorder), "a reader of a queued dump recorded it inside the render pass");
    recorder.Sync();
    Require(samples(words[0]) > base && samples(words[0]) < recorder.SamplesTotal(), "the dump a reader recorded in the pass counted the wrong draws");
    base = recorder.SamplesTotal();
    constexpr std::size_t slots = 160;
    std::fill(words, words + slots * 32, 0);
    std::size_t continued = 0;
    for (std::size_t i = 0; i < slots; ++i) {
        if (cover.Draw(recorder)) ++continued;
        Require(recorder.DumpSamples(target + 256 * i, address + 256 * i), "a dump of many in one pass was not made on the GPU");
    }
    recorder.Sync();
    Require(continued + 2 >= slots, "draws between dumps did not keep their render pass");
    std::uint64_t previous = base;
    for (std::size_t i = 0; i < slots; ++i) {
        const auto value = samples(words[32 * i]);
        Require((words[32 * i] & SampleReady) != 0 && value > previous, "a dump past one query pool's slots counted the wrong draws");
        if (context.occlusionQueryPrecise) Require(value == base + (i + 1) * CoverPass::Samples, "a dump past one query pool's slots counted the wrong samples");
        previous = value;
    }
    base = recorder.SamplesTotal();
    constexpr std::size_t passes = 70;
    for (std::size_t i = 0; i < passes; ++i) {
        Require(!cover.Draw(recorder), "a draw after other work continued a pass");
        recorder.Commands();
    }
    recorder.Submit();
    Require(recorder.DumpSamples(target, address), "a dump after many passes was not made on the GPU");
    recorder.Sync();
    const auto folded = samples(words[0]);
    Require(folded > base && folded == recorder.SamplesTotal(), "segments folded at Submit without a dump were lost");
    if (context.occlusionQueryPrecise) Require(folded == base + passes * CoverPass::Samples, "segments folded at Submit without a dump counted the wrong samples");
}

void pendingKeyStoreTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: pending key stores not tested\n";
        return;
    }
    constexpr std::size_t surfaceBytes = 65536;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the pending key store block");
    auto* keys = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the pending key store block refused: pending key stores not tested\n";
        return;
    }
    recorder.Sync();
    std::memset(keys, 0x00, bytes);
    MarkDccUncompressed(context, address, surfaceBytes);
    Require(recorder.PendingWriteOverlaps(address, keyCount) && keys[0] == 0x00, "the uncompressed key store did not stay pending");
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed, "the keys after a pending uncompressed store do not read as uncompressed");
    Require(recorder.PendingWriteOverlaps(address, keyCount), "reading keys the driver's own pending store wrote waited for the GPU");
    Require(CurrentDccKeys(address + 16, surfaceBytes / 2) == DccKeys::Uncompressed, "a key range inside the pending store does not read as uncompressed");
    recorder.NotePendingWrite(address + 16, 16);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a key read with a later writer over the pending store did not wait for it");
    Require(std::all_of(keys, keys + keyCount, [](std::uint8_t key) { return key == 0xff; }), "the uncompressed key store did not land");
    recorder.NotePendingWrite(address, keyCount);
    NoteKeysFillOnGpu(address, keyCount, DccKeys::Clear0001);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of a pending fill did not read as its keys without a wait");
    Require(KnownPendingDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of a pending fill were not known without a wait");
    Require(KnownPendingDccKeys(address + 16, surfaceBytes / 2) == DccKeys::Clear0001, "a key range inside a pending fill was not known");
    Require(!KnownPendingDccKeys(address, 2 * surfaceBytes).has_value() && recorder.PendingWriteOverlaps(address, keyCount), "a range wider than the pending fill was known, or waited for");
    recorder.NotePendingWrite(address, 2 * keyCount);
    Require(!KnownPendingDccKeys(address, surfaceBytes).has_value() && recorder.PendingWriteOverlaps(address, keyCount), "the keys under a later writer over a pending fill were known");
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a later wider writer over a pending fill was not waited for");
    Require(!KnownPendingDccKeys(address, surfaceBytes).has_value(), "keys nothing writes were known as pending keys");
    recorder.NotePendingFill(address, keyCount / 2, 0x40);
    recorder.NotePendingFill(address + keyCount / 2, keyCount / 2, 0x40);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of pending 0001 fills over both halves did not read as 0001 without a wait");
    recorder.NotePendingWrite(address + keyCount / 2, 16);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a later writer of unknown bytes over pending fills was not waited for");
    recorder.NotePendingFill(address, keyCount / 2, 0x00);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Mixed && recorder.PendingWriteOverlaps(address, keyCount), "a pending 0000 fill over half of uncompressed keys did not read as mixed without a wait");
    recorder.Sync();
#ifdef _WIN32
    auto* edgeBlock = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * bytes, MEM_RESERVE, PAGE_NOACCESS));
    Require(edgeBlock != nullptr && VirtualAlloc(edgeBlock, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr, "cannot reserve the key block past committed memory");
#else
    void* edgeMapping = mmap(nullptr, 2 * bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(edgeMapping != MAP_FAILED && mprotect(edgeMapping, bytes, PROT_READ | PROT_WRITE) == 0, "cannot reserve the key block past committed memory");
    auto* edgeBlock = static_cast<std::uint8_t*>(edgeMapping);
#endif
    const auto edge = reinterpret_cast<std::uint64_t>(edgeBlock) + bytes - keyCount / 2;
    Require(!AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(edge), keyCount), "keys past the end of committed memory read as accessible");
    recorder.NotePendingFill(edge, keyCount / 2, 0x40);
    Require(CurrentDccKeys(edge, surfaceBytes) == DccKeys::Unreadable && !recorder.PendingWriteOverlaps(edge, keyCount), "keys running past the end of committed memory, with a pending fill over their committed half, were not read as unreadable after a wait");
#ifdef _WIN32
    VirtualFree(edgeBlock, 0, MEM_RELEASE);
#else
    munmap(edgeBlock, 2 * bytes);
#endif
    constexpr std::uint64_t displayBytes = 15u * 9u * 65536u;
    constexpr std::size_t consoleKeys = 49152;
    const auto displayKeys = DccKeyCount(TextureTileMode::kR64KBX, 4, 1920, 1080, displayBytes);
    std::memset(keys, 0x00, consoleKeys);
    MarkDccUncompressed(context, address, displayBytes, displayKeys);
    Require(recorder.PendingWriteOverlaps(address, consoleKeys) && keys[consoleKeys - 1] == 0x00, "the uncompressed key store over a 1920x1080 surface did not stay pending");
    Require(CurrentDccKeys(address + displayBytes / 256, (consoleKeys - displayBytes / 256) * 256) == DccKeys::Uncompressed && recorder.PendingWriteOverlaps(address, consoleKeys), "the keys past one per 256 bytes of a pending 1920x1080 store did not read as uncompressed without a wait");
    Require(CurrentDccKeys(address, displayBytes, displayKeys) == DccKeys::Uncompressed && recorder.PendingWriteOverlaps(address, consoleKeys), "the keys of a pending 1920x1080 store did not read as uncompressed over the console's extent without a wait");
    recorder.NotePendingWrite(address + 40000, 1);
    Require(CurrentDccKeys(address, displayBytes, displayKeys) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address + 40000, 1), "a later writer of key 40000 over a pending 1920x1080 store was not waited for");
    Require(std::all_of(keys, keys + consoleKeys, [](std::uint8_t key) { return key == 0xff; }), "the uncompressed key store over a 1920x1080 surface did not land on all 49152 keys");
    recorder.Sync();
}

void sampleDumpTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0 || !context.bufferDeviceAddress) {
        std::cout << "host imports or buffer device addresses unavailable: occlusion counter dumps on the GPU not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the occlusion counter block");
    auto* words = static_cast<std::uint64_t*>(block);
    constexpr std::uint64_t untouched = 0xaaaaaaaaaaaaaaaaull;
    std::fill(words, words + 64, untouched);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr || import->address == 0) {
        std::cout << "host import of the occlusion counter block refused: occlusion counter dumps on the GPU not tested\n";
        return;
    }
    const auto target = import->address + (address - import->base);
    recorder.Sync();
    constexpr std::uint64_t ready = 1ull << 63u;
    Require(recorder.DumpSamples(target, address), "the occlusion counters were not dumped on the GPU");
    Require(words[0] == untouched && !recorder.Idle(), "the occlusion counter dump waited for the GPU or landed before its batch ran");
    recorder.Sync();
    const auto begin = recorder.SamplesTotal();
    for (std::size_t db = 0; db < 16; ++db) {
        Require(words[db * 2] == ((db == 0 ? begin : 0) | ready), "an occlusion counter dump stored the wrong value");
        Require(words[db * 2 + 1] == untouched, "an occlusion counter dump stored over the next counter");
    }
    Require(recorder.DumpSamples(target + 8, address + 8), "the second occlusion counter dump was not made on the GPU");
    recorder.Submit();
    recorder.Sync();
    for (std::size_t db = 0; db < 16; ++db) {
        Require(words[db * 2 + 1] == ((db == 0 ? begin : 0) | ready), "the counters changed with nothing drawn between two dumps");
        Require(words[db * 2] == ((db == 0 ? begin : 0) | ready), "the second dump stored over the first");
    }
    Require(recorder.SamplesTotal() == begin, "the sample total moved with nothing drawn");
    for (int i = 0; i < 40; ++i) Require(recorder.DumpSamples(target, address), "a dump past one batch's query slots was not made on the GPU");
    recorder.Sync();
    Require(words[0] == (begin | ready) && recorder.SamplesTotal() == begin, "dumps past one batch's query slots moved the total");
    Require(recorder.QueuesSampleDumps(), "occlusion dumps are recorded at once, not queued with the batch's stores");
    queuedSampleDumpTests(recorder, *import, words, address, target);
    if (!device.Graphics()) {
        std::cout << "the test queue has no graphics: occlusion dumps inside a render pass not tested\n";
        return;
    }
    CoverPass cover(context);
    inPassSampleDumpTests(recorder, context, cover, words, address, target);
}

void cmaskPassTests(const Device& device, Recorder& recorder) {
    namespace GuestMemory = AgcDriver::GuestMemory;
    auto context = device.GetContext();
    context.hostImportAlignment = 0;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    constexpr std::size_t surfaceBytes = 256 * 256 * 4;
    constexpr std::size_t bytes = surfaceBytes + 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the CMASK pass block");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Release {
        const Context& context;
        void* block;
        ~Release() {
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } release{context, block};
    Require(HostImportFor(context, address, bytes) == nullptr, "CMASK copy-path test imported its memory");
    std::memset(block, 0x55, surfaceBytes);
    std::memset(static_cast<std::byte*>(block) + surfaceBytes, 0, 4096);
    ColorTarget color{};
    color.address = address;
    color.extent = {256, 256};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.bytes = surfaceBytes;
    color.componentMapping = 0xe4u;
    color.tileMode = ColorTileMode::RenderTarget;
    color.cmaskAddress = address + surfaceBytes;
    color.cmaskBytes = CmaskBytes(color.extent.width, color.extent.height);
    color.clearWords = {0x80402010u, 0};
    const auto run = [&] { RunColorMetadataPass(context, {ColorMetadataPass::Mode::EliminateFastClear, {color}}); };
    auto broken = context;
    broken.deviceProc = nullptr;
    bool refused = false;
    try {
        RunColorMetadataPass(broken, {ColorMetadataPass::Mode::EliminateFastClear, {color}});
    } catch (const std::runtime_error& error) {
        refused = std::string_view(error.what()).find("missing Vulkan device function resolver") != std::string_view::npos;
        if (!refused) throw;
    }
    Require(refused, "CMASK storage creation failure was silently ignored");
    Require(*static_cast<std::uint8_t*>(block) == 0x55 && ReadDccKeys(color.cmaskAddress, color.cmaskBytes * 256u) == DccKeys::Clear0000, "failed CMASK storage creation changed guest memory");
    run();
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr && *static_cast<std::uint8_t*>(block) == 0x55, "CMASK clear without host imports did not stay on the GPU");
    Require(ReadDccKeys(color.cmaskAddress, color.cmaskBytes * 256u) == DccKeys::Uncompressed, "resident CMASK clear left unresolved metadata");
    color.clearWords[0] = 0x12345678;
    run();
    std::vector<std::uint32_t> stored(surfaceBytes / 4);
    GuestMemory::Read(address, std::as_writable_bytes(std::span(stored)));
    Require(std::ranges::all_of(stored, [](auto word) { return word == 0x80402010u; }), "expanded CMASK changed a pending clear or copy write-back lost it");
    const std::vector<std::byte> clearKeys(4096);
    GuestMemory::Write(color.cmaskAddress, clearKeys);
    run();
    GuestMemory::Read(address, std::as_writable_bytes(std::span(stored)));
    Require(std::ranges::all_of(stored, [](auto word) { return word == 0x12345678u; }), "a second resident CMASK clear retained the first frame");
    recorder.Sync();
}

void metadataPassTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: resident metadata passes not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the metadata pass block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the metadata pass block refused: resident metadata passes not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    ColorTarget color{};
    color.address = address;
    color.extent = {side, side};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.bytes = surfaceBytes;
    color.componentMapping = 0xe4u;
    color.tileMode = ColorTileMode::RenderTarget;
    color.elementBytes = 4;
    color.dccAddress = address + surfaceBytes;
    color.dccPipeAligned = true;
    color.clearWords = {0x80402010u, 0};
    const ColorMetadataPass pass{ColorMetadataPass::Mode::EliminateFastClear, {color}};
    const auto memoryHolds = [&](std::array<std::uint8_t, 4> texel) {
        StorageTexture::FlushPending(address, surfaceBytes, nullptr, "test");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        std::vector<std::uint8_t> stored(surfaceBytes);
        AgcDriver::GuestMemory::Read(address, std::as_writable_bytes(std::span(stored)), 1);
        for (std::size_t i = 0; i < surfaceBytes; ++i) {
            if (stored[i] != texel[i % 4]) return false;
        }
        return true;
    };
    const auto keysUncompressed = [&] { return ReadDccKeys(color.dccAddress, surfaceBytes) == DccKeys::Uncompressed; };
    const auto extent = DccKeyCount(TextureTileMode::kR64KBX, 4, side, side, surfaceBytes);
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0x20, extent);
    RunColorMetadataPass(context, pass);
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr && texels[0] == 0x55, "the register fast clear eliminate did not stay in the resident image");
    Require(keysUncompressed(), "the register fast clear eliminate left the keys compressed");
    Require(CurrentDccKeys(color.dccAddress, surfaceBytes, extent) == DccKeys::Uncompressed, "the register fast clear eliminate did not store uncompressed keys over the target's console DCC extent");
    Require(memoryHolds({0x10, 0x20, 0x40, 0x80}), "the register fast clear eliminate did not store CB_COLOR_CLEAR_WORD");
    Require(std::all_of(keys, keys + extent, [](std::uint8_t key) { return key == 0xff; }), "the stored keys are not uncompressed");
    std::memset(keys, 0xc0, keyCount);
    RunColorMetadataPass(context, {ColorMetadataPass::Mode::DccDecompress, {color}});
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr, "the DCC decompress did not stay in the resident image");
    Require(keysUncompressed(), "the DCC decompress left the keys compressed");
    Require(memoryHolds({0xff, 0xff, 0xff, 0xff}), "the DCC decompress did not store the 1111 value");
    {
        const auto* import = HostImportFor(base, address, bytes);
        Require(import != nullptr, "the metadata pass block lost its import");
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, import->buffer, color.dccAddress - import->base, keyCount, 0x20202020u);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
        recorder.NotePendingWrite(color.dccAddress, keyCount);
        Require(keys[0] == 0xff, "the recorded key fill landed before its batch ran");
        Require(CurrentDccKeys(color.dccAddress, surfaceBytes) == DccKeys::ClearRegister, "the keys read past a pending fast-clear fill");
    }
    std::memset(texels, 0x77, surfaceBytes);
    RunColorMetadataPass(context, pass);
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr && texels[0] == 0x77, "a fast clear recorded before the pass fell back to CPU write-back");
    Require(keysUncompressed() && memoryHolds({0x10, 0x20, 0x40, 0x80}), "a fast clear recorded before the pass was not eliminated");
    std::memset(texels, 0x66, surfaceBytes);
    RunColorMetadataPass(context, pass);
    Require(memoryHolds({0x66, 0x66, 0x66, 0x66}), "a pass over uncompressed keys changed the texels");
    ClearCachedTextures(context.device);
    GuestTextureResource plain{};
    plain.baseAddress = address;
    plain.width = side;
    plain.height = side;
    plain.mipCount = 1;
    plain.tileMode = TextureTileMode::kR64KBX;
    plain.dimension = TextureDimension::k2D;
    plain.format = 56;
    plain.dstSelX = 4;
    plain.dstSelY = 5;
    plain.dstSelZ = 6;
    plain.dstSelW = 7;
    const auto unkeyed = CachedStorageSurface(context, plain);
    std::memset(keys, 0xc0, keyCount);
    RunColorMetadataPass(context, pass);
    const auto keyed = StorageTexture::FindPending(address, surfaceBytes);
    Require(keyed != nullptr && keyed != unkeyed && keyed->Descriptor().dccAddress == color.dccAddress, "a pass over a resident image under other keys did not remake it under the target's keys");
    Require(keysUncompressed() && memoryHolds({0xff, 0xff, 0xff, 0xff}), "a pass over a remade resident image did not store the 1111 value");
    {
        const auto* import = HostImportFor(base, address, bytes);
        Require(import != nullptr, "the metadata pass block lost its import");
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, import->buffer, color.dccAddress - import->base, keyCount, 0x20202020u);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
        recorder.NotePendingWrite(color.dccAddress, keyCount);
        NoteKeysFillOnGpu(color.dccAddress, keyCount, DccKeys::ClearRegister);
    }
    Require(KnownPendingDccKeys(color.dccAddress, surfaceBytes) == DccKeys::ClearRegister, "the keys of a pending register clear store were not known");
    const std::array<std::uint32_t, 4> clearPattern{0x80402010u, 0x80402010u, 0x80402010u, 0x80402010u};
    const char* refusal = nullptr;
    Require(!keyed->FillClear(clearPattern, StorageTexture::WholeImage, refusal) && refusal != nullptr && std::strcmp(refusal, "keys pending") == 0, "a clear over pending keys was not refused without the proof");
    refusal = nullptr;
    Require(keyed->FillClear(clearPattern, StorageTexture::WholeImage, refusal, true), "a clear over known pending keys was refused");
    Require(recorder.PendingWriteOverlaps(color.dccAddress, keyCount) && keys[0] == 0xff, "the clear over known pending keys waited for the key store");
    MarkDccUncompressed(context, color.dccAddress, surfaceBytes);
    Require(memoryHolds({0x10, 0x20, 0x40, 0x80}), "the clear recorded behind a pending key store did not store CB_COLOR_CLEAR_WORD");
    Require(std::all_of(keys, keys + keyCount, [](std::uint8_t key) { return key == 0xff; }), "the keys after a clear behind a pending key store are not uncompressed");
}

// The data word positions of a dispatch-cache variant (Driver.cpp's data-only hits): leaves
// located among the runs' words, aliased, unaligned, out-of-run and mismatched ones skipped.

std::array<std::uint32_t, 8> linearTextureWords(std::uint64_t address, std::uint32_t width, std::uint32_t height, std::uint32_t format) {
    const auto base40 = address >> 8u;
    std::array<std::uint32_t, 8> words{};
    words[0] = static_cast<std::uint32_t>(base40);
    words[1] = static_cast<std::uint32_t>((base40 >> 32u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u);
    words[2] = (((width - 1u) >> 2u) & 0xfffu) | ((height - 1u) << 14u);
    words[3] = 4u | (5u << 3u) | (6u << 6u) | (7u << 9u) | (9u << 28u);
    return words;
}

void imageMemoTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    if (!device.NullDescriptors()) {
        std::cout << "nullDescriptor unavailable: image memo not tested\n";
        return;
    }
    context.nullDescriptors = true;
    constexpr std::uint32_t width = 64;
    constexpr std::uint32_t height = 4;
    constexpr std::size_t surfaceBytes = width * height * 4;
    constexpr std::size_t bytes = 2 * 65536;
    void* block = AllocateWatched(bytes, 65536);
    const bool watched = block != nullptr;
#ifdef _WIN32
    if (!watched) block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    if (!watched) block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the image memo block");
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    auto* first = static_cast<std::uint8_t*>(block);
    auto* second = first + 65536;
    std::memset(first, 0x11, surfaceBytes);
    std::memset(second, 0x22, surfaceBytes);
    std::vector<std::shared_ptr<Texture>> kept;
    const auto lookup = [&](const std::array<std::uint32_t, 8>& words) {
        std::lock_guard gpu(GpuMutex());
        auto texture = LookupDeviceTableTexture(context, words, DeviceTableView{}, 0u, false).texture;
        Require(texture != nullptr, "the lookup has no sampled texture");
        kept.push_back(texture);
        return texture;
    };
    const auto sample = [&](const std::uint8_t* surface) { return lookup(linearTextureWords(reinterpret_cast<std::uint64_t>(surface), width, height, 56)); };
    const auto made = sample(first);
    Require(sample(first) == made, "an unchanged sampled texture was made again");
    const auto other = sample(second);
    Require(other != made && sample(first) == made, "a texture of another descriptor answered for the first");
    ClearCachedTextures(context.device);
    const auto remade = sample(first);
    Require(remade != made && remade != other, "a texture the cache dropped was bound again");
    Require(sample(first) == remade, "the remade texture is not the cache's");
    const auto replaced = sample(second);
    Require(replaced != other && sample(second) == replaced && sample(first) == remade, "the cache does not answer with the replacements");
    const auto firstWords = linearTextureWords(reinterpret_cast<std::uint64_t>(first), width, height, 56);
    const auto secondWords = linearTextureWords(reinterpret_cast<std::uint64_t>(second), width, height, 56);
    std::vector<std::shared_ptr<Texture>> repeated;
    for (const auto& words : {firstWords, firstWords, secondWords, firstWords}) repeated.push_back(lookup(words));
    Require(repeated[0] == remade && repeated[1] == remade && repeated[2] == replaced && repeated[3] == remade, "repeated lookups do not answer with the cache's textures");
    recorder.Submit();
    device.WaitQueue();
    recorder.Sync();
    kept.clear();
    ClearCachedTextures(context.device);
    GuestAllocations::Mutation().Remove(block);
    if (watched) ReleaseWatched(block, bytes);
#ifdef _WIN32
    else VirtualFree(block, 0, MEM_RELEASE);
#else
    else std::free(block);
#endif
}

void multisampleGuestAccessTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0 || !device.NullDescriptors()) {
        std::cout << "4x color targets or nullDescriptor unavailable: multisampled target guest access not tested\n";
        return;
    }
    constexpr std::uint32_t side = 64;
    constexpr std::size_t targetBytes = 65536u;
    constexpr std::size_t bytes = 5u * 65536u;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536u);
    if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* block = nullptr;
    void* raw = mmap(nullptr, bytes + 65536u, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw != MAP_FAILED) {
        const auto begin = reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned = (begin + 65535u) & ~std::uintptr_t{65535u};
        if (aligned != begin) munmap(raw, aligned - begin);
        if (aligned + bytes != begin + bytes + 65536u) munmap(reinterpret_cast<void*>(aligned + bytes), begin + 65536u - aligned);
        block = reinterpret_cast<void*>(aligned);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
    }
#endif
    Require(block != nullptr, "cannot allocate the multisampled target block");
    std::memset(block, 0, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            ClearMultisampleTargets(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            ReleaseWatched(block, bytes);
        }
    } unregister{base, block};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    context.nullDescriptors = true;
    const auto clearTargets = [&] {
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        ClearMultisampleTargets(context.device);
    };
    const auto refused = [](const std::function<void()>& access, const std::string& expected, const std::string& what) {
        std::string message;
        try {
            access();
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find(expected) != std::string::npos && message.find("multisampled color target") != std::string::npos, what + " of a resolved-only multisampled target's memory did not throw: " + message);
    };
    ColorTarget target{};
    target.address = address;
    target.surfaceAddress = address;
    target.extent = {side, side};
    target.surfaceExtent = target.extent;
    target.format = VK_FORMAT_R8G8B8A8_UNORM;
    target.bytes = targetBytes;
    target.samples = 4;
    {
        std::lock_guard gpu(GpuMutex());
        Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the multisampled target has no view");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::array<std::byte, 64> texels{};
    refused([&] { AgcDriver::GuestMemory::Read(address + 256u, texels); }, "a CPU read", "a CPU read");
    refused([&] { AgcDriver::GuestMemory::ReadCommitted(address + 256u, texels); }, "a CPU read", "a committed CPU read");
    refused([&] { AgcDriver::GuestMemory::ReadCommittedUnsynced(address + 256u, texels); }, "a CPU read", "a committed CPU read without the flush hook");
    refused([&] { RequireNoMultisampleTarget(address, 64u, "a DMA copy"); }, "a DMA copy", "a DMA copy");
    const std::array<std::uint32_t, 8> plainWords{
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (9u << 28u),
        0u,
        0u,
        0u,
        0u,
    };
    refused([&] {
        std::lock_guard gpu(GpuMutex());
        static_cast<void>(LookupDeviceTableTexture(context, plainWords, DeviceTableView{}, 0u, true));
    }, "a texture read", "a texture read");
    const auto multisampledWords = [&](std::uint32_t width, std::uint32_t sampleShift, std::uint64_t base) {
        return std::vector<std::uint32_t>{
            static_cast<std::uint32_t>(base >> 8u),
            static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (56u << 20u) | (((width - 1u) & 3u) << 30u),
            ((width - 1u) >> 2u) | ((side - 1u) << 14u),
            0xfacu | (sampleShift << 16u) | (14u << 28u),
            0u,
            sampleShift << 4u,
            0u,
            0u,
        };
    };
    const auto sampleTarget = [&](ShaderRecompiler::DescriptorKind kind, const std::vector<std::uint32_t>& words) -> std::shared_ptr<Texture> {
        DeviceTableView view;
        view.shape = ShaderRecompiler::DescriptorImageShape::Image2DMultisample;
        const std::span<const std::uint32_t, 8> descriptor(words.data(), 8);
        if (kind == ShaderRecompiler::DescriptorKind::StorageImage) {
            static_cast<void>(LookupDeviceTableStorage(context, descriptor, view));
            return nullptr;
        }
        return LookupDeviceTableTexture(context, descriptor, view, 0u, true).texture;
    };
    const auto refusedSample = [&](ShaderRecompiler::DescriptorKind kind, const std::vector<std::uint32_t>& words, const std::string& expected, const std::string& what) {
        std::string message;
        try {
            std::lock_guard gpu(GpuMutex());
            static_cast<void>(sampleTarget(kind, words));
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find(expected) != std::string::npos, what + " did not throw: " + message);
    };
    {
        std::lock_guard gpu(GpuMutex());
        const auto first = sampleTarget(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address));
        const auto second = sampleTarget(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address));
        Require(first != nullptr, "the multisampled target was not bound as a sampled texture");
        Require(first->Layout() == VK_IMAGE_LAYOUT_GENERAL && first->ViewFormat() == target.format, "the sampled multisampled target is not a view of the target's image in its layout");
        Require(second == first, "a second lookup of the multisampled target made another texture");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 3u, address), "as a 64x64 texture of 8 samples", "sampling a 4x target as an 8x texture");
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side / 2u, 2u, address), "as a 32x64 texture of 4 samples", "sampling a 4x target as a narrower texture");
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address + 4u * 65536u), "which no draw rendered to and whose tile mode 0 has no modeled multisampled color footprint", "sampling a linear multisampled texture no draw rendered to");
    refusedSample(ShaderRecompiler::DescriptorKind::StorageImage, multisampledWords(side, 2u, address), "storage image access to a multisampled texture", "storage access to a multisampled texture");
    refused([&] { AgcDriver::GuestMemory::Read(address + 3u * 65536u + 256u, texels); }, "a CPU read", "a CPU read in the padding of a later sample plane");
    AgcDriver::GuestMemory::Read(address + 4u * 65536u, texels);
    clearTargets();
    target.tileMode = ColorTileMode::RenderTarget;
    {
        std::lock_guard gpu(GpuMutex());
        Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the tiled multisampled target has no view");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    refused([&] { AgcDriver::GuestMemory::Read(address + targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the tiled multisampled target");
    AgcDriver::GuestMemory::Read(address + targetBytes, texels);
    clearTargets();
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_2_BIT) != 0) {
        target.extent = {128, 64};
        target.surfaceExtent = target.extent;
        target.samples = 2;
        {
            std::lock_guard gpu(GpuMutex());
            Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the 2x tiled multisampled target has no view");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        refused([&] { AgcDriver::GuestMemory::Read(address + 2u * targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the 2x tile-height padding");
        AgcDriver::GuestMemory::Read(address + 2u * targetBytes, texels);
        clearTargets();
    }
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_8_BIT) != 0) {
        target.extent = {64, 32};
        target.surfaceExtent = target.extent;
        target.samples = 8;
        {
            std::lock_guard gpu(GpuMutex());
            Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the 8x tiled multisampled target has no view");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        refused([&] { AgcDriver::GuestMemory::Read(address + 2u * targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the 8x tile-height padding");
        AgcDriver::GuestMemory::Read(address + 2u * targetBytes, texels);
        clearTargets();
    }
    AgcDriver::GuestMemory::Read(address + 256u, texels);
    target.extent = {side, side};
    target.surfaceExtent = target.extent;
    target.samples = 4;
    VkImageView previous = VK_NULL_HANDLE;
    std::shared_ptr<Texture> previousTexture;
    for (const bool removeAllocation : {false, true}) {
        VkImageView view = VK_NULL_HANDLE;
        std::shared_ptr<Texture> texture;
        {
            std::lock_guard gpu(GpuMutex());
            view = MultisampleTargetView(context, target);
            Require(view != VK_NULL_HANDLE && view != previous, "a remapped allocation reused its retired multisampled image");
            texture = sampleTarget(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address));
            Require(texture != nullptr && texture != previousTexture, "a remapped allocation sampled its retired multisampled image");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        {
            GuestAllocations::Mutation mutation;
            mutation.Protect(block, bytes, true, true, true, [] {});
        }
        refused([&] { RequireNoMultisampleTarget(address, 64u, "a CPU read"); }, "a CPU read", "a read after protection changed");
        bool failed = false;
        try {
            GuestAllocations::Mutation mutation;
            mutation.Unmap(block, bytes, [](const void*, std::size_t, const void*, bool) { throw std::runtime_error("unmap failed"); });
        } catch (const std::runtime_error&) {
            failed = true;
        }
        Require(failed, "the failed host unmap did not throw");
        refused([&] { RequireNoMultisampleTarget(address, 64u, "a CPU read"); }, "a CPU read", "a read after a failed unmap");
        {
            GuestAllocations::Mutation mutation;
            if (removeAllocation) mutation.Remove(block);
            else mutation.Unmap(block, bytes, [](const void*, std::size_t, const void*, bool) {});
            mutation.Add(block, bytes, true, true, true);
        }
        RequireNoMultisampleTarget(address, bytes, "a texture read");
        AgcDriver::GuestMemory::Read(address + 256u, texels);
        refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address), "which no draw rendered to", "sampling a multisampled target whose allocation was unmapped");
        previous = view;
        previousTexture = texture;
    }
    clearTargets();
    std::cout << "Multisampled target guest access tests passed\n";
}

void fmaskDecompressMsaaTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0 || base.hostImportAlignment == 0) {
        std::cout << "4x color targets or host imports unavailable: FMASK decompress not tested\n";
        return;
    }
    constexpr std::uint32_t side = 128;
    constexpr std::size_t targetBytes = 65536u;
    constexpr std::size_t cmaskBytes = 4096u;
    constexpr std::size_t destBytes = targetBytes;
    constexpr std::size_t bytes = 5u * 65536u + cmaskBytes + 2u * destBytes;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the FMASK decompress block");
    std::memset(block, 0, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto cmaskAddress = address + 5u * 65536u;
    const auto destAddress = cmaskAddress + cmaskBytes;
    const auto destAddress2 = destAddress + destBytes;
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            ClearMultisampleTargets(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } unregister{base, block};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    ColorTarget source{};
    source.address = address;
    source.surfaceAddress = address;
    source.extent = {side, side};
    source.surfaceExtent = source.extent;
    source.format = VK_FORMAT_R8G8B8A8_UNORM;
    source.elementBytes = 4;
    source.bytes = targetBytes;
    source.samples = 4;
    source.cmaskAddress = cmaskAddress;
    source.cmaskBytes = cmaskBytes;
    source.cmaskFastClear = true;
    source.clearWords = {0x11223344u, 0};
    ColorTarget destination{};
    destination.address = destAddress;
    destination.surfaceAddress = destAddress;
    destination.extent = {side, side};
    destination.surfaceExtent = destination.extent;
    destination.format = VK_FORMAT_R8G8B8A8_UNORM;
    destination.elementBytes = 4;
    destination.bytes = destBytes;
    destination.samples = 1;
    destination.tileMode = ColorTileMode::RenderTarget;
    {
        std::lock_guard gpu(GpuMutex());
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::FmaskDecompress, {source}});
        auto recolored = source;
        recolored.clearWords = {0x55667788u, 0};
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination}, recolored});
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::vector<std::uint32_t> resolved(destBytes / 4);
    AgcDriver::GuestMemory::Read(destAddress, std::as_writable_bytes(std::span(resolved)));
    Require(std::ranges::all_of(resolved, [](auto word) { return word == 0x11223344u; }), "FMASK decompress did not materialize the seeded clear to the host multisampled image");
    source.clearWords = {0x55667788u, 0};
    ColorTarget destination2 = destination;
    destination2.address = destAddress2;
    destination2.surfaceAddress = destAddress2;
    {
        std::lock_guard gpu(GpuMutex());
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::FmaskDecompress, {source}});
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination2}, source});
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::vector<std::uint32_t> resolved2(destBytes / 4);
    AgcDriver::GuestMemory::Read(destAddress2, std::as_writable_bytes(std::span(resolved2)));
    Require(std::ranges::all_of(resolved2, [](auto word) { return word == 0x11223344u; }), "subsequent FMASK decompress re-cleared the multisampled target");
    std::cout << "FMASK decompress MSAA clear and resolve tests passed\n";
}

void tiledResolveTests(const Device& device, Recorder& recorder) {
    namespace GuestMemory = AgcDriver::GuestMemory;
    auto context = device.GetContext();
    if ((context.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0 || !device.Graphics()) {
        std::cout << "4x color targets or graphics unavailable: CB resolves into tiled targets not tested\n";
        return;
    }
    context.hostImportAlignment = 0;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    constexpr std::uint32_t side = 128;
    constexpr std::size_t targetBytes = side * side * 4u;
    constexpr std::size_t sourceBytes = targetBytes * 4u;
    constexpr std::size_t bytes = targetBytes + sourceBytes + 65536u;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the tiled resolve block");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Release {
        const Context& context;
        void* block;
        ~Release() {
            ClearMultisampleTargets(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } release{context, block};
    ColorTarget source{};
    source.address = address + targetBytes;
    source.surfaceAddress = source.address;
    source.extent = {side, side};
    source.surfaceExtent = source.extent;
    source.format = VK_FORMAT_R8G8B8A8_UNORM;
    source.bytes = targetBytes;
    source.componentMapping = 0xe4u;
    source.tileMode = ColorTileMode::RenderTarget;
    source.samples = 4;
    source.cmaskAddress = address + targetBytes + sourceBytes;
    source.cmaskBytes = 4096;
    source.cmaskFastClear = true;
    const std::array<std::pair<ColorTileMode, std::uint32_t>, 3> modes{{{ColorTileMode::Standard4KB, 0x80402010u}, {ColorTileMode::Standard64KB, 0x10204080u}, {ColorTileMode::RenderTarget, 0x40801020u}}};
    for (const auto& [mode, clear] : modes) {
        const auto name = std::to_string(static_cast<std::uint32_t>(mode));
        std::memset(block, 0x55, targetBytes);
        std::memset(static_cast<std::byte*>(block) + targetBytes + sourceBytes, 0, 4096);
        ColorTarget destination{};
        destination.address = address;
        destination.surfaceAddress = address;
        destination.extent = {side, side};
        destination.surfaceExtent = destination.extent;
        destination.format = VK_FORMAT_R8G8B8A8_UNORM;
        destination.bytes = targetBytes;
        destination.componentMapping = 0xe4u;
        destination.tileMode = mode;
        source.clearWords = {clear, 0};
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination}, source});
        std::vector<std::uint32_t> resolved(targetBytes / 4);
        GuestMemory::Read(address, std::as_writable_bytes(std::span(resolved)));
        const auto texels = std::ranges::count(resolved, clear);
        Require(texels == static_cast<std::ptrdiff_t>(resolved.size()), "a CB resolve into a tile mode " + name + " target left " + std::to_string(resolved.size() - static_cast<std::size_t>(texels)) + " texels without the resolved color");
        recorder.Sync();
        ClearMultisampleTargets(context.device);
        ClearCachedTextures(context.device);
    }
    {
        ColorTarget destination{};
        destination.address = address;
        destination.extent = {side, side};
        destination.format = VK_FORMAT_R8G8B8A8_UNORM;
        destination.bytes = targetBytes;
        destination.tileMode = ColorTileMode::Linear;
        std::string message;
        try {
            RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination}, source});
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find("without a resident image is not implemented") != std::string::npos, "a CB resolve into a linear target did not throw: " + message);
    }
    recorder.Sync();
}

}

class SampleProgram {
public:
    SampleProgram(const Context& context, Recorder& recorder, std::span<const std::uint32_t> code = SAMPLE_LOD_SPV) : context(context), recorder(recorder), result(context, sizeof(float) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 2};
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &setLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &push;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = code.size_bytes();
        moduleInfo.pCode = code.data();
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        pipelineInfo.layout = pipelineLayout;
        Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        Check(context.Function<PFN_vkCreateSampler>("vkCreateSampler")(context.device, &samplerInfo, nullptr, &sampler), "vkCreateSampler");
        const VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = sizes;
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
    }
    ~SampleProgram() {
        context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        context.Function<PFN_vkDestroySampler>("vkDestroySampler")(context.device, sampler, nullptr);
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, pipelineLayout, nullptr);
        context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }
    SampleProgram(const SampleProgram&) = delete;
    SampleProgram& operator=(const SampleProgram&) = delete;

    float Red(VkImageView view, VkImageLayout layout, float lod, float layer = 0) {
        VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = pool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &setLayout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocateInfo, &set), "vkAllocateDescriptorSets");
        const VkDescriptorImageInfo image{sampler, view, layout};
        const VkDescriptorBufferInfo buffer{result.Handle(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        writes[0].dstSet = set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &image;
        writes[1].dstSet = set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &buffer;
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 2, writes, 0, nullptr);
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        const float parameters[]{lod, layer};
        context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(parameters), parameters);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, 1, 1, 1);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        recorder.Sync();
        Check(context.Function<PFN_vkFreeDescriptorSets>("vkFreeDescriptorSets")(context.device, pool, 1, &set), "vkFreeDescriptorSets");
        float texel[4];
        std::memcpy(texel, result.Bytes().data(), sizeof(texel));
        return texel[0];
    }

private:
    const Context& context;
    Recorder& recorder;
    Buffer result;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
};

void expectRed(float got, float want, const char* what) {
    if (std::abs(got - want) > 1.5f / 255.0f) throw std::runtime_error(std::string(what) + ": read " + std::to_string(got) + ", expected " + std::to_string(want));
}

void minLodTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (!context.imageViewMinLod) {
        std::cout << "VK_EXT_image_view_min_lod unavailable: minimum LOD clamp not tested\n";
        return;
    }
    TextureDetiler detiler(context);
    GuestTextureResource resource{};
    resource.baseAddress = 0x100000;
    resource.width = 8;
    resource.height = 8;
    resource.mipCount = 4;
    resource.baseLevel = 0;
    resource.lastLevel = 3;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    const auto geometry = DescribeSurface(resource);
    Require(geometry.mips.size() == 4, "the min LOD test surface has an unexpected mip count");
    const std::array<std::uint8_t, 4> levels{0x00, 0x40, 0x80, 0xff};
    std::vector<std::byte> snapshot(static_cast<std::size_t>(geometry.guestBytes));
    for (std::size_t level = 0; level < levels.size(); ++level) {
        const auto& mip = geometry.mips[level];
        std::fill_n(snapshot.begin() + static_cast<std::ptrdiff_t>(mip.tiledOffset), static_cast<std::size_t>(mip.tiledSize), std::byte{levels[level]});
    }
    SampleProgram program(context, recorder);
    const VkComponentMapping identity{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    const auto sample = [&](std::uint32_t minLod, float lod, std::uint32_t baseLevel = 0) {
        auto described = resource;
        described.minLod = minLod;
        described.baseLevel = baseLevel;
        Texture texture(context, detiler, described, identity, snapshot);
        return program.Red(texture.View(), texture.Layout(), lod);
    };
    const auto unorm = [&](std::size_t level) { return levels[level] / 255.0f; };
    expectRed(sample(0, 0.0f), unorm(0), "minimum LOD clamp: no clamp reads level 0");
    expectRed(sample(0x100, 0.0f), unorm(1), "minimum LOD clamp: MIN_LOD 1 reads level 1 at LOD 0");
    const auto fractional = sample(0x180, 0.0f);
    if (std::abs(fractional - unorm(1)) <= 1.5f / 255.0f) {
        std::cout << "minimum LOD clamp: the device takes the integer part of a fractional view minimum LOD\n";
    } else {
        expectRed(fractional, (unorm(1) + unorm(2)) / 2.0f, "minimum LOD clamp: MIN_LOD 1.5 blends levels 1 and 2");
    }
    expectRed(sample(0x100, 2.0f), unorm(2), "minimum LOD clamp: MIN_LOD 1 lowered LOD 2");
    expectRed(sample(0xfff, 0.0f), unorm(3), "minimum LOD clamp: MIN_LOD past the last level reads the last level");
    expectRed(sample(0x200, 0.0f, 1), unorm(2), "minimum LOD clamp: MIN_LOD 2 over a view from level 1 reads level 2");
    expectRed(sample(0x100, 0.0f, 1), unorm(1), "minimum LOD clamp: MIN_LOD at the view's base level reads its base level");
}

void singleCubeTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    SampleProgram program(context, recorder, SAMPLE_ARRAY_SPV);
    const VkComponentMapping identity{};
    for (const auto first : {0u, 5u, 6u}) {
        std::array<std::uint32_t, 8> words{0x1000u, (56u << 20u) | (3u << 30u), 15u | (63u << 14u), 0xb0000facu, first | (first << 16u), 0, 0, 0};
        auto resource = DecodeTextureResource(words);
        const auto geometry = DescribeSurface(resource);
        std::vector<std::byte> memory(static_cast<std::size_t>(geometry.guestBytes) + 256);
        auto* surface = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255});
        for (std::uint32_t layer = 0; layer < geometry.imageLayers; ++layer) {
            std::memset(surface + geometry.GuestLayerOffset(layer), 16 * (layer + 1), static_cast<std::size_t>(geometry.layerBytes));
        }
        resource.baseAddress = reinterpret_cast<std::uint64_t>(surface);
        const std::span<const std::byte> snapshot(surface, static_cast<std::size_t>(geometry.guestBytes));
        Texture texture(context, detiler, resource, identity, snapshot);
        auto storage = StorageTexture::Create(context, detiler, resource, 0);
        recorder.Keep(storage);
        Texture storageView(context, storage, resource, identity);
        for (std::uint32_t face = 0; face < 6; ++face) {
            const auto expected = 16.0f * (first + face + 1) / 255.0f;
            expectRed(program.Red(texture.View(), texture.Layout(), 0, face), expected, "single cube snapshot sampled the wrong face");
            expectRed(program.Red(storageView.View(), storageView.Layout(), 0, face), expected, "single cube storage view sampled the wrong face");
        }
    }
}

void firstLayerViewTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    TextureDetiler detiler(context);
    auto withDetiler = context;
    withDetiler.detiler = &detiler;
    GuestTextureResource resource{};
    resource.width = 64;
    resource.height = 4;
    resource.depthOrLastArray = 2;
    resource.baseArray = 1;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2DArray;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    const auto geometry = DescribeSurface(resource);
    std::vector<std::uint8_t> memory(static_cast<std::size_t>(geometry.guestBytes) + 256);
    auto* surface = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255});
    for (std::uint32_t layer = 0; layer < 3; ++layer) std::memset(surface + geometry.GuestLayerOffset(layer), 0x20 * (layer + 1), static_cast<std::size_t>(geometry.layerBytes));
    resource.baseAddress = reinterpret_cast<std::uint64_t>(surface);
    auto image = StorageTexture::Create(withDetiler, detiler, resource, 0);
    recorder.Keep(image);
    SampleProgram program(context, recorder);
    expectRed(program.Red(image->FirstLayerView(0), VK_IMAGE_LAYOUT_GENERAL, 0.0f), 0x40 / 255.0f, "a first-layer view does not read the BASE_ARRAY layer");
    Require(image->FirstLayerView(0) == image->FirstLayerView(0), "first-layer views are not reused");
    const VkComponentMapping identity{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    Texture snapshotTexture(context, detiler, resource, identity, std::span<const std::byte>(reinterpret_cast<const std::byte*>(surface), static_cast<std::size_t>(geometry.guestBytes)));
    Require(snapshotTexture.FirstLayerView() != VK_NULL_HANDLE, "a sampled 2D array texture has no first-layer view");
    expectRed(program.Red(snapshotTexture.FirstLayerView(), snapshotTexture.Layout(), 0.0f), 0x40 / 255.0f, "a sampled first-layer view does not read the BASE_ARRAY layer");
    Texture storageView(context, image, resource, identity);
    Require(storageView.FirstLayerView() != VK_NULL_HANDLE, "a sampled view of a 2D array storage image has no first-layer view");
    expectRed(program.Red(storageView.FirstLayerView(), storageView.Layout(), 0.0f), 0x40 / 255.0f, "a sampled first-layer view of a storage image does not read the BASE_ARRAY layer");
    auto flat = resource;
    flat.dimension = TextureDimension::k2D;
    flat.depthOrLastArray = 0;
    flat.baseArray = 0;
    Texture flatTexture(context, detiler, flat, identity, std::span<const std::byte>(reinterpret_cast<const std::byte*>(surface), static_cast<std::size_t>(DescribeSurface(flat).guestBytes)));
    Require(flatTexture.FirstLayerView() == VK_NULL_HANDLE, "a 2D texture has a first-layer view");
    expectRed(program.Red(flatTexture.View(), flatTexture.Layout(), 0.0f), 0x20 / 255.0f, "a 2D texture over the surface does not read its first layer");
}

void sliceDirtyTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    constexpr std::uint32_t side = 128;
    constexpr std::size_t sliceBytes = 65536;
    constexpr std::size_t bytes = 4 * sliceBytes;
    constexpr std::uint8_t untouched = 0x11;
    for (const auto tileMode : {TextureTileMode::kLinear, TextureTileMode::kR64KBX}) {
        const std::string what = tileMode == TextureTileMode::kLinear ? "a linear two-slice image" : "a tiled two-slice image";
        void* block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: slice dirty marking not tested\n";
            return;
        }
        auto* memory = static_cast<std::uint8_t*>(block);
        std::memset(memory, untouched, bytes);
        const auto address = reinterpret_cast<std::uint64_t>(block);
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(block, bytes, true, true, true);
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                ReleaseWatched(block, bytes);
            }
        } unregister{base, block, address, bytes};
        AgcDriver::GuestMemory::CollectWritesUncached(address, bytes);
        TextureDetiler detiler(base);
        auto context = base;
        context.hostImportAlignment = 0;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = side;
        resource.height = side;
        resource.depthOrLastArray = 1;
        resource.mipCount = 1;
        resource.tileMode = tileMode;
        resource.dimension = TextureDimension::k2DArray;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        Require(geometry.imageLayers == 2 && geometry.layerBytes == sliceBytes && geometry.guestBytes == 2 * sliceBytes, what + " does not hold two 64 KiB slices");
        const auto stored = [&](std::uint32_t slice, std::array<std::uint8_t, 4> color) {
            std::vector<std::byte> read(sliceBytes);
            AgcDriver::GuestMemory::Read(address + geometry.GuestLayerOffset(slice), read);
            for (std::size_t at = 0; at < read.size(); ++at) {
                if (std::to_integer<std::uint8_t>(read[at]) != color[at % 4]) return false;
            }
            return true;
        };
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
            const VkClearColorValue green{{0.0f, 1.0f, 0.0f, 1.0f}};
            const VkImageSubresourceRange first{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const VkImageSubresourceRange second{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1, 1};
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &first);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &green, 1, &second);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkSliceDirty(0);
            for (std::uint32_t slice = 0; slice < 2; ++slice) {
                const auto display = address + geometry.GuestLayerOffset(slice);
                const auto found = StorageTexture::FindPendingSlice(display, sliceBytes);
                Require(found == image, what + ": a display buffer that is slice " + std::to_string(slice) + " does not find the pending image");
                Require(AgcDriver::ResidentPresentSlice(found->Descriptor().baseAddress, found->GuestBytes(), found->ImageLayers(), display, sliceBytes) == slice, what + ": a display buffer that is slice " + std::to_string(slice) + " is not presented from that layer");
            }
            Require(StorageTexture::FindPending(address + geometry.GuestLayerOffset(1), sliceBytes) == nullptr, what + ": the lookup by base address finds a display buffer in slice 1");
            Require(StorageTexture::FindPendingSlice(address + 2 * sliceBytes, sliceBytes) == nullptr && StorageTexture::FindPendingSlice(address + geometry.GuestLayerOffset(1), sliceBytes + 1) == nullptr, what + ": a range past the image's last slice finds the pending image");
            image->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(stored(0, {255, 0, 0, 255}), what + ": the slice a draw wrote was not stored");
            Require(stored(1, {untouched, untouched, untouched, untouched}), what + ": a slice no draw wrote was stored over guest memory");
            static_cast<void>(recorder.Commands());
            recorder.Keep(image);
            image->MarkSliceDirty(1);
            image->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(stored(0, {255, 0, 0, 255}) && stored(1, {0, 255, 0, 255}), what + ": the second slice was not stored once a draw wrote it");
            auto other = StorageTexture::Create(context, detiler, resource, 0);
            static_cast<void>(recorder.Commands());
            recorder.Keep(image);
            recorder.Keep(other);
            image->MarkSliceDirty(1);
            other->MarkSliceDirty(1);
            std::string ambiguous;
            try {
                static_cast<void>(StorageTexture::FindPendingSlice(address + geometry.GuestLayerOffset(1), sliceBytes));
            } catch (const std::runtime_error& error) {
                ambiguous = error.what();
            }
            Require(ambiguous.find("is a slice of the pending array images") != std::string::npos, what + ": a slice of two pending array images did not throw: " + ambiguous);
            image->WriteBack();
            other->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        recorder.Sync();
    }
    std::cout << "slice dirty marking: ok\n";
}

void atomicViewTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    TextureDetiler detiler(context);
    auto withDetiler = context;
    withDetiler.detiler = &detiler;
    for (const std::uint32_t format : {20u, 21u, 22u, 56u}) {
        GuestTextureResource resource{};
        resource.width = 64;
        resource.height = 4;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kLinear;
        resource.dimension = TextureDimension::k2D;
        resource.format = format;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        std::vector<std::uint8_t> memory(static_cast<std::size_t>(geometry.guestBytes) + 256);
        resource.baseAddress = (reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255};
        auto image = StorageTexture::Create(withDetiler, detiler, resource, 0);
        recorder.Keep(image);
        if (format == 56u) {
            bool refused = false;
            try {
                static_cast<void>(image->AtomicView(0, StorageTexture::LayersOf(resource, false)));
            } catch (const std::exception&) {
                refused = true;
            }
            Require(refused, "an 8_8_8_8 storage image has an atomic view");
            continue;
        }
        const auto atomic = image->AtomicView(0, StorageTexture::LayersOf(resource, false));
        Require(atomic != VK_NULL_HANDLE && atomic == image->AtomicView(0, StorageTexture::LayersOf(resource, false)), "atomic storage views are not reused");
        Require((atomic == image->View(0)) == (format == 20u), "only a 32_UINT storage image serves atomics through its own view");
    }
}

void pipelineCacheTests(const Device& device, bool benchmark) {
    std::thread establishThreadedProcess([] {});
    establishThreadedProcess.join();
    std::lock_guard gpu(GpuMutex());
    const auto& context = device.GetContext();
    struct Cleanup {
        VkDevice device;
        ~Cleanup() { ClearCachedPipelines(device); }
    } cleanup{context.device};
    ShaderRecompiler::RecompileResult vertex, fragment;
    vertex.variantId = 1;
    fragment.variantId = 2;
    vertex.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_vert_SPV), std::end(TRIANGLE_vert_SPV));
    fragment.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_frag_SPV), std::end(TRIANGLE_frag_SPV));
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    State state{};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.hasColorTarget = true;
    state.color.format = VK_FORMAT_R8G8B8A8_UNORM;
    state.colors.push_back(state.color);
    state.blend.colorWriteMask = 15;
    state.blends.push_back(state.blend);
    ShaderResources resources(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), 0, 0);
    const auto lookup = [&](const VertexInputLayout& input) {
        return CachedPipeline(context, state, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL);
    };
    VertexInputLayout input;
    const auto first = lookup(input);
    Require(lookup(input) == first, "pipeline cache did not retain an identical key");
    state.cullMode = VK_CULL_MODE_BACK_BIT;
    Require(lookup(input) != first, "pipeline cache ignored changed raster state");
    state.cullMode = 0;
    const auto largest = std::min({32u, context.limits.maxVertexInputBindings, context.limits.maxVertexInputAttributes});
    for (const auto attributes : {0u, 8u, largest}) {
        input.bindings.clear();
        input.attributes.clear();
        for (std::uint32_t i = 0; i < attributes; ++i) {
            input.bindings.push_back({i, 16, VK_VERTEX_INPUT_RATE_VERTEX});
            input.attributes.push_back({i, i, VK_FORMAT_R32G32B32A32_SFLOAT, 0});
        }
        const auto expected = lookup(input);
        if (attributes != 0) Require(expected != first, "pipeline cache ignored changed vertex input");
        Require(lookup(input) == expected, "pipeline cache lost the larger key");
        if (!benchmark) continue;
        std::array<double, 9> times;
        for (unsigned pass = 0; pass <= times.size(); ++pass) {
            const auto started = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < 65536; ++i) Require(lookup(input) == expected, "pipeline cache benchmark missed a warmed key");
            if (pass != 0) times[pass - 1] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count() / 65536;
        }
        std::ranges::sort(times);
        std::cout << "Pipeline cache " << attributes << " attributes: median " << times[4] << " us/lookup, p95 pass " << times.back() << " us\n";
    }
    input = {};
    Require(lookup(input) == first, "pipeline cache retained bytes from a larger key");
    vertex.variantId = 0;
    Require(lookup(input) == first, "pipeline cache keyed on the variant id instead of the module");
    vertex.spirv[2] ^= 1u;
    const auto other = lookup(input);
    Require(other != first && lookup(input) == other, "pipeline cache ignored another module");
    vertex.spirv[2] ^= 1u;
    vertex.variantId = 1;
    Require(lookup(input) == first, "another module disturbed the retained pipeline");
    ClearCachedPipelines(context.device);
    Require(lookup(input) != first, "clearing the pipeline cache retained its old entry");
    std::cout << "Pipeline cache key and retained lifetime tests passed\n";
}

void descriptorPoolTests(const Device& device) {
    DescriptorCache cache(device.GetContext());
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    const std::array<std::uint32_t, 4> key{binding.binding, binding.descriptorType, binding.descriptorCount, binding.stageFlags};
    const auto layout = cache.Layout(key, std::span(&binding, 1));
    const VkDescriptorPoolSize size{binding.descriptorType, binding.descriptorCount};
    const auto held = cache.Allocate(layout, std::span(&size, 1));
    std::uint64_t initialPools = 0;
    for (unsigned round = 0; round < 3; ++round) {
        std::vector<DescriptorCache::SetAllocation> sets;
        for (unsigned i = 0; i < 1100; ++i) {
            const auto allocation = cache.Allocate(layout, std::span(&size, 1));
            Require(allocation.set != VK_NULL_HANDLE, "descriptor pool rollover refused a valid layout");
            sets.push_back(allocation);
        }
        const auto pools = cache.Counters().pools;
        if (round == 0) {
            initialPools = pools;
        } else {
            Require(pools <= initialPools + 1, "descriptor pool recycling grows behind one retained set");
        }
        for (const auto& allocation : sets) cache.Free(allocation);
    }
    cache.Free(held);
    std::cout << "Descriptor pool recycling passed across 3,301 sets (" << initialPools << " initial pools, " << cache.Counters().pools << " final pools)\n";
}

void keysFillTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: DCC key fills not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the key fill block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto keysAddress = address + surfaceBytes;
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the key fill block refused: DCC key fills not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = keysAddress;
    Require(DescribeSurface(resource).guestBytes == surfaceBytes, "the key fill surface has an unexpected size");
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto draw = [&](VkClearColorValue value) {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        };
        const auto holds = [&](std::array<std::uint8_t, 4> texel) {
            Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                if (std::to_integer<std::uint8_t>(pixels[i]) != texel[i % 4]) return false;
            }
            return true;
        };
        const auto fill = [&](std::uint8_t key) {
            recorder.Sync();
            std::memset(keys, key, keyCount);
            return StorageTexture::NoteKeysFill(keysAddress, keyCount, key);
        };
        Require(holds({0, 0, 0, 0}), "a surface under 0000 keys was not cleared");
        draw({{1.0f, 0.0f, 0.0f, 1.0f}});
        Require(fill(0x00) == 1, "a 0000 key fill did not cover the surface");
        Require(image->FilledKeys() == DccKeys::Clear0000, "a key fill over pending results was not recorded");
        image->Refresh();
        Require(holds({0, 0, 0, 0}), "a key fill did not clear the results made before it at the next refresh");
        Require(image->FilledKeys() == DccKeys::Uncompressed, "the image cleared by a refresh still holds the fill");
        draw({{0.0f, 0.0f, 1.0f, 1.0f}});
        Require(fill(0x00) == 1, "a second 0000 key fill did not cover the surface");
#ifdef _WIN32
        _putenv_s("APS5_KEYS_FILL_CLEAR", "1");
#else
        setenv("APS5_KEYS_FILL_CLEAR", "1", 1);
#endif
        Require(StorageTexture::ClearByKeysFill(keysAddress, keyCount, 0x00) == 1, "a 0000 key fill did not clear the surface at once");
        Require(holds({0, 0, 0, 0}), "a key fill cleared at once left results made before it");
        draw({{0.0f, 1.0f, 0.0f, 1.0f}});
        Require(image->FilledKeys() == DccKeys::Uncompressed, "results drawn after a key fill cleared at once are held under the fill's code");
        Require(holds({0, 255, 0, 255}), "results drawn after a key fill cleared at once were lost");
    }
    recorder.Sync();
}

void integerClearTests(const Device& device) {
    const auto& context = device.GetContext();
    std::size_t tested = 0;
    for (const std::uint32_t format : {5u, 11u, 20u}) {
        if (!StorageFormatAvailable(context, format)) continue;
        ++tested;
        Require(StorageClearAvailable(context, format, DccKeys::Clear0000), "an integer storage format refused the 0000 clear");
        for (const auto keys : {DccKeys::Clear0001, DccKeys::Clear1111, DccKeys::ClearRegister, DccKeys::Uncompressed}) {
            Require(!StorageClearAvailable(context, format, keys), "an integer storage format took a clear code other than 0000");
        }
    }
    Require(tested == 3, "an integer storage format has no storage form");
    std::cout << "integer storage clears: " << tested << " formats ok\n";
}

void denormalClearTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: denormal fill clears not tested\n";
        return;
    }
    constexpr std::uint32_t side = 64;
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the denormal clear block");
    std::memset(block, 0x55, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the denormal clear block refused: denormal fill clears not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 29;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    Require(ResolveTextureFormat(resource.format) == VK_FORMAT_R16G16_SFLOAT, "guest format 29 is not R16G16_SFLOAT");
    const bool keeps = ClearKeepsDenormals(context, VK_FORMAT_R16G16_SFLOAT);
    std::cout << "denormal half clears are " << (keeps ? "kept" : "flushed") << " by this device\n";
    Require(ClearKeepsDenormals(context, VK_FORMAT_R16G16_SFLOAT) == keeps, "the denormal clear probe changed its answer");
    Require(!ClearKeepsDenormals(context, VK_FORMAT_R32G32B32A32_SFLOAT), "a format without small floats probed as keeping denormals");
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const auto holds = [&](std::uint32_t texel) {
            Buffer readback(context, side * side * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            for (std::size_t at = 0; at + 4 <= pixels.size(); at += 4) {
                std::uint32_t value = 0;
                std::memcpy(&value, pixels.data() + at, 4);
                if (value != texel) return false;
            }
            return true;
        };
        const char* refusal = nullptr;
        const std::array<std::uint32_t, 4> normal{0x3c00bc00u, 0x3c00bc00u, 0x3c00bc00u, 0x3c00bc00u};
        Require(image->FillClear(normal, StorageTexture::WholeImage, refusal), "a normal half pattern was refused");
        Require(holds(0x3c00bc00u), "a normal half fill clear stored other bits");
        const std::array<std::uint32_t, 4> denormal{0x83ff0001u, 0x83ff0001u, 0x83ff0001u, 0x83ff0001u};
        refusal = nullptr;
        const bool cleared = image->FillClear(denormal, StorageTexture::WholeImage, refusal);
        Require(cleared == keeps, keeps ? "a denormal half pattern was refused on a device whose clears keep denormals" : "a denormal half pattern was cleared on a device whose clears flush them");
        if (cleared) Require(holds(0x83ff0001u), "a denormal half fill clear stored other bits");
        const std::array<std::uint32_t, 4> nan{0x7e007e00u, 0x7e007e00u, 0x7e007e00u, 0x7e007e00u};
        refusal = nullptr;
        Require(!image->FillClear(nan, StorageTexture::WholeImage, refusal), "a NaN half pattern was cleared");
    }
    recorder.Sync();
}

void pipelineLibraryTests(const Device& device) {
    std::lock_guard gpu(GpuMutex());
    const auto& context = device.GetContext();
    struct Cleanup {
        VkDevice device;
        ~Cleanup() { ClearCachedPipelines(device); }
    } cleanup{context.device};
    ClearCachedPipelines(context.device);
    ShaderRecompiler::RecompileResult vertex, fragment;
    vertex.variantId = 11;
    fragment.variantId = 12;
    vertex.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_vert_SPV), std::end(TRIANGLE_vert_SPV));
    fragment.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_frag_SPV), std::end(TRIANGLE_frag_SPV));
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    State state{};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.hasColorTarget = true;
    state.color.format = VK_FORMAT_R8G8B8A8_UNORM;
    state.colors.push_back(state.color);
    state.blend.colorWriteMask = 15;
    state.blends.push_back(state.blend);
    ShaderResources resources(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), 0, 0);
    VertexInputLayout input;
    const auto lookup = [&] { return CachedPipeline(context, state, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL); };
    const auto counters = [&] { return PipelineLibraryCountersOf(context.device); };
    const auto first = lookup();
    if (!context.graphicsPipelineLibrary) {
        Require(counters().linked == 0, "a pipeline was linked from libraries without VK_EXT_graphics_pipeline_library");
        std::cout << "graphics pipeline libraries unavailable or disabled: pipelines are created whole\n";
        return;
    }
    const auto built = counters();
    Require(built.linked == 1 && built.built == std::array<std::uint64_t, 4>{1, 1, 1, 1}, "the first pipeline was not linked from four new libraries");
    state.blends.front().blendEnable = VK_TRUE;
    state.blends.front().srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    state.blends.front().dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    const auto blended = lookup();
    Require(blended != first && counters().built == std::array<std::uint64_t, 4>{1, 1, 1, 2}, "a blend change rebuilt more than the fragment output library");
    state.cullMode = VK_CULL_MODE_BACK_BIT;
    state.frontFace = VK_FRONT_FACE_CLOCKWISE;
    const auto culled = lookup();
    Require(culled != blended && counters().built == std::array<std::uint64_t, 4>{1, 1, 1, 2}, "a cull mode change rebuilt a library instead of setting dynamic state");
    input.bindings.push_back({0, 16, VK_VERTEX_INPUT_RATE_VERTEX});
    input.attributes.push_back({0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0});
    static_cast<void>(lookup());
    Require(counters().built == std::array<std::uint64_t, 4>{2, 1, 1, 2} && counters().linked == 4, "a vertex input change rebuilt more than the vertex input library");
    Require(StagePushOffset(88, ShaderRecompiler::ShaderStage::Fragment, true) == PipelinePushSlotBytes && StagePushOffset(88, ShaderRecompiler::ShaderStage::Fragment, false) == 88 && StagePushOffset(40, ShaderRecompiler::ShaderStage::TessellationEvaluation, true) == 40, "a stage got the wrong push offset");
    const auto beforeSlots = counters().built;
    ShaderRecompiler::RecompileResult otherVertex = vertex;
    otherVertex.spirv[2] ^= 1u;
    std::array slotted{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, PipelinePushSlotBytes}};
    static_cast<void>(CachedPipeline(context, state, input, resources, slotted, VK_IMAGE_LAYOUT_GENERAL));
    slotted.front().program = &otherVertex;
    static_cast<void>(CachedPipeline(context, state, input, resources, slotted, VK_IMAGE_LAYOUT_GENERAL));
    Require(counters().built[2] == beforeSlots[2] + 1u && counters().built[1] == beforeSlots[1] + 1u, "a pixel shader in its fixed slot built its library again for another vertex shader");
    fragment.spirv[2] ^= 1u;
    static_cast<void>(lookup());
    Require(counters().built == std::array<std::uint64_t, 4>{2, 2, 3, 2}, "a new pixel shader rebuilt more than the fragment shader library");
    state.colors.front().format = VK_FORMAT_R16G16B16A16_SFLOAT;
    static_cast<void>(lookup());
    Require(counters().built == std::array<std::uint64_t, 4>{2, 2, 3, 3}, "another target format rebuilt more than the fragment output library");
    state.samples = 4;
    state.colors.front().samples = 4;
    const auto multisampled = lookup();
    Require(counters().built == std::array<std::uint64_t, 4>{2, 2, 4, 4}, "a 4x pipeline shared the 1x fragment shader or fragment output library");
    state.samples = 1;
    state.colors.front().samples = 1;
    Require(lookup() != multisampled && counters().built == std::array<std::uint64_t, 4>{2, 2, 4, 4}, "the 1x pipeline after a 4x one built its libraries again");
    state.depth = DepthTarget{0, 0, {64, 64}, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    state.depthTest = true;
    state.depthCompare = VK_COMPARE_OP_LESS;
    static_cast<void>(lookup());
    Require(counters().built == std::array<std::uint64_t, 4>{2, 3, 5, 5}, "a depth target shared the shader libraries built with no depth format");
    state.depth.reset();
    state.depthTest = false;
    if ((context.sampleLocationSampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0) {
        state.samples = 4;
        state.colors.front().samples = 4;
        state.customSampleLocations = true;
        state.sampleLocations = {0x10, 0x21, 0x32, 0x43};
        const auto pattern = lookup();
        const auto beforePatternChange = counters().built;
        state.sampleLocations[0] = 0x20;
        const auto otherPattern = lookup();
        Require(otherPattern != pattern && counters().built[2] == beforePatternChange[2] + 1u && counters().built[3] == beforePatternChange[3] + 1u && counters().built[0] == beforePatternChange[0] && counters().built[1] == beforePatternChange[1], "a custom sample pattern did not rebuild exactly the fragment shader and fragment output libraries");
        state.sampleLocations[0] = 0x10;
        Require(lookup() == pattern && counters().built[2] == beforePatternChange[2] + 1u && counters().built[3] == beforePatternChange[3] + 1u, "a cached custom sample pattern was not reused");
        if ((context.sampleLocationGridSampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0) {
            state.sampleLocationGrid = true;
            state.sampleLocations[3u * 8u] = 0x20;
            const auto gridPattern = lookup();
            Require(gridPattern != pattern && counters().built[2] == beforePatternChange[2] + 2u && counters().built[3] == beforePatternChange[3] + 2u, "a pattern varying between pixels shared a uniform pattern's fragment libraries");
            state.sampleLocationGrid = false;
            state.sampleLocations[3u * 8u] = 0;
            Require(lookup() == pattern, "the uniform pattern was not reused after a grid pattern");
        }

        std::optional<VkFormat> compatibleDepth, incompatibleDepth;
        state.customSampleLocations = false;
        for (const auto format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
            state.depth = DepthTarget{};
            state.depth->format = format;
            state.depth->samples = state.samples;
            (ValidateSampleLocations(context, state) ? compatibleDepth : incompatibleDepth) = format;
        }
        if (compatibleDepth && incompatibleDepth) {
            state.depth->format = *compatibleDepth;
            const auto compatible = lookup();
            const auto beforeDepthChange = counters().built;
            state.depth->format = *incompatibleDepth;
            const auto incompatible = lookup();
            Require(incompatible != compatible && counters().built[2] == beforeDepthChange[2] + 1u && counters().built[3] == beforeDepthChange[3] + 1u, "depth-format sample-location enablement did not rebuild both fragment libraries");
        }
        state.customSampleLocations = true;
    }
    WaitForOptimizedPipelines(context.device);
    const auto lto = std::getenv("APS5_NO_PIPELINE_LTO") == nullptr && context.pipelineLinkTimeOptimization;
    Require(counters().optimized == (lto ? counters().linked : 0u), "a linked pipeline was not optimized in the background");
    Require(first->Optimized() == lto && culled->Optimized() == lto, "a pipeline did not switch to its optimized link");
    ClearCachedPipelines(context.device);
    Require(counters().linked == 0, "clearing the pipelines kept the device's libraries");
    if (context.pipelineCreationFeedback) {
        PipelineCache cache(context, device.Properties(), false);
        auto warm = context;
        warm.pipelineCache = &cache;
        const auto warmLookup = [&] { return CachedPipeline(warm, state, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL); };
        static_cast<void>(warmLookup());
        ClearCachedPipelines(context.device);
        const auto before = PipelineCreationFeedbackTotals();
        static_cast<void>(warmLookup());
        const auto after = PipelineCreationFeedbackTotals();
        ClearCachedPipelines(context.device);
        constexpr auto shaderParts = static_cast<std::size_t>(PipelineCreationKind::LibraryShaderPart);
        constexpr auto interfaceParts = static_cast<std::size_t>(PipelineCreationKind::LibraryInterfacePart);
        Require(after.unreported[shaderParts] == before.unreported[shaderParts] && after.unreported[interfaceParts] == before.unreported[interfaceParts], "the driver reported no creation feedback for a library part");
        Require(after.hits[shaderParts] - before.hits[shaderParts] == 2u && after.misses[shaderParts] == before.misses[shaderParts], "the pre-rasterization and fragment shader library parts created again from a warm pipeline cache reported " + std::to_string(after.hits[shaderParts] - before.hits[shaderParts]) + " hits and " + std::to_string(after.misses[shaderParts] - before.misses[shaderParts]) + " misses");
        Require((after.hits[interfaceParts] - before.hits[interfaceParts]) + (after.misses[interfaceParts] - before.misses[interfaceParts]) == 2u, "the vertex input and fragment output library parts created again did not each report feedback");
    }
    std::cout << "Pipeline library reuse tests passed\n";
}

#ifdef _WIN32
namespace StorageCpuSnapshotTest {
constexpr std::uint32_t Side = 256;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileR64KBX = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t SurfaceBytes = std::size_t{Side} * Side * 4u;
constexpr std::size_t BlockBytes = 12 * SurfaceBytes;

class Block {
public:
    explicit Block(const Context& context, bool watched = false) : context(context), watched(watched) {
        if (watched) {
            memory = GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536);
            const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, BlockBytes, nullptr);
            Require(section != nullptr, "cannot allocate the shared texture section");
            try { GuestArena::GuestArenaMap_nid_postfix(memory, BlockBytes, section, 0, PAGE_READWRITE); }
            catch (...) { CloseHandle(section); throw; }
            CloseHandle(section);
        } else memory = VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        Require(memory != nullptr, "cannot allocate the texture block");
        std::memset(memory, 0x5a, BlockBytes);
        GuestAllocations::Mutation().Add(memory, BlockBytes, true, true, true);
    }

    ~Block() {
        ClearCachedTextures(context.device);
        GuestAllocations::Mutation().Remove(memory);
        HostImportFor(context, Surface(0), BlockBytes);
        if (watched) {
            GuestArena::GuestArenaReset_nid_postfix(memory, BlockBytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, BlockBytes);
        } else VirtualFree(memory, 0, MEM_RELEASE);
    }

    std::uint64_t Surface(std::size_t surface) const { return reinterpret_cast<std::uint64_t>(memory) + surface * SurfaceBytes; }

private:
    const Context& context;
    bool watched;
    void* memory = nullptr;
};

std::array<std::uint32_t, 8> Descriptor(std::uint64_t address, std::uint32_t side = Side) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

GuestTextureResource Resource(std::uint64_t address, std::uint32_t side = Side) {
    const auto words = Descriptor(address, side);
    return DecodeTextureResource(words);
}

void Run(Context context) {
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    for (const bool watched : {true, false}) {
        Block block(context, watched);
        const auto address = block.Surface(0);
        Require(AgcDriver::GuestMemory::Watched(address, SurfaceBytes) == watched, "snapshot fixture has the wrong write tracking");
        Require(HostImportFor(context, address, SurfaceBytes) != nullptr, "snapshot fixture requires a host import");
        Require(AgcDriver::GuestMemory::Watched(address, SurfaceBytes) == watched, "host import changed the fixture's write tracking");
        auto image = StorageTexture::Create(context, *context.detiler, Resource(address), 0);
        const auto expectedBytes = watched ? 0u : SurfaceBytes;
        Require(image->OriginalSnapshotBytes() == expectedBytes, "watched GPU-direct upload retained an unnecessary CPU surface snapshot");
        const auto readPixel = [&] {
            Buffer result(context, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            CommandBatch batch(context);
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {1, 1, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(batch.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, result.Handle(), 1, &copy);
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            batch.SubmitAndWait();
            result.Invalidate();
            std::uint32_t pixel = 0;
            std::memcpy(&pixel, result.Bytes().data(), sizeof(pixel));
            return pixel;
        };
        Require(readPixel() == 0x5a5a5a5au, "snapshot upload returned incorrect GPU texels");
        std::memset(reinterpret_cast<void*>(address), 0x6b, SurfaceBytes);
        image->Refresh();
        Require(readPixel() == 0x6b6b6b6bu, "snapshot refresh missed a CPU write");
        Require(image->OriginalSnapshotBytes() == expectedBytes, "watched refresh allocated a CPU surface snapshot");
        {
            CommandBatch batch(context);
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkClearColorValue clear{};
            clear.uint32[0] = 0x7c7c7c7cu;
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(batch.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            batch.SubmitAndWait();
        }
        image->MarkDirty();
        image->WriteBack();
        std::uint32_t pixel = 0;
        AgcDriver::GuestMemory::ReadCommitted(address, std::as_writable_bytes(std::span(&pixel, 1)));
        Require(pixel == 0x7c7c7c7cu, "snapshot write-back lost GPU texels");
        Require(image->OriginalSnapshotBytes() == expectedBytes, "GPU-direct write-back allocated a CPU surface snapshot");
        if (watched) {
            AgcDriver::GuestMemory::Unwatch(address, SurfaceBytes);
            Require(!AgcDriver::GuestMemory::Watched(address, SurfaceBytes), "snapshot fixture is still write-watched");
            std::memset(reinterpret_cast<void*>(address), 0x8d, SurfaceBytes);
            image->Refresh();
            Require(image->OriginalSnapshotBytes() == SurfaceBytes, "unwatched refresh did not allocate its comparison snapshot");
            Require(readPixel() == 0x8d8d8d8du, "unwatched refresh returned incorrect GPU texels");
        }
        image.reset();
        Context cpu = context;
        cpu.hostImportAlignment = 0;
        image = StorageTexture::Create(cpu, *context.detiler, Resource(address), 0);
        Require(image->OriginalSnapshotBytes() == SurfaceBytes, "CPU upload has no guest-byte snapshot");
        const auto expectedPixel = watched ? 0x8d8d8d8du : 0x7c7c7c7cu;
        Require(readPixel() == expectedPixel, "CPU upload returned incorrect GPU texels");
        {
            CommandBatch batch(context);
            VkClearColorValue clear{};
            clear.uint32[0] = 0x9e9e9e9eu;
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(batch.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            batch.SubmitAndWait();
        }
        image->MarkDirty();
        image->WriteBack();
        AgcDriver::GuestMemory::ReadCommitted(address, std::as_writable_bytes(std::span(&pixel, 1)));
        Require(pixel == 0x9e9e9e9eu, "CPU write-back lost guest texels");
        std::printf("%s snapshot fixture: direct %zu bytes, CPU fallback %zu bytes\n", watched ? "watched" : "unwatched", expectedBytes, image->OriginalSnapshotBytes());
    }
}
}
#endif

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--queued-write-groups-only") {
            queuedWriteGroupTests();
            std::cout << "queued write group tests passed\n";
            return 0;
        }
        std::unique_ptr<Device> created;
        try {
            created = std::make_unique<Device>();
        } catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::cout << "skipped, no usable Vulkan device: " << error.what() << '\n';
            return 77;
        }
        Device& device = *created;
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-retirement-only") {
            const auto alignment = device.GetContext().hostImportAlignment;
            if (alignment == 0 || alignment > 65536) {
                std::cout << "skipped, host import retirement requires host imports aligned within 64 KiB; alignment=" << alignment << '\n';
                return 77;
            }
            hostImportRetirementTests(device);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--pipeline-library-only") {
            pipelineLibraryTests(device);
            return 0;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--benchmark-pipeline-cache" || std::string_view(argv[1]) == "--pipeline-cache-only")) {
            pipelineCacheTests(device, std::string_view(argv[1]) == "--benchmark-pipeline-cache");
            return 0;
        }

        if (argc == 2 && std::string_view(argv[1]) == "--descriptor-pool-only") {
            descriptorPoolTests(device);
            return 0;
        }
        std::unique_lock gpu(GpuMutex());
#ifdef _WIN32
        if (argc == 2 && std::string_view(argv[1]) == "--storage-cpu-snapshot-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                Require(std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr, "storage snapshot test requires host imports");
                std::cout << "skipped, host imports unavailable\n";
                return 77;
            }
            StorageCpuSnapshotTest::Run(device.GetContext());
            std::cout << "Storage CPU snapshot tests passed\n";
            return 0;
        }
#endif
        std::cout << "host imports " << (PrepareImportWatch(device.GetContext()) == ImportWatch::Unwatch ? "are compared" : "stay watched") << '\n';
        Recorder recorder(device.GetContext());
        recorder.Activate();
        if (argc == 2 && std::string_view(argv[1]) == "--pending-write-snapshot-only") {
            pendingWriteSnapshotTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--benchmark-pending-write-snapshot") {
            pendingWriteSnapshotBenchmark(recorder);
            return 0;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--storage-upload-lifetime-only" || std::string_view(argv[1]) == "--storage-upload-failure-only")) {
            storageUploadLifetimeTests(device, recorder, std::string_view(argv[1]) == "--storage-upload-failure-only");
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--tiled-resolve-only") {
            if ((device.GetContext().limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0 || !device.Graphics()) {
                std::cout << "skipped, the device lacks 4x color targets or a graphics queue\n";
                return 77;
            }
            tiledResolveTests(device, recorder);
            std::cout << "CB resolves into tiled targets tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--cube-only") {
            singleCubeTests(device, recorder);
            std::cout << "Single cube snapshot and storage sampling tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--unchanged-cpu-stamps-only") {
            unchangedCpuStampTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--kept-unit-settle-only") {
            keptUnitSettleTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--integer-clear-only") {
            integerClearTests(device);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--settled-baseline-only") {
            settledBaselineTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--completion-labels-only") {
            completionCountTests(device, recorder);
            afterRecordedWorkTests(device, recorder);
            completionWaitTests(device, recorder);
            directCompletionLabelTests(recorder);
            labelTests(recorder);
            labelRegistryTests(recorder);
            largeLabelTests(recorder);
            std::cout << "Completion label storage and ordering tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--written-buffer-order-only") {
            if (!writtenBufferOrderTests(device, recorder)) return 77;
            std::cout << "written buffer order tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--label-match-only") {
            labelMatchTests(recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-unmap-only") {
            gpu.unlock();
            hostImportUnmapTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--slice-dirty-only") {
            sliceDirtyTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--resident-present-only") {
            RunResidentPresentTests(device.GetContext());
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--parked-import-unmap-only") {
            gpu.unlock();
            parkedImportUnmapTests(device);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--import-watch-only") {
            importWatchTests(device);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--split-writeback-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            splitWriteBackTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--cmask-only") {
            cmaskPassTests(device, recorder);
            std::cout << "CMASK copied clear, write-back and failure tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--metadata-pass-only") {
            metadataPassTests(device, recorder);
            std::cout << "Color metadata pass tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--pending-key-stores-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            pendingKeyStoreTests(device, recorder);
            std::cout << "Pending DCC key store tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--sample-dumps-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            if (!device.GetContext().bufferDeviceAddress) {
                std::cout << "skipped, the device lacks buffer device addresses\n";
                return 77;
            }
            sampleDumpTests(device, recorder);
            std::cout << "Occlusion counter dump tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--draw-inputs-in-place-only") {
            drawInputInPlaceTests(device, recorder);
            std::cout << "In-place draw input tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--multisample-only") {
            multisampleGuestAccessTests(device, recorder);
            fmaskDecompressMsaaTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--storage-refresh-only") {
            storageRefreshTests(device, recorder, false);
            storageRefreshTests(device, recorder, true);
            std::cout << "Storage refresh tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--write-back-padding-only") {
            writeBackPaddingTests(device, recorder, false);
            writeBackPaddingTests(device, recorder, true);
            std::cout << "Write-back padding tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--single-pass-only") {
            singlePassTests(device, recorder, false);
            singlePassTests(device, recorder, true);
            std::cout << "Single-pass storage tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--stale-generation-only") {
            staleGenerationTests(device, recorder);
            std::cout << "Stale generation tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--draw-descriptors-only") {
            drawDescriptorWriteTests(device, recorder);
            std::cout << "Draw descriptor writes passed\n";
            return 0;
        }

#ifdef _WIN32
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-second-mapping-only") {
            secondMappingImportTests(device);
            return 0;
        }
#endif
        readTrackingTests(device, recorder);
        writeSettledTests(device, recorder);
        completionCountTests(device, recorder);
        afterRecordedWorkTests(device, recorder);
        directCompletionLabelTests(recorder);
        batchStampTests(recorder);
        labelTests(recorder);
        labelRegistryTests(recorder);
        largeLabelTests(recorder);
        unchangedSinceTests();
        keyProofTests(device, recorder);
        targetKeyProofTests(device, recorder);
        refreshProofTests(device, recorder);
        resourceReadTests(device, recorder);
        multisampleGuestAccessTests(device, recorder);
        tiledResolveTests(device, recorder);
        fmaskDecompressMsaaTests(device, recorder);
        drawDescriptorWriteTests(device, recorder);
        drawInputInPlaceTests(device, recorder);
        RunResidentPresentTests(device.GetContext());
        sliceDirtyTests(device, recorder);
        storeRunTests(device, recorder);
        queuedWriteGroupTests();
        remappedImportTests(device);
#ifdef _WIN32
        secondMappingImportTests(device);
#endif
        movedMetadataTests(device, recorder);
        viewPastLastMipTests(device, recorder);
        keysFillTests(device, recorder);
        integerClearTests(device);
        denormalClearTests(device, recorder);
        storageRefreshTests(device, recorder, false);
        storageRefreshTests(device, recorder, true);
        static_cast<void>(writtenBufferOrderTests(device, recorder));
        writeBackPaddingTests(device, recorder, false);
        writeBackPaddingTests(device, recorder, true);
        splitWriteBackTests(device, recorder);
        singlePassTests(device, recorder, false);
        singlePassTests(device, recorder, true);
        unchangedCpuStampTests(device, recorder);
        keptUnitSettleTests(device, recorder);
        settledBaselineTests(device, recorder);
        importWatchTests(device);
        staleGenerationTests(device, recorder);
        importWindowTests(device, recorder);
        imageMemoTests(device, recorder);
        minLodTests(device, recorder);
        firstLayerViewTests(device, recorder);
        singleCubeTests(device, recorder);
        atomicViewTests(device, recorder);
        metadataPassTests(device, recorder);
        cmaskPassTests(device, recorder);
        pendingKeyStoreTests(device, recorder);
        sampleDumpTests(device, recorder);
        gpu.unlock();
        hostImportUnmapTests(device, recorder);
        gpu.lock();
        recorder.Activate();
        std::cout << "Recorder read tracking and label tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
