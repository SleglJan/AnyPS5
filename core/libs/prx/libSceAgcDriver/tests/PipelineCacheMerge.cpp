#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "PipelineCacheFill_spv.h"
#include <SDL_loadso.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Context;
using AgcDriver::Graphics::PipelineCache;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

void SetVariable(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) unsetenv(name);
    else setenv(name, value.c_str(), 1);
#endif
}

struct Device {
    void* library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr deviceProc = nullptr;

    template <class TFunction>
    TFunction Instance(const char* name) const { return reinterpret_cast<TFunction>(instanceProc(instance, name)); }

    Device() {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        Require(library != nullptr, "cannot load Vulkan");
        instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
        Require(instanceProc != nullptr, "missing vkGetInstanceProcAddr");
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &application;
        Require(Instance<PFN_vkCreateInstance>("vkCreateInstance")(&instanceInfo, nullptr, &instance) == VK_SUCCESS, "vkCreateInstance");
        std::uint32_t count = 0;
        const auto enumerate = Instance<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
        Require(enumerate(instance, &count, nullptr) == VK_SUCCESS && count != 0, "no Vulkan device");
        std::vector<VkPhysicalDevice> devices(count);
        enumerate(instance, &count, devices.data());
        const auto extensionsOf = Instance<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        const auto propertiesOf = Instance<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
        int rank = -1;
        for (const auto candidate : devices) {
            std::uint32_t extensions = 0;
            extensionsOf(candidate, nullptr, &extensions, nullptr);
            std::vector<VkExtensionProperties> list(extensions);
            extensionsOf(candidate, nullptr, &extensions, list.data());
            bool control = false;
            for (const auto& extension : list) control = control || std::strcmp(extension.extensionName, VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME) == 0;
            VkPhysicalDeviceProperties candidateProperties{};
            propertiesOf(candidate, &candidateProperties);
            const int candidateRank = candidateProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : candidateProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
            if (!control || candidateRank <= rank) continue;
            rank = candidateRank;
            physical = candidate;
            properties = candidateProperties;
        }
        Require(physical != VK_NULL_HANDLE, "no Vulkan device offers VK_EXT_pipeline_creation_cache_control");
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkPhysicalDevicePipelineCreationCacheControlFeaturesEXT control{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT, nullptr, VK_TRUE};
        const char* extension = VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME;
        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &control};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queue;
        deviceInfo.enabledExtensionCount = 1;
        deviceInfo.ppEnabledExtensionNames = &extension;
        Require(Instance<PFN_vkCreateDevice>("vkCreateDevice")(physical, &deviceInfo, nullptr, &device) == VK_SUCCESS, "vkCreateDevice");
        deviceProc = Instance<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
    }

    ~Device() {
        if (device != VK_NULL_HANDLE) reinterpret_cast<PFN_vkDestroyDevice>(deviceProc(device, "vkDestroyDevice"))(device, nullptr);
        if (instance != VK_NULL_HANDLE) Instance<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        if (library != nullptr) SDL_UnloadObject(library);
    }
};

}

int main() {
    Device gpu;
    Context context{};
    context.device = gpu.device;
    context.physical = gpu.physical;
    context.deviceProc = gpu.deviceProc;
    const auto function = [&](const char* name) { return gpu.deviceProc(gpu.device, name); };
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = sizeof(PIPELINE_CACHE_FILL_SPV);
    moduleInfo.pCode = PIPELINE_CACHE_FILL_SPV;
    VkShaderModule module = VK_NULL_HANDLE;
    Require(reinterpret_cast<PFN_vkCreateShaderModule>(function("vkCreateShaderModule"))(gpu.device, &moduleInfo, nullptr, &module) == VK_SUCCESS, "vkCreateShaderModule");
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setInfo.bindingCount = 1;
    setInfo.pBindings = &binding;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    Require(reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(function("vkCreateDescriptorSetLayout"))(gpu.device, &setInfo, nullptr, &setLayout) == VK_SUCCESS, "vkCreateDescriptorSetLayout");
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    Require(reinterpret_cast<PFN_vkCreatePipelineLayout>(function("vkCreatePipelineLayout"))(gpu.device, &layoutInfo, nullptr, &layout) == VK_SUCCESS, "vkCreatePipelineLayout");
    const auto destroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(function("vkDestroyPipeline"));
    const auto seed = static_cast<std::uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    const auto build = [&](const Context& with, std::uint32_t value) {
        const VkSpecializationMapEntry entry{0, 0, sizeof(value)};
        const VkSpecializationInfo specialization{1, &entry, sizeof(value), &value};
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", &specialization};
        info.layout = layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        Require(AgcDriver::Graphics::CreateComputePipeline(with, info, &pipeline) == VK_SUCCESS && pipeline != VK_NULL_HANDLE, "a compute pipeline was not created through the pipeline cache");
        destroyPipeline(gpu.device, pipeline, nullptr);
    };
    const auto directory = std::filesystem::temp_directory_path() / ("anyps5-pipeline-cache-merge-" + std::to_string(seed));
    std::filesystem::create_directories(directory);
    SetVariable("ANYPS5_NO_SHADER_CACHE", "");
    SetVariable("ANYPS5_SHADER_CACHE_DIR", directory.string());
    constexpr std::uint32_t Pipelines = 6;
    constexpr std::uint32_t SecondOffset = 1000;
    std::filesystem::path base;
    bool kept = true;
    {
        auto first = std::make_unique<PipelineCache>(context, gpu.properties, true);
        auto second = std::make_unique<PipelineCache>(context, gpu.properties, true);
        Require(first->BasePath() == second->BasePath(), "two pipeline caches of one device do not share their base file");
        base = first->BasePath();
        auto withFirst = context;
        withFirst.pipelineCache = first.get();
        auto withSecond = context;
        withSecond.pipelineCache = second.get();
        for (std::uint32_t index = 0; index < Pipelines; ++index) build(withFirst, seed + index);
        for (std::uint32_t index = 0; index < Pipelines; ++index) build(withSecond, seed + SecondOffset + index);
        first->Save();
        second->Save();
        kept = std::filesystem::exists(first->SessionPath()) && std::filesystem::exists(second->SessionPath());
        first.reset();
        Require(!kept || std::filesystem::exists(base), "the first cache's teardown wrote no base file");
        second.reset();
    }
    if (!kept) {
        std::printf("pipeline cache merge: the device keeps no pipeline data in its caches; the merge checks were not run\n");
        std::filesystem::remove_all(directory);
        return 0;
    }
    {
        PipelineCache cache(context, gpu.properties, true);
        auto with = context;
        with.pipelineCache = &cache;
        for (std::uint32_t index = 0; index < Pipelines; ++index) build(with, seed + index);
        cache.Save();
        Require(!std::filesystem::exists(cache.SessionPath()), "the pipelines the first cache built are missing from the base file after the second cache's teardown: the second teardown overwrote the file instead of merging with it");
        for (std::uint32_t index = 0; index < Pipelines; ++index) build(with, seed + SecondOffset + index);
        cache.Save();
        Require(!std::filesystem::exists(cache.SessionPath()), "the pipelines the second cache built are missing from the base file after its teardown");
    }
    reinterpret_cast<PFN_vkDestroyPipelineLayout>(function("vkDestroyPipelineLayout"))(gpu.device, layout, nullptr);
    reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(function("vkDestroyDescriptorSetLayout"))(gpu.device, setLayout, nullptr);
    reinterpret_cast<PFN_vkDestroyShaderModule>(function("vkDestroyShaderModule"))(gpu.device, module, nullptr);
    std::filesystem::remove_all(directory);
    std::printf("pipeline cache merge: two caches on one file keep both caches' pipelines after their teardowns\n");
    return 0;
}
