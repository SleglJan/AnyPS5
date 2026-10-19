#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

struct CacheUse {
    std::atomic<int> readers{0};
    std::atomic<int> writers{0};
};

struct FakeDevice {
    std::mutex mutex;
    std::map<VkPipelineCache, std::unique_ptr<CacheUse>> caches;
    std::uintptr_t next = 0x1000;
    std::atomic<std::uint64_t> overlaps{0};
    std::atomic<std::uint64_t> merged{0};
    std::atomic<std::uint64_t> created{0};
    VkPipelineCache base = VK_NULL_HANDLE;
    VkPipelineCache session = VK_NULL_HANDLE;
    std::atomic<bool> holdSession{false};
    std::atomic<bool> sessionEntered{false};
    std::atomic<bool> releaseSession{false};
    std::mutex holdMutex;
    std::condition_variable holdWake;

    CacheUse& use(VkPipelineCache cache) {
        std::lock_guard lock(mutex);
        auto& slot = caches[cache];
        if (!slot) slot = std::make_unique<CacheUse>();
        return *slot;
    }
};

FakeDevice* fake = nullptr;

struct ReadUse {
    CacheUse& use;
    explicit ReadUse(CacheUse& use) : use(use) {
        use.readers.fetch_add(1);
        if (use.writers.load() != 0) fake->overlaps.fetch_add(1);
    }
    ~ReadUse() { use.readers.fetch_sub(1); }
};

struct WriteUse {
    CacheUse& use;
    explicit WriteUse(CacheUse& use) : use(use) {
        if (use.writers.fetch_add(1) != 0) fake->overlaps.fetch_add(1);
        if (use.readers.load() != 0) fake->overlaps.fetch_add(1);
    }
    ~WriteUse() { use.writers.fetch_sub(1); }
};

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(VkDevice, const VkPipelineCacheCreateInfo*, const VkAllocationCallbacks*, VkPipelineCache* cache) {
    std::lock_guard lock(fake->mutex);
    *cache = reinterpret_cast<VkPipelineCache>(fake->next);
    fake->next += 0x10;
    fake->caches[*cache] = std::make_unique<CacheUse>();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(VkDevice, VkPipelineCache cache, const VkAllocationCallbacks*) {
    WriteUse write(fake->use(cache));
}

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineCacheData(VkDevice, VkPipelineCache cache, std::size_t* size, void*) {
    ReadUse read(fake->use(cache));
    *size = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL MergePipelineCaches(VkDevice, VkPipelineCache destination, std::uint32_t count, const VkPipelineCache* sources) {
    WriteUse write(fake->use(destination));
    std::vector<std::unique_ptr<ReadUse>> reads;
    for (std::uint32_t index = 0; index < count; ++index) reads.push_back(std::make_unique<ReadUse>(fake->use(sources[index])));
    std::this_thread::yield();
    fake->merged.fetch_add(count);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice, VkPipelineCache cache, std::uint32_t, const VkComputePipelineCreateInfo* infos, const VkAllocationCallbacks*, VkPipeline* pipelines) {
    ReadUse read(fake->use(cache));
    const bool lookup = (infos[0].flags & VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT) != 0;
    if (lookup && (cache == fake->base || cache == fake->session)) return VK_PIPELINE_COMPILE_REQUIRED_EXT;
    if (cache == fake->session && fake->holdSession.exchange(false)) {
        fake->sessionEntered.store(true);
        fake->holdWake.notify_all();
        std::unique_lock lock(fake->holdMutex);
        fake->holdWake.wait(lock, [] { return fake->releaseSession.load(); });
    }
    std::this_thread::yield();
    fake->created.fetch_add(1);
    pipelines[0] = reinterpret_cast<VkPipeline>(std::uintptr_t{0x10});
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL DeviceProc(VkDevice, const char* name) {
    const std::string function(name);
    if (function == "vkCreatePipelineCache") return reinterpret_cast<PFN_vkVoidFunction>(&CreatePipelineCache);
    if (function == "vkDestroyPipelineCache") return reinterpret_cast<PFN_vkVoidFunction>(&DestroyPipelineCache);
    if (function == "vkGetPipelineCacheData") return reinterpret_cast<PFN_vkVoidFunction>(&GetPipelineCacheData);
    if (function == "vkMergePipelineCaches") return reinterpret_cast<PFN_vkVoidFunction>(&MergePipelineCaches);
    if (function == "vkCreateComputePipelines") return reinterpret_cast<PFN_vkVoidFunction>(&CreateComputePipelines);
    return nullptr;
}

VkComputePipelineCreateInfo Info(const char* name) {
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, VK_NULL_HANDLE, name, nullptr};
    return info;
}

}

int main() {
    FakeDevice device;
    fake = &device;
    const auto directory = std::filesystem::temp_directory_path() / ("anyps5-pipeline-cache-merge-access-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    SetVariable("ANYPS5_NO_SHADER_CACHE", "");
    SetVariable("ANYPS5_SHADER_CACHE_DIR", directory.string());
    Context context{};
    context.device = reinterpret_cast<VkDevice>(std::uintptr_t{0x8});
    context.deviceProc = &DeviceProc;
    VkPhysicalDeviceProperties properties{};
    const std::vector<std::byte> data(64, std::byte{1});
    {
        auto cache = std::make_unique<PipelineCache>(context, properties, true);
        auto with = context;
        with.pipelineCache = cache.get();
        {
            std::lock_guard lock(device.mutex);
            Require(device.caches.size() == 2, "the pipeline cache did not create one base and one session cache");
            device.base = device.caches.begin()->first;
            device.session = std::prev(device.caches.end())->first;
        }
        device.holdSession.store(true);
        std::thread creator([&] {
            VkPipeline pipeline = VK_NULL_HANDLE;
            Require(cache->CreateCompute(with, Info("main"), &pipeline) == VK_SUCCESS, "a creation from the session cache failed");
        });
        {
            std::unique_lock lock(device.holdMutex);
            device.holdWake.wait(lock, [&] { return device.sessionEntered.load(); });
        }
        VkPipeline pipeline = VK_NULL_HANDLE;
        Require(cache->CreateComputeFrom(with, Info("compiled"), &pipeline, data) == VK_SUCCESS, "a creation from compiled cache data failed");
        Require(device.overlaps.load() == 0, "vkMergePipelineCaches wrote the session cache while another thread was creating a pipeline from it");
        {
            std::lock_guard lock(device.holdMutex);
            device.releaseSession.store(true);
        }
        device.holdWake.notify_all();
        creator.join();

        constexpr int Threads = 8;
        constexpr int PerThread = 400;
        const auto started = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        for (int thread = 0; thread < Threads; ++thread) {
            threads.emplace_back([&, thread] {
                for (int index = 0; index < PerThread; ++index) {
                    VkPipeline created = VK_NULL_HANDLE;
                    if ((index + thread) % 4 == 0) Require(cache->CreateComputeFrom(with, Info("compiled"), &created, data) == VK_SUCCESS, "a creation from compiled cache data failed");
                    else Require(cache->CreateCompute(with, Info("main"), &created) == VK_SUCCESS, "a creation from the session cache failed");
                }
            });
        }
        for (auto& thread : threads) thread.join();
        const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
        std::printf("pipeline cache merge access: %d creations on %d threads, %.2f us per creation, %llu merged before teardown\n", Threads * PerThread, Threads, elapsed / (Threads * PerThread), static_cast<unsigned long long>(device.merged.load()));
        Require(device.overlaps.load() == 0, "vkMergePipelineCaches overlapped another use of its destination cache under concurrent creations");
    }
    const std::uint64_t expectedMerges = 1 + 8 * 400 / 4;
    Require(device.merged.load() == expectedMerges, "a compiled cache was not merged into the session cache by teardown: " + std::to_string(device.merged.load()) + " of " + std::to_string(expectedMerges));
    Require(device.overlaps.load() == 0, "a pipeline cache was used during a merge into it");
    std::filesystem::remove_all(directory);
    std::printf("pipeline cache merge access: no merge overlapped a creation from its destination cache\n");
    return 0;
}
