#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINECACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINECACHE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace AgcDriver::Graphics {

class PipelineCache {
public:
    static constexpr int SaveIntervalSeconds = 10;

    PipelineCache(const Context& context, const VkPhysicalDeviceProperties& properties, bool creationCacheControl);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    VkResult CreateCompute(const Context& caller, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words = 0) const;
    VkResult CreateGraphics(const Context& caller, const VkGraphicsPipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words = 0) const;
    bool ChecksHits() const { return cacheControl; }
    VkResult LookupCompute(const Context& caller, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline) const;
    VkResult CreateComputeFrom(const Context& caller, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::span<const std::byte> data) const;
    void Save();
    const std::filesystem::path& BasePath() const { return path; }
    const std::filesystem::path& SessionPath() const { return sessionPath; }

private:
    bool read(const std::filesystem::path& file, std::vector<std::byte>& bytes, std::span<const std::byte>& data) const;
    VkPipelineCache create(std::span<const std::byte> data) const;
    bool write(VkPipelineCache source, const std::filesystem::path& file, std::size_t& bytes, std::uint64_t* hash = nullptr) const;
    void save(bool final);
    void run();

    Context context;
    VkPhysicalDeviceProperties properties{};
    VkPipelineCache base = VK_NULL_HANDLE;
    VkPipelineCache session = VK_NULL_HANDLE;
    std::filesystem::path path;
    std::filesystem::path sessionPath;
    std::vector<std::filesystem::path> mergedSessions;
    bool baseStale = false;
    bool cacheControl = false;
    std::uint64_t fileHash = 0;
    std::size_t savedSessionBytes = 0;
    std::size_t initialSessionBytes = 0;
    std::mutex saveMutex;
    std::mutex mutex;
    std::condition_variable wake;
    bool stopping = false;
    std::thread saver;
};

VkResult CreateComputePipeline(const Context& context, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words = 0);
VkResult CreateGraphicsPipeline(const Context& context, const VkGraphicsPipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words = 0);
bool PipelineCacheChecksHits(const Context& context);
VkResult LookupComputePipeline(const Context& context, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline);
VkResult CreateComputePipelineFrom(const Context& context, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::span<const std::byte> data);

}

#endif
