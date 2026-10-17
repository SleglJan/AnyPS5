#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include <optional>
#include "ShaderCacheDirectory.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

constexpr std::uint32_t FileMagic = 0x50565041u;
constexpr std::uint32_t FileFormat = 1;

struct FileHeader {
    std::uint32_t magic;
    std::uint32_t format;
    std::uint64_t dataBytes;
    std::uint64_t dataHash;
};
static_assert(sizeof(FileHeader) == 24);

bool profiling() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

double millisecondsSince(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

std::mutex& fileMutex() {
    static std::mutex mutex;
    return mutex;
}

std::string sessionPrefix(const std::filesystem::path& base) {
    return base.stem().string() + ".session-";
}

}

PipelineCache::PipelineCache(const Context& context, const VkPhysicalDeviceProperties& properties, bool creationCacheControl) : context(context), properties(properties), cacheControl(creationCacheControl) {
    const auto directory = ShaderRecompiler::ShaderCacheDirectory();
    if (!directory.empty() && !creationCacheControl) std::fprintf(stderr, "[pipeline-cache] not kept on disk: the device lacks pipelineCreationCacheControl (VK_EXT_pipeline_creation_cache_control)\n");
    if (!directory.empty() && creationCacheControl) {
        std::string uuid;
        for (const auto byte : properties.pipelineCacheUUID) {
            char digits[3];
            std::snprintf(digits, sizeof(digits), "%02x", static_cast<unsigned>(byte));
            uuid += digits;
        }
        char name[160];
        std::snprintf(name, sizeof(name), "vk-pipelines-%04x-%04x-%08x-%s.bin", properties.vendorID, properties.deviceID, properties.driverVersion, uuid.c_str());
        path = directory / name;
    }
    {
        std::vector<std::byte> file;
        std::span<const std::byte> data;
        if (!path.empty() && read(path, file, data)) {
            fileHash = ShaderRecompiler::HashBytes(data);
            std::fprintf(stderr, "[pipeline-cache] loaded %.1f KiB from %s\n", static_cast<double>(data.size()) / 1024.0, path.string().c_str());
        }
        base = create(data);
    }
    if (path.empty()) return;
    const auto merge = context.Function<PFN_vkMergePipelineCaches>("vkMergePipelineCaches");
    const auto destroy = context.Function<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache");
    const auto prefix = sessionPrefix(path);
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path(), error)) {
        const auto name = entry.path().filename().string();
        if (!name.starts_with(prefix) || entry.path().extension() != ".bin") continue;
        std::vector<std::byte> file;
        std::span<const std::byte> data;
        if (read(entry.path(), file, data)) {
            if (const auto earlier = create(data); earlier != VK_NULL_HANDLE) {
                Check(merge(context.device, base, 1, &earlier), "vkMergePipelineCaches session");
                destroy(context.device, earlier, nullptr);
                std::fprintf(stderr, "[pipeline-cache] merged %.1f KiB from %s\n", static_cast<double>(data.size()) / 1024.0, entry.path().string().c_str());
            }
        }
        mergedSessions.push_back(entry.path());
    }
    baseStale = !mergedSessions.empty();
    session = create({});
    const auto getData = context.Function<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData");
    Check(getData(context.device, session, &savedSessionBytes, nullptr), "vkGetPipelineCacheData session");
    initialSessionBytes = savedSessionBytes;
    char id[17];
    std::snprintf(id, sizeof(id), "%016llx", static_cast<unsigned long long>((static_cast<std::uint64_t>(std::random_device{}()) << 32u) ^ static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count())));
    sessionPath = path.parent_path() / (prefix + id + ".bin");
    saver = std::thread([this] { run(); });
}

PipelineCache::~PipelineCache() {
    if (saver.joinable()) {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        saver.join();
        save(true);
    }
    const auto destroy = context.Function<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache");
    if (session != VK_NULL_HANDLE) destroy(context.device, session, nullptr);
    destroy(context.device, base, nullptr);
}

VkResult PipelineCache::CreateCompute(const Context& caller, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words) const {
    const auto create = caller.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    std::optional<CreationGate> gate;
    if (session == VK_NULL_HANDLE) {
        if (words != 0) gate.emplace(words);
        return CreationThread::Run(words, [&] { return create(caller.device, base, 1, &info, nullptr, pipeline); });
    }
    auto lookup = info;
    lookup.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT;
    const auto found = create(caller.device, base, 1, &lookup, nullptr, pipeline);
    if (found != VK_PIPELINE_COMPILE_REQUIRED_EXT) return found;
    if (words != 0) gate.emplace(words);
    return CreationThread::Run(words, [&] { return create(caller.device, session, 1, &info, nullptr, pipeline); });
}

VkResult PipelineCache::CreateGraphics(const Context& caller, const VkGraphicsPipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words) const {
    const auto create = caller.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines");
    std::optional<CreationGate> gate;
    if (session == VK_NULL_HANDLE) {
        if (words != 0) gate.emplace(words);
        return CreationThread::Run(words, [&] { return create(caller.device, base, 1, &info, nullptr, pipeline); });
    }
    auto lookup = info;
    lookup.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT;
    const auto found = create(caller.device, base, 1, &lookup, nullptr, pipeline);
    if (found != VK_PIPELINE_COMPILE_REQUIRED_EXT) return found;
    if (words != 0) gate.emplace(words);
    return CreationThread::Run(words, [&] { return create(caller.device, session, 1, &info, nullptr, pipeline); });
}

VkResult PipelineCache::LookupCompute(const Context& caller, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline) const {
    if (!cacheControl) return VK_PIPELINE_COMPILE_REQUIRED_EXT;
    const auto create = caller.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    auto lookup = info;
    lookup.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT;
    for (const auto cache : {base, session}) {
        if (cache == VK_NULL_HANDLE) continue;
        const auto found = create(caller.device, cache, 1, &lookup, nullptr, pipeline);
        if (found != VK_PIPELINE_COMPILE_REQUIRED_EXT) return found;
    }
    return VK_PIPELINE_COMPILE_REQUIRED_EXT;
}

VkResult PipelineCache::CreateComputeFrom(const Context& caller, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::span<const std::byte> data) const {
    if (!cacheControl || data.empty()) return VK_PIPELINE_COMPILE_REQUIRED_EXT;
    const auto compiled = create(data);
    if (compiled == VK_NULL_HANDLE) return VK_PIPELINE_COMPILE_REQUIRED_EXT;
    auto lookup = info;
    lookup.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT;
    const auto created = caller.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(caller.device, compiled, 1, &lookup, nullptr, pipeline);
    if (created == VK_SUCCESS) Check(context.Function<PFN_vkMergePipelineCaches>("vkMergePipelineCaches")(context.device, session != VK_NULL_HANDLE ? session : base, 1, &compiled), "vkMergePipelineCaches compiled");
    context.Function<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache")(context.device, compiled, nullptr);
    return created;
}

void PipelineCache::Save() {
    if (!path.empty()) save(false);
}

bool PipelineCache::read(const std::filesystem::path& file, std::vector<std::byte>& bytes, std::span<const std::byte>& data) const {
    if (!ShaderRecompiler::ReadWholeFile(file, bytes)) return false;
    const auto reject = [&](const char* why) {
        std::fprintf(stderr, "[pipeline-cache] ignoring %s: %s\n", file.string().c_str(), why);
        return false;
    };
    FileHeader header{};
    if (bytes.size() < sizeof(header)) return reject("truncated");
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (header.magic != FileMagic || header.format != FileFormat) return reject("another format");
    if (header.dataBytes != bytes.size() - sizeof(header)) return reject("truncated");
    const auto payload = std::span<const std::byte>(bytes).subspan(sizeof(header));
    if (ShaderRecompiler::HashBytes(payload) != header.dataHash) return reject("checksum mismatch");
    struct VulkanHeader {
        std::uint32_t headerSize;
        std::uint32_t headerVersion;
        std::uint32_t vendorID;
        std::uint32_t deviceID;
        std::uint8_t uuid[VK_UUID_SIZE];
    } vulkan{};
    static_assert(sizeof(VulkanHeader) == 32);
    if (payload.size() < sizeof(vulkan)) return reject("no Vulkan header");
    std::memcpy(&vulkan, payload.data(), sizeof(vulkan));
    if (vulkan.headerSize < sizeof(vulkan) || vulkan.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE || vulkan.vendorID != properties.vendorID || vulkan.deviceID != properties.deviceID || std::memcmp(vulkan.uuid, properties.pipelineCacheUUID, VK_UUID_SIZE) != 0) return reject("made for another device or driver");
    data = payload;
    return true;
}

VkPipelineCache PipelineCache::create(std::span<const std::byte> data) const {
    const auto createCache = context.Function<PFN_vkCreatePipelineCache>("vkCreatePipelineCache");
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = data.size();
    info.pInitialData = data.empty() ? nullptr : data.data();
    VkPipelineCache result = VK_NULL_HANDLE;
    const auto created = createCache(context.device, &info, nullptr, &result);
    if (created == VK_SUCCESS) return result;
    if (data.empty()) Check(created, "vkCreatePipelineCache");
    std::fprintf(stderr, "[pipeline-cache] the driver refused %.1f KiB of cache data (Vulkan result %d)\n", static_cast<double>(data.size()) / 1024.0, static_cast<int>(created));
    return base == VK_NULL_HANDLE ? create({}) : VK_NULL_HANDLE;
}

bool PipelineCache::write(VkPipelineCache source, const std::filesystem::path& file, std::size_t& bytes, std::uint64_t* hash) const {
    const auto getData = context.Function<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData");
    std::size_t size = 0;
    if (getData(context.device, source, &size, nullptr) != VK_SUCCESS) return false;
    std::vector<std::byte> contents(sizeof(FileHeader) + size);
    const auto result = getData(context.device, source, &size, contents.data() + sizeof(FileHeader));
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) return false;
    contents.resize(sizeof(FileHeader) + size);
    const FileHeader header{FileMagic, FileFormat, size, ShaderRecompiler::HashBytes(std::span(contents).subspan(sizeof(FileHeader)))};
    std::memcpy(contents.data(), &header, sizeof(header));
    if (!ShaderRecompiler::WriteFileAtomically(file, contents)) {
        std::fprintf(stderr, "[pipeline-cache] cannot write %s\n", file.string().c_str());
        return false;
    }
    bytes = size;
    if (hash != nullptr) *hash = header.dataHash;
    return true;
}

void PipelineCache::save(bool final) {
    std::lock_guard lock(saveMutex);
    const auto getData = context.Function<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData");
    std::size_t size = 0;
    if (getData(context.device, session, &size, nullptr) != VK_SUCCESS) return;
    const auto started = std::chrono::steady_clock::now();
    if (final) {
        if (!baseStale && size == initialSessionBytes) return;
        const auto merge = context.Function<PFN_vkMergePipelineCaches>("vkMergePipelineCaches");
        Check(merge(context.device, base, 1, &session), "vkMergePipelineCaches teardown");
        std::lock_guard fileLock(fileMutex());
        std::vector<std::byte> file;
        std::span<const std::byte> data;
        VkPipelineCache merged = VK_NULL_HANDLE;
        if (read(path, file, data) && ShaderRecompiler::HashBytes(data) != fileHash) merged = create(data);
        if (merged != VK_NULL_HANDLE) Check(merge(context.device, merged, 1, &base), "vkMergePipelineCaches file");
        std::size_t bytes = 0;
        const bool written = write(merged != VK_NULL_HANDLE ? merged : base, path, bytes, &fileHash);
        if (merged != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache")(context.device, merged, nullptr);
        if (!written) return;
        std::error_code error;
        for (const auto& merged : mergedSessions) std::filesystem::remove(merged, error);
        std::filesystem::remove(sessionPath, error);
        std::fprintf(stderr, "[pipeline-cache] saved %.1f KiB to %s with this session and %zu earlier ones merged in %.1f ms (teardown)\n", static_cast<double>(bytes) / 1024.0, path.string().c_str(), mergedSessions.size(), millisecondsSince(started));
        mergedSessions.clear();
        baseStale = false;
        return;
    }
    if (size == savedSessionBytes) return;
    std::size_t bytes = 0;
    if (!write(session, sessionPath, bytes)) return;
    savedSessionBytes = bytes;
    if (profiling()) std::fprintf(stderr, "[pipeline-cache] saved this session's %.1f KiB to %s in %.1f ms\n", static_cast<double>(bytes) / 1024.0, sessionPath.string().c_str(), millisecondsSince(started));
}

void PipelineCache::run() {
    std::unique_lock lock(mutex);
    while (!wake.wait_for(lock, std::chrono::seconds(SaveIntervalSeconds), [&] { return stopping; })) {
        lock.unlock();
        save(false);
        lock.lock();
    }
}

namespace {

struct FeedbackTotals {
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(PipelineCreationKind::Count)> hits{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(PipelineCreationKind::Count)> misses{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(PipelineCreationKind::Count)> unreported{};
};

FeedbackTotals& Feedback() {
    static auto* totals = new FeedbackTotals;
    return *totals;
}

PipelineCreationKind GraphicsKind(const VkGraphicsPipelineCreateInfo& info) {
    if ((info.flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) != 0) return info.stageCount != 0 ? PipelineCreationKind::LibraryShaderPart : PipelineCreationKind::LibraryInterfacePart;
    for (auto* next = static_cast<const VkBaseInStructure*>(info.pNext); next != nullptr; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR) return PipelineCreationKind::Linked;
    }
    return PipelineCreationKind::Graphics;
}

template<typename TInfo, typename TCreate>
VkResult WithFeedback(const Context& context, const TInfo& info, PipelineCreationKind kind, TCreate&& create, PipelineCacheOutcome* outcome = nullptr) {
    if (outcome != nullptr) *outcome = PipelineCacheOutcome::Unreported;
    if (!context.pipelineCreationFeedback) return create(info);
    VkPipelineCreationFeedbackEXT feedback{};
    VkPipelineCreationFeedbackCreateInfoEXT chained{VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT, info.pNext, &feedback, 0, nullptr};
    auto withFeedback = info;
    withFeedback.pNext = &chained;
    const auto result = create(withFeedback);
    if (result != VK_SUCCESS) return result;
    auto& totals = Feedback();
    const auto index = static_cast<std::size_t>(kind);
    const auto observed = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT_EXT) == 0 ? PipelineCacheOutcome::Unreported : (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT_EXT) != 0 ? PipelineCacheOutcome::Hit : PipelineCacheOutcome::Miss;
    if (outcome != nullptr) *outcome = observed;
    if (observed == PipelineCacheOutcome::Unreported) totals.unreported[index].fetch_add(1, std::memory_order_relaxed);
    else if (observed == PipelineCacheOutcome::Hit) totals.hits[index].fetch_add(1, std::memory_order_relaxed);
    else totals.misses[index].fetch_add(1, std::memory_order_relaxed);
    return result;
}

VkResult CreateComputeUnobserved(const Context& context, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words) {
    if (context.pipelineCache != nullptr) return context.pipelineCache->CreateCompute(context, info, pipeline, words);
    std::optional<CreationGate> gate;
    if (words != 0) gate.emplace(words);
    const auto create = context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    return CreationThread::Run(words, [&] { return create(context.device, VK_NULL_HANDLE, 1, &info, nullptr, pipeline); });
}

VkResult CreateGraphicsUnobserved(const Context& context, const VkGraphicsPipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words) {
    if (context.pipelineCache != nullptr) return context.pipelineCache->CreateGraphics(context, info, pipeline, words);
    std::optional<CreationGate> gate;
    if (words != 0) gate.emplace(words);
    const auto create = context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines");
    return CreationThread::Run(words, [&] { return create(context.device, VK_NULL_HANDLE, 1, &info, nullptr, pipeline); });
}

}

VkResult CreateComputePipeline(const Context& context, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words) {
    return CreationPool::AtUse([&] { return WithFeedback(context, info, PipelineCreationKind::Compute, [&](const VkComputePipelineCreateInfo& observed) { return CreateComputeUnobserved(context, observed, pipeline, words); }); });
}

VkResult CreateGraphicsPipeline(const Context& context, const VkGraphicsPipelineCreateInfo& info, VkPipeline* pipeline, std::size_t words) {
    return CreationPool::AtUse([&] { return WithFeedback(context, info, GraphicsKind(info), [&](const VkGraphicsPipelineCreateInfo& observed) { return CreateGraphicsUnobserved(context, observed, pipeline, words); }); });
}

PipelineCreationFeedbackCounts PipelineCreationFeedbackTotals() {
    auto& totals = Feedback();
    PipelineCreationFeedbackCounts counts;
    for (std::size_t kind = 0; kind < counts.hits.size(); ++kind) {
        counts.hits[kind] = totals.hits[kind].load(std::memory_order_relaxed);
        counts.misses[kind] = totals.misses[kind].load(std::memory_order_relaxed);
        counts.unreported[kind] = totals.unreported[kind].load(std::memory_order_relaxed);
    }
    return counts;
}

std::string PipelineCreationFeedbackReport() {
    const auto counts = PipelineCreationFeedbackTotals();
    constexpr std::array<const char*, static_cast<std::size_t>(PipelineCreationKind::Count)> names{"compute", "graphics", "library shader parts", "library interface parts", "linked"};
    std::string line = " pipeline cache feedback:";
    for (std::size_t kind = 0; kind < names.size(); ++kind) {
        char text[128];
        std::snprintf(text, sizeof(text), " %s %llu hits, %llu misses, %llu unreported;", names[kind], static_cast<unsigned long long>(counts.hits[kind]), static_cast<unsigned long long>(counts.misses[kind]), static_cast<unsigned long long>(counts.unreported[kind]));
        line += text;
    }
    return line;
}

bool PipelineCacheChecksHits(const Context& context) {
    return context.pipelineCache != nullptr && context.pipelineCache->ChecksHits();
}

VkResult LookupComputePipeline(const Context& context, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline) {
    if (context.pipelineCache == nullptr) return VK_PIPELINE_COMPILE_REQUIRED_EXT;
    return CreationPool::AtUse([&] { return context.pipelineCache->LookupCompute(context, info, pipeline); });
}

VkResult CreateComputePipelineFrom(const Context& context, const VkComputePipelineCreateInfo& info, VkPipeline* pipeline, std::span<const std::byte> data, PipelineCacheOutcome* outcome) {
    if (outcome != nullptr) *outcome = PipelineCacheOutcome::Unreported;
    if (context.pipelineCache == nullptr) return VK_PIPELINE_COMPILE_REQUIRED_EXT;
    return CreationPool::AtUse([&] { return WithFeedback(context, info, PipelineCreationKind::Compute, [&](const VkComputePipelineCreateInfo& observed) { return context.pipelineCache->CreateComputeFrom(context, observed, pipeline, data); }, outcome); });
}


}
