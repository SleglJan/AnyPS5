#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERRESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERRESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <array>
#include <chrono>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace AgcDriver { class VulkanDevice; }

namespace AgcDriver::Graphics {

class Recorder;

inline constexpr std::size_t RuntimeFormatTableBytes = ShaderRecompiler::RuntimeAbi::FormatTableWords * sizeof(std::uint32_t);
std::unique_ptr<Buffer> CreateRuntimeFormatTable(const Context& context);

void FlushCachedTextures(VkDevice device);
void ClearCachedTextures(VkDevice device);
std::uint64_t TextureCacheBudget(const VkPhysicalDeviceMemoryProperties& memory);
std::uint64_t StorageTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported);
std::uint64_t SampledTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t textureBytes);
bool SampledBudgetReportDue(std::uint64_t reported, std::uint64_t budget, std::chrono::steady_clock::duration sinceReport);
struct TextureCacheUse {
    std::size_t sampledEntries = 0;
    std::uint64_t sampledBytes = 0;
    std::uint64_t storageBytes = 0;
};
TextureCacheUse TextureCacheUsage();
std::uint64_t SampledTextureCacheBudget(const Context& context);
std::uint64_t StorageTextureCacheBudget(const Context& context);
std::shared_ptr<Texture> CachedSampledTexture(const Context& context, std::span<const std::uint32_t> words);
bool SampledTexturesShareEntry(std::span<const std::uint32_t> first, std::span<const std::uint32_t> second);
void CompletePendingCpuWrites(const Context& context, std::uint64_t address, std::size_t bytes);

// The cached storage image of a surface (render targets use it as their resident image); brought up
// to date with guest memory before it is returned.
std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource);
// Whether `image` is still the storage cache's image of its surface (what CachedStorageSurface
// would return); a true answer counts as a use for the cache's eviction order, as a lookup would.
bool StorageImageCached(const Context& context, const StorageTexture* image);
bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress);

bool PendingStorageOverlaps(std::uint64_t address, std::size_t bytes, const StorageTexture* except);

struct PipelineSetLayouts {
    VkDescriptorSetLayout main = VK_NULL_HANDLE;
    VkDescriptorSetLayout fragment = VK_NULL_HANDLE;
    bool split = false;
    std::span<const std::uint32_t> mainKey;
    std::span<const std::uint32_t> fragmentKey;
};

class ProgramResourceCache;

struct ProgramResourceCounts {
    std::uint64_t programs = 0;
    std::uint64_t uses = 0;
    std::uint64_t dispatchSets = 0;
    std::uint64_t dispatchData = 0;
};
ProgramResourceCounts ProgramResourceCounters();

class DescriptorCache {
public:
    explicit DescriptorCache(const Context& context);
    ~DescriptorCache();
    DescriptorCache(const DescriptorCache&) = delete;
    DescriptorCache& operator=(const DescriptorCache&) = delete;

    // The layout for `key` (binding, type, count, stage flags per binding, as ShaderResources builds
    // it from `bindings`), created on first use.
    VkDescriptorSetLayout Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind = false, std::span<const VkDescriptorBindingFlags> flags = {});
    struct SetAllocation {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::size_t poolIndex = 0;
        bool updateAfterBind = false;
    };
    SetAllocation Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind = false, std::optional<std::uint32_t> variableCount = std::nullopt);
    void Free(const SetAllocation& allocation) noexcept;
    // APS5_PROFILE_DRAW counters: layouts served from the map / created, sets allocated, pools opened.
    struct Stats {
        std::uint64_t layoutHits = 0;
        std::uint64_t layoutMisses = 0;
        std::uint64_t sets = 0;
        std::uint64_t pools = 0;
        std::uint64_t dedicatedPools = 0;
    };
    Stats Counters() const;
    ProgramResourceCache& Programs() { return *programs; }

private:
    std::unique_ptr<ProgramResourceCache> programs;
    Context context;
    PFN_vkDestroyDescriptorSetLayout destroyLayout;
    PFN_vkDestroyDescriptorPool destroyPool;
    PFN_vkResetDescriptorPool resetPool;
    mutable std::mutex mutex;
    std::map<std::vector<std::uint32_t>, VkDescriptorSetLayout> layouts;
    struct Pool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        std::size_t liveSets = 0;
        bool exhausted = false;
    };
    std::vector<Pool> pools;
    std::vector<Pool> updateAfterBindPools;
    std::vector<VkDescriptorPool> dedicated;
    Stats stats;
};

struct SampledTextureCounts {
    std::uint64_t gathers;
    std::uint64_t clears;
};
SampledTextureCounts SampledTextureSources();

struct DeviceTableView {
    ShaderRecompiler::DescriptorImageShape shape = ShaderRecompiler::DescriptorImageShape::Image2D;
    bool depthCompare = false;
    bool identitySwizzle = false;
    bool decodedCopy = false;
    bool integerView = false;
    bool unnormalized = false;
    bool exactAtBase = false;
    bool atomic = false;
    bool atomic64 = false;
    bool mipRun = false;
    bool srgb = false;
    bool written = false;
};

struct DeviceTableStorage {
    std::shared_ptr<StorageTexture> image;
    std::uint32_t slot = 0;
    std::uint64_t address = 0;
    std::uint64_t guestBytes = 0;
};

DeviceTableStorage LookupDeviceTableStorage(const Context& context, std::span<const std::uint32_t, 8> descriptor, const DeviceTableView& view);

struct DeviceTableTexture {
    std::shared_ptr<Texture> texture;
    std::uint32_t slot = 0;
    std::uint64_t address = 0;
    std::uint64_t guestBytes = 0;
    std::uint64_t generation = 0;
    bool snapshot = false;
    bool perUse = false;
};

DeviceTableTexture LookupDeviceTableTexture(const Context& context, std::span<const std::uint32_t, 8> descriptor, const DeviceTableView& view, std::uint32_t samplerClass, bool perUse);
std::uint32_t DeviceTableSamplerClass(const Sampler* sampler, std::span<const std::uint32_t, 4> words, bool unnormalized, bool integerBorder);

class ShaderResources {
public:
    ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes);
    ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes);
    ShaderResources(const Context& context, const CompiledShader& compute);
    // Two-stage build for dispatches (see build): with `deferred` the constructor runs stage A only,
    // which needs no device lock, and Complete() runs stage B under GuestMemory::GpuMutex; until
    // then no other member may be used. `compute` and `snapshots` must outlive Complete(). An
    // address-based shader (BDA tables) builds entirely in Complete(): its lease acquisition
    // reconciles imports and refreshes mirrors, which needs the lock.
    ShaderResources(const Context& context, const CompiledShader& compute, bool deferred, std::uint64_t dispatchThreads = 0, std::uint64_t linearThreads = 0);
    void Complete();
    bool Completed() const { return completed; }
    ~ShaderResources();
    ShaderResources(const ShaderResources&) = delete;
    ShaderResources& operator=(const ShaderResources&) = delete;
    VkDescriptorSetLayout Layout() const;
    std::vector<VkDescriptorSetLayoutBinding> LayoutBindings() const;
    bool LayoutUpdateAfterBind() const;
    std::span<const VkDescriptorBindingFlags> LayoutBindingFlags() const;
    VkDescriptorSetLayout FragmentLayout() const;
    bool SplitSets() const;
    PipelineSetLayouts SetLayouts() const;
    void Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const;
    void WriteBack();
    // Deferred completion: MarkGpuWrites registers the results the recorded work leaves on the GPU
    // (storage images stay there; buffer ranges are noted so CPU reads wait); WriteBackBuffers runs
    // once the work completed.
    void MarkGpuWrites(Recorder& recorder);
    void WriteBackBuffers();
    bool NeedsCompletion() const { return bda != nullptr && faultChecks; }
    std::uint64_t ImageStoreRecord() const { return imageStoresRecorded && bda != nullptr ? bda->StoreRecord() : 0; }
    bool HoldsLease() const { return guestMemory.HoldsLease(); }
    bool UsesAddressTable() const { return usesBda; }
    bool MemoryReadsUnknown() const { return usesBda && accessesUnbounded; }
    bool MemoryWritesUnknown() const { return pageTableStores && storesUnbounded; }
    bool MemoryAccessesUnknown() const { return MemoryReadsUnknown() || MemoryWritesUnknown(); }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> MemoryReads() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> MemoryWrites() const;
    struct WritableSet {
        std::vector<ShaderRecompiler::BdaAbi::Grant> ranges;
        bool writableImports = false;
    };
    WritableSet Writable() const { return {guestMemory.WriteGrants(), pageTableStores}; }
    bool UsesDeviceImages() const { return deviceImages; }
    void ExpectDeviceImages(DeviceImageTable& table, ShaderRecompiler::ShaderStage stage, const ShaderRecompiler::RecompileResult& program) const;
    bool WritesOverlap(std::uint64_t address, std::size_t bytes) const;
    // Whether a region the recorded work reads in place through a host import overlaps the range.
    bool ReadsOverlap(std::uint64_t address, std::size_t bytes) const;
    // The recorder's hazard tracker inputs (Recorder::NoteAccess): the written elements' guest
    // ranges, the regions read in place, and the storage images with whether each is written.
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& GpuWrites() const { return guestMemory.Writes(); }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> InPlaceReads() const { return guestMemory.InPlaceReads(); }
    std::vector<std::pair<VkImage, bool>> StorageImages() const;
    bool ReadsImage(const StorageTexture* image) const;
    bool HasCachedStorageImages() const;
    const std::vector<std::uint32_t>& LayoutKey() const;
    const std::vector<std::uint32_t>& FragmentLayoutKey() const;
    // Debug aid: each bound guest resource with the fraction of sampled bytes that are nonzero.
    std::string Describe() const;

    const std::vector<std::shared_ptr<Texture>>& SampledTextures() const { return textures; }
    bool ChecksFaults() const { return bda != nullptr; }
    void CheckFaults() const;
    void UseBatchFaultRecord(Recorder& recorder);
    void UseOwnFaultRecord();
    void PrepareDeviceRecords(Recorder* recorder, VkCommandBuffer commands);
    bool NeedsDeviceTransfers(const Recorder& recorder) const;
    void FinishDeviceRecords(VkCommandBuffer commands) const;
    static void RecordBatchFaults(Recorder& recorder, const std::shared_ptr<ShaderResources>& resources);
    void NameProgram(std::uint64_t address) { program = address; }
    void PatchDispatchThreadLimit(const ShaderRecompiler::RecompileResult& program, VkDeviceAddress address, std::span<std::byte, PipelinePushConstantBytes> push);
    void PatchPushConstants(std::span<std::byte, PipelinePushConstantBytes> bytes) const;
    // APS5_PROFILE_DRAW: the build's sub-phases ([resources] totals) and their times in
    // milliseconds (zero when not profiling), so the draw path can total them separately from
    // dispatches. bindingsMs covers the binding plan (stage A) and the image lookups (stage B);
    // prepareMs/completeMs are the two stages' totals.
    enum class BuildPhase : std::size_t { Bindings, Upload, Descriptors, StageA, Images, Bda, StageB, Count };
    struct BuildTiming {
        double bindingsMs = 0;
        double uploadMs = 0;
        double descriptorsMs = 0;
        double prepareMs = 0;
        double completeMs = 0;
    };
    const BuildTiming& Timing() const { return timing; }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> PresyncSurfaces() const;

private:
    friend class AgcDriver::VulkanDevice;
    void checkBdaFault() const;
    std::uint64_t program = 0;
    struct DescribedRange {
        const char* kind;
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t format = 0;
        int tileMode = 0;
        std::uint64_t dccAddress = 0;
    };
    std::vector<DescribedRange> describedRanges;
    struct Allocation {
        std::uint64_t address;
        std::size_t size;
        bool guest;
        std::unique_ptr<Buffer> buffer;
        ShaderRecompiler::DescriptorRole role = ShaderRecompiler::DescriptorRole::ShaderData;
        // Guest buffers: whether the shader may store to the range (see addGuestBuffer).
        bool written = true;
        // Data buffers: the words the buffer holds once the recorded work ran (see RefreshData).
        std::vector<std::uint32_t> dataWords;
        std::uint32_t adjustment = 0;
        std::int32_t pushByte = -1;
        std::int64_t dataAllocation = -1;
        std::uint32_t dataByte = 0;
        bool deviceAddressed = false;
        std::int32_t addressPushByte = -1;
        std::uint32_t addressDataByte = 0;
    };
    struct DataPatch {
        std::size_t allocation;
        std::uint32_t byte;
        std::uint32_t adjustment;
    };

    struct Binding {
        VkDescriptorSetLayoutBinding layout;
        std::vector<std::size_t> allocations;
        VkDescriptorBindingFlags flags = 0;
        std::int32_t set = -1;
        bool noDescriptor = false;
    };

    struct TableStage {
        ShaderRecompiler::ShaderStage stage = ShaderRecompiler::ShaderStage::Compute;
        bool images = false;
        std::vector<std::vector<std::uint32_t>> samplerElements;
        std::size_t mapAllocation = std::numeric_limits<std::size_t>::max();
        std::vector<std::uint32_t> map;
        std::size_t firstSampler = 0;
        std::size_t samplerCount = 0;
    };
    struct SetObjects {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        bool ownsLayout = false;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        DescriptorCache::SetAllocation cacheAllocation;
    };
    struct SamplerSource {
        std::array<std::uint32_t, 4> words{};
        bool compareEnable = false;
        bool unnormalized = false;
        bool integerBorder = false;
    };
    struct SamplerSlotPatch {
        std::size_t sampler;
        std::int32_t pushByte = -1;
        std::int64_t dataAllocation = -1;
        std::uint32_t dataByte = 0;
    };
    struct DispatchWords {
        std::size_t allocation;
        std::size_t shader;
        std::size_t binding;
    };
    struct Program;
    friend class ProgramResourceCache;
    static std::shared_ptr<const Program> BuildProgram(const Context& context, std::span<const CompiledShader> shaders, std::uint32_t colorAttachments);
    static std::shared_ptr<const Program> ProgramFor(const Context& context, std::span<const CompiledShader> shaders, std::uint32_t colorAttachments);
    std::shared_ptr<const Program> shared;
    static SetObjects AllocateSet(const Context& context, VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind);
    static bool DispatchAllocation(const Allocation& allocation);

    void build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes);
    void buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes);
    void buildComplete();
    // APS5_PROFILE_DRAW: closes the current sub-phase of the build into the [resources] totals.
    double phase(BuildPhase which);
    void forgetDeferredInputs();
    void release() noexcept;
    void prepareAddressBindings(std::span<const CompiledShader> shaders);
    void selectMemoryImages(std::span<const CompiledShader> shaders);
    VkDescriptorBufferInfo descriptor(std::size_t index);
    void patchShaderData(std::int32_t pushByte, std::int64_t dataAllocation, std::uint32_t dataByte, std::span<const std::uint32_t> words);
    void reportDescriptorCaches() const;
    Buffer* dataBuffer(std::size_t index) const;
    VkDescriptorSet mainSet() const;
    VkDescriptorSet fragmentSet() const;
    Context context;
    GuestBufferMemory guestMemory;
    std::unique_ptr<BdaResources> bda;
    bool usesBda = false;
    bool pageTableStores = false;
    bool storesUnbounded = false;
    bool computeBuild = false;
    bool accessesUnbounded = false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> readRanges;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> accessRanges;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> storeRanges;
    std::optional<std::pair<std::uint64_t, std::uint64_t>> linearStoreRange() const;
    bool usesFaultBuffer = false;
    bool faultChecks = false;
    bool imageStoresRecorded = false;
    bool deviceImages = false;
    VkDescriptorSet _set = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // The cache pool the set was allocated from, freed back to it on release.
    DescriptorCache::SetAllocation cacheAllocation;
    SetObjects fragment;
    std::vector<std::unique_ptr<Buffer>> dispatchData;
    std::vector<VkDescriptorBufferInfo> descriptorBuffers;
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    friend class BatchFaults;
    std::shared_ptr<BatchFaultRecord> batchFaultRecord();
    void rebindFaultRecord(std::shared_ptr<BatchFaultRecord> record);
    struct FaultDescriptor {
        VkDescriptorSet set;
        std::uint32_t binding;
        std::uint32_t element;
    };
    std::vector<FaultDescriptor> faultDescriptors;
    std::uint32_t writtenChunks = 0;
    bool drawBuild = false;
    std::uint64_t dispatchThreads = 0;
    std::uint64_t linearThreads = 0;
    std::optional<ShaderRecompiler::LinearStore> linearStore;
    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> stageShaders;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pushPatches;
    BuildTiming timing;
    // The compute constructor's shader and captured regions: the caller's objects, valid only until
    // the build (Complete() for a deferred one) is done, and reset then (forgetDeferredInputs).
    CompiledShader deferredCompute{};
    // The deferred build is address-based: everything runs in Complete().
    bool lockedBuild = false;
    // Stage A runs without the device lock (a deferred, non-address-based build).
    bool unlockedPrepare = false;
    bool completed = false;
    std::chrono::steady_clock::time_point phaseStart;
};

class BatchFaults {
public:
    void Add(std::shared_ptr<ShaderResources> user);
    void Complete();

private:
    std::vector<std::shared_ptr<ShaderResources>> users;
};

}

#endif
