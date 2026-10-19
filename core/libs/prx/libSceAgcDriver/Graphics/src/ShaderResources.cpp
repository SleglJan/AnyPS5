#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "ThreadOwned.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyMirrors.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DescriptorHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libc/include/General.hpp"
#include "Optimization/include/Optimization/ShaderStageInputInfo.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include "DeviceImageTableAbi.hpp"
#include "ImageTableAbi.hpp"
#include "Optimization/DescriptorBindingBuilder.hpp"
#include "StorageImageMemory.hpp"
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "prx/libc/include/GuestAllocations.hpp"

namespace AgcDriver::Graphics {
namespace {


bool overlap(std::uint64_t first, std::size_t firstSize, std::uint64_t second, std::size_t secondSize) {
    return first < second + secondSize && second < first + firstSize;
}

VkComponentSwizzle ComponentSwizzleFor(std::uint8_t dstSel) {
    switch (dstSel) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return VK_COMPONENT_SWIZZLE_R;
        case 5: return VK_COMPONENT_SWIZZLE_G;
        case 6: return VK_COMPONENT_SWIZZLE_B;
        case 7: return VK_COMPONENT_SWIZZLE_A;
        default: throw std::runtime_error("AGC graphics: guest texture descriptor has an invalid destination channel selector " + std::to_string(dstSel));
    }
}

VkComponentMapping ViewComponents(const GuestTextureResource& resource) {
    if (IsConvertedTextureFormat(resource.format)) return {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    const auto channel = [&](std::uint8_t dstSel) { return TextureComponentChannel(resource.format, ComponentSwizzleFor(dstSel)); };
    return {channel(resource.dstSelX), channel(resource.dstSelY), channel(resource.dstSelZ), channel(resource.dstSelW)};
}

VkComponentMapping DecodedCopyComponents(const GuestTextureResource& resource) {
    const auto channel = [&](std::uint8_t dstSel) { return DecodedCopyChannel(resource.format, ComponentSwizzleFor(dstSel)); };
    return {channel(resource.dstSelX), channel(resource.dstSelY), channel(resource.dstSelZ), channel(resource.dstSelW)};
}

VkComponentMapping SampledViewComponents(const GuestTextureResource& resource, bool decodedCopy) {
    return decodedCopy ? DecodedCopyComponents(resource) : ViewComponents(resource);
}

// Sampled textures are reused across draws and dispatches while their guest bytes are unchanged; a
// byte copy of the guest surface validates each reuse, so CPU or GPU writes to it force a re-upload.
// The key is everything a lookup matches: the device, the eight descriptor words and the swizzle.
struct TextureKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    std::array<std::uint32_t, 4> components;
    bool depthCompare = false;
    bool decodedCopy = false;
    bool operator==(const TextureKey&) const = default;
};

std::uint64_t hashWords(std::uint64_t hash, const std::uint32_t* words, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) hash = (hash ^ words[i]) * 1099511628211ull;
    return hash;
}

struct TextureKeyHash {
    std::size_t operator()(const TextureKey& key) const noexcept {
        auto hash = 14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device);
        hash = hashWords(hash, key.words.data(), key.words.size()) ^ static_cast<std::uint64_t>(key.depthCompare) ^ (static_cast<std::uint64_t>(key.decodedCopy) << 1u);
        return static_cast<std::size_t>(hashWords(hash, key.components.data(), key.components.size()));
    }
};

TextureKey MakeTextureKey(VkDevice device, std::span<const std::uint32_t> words, VkComponentMapping components, bool depthCompare = false, bool decodedCopy = false) {
    TextureKey key{device, {}, {static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a)}};
    key.depthCompare = depthCompare;
    key.decodedCopy = decodedCopy;
    std::copy(words.begin(), words.end(), key.words.begin());
    key.words[5] &= ~((0xfffu << 8u) | (1u << 25u));
    key.words[6] &= ~0xffu;
    return key;
}

struct CachedTexture {
    TextureKey key;
    std::uint64_t address;
    std::uint64_t snapshotBytes = 0;
    std::shared_ptr<Texture> texture;
    // A fast-cleared surface is cached as its clear texels; it stays valid while the keys are unchanged.
    DccKeys keys = DccKeys::Uncompressed;
    std::uint64_t generation = 0;
    // A texture viewing a storage image on the GPU has no snapshot; it stays valid while that image
    // is still its surface's cached image and nothing wrote the guest memory since it matched.
    std::shared_ptr<StorageTexture> source;
    std::uint64_t sourceVersion = 0;
    std::uint64_t accounted = 0;
    std::uint64_t present = 0;
    bool gathered = false;
    std::uint64_t mirrorSerial = 0;
};

// Entries in use order (front = most recent) with a hash index by key: a lookup is O(1) and the
// eviction takes the back. APS5_NO_TEXTURE_HASH=1 finds entries by scanning the list (the index is
// still kept), the linear lookup of before.
struct TextureCache {
    std::mutex mutex;
    std::list<CachedTexture> entries;
    std::unordered_map<TextureKey, std::list<CachedTexture>::iterator, TextureKeyHash> index;
    std::unordered_multimap<const StorageTexture*, std::list<CachedTexture>::iterator> views;
    std::uint64_t bytes = 0;
    std::size_t passed = 0;
    std::chrono::steady_clock::time_point passedSince{};
    std::uint64_t firstUsesPresent = 0;
    std::unordered_set<TextureKey, TextureKeyHash> firstUses;
    struct Counts {
        std::uint64_t inserted = 0;
        std::uint64_t uncached = 0;
        std::uint64_t evicted = 0;
        std::uint64_t passed = 0;
        std::uint64_t stale = 0;
    } counts;
    std::chrono::steady_clock::time_point countsReported = std::chrono::steady_clock::now();
    VkDevice budgetDevice = VK_NULL_HANDLE;
    VkPhysicalDevice budgetPhysical = VK_NULL_HANDLE;
    std::chrono::steady_clock::time_point budgetRead{};
    std::uint64_t budget = 0;
    std::uint64_t reportedBudget = 0;
    std::chrono::steady_clock::time_point reportedAt{};
    std::uint64_t erasures = 0;
};

TextureCache& Textures() {
    static TextureCache cache;
    return cache;
}

struct UncachedImages {
    std::mutex mutex;
    std::vector<std::weak_ptr<StorageTexture>> images;
    std::atomic<bool> any{false};
};

UncachedImages& Uncached() {
    static UncachedImages uncached;
    return uncached;
}

void noteUncached(const std::shared_ptr<StorageTexture>& image) {
    auto& uncached = Uncached();
    std::lock_guard lock(uncached.mutex);
    uncached.images.push_back(image);
    uncached.any.store(true, std::memory_order_release);
}

bool SamplerCacheDisabled() {
    static const bool disabled = std::getenv("APS5_NO_SAMPLER_CACHE") != nullptr;
    return disabled;
}

bool TextureHashEnabled() {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_HASH") != nullptr;
    return !disabled;
}

std::list<CachedTexture>::iterator findTexture(TextureCache& cache, const TextureKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

void eraseTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    cache.bytes -= it->accounted;
    if (it->source != nullptr) {
        for (auto [view, last] = cache.views.equal_range(it->source.get()); view != last; ++view) {
            if (view->second == it) {
                cache.views.erase(view);
                break;
            }
        }
    }
    cache.index.erase(it->key);
    cache.entries.erase(it);
    ++cache.erasures;
}

void dropUncachedViews(TextureCache& cache) {
    auto& uncached = Uncached();
    if (!uncached.any.load(std::memory_order_acquire)) return;
    std::vector<std::weak_ptr<StorageTexture>> images;
    {
        std::lock_guard lock(uncached.mutex);
        images.swap(uncached.images);
        uncached.any.store(false, std::memory_order_release);
    }
    for (const auto& weak : images) {
        const auto image = weak.lock();
        if (image == nullptr) continue;
        for (auto view = cache.views.find(image.get()); view != cache.views.end(); view = cache.views.find(image.get())) eraseTexture(cache, view->second);
    }
}

// Moves an entry to the front (most recently used).
void touchTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    it->present = Recorder::Presents();
}

bool admitTexture(TextureCache& cache, const TextureKey& key, std::uint64_t incoming, std::uint64_t budget) {
    if (cache.bytes + incoming <= budget) return true;
    if (const auto now = std::chrono::steady_clock::now(); now - cache.passedSince >= std::chrono::seconds(1)) {
        cache.passed = 0;
        cache.passedSince = now;
    }
    const auto present = Recorder::Presents();
    if (cache.firstUsesPresent != present) {
        cache.firstUses.clear();
        cache.firstUsesPresent = present;
    }
    const bool firstUse = incoming != 0 && present != 0 && !cache.firstUses.contains(key);
    const auto ceiling = budget + budget / 4u;
    while (!cache.entries.empty() && cache.bytes + incoming > budget) {
        const auto back = std::prev(cache.entries.end());
        if (firstUse && back->present + 1u >= present) {
            cache.firstUses.insert(key);
            ++cache.counts.uncached;
            return false;
        }
        const bool over = cache.bytes + incoming > ceiling;
        const bool keep = back->accounted == 0 || (!over && back->texture.use_count() > 1);
        if (keep && cache.passed < cache.entries.size()) {
            cache.entries.splice(cache.entries.begin(), cache.entries, back);
            ++cache.passed;
            ++cache.counts.passed;
        } else if (keep && !over) {
            break;
        } else {
            eraseTexture(cache, back);
            ++cache.counts.evicted;
        }
    }
    return true;
}

void reportCacheCounts(TextureCache& cache, std::uint64_t budget) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - cache.countsReported < std::chrono::seconds(10)) return;
    cache.countsReported = now;
    const auto counts = std::exchange(cache.counts, {});
    AgcDriver::ProfilePrint_nid_no_patch("[texture-cache] (10 s): %llu textures cached, %llu used uncached (first use this frame, the least recently used entry used this frame or the last), %llu evicted, %llu held ones passed over, %llu found changed; %llu MiB in %zu entries, budget %llu MiB, frame %llu\n", static_cast<unsigned long long>(counts.inserted), static_cast<unsigned long long>(counts.uncached), static_cast<unsigned long long>(counts.evicted), static_cast<unsigned long long>(counts.passed), static_cast<unsigned long long>(counts.stale), static_cast<unsigned long long>(cache.bytes >> 20u), cache.entries.size(), static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(Recorder::Presents()));
}

struct ImageMemoEntry {
    VkDevice device = VK_NULL_HANDLE;
    std::array<std::uint32_t, 8> words{};
    GuestTextureResource resource{};
    std::uint64_t guestBytes = 0;
    VkComponentMapping components{};
    bool hasEntry = false;
    std::list<CachedTexture>::iterator entry{};
    std::uint64_t erasures = 0;
};

constexpr std::size_t ImageMemoSlots = 1024;
struct ImageMemoTag {};

ImageMemoEntry& imageMemoSlot(std::span<const std::uint32_t> words) {
    auto& imageMemo = HostThreadLocal<std::unique_ptr<std::array<ImageMemoEntry, ImageMemoSlots>>, ImageMemoTag>();
    if (imageMemo == nullptr) imageMemo = std::make_unique<std::array<ImageMemoEntry, ImageMemoSlots>>();
    const auto hash = hashWords(14695981039346656037ull, words.data(), words.size());
    return (*imageMemo)[(hash ^ (hash >> 29u)) % ImageMemoSlots];
}

const ImageMemoEntry& decodedImage(VkDevice device, std::span<const std::uint32_t> words) {
    auto& slot = imageMemoSlot(words);
    if (slot.device == device && std::equal(words.begin(), words.end(), slot.words.begin())) return slot;
    slot.device = VK_NULL_HANDLE;
    slot.resource = DecodeTextureResource(words);
    slot.guestBytes = DescribeSurface(slot.resource).guestBytes;
    slot.components = ViewComponents(slot.resource);
    std::copy(words.begin(), words.end(), slot.words.begin());
    slot.hasEntry = false;
    slot.device = device;
    return slot;
}

std::list<CachedTexture>::iterator findSampledEntry(TextureCache& cache, VkDevice device, const std::array<std::uint32_t, 8>& words, VkComponentMapping components) {
    auto& slot = imageMemoSlot(words);
    const bool same = slot.device == device && slot.words == words && slot.components.r == components.r && slot.components.g == components.g && slot.components.b == components.b && slot.components.a == components.a;
    if (same && slot.hasEntry && slot.erasures == cache.erasures) return slot.entry;
    const auto it = findTexture(cache, MakeTextureKey(device, words, components));
    if (same) {
        slot.hasEntry = it != cache.entries.end();
        slot.entry = it;
        slot.erasures = cache.erasures;
    }
    return it;
}

struct TextureCounters {
    std::atomic<std::uint64_t> fromStorage{0};
    std::atomic<std::uint64_t> gathers{0};
    std::atomic<std::uint64_t> clears{0};
    std::atomic<std::uint64_t> storageFallbacks{0};
    std::atomic<std::uint64_t> records{0};
    std::atomic<std::uint64_t> fastHits{0};
    std::atomic<std::uint64_t> fastMisses{0};
    std::atomic<std::uint64_t> storageHits{0};
    std::atomic<std::uint64_t> storageCreated{0};
    std::atomic<std::int64_t> lastReport{0};
};

TextureCounters& TextureCounts() {
    static TextureCounters counters;
    return counters;
}

std::vector<TextureGatherCopy> gatherSurface(const Context& context, std::uint64_t address, std::uint64_t bytes) {
    const MirrorSpan span(ResidencyDriver::Active(), address, address + bytes, ResidencyAccess::Read);
    std::vector<TextureGatherCopy> copies;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> importReads;
    for (auto& piece : GuestDevicePieces(context, span, address, address + bytes, "sampled texture")) {
        if (piece.imported) importReads.emplace_back(piece.begin, piece.end);
        copies.push_back({piece.buffer, piece.offset, piece.begin - address, piece.end - piece.begin, std::move(piece.mirror)});
    }
    if (auto* recorder = Recorder::Active(); recorder != nullptr && !importReads.empty()) recorder->NotePendingReads(importReads, Recorder::ReadKind::GpuCopy);
    return copies;
}

void reportTextureCounters() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& counters = TextureCounts();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto count = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
    AgcDriver::ProfilePrint_nid_no_patch("[textures] sampled created: %llu from storage images, %llu gathered, %llu cleared, %llu storage-path fallbacks; stage-A records %llu: %llu fast hits, %llu full lookups; storage images %llu hits, %llu created\n", count(counters.fromStorage), count(counters.gathers), count(counters.clears), count(counters.storageFallbacks), count(counters.records), count(counters.fastHits), count(counters.fastMisses), count(counters.storageHits), count(counters.storageCreated));
}

struct LookupRecord {
    const void* object;
    GuestTextureResource resource;
    std::uint64_t bytes;
    DccKeys keys;
    std::uint64_t generation;
    const StorageTexture* source;
    bool depth = false;
};

thread_local std::vector<LookupRecord>* lookupLogSlot = nullptr;

std::vector<LookupRecord>& lookupLog() {
    return ShaderRecompiler::ThreadOwned(lookupLogSlot);
}

void logLookup(const LookupRecord& record) {
    // The bound drops the oldest half, never everything: one build's lookups (a few dozen at most)
    // are the newest records, and a capture must find them even when the bound trips mid-build.
    constexpr std::size_t bound = 1024;
    auto& log = lookupLog();
    if (log.size() >= bound) log.erase(log.begin(), log.begin() + bound / 2);
    log.push_back(record);
}

std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip, std::uint64_t guestBytes = 0, bool written = false);

std::uint64_t heldBytes(const StorageTexture& image) {
    return std::max<std::uint64_t>(image.AllocationBytes(), image.GuestBytes());
}

bool MetadataMoved(const StorageTexture& image, const GuestTextureResource& resource) {
    return resource.dccAddress != 0 && image.Descriptor().dccAddress != resource.dccAddress && !image.ServesKeysAt(resource.dccAddress);
}

// Whether a sampled texture over `resource` can be a view of the surface's cached storage image
// instead of a CPU snapshot (see cachedTexture): the format has a storage form and is not block
// compressed, and the surface lives in host-imported memory, where the image uploads and refreshes
// GPU-direct (a snapshot of such a surface reads and compares its bytes on the CPU, waiting for
// the recorded work that wrote them). APS5_NO_SAMPLED_FROM_STORAGE=1 keeps every sampled texture a
// snapshot as before.
bool SampledFromStorageEligible(const Context& context, std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes, bool importOnDemand = false) {
    static const bool disabled = std::getenv("APS5_NO_SAMPLED_FROM_STORAGE") != nullptr;
    if (disabled || IsBlockCompressed(format) || !StorageFormatAvailable(context, format)) return false;
    if (importOnDemand) return HostImportFor(context, address, static_cast<std::size_t>(guestBytes)) != nullptr;
    return HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
}

bool SampledFromStorageEligible(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes, bool importOnDemand = false) {
    return SampledFromStorageEligible(context, resource.format, resource.baseAddress, guestBytes, importOnDemand);
}

std::uint64_t DepthCopyMinBytes() {
    static const std::uint64_t bytes = [] {
        const char* text = std::getenv("APS5_DEPTH_COPY_MIN_KIB");
        return (text != nullptr ? std::strtoull(text, nullptr, 10) : 1024ull) << 10u;
    }();
    return bytes;
}

// Surfaces whose storage image could not be made (no writable committed pages, say): remembered
// so the sampled lookup does not throw and fall back on every use. A failure is keyed by the surface
// (address and size: the heap reuses addresses) and holds only while the guest mappings are what
// they were when it failed (the allocation generation): pages committed later, or another surface
// at the address, get a fresh attempt.
struct StorageFailures {
    std::mutex mutex;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> generations;
};

StorageFailures& StorageFailed() {
    static StorageFailures failures;
    return failures;
}

// The cached storage image for a sampled descriptor, or null when it cannot be made (then the
// snapshot path serves the descriptor, as before). Every null return counts as a fallback.
std::shared_ptr<StorageTexture> sampledStorageSource(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    auto& counters = TextureCounts();
    auto& failures = StorageFailed();
    const auto surface = std::make_pair(resource.baseAddress, guestBytes);
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    {
        std::lock_guard lock(failures.mutex);
        if (const auto it = failures.generations.find(surface); it != failures.generations.end()) {
            if (it->second == generation) {
                counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
            failures.generations.erase(it);
        }
    }
    try {
        return cachedStorageTexture(context, {}, resource, 0, guestBytes);
    } catch (const std::exception& error) {
        counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(failures.mutex);
        if (failures.generations.size() < 4096 && failures.generations.emplace(surface, generation).second) std::fprintf(stderr, "[textures] sampled texture 0x%llx (%ux%u format %u) keeps the snapshot path: %s\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, error.what());
        return nullptr;
    }
}

std::uint64_t storageCacheBytes();

std::optional<std::uint32_t> largestDeviceLocalHeap(const VkPhysicalDeviceMemoryProperties& memory) {
    std::optional<std::uint32_t> heap;
    for (std::uint32_t i = 0; i < std::min<std::uint32_t>(memory.memoryHeapCount, VK_MAX_MEMORY_HEAPS); ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && (!heap || memory.memoryHeaps[i].size > memory.memoryHeaps[*heap].size)) heap = i;
    }
    return heap;
}

std::uint64_t forcedCacheBudget() {
    static const std::uint64_t forced = [] {
        const char* text = std::getenv("APS5_TEXTURE_CACHE_MIB");
        return text != nullptr ? std::strtoull(text, nullptr, 10) << 20u : 0u;
    }();
    return forced;
}

struct StorageBudgetCache {
    std::mutex mutex;
    VkDevice device{VK_NULL_HANDLE};
    VkPhysicalDevice physical{VK_NULL_HANDLE};
    std::chrono::steady_clock::time_point read{};
    std::uint64_t budget{0};
};

StorageBudgetCache& StorageBudget() {
    static StorageBudgetCache cache;
    return cache;
}

std::uint64_t storageBudget(const Context& context) {
    const auto forced = forcedCacheBudget();
    if (forced != 0) return forced;
    if (context.memoryProperties2 == nullptr || context.physical == VK_NULL_HANDLE) return TextureCacheBudget(context.memory);
    auto& cached = StorageBudget();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(cached.mutex);
    if (cached.device == context.device && cached.physical == context.physical && now - cached.read < std::chrono::seconds(1)) return cached.budget;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    cached.device = context.device;
    cached.physical = context.physical;
    cached.read = now;
    cached.budget = StorageTextureBudget(context.memory, &reported);
    return cached.budget;
}

std::uint64_t sampledBudget(const Context& context, TextureCache& cache) {
    if (const auto forced = forcedCacheBudget(); forced != 0) return forced;
    if (context.memoryProperties2 == nullptr || context.physical == VK_NULL_HANDLE) return TextureCacheBudget(context.memory);
    const auto now = std::chrono::steady_clock::now();
    if (cache.budgetDevice == context.device && cache.budgetPhysical == context.physical && now - cache.budgetRead < std::chrono::seconds(1)) return cache.budget;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    const auto sampled = SampledTextureMemory();
    const auto storage = storageCacheBytes();
    const auto budget = SampledTextureBudget(context.memory, &reported, sampled + storage);
    const bool fresh = cache.budgetDevice != context.device || cache.budgetPhysical != context.physical;
    if (const auto heap = largestDeviceLocalHeap(context.memory); heap && (fresh || SampledBudgetReportDue(cache.reportedBudget, budget, now - cache.reportedAt))) {
        char was[48] = "";
        if (!fresh) std::snprintf(was, sizeof(was), " (was %llu MiB)", static_cast<unsigned long long>(cache.reportedBudget >> 20u));
        std::fprintf(stderr, "[gpu] sampled texture cache budget %llu MiB%s: device-local heap %u has a VK_EXT_memory_budget budget of %llu MiB and uses %llu MiB, %llu MiB of it sampled textures (%llu MiB in the cache) and %llu MiB cached storage images; storage image cache budget %llu MiB\n", static_cast<unsigned long long>(budget >> 20u), was, *heap, static_cast<unsigned long long>(reported.heapBudget[*heap] >> 20u), static_cast<unsigned long long>(reported.heapUsage[*heap] >> 20u), static_cast<unsigned long long>(sampled >> 20u), static_cast<unsigned long long>(cache.bytes >> 20u), static_cast<unsigned long long>(storage >> 20u), static_cast<unsigned long long>(storageBudget(context) >> 20u));
        cache.reportedBudget = budget;
        cache.reportedAt = now;
    }
    cache.budgetDevice = context.device;
    cache.budgetPhysical = context.physical;
    cache.budgetRead = now;
    cache.budget = budget;
    return budget;
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<Texture> cachedTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, std::uint64_t guestBytes = 0, bool depthCompare = false, bool decodedCopy = false) {
    if (resource.dimension == TextureDimension::k2DMsaa) {
        Require(!depthCompare, "native multisampled depth comparison sampling is not implemented");
        if (auto depth = DepthSurfaceTexture(context, words, resource, components)) {
            logLookup({depth.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
            return depth;
        }
        auto target = MultisampleTargetTexture(context, resource, components);
        logLookup({target.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
        return target;
    }
    CaptureTrace::Log("sampled-lookup address=%llx width=%u height=%u dcc=%llx", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, static_cast<unsigned long long>(resource.dccAddress));
    RequireNoMultisampleTarget(resource.baseAddress, guestBytes != 0 ? static_cast<std::size_t>(guestBytes) : 1u, "a texture read");
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    if (auto depth = DepthSurfaceTexture(context, words, resource, components)) {
        logLookup({depth.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
        return depth;
    }
    const auto depthBitsWidth = words.size() >= 4 ? ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) : 0u;
    if (depthBitsWidth == 32u) {
        char text[160];
        std::snprintf(text, sizeof(text), "AGC graphics: 32-bit integer read of the depth-layout texture 0x%llx, which is no depth surface drawn with, is not implemented", static_cast<unsigned long long>(resource.baseAddress));
        throw std::runtime_error(text);
    }
    constexpr auto unorm16 = static_cast<std::uint32_t>(ShaderRecompiler::IrBufferFormat::Format16UNorm);
    if (depthBitsWidth == 16u && resource.format != unorm16) {
        auto normalized = resource;
        normalized.format = unorm16;
        return cachedTexture(context, words, normalized, components, guestBytes, depthCompare, decodedCopy);
    }
    const bool profile = LookupOutcomes::Profiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto scanKeys = [&] {
        const auto scanStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto keys = TextureClearKeys(resource, guestBytes);
        if (profile && resource.dccAddress != 0) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
        return keys;
    };
    if (guestBytes == 0) guestBytes = DescribeSurface(resource).guestBytes;
    auto& counters = TextureCounts();
    const auto address = resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(guestBytes);
    // A storage image whose results for this surface (or for a mip chain containing it) are still on
    // the GPU supplies the texture by a view of it; anything else needs those results in guest
    // memory first.
    const bool depthCopy = depthCompare && guestBytes >= DepthCopyMinBytes() && Texture::DepthCopyFormat(ResolveTextureFormat(resource.format)) != VK_FORMAT_UNDEFINED && resource.dimension != TextureDimension::k3D;
    const auto servesCopy = [&](const StorageTexture& candidate) {
        return depthCompare ? depthCopy && Texture::CanDepthCopyFrom(candidate, resource) : decodedCopy ? Texture::CanDecodedCopyFrom(candidate, resource) : Texture::CanCopyFrom(candidate, resource);
    };
    auto source = StorageTexture::FindPending(address, guestBytes);
    if (source != nullptr && !servesCopy(*source)) source.reset();
    if (source != nullptr && MetadataMoved(*source, resource)) source.reset();
    std::optional<DccKeys> keys;
    bool clearThroughKeys = false;
    if (source != nullptr && resource.dccAddress != 0 && IsDccClear(source->FilledKeys())) {
        keys = scanKeys();
        if (*keys == source->FilledKeys()) {
            std::array<std::byte, 16> probe{};
            if (!FillDccClear(ResolveTextureFormat(resource.format), *keys, resource.dccAlphaOnMsb, probe)) {
                char text[256];
                std::snprintf(text, sizeof(text), "AGC graphics: sampled texture 0x%llx (%ux%u format %u, dcc 0x%llx) reads %s DCC keys filled over its pending image, a clear value the format has no encoding for", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
                throw std::runtime_error(text);
            }
            static const bool traceKeys = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
            if (traceKeys) std::fprintf(stderr, "[dcc-keys] sampled 0x%llx through keys 0x%llx reads the %s fill, not the pending image\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
            source.reset();
            clearThroughKeys = true;
        }
    }
    if (source != nullptr && resource.dccAddress != 0 && (keys.has_value() ? *keys : scanKeys()) == DccKeys::ClearSingle) source.reset();

    // Otherwise a surface in host-imported memory is viewed through its cached storage image (made
    // here when there is none): its refresh after a CPU or GPU write is a GPU-direct detile from the
    // import, recorded behind the producer, so no bytes are read or compared on the CPU and nothing
    // waits for the producer. A fast-cleared surface (keys) is viewed only through an image whose own
    // descriptor carries the DCC address (below); it stays a snapshot otherwise, its texels not read.
    if ((!depthCompare || depthCopy) && source == nullptr && !clearThroughKeys && SampledFromStorageEligible(context, resource, guestBytes, true)) {

        keys = scanKeys();
        if (*keys == DccKeys::Uncompressed) {
            source = sampledStorageSource(context, resource, guestBytes);
            if (source != nullptr && ((depthCompare && !Texture::CanDepthCopyFrom(*source, resource)) || (decodedCopy && !Texture::CanDecodedCopyFrom(*source, resource)))) source.reset();
        } else if (!depthCompare && StorageClearAvailable(context, resource.format, *keys)) {
            // A fast-cleared surface is viewed as well, through its image cleared on the GPU (the
            // lookup's Refresh, see StorageTexture::upload), when the image's own descriptor names
            // the same DCC metadata so the refresh sees the keys. A snapshot of the clear texels
            // (filled and uploaded on the CPU, then replaced by a view once results are pending: two
            // 4K uploads per clear at the movie stage) serves the surface otherwise, as before.
            auto candidate = sampledStorageSource(context, resource, guestBytes);
            if (candidate != nullptr && candidate->Descriptor().dccAddress == resource.dccAddress && (!decodedCopy || Texture::CanDecodedCopyFrom(*candidate, resource))) source = std::move(candidate);
        }
        // The lookup's Refresh may have flushed another image's results over the memory (which
        // marks the keys uncompressed): a surface that stays a snapshot re-reads them behind that
        // flush, as it does behind its own. APS5_NO_KEYS_RESCAN=1 keeps the keys scanned above.
        static const bool rescan = std::getenv("APS5_NO_KEYS_RESCAN") == nullptr;
        if (rescan && source == nullptr) keys.reset();
    }
    // Pages of the surface written since they were last collected are stamped now, before any
    // UnchangedSince (here and on a cache hit) looks at them: the checks only compare stamped blocks,
    // so a write the game made since the last walk is invisible until a collect stamps its page. The
    // walk is memoized per packet, so a texture viewed from a storage image repeats it for free.
    auto generation = GuestMemory::CollectWrites(address, bytes);
    if (source != nullptr && !GuestMemory::UnchangedSince(address, bytes, source->Generation())) {
        // The CPU wrote the memory while results were pending: Refresh merges them the usual way.
        source->Refresh();
    }
    // Results stored to guest memory just now are stamped newer than `generation`; a snapshot read
    // after them is current at the generation of a second (memoized) collect. The store marks the
    // surface's DCC keys uncompressed, so the keys are read after it.
    const auto flushStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto observedKeys = keys.has_value() ? *keys : scanKeys();
    const bool stagedClear = source != nullptr && resource.dccAddress != 0 && (IsDccClear(observedKeys) || observedKeys == DccKeys::ClearSingle);
    bool stagedFlushed = false;
    if ((source == nullptr || stagedClear) && !clearThroughKeys && StorageTexture::FlushPending(address, bytes, nullptr, "sampled texture")) {
        if (profile) LookupOutcomes::Add(LookupOutcomes::PendingFlush, flushStart);
        generation = GuestMemory::CollectWrites(address, bytes);
        if (stagedClear) {
            stagedFlushed = true;
            keys = DccKeys::Uncompressed;
        } else {
            keys.reset();
        }
        if (stagedFlushed) source.reset();
    }
    if (!keys.has_value()) keys = scanKeys();
    const auto storageTexture = [&] {
        if (!depthCompare && !decodedCopy) return std::make_shared<Texture>(context, source, resource, components);
        auto texture = depthCompare ? std::make_shared<Texture>(context, source, resource, components, Texture::DepthCopyTag{}) : std::make_shared<Texture>(context, source, resource, components, Texture::DecodedCopyTag{});
        texture->RefreshCopy();
        return texture;
    };
    const auto surfaceTexture = [&](CachedTexture& made) {
        if (IsDccClear(*keys)) {
            made.texture = std::make_shared<Texture>(context, *context.detiler, resource, components, std::span<const TextureGatherCopy>(), *keys, depthCompare, decodedCopy);
            counters.clears.fetch_add(1, std::memory_order_relaxed);
        } else {
            const auto gather = gatherSurface(context, address, bytes);
            made.gathered = true;
            made.mirrorSerial = ImageMirrorSerial(context, address, bytes);
            made.texture = std::make_shared<Texture>(context, *context.detiler, resource, components, std::span<const TextureGatherCopy>(gather), *keys, depthCompare, decodedCopy);
            counters.gathers.fetch_add(1, std::memory_order_relaxed);
        }
        made.accounted = made.texture->AllocationBytes();
    };
    auto& cache = Textures();
    const auto key = MakeTextureKey(context.device, words, components, depthCompare, decodedCopy);
    std::lock_guard lock(cache.mutex);
    dropUncachedViews(cache);
    if (auto it = findTexture(cache, key); it != cache.entries.end()) {
        if (it->source != nullptr) {
            // The view follows the storage image, whatever the GPU wrote to it since; guest memory
            // written meanwhile is taken in by refreshing the image. A fast clear the image cannot
            // see (keys, and no pending results to prefer) ends the view: a snapshot holds the clear.
            if (!stagedFlushed && (source == nullptr || source == it->source) && (source != nullptr || *keys == DccKeys::Uncompressed) && StorageImageServesKeys(*it->source, resource.dccAddress)) {
                if (!GuestMemory::UnchangedSince(address, bytes, it->source->Generation())) it->source->Refresh();
                it->texture->RefreshCopy();
                it->keys = *keys;
                touchTexture(cache, it);
                logLookup({it->texture.get(), resource, guestBytes, *keys, it->source->Generation(), it->source.get()});
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(*keys != DccKeys::Uncompressed ? LookupOutcomes::SampledHitClearedView : LookupOutcomes::SampledHitView, start);
                return it->texture;
            }
        } else if (source == nullptr && it->gathered && it->snapshotBytes == guestBytes && it->keys == *keys && (GuestMemory::UnchangedSince(address, bytes, it->generation) || (it->mirrorSerial != 0 && ImageMirrorSerial(context, address, bytes) == it->mirrorSerial))) {
            it->generation = generation;
            touchTexture(cache, it);
            logLookup({it->texture.get(), resource, guestBytes, *keys, generation, nullptr});
            reportTextureCounters();
            if (profile) LookupOutcomes::Add(LookupOutcomes::SampledHitSnapshot, start);
            return it->texture;
        } else if (source == nullptr && !it->gathered && it->snapshotBytes == guestBytes && it->keys == *keys && IsDccClear(*keys)) {
            it->generation = generation;
            touchTexture(cache, it);
            logLookup({it->texture.get(), resource, guestBytes, *keys, generation, nullptr});
            reportTextureCounters();
            if (profile) LookupOutcomes::Add(LookupOutcomes::SampledHitSnapshot, start);
            return it->texture;
        }
        eraseTexture(cache, it);
        ++cache.counts.stale;
    }
    CachedTexture entry{key, address, source != nullptr ? 0u : bytes, nullptr, *keys, generation};
    if (source != nullptr) {
        entry.source = source;
        entry.sourceVersion = source->Version();
        entry.texture = storageTexture();
        entry.accounted = (source->Cached() ? 0u : heldBytes(*source)) + (decodedCopy ? entry.texture->AllocationBytes() : 0u);
        counters.fromStorage.fetch_add(1, std::memory_order_relaxed);
    } else {
        surfaceTexture(entry);
    }
    const auto budget = sampledBudget(context, cache);
    auto texture = entry.texture;
    if (!admitTexture(cache, key, entry.accounted, budget)) {
        logLookup({texture.get(), resource, guestBytes, *keys, source != nullptr ? source->Generation() : generation, source.get()});
        reportCacheCounts(cache, budget);
        reportTextureCounters();
        if (profile) LookupOutcomes::Add(source != nullptr ? LookupOutcomes::SampledMadeView : LookupOutcomes::SampledMadeSnapshot, start);
        return texture;
    }
    cache.bytes += entry.accounted;
    entry.present = Recorder::Presents();
    ++cache.counts.inserted;
    logLookup({texture.get(), resource, guestBytes, *keys, source != nullptr ? source->Generation() : generation, source.get()});
    cache.entries.push_front(std::move(entry));
    cache.index[key] = cache.entries.begin();
    if (source != nullptr) cache.views.emplace(source.get(), cache.entries.begin());
    reportCacheCounts(cache, budget);
    reportTextureCounters();
    if (profile) LookupOutcomes::Add(source != nullptr ? LookupOutcomes::SampledMadeView : LookupOutcomes::SampledMadeSnapshot, start);
    return texture;
}

// Storage images stay on the GPU between dispatches: while guest memory still holds what an image was
// last uploaded from or written back as, its next use skips the upload and detile.
struct StorageKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    bool operator==(const StorageKey&) const = default;
};

struct StorageKeyHash {
    std::size_t operator()(const StorageKey& key) const noexcept {
        return static_cast<std::size_t>(hashWords(14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device), key.words.data(), key.words.size()));
    }
};

struct CachedStorageTexture {
    StorageKey key;
    std::uint32_t mip;
    std::shared_ptr<StorageTexture> texture;
    std::uint64_t accounted = 0;
};

// As TextureCache: use order with a hash index by key, plus one by image for StorageImageCached.
struct StorageTextureCache {
    std::mutex mutex;
    std::list<CachedStorageTexture> entries;
    std::unordered_map<StorageKey, std::list<CachedStorageTexture>::iterator, StorageKeyHash> index;
    std::unordered_map<const StorageTexture*, std::list<CachedStorageTexture>::iterator> byImage;
    std::atomic<std::uint64_t> bytes{0};
};

StorageTextureCache& StorageTextures() {
    static StorageTextureCache cache;
    return cache;
}

std::uint64_t storageCacheBytes() {
    return StorageTextures().bytes.load(std::memory_order_relaxed);
}

std::list<CachedStorageTexture>::iterator findStorage(StorageTextureCache& cache, const StorageKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

std::list<CachedStorageTexture>::iterator findStorageByImage(StorageTextureCache& cache, VkDevice device, const StorageTexture* image) {
    if (TextureHashEnabled()) {
        const auto found = cache.byImage.find(image);
        return found == cache.byImage.end() || found->second->key.device != device ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key.device == device && it->texture.get() == image) return it;
    }
    return cache.entries.end();
}

// Evicts an entry: its pending results go to guest memory first (the image may die with the entry).
std::shared_ptr<StorageTexture> evictStorage(StorageTextureCache& cache, std::list<CachedStorageTexture>::iterator it) {
    auto texture = it->texture;
    it->texture->SetCached(false);
    noteUncached(it->texture);
    it->texture->Flush();
    cache.bytes -= it->accounted;
    cache.index.erase(it->key);
    cache.byImage.erase(it->texture.get());
    cache.entries.erase(it);
    return texture;
}

bool ArraySliceViews(const GuestTextureResource& resource) {
    return resource.dimension == TextureDimension::k2DArray || resource.dimension == TextureDimension::k1DArray;
}

// Storage images are shared by every descriptor of one surface (address, extent, layers, format, tile
// mode): the image holds the whole mip chain, and render targets in the same memory attach to it.
std::array<std::uint32_t, 8> SurfaceKey(const Context& context, const GuestTextureResource& resource) {
    const bool slices = ArraySliceViews(resource);
    // Guest formats that store in the same Vulkan format share the image (views carry the difference).
    return {static_cast<std::uint32_t>(resource.baseAddress | resource.pipeBankXor), static_cast<std::uint32_t>(resource.baseAddress >> 32u), resource.width, resource.height, ((slices ? 0u : resource.depthOrLastArray) << 16u) | (resource.mipCount & 0xffffu), (static_cast<std::uint32_t>(resource.tileMode) << 12u) | (static_cast<std::uint32_t>(resource.dimension) << 20u), slices ? 0u : resource.baseArray, static_cast<std::uint32_t>(StorageFormatForGuest(context, resource.format))};
}

bool HoldsSlices(const StorageTexture& image, const GuestTextureResource& resource) {
    return image.Descriptor().depthOrLastArray >= resource.depthOrLastArray;
}

struct ExtendedSurfaces {
    std::mutex mutex;
    std::map<std::array<std::uint32_t, 8>, std::uint32_t> levels;
    std::atomic<bool> any{false};
};

ExtendedSurfaces& ExtendedChains() {
    static ExtendedSurfaces surfaces;
    return surfaces;
}

std::uint32_t AllocatedLevels(const GuestTextureResource& resource) {
    return resource.allocatedMipCount != 0 ? resource.allocatedMipCount : resource.mipCount;
}

GuestTextureResource StorageSurface(const Context& context, const GuestTextureResource& viewed) {
    auto surface = viewed;
    if (ArraySliceViews(viewed)) surface.baseArray = 0;
    auto& surfaces = ExtendedChains();
    const bool extended = viewed.mipCount > AllocatedLevels(viewed);
    if (!extended && !surfaces.any.load(std::memory_order_acquire)) return surface;
    surface.mipCount = AllocatedLevels(viewed);
    const auto identity = SurfaceKey(context, surface);
    std::lock_guard lock(surfaces.mutex);
    if (extended) {
        auto& levels = surfaces.levels[identity];
        levels = std::max(levels, viewed.mipCount);
        surfaces.any.store(true, std::memory_order_release);
        surface.mipCount = levels;
    } else if (const auto it = surfaces.levels.find(identity); it != surfaces.levels.end()) {
        surface.mipCount = it->second;
    } else {
        surface.mipCount = viewed.mipCount;
    }
    return surface;
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& viewed, std::uint32_t mip, std::uint64_t guestBytes, bool written) {
    Require(viewed.dimension != TextureDimension::k2DMsaa, "storage image access to a multisampled texture is not implemented");
    RequireNoMultisampleTarget(viewed.baseAddress, guestBytes != 0 ? static_cast<std::size_t>(guestBytes) : 1u, "a storage image access");
    if (auto depth = DepthSurfaceStorage(context, words, viewed, mip)) return depth;
    if (written) RetireDepthSurfaces(context.device, viewed.baseAddress, guestBytes != 0 ? guestBytes : DescribeSurface(viewed).guestBytes);
    static_cast<void>(words);
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    const bool profile = LookupOutcomes::Profiled();
    auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& counters = TextureCounts();
    const auto resource = StorageSurface(context, viewed);
    const StorageKey key{context.device, SurfaceKey(context, resource)};
    auto& cache = StorageTextures();
    std::vector<std::shared_ptr<StorageTexture>> evicted;
    std::shared_ptr<StorageTexture> texture;
    {
        std::lock_guard lock(cache.mutex);
        if (resource.mipCount > AllocatedLevels(resource)) {
            auto allocated = resource;
            allocated.mipCount = AllocatedLevels(resource);
            if (const auto found = findStorage(cache, {context.device, SurfaceKey(context, allocated)}); found != cache.entries.end()) evicted.push_back(evictStorage(cache, found));
        }
        auto it = findStorage(cache, key);
        if (it != cache.entries.end() && (MetadataMoved(*it->texture, resource) || !HoldsSlices(*it->texture, resource))) {
            evicted.push_back(evictStorage(cache, it));
            it = cache.entries.end();
        }
        if (it != cache.entries.end()) {
            it->texture->Refresh();
            cache.entries.splice(cache.entries.begin(), cache.entries, it);
            texture = it->texture;
            counters.storageHits.fetch_add(1, std::memory_order_relaxed);
            if (profile) LookupOutcomes::Add(LookupOutcomes::StorageHit, start);
        } else {
            if (guestBytes == 0) guestBytes = DescribeSurface(resource).guestBytes;
            if (StorageTexture::FlushPending(resource.baseAddress, static_cast<std::size_t>(guestBytes), nullptr, "storage image creation", PublishScope::None) && profile) start = LookupOutcomes::Add(LookupOutcomes::PendingFlush, start);
            CachedStorageTexture entry{key, mip, StorageTexture::Create(context, *context.detiler, resource, mip)};
            static const bool keepNew = std::getenv("APS5_NO_KEEP_NEW_STORAGE") == nullptr;
            if (auto* recorder = Recorder::Active(); keepNew && recorder != nullptr && GuestMemory::GpuMutex().HeldByThisThread() && recorder->Recording()) recorder->Keep(entry.texture);
            entry.accounted = heldBytes(*entry.texture);
            const auto budget = storageBudget(context);
            while (!cache.entries.empty() && cache.bytes + entry.accounted > budget) evicted.push_back(evictStorage(cache, std::prev(cache.entries.end())));
            cache.bytes += entry.accounted;
            texture = entry.texture;
            cache.entries.push_front(std::move(entry));
            cache.index[key] = cache.entries.begin();
            cache.byImage[texture.get()] = cache.entries.begin();
            texture->SetCached(true);
            counters.storageCreated.fetch_add(1, std::memory_order_relaxed);
            if (profile) LookupOutcomes::Add(LookupOutcomes::StorageMade, start);
        }
    }
    return texture;
}

}

std::uint64_t TextureCacheBudget(const VkPhysicalDeviceMemoryProperties& memory) {
    const auto heap = largestDeviceLocalHeap(memory);
    return std::max<std::uint64_t>(2048ull << 20u, heap ? memory.memoryHeaps[*heap].size / 4u : 0u);
}

std::uint64_t StorageTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported) {
    const auto heap = largestDeviceLocalHeap(memory);
    if (reported == nullptr || !heap || reported->heapBudget[*heap] == 0) return TextureCacheBudget(memory);
    return std::max<std::uint64_t>(512ull << 20u, std::min<std::uint64_t>(TextureCacheBudget(memory), reported->heapBudget[*heap] / 4u));
}

std::uint64_t SampledTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t textureBytes) {
    const auto heap = largestDeviceLocalHeap(memory);
    if (reported == nullptr || !heap || reported->heapBudget[*heap] == 0) return TextureCacheBudget(memory);
    const auto budget = reported->heapBudget[*heap];
    const auto usage = reported->heapUsage[*heap];
    const auto reserved = usage - std::min<std::uint64_t>(usage, textureBytes) + TextureCacheBudget(memory) + budget / 8u;
    return std::max<std::uint64_t>(2048ull << 20u, budget > reserved ? budget - reserved : 0u);
}

bool SampledBudgetReportDue(std::uint64_t reported, std::uint64_t budget, std::chrono::steady_clock::duration sinceReport) {
    const auto change = budget > reported ? budget - reported : reported - budget;
    return change > reported / 10u && sinceReport >= std::chrono::seconds(10);
}

TextureCacheUse TextureCacheUsage() {
    TextureCacheUse use;
    {
        auto& sampled = Textures();
        std::lock_guard lock(sampled.mutex);
        dropUncachedViews(sampled);
        use.sampledEntries = sampled.entries.size();
        use.sampledBytes = sampled.bytes;
    }
    use.storageBytes = storageCacheBytes();
    return use;
}

std::uint64_t SampledTextureCacheBudget(const Context& context) {
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    return sampledBudget(context, cache);
}

std::uint64_t StorageTextureCacheBudget(const Context& context) {
    return storageBudget(context);
}

std::shared_ptr<Texture> CachedSampledTexture(const Context& context, std::span<const std::uint32_t> words) {
    const auto resource = DecodeTextureResource(words);
    return cachedTexture(context, words, resource, ViewComponents(resource));
}

bool SampledTexturesShareEntry(std::span<const std::uint32_t> first, std::span<const std::uint32_t> second) {
    Require(first.size() == 8 && second.size() == 8, "sampled texture keys take eight descriptor words");
    const VkComponentMapping components{VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    const auto firstKey = MakeTextureKey(VK_NULL_HANDLE, first, components);
    const auto secondKey = MakeTextureKey(VK_NULL_HANDLE, second, components);
    return firstKey == secondKey && TextureKeyHash{}(firstKey) == TextureKeyHash{}(secondKey);
}

void FlushCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot flush textures without a Vulkan device");
    GuestMemory::AssertGpuLockHeld("FlushCachedTextures");
    {
        auto& cache = StorageTextures();
        std::lock_guard lock(cache.mutex);
        for (const auto& entry : cache.entries) {
            if (entry.key.device == device) entry.texture->Flush();
        }
    }
    StorageTexture::FlushAllPending("cached texture flush");
}

void ClearCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot clear textures without a Vulkan device");
    auto& sampled = Textures();
    {
        std::lock_guard lock(sampled.mutex);
        for (auto it = sampled.entries.begin(); it != sampled.entries.end();) {
            if (it->key.device == device) eraseTexture(sampled, it++);
            else ++it;
        }
        if (sampled.budgetDevice == device) {
            sampled.budgetDevice = VK_NULL_HANDLE;
            sampled.budgetPhysical = VK_NULL_HANDLE;
        }
    }
    {
        auto& budget = StorageBudget();
        std::lock_guard lock(budget.mutex);
        if (budget.device == device) {
            budget.device = VK_NULL_HANDLE;
            budget.physical = VK_NULL_HANDLE;
        }
    }
    auto& storage = StorageTextures();
    std::vector<std::shared_ptr<StorageTexture>> cleared;
    {
        std::lock_guard lock(storage.mutex);
        for (auto it = storage.entries.begin(); it != storage.entries.end();) {
            if (it->key.device != device) {
                ++it;
                continue;
            }
            cleared.push_back(it->texture);
            it->texture->SetCached(false);
            noteUncached(it->texture);
            storage.bytes -= it->accounted;
            storage.index.erase(it->key);
            storage.byImage.erase(it->texture.get());
            it = storage.entries.erase(it);
        }
    }
}

bool StorageImageCached(const Context& context, const StorageTexture* image) {
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    const auto it = findStorageByImage(cache, context.device, image);
    if (it == cache.entries.end()) return false;
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    return true;
}

namespace {

// StorageImageCached for several images under one acquisition of the cache mutex.
bool StorageImagesCached(const Context& context, std::span<const StorageTexture* const> images) {
    if (images.empty()) return true;
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    for (const auto* image : images) {
        const auto it = findStorageByImage(cache, context.device, image);
        if (it == cache.entries.end()) return false;
        cache.entries.splice(cache.entries.begin(), cache.entries, it);
    }
    return true;
}

}

std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource) {
    return cachedStorageTexture(context, {}, resource, 0);
}

bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress) {
    GuestTextureResource resource{};
    resource.dccAddress = dccAddress;
    return !MetadataMoved(image, resource);
}

std::unique_ptr<Buffer> CreateRuntimeFormatTable(const Context& context) {
    const auto words = ShaderRecompiler::RuntimeFormatTable();
    Require(RuntimeFormatTableBytes <= context.limits.maxStorageBufferRange, "the format table exceeds the storage buffer range limit");
    auto table = std::make_unique<Buffer>(context, RuntimeFormatTableBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer staging(context, RuntimeFormatTableBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(staging.Bytes().data(), words.data(), RuntimeFormatTableBytes);
    CommandBatch batch(context);
    CopyBuffer(context, batch.Handle(), staging.Handle(), 0, table->Handle(), 0, RuntimeFormatTableBytes);
    RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    batch.SubmitAndWait();
    return table;
}

namespace {

const char* roleName(ShaderRecompiler::DescriptorRole role) {
    switch (role) {
        case ShaderRecompiler::DescriptorRole::GuestSamplers: return "GuestSamplers";
        case ShaderRecompiler::DescriptorRole::Gds: return "Gds";
        case ShaderRecompiler::DescriptorRole::BdaPagetable: return "BdaPagetable";
        case ShaderRecompiler::DescriptorRole::FaultBuffer: return "FaultBuffer";
        case ShaderRecompiler::DescriptorRole::FlattenedSrt: return "FlattenedSrt";
        case ShaderRecompiler::DescriptorRole::ShaderData: return "ShaderData";
        case ShaderRecompiler::DescriptorRole::ImageTable: return "ImageTable";
        case ShaderRecompiler::DescriptorRole::ImageTableMap: return "ImageTableMap";
        case ShaderRecompiler::DescriptorRole::FormatTable: return "FormatTable";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor role");
}

const char* kindName(ShaderRecompiler::DescriptorKind kind) {
    switch (kind) {
        case ShaderRecompiler::DescriptorKind::UniformBuffer: return "UniformBuffer";
        case ShaderRecompiler::DescriptorKind::StorageBuffer: return "StorageBuffer";
        case ShaderRecompiler::DescriptorKind::UniformTexelBuffer: return "UniformTexelBuffer";
        case ShaderRecompiler::DescriptorKind::StorageTexelBuffer: return "StorageTexelBuffer";
        case ShaderRecompiler::DescriptorKind::SampledImage: return "SampledImage";
        case ShaderRecompiler::DescriptorKind::StorageImage: return "StorageImage";
        case ShaderRecompiler::DescriptorKind::Sampler: return "Sampler";
        case ShaderRecompiler::DescriptorKind::DeviceAddress: return "DeviceAddress";
        case ShaderRecompiler::DescriptorKind::SamplerSlot: return "SamplerSlot";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor kind");
}

// Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
// storage writes), so the second and later ones share the first one's image lookup.
// APS5_NO_STORAGE_DEDUPE=1 looks each one up (and refreshes the image) separately as before.
bool StorageDedupeEnabled() {
    static const bool disabled = std::getenv("APS5_NO_STORAGE_DEDUPE") != nullptr;
    return !disabled;
}

// Whether storage element `element` of `binding` repeats the previous element's eight words.
bool SameAsPreviousStorageElement(const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t element) {
    if (element == 0) return false;
    const auto words = binding.guestDescriptor.begin() + static_cast<std::size_t>(element) * 8u;
    return std::equal(words, words + 8, words - 8);
}

}

ShaderResources::ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes) : ShaderResources(context, std::array<CompiledShader, 2>{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, {ShaderRecompiler::ShaderStage::Fragment, &fragment, static_cast<std::uint32_t>(vertex.pushConstants.size())}}}, target, 1, indexAddress, indexBytes) {}

ShaderResources::ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes) : context(context), guestMemory(context) {
    drawBuild = true;
    for (const auto& shader : shaders) {
        if (shader.codeAddress != 0) stageShaders.emplace_back(static_cast<std::uint32_t>(shader.stage), shader.codeAddress);
    }
    prepareAddressBindings(shaders);
    build(shaders, &target, colorAttachments, indexAddress, indexBytes);
}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute) : ShaderResources(context, compute, false) {}

namespace {

// Whether a compute stage maps registered guest memory through BDA tables (see prepareAddressBindings).
bool UsesAddressTables(const CompiledShader& compute) {
    Require(compute.program != nullptr, "missing compiled shader");
    return std::any_of(compute.program->bindings.begin(), compute.program->bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; });
}

}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, bool deferred, std::uint64_t dispatchThreads, std::uint64_t linearThreads) : context(context), guestMemory(context), dispatchThreads(dispatchThreads), deferredCompute(compute) {
    Require(compute.stage == ShaderRecompiler::ShaderStage::Compute, "compute resources require a compute shader");
    computeBuild = true;
    imageStoresRecorded = compute.program != nullptr && compute.program->imageStoresRecorded;
    this->linearThreads = linearThreads;
    if (compute.program != nullptr) linearStore = compute.program->linearStore;
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    for (const auto& shader : shaders) {
        if (shader.codeAddress != 0) stageShaders.emplace_back(static_cast<std::uint32_t>(shader.stage), shader.codeAddress);
    }
    if (!deferred) {
        prepareAddressBindings(shaders);
        build(shaders, nullptr, 0, 0, 0);
        forgetDeferredInputs();
        return;
    }
    lockedBuild = UsesAddressTables(compute);
    if (lockedBuild) {
        prepareAddressBindings(shaders);
        return;
    }
    unlockedPrepare = true;
    prepareAddressBindings(shaders);
    buildPrepare(shaders, nullptr, 0, 0, 0);
}

void ShaderResources::Complete() {
    Require(!completed, "shader resources were already completed");
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (lockedBuild) buildPrepare(shaders, nullptr, 0, 0, 0);
    buildComplete();
    forgetDeferredInputs();
}

void ShaderResources::forgetDeferredInputs() {
    // The compiled shader and the captured regions belong to the dispatch, which returns while this
    // object lives on in the resource cache and the recorder: nothing may reach for them after the
    // build, so they are dropped rather than left dangling.
    deferredCompute = {};
}

void ShaderResources::build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes) {
    buildPrepare(shaders, target, colorAttachments, indexAddress, indexBytes);
    buildComplete();
}

namespace {

VkDescriptorSetLayout CreateSetLayout(const Context& context, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind, std::span<const VkDescriptorBindingFlags> flags) {
    Require(flags.size() == bindings.size(), "descriptor binding flag count differs from layout");
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlags.bindingCount = static_cast<std::uint32_t>(flags.size());
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.pNext = std::ranges::any_of(flags, [](auto value) { return value != 0; }) ? &bindingFlags : nullptr;
    info.flags = updateAfterBind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0;
    info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &layout), "vkCreateDescriptorSetLayout");
    return layout;
}

std::vector<VkDescriptorPoolSize> PoolSizes(std::span<const VkDescriptorSetLayoutBinding> bindings) {
    std::map<VkDescriptorType, std::uint32_t> counts;
    for (const auto& binding : bindings) counts[binding.descriptorType] += binding.descriptorCount;
    std::vector<VkDescriptorPoolSize> sizes;
    for (const auto& [type, count] : counts) sizes.push_back({type, count});
    return sizes;
}

VkDescriptorPool CreateSetPool(const Context& context, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind) {
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.flags = updateAfterBind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
    info.maxSets = 1;
    info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    info.pPoolSizes = sizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &info, nullptr, &pool), "vkCreateDescriptorPool");
    return pool;
}

// APS5_PROFILE_DRAW: the [resources] phase totals. Each thread accumulates its builds' phases in
// arrays of its own and merges them into the shared totals every 1000 of its builds (and when it
// ends), so no build takes the shared mutex per phase; the totals lag by up to 999 builds per
// worker.
constexpr std::size_t BuildPhaseCount = static_cast<std::size_t>(ShaderResources::BuildPhase::Count);
constexpr std::array<const char*, BuildPhaseCount> BuildPhaseNames{"bindings", "guest memory upload", "descriptors", "stage A", "images", "bda", "stage B"};

struct BuildProfile {
    std::mutex mutex;
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
};

BuildProfile& Builds() {
    static BuildProfile profile;
    return profile;
}

bool BuildProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

struct ThreadBuildProfile {
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
    ~ThreadBuildProfile() { merge(); }
    void merge() {
        auto& profile = Builds();
        std::lock_guard lock(profile.mutex);
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) profile.ms[i] += ms[i];
        profile.builds += builds;
        ms = {};
        if (builds == 0) return;
        builds = 0;
        std::string report;
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) report += " " + std::string(BuildPhaseNames[i]) + "=" + std::to_string(static_cast<long long>(profile.ms[i])) + "ms";
        AgcDriver::ProfilePrint_nid_no_patch("[resources] %llu builds, phase totals:%s\n", static_cast<unsigned long long>(profile.builds), report.c_str());
    }
};

ThreadBuildProfile& ThreadBuilds() {
    thread_local ThreadBuildProfile* profileSlot = nullptr;
    auto& profile = ShaderRecompiler::ThreadOwned(profileSlot);
    return profile;
}

void addBuildPhase(ShaderResources::BuildPhase which, double ms) {
    ThreadBuilds().ms[static_cast<std::size_t>(which)] += ms;
}

}

struct ShaderResources::Program {
    explicit Program(const Context& context) : context(context) {}
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    ~Program();
    std::uint32_t setOf(const Binding& binding) const;
    Context context;
    std::vector<Binding> bindings;
    std::vector<Allocation> allocations;
    std::vector<DispatchWords> dispatchWords;
    std::vector<DataPatch> dataPatches;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pushPatches;
    std::vector<std::shared_ptr<Sampler>> samplers;
    std::vector<SamplerSource> samplerSources;
    std::vector<SamplerSlotPatch> samplerSlotPatches;
    std::vector<TableStage> tableStages;
    std::vector<std::uint32_t> imageTableStages;
    std::vector<std::uint32_t> layoutKey;
    std::vector<std::uint32_t> fragmentLayoutKey;
    std::vector<VkDescriptorBindingFlags> layoutBindingFlags;
    std::vector<VkDescriptorBindingFlags> fragmentBindingFlags;
    std::uint64_t storageBuffers = 0;
    std::int64_t computeShaderData = -1;
    bool updateAfterBind = false;
    bool splitSets = false;
    bool dispatchSets = false;
    std::array<std::vector<VkDescriptorPoolSize>, 2> poolSizes;
    std::array<bool, 2> sets{};
    SetObjects main;
    SetObjects fragment;
    std::vector<VkDescriptorBufferInfo> descriptorBuffers;
    std::vector<VkWriteDescriptorSet> descriptorWrites;
};

ShaderResources::Program::~Program() {
    for (auto* objects : {&main, &fragment}) {
        if (objects->cacheAllocation.pool && context.descriptorCache != nullptr) context.descriptorCache->Free(objects->cacheAllocation);
        if (objects->pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, objects->pool, nullptr);
        if (objects->layout && objects->ownsLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, objects->layout, nullptr);
    }
}

std::uint32_t ShaderResources::Program::setOf(const Binding& binding) const {
    constexpr auto count = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    if (binding.set >= 0) return static_cast<std::uint32_t>(binding.set);
    return splitSets && binding.layout.binding / count == static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Stage::Fragment) ? 1u : 0u;
}

double ShaderResources::phase(BuildPhase which) {
    if (!BuildProfiled()) return 0.0;
    const auto now = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration<double, std::milli>(now - phaseStart).count();
    addBuildPhase(which, ms);
    if (ms > 50) AgcDriver::ProfilePrint_nid_no_patch("[resources] %s took %.0f ms (%zu textures, %zu buffers, bda %d)\n", BuildPhaseNames[static_cast<std::size_t>(which)], ms, textures.size(), shared != nullptr ? shared->allocations.size() : std::size_t{0}, usesBda ? 1 : 0);
    phaseStart = now;
    return ms;
}

namespace {

std::uint32_t IrStage(ShaderRecompiler::ShaderStage stage) {
    using Ir = ShaderRecompiler::IrShaderStage;
    switch (stage) {
        case ShaderRecompiler::ShaderStage::Compute: return static_cast<std::uint32_t>(Ir::Compute);
        case ShaderRecompiler::ShaderStage::Vertex: return static_cast<std::uint32_t>(Ir::Vertex);
        case ShaderRecompiler::ShaderStage::Fragment: return static_cast<std::uint32_t>(Ir::Pixel);
        case ShaderRecompiler::ShaderStage::Mesh: return static_cast<std::uint32_t>(Ir::Mesh);
        case ShaderRecompiler::ShaderStage::Local: return static_cast<std::uint32_t>(Ir::Local);
        case ShaderRecompiler::ShaderStage::TessellationControl: return static_cast<std::uint32_t>(Ir::TessellationControl);
        case ShaderRecompiler::ShaderStage::TessellationEvaluation: return static_cast<std::uint32_t>(Ir::TessellationEvaluation);
        case ShaderRecompiler::ShaderStage::Geometry: break;
    }
    return static_cast<std::uint32_t>(Ir::Unknown);
}

std::string HexText(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(value));
    return text;
}

void ValidateDescriptorLayout(const Context& context, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind) {
    const auto maxStageResources = updateAfterBind ? context.descriptorIndexingLimits.maxPerStageUpdateAfterBindResources : context.limits.maxPerStageResources;
    std::map<VkDescriptorType, std::uint64_t> heapTotals;
    std::map<std::pair<VkShaderStageFlags, VkDescriptorType>, std::uint64_t> stageTotals;
    std::map<VkShaderStageFlags, std::uint64_t> stageResources;
    for (const auto& layout : bindings) {
        heapTotals[layout.descriptorType] += layout.descriptorCount;
        for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
            const auto stage = VkShaderStageFlags{1u} << bit;
            if ((layout.stageFlags & stage) == 0u) continue;
            const auto count = stageTotals[{stage, layout.descriptorType}] += layout.descriptorCount;
            const auto all = stageResources[stage] += layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ? 0u : layout.descriptorCount;
            Require(all <= maxStageResources, "typed heaps exceed per-stage resource capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) Require(count <= context.limits.maxPerStageDescriptorSampledImages, "typed heaps exceed per-stage sampled image capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) Require(count <= context.limits.maxPerStageDescriptorStorageImages, "typed heaps exceed per-stage storage image capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) Require(count <= context.limits.maxPerStageDescriptorSamplers, "typed heaps exceed per-stage sampler capacity");
        }
    }
    Require(heapTotals[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] <= context.limits.maxDescriptorSetSampledImages && heapTotals[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] <= context.limits.maxDescriptorSetStorageImages && heapTotals[VK_DESCRIPTOR_TYPE_SAMPLER] <= context.limits.maxDescriptorSetSamplers, "typed heaps exceed descriptor set capacity");
}

void ValidateRuntimeResources(const CompiledShader& shader, std::span<const std::uint32_t> words) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    Require(words.size() == program.shaderDataDwords, "invalid compact shader data size");
}

}

namespace {

constexpr std::size_t ProgramResourceCapacity = 8192;

struct ResourceCounterSet {
    std::atomic<std::uint64_t> programs{0};
    std::atomic<std::uint64_t> uses{0};
    std::atomic<std::uint64_t> dispatchSets{0};
    std::atomic<std::uint64_t> dispatchData{0};
};

ResourceCounterSet& ResourceCounters() {
    static ResourceCounterSet counters;
    return counters;
}

bool DispatchWordsBinding(const ShaderRecompiler::RecompileResult& program, const ShaderRecompiler::DescriptorBinding& binding) {
    return binding.role == ShaderRecompiler::DescriptorRole::ShaderData && program.pushConstants.empty();
}

std::vector<std::uint32_t> ProgramKey(const Context& context, std::span<const CompiledShader> shaders, std::uint32_t colorAttachments) {
    std::vector<std::uint32_t> key;
    const auto pointer = [&](const void* value) {
        const auto wide = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
        key.insert(key.end(), {static_cast<std::uint32_t>(wide), static_cast<std::uint32_t>(wide >> 32u)});
    };
    const auto bits = [&](const std::vector<bool>& values) {
        key.push_back(static_cast<std::uint32_t>(values.size()));
        for (const bool value : values) key.push_back(value ? 1u : 0u);
    };
    const auto& limits = context.limits;
    const auto& indexing = context.descriptorIndexingLimits;
    key.insert(key.end(), {colorAttachments, static_cast<std::uint32_t>(shaders.size()), limits.maxBoundDescriptorSets, limits.maxPerStageResources, limits.maxPerStageDescriptorStorageBuffers, limits.maxStorageBufferRange, limits.maxDescriptorSetStorageBuffers, indexing.maxPerStageDescriptorUpdateAfterBindStorageBuffers, indexing.maxPerStageUpdateAfterBindResources, indexing.maxDescriptorSetUpdateAfterBindStorageBuffers, context.graphicsPipelineLibrary ? 1u : 0u});
    pointer(context.formatTable);
    pointer(context.deviceImages);
    pointer(context.samplerCache);
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        const auto& program = *shader.program;
        key.insert(key.end(), {static_cast<std::uint32_t>(shader.stage), shader.pushConstantOffset, program.runtimeAbiVersion, static_cast<std::uint32_t>(program.pushConstants.size()), program.samplerSlotDword, program.shaderDataDwords, static_cast<std::uint32_t>(program.bindings.size())});
        for (const auto& binding : program.bindings) {
            key.insert(key.end(), {static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count, binding.readOnly ? 1u : 0u, static_cast<std::uint32_t>(binding.guestDescriptor.size())});
            if (!DispatchWordsBinding(program, binding)) key.insert(key.end(), binding.guestDescriptor.begin(), binding.guestDescriptor.end());
            bits(binding.samplerDepthCompare);
            bits(binding.samplerUnnormalized);
            bits(binding.samplerIntegerBorder);
            bits(binding.samplerUsed);
            key.push_back(static_cast<std::uint32_t>(binding.tableSamplerElements.size()));
            for (const auto& elements : binding.tableSamplerElements) {
                key.push_back(static_cast<std::uint32_t>(elements.size()));
                key.insert(key.end(), elements.begin(), elements.end());
            }
        }
    }
    return key;
}

}

ProgramResourceCounts ProgramResourceCounters() {
    const auto& counters = ResourceCounters();
    return {counters.programs.load(std::memory_order_relaxed), counters.uses.load(std::memory_order_relaxed), counters.dispatchSets.load(std::memory_order_relaxed), counters.dispatchData.load(std::memory_order_relaxed)};
}


class ProgramResourceCache {
public:
    std::shared_ptr<const ShaderResources::Program> Get(const Context& context, std::span<const CompiledShader> shaders, std::uint32_t colorAttachments) {
        auto key = ProgramKey(context, shaders, colorAttachments);
        {
            std::lock_guard lock(mutex);
            if (const auto found = entries.find(key); found != entries.end()) {
                order.splice(order.begin(), order, found->second.order);
                return found->second.program;
            }
        }
        auto built = ShaderResources::BuildProgram(context, shaders, colorAttachments);
        std::lock_guard lock(mutex);
        const auto [found, inserted] = entries.try_emplace(std::move(key), Entry{built, order.end()});
        if (!inserted) {
            order.splice(order.begin(), order, found->second.order);
            return found->second.program;
        }
        order.push_front(&found->first);
        found->second.order = order.begin();
        while (entries.size() > ProgramResourceCapacity) {
            const auto* oldest = order.back();
            order.pop_back();
            entries.erase(*oldest);
        }
        return built;
    }

private:
    struct KeyHash {
        std::size_t operator()(const std::vector<std::uint32_t>& key) const noexcept {
            return static_cast<std::size_t>(hashWords(0xcbf29ce484222325ull, key.data(), key.size()));
        }
    };
    using Order = std::list<const std::vector<std::uint32_t>*>;
    struct Entry {
        std::shared_ptr<const ShaderResources::Program> program;
        Order::iterator order;
    };
    std::mutex mutex;
    std::unordered_map<std::vector<std::uint32_t>, Entry, KeyHash> entries;
    Order order;
};

std::shared_ptr<const ShaderResources::Program> ShaderResources::ProgramFor(const Context& context, std::span<const CompiledShader> shaders, std::uint32_t colorAttachments) {
    if (context.descriptorCache == nullptr) return BuildProgram(context, shaders, colorAttachments);
    return context.descriptorCache->Programs().Get(context, shaders, colorAttachments);
}

ShaderResources::SetObjects ShaderResources::AllocateSet(const Context& context, VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind) {
    static const bool noPoolCache = std::getenv("APS5_NO_POOL_CACHE") != nullptr;
    SetObjects objects;
    if (context.descriptorCache != nullptr && !noPoolCache) {
        objects.cacheAllocation = context.descriptorCache->Allocate(layout, sizes, updateAfterBind, std::nullopt);
        objects.set = objects.cacheAllocation.set;
    }
    if (objects.set == VK_NULL_HANDLE) {
        objects.pool = CreateSetPool(context, sizes, updateAfterBind);
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = objects.pool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &layout;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &objects.set), "vkAllocateDescriptorSets");
    }
    return objects;
}

bool ShaderResources::DispatchAllocation(const Allocation& allocation) {
    return allocation.guest || (allocation.buffer == nullptr && allocation.role != ShaderRecompiler::DescriptorRole::FormatTable);
}

std::shared_ptr<const ShaderResources::Program> ShaderResources::BuildProgram(const Context& context, std::span<const CompiledShader> shaders, std::uint32_t colorAttachments) {
    auto built = std::make_shared<Program>(context);
    auto& program = *built;
    auto& allocations = program.allocations;
    auto& bindings = program.bindings;
    auto& samplers = program.samplers;
    auto& updateAfterBind = program.updateAfterBind;
    auto& storageBuffers = program.storageBuffers;
    const auto addDataBuffer = [&](std::span<const std::uint32_t> words) {
        const auto size = words.size() * sizeof(std::uint32_t);
        Require(size <= context.limits.maxStorageBufferRange, "shader data buffer exceeds descriptor range limit");
        auto buffer = std::make_unique<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memcpy(buffer->Bytes().data(), words.data(), size);
        allocations.push_back({0, size, false, std::move(buffer)});
        return allocations.size() - 1;
    };
    const auto addGuestSamplers = [&](const ShaderRecompiler::DescriptorBinding& binding, const CompiledShader& shader) {
        Require(binding.kind == ShaderRecompiler::DescriptorKind::SamplerSlot, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role GuestSamplers: guest samplers take slots in the descriptor heap");
        Require(binding.count != 0, "empty descriptor binding");
        Require(binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 4u, "guest sampler descriptor must contain 4 dwords");
        Require(binding.samplerDepthCompare.size() == binding.count, "guest sampler binding is missing depth comparison metadata");
        Require(binding.samplerUnnormalized.empty() || binding.samplerUnnormalized.size() == binding.count, "guest sampler binding has unnormalized coordinate metadata of another size");
        Require(binding.samplerUsed.size() == binding.count, "guest sampler binding is missing the elements its shader reads");
        const auto& push = shader.program->pushConstants;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 4u, 4u);
            const bool compareEnable = binding.samplerDepthCompare.at(element);
            const bool unnormalized = element < binding.samplerUnnormalized.size() && binding.samplerUnnormalized[element];
            const bool integerBorder = element < binding.samplerIntegerBorder.size() && binding.samplerIntegerBorder[element];
            if (!binding.samplerUsed[element]) {
                samplers.push_back(nullptr);
            } else if (context.samplerCache != nullptr && !SamplerCacheDisabled()) {
                samplers.push_back(context.samplerCache->Get(context, words, compareEnable, unnormalized, true, integerBorder));
            } else {
                auto resource = DecodeSamplerResource(words, unnormalized, true, integerBorder);
                resource.compareEnable = compareEnable;
                samplers.push_back(std::make_shared<Sampler>(context, resource));
            }
            program.samplerSources.push_back({{words[0], words[1], words[2], words[3]}, compareEnable, unnormalized, integerBorder});
            const auto position = (shader.program->samplerSlotDword + element) * 4u;
            SamplerSlotPatch patch{binding.samplerUsed[element] ? samplers.size() - 1u : std::numeric_limits<std::size_t>::max()};
            if (!push.empty()) {
                Require(position + 4u <= push.size(), "a guest sampler slot lies outside the shader's push constants");
                patch.pushByte = static_cast<std::int32_t>(shader.pushConstantOffset + position);
            } else {
                patch.dataByte = position;
            }
            program.samplerSlotPatches.push_back(patch);
        }
        Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_SAMPLER, binding.count, 0, nullptr}, {}, {}};
        item.noDescriptor = true;
        bindings.push_back(std::move(item));
    };
    Require(!shaders.empty() && context.limits.maxBoundDescriptorSets > ShaderRecompiler::RuntimeAbi::HeapDescriptorSet, "shader descriptor set exceeds device limits");
    std::set<std::uint32_t> occupied;
    for (std::size_t shaderIndex = 0; shaderIndex < shaders.size(); ++shaderIndex) {
        const auto& shader = shaders[shaderIndex];
        Require(shader.program != nullptr, "missing compiled shader");
        ShaderRecompiler::RuntimeAbi::RequireVersion(shader.program->runtimeAbiVersion);
        const VkShaderStageFlags flags = VulkanStage(shader.stage);
        std::uint64_t stageStorageBuffers = 0;
        std::uint64_t stageResources = shader.stage == ShaderRecompiler::ShaderStage::Fragment ? colorAttachments : 0;
        Require(stageResources <= context.limits.maxPerStageResources, "shader descriptors exceed per-stage limits");
        std::int64_t shaderData = -1;
        const auto firstSampler = samplers.size();
        const auto firstSlotPatch = program.samplerSlotPatches.size();
        std::optional<std::size_t> stageTable;
        const auto table = [&]() -> TableStage& {
            if (!stageTable.has_value()) {
                program.tableStages.push_back({shader.stage});
                stageTable = program.tableStages.size() - 1u;
            }
            return program.tableStages[*stageTable];
        };
        const auto requireStageLimits = [&] {
            if (stageStorageBuffers <= context.limits.maxPerStageDescriptorStorageBuffers && stageResources <= context.limits.maxPerStageResources) return;
            const auto& limits = context.descriptorIndexingLimits;
            Require(stageStorageBuffers <= limits.maxPerStageDescriptorUpdateAfterBindStorageBuffers && stageResources <= limits.maxPerStageUpdateAfterBindResources, "shader descriptors exceed per-stage limits");
            updateAfterBind = true;
        };
        for (std::size_t bindingIndex = 0; bindingIndex < shader.program->bindings.size(); ++bindingIndex) {
            const auto& binding = shader.program->bindings[bindingIndex];
            Require(binding.descriptorSet == ShaderRecompiler::RuntimeAbi::DescriptorSet || (shader.stage == ShaderRecompiler::ShaderStage::Fragment && binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet), "unexpected descriptor set: a shader resource must use its stage's descriptor set");
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable) {
                Require(binding.binding == ShaderRecompiler::RuntimeAbi::ImageTableBinding && binding.kind == ShaderRecompiler::DescriptorKind::SampledImage && binding.count == 0u && binding.guestDescriptor.empty(), "image table binding is malformed");
                auto& stage = table();
                Require(!stage.images, "a stage binds two image tables");
                stage.images = true;
                stage.samplerElements = binding.tableSamplerElements;
                stage.firstSampler = firstSampler;
                continue;
            }
            Require(occupied.insert(binding.binding).second, "duplicate shader binding");
            if (binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers) {
                addGuestSamplers(binding, shader);
                continue;
            }
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap) {
                Require(binding.kind == ShaderRecompiler::DescriptorKind::StorageBuffer && binding.count == 1 && binding.guestDescriptor.size() >= ShaderRecompiler::ImageTableAbi::HeaderWords && !binding.readOnly && binding.guestDescriptor[ShaderRecompiler::ImageTableAbi::HeaderVersion] == ShaderRecompiler::ImageTableAbi::Version, "image table map binding is malformed");
                const auto size = binding.guestDescriptor.size() * sizeof(std::uint32_t);
                Require(size <= context.limits.maxStorageBufferRange, "image table map exceeds the storage buffer range limit");
                stageStorageBuffers += 1;
                stageResources += 1;
                storageBuffers += 1;
                requireStageLimits();
                auto& stage = table();
                Require(stage.mapAllocation == std::numeric_limits<std::size_t>::max(), "a stage binds two image table maps");
                stage.map = binding.guestDescriptor;
                allocations.push_back({0, size, false, std::make_unique<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), ShaderRecompiler::DescriptorRole::ImageTableMap});
                stage.mapAllocation = allocations.size() - 1u;
                bindings.push_back({{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags, nullptr}, {allocations.size() - 1u}});
                continue;
            }
            if (binding.role == ShaderRecompiler::DescriptorRole::FormatTable) {
                Require(binding.kind == ShaderRecompiler::DescriptorKind::StorageBuffer && binding.count == 1 && binding.guestDescriptor.empty() && binding.readOnly, "format table binding is malformed");
                Require(context.formatTable != nullptr, "the format table was not created at device setup");
                stageStorageBuffers += 1;
                stageResources += 1;
                storageBuffers += 1;
                requireStageLimits();
                allocations.push_back({0, RuntimeFormatTableBytes, false, nullptr, ShaderRecompiler::DescriptorRole::FormatTable});
                bindings.push_back({{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags, nullptr}, {allocations.size() - 1u}});
                continue;
            }
            const bool addressRole = binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable || binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer;
            const bool bufferRole = addressRole || binding.role == ShaderRecompiler::DescriptorRole::ShaderData || binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt || binding.role == ShaderRecompiler::DescriptorRole::Gds;
            if (!bufferRole) Require(false, std::string("unsupported descriptor role ") + roleName(binding.role));
            if (binding.kind != ShaderRecompiler::DescriptorKind::StorageBuffer) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role) + ": only StorageBuffer is supported");
            Require(!binding.readOnly, std::string("a read-only descriptor binding is allowed only for the format table, not for role ") + roleName(binding.role));
            Require(binding.count != 0, "empty descriptor binding");
            stageStorageBuffers += binding.count;
            stageResources += binding.count;
            storageBuffers += binding.count;
            requireStageLimits();
            Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding.count, flags, nullptr}, {}};
            if (addressRole) {
                item.allocations.push_back(allocations.size());
                allocations.push_back({0, 0, false, nullptr, binding.role});
            } else if (binding.role == ShaderRecompiler::DescriptorRole::Gds) {
                Require(binding.count == 1 && binding.guestDescriptor.empty(), "invalid GDS descriptor contract");
                item.allocations.push_back(allocations.size());
                allocations.push_back({Pm4::GdsAddress(), Pm4::GdsBytes, true, nullptr, ShaderRecompiler::DescriptorRole::ShaderData, true});
            } else {
                Require(binding.count == 1, "shader data and flattened SRT descriptors must not be arrays");
                Require(!binding.guestDescriptor.empty(), "empty shader data descriptor");
                if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) {
                    ValidateRuntimeResources(shader, binding.guestDescriptor);
                }
                if (DispatchWordsBinding(*shader.program, binding)) {
                    const auto size = binding.guestDescriptor.size() * sizeof(std::uint32_t);
                    Require(size <= context.limits.maxStorageBufferRange, "shader data buffer exceeds descriptor range limit");
                    allocations.push_back({0, size, false, nullptr});
                    program.dispatchWords.push_back({allocations.size() - 1u, shaderIndex, bindingIndex});
                    item.allocations.push_back(allocations.size() - 1u);
                } else {
                    item.allocations.push_back(addDataBuffer(binding.guestDescriptor));
                }
                if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) shaderData = static_cast<std::int64_t>(item.allocations.back());
            }
            bindings.push_back(std::move(item));
        }
        if (shader.stage == ShaderRecompiler::ShaderStage::Compute) program.computeShaderData = shaderData;
        for (auto index = firstSlotPatch; index < program.samplerSlotPatches.size(); ++index) program.samplerSlotPatches[index].dataAllocation = shaderData;
        if (stageTable.has_value() && program.tableStages[*stageTable].images) program.tableStages[*stageTable].samplerCount = samplers.size() - program.tableStages[*stageTable].firstSampler;
    }
    program.splitSets = SplitsSets(context, shaders);
    for (auto& stage : program.tableStages) {
        Require(stage.mapAllocation != std::numeric_limits<std::size_t>::max(), "an image table stage has no map");
        Require(stage.map.size() >= ShaderRecompiler::ImageTableAbi::HeaderWords, "image table map is truncated");
    }
    if (storageBuffers > context.limits.maxDescriptorSetStorageBuffers) {
        Require(storageBuffers <= context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindStorageBuffers, "pipeline descriptors exceed device limits");
        updateAfterBind = true;
    }
    std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> descriptions;
    for (const auto& binding : bindings) {
        if (binding.noDescriptor) continue;
        const auto set = program.setOf(binding);
        descriptions[set].push_back(binding.layout);
        (set == 0 ? program.layoutBindingFlags : program.fragmentBindingFlags).push_back(binding.flags | (updateAfterBind && binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0));
        auto& key = set == 0 ? program.layoutKey : program.fragmentLayoutKey;
        key.insert(key.end(), {binding.layout.binding, static_cast<std::uint32_t>(binding.layout.descriptorType), binding.layout.descriptorCount, binding.layout.stageFlags});
        for (const auto index : binding.allocations) program.dispatchSets = program.dispatchSets || DispatchAllocation(allocations[index]);
    }
    if (updateAfterBind) {
        program.layoutKey.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
        if (program.splitSets) program.fragmentLayoutKey.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
    }
    static const bool noLayoutCache = std::getenv("APS5_NO_LAYOUT_CACHE") != nullptr;
    const auto createLayout = [&](const std::vector<std::uint32_t>& key, const std::vector<VkDescriptorSetLayoutBinding>& description, const std::vector<VkDescriptorBindingFlags>& flags, bool& owns) {
        if (context.descriptorCache != nullptr && !noLayoutCache) return context.descriptorCache->Layout(key, description, updateAfterBind, flags);
        ValidateDescriptorLayout(context, description, updateAfterBind);
        owns = true;
        return CreateSetLayout(context, description, updateAfterBind, flags);
    };
    program.main.layout = createLayout(program.layoutKey, descriptions[0], program.layoutBindingFlags, program.main.ownsLayout);
    if (program.splitSets) {
        program.fragment.layout = createLayout(program.fragmentLayoutKey, descriptions[1], program.fragmentBindingFlags, program.fragment.ownsLayout);
        for (std::uint32_t set = 0; set < 2u; ++set) {
            std::map<VkDescriptorType, std::uint32_t> counts;
            for (const auto& binding : bindings) {
                if (binding.noDescriptor || program.setOf(binding) != set) continue;
                counts[binding.layout.descriptorType] += binding.layout.descriptorCount;
            }
            for (const auto& [type, count] : counts) program.poolSizes[set].push_back({type, count});
            program.sets[set] = true;
        }
    } else if (std::any_of(bindings.begin(), bindings.end(), [](const Binding& binding) { return !binding.noDescriptor; })) {
        if (storageBuffers != 0) program.poolSizes[0].push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<std::uint32_t>(storageBuffers)});
        program.sets[0] = true;
    }
    for (const auto& patch : program.samplerSlotPatches) {
        const std::array<std::uint32_t, 1> slot{patch.sampler == std::numeric_limits<std::size_t>::max() ? 0u : samplers[patch.sampler]->Slot()};
        const auto raw = std::as_bytes(std::span(slot));
        if (patch.pushByte >= 0) {
            for (std::size_t byte = 0; byte < raw.size(); ++byte) program.pushPatches.emplace_back(static_cast<std::uint32_t>(patch.pushByte) + static_cast<std::uint32_t>(byte), static_cast<std::uint32_t>(raw[byte]));
            continue;
        }
        Require(patch.dataAllocation >= 0, "a shader data patch in a shader without shader data");
        auto& data = allocations[static_cast<std::size_t>(patch.dataAllocation)];
        Require(patch.dataByte + raw.size() <= data.size, "a shader data patch lies outside the shader's data buffer");
        for (std::size_t byte = 0; byte < raw.size(); ++byte) {
            if (data.buffer != nullptr) data.buffer->Bytes()[patch.dataByte + byte] = raw[byte];
            program.dataPatches.push_back({static_cast<std::size_t>(patch.dataAllocation), patch.dataByte + static_cast<std::uint32_t>(byte), static_cast<std::uint32_t>(raw[byte])});
        }
    }
    if (!program.tableStages.empty()) {
        namespace Abi = ShaderRecompiler::ImageTableAbi;
        Require(context.deviceImages != nullptr, "a program reading images through the device image table needs the device's table");
        const auto header = context.deviceImages->HeaderAddress();
        for (auto& stage : program.tableStages) {
            auto& map = stage.map;
            map[Abi::HeaderDeviceLow] = static_cast<std::uint32_t>(header);
            map[Abi::HeaderDeviceHigh] = static_cast<std::uint32_t>(header >> 32u);
            if (stage.images) {
                for (std::size_t sampler = 0; sampler < stage.samplerElements.size(); ++sampler) {
                    std::uint32_t bits = 0u;
                    for (const auto element : stage.samplerElements[sampler]) {
                        Require(element < stage.samplerCount, "a device-table image is paired with sampler element " + std::to_string(element) + ", which its shader does not bind");
                        const auto& source = program.samplerSources.at(stage.firstSampler + element);
                        bits |= DeviceTableSamplerClass(samplers[stage.firstSampler + element].get(), source.words, source.unnormalized, source.integerBorder);
                    }
                    Require(Abi::HeaderWords + sampler < map.size(), "image table map has no room for its sampler classes");
                    map[Abi::HeaderWords + sampler] = bits;
                }
            }
            auto& allocation = allocations.at(stage.mapAllocation);
            Require(allocation.buffer != nullptr && allocation.size == map.size() * sizeof(std::uint32_t), "image table map buffer disagrees with the map");
            std::memcpy(allocation.buffer->Bytes().data(), map.data(), allocation.size);
            program.imageTableStages.push_back(IrStage(stage.stage));
        }
    }
    if (!program.dispatchSets && (program.sets[0] || program.sets[1])) {
        for (std::size_t set = 0; set < 2u; ++set) {
            if (!program.sets[set]) continue;
            auto& objects = set == 0 ? program.main : program.fragment;
            const auto allocated = AllocateSet(context, objects.layout, program.poolSizes[set], updateAfterBind);
            objects.set = allocated.set;
            objects.pool = allocated.pool;
            objects.cacheAllocation = allocated.cacheAllocation;
        }
        std::size_t bufferCount = 0;
        for (const auto& binding : bindings) bufferCount += binding.allocations.size();
        program.descriptorBuffers.reserve(bufferCount);
        for (const auto& binding : bindings) {
            if (binding.noDescriptor) continue;
            Require(binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, "AGC graphics: ShaderResources encountered an unknown descriptor type while writing the descriptor set");
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = program.setOf(binding) == 0 ? program.main.set : program.fragment.set;
            write.dstBinding = binding.layout.binding;
            write.descriptorCount = binding.layout.descriptorCount;
            write.descriptorType = binding.layout.descriptorType;
            write.pBufferInfo = program.descriptorBuffers.data() + program.descriptorBuffers.size();
            for (const auto index : binding.allocations) {
                const auto& allocation = allocations[index];
                if (allocation.role == ShaderRecompiler::DescriptorRole::FormatTable) program.descriptorBuffers.push_back({context.formatTable->Handle(), 0, RuntimeFormatTableBytes});
                else program.descriptorBuffers.push_back({allocation.buffer->Handle(), 0, allocation.size});
            }
            program.descriptorWrites.push_back(write);
        }
        if (!program.descriptorWrites.empty()) context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(program.descriptorWrites.size()), program.descriptorWrites.data(), 0, nullptr);
    }
    ResourceCounters().programs.fetch_add(1, std::memory_order_relaxed);
    return built;
}

void ShaderResources::buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes) {
    PerformanceTimer frameTiming("Resources.Prepare");
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    if (BuildProfiled()) {
        auto& profile = ThreadBuilds();
        if (++profile.builds % 1000 == 0) profile.merge();
    }
    try {
        shared = ProgramFor(context, shaders, colorAttachments);
        const auto& program = *shared;
        ResourceCounters().uses.fetch_add(1, std::memory_order_relaxed);
        for (const auto& allocation : program.allocations) {
            if (allocation.guest) guestMemory.AddInPlace(allocation.address, allocation.size, true);
        }
        if (!program.dispatchWords.empty()) {
            dispatchData.resize(program.allocations.size());
            for (const auto& words : program.dispatchWords) {
                const auto& shader = shaders[words.shader];
                const auto& source = shader.program->bindings.at(words.binding).guestDescriptor;
                ValidateRuntimeResources(shader, source);
                const auto size = source.size() * sizeof(std::uint32_t);
                Require(size == program.allocations[words.allocation].size, "a dispatch's shader data differs in size from its program's");
                auto buffer = std::make_unique<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                std::memcpy(buffer->Bytes().data(), source.data(), size);
                for (const auto& patch : program.dataPatches) {
                    if (patch.allocation == words.allocation) buffer->Bytes()[patch.byte] = static_cast<std::byte>(patch.adjustment);
                }
                dispatchData[words.allocation] = std::move(buffer);
                ResourceCounters().dispatchData.fetch_add(1, std::memory_order_relaxed);
            }
        }
        timing.bindingsMs = phase(BuildPhase::Bindings);
        guestMemory.UploadPrepare();
        timing.uploadMs = phase(BuildPhase::Upload);
        if (program.dispatchSets) {
            if (program.sets[0]) {
                const auto allocated = AllocateSet(context, program.main.layout, program.poolSizes[0], program.updateAfterBind);
                _set = allocated.set;
                pool = allocated.pool;
                cacheAllocation = allocated.cacheAllocation;
            }
            if (program.sets[1]) {
                const auto allocated = AllocateSet(context, program.fragment.layout, program.poolSizes[1], program.updateAfterBind);
                fragment.set = allocated.set;
                fragment.pool = allocated.pool;
                fragment.cacheAllocation = allocated.cacheAllocation;
            }
            ResourceCounters().dispatchSets.fetch_add(1, std::memory_order_relaxed);
        }
        timing.descriptorsMs = phase(BuildPhase::Descriptors);
        if (BuildProfiled()) {
            timing.prepareMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageA, timing.prepareMs);
        }
    } catch (...) {
        release();
        throw;
    }
}

bool SplitsSets(const Context& context, std::span<const CompiledShader> shaders) {
    const auto fragmentSet = [](const CompiledShader& shader) { return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const auto& binding) { return binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet; }); };
    const auto fragmentStage = [](const CompiledShader& shader) { return shader.stage == ShaderRecompiler::ShaderStage::Fragment; };
    return std::any_of(shaders.begin(), shaders.end(), fragmentSet) || (context.graphicsPipelineLibrary && std::any_of(shaders.begin(), shaders.end(), fragmentStage));
}

DescribedSetLayouts DescribeSetLayouts(const Context& context, std::span<const CompiledShader> shaders) {
    Require(context.descriptorCache != nullptr, "described set layouts need the descriptor cache");
    DescribedSetLayouts result;
    std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> descriptions;
    const bool split = SplitsSets(context, shaders);
    std::array<std::vector<VkDescriptorBindingFlags>, 2> bindingFlags;
    bool updateAfterBind = false;
    std::uint64_t storageBuffers = 0;
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        const VkShaderStageFlags flags = VulkanStage(shader.stage);
        std::uint64_t stageStorageBuffers = 0;
        std::uint64_t stageResources = 0;
        for (const auto& binding : shader.program->bindings) {
            const auto set = binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet ? 1 : 0;
            if (binding.kind == ShaderRecompiler::DescriptorKind::DeviceAddress || binding.kind == ShaderRecompiler::DescriptorKind::SamplerSlot) continue;
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable) continue;
            const auto type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            stageResources += binding.count;
            if (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                stageStorageBuffers += binding.count;
                storageBuffers += binding.count;
            }
            if (stageStorageBuffers > context.limits.maxPerStageDescriptorStorageBuffers || stageResources > context.limits.maxPerStageResources) updateAfterBind = true;
            descriptions[set].push_back({binding.binding, type, binding.count, flags, nullptr});
            bindingFlags[set].push_back(0);
        }
    }
    if (storageBuffers > context.limits.maxDescriptorSetStorageBuffers) updateAfterBind = true;
    for (std::size_t set = 0; set < descriptions.size(); ++set) {
        for (std::size_t index = 0; index < descriptions[set].size(); ++index) {
            if (updateAfterBind && descriptions[set][index].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) bindingFlags[set][index] |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        }
    }
    for (std::size_t set = 0; set < descriptions.size(); ++set) {
        auto& key = set == 0 ? result.mainKey : result.fragmentKey;
        for (const auto& binding : descriptions[set]) key.insert(key.end(), {binding.binding, static_cast<std::uint32_t>(binding.descriptorType), binding.descriptorCount, binding.stageFlags});
        if (updateAfterBind && (set == 0 || split)) key.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
    }
    result.layouts.split = split;
    result.layouts.main = context.descriptorCache->Layout(result.mainKey, descriptions[0], updateAfterBind, bindingFlags[0]);
    if (split) result.layouts.fragment = context.descriptorCache->Layout(result.fragmentKey, descriptions[1], updateAfterBind, bindingFlags[1]);
    return result;
}

void ShaderResources::buildComplete() {
    PerformanceTimer frameTiming("Resources.Complete");
    Require(!completed, "cannot update a published descriptor heap");
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    try {
        const auto& program = *shared;
        timing.bindingsMs += phase(BuildPhase::Images);
        guestMemory.UploadFinish();
        timing.uploadMs += phase(BuildPhase::Upload);
        if (usesBda) {
            auto* residency = ResidencyDriver::Active();
            if (residency != nullptr) {
                residency->BeforeDispatch();
                residency->UseForRead(readRanges);
            }
            if (residency != nullptr && pageTableStores && (storesUnbounded || !storeRanges.empty())) {
                std::vector<ResidencyRange> grants;
                if (!storesUnbounded) {
                    for (const auto& [begin, end] : storeRanges) grants.push_back({begin, end - begin});
                }
                residency->BeforeWritingDispatch(grants, storesUnbounded);
            }
            bda = std::make_unique<BdaResources>(context, guestMemory, std::span<const ShaderRecompiler::BdaAbi::Range>{}, batchFaultRecord());
            if (residency != nullptr && bda->PageTable() != nullptr) residency->CountMixedRanges(*bda->PageTable(), readRanges);
        } else if (usesFaultBuffer) {
            bda = std::make_unique<BdaResources>(context);
        }
        if (bda != nullptr) bda->NameStages(stageShaders);
        if (bda != nullptr && !program.imageTableStages.empty()) bda->SetImageTables(std::vector<std::uint32_t>(program.imageTableStages));
        phase(BuildPhase::Bda);
        if (program.dispatchSets && (_set != VK_NULL_HANDLE || fragment.set != VK_NULL_HANDLE)) {
            // One update call for the whole set: the info arrays are sized up front so every write's
            // pointer into them stays valid until the call.
            std::size_t bufferCount = 0;
            for (const auto& binding : program.bindings) bufferCount += binding.allocations.size();
            auto& buffers = descriptorBuffers;
            buffers.reserve(bufferCount);
            auto& writes = descriptorWrites;
            writes.reserve(program.bindings.size());
            for (const auto& binding : program.bindings) {
                if (binding.noDescriptor) continue;
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = program.setOf(binding) == 0 ? _set : fragment.set;
                write.dstBinding = binding.layout.binding;
                write.descriptorCount = binding.layout.descriptorCount;
                write.descriptorType = binding.layout.descriptorType;
                switch (binding.layout.descriptorType) {
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        write.pBufferInfo = buffers.data() + buffers.size();
                        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
                            const auto index = binding.allocations[element];
                            if (program.allocations[index].role == ShaderRecompiler::DescriptorRole::FaultBuffer) faultDescriptors.push_back({write.dstSet, write.dstBinding, static_cast<std::uint32_t>(element)});
                            buffers.push_back(descriptor(index));
                        }
                        break;
                    default: throw std::runtime_error("AGC graphics: ShaderResources encountered an unknown descriptor type while writing the descriptor set");
                }
                writes.push_back(write);
            }
            context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
        timing.descriptorsMs += phase(BuildPhase::Descriptors);
        completed = true;
        if (BuildProfiled()) {
            timing.completeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageB, timing.completeMs);
            reportDescriptorCaches();
        }
    } catch (...) {
        release();
        throw;
    }
}

// APS5_PROFILE_DRAW: the per-device descriptor caches' counters, every 10 s.
void ShaderResources::reportDescriptorCaches() const {
    static std::mutex reportMutex;
    static auto lastReport = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(reportMutex);
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    const auto descriptors = context.descriptorCache != nullptr ? context.descriptorCache->Counters() : DescriptorCache::Stats{};
    const auto samplerHits = context.samplerCache != nullptr ? context.samplerCache->Hits() : 0;
    const auto samplerMisses = context.samplerCache != nullptr ? context.samplerCache->Misses() : 0;
    AgcDriver::ProfilePrint_nid_no_patch("[descriptors] layouts %llu hits / %llu created, sets %llu from %llu pools and %llu dedicated pools, samplers %llu hits / %llu created\n", static_cast<unsigned long long>(descriptors.layoutHits), static_cast<unsigned long long>(descriptors.layoutMisses), static_cast<unsigned long long>(descriptors.sets), static_cast<unsigned long long>(descriptors.pools), static_cast<unsigned long long>(descriptors.dedicatedPools), static_cast<unsigned long long>(samplerHits), static_cast<unsigned long long>(samplerMisses));
}

void CompletePendingCpuWrites(const Context& context, std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || Recorder::InCompletion()) return;
    auto* recorder = Recorder::Active();
    if (recorder == nullptr) return;
    if (Recorder::PendingCompletionLabels() == 0) return;
    if (!recorder->PendingWriteOverlaps(address, bytes)) return;
    if (recorder->LabelStoredAtCompletionIn(address, bytes)) {
        Recorder::CountSync(4);
        recorder->SyncThrough(address, bytes);
    }
}

DescriptorCache::DescriptorCache(const Context& context) : context(context), destroyLayout(context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")), destroyPool(context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")), resetPool(context.Function<PFN_vkResetDescriptorPool>("vkResetDescriptorPool")) {
    programs = std::make_unique<ProgramResourceCache>();
}

DescriptorCache::~DescriptorCache() {
    programs.reset();
    for (const auto& pool : pools) destroyPool(context.device, pool.handle, nullptr);
    for (const auto& pool : updateAfterBindPools) destroyPool(context.device, pool.handle, nullptr);
    for (const auto pool : dedicated) destroyPool(context.device, pool, nullptr);
    for (const auto& [key, layout] : layouts) destroyLayout(context.device, layout, nullptr);
}

VkDescriptorSetLayout DescriptorCache::Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind, std::span<const VkDescriptorBindingFlags> flags) {
    std::vector<VkDescriptorBindingFlags> defaults;
    if (flags.empty()) {
        for (const auto& binding : bindings) defaults.push_back(updateAfterBind && binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0);
        flags = defaults;
    }
    Require(flags.size() == bindings.size(), "descriptor binding flag count differs from layout");
    std::lock_guard lock(mutex);
    std::vector<std::uint32_t> keyCopy(key.begin(), key.end());
    keyCopy.push_back(static_cast<std::uint32_t>(flags.size()));
    keyCopy.insert(keyCopy.end(), flags.begin(), flags.end());
    if (const auto found = layouts.find(keyCopy); found != layouts.end()) {
        ++stats.layoutHits;
        return found->second;
    }
    ValidateDescriptorLayout(context, bindings, updateAfterBind);
    ++stats.layoutMisses;
    const auto layout = CreateSetLayout(context, bindings, updateAfterBind, flags);
    layouts.emplace(std::move(keyCopy), layout);
    return layout;
}

namespace {
std::uint32_t SetDescriptorLimit(const VkPhysicalDeviceLimits& limits, VkDescriptorType type) {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return limits.maxDescriptorSetStorageBuffers;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return limits.maxDescriptorSetSampledImages;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return limits.maxDescriptorSetStorageImages;
        case VK_DESCRIPTOR_TYPE_SAMPLER: return limits.maxDescriptorSetSamplers;
        default: throw std::runtime_error("AGC graphics: descriptor set uses an unsupported descriptor type " + std::to_string(type));
    }
}

constexpr std::uint32_t ChainPoolSets = 1024;
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolCreateSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 + 2}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};
}

DescriptorCache::SetAllocation DescriptorCache::Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind, std::optional<std::uint32_t> variableCount) {
    bool fits = true;
    for (const auto& size : sizes) {
        const auto limit = updateAfterBind && size.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindStorageBuffers : SetDescriptorLimit(context.limits, size.type);
        Require(size.descriptorCount <= limit, "descriptor set exceeds the device's per-set descriptor limit");
        const auto capacity = std::find_if(ChainPoolSizes.begin(), ChainPoolSizes.end(), [&](const auto& item) { return item.type == size.type; });
        if (capacity == ChainPoolSizes.end() || size.descriptorCount > capacity->descriptorCount) fits = false;
    }
    std::lock_guard lock(mutex);
    const auto allocate = context.Resolved(&DeviceFunctions::allocateDescriptorSets, "vkAllocateDescriptorSets");
    const std::uint32_t variable = variableCount.value_or(0u);
    VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
    variableInfo.descriptorSetCount = 1;
    variableInfo.pDescriptorCounts = &variable;
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.pNext = variableCount.has_value() ? &variableInfo : nullptr;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &layout;
    if (!fits) {
        dedicated.reserve(dedicated.size() + 1);
        const auto pool = CreateSetPool(context, sizes, updateAfterBind);
        allocation.descriptorPool = pool;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result != VK_SUCCESS) {
            destroyPool(context.device, pool, nullptr);
            Check(result, "vkAllocateDescriptorSets");
        }
        dedicated.push_back(pool);
        ++stats.dedicatedPools;
        ++stats.sets;
        return {set, pool};
    }
    auto& chain = updateAfterBind ? updateAfterBindPools : pools;
    for (std::size_t index = chain.size(); index != 0; --index) {
        auto& pool = chain[index - 1];
        if (pool.exhausted) continue;
        allocation.descriptorPool = pool.handle;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result == VK_SUCCESS) {
            ++stats.sets;
            ++pool.liveSets;
            return {set, pool.handle, index - 1, updateAfterBind};
        }
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL) Check(result, "vkAllocateDescriptorSets");
        pool.exhausted = true;
    }
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = updateAfterBind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
    poolInfo.maxSets = ChainPoolSets;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(ChainPoolCreateSizes.size());
    poolInfo.pPoolSizes = ChainPoolCreateSizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
    chain.push_back({pool});
    ++stats.pools;
    allocation.descriptorPool = pool;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Check(allocate(context.device, &allocation, &set), "vkAllocateDescriptorSets");
    chain.back().liveSets = 1;
    ++stats.sets;
    return {set, pool, chain.size() - 1, updateAfterBind};
}

void DescriptorCache::Free(const SetAllocation& allocation) noexcept {
    if (allocation.set == VK_NULL_HANDLE || allocation.pool == VK_NULL_HANDLE) return;
    std::lock_guard lock(mutex);
    if (const auto found = std::find(dedicated.begin(), dedicated.end(), allocation.pool); found != dedicated.end()) {
        dedicated.erase(found);
        destroyPool(context.device, allocation.pool, nullptr);
        return;
    }
    auto& chain = allocation.updateAfterBind ? updateAfterBindPools : pools;
    if (allocation.poolIndex >= chain.size()) std::terminate();
    auto& pool = chain[allocation.poolIndex];
    if (pool.handle != allocation.pool || pool.liveSets == 0) std::terminate();
    if (--pool.liveSets != 0) return;
    if (resetPool(context.device, pool.handle, 0) != VK_SUCCESS) std::terminate();
    pool.exhausted = false;
}

DescriptorCache::Stats DescriptorCache::Counters() const {
    std::lock_guard lock(mutex);
    return stats;
}


std::string ShaderResources::Describe() const {
    const auto sample = [](std::uint64_t address, std::uint64_t bytes) {
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes))) return -1.0;
        std::size_t nonzero = 0;
        std::size_t samples = 0;
        const auto* data = reinterpret_cast<const std::uint32_t*>(address);
        const auto words = bytes / 4;
        const auto step = std::max<std::uint64_t>(1, words / 4096);
        for (std::uint64_t i = 0; i < words; i += step, ++samples) nonzero += data[i] != 0;
        return samples == 0 ? 0.0 : static_cast<double>(nonzero) / samples;
    };
    std::string text;
    char line[160];
    for (const auto& range : describedRanges) {
        std::snprintf(line, sizeof(line), " %s 0x%llx+0x%llx(%ux%u f%u t%d) nz=%.2f", range.kind, static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(range.bytes), range.width, range.height, range.format, range.tileMode, sample(range.address, range.bytes));
        text += line;
        if (range.dccAddress != 0) {
            const auto keys = ReadDccKeys(range.dccAddress, range.bytes);
            std::snprintf(line, sizeof(line), " dcc=%s@0x%llx", DccKeysName(keys), static_cast<unsigned long long>(range.dccAddress));
            text += line;
            if (keys == DccKeys::Mixed) {
                const auto* bytes = reinterpret_cast<const std::uint8_t*>(range.dccAddress);
                const auto count = static_cast<std::size_t>(range.bytes / 256u);
                std::size_t run = 1;
                while (run < count && bytes[run] == bytes[0]) ++run;
                std::snprintf(line, sizeof(line), "(%02x x%zu then %02x of %zu)", bytes[0], run, run < count ? bytes[run] : 0u, count);
                text += line;
            }
        }
    }
    static const bool words = [] { const char* value = std::getenv("APS5_TRACE_DISPATCH_IO"); return value != nullptr && value[0] == '2'; }();
    const auto appendWords = [&](const std::uint32_t* data, std::size_t bytes) {
        if (!words || bytes > 0x200) return;
        text += " [";
        for (std::size_t i = 0; i < bytes / 4; ++i) {
            char word[12];
            std::snprintf(word, sizeof(word), "%s%08x", i == 0 ? "" : " ", data[i]);
            text += word;
        }
        text += "]";
    };
    for (std::size_t index = 0; index < shared->allocations.size(); ++index) {
        const auto& allocation = shared->allocations[index];
        if (allocation.guest) {
            std::snprintf(line, sizeof(line), " buffer%s 0x%llx+0x%zx nz=%.2f", allocation.written ? "" : "(ro)", static_cast<unsigned long long>(allocation.address), allocation.size, sample(allocation.address, allocation.size));
            text += line;
            if (GuestMemory::Accessible(reinterpret_cast<const void*>(allocation.address), allocation.size)) appendWords(reinterpret_cast<const std::uint32_t*>(allocation.address), allocation.size);
        } else if (auto* buffer = dataBuffer(index); buffer != nullptr) {
            const auto bytes = buffer->Bytes();
            std::snprintf(line, sizeof(line), " data+0x%zx nz=%.2f", allocation.size, sample(reinterpret_cast<std::uint64_t>(bytes.data()), allocation.size));
            text += line;
            appendWords(reinterpret_cast<const std::uint32_t*>(bytes.data()), allocation.size);
        }
    }
    return text;
}

namespace {

bool SameKeySurface(const StorageTexture* source, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    if (source == nullptr) return false;
    const auto& own = source->Descriptor();
    return own.dccAddress == resource.dccAddress && source->GuestBytes() == guestBytes && own.format == resource.format && own.dccAlphaOnMsb == resource.dccAlphaOnMsb;
}

}

SampledTextureCounts SampledTextureSources() {
    const auto& counters = TextureCounts();
    return {counters.gathers.load(std::memory_order_relaxed), counters.clears.load(std::memory_order_relaxed)};
}

DeviceTableTexture LookupDeviceTableTexture(const Context& context, std::span<const std::uint32_t, 8> descriptor, const DeviceTableView& view, std::uint32_t samplerClass, bool perUse) {
    namespace Abi = ShaderRecompiler::DeviceImageTableAbi;
    std::array<std::uint32_t, 8> words{};
    std::copy(descriptor.begin(), descriptor.end(), words.begin());
    if (view.identitySwizzle) words[3] = (words[3] & ~0xfffu) | ShaderRecompiler::ShaderImageIdentitySwizzle;
    const auto resource = DecodeTextureResource(words);
    const bool firstLayer = view.shape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
    if (!firstLayer && !MatchesGuestDimension(view.shape, resource.dimension)) throw std::runtime_error("AGC graphics: guest texture dimension disagrees with the table entry's image shape (shape " + std::to_string(static_cast<int>(view.shape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
    const auto guestBytes = DescribeSurface(resource).guestBytes;
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(resource.baseAddress), static_cast<std::size_t>(guestBytes)) && GuestMemory::CommittedRanges(resource.baseAddress, static_cast<std::size_t>(guestBytes), false).empty()) throw std::runtime_error("AGC graphics: the table entry's surface at 0x" + HexText(resource.baseAddress) + " (0x" + HexText(guestBytes) + " bytes) is not mapped");
    static_cast<void>(ResolveTextureFormat(resource.format));
    CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
    if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
    auto texture = cachedTexture(context, words, resource, SampledViewComponents(resource, view.decodedCopy), guestBytes, view.depthCompare, view.decodedCopy);
    const auto& log = lookupLog();
    const auto* looked = !log.empty() && log.back().object == texture.get() ? &log.back() : nullptr;
    const bool depth = looked != nullptr && looked->depth;
    const bool copied = texture->CopyKind() != Texture::PerUseCopy::None;
    if (depth && !perUse) throw std::runtime_error("AGC graphics: the table entry's surface at 0x" + HexText(resource.baseAddress) + " is a depth surface, which the device image table does not serve: a dispatch sampling it through the table would need the depth image's layout transitions");
    if (copied && !perUse) throw std::runtime_error("AGC graphics: the table entry's texture at 0x" + HexText(resource.baseAddress) + " is a copy remade per use, which the device image table does not serve");
    if ((samplerClass & Abi::SamplerFilterMinmax) != 0u) {
        VkFormatProperties properties{};
        context.formatProperties(context.physical, texture->ViewFormat(), &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT_EXT) == 0) throw std::runtime_error("a sampled texture whose format " + std::to_string(texture->ViewFormat()) + " does not support min/max filtering is sampled through a min or max reduction sampler with linear filtering");
    }
    if ((samplerClass & Abi::SamplerOpaqueBlackBorder) != 0u && resource.bcSwizzle != 0u && resource.bcSwizzle != 4u) throw std::runtime_error("a sampled texture whose BC swizzle " + std::to_string(resource.bcSwizzle) + " moves the alpha channel is sampled through an opaque black border, which is not implemented");
    if ((samplerClass & Abi::SamplerNonSeamlessCube) != 0u && resource.dimension == TextureDimension::kCube) throw std::runtime_error("a cube texture is sampled through a sampler that disables seamless cube filtering, which needs VK_EXT_non_seamless_cube_map");
    const bool integerImage = view.integerView;
    if (((samplerClass & Abi::SamplerFloatBorder) != 0u && integerImage) || ((samplerClass & Abi::SamplerIntegerBorder) != 0u && !integerImage)) throw std::runtime_error("an image table entry's sampler border color numeric class disagrees with its sampled image class, which needs dynamic pairing and is not implemented");
    if ((samplerClass & Abi::SamplerAnisoOverride) != 0u) throw std::runtime_error("an image table entry paired with anisotropic ANISO_OVERRIDE needs dynamic texture mip-level pairing, which is not implemented");
    bool baseLevel = false;
    if (view.unnormalized) {
        const auto range = texture->SampledViewRange(firstLayer);
        const bool singleLevel = range.levels == 1u && resource.baseLevel == 0u;
        const bool supported = range.layers == 1u && EffectiveMinLod(resource) == 0.0f && (singleLevel || view.exactAtBase);
        if (!supported || (range.type != VK_IMAGE_VIEW_TYPE_1D && range.type != VK_IMAGE_VIEW_TYPE_2D)) throw std::runtime_error("AGC graphics: guest texture sampled with unnormalized coordinates is not a single-layer 1D or 2D view without a minimum LOD clamp, and is neither a single-level view nor read at its base level only");
        baseLevel = range.levels > 1u;
    }
    if (view.srgb && !texture->HasSrgbView()) throw std::runtime_error("AGC graphics: a texture of guest format " + std::to_string(resource.format) + ", which is not sRGB and has no sRGB view, is sampled through a sampler that forces sRGB decoding, which is not implemented");
    DeviceTableTexture result;
    const auto which = baseLevel ? Texture::SampledView::BaseLevel : firstLayer ? Texture::SampledView::FirstLayer : Texture::SampledView::Whole;
    result.slot = view.srgb ? texture->SrgbSampledSlot(which) : texture->SampledSlot(which);
    result.address = resource.baseAddress;
    result.guestBytes = guestBytes;
    result.snapshot = looked != nullptr && looked->source == nullptr && looked->keys == DccKeys::Uncompressed && resource.dccAddress == 0 && looked->generation != 0;
    result.perUse = depth || copied;
    result.generation = result.snapshot ? looked->generation : GuestMemory::CollectWrites(resource.baseAddress, static_cast<std::size_t>(guestBytes));
    result.texture = std::move(texture);
    return result;
}

DeviceTableStorage LookupDeviceTableStorage(const Context& context, std::span<const std::uint32_t, 8> descriptor, const DeviceTableView& view) {
    std::array<std::uint32_t, 8> words{};
    std::copy(descriptor.begin(), descriptor.end(), words.begin());
    const auto resource = DecodeTextureResource(words);
    const bool firstLayer = view.shape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
    if (!firstLayer && !MatchesGuestDimension(view.shape, resource.dimension)) throw std::runtime_error("AGC graphics: guest storage texture dimension disagrees with the table entry's image shape (shape " + std::to_string(static_cast<int>(view.shape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
    const auto mip = std::min(resource.baseLevel, resource.mipCount - 1u);
    Require(resource.minLod <= mip * 256u, "guest storage texture descriptor clamps its minimum LOD above the level it addresses, which is not implemented");
    const auto guestBytes = DescribeSurface(resource).guestBytes;
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(resource.baseAddress), static_cast<std::size_t>(guestBytes)) && GuestMemory::CommittedRanges(resource.baseAddress, static_cast<std::size_t>(guestBytes), false).empty()) throw std::runtime_error("AGC graphics: the table entry's storage surface at 0x" + HexText(resource.baseAddress) + " is not mapped");
    CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
    if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
    auto image = cachedStorageTexture(context, words, resource, mip, guestBytes, view.written);
    const auto layers = StorageTexture::LayersOf(resource, firstLayer);
    DeviceTableStorage result;
    if (view.mipRun) {
        Require(!view.atomic && !view.atomic64, "a storage image with dynamic mips is accessed atomically through the device image table, which is not implemented");
        Require(resource.lastLevel >= resource.baseLevel && resource.lastLevel - resource.baseLevel < ShaderRecompiler::RuntimeAbi::StorageMipSlots, "guest storage texture descriptor names an invalid range of mip levels");
        result.slot = image->StorageSlotRun(resource.baseLevel, resource.lastLevel, layers);
    } else {
        result.slot = image->StorageSlot(mip, layers, view.atomic64 ? StorageTexture::StorageViewKind::Atomic64 : view.atomic ? StorageTexture::StorageViewKind::Atomic : StorageTexture::StorageViewKind::Storage);
    }
    result.address = resource.baseAddress;
    result.guestBytes = guestBytes;
    result.image = std::move(image);
    return result;
}

std::uint32_t DeviceTableSamplerClass(const Sampler* sampler, std::span<const std::uint32_t, 4> words, bool unnormalized, bool integerBorder) {
    namespace Abi = ShaderRecompiler::DeviceImageTableAbi;
    if (sampler == nullptr) return 0u;
    std::uint32_t bits = 0u;
    if (sampler->RequiresFilterMinmax()) bits |= Abi::SamplerFilterMinmax;
    if (sampler->ReadsOpaqueBlackBorder()) bits |= Abi::SamplerOpaqueBlackBorder;
    if (sampler->RequiresNonSeamlessCube()) bits |= Abi::SamplerNonSeamlessCube;
    const auto descriptor = DecodeSamplerResource(words, unnormalized, true, integerBorder);
    const bool border = descriptor.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || descriptor.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || descriptor.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    if (border) bits |= integerBorder ? Abi::SamplerIntegerBorder : Abi::SamplerFloatBorder;
    if (SingleLevelSamplerWords(words, true, false).has_value()) bits |= Abi::SamplerAnisoOverride;
    if (ShaderRecompiler::ReducesBetweenTexels(words[0], words[2])) bits |= Abi::SamplerReduction;
    return bits;
}

void ShaderResources::ExpectDeviceImages(DeviceImageTable& table, ShaderRecompiler::ShaderStage stageOf, const ShaderRecompiler::RecompileResult& program) const {
    namespace Abi = ShaderRecompiler::ImageTableAbi;
    namespace Keys = ShaderRecompiler::DeviceImageTableAbi;
    const auto& expected = program.deviceImages;
    if (std::ranges::none_of(expected, [](const Keys::Expectation& image) { return image.mapSampler != Keys::NoMapSampler; })) {
        table.Expect(expected, program.deviceImagesComplete);
        return;
    }
    const auto& tableStages = shared->tableStages;
    const auto stage = std::ranges::find_if(tableStages, [&](const TableStage& candidate) { return candidate.stage == stageOf; });
    Require(stage != tableStages.end() && stage->map.size() >= Abi::HeaderWords, "a device image expectation names a sampler class, but its stage has no image table map with sampler classes");
    const auto& map = stage->map;
    auto keys = expected;
    for (auto& image : keys) {
        if (image.mapSampler == Keys::NoMapSampler) continue;
        const auto at = static_cast<std::size_t>(Abi::HeaderWords) + image.mapSampler;
        Require(at < map.size(), "a device image expectation names sampler " + std::to_string(image.mapSampler) + ", which the image table map has no class for");
        image.key[Keys::SamplerClassWord] = map[at];
    }
    table.Expect(keys, program.deviceImagesComplete);
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::PresyncSurfaces() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> surfaces;
    // A surface's lookup reads guest memory on the CPU unless it is served GPU-direct from a host
    // import: a storage image of imported memory uploads and refreshes from the import, and a
    // sampled texture of one views that image (SampledFromStorageEligible), so neither waits for
    // the producer.
    const auto consider = [&](std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes, bool sampled) {
        const bool gpuDirect = sampled ? SampledFromStorageEligible(context, format, address, guestBytes) : HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
        if (!gpuDirect) surfaces.emplace_back(address, guestBytes);
    };
    if (completed) {
        std::size_t textureIndex = 0;
        for (const auto& range : describedRanges) {
            if (std::strcmp(range.kind, "texture") == 0 && textureIndex < textures.size()) {
                const auto& texture = textures[textureIndex++];
                if (texture->ViewsStorageImage()) consider(range.format, range.address, range.bytes, false);
                else consider(range.format, range.address, range.bytes, true);
            }
        }
        return surfaces;
    }
    return surfaces;
}

ShaderResources::~ShaderResources() {
    release();
}

void ShaderResources::release() noexcept {
    // A set from the cache's pool chain goes back to it; a dedicated pool dies with its set.
    if (cacheAllocation.pool && context.descriptorCache != nullptr) context.descriptorCache->Free(cacheAllocation);
    if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (fragment.cacheAllocation.pool && context.descriptorCache != nullptr) context.descriptorCache->Free(fragment.cacheAllocation);
    if (fragment.pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, fragment.pool, nullptr);
    cacheAllocation = {};
    pool = VK_NULL_HANDLE;
    _set = VK_NULL_HANDLE;
    fragment = {};
}

VkDescriptorSetLayout ShaderResources::Layout() const {
    return shared != nullptr ? shared->main.layout : VK_NULL_HANDLE;
}

std::vector<VkDescriptorSetLayoutBinding> ShaderResources::LayoutBindings() const {
    std::vector<VkDescriptorSetLayoutBinding> result;
    for (const auto& binding : shared->bindings) {
        if (!binding.noDescriptor) result.push_back(binding.layout);
    }
    return result;
}

bool ShaderResources::LayoutUpdateAfterBind() const {
    return shared->updateAfterBind;
}

std::span<const VkDescriptorBindingFlags> ShaderResources::LayoutBindingFlags() const {
    return shared->layoutBindingFlags;
}

VkDescriptorSetLayout ShaderResources::FragmentLayout() const {
    return shared->fragment.layout;
}

bool ShaderResources::SplitSets() const {
    return shared->splitSets;
}

PipelineSetLayouts ShaderResources::SetLayouts() const {
    return {shared->main.layout, shared->fragment.layout, shared->splitSets, shared->layoutKey, shared->fragmentLayoutKey};
}

const std::vector<std::uint32_t>& ShaderResources::LayoutKey() const {
    return shared->layoutKey;
}

const std::vector<std::uint32_t>& ShaderResources::FragmentLayoutKey() const {
    return shared->fragmentLayoutKey;
}

void ShaderResources::PatchPushConstants(std::span<std::byte, PipelinePushConstantBytes> bytes) const {
    for (const auto& [position, adjustment] : shared->pushPatches) bytes[position] = static_cast<std::byte>(adjustment);
    for (const auto& [position, adjustment] : pushPatches) bytes[position] = static_cast<std::byte>(adjustment);
}

VkDescriptorBufferInfo ShaderResources::descriptor(std::size_t index) {
    const auto& allocation = shared->allocations[index];
    Require(!allocation.deviceAddressed, "a guest buffer read through its device address has no descriptor");
    if (allocation.guest) {
        std::uint32_t adjustment = 0;
        const auto info = guestMemory.Descriptor(allocation.address, allocation.size, adjustment);
        Require(adjustment == 0, "a guest buffer off the storage buffer offset alignment in a shader without shader data is not implemented");
        return info;
    }
    if (allocation.role == ShaderRecompiler::DescriptorRole::BdaPagetable || allocation.role == ShaderRecompiler::DescriptorRole::FaultBuffer) {
        Require(bda != nullptr, "BDA descriptors have no memory owner");
        return allocation.role == ShaderRecompiler::DescriptorRole::BdaPagetable ? bda->Table() : bda->Fault();
    }
    if (allocation.role == ShaderRecompiler::DescriptorRole::FormatTable) return {context.formatTable->Handle(), 0, RuntimeFormatTableBytes};
    const auto* buffer = dataBuffer(index);
    Require(buffer != nullptr, "shader data has no buffer owner");
    return {buffer->Handle(), 0, allocation.size};
}

Buffer* ShaderResources::dataBuffer(std::size_t index) const {
    if (index < dispatchData.size() && dispatchData[index] != nullptr) return dispatchData[index].get();
    return shared->allocations[index].buffer.get();
}

VkDescriptorSet ShaderResources::mainSet() const {
    return shared->dispatchSets ? _set : shared->main.set;
}

VkDescriptorSet ShaderResources::fragmentSet() const {
    return shared->dispatchSets ? fragment.set : shared->fragment.set;
}

void ShaderResources::Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const {
    const auto main = mainSet();
    const auto second = fragmentSet();
    if (main != VK_NULL_HANDLE) context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, layout, 0, 1, &main, 0, nullptr);
    if (second != VK_NULL_HANDLE) context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, layout, ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet, 1, &second, 0, nullptr);
}

void ShaderResources::WriteBack() {
    WriteBackBuffers();
}

void ShaderResources::MarkGpuWrites(Recorder& recorder) {
    // The ranges this use reads in place through their host imports (read-only and written elements
    // alike, and an address-based build's whole leased heaps), before the writes: a CPU store into
    // one of them (the copy HLE) must not land before the recorded work read it.
    recorder.NotePendingReads(MemoryReads(), MemoryReadsUnknown() ? Recorder::ReadKind::AddressBased : Recorder::ReadKind::DispatchElement);
    recorder.NotePendingWrites(guestMemory.Writes());
    if (pageTableStores && storesUnbounded) recorder.NotePendingWrite(0, std::numeric_limits<std::size_t>::max());
    else if (pageTableStores) {
        for (const auto& [begin, end] : storeRanges) recorder.NotePendingWrite(begin, static_cast<std::size_t>(end - begin));
    }
    guestMemory.MarkDirectWrites();
}

std::optional<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::linearStoreRange() const {
    if (!linearStore.has_value() || linearThreads == 0 || linearStore->stride == 0) return std::nullopt;
    const auto& linear = *linearStore;
    const auto elements = std::min<std::uint64_t>(linearThreads, linear.records);
    if (elements == 0 || elements > std::numeric_limits<std::uint64_t>::max() / linear.stride) return std::nullopt;
    const auto bytes = elements * linear.stride + (linear.elementBytes > linear.stride ? linear.elementBytes - linear.stride : 0u);
    if (linear.base > std::numeric_limits<std::uint64_t>::max() - bytes) return std::nullopt;
    return std::pair{linear.base, linear.base + bytes};
}

void ShaderResources::checkBdaFault() const {
    try {
        bda->CheckFault();
    } catch (const std::runtime_error& error) {
        if (program == 0) throw;
        throw std::runtime_error(std::string(error.what()) + " (program 0x" + HexText(program) + ")");
    }
}

void ShaderResources::CheckFaults() const {
    if (bda) checkBdaFault();
}

void ShaderResources::WriteBackBuffers() {
    if (bda) checkBdaFault();
    guestMemory.Release();
}

std::shared_ptr<BatchFaultRecord> ShaderResources::batchFaultRecord() {
    namespace Abi = ShaderRecompiler::BdaAbi;
    auto* recorder = Recorder::Active();
    if (recorder == nullptr || !faultChecks || !shared->tableStages.empty() || MemoryWritesUnknown()) return nullptr;
    auto writes = MemoryWrites();
    std::sort(writes.begin(), writes.end());
    std::uint64_t chunks = 0;
    std::uint64_t counted = 0;
    for (const auto& [begin, end] : writes) {
        if (end <= begin) continue;
        const auto first = std::max(begin >> Abi::WrittenChunkShift, counted);
        const auto last = ((end - 1u) >> Abi::WrittenChunkShift) + 1u;
        if (last > first) chunks += last - first;
        counted = std::max(counted, last);
        if (chunks > BatchFaultRecord::Capacity) return nullptr;
    }
    writtenChunks = static_cast<std::uint32_t>(chunks);
    return recorder->CallerFaultRecord(writtenChunks);
}

void ShaderResources::rebindFaultRecord(std::shared_ptr<BatchFaultRecord> record) {
    Require(!faultDescriptors.empty(), "a build's fault record has no descriptor to rebind");
    bda->UseRecord(context, std::move(record));
    const auto info = bda->Fault();
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(faultDescriptors.size());
    for (const auto& slot : faultDescriptors) {
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = slot.set;
        write.dstBinding = slot.binding;
        write.dstArrayElement = slot.element;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        writes.push_back(write);
    }
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void ShaderResources::UseBatchFaultRecord(Recorder& recorder) {
    if (bda == nullptr || bda->SharedRecord() == nullptr || recorder.CallerFaultRecordCurrent(*bda->SharedRecord())) return;
    rebindFaultRecord(recorder.CallerFaultRecord(writtenChunks));
}

void ShaderResources::UseOwnFaultRecord() {
    if (bda != nullptr && bda->SharedRecord() != nullptr) rebindFaultRecord(nullptr);
}

void ShaderResources::PrepareDeviceRecords(Recorder* recorder, VkCommandBuffer commands) {
    if (bda != nullptr) bda->PrepareUse(context, commands, recorder);
}

bool ShaderResources::NeedsDeviceTransfers(const Recorder& recorder) const {
    return bda != nullptr && bda->NeedsTransfers(recorder);
}

void ShaderResources::FinishDeviceRecords(VkCommandBuffer commands) const {
    if (bda != nullptr) bda->FinishUse(context, commands);
}

void ShaderResources::RecordBatchFaults(Recorder& recorder, const std::shared_ptr<ShaderResources>& resources) {
    if (resources->bda != nullptr && resources->bda->SharedRecord() != nullptr) Require(recorder.CallerFaultRecordCurrent(*resources->bda->SharedRecord()), "a build was recorded into a batch other than its fault record's");
    recorder.CallerBatchFaults().Add(resources);
}

void BatchFaults::Add(std::shared_ptr<ShaderResources> user) {
    users.push_back(std::move(user));
}

void BatchFaults::Complete() {
    const auto failed = [](const std::exception& error) { std::fprintf(stderr, "[gpu] deferred write-back failed: %s\n", error.what()); };
    std::vector<const BatchFaultRecord*> records;
    for (const auto& user : users) {
        const auto* record = user->bda != nullptr ? user->bda->SharedRecord().get() : nullptr;
        if (record != nullptr && std::find(records.begin(), records.end(), record) == records.end()) records.push_back(record);
    }
    for (const auto* record : records) {
        std::vector<BatchFaultUser> sharing;
        for (const auto& user : users) {
            if (user->bda == nullptr || user->bda->SharedRecord().get() != record) continue;
            sharing.push_back({user->bda.get(), user->program, user->MemoryReads(), user->MemoryWrites()});
        }
        try {
            record->Check(sharing);
        } catch (const std::exception& error) {
            failed(error);
        }
    }
    for (const auto& user : users) {
        try {
            user->WriteBackBuffers();
        } catch (const std::exception& error) {
            failed(error);
        }
    }
}

void ShaderResources::PatchDispatchThreadLimit(const ShaderRecompiler::RecompileResult& program, VkDeviceAddress address, std::span<std::byte, PipelinePushConstantBytes> push) {
    const std::array<std::uint32_t, 2> words{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)};
    const auto position = program.dispatchThreadLimitDword * 4u;
    if (!program.pushConstants.empty()) {
        Require(position + sizeof(words) <= program.pushConstants.size() && position + sizeof(words) <= push.size(), "a dispatch thread limit lies outside the shader's push constants");
        std::memcpy(push.data() + position, words.data(), sizeof(words));
        return;
    }
    patchShaderData(-1, shared->computeShaderData, position, words);
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::MemoryReads() const {
    if (!usesBda || MemoryReadsUnknown()) return guestMemory.InPlaceReads();
    auto reads = guestMemory.RegionReads();
    reads.insert(reads.end(), accessRanges.begin(), accessRanges.end());
    return reads;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::MemoryWrites() const {
    auto writes = guestMemory.Writes();
    if (pageTableStores && !storesUnbounded) writes.insert(writes.end(), storeRanges.begin(), storeRanges.end());
    return writes;
}

bool ShaderResources::ReadsOverlap(std::uint64_t address, std::size_t bytes) const {
    const auto reads = MemoryReads();
    return std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return address < range.second && range.first < address + bytes; });
}

bool ShaderResources::WritesOverlap(std::uint64_t address, std::size_t bytes) const {
    if (MemoryWritesUnknown() || guestMemory.WritesOverlap(address, bytes)) return true;
    return pageTableStores && std::any_of(storeRanges.begin(), storeRanges.end(), [&](const auto& range) { return address < range.second && range.first < address + bytes; });
}

std::vector<std::pair<VkImage, bool>> ShaderResources::StorageImages() const {
    std::vector<std::pair<VkImage, bool>> images;
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() != nullptr) images.emplace_back(texture->StorageSource()->Image(), false);
    }
    return images;
}

bool ShaderResources::HasCachedStorageImages() const {
    return std::none_of(textures.begin(), textures.end(), [](const auto& texture) { return texture != nullptr && texture->StorageSource() != nullptr && !texture->StorageSource()->Cached(); });
}

bool ShaderResources::ReadsImage(const StorageTexture* image) const {
    if (image == nullptr) return false;
    return std::any_of(textures.begin(), textures.end(), [&](const auto& texture) { return texture != nullptr && texture->StorageSource() == image; });
}

}
