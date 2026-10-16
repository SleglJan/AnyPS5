#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CommittedImports.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "ThreadOwned.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <array>
#include <atomic>
#include <tuple>
#include <utility>
#include <functional>
#include <bit>
#include <condition_variable>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <stop_token>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace AgcDriver::Graphics {

struct ImageMirror {
    std::uint64_t base = 0;
    std::uint64_t bytes = 0;
    std::weak_ptr<const GuestAllocations::Range> range;
    std::shared_ptr<Buffer> buffer;
    std::uint64_t serial = 0;
};

struct GuestBufferMemory::AddressSpace {
    std::uint64_t generation = 0;
    std::uint64_t importsEpoch = 0;
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t serial = 0;
    GuestAllocations::Lease lease;
    std::vector<Region> base;
    std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
    mutable std::mutex tableMutex;
    mutable std::shared_ptr<const BdaPageTable> pageTable;
    mutable std::vector<ShaderRecompiler::BdaAbi::Range> resident;
    mutable std::shared_ptr<const BdaPageTable> lastOverride;
    mutable std::uint64_t residencySerial = 0;
    struct OverrideTable {
        std::shared_ptr<const BdaPageTable> base;
        std::vector<ShaderRecompiler::BdaAbi::Range> overrides;
        std::shared_ptr<const BdaPageTable> table;
    };
    mutable std::array<OverrideTable, 16> overrideTables;
    mutable std::size_t nextOverrideTable = 0;
    mutable std::uint64_t stamped = 0;
    std::shared_ptr<const void> retired;
    mutable std::shared_ptr<const BdaPageTable> inherited;
};

namespace {

struct RetiredImport;

struct LeasedRange {
    std::uint64_t address;
    std::uint64_t bytes;
    std::weak_ptr<const GuestAllocations::Range> range;
};

std::vector<LeasedRange> leasedRanges(const GuestAllocations::Lease& lease) {
    std::vector<LeasedRange> ranges;
    ranges.reserve(lease.size());
    for (const auto& range : lease) ranges.push_back({range->address, range->bytes, range});
    return ranges;
}

struct RetiredSet {
    std::vector<std::shared_ptr<void>> held;
    std::vector<const HostImport*> entries;
    std::shared_ptr<RetiredSet> next;
};

struct RetiredImports {
    std::mutex mutex;
    std::shared_ptr<RetiredSet> current = std::make_shared<RetiredSet>();
};

RetiredImports& Retired() {
    static RetiredImports retired;
    return retired;
}

std::shared_ptr<const void> RetiredSince() {
    auto& retired = Retired();
    std::lock_guard lock(retired.mutex);
    return retired.current;
}

std::vector<const HostImport*> RetiredImportsSince(const std::shared_ptr<const void>& token) {
    std::vector<const HostImport*> entries;
    auto& retired = Retired();
    std::lock_guard lock(retired.mutex);
    for (const auto* set = static_cast<const RetiredSet*>(token.get()); set != nullptr; set = set->next.get()) entries.insert(entries.end(), set->entries.begin(), set->entries.end());
    std::sort(entries.begin(), entries.end());
    return entries;
}

struct AddressSpaceCache {
    std::atomic<std::shared_ptr<const GuestBufferMemory::AddressSpace>> current;
    std::atomic<std::uint64_t> serials{0};
    // Set by the waiter's drop, cleared by the next publish: the rebuild's reason.
    std::atomic<bool> droppedByWaiter{false};
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> rebuiltFirst{0};
    std::atomic<std::uint64_t> rebuiltGeneration{0};
    std::atomic<std::uint64_t> rebuiltEpoch{0};
    std::atomic<std::uint64_t> rebuiltDevice{0};
    std::atomic<std::uint64_t> rebuiltWaiterDrop{0};
    std::atomic<std::uint64_t> unpublished{0};
    std::atomic<std::uint64_t> dissolvedOverlap{0};
    std::atomic<std::uint64_t> dissolvedImports{0};
    std::atomic<std::uint64_t> waiterDrops{0};
};

AddressSpaceCache& Spaces() {
    static AddressSpaceCache cache;
    return cache;
}

bool addressSpaceCacheEnabled() {
    static const bool disabled = std::getenv("APS5_NO_ADDRESS_SPACE_CACHE") != nullptr;
    return !disabled;
}

struct ImportFailure {
    std::uint64_t base;
    std::uint64_t bytes;
    GuestAllocations::Mapped ranges;
};

// Imports persist across draws, keyed by allocation base, and are dropped when their allocation leaves
// the registered set; a dropped import is destroyed once the recorded work that may read it completed.
struct HostImports {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkFreeMemory freeMemory = nullptr;
    std::map<std::uint64_t, HostImport> imports;
    std::map<VkDevice, std::vector<std::shared_ptr<RetiredImport>>> parked;
    std::uint64_t liveBytes = 0;
    std::map<std::uint64_t, ImportFailure> failed;
    std::map<std::uint64_t, ImportFailure> failedJoins;
    std::set<std::uint64_t> unaliasedSpans;
    // Registry generation the imports were last reconciled with.
    std::uint64_t refreshedGeneration = 0;
    // Bumped whenever an import is dropped, so HostImport pointers taken under the lock earlier can be
    // reused while it is unchanged.
    std::atomic<std::uint64_t> epoch{1};
    VkDevice watchDevice = VK_NULL_HANDLE;
    bool unwatchImports = false;
    bool unwatchDmaBufImports = false;
    std::uint64_t useCounter = 0;
    std::uint64_t buildFloor = 0;
    bool retireAllowed = true;
    bool retireRefused = false;
    std::vector<LeasedRange> refreshedLease;
    bool refreshedLeaseValid = false;
};

struct RetireScope {
    HostImports& state;
    RetireScope(HostImports& state, bool allowed) : state(state) {
        state.retireAllowed = allowed;
        state.retireRefused = false;
    }
    ~RetireScope() { state.retireAllowed = true; }
    RetireScope(const RetireScope&) = delete;
    RetireScope& operator=(const RetireScope&) = delete;
};

HostImports& Imports() {
    static HostImports imports;
    return imports;
}

void destroyImport(const Context& context, const HostImport& entry) {
    context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, entry.buffer, nullptr);
    context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, entry.memory, nullptr);
#ifdef _WIN32
    GuestArena::GuestArenaUnmapAlias_nid_postfix(entry.alias);
#endif
}

using ImportNode = std::map<std::uint64_t, HostImport>::node_type;

// Frees a dropped import's Vulkan objects when it is released. Never copied: a copy would destroy the
// same handles twice (and a temporary would destroy them at once).
struct RetiredImport {
    RetiredImport(VkDevice device, PFN_vkDestroyBuffer destroyBuffer, PFN_vkFreeMemory freeMemory, ImportNode node)
        : device(device), destroyBuffer(destroyBuffer), freeMemory(freeMemory), node(std::move(node)) {}
    RetiredImport(const RetiredImport&) = delete;
    RetiredImport& operator=(const RetiredImport&) = delete;
    ~RetiredImport() {
        const auto& entry = node.mapped();
        destroyBuffer(device, entry.buffer, nullptr);
        freeMemory(device, entry.memory, nullptr);
#ifdef _WIN32
        GuestArena::GuestArenaUnmapAlias_nid_postfix(entry.alias);
#endif
        node.mapped().range.reset();
        node.mapped().ranges.clear();
        GuestAllocations::GuestAllocationsNotifyUnpinned_nid_postfix();
    }
    VkDevice device;
    PFN_vkDestroyBuffer destroyBuffer;
    PFN_vkFreeMemory freeMemory;
    ImportNode node;
};

const GuestAllocations::Range* containingRange(const GuestAllocations::Lease& lease, std::uint64_t begin, std::uint64_t end);
bool runCovers(const GuestAllocations::Lease& lease, std::uint64_t begin, std::uint64_t end);
bool joinedCovers(const GuestAllocations::Lease& lease, std::uint64_t begin, std::uint64_t end);

void retireImport(VkDevice device, HostImports& state, std::map<std::uint64_t, HostImport>::iterator it, const GuestAllocations::Lease& lease, bool hold = true) {
    Require(state.retireAllowed, "a host import was retired outside the device lock");
    // Results shadowed for the import reach its (old) buffer first where the memory is still a
    // readable registered range (the import retires because its registration vanished or changed
    // size); the holder below outlives the batch that copies them.
    RetireShadow(device, it->second, [&lease, joined = it->second.joined](std::uint64_t begin, std::uint64_t end) { return runCovers(lease, begin, end) || (joined && joinedCovers(lease, begin, end)) || GuestAllocations::GuestAllocationsStillMapped_nid_postfix(begin, end); });
    ++state.epoch;
    state.liveBytes -= it->second.bytes;
    auto holder = std::make_shared<RetiredImport>(state.device, state.destroyBuffer, state.freeMemory, state.imports.extract(it));
    RetainRetiredImport(holder, &holder->node.mapped());
    auto* recorder = Recorder::Active();
    if (recorder == nullptr) return;
    if (hold) recorder->KeepThroughHold(std::move(holder));
    else if (!recorder->Idle()) recorder->Keep(std::move(holder));
}

bool releaseParkedImports(HostImports& state, std::uint64_t begin, std::uint64_t end) {
    std::vector<std::shared_ptr<RetiredImport>> released;
    {
        std::lock_guard lock(state.mutex);
        for (auto device = state.parked.begin(); device != state.parked.end();) {
            auto& list = device->second;
            for (auto it = list.begin(); it != list.end();) {
                if ((*it)->node.mapped().base < end && begin < (*it)->node.mapped().base + (*it)->node.mapped().bytes) {
                    released.push_back(std::move(*it));
                    it = list.erase(it);
                } else {
                    ++it;
                }
            }
            device = list.empty() ? state.parked.erase(device) : std::next(device);
        }
    }
    return !released.empty();
}

const char* bindImport(const Context& context, HostImport& entry, VkExternalMemoryHandleTypeFlagBits handleType, const void* import, std::uint32_t importTypes, bool& allocated, VkResult& failure) {
    const auto failed = [&](const char* step, VkResult result) -> const char* {
        if (entry.buffer) context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, entry.buffer, nullptr);
        if (entry.memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, entry.memory, nullptr);
        entry.buffer = VK_NULL_HANDLE;
        entry.memory = VK_NULL_HANDLE;
        failure = result;
        return step;
    };
    allocated = false;
    const VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr, static_cast<VkExternalMemoryHandleTypeFlags>(handleType)};
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = entry.bytes;
    // INDIRECT_BUFFER: DISPATCH_INDIRECT group counts are read in place (VulkanDevice::DispatchIndirect).
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (const auto result = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &entry.buffer); result != VK_SUCCESS) return failed("vkCreateBuffer", result);
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, entry.buffer, &requirements);
    const auto types = requirements.memoryTypeBits & importTypes;
    if (types == 0) return failed("memory type selection", VK_ERROR_FORMAT_NOT_SUPPORTED);
    const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, import, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
    allocation.allocationSize = entry.bytes;
    allocation.memoryTypeIndex = static_cast<std::uint32_t>(std::countr_zero(types));
    if (const auto result = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &entry.memory); result != VK_SUCCESS) return failed("vkAllocateMemory", result);
    allocated = true;
    if (const auto result = context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, entry.buffer, entry.memory, 0); result != VK_SUCCESS) return failed("vkBindBufferMemory", result);
    const VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, entry.buffer};
    entry.address = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &addressInfo);
    if (entry.address == 0) return failed("vkGetBufferDeviceAddressKHR", VK_ERROR_UNKNOWN);
    return nullptr;
}

const char* createHostPointerImport(const Context& context, HostImport& entry, VkResult& failure) {
    void* const host = entry.alias != nullptr ? entry.alias : reinterpret_cast<void*>(entry.base);
    VkMemoryHostPointerPropertiesEXT pointer{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (const auto result = context.Function<PFN_vkGetMemoryHostPointerPropertiesEXT>("vkGetMemoryHostPointerPropertiesEXT")(context.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &pointer); result != VK_SUCCESS) {
        failure = result;
        return "vkGetMemoryHostPointerPropertiesEXT";
    }
    const VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host};
    bool allocated = false;
    return bindImport(context, entry, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, &import, pointer.memoryTypeBits, allocated, failure);
}

#ifndef _WIN32
int udmabufDevice() {
    static const int device = [] {
        const int result = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
        if (result < 0) {
            const int error = errno;
            std::fprintf(stderr, "[gpu] open /dev/udmabuf: %s; shared direct memory is copied instead of imported, and GPU stores through FLAT/GLOBAL addresses do not reach it%s\n",
                std::strerror(error), error == EACCES ? " (give the user read-write access to /dev/udmabuf, for example through the kvm group)" : "");
        }
        return result;
    }();
    return device;
}

const char* createDmaBufImport(const Context& context, HostImport& entry, int file, std::uint64_t offset, VkResult& failure) {
    udmabuf_create request{};
    request.memfd = static_cast<std::uint32_t>(file);
    request.flags = UDMABUF_FLAGS_CLOEXEC;
    request.offset = offset;
    request.size = entry.bytes;
    const int device = udmabufDevice();
    const int buffer = device < 0 ? -1 : ioctl(device, UDMABUF_CREATE, &request);
    close(file);
    if (buffer < 0) {
        failure = VK_ERROR_INVALID_EXTERNAL_HANDLE;
        return device < 0 ? "open /dev/udmabuf" : "UDMABUF_CREATE";
    }
    VkMemoryFdPropertiesKHR properties{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    if (const auto result = context.Function<PFN_vkGetMemoryFdPropertiesKHR>("vkGetMemoryFdPropertiesKHR")(context.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, buffer, &properties); result != VK_SUCCESS) {
        close(buffer);
        failure = result;
        return "vkGetMemoryFdPropertiesKHR";
    }
    const VkImportMemoryFdInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, buffer};
    bool allocated = false;
    const char* step = bindImport(context, entry, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, &import, properties.memoryTypeBits, allocated, failure);
    if (!allocated) close(buffer);
    entry.dmaBuf = step == nullptr;
    return step;
}
#endif

#ifndef _WIN32
constexpr std::uint64_t ImportChunkBytes = std::uint64_t{64} << 20u;

struct ImportChunk {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkFreeMemory freeMemory = nullptr;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::uint32_t memoryType = 0;
    ImportChunk() = default;
    ImportChunk(const ImportChunk&) = delete;
    ImportChunk& operator=(const ImportChunk&) = delete;
    ~ImportChunk() {
        if (memory != VK_NULL_HANDLE) freeMemory(device, memory, nullptr);
    }
};

struct ImportChunks {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::map<std::tuple<dev_t, ino_t, std::uint64_t>, std::weak_ptr<ImportChunk>> chunks;
};

ImportChunks& Chunks() {
    static ImportChunks chunks;
    return chunks;
}

std::shared_ptr<ImportChunk> importChunk(const Context& context, int file, std::uint64_t index, std::uint32_t bufferTypes, const char*& step, VkResult& failure) {
    struct stat info {};
    if (fstat(file, &info) != 0) {
        failure = VK_ERROR_INVALID_EXTERNAL_HANDLE;
        step = "fstat of the shared backing";
        return nullptr;
    }
    auto& chunks = Chunks();
    std::lock_guard lock(chunks.mutex);
    if (chunks.device != context.device) {
        chunks.chunks.clear();
        chunks.device = context.device;
    }
    const auto key = std::make_tuple(info.st_dev, info.st_ino, index);
    if (const auto found = chunks.chunks.find(key); found != chunks.chunks.end()) {
        if (auto chunk = found->second.lock()) return chunk;
        chunks.chunks.erase(found);
    }
    if (static_cast<std::uint64_t>(info.st_size) < (index + 1) * ImportChunkBytes) {
        failure = VK_ERROR_INVALID_EXTERNAL_HANDLE;
        step = "a chunk past the end of the shared backing";
        return nullptr;
    }
    udmabuf_create request{};
    request.memfd = static_cast<std::uint32_t>(file);
    request.flags = UDMABUF_FLAGS_CLOEXEC;
    request.offset = index * ImportChunkBytes;
    request.size = ImportChunkBytes;
    const int device = udmabufDevice();
    const int buffer = device < 0 ? -1 : ioctl(device, UDMABUF_CREATE, &request);
    if (buffer < 0) {
        failure = VK_ERROR_INVALID_EXTERNAL_HANDLE;
        step = device < 0 ? "open /dev/udmabuf" : "UDMABUF_CREATE";
        return nullptr;
    }
    VkMemoryFdPropertiesKHR properties{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    if (const auto result = context.Function<PFN_vkGetMemoryFdPropertiesKHR>("vkGetMemoryFdPropertiesKHR")(context.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, buffer, &properties); result != VK_SUCCESS) {
        close(buffer);
        failure = result;
        step = "vkGetMemoryFdPropertiesKHR";
        return nullptr;
    }
    const auto types = properties.memoryTypeBits & bufferTypes;
    if (types == 0) {
        close(buffer);
        failure = VK_ERROR_FORMAT_NOT_SUPPORTED;
        step = "memory type selection";
        return nullptr;
    }
    const VkImportMemoryFdInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, buffer};
    const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &import, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
    allocation.allocationSize = ImportChunkBytes;
    allocation.memoryTypeIndex = static_cast<std::uint32_t>(std::countr_zero(types));
    auto chunk = std::make_shared<ImportChunk>();
    if (const auto result = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &chunk->memory); result != VK_SUCCESS) {
        close(buffer);
        failure = result;
        step = "vkAllocateMemory";
        return nullptr;
    }
    chunk->device = context.device;
    chunk->freeMemory = context.Function<PFN_vkFreeMemory>("vkFreeMemory");
    chunk->memoryType = allocation.memoryTypeIndex;
    for (auto it = chunks.chunks.begin(); it != chunks.chunks.end();) it = it->second.expired() ? chunks.chunks.erase(it) : std::next(it);
    chunks.chunks[key] = chunk;
    return chunk;
}

const char* createChunkedDmaBufImport(const Context& context, HostImport& entry, int file, std::uint64_t offset, bool& fits, VkResult& failure) {
    const auto index = offset / ImportChunkBytes;
    const auto within = offset - index * ImportChunkBytes;
    fits = entry.bytes != 0 && within + entry.bytes <= ImportChunkBytes;
    if (!fits) return nullptr;
    const VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = entry.bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (const auto result = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer); result != VK_SUCCESS) {
        failure = result;
        return "vkCreateBuffer";
    }
    const auto destroy = [&] { context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr); };
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
    if (requirements.alignment == 0 || within % requirements.alignment != 0 || within + requirements.size > ImportChunkBytes) {
        destroy();
        fits = false;
        return nullptr;
    }
    const char* step = nullptr;
    auto chunk = importChunk(context, file, index, requirements.memoryTypeBits, step, failure);
    if (chunk == nullptr) {
        destroy();
        return step;
    }
    if (((requirements.memoryTypeBits >> chunk->memoryType) & 1u) == 0) {
        destroy();
        fits = false;
        return nullptr;
    }
    if (const auto result = context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, chunk->memory, within); result != VK_SUCCESS) {
        destroy();
        failure = result;
        return "vkBindBufferMemory";
    }
    const VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buffer};
    entry.buffer = buffer;
    entry.memory = VK_NULL_HANDLE;
    entry.address = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &addressInfo);
    entry.chunk = std::move(chunk);
    entry.dmaBuf = true;
    return nullptr;
}
#endif

#ifndef _WIN32
struct BackingFile {
    BackingFile() = default;
    BackingFile(const BackingFile&) = delete;
    BackingFile& operator=(const BackingFile&) = delete;
    ~BackingFile() {
        if (file >= 0) close(file);
    }
    int file = -1;
    std::uint64_t offset = 0;
};

bool sharedBacking(const Context& context, const HostImport& entry, BackingFile& backing) {
    return context.dmaBufImport && GuestArena::GuestArenaSharedBacking_nid_postfix(static_cast<std::uintptr_t>(entry.base), static_cast<std::size_t>(entry.bytes), &backing.file, &backing.offset);
}

const char* createBackingImport(const Context& context, HostImport& entry, BackingFile& backing, VkResult& failure) {
    const int file = std::exchange(backing.file, -1);
    bool fits = false;
    VkResult chunkFailure = VK_SUCCESS;
    if (createChunkedDmaBufImport(context, entry, file, backing.offset, fits, chunkFailure) == nullptr && fits) {
        close(file);
        return nullptr;
    }
    return createDmaBufImport(context, entry, file, backing.offset, failure);
}
#endif

const char* createImport(const Context& context, HostImport& entry, VkResult& failure) {
    const char* step = createHostPointerImport(context, entry, failure);
#ifndef _WIN32
    BackingFile backing;
    if (step != nullptr && sharedBacking(context, entry, backing)) step = createBackingImport(context, entry, backing, failure);
#endif
    return step;
}

#ifndef _WIN32
enum class ImportWatchRequest : std::uint8_t { Probe, Watch, Unwatch };

ImportWatchRequest importWatchRequest() {
    static const auto request = [] {
        const char* value = std::getenv("APS5_WRITE_WATCH_IMPORTS");
        if (value == nullptr || std::strcmp(value, "probe") == 0) return ImportWatchRequest::Probe;
        if (std::strcmp(value, "watch") == 0) return ImportWatchRequest::Watch;
        if (std::strcmp(value, "unwatch") == 0) return ImportWatchRequest::Unwatch;
        throw std::runtime_error(std::string("APS5_WRITE_WATCH_IMPORTS=") + value + ": expected probe, watch or unwatch");
    }();
    return request;
}
#endif

void decideImportWatch(const Context& context, HostImports& state) {
    if (state.watchDevice == context.device) return;
#ifdef _WIN32
    state.watchDevice = context.device;
    state.unwatchImports = false;
    state.unwatchDmaBufImports = false;
    if (context.hostImportAlignment == 0 || !GuestMemory::WriteWatched()) return;
    const auto probe = ProbeImportWriteProtection(context);
    state.unwatchImports = probe.failure != nullptr || probe.writtenAfterSubmit != 0 || probe.writtenAfterFill != 0 || probe.writtenByCpu == 0;
    if (probe.failure != nullptr) std::fprintf(stderr, "[write-watch] host imports and the write watch: unknown (probe failed at %s, %d); imported ranges are compared\n", probe.failure, static_cast<int>(probe.result));
    else std::fprintf(stderr, "[write-watch] host imports and the write watch: %u of %u scratch pages marked inside the import, %u after a GPU read, %u after a GPU write, %u after a CPU store; imported ranges %s\n", probe.writtenAtImport, probe.pages, probe.writtenAfterSubmit, probe.writtenAfterFill, probe.writtenByCpu, state.unwatchImports ? "are compared" : "stay watched");
#else
    const auto request = importWatchRequest();
    state.watchDevice = context.device;
    state.unwatchImports = false;
    state.unwatchDmaBufImports = false;
    if (context.hostImportAlignment == 0 || !GuestMemory::WriteWatched()) return;
    if (request == ImportWatchRequest::Watch) {
        std::fprintf(stderr, "[write-watch] host imports stay watched (APS5_WRITE_WATCH_IMPORTS=watch)\n");
        return;
    }
    if (request == ImportWatchRequest::Unwatch) {
        state.unwatchImports = true;
        state.unwatchDmaBufImports = true;
        std::fprintf(stderr, "[write-watch] host imports are compared, not watched (APS5_WRITE_WATCH_IMPORTS=unwatch)\n");
        return;
    }
    const auto probe = ProbeImportWriteProtection(context);
    if (probe.failure != nullptr) {
        state.unwatchImports = true;
        std::fprintf(stderr, "[write-watch] host imports resolve write protection: unknown (probe failed at %s, %d); imported ranges are compared\n", probe.failure, static_cast<int>(probe.result));
    } else {
        state.unwatchImports = probe.writtenAfterSubmit != 0 || probe.writtenAfterFill != 0;
        std::fprintf(stderr, "[write-watch] host imports resolve write protection: %s (%u of %u scratch pages written after a GPU read, %u after a GPU write, %u after the import); imported ranges %s\n", state.unwatchImports ? "yes" : "no", probe.writtenAfterSubmit, probe.pages, probe.writtenAfterFill, probe.writtenAtImport, state.unwatchImports ? "are compared" : "stay watched");
    }
    state.unwatchDmaBufImports = state.unwatchImports;
    if (!context.dmaBufImport) return;
    const auto dmaBuf = ProbeDmaBufImportWriteProtection(context);
    if (dmaBuf.failure != nullptr) {
        state.unwatchDmaBufImports = true;
        std::fprintf(stderr, "[write-watch] dma-buf imports keep write protection: unknown (probe failed at %s, %d); imported ranges are compared\n", dmaBuf.failure, static_cast<int>(dmaBuf.result));
        return;
    }
    state.unwatchDmaBufImports = dmaBuf.writtenAfterSubmit != 0 || dmaBuf.writtenAfterFill != 0 || dmaBuf.writtenByCpu == 0;
    std::fprintf(stderr, "[write-watch] dma-buf imports keep write protection: %s (%u of %u scratch pages written after a GPU read, %u after a GPU write, %u after the import, %u seen after a CPU store); imported ranges %s\n", state.unwatchDmaBufImports ? "no" : "yes", dmaBuf.writtenAfterSubmit, dmaBuf.pages, dmaBuf.writtenAfterFill, dmaBuf.writtenAtImport, dmaBuf.writtenByCpu, state.unwatchDmaBufImports ? "are compared" : "stay watched");
#endif
}

std::shared_ptr<const GuestAllocations::Range> leasedRangeOwner(const GuestAllocations::Lease& lease, std::uint64_t base) {
    const auto found = std::lower_bound(lease.begin(), lease.end(), base, [](const auto& range, std::uint64_t value) { return range->address < value; });
    return found != lease.end() && (*found)->address == base ? *found : nullptr;
}

std::shared_ptr<const GuestAllocations::Range> leasedRangeOwner(const GuestAllocations::Lease& lease, std::uint64_t begin, std::uint64_t end) {
    const auto found = std::upper_bound(lease.begin(), lease.end(), begin, [](std::uint64_t value, const auto& range) { return value < range->address; });
    if (found == lease.begin()) return nullptr;
    const auto& range = *std::prev(found);
    return begin >= range->address && end <= range->address + range->bytes ? range : nullptr;
}

struct ImportRun {
    std::uint64_t base = 0;
    std::uint64_t bytes = 0;
    std::vector<std::shared_ptr<const GuestAllocations::Range>> ranges;
};

bool continuesRun(const GuestAllocations::Range& before, const GuestAllocations::Range& after) {
    return before.address + before.bytes == after.address && before.readable && after.readable && before.writable == after.writable && before.gpu == after.gpu;
}

std::size_t leasedIndexContaining(const GuestAllocations::Lease& lease, std::uint64_t address) {
    const auto found = std::upper_bound(lease.begin(), lease.end(), address, [](std::uint64_t value, const auto& range) { return value < range->address; });
    if (found == lease.begin()) return lease.size();
    const auto index = static_cast<std::size_t>(std::prev(found) - lease.begin());
    const auto& range = *lease[index];
    return range.readable && address < range.address + range.bytes ? index : lease.size();
}

std::atomic<std::uint64_t> importWindow{0};

ImportRun importRun(const GuestAllocations::Lease& lease, std::uint64_t address) {
    const auto index = leasedIndexContaining(lease, address);
    if (index == lease.size()) return {};
    const auto& containing = *lease[index];
    std::uint64_t low = 0;
    std::uint64_t high = std::numeric_limits<std::uint64_t>::max();
    const auto window = importWindow.load(std::memory_order_relaxed);
    if (window != 0) {
        low = address / window * window;
        high = low + window;
        if (containing.address < low || containing.address + containing.bytes > high) {
            low = containing.address;
            high = containing.address + containing.bytes;
        }
    }
    auto first = index;
    while (first > 0 && lease[first - 1]->address >= low && continuesRun(*lease[first - 1], *lease[first])) --first;
    auto last = index;
    while (last + 1 < lease.size() && lease[last + 1]->address + lease[last + 1]->bytes <= high && continuesRun(*lease[last], *lease[last + 1])) ++last;
    if (window != 0 && (lease[first]->address != low || lease[last]->address + lease[last]->bytes != high)) first = last = index;
    if (first != last && Imports().unaliasedSpans.contains(lease[first]->address)) return {lease[index]->address, lease[index]->bytes, {lease[index]}};
    ImportRun run{lease[first]->address, lease[last]->address + lease[last]->bytes - lease[first]->address, {}};
    run.ranges.assign(lease.begin() + static_cast<std::ptrdiff_t>(first), lease.begin() + static_cast<std::ptrdiff_t>(last) + 1);
    return run;
}

bool runCovers(const GuestAllocations::Lease& lease, std::uint64_t begin, std::uint64_t end) {
    const auto run = importRun(lease, begin);
    return !run.ranges.empty() && end <= run.base + run.bytes;
}

ImportRun joinedRun(const GuestAllocations::Lease& lease, std::uint64_t begin, std::uint64_t end) {
    const auto window = importWindow.load(std::memory_order_relaxed);
    auto last = leasedIndexContaining(lease, begin);
    if (window == 0 || last == lease.size()) return {};
    const auto first = last;
    for (;;) {
        if (lease[last]->bytes <= window) return {};
        if (lease[last]->address + lease[last]->bytes >= end) break;
        if (last + 1 == lease.size() || !continuesRun(*lease[last], *lease[last + 1])) return {};
        ++last;
    }
    if (first == last) return {};
    ImportRun run{lease[first]->address, lease[last]->address + lease[last]->bytes - lease[first]->address, {}};
    run.ranges.assign(lease.begin() + static_cast<std::ptrdiff_t>(first), lease.begin() + static_cast<std::ptrdiff_t>(last) + 1);
    return run;
}

bool joinedCovers(const GuestAllocations::Lease& lease, std::uint64_t begin, std::uint64_t end) {
    return !joinedRun(lease, begin, end).ranges.empty();
}

template<typename Entry>
bool sameRun(const Entry& entry, const ImportRun& run) {
    if (run.base != entry.base || run.bytes != entry.bytes || run.ranges.size() != entry.ranges.size()) return false;
    for (std::size_t index = 0; index < run.ranges.size(); ++index) {
        if (entry.ranges[index].owner_before(run.ranges[index]) || run.ranges[index].owner_before(entry.ranges[index])) return false;
    }
    return true;
}

bool wouldRetireImport(const HostImports& state, const ImportRun& run) {
    if (run.ranges.empty()) return false;
    const auto next = state.imports.lower_bound(run.base);
    if (next != state.imports.end()) {
        if (next->first == run.base && sameRun(next->second, run)) return false;
        if (next->first < run.base + run.bytes) return true;
    }
    if (next == state.imports.begin()) return false;
    const auto& previous = std::prev(next)->second;
    return previous.base + previous.bytes > run.base;
}

bool sameRange(const HostImport& entry, const GuestAllocations::Lease& lease) {
    if (entry.ranges.size() == 1 && !entry.joined && entry.bytes < entry.ranges.front()->bytes) {
        const auto current = leasedRangeOwner(lease, entry.base, entry.base + entry.bytes);
        return current != nullptr && !entry.ranges.front().owner_before(current) && !current.owner_before(entry.ranges.front());
    }
    return sameRun(entry, entry.joined ? joinedRun(lease, entry.base, entry.base + entry.bytes) : importRun(lease, entry.base));
}

bool pointerHeld(const HostImports& state, const HostImport& entry) {
    return entry.lastUse >= state.buildFloor;
}

bool overlapsHeldImport(const HostImports& state, std::uint64_t begin, std::uint64_t end) {
    auto it = state.imports.lower_bound(begin);
    if (it != state.imports.begin() && std::prev(it)->second.base + std::prev(it)->second.bytes > begin) --it;
    for (; it != state.imports.end() && it->first < end; ++it) {
        if (pointerHeld(state, it->second)) return true;
    }
    return false;
}

bool evictable(const HostImports& state, const HostImport& entry) {
    constexpr std::uint64_t protectedUses = 256;
    return !pointerHeld(state, entry) && entry.lastUse + protectedUses < state.useCounter;
}

std::uint64_t retireLeastRecent(const Context& context, HostImports& state, const GuestAllocations::Lease& lease, std::uint64_t wanted) {
    std::uint64_t freed = 0;
    while (freed < wanted) {
        auto oldest = state.imports.end();
        for (auto it = state.imports.begin(); it != state.imports.end(); ++it) {
            if (evictable(state, it->second) && (oldest == state.imports.end() || it->second.lastUse < oldest->second.lastUse)) oldest = it;
        }
        if (oldest == state.imports.end()) break;
        freed += oldest->second.bytes;
        retireImport(context.device, state, oldest, lease);
    }
    return freed;
}

std::atomic<std::uint64_t> budgetOverride{0};

std::uint64_t importBudget(const Context& context) {
    if (const auto set = budgetOverride.load(std::memory_order_relaxed); set != 0) return set;
    return GuestImportBudget(context);
}

std::atomic<std::uint64_t> rangeImportWindow{64ull << 20u};

ImportRun windowRun(const HostImports& state, ImportRun run, std::uint64_t begin, std::uint64_t end, std::uint64_t alignment) {
    const auto window = rangeImportWindow.load(std::memory_order_relaxed);
    if (run.ranges.size() != 1 || window == 0 || alignment == 0 || window % alignment != 0) return run;
    const auto& range = *run.ranges.front();
    if (range.bytes <= window || begin < range.address || end > range.address + range.bytes || end <= begin) return run;
    const auto lastCell = range.bytes / window - 1;
    const auto cell = [&](std::uint64_t address) { return std::min((address - range.address) / window, lastCell); };
    auto windowBegin = range.address + cell(begin) * window;
    const auto endCell = cell(end - 1);
    auto windowEnd = endCell == lastCell ? range.address + range.bytes : range.address + (endCell + 1) * window;
    for (bool grown = true; grown;) {
        grown = false;
        auto it = state.imports.upper_bound(windowBegin);
        if (it != state.imports.begin()) --it;
        for (; it != state.imports.end() && it->first < windowEnd; ++it) {
            const auto importEnd = it->second.base + it->second.bytes;
            if (importEnd <= windowBegin || it->second.base < range.address || importEnd > range.address + range.bytes) continue;
            if (it->second.base < windowBegin || importEnd > windowEnd) grown = true;
            windowBegin = std::min(windowBegin, it->second.base);
            windowEnd = std::max(windowEnd, importEnd);
        }
    }
    if (windowBegin % alignment != 0 || (windowEnd - windowBegin) % alignment != 0) return run;
    run.base = windowBegin;
    run.bytes = windowEnd - windowBegin;
    return run;
}

const HostImport* findImport(HostImports& state, std::uint64_t begin, std::uint64_t end);
bool importsStale(const Context& context, const HostImports& state);
const HostImport* importRunAt(const Context& context, HostImports& state, ImportRun run, bool joined, std::uint64_t address, const GuestAllocations::Lease& lease, std::uint64_t end, bool evict);

std::uint64_t evictableBytes(const HostImports& state, std::uint64_t begin, std::uint64_t end) {
    std::uint64_t bytes = 0;
    for (const auto& [base, entry] : state.imports) {
        if ((entry.base + entry.bytes <= begin || entry.base >= end) && evictable(state, entry)) bytes += entry.bytes;
    }
    return bytes;
}

const HostImport* importAllocation(const Context& context, HostImports& state, std::uint64_t address, const GuestAllocations::Lease& lease, std::uint64_t end = 0, bool evict = true) {
    const bool fresh = !importsStale(context, state);
    if (fresh && end != 0) {
        if (const auto* current = findImport(state, address, end)) return current;
    } else if (!fresh) {
        if (const auto* current = findImport(state, address, address + 1); current != nullptr && current->joined && sameRange(*current, lease)) return current;
    }
    auto run = importRun(lease, address);
    if (fresh && end == 0) {
        if (const auto* current = findImport(state, address, run.ranges.empty() ? address + 1 : run.base + run.bytes)) return current;
    }
    if (end != 0) run = windowRun(state, std::move(run), address, end, context.hostImportAlignment);
    if (run.ranges.empty()) return nullptr;
    if (end == 0 && overlapsHeldImport(state, run.base, run.base + run.bytes)) {
        const auto own = *std::find_if(run.ranges.begin(), run.ranges.end(), [&](const auto& range) { return address >= range->address && address < range->address + range->bytes; });
        if (const auto* committed = CommittedImportCovering(state.device, own->address, own->address + own->bytes)) return committed;
        if (const auto* current = findImport(state, own->address, own->address + own->bytes); current != nullptr && (fresh || std::find(current->ranges.begin(), current->ranges.end(), own) != current->ranges.end())) return current;
        run = ImportRun{own->address, own->bytes, {own}};
    }
    return importRunAt(context, state, std::move(run), false, address, lease, end, evict);
}

const HostImport* importJoined(const Context& context, HostImports& state, std::uint64_t begin, std::uint64_t end, const GuestAllocations::Lease& lease) {
    if (const auto* low = findImport(state, begin, begin + 1); low != nullptr && low->joined) begin = low->base;
    if (const auto* high = findImport(state, end - 1, end); high != nullptr && high->joined) end = high->base + high->bytes;
    auto run = joinedRun(lease, begin, end);
    if (run.ranges.empty()) return nullptr;
    if (const auto failed = state.failedJoins.find(run.base); failed != state.failedJoins.end() && sameRun(failed->second, run)) return nullptr;
    return importRunAt(context, state, std::move(run), true, begin, lease, end, true);
}

const HostImport* importRunAt(const Context& context, HostImports& state, ImportRun run, bool joined, std::uint64_t address, const GuestAllocations::Lease& lease, std::uint64_t end, bool evict) {
    const auto base = run.base;
    const auto bytes = run.bytes;
    auto& failures = joined ? state.failedJoins : state.failed;
    const auto alignment = context.hostImportAlignment;
    if (joined && (alignment == 0 || base % alignment != 0 || bytes % alignment != 0)) {
        failures.insert_or_assign(base, ImportFailure{base, bytes, GuestAllocations::Mapped(run.ranges.begin(), run.ranges.end())});
        return nullptr;
    }
    if (const auto found = state.imports.find(base); found != state.imports.end() && sameRun(found->second, run)) return &found->second;
    if (alignment == 0 || base % alignment != 0 || bytes % alignment != 0) return nullptr;
    if (const auto failed = failures.find(base); failed != failures.end()) {
        if (sameRun(failed->second, run)) return nullptr;
        failures.erase(failed);
    }
    auto first = state.imports.lower_bound(base);
    if (first != state.imports.begin() && std::prev(first)->second.base + std::prev(first)->second.bytes > base) --first;
    std::uint64_t replaced = 0;
    const auto held = [&](const HostImport& entry) { return !joined && pointerHeld(state, entry); };
    for (auto it = first; it != state.imports.end() && it->first < base + bytes; ++it) {
        if (held(it->second)) return nullptr;
        replaced += it->second.bytes;
    }
    const auto budget = importBudget(context);
    const auto live = state.liveBytes - replaced + CommittedImportPinnedBytes();
    if (live + bytes > budget && (!evict || live + bytes - evictableBytes(state, base, base + bytes) > budget)) {
        // A refused import turns every later use of the range into CPU copies, so say so.
        static std::uint64_t refused = 0, refusedBytes = 0;
        static auto lastReport = std::chrono::steady_clock::now() - std::chrono::seconds(60);
        ++refused;
        refusedBytes += bytes;
        if (std::chrono::steady_clock::now() - lastReport > std::chrono::seconds(10)) {
            lastReport = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[gpu] host import budget: %llu MiB live and %llu MiB of committed imports of %llu MiB; %llu imports (%llu MiB) refused so far, last 0x%llx+0x%llx\n", static_cast<unsigned long long>(state.liveBytes >> 20u), static_cast<unsigned long long>(CommittedImportPinnedBytes() >> 20u), static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(refused), static_cast<unsigned long long>(refusedBytes >> 20u), static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes));
        }
        return nullptr;
    }
    if (!state.retireAllowed && ((first != state.imports.end() && first->first < base + bytes) || live + bytes > budget)) {
        state.retireRefused = true;
        return nullptr;
    }
    for (auto it = first; it != state.imports.end() && it->first < base + bytes;) {
        const auto next = std::next(it);
        retireImport(context.device, state, it, lease);
        it = next;
    }
    if (state.liveBytes + CommittedImportPinnedBytes() + bytes > budget) retireLeastRecent(context, state, lease, state.liveBytes + CommittedImportPinnedBytes() + bytes - budget);
    HostImport entry{base, bytes, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
    entry.range = run.ranges.front();
    entry.ranges = std::move(run.ranges);
#ifdef _WIN32
    // Drivers pin imported pages, so every page must be committed and accessible.
    bool writable = true;
    bool readOnly = true;
    MEMORY_BASIC_INFORMATION refused{};
    for (std::uint64_t cursor = base; cursor < base + bytes;) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) == 0 || info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
            failures.insert_or_assign(base, ImportFailure{base, bytes, GuestAllocations::Mapped(entry.ranges.begin(), entry.ranges.end())});
            return nullptr;
        }
        const auto protection = info.Protect & 0xffu;
        const bool pageWritable = (info.Type == MEM_PRIVATE || info.Type == MEM_IMAGE) && (protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE || protection == PAGE_WRITECOPY || protection == PAGE_EXECUTE_WRITECOPY);
        const bool pageReadOnly = info.Type != MEM_MAPPED && (protection == PAGE_READONLY || protection == PAGE_EXECUTE_READ);
        if (!pageReadOnly) readOnly = false;
        if (info.Type != MEM_MAPPED && !pageWritable && !pageReadOnly && refused.BaseAddress == nullptr) refused = info;
        if (!pageWritable) writable = false;
        cursor = reinterpret_cast<std::uint64_t>(info.BaseAddress) + info.RegionSize;
    }
    if (!writable && readOnly) {
        failures.insert_or_assign(base, ImportFailure{base, bytes, GuestAllocations::Mapped(entry.ranges.begin(), entry.ranges.end())});
        return nullptr;
    }
    if (!writable) {
        if (refused.BaseAddress != nullptr) {
            char text[256];
            std::snprintf(text, sizeof(text), "AGC graphics: host import of 0x%llx+0x%llx: memory at 0x%llx (type 0x%lx, protection 0x%lx) is neither read-write, read-only nor a shared mapping", static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), reinterpret_cast<unsigned long long>(refused.BaseAddress), refused.Type, refused.Protect);
            throw std::runtime_error(text);
        }
        try {
            entry.alias = GuestArena::GuestArenaMapAlias_nid_postfix(static_cast<std::uintptr_t>(base), static_cast<std::size_t>(bytes));
        } catch (const std::runtime_error&) {
            if (entry.ranges.size() == 1) throw;
            if (joined) {
                failures.insert_or_assign(base, ImportFailure{base, bytes, GuestAllocations::Mapped(entry.ranges.begin(), entry.ranges.end())});
                return nullptr;
            }
            state.unaliasedSpans.insert(base);
            return importAllocation(context, state, address, lease, end, evict);
        }
    }
#else
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(base), static_cast<std::size_t>(bytes), true)) {
        failures.insert_or_assign(base, ImportFailure{base, bytes, GuestAllocations::Mapped(entry.ranges.begin(), entry.ranges.end())});
        return nullptr;
    }
#endif
    decideImportWatch(context, state);
    VkResult result = VK_SUCCESS;
    const char* step = nullptr;
#ifdef _WIN32
    const bool watchedHostPointer = entry.alias != nullptr || !state.unwatchImports;
    const bool watchedDmaBuf = watchedHostPointer;
#else
    const bool watchedHostPointer = !state.unwatchImports;
    const bool watchedDmaBuf = !state.unwatchDmaBufImports;
#endif
    if (!watchedHostPointer && CommittedImportsOverlap(context.device, base, base + bytes)) {
#ifdef _WIN32
        GuestArena::GuestArenaUnmapAlias_nid_postfix(entry.alias);
#endif
        failures.insert_or_assign(base, ImportFailure{base, bytes, GuestAllocations::Mapped(entry.ranges.begin(), entry.ranges.end())});
        return nullptr;
    }
    GuestMemory::ImportWatched(base, bytes, [&] {
        step = watchedDmaBuf == watchedHostPointer ? createImport(context, entry, result) : createHostPointerImport(context, entry, result);
        return step == nullptr;
    }, watchedHostPointer);
#ifndef _WIN32
    if (BackingFile backing; step != nullptr && watchedDmaBuf != watchedHostPointer && sharedBacking(context, entry, backing)) {
        GuestMemory::ImportWatched(base, bytes, [&] {
            step = createBackingImport(context, entry, backing, result);
            return step == nullptr;
        }, watchedDmaBuf);
    }
#endif
    entry.unwatched = step == nullptr && !(entry.dmaBuf ? watchedDmaBuf : watchedHostPointer);
    if (step != nullptr) {
#ifdef _WIN32
        GuestArena::GuestArenaUnmapAlias_nid_postfix(entry.alias);
#endif
        failures.insert_or_assign(base, ImportFailure{base, bytes, GuestAllocations::Mapped(entry.ranges.begin(), entry.ranges.end())});
        std::fprintf(stderr, "[gpu] host import of 0x%llx+0x%llx failed at %s (%d); falling back to copies\n", static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), step, static_cast<int>(result));
#ifdef _WIN32
        static const bool trace = std::getenv("APS5_TRACE_HOST_IMPORT") != nullptr;
        for (std::uint64_t cursor = base; trace && cursor < base + bytes;) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) == 0) break;
            const auto regionEnd = reinterpret_cast<std::uint64_t>(info.BaseAddress) + info.RegionSize;
            std::fprintf(stderr, "[gpu]   0x%llx+0x%llx state 0x%lx protect 0x%lx type 0x%lx allocation 0x%llx\n", static_cast<unsigned long long>(cursor), static_cast<unsigned long long>(std::min(regionEnd, base + bytes) - cursor), info.State, info.Protect, info.Type, reinterpret_cast<unsigned long long>(info.AllocationBase));
            cursor = regionEnd;
        }
#else
        static const bool trace = std::getenv("APS5_TRACE_HOST_IMPORT") != nullptr;
        if (std::FILE* maps = trace ? std::fopen("/proc/self/maps", "r") : nullptr) {
            char line[512];
            while (std::fgets(line, sizeof(line), maps) != nullptr) {
                unsigned long long first = 0;
                unsigned long long last = 0;
                if (std::sscanf(line, "%llx-%llx", &first, &last) != 2 || last <= base || first >= base + bytes) continue;
                std::fprintf(stderr, "[gpu]   %s", line);
            }
            std::fclose(maps);
        }
#endif
        return nullptr;
    }
    static std::uint64_t importedBytes = 0;
    importedBytes += bytes;
    static const bool trace = std::getenv("APS5_TRACE_HOST_IMPORT") != nullptr;
    state.liveBytes += bytes;
    if (trace) std::fprintf(stderr, "[gpu] host import of 0x%llx+0x%llx ok (%zu live, %.1f MiB live, %.1f MiB ever)%s\n", static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), state.imports.size() + 1, state.liveBytes / 1048576.0, importedBytes / 1048576.0, joined ? ", joined at bind" : "");
    entry.joined = joined;
    entry.lastUse = ++state.useCounter;
    return &state.imports.emplace(base, entry).first->second;
}

// The readable registered range containing [begin, end), or null.
const GuestAllocations::Range* containingRange(const GuestAllocations::Lease& lease, std::uint64_t begin, std::uint64_t end) {
    const auto found = std::upper_bound(lease.begin(), lease.end(), begin, [](std::uint64_t value, const auto& range) { return value < range->address; });
    if (found == lease.begin()) return nullptr;
    const auto& range = *std::prev(found);
    return range->readable && begin >= range->address && end <= range->address + range->bytes ? range.get() : nullptr;
}

// Drops imports whose registered range changed or disappeared: their pages may no longer back the
// guest addresses. Walks the imports only when the registry changed since the last walk.
using Intervals = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

Intervals changedIntervals(const std::vector<LeasedRange>& before, const GuestAllocations::Lease& after) {
    Intervals changed;
    const auto add = [&](std::uint64_t begin, std::uint64_t bytes) {
        const auto end = begin + bytes;
        if (!changed.empty() && begin <= changed.back().second) changed.back().second = std::max(changed.back().second, end);
        else changed.emplace_back(begin, end);
    };
    std::size_t old = 0;
    std::size_t now = 0;
    while (old < before.size() || now < after.size()) {
        if (old < before.size() && now < after.size() && !before[old].range.owner_before(after[now]) && !after[now].owner_before(before[old].range)) {
            ++old;
            ++now;
            continue;
        }
        if (now == after.size() || (old < before.size() && before[old].address <= after[now]->address)) {
            add(before[old].address, before[old].bytes);
            ++old;
        } else {
            add(after[now]->address, after[now]->bytes);
            ++now;
        }
    }
    return changed;
}

Intervals mergedIntervals(Intervals intervals) {
    std::sort(intervals.begin(), intervals.end());
    Intervals merged;
    for (const auto& [begin, end] : intervals) {
        if (!merged.empty() && begin <= merged.back().second) merged.back().second = std::max(merged.back().second, end);
        else merged.emplace_back(begin, end);
    }
    return merged;
}

bool nearInterval(const Intervals& intervals, std::uint64_t begin, std::uint64_t end) {
    const auto it = std::lower_bound(intervals.begin(), intervals.end(), begin, [](const std::pair<std::uint64_t, std::uint64_t>& interval, std::uint64_t value) { return interval.second < value; });
    return it != intervals.end() && it->first <= end;
}

template<typename TVisit>
void forEachImportNear(HostImports& state, const Intervals& intervals, TVisit&& visit) {
    if (state.imports.empty()) return;
    auto next = state.imports.begin();
    for (const auto& [begin, end] : intervals) {
        if (next == state.imports.end()) return;
        auto it = state.imports.upper_bound(begin);
        if (it != state.imports.begin()) --it;
        if (it->first < next->first) it = next;
        for (; it != state.imports.end() && it->first <= end; ++it) {
            next = std::next(it);
            if (it->second.base + it->second.bytes < begin) continue;
            visit(it);
        }
    }
}

void refreshChangedImports(const Context& context, HostImports& state, const GuestAllocations::Lease& lease, std::uint64_t generation) {
    const auto changed = changedIntervals(state.refreshedLease, lease);
    std::vector<std::map<std::uint64_t, HostImport>::iterator> stale;
    forEachImportNear(state, changed, [&](std::map<std::uint64_t, HostImport>::iterator it) {
        if (!sameRange(it->second, lease)) stale.push_back(it);
    });
    if (!state.retireAllowed && !stale.empty()) {
        state.retireRefused = true;
        return;
    }
    state.refreshedGeneration = generation;
    state.refreshedLease = leasedRanges(lease);
    static const bool trace = std::getenv("APS5_TRACE_HOST_IMPORT") != nullptr;
    for (const auto it : stale) {
        if (trace && leasedRangeOwner(lease, it->first, it->first + it->second.bytes) != nullptr) std::fprintf(stderr, "[gpu] host import of 0x%llx+0x%llx retired: the range was mapped again\n", static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(it->second.bytes));
        retireImport(context.device, state, it, lease);
    }
    const auto unwatch = [](std::map<std::uint64_t, HostImport>::iterator it) {
        if (it->second.unwatched) GuestMemory::Unwatch(it->first, it->second.bytes);
    };
    const auto restored = GuestMemory::TakeRestoredWatches();
    if (restored.size() == 1 && restored.front().first == 0 && restored.front().second == std::numeric_limits<std::uint64_t>::max()) {
        for (auto it = state.imports.begin(); it != state.imports.end(); ++it) unwatch(it);
    } else {
        forEachImportNear(state, changed, unwatch);
        if (!restored.empty()) forEachImportNear(state, mergedIntervals(restored), unwatch);
    }
    for (auto it = state.failed.begin(); it != state.failed.end();) it = !nearInterval(changed, it->second.base, it->second.base + it->second.bytes) || sameRun(it->second, importRun(lease, it->first)) ? std::next(it) : state.failed.erase(it);
    for (auto it = state.failedJoins.begin(); it != state.failedJoins.end();) it = !nearInterval(changed, it->second.base, it->second.base + it->second.bytes) || sameRun(it->second, joinedRun(lease, it->second.base, it->second.base + it->second.bytes)) ? std::next(it) : state.failedJoins.erase(it);
}

void refreshImports(const Context& context, HostImports& state, const GuestAllocations::Lease& lease) {
    ReleaseRetiredCommittedImports();
    if (state.device != context.device) {
        while (!state.imports.empty()) {
            auto node = state.imports.extract(state.imports.begin());
            if (state.device != VK_NULL_HANDLE && state.destroyBuffer != nullptr && state.freeMemory != nullptr) {
                state.parked[state.device].push_back(std::make_shared<RetiredImport>(state.device, state.destroyBuffer, state.freeMemory, std::move(node)));
                continue;
            }
#ifdef _WIN32
            GuestArena::GuestArenaUnmapAlias_nid_postfix(node.mapped().alias);
#endif
        }
        state.liveBytes = 0;
        state.failed.clear();
        state.failedJoins.clear();
        state.device = context.device;
        state.destroyBuffer = context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer");
        state.freeMemory = context.Function<PFN_vkFreeMemory>("vkFreeMemory");
        state.refreshedGeneration = 0;
        state.refreshedLease.clear();
        state.refreshedLeaseValid = false;
        ++state.epoch;
    }
    if (state.destroyBuffer == nullptr) state.destroyBuffer = context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    if (state.freeMemory == nullptr) state.freeMemory = context.Function<PFN_vkFreeMemory>("vkFreeMemory");
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (generation == state.refreshedGeneration) return;
    if (state.refreshedLeaseValid) {
        refreshChangedImports(context, state, lease, generation);
        return;
    }
    if (!state.retireAllowed && std::any_of(state.imports.begin(), state.imports.end(), [&](const auto& item) { return !sameRange(item.second, lease); })) {
        state.retireRefused = true;
        return;
    }
    state.refreshedGeneration = generation;
    state.refreshedLease = leasedRanges(lease);
    state.refreshedLeaseValid = true;
    static_cast<void>(GuestMemory::TakeRestoredWatches());
    for (auto it = state.imports.begin(); it != state.imports.end();) {
        if (sameRange(it->second, lease)) {
            if (it->second.unwatched) GuestMemory::Unwatch(it->first, it->second.bytes);
            ++it;
            continue;
        }
        static const bool trace = std::getenv("APS5_TRACE_HOST_IMPORT") != nullptr;
        if (trace && leasedRangeOwner(lease, it->first, it->first + it->second.bytes) != nullptr) std::fprintf(stderr, "[gpu] host import of 0x%llx+0x%llx retired: the range was mapped again\n", static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(it->second.bytes));
        const auto next = std::next(it);
        retireImport(context.device, state, it, lease);
        it = next;
    }
    for (auto it = state.failed.begin(); it != state.failed.end();) it = sameRun(it->second, importRun(lease, it->first)) ? std::next(it) : state.failed.erase(it);
    for (auto it = state.failedJoins.begin(); it != state.failedJoins.end();) it = sameRun(it->second, joinedRun(lease, it->second.base, it->second.base + it->second.bytes)) ? std::next(it) : state.failedJoins.erase(it);
}

const HostImport* findImport(HostImports& state, std::uint64_t begin, std::uint64_t end) {
    if (const auto* committed = CommittedImportCovering(state.device, begin, end)) return committed;
    auto found = state.imports.upper_bound(begin);
    if (found == state.imports.begin()) return nullptr;
    --found;
    const auto& entry = found->second;
    if (begin < entry.base || end > entry.base + entry.bytes) return nullptr;
    entry.lastUse = ++state.useCounter;
    return &entry;
}

const HostImport* importAccess(const Context& context, HostImports& state, std::uint64_t begin, std::uint64_t end, const GuestAllocations::Lease& lease) {
    if (!runCovers(lease, begin, end)) return findImport(state, begin, end);
    const auto* entry = importAllocation(context, state, begin, lease, end);
    return entry != nullptr && findImport(state, begin, end) == entry ? entry : nullptr;
}

// Whether the imports need reconciling before a lookup: the registry changed since the last walk.
bool importsStale(const Context& context, const HostImports& state) {
    return state.device != context.device || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != state.refreshedGeneration;
}

struct ImageMirrors {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::map<std::uint64_t, std::shared_ptr<ImageMirror>> entries;
    std::uint64_t builds = 0;
    std::uint64_t rebuilds = 0;
    std::uint64_t serials = 0;
    std::uint64_t sweeps = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

struct AddressBuildTiming {
    double leaseUs = 0;
    double importsUs = 0;
    double mirrorsUs = 0;
};

AddressBuildTiming& ThreadAddressTiming() {
    thread_local AddressBuildTiming timing;
    return timing;
}

struct AddressBuildTotals {
    std::mutex mutex;
    std::uint64_t builds = 0;
    AddressBuildTiming sums;
    BdaPageTable::Counts tableSeen;
    AddressSpaceStats spaceSeen;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

AddressBuildTotals& AddressBuilds() {
    static AddressBuildTotals totals;
    return totals;
}

ImageMirrors& Mirrors() {
    static ImageMirrors mirrors;
    return mirrors;
}

bool sameRange(const std::weak_ptr<const GuestAllocations::Range>& mirrored, const std::shared_ptr<const GuestAllocations::Range>& range) {
    return !mirrored.owner_before(range) && !range.owner_before(mirrored);
}

std::shared_ptr<ImageMirror> acquireMirror(const Context& context, const std::shared_ptr<const GuestAllocations::Range>& range) {
    auto& state = Mirrors();
    {
        std::lock_guard lock(state.mutex);
        if (state.device != context.device) {
            state.entries.clear();
            state.device = context.device;
        }
        const auto found = state.entries.find(range->address);
        if (found != state.entries.end() && sameRange(found->second->range, range) && found->second->bytes == range->bytes) return found->second;
    }
    char where[128];
    std::snprintf(where, sizeof(where), "read-only image mirror of 0x%llx+0x%llx: ", static_cast<unsigned long long>(range->address), static_cast<unsigned long long>(range->bytes));
    Require(GuestMemory::Accessible(reinterpret_cast<const void*>(range->address), range->bytes, false), std::string(where) + "its pages are not readable");
    auto mirror = std::make_shared<ImageMirror>();
    mirror->base = range->address;
    mirror->bytes = range->bytes;
    mirror->range = range;
    mirror->buffer = std::make_shared<Buffer>(context, static_cast<std::size_t>(mirror->bytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    {
        const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::MirrorRefresh);
        GuestMemory::Read(range->address, mirror->buffer->Bytes().first(static_cast<std::size_t>(mirror->bytes)));
    }
    std::lock_guard lock(state.mutex);
    ++state.rebuilds;
    mirror->serial = (1ull << 63u) | ++state.serials;
    state.entries[range->address] = mirror;
    return mirror;
}

void sweepMirrors() {
    auto& state = Mirrors();
    std::lock_guard lock(state.mutex);
    ++state.sweeps;
    for (auto it = state.entries.begin(); it != state.entries.end();) {
        if (it->second->range.expired()) it = state.entries.erase(it);
        else ++it;
    }
}

void reportMirrors() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& state = Mirrors();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(state.mutex);
    if (now - state.lastReport < std::chrono::seconds(10)) return;
    state.lastReport = now;
    std::uint64_t bytes = 0;
    for (const auto& [base, mirror] : state.entries) bytes += mirror->bytes;
    AgcDriver::ProfilePrint_nid_no_patch("[buffers] read-only image mirrors: %zu ranges (%.1f MiB), %llu address-based builds mapped them, %llu built\n", state.entries.size(), bytes / 1048576.0, static_cast<unsigned long long>(state.builds), static_cast<unsigned long long>(state.rebuilds));
}

struct LeaseState {
    std::mutex mutex;
    LeaseStats stats;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    const Recorder* leaseRecorder = nullptr;
    Recorder::Token newestLeaseToken{};
};

LeaseState& Leases() {
    static LeaseState state;
    return state;
}

struct PinReleaseQueue {
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pending;
    bool started = false;
};

PinReleaseQueue& PinReleases() {
    static auto& queue = *new PinReleaseQueue();
    return queue;
}

bool overlapsImport(const HostImports& state, std::uint64_t begin, std::uint64_t end) {
    auto it = state.imports.upper_bound(begin);
    if (it != state.imports.begin() && std::prev(it)->second.base + std::prev(it)->second.bytes > begin) return true;
    return it != state.imports.end() && it->first < end;
}

void releasePins(std::uint64_t address, std::uint64_t end) {
    const auto start = std::chrono::steady_clock::now();
    bool synced = false;
    bool drained = false;
    auto& leases = Leases();
    auto* recorder = Recorder::Active();
    const auto target = leases.newestLeaseToken;
    if (recorder != nullptr && recorder == leases.leaseRecorder && !recorder->Reached(target)) {
        Recorder::CountSync(4);
        recorder->Submit();
        recorder->WaitToken(target);
        recorder->Reap();
        synced = true;
    } else if (recorder != nullptr && !recorder->Idle()) {
        Recorder::CountSync(4);
        recorder->Sync();
        synced = true;
        drained = true;
    }
    if (recorder != nullptr) recorder->FlushDeferredReleases();
    auto& imports = Imports();
    bool retired = releaseParkedImports(imports, address, end);
    bool overlapping = false;
    {
        std::lock_guard importsLock(imports.mutex);
        overlapping = overlapsImport(imports, address, end);
    }
    if (overlapping) {
        const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
        std::lock_guard importsLock(imports.mutex);
        auto it = imports.imports.upper_bound(address);
        if (it != imports.imports.begin() && std::prev(it)->second.base + std::prev(it)->second.bytes > address) --it;
        while (it != imports.imports.end() && it->first < end) {
            const auto next = std::next(it);
            if (it->second.base + it->second.bytes > address) {
                retireImport(imports.device, imports, it, lease, false);
                retired = true;
            }
            it = next;
        }
    }
    if (retired && recorder != nullptr && !recorder->Idle()) {
        Recorder::CountSync(4);
        recorder->Sync();
        synced = true;
        drained = true;
    }
    if (retired && recorder != nullptr) recorder->FlushDeferredReleases();
    std::lock_guard lock(leases.mutex);
    if (synced) ++leases.stats.contentionSyncs;
    if (drained) ++leases.stats.contentionDrains;
    leases.stats.contentionMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void pinReleaseWorker() {
    auto& queue = PinReleases();
    for (;;) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> work;
        {
            std::unique_lock lock(queue.mutex);
            queue.wake.wait(lock, [&] { return !queue.pending.empty(); });
            work.swap(queue.pending);
        }
        std::erase_if(work, [](const std::pair<std::uint64_t, std::uint64_t>& range) { return !GuestAllocations::GuestAllocationsPinned_nid_postfix(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(range.first)), static_cast<std::size_t>(range.second - range.first)); });
        if (work.empty()) continue;
        try {
            std::lock_guard lock(GuestMemory::GpuMutex());
            for (const auto& [begin, end] : work) releasePins(begin, end);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[gpu] pin release failed: %s\n", error.what());
        }
        GuestAllocations::GuestAllocationsNotifyUnpinned_nid_postfix();
    }
}

bool RequestPinRelease(std::uintptr_t address, std::size_t bytes) noexcept {
    try {
        auto dropped = Spaces().current.exchange(nullptr);
        if (dropped != nullptr) {
            Spaces().droppedByWaiter.store(true, std::memory_order_relaxed);
            Spaces().waiterDrops.fetch_add(1, std::memory_order_relaxed);
        }
        {
            auto& leases = Leases();
            std::lock_guard lock(leases.mutex);
            ++leases.stats.contentionWaits;
            if (dropped != nullptr) ++leases.stats.cacheDrops;
        }
        const auto end = static_cast<std::uint64_t>(address) + bytes;
        if (dropped != nullptr) {
            dropped.reset();
            GuestAllocations::GuestAllocationsNotifyUnpinned_nid_postfix();
        }
        if (GuestMemory::GpuMutex().HeldByThisThread()) {
            releasePins(address, end);
            GuestAllocations::GuestAllocationsNotifyUnpinned_nid_postfix();
            return true;
        }
        auto& queue = PinReleases();
        {
            std::lock_guard lock(queue.mutex);
            const std::pair<std::uint64_t, std::uint64_t> request{address, end};
            if (std::find(queue.pending.begin(), queue.pending.end(), request) == queue.pending.end()) queue.pending.push_back(request);
            if (!queue.started) {
                std::thread(pinReleaseWorker).detach();
                queue.started = true;
            }
        }
        queue.wake.notify_one();
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[gpu] pin release request failed: %s\n", error.what());
        return false;
    }
}

struct LiveSpaces {
    std::mutex mutex;
    std::vector<std::weak_ptr<const GuestBufferMemory::AddressSpace>> spaces;
};

LiveSpaces& Live() {
    static LiveSpaces live;
    return live;
}

void trackSpace(const std::shared_ptr<const GuestBufferMemory::AddressSpace>& space) {
    auto& live = Live();
    std::lock_guard lock(live.mutex);
    std::erase_if(live.spaces, [](const auto& entry) { return entry.expired(); });
    live.spaces.push_back(space);
}

void DescribePinHolders(std::uintptr_t address, std::size_t bytes, char* text, std::size_t size) noexcept {
    try {
        const auto end = static_cast<std::uint64_t>(address) + bytes;
        const auto overlaps = [&](std::uint64_t begin, std::uint64_t finish) { return begin < end && address < finish; };
        std::vector<std::shared_ptr<const GuestBufferMemory::AddressSpace>> holding;
        {
            std::lock_guard lock(Live().mutex);
            for (const auto& entry : Live().spaces) {
                auto space = entry.lock();
                if (space != nullptr && std::any_of(space->lease.begin(), space->lease.end(), [&](const auto& range) { return overlaps(range->address, range->address + range->bytes); })) holding.push_back(std::move(space));
            }
        }
        std::string out;
        const auto current = Spaces().current.load();
        for (const auto& space : holding) {
            const bool cached = space == current;
            const auto builds = static_cast<long long>(space.use_count()) - 1 - (cached ? 2 : 0);
            char line[160];
            std::snprintf(line, sizeof(line), "%saddress space %llu held by %lld build%s%s", out.empty() ? "" : ", ", static_cast<unsigned long long>(space->serial), builds, builds == 1 ? "" : "s", cached ? " and the address space cache" : "");
            out += line;
        }
        std::size_t imports = 0;
        std::size_t parked = 0;
        std::uint64_t first = 0;
        std::uint64_t firstBytes = 0;
        {
            auto& state = Imports();
            std::lock_guard lock(state.mutex);
            for (const auto& [base, entry] : state.imports) {
                if (!overlaps(entry.base, entry.base + entry.bytes)) continue;
                if (imports++ == 0) {
                    first = entry.base;
                    firstBytes = entry.bytes;
                }
            }
            for (const auto& [device, list] : state.parked) {
                for (const auto& retired : list) {
                    if (overlaps(retired->node.mapped().base, retired->node.mapped().base + retired->node.mapped().bytes)) ++parked;
                }
            }
        }
        if (imports != 0) {
            char line[128];
            std::snprintf(line, sizeof(line), "%s%zu host import%s (first 0x%llx+0x%llx)", out.empty() ? "" : ", ", imports, imports == 1 ? "" : "s", static_cast<unsigned long long>(first), static_cast<unsigned long long>(firstBytes));
            out += line;
        }
        if (parked != 0) {
            char line[64];
            std::snprintf(line, sizeof(line), "%s%zu parked import%s", out.empty() ? "" : ", ", parked, parked == 1 ? "" : "s");
            out += line;
        }
        if (out.empty()) out = "no build, address space or import the driver tracks";
        std::snprintf(text, size, "%s", out.c_str());
    } catch (...) {
        std::snprintf(text, size, "holders unknown");
    }
}

void ensurePinWaiter() {
    static const bool registered = [] {
        GuestAllocations::GuestAllocationsSetPinWaiter_nid_postfix(&RequestPinRelease);
        GuestAllocations::GuestAllocationsSetPinDescriber_nid_postfix(&DescribePinHolders);
        return true;
    }();
    static_cast<void>(registered);
}

}

bool SyncLeaseWork() {
    static const bool sync = std::getenv("APS5_SYNC_LEASE_DISPATCH") != nullptr;
    return sync;
}

void CountLeaseOutcome(bool synced, const std::array<std::uint64_t, MaxRecorderStreams>& batch) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto& state = Leases();
    if (!synced) {
        GuestMemory::AssertGpuLockHeld("CountLeaseOutcome");
        if (const auto* recorder = Recorder::Active(); recorder != state.leaseRecorder) {
            state.leaseRecorder = recorder;
            state.newestLeaseToken = {};
        }
        for (std::size_t stream = 0; stream < batch.size(); ++stream) state.newestLeaseToken[stream] = std::max(state.newestLeaseToken[stream], batch[stream]);
    }
    std::lock_guard lock(state.mutex);
    ++(synced ? state.stats.synced : state.stats.deferred);
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - state.lastReport < std::chrono::seconds(10)) return;
    state.lastReport = now;
    const auto& stats = state.stats;
    AgcDriver::ProfilePrint_nid_no_patch("[address-sync] leases: %llu released at completion, %llu synced at once; %llu pin-contention waits by guest threads (%llu finished the lease batch, %llu drained the recorder, %llu cache-only drops) %.1f s\n", static_cast<unsigned long long>(stats.deferred), static_cast<unsigned long long>(stats.synced), static_cast<unsigned long long>(stats.contentionWaits), static_cast<unsigned long long>(stats.contentionSyncs), static_cast<unsigned long long>(stats.contentionDrains), static_cast<unsigned long long>(stats.cacheDrops), stats.contentionMs / 1000);
}

LeaseStats LeaseCounters() {
    auto& state = Leases();
    std::lock_guard lock(state.mutex);
    return state.stats;
}

AddressSpaceStats AddressSpaceCounters() {
    const auto& cache = Spaces();
    const auto load = [](const std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
    return {addressSpaceCacheEnabled(), load(cache.hits), load(cache.rebuiltFirst), load(cache.rebuiltGeneration), load(cache.rebuiltEpoch), load(cache.rebuiltDevice), load(cache.rebuiltWaiterDrop), load(cache.unpublished), load(cache.dissolvedOverlap), load(cache.dissolvedImports), load(cache.waiterDrops)};
}

MirrorStats MirrorCounters() {
    auto& state = Mirrors();
    std::lock_guard lock(state.mutex);
    std::uint64_t bytes = 0;
    for (const auto& [base, mirror] : state.entries) bytes += mirror->bytes;
    return {static_cast<std::uint64_t>(state.entries.size()), bytes, state.rebuilds, state.sweeps};
}

void ClearImageMirrors(VkDevice device) {
    if (auto current = Spaces().current.load(); current != nullptr && current->device == device) Spaces().current.compare_exchange_strong(current, std::shared_ptr<const GuestBufferMemory::AddressSpace>{});
    auto& state = Mirrors();
    std::map<std::uint64_t, std::shared_ptr<ImageMirror>> entries;
    {
        std::lock_guard lock(state.mutex);
        if (state.device != device) return;
        entries.swap(state.entries);
        state.device = VK_NULL_HANDLE;
    }
    Spaces().current.store(nullptr);
}

void BumpHostImportsEpoch() {
    Imports().epoch.fetch_add(1, std::memory_order_acq_rel);
}

void ClearHostImports(VkDevice device) {
    DetachCommittedImports(device);
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    state.parked.erase(device);
    if (state.device != device) return;
    for (const auto& [address, entry] : state.imports) {
        state.destroyBuffer(state.device, entry.buffer, nullptr);
        state.freeMemory(state.device, entry.memory, nullptr);
#ifdef _WIN32
        GuestArena::GuestArenaUnmapAlias_nid_postfix(entry.alias);
#endif
    }
    state.imports.clear();
    state.liveBytes = 0;
    state.failed.clear();
    state.failedJoins.clear();
    state.device = VK_NULL_HANDLE;
    state.refreshedGeneration = 0;
    ++state.epoch;
}

namespace {

void runImportProbe(const Context& context, std::uint64_t base, std::uint64_t bytes, const std::function<const char*(HostImport&, VkResult&)>& importStep, ImportProbe& probe) {
#ifndef _WIN32
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(reinterpret_cast<const void*>(base), bytes);
#endif
    HostImport import{base, bytes, VK_NULL_HANDLE, VK_NULL_HANDLE, 0};
    VkBuffer destination = VK_NULL_HANDLE;
    VkDeviceMemory destinationMemory = VK_NULL_HANDLE;
    VkCommandBuffer commands[2]{};
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    const auto collect = [&](std::uint32_t& pages) {
        pages = 0;
#ifdef _WIN32
        std::vector<void*> written(static_cast<std::size_t>(bytes / 4096));
        ULONG_PTR count = written.size();
        DWORD granularity = 0;
        if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<void*>(base), static_cast<SIZE_T>(bytes), written.data(), &count, &granularity) != 0) return false;
        pages = static_cast<std::uint32_t>(count);
        return true;
#else
        return GuestWriteWatch::GuestWriteWatchCollect_nid_postfix(static_cast<std::uintptr_t>(base), static_cast<std::size_t>(bytes), [](void* count, std::uintptr_t begin, std::uintptr_t end) { *static_cast<std::uint32_t*>(count) += static_cast<std::uint32_t>((end - begin) / 4096); }, &pages);
#endif
    };
    const auto run = [&]() -> const char* {
        std::uint32_t quiet = 0;
        if (!collect(quiet)) return "the first collect";
        if (!collect(quiet)) return "the second collect";
        if (quiet != 0) return "an unwritten scratch range";
        if (const char* step = importStep(import, probe.result)) return step;
        if (!collect(probe.writtenAtImport)) return "the collect after the import";
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if ((probe.result = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &destination)) != VK_SUCCESS) return "vkCreateBuffer";
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, destination, &requirements);
        if (requirements.memoryTypeBits == 0) return "the destination memory type";
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = static_cast<std::uint32_t>(std::countr_zero(requirements.memoryTypeBits));
        if ((probe.result = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &destinationMemory)) != VK_SUCCESS) return "vkAllocateMemory";
        if ((probe.result = context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, destination, destinationMemory, 0)) != VK_SUCCESS) return "vkBindBufferMemory";
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = context.pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 2;
        if ((probe.result = context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &allocate, commands)) != VK_SUCCESS) return "vkAllocateCommandBuffers";
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if ((probe.result = context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands[0], &begin)) != VK_SUCCESS) return "vkBeginCommandBuffer";
        const VkBufferCopy region{0, 0, bytes};
        context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands[0], import.buffer, destination, 1, &region);
        if ((probe.result = context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands[0])) != VK_SUCCESS) return "vkEndCommandBuffer";
        if ((probe.result = context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands[1], &begin)) != VK_SUCCESS) return "vkBeginCommandBuffer";
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands[1], import.buffer, 0, bytes, 0x02020202u);
        if ((probe.result = context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands[1])) != VK_SUCCESS) return "vkEndCommandBuffer";
        const VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if ((probe.result = context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &fenceInfo, nullptr, &fence)) != VK_SUCCESS) return "vkCreateFence";
        const auto execute = [&](VkCommandBuffer buffer) -> const char* {
            if ((probe.result = context.Function<PFN_vkResetFences>("vkResetFences")(context.device, 1, &fence)) != VK_SUCCESS) return "vkResetFences";
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &buffer;
            if ((probe.result = context.Function<PFN_vkQueueSubmit>("vkQueueSubmit")(context.queue, 1, &submit, fence)) != VK_SUCCESS) return "vkQueueSubmit";
            submitted = true;
            if ((probe.result = context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, 10'000'000'000ull)) != VK_SUCCESS) return "vkWaitForFences";
            submitted = false;
            return nullptr;
        };
        if (const char* step = execute(commands[0])) return step;
        if (!collect(probe.writtenAfterSubmit)) return "the collect after the submission";
        if (const char* step = execute(commands[1])) return step;
        if (!collect(probe.writtenAfterFill)) return "the collect after the fill";
        *reinterpret_cast<volatile std::uint8_t*>(base) = 2;
        if (!collect(probe.writtenByCpu)) return "the collect after a CPU store";
        return nullptr;
    };
    probe.failure = run();
    if (submitted) context.Function<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(context.queue);
    if (fence != VK_NULL_HANDLE) context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
    if (commands[0] != VK_NULL_HANDLE) context.Function<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(context.device, context.pool, 2, commands);
    if (destination != VK_NULL_HANDLE) context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, destination, nullptr);
    if (destinationMemory != VK_NULL_HANDLE) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, destinationMemory, nullptr);
    if (import.buffer != VK_NULL_HANDLE) destroyImport(context, import);
#ifndef _WIN32
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(reinterpret_cast<const void*>(base), bytes);
#endif
}

}

ImportProbe ProbeImportWriteProtection(const Context& context) {
    ImportProbe probe;
#ifdef _WIN32
    if (context.hostImportAlignment == 0) {
        probe.failure = "host import support";
        return probe;
    }
    constexpr std::uint64_t page = 4096;
    const std::uint64_t alignment = std::max<std::uint64_t>(context.hostImportAlignment, page);
    const std::uint64_t bytes = (4 * page + alignment - 1) / alignment * alignment;
    probe.pages = static_cast<std::uint32_t>(bytes / page);
    void* raw = VirtualAlloc(nullptr, static_cast<SIZE_T>(bytes), MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
    if (raw == nullptr) {
        probe.failure = "VirtualAlloc";
        return probe;
    }
    const auto base = reinterpret_cast<std::uint64_t>(raw);
    auto* scratch = static_cast<volatile std::uint8_t*>(raw);
    for (std::uint64_t offset = 0; offset < bytes; offset += page) scratch[offset] = 1;
    runImportProbe(context, base, bytes, [&](HostImport& import, VkResult& result) { return createHostPointerImport(context, import, result); }, probe);
    VirtualFree(raw, 0, MEM_RELEASE);
    return probe;
#else
    if (context.hostImportAlignment == 0) {
        probe.failure = "host import support";
        return probe;
    }
    if (!GuestWriteWatch::GuestWriteWatchAvailable_nid_postfix()) {
        probe.failure = "the write watch";
        return probe;
    }
    constexpr std::uint64_t page = 4096;
    const std::uint64_t alignment = std::max<std::uint64_t>(context.hostImportAlignment, page);
    const std::uint64_t bytes = (4 * page + alignment - 1) / alignment * alignment;
    probe.pages = static_cast<std::uint32_t>(bytes / page);
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        probe.failure = "mmap";
        return probe;
    }
    const auto base = (reinterpret_cast<std::uint64_t>(raw) + alignment - 1) / alignment * alignment;
    auto* scratch = reinterpret_cast<volatile std::uint8_t*>(base);
    for (std::uint64_t offset = 0; offset < bytes; offset += page) scratch[offset] = 1;
    runImportProbe(context, base, bytes, [&](HostImport& import, VkResult& result) { return createImport(context, import, result); }, probe);
    munmap(raw, bytes + alignment);
    return probe;
#endif
}

ImportProbe ProbeDmaBufImportWriteProtection(const Context& context) {
    ImportProbe probe;
#ifdef _WIN32
    static_cast<void>(context);
    probe.failure = "the Linux write watch";
    return probe;
#else
    if (!context.dmaBufImport) {
        probe.failure = "dma-buf import support";
        return probe;
    }
    if (!GuestWriteWatch::GuestWriteWatchAvailable_nid_postfix()) {
        probe.failure = "the write watch";
        return probe;
    }
    constexpr std::uint64_t page = 4096;
    const std::uint64_t alignment = std::max<std::uint64_t>(context.hostImportAlignment, page);
    const std::uint64_t bytes = (4 * page + alignment - 1) / alignment * alignment;
    probe.pages = static_cast<std::uint32_t>(bytes / page);
    const int file = memfd_create("aps5-dma-buf-import-probe", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (file < 0) {
        probe.failure = "memfd_create";
        return probe;
    }
    void* mapped = MAP_FAILED;
    if (ftruncate(file, static_cast<off_t>(bytes)) == 0 && fcntl(file, F_ADD_SEALS, F_SEAL_SHRINK) == 0) mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
    if (mapped == MAP_FAILED) {
        close(file);
        probe.failure = "the shared scratch mapping";
        return probe;
    }
    const auto base = reinterpret_cast<std::uint64_t>(mapped);
    auto* scratch = reinterpret_cast<volatile std::uint8_t*>(base);
    for (std::uint64_t offset = 0; offset < bytes; offset += page) scratch[offset] = 1;
    runImportProbe(context, base, bytes, [&](HostImport& import, VkResult& result) -> const char* {
        const int copy = dup(file);
        if (copy < 0) {
            result = VK_ERROR_INITIALIZATION_FAILED;
            return "dup";
        }
        return createDmaBufImport(context, import, copy, 0, result);
    }, probe);
    munmap(mapped, bytes);
    close(file);
    return probe;
#endif
}

ImportWatch PrepareImportWatch(const Context& context) {
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    decideImportWatch(context, state);
    return state.unwatchImports ? ImportWatch::Unwatch : ImportWatch::Watch;
}

void SetImportWatch(const Context& context, ImportWatch watch) {
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    state.watchDevice = context.device;
    state.unwatchImports = watch == ImportWatch::Unwatch;
    state.unwatchDmaBufImports = state.unwatchImports;
}

void SetHostImportWindow(std::uint64_t bytes) {
    importWindow.store(bytes, std::memory_order_relaxed);
}

void SetHostImportRangeWindow(std::uint64_t bytes) {
    rangeImportWindow.store(bytes, std::memory_order_relaxed);
}

void SetHostImportBudget(std::uint64_t bytes) {
    budgetOverride.store(bytes, std::memory_order_relaxed);
}

std::uint64_t HostImportLiveBytes() {
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    return state.liveBytes;
}

const HostImport* HostImportFor(const Context& context, std::uint64_t address, std::size_t bytes) {
    if (context.hostImportAlignment == 0 || bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) return nullptr;
    ensurePinWaiter();
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    // A hit is only valid while the registry has not changed since the imports were reconciled.
    if (!importsStale(context, state)) {
        if (const auto* entry = findImport(state, address, address + bytes)) {
            entry->lastUse = ++state.useCounter;
            return entry;
        }
    }
    const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
    refreshImports(context, state, lease);
    return importAccess(context, state, address, address + bytes, lease);
}

bool ImportMappedRanges(const Context& context, const GuestAllocations::Mapped& ranges, std::uint64_t generation, bool adoptDevice) {
    if (context.hostImportAlignment == 0) return true;
    ensurePinWaiter();
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    if (!adoptDevice && (state.device != context.device || state.watchDevice != context.device)) return false;
    const bool adopting = state.device != context.device;
    GuestAllocations::Lease current;
    if (adopting) {
        current = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
        refreshImports(context, state, current);
    } else if (state.refreshedGeneration > generation) {
        return true;
    }
    if (!adopting) current = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
    for (const auto& mapped : ranges) {
        const auto range = mapped.lock();
        if (range == nullptr || leasedRangeOwner(current, range->address) != range) continue;
        try {
            if (!GuestMemory::GpuMutex().HeldByThisThread() && wouldRetireImport(state, importRun(current, range->address))) return false;
            if (importAllocation(context, state, range->address, current, 0, false) == nullptr && GuestAllocations::GuestAllocationsGeneration_nid_postfix() != generation) state.failed.erase(importRun(current, range->address).base);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[gpu] import of GPU memory mapped at 0x%llx+0x%zx failed: %s\n", static_cast<unsigned long long>(range->address), range->bytes, error.what());
        }
    }
    return true;
}

bool RegisteredReadableCovers(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) return false;
    const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
    return containingRange(lease, address, address + bytes) != nullptr;
}

std::uint64_t RegisteredReadableEnd(std::uint64_t address) {
    if (address == std::numeric_limits<std::uint64_t>::max()) return 0;
    const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
    const auto* range = containingRange(lease, address, address + 1);
    return range != nullptr ? range->address + range->bytes : 0;
}

bool HostImportCovers(const Context& context, std::uint64_t address, std::size_t bytes) {
    if (context.hostImportAlignment == 0 || bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) return false;
    if (CommittedImportCovering(context.device, address, address + bytes) != nullptr) return true;
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    // Imports of a replaced device are not this device's; a stale registry only means an import
    // may be dropped at the next reconcile, which the binding path handles.
    return state.device == context.device && findImport(state, address, address + bytes) != nullptr;
}

GuestBufferMemory::GuestBufferMemory(const Context& context) : context(context) {}

bool GuestBufferMemory::WritesOverlap(std::uint64_t address, std::size_t bytes) const {
    return std::any_of(writes.begin(), writes.end(), [&](const auto& range) { return address < range.second && range.first < address + bytes; });
}

void GuestBufferMemory::CountAddressBuild() {
    auto& totals = AddressBuilds();
    auto& timing = ThreadAddressTiming();
    std::lock_guard lock(totals.mutex);
    ++totals.builds;
    totals.sums.leaseUs += timing.leaseUs;
    totals.sums.importsUs += timing.importsUs;
    totals.sums.mirrorsUs += timing.mirrorsUs;
    timing = {};
    const auto now = std::chrono::steady_clock::now();
    if (now - totals.lastReport < std::chrono::seconds(10)) return;
    totals.lastReport = now;
    const auto per = [&](double us) { return us / static_cast<double>(totals.builds); };
    const auto table = BdaPageTable::Totals();
    const auto& seen = totals.tableSeen;
    const auto delta = [](std::uint64_t now, std::uint64_t before) { return static_cast<unsigned long long>(now - before); };
    const auto space = AddressSpaceCounters();
    const auto& spaceSeen = totals.spaceSeen;
    AgcDriver::ProfilePrint_nid_no_patch("[address] %llu address-based builds (10 s), us per build: lease %.0f, imports pass %.0f, mirrors %.0f; space hits %llu / rebuilds: generation %llu, epoch %llu, device %llu, waiter drop %llu, first %llu; unpublished %llu, dissolved: overlap %llu, imports %llu%s; [bda-table] page table versions %llu (windows encoded %llu, shared %llu)\n", static_cast<unsigned long long>(totals.builds), per(totals.sums.leaseUs), per(totals.sums.importsUs), per(totals.sums.mirrorsUs), delta(space.hits, spaceSeen.hits), delta(space.rebuiltGeneration, spaceSeen.rebuiltGeneration), delta(space.rebuiltEpoch, spaceSeen.rebuiltEpoch), delta(space.rebuiltDevice, spaceSeen.rebuiltDevice), delta(space.rebuiltWaiterDrop, spaceSeen.rebuiltWaiterDrop), delta(space.rebuiltFirst, spaceSeen.rebuiltFirst), delta(space.unpublished, spaceSeen.unpublished), delta(space.dissolvedOverlap, spaceSeen.dissolvedOverlap), delta(space.dissolvedImports, spaceSeen.dissolvedImports), space.enabled ? "" : " (cache off)", delta(table.tables, seen.tables), delta(table.encodedWindows, seen.encodedWindows), delta(table.sharedWindows, seen.sharedWindows));
    totals.tableSeen = table;
    totals.spaceSeen = space;
    totals.builds = 0;
    totals.sums = {};
}

namespace {

std::string unservedRange(const GuestAllocations::Range& range, const char* why) {
    char text[256];
    std::snprintf(text, sizeof(text), "guest memory 0x%llx+0x%llx mapped for the GPU (%s%s) has no host import: %s", static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(range.bytes), range.writable ? "writable" : "read-only", range.releasable ? "" : ", main image", why);
    return text;
}

std::string unservedView(std::uint64_t begin, std::uint64_t end, const char* why) {
    char text[192];
    std::snprintf(text, sizeof(text), "guest memory 0x%llx+0x%llx bound for the GPU has no host import: %s", static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin), why);
    return text;
}

}

void RetainRetiredImport(std::shared_ptr<void> holder, const HostImport* entry) {
    auto& retired = Retired();
    std::lock_guard lock(retired.mutex);
    retired.current->held.push_back(std::move(holder));
    retired.current->entries.push_back(entry);
    auto next = std::make_shared<RetiredSet>();
    retired.current->next = next;
    retired.current = std::move(next);
}

void GuestBufferMemory::Prewarm(const Context& context) {
    if (!context.bufferDeviceAddress || !addressSpaceCacheEnabled()) return;
    const auto current = Spaces().current.load();
    if (current == nullptr || current->device != context.device) return;
    if (current->generation == GuestAllocations::GuestAllocationsGeneration_nid_postfix() && current->importsEpoch == Imports().epoch.load(std::memory_order_acquire)) return;
    GuestBufferMemory warm(context);
    warm.AcquireRegistered();
}

void GuestBufferMemory::AcquireRegistered() {
    Require(!uploaded && regions.empty() && lease.empty() && space == nullptr, "guest allocation lease must precede resource registration");
    ensurePinWaiter();
    if (!GuestMemory::GpuMutex().HeldByThisThread()) GuestAllocations::GuestAllocationsAwaitPinWaits_nid_postfix();
    retiredSince = RetiredSince();
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto& timing = ThreadAddressTiming();
    const auto lap = [&](double& into, std::chrono::steady_clock::time_point& from) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        into += std::chrono::duration<double, std::micro>(now - from).count();
        from = now;
    };
    auto at = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool importable = context.hostImportAlignment != 0;
    auto& spaces = Spaces();
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (addressSpaceCacheEnabled()) {
        auto current = spaces.current.load();
        bool hit = false;
        {
            auto& state = Imports();
            std::lock_guard lock(state.mutex);
            if (current == nullptr) (spaces.droppedByWaiter.exchange(false, std::memory_order_relaxed) ? spaces.rebuiltWaiterDrop : spaces.rebuiltFirst).fetch_add(1, std::memory_order_relaxed);
            else if (current->device != context.device) spaces.rebuiltDevice.fetch_add(1, std::memory_order_relaxed);
            else if (current->generation != generation) spaces.rebuiltGeneration.fetch_add(1, std::memory_order_relaxed);
            else if (current->importsEpoch != state.epoch) spaces.rebuiltEpoch.fetch_add(1, std::memory_order_relaxed);
            else hit = true;
            importFloor = state.useCounter + 1;
            constexpr std::uint64_t stampInterval = 32;
            if (hit && (current->stamped == 0 || state.useCounter >= current->stamped + stampInterval)) {
                const auto stamp = ++state.useCounter;
                for (const auto& region : current->base) {
                    if (region.direct != nullptr) region.direct->lastUse = stamp;
                }
                current->stamped = stamp;
            }
        }
        if (hit) {
            spaces.hits.fetch_add(1, std::memory_order_relaxed);
            space = std::move(current);
            regionsSorted = true;
            lap(timing.leaseUs, at);
            reportMirrors();
            return;
        }
    }
    lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
    lap(timing.leaseUs, at);
    std::unique_lock<GuestMemory::GpuMutexType> deviceLock(GuestMemory::GpuMutex(), std::defer_lock);
    const bool retiring = GuestMemory::GpuMutex().HeldByThisThread();
    const auto withImports = [&](const auto& body) {
        auto& state = Imports();
        for (;;) {
            {
                std::lock_guard lock(state.mutex);
                const RetireScope scope(state, retiring || deviceLock.owns_lock());
                body(state);
                if (!state.retireRefused) return;
            }
            Require(!retiring && !deviceLock.owns_lock(), "a host import refused to retire under the device lock");
            deviceLock.lock();
        }
    };
    withImports([&](HostImports& state) {
        if (importable) refreshImports(context, state, lease);
        if (importFloor == 0) importFloor = state.useCounter + 1;
        state.buildFloor = importFloor;
        importsEpoch = state.epoch;
    });
    lap(timing.importsUs, at);
    regions.reserve(lease.size());
    bool mirrored = false;
    const auto previous = addressSpaceCacheEnabled() ? spaces.current.load() : nullptr;
    const bool reuse = previous != nullptr && previous->device == context.device;
    const auto retiredImports = reuse ? RetiredImportsSince(previous->retired) : std::vector<const HostImport*>{};
    const auto window = reuse ? importWindow.load(std::memory_order_relaxed) : 0u;
    const auto windowOf = [&](const GuestAllocations::Range& range) -> std::optional<std::uint64_t> {
        if (window == 0 || range.bytes == 0 || (range.address + range.bytes - 1u) / window != range.address / window) return std::nullopt;
        return range.address / window;
    };
    std::vector<std::uint64_t> addedWindows;
    if (window != 0) {
        const auto& oldLease = previous->lease;
        std::size_t old = 0;
        for (const auto& range : lease) {
            while (old < oldLease.size() && oldLease[old]->address < range->address) ++old;
            if (old < oldLease.size() && oldLease[old] == range) continue;
            if (const auto added = windowOf(*range)) addedWindows.push_back(*added);
        }
    }
    std::size_t oldRange = 0;
    std::size_t oldRegion = 0;
    const auto reusable = [&](const std::shared_ptr<const GuestAllocations::Range>& range) -> const Region* {
        if (!reuse) return nullptr;
        const auto& oldLease = previous->lease;
        while (oldRange < oldLease.size() && oldLease[oldRange]->address < range->address) ++oldRange;
        if (oldRange == oldLease.size() || oldLease[oldRange] != range) return nullptr;
        if (const auto own = windowOf(*range); own && std::binary_search(addedWindows.begin(), addedWindows.end(), *own)) return nullptr;
        const auto& base = previous->base;
        while (oldRegion < base.size() && base[oldRegion].end <= range->address) ++oldRegion;
        if (oldRegion == base.size() || base[oldRegion].begin > range->address || base[oldRegion].end < range->address + range->bytes) return nullptr;
        const auto& region = base[oldRegion];
        if (region.direct == nullptr && region.mirror == nullptr) return nullptr;
        if (region.direct != nullptr && std::binary_search(retiredImports.begin(), retiredImports.end(), region.direct)) return nullptr;
        return &region;
    };
    std::vector<std::pair<std::size_t, std::shared_ptr<const GuestAllocations::Range>>> pending;
    std::vector<std::pair<std::size_t, std::shared_ptr<const GuestAllocations::Range>>> reused;
    for (const auto& range : lease) {
        if (!range->readable || !range->gpu) continue;
        validate(range->address, range->bytes);
        Region region{range->address, range->address + range->bytes, range->writable};
        if (const auto* old = reusable(range)) {
            region.direct = old->direct;
            region.mirror = old->mirror;
            if (region.direct != nullptr) reused.emplace_back(regions.size(), range);
            else mirrored = true;
        } else {
            pending.emplace_back(regions.size(), range);
        }
        regions.push_back(std::move(region));
    }
    std::vector<std::uint64_t> pendingWindows;
    for (const auto& [index, range] : pending) {
        if (const auto own = windowOf(*range)) pendingWindows.push_back(*own);
    }
    std::sort(pendingWindows.begin(), pendingWindows.end());
    std::vector<const HostImport*> kept;
    bool moved = false;
    for (const auto& [index, range] : reused) {
        if (const auto own = windowOf(*range); own && std::binary_search(pendingWindows.begin(), pendingWindows.end(), *own)) {
            regions[index].direct = nullptr;
            pending.emplace_back(index, range);
            moved = true;
            continue;
        }
        kept.push_back(regions[index].direct);
    }
    if (moved) std::sort(pending.begin(), pending.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
    if (!kept.empty()) {
        withImports([&](HostImports& state) {
            const auto stamp = ++state.useCounter;
            for (const auto* entry : kept) entry->lastUse = stamp;
        });
    }
    const HostImport* runEntry = nullptr;
    const GuestAllocations::Range* runRange = nullptr;
    for (const auto& [index, range] : pending) {
        auto& region = regions[index];
        const HostImport* entry = nullptr;
        if (runEntry != nullptr && continuesRun(*runRange, *range) && range->address + range->bytes <= runEntry->base + runEntry->bytes) entry = runEntry;
        else if (importable) withImports([&](HostImports& state) { entry = importAllocation(context, state, range->address, lease, 0, false); });
        lap(timing.importsUs, at);
        runEntry = entry;
        runRange = range.get();
        if (entry != nullptr) {
            region.direct = entry;
            continue;
        }
        Require(!range->writable && !range->releasable, unservedRange(*range, "only a read-only range of the main image is mirrored"));
        region.mirror = acquireMirror(context, range);
        lap(timing.mirrorsUs, at);
        mirrored = true;
    }
    regionsSorted = true;
    if (mirrored) {
        std::lock_guard lock(Mirrors().mutex);
        ++Mirrors().builds;
    }
    if (addressSpaceCacheEnabled()) {
        std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
        bool publish = true;
        {
            auto& state = Imports();
            std::lock_guard lock(state.mutex);
            if (state.epoch != importsEpoch && std::all_of(regions.begin(), regions.end(), [&](const Region& region) { return region.direct == nullptr || findImport(state, region.begin, region.end) == region.direct; })) importsEpoch = state.epoch;
            if (state.epoch != importsEpoch) {
                publish = false;
                spaces.unpublished.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::size_t kept = 0;
                for (std::size_t index = 0; index < regions.size(); ++index) {
                    auto& region = regions[index];
                    if (kept != 0) {
                        auto& previous = regions[kept - 1];
                        if (region.direct != nullptr && previous.direct == region.direct && previous.end == region.begin && previous.writable == region.writable && region.end <= region.direct->base + region.direct->bytes) {
                            previous.end = region.end;
                            continue;
                        }
                    }
                    if (kept != index) regions[kept] = std::move(region);
                    ++kept;
                }
                regions.erase(regions.begin() + static_cast<std::ptrdiff_t>(kept), regions.end());
                for (const auto& region : regions) ranges.push_back(addressRange(region));
            }
        }
        if (publish) {
            auto built = std::make_shared<AddressSpace>();
            built->generation = generation;
            built->importsEpoch = importsEpoch;
            built->device = context.device;
            built->serial = spaces.serials.fetch_add(1, std::memory_order_relaxed) + 1;
            built->lease = std::move(lease);
            built->ranges = std::move(ranges);
            built->base = std::move(regions);
            built->retired = retiredSince;
            if (reuse) {
                std::lock_guard tableLock(previous->tableMutex);
                built->inherited = previous->pageTable != nullptr ? previous->pageTable : previous->inherited;
            }
            regions.clear();
            lease.clear();
            space = std::move(built);
            spaces.droppedByWaiter.store(false, std::memory_order_relaxed);
            spaces.current.store(space);
            trackSpace(space);
        }
    }
    sweepMirrors();
    reportMirrors();
}

void GuestBufferMemory::validate(std::uint64_t address, std::size_t bytes) const {
    Require(!uploaded, "guest memory ownership is frozen for GPU execution");
    Require(address != 0 && bytes != 0, "empty guest memory range");
    Require(bytes <= std::numeric_limits<std::uint64_t>::max() - address, "guest memory range overflow");
}

GuestBufferMemory::BaseOverlap GuestBufferMemory::baseOverlap(std::uint64_t begin, std::uint64_t end, const Region** owner) const {
    if (space == nullptr || space->base.empty()) return BaseOverlap::None;
    const auto& base = space->base;
    const auto found = std::upper_bound(base.begin(), base.end(), begin, [](std::uint64_t value, const Region& region) { return value < region.begin; });
    if (found != base.begin()) {
        const auto& previous = *std::prev(found);
        if (begin < previous.end) {
            if (end > previous.end) return BaseOverlap::Partial;
            if (owner != nullptr) *owner = &previous;
            return BaseOverlap::Inside;
        }
    }
    return found != base.end() && found->begin < end ? BaseOverlap::Partial : BaseOverlap::None;
}

void GuestBufferMemory::dissolveSpace(bool resolve) {
    Require(space != nullptr, "no address space to dissolve");
    std::vector<Region> merged;
    merged.reserve(space->base.size() + regions.size());
    if (prepared) {
        std::merge(space->base.begin(), space->base.end(), std::make_move_iterator(regions.begin()), std::make_move_iterator(regions.end()), std::back_inserter(merged), [](const Region& left, const Region& right) { return left.begin < right.begin; });
    } else {
        merged.assign(space->base.begin(), space->base.end());
        merged.insert(merged.end(), std::make_move_iterator(regions.begin()), std::make_move_iterator(regions.end()));
        regionsSorted = false;
    }
    if (resolve) {
        for (auto& region : merged) {
            if (region.direct != nullptr) region.pending = true;
        }
    }
    lease = space->lease;
    importsEpoch = space->importsEpoch;
    regions = std::move(merged);
    space.reset();
}

const GuestBufferMemory::Region* GuestBufferMemory::owner(std::uint64_t address) const {
    const auto candidate = [address](const std::vector<Region>& list) -> const Region* {
        const auto found = std::upper_bound(list.begin(), list.end(), address, [](std::uint64_t value, const Region& region) { return value < region.begin; });
        return found == list.begin() ? nullptr : &*std::prev(found);
    };
    const auto* own = candidate(regions);
    const auto* shared = space != nullptr ? candidate(space->base) : nullptr;
    if (own == nullptr || (shared != nullptr && shared->begin > own->begin)) return shared;
    return own;
}

void GuestBufferMemory::AddInPlace(std::uint64_t address, std::size_t bytes, bool written) {
    validate(address, bytes);
    const auto begin = address & ~std::uint64_t{3};
    const auto end = (address + bytes + 3) & ~std::uint64_t{3};
    if (written) writes.emplace_back(address, address + bytes);
    const Region* inside = nullptr;
    switch (baseOverlap(begin, end, &inside)) {
        case BaseOverlap::Inside:
            Require(inside->direct != nullptr || !written, unservedView(begin, end, "a written range inside a read-only image mirror"));
            return;
        case BaseOverlap::Partial:
            Spaces().dissolvedOverlap.fetch_add(1, std::memory_order_relaxed);
            dissolveSpace(false);
            break;
        case BaseOverlap::None:
            break;
    }
    regions.push_back({begin, end, true});
    regionsSorted = false;
}

void GuestBufferMemory::Upload() {
    UploadPrepare();
    UploadFinish();
}

void GuestBufferMemory::UploadPrepare() {
    Require(!prepared && !uploaded, "guest memory was already uploaded");
    prepared = true;
    std::sort(regions.begin(), regions.end(), [](const Region& left, const Region& right) { return left.begin < right.begin; });
    std::vector<Region> merged;
    merged.reserve(regions.size());
    for (auto& region : regions) {
        if (!merged.empty() && region.begin < merged.back().end) {
            auto& previous = merged.back();
            Require(previous.mirror == nullptr && region.mirror == nullptr, unservedView(region.begin, region.end, "a bound range overlaps a read-only image mirror"));
            if (region.end > previous.end) {
                previous.end = region.end;
                previous.direct = nullptr;
            }
            previous.writable = previous.writable || region.writable;
            continue;
        }
        merged.push_back(std::move(region));
    }
    regions = std::move(merged);
    regionsSorted = true;
    for (auto& region : regions) {
        if (region.mirror == nullptr) region.pending = true;
    }
}

void GuestBufferMemory::UploadFinish() {
    Require(prepared && !uploaded, "guest memory upload was not prepared");
    for (const auto& region : regions) CompletePendingCpuWrites(context, region.begin, static_cast<std::size_t>(region.end - region.begin));
    if (space != nullptr && Recorder::PendingCompletionLabels() != 0) {
        if (auto* recorder = Recorder::Active()) {
            for (const auto& [address, bytes] : recorder->CompletionLabelRanges()) {
                const bool read = accessUnbounded || std::ranges::any_of(accessRanges, [&](const auto& range) { return address < range.second && range.first < address + bytes; });
                if (read) CompletePendingCpuWrites(context, address, bytes);
            }
        }
    }
    uploaded = true;
    GuestAllocations::Lease acquired;
    const auto importRanges = [&]() -> const GuestAllocations::Lease& {
        if (!lease.empty()) return lease;
        if (space != nullptr) return space->lease;
        if (acquired.empty()) acquired = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
        return acquired;
    };
    const bool importable = context.hostImportAlignment != 0;
    if (importable) {
        auto& state = Imports();
        std::lock_guard lock(state.mutex);
        state.buildFloor = importFloor != 0 ? importFloor : state.useCounter + 1;
        if (importsStale(context, state)) refreshImports(context, state, importRanges());
        if (space != nullptr && state.epoch != space->importsEpoch) {
            Spaces().dissolvedImports.fetch_add(1, std::memory_order_relaxed);
            dissolveSpace(true);
        }
        for (const auto& region : regions) {
            if (!region.pending || region.direct != nullptr) continue;
            if (findImport(state, region.begin, region.end) != nullptr || runCovers(importRanges(), region.begin, region.end)) continue;
            static_cast<void>(importJoined(context, state, region.begin, region.end, importRanges()));
        }
    }
    for (auto& region : regions) {
        if (!region.pending) continue;
        region.pending = false;
        Require(importable, unservedView(region.begin, region.end, "the device imports no host memory"));
        auto& state = Imports();
        std::lock_guard lock(state.mutex);
        const HostImport* entry = region.direct != nullptr && state.epoch == importsEpoch ? region.direct : nullptr;
        if (entry == nullptr) entry = findImport(state, region.begin, region.end);
        if (entry == nullptr) entry = importAccess(context, state, region.begin, region.end, importRanges());
        Require(entry != nullptr, unservedView(region.begin, region.end, "no import covers it"));
        region.direct = entry;
    }
    shadowOverrides.clear();
    shadowSlabs.clear();
    if (space != nullptr || !lease.empty()) {
        if (accessUnbounded) StorageTexture::FlushPending(0, std::numeric_limits<std::size_t>::max(), nullptr, "page-table build", PublishScope::None);
        else for (const auto& [begin, end] : accessRanges) StorageTexture::FlushPending(begin, static_cast<std::size_t>(end - begin), nullptr, "page-table access", PublishScope::None);
        for (const auto& [begin, end] : writes) StorageTexture::PublishShadowsOnly(begin, static_cast<std::size_t>(end - begin), PublishScope::Whole, "page-table build");
    }
    const auto regionScope = space != nullptr || !lease.empty() ? PublishScope::None : PublishScope::Whole;
    for (const auto& region : regions) StorageTexture::FlushPending(region.begin, static_cast<std::size_t>(region.end - region.begin), nullptr, "imported buffer region", regionScope);
    if (space != nullptr || !lease.empty()) mapShadows();
}

namespace {

struct ShadowSnapshot {
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t serial = 0;
    std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
    std::vector<std::weak_ptr<ShadowSlab>> slabs;
};

struct ShadowSnapshotTag {};

}

void GuestBufferMemory::mapShadows() {
    constexpr auto everything = std::numeric_limits<std::size_t>::max();
    if (!AnyShadowedOverlaps(0, everything)) return;
    auto& snapshot = HostThreadLocal<ShadowSnapshot, ShadowSnapshotTag>();
    std::vector<std::shared_ptr<ShadowSlab>> slabs;
    const bool current = !ShadowWritesQueued() && snapshot.device == context.device && snapshot.serial == ShadowSerial() && std::all_of(snapshot.slabs.begin(), snapshot.slabs.end(), [&](const std::weak_ptr<ShadowSlab>& slab) {
        slabs.push_back(slab.lock());
        return slabs.back() != nullptr;
    });
    if (!current) {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        const auto mappings = ShadowMappings(0, everything);
        snapshot.ranges.clear();
        snapshot.slabs.clear();
        slabs.clear();
        for (const auto& mapping : mappings) {
            snapshot.ranges.push_back({mapping.begin, mapping.end, mapping.device, ShaderRecompiler::BdaAbi::Read, 0u});
            snapshot.slabs.push_back(mapping.slab);
            slabs.push_back(mapping.slab);
        }
        snapshot.serial = ShadowSerial();
        snapshot.device = context.device;
    }
    for (std::size_t index = 0; index < snapshot.ranges.size(); ++index) {
        auto range = snapshot.ranges[index];
        if (writableOver(range.begin, range.end)) range.permissions |= ShaderRecompiler::BdaAbi::Write;
        shadowOverrides.push_back(range);
        shadowSlabs.push_back(std::move(slabs[index]));
    }
}

bool GuestBufferMemory::writableOver(std::uint64_t begin, std::uint64_t end) const {
    std::vector<ShaderRecompiler::BdaAbi::Range> local;
    std::span<const ShaderRecompiler::BdaAbi::Range> ranges;
    if (space != nullptr) {
        ranges = space->ranges;
    } else {
        for (const auto& region : regions) local.push_back(addressRange(region));
        std::sort(local.begin(), local.end(), [](const auto& left, const auto& right) { return left.begin < right.begin; });
        ranges = local;
    }
    auto found = std::upper_bound(ranges.begin(), ranges.end(), begin, [](std::uint64_t address, const ShaderRecompiler::BdaAbi::Range& range) { return address < range.begin; });
    if (found == ranges.begin()) return false;
    auto cursor = begin;
    for (--found; found != ranges.end() && found->begin <= cursor && cursor < end; ++found) {
        if (found->end <= cursor || (found->permissions & ShaderRecompiler::BdaAbi::Write) == 0u) return false;
        cursor = found->end;
    }
    return cursor >= end;
}

VkDescriptorBufferInfo GuestBufferMemory::Descriptor(std::uint64_t address, std::size_t bytes, std::uint32_t& adjustment) const {
    Require(uploaded && !committed, "guest GPU memory is not available");
    Require(bytes != 0 && bytes <= std::numeric_limits<std::uint64_t>::max() - address, "invalid guest buffer view");
    const auto* found = owner(address);
    Require(found != nullptr && address >= found->begin && address + bytes <= found->end && found->direct != nullptr, unservedView(address, address + bytes, "no import serves the bound range"));
    const auto& region = *found;
    const auto offset = address - region.direct->base;
    Require(context.limits.minStorageBufferOffsetAlignment != 0, "no storage buffer offset alignment");
    Require(region.direct->base % 4 == 0, "a guest buffer view in an import that does not start at a DWORD boundary is not implemented");
    adjustment = static_cast<std::uint32_t>(offset % std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4));
    const auto range = ViewBytes(bytes, adjustment);
    Require(address - adjustment + range <= region.direct->base + region.direct->bytes, unservedView(address, address + bytes, "rounded up to its view range it exceeds its import"));
    Require(range <= context.limits.maxStorageBufferRange, "guest buffer view exceeds descriptor range limit");
    return {region.direct->buffer, offset - adjustment, range};
}

std::pair<VkDeviceAddress, std::uint64_t> GuestBufferMemory::Address(std::uint64_t address, std::size_t bytes, std::uint32_t& adjustment) const {
    Require(uploaded && !committed, "guest GPU memory is not available");
    Require(bytes != 0 && bytes <= std::numeric_limits<std::uint64_t>::max() - address, "invalid guest buffer view");
    const auto* found = owner(address);
    Require(found != nullptr && address >= found->begin && address + bytes <= found->end, unservedView(address, address + bytes, "no import or mirror serves the range"));
    const auto& region = *found;
    const auto base = region.direct != nullptr ? region.direct->base : region.mirror->base;
    const auto offset = address - base;
    Require(base % 4 == 0, "a guest buffer view in a GPU owner that does not start at a DWORD boundary is not implemented");
    adjustment = static_cast<std::uint32_t>(offset % AddressAlignment);
    const auto range = ViewBytes(bytes, adjustment);
    const auto end = region.direct != nullptr ? region.direct->base + region.direct->bytes : region.mirror->base + region.mirror->bytes;
    Require(address - adjustment + range <= end, unservedView(address, address + bytes, "rounded up to its view range it exceeds its GPU owner"));
    const auto owned = region.direct != nullptr ? region.direct->address : region.mirror->buffer->DeviceAddress();
    const auto device = owned + offset - adjustment;
    Require(device % AddressAlignment == 0, "a guest buffer's GPU owner is not aligned for device address access");
    return {device, range};
}

std::uint64_t GuestBufferMemory::ViewBytes(std::uint64_t bytes, std::uint32_t adjustment) {
    return adjustment % 4 == 0 ? bytes + adjustment : 4 * (adjustment / 4 + bytes / 4 + 1);
}

ShaderRecompiler::BdaAbi::Range GuestBufferMemory::addressRange(const Region& region) {
    Require(region.direct != nullptr || region.mirror != nullptr, "incomplete guest GPU upload");
    const auto address = region.direct != nullptr ? region.direct->address + (region.begin - region.direct->base) : region.mirror->buffer->DeviceAddress() + (region.begin - region.mirror->base);
    Require(region.end - region.begin <= std::numeric_limits<std::uint64_t>::max() - address, "GPU address range overflow");
    const auto permissions = ShaderRecompiler::BdaAbi::Read | (region.direct != nullptr && region.writable ? ShaderRecompiler::BdaAbi::Write : 0u);
    const auto end = region.direct != nullptr ? region.end : std::min(region.end, region.mirror->base + region.mirror->bytes);
    return {region.begin, end, address, permissions, 0};
}

std::vector<ShaderRecompiler::BdaAbi::Range> GuestBufferMemory::AddressRanges() const {
    Require(uploaded && !committed, "guest GPU address ranges are not available");
    std::vector<ShaderRecompiler::BdaAbi::Range> result;
    result.reserve(regions.size() + (space != nullptr ? space->ranges.size() : 0));
    for (const auto& region : regions) result.push_back(addressRange(region));
    if (space != nullptr) {
        std::vector<ShaderRecompiler::BdaAbi::Range> combined;
        combined.reserve(result.size() + space->ranges.size());
        std::merge(space->ranges.begin(), space->ranges.end(), result.begin(), result.end(), std::back_inserter(combined), [](const auto& left, const auto& right) { return left.begin < right.begin; });
        result = std::move(combined);
    }
    return result;
}

std::vector<ShaderRecompiler::BdaAbi::Grant> GuestBufferMemory::WriteGrants() const {
    Require(uploaded && !committed, "guest GPU write grants are not available");
    auto sorted = writes;
    std::sort(sorted.begin(), sorted.end());
    std::vector<ShaderRecompiler::BdaAbi::Grant> grants;
    for (const auto& [begin, end] : sorted) {
        if (!grants.empty() && begin <= grants.back().end) grants.back().end = std::max(grants.back().end, end);
        else grants.push_back({begin, end});
    }
    return grants;
}

std::vector<ShaderRecompiler::BdaAbi::Range> GuestBufferMemory::OwnRanges() const {
    Require(uploaded && !committed, "guest GPU address ranges are not available");
    return {};
}

std::shared_ptr<const BdaPageTable> GuestBufferMemory::PageTable(const Context& context) const {
    Require(uploaded && !committed, "guest GPU address ranges are not available");
    auto* residency = ResidencyDriver::Active();
    const auto serial = residency != nullptr ? residency->Manager().Serial() : 0;
    if (space != nullptr) {
        if (space->ranges.empty()) return nullptr;
        std::lock_guard lock(space->tableMutex);
        if (space->pageTable == nullptr || space->residencySerial != serial) {
            space->resident = WithResidentPages(space->ranges);
            const std::array<std::shared_ptr<const BdaPageTable>, 2> shares{space->pageTable, space->inherited};
            space->pageTable = BdaPageTable::Build(context, space->resident, shares);
            space->inherited.reset();
            space->residencySerial = serial;
        }
        return space->pageTable;
    }
    std::vector<ShaderRecompiler::BdaAbi::Range> mapped;
    for (const auto& region : regions) mapped.push_back(addressRange(region));
    return mapped.empty() ? nullptr : BdaPageTable::Build(context, WithResidentPages(mapped));
}

std::vector<GuestBufferMemory::ImportSegment> GuestBufferMemory::WritableImportSegments(std::uint64_t begin, std::uint64_t end) const {
    Require(uploaded && !committed, "guest GPU address ranges are not available");
    std::vector<ImportSegment> found;
    const auto add = [&](const Region& region) {
        if (region.direct == nullptr || !region.writable || region.end <= begin || end <= region.begin) return;
        const auto low = std::max(begin, region.begin);
        const auto high = std::min(end, region.end);
        found.push_back({low, high, region.direct->buffer, low - region.direct->base});
    };
    if (space != nullptr) {
        for (const auto& region : space->base) add(region);
    }
    for (const auto& region : regions) add(region);
    std::sort(found.begin(), found.end(), [](const ImportSegment& left, const ImportSegment& right) { return left.begin < right.begin; });
    std::vector<ImportSegment> segments;
    for (const auto& segment : found) {
        if (!segments.empty() && segment.begin < segments.back().end) continue;
        segments.push_back(segment);
    }
    return segments;
}

namespace {

bool sameRanges(std::span<const ShaderRecompiler::BdaAbi::Range> left, std::span<const ShaderRecompiler::BdaAbi::Range> right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(), [](const auto& a, const auto& b) { return a.begin == b.begin && a.end == b.end && a.deviceAddress == b.deviceAddress && a.permissions == b.permissions; });
}

}

std::shared_ptr<const BdaPageTable> GuestBufferMemory::PageTableWith(const Context& context, std::span<const ShaderRecompiler::BdaAbi::Range> overrides) const {
    Require(uploaded && !committed, "guest GPU address ranges are not available");
    if (space == nullptr) return buildWith(context, overrides);
    const auto base = PageTable(context);
    {
        std::lock_guard lock(space->tableMutex);
        for (const auto& cached : space->overrideTables) {
            if (base != nullptr && cached.base == base && cached.table != nullptr && sameRanges(cached.overrides, overrides)) return cached.table;
        }
    }
    auto table = buildWith(context, overrides);
    std::lock_guard lock(space->tableMutex);
    auto& slot = space->overrideTables[space->nextOverrideTable];
    space->nextOverrideTable = (space->nextOverrideTable + 1u) % space->overrideTables.size();
    slot.base = base;
    slot.overrides.assign(overrides.begin(), overrides.end());
    slot.table = table;
    return table;
}

std::shared_ptr<const BdaPageTable> GuestBufferMemory::buildWith(const Context& context, std::span<const ShaderRecompiler::BdaAbi::Range> overrides) const {
    std::vector<ShaderRecompiler::BdaAbi::Range> resident;
    std::array<std::shared_ptr<const BdaPageTable>, 2> shares{};
    if (space != nullptr) {
        std::lock_guard lock(space->tableMutex);
        resident = space->resident;
        shares = {space->lastOverride, space->pageTable};
    } else {
        std::vector<ShaderRecompiler::BdaAbi::Range> mapped;
        for (const auto& region : regions) mapped.push_back(addressRange(region));
        resident = WithResidentPages(mapped);
    }
    std::vector<ShaderRecompiler::BdaAbi::Range> spliced;
    spliced.reserve(resident.size() + 2u * overrides.size());
    std::size_t firstCut = 0;
    for (const auto& range : resident) {
        while (firstCut < overrides.size() && overrides[firstCut].end <= range.begin) ++firstCut;
        auto cursor = range.begin;
        for (auto index = firstCut; index < overrides.size() && overrides[index].begin < range.end; ++index) {
            const auto& cut = overrides[index];
            if (cut.end <= cursor) continue;
            if (cut.begin > cursor) {
                auto part = range;
                part.begin = cursor;
                part.end = cut.begin;
                part.deviceAddress = range.deviceAddress + (cursor - range.begin);
                spliced.push_back(part);
            }
            cursor = std::max(cursor, std::min(cut.end, range.end));
        }
        if (cursor < range.end) {
            auto part = range;
            part.begin = cursor;
            part.deviceAddress = range.deviceAddress + (cursor - range.begin);
            spliced.push_back(part);
        }
    }
    spliced.insert(spliced.end(), overrides.begin(), overrides.end());
    std::sort(spliced.begin(), spliced.end(), [](const auto& left, const auto& right) { return left.begin < right.begin; });
    if (spliced.empty()) return nullptr;
    if (space == nullptr) return BdaPageTable::Build(context, spliced);
    auto table = BdaPageTable::Build(context, spliced, shares);
    std::lock_guard lock(space->tableMutex);
    space->lastOverride = table;
    return table;
}

void GuestBufferMemory::MarkDirectWrites() const {
    for (const auto& [begin, end] : writes) GuestMemory::MarkWritten(begin, static_cast<std::size_t>(end - begin));
}

void GuestBufferMemory::Release() {
    Require(uploaded && !committed, "guest memory cannot be released twice or before upload");
    committed = true;
    lease.clear();
    space.reset();
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> GuestBufferMemory::InPlaceReads() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
    if (!uploaded || committed) return result;
    if (space != nullptr) {
        for (const auto& region : space->base) {
            if (region.direct != nullptr) result.emplace_back(region.begin, region.end);
        }
    }
    for (const auto& region : regions) {
        if (region.direct != nullptr) result.emplace_back(region.begin, region.end);
    }
    return result;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> GuestBufferMemory::RegionReads() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
    if (!uploaded || committed) return result;
    for (const auto& region : regions) {
        if (region.direct != nullptr) result.emplace_back(region.begin, region.end);
    }
    return result;
}

std::uint64_t ImageMirrorSerial(const Context& context, std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) return 0;
    auto& state = Mirrors();
    std::lock_guard lock(state.mutex);
    if (state.device != context.device || state.entries.empty()) return 0;
    auto found = state.entries.upper_bound(address);
    if (found == state.entries.begin()) return 0;
    --found;
    const auto& mirror = found->second;
    if (address < mirror->base || address + bytes > mirror->base + mirror->bytes || mirror->range.expired()) return 0;
    return mirror->serial;
}

std::uint64_t HostImportSerial(const Context& context, std::uint64_t address, std::size_t bytes, bool reconcile) {
    if (context.hostImportAlignment == 0 || bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) return 0;
    if (const auto* committed = CommittedImportCovering(context.device, address, address + bytes)) return committed->serial;
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    if (reconcile && GuestAllocations::GuestAllocationsGeneration_nid_postfix() != state.refreshedGeneration) static_cast<void>(refreshImports(context, state, GuestAllocations::GuestAllocationsAcquire_nid_postfix()));
    auto found = state.imports.upper_bound(address);
    if (found == state.imports.begin()) return 0;
    --found;
    auto& entry = found->second;
    if (address < entry.base || address + bytes > entry.base + entry.bytes) return 0;
    if (entry.serial == 0) entry.serial = NewHostImportSerial();
    return entry.serial;
}

bool HostImportsUnchanged(const Context& context, const HostImportsProof& proof) {
    if (proof.device == VK_NULL_HANDLE || proof.device != context.device) return false;
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    return state.device == proof.device && state.epoch == proof.epoch && state.refreshedGeneration == proof.refreshedGeneration && !importsStale(context, state);
}

HostImportsProof HostImportsIdentity(const Context& context) {
    auto& state = Imports();
    std::lock_guard lock(state.mutex);
    if (importsStale(context, state)) return {};
    return {state.device, state.epoch, state.refreshedGeneration};
}

}
