#include "prx/libSceAgcDriver/Graphics/include/CommittedImports.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

constexpr std::size_t ImportStackBytes = 256 * 1024;
constexpr std::uint64_t PageBytes = std::uint64_t{1} << GuestMemory::ResidencyPageShift;
constexpr VkBufferUsageFlags ImportUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

std::atomic<std::uint64_t> importSerials{0};
std::atomic<bool> attached{false};
std::atomic<std::uint64_t> pinnedBytes{0};
std::atomic<std::uint64_t> budgetOverride{0};

std::uint64_t physicalMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullTotalPhys : 0;
#else
    const auto pages = sysconf(_SC_PHYS_PAGES);
    const auto pageSize = sysconf(_SC_PAGESIZE);
    return pages > 0 && pageSize > 0 ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(pageSize) : 0;
#endif
}

std::uint64_t largestHostHeap(const Context& context) {
    std::uint64_t heap = 0;
    for (std::uint32_t index = 0; index < context.memory.memoryHeapCount; ++index) {
        if ((context.memory.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) heap = std::max<std::uint64_t>(heap, context.memory.memoryHeaps[index].size);
    }
    return heap;
}

constexpr std::uint64_t HostHeapReserve = 2048ull << 20u;

struct ImportMemory {
    ImportMemory(VkDevice device, PFN_vkFreeMemory freeMemory, VkDeviceMemory memory, std::uint64_t base, std::uint64_t bytes, void* alias)
        : device(device), freeMemory(freeMemory), memory(memory), base(base), bytes(bytes), alias(alias) {
        pinnedBytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    ImportMemory(const ImportMemory&) = delete;
    ImportMemory& operator=(const ImportMemory&) = delete;
    ~ImportMemory() {
        freeMemory(device, memory, nullptr);
#ifdef _WIN32
        GuestArena::GuestArenaUnmapAlias_nid_postfix(alias);
#endif
        pinnedBytes.fetch_sub(bytes, std::memory_order_relaxed);
    }
    VkDevice device;
    PFN_vkFreeMemory freeMemory;
    VkDeviceMemory memory;
    std::uint64_t base;
    std::uint64_t bytes;
    void* alias;
};

using Views = std::map<std::uint64_t, HostImport>;
using ViewNode = Views::node_type;

struct RetiredView {
    RetiredView(VkDevice device, PFN_vkDestroyBuffer destroyBuffer, ViewNode node) : device(device), destroyBuffer(destroyBuffer), node(std::move(node)) {}
    RetiredView(const RetiredView&) = delete;
    RetiredView& operator=(const RetiredView&) = delete;
    ~RetiredView() { destroyBuffer(device, node.mapped().buffer, nullptr); }
    VkDevice device;
    PFN_vkDestroyBuffer destroyBuffer;
    ViewNode node;
};

struct Parked {
    Context context{};
    Views views;
    std::vector<ViewNode> retired;
};

struct PendingImport {
    std::uint64_t end = 0;
    void* alias = nullptr;
    bool started = false;
};

struct Committed {
    ~Committed() {
        std::unique_lock lock(mutex);
        stopping = true;
        changed.notify_all();
        lock.unlock();
        if (worker.joinable()) worker.join();
    }
    std::mutex mutex;
    std::condition_variable changed;
    std::map<std::uint64_t, PendingImport> pending;
    std::uint64_t pendingBytes = 0;
    std::thread worker;
    bool stopping = false;
    std::exception_ptr failure;
    std::uint64_t deferred = 0;
    std::uint64_t pendingWaits = 0;
    std::uint64_t pendingWaitNanoseconds = 0;
    std::uint64_t cancelled = 0;
    VkDevice device = VK_NULL_HANDLE;
    Context context{};
    bool watched = true;
    Views views;
    std::vector<ViewNode> retired;
    std::uint64_t viewBytes = 0;
    std::uint64_t generation = 1;
    std::map<VkDevice, Parked> parked;
};

Committed& CommittedState() {
    static Committed state;
    return state;
}

std::uint64_t budgetOf(const Context& context) {
    if (const auto set = budgetOverride.load(std::memory_order_relaxed); set != 0) return set;
    const auto fromMemory = physicalMemoryBytes() / 2;
    const auto heap = largestHostHeap(context);
    return heap > HostHeapReserve ? std::min(fromMemory, heap - HostHeapReserve) : fromMemory;
}

[[noreturn]] void refuse(const Context& context, std::uint64_t base, std::uint64_t bytes, const char* step, VkResult result) {
    char text[384];
    if (step == nullptr) {
        std::snprintf(text, sizeof(text), "guest memory import of 0x%llx+0x%llx refused: %.1f MiB imported + %.1f MiB > budget %.1f MiB (physical %llu MiB / 2, host heap %llu MiB - 2048)", static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), static_cast<double>(pinnedBytes.load()) / 1048576.0, static_cast<double>(bytes) / 1048576.0, static_cast<double>(budgetOf(context)) / 1048576.0, static_cast<unsigned long long>(physicalMemoryBytes() >> 20u), static_cast<unsigned long long>(largestHostHeap(context) >> 20u));
    } else {
        std::snprintf(text, sizeof(text), "guest memory import of 0x%llx+0x%llx failed at %s (%d) with %llu MiB imported", static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), step, static_cast<int>(result), static_cast<unsigned long long>(pinnedBytes.load() >> 20u));
    }
    throw std::runtime_error(text);
}

VkBuffer createViewBuffer(const Context& context, std::uint64_t bytes, VkResult& result) {
    const VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = bytes;
    info.usage = ImportUsage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    result = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer);
    return result == VK_SUCCESS ? buffer : VK_NULL_HANDLE;
}

HostImport makeView(const Context& context, const std::shared_ptr<ImportMemory>& memory, VkBuffer buffer, std::uint64_t begin, std::uint64_t end) {
    VkResult result = VK_SUCCESS;
    if (buffer == VK_NULL_HANDLE && (buffer = createViewBuffer(context, end - begin, result)) == VK_NULL_HANDLE) refuse(context, begin, end - begin, "vkCreateBuffer", result);
    const auto destroy = [&] { context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr); };
    if ((result = context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory->memory, begin - memory->base)) != VK_SUCCESS) {
        destroy();
        refuse(context, begin, end - begin, "vkBindBufferMemory", result);
    }
    const VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buffer};
    const auto address = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &addressInfo);
    if (address == 0) {
        destroy();
        refuse(context, begin, end - begin, "vkGetBufferDeviceAddressKHR", VK_ERROR_UNKNOWN);
    }
    HostImport view{begin, end - begin, buffer, memory->memory, address};
    view.serial = NewHostImportSerial();
    view.chunk = memory;
    return view;
}

void* mapAlias(std::uint64_t base, std::uint64_t bytes) {
#ifdef _WIN32
    return GuestArena::GuestArenaMapAlias_nid_postfix(static_cast<std::uintptr_t>(base), static_cast<std::size_t>(bytes));
#else
    static_cast<void>(base);
    static_cast<void>(bytes);
    return nullptr;
#endif
}

void unmapAlias(void* alias) {
#ifdef _WIN32
    GuestArena::GuestArenaUnmapAlias_nid_postfix(alias);
#else
    static_cast<void>(alias);
#endif
}

HostImport importHost(const Context& context, std::uint64_t base, std::uint64_t bytes, void* alias) {
    const auto unmapAlias = [&] {
#ifdef _WIN32
        GuestArena::GuestArenaUnmapAlias_nid_postfix(alias);
#endif
    };
    void* const host = alias != nullptr ? alias : reinterpret_cast<void*>(base);
    VkMemoryHostPointerPropertiesEXT pointer{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (const auto result = context.Function<PFN_vkGetMemoryHostPointerPropertiesEXT>("vkGetMemoryHostPointerPropertiesEXT")(context.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &pointer); result != VK_SUCCESS) {
        unmapAlias();
        refuse(context, base, bytes, "vkGetMemoryHostPointerPropertiesEXT", result);
    }
    VkResult result = VK_SUCCESS;
    const VkBuffer buffer = createViewBuffer(context, bytes, result);
    if (buffer == VK_NULL_HANDLE) {
        unmapAlias();
        refuse(context, base, bytes, "vkCreateBuffer", result);
    }
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
    const auto types = requirements.memoryTypeBits & pointer.memoryTypeBits;
    const auto fail = [&](const char* step, VkResult code) {
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        unmapAlias();
        refuse(context, base, bytes, step, code);
    };
    if (types == 0) fail("memory type selection", VK_ERROR_FORMAT_NOT_SUPPORTED);
    const VkImportMemoryHostPointerInfoEXT hostPointer{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host};
    const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &hostPointer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
    allocation.allocationSize = bytes;
    allocation.memoryTypeIndex = static_cast<std::uint32_t>(std::countr_zero(types));
    if (alias == nullptr) GuestMemory::NoteHostImport(base, bytes);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (const auto code = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory); code != VK_SUCCESS) fail("vkAllocateMemory", code);
    const auto owner = std::make_shared<ImportMemory>(context.device, context.Function<PFN_vkFreeMemory>("vkFreeMemory"), memory, base, bytes, alias);
    return makeView(context, owner, buffer, base, base + bytes);
}

HostImport importRange(const Context& context, std::uint64_t base, std::uint64_t bytes, bool shared) {
    if (pinnedBytes.load(std::memory_order_relaxed) + bytes > budgetOf(context)) refuse(context, base, bytes, nullptr, VK_SUCCESS);
    return importHost(context, base, bytes, shared ? mapAlias(base, bytes) : nullptr);
}

bool pendingOverlaps(const Committed& state, std::uint64_t begin, std::uint64_t end) {
    auto it = state.pending.upper_bound(begin);
    if (it != state.pending.begin()) --it;
    for (; it != state.pending.end() && it->first < end; ++it) {
        if (it->second.end > begin) return true;
    }
    return false;
}

void awaitImports(Committed& state, std::unique_lock<std::mutex>& lock, std::uint64_t begin, std::uint64_t end) {
    if (pendingOverlaps(state, begin, end)) {
        const auto start = std::chrono::steady_clock::now();
        state.changed.wait(lock, [&] { return !pendingOverlaps(state, begin, end); });
        ++state.pendingWaits;
        state.pendingWaitNanoseconds += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    }
    if (state.failure) std::rethrow_exception(state.failure);
}

void cancelImports(Committed& state, std::unique_lock<std::mutex>& lock, std::uint64_t begin, std::uint64_t end) {
    for (auto it = state.pending.lower_bound(begin); it != state.pending.end() && it->first < end;) {
        if (it->second.started || it->second.end > end) {
            ++it;
            continue;
        }
        unmapAlias(it->second.alias);
        state.pendingBytes -= it->second.end - it->first;
        ++state.cancelled;
        ++state.generation;
        it = state.pending.erase(it);
    }
    state.changed.wait(lock, [&] { return !pendingOverlaps(state, begin, end); });
}

void insert(Committed& state, HostImport view);
void destroyView(const Context& context, HostImport& view);

void runImports(Committed& state) {
    std::unique_lock lock(state.mutex);
    for (;;) {
        auto next = state.pending.end();
        state.changed.wait(lock, [&] {
            if (state.stopping) return true;
            next = std::find_if(state.pending.begin(), state.pending.end(), [](const auto& entry) { return !entry.second.started; });
            return next != state.pending.end();
        });
        if (state.stopping) return;
        next->second.started = true;
        const auto base = next->first;
        const auto end = next->second.end;
        void* const alias = next->second.alias;
        const auto context = state.context;
        lock.unlock();
        std::optional<HostImport> view;
        std::exception_ptr failure;
        try {
            view = importHost(context, base, end - base, alias);
        } catch (...) {
            failure = std::current_exception();
        }
        lock.lock();
        state.pending.erase(base);
        state.pendingBytes -= end - base;
        ++state.generation;
        if (failure) {
            if (!state.failure) state.failure = failure;
        } else if (state.device == context.device) {
            insert(state, std::move(*view));
        } else {
            destroyView(context, *view);
        }
        state.changed.notify_all();
    }
}

void stopImports(Committed& state, std::unique_lock<std::mutex>& lock) {
    cancelImports(state, lock, 0, ~std::uint64_t{0});
    if (!state.worker.joinable()) return;
    state.stopping = true;
    state.changed.notify_all();
    auto worker = std::move(state.worker);
    lock.unlock();
    worker.join();
    lock.lock();
    state.stopping = false;
}

void destroyView(const Context& context, HostImport& view) {
    context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, view.buffer, nullptr);
    view.chunk.reset();
}

void insert(Committed& state, HostImport view) {
    state.viewBytes += view.bytes;
    ++state.generation;
    state.views.emplace(view.base, std::move(view));
}

bool importable(std::uint32_t protection) {
    const auto access = protection & 0xffu;
#ifdef _WIN32
    return (protection & (PAGE_GUARD | PAGE_NOCACHE | PAGE_WRITECOMBINE)) == 0 && (access == PAGE_READWRITE || access == PAGE_EXECUTE_READWRITE);
#else
    return access == 0x04u;
#endif
}

bool importableMapping(GuestArena::MappingChange change, std::uint32_t protection) {
#ifdef _WIN32
    return change == GuestArena::MappingChange::Mapped || importable(protection);
#else
    return change != GuestArena::MappingChange::Released && importable(protection);
#endif
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> uncovered(const Committed& state, std::uint64_t begin, std::uint64_t end) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pieces;
    auto cursor = begin;
    auto it = state.views.upper_bound(begin);
    if (it != state.views.begin()) --it;
    for (; it != state.views.end() && it->first < end; ++it) {
        const auto viewEnd = it->second.base + it->second.bytes;
        if (viewEnd <= cursor) continue;
        if (it->first > cursor) pieces.emplace_back(cursor, it->first);
        cursor = std::max(cursor, viewEnd);
    }
    if (cursor < end) pieces.emplace_back(cursor, end);
    return pieces;
}

void release(Committed& state, std::uint64_t begin, std::uint64_t end) {
    auto it = state.views.upper_bound(begin);
    if (it != state.views.begin()) --it;
    std::vector<std::pair<std::shared_ptr<ImportMemory>, std::pair<std::uint64_t, std::uint64_t>>> kept;
    bool any = false;
    while (it != state.views.end() && it->first < end) {
        const auto viewBegin = it->second.base;
        const auto viewEnd = viewBegin + it->second.bytes;
        if (viewEnd <= begin) {
            ++it;
            continue;
        }
        auto memory = std::static_pointer_cast<ImportMemory>(it->second.chunk);
        if (viewBegin < begin) kept.push_back({memory, {viewBegin, begin}});
        if (viewEnd > end) kept.push_back({memory, {end, viewEnd}});
        state.viewBytes -= it->second.bytes;
        const auto next = std::next(it);
        ++state.generation;
        state.retired.push_back(state.views.extract(it));
        it = next;
        any = true;
    }
    if (!any) return;
    BumpHostImportsEpoch();
    for (const auto& [memory, piece] : kept) {
        auto view = makeView(state.context, memory, VK_NULL_HANDLE, piece.first, piece.second);
        state.viewBytes += view.bytes;
        state.views.emplace(piece.first, std::move(view));
    }
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> importPieces(std::uint64_t address, std::size_t bytes, bool shared, bool& watched) {
    auto& state = CommittedState();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pieces;
    Context context{};
    {
        std::lock_guard lock(state.mutex);
        if (state.device == VK_NULL_HANDLE) return {};
        pieces = uncovered(state, address, address + bytes);
        context = state.context;
        watched = state.watched;
    }
    std::vector<HostImport> made;
    try {
        for (const auto& [begin, end] : pieces) made.push_back(importRange(context, begin, end - begin, shared));
    } catch (...) {
        for (auto& view : made) destroyView(context, view);
        throw;
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> imported;
    std::lock_guard lock(state.mutex);
    for (auto& view : made) {
        const auto end = view.base + view.bytes;
        if (state.device != context.device || uncovered(state, view.base, end) != std::vector<std::pair<std::uint64_t, std::uint64_t>>{{view.base, end}}) {
            destroyView(context, view);
            continue;
        }
        imported.emplace_back(view.base, end);
        insert(state, std::move(view));
    }
    return imported;
}

void importFresh(std::uint64_t address, std::size_t bytes) {
    bool watched = true;
    for (const auto& [begin, end] : importPieces(address, bytes, false, watched)) {
        if (watched) GuestMemory::ConsumeImportMarks(begin, static_cast<std::size_t>(end - begin));
        else GuestMemory::Unwatch(begin, static_cast<std::size_t>(end - begin));
    }
}

void importShared(std::uint64_t address, std::size_t bytes) {
#ifndef _WIN32
    importFresh(address, bytes);
#else
    auto& state = CommittedState();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pieces;
    VkDevice device = VK_NULL_HANDLE;
    {
        std::unique_lock lock(state.mutex);
        if (state.device == VK_NULL_HANDLE) return;
        awaitImports(state, lock, address, address + bytes);
        pieces = uncovered(state, address, address + bytes);
        device = state.device;
        std::uint64_t total = 0;
        for (const auto& [begin, end] : pieces) {
            total += end - begin;
            if (pinnedBytes.load(std::memory_order_relaxed) + state.pendingBytes + total > budgetOf(state.context)) refuse(state.context, begin, end - begin, nullptr, VK_SUCCESS);
        }
    }
    std::vector<void*> aliases;
    aliases.reserve(pieces.size());
    for (const auto& [begin, end] : pieces) aliases.push_back(mapAlias(begin, end - begin));
    std::unique_lock lock(state.mutex);
    for (std::size_t index = 0; index < pieces.size(); ++index) {
        const auto [begin, end] = pieces[index];
        if (state.device != device || pendingOverlaps(state, begin, end) || uncovered(state, begin, end) != std::vector<std::pair<std::uint64_t, std::uint64_t>>{{begin, end}}) {
            unmapAlias(aliases[index]);
            continue;
        }
        state.pending.emplace(begin, PendingImport{end, aliases[index], false});
        state.pendingBytes += end - begin;
        ++state.deferred;
        ++state.generation;
    }
    if (!state.worker.joinable()) state.worker = std::thread([&state] { runImports(state); });
    state.changed.notify_all();
#endif
}

void releaseRange(std::uint64_t address, std::size_t bytes) {
    auto& state = CommittedState();
    std::unique_lock lock(state.mutex);
    if (state.device == VK_NULL_HANDLE) return;
    cancelImports(state, lock, address, address + bytes);
    release(state, address, address + bytes);
}

void destroyViews(const Context& context, Views& views, std::vector<ViewNode>& retired) {
    for (auto& node : retired) destroyView(context, node.mapped());
    retired.clear();
    for (auto& [base, view] : views) destroyView(context, view);
    views.clear();
}

bool overlapsLocked(const Committed& state, std::uint64_t begin, std::uint64_t end) {
    auto it = state.views.upper_bound(begin);
    if (it != state.views.begin()) --it;
    for (; it != state.views.end() && it->first < end; ++it) {
        if (it->second.base + it->second.bytes > begin) return true;
    }
    return false;
}

const HostImport* coveringLocked(const Committed& state, std::uint64_t begin, std::uint64_t end) {
    auto found = state.views.upper_bound(begin);
    if (found == state.views.begin()) return nullptr;
    --found;
    const auto& view = found->second;
    return begin >= view.base && end <= view.base + view.bytes ? &view : nullptr;
}

}

std::uint64_t GuestImportBudget(const Context& context) {
    return budgetOf(context);
}

std::uint64_t NewHostImportSerial() {
    return importSerials.fetch_add(1, std::memory_order_relaxed) + 1;
}

void SetCommittedImportBudget(std::uint64_t bytes) {
    budgetOverride.store(bytes, std::memory_order_relaxed);
}

std::uint64_t CommittedImportPinnedBytes() {
    return pinnedBytes.load(std::memory_order_relaxed);
}

CommittedImportStats CommittedImportCounters() {
    auto& state = CommittedState();
    std::lock_guard lock(state.mutex);
    return {state.views.size(), state.viewBytes, pinnedBytes.load(std::memory_order_relaxed), state.retired.size(), state.device != VK_NULL_HANDLE ? budgetOf(state.context) : 0, state.deferred, state.pendingWaits, state.pendingWaitNanoseconds, state.cancelled, state.pendingBytes};
}

void NoteGuestMapping(std::uint64_t address, std::size_t bytes, GuestArena::MappingChange change, std::uint32_t protection) {
    if (bytes == 0 || !attached.load(std::memory_order_acquire)) return;
    if (change != GuestArena::MappingChange::Released && !importableMapping(change, protection)) return;
    if (change == GuestArena::MappingChange::Released) {
        auto& state = CommittedState();
        std::lock_guard lock(state.mutex);
        if (!overlapsLocked(state, address, address + bytes) && !pendingOverlaps(state, address, address + bytes)) return;
    }
    GuestMemory::RunWithStack(ImportStackBytes, [&] {
        switch (change) {
        case GuestArena::MappingChange::Committed:
            importFresh(address, bytes);
            return;
        case GuestArena::MappingChange::Mapped:
            importShared(address, bytes);
            return;
        case GuestArena::MappingChange::Released:
            releaseRange(address, bytes);
            return;
        }
    });
}

void AttachCommittedImports(const Context& context, bool watched) {
    if (context.hostImportAlignment == 0) return;
    GuestMemory::SetMappingListener(&NoteGuestMapping);
    auto& state = CommittedState();
    std::unique_lock lock(state.mutex);
    if (state.device == context.device) return;
    if (state.device != VK_NULL_HANDLE) {
        stopImports(state, lock);
        auto& parked = state.parked[state.device];
        parked.context = state.context;
        parked.views.merge(state.views);
        for (auto& [base, view] : state.views) destroyView(state.context, view);
        for (auto& node : state.retired) parked.retired.push_back(std::move(node));
        state.views.clear();
        state.retired.clear();
        state.viewBytes = 0;
        BumpHostImportsEpoch();
    }
    state.device = context.device;
    state.context = context;
    state.watched = watched;
    attached.store(true, std::memory_order_release);
#ifdef _WIN32
    GuestArena::GuestArenaSetSharedImports_nid_postfix(true);
#endif
    struct Walk {
        Committed& state;
        std::exception_ptr failure;
    } walk{state, nullptr};
    GuestArena::GuestArenaForEachMapping_nid_postfix([](void* context, std::uintptr_t address, std::size_t bytes, GuestArena::MappingChange change, std::uint32_t protection) {
        auto& walk = *static_cast<Walk*>(context);
        if (walk.failure) return;
        try {
            if (importableMapping(change, protection)) {
                for (const auto& [begin, end] : uncovered(walk.state, address, address + bytes)) insert(walk.state, importRange(walk.state.context, begin, end - begin, change == GuestArena::MappingChange::Mapped));
            }
        } catch (...) {
            walk.failure = std::current_exception();
        }
    }, &walk);
    if (walk.failure) std::rethrow_exception(walk.failure);
}

void DetachCommittedImports(VkDevice device) {
    auto& state = CommittedState();
    std::unique_lock lock(state.mutex);
    if (device == VK_NULL_HANDLE) return;
    if (state.device == device) stopImports(state, lock);
    if (const auto parked = state.parked.find(device); parked != state.parked.end()) {
        destroyViews(parked->second.context, parked->second.views, parked->second.retired);
        state.parked.erase(parked);
    }
    if (state.device != device) return;
    destroyViews(state.context, state.views, state.retired);
    state.viewBytes = 0;
    attached.store(false, std::memory_order_release);
#ifdef _WIN32
    GuestArena::GuestArenaSetSharedImports_nid_postfix(false);
#endif
    state.device = VK_NULL_HANDLE;
    state.context = Context{};
    BumpHostImportsEpoch();
}

const HostImport* CommittedImportCovering(VkDevice device, std::uint64_t begin, std::uint64_t end) {
    auto& state = CommittedState();
    std::unique_lock lock(state.mutex);
    if (device == VK_NULL_HANDLE || state.device != device || end <= begin) return nullptr;
    awaitImports(state, lock, begin, end);
    if (state.device != device) return nullptr;
    return coveringLocked(state, begin, end);
}

bool CommittedImportsOverlap(VkDevice device, std::uint64_t begin, std::uint64_t end) {
    auto& state = CommittedState();
    std::unique_lock lock(state.mutex);
    if (device == VK_NULL_HANDLE || state.device != device || end <= begin) return false;
    awaitImports(state, lock, begin, end);
    if (state.device != device) return false;
    return overlapsLocked(state, begin, end);
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> CommittedImportRanges(VkDevice device, std::uint64_t& generation) {
    auto& state = CommittedState();
    std::lock_guard lock(state.mutex);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    if (device == VK_NULL_HANDLE || state.device != device) return ranges;
    generation = state.generation;
    ranges.reserve(state.views.size() + state.pending.size());
    const auto add = [&](std::uint64_t begin, std::uint64_t end) {
        if (!ranges.empty() && ranges.back().second == begin) ranges.back().second = end;
        else ranges.emplace_back(begin, end);
    };
    auto pending = state.pending.begin();
    for (const auto& [base, view] : state.views) {
        for (; pending != state.pending.end() && pending->first < base; ++pending) add(pending->first, pending->second.end);
        add(view.base, view.base + view.bytes);
    }
    for (; pending != state.pending.end(); ++pending) add(pending->first, pending->second.end);
    return ranges;
}

std::uint64_t CommittedImportGeneration(VkDevice device) {
    auto& state = CommittedState();
    std::lock_guard lock(state.mutex);
    return device == VK_NULL_HANDLE || state.device != device ? 0 : state.generation;
}

std::optional<ImportedPage> ImportOf(VkDevice device, std::uint64_t page) {
    const auto pageBegin = page << GuestMemory::ResidencyPageShift;
    const auto pageEnd = pageBegin + PageBytes;
    ImportedPage result;
    {
        auto& state = CommittedState();
        std::unique_lock lock(state.mutex);
        if (device == VK_NULL_HANDLE || state.device != device) return std::nullopt;
        awaitImports(state, lock, pageBegin, pageEnd);
        if (state.device != device) return std::nullopt;
        const auto* view = coveringLocked(state, pageBegin, pageBegin + 1);
        if (view == nullptr) return std::nullopt;
        result.buffer = view->buffer;
        result.offset = pageBegin - view->base;
        result.address = view->address + result.offset;
        result.bytes = std::min(pageEnd, view->base + view->bytes) - pageBegin;
        result.serial = view->serial;
    }
    const auto end = pageBegin + result.bytes;
    for (auto cursor = pageBegin; cursor < end;) {
        const auto range = GuestAllocations::GuestAllocationsLookup_nid_postfix(static_cast<std::uintptr_t>(cursor), false);
        if (range == nullptr || range->writable) {
            result.cpuWritable = true;
            return result;
        }
        cursor = range->address + range->bytes;
    }
    result.cpuWritable = false;
    return result;
}

void ReleaseRetiredCommittedImports() {
    if (!GuestMemory::GpuMutex().HeldByThisThread()) return;
    auto& state = CommittedState();
    std::vector<ViewNode> retired;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    {
        std::lock_guard lock(state.mutex);
        if (state.retired.empty() || state.device == VK_NULL_HANDLE) return;
        retired.swap(state.retired);
        device = state.device;
        destroyBuffer = state.context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    }
    auto* recorder = Recorder::Active();
    for (auto& node : retired) {
        auto holder = std::make_shared<RetiredView>(device, destroyBuffer, std::move(node));
        RetainRetiredImport(holder, &holder->node.mapped());
        if (recorder != nullptr) recorder->KeepThroughHold(std::move(holder));
    }
}

}
