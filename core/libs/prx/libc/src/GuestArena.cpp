#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libc/include/WindowsMappings.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#ifndef _WIN32
#include <cerrno>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include <cstdio>
#include <iterator>
#include <map>
#include <set>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace GuestArena {
namespace {

std::atomic<std::uint64_t> commitGeneration{1};
std::atomic<void (*)(std::uintptr_t, std::size_t, std::uint64_t, MappingChange, std::uint32_t)> mappingObserver{nullptr};

void notifyMapping(std::uintptr_t address, std::size_t bytes, std::uint64_t generation, MappingChange change, std::uint32_t protection) {
    if (const auto callback = mappingObserver.load(std::memory_order_acquire)) callback(address, bytes, generation, change, protection);
}

#ifdef _WIN32
void notifyRefusalClears();

void notifyDemoted() {
    for (const auto& run : WindowsMappings::Get().TakeDemoted()) {
        const auto generation = commitGeneration.load(std::memory_order_acquire);
        notifyMapping(run.begin, run.end - run.begin, generation, MappingChange::Released, 0);
        notifyMapping(run.begin, run.end - run.begin, generation, MappingChange::Mapped, run.protection);
    }
    notifyRefusalClears();
}

std::atomic<void (*)(const std::uintptr_t*, std::size_t)> writeGuardObserver{nullptr};
std::atomic<void (*)(std::uintptr_t)> accessGuardObserver{nullptr};

void notifyGuarded(const std::vector<std::uintptr_t>& pages) {
    if (pages.empty()) return;
    if (const auto callback = writeGuardObserver.load(std::memory_order_acquire)) callback(pages.data(), pages.size());
}

std::atomic<void (*)(std::uintptr_t, std::size_t, std::uint32_t)> refusalObserver{nullptr};

void notifyRefusalClears() {
    const auto clears = WindowsMappings::Get().TakeRefusalClears();
    const auto callback = refusalObserver.load(std::memory_order_acquire);
    if (callback == nullptr) return;
    for (const auto& clear : clears) callback(clear.address, clear.bytes, clear.cleared);
}
#endif

#ifdef ANYPS5_ENABLE_ASAN
constexpr std::uintptr_t GuestArenaOffset = 0x0000100000000000ull;
#else
constexpr std::uintptr_t GuestArenaOffset = 0;
#endif
constexpr std::uintptr_t ArenaStart = 0x0000000200000000ull + GuestArenaOffset;
constexpr std::uintptr_t SystemReservedStart = 0x00000007FFFFC000ull + GuestArenaOffset;
constexpr std::uintptr_t SystemReservedEnd = 0x0000001000000000ull + GuestArenaOffset;
constexpr std::uintptr_t ApplicationAreaEnd = 0x000000FC00000000ull + GuestArenaOffset;

std::uintptr_t alignUp(std::uintptr_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
}

class Arena {
public:
    static Arena& Get() {
        static auto* arena = new Arena;
        return *arena;
    }

    bool Available() const {
        return _base != 0;
    }

    bool Contains(const void* pointer, std::size_t bytes) const {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        return _base != 0 && address >= _base && bytes <= _end - _base && address - _base <= (_end - _base) - bytes;
    }

    void* Allocate(std::size_t bytes, std::size_t alignment) {
        return AllocateAtOrAbove(0, bytes, alignment);
    }

    void* AllocateAtOrAbove(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) {
        RequireAllocatable(alignment);
        std::lock_guard lock(_lock);
        if (hint >= _end) throw std::runtime_error("mapping address hint is above the guest address space arena");
        return Claim(FindAtOrAbove(hint, bytes, alignment), bytes);
    }

    void* AllocateFromHint(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) {
        RequireAllocatable(alignment);
        std::lock_guard lock(_lock);
        auto candidate = hint < _end ? FindAtOrAbove(hint, bytes, alignment) : 0;
        if (candidate == 0 && hint > _base) candidate = FindAtOrAbove(_base, bytes, alignment);
        return Claim(candidate, bytes);
    }

    std::uintptr_t HighestFree(std::size_t bytes, std::size_t alignment) {
        if (_base == 0) throw std::runtime_error("guest address space arena is unavailable");
        if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) throw std::invalid_argument("invalid guest arena window");
        std::lock_guard lock(_lock);
        auto limit = _end;
        while (limit >= _base && bytes <= limit - _base) {
            const auto candidate = (limit - bytes) & ~(alignment - 1);
            if (candidate < _base) break;
            const auto next = _used.lower_bound(candidate + bytes);
            if (next == _used.begin() || std::prev(next)->second <= candidate) return candidate;
            limit = std::prev(next)->first;
        }
        return 0;
    }

    bool MarkUsedForCommit(const void* pointer, std::size_t bytes) {
        std::lock_guard lock(_lock);
        const auto start = reinterpret_cast<std::uintptr_t>(pointer);
        const auto end = start + bytes;
        if (OverlapsHostRegion(start, bytes)) {
            char message[160];
            std::snprintf(message, sizeof(message), "guest arena commit 0x%llx+0x%zx overlaps a host region", static_cast<unsigned long long>(start), bytes);
            throw std::runtime_error(message);
        }
        const auto next = _used.lower_bound(start);
        auto covering = _used.upper_bound(start);
        if (covering != _used.begin()) {
            --covering;
            auto reach = covering->second;
            for (auto chain = std::next(covering); reach < end && chain != _used.end() && chain->first == reach; ++chain) reach = chain->second;
            if (reach >= end) return false;
        }
        if ((next != _used.end() && next->first < end) || (next != _used.begin() && std::prev(next)->second > start)) {
            char message[160];
            std::snprintf(message, sizeof(message), "guest arena commit 0x%llx+0x%zx overlaps a used range", static_cast<unsigned long long>(start), bytes);
            throw std::runtime_error(message);
        }
        _used.emplace(start, end);
        return true;
    }

    void MarkUsed(const void* pointer, std::size_t bytes) {
        std::lock_guard lock(_lock);
        const auto start = reinterpret_cast<std::uintptr_t>(pointer);
        const auto next = _used.lower_bound(start);
        if ((next != _used.end() && next->first < start + bytes) || (next != _used.begin() && std::prev(next)->second > start)) {
            char message[160];
            std::snprintf(message, sizeof(message), "fixed guest mapping 0x%llx+0x%zx overlaps %s", static_cast<unsigned long long>(start), bytes, OverlapsHostRegion(start, bytes) ? "a host region in the guest arena" : "a guest arena range");
            throw std::runtime_error(message);
        }
        _used.emplace(start, start + bytes);
    }

    void Release(const void* pointer, std::size_t bytes) {
        std::lock_guard lock(_lock);
        const auto start = reinterpret_cast<std::uintptr_t>(pointer);
        const auto end = start + bytes;
        auto it = _used.upper_bound(start);
        if (it != _used.begin()) --it;
        while (it != _used.end() && it->first < end) {
            const auto rangeStart = it->first;
            const auto rangeEnd = it->second;
            if (rangeEnd <= start) {
                ++it;
                continue;
            }
            it = _used.erase(it);
            if (rangeStart < start) _used.emplace(rangeStart, start);
            if (rangeEnd > end) _used.emplace(end, rangeEnd);
        }
        for (const auto& [holeStart, holeEnd] : _holes) {
            if (holeStart < end && start < holeEnd) _used.emplace(holeStart, holeEnd);
        }
    }

private:
    Arena() {
#if defined(__linux__)
        SeedLinuxUsedRanges();
#elif defined(_WIN32)
        _writeWatched = std::getenv("APS5_NO_WRITE_WATCH") == nullptr;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const std::uintptr_t granularity = system.dwAllocationGranularity;
        const std::uintptr_t end = ApplicationAreaEnd;
        std::uint32_t retries = 0;
        for (std::uintptr_t cursor = ArenaStart; cursor < end;) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) == 0) throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "query the guest arena range");
            const auto regionEnd = std::min(end, reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize);
            const auto first = info.State == MEM_FREE ? std::min(regionEnd, alignUp(cursor, granularity)) : regionEnd;
            const auto last = std::max(first, regionEnd & ~(granularity - 1));
            if (first < last && WindowsMappings::Get().Reserve(reinterpret_cast<void*>(first), last - first) == nullptr) {
                const auto error = GetLastError();
                if (error == ERROR_INVALID_ADDRESS && ++retries < 64u) continue;
                throw std::system_error(static_cast<int>(error), std::system_category(), "reserve the guest arena range");
            }
            retries = 0;
            if (cursor < first) _holes.emplace_back(cursor, first);
            if (last < regionEnd) _holes.emplace_back(last, regionEnd);
            cursor = regionEnd;
        }
        _hostRegions = _holes;
        _holes.emplace_back(SystemReservedStart, SystemReservedEnd);
        for (const auto& [holeStart, holeEnd] : _holes) _used.emplace(holeStart, holeEnd);
        _base = ArenaStart;
        _end = end;
#endif
    }

#if defined(__linux__)
    void SeedLinuxUsedRanges() {
        std::ifstream maps("/proc/self/maps");
        if (!maps.is_open()) throw std::runtime_error("guest arena: cannot open /proc/self/maps");
        std::string line;
        while (std::getline(maps, line)) {
            std::istringstream fields(line);
            std::uintptr_t begin = 0, end = 0;
            char dash = 0;
            if (!(fields >> std::hex >> begin >> dash >> end) || dash != '-' || end <= begin) throw std::runtime_error("guest arena: malformed /proc/self/maps line: " + line);
            if (end <= ArenaStart || begin >= ApplicationAreaEnd) continue;
            const auto first = std::max(begin, ArenaStart);
            const auto last = std::min(end, ApplicationAreaEnd);
            _used.emplace(first, last);
            _hostRegions.emplace_back(first, last);
        }
        _used.emplace(SystemReservedStart, SystemReservedEnd);
        _hostRegions.emplace_back(SystemReservedStart, SystemReservedEnd);
        ReserveFreeRanges();
        _base = ArenaStart;
        _end = ApplicationAreaEnd;
    }

    static void TryReserve(std::vector<std::pair<std::uintptr_t, std::uintptr_t>>& occupied, std::uintptr_t from, std::uintptr_t to) {
        constexpr std::uintptr_t kChunk = 1ULL << 30;
        for (std::uintptr_t cursor = from; cursor < to; cursor += kChunk) {
            const auto end = std::min(cursor + kChunk, to);
            void* const wanted = reinterpret_cast<void*>(cursor);
            void* const placed = mmap(wanted, end - cursor, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
            if (placed == wanted) continue;
            const int failed = placed == MAP_FAILED ? errno : EINVAL;
            if (placed != MAP_FAILED) munmap(placed, end - cursor);
            if (failed != EEXIST) {
                char message[128];
                std::snprintf(message, sizeof(message), "guest arena: cannot reserve 0x%llx+0x%llx",
                    static_cast<unsigned long long>(cursor), static_cast<unsigned long long>(end - cursor));
                throw std::system_error(failed, std::system_category(), message);
            }
            occupied.emplace_back(cursor, end);
        }
    }

    void ReserveFreeRanges() {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> occupied;
        std::uintptr_t cursor = ArenaStart;
        for (const auto& [usedStart, usedEnd] : _used) {
            if (usedStart > ApplicationAreaEnd) break;
            if (usedStart > cursor) TryReserve(occupied, cursor, std::min(usedStart, ApplicationAreaEnd));
            if (usedEnd > cursor) cursor = usedEnd;
            if (cursor >= ApplicationAreaEnd) break;
        }
        if (cursor < ApplicationAreaEnd) TryReserve(occupied, cursor, ApplicationAreaEnd);
        for (const auto& [from, to] : occupied) {
            _used.emplace(from, to);
            _hostRegions.emplace_back(from, to);
        }
    }
#endif

    std::mutex _lock;
    std::map<std::uintptr_t, std::uintptr_t> _used;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> _holes;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> _hostRegions;
    void RequireAllocatable(std::size_t alignment) const {
        if (_base == 0) throw std::runtime_error("guest address space arena is unavailable");
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) throw std::invalid_argument("invalid guest arena alignment");
    }

    std::uintptr_t FindAtOrAbove(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) const {
        std::uintptr_t candidate = alignUp(std::max(_base, hint), alignment);
        auto it = _used.upper_bound(candidate);
        if (it != _used.begin() && std::prev(it)->second > candidate) --it;
        for (; it != _used.end(); ++it) {
            const auto& [start, end] = *it;
            if (candidate <= _end && bytes <= _end - candidate && candidate + bytes <= start) break;
            candidate = std::max(candidate, alignUp(end, alignment));
        }
        if (candidate > _end || bytes > _end - candidate) return 0;
        return candidate;
    }

    void* Claim(std::uintptr_t candidate, std::size_t bytes) {
        if (candidate == 0) throw std::runtime_error("guest address space arena exhausted");
        _used.emplace(candidate, candidate + bytes);
        return reinterpret_cast<void*>(candidate);
    }

    std::uintptr_t _base = 0;
    std::uintptr_t _end = 0;
    bool _writeWatched = false;

public:
    std::uintptr_t Base() const { return _base; }
    std::size_t Size() const { return _end - _base; }
    bool WriteWatched() const { return _writeWatched; }
    bool OverlapsHostRegion(std::uintptr_t start, std::size_t bytes) const {
        for (const auto& [regionStart, regionEnd] : _hostRegions) {
            if (regionStart < start + bytes && start < regionEnd) return true;
        }
        return false;
    }
    bool OverlapsHole(std::uintptr_t start, std::size_t bytes) const {
        for (const auto& [holeStart, holeEnd] : _holes) {
            if (holeStart < start + bytes && start < holeEnd) return true;
        }
        return false;
    }
};

const bool g_reserved = (Arena::Get(), true);

}

bool GuestArenaAvailable_nid_postfix() {
    return Arena::Get().Available();
}

bool GuestArenaContains_nid_postfix(const void* pointer, std::size_t bytes) {
    return Arena::Get().Contains(pointer, bytes);
}

void* GuestArenaAllocate_nid_postfix(std::size_t bytes, std::size_t alignment) {
    return Arena::Get().Allocate(bytes, alignment);
}

void* GuestArenaAllocateAtOrAbove_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) {
    return Arena::Get().AllocateAtOrAbove(hint, bytes, alignment);
}

void* GuestArenaAllocateFromHint_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) {
    return Arena::Get().AllocateFromHint(hint, bytes, alignment);
}

std::uintptr_t GuestArenaHighestFree_nid_no_patch(std::size_t bytes, std::size_t alignment) {
    return Arena::Get().HighestFree(bytes, alignment);
}

void GuestArenaMarkUsed_nid_postfix(const void* pointer, std::size_t bytes) {
    Arena::Get().MarkUsed(pointer, bytes);
}

void GuestArenaRelease_nid_postfix(const void* pointer, std::size_t bytes) {
    Arena::Get().Release(pointer, bytes);
}

void GuestArenaRange_nid_postfix(std::uintptr_t* base, std::size_t* bytes) {
    *base = Arena::Get().Base();
    *bytes = Arena::Get().Size();
}

namespace {

std::invalid_argument OutsideArena(const char* operation, const void* pointer, std::size_t bytes) {
    char message[128];
    std::snprintf(message, sizeof(message), "%s 0x%llx+0x%zx outside the guest arena", operation, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(pointer)), bytes);
    return std::invalid_argument(message);
}

}

#ifdef _WIN32
void GuestArenaSetProtection_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint32_t protection) {
    WindowsMappings::Get().SetProtection(address, bytes, protection);
    notifyGuarded(WindowsMappings::Get().TakeGuardReports());
}

bool GuestArenaHandleWrite_nid_postfix(std::uintptr_t address) {
    const bool handled = WindowsMappings::Get().HandleWrite(address);
    if (handled) notifyGuarded(WindowsMappings::Get().TakeGuardReports());
    return handled;
}

GuardRefusal GuestArenaGuardWrites_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!Arena::Get().Contains(reinterpret_cast<const void*>(address), bytes)) throw OutsideArena("write guard", reinterpret_cast<const void*>(address), bytes);
    return WindowsMappings::Get().Guard(address, bytes);
}

void GuestArenaUnguardWrites_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    WindowsMappings::Get().Unguard(address, bytes);
}

bool GuestArenaWriteGuarded_nid_postfix(std::uintptr_t page) {
    return WindowsMappings::Get().IsGuarded(page);
}

GuardRefusal GuestArenaGuardAccess_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!Arena::Get().Contains(reinterpret_cast<const void*>(address), bytes)) throw OutsideArena("access guard", reinterpret_cast<const void*>(address), bytes);
    return WindowsMappings::Get().GuardAccess(address, bytes);
}

void GuestArenaNoteImport_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (bytes == 0 || !Arena::Get().Contains(reinterpret_cast<const void*>(address), bytes)) return;
    WindowsMappings::Get().NoteImport(address, bytes);
    notifyGuarded(WindowsMappings::Get().TakeGuardReports());
}

std::uint64_t GuestArenaImportStamp_nid_postfix() {
    return WindowsMappings::Get().ImportStamp();
}

void GuestArenaSettleImports_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint64_t stamp) {
    if (bytes == 0 || !Arena::Get().Contains(reinterpret_cast<const void*>(address), bytes)) return;
    WindowsMappings::Get().SettleImports(address, bytes, stamp);
    notifyRefusalClears();
}

bool GuestArenaImportSettled_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (bytes == 0 || !Arena::Get().Contains(reinterpret_cast<const void*>(address), bytes)) return true;
    return WindowsMappings::Get().ImportSettled(address, bytes);
}

void GuestArenaUnguardAccess_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    WindowsMappings::Get().UnguardAccess(address, bytes);
}

bool GuestArenaAccessGuarded_nid_postfix(std::uintptr_t page) {
    return WindowsMappings::Get().AccessGuarded(page);
}

bool GuestArenaHandleAccess_nid_postfix(std::uintptr_t address, bool write) {
    static_cast<void>(write);
    if (!Arena::Get().Contains(reinterpret_cast<const void*>(address), 1) || !WindowsMappings::Get().AccessGuarded(address)) return false;
    const auto observer = accessGuardObserver.load(std::memory_order_acquire);
    if (observer == nullptr) return false;
    observer(address);
    return !WindowsMappings::Get().AccessGuarded(address);
}

void GuestArenaSetAccessGuardObserver_nid_postfix(void (*callback)(std::uintptr_t page)) {
    accessGuardObserver.store(callback, std::memory_order_release);
}

void GuestArenaSetWriteGuardObserver_nid_postfix(void (*callback)(const std::uintptr_t*, std::size_t)) {
    writeGuardObserver.store(callback, std::memory_order_release);
}

void GuestArenaSetGuardRefusalObserver_nid_postfix(void (*callback)(std::uintptr_t, std::size_t, std::uint32_t)) {
    refusalObserver.store(callback, std::memory_order_release);
}

void GuestArenaAliases_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*callback)(void*, std::uintptr_t, std::size_t), void* context) {
    for (const auto& [alias, size] : WindowsMappings::Get().Aliases(address, bytes)) callback(context, alias, size);
}

bool GuestArenaBacking_nid_postfix(std::uintptr_t page, std::uint64_t* section, std::uint64_t* offset, bool* uniform, bool* aliased) {
    return WindowsMappings::Get().Backing(page, section, offset, uniform, aliased);
}

bool GuestArenaHandleRead_nid_postfix(std::uintptr_t address) {
    return WindowsMappings::Get().HandleRead(address);
}

void GuestArenaPinWritable_nid_postfix(const void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) return;
    WindowsMappings::Get().Pin(reinterpret_cast<std::uintptr_t>(pointer), bytes);
    notifyGuarded(WindowsMappings::Get().TakeGuardReports());
}

void GuestArenaUnpinWritable_nid_postfix(const void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) return;
    WindowsMappings::Get().Unpin(reinterpret_cast<std::uintptr_t>(pointer), bytes);
    notifyRefusalClears();
}

bool GuestArenaProtection_nid_postfix(std::uintptr_t address, std::uint32_t* protection, std::uintptr_t* end) {
    return WindowsMappings::Get().Protection(address, protection, end);
}

bool GuestArenaCollectWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
    return WindowsMappings::Get().Collect(address, bytes, pages, count, clear);
}

bool GuestArenaCollectArmedWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
    return WindowsMappings::Get().Collect(address, bytes, pages, count, clear, WindowsMappings::Collection::Armed);
}

bool GuestArenaCollectFreshWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
    return WindowsMappings::Get().Collect(address, bytes, pages, count, clear, WindowsMappings::Collection::Fresh);
}

void GuestArenaCollectWork_nid_postfix(std::uint64_t* systemCalls, std::uint64_t* pieces) {
    WindowsMappings::Get().CollectWork(systemCalls, pieces);
}

bool GuestArenaHostRegionOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    return Arena::Get().OverlapsHostRegion(address, bytes);
}

void GuestArenaForEachMapping_nid_postfix(void (*callback)(void*, std::uintptr_t, std::size_t, MappingChange, std::uint32_t), void* context) {
    auto& arena = Arena::Get();
    const auto end = arena.Base() + arena.Size();
    std::uintptr_t runBegin = 0;
    std::uintptr_t runEnd = 0;
    MappingChange runChange = MappingChange::Committed;
    std::uint32_t runProtection = 0;
    const auto flush = [&] {
        if (runEnd > runBegin) callback(context, runBegin, runEnd - runBegin, runChange, runProtection);
        runBegin = runEnd = 0;
    };
    for (auto cursor = arena.Base(); cursor < end;) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) == 0) throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "query the guest arena mappings");
        const auto regionEnd = std::min(end, reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize);
        const bool guest = info.State == MEM_COMMIT && (info.Type == MEM_PRIVATE || info.Type == MEM_MAPPED) && !arena.OverlapsHole(cursor, regionEnd - cursor);
        if (!guest) {
            flush();
        } else {
            const auto change = info.Type == MEM_MAPPED ? MappingChange::Mapped : MappingChange::Committed;
            const auto protection = static_cast<std::uint32_t>(info.Protect);
            if (runEnd != cursor || change != runChange || (change == MappingChange::Committed && protection != runProtection)) {
                flush();
                runBegin = cursor;
                runChange = change;
                runProtection = protection;
            }
            runEnd = regionEnd;
        }
        cursor = regionEnd;
    }
    flush();
}

void GuestArenaCommit_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection, std::size_t granule) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("commit", pointer, bytes);
    const auto generation = commitGeneration.fetch_add(1, std::memory_order_acq_rel) + 1;
    const auto created = WindowsMappings::Get().Commit(pointer, bytes, protection, granule, Arena::Get().WriteWatched());
    notifyGuarded(WindowsMappings::Get().TakeGuardReports());
    for (const auto& [address, size] : created) notifyMapping(address, size, generation, MappingChange::Committed, protection);
    notifyDemoted();
}

void GuestArenaReset_nid_postfix(void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("reset", pointer, bytes);
    WindowsMappings::Get().Reset(pointer, bytes);
    notifyMapping(reinterpret_cast<std::uintptr_t>(pointer), bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Released, 0);
    notifyDemoted();
}

void GuestArenaMap_nid_postfix(void* pointer, std::size_t bytes, MappingHandle section, std::uint64_t offset, std::uint32_t protection) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("shared mapping", pointer, bytes);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    notifyMapping(address, bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Released, 0);
    WindowsMappings::Get().Map(pointer, bytes, section, offset, protection, Arena::Get().WriteWatched(), false);
    notifyMapping(address, bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Mapped, protection);
    notifyDemoted();
}

void GuestArenaMapDirect_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("direct memory mapping", pointer, bytes);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    notifyMapping(address, bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Released, 0);
    WindowsMappings::Get().Map(pointer, bytes, section, offset, protection, Arena::Get().WriteWatched(), true);
    notifyMapping(address, bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Mapped, protection);
    notifyDemoted();
}

void* GuestArenaMapAlias_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    auto* alias = WindowsMappings::Get().MapAlias(address, bytes);
    static_cast<void>(WindowsMappings::Get().TakeDemoted());
    return alias;
}

void GuestArenaUnmapAlias_nid_postfix(void* alias) {
    WindowsMappings::Get().UnmapAlias(alias);
}

bool GuestArenaResidentSpan_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uintptr_t* start, std::uintptr_t* end) {
    return WindowsMappings::Get().ResidentSpan(address, bytes, start, end);
}

void GuestArenaForgetSection_nid_postfix(void* section) {
    WindowsMappings::Get().ForgetSection(section);
}

void GuestArenaSetSharedImports_nid_postfix(bool imported) {
    WindowsMappings::Get().SetSharedImports(imported);
}
#endif

#if defined(__linux__)
namespace {
int PosixProtection(std::uint32_t windowsProtection) {
    switch (windowsProtection) {
    case 0x01: return PROT_NONE;
    case 0x02: return PROT_READ;
    case 0x04: return PROT_READ | PROT_WRITE;
    case 0x08: return PROT_READ | PROT_WRITE;
    case 0x10: return PROT_EXEC;
    case 0x20: return PROT_READ | PROT_EXEC;
    case 0x40: return PROT_READ | PROT_WRITE | PROT_EXEC;
    default: throw std::runtime_error("guest arena: unmapped protection value");
    }
}

void* PlaceAt(void* pointer, std::size_t bytes, std::uint32_t protection) {
    return mmap(pointer, bytes, PosixProtection(protection), MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
}

enum class PlacementKind : std::uint8_t { Anonymous, Shared };

struct Placement {
    std::uintptr_t end;
    PlacementKind kind;
    std::uint32_t protection;
};

struct Placements {
    std::mutex mutex;
    std::map<std::uintptr_t, Placement> ranges;
};

Placements& PlacementTable() {
    static auto& placements = *new Placements;
    return placements;
}

void cutPlacements(std::map<std::uintptr_t, Placement>& ranges, std::uintptr_t begin, std::uintptr_t end) {
    auto it = ranges.upper_bound(begin);
    if (it != ranges.begin()) --it;
    while (it != ranges.end() && it->first < end) {
        const auto rangeBegin = it->first;
        const auto range = it->second;
        if (range.end <= begin) {
            ++it;
            continue;
        }
        it = ranges.erase(it);
        if (rangeBegin < begin) ranges.emplace(rangeBegin, Placement{begin, range.kind, range.protection});
        if (range.end > end) it = ranges.emplace(end, Placement{range.end, range.kind, range.protection}).first;
    }
}

void RecordPlacement(std::uintptr_t begin, std::size_t bytes, PlacementKind kind, std::uint32_t protection) {
    auto& table = PlacementTable();
    std::lock_guard lock(table.mutex);
    cutPlacements(table.ranges, begin, begin + bytes);
    table.ranges.emplace(begin, Placement{begin + bytes, kind, protection});
}

void ErasePlacements(std::uintptr_t begin, std::size_t bytes) {
    auto& table = PlacementTable();
    std::lock_guard lock(table.mutex);
    cutPlacements(table.ranges, begin, begin + bytes);
}

std::vector<std::pair<std::uintptr_t, Placement>> PlacementsWithin(std::uintptr_t begin, std::uintptr_t end) {
    std::vector<std::pair<std::uintptr_t, Placement>> within;
    auto& table = PlacementTable();
    std::lock_guard lock(table.mutex);
    auto it = table.ranges.upper_bound(begin);
    if (it != table.ranges.begin()) --it;
    for (; it != table.ranges.end() && it->first < end; ++it) {
        if (it->second.end <= begin) continue;
        within.emplace_back(std::max(begin, it->first), Placement{std::min(end, it->second.end), it->second.kind, it->second.protection});
    }
    return within;
}

bool OpensForImport(const Placement& placement, std::uint32_t protection) {
    return placement.protection != 0x04 && protection == 0x04;
}
struct SharedView {
    std::uintptr_t end;
    std::uint64_t backing;
    std::uint64_t offset;
};

struct BackingViews {
    std::multimap<std::uint64_t, std::pair<std::uintptr_t, std::uint64_t>> byOffset;
    std::multiset<std::uint64_t> lengths;
};

struct SharedViews {
    std::mutex mutex;
    std::map<std::uintptr_t, SharedView> views;
    std::map<std::uint64_t, BackingViews> backings;

    void insert(std::uintptr_t begin, const SharedView& view) {
        views.emplace(begin, view);
        auto& backing = backings[view.backing];
        const auto length = static_cast<std::uint64_t>(view.end - begin);
        backing.byOffset.emplace(view.offset, std::pair{begin, view.offset + length});
        backing.lengths.insert(length);
    }

    std::map<std::uintptr_t, SharedView>::iterator erase(std::map<std::uintptr_t, SharedView>::iterator it) {
        const auto key = it->second.backing;
        auto& backing = backings.at(key);
        for (auto [entry, last] = backing.byOffset.equal_range(it->second.offset); entry != last; ++entry) {
            if (entry->second.first != it->first) continue;
            backing.byOffset.erase(entry);
            break;
        }
        backing.lengths.erase(backing.lengths.find(static_cast<std::uint64_t>(it->second.end - it->first)));
        if (backing.byOffset.empty()) backings.erase(key);
        return views.erase(it);
    }

    template <class Fn>
    void forEachOverlapping(std::uint64_t key, std::uintptr_t exceptBegin, std::uint64_t low, std::uint64_t high, Fn&& fn) const {
        const auto found = backings.find(key);
        if (found == backings.end()) return;
        const auto& backing = found->second;
        const auto longest = backing.lengths.empty() ? 0 : *backing.lengths.rbegin();
        const auto from = low > longest ? low - longest : 0;
        for (auto it = backing.byOffset.lower_bound(from); it != backing.byOffset.end() && it->first < high; ++it) {
            const auto [otherBegin, otherEnd] = it->second;
            if (otherBegin == exceptBegin || otherEnd <= low) continue;
            if (!fn(otherBegin, it->first, otherEnd)) return;
        }
    }
};

SharedViews& Shared() {
    static auto& shared = *new SharedViews;
    return shared;
}

std::uint64_t BackingOf(int descriptor) {
    struct stat status {};
    if (fstat(descriptor, &status) != 0) throw std::system_error(errno, std::generic_category(), "guest arena: cannot identify the shared mapping's backing");
    return (static_cast<std::uint64_t>(status.st_dev) << 32) ^ static_cast<std::uint64_t>(status.st_ino);
}

void forgetViews(SharedViews& shared, std::uintptr_t begin, std::uintptr_t end) {
    auto& views = shared.views;
    auto it = views.upper_bound(begin);
    if (it != views.begin()) --it;
    while (it != views.end() && it->first < end) {
        const auto viewBegin = it->first;
        const auto view = it->second;
        if (view.end <= begin) {
            ++it;
            continue;
        }
        it = shared.erase(it);
        if (viewBegin < begin) shared.insert(viewBegin, SharedView{begin, view.backing, view.offset});
        if (view.end > end) {
            shared.insert(end, SharedView{view.end, view.backing, view.offset + (end - viewBegin)});
            it = views.find(end);
        }
    }
}

void ForgetSharedViews(std::uintptr_t begin, std::size_t bytes) {
    auto& shared = Shared();
    std::lock_guard lock(shared.mutex);
    forgetViews(shared, begin, begin + bytes);
}

void RecordSharedView(std::uintptr_t begin, std::size_t bytes, std::uint64_t backing, std::uint64_t offset) {
    auto& shared = Shared();
    std::lock_guard lock(shared.mutex);
    forgetViews(shared, begin, begin + bytes);
    shared.insert(begin, SharedView{begin + bytes, backing, offset});
}

const SharedView* viewAt(const std::map<std::uintptr_t, SharedView>& views, std::uintptr_t address, std::uintptr_t* begin) {
    auto it = views.upper_bound(address);
    if (it == views.begin()) return nullptr;
    --it;
    if (it->second.end <= address) return nullptr;
    *begin = it->first;
    return &it->second;
}

constexpr std::uintptr_t SharedPageBytes = 0x4000;
constexpr std::uintptr_t ResidencyPageBytes = 0x10000;
}
#endif

#if defined(__linux__)
void GuestArenaSetProtection_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint32_t protection) {
    if (!Arena::Get().Contains(reinterpret_cast<void*>(address), bytes)) throw OutsideArena("set protection", reinterpret_cast<void*>(address), bytes);
    const auto native = PosixProtection(protection);
    const auto placed = PlacementsWithin(address, address + bytes);
    if (mprotect(reinterpret_cast<void*>(address), bytes, native) != 0)
        throw std::system_error(errno, std::generic_category(), "guest arena protection failed");
    const auto generation = commitGeneration.fetch_add(1, std::memory_order_acq_rel) + 1;
    try {
        for (const auto& [begin, placement] : placed) {
            if (!OpensForImport(placement, protection)) continue;
            notifyMapping(begin, static_cast<std::size_t>(placement.end - begin), generation, placement.kind == PlacementKind::Anonymous ? MappingChange::Committed : MappingChange::Mapped, protection);
        }
    } catch (...) {
        for (const auto& [begin, placement] : placed) mprotect(reinterpret_cast<void*>(begin), placement.end - begin, PosixProtection(placement.protection));
        throw;
    }
    for (const auto& [begin, placement] : placed) RecordPlacement(begin, placement.end - begin, placement.kind, protection);
}

void GuestArenaCommit_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection, std::size_t granule) {
    (void)granule;
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("commit", pointer, bytes);
    static_cast<void>(PosixProtection(protection));
    const bool marked = Arena::Get().MarkUsedForCommit(pointer, bytes);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto generation = commitGeneration.fetch_add(1, std::memory_order_acq_rel) + 1;
    try {
        ForgetSharedViews(reinterpret_cast<std::uintptr_t>(pointer), bytes);
        if (PlaceAt(pointer, bytes, protection) == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "guest arena commit failed");
        RecordPlacement(address, bytes, PlacementKind::Anonymous, protection);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(pointer, bytes);
        notifyMapping(address, bytes, generation, MappingChange::Committed, protection);
    } catch (...) {
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(pointer, bytes);
        ErasePlacements(address, bytes);
        PlaceAt(pointer, bytes, 0x01);
        if (marked) Arena::Get().Release(pointer, bytes);
        throw;
    }
}

void GuestArenaReset_nid_postfix(void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("reset", pointer, bytes);
    ForgetSharedViews(reinterpret_cast<std::uintptr_t>(pointer), bytes);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(pointer, bytes);
    ErasePlacements(reinterpret_cast<std::uintptr_t>(pointer), bytes);
    if (PlaceAt(pointer, bytes, 0x01) == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "guest arena reset failed");
    notifyMapping(reinterpret_cast<std::uintptr_t>(pointer), bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Released, 0);
}

void GuestArenaDetach_nid_postfix(void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("detach", pointer, bytes);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(pointer, bytes);
    ErasePlacements(address, bytes);
    ForgetSharedViews(address, bytes);
    notifyMapping(address, bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Released, 0);
}

void GuestArenaMap_nid_postfix(void* pointer, std::size_t bytes, MappingHandle section, std::uint64_t offset, std::uint32_t protection) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("shared mapping", pointer, bytes);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto backing = BackingOf(section);
    notifyMapping(address, bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Released, 0);
    ForgetSharedViews(address, bytes);
    if (mmap(pointer, bytes, PosixProtection(protection), MAP_SHARED | MAP_FIXED, section, static_cast<off_t>(offset)) == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "guest arena shared mapping failed");
    RecordPlacement(address, bytes, PlacementKind::Shared, protection);
    RecordSharedView(address, bytes, backing, offset);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(pointer, bytes);
    if (protection != 0x01) notifyMapping(address, bytes, commitGeneration.load(std::memory_order_acquire), MappingChange::Mapped, protection);
}

bool GuestArenaBacking_nid_postfix(std::uintptr_t page, std::uint64_t* section, std::uint64_t* offset, bool* uniform, bool* aliased) {
    auto& shared = Shared();
    std::lock_guard lock(shared.mutex);
    const auto base = page & ~(ResidencyPageBytes - 1);
    *uniform = true;
    *aliased = false;
    std::uintptr_t firstBegin = 0;
    const auto* first = viewAt(shared.views, base, &firstBegin);
    if (first == nullptr) {
        *section = 0;
        *offset = base;
        for (auto piece = base; piece < base + ResidencyPageBytes; piece += SharedPageBytes) {
            std::uintptr_t begin = 0;
            if (viewAt(shared.views, piece, &begin) != nullptr) *uniform = false;
        }
        return false;
    }
    *section = first->backing;
    *offset = first->offset + (base - firstBegin);
    for (auto piece = base; piece < base + ResidencyPageBytes; piece += SharedPageBytes) {
        std::uintptr_t begin = 0;
        const auto* view = viewAt(shared.views, piece, &begin);
        if (view == nullptr || view->backing != *section || view->offset + (piece - begin) != *offset + (piece - base)) {
            *uniform = false;
            continue;
        }
        if (*aliased) continue;
        const auto pieceOffset = view->offset + (piece - begin);
        shared.forEachOverlapping(view->backing, begin, pieceOffset, pieceOffset + 1, [&](std::uintptr_t, std::uint64_t, std::uint64_t) {
            *aliased = true;
            return false;
        });
    }
    return true;
}

void GuestArenaAliases_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*callback)(void* context, std::uintptr_t address, std::size_t bytes), void* context) {
    std::vector<std::pair<std::uintptr_t, std::size_t>> ranges;
    {
        auto& shared = Shared();
        std::lock_guard lock(shared.mutex);
        const auto end = address + bytes;
        auto it = shared.views.upper_bound(address);
        if (it != shared.views.begin()) --it;
        for (; it != shared.views.end() && it->first < end; ++it) {
            const auto viewBegin = it->first;
            const auto& view = it->second;
            if (view.end <= address) continue;
            const auto from = std::max(address, viewBegin);
            const auto to = std::min(end, view.end);
            const auto fromOffset = view.offset + (from - viewBegin);
            const auto toOffset = view.offset + (to - viewBegin);
            shared.forEachOverlapping(view.backing, viewBegin, fromOffset, toOffset, [&](std::uintptr_t otherBegin, std::uint64_t otherOffset, std::uint64_t otherEndOffset) {
                const auto low = std::max(fromOffset, otherOffset);
                const auto high = std::min(toOffset, otherEndOffset);
                ranges.emplace_back(otherBegin + (low - otherOffset), static_cast<std::size_t>(high - low));
                return true;
            });
        }
    }
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<std::uintptr_t, std::size_t>> merged;
    for (const auto& range : ranges) {
        if (!merged.empty() && merged.back().first + merged.back().second >= range.first) merged.back().second = std::max(merged.back().first + merged.back().second, range.first + range.second) - merged.back().first;
        else merged.push_back(range);
    }
    for (const auto& [alias, size] : merged) callback(context, alias, size);
}

namespace {

std::uint32_t WindowsProtection(const std::string& permissions) {
    const bool read = permissions.size() > 0 && permissions[0] == 'r';
    const bool write = permissions.size() > 1 && permissions[1] == 'w';
    const bool execute = permissions.size() > 2 && permissions[2] == 'x';
    if (execute) return write ? 0x40u : read ? 0x20u : 0x10u;
    if (write) return 0x04u;
    return read ? 0x02u : 0x01u;
}

}

void GuestArenaForEachMapping_nid_postfix(void (*callback)(void*, std::uintptr_t, std::size_t, MappingChange, std::uint32_t), void* context) {
    auto& arena = Arena::Get();
    const auto arenaEnd = arena.Base() + arena.Size();
    std::ifstream maps("/proc/self/maps");
    if (!maps.is_open()) throw std::system_error(errno, std::generic_category(), "query the guest arena mappings");
    std::uintptr_t runBegin = 0;
    std::uintptr_t runEnd = 0;
    MappingChange runChange = MappingChange::Committed;
    std::uint32_t runProtection = 0;
    const auto flush = [&] {
        if (runEnd > runBegin) callback(context, runBegin, runEnd - runBegin, runChange, runProtection);
        runBegin = runEnd = 0;
    };
    const auto report = [&](std::uintptr_t pieceBegin, std::uintptr_t pieceEnd, MappingChange change, std::uint32_t protection) {
        if (runEnd != pieceBegin || change != runChange || protection != runProtection) {
            flush();
            runBegin = pieceBegin;
            runChange = change;
            runProtection = protection;
        }
        runEnd = pieceEnd;
    };
    std::string line;
    while (std::getline(maps, line)) {
        std::istringstream fields(line);
        std::uintptr_t begin = 0, end = 0, offset = 0;
        char dash = 0;
        std::string permissions, device, path;
        std::uint64_t inode = 0;
        if (!(fields >> std::hex >> begin >> dash >> end >> permissions >> offset >> device >> std::dec >> inode) || dash != '-' || end <= begin)
            throw std::runtime_error("guest arena: malformed /proc/self/maps line");
        fields >> path;
        if (end <= arena.Base() || begin >= arenaEnd) continue;
        begin = std::max(begin, arena.Base());
        end = std::min(end, arenaEnd);
        const bool shared = permissions.size() > 3 && permissions[3] == 's';
        const auto protection = WindowsProtection(permissions);
        if (protection == 0x01u || arena.OverlapsHostRegion(begin, end - begin)) {
            flush();
            continue;
        }
        if (shared) {
            auto cursor = begin;
            for (const auto& [placedBegin, placement] : PlacementsWithin(begin, end)) {
                if (placement.kind != PlacementKind::Shared) continue;
                if (placedBegin > cursor) flush();
                report(placedBegin, placement.end, MappingChange::Mapped, protection);
                cursor = placement.end;
            }
            if (cursor < end) flush();
            continue;
        }
        if (inode != 0 || !path.empty()) {
            flush();
            continue;
        }
        report(begin, end, MappingChange::Committed, protection);
    }
    flush();
}
#endif

bool GuestArenaWriteWatched_nid_postfix() {
    return Arena::Get().WriteWatched();
}

std::uint64_t GuestArenaCommitGeneration_nid_postfix() {
    return commitGeneration.load(std::memory_order_acquire);
}

void GuestArenaSetMappingObserver_nid_postfix(void (*callback)(std::uintptr_t, std::size_t, std::uint64_t, MappingChange, std::uint32_t)) {
    mappingObserver.store(callback, std::memory_order_release);
}


bool GuestArenaBeginHostWrite_nid_postfix(void* pointer, std::size_t bytes) {
#ifdef _WIN32
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    auto& mappings = WindowsMappings::Get();
    if (!mappings.ReserveHostWrite(address, bytes)) return false;
    for (auto guarded = mappings.FirstAccessGuarded(address, bytes); guarded != 0; guarded = mappings.FirstAccessGuarded(address, bytes)) {
        if (!GuestArenaHandleAccess_nid_postfix(guarded, true)) {
            mappings.CancelHostWrite(address, bytes);
            notifyRefusalClears();
            return false;
        }
    }
    const bool open = mappings.OpenHostWrite(address, bytes);
    if (!open) mappings.CancelHostWrite(address, bytes);
    notifyGuarded(mappings.TakeGuardReports());
    notifyRefusalClears();
    return open;
#else
    GuestWriteWatch::GuestWriteWatchBeginHostWrite_nid_postfix(pointer, bytes);
    return true;
#endif
}

void GuestArenaEndHostWrite_nid_postfix(void* pointer, std::size_t bytes) {
#ifdef _WIN32
    WindowsMappings::Get().EndHostWrite(reinterpret_cast<std::uintptr_t>(pointer), bytes);
    notifyRefusalClears();
#else
    GuestWriteWatch::GuestWriteWatchEndHostWrite_nid_postfix(pointer, bytes);
#endif
}

#ifndef _WIN32
namespace {

std::atomic<SharedBackingResolver> sharedBackingResolver{nullptr};
std::atomic<SharedBackingWriter> sharedBackingWriter{nullptr};

}

void GuestArenaSetSharedBacking_nid_postfix(SharedBackingResolver resolver) {
    sharedBackingResolver.store(resolver, std::memory_order_release);
}

void GuestArenaSetSharedBackingWriter_nid_no_patch(SharedBackingWriter writer) {
    sharedBackingWriter.store(writer, std::memory_order_release);
}

bool GuestArenaWriteSharedBacking_nid_no_patch(std::uintptr_t address, const void* source, std::size_t bytes) {
    const auto writer = sharedBackingWriter.load(std::memory_order_acquire);
    return writer != nullptr && writer(address, source, bytes);
}

bool GuestArenaSharedBacking_nid_postfix(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset) {
    const auto resolver = sharedBackingResolver.load(std::memory_order_acquire);
    return resolver != nullptr && resolver(address, bytes, file, offset);
}
#endif

}
