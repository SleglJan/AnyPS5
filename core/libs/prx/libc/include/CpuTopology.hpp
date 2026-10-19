#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_CPUTOPOLOGY_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_CPUTOPOLOGY_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <unistd.h>
#endif

// Hybrid CPU layout (Intel P/E cores) from the OS: which logical processors belong to the lowest and
// the highest efficiency class. Non-hybrid machines report no layout and every pin is a no-op unless
// a mask is given explicitly. Header-only so libkernel and the AGC driver share it without a link.
namespace CpuTopology {

struct Layout {
    std::uint64_t process = 0;
    std::uint64_t efficient = 0;
    std::uint64_t performant = 0;
    bool hybrid = false;
};

#ifdef __linux__
inline std::uint64_t MaskFromCpuList(const char* path) {
    std::ifstream file(path);
    std::string text;
    if (!std::getline(file, text)) return 0;
    std::uint64_t mask = 0;
    const char* cursor = text.c_str();
    while (*cursor != '\0') {
        char* end = nullptr;
        const unsigned long first = std::strtoul(cursor, &end, 10);
        if (end == cursor) return 0;
        unsigned long last = first;
        cursor = end;
        if (*cursor == '-') {
            last = std::strtoul(cursor + 1, &end, 10);
            if (end == cursor + 1 || last < first) return 0;
            cursor = end;
        }
        for (unsigned long cpu = first; cpu <= last && cpu < 64; ++cpu) mask |= std::uint64_t{1} << cpu;
        if (*cursor == ',') ++cursor;
        else if (*cursor != '\0') return 0;
    }
    return mask;
}
#endif

inline const Layout& Get() {
    static const Layout layout = [] {
        Layout result{};
#ifdef _WIN32
        DWORD_PTR processMask = 0;
        DWORD_PTR systemMask = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) result.process = static_cast<std::uint64_t>(processMask);
        DWORD length = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) return result;
        std::vector<unsigned char> buffer(length);
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()), &length)) return result;
        std::uint64_t byClass[256] = {};
        bool seen[256] = {};
        for (DWORD offset = 0; offset + 8 <= length;) {
            const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
            if (entry->Size == 0 || offset + entry->Size > length) break;
            if (entry->Relationship == RelationProcessorCore && entry->Processor.GroupCount >= 1 && entry->Processor.GroupMask[0].Group == 0) {
                const unsigned cls = entry->Processor.EfficiencyClass;
                byClass[cls] |= static_cast<std::uint64_t>(entry->Processor.GroupMask[0].Mask);
                seen[cls] = true;
            }
            offset += entry->Size;
        }
        int lowest = -1;
        int highest = -1;
        for (int cls = 0; cls < 256; ++cls) {
            if (!seen[cls]) continue;
            if (lowest < 0) lowest = cls;
            highest = cls;
        }
        if (lowest < 0 || lowest == highest) return result;
        result.efficient = byClass[lowest];
        result.performant = byClass[highest];
        result.hybrid = result.efficient != 0 && result.performant != 0;
#elif defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            for (unsigned cpu = 0; cpu < 64; ++cpu)
                if (CPU_ISSET(cpu, &set)) result.process |= std::uint64_t{1} << cpu;
        }
        result.performant = MaskFromCpuList("/sys/devices/cpu_core/cpus");
        result.efficient = MaskFromCpuList("/sys/devices/cpu_atom/cpus");
        result.hybrid = result.efficient != 0 && result.performant != 0 && (result.efficient & result.performant) == 0;
#endif
        return result;
    }();
    return layout;
}

// A hex mask from the environment, 0 when unset or empty.
inline std::uint64_t MaskFromEnvironment(const char* name) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') return 0;
    return std::strtoull(text, nullptr, 16);
}

inline bool Trace() {
    static const bool trace = std::getenv("APS5_TRACE_AFFINITY") != nullptr;
    return trace;
}

inline unsigned long ThreadId(void* handle) {
#ifdef _WIN32
    return handle != nullptr ? GetThreadId(static_cast<HANDLE>(handle)) : GetCurrentThreadId();
#elif defined(__linux__)
    return handle == nullptr ? static_cast<unsigned long>(gettid()) : 0;
#else
    (void)handle;
    return 0;
#endif
}

// Restricts the thread (`nullptr` is the calling thread) to `mask` clipped to the process affinity;
// returns the mask applied, 0 when nothing was.
inline std::uint64_t Pin(void* handle, std::uint64_t mask) {
#ifdef _WIN32
    const auto& layout = Get();
    if (layout.process != 0) mask &= layout.process;
    if (mask == 0) return 0;
    HANDLE thread = handle != nullptr ? static_cast<HANDLE>(handle) : GetCurrentThread();
    return SetThreadAffinityMask(thread, static_cast<DWORD_PTR>(mask)) != 0 ? mask : 0;
#elif defined(__linux__)
    const auto& layout = Get();
    if (layout.process != 0) mask &= layout.process;
    if (mask == 0 || handle != nullptr) return 0;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (unsigned cpu = 0; cpu < 64; ++cpu)
        if ((mask >> cpu) & 1u) CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0 ? mask : 0;
#else
    (void)handle;
    (void)mask;
    return 0;
#endif
}

// Pin plus the APS5_TRACE_AFFINITY=1 line naming the thread.
inline std::uint64_t PinTraced(const char* role, void* handle, std::uint64_t mask) {
    const auto applied = Pin(handle, mask);
    if (Trace()) std::fprintf(stderr, "[affinity] %s tid=%lu mask=0x%llx%s\n", role, ThreadId(handle), static_cast<unsigned long long>(mask), applied != 0 ? "" : " (not applied)");
    return applied;
}

// Helper threads (deferred release, mirror refresh) are pinned only when APS5_HELPER_AFFINITY_MASK
// names their cores.
inline void PinHelperThread(const char* role) {
    static const std::uint64_t mask = MaskFromEnvironment("APS5_HELPER_AFFINITY_MASK");
    if (mask != 0) PinTraced(role, nullptr, mask);
}

inline constexpr unsigned GUEST_CPU_COUNT = 14;
inline constexpr unsigned OWN_CPU_MINIMUM = 2;
inline constexpr int GUEST_PRIORITY_HIGHEST = 256;
inline constexpr int GUEST_PRIORITY_LOWEST = 767;

struct GuestPlacement {
    std::uint64_t guestCpu[GUEST_CPU_COUNT] = {};
    std::uint64_t reserved = 0;
    std::uint64_t own = 0;
};

inline std::vector<std::uint64_t> CoreMasks(std::uint64_t process) {
    std::vector<std::uint64_t> cores;
#ifdef _WIN32
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) return cores;
    std::vector<unsigned char> buffer(length);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()), &length)) return cores;
    for (DWORD offset = 0; offset + 8 <= length;) {
        const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (entry->Size == 0 || offset + entry->Size > length) break;
        if (entry->Relationship == RelationProcessorCore && entry->Processor.GroupCount >= 1 && entry->Processor.GroupMask[0].Group == 0) {
            const auto mask = static_cast<std::uint64_t>(entry->Processor.GroupMask[0].Mask) & process;
            if (mask != 0) cores.push_back(mask);
        }
        offset += entry->Size;
    }
#elif defined(__linux__)
    std::uint64_t seen = 0;
    for (unsigned cpu = 0; cpu < 64; ++cpu) {
        const std::uint64_t bit = std::uint64_t{1} << cpu;
        if ((process & bit) == 0 || (seen & bit) != 0) continue;
        const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list";
        std::uint64_t mask = MaskFromCpuList(path.c_str()) & process;
        if ((mask & bit) == 0) mask = bit;
        seen |= mask;
        cores.push_back(mask);
    }
#endif
    return cores;
}

inline GuestPlacement PlaceGuestCpus(std::uint64_t process, const std::vector<std::uint64_t>& cores) {
    GuestPlacement placement{};
    std::vector<std::uint64_t> order;
    for (unsigned rank = 0; rank < 64; ++rank) {
        bool any = false;
        for (const auto core : cores) {
            std::uint64_t rest = core;
            for (unsigned skip = 0; skip < rank && rest != 0; ++skip) rest &= rest - 1;
            if (rest == 0) continue;
            any = true;
            order.push_back(rest & (~rest + 1));
        }
        if (!any) break;
    }
    if (order.empty()) {
        for (unsigned cpu = 0; cpu < 64; ++cpu)
            if ((process >> cpu) & 1u) order.push_back(std::uint64_t{1} << cpu);
    }
    if (order.empty()) return placement;
    const bool reserve = order.size() >= GUEST_CPU_COUNT + OWN_CPU_MINIMUM;
    for (unsigned cpu = 0; cpu < GUEST_CPU_COUNT; ++cpu) placement.guestCpu[cpu] = order[cpu % order.size()];
    if (reserve) {
        for (const auto bit : placement.guestCpu) placement.reserved |= bit;
    }
    placement.own = process & ~placement.reserved;
    return placement;
}

inline const GuestPlacement& Placement() {
    static const GuestPlacement placement = [] {
        const auto process = Get().process;
        return PlaceGuestCpus(process, CoreMasks(process));
    }();
    return placement;
}

inline std::uint64_t HostMaskForGuest(std::uint64_t guestMask) {
    const auto& placement = Placement();
    std::uint64_t host = 0;
    for (unsigned cpu = 0; cpu < GUEST_CPU_COUNT; ++cpu)
        if ((guestMask >> cpu) & 1u) host |= placement.guestCpu[cpu];
    if (host == 0)
        for (const auto bit : placement.guestCpu) host |= bit;
    return host;
}

#ifdef _WIN32
inline int HostPriorityForGuest(int guestPriority) {
    if (guestPriority < GUEST_PRIORITY_HIGHEST) guestPriority = GUEST_PRIORITY_HIGHEST;
    if (guestPriority > GUEST_PRIORITY_LOWEST) guestPriority = GUEST_PRIORITY_LOWEST;
    if (guestPriority < 448) return THREAD_PRIORITY_HIGHEST;
    if (guestPriority < 576) return THREAD_PRIORITY_ABOVE_NORMAL;
    if (guestPriority < 704) return THREAD_PRIORITY_NORMAL;
    if (guestPriority < 736) return THREAD_PRIORITY_BELOW_NORMAL;
    return THREAD_PRIORITY_LOWEST;
}
#endif

inline bool PlaceThreadAsGuest(void* handle, std::uint64_t guestMask, int guestPriority) {
#ifdef _WIN32
    HANDLE thread = handle != nullptr ? static_cast<HANDLE>(handle) : GetCurrentThread();
    const bool pinned = SetThreadAffinityMask(thread, static_cast<DWORD_PTR>(HostMaskForGuest(guestMask))) != 0;
    return SetThreadPriority(thread, HostPriorityForGuest(guestPriority)) != 0 && pinned;
#elif defined(__linux__)
    (void)guestPriority;
    const std::uint64_t mask = HostMaskForGuest(guestMask);
    cpu_set_t set;
    CPU_ZERO(&set);
    for (unsigned cpu = 0; cpu < 64; ++cpu)
        if ((mask >> cpu) & 1u) CPU_SET(cpu, &set);
    const pthread_t thread = handle != nullptr ? *static_cast<const pthread_t*>(handle) : pthread_self();
    return pthread_setaffinity_np(thread, sizeof(set), &set) == 0;
#else
    (void)handle;
    (void)guestMask;
    (void)guestPriority;
    return false;
#endif
}

inline bool PlaceThreadAsOwn(void* handle) {
    const auto& placement = Placement();
    if (placement.reserved == 0 || placement.own == 0) return true;
#ifdef _WIN32
    HANDLE thread = handle != nullptr ? static_cast<HANDLE>(handle) : GetCurrentThread();
    return SetThreadAffinityMask(thread, static_cast<DWORD_PTR>(placement.own)) != 0;
#else
    (void)handle;
    return false;
#endif
}

}

#endif
