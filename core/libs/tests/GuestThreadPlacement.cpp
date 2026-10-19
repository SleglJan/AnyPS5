#include "SceTypes.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

extern "C" {
int APS5_VABI scePthreadCreate(Pthread* thread, const PthreadAttr* attr, PthreadEntry entry, void* arg, const char* name);
int APS5_VABI scePthreadJoin(Pthread thread, void** retval);
int APS5_VABI scePthreadSetprio(Pthread thread, int prio);
int APS5_VABI scePthreadSetaffinity(Pthread thread, KernelCpumask mask);
int APS5_VABI scePthreadGetaffinity(Pthread thread, KernelCpumask* mask);
int APS5_VABI scePthreadAttrInit(PthreadAttr* attr);
int APS5_VABI scePthreadAttrDestroy(PthreadAttr* attr);
int APS5_VABI scePthreadAttrSetinheritsched(PthreadAttr* attr, int inheritSched);
int APS5_VABI scePthreadAttrSetschedparam(PthreadAttr* attr, const KernelSchedParam* param);
int APS5_VABI scePthreadAttrSetaffinity(PthreadAttr* attr, KernelCpumask mask);
Pthread APS5_VABI scePthreadSelf();
}

static constexpr int SCE_OK = 0;
static constexpr int EXPLICIT_SCHED = 0;
static constexpr KernelCpumask TWO_CPU_MASK = 0xa00;
static constexpr KernelCpumask MOVED_MASK = 0x5;
static constexpr auto BUSY_TIME = std::chrono::milliseconds(300);

static int failures = 0;

static void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "guest thread placement: %s\n", what);
    ++failures;
}

static std::uint64_t CurrentHostCpu() {
#ifdef _WIN32
    PROCESSOR_NUMBER number{};
    GetCurrentProcessorNumberEx(&number);
    return number.Group == 0 ? std::uint64_t{1} << number.Number : 0;
#elif defined(__linux__)
    const int cpu = sched_getcpu();
    return cpu >= 0 && cpu < 64 ? std::uint64_t{1} << cpu : 0;
#else
    return 0;
#endif
}

static std::uint64_t CurrentHostMask() {
#ifdef _WIN32
    GROUP_AFFINITY affinity{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &affinity) || affinity.Group != 0) return 0;
    return static_cast<std::uint64_t>(affinity.Mask);
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) != 0) return 0;
    std::uint64_t mask = 0;
    for (unsigned cpu = 0; cpu < 64; ++cpu)
        if (CPU_ISSET(cpu, &set)) mask |= std::uint64_t{1} << cpu;
    return mask;
#else
    return 0;
#endif
}

static std::uint64_t BusyRanOn() {
    std::uint64_t seen = 0;
    const auto end = std::chrono::steady_clock::now() + BUSY_TIME;
    while (std::chrono::steady_clock::now() < end) seen |= CurrentHostCpu();
    return seen;
}

struct AffinityProbe {
    std::uint64_t maskBefore = 0;
    std::uint64_t seenBefore = 0;
    std::atomic<bool> moved{false};
    std::atomic<bool> movedSeen{false};
    std::uint64_t maskAfter = 0;
    std::uint64_t seenAfter = 0;
};

static void* APS5_VABI AffinityWorker(void* opaque) {
    auto* probe = static_cast<AffinityProbe*>(opaque);
    probe->maskBefore = CurrentHostMask();
    probe->seenBefore = BusyRanOn();
    probe->movedSeen.store(true, std::memory_order_release);
    while (!probe->moved.load(std::memory_order_acquire)) std::this_thread::yield();
    probe->maskAfter = CurrentHostMask();
    probe->seenAfter = BusyRanOn();
    return nullptr;
}

static void CheckTwoCpuMask() {
    const auto& placement = CpuTopology::Placement();
    const std::uint64_t expected = CpuTopology::HostMaskForGuest(TWO_CPU_MASK);
    Require(expected == (placement.guestCpu[9] | placement.guestCpu[11]), "mask 0xa00 maps to guest CPUs 9 and 11");
    if (placement.reserved != 0) {
        Require(std::popcount(expected) == 2, "mask 0xa00 maps to two host CPUs");
        Require((expected & ~placement.reserved) == 0, "mask 0xa00 maps into the reserved CPUs");
    }
    PthreadAttr attr = nullptr;
    Require(scePthreadAttrInit(&attr) == SCE_OK, "attr init");
    Require(scePthreadAttrSetaffinity(&attr, TWO_CPU_MASK) == SCE_OK, "attr setaffinity");
    AffinityProbe probe;
    Pthread thread = nullptr;
    Require(scePthreadCreate(&thread, &attr, AffinityWorker, &probe, "placement") == SCE_OK, "create");
    Require(scePthreadAttrDestroy(&attr) == SCE_OK, "attr destroy");
    while (!probe.movedSeen.load(std::memory_order_acquire)) std::this_thread::yield();
    Require(scePthreadSetaffinity(thread, MOVED_MASK) == SCE_OK, "setaffinity");
    KernelCpumask reported = 0;
    Require(scePthreadGetaffinity(thread, &reported) == SCE_OK && reported == MOVED_MASK, "getaffinity reports the guest mask");
    probe.moved.store(true, std::memory_order_release);
    Require(scePthreadJoin(thread, nullptr) == SCE_OK, "join");
    std::fprintf(stderr, "0xa00: host mask 0x%llx ran on 0x%llx; 0x5: host mask 0x%llx ran on 0x%llx\n", static_cast<unsigned long long>(probe.maskBefore), static_cast<unsigned long long>(probe.seenBefore), static_cast<unsigned long long>(probe.maskAfter), static_cast<unsigned long long>(probe.seenAfter));
    Require(probe.maskBefore == expected, "a thread created with mask 0xa00 has its two mapped host CPUs as affinity");
    Require(probe.seenBefore != 0 && (probe.seenBefore & ~expected) == 0, "a thread created with mask 0xa00 runs only on its mapped host CPUs");
    const std::uint64_t moved = CpuTopology::HostMaskForGuest(MOVED_MASK);
    Require(probe.maskAfter == moved, "scePthreadSetaffinity moves the thread to its mapped host CPUs");
    Require(probe.seenAfter != 0 && (probe.seenAfter & ~moved) == 0, "after scePthreadSetaffinity the thread runs only on its mapped host CPUs");
}

#ifdef _WIN32
struct PriorityProbe {
    int initial = 0;
    int raised = 0;
};

static void* APS5_VABI PriorityWorker(void* opaque) {
    auto* probe = static_cast<PriorityProbe*>(opaque);
    probe->initial = GetThreadPriority(GetCurrentThread());
    Require(scePthreadSetprio(scePthreadSelf(), CpuTopology::GUEST_PRIORITY_HIGHEST) == SCE_OK, "setprio");
    probe->raised = GetThreadPriority(GetCurrentThread());
    return nullptr;
}

static int HostPriorityOfNewThread(int guestPriority, PriorityProbe& probe) {
    PthreadAttr attr = nullptr;
    Require(scePthreadAttrInit(&attr) == SCE_OK, "attr init");
    Require(scePthreadAttrSetinheritsched(&attr, EXPLICIT_SCHED) == SCE_OK, "attr setinheritsched");
    const KernelSchedParam param{guestPriority};
    Require(scePthreadAttrSetschedparam(&attr, &param) == SCE_OK, "attr setschedparam");
    Pthread thread = nullptr;
    Require(scePthreadCreate(&thread, &attr, PriorityWorker, &probe, "priority") == SCE_OK, "create");
    Require(scePthreadAttrDestroy(&attr) == SCE_OK, "attr destroy");
    Require(scePthreadJoin(thread, nullptr) == SCE_OK, "join");
    return probe.initial;
}

static void CheckPriorityOrder() {
    PriorityProbe high;
    PriorityProbe low;
    PriorityProbe middle;
    const int highHost = HostPriorityOfNewThread(CpuTopology::GUEST_PRIORITY_HIGHEST, high);
    const int middleHost = HostPriorityOfNewThread(700, middle);
    const int lowHost = HostPriorityOfNewThread(CpuTopology::GUEST_PRIORITY_LOWEST, low);
    std::fprintf(stderr, "guest 256 -> host %d, 700 -> %d, 767 -> %d; 767 raised to 256 -> %d\n", highHost, middleHost, lowHost, low.raised);
    Require(highHost > middleHost && middleHost > lowHost, "a higher guest priority gives a higher host priority");
    Require(low.raised == highHost, "scePthreadSetprio raises the host priority");
    int previous = THREAD_PRIORITY_TIME_CRITICAL;
    for (int guest = CpuTopology::GUEST_PRIORITY_HIGHEST; guest <= CpuTopology::GUEST_PRIORITY_LOWEST; ++guest) {
        const int host = CpuTopology::HostPriorityForGuest(guest);
        Require(host <= previous, "guest priorities map monotonically onto host priorities");
        previous = host;
    }
}

static std::uint64_t ThreadMask(HANDLE thread) {
    GROUP_AFFINITY affinity{};
    if (!GetThreadGroupAffinity(thread, &affinity) || affinity.Group != 0) return 0;
    return static_cast<std::uint64_t>(affinity.Mask);
}

static void CheckOwnThreadsAvoidReserved(std::uint64_t mainMask) {
    const auto& placement = CpuTopology::Placement();
    if (placement.reserved == 0) {
        std::fprintf(stderr, "fewer than %u host CPUs: nothing is reserved\n", CpuTopology::GUEST_CPU_COUNT + CpuTopology::OWN_CPU_MINIMUM);
        return;
    }
    Require(mainMask != 0 && (mainMask & placement.reserved) == 0, "the process's first thread is kept off the reserved CPUs");
    std::uint64_t ownMask = 0;
    std::uint64_t ownSeen = 0;
    std::thread own([&] {
        ownMask = CurrentHostMask();
        ownSeen = BusyRanOn();
    });
    own.join();
    std::fprintf(stderr, "reserved 0x%llx own 0x%llx; own thread mask 0x%llx ran on 0x%llx\n", static_cast<unsigned long long>(placement.reserved), static_cast<unsigned long long>(placement.own), static_cast<unsigned long long>(ownMask), static_cast<unsigned long long>(ownSeen));
    Require(ownMask == placement.own, "an emulator thread has the non-reserved CPUs as affinity");
    Require(ownSeen != 0 && (ownSeen & placement.reserved) == 0, "an emulator thread never runs on a reserved CPU");
}
#endif

int main() {
#ifdef _WIN32
    const std::uint64_t mainMask = ThreadMask(GetCurrentThread());
#endif
    const auto& placement = CpuTopology::Placement();
    for (unsigned cpu = 0; cpu < CpuTopology::GUEST_CPU_COUNT; ++cpu)
        std::fprintf(stderr, "guest CPU %u -> host 0x%llx\n", cpu, static_cast<unsigned long long>(placement.guestCpu[cpu]));
#ifdef _WIN32
    CheckOwnThreadsAvoidReserved(mainMask);
    CheckPriorityOrder();
#endif
    CheckTwoCpuMask();
    if (failures != 0) std::abort();
    return 0;
}
