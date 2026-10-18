#include "prx/libSceAgcDriver/Execution/include/AsyncLog.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelCompare.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/shaders/MeshArguments_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/DispatchThreads_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/SampleCounter_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/SampleDumps_spv.h"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4Opcodes.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include "prx/libc/include/general/AtomicSharedPtr.hpp"

namespace AgcDriver::Graphics {

namespace {

const void* RecordingThreadToken() {
    static thread_local char token;
    return &token;
}

thread_local bool ownStreamRecording = false;

std::mutex orderWaitMutex;
std::uint64_t orderWaitGeneration = 0;
std::atomic<std::uint32_t> orderWaiters{0};

std::condition_variable& OrderWaitChanged() {
    static auto* const changed = new std::condition_variable;
    return *changed;
}

void NotifyOrderWaiters() {
    if (orderWaiters.load(std::memory_order_acquire) == 0) return;
    {
        std::lock_guard lock(orderWaitMutex);
        ++orderWaitGeneration;
    }
    OrderWaitChanged().notify_all();
}

class RecordingHold {
public:
    explicit RecordingHold(RecordingMutex& mutex) : mutex(&mutex) { mutex.lock(); }
    ~RecordingHold() {
        if (mutex != nullptr) mutex->unlock();
    }
    RecordingHold(const RecordingHold&) = delete;
    RecordingHold& operator=(const RecordingHold&) = delete;
    void Abandon() { mutex = nullptr; }

private:
    RecordingMutex* mutex;
};

}

void RecordingMutex::lock() {
    if (owner.load(std::memory_order_relaxed) == RecordingThreadToken()) {
        ++depth;
        return;
    }
    mutex.lock();
    owner.store(RecordingThreadToken(), std::memory_order_relaxed);
    depth = 1;
}

void RecordingMutex::unlock() {
    if (--depth != 0) return;
    owner.store(nullptr, std::memory_order_relaxed);
    mutex.unlock();
}

bool RecordingMutex::HeldByThisThread() const {
    return owner.load(std::memory_order_relaxed) == RecordingThreadToken();
}

std::uint32_t RecordingMutex::ReleaseAll() {
    if (!HeldByThisThread()) return 0;
    const auto held = depth;
    depth = 0;
    owner.store(nullptr, std::memory_order_relaxed);
    mutex.unlock();
    return held;
}

void RecordingMutex::Reacquire(std::uint32_t held) {
    if (held == 0) return;
    mutex.lock();
    owner.store(RecordingThreadToken(), std::memory_order_relaxed);
    depth = held;
}

namespace {

Recorder* activeRecorder = nullptr;

bool DrawProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profiled;
}

std::uint64_t syncCounts[5] = {};
// Fence wait time by sync source (APS5_PROFILE_DRAW), and the source CountSync announced for the
// Sync/SyncThrough that follows it on this thread.
double syncWaitedMs[5] = {};
thread_local int announcedSource = 4;
// The call site CountSync named for that sync (nullptr: none; the sync's own return address is
// taken then), and the fence-wait table per (source, site) it feeds (APS5_PROFILE_DRAW, under the
// GpuMutex like finish()): which caller's syncs wait, when the source alone does not say (source
// 0 is every WaitIdle and drain). `activeSyncSite` is the entry of the sync in progress on this
// thread (an index: a nested sync from a completion replaces and restores it), so finish() charges
// each batch's wait to it.
thread_local const void* announcedSite = nullptr;
struct SyncSiteWaits {
    int source;
    const void* site;
    std::uint64_t syncs;
    std::uint64_t batches;
    double waitedMs;
};
std::vector<SyncSiteWaits> syncSites;
constexpr std::size_t NoSyncSite = static_cast<std::size_t>(-1);
constexpr std::size_t SyncSiteLimit = 48;
thread_local std::size_t activeSyncSite = NoSyncSite;

bool SyncSitesProfiled() {
    // APS5_NO_SYNC_SITES=1 leaves only the per-source and per-thread counts.
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr && std::getenv("APS5_NO_SYNC_SITES") == nullptr;
    return profiled;
}

// Begins a sync's attribution: returns the previous active entry for the caller to restore, after
// counting the sync under (source, site). Beyond the table's limit every further site shares the
// last entry (a nullptr site), so the table stays small.
std::size_t BeginSyncSite(int source, const void* site) {
    const auto previous = activeSyncSite;
    if (!SyncSitesProfiled()) return previous;
    auto it = std::find_if(syncSites.begin(), syncSites.end(), [&](const SyncSiteWaits& entry) { return entry.source == source && entry.site == site; });
    if (it == syncSites.end()) {
        if (syncSites.size() >= SyncSiteLimit) {
            it = std::find_if(syncSites.begin(), syncSites.end(), [&](const SyncSiteWaits& entry) { return entry.source == source && entry.site == nullptr; });
            if (it == syncSites.end()) it = syncSites.insert(syncSites.end(), SyncSiteWaits{source, nullptr, 0, 0, 0});
        } else {
            it = syncSites.insert(syncSites.end(), SyncSiteWaits{source, site, 0, 0, 0});
        }
    }
    ++it->syncs;
    activeSyncSite = static_cast<std::size_t>(it - syncSites.begin());
    return previous;
}

void CountSiteWait(double ms) {
    if (activeSyncSite == NoSyncSite || activeSyncSite >= syncSites.size()) return;
    auto& entry = syncSites[activeSyncSite];
    ++entry.batches;
    entry.waitedMs += ms;
}

// The sites by wait, longest first, as "source@+offset syncs/batches/wait" (offsets symbolize with
// nm against the driver's image like the [guestmem] callers; +0x0 is the overflow entry).
std::string SyncSiteReport() {
    static const char* const names[5] = {"idle", "pending-write", "recorded-store", "address-based", "other"};
    std::vector<const SyncSiteWaits*> order;
    for (const auto& entry : syncSites) order.push_back(&entry);
    std::sort(order.begin(), order.end(), [](const SyncSiteWaits* a, const SyncSiteWaits* b) { return a->waitedMs > b->waitedMs; });
    std::string report;
    for (std::size_t i = 0; i < order.size() && i < 8; ++i) {
        char text[96];
        std::snprintf(text, sizeof(text), " %s@+0x%llx %llu/%llu/%.1fs", names[order[i]->source], order[i]->site != nullptr ? GuestMemory::CodeOffset(order[i]->site) : 0ull, static_cast<unsigned long long>(order[i]->syncs), static_cast<unsigned long long>(order[i]->batches), order[i]->waitedMs / 1000);
        report += text;
    }
    return report;
}
// Fence waits by thread and source (APS5_PROFILE_DRAW), keyed by the thread's GpuMutex queue tag:
// a sync waits for the GPU under the mutex, so which worker's syncs (and which kind) hold it against
// queue 0 has to be visible. Its own mutex: the table is read while the report line is printed.
struct ThreadSyncs {
    std::uint32_t tag;
    std::array<std::uint64_t, 5> counts;
    std::array<double, 5> waitedMs;
};
std::vector<ThreadSyncs> threadSyncs;
std::mutex threadSyncsMutex;

// This thread's fence and timeline waits in total (Recorder::ThreadWaitedMs): a caller times a span
// of its own work and reads the difference to learn how much of it was waiting for the GPU.
thread_local double threadWaitedMs = 0;

void CountThreadSync(int source, double ms) {
    threadWaitedMs += ms;
    const auto tag = GuestMemory::GpuLockThreadTag();
    std::lock_guard lock(threadSyncsMutex);
    auto it = std::find_if(threadSyncs.begin(), threadSyncs.end(), [&](const ThreadSyncs& thread) { return thread.tag == tag; });
    if (it == threadSyncs.end()) it = threadSyncs.insert(threadSyncs.end(), ThreadSyncs{tag, {}, {}});
    ++it->counts[source];
    it->waitedMs[source] += ms;
}

std::string ThreadSyncReport() {
    static const char* const names[5] = {"idle", "pending-write", "recorded-store", "address-based", "other"};
    std::string report;
    std::lock_guard lock(threadSyncsMutex);
    for (const auto& thread : threadSyncs) {
        char text[64];
        if (thread.tag == 0xffffffffu) std::snprintf(text, sizeof(text), " untagged:");
        else std::snprintf(text, sizeof(text), " queue 0x%x:", thread.tag);
        report += text;
        for (int source = 0; source < 5; ++source) {
            if (thread.counts[source] == 0) continue;
            std::snprintf(text, sizeof(text), " %s %llu/%.1fs", names[source], static_cast<unsigned long long>(thread.counts[source]), thread.waitedMs[source] / 1000);
            report += text;
        }
    }
    return report;
}
// Unlocked timeline waits (WaitSerial): count and time, for the [recorder] line.
std::atomic<std::uint64_t> unlockedWaits{0}, unlockedWaitedUs{0};
// vkQueueSubmit calls and their time (APS5_PROFILE_DRAW; Submit runs under the GpuMutex).
std::uint64_t submitCount = 0;
double submitUs = 0, submitMaxUs = 0;
// Threads inside a timeline wait with the GpuMutex released (WaitSerial, syncThroughUnlocked): a
// recorder's teardown, after its Sync() made every wait satisfiable, waits until they left the
// semaphore before destroying it. Counted per recorder id (a small fixed array; the waiter cannot
// touch a recorder that may be dying, so it is keyed by the id it copied out), so a replaced
// device's old recorder does not spin on the new device's in-flight waits.
constexpr std::size_t WaiterSlots = 16;
std::atomic<int> unlockedWaiters[WaiterSlots]{};
std::atomic<int>& WaitersOf(std::uint64_t id) { return unlockedWaiters[id % WaiterSlots]; }
// Recorders alive, by id (own mutex, taken under the GpuMutex or with nothing held): a thread that
// released the GpuMutex around a wait learns whether its recorder still exists before touching it.
std::mutex liveRecordersMutex;
std::vector<std::uint64_t> liveRecorders;
std::atomic<std::uint64_t> nextRecorderId{1};
std::atomic<std::uint64_t> samplesPassed{0};
constexpr std::uint32_t SampleFoldCapacity = 64;
constexpr std::uint32_t SampleCounterPushBytes = 24;
constexpr std::uint32_t SegmentSlots = 128;
constexpr std::uint32_t SegmentFoldCapacity = 1024;
constexpr std::uint32_t DumpFoldCapacity = 1024;
constexpr std::uint32_t SampleDumpBytes = 15 * 16 + 8;
constexpr VkDeviceSize DumpListOffset = 16 + 8 * SegmentFoldCapacity;
constexpr std::uint32_t SampleDumpPushBytes = 40;
std::atomic<std::uint64_t> dumpsQueued{0}, dumpsQueuedInPass{0}, dumpsAtOnce{0}, segmentsClosed{0}, dumpFolds{0};

struct SamplePoolCache {
    VkDevice device;
    PFN_vkDestroyQueryPool destroy;
    std::mutex mutex;
    std::vector<VkQueryPool> free;
    bool closed = false;
    void put(VkQueryPool pool) {
        {
            std::lock_guard lock(mutex);
            if (!closed && free.size() < 256) {
                free.push_back(pool);
                return;
            }
        }
        destroy(device, pool, nullptr);
    }
    VkQueryPool take() {
        std::lock_guard lock(mutex);
        if (free.empty()) return VK_NULL_HANDLE;
        const auto pool = free.back();
        free.pop_back();
        return pool;
    }
    void close() {
        std::lock_guard lock(mutex);
        closed = true;
        for (const auto pool : free) destroy(device, pool, nullptr);
        free.clear();
    }
};

struct SamplePool {
    SamplePool(std::shared_ptr<SamplePoolCache> cache, VkQueryPool pool) : cache(std::move(cache)), pool(pool) {}
    SamplePool(const SamplePool&) = delete;
    SamplePool& operator=(const SamplePool&) = delete;
    ~SamplePool() { cache->put(pool); }
    std::shared_ptr<SamplePoolCache> cache;
    VkQueryPool pool;
};

bool RecorderAlive(std::uint64_t id) {
    std::lock_guard lock(liveRecordersMutex);
    return std::find(liveRecorders.begin(), liveRecorders.end(), id) != liveRecorders.end();
}

// Completion actions in progress on this thread (finish()): a pending-write sync the flush hook
// makes from inside one (a copied buffer's write-back store) is skipped, see SyncThrough.
thread_local int completionDepth = 0;
// This thread's hook syncs that waited for an unsignaled target (Recorder::ThreadHookWaits).
thread_local std::uint64_t hookRealWaits = 0;

}

bool Recorder::InCompletion() {
    return completionDepth != 0;
}

namespace {

// APS5_NO_HOOK_COMPLETION_GUARD=1: the flush hook records the accessing worker's queued labels
// even from inside a completion action, as before (see FlushForAccess).
bool HookCompletionGuard() {
    static const bool guard = std::getenv("APS5_NO_HOOK_COMPLETION_GUARD") == nullptr;
    return guard;
}

// APS5_COMPLETION_STORE_SYNC=1: a store made by a completion action waits for later batches that
// note its range, as before (the wait runs under whichever hold reaped the batch).
bool CompletionStoreSyncs() {
    static const bool enabled = std::getenv("APS5_COMPLETION_STORE_SYNC") != nullptr;
    return enabled;
}

// APS5_HOOK_LOCKED_WAIT=1: the flush hook's pending-write wait stays under the GpuMutex as before.
bool HookLockedWait() {
    static const bool locked = std::getenv("APS5_HOOK_LOCKED_WAIT") != nullptr;
    return locked;
}

// APS5_HOOK_FULL_SYNC=1: the flush hook's recorded-store wait is a full Sync() under the GpuMutex
// (every batch in flight) as before, instead of SyncThrough's targeted, unlocked wait (H1).
bool HookFullSync() {
    static const bool hookFullSync = std::getenv("APS5_HOOK_FULL_SYNC") != nullptr;
    return hookFullSync;
}

// APS5_HOOK_FLUSH_CPU_BLOCKS=0: an access whose every 64 KiB block the CPU wrote since the
// overlapping pending images' generations flushes them as before, instead of the C4 skip (H2:
// StorageTexture::AccessKeptByCpu).
bool HookFlushCpuBlocks() {
    static const bool hookFlushCpuBlocks = [] {
        const char* text = std::getenv("APS5_HOOK_FLUSH_CPU_BLOCKS");
        return text != nullptr && text[0] == '0';
    }();
    return hookFlushCpuBlocks;
}

// What the holds spend on the recorder (APS5_PROFILE_DRAW, all under the GpuMutex, printed at the
// end of the [recorder] line): completions and their time (of which releasing the kept objects),
// pending-write syncs from inside completions (skipped, or waited with the kill switch), reaps and
// the batches they retired, and the flush hook's unlocked waits (with those that found the
// recorder torn down when they retook the mutex, and those that had to wait locked).
struct HoldCounters {
    std::uint64_t completions = 0;
    double completionMs = 0;
    double keptReleaseMs = 0;
    std::uint64_t completionSyncsSkipped = 0;
    std::uint64_t completionSyncsWaited = 0;
    double completionSyncWaitMs = 0;
    std::uint64_t reaps = 0;
    std::uint64_t reapsWithWork = 0;
    std::uint64_t reapBatches = 0;
    double reapMs = 0;
    std::uint64_t hookUnlockedWaits = 0;
    // The GPU wait alone (timeline wait with the mutex released) and, separately, the relock's
    // own wait for the GpuMutex (also counted by the [lock] line under 'hook'), so the two
    // remaining costs, GPU latency and hold contention, stay apart.
    double hookUnlockedWaitMs = 0;
    double hookRelockMs = 0;
    // The same by sync source (1 pending write, 2 recorded store).
    std::array<std::uint64_t, 5> hookUnlockedWaitsBySource{};
    std::array<double, 5> hookUnlockedWaitMsBySource{};
    std::uint64_t hookUnlockedTornDown = 0;
    // Genuine nested hooks (depth >= 2 outside a completion action) that had to wait locked.
    std::uint64_t hookLockedWaits = 0;
};
HoldCounters holdCounters;

// Deferred release of a finished batch's kept objects (and its completion actions, whose captures
// hold some of the same objects): finish() moves them to this list of the finishing thread instead
// of destroying them under the mutex, and the GpuMutex unlock hook (ReleaseDeferredKeeps, set by
// the constructor) hands the list to the release thread once the thread gave up its outermost
// hold. Destroying a ShaderResources with its buffers, descriptor set, textures and pipelines cost
// ~1.5 ms per reap with work, inside whichever hold reaped ('kept objects released', 15.9 s per
// 200 s run), and most reaps are queue 0's own (ReapRecorded before its dispatches), so a release
// on the unlocking thread would still be the frame's time. The destructors touch only their own
// caches' mutexes and the device, never the recorder or GpuMutex, so they need no hold and no
// particular thread (every Vulkan object is externally synchronized per object, and its cache's
// mutex covers its pool); they need the device alive, which ~Recorder guarantees by draining its
// own thread's list, stopping and joining the release thread (it drains the queue before it
// leaves) and waiting for every release still in progress on another thread (`deferredPending`:
// counted up under the mutex, down after each batch's objects are gone) before the device goes.
// The queue is bounded (APS5_RELEASE_QUEUE_MAX batches, default 256): a thread whose hand-off
// would exceed it destroys its batches itself, as it did before the release thread, so memory
// cannot grow behind a release thread that falls behind.
// APS5_RELEASE_UNDER_LOCK=1 destroys them in finish() as before; APS5_RELEASE_ON_UNLOCK=1 destroys
// them on the unlocking thread (no release thread).
struct DeferredBatch {
    std::vector<std::shared_ptr<void>> kept;
    std::vector<std::function<void()>> completions;
};
struct DeferredBatchesTag {};
auto& DeferredBatches() { return HostThreadLocal<std::vector<DeferredBatch>, DeferredBatchesTag>(); }
struct DeferredBytesTag {};
auto& DeferredBytes() { return HostThreadLocal<std::size_t, DeferredBytesTag>(); }
std::vector<DeferredBatch> TakeDeferred() {
    auto taken = std::move(DeferredBatches());
    DeferredBatches().clear();
    DeferredBytes() = 0;
    return taken;
}
std::atomic<std::uint64_t> deferredPending{0};
std::atomic<std::uint64_t> threadReleases{0}, threadObjects{0}, threadReleaseUs{0};
std::atomic<std::uint64_t> inlineReleases{0}, inlineObjects{0}, inlineReleaseUs{0}, inlineOverBound{0};
std::atomic<std::uint64_t> releaseQueueMax{0};

bool ReleaseUnderLock() {
    static const bool locked = std::getenv("APS5_RELEASE_UNDER_LOCK") != nullptr;
    return locked;
}

bool ReleaseOnUnlock() {
    static const bool onUnlock = std::getenv("APS5_RELEASE_ON_UNLOCK") != nullptr;
    return onUnlock;
}

std::size_t ReleaseQueueBound() {
    static const std::size_t bound = [] {
        const char* value = std::getenv("APS5_RELEASE_QUEUE_MAX");
        const auto parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : std::size_t{256};
    }();
    return bound;
}

// Destroys deferred batches, on the release thread (`onThread`) or on the thread that deferred
// them, and counts each batch off `deferredPending` only once its objects are gone: ~Recorder
// waits for that count.
void DestroyDeferred(std::vector<DeferredBatch> releasing, bool onThread) {
    if (releasing.empty()) return;
    const bool profile = DrawProfiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto count = releasing.size();
    std::uint64_t objects = 0;
    for (auto& batch : releasing) {
        objects += batch.kept.size();
        // The completions first (their captures hold some of the objects), then the objects.
        batch.completions.clear();
        batch.kept.clear();
        deferredPending.fetch_sub(1, std::memory_order_acq_rel);
    }
    releasing.clear();
    if (profile) {
        const auto us = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        (onThread ? threadReleases : inlineReleases).fetch_add(count, std::memory_order_relaxed);
        (onThread ? threadObjects : inlineObjects).fetch_add(objects, std::memory_order_relaxed);
        (onThread ? threadReleaseUs : inlineReleaseUs).fetch_add(us, std::memory_order_relaxed);
    }
}

// The release thread's queue: one for the process. The thread is started at the first hand-off and
// stopped and joined by ~Recorder (JoinReleaseThread), after which the next hand-off starts it
// again (a device replacement makes a new recorder). The queue object is leaked so that a static
// destructor never races the thread: at process exit the thread is idle on its condition variable
// (every ~Recorder joined it) or, when no ~Recorder ran, mid-release like any worker thread. Nothing
// is done under `mutex` but the hand-off and the take; `joinMutex` serializes joiners and orders
// before `mutex`; neither is ever taken by a kept object's destructor.
struct ReleaseQueue {
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<DeferredBatch> items;
    std::thread thread;
    bool started = false;
    // Set by a joiner under `mutex` until the thread was joined: the thread leaves once the queue
    // is empty, and hand-offs meanwhile destroy inline (nothing may be queued without a taker).
    bool stop = false;
    std::mutex joinMutex;
};

ReleaseQueue& ReleaseThreadQueue() {
    static ReleaseQueue* const queue = new ReleaseQueue;
    return *queue;
}

void ReleaseThreadMain() {
    CpuTopology::PinHelperThread("release thread");
    auto& queue = ReleaseThreadQueue();
    std::unique_lock lock(queue.mutex);
    for (;;) {
        queue.wake.wait(lock, [&] { return !queue.items.empty() || queue.stop; });
        if (queue.items.empty()) return;
        auto items = std::move(queue.items);
        queue.items.clear();
        lock.unlock();
        DestroyDeferred(std::move(items), true);
        lock.lock();
    }
}

// Stops the release thread once it emptied the queue and joins it (~Recorder, before its device
// goes). Hand-offs made while the thread stops destroy inline, so nothing can sit in the queue
// without a taker; a later hand-off starts the thread again.
void JoinReleaseThread() {
    auto& queue = ReleaseThreadQueue();
    std::lock_guard joining(queue.joinMutex);
    std::thread worker;
    {
        std::lock_guard lock(queue.mutex);
        if (!queue.started) return;
        queue.stop = true;
        worker = std::move(queue.thread);
    }
    queue.wake.notify_all();
    worker.join();
    std::lock_guard lock(queue.mutex);
    queue.started = false;
    queue.stop = false;
}

// The unlock hook: hands what this thread deferred to the release thread, or destroys it here
// (APS5_RELEASE_ON_UNLOCK=1, the queue is full or stopping, or the hand-off cannot be made).
void ReleaseDeferredKeeps() {
    NotifyOrderWaiters();
    if (DeferredBatches().empty()) return;
    // Taken off the thread's list first: a destructor that took and released the mutex (none is
    // known to) would re-enter here and must find nothing.
    auto releasing = TakeDeferred();
    if (ReleaseOnUnlock()) {
        DestroyDeferred(std::move(releasing), false);
        return;
    }
    auto& queue = ReleaseThreadQueue();
    const bool profile = DrawProfiled();
    bool handedOff = false;
    bool overBound = false;
    try {
        std::lock_guard lock(queue.mutex);
        if (queue.items.size() + releasing.size() > ReleaseQueueBound()) {
            overBound = true;
        } else if (!queue.stop) {
            // The thread before the items: items queued with no thread to take them would keep
            // `deferredPending` up and ~Recorder waiting forever.
            if (!queue.started) {
                queue.thread = std::thread(&ReleaseThreadMain);
                queue.started = true;
            }
            // The allocation before any move: a failure here leaves `releasing` intact for the
            // fallback below, and the moves cannot throw.
            queue.items.reserve(queue.items.size() + releasing.size());
            for (auto& batch : releasing) queue.items.push_back(std::move(batch));
            handedOff = true;
            if (profile) {
                const auto queued = static_cast<std::uint64_t>(queue.items.size());
                auto seen = releaseQueueMax.load(std::memory_order_relaxed);
                while (queued > seen && !releaseQueueMax.compare_exchange_weak(seen, queued, std::memory_order_relaxed)) {
                }
            }
        }
    } catch (...) {
    }
    if (!handedOff) {
        if (profile && overBound) inlineOverBound.fetch_add(releasing.size(), std::memory_order_relaxed);
        DestroyDeferred(std::move(releasing), false);
        return;
    }
    queue.wake.notify_one();
}
// Lock-free state of the active recorder's open batch, read by the queue workers between packets
// and inside WAIT_REG_MEM polls (see the static readers in Recorder.hpp). Written under the mutex.
constexpr std::int64_t NoPendingLabel = std::numeric_limits<std::int64_t>::min();
std::atomic<std::int64_t> pendingLabelSince{NoPendingLabel};
std::atomic<std::uint64_t> writeGeneration{0};
std::atomic<std::uint64_t> publishGeneration{0};
std::atomic<std::uint64_t> completionLabels{0};
std::atomic<std::uint64_t> writeBackCompletions{0};
std::atomic<std::uint64_t> completionStoresSkipped{0};
std::atomic<std::uint64_t> completionStoresRun{0};
// Write-backs noted over a tracked label dword (NoteWrittenBack): a CPU store over a GPU label.
std::atomic<std::uint64_t> writeBacksOverLabels{0};
// Completion labels that entered the pending count at a write-back (see NoteWrittenBack).
std::atomic<std::uint64_t> completionLabelsCountedLate{0};
std::atomic<std::uint64_t> workSinceSubmit{0};

// APS5_COUNT_ALL_COMPLETION_LABELS=1: a completion label also stored on the GPU counts as pending
// from its registration, as before (see Recorder::AfterCompletions).
bool CountAllCompletionLabels() {
    static const bool all = std::getenv("APS5_COUNT_ALL_COMPLETION_LABELS") != nullptr;
    return all;
}

// APS5_NO_PROC_TABLE=1: the recorder resolves its Vulkan entry points per call, as before.
bool ProcTable() {
    static const bool table = std::getenv("APS5_NO_PROC_TABLE") == nullptr;
    return table;
}
// The pending-label table's own mutex: every mutation of a recorder's `labels` holds it (under the
// GPU mutex), so a WAIT_REG_MEM looks a label up with this small lock alone (Recorder::LookupLabel)
// instead of queueing behind a dispatch. `labelTableOwner` is the active recorder while it lives,
// set and cleared under this mutex, so a lookup never touches a table being destroyed. Nothing
// under this mutex takes the GPU mutex (no lock-order cycle).
std::mutex labelTableMutex;
std::atomic<std::uint64_t> labelTableLookups{0};
Recorder* labelTableOwner = nullptr;
// The ranges of the calling worker's queued labels (NoteQueuedLabel, cleared by
// ForgetQueuedLabels): the flush hook, on the same thread, has them recorded before an access that
// overlaps one, through the function the driver installed.
struct QueuedLabelRangesTag {};
auto& QueuedLabelRanges() { return HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, QueuedLabelRangesTag>(); }
std::atomic<void (*)()> queuedLabelRecorder{nullptr};
// Store-run and queued-label counters (Recorder::StoreCounts), relaxed: they are only reported.
std::atomic<std::uint64_t> storeCount{0}, storeRuns{0}, storesJoined{0}, storesReplaced{0}, storeWawBarriers{0}, storeJoinsRefused{0};
std::atomic<std::uint64_t> keyStoreCount{0}, keyStoreRuns{0}, keyStoreRunsForWriter{0}, keyStoresJoined{0};

// Debug aid: APS5_DCC_KEYS_EACH=1 records every DCC key store at once (see Recorder::QueueKeyStore).
bool KeyStoresEach() {
    static const bool each = std::getenv("APS5_DCC_KEYS_EACH") != nullptr;
    return each;
}
std::atomic<std::uint64_t> queuedLabelsNoted{0}, queuedLabelsOverRecorded{0}, queuedLabelHits{0}, queuedLabelHookRecords{0}, queuedLabelHookInCompletion{0};
// Read-tracking counters (Recorder::ReadCounts), relaxed: they are only reported.
constexpr std::size_t ReadKinds = static_cast<std::size_t>(Recorder::ReadKind::Count);
std::atomic<std::uint64_t> readsNoted{0}, readQueries{0}, readStaleIgnored{0};
std::atomic<std::uint64_t> readHits[ReadKinds]{};

bool ReadTrackingEnabled() {
    // APS5_COPY_READ_TRACKING=0: no in-place reads are noted; the copy HLE's CPU path then requires
    // an idle recorder (VulkanDevice::CopyBuffer), never less.
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_COPY_READ_TRACKING");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool LabelRunsEnabled() {
    // Debug aid: APS5_NO_LABEL_RUNS=1 records every store between a barrier pair of its own
    // (APS5_NO_LABEL_BATCHING=1 turns off every label-batching change of Driver.cpp too).
    static const bool enabled = std::getenv("APS5_NO_LABEL_RUNS") == nullptr && std::getenv("APS5_NO_LABEL_BATCHING") == nullptr;
    return enabled;
}

// One store run per batch, recorded at Submit (see Recorder::RecordStore). Debug aid:
// APS5_LABEL_RUNS_INLINE=1 records the stores at once and closes the run at the next command.
bool LabelRunsPerBatch() {
    static const bool perBatch = LabelRunsEnabled() && std::getenv("APS5_LABEL_RUNS_INLINE") == nullptr;
    return perBatch;
}
std::atomic<std::uint64_t> storeRunsAtSubmit{0}, storeRunsForced{0};

// APS5_NO_JOIN_WAW_CHECK=1: a contiguous store joins the pending one without the check against
// the run's recorded stores, as before (see RecordStore).
bool JoinWawCheck() {
    static const bool check = std::getenv("APS5_NO_JOIN_WAW_CHECK") == nullptr;
    return check;
}

bool QueuedLabelOverlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    for (const auto& [begin, finish] : QueuedLabelRanges()) {
        if (address < finish && begin < end) return true;
    }
    return false;
}
// Flush hook statistics: calls, calls whose snapshot check overlapped (the GpuMutex was taken),
// targeted syncs and the batches they left in flight.
std::atomic<std::uint64_t> hookCalls{0}, hookLocks{0}, targetedSyncs{0}, batchesLeftInFlight{0};
// Snapshot maintenance (under the GpuMutex): notes whose range the snapshot already covered (no
// rebuild), rebuilds, and the time the rebuilds took (APS5_PROFILE_DRAW), so their cost is visible.
std::uint64_t snapshotCovered = 0, snapshotRebuilds = 0;
double snapshotRebuildMs = 0;

using WriteRanges = Recorder::WriteRanges;
// The sorted, merged union of the guest ranges every unfinished batch will write. Rebuilt under
// GuestMemory::GpuMutex (the only writer) and read by the flush hook without it: the lock is the
// graphics worker's main stall, so a no-overlap access must not wait for another queue's device
// work. A range leaves the snapshot only after its batch's completions (CPU write-backs) ran, so a
// reader that sees no overlap either precedes the note (the queues are unordered then, as on the
// GPU) or follows the write-back.
AtomicSharedPtr<const WriteRanges> pendingWrites;

bool HookSnapshotEnabled() {
    // Debug aid: APS5_NO_HOOK_SNAPSHOT=1 takes the GpuMutex on every access as before.
    static const bool enabled = std::getenv("APS5_NO_HOOK_SNAPSHOT") == nullptr;
    return enabled;
}

bool SnapshotOverlaps(const WriteRanges* snapshot, std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || snapshot == nullptr || snapshot->empty()) return false;
    // Merged ranges are ordered by both bounds: the first one ending past the access decides.
    const auto it = std::partition_point(snapshot->begin(), snapshot->end(), [&](const auto& range) { return range.second <= address; });
    return it != snapshot->end() && it->first < address + bytes;
}

bool SnapshotOverlaps(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0) return false;
    const auto snapshot = pendingWrites.load(std::memory_order_acquire);
    return SnapshotOverlaps(snapshot.get(), address, bytes);
}

// Whether one merged range of the snapshot contains [address, end) entirely.
bool SnapshotCovers(std::uint64_t address, std::uint64_t end) {
    const auto snapshot = pendingWrites.load(std::memory_order_acquire);
    if (snapshot == nullptr || snapshot->empty()) return false;
    const auto it = std::partition_point(snapshot->begin(), snapshot->end(), [&](const auto& range) { return range.second <= address; });
    return it != snapshot->end() && it->first <= address && end <= it->second;
}

// Attribution of the pending-write syncs the hook makes (the [hooksync] line every 10 s, under
// APS5_PROFILE_DRAW; APS5_NO_HOOKSYNC_PROFILE=1 leaves only the plain counts). Every sync is charged
// to the packet the accessing thread executes and the read site it named (or, when it named none,
// its return addresses), together with the noted range it hit, the batch that noted it and the
// time waited. A small read (up to 64 KiB) has its bytes compared before and after the wait: a
// sync whose bytes come out unchanged although its target batch had not run yet was not needed
// for that read (the noted range was larger than what the GPU wrote, or the GPU wrote the same
// values), which separates false overlaps from real producer/consumer dependencies; see
// HookSyncOutcomes for the cases that prove nothing. All state lives under the GpuMutex, which
// the hook holds.
// Debug aid: APS5_NO_SYNC_THROUGH=1 makes every pending-write sync a full Sync (SyncThrough and
// the attribution's DescribePendingWrite read the same switch).
bool SyncThroughEnabled() {
    static const bool enabled = std::getenv("APS5_NO_SYNC_THROUGH") == nullptr;
    return enabled;
}

// Six frames: whether the hook's constructor and FlushGpuWrites are inlined into the GuestMemory
// entry decides at which frame the driver caller sits, so enough are kept to see its own caller.
constexpr std::size_t HookSyncFrames = 6;

struct HookSyncKey {
    std::uint32_t queue;
    std::uint32_t opcode;
    GuestMemory::ReadSite site;
    std::array<unsigned long long, HookSyncFrames> frames;
    bool operator<(const HookSyncKey& other) const {
        return std::tie(queue, opcode, site, frames) < std::tie(other.queue, other.opcode, other.site, other.frames);
    }
};

// Whether the bytes of a small access changed over the wait, split by the state of the target
// batch: only an unchanged access whose target had not run yet (open, or in flight with its fence
// unsignaled) shows the sync was not needed; against a signaled fence the copy made before the
// wait already held the GPU's values, so unchanged proves nothing. A store (ReadSite::Store) is
// counted apart: its sync orders a CPU write after the GPU's, whatever the bytes do.
struct HookSyncOutcomes {
    std::uint64_t unchangedOpen = 0;
    std::uint64_t unchangedPending = 0;
    std::uint64_t unchangedSignaled = 0;
    std::uint64_t changed = 0;
    std::uint64_t stores = 0;
    std::uint64_t unchecked = 0;
};

struct HookSyncTotals {
    std::uint64_t count = 0;
    HookSyncOutcomes outcomes;
    std::uint64_t rangeBytes = 0;
    std::uint64_t accessBytes = 0;
    double waitedMs = 0;
};

// Size buckets for the noted ranges and the accesses: <=4K, <=64K, <=1M, <=16M, <=256M, larger.
constexpr std::size_t SizeBuckets = 6;
constexpr const char* SizeBucketNames[SizeBuckets] = {"<=4K", "<=64K", "<=1M", "<=16M", "<=256M", ">256M"};

std::size_t SizeBucket(std::uint64_t bytes) {
    if (bytes <= 4096) return 0;
    if (bytes <= 65536) return 1;
    if (bytes <= (1u << 20u)) return 2;
    if (bytes <= (16u << 20u)) return 3;
    if (bytes <= (256u << 20u)) return 4;
    return 5;
}

struct HookSyncStats {
    std::map<HookSyncKey, HookSyncTotals> byKey;
    std::uint64_t count = 0, openTargets = 0, signaledTargets = 0, batchesFinished = 0;
    HookSyncOutcomes outcomes;
    double waitedMs = 0;
    std::array<std::uint64_t, SizeBuckets> rangeBuckets{};
    std::array<std::uint64_t, SizeBuckets> accessBuckets{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

HookSyncStats& HookSyncs() {
    static HookSyncStats stats;
    return stats;
}

bool HookSyncProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr && std::getenv("APS5_NO_HOOKSYNC_PROFILE") == nullptr;
    return profiled;
}

std::string PacketName(std::uint32_t opcode) {
    if (opcode == GuestMemory::NoPacket) return "no-packet";
    if (opcode == 0xffffu) return "flip";
    // 0xfffd is reserved for the deferred-label stores (recordDeferredLabels), once its owner
    // brackets them with SetCurrentPacket; until then they are charged to the following packet.
    if (opcode == 0xfffdu) return "deferred-labels";
    for (const auto& entry : Pm4::Opcodes) {
        if (entry.value == opcode) return std::string(entry.name);
    }
    char text[24];
    std::snprintf(text, sizeof(text), "op 0x%x", opcode);
    return text;
}

void ReportHookSyncs(HookSyncStats& stats) {
    std::vector<std::pair<const HookSyncKey*, const HookSyncTotals*>> hot;
    hot.reserve(stats.byKey.size());
    for (const auto& [key, totals] : stats.byKey) hot.emplace_back(&key, &totals);
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.second->waitedMs > b.second->waitedMs; });
    std::string report;
    char text[512];
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto& o = stats.outcomes;
    std::snprintf(text, sizeof(text), "[hooksync] %llu pending-write syncs waited %.0f ms (10 s); read bytes after the wait: unchanged %llu (target open %llu, in flight unsignaled %llu, signaled %llu), changed %llu; stores %llu, unchecked %llu; targets: %llu open, %llu already signaled, %llu batches finished; noted range:", count(stats.count), stats.waitedMs, count(o.unchangedOpen + o.unchangedPending + o.unchangedSignaled), count(o.unchangedOpen), count(o.unchangedPending), count(o.unchangedSignaled), count(o.changed), count(o.stores), count(o.unchecked), count(stats.openTargets), count(stats.signaledTargets), count(stats.batchesFinished));
    report += text;
    for (std::size_t i = 0; i < SizeBuckets; ++i) {
        if (stats.rangeBuckets[i] == 0) continue;
        std::snprintf(text, sizeof(text), " %s %llu", SizeBucketNames[i], static_cast<unsigned long long>(stats.rangeBuckets[i]));
        report += text;
    }
    report += "; access:";
    for (std::size_t i = 0; i < SizeBuckets; ++i) {
        if (stats.accessBuckets[i] == 0) continue;
        std::snprintf(text, sizeof(text), " %s %llu", SizeBucketNames[i], static_cast<unsigned long long>(stats.accessBuckets[i]));
        report += text;
    }
    report += "; top by wait (queue packet site frames: count/ms, unchanged open/unsignaled/signaled, changed, stores, avg range/access):";
    for (std::size_t i = 0; i < hot.size() && i < 10; ++i) {
        const auto& key = *hot[i].first;
        const auto& totals = *hot[i].second;
        const auto& k = totals.outcomes;
        std::snprintf(text, sizeof(text), " [0x%x %s %s +0x%llx/+0x%llx/+0x%llx/+0x%llx/+0x%llx/+0x%llx: %llu/%.0fms u%llu/%llu/%llu c%llu s%llu %.0fK/%.0fK]", key.queue, PacketName(key.opcode).c_str(), GuestMemory::ReadSiteName(key.site), key.frames[0], key.frames[1], key.frames[2], key.frames[3], key.frames[4], key.frames[5], count(totals.count), totals.waitedMs, count(k.unchangedOpen), count(k.unchangedPending), count(k.unchangedSignaled), count(k.changed), count(k.stores), totals.rangeBytes / 1024.0 / totals.count, totals.accessBytes / 1024.0 / totals.count);
        report += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", report.c_str());
    stats = HookSyncStats{};
}

// Wraps one pending-write sync (constructed before it, under the GpuMutex): gathers what the sync
// waits for and, in the destructor, the time it took and whether the bytes changed.
class HookSyncScope {
public:
    HookSyncScope(const Recorder& recorder, std::uint64_t address, std::size_t bytes) : enabled(HookSyncProfiled()), address(address), bytes(bytes) {
        if (!enabled) return;
        info = recorder.DescribePendingWrite(address, bytes);
        const auto packet = GuestMemory::CurrentPacket();
        key = HookSyncKey{packet.queue, packet.opcode, GuestMemory::CurrentReadSite(), {}};
        // Frame 0 is the hook's caller (the GuestMemory entry, or the driver code when the entry
        // was inlined); the driver code that made the access is one of the next frames.
        GuestMemory::CaptureCallerOffsets(key.frames, 1);
        // The access's bytes are copied raw: a Read here would re-enter the hook. The same access
        // check the accessing code makes guards the copy; a range that fails it is left unchecked.
        // A store's bytes are not compared (counted as a store instead).
        if (key.site != GuestMemory::ReadSite::Store && bytes <= CompareLimit && GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) {
            before.resize(bytes);
            std::memcpy(before.data(), reinterpret_cast<const void*>(address), bytes);
        }
        start = std::chrono::steady_clock::now();
    }
    ~HookSyncScope() {
        if (!enabled) return;
        const auto now = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration<double, std::milli>(now - start).count();
        auto& stats = HookSyncs();
        auto& totals = stats.byKey[key];
        ++totals.count;
        totals.waitedMs += ms;
        totals.accessBytes += bytes;
        ++stats.count;
        stats.waitedMs += ms;
        stats.accessBuckets[SizeBucket(bytes)] += 1;
        if (info.has_value()) {
            const auto rangeBytes = info->rangeEnd - info->rangeBegin;
            totals.rangeBytes += rangeBytes;
            stats.rangeBuckets[SizeBucket(rangeBytes)] += 1;
            if (info->open) ++stats.openTargets;
            if (info->signaled) ++stats.signaledTargets;
            stats.batchesFinished += info->batchesToFinish;
        }
        // One outcome per sync, on both the key's and the interval's counters.
        const auto outcome = [&]() -> std::uint64_t HookSyncOutcomes::* {
            if (key.site == GuestMemory::ReadSite::Store) return &HookSyncOutcomes::stores;
            if (before.empty() || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) return &HookSyncOutcomes::unchecked;
            if (std::memcmp(before.data(), reinterpret_cast<const void*>(address), bytes) != 0) return &HookSyncOutcomes::changed;
            // A target already signaled (or no target found: the batch is finishing) had run
            // before the copy, so its unchanged bytes are not evidence of a false overlap.
            if (!info.has_value() || info->signaled) return &HookSyncOutcomes::unchangedSignaled;
            return info->open ? &HookSyncOutcomes::unchangedOpen : &HookSyncOutcomes::unchangedPending;
        }();
        ++(totals.outcomes.*outcome);
        ++(stats.outcomes.*outcome);
        if (now - stats.lastReport > std::chrono::seconds(10)) ReportHookSyncs(stats);
    }
    HookSyncScope(const HookSyncScope&) = delete;
    HookSyncScope& operator=(const HookSyncScope&) = delete;

private:
    static constexpr std::size_t CompareLimit = 65536;
    bool enabled;
    std::uint64_t address;
    std::size_t bytes;
    std::optional<Recorder::PendingWriteInfo> info;
    HookSyncKey key{};
    std::vector<std::byte> before;
    std::chrono::steady_clock::time_point start;
};

// Attribution of the hook's recorded-store syncs (source 2: a pending image's write-back or a unit
// publish was recorded for the access and SyncThrough waits for it), the [hooksync] recorded-store
// line every 10 s under the same switch as HookSyncScope: per (queue, packet, read site), the syncs,
// their wait, the batches in flight at the sync, the C4 classification of the access (every 64 KiB
// block CPU-written since the overlapping images' generations: the store would not change a byte
// the CPU reads), whether the bytes changed over the wait, whether the images overlapped had a
// consumer proof within StorageTexture::DeadImagePresents presents, and the accesses the C4 skip
// took instead of a flush (with the images that left the registry there as an empty write-back
// would have left them). The skips are counted without the GpuMutex (their own mutex); the syncs
// under it.
struct RecordedStoreTotals {
    std::uint64_t count = 0;
    double waitedMs = 0;
    std::uint64_t batchesFinished = 0;
    std::uint64_t cpuWrittenYes = 0, cpuWrittenNo = 0, unclassified = 0, publishOnly = 0;
    std::uint64_t changed = 0, unchanged = 0, unchecked = 0;
    std::uint64_t imagesDead = 0, imagesLive = 0;
    std::uint64_t skipped = 0, flushed = 0, evicted = 0;
};

struct RecordedStoreStats {
    std::mutex mutex;
    std::map<HookSyncKey, RecordedStoreTotals> byKey;
    RecordedStoreTotals totals;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

RecordedStoreStats& RecordedStoreSyncs() {
    static RecordedStoreStats stats;
    return stats;
}

void ReportRecordedStoreSyncs(RecordedStoreStats& stats) {
    std::vector<std::pair<const HookSyncKey*, const RecordedStoreTotals*>> hot;
    for (const auto& [key, totals] : stats.byKey) hot.emplace_back(&key, &totals);
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return std::tie(a.second->waitedMs, a.second->skipped) > std::tie(b.second->waitedMs, b.second->skipped); });
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto& t = stats.totals;
    const auto skips = StorageTexture::TakeHookSkipCounts();
    char text[640];
    std::snprintf(text, sizeof(text), "[hooksync] recorded-store syncs (10 s): %llu waited %.0f ms, %llu batches in flight at them; skipped (cpu-written blocks) %llu / flushed %llu / evicted stale %llu; flushed later after a skip %llu (by the same site %llu); C4 every block CPU-written: yes %llu / no %llu / unclassified %llu (publish only %llu); bytes after the wait: changed %llu, unchanged %llu, unchecked %llu; images overlapped: dead %llu / live %llu (no consumer proof for %llu presents); by queue packet site (count/ms, skipped, cpu-written yes/no, changed/unchanged, dead/live):", count(t.count), t.waitedMs, count(t.batchesFinished), count(t.skipped), count(t.flushed), count(t.evicted), count(skips.flushedAfterSkip), count(skips.flushedAfterSkipSameSite), count(t.cpuWrittenYes), count(t.cpuWrittenNo), count(t.unclassified), count(t.publishOnly), count(t.changed), count(t.unchanged), count(t.unchecked), count(t.imagesDead), count(t.imagesLive), count(StorageTexture::DeadImagePresents));
    std::string report = text;
    for (std::size_t i = 0; i < hot.size() && i < 10; ++i) {
        const auto& key = *hot[i].first;
        const auto& k = *hot[i].second;
        std::snprintf(text, sizeof(text), " [0x%x %s %s +0x%llx/+0x%llx/+0x%llx: %llu/%.0fms k%llu y%llu/n%llu c%llu/u%llu d%llu/l%llu]", key.queue, PacketName(key.opcode).c_str(), GuestMemory::ReadSiteName(key.site), key.frames[0], key.frames[1], key.frames[2], count(k.count), k.waitedMs, count(k.skipped), count(k.cpuWrittenYes), count(k.cpuWrittenNo), count(k.changed), count(k.unchanged), count(k.imagesDead), count(k.imagesLive));
        report += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", report.c_str());
    stats.byKey.clear();
    stats.totals = {};
}

// The key of the access the hook handles now: always inlined into FlushForAccess so that frame 0
// is the hook's caller for the skip counters and the sync scope alike, whatever the inliner does
// with the scope's constructor.
[[gnu::always_inline]] inline HookSyncKey RecordedStoreKey() {
    const auto packet = GuestMemory::CurrentPacket();
    HookSyncKey key{packet.queue, packet.opcode, GuestMemory::CurrentReadSite(), {}};
    GuestMemory::CaptureCallerOffsets(key.frames, 1);
    return key;
}

void CountRecordedStoreSkip(const HookSyncKey& key, std::size_t evicted) {
    auto& stats = RecordedStoreSyncs();
    std::lock_guard lock(stats.mutex);
    for (auto* totals : {&stats.byKey[key], &stats.totals}) {
        ++totals->skipped;
        totals->evicted += evicted;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.lastReport > std::chrono::seconds(10)) {
        stats.lastReport = now;
        ReportRecordedStoreSyncs(stats);
    }
}

class RecordedStoreSyncScope {
public:
    RecordedStoreSyncScope(const Recorder& recorder, std::uint64_t address, std::size_t bytes, const StorageTexture::AccessClassification& classification, bool stored, const HookSyncKey& key) : enabled(HookSyncProfiled()), address(address), bytes(bytes), classification(classification), stored(stored), key(key) {
        if (!enabled) return;
        batchesInFlight = recorder.InFlightBatches();
        if (key.site != GuestMemory::ReadSite::Store && bytes <= CompareLimit && GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) {
            before.resize(bytes);
            std::memcpy(before.data(), reinterpret_cast<const void*>(address), bytes);
        }
        start = std::chrono::steady_clock::now();
    }
    ~RecordedStoreSyncScope() {
        if (!enabled) return;
        const auto now = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration<double, std::milli>(now - start).count();
        auto& stats = RecordedStoreSyncs();
        std::lock_guard lock(stats.mutex);
        const auto add = [&](RecordedStoreTotals& totals) {
            ++totals.count;
            totals.waitedMs += ms;
            totals.batchesFinished += batchesInFlight;
            if (stored) ++totals.flushed;
            if (classification.images == 0) ++totals.publishOnly;
            else if (!classification.checked) ++totals.unclassified;
            else ++(classification.allCpuWritten ? totals.cpuWrittenYes : totals.cpuWrittenNo);
            totals.imagesDead += classification.dead;
            totals.imagesLive += classification.live;
            if (before.empty() || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) ++totals.unchecked;
            else if (std::memcmp(before.data(), reinterpret_cast<const void*>(address), bytes) != 0) ++totals.changed;
            else ++totals.unchanged;
        };
        add(stats.byKey[key]);
        add(stats.totals);
        if (now - stats.lastReport > std::chrono::seconds(10)) {
            stats.lastReport = now;
            ReportRecordedStoreSyncs(stats);
        }
    }
    RecordedStoreSyncScope(const RecordedStoreSyncScope&) = delete;
    RecordedStoreSyncScope& operator=(const RecordedStoreSyncScope&) = delete;

private:
    static constexpr std::size_t CompareLimit = 65536;
    bool enabled;
    std::uint64_t address;
    std::size_t bytes;
    StorageTexture::AccessClassification classification;
    bool stored;
    HookSyncKey key;
    std::size_t batchesInFlight = 0;
    std::vector<std::byte> before;
    std::chrono::steady_clock::time_point start;
};

// GuestMemory flush hook: a CPU access to memory that recorded GPU work will write waits for that
// work first; then storage image results pending for the range are stored. The recorder is only
// dereferenced under the GpuMutex (it is destroyed with its device under that lock).
void FlushForAccess(std::uint64_t address, std::size_t bytes) {
    hookCalls.fetch_add(1, std::memory_order_relaxed);
    // A label this worker queued but has not recorded writes the range: in queue order the label
    // precedes this access, so the group is recorded first (it then notes the range as a pending
    // write of the open batch, and the check below syncs on it like any recorded store). Not from
    // inside a completion action (a write-back store made while a batch is reaped): the record
    // would reap nested in the reap in progress, landing later batches' write-backs before the
    // rest of this one's, and this queue's queued label follows every batch of its own in flight
    // anyway, so the write-back landing first is the order the hardware gives.
    if (!QueuedLabelRanges().empty() && QueuedLabelOverlaps(address, bytes)) {
        if (completionDepth != 0 && HookCompletionGuard()) {
            queuedLabelHookInCompletion.fetch_add(1, std::memory_order_relaxed);
        } else if (auto* record = queuedLabelRecorder.load(std::memory_order_acquire); record != nullptr) {
            queuedLabelHookRecords.fetch_add(1, std::memory_order_relaxed);
            record();
        }
    }
    if (!HookSnapshotEnabled() || SnapshotOverlaps(address, bytes)) {
        hookLocks.fetch_add(1, std::memory_order_relaxed);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard gpu(GuestMemory::GpuMutex());
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->PendingWriteOverlaps(address, bytes)) {
            const HookSyncScope attribution(*recorder, address, bytes);
            Recorder::CountSync(1);
            // The wait runs without the mutex when this acquisition is the outermost one (a stage-A
            // read, a game thread); the recorder is not dereferenced after the call.
            recorder->SyncThrough(address, bytes, true);
        }
    }
    // A reader takes every shadowed unit of the range; a store only the units it covers partly
    // (its own stamp makes the others stale). Inside a completion action a store has no publish
    // recorded: the batch is not waited for, and the copy would land over the store later.
    const auto site = GuestMemory::CurrentReadSite();
    const auto scope = site != GuestMemory::ReadSite::Store ? PublishScope::Whole : completionDepth != 0 ? PublishScope::None : PublishScope::PartialUnits;
    bool published = false;
    // The classification is made before the store (the store changes what it measures); profile only.
    StorageTexture::AccessClassification classification{};
    if (HookSyncProfiled()) StorageTexture::ClassifyAccess(address, bytes, classification);
    // C4: an access whose every 64 KiB block the CPU wrote since the overlapping images' generations
    // reads the same bytes with or without their store (the store keeps those blocks), so the
    // images stay pending (short of the layers the store would have left empty-handed anyway) and
    // only the unit shadows over the range are published; with no image over the range that
    // publish is all FlushPending would do (one registry scan, not two).
    bool stored = false;
    std::size_t images = 0, evicted = 0;
    const bool kept = !HookFlushCpuBlocks() && StorageTexture::AccessKeptByCpu(address, bytes, &images, &evicted);
    if (kept || (images == 0 && !HookFlushCpuBlocks())) {
        published = StorageTexture::WriteBackMirrors(address, bytes, scope);
        if (kept && HookSyncProfiled()) CountRecordedStoreSkip(RecordedStoreKey(), evicted);
    } else {
        stored = StorageTexture::FlushPending(address, bytes, nullptr, "memory access", scope, &published);
    }
    if (stored || published) {
        // Stores into imported memory were only recorded; the CPU is about to read them. The wait
        // targets the batch holding them (the open one: SyncThrough submits it) and runs with the
        // mutex released when this acquisition is the outermost; a nested hook waits locked through
        // that batch ([lock] 'locked GPU waits'). Inside a completion action the store must land
        // before the completing batch's CPU write-back (an older result under a newer one), which
        // only the full Sync() gives there.
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard gpu(GuestMemory::GpuMutex());
        if (auto* recorder = Recorder::Active(); recorder != nullptr) {
            const RecordedStoreSyncScope attribution(*recorder, address, bytes, classification, stored, HookSyncProfiled() ? RecordedStoreKey() : HookSyncKey{});
            Recorder::CountSync(2);
            if (HookFullSync() || completionDepth != 0) recorder->Sync();
            else recorder->SyncThrough(address, bytes, true);
        }
    }
}

}

Recorder::Recorder(const Context& context, bool timelineSemaphores) : context(context), id(nextRecorderId.fetch_add(1)) {
    labelSince.fill(NoPendingLabel);
    {
        std::lock_guard lock(liveRecordersMutex);
        liveRecorders.push_back(id);
    }
    // Idempotent: the same hook for every recorder (the deferred lists are per thread, not per recorder).
    GuestMemory::SetGpuUnlockHook(&ReleaseDeferredKeeps);
    if (ProcTable() && context.deviceProc != nullptr) {
        getFenceStatus = context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus");
        resetFences = context.Function<PFN_vkResetFences>("vkResetFences");
        waitForFences = context.Function<PFN_vkWaitForFences>("vkWaitForFences");
        cmdUpdateBuffer = context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
        beginCommandBuffer = context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer");
        endCommandBuffer = context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer");
        queueSubmit = context.Function<PFN_vkQueueSubmit>("vkQueueSubmit");
        cmdPipelineBarrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        cmdBeginQuery = context.Function<PFN_vkCmdBeginQuery>("vkCmdBeginQuery");
        cmdEndQuery = context.Function<PFN_vkCmdEndQuery>("vkCmdEndQuery");
        cmdResetQueryPool = context.Function<PFN_vkCmdResetQueryPool>("vkCmdResetQueryPool");
        cmdCopyQueryPoolResults = context.Function<PFN_vkCmdCopyQueryPoolResults>("vkCmdCopyQueryPoolResults");
        cmdBindPipeline = context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
        cmdPushConstants = context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        cmdDispatch = context.Function<PFN_vkCmdDispatch>("vkCmdDispatch");
    }
    streams.resize(1);
    streams.front().queue = context.queue;
    if (!timelineSemaphores) return;
    // The timeline starts at 0 and every Submit signals its serial (1, 2, ...): a value that only
    // grows, so a thread waiting for it outside the mutex can never observe a reset or a reused
    // handle (the batch fences are pooled and reset in release()). A failure here only disables the
    // unlocked waits; the locked Sync path still works.
    VkSemaphoreTypeCreateInfoKHR type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
    type.initialValue = 0;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    const auto createSemaphore = context.Function<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    const auto result = createSemaphore(context.device, &info, nullptr, &streams.front().timeline);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "[gpu] timeline semaphore creation failed (Vulkan result %d); drains wait under the GPU mutex\n", static_cast<int>(result));
        streams.front().timeline = VK_NULL_HANDLE;
        return;
    }
    const auto queues = std::clamp<std::uint32_t>(context.streamQueueCount, 1, static_cast<std::uint32_t>(MaxRecorderStreams));
    for (std::uint32_t index = 1; index < queues; ++index) {
        Stream stream;
        stream.queue = context.streamQueues[index];
        Check(createSemaphore(context.device, &info, nullptr, &stream.timeline), "vkCreateSemaphore recorder stream");
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = context.streamQueueFamily;
        const auto created = context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &poolInfo, nullptr, &stream.pool);
        if (created != VK_SUCCESS) context.Function<PFN_vkDestroySemaphore>("vkDestroySemaphore")(context.device, stream.timeline, nullptr);
        Check(created, "vkCreateCommandPool recorder stream");
        streams.push_back(stream);
    }
}

Recorder::~Recorder() {
    try {
        Sync();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[gpu] recorder teardown: %s\n", error.what());
    }
    holdKept.clear();
    if (activeRecorder == this) {
        activeRecorder = nullptr;
        pendingWrites.store(nullptr, std::memory_order_release);
        publishGeneration.fetch_add(1, std::memory_order_release);
        pendingLabelSince.store(NoPendingLabel, std::memory_order_release);
        completionLabels.store(0, std::memory_order_release);
        writeBackCompletions.store(0, std::memory_order_release);
        workSinceSubmit.store(0, std::memory_order_release);
    }
    {
        // A thread that released the mutex around a wait on this recorder re-checks the id when it
        // retakes the mutex and touches nothing then. That check is only sound while the device
        // (and so this destructor) is torn down under the GpuMutex, as the driver's drain and
        // failure paths do; a worker dropping the last device reference after releasing its lock
        // would run this without the mutex, concurrently with a relocked waiter.
        std::lock_guard lock(liveRecordersMutex);
        liveRecorders.erase(std::remove(liveRecorders.begin(), liveRecorders.end(), id), liveRecorders.end());
    }
    // Sync() above completed every batch, so every timeline wait in progress on this recorder
    // returns now; the semaphore is destroyed only once they all left it.
    while (WaitersOf(id).load(std::memory_order_acquire) != 0) std::this_thread::yield();
    // The kept objects of every batch this thread finished (Sync() above included) belong to the
    // device that is going away: destroyed now, here (no hand-off: nothing should stay in flight
    // behind this thread); the release thread is then stopped and joined (it drains its queue
    // first, holding no mutex this thread holds: the destructors take only their caches' own
    // mutexes, which order after the GpuMutex), and the releases still in progress inline on
    // other threads (an unlock hook that found the queue stopping or full) are waited for before
    // the caller destroys the device. The count is global: a newer recorder's batches (device
    // replacement) are waited for too, which only prolongs the spin.
    if (!DeferredBatches().empty()) DestroyDeferred(TakeDeferred(), false);
    JoinReleaseThread();
    while (deferredPending.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    {
        // Unlocked lookups stop before `labels` is destroyed (after this body).
        std::lock_guard tableLock(labelTableMutex);
        if (labelTableOwner == this) labelTableOwner = nullptr;
    }
    if (meshArgumentPipeline != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, meshArgumentPipeline, nullptr);
    if (meshArgumentLayout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, meshArgumentLayout, nullptr);
    if (dispatchThreadsPipeline != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, dispatchThreadsPipeline, nullptr);
    if (dispatchThreadsLayout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, dispatchThreadsLayout, nullptr);
    if (sampleCounterState > 0) {
        samplesPassed.store(SamplesTotal(), std::memory_order_release);
        if (samplePools != nullptr) std::static_pointer_cast<SamplePoolCache>(samplePools)->close();
        if (segmentPools != nullptr) std::static_pointer_cast<SamplePoolCache>(segmentPools)->close();
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, samplePipeline, nullptr);
        context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, sampleLayout, nullptr);
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, dumpPipeline, nullptr);
        context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, dumpLayout, nullptr);
        sampleCounter.reset();
    }
    for (std::uint32_t stream = 0; stream < streams.size(); ++stream) {
        for (auto& [commands, fence] : streams[stream].spare) {
            context.Function<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(context.device, streamPool(stream), 1, &commands);
            context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
        }
        streams[stream].spare.clear();
        if (streams[stream].pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, streams[stream].pool, nullptr);
        streams[stream].pool = VK_NULL_HANDLE;
    }
    for (const auto pool : sparePools) context.Function<PFN_vkDestroyQueryPool>("vkDestroyQueryPool")(context.device, pool, nullptr);
    sparePools.clear();
    // Sync() above waited for every batch, so no submission still signals the timeline.
    for (auto& stream : streams) {
        if (stream.timeline != VK_NULL_HANDLE) context.Function<PFN_vkDestroySemaphore>("vkDestroySemaphore")(context.device, stream.timeline, nullptr);
        stream.timeline = VK_NULL_HANDLE;
    }
}

std::optional<std::chrono::steady_clock::time_point> Recorder::PendingLabelSince() {
    const auto since = pendingLabelSince.load(std::memory_order_acquire);
    if (since == NoPendingLabel) return std::nullopt;
    return std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(since));
}

std::uint64_t Recorder::WriteGeneration() {
    return writeGeneration.load(std::memory_order_acquire);
}

std::uint64_t Recorder::PublishGeneration() {
    return publishGeneration.load(std::memory_order_acquire);
}

std::uint64_t Recorder::PendingCompletionLabels() {
    return completionLabels.load(std::memory_order_acquire);
}

std::uint64_t Recorder::PendingWriteBackCompletions() {
    return writeBackCompletions.load(std::memory_order_acquire);
}

std::uint64_t Recorder::RecordedWorkSinceSubmit() {
    return workSinceSubmit.load(std::memory_order_relaxed);
}

void Recorder::CountRecordedWork() {
    workSinceSubmit.fetch_add(1, std::memory_order_relaxed);
}

Recorder* Recorder::Active() {
    return activeRecorder;
}

void Recorder::Activate() {
    const std::lock_guard recordingGuard(recording);
    activeRecorder = this;
    {
        std::lock_guard tableLock(labelTableMutex);
        labelTableOwner = this;
    }
    GuestMemory::SetFlushHook(&FlushForAccess);
}

std::optional<Recorder::LabelHit> Recorder::LookupLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, const LabelAccepts& accepts, bool* behindCompletion) {
    std::lock_guard tableLock(labelTableMutex);
    if (behindCompletion != nullptr) *behindCompletion = false;
    if (labelTableOwner == nullptr) return std::nullopt;
    return labelTableOwner->lookupLabel(address, bytes, afterStamp, accepts, behindCompletion);
}

bool Recorder::WideLabelIn(std::uint64_t address, std::size_t bytes) {
    std::lock_guard tableLock(labelTableMutex);
    return labelTableOwner != nullptr && labelTableOwner->wideLabelInLocked(address, bytes);
}

bool Recorder::wideLabelInLocked(std::uint64_t address, std::size_t bytes) const {
    if (wideLabels.empty() || bytes == 0) return false;
    const auto end = address + bytes;
    for (auto it = wideLabels.lower_bound(address >= wideLabelBytes ? address - wideLabelBytes + 1 : 0); it != wideLabels.end() && it->first < end; ++it) {
        if (it->second.end > address) return true;
    }
    return false;
}

bool Recorder::PendingLabelsMatchMemory(std::uint64_t address, std::size_t bytes) {
    std::lock_guard tableLock(labelTableMutex);
    return labelTableOwner == nullptr || labelTableOwner->labelsMatchMemoryLocked(address, bytes);
}

bool Recorder::labelsMatchMemoryLocked(std::uint64_t address, std::size_t bytes) const {
    if (bytes == 0) return true;
    const auto end = address + bytes;
    const auto overlapsTable = [&](const auto& table) {
        const auto first = table.lower_bound(address >= 3 ? address - 3 : 0);
        return first != table.end() && first->first < end;
    };
    const bool tabled = overlapsTable(labels) || overlapsTable(queuedLabels);
    if (!tabled && !wideLabelInLocked(address, bytes)) return true;
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address & ~3ull), static_cast<std::size_t>(((end + 3) & ~3ull) - (address & ~3ull)))) return false;
    const auto matches = [](std::uint64_t dword, const LabelWriter& writer) { return std::memcmp(&writer.value, reinterpret_cast<const void*>(dword), 4) == 0; };
    for (auto it = labels.lower_bound(address >= 3 ? address - 3 : 0); it != labels.end() && it->first < end; ++it) {
        if (it->first + 4 <= address) continue;
        for (const auto& writer : it->second) {
            if (!matches(it->first, writer)) return false;
        }
    }
    for (auto it = queuedLabels.lower_bound(address >= 3 ? address - 3 : 0); it != queuedLabels.end() && it->first < end; ++it) {
        if (it->first + 4 > address && !matches(it->first, it->second)) return false;
    }
    if (!wideLabels.empty()) {
        for (auto it = wideLabels.lower_bound(address >= wideLabelBytes ? address - wideLabelBytes + 1 : 0); it != wideLabels.end() && it->first < end; ++it) {
            const auto from = std::max(address, it->first);
            const auto to = std::min(end, it->second.end);
            if (from >= to) continue;
            if (std::memcmp(it->second.bytes.data() + (from - it->first), reinterpret_cast<const void*>(from), static_cast<std::size_t>(to - from)) != 0) return false;
        }
    }
    return true;
}

std::optional<std::uint64_t> Recorder::LookupLabelValue(std::uint64_t address, std::size_t bytes) {
    std::lock_guard tableLock(labelTableMutex);
    if (labelTableOwner == nullptr || (bytes != 4 && bytes != 8) || address % 4 != 0) return std::nullopt;
    const auto tag = GuestMemory::GpuLockThreadTag();
    const auto& queued = labelTableOwner->queuedLabels;
    const auto& recorded = labelTableOwner->labels;
    std::uint64_t value = 0;
    for (std::size_t offset = 0; offset < bytes; offset += 4) {
        const LabelWriter* newest = nullptr;
        if (const auto own = queued.find(address + offset); own != queued.end() && own->second.queue == tag) {
            newest = &own->second;
        } else if (const auto found = recorded.find(address + offset); found != recorded.end()) {
            for (const auto& writer : found->second) {
                if (newest == nullptr || writer.stamp > newest->stamp) newest = &writer;
            }
        }
        if (newest == nullptr) return std::nullopt;
        value |= static_cast<std::uint64_t>(newest->value) << (offset * 8u);
    }
    return value;
}

void Recorder::NoteQueuedLabel(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue) {
    if (bytes.size() < 4 || address % 4 != 0) return;
    // The range before the table: the hook covers the group whether or not a table exists to
    // enter it in (no owner: the callback records it outright, by a CPU store when need be).
    QueuedLabelRanges().emplace_back(address, address + bytes.size());
    queuedLabelsNoted.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard tableLock(labelTableMutex);
    if (labelTableOwner == nullptr) return;
    const auto& recorded = labelTableOwner->labels;
    for (std::size_t offset = 0; offset + 4 <= bytes.size(); offset += 4) {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, 4);
        if (recorded.contains(address + offset)) queuedLabelsOverRecorded.fetch_add(1, std::memory_order_relaxed);
        labelTableOwner->queuedLabels.insert_or_assign(address + offset, LabelWriter{NoStream, 0, queue, stamp, value});
    }
}

void Recorder::ForgetQueuedLabels() {
    if (QueuedLabelRanges().empty()) return;
    auto ranges = std::move(QueuedLabelRanges());
    QueuedLabelRanges().clear();
    std::lock_guard tableLock(labelTableMutex);
    if (labelTableOwner == nullptr) return;
    auto& queued = labelTableOwner->queuedLabels;
    const auto tag = GuestMemory::GpuLockThreadTag();
    for (const auto& [begin, end] : ranges) {
        for (auto dword = begin; dword + 4 <= end; dword += 4) {
            if (const auto own = queued.find(dword); own != queued.end() && own->second.queue == tag) queued.erase(own);
        }
    }
}

void Recorder::SetQueuedLabelRecorder(void (*recorder)()) {
    queuedLabelRecorder.store(recorder, std::memory_order_release);
}

Recorder::StoreStatistics Recorder::StoreCounts() {
    return StoreStatistics{storeCount.load(std::memory_order_relaxed), storeRuns.load(std::memory_order_relaxed), storesJoined.load(std::memory_order_relaxed), storesReplaced.load(std::memory_order_relaxed), storeWawBarriers.load(std::memory_order_relaxed), storeJoinsRefused.load(std::memory_order_relaxed), queuedLabelsNoted.load(std::memory_order_relaxed), queuedLabelsOverRecorded.load(std::memory_order_relaxed), queuedLabelHits.load(std::memory_order_relaxed), queuedLabelHookRecords.load(std::memory_order_relaxed), queuedLabelHookInCompletion.load(std::memory_order_relaxed), keyStoreCount.load(std::memory_order_relaxed), keyStoreRuns.load(std::memory_order_relaxed), keyStoreRunsForWriter.load(std::memory_order_relaxed), keyStoresJoined.load(std::memory_order_relaxed), storeRunsAtSubmit.load(std::memory_order_relaxed), storeRunsForced.load(std::memory_order_relaxed)};
}

std::uint64_t Recorder::ThreadHookWaits() {
    return hookRealWaits;
}

bool Recorder::SnapshotWriteOverlaps(std::uint64_t address, std::size_t bytes) {
    return AgcDriver::Graphics::SnapshotOverlaps(address, bytes);
}

std::shared_ptr<const Recorder::WriteRanges> Recorder::PendingWriteSnapshot() {
    return pendingWrites.load(std::memory_order_acquire);
}

bool Recorder::SnapshotOverlaps(const WriteRanges* snapshot, std::uint64_t address, std::size_t bytes) {
    return AgcDriver::Graphics::SnapshotOverlaps(snapshot, address, bytes);
}

void Recorder::CountSync(int source, const void* site) {
    if (source < 0 || source >= 5) return;
    ++syncCounts[source];
    announcedSource = source;
    announcedSite = site;
}

void Recorder::AnnounceSyncSite(const void* site) {
    announcedSite = site;
}

double Recorder::ThreadWaitedMs() {
    return threadWaitedMs;
}

std::uint64_t Recorder::ReapsWithWork() {
    return holdCounters.reapsWithWork;
}

std::uint32_t Recorder::streamOfCaller() const {
    const std::lock_guard recordingGuard(recording);
    if (submittingStream != NoStream) return submittingStream;
    if (streams.size() == 1) return 0;
    const auto tag = GuestMemory::GpuLockThreadTag();
    if (tag == 0 || tag >= queueStreams.size()) return 0;
    auto& slot = queueStreams[tag];
    if (slot == 0) {
        slot = static_cast<std::uint8_t>(nextStream + 1);
        nextStream = nextStream + 1 < streams.size() ? nextStream + 1 : 1;
    }
    return slot - 1u;
}

std::unique_ptr<Recorder::Batch>& Recorder::callerOpen() {
    const std::lock_guard recordingGuard(recording);
    return opens[streamOfCaller()];
}

const Recorder::Batch* Recorder::callerBatch() const {
    const std::lock_guard recordingGuard(recording);
    return opens[streamOfCaller()].get();
}

bool Recorder::CallerBatchValue(std::uint32_t& stream, std::uint64_t& value) const {
    const std::lock_guard recordingGuard(recording);
    stream = streamOfCaller();
    const auto* open = opens[stream].get();
    if (open == nullptr) return false;
    value = open->value;
    return true;
}

bool Recorder::streamOwnedByCaller(std::uint32_t stream) const {
    const auto tag = GuestMemory::GpuLockThreadTag();
    if (stream == 0 || tag == 0 || tag >= queueStreams.size() || queueStreams[tag] != stream + 1u) return false;
    for (std::size_t other = 1; other < queueStreams.size(); ++other) {
        if (other != tag && queueStreams[other] == stream + 1u) return false;
    }
    return true;
}

VkCommandPool Recorder::streamPool(std::uint32_t stream) const {
    return streams[stream].pool != VK_NULL_HANDLE ? streams[stream].pool : context.pool;
}

bool Recorder::OwnsStream() const {
    const std::lock_guard recordingGuard(recording);
    return streams.size() > 1 && submittingStream == NoStream && streamOwnedByCaller(streamOfCaller());
}

Recorder::OwnStreamOutcome Recorder::RecordLabelsOnOwnStream(std::span<const OwnStreamLabel> labels, const std::function<bool()>& unchanged, bool submit) {
    RecordingHold recordingHold(recording);
    if (labels.empty() || streams.size() == 1 || submittingStream != NoStream || !LabelRunsPerBatch() || FlipReadCheck()) return OwnStreamOutcome::Refused;
    const auto stream = streamOfCaller();
    if (!streamOwnedByCaller(stream) || streams[stream].pool == VK_NULL_HANDLE || Idle()) return OwnStreamOutcome::Refused;
    for (std::uint32_t other = 0; other < streams.size(); ++other) {
        const auto after = streams[other].barrier;
        if (other != stream && after > streams[other].finished && opens[other] != nullptr && opens[other]->value <= after) return OwnStreamOutcome::Refused;
    }
    if (const auto& open = opens[stream]; open != nullptr && (open->renderPass.open || open->signalPublicationOwed)) return OwnStreamOutcome::Refused;
    if (!unchanged()) return OwnStreamOutcome::Refused;
    struct Scope {
        Scope() { ownStreamRecording = true; }
        ~Scope() { ownStreamRecording = false; }
    } scope;
    for (const auto& label : labels) {
        RecordStore(label.buffer, label.offset, label.bytes, label.address);
        NoteLabel(label.address, label.bytes, label.stamp, label.queue);
    }
    const auto submittable = [&] {
        const auto& open = opens[stream];
        return open != nullptr && open->labelsOnly && open->keyStores.empty() && !open->sampleActive && !open->segmentActive;
    };
    if (!submit || !submittable()) return OwnStreamOutcome::Recorded;
    std::unique_lock device(GuestMemory::GpuMutex(), std::defer_lock);
    const auto myId = id;
    while (opens[stream] != nullptr && awaitsUnsubmitted(*opens[stream])) {
        std::unique_lock waitLock(orderWaitMutex);
        const auto generation = orderWaitGeneration;
        orderWaiters.fetch_add(1, std::memory_order_acq_rel);
        waitLock.unlock();
        if (device.try_lock()) {
            orderWaiters.fetch_sub(1, std::memory_order_acq_rel);
            break;
        }
        const auto held = recording.ReleaseAll();
        waitLock.lock();
        OrderWaitChanged().wait(waitLock, [&] { return orderWaitGeneration != generation; });
        orderWaiters.fetch_sub(1, std::memory_order_acq_rel);
        waitLock.unlock();
        if (!RecorderAlive(myId)) {
            recordingHold.Abandon();
            return OwnStreamOutcome::Recorded;
        }
        recording.Reacquire(held);
    }
    if (opens[stream] == nullptr) return OwnStreamOutcome::Submitted;
    if (!submittable()) return OwnStreamOutcome::Recorded;
    if (device.owns_lock()) ownStreamRecording = false;
    submitStream(stream);
    return OwnStreamOutcome::Submitted;
}

bool Recorder::anyOpen() const {
    const std::lock_guard recordingGuard(recording);
    return std::any_of(opens.begin(), opens.begin() + static_cast<std::ptrdiff_t>(streams.size()), [](const auto& batch) { return batch != nullptr; });
}

void Recorder::DeclareSection(std::span<const SectionObject> objects) {
    const std::lock_guard recordingGuard(recording);
    auto& declaration = declarations[streamOfCaller()];
    declaration.objects.assign(objects.begin(), objects.end());
    declaration.declared = true;
}

void Recorder::dependOn(const Epoch& after) {
    const std::lock_guard recordingGuard(recording);
    const auto stream = streamOfCaller();
    Epoch waits = opens[stream] != nullptr ? opens[stream]->waits : Epoch{};
    for (std::uint32_t other = 0; other < streams.size(); ++other) {
        if (other == stream || after[other] <= streams[other].finished) continue;
        const bool submit = opens[other] != nullptr && opens[other]->value <= after[other];
        if (submit) {
            ++orderSubmits;
            submitStream(other);
        }
        waits[other] = std::max(waits[other], after[other]);
    }
    ensureOpen();
    auto& open = *opens[stream];
    for (std::uint32_t other = 0; other < streams.size(); ++other) open.waits[other] = std::max(open.waits[other], waits[other]);
}

void Recorder::beginSection() {
    const std::lock_guard recordingGuard(recording);
    auto& declaration = declarations[streamOfCaller()];
    const bool declared = std::exchange(declaration.declared, false);
    std::vector<SectionObject> objects;
    objects.swap(declaration.objects);
    ensureOpen();
    if (streams.size() == 1 || submittingStream != NoStream) return;
    const auto stream = streamOfCaller();
    Epoch after{};
    if (!declared) {
        for (std::uint32_t other = 0; other < streams.size(); ++other) after[other] = opens[other] != nullptr ? opens[other]->value : streams[other].submitted;
        dependOn(after);
        streams[stream].barrier = opens[stream]->value;
        return;
    }
    for (std::uint32_t other = 0; other < streams.size(); ++other) after[other] = streams[other].barrier;
    for (const auto& [object, written] : objects) {
        const auto& use = objectUses[object];
        for (std::uint32_t other = 0; other < streams.size(); ++other) after[other] = std::max(after[other], written ? std::max(use.read[other], use.written[other]) : use.written[other]);
    }
    dependOn(after);
    const auto value = opens[stream]->value;
    for (const auto& [object, written] : objects) {
        auto& use = objectUses[object];
        (written ? use.written : use.read)[stream] = value;
    }
    if (objectUses.size() > objectSweepAt) sweepObjectUses();
}

void Recorder::sweepObjectUses() {
    const std::lock_guard recordingGuard(recording);
    for (auto it = objectUses.begin(); it != objectUses.end();) {
        bool live = false;
        for (std::uint32_t stream = 0; stream < streams.size() && !live; ++stream) live = std::max(it->second.read[stream], it->second.written[stream]) > streams[stream].finished;
        it = live ? std::next(it) : objectUses.erase(it);
    }
    objectSweepAt = std::max<std::size_t>(ObjectSweepMinimum, objectUses.size() * 2);
}

void Recorder::OrderAfterLabel(std::uint64_t address, std::size_t bytes) {
    const std::lock_guard recordingGuard(recording);
    if (streams.size() == 1 || bytes == 0) return;
    const auto stream = streamOfCaller();
    Epoch after{};
    {
        std::lock_guard tableLock(labelTableMutex);
        labelTableLookups.fetch_add(1, std::memory_order_relaxed);
        for (auto found = labels.lower_bound(address & ~std::uint64_t{3}); found != labels.end() && found->first < address + bytes; ++found) {
            if ((found->first & 3u) != 0u) continue;
            for (const auto& writer : found->second) {
                if (writer.stream != stream) after[writer.stream] = std::max(after[writer.stream], writer.batch);
            }
        }
    }
    bool any = false;
    for (std::uint32_t other = 0; other < streams.size(); ++other) any = any || after[other] > streams[other].finished;
    if (!any) return;
    dependOn(after);
}

std::uint64_t Recorder::LabelTableLookups() {
    return labelTableLookups.load(std::memory_order_relaxed);
}

Recorder::Token Recorder::CallerToken() {
    const std::lock_guard recordingGuard(recording);
    ensureOpen();
    Token token{};
    const auto stream = streamOfCaller();
    token[stream] = opens[stream]->value;
    return token;
}

Recorder::Token Recorder::AllToken() const {
    const std::lock_guard recordingGuard(recording);
    Token token{};
    for (std::uint32_t stream = 0; stream < streams.size(); ++stream) token[stream] = opens[stream] != nullptr ? opens[stream]->value : streams[stream].submitted;
    return token;
}

bool Recorder::Reached(const Token& token) const {
    const std::lock_guard recordingGuard(recording);
    for (std::uint32_t stream = 0; stream < streams.size(); ++stream) {
        if (token[stream] > streams[stream].finished) return false;
    }
    return true;
}

void Recorder::OrderAfterBatch(std::uint32_t stream, std::uint64_t value) {
    const std::lock_guard recordingGuard(recording);
    if (streams.size() == 1 || stream >= streams.size() || stream == streamOfCaller() || value <= streams[stream].finished) return;
    Epoch after{};
    after[stream] = value;
    dependOn(after);
}

bool Recorder::TokenSignaled(const Token& token) const {
    const auto counter = context.Function<PFN_vkGetSemaphoreCounterValueKHR>("vkGetSemaphoreCounterValueKHR");
    for (std::uint32_t stream = 0; stream < streams.size(); ++stream) {
        if (token[stream] == 0) continue;
        std::uint64_t value = 0;
        if (streams[stream].timeline == VK_NULL_HANDLE || counter(context.device, streams[stream].timeline, &value) != VK_SUCCESS || value < token[stream]) return false;
    }
    return true;
}

void Recorder::OrderAfterQueue(std::uint32_t queue) {
    const std::lock_guard recordingGuard(recording);
    if (streams.size() == 1) return;
    const std::uint32_t other = queue == 0 || queue >= queueStreams.size() || queueStreams[queue] == 0 ? 0 : queueStreams[queue] - 1u;
    if (other == streamOfCaller()) return;
    const auto value = opens[other] != nullptr ? opens[other]->value : streams[other].submitted;
    if (value <= streams[other].finished) return;
    Epoch after{};
    after[other] = value;
    dependOn(after);
}

bool Recorder::OrderAfterQueueOnOwnStream(std::uint32_t queue) {
    const std::lock_guard recordingGuard(recording);
    if (streams.size() == 1) return true;
    if (submittingStream != NoStream) return false;
    const auto stream = streamOfCaller();
    if (!streamOwnedByCaller(stream) || streams[stream].pool == VK_NULL_HANDLE) return false;
    const std::uint32_t other = queue == 0 || queue >= queueStreams.size() || queueStreams[queue] == 0 ? 0 : queueStreams[queue] - 1u;
    if (other == stream) return true;
    const auto value = opens[other] != nullptr ? opens[other]->value : streams[other].submitted;
    if (value <= streams[other].finished) return true;
    struct Scope {
        Scope() { ownStreamRecording = true; }
        ~Scope() { ownStreamRecording = false; }
    } scope;
    ensureOpen();
    auto& waits = opens[stream]->waits;
    waits[other] = std::max(waits[other], value);
    return true;
}

bool Recorder::awaitsUnsubmitted(const Batch& batch) const {
    const std::lock_guard recordingGuard(recording);
    for (std::uint32_t other = 0; other < streams.size(); ++other) {
        if (other != batch.stream && batch.waits[other] > streams[other].submitted) return true;
    }
    return false;
}

void Recorder::OrderAfterValue(std::uint64_t address, std::size_t bytes) {
    const std::lock_guard recordingGuard(recording);
    if (streams.size() == 1 || bytes == 0) return;
    const auto stream = streamOfCaller();
    const auto end = address + bytes;
    Epoch after{};
    std::array<bool, MaxRecorderStreams> labelled{};
    {
        std::lock_guard tableLock(labelTableMutex);
        labelTableLookups.fetch_add(1, std::memory_order_relaxed);
        for (auto found = labels.lower_bound(address & ~std::uint64_t{3}); found != labels.end() && found->first < end; ++found) {
            if ((found->first & 3u) != 0u) continue;
            const auto dword = found->first;
            for (const auto& writer : found->second) {
                if (writer.stream == stream) continue;
                labelled[writer.stream] = true;
                if (std::memcmp(&writer.value, reinterpret_cast<const void*>(dword), 4) == 0) after[writer.stream] = std::max(after[writer.stream], writer.batch);
            }
        }
    }
    const auto unlabelled = [&](const Batch& batch) { return batch.stream != stream && !labelled[batch.stream] && overlaps(batch, address, end); };
    for (std::uint32_t other = 0; other < streams.size(); ++other) {
        if (opens[other] != nullptr && unlabelled(*opens[other])) after[other] = std::max(after[other], opens[other]->value);
    }
    for (const auto& batch : inFlight) {
        if (unlabelled(*batch)) after[batch->stream] = std::max(after[batch->stream], batch->value);
    }
    bool any = false;
    for (std::uint32_t other = 0; other < streams.size(); ++other) any = any || after[other] > streams[other].finished;
    if (!any) return;
    dependOn(after);
}

void Recorder::keepOnOtherStreams(const std::shared_ptr<void>& object) {
    const std::lock_guard recordingGuard(recording);
    const auto stream = streamOfCaller();
    for (std::uint32_t other = 0; other < streams.size(); ++other) {
        if (other == stream) continue;
        Batch* newest = opens[other] != nullptr ? opens[other].get() : streams[other].submitted > streams[other].finished ? streams[other].newest : nullptr;
        if (newest != nullptr) newest->kept.push_back(object);
    }
}

void Recorder::closeEpoch(Epoch& epoch) const {
    const std::lock_guard recordingGuard(recording);
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
        const auto& batch = **it;
        if (batch.value > epoch[batch.stream]) continue;
        for (std::uint32_t other = 0; other < streams.size(); ++other) epoch[other] = std::max(epoch[other], batch.waits[other]);
    }
}

Recorder::Epoch Recorder::syncTargets(std::uint64_t address, std::uint64_t end) {
    const std::lock_guard recordingGuard(recording);
    Epoch epoch{};
    for (std::uint32_t stream = 0; stream < streams.size(); ++stream) {
        if (opens[stream] == nullptr || !overlaps(*opens[stream], address, end)) continue;
        epoch[stream] = opens[stream]->value;
        submitStream(stream);
    }
    for (const auto& batch : inFlight) {
        if (overlaps(*batch, address, end)) epoch[batch->stream] = std::max(epoch[batch->stream], batch->value);
    }
    return epoch;
}

bool Recorder::epochUnsignaled(const Epoch& epoch) const {
    const std::lock_guard recordingGuard(recording);
    return std::any_of(inFlight.begin(), inFlight.end(), [&](const auto& batch) { return batch->value <= epoch[batch->stream] && fenceStatus(batch->fence) != VK_SUCCESS; });
}

void Recorder::finishEpoch(Epoch epoch, int source) {
    const std::lock_guard recordingGuard(recording);
    closeEpoch(epoch);
    for (;;) {
        const auto found = std::find_if(inFlight.begin(), inFlight.end(), [&](const auto& batch) { return batch->value <= epoch[batch->stream]; });
        if (found == inFlight.end()) return;
        auto batch = std::move(*found);
        inFlight.erase(found);
        finish(std::move(batch), true, source);
    }
}

VkCommandBuffer Recorder::Commands(VkAccessFlags* coveredAccess) {
    const std::lock_guard recordingGuard(recording);
    beginSection();
    return batchCommands(coveredAccess);
}

VkCommandBuffer Recorder::batchCommands(VkAccessFlags* coveredAccess) {
    const std::lock_guard recordingGuard(recording);
    ensureOpen();
    auto& open = callerOpen();
    open->labelsOnly = false;
    // Work recorded after a draw's render pass or an inline store run must see their writes: the
    // pass's end and the run's trailing barrier go in first (a per-batch run waits for Submit, or
    // for a caller whose ranges overlap a queued store: FlushStores).
    if (open->renderPass.open) endOpenRenderPass();
    if (open->run.open && !LabelRunsPerBatch()) closeStoreRun();
    if (coveredAccess != nullptr) *coveredAccess = open->coveredAccess;
    open->coveredAccess = 0;
    return open->commands;
}

void Recorder::MarkCovered(VkAccessFlags access) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (open != nullptr) open->coveredAccess = access;
}

Recorder::DrawPassStart Recorder::StartDrawPass(std::uint64_t key, bool forced, bool fenced, const PassAccess& access) {
    const std::lock_guard recordingGuard(recording);
    beginSection();
    auto& open = callerOpen();
    auto& pass = open->renderPass;
    if (!pass.open || forced || (countingSamples && segmentMode() && !segmentSlotReady(*open))) {
        const auto commands = batchCommands();
        open->passHazards.Clear();
        open->passHazards.BeginPass();
        return {commands, false, true};
    }
    const auto hazard = open->passHazards.Check(access, pass.key == key);
    if (hazard == PassHazard::None && pass.key == key) return {open->commands, true, false};
    const bool merge = MergeBarriers() && !fenced && !pass.fenced && !pass.afterPass;
    const bool elide = merge && hazard == PassHazard::None;
    endOpenRenderPass(!merge);
    const auto commands = batchCommands();
    if (!elide) open->passHazards.Clear();
    open->passHazards.BeginPass();
    return {commands, false, !elide};
}

void Recorder::LeaveRenderPassOpen(std::uint64_t key, std::uint32_t timing, bool fenced, const PassAccess& access, std::function<void(VkCommandBuffer)> afterPass) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    Require(open != nullptr, "no batch is open for the render pass");
    auto& pass = open->renderPass;
    if (!pass.open) {
        pass.timing = timing;
        pass.afterPass = std::move(afterPass);
    }
    pass.open = true;
    pass.key = key;
    pass.fenced = pass.fenced || fenced;
    open->passHazards.Add(access);
}

void Recorder::endOpenRenderPass(bool barrier) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    auto& pass = open->renderPass;
    if (open->segmentActive) endSegment(*open);
    EndRenderPass(context, open->commands);
    // The pass's attachment and shader writes are visible to everything recorded after it (the
    // host sees them at the batch's fence).
    if (barrier) {
        recordBarrier(open->commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        CountBarriers(CommandClass::Draw);
    }
    if (pass.afterPass) pass.afterPass(open->commands);
    EndGpuTiming(pass.timing);
    open->coveredAccess = barrier ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT : 0;
    open->hostReadOwed = true;
    pass = {};
}

void Recorder::QueueKeyStore(VkBuffer buffer, VkDeviceSize first, VkDeviceSize last, std::shared_ptr<void> seed, std::uint64_t begin, std::uint64_t end) {
    const std::lock_guard recordingGuard(recording);
    DeclareSection({});
    beginSection();
    auto& open = callerOpen();
    // A queued label store over the keys lands first (program order: the key store is later).
    FlushStoresOverlapping(begin, static_cast<std::size_t>(end - begin));
    keyStoreCount.fetch_add(1, std::memory_order_relaxed);
    if (seed != nullptr) open->kept.push_back(seed);
    // The same range queued twice lands once: nothing wrote over it in between, or the writer
    // would have recorded the first store first.
    for (const auto& store : open->keyStores) {
        if (store.buffer == buffer && store.first == first && store.last == last) {
            keyStoresJoined.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    open->keyStores.push_back({buffer, first, last, std::move(seed), begin, end});
    if (KeyStoresEach()) recordKeyStores(false);
}

bool Recorder::QueuedKeyStoreOverlaps(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    const auto* open = callerBatch();
    if (open == nullptr || bytes == 0) return false;
    const auto end = address + bytes;
    return std::any_of(open->keyStores.begin(), open->keyStores.end(), [&](const Batch::KeyStore& store) { return address < store.end && store.begin < end; });
}

bool Recorder::AnyQueuedKeyStore(const std::function<bool(std::uint64_t, std::uint64_t)>& overlaps) const {
    const std::lock_guard recordingGuard(recording);
    const auto* open = callerBatch();
    if (open == nullptr) return false;
    return std::any_of(open->keyStores.begin(), open->keyStores.end(), [&](const Batch::KeyStore& store) { return overlaps(store.begin, store.end); });
}

void Recorder::FlushKeyStores() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (open == nullptr || open->keyStores.empty()) return;
    recordKeyStores(true);
}

void Recorder::recordKeyStores(bool forWriter) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    auto stores = std::move(open->keyStores);
    open->keyStores.clear();
    if (stores.empty()) return;
    const auto commands = batchCommands();
    const auto timing = beginTiming(ClassKey(CommandClass::DccKeyStore));
    // Ordered behind every earlier recorded read or write of the ranges (the title's DCC clear or
    // decompress kernel storing keys through a V#), and visible to the work after and to the host.
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    const auto fill = context.Resolved(&DeviceFunctions::cmdFillBuffer, "vkCmdFillBuffer");
    const auto copy = context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer");
    std::uint64_t bytes = 0;
    for (const auto& store : stores) {
        const VkDeviceSize fillBegin = std::min((store.first + 3) & ~VkDeviceSize{3}, store.last);
        const VkDeviceSize fillEnd = std::max(store.last & ~VkDeviceSize{3}, fillBegin);
        if (fillEnd > fillBegin) fill(commands, store.buffer, fillBegin, fillEnd - fillBegin, 0xffffffffu);
        if (store.seed != nullptr) {
            VkBufferCopy copies[2];
            std::uint32_t copyCount = 0;
            if (fillBegin > store.first) copies[copyCount++] = {0, store.first, fillBegin - store.first};
            if (store.last > fillEnd) copies[copyCount++] = {0, fillEnd, store.last - fillEnd};
            if (copyCount != 0) copy(commands, static_cast<Buffer*>(store.seed.get())->Handle(), store.buffer, copyCount, copies);
        }
        bytes += store.last - store.first;
    }
    recordBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
    CountBarriers(CommandClass::DccKeyStore, 2);
    EndGpuTiming(timing, bytes);
    open->coveredAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    keyStoreRuns.fetch_add(1, std::memory_order_relaxed);
    if (forWriter) keyStoreRunsForWriter.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::ensureOpen() {
    const std::lock_guard recordingGuard(recording);
    if (!ownStreamRecording) GuestMemory::AssertGpuLockHeld("Recorder::Commands");
    const auto stream = streamOfCaller();
    auto& open = opens[stream];
    if (open == nullptr) {
        auto batch = std::make_unique<Batch>();
        batch->stream = stream;
        batch->labelsOnly = ownStreamRecording;
        try {
            if (auto& spare = streams[stream].spare; !spare.empty()) {
                // The pool resets command buffers on begin; the fence was reset when the batch completed.
                std::tie(batch->commands, batch->fence) = spare.back();
                spare.pop_back();
            } else {
                VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                allocation.commandPool = streamPool(stream);
                allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocation.commandBufferCount = 1;
                Check(context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &allocation, &batch->commands), "vkAllocateCommandBuffers recorder");
                VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                Check(context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &info, nullptr, &batch->fence), "vkCreateFence recorder");
            }
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Check(function(beginCommandBuffer, "vkBeginCommandBuffer")(batch->commands, &begin), "vkBeginCommandBuffer recorder");
        } catch (...) {
            release(*batch);
            throw;
        }
        batch->queue = GuestMemory::GpuLockThreadTag();
        batch->value = ++streams[stream].opened;
        if (countingSamples && !segmentMode() && stream == 0) beginSamples(*batch);
        dropStaleHoldKept();
        batch->kept.insert(batch->kept.end(), holdKept.begin(), holdKept.end());
        open = std::move(batch);
        if (PerformanceTrace::Recording()) {
            static std::atomic<std::uint64_t> traceIds{0};
            open->traceId = traceIds.fetch_add(1, std::memory_order_relaxed) + 1;
        }
        TraceBatchTag = open->traceId;
        // The whole-batch range: its start stamp waits for the previous batches like any
        // bottom-of-pipe stamp, so it marks when this batch's execution began.
        if (BatchStampsEnabled()) open->batchTiming = beginTiming(BatchTimingKey);
    }
}

bool Recorder::DeferSignalPublication() {
    const std::lock_guard recordingGuard(recording);
    GuestMemory::AssertGpuLockHeld("Recorder::DeferSignalPublication");
    if (!LabelRunsPerBatch()) return false;
    auto& open = callerOpen();
    if (open == nullptr || !open->renderPass.open) return false;
    open->signalPublicationOwed = true;
    return true;
}

void Recorder::RecordStore(VkBuffer buffer, VkDeviceSize offset, std::span<const std::byte> bytes, std::uint64_t address) {
    const std::lock_guard recordingGuard(recording);
    if (bytes.empty()) return;
    DeclareSection({});
    beginSection();
    auto& open = callerOpen();
    auto& run = open->run;
    const VkDeviceSize end = offset + bytes.size();
    storeCount.fetch_add(1, std::memory_order_relaxed);
    if (LabelRunsPerBatch()) {
        // Queued for the batch's run: a store inside the last queued one replaces its bytes, a
        // contiguous one joins it (a joined range overlapping an earlier queued store takes the
        // WAW barrier when the run is recorded, so no join is refused).
        if (run.queued.empty()) storeRuns.fetch_add(1, std::memory_order_relaxed);
        else if (auto& last = run.queued.back(); last.buffer == buffer) {
            const VkDeviceSize lastEnd = last.offset + last.bytes.size();
            if (offset >= last.offset && end <= lastEnd) {
                std::memcpy(last.bytes.data() + (offset - last.offset), bytes.data(), bytes.size());
                storesReplaced.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (offset == lastEnd && last.bytes.size() + bytes.size() <= 65536) {
                last.bytes.insert(last.bytes.end(), bytes.begin(), bytes.end());
                storesJoined.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        run.queued.push_back({buffer, offset, std::vector<std::byte>(bytes.begin(), bytes.end()), address});
        return;
    }
    // A label says the work before it is done: the queued key stores (results of that work) land
    // ahead of every store of the run.
    if (!open->keyStores.empty()) recordKeyStores(true);
    if (open->renderPass.open) endOpenRenderPass();
    open->coveredAccess = 0;
    // Two transfers to the same bytes have no order of their own within the run.
    const auto overlapsRecorded = [&] { return std::any_of(run.recorded.begin(), run.recorded.end(), [&](const auto& store) { return std::get<0>(store) == buffer && offset < std::get<2>(store) && std::get<1>(store) < end; }); };
    if (!run.open) {
        run.timing = beginTiming(ClassKey(CommandClass::LabelRun));
        run.storedBytes = 0;
        // The run's leading barrier: every earlier write of the range (a shader's, a transfer's, a
        // color attachment's) lands before the stores of the run.
        recordBarrier(open->commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        CountBarriers(CommandClass::LabelRun);
        run.open = true;
        storeRuns.fetch_add(1, std::memory_order_relaxed);
    } else if (run.buffer == buffer && !run.bytes.empty()) {
        const VkDeviceSize pendingEnd = run.offset + run.bytes.size();
        if (offset >= run.offset && end <= pendingEnd) {
            std::memcpy(run.bytes.data() + (offset - run.offset), bytes.data(), bytes.size());
            storesReplaced.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (offset == pendingEnd && run.bytes.size() + bytes.size() <= 65536) {
            // A joined store shares the pending vkCmdUpdateBuffer, which no barrier precedes: one
            // that overlaps a store the run already recorded takes the WAW barrier below instead.
            if (!JoinWawCheck() || !overlapsRecorded()) {
                run.bytes.insert(run.bytes.end(), bytes.begin(), bytes.end());
                storesJoined.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            storeJoinsRefused.fetch_add(1, std::memory_order_relaxed);
        }
    }
    flushPendingStore();
    const bool waw = overlapsRecorded();
    if (waw) {
        recordBarrier(open->commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        CountBarriers(CommandClass::LabelRun);
        storeWawBarriers.fetch_add(1, std::memory_order_relaxed);
    }
    run.buffer = buffer;
    run.offset = offset;
    run.bytes.assign(bytes.begin(), bytes.end());
    if (!LabelRunsEnabled()) closeStoreRun();
}

void Recorder::recordBarrier(VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) const {
    const std::lock_guard recordingGuard(recording);
    if (cmdPipelineBarrier == nullptr) {
        RecordMemoryBarrier(context, commands, sourceStage, destinationStage, sourceAccess, destinationAccess);
        return;
    }
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, sourceAccess, destinationAccess};
    cmdPipelineBarrier(commands, sourceStage, destinationStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Recorder::flushPendingStore() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    auto& run = open->run;
    if (run.bytes.empty()) return;
    function(cmdUpdateBuffer, "vkCmdUpdateBuffer")(open->commands, run.buffer, run.offset, run.bytes.size(), run.bytes.data());
    run.recorded.emplace_back(run.buffer, run.offset, run.offset + run.bytes.size());
    run.storedBytes += run.bytes.size();
    run.bytes.clear();
}

bool Recorder::closeStoreRun(bool atSubmit) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    auto& run = open->run;
    if (LabelRunsPerBatch()) {
        auto stores = std::move(run.queued);
        run.queued.clear();
        open->queuedDumps = 0;
        if (stores.empty()) return false;
        // A label says the work before it is done: the queued key stores (results of that work)
        // land ahead of every store of the run, and the pass a draw left open ends.
        if (!open->keyStores.empty()) recordKeyStores(true);
        if (open->renderPass.open) endOpenRenderPass();
        const auto commands = open->commands;
        run.timing = beginTiming(ClassKey(CommandClass::LabelRun));
        if (BarrierValidate()) {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
            for (const auto& store : stores) ranges.emplace_back(store.address, store.End());
            NoteAccess(CommandClass::LabelRun, Access{{}, ranges, {}, VK_PIPELINE_STAGE_TRANSFER_BIT});
        }
        // At Submit every result of the batch reaches the host before the stores (the title's
        // pollers read results after the label); in place, later work sees the stores as well.
        if (atSubmit) recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
        else recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        std::uint32_t barriers = 2;
        std::uint64_t storedBytes = 0;
        const auto update = function(cmdUpdateBuffer, "vkCmdUpdateBuffer");
        std::vector<QueuedWrite> writes;
        writes.reserve(stores.size());
        for (const auto& store : stores) writes.push_back({store.address, store.End(), store.dumpTarget != 0});
        const bool dumps = std::any_of(writes.begin(), writes.end(), [](const QueuedWrite& write) { return write.computed; });
        if (dumps) recordQueuedDumps(commands, stores);
        std::size_t first = 0;
        std::uint32_t dumpIndex = 0, consumed = 0;
        for (const auto end : QueuedWriteGroups(writes)) {
            if (first != 0) {
                recordBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                ++barriers;
                run.recorded.clear();
            }
            for (std::size_t i = first; i < end; ++i) {
                const auto& store = stores[i];
                if (store.dumpTarget != 0) continue;
                const VkDeviceSize storeEnd = store.offset + store.bytes.size();
                if (std::any_of(run.recorded.begin(), run.recorded.end(), [&](const auto& earlier) { return std::get<0>(earlier) == store.buffer && store.offset < std::get<2>(earlier) && std::get<1>(earlier) < storeEnd; })) {
                    recordBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                    ++barriers;
                    storeWawBarriers.fetch_add(1, std::memory_order_relaxed);
                }
                update(commands, store.buffer, store.offset, store.bytes.size(), store.bytes.data());
                run.recorded.emplace_back(store.buffer, store.offset, storeEnd);
                storedBytes += store.bytes.size();
            }
            const auto before = dumpIndex;
            foldQueuedDumps(commands, stores, first, end, dumpIndex, consumed);
            if (dumpIndex != before) barriers += 3;
            first = end;
        }
        if (dumps) {
            retireSegments(consumed);
            dumpFolds.fetch_add(1, std::memory_order_relaxed);
        }
        const VkPipelineStageFlags written = VK_PIPELINE_STAGE_TRANSFER_BIT | (dumps ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0u);
        const VkAccessFlags writtenAccess = VK_ACCESS_TRANSFER_WRITE_BIT | (dumps ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
        if (atSubmit) {
            recordBarrier(commands, written, VK_PIPELINE_STAGE_HOST_BIT, writtenAccess, VK_ACCESS_HOST_READ_BIT);
            open->coveredAccess = 0;
            storeRunsAtSubmit.fetch_add(1, std::memory_order_relaxed);
        } else {
            recordBarrier(commands, written, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, writtenAccess, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            open->coveredAccess = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            storeRunsForced.fetch_add(1, std::memory_order_relaxed);
        }
        CountBarriers(CommandClass::LabelRun, barriers);
        EndGpuTiming(run.timing, storedBytes);
        run.timing = NoTiming;
        run.recorded.clear();
        return atSubmit;
    }
    flushPendingStore();
    // The run's stores are visible to the host and to everything recorded after them.
    recordBarrier(open->commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    CountBarriers(CommandClass::LabelRun);
    EndGpuTiming(run.timing, run.storedBytes);
    run.timing = NoTiming;
    open->coveredAccess = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    run.open = false;
    run.buffer = VK_NULL_HANDLE;
    run.recorded.clear();
    return false;
}

bool Recorder::QueuedStoreOverlaps(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    const auto* open = callerBatch();
    if (open == nullptr || bytes == 0) return false;
    const auto end = address + bytes;
    return std::any_of(open->run.queued.begin(), open->run.queued.end(), [&](const Batch::StoreRun::Queued& store) { return address < store.End() && store.address < end; });
}

bool Recorder::AnyQueuedStore(const std::function<bool(std::uint64_t, std::uint64_t)>& overlaps) const {
    const std::lock_guard recordingGuard(recording);
    const auto* open = callerBatch();
    if (open == nullptr) return false;
    return std::any_of(open->run.queued.begin(), open->run.queued.end(), [&](const Batch::StoreRun::Queued& store) { return overlaps(store.address, store.End()); });
}

void Recorder::FlushStores() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (open == nullptr || open->run.queued.empty()) return;
    closeStoreRun(false);
}

namespace {

constexpr std::uint32_t MaxTimedRanges = 512;
constexpr std::size_t CommandClasses = static_cast<std::size_t>(Recorder::CommandClass::Count);
constexpr const char* CommandClassNames[CommandClasses] = {"dispatch-lead", "dispatch-trail", "indirect-args", "label-run", "fill", "fill-clear", "copy", "staging-in", "staging-out", "draw", "storage-upload", "storage-writeback", "dcc-clear", "dcc-keys", "present-blit", "shadow-publish", "template-refresh"};
// [barriers] counters by class (relaxed: only reported) and the [gputime] totals, which the
// presenter's thread adds to without the GPU mutex (AddGpuTiming).
std::atomic<std::uint64_t> classBarriers[CommandClasses]{};
std::atomic<std::uint64_t> classMerged[CommandClasses]{};
// The hazard tracker's counts (APS5_BARRIER_VALIDATE=1): simulated barriers by class and by
// hazard kind, leading barriers it would have skipped by class, batch-end barriers, and the
// APS5_TRACE_BARRIERS budget left.
constexpr std::size_t HazardKinds = 5;
constexpr const char* HazardNames[HazardKinds] = {"raw", "waw", "war", "image", "conservative"};
std::atomic<std::uint64_t> validateEmitted[CommandClasses]{}, validateSkipped[CommandClasses]{}, validateKinds[HazardKinds]{};
std::atomic<std::uint64_t> validateBatchEnds{0};
std::atomic<std::int64_t> traceBarriersLeft{[] {
    const char* text = std::getenv("APS5_TRACE_BARRIERS");
    return text != nullptr ? static_cast<std::int64_t>(std::strtoll(text, nullptr, 10)) : std::int64_t{0};
}()};
std::atomic<std::uint64_t> timingPresents{0};
std::atomic<std::uint64_t> presentSerial{0};
std::atomic<std::uint64_t> timingDropped{0};
struct TimingTotals {
    std::uint64_t count = 0;
    double ms = 0;
    std::uint64_t bytes = 0;
    std::array<std::uint32_t, 3> groups{};
    bool mixedGroups = false;
};
std::mutex timingMutex;
std::map<std::uint64_t, TimingTotals> timingByKey;
double timingProgramMs = 0, timingClassMs = 0, timingUnionMs = 0, timingBatchMs = 0;
std::uint64_t timingBatches = 0;

bool DrawOrGpuProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr || Recorder::GpuTimingEnabled();
    return profiled;
}

void reportBarriers() {
    if (!DrawOrGpuProfiled()) return;
    static auto lastReport = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::uint64_t total = 0, merged = 0;
    std::string line, mergedLine;
    for (std::size_t i = 0; i < CommandClasses; ++i) {
        const auto count = classBarriers[i].exchange(0, std::memory_order_relaxed);
        total += count;
        char text[64];
        std::snprintf(text, sizeof(text), " %s %llu", CommandClassNames[i], static_cast<unsigned long long>(count));
        line += text;
        const auto mergedCount = classMerged[i].exchange(0, std::memory_order_relaxed);
        if (mergedCount == 0) continue;
        merged += mergedCount;
        std::snprintf(text, sizeof(text), " %s %llu", CommandClassNames[i], static_cast<unsigned long long>(mergedCount));
        mergedLine += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[barriers] %llu recorded (10 s) by class:%s; merged %llu:%s", static_cast<unsigned long long>(total), line.c_str(), static_cast<unsigned long long>(merged), mergedLine.c_str());
    if (Recorder::BarrierValidate()) {
        std::uint64_t emitted = 0, skipped = 0;
        std::string classes, kinds;
        for (std::size_t i = 0; i < CommandClasses; ++i) {
            const auto emittedCount = validateEmitted[i].exchange(0, std::memory_order_relaxed);
            const auto skippedCount = validateSkipped[i].exchange(0, std::memory_order_relaxed);
            emitted += emittedCount;
            skipped += skippedCount;
            if (emittedCount == 0 && skippedCount == 0) continue;
            char text[96];
            std::snprintf(text, sizeof(text), " %s %llu/%llu", CommandClassNames[i], static_cast<unsigned long long>(emittedCount), static_cast<unsigned long long>(skippedCount));
            classes += text;
        }
        for (std::size_t i = 0; i < HazardKinds; ++i) {
            char text[64];
            std::snprintf(text, sizeof(text), " %s %llu", HazardNames[i], static_cast<unsigned long long>(validateKinds[i].exchange(0, std::memory_order_relaxed)));
            kinds += text;
        }
        AgcDriver::ProfilePrint_nid_no_patch("; validate: would emit %llu (%s) + %llu batch ends, would skip %llu; emitted/skipped by class:%s", static_cast<unsigned long long>(emitted), kinds.c_str() + 1, static_cast<unsigned long long>(validateBatchEnds.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(skipped), classes.c_str());
    }
    AgcDriver::ProfilePrint_nid_no_patch("\n");
    AgcDriver::ProfilePrint_nid_no_patch("[samples] occlusion dumps (cumulative): %llu queued (%llu inside an open pass), %llu recorded at once; %llu segments closed; %llu folds of queued dumps\n", static_cast<unsigned long long>(dumpsQueued.load(std::memory_order_relaxed)), static_cast<unsigned long long>(dumpsQueuedInPass.load(std::memory_order_relaxed)), static_cast<unsigned long long>(dumpsAtOnce.load(std::memory_order_relaxed)), static_cast<unsigned long long>(segmentsClosed.load(std::memory_order_relaxed)), static_cast<unsigned long long>(dumpFolds.load(std::memory_order_relaxed)));
}

}

bool Recorder::GpuTimingEnabled() {
    static const bool enabled = std::getenv("APS5_PROFILE_GPU") != nullptr;
    return enabled;
}

bool Recorder::BatchStampsEnabled() {
    return GpuTimingEnabled() || DrawProfiled() || PerformanceTrace::Configured();
}

void Recorder::CountBarriers(CommandClass which, std::uint32_t count) {
    if (!DrawOrGpuProfiled()) return;
    classBarriers[static_cast<std::size_t>(which)].fetch_add(count, std::memory_order_relaxed);
}

bool Recorder::MergeBarriers() {
    static const bool merge = std::getenv("APS5_FULL_BARRIERS") == nullptr && std::getenv("APS5_NO_BARRIER_ELISION") == nullptr;
    return merge;
}

void Recorder::CountMerged(CommandClass which) {
    if (!DrawOrGpuProfiled()) return;
    classMerged[static_cast<std::size_t>(which)].fetch_add(1, std::memory_order_relaxed);
}

bool Recorder::BarrierValidate() {
    static const bool validate = std::getenv("APS5_BARRIER_VALIDATE") != nullptr;
    return validate;
}

void Recorder::NoteAccess(CommandClass which, const Access& access) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (!BarrierValidate()) return;
    ensureOpen();
    auto& tracker = open->tracker;
    const auto overlapsAny = [](const auto& ranges, const std::pair<std::uint64_t, std::uint64_t>& range) {
        return std::any_of(ranges.begin(), ranges.end(), [&](const auto& other) { return range.first < other.end && other.begin < range.second; });
    };
    std::size_t kind = HazardKinds;
    if (access.conservative) kind = 4;
    for (const auto& range : access.reads) {
        if (kind != HazardKinds) break;
        if (overlapsAny(tracker.writes, range)) kind = 0;
    }
    for (const auto& range : access.writes) {
        if (kind != HazardKinds) break;
        if (overlapsAny(tracker.writes, range)) kind = 1;
        else if (overlapsAny(tracker.reads, range)) kind = 2;
    }
    for (const auto& [image, written] : access.images) {
        if (kind != HazardKinds) break;
        if (std::any_of(tracker.images.begin(), tracker.images.end(), [&](const auto& seen) { return seen.image == image && (seen.written || written); })) kind = 3;
    }
    const auto index = static_cast<std::size_t>(which);
    if (kind != HazardKinds) {
        validateEmitted[index].fetch_add(1, std::memory_order_relaxed);
        validateKinds[kind].fetch_add(1, std::memory_order_relaxed);
        tracker.reads.clear();
        tracker.writes.clear();
        tracker.images.clear();
    } else {
        validateSkipped[index].fetch_add(1, std::memory_order_relaxed);
        if (traceBarriersLeft.load(std::memory_order_relaxed) > 0 && traceBarriersLeft.fetch_sub(1, std::memory_order_relaxed) > 0) {
            const auto first = [](const auto& ranges) { return ranges.empty() ? std::pair<std::uint64_t, std::uint64_t>{0, 0} : std::pair<std::uint64_t, std::uint64_t>{ranges.front().first, ranges.front().second}; };
            const auto reads = first(access.reads);
            const auto writes = first(access.writes);
            std::fprintf(stderr, "[barriers] would skip %s: reads %zu (first 0x%llx+0x%llx), writes %zu (first 0x%llx+0x%llx), images %zu; since the last barrier: %zu reads, %zu writes, %zu images\n", CommandClassNames[index], access.reads.size(), static_cast<unsigned long long>(reads.first), static_cast<unsigned long long>(reads.second - reads.first), access.writes.size(), static_cast<unsigned long long>(writes.first), static_cast<unsigned long long>(writes.second - writes.first), access.images.size(), tracker.reads.size(), tracker.writes.size(), tracker.images.size());
        }
    }
    for (const auto& [begin, end] : access.reads) tracker.reads.push_back({begin, end, access.stages});
    for (const auto& [begin, end] : access.writes) tracker.writes.push_back({begin, end, access.stages});
    for (const auto& [image, written] : access.images) tracker.images.push_back({image, written, access.stages});
}

void Recorder::CountPresent() {
    timingPresents.fetch_add(1, std::memory_order_relaxed);
    presentSerial.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t Recorder::Presents() {
    return presentSerial.load(std::memory_order_relaxed);
}

void Recorder::AddGpuTiming(CommandClass which, double nanoseconds, std::uint64_t bytes) {
    std::lock_guard lock(timingMutex);
    auto& totals = timingByKey[ClassKey(which)];
    ++totals.count;
    totals.ms += nanoseconds / 1e6;
    totals.bytes += bytes;
    timingClassMs += nanoseconds / 1e6;
}

std::uint32_t Recorder::BeginGpuTiming(std::uint64_t key) {
    const std::lock_guard recordingGuard(recording);
    if (!GpuTimingEnabled()) return NoTiming;
    batchCommands();
    return beginTiming(key);
}

std::uint32_t Recorder::beginTiming(std::uint64_t key) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    const bool full = GpuTimingEnabled();
    if (!full && !(key == BatchTimingKey && BatchStampsEnabled())) return NoTiming;
    const auto commands = open->commands;
    const std::uint32_t queryCount = full ? MaxTimedRanges * 2 : 2;
    if (open->queries == VK_NULL_HANDLE) {
        if (!full && !sparePools.empty()) {
            open->queries = sparePools.back();
            sparePools.pop_back();
        } else {
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = queryCount;
            if (context.Function<PFN_vkCreateQueryPool>("vkCreateQueryPool")(context.device, &info, nullptr, &open->queries) != VK_SUCCESS) {
                open->queries = VK_NULL_HANDLE;
                return NoTiming;
            }
        }
        function(cmdResetQueryPool, "vkCmdResetQueryPool")(commands, open->queries, 0, queryCount);
    }
    if (open->timedKeys.size() >= queryCount / 2) {
        timingDropped.fetch_add(1, std::memory_order_relaxed);
        return NoTiming;
    }
    const auto index = static_cast<std::uint32_t>(open->timedKeys.size());
    open->timedKeys.push_back(key);
    open->timedBytes.push_back(0);
    open->timedGroups.push_back({});
    // Both stamps wait for everything before them to complete, so the range is the timed work alone
    // (a top-of-pipe stamp is not held back by the barrier that precedes the work).
    context.Function<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp")(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, open->queries, index * 2);
    return index;
}

void Recorder::EndGpuTiming(std::uint32_t index, std::uint64_t bytes) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (index == NoTiming || open == nullptr || open->queries == VK_NULL_HANDLE) return;
    if (index < open->timedBytes.size()) open->timedBytes[index] += bytes;
    context.Function<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp")(open->commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, open->queries, index * 2 + 1);
}

void Recorder::NoteGpuTimingGroups(std::uint32_t index, std::array<std::uint32_t, 3> groups) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (index == NoTiming || open == nullptr || index >= open->timedGroups.size()) return;
    open->timedGroups[index] = groups;
}

void Recorder::CountSamples() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    GuestMemory::AssertGpuLockHeld("Recorder::CountSamples");
    gpuSampleCounter();
    countingSamples = true;
    if (segmentMode() || open == nullptr || open->samples != VK_NULL_HANDLE) return;
    if (open->renderPass.open) endOpenRenderPass();
    beginSamples(*open);
}

std::uint64_t Recorder::SamplesTotal() {
    const std::lock_guard recordingGuard(recording);
    if (sampleCounterState <= 0) return samplesPassed.load(std::memory_order_acquire);
    if (segmentMode() && !Idle()) Sync();
    std::uint64_t total = 0;
    std::memcpy(&total, sampleCounter->Bytes().data(), sizeof(total));
    for (const auto& segment : pendingSamples) {
        std::uint64_t samples = 0;
        Check(context.Function<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(context.device, segment.handle, segment.slot, 1, sizeof(samples), &samples, sizeof(samples), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults occlusion");
        total += samples;
    }
    pendingSamples.clear();
    std::memcpy(sampleCounter->Bytes().data(), &total, sizeof(total));
    return total;
}

void Recorder::NoteSampledDraw(VkCommandBuffer commands) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (open == nullptr) return;
    if (segmentMode()) {
        if (countingSamples && commands == open->commands && !open->segmentActive) beginSegment(*open);
        return;
    }
    if (open->sampleActive) open->samplesDrawn = true;
}

bool Recorder::gpuSampleCounter() {
    const std::lock_guard recordingGuard(recording);
    if (sampleCounterState != 0) return sampleCounterState > 0;
    sampleCounterState = -1;
    if (!context.bufferDeviceAddress) return false;
    VkShaderModule module = VK_NULL_HANDLE;
    try {
        sampleCounter = std::make_unique<Buffer>(context, DumpListOffset + 16 * DumpFoldCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        std::memset(sampleCounter->Bytes().data(), 0, sampleCounter->Bytes().size());
        const auto initial = samplesPassed.load(std::memory_order_acquire);
        std::memcpy(sampleCounter->Bytes().data(), &initial, sizeof(initial));
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, SampleCounterPushBytes};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &push;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &sampleLayout), "vkCreatePipelineLayout sample counter");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(SAMPLE_COUNTER_SPV);
        moduleInfo.pCode = SAMPLE_COUNTER_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule sample counter");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = sampleLayout;
        Check(CreateComputePipeline(context, pipelineInfo, &samplePipeline), "vkCreateComputePipelines sample counter");
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        module = VK_NULL_HANDLE;
        const VkPushConstantRange dumpPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, SampleDumpPushBytes};
        layoutInfo.pPushConstantRanges = &dumpPush;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &dumpLayout), "vkCreatePipelineLayout sample dumps");
        moduleInfo.codeSize = sizeof(SAMPLE_DUMPS_SPV);
        moduleInfo.pCode = SAMPLE_DUMPS_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule sample dumps");
        pipelineInfo.stage.module = module;
        pipelineInfo.layout = dumpLayout;
        Check(CreateComputePipeline(context, pipelineInfo, &dumpPipeline), "vkCreateComputePipelines sample dumps");
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        module = VK_NULL_HANDLE;
    } catch (const std::exception& error) {
        if (module != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        if (samplePipeline != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, samplePipeline, nullptr);
        if (dumpLayout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, dumpLayout, nullptr);
        if (sampleLayout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, sampleLayout, nullptr);
        samplePipeline = VK_NULL_HANDLE;
        dumpLayout = VK_NULL_HANDLE;
        sampleLayout = VK_NULL_HANDLE;
        sampleCounter.reset();
        std::fprintf(stderr, "[gpu] occlusion counter dumps drain the device: %s\n", error.what());
        return false;
    }
    sampleCounterState = 1;
    return true;
}

std::uint64_t Recorder::Batch::StoreRun::Queued::End() const {
    return address + (dumpTarget != 0 ? SampleDumpBytes : bytes.size());
}

std::vector<std::size_t> Recorder::QueuedWriteGroups(std::span<const QueuedWrite> writes) {
    std::vector<std::size_t> ends;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> computed;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        const auto& write = writes[i];
        if (!write.computed && !computed.empty()) {
            ends.push_back(i);
            computed.clear();
        }
        if (write.computed) computed.emplace_back(write.begin, write.end);
    }
    if (!writes.empty()) ends.push_back(writes.size());
    return ends;
}

bool Recorder::segmentMode() const {
    const std::lock_guard recordingGuard(recording);
    return sampleCounterState > 0;
}

bool Recorder::QueuesSampleDumps() const {
    const std::lock_guard recordingGuard(recording);
    return segmentMode() && LabelRunsPerBatch();
}

Recorder::SampleDumpStatistics Recorder::SampleDumpCounts() {
    return SampleDumpStatistics{dumpsQueued.load(std::memory_order_relaxed), dumpsQueuedInPass.load(std::memory_order_relaxed), dumpsAtOnce.load(std::memory_order_relaxed), segmentsClosed.load(std::memory_order_relaxed), dumpFolds.load(std::memory_order_relaxed)};
}

bool Recorder::segmentSlotReady(const Batch& batch) const {
    const std::lock_guard recordingGuard(recording);
    return batch.segmentActive || (batch.segments != VK_NULL_HANDLE && batch.segmentNext < SegmentSlots);
}

void Recorder::PrepareSampleSlot() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    GuestMemory::AssertGpuLockHeld("Recorder::PrepareSampleSlot");
    if (!countingSamples || !segmentMode()) return;
    ensureOpen();
    Require(!open->renderPass.open, "a sample slot is prepared inside a render pass");
    if (!segmentSlotReady(*open)) takeSegmentPool(*open);
}

void Recorder::EndPassSamples() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (open != nullptr && open->segmentActive) endSegment(*open);
}

void Recorder::takeSegmentPool(Batch& batch) {
    const std::lock_guard recordingGuard(recording);
    auto cache = std::static_pointer_cast<SamplePoolCache>(segmentPools);
    if (cache == nullptr) {
        cache = std::make_shared<SamplePoolCache>();
        cache->device = context.device;
        cache->destroy = context.Function<PFN_vkDestroyQueryPool>("vkDestroyQueryPool");
        segmentPools = cache;
    }
    if (batch.segmentPool != nullptr) batch.kept.push_back(std::move(batch.segmentPool));
    batch.segmentPool.reset();
    batch.segments = cache->take();
    if (batch.segments == VK_NULL_HANDLE) {
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_OCCLUSION;
        info.queryCount = SegmentSlots;
        Check(context.Function<PFN_vkCreateQueryPool>("vkCreateQueryPool")(context.device, &info, nullptr, &batch.segments), "vkCreateQueryPool occlusion segments");
    }
    try {
        batch.segmentPool = std::make_shared<SamplePool>(cache, batch.segments);
    } catch (...) {
        cache->put(batch.segments);
        batch.segments = VK_NULL_HANDLE;
        throw;
    }
    batch.segmentNext = 0;
    function(cmdResetQueryPool, "vkCmdResetQueryPool")(batch.commands, batch.segments, 0, SegmentSlots);
}

void Recorder::beginSegment(Batch& batch) {
    const std::lock_guard recordingGuard(recording);
    Require(batch.segments != VK_NULL_HANDLE && batch.segmentNext < SegmentSlots, "an occlusion segment begins without a reset query slot");
    batch.segmentSlot = batch.segmentNext++;
    function(cmdBeginQuery, "vkCmdBeginQuery")(batch.commands, batch.segments, batch.segmentSlot, context.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0u);
    batch.segmentActive = true;
}

void Recorder::endSegment(Batch& batch) {
    const std::lock_guard recordingGuard(recording);
    function(cmdEndQuery, "vkCmdEndQuery")(batch.commands, batch.segments, batch.segmentSlot);
    batch.segmentActive = false;
    pendingSamples.push_back({batch.segmentPool, batch.segments, batch.segmentSlot});
    segmentsClosed.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::copySegments(VkCommandBuffer commands, std::uint32_t count) {
    const std::lock_guard recordingGuard(recording);
    Require(count <= pendingSamples.size() && count <= SegmentFoldCapacity, "occlusion segments exceed the fold capacity");
    const auto copy = function(cmdCopyQueryPoolResults, "vkCmdCopyQueryPoolResults");
    for (std::uint32_t i = 0; i < count;) {
        const auto& first = pendingSamples[i];
        std::uint32_t run = 1;
        while (i + run < count && pendingSamples[i + run].handle == first.handle && pendingSamples[i + run].slot == first.slot + run) ++run;
        copy(commands, first.handle, first.slot, run, sampleCounter->Handle(), 16 + 8 * static_cast<VkDeviceSize>(i), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        i += run;
    }
}

void Recorder::retireSegments(std::uint32_t count) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    for (std::uint32_t i = 0; i < count; ++i) {
        if (pendingSamples[i].pool != nullptr) open->kept.push_back(std::move(pendingSamples[i].pool));
    }
    pendingSamples.erase(pendingSamples.begin(), pendingSamples.begin() + count);
}

void Recorder::recordQueuedDumps(VkCommandBuffer commands, std::span<const Batch::StoreRun::Queued> writes) {
    const std::lock_guard recordingGuard(recording);
    std::uint32_t segments = 0;
    std::vector<std::uint32_t> list;
    for (const auto& write : writes) {
        if (write.dumpTarget == 0) continue;
        segments = write.dumpSegments;
        list.insert(list.end(), {static_cast<std::uint32_t>(write.dumpTarget), static_cast<std::uint32_t>(write.dumpTarget >> 32u), write.dumpSegments, 0u});
    }
    Require(list.size() <= 4u * DumpFoldCapacity, "queued occlusion dumps exceed the fold capacity");
    copySegments(commands, segments);
    function(cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, sampleCounter->Handle(), DumpListOffset, list.size() * sizeof(std::uint32_t), list.data());
}

void Recorder::dispatchDumps(VkCommandBuffer commands, std::uint32_t firstDump, std::uint32_t dumps, std::uint32_t firstSegment, std::uint32_t lastSegment) {
    const std::lock_guard recordingGuard(recording);
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    function(cmdBindPipeline, "vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, dumpPipeline);
    struct {
        VkDeviceAddress counter;
        VkDeviceAddress list;
        std::uint32_t firstDump;
        std::uint32_t dumps;
        std::uint32_t firstSegment;
        std::uint32_t lastSegment;
        std::uint32_t phase;
    } parameters{sampleCounter->DeviceAddress(), sampleCounter->DeviceAddress() + DumpListOffset, firstDump, dumps, firstSegment, lastSegment, 0u};
    static_assert(sizeof(parameters) == SampleDumpPushBytes);
    const auto push = function(cmdPushConstants, "vkCmdPushConstants");
    const auto dispatch = function(cmdDispatch, "vkCmdDispatch");
    push(commands, dumpLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, SampleDumpPushBytes, &parameters);
    dispatch(commands, 1, 1, 1);
    if (dumps == 0) return;
    const auto publish = [&](std::uint32_t phase) {
        recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
        parameters.phase = phase;
        push(commands, dumpLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, SampleDumpPushBytes, &parameters);
        dispatch(commands, 1, 1, 1);
    };
    publish(1u);
    publish(2u);
}

void Recorder::foldQueuedDumps(VkCommandBuffer commands, std::span<const Batch::StoreRun::Queued> writes, std::size_t first, std::size_t end, std::uint32_t& dumpIndex, std::uint32_t& consumed) {
    const std::lock_guard recordingGuard(recording);
    std::uint32_t count = 0, segments = consumed;
    for (std::size_t i = first; i < end; ++i) {
        if (writes[i].dumpTarget == 0) continue;
        ++count;
        segments = writes[i].dumpSegments;
    }
    if (count == 0) return;
    dispatchDumps(commands, dumpIndex, count, consumed, segments);
    dumpIndex += count;
    consumed = segments;
}

void Recorder::foldSegments() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    const auto commands = open->commands;
    const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(pendingSamples.size(), SegmentFoldCapacity));
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    copySegments(commands, count);
    dispatchDumps(commands, 0, 0, 0, count);
    recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
    CountBarriers(CommandClass::Copy, 3);
    retireSegments(count);
}

void Recorder::endSamples(Batch& batch) {
    const std::lock_guard recordingGuard(recording);
    if (!batch.sampleActive) return;
    function(cmdEndQuery, "vkCmdEndQuery")(batch.commands, batch.samples, 0);
    batch.sampleActive = false;
    if (sampleCounterState > 0 && batch.samplesDrawn) pendingSamples.push_back({batch.samplePool, batch.samples});
    batch.samplesDrawn = false;
}

void Recorder::foldSamples(Batch& batch, VkDeviceAddress target) {
    const std::lock_guard recordingGuard(recording);
    const auto commands = batch.commands;
    const auto copy = function(cmdCopyQueryPoolResults, "vkCmdCopyQueryPoolResults");
    const auto bind = function(cmdBindPipeline, "vkCmdBindPipeline");
    const auto push = function(cmdPushConstants, "vkCmdPushConstants");
    const auto dispatch = function(cmdDispatch, "vkCmdDispatch");
    std::size_t done = 0;
    do {
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(SampleFoldCapacity, pendingSamples.size() - done));
        if (done != 0) recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& segment = pendingSamples[done + i];
            copy(commands, segment.handle, segment.slot, 1, sampleCounter->Handle(), 16 + 8 * i, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (segment.pool != nullptr) batch.kept.push_back(segment.pool);
        }
        done += count;
        const bool last = done == pendingSamples.size();
        recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        bind(commands, VK_PIPELINE_BIND_POINT_COMPUTE, samplePipeline);
        struct {
            VkDeviceAddress counter;
            VkDeviceAddress target;
            std::uint32_t count;
            std::uint32_t store;
        } parameters{sampleCounter->DeviceAddress(), target, count, last && target != 0 ? 1u : 0u};
        static_assert(sizeof(parameters) == SampleCounterPushBytes);
        push(commands, sampleLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, SampleCounterPushBytes, &parameters);
        dispatch(commands, 1, 1, 1);
        if (parameters.store == 1u) {
            const auto publish = [&](std::uint32_t store) {
                recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
                parameters.store = store;
                push(commands, sampleLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, SampleCounterPushBytes, &parameters);
                dispatch(commands, 1, 1, 1);
            };
            publish(3u);
            publish(2u);
        }
    } while (done < pendingSamples.size());
    recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_HOST_READ_BIT);
    CountBarriers(CommandClass::Copy, target != 0 ? 4u : 2u);
    pendingSamples.clear();
}

bool Recorder::DumpSamples(VkDeviceAddress target, std::uint64_t address) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    GuestMemory::AssertGpuLockHeld("Recorder::DumpSamples");
    if (target == 0 || !gpuSampleCounter()) return false;
    countingSamples = true;
    if (segmentMode()) {
        ensureOpen();
        if (address != 0 && LabelRunsPerBatch() && pendingSamples.size() < SegmentFoldCapacity && open->queuedDumps < DumpFoldCapacity) {
            if (open->segmentActive) endSegment(*open);
            if (open->renderPass.open) dumpsQueuedInPass.fetch_add(1, std::memory_order_relaxed);
            if (open->run.queued.empty()) storeRuns.fetch_add(1, std::memory_order_relaxed);
            auto& queued = open->run.queued.emplace_back();
            queued.address = address;
            queued.dumpTarget = target;
            queued.dumpSegments = static_cast<std::uint32_t>(pendingSamples.size());
            ++open->queuedDumps;
            dumpsQueued.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        FlushStores();
        Commands();
        foldSamples(*open, target);
        dumpsAtOnce.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    Commands();
    endSamples(*open);
    foldSamples(*open, target);
    if (open->samplePool != nullptr) open->kept.push_back(std::move(open->samplePool));
    open->samplePool.reset();
    open->samples = VK_NULL_HANDLE;
    beginSamples(*open);
    return true;
}

bool Recorder::RecordDispatchThreads(VkCommandBuffer commands, VkDeviceAddress threads, VkDeviceAddress scratch, std::span<const std::uint32_t, 3> size) {
    const std::lock_guard recordingGuard(recording);
    constexpr std::uint32_t PushBytes = 32;
    if (dispatchThreadsState == 0) {
        dispatchThreadsState = -1;
        VkShaderModule module = VK_NULL_HANDLE;
        try {
            const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, PushBytes};
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &push;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &dispatchThreadsLayout), "vkCreatePipelineLayout dispatch threads");
            VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            moduleInfo.codeSize = sizeof(DISPATCH_THREADS_SPV);
            moduleInfo.pCode = DISPATCH_THREADS_SPV;
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule dispatch threads");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = module;
            pipelineInfo.stage.pName = "main";
            pipelineInfo.layout = dispatchThreadsLayout;
            Check(CreateComputePipeline(context, pipelineInfo, &dispatchThreadsPipeline), "vkCreateComputePipelines dispatch threads");
            context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            dispatchThreadsState = 1;
        } catch (const std::exception& error) {
            if (module != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            std::fprintf(stderr, "[gpu] indirect dispatches with thread dimensions read their arguments on the CPU: %s\n", error.what());
        }
    }
    if (dispatchThreadsState < 0) return false;
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    function(cmdBindPipeline, "vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, dispatchThreadsPipeline);
    struct {
        VkDeviceAddress threads;
        VkDeviceAddress scratch;
        std::array<std::uint32_t, 4> size;
    } parameters{threads, scratch, {size[0], size[1], size[2], 0u}};
    static_assert(sizeof(parameters) == PushBytes);
    function(cmdPushConstants, "vkCmdPushConstants")(commands, dispatchThreadsLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, PushBytes, &parameters);
    function(cmdDispatch, "vkCmdDispatch")(commands, 1, 1, 1);
    recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    return true;
}

bool Recorder::RecordMeshArguments(VkCommandBuffer commands, VkDeviceAddress record, VkDeviceAddress arguments, std::span<const std::uint32_t, 7> rules) {
    const std::lock_guard recordingGuard(recording);
    constexpr std::uint32_t PushBytes = 48;
    if (meshArgumentState == 0) {
        meshArgumentState = -1;
        VkShaderModule module = VK_NULL_HANDLE;
        try {
            const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, PushBytes};
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &push;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &meshArgumentLayout), "vkCreatePipelineLayout mesh arguments");
            VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            moduleInfo.codeSize = sizeof(MESH_ARGUMENTS_SPV);
            moduleInfo.pCode = MESH_ARGUMENTS_SPV;
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule mesh arguments");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = module;
            pipelineInfo.stage.pName = "main";
            pipelineInfo.layout = meshArgumentLayout;
            Check(CreateComputePipeline(context, pipelineInfo, &meshArgumentPipeline), "vkCreateComputePipelines mesh arguments");
            context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            meshArgumentState = 1;
        } catch (const std::exception& error) {
            if (module != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            std::fprintf(stderr, "[gpu] mesh indirect draws read their records on the CPU: %s\n", error.what());
        }
    }
    if (meshArgumentState < 0) return false;
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    function(cmdBindPipeline, "vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, meshArgumentPipeline);
    struct {
        VkDeviceAddress record;
        VkDeviceAddress arguments;
        std::array<std::uint32_t, 8> rules;
    } parameters{record, arguments, {rules[0], rules[1], rules[2], rules[3], rules[4], rules[5], rules[6], 0u}};
    static_assert(sizeof(parameters) == PushBytes);
    function(cmdPushConstants, "vkCmdPushConstants")(commands, meshArgumentLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, PushBytes, &parameters);
    function(cmdDispatch, "vkCmdDispatch")(commands, 1, 1, 1);
    recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    CountBarriers(CommandClass::Draw, 2);
    return true;
}

void Recorder::beginSamples(Batch& batch) {
    const std::lock_guard recordingGuard(recording);
    auto cache = std::static_pointer_cast<SamplePoolCache>(samplePools);
    if (sampleCounterState > 0 && cache == nullptr) {
        cache = std::make_shared<SamplePoolCache>();
        cache->device = context.device;
        cache->destroy = context.Function<PFN_vkDestroyQueryPool>("vkDestroyQueryPool");
        samplePools = cache;
    }
    batch.samples = sampleCounterState > 0 ? cache->take() : VK_NULL_HANDLE;
    if (batch.samples == VK_NULL_HANDLE) {
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_OCCLUSION;
        info.queryCount = 1;
        Check(context.Function<PFN_vkCreateQueryPool>("vkCreateQueryPool")(context.device, &info, nullptr, &batch.samples), "vkCreateQueryPool occlusion");
    }
    if (sampleCounterState > 0) {
        try {
            batch.samplePool = std::make_shared<SamplePool>(cache, batch.samples);
        } catch (...) {
            cache->put(batch.samples);
            batch.samples = VK_NULL_HANDLE;
            throw;
        }
    }
    function(cmdResetQueryPool, "vkCmdResetQueryPool")(batch.commands, batch.samples, 0, 1);
    function(cmdBeginQuery, "vkCmdBeginQuery")(batch.commands, batch.samples, 0, context.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0u);
    batch.sampleActive = true;
    batch.samplesDrawn = false;
}

void Recorder::readSamples(Batch& batch) {
    const std::lock_guard recordingGuard(recording);
    if (sampleCounterState > 0) return;
    if (batch.samples == VK_NULL_HANDLE || !batch.submitted) return;
    std::uint64_t samples = 0;
    Check(context.Function<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(context.device, batch.samples, 0, 1, sizeof(samples), &samples, sizeof(samples), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults occlusion");
    samplesPassed.fetch_add(samples, std::memory_order_acq_rel);
}

void Recorder::readGpuTiming(Batch& batch) {
    const std::lock_guard recordingGuard(recording);
    if (batch.queries == VK_NULL_HANDLE || batch.timedKeys.empty()) return;
    std::vector<std::uint64_t> stamps(batch.timedKeys.size() * 2);
    const auto result = context.Function<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(context.device, batch.queries, 0, static_cast<std::uint32_t>(stamps.size()), stamps.size() * sizeof(std::uint64_t), stamps.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) return;
    const auto period = context.limits.timestampPeriod;
    if (!GpuTimingEnabled()) {
        if (batch.batchTiming == 0 && stamps[1] >= stamps[0]) {
            batch.gpuStartNs = static_cast<double>(stamps[0]) * period;
            batch.gpuEndNs = static_cast<double>(stamps[1]) * period;
            PerformanceTrace::GpuRange("batch", batch.gpuStartNs, batch.gpuEndNs, batch.traceId);
        }
        return;
    }
    static auto lastReport = std::chrono::steady_clock::now();
    // The union of the timed ranges (class ranges nest program ranges: a copy's transfer inside
    // its class range) against the batch span gives what no range covers.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
    intervals.reserve(batch.timedKeys.size());
    std::unique_lock lock(timingMutex);
    for (std::size_t i = 0; i < batch.timedKeys.size(); ++i) {
        const auto ns = static_cast<double>(stamps[i * 2 + 1] - stamps[i * 2]) * period;
        if (i == batch.batchTiming) {
            // The whole batch: apart from the per-program totals it contains.
            batch.gpuStartNs = static_cast<double>(stamps[i * 2]) * period;
            batch.gpuEndNs = static_cast<double>(stamps[i * 2 + 1]) * period;
            PerformanceTrace::GpuRange("batch", batch.gpuStartNs, batch.gpuEndNs, batch.traceId);
            timingBatchMs += ns / 1e6;
            continue;
        }
        const auto key = batch.timedKeys[i];
        if (key >= ClassKey(CommandClass::DispatchLeading) && key < ClassKey(CommandClass::Count)) PerformanceTrace::GpuRange(CommandClassNames[key - ClassKey(CommandClass::DispatchLeading)], static_cast<double>(stamps[i * 2]) * period, static_cast<double>(stamps[i * 2 + 1]) * period, batch.traceId);
        auto& totals = timingByKey[key];
        ++totals.count;
        totals.ms += ns / 1e6;
        totals.bytes += batch.timedBytes[i];
        if (const auto& groups = batch.timedGroups[i]; groups[0] != 0) {
            if (totals.groups[0] != 0 && totals.groups != groups) totals.mixedGroups = true;
            totals.groups = groups;
        }
        (key >= ClassKey(CommandClass::DispatchLeading) && key < ClassKey(CommandClass::Count) ? timingClassMs : timingProgramMs) += ns / 1e6;
        intervals.emplace_back(stamps[i * 2], stamps[i * 2 + 1]);
    }
    std::sort(intervals.begin(), intervals.end());
    std::uint64_t coveredTicks = 0, unionEnd = 0;
    for (const auto& [begin, end] : intervals) {
        const auto from = std::max(begin, unionEnd);
        if (end > from) coveredTicks += end - from;
        unionEnd = std::max(unionEnd, end);
    }
    timingUnionMs += static_cast<double>(coveredTicks) * period / 1e6;
    ++timingBatches;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    const auto presents = timingPresents.exchange(0, std::memory_order_relaxed);
    const double perPresent = presents != 0 ? 1.0 / static_cast<double>(presents) : 0.0;
    std::vector<std::pair<std::uint64_t, TimingTotals>> hot;
    std::string classes;
    for (const auto& [key, totals] : timingByKey) {
        if (key >= ClassKey(CommandClass::DispatchLeading) && key < ClassKey(CommandClass::Count)) {
            char text[160];
            std::snprintf(text, sizeof(text), " %s x%llu %.1fms %.1fMiB (per present x%.1f %.2fms %.2fMiB)", CommandClassNames[key - ClassKey(CommandClass::DispatchLeading)], static_cast<unsigned long long>(totals.count), totals.ms, totals.bytes / 1048576.0, static_cast<double>(totals.count) * perPresent, totals.ms * perPresent, totals.bytes / 1048576.0 * perPresent);
            classes += text;
        } else {
            hot.emplace_back(key, totals);
        }
    }
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    // The first field stays the program sum: the measure scripts match on it.
    AgcDriver::ProfilePrint_nid_no_patch("[gputime] %.0f ms of GPU time in %llu batches over 10 s (batch %.0f ms first to last command; classes %.0f ms, all ranges %.0f ms, untimed %.0f ms outside every range; %llu presents; %llu ranges dropped at the %u cap); by program:", timingProgramMs, static_cast<unsigned long long>(timingBatches), timingBatchMs, timingClassMs, timingUnionMs, timingBatchMs - timingUnionMs, static_cast<unsigned long long>(presents), static_cast<unsigned long long>(timingDropped.exchange(0, std::memory_order_relaxed)), MaxTimedRanges);
    for (std::size_t i = 0; i < hot.size() && i < 12; ++i) {
        const auto& totals = hot[i].second;
        if (totals.groups[0] == 0) AgcDriver::ProfilePrint_nid_no_patch(" 0x%llx x%llu %.0fms", static_cast<unsigned long long>(hot[i].first), static_cast<unsigned long long>(totals.count), totals.ms);
        else AgcDriver::ProfilePrint_nid_no_patch(" 0x%llx x%llu %.0fms %ux%ux%u%s", static_cast<unsigned long long>(hot[i].first), static_cast<unsigned long long>(totals.count), totals.ms, totals.groups[0], totals.groups[1], totals.groups[2], totals.mixedGroups ? " (last of several)" : "");
    }
    AgcDriver::ProfilePrint_nid_no_patch("; by class:%s\n", classes.c_str());
    timingByKey.clear();
    timingProgramMs = timingClassMs = timingUnionMs = timingBatchMs = 0;
    timingBatches = 0;
    lock.unlock();
    reportBarriers();
}

bool Recorder::FlipReadCheck() {
    static const bool enabled = std::getenv("APS5_FLIP_READ_CHECK") != nullptr;
    return enabled;
}

std::vector<Recorder::Completed> Recorder::CompletedBatches(std::uint64_t afterSerial, std::uint64_t throughSerial, std::size_t& missing) const {
    std::vector<Completed> found;
    missing = 0;
    if (throughSerial <= afterSerial) return found;
    std::lock_guard lock(completedMutex);
    for (auto serial = afterSerial + 1; serial <= throughSerial; ++serial) {
        const auto& entry = completed[serial % completed.size()];
        if (entry.serial == serial) found.push_back(entry);
        else ++missing;
    }
    return found;
}

std::uint64_t Recorder::NewestSubmitted(std::chrono::steady_clock::time_point* submittedAt) const {
    std::lock_guard lock(completedMutex);
    if (submittedAt != nullptr) *submittedAt = newestSubmittedAt;
    return newestSubmitted;
}

std::size_t Recorder::UnsignaledBatches() const {
    const std::lock_guard recordingGuard(recording);
    return static_cast<std::size_t>(std::count_if(inFlight.begin(), inFlight.end(), [&](const auto& batch) { return !signaled(*batch); }));
}

void Recorder::Keep(std::shared_ptr<void> object, std::size_t bytes) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    ensureOpen();
    if (streams.size() > 1) keepOnOtherStreams(object);
    open->kept.push_back(std::move(object));
    open->keptBytes += bytes;
}

void Recorder::KeepThroughHold(std::shared_ptr<void> object) {
    const std::lock_guard recordingGuard(recording);
    GuestMemory::AssertGpuLockHeld("Recorder::KeepThroughHold");
    if (const auto hold = GuestMemory::GpuHoldSerial(); hold != holdKeptSerial) {
        holdKept.clear();
        holdKeptSerial = hold;
    }
    Keep(object);
    holdKept.push_back(std::move(object));
}

void Recorder::dropStaleHoldKept() {
    const std::lock_guard recordingGuard(recording);
    if (!holdKept.empty() && holdKeptSerial != GuestMemory::GpuHoldSerial() && GuestMemory::GpuMutex().HeldByThisThread()) holdKept.clear();
}

void Recorder::KeepBytes(const void* owner, std::size_t bytes) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (owner == nullptr || bytes == 0) return;
    ensureOpen();
    if (open->keptOwners.insert(owner).second) open->keptBytes += bytes;
}

void Recorder::BoundKeptBytes() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (!GuestMemory::GpuMutex().HeldByThisThread()) return;
    if (open != nullptr && open->keptBytes >= KeptBytesBudget) Submit();
    if (DeferredBytes() != 0 && inFlightKeptBytes + DeferredBytes() > 2 * KeptBytesBudget) DestroyDeferred(TakeDeferred(), false);
    while (!inFlight.empty() && inFlightKeptBytes > 2 * KeptBytesBudget) {
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, 4, true);
    }
}

bool Recorder::KnownHighestIndex(std::uint64_t address, std::size_t bytes, IndexKind use, std::optional<std::uint32_t>& highest) {
    const std::lock_guard recordingGuard(recording);
    const auto found = indexRanges.find({address, use, bytes});
    if (found == indexRanges.end()) return false;
    if (found->second.registryGeneration != GuestAllocations::GuestAllocationsGeneration_nid_postfix() || !GuestMemory::UnchangedSince(address, bytes, found->second.generation)) {
        indexRangeRecency.erase(found->second.recent);
        indexRanges.erase(found);
        return false;
    }
    indexRangeRecency.splice(indexRangeRecency.end(), indexRangeRecency, found->second.recent);
    highest = found->second.highest;
    return true;
}

void Recorder::KeepHighestIndex(std::uint64_t address, std::size_t bytes, IndexKind use, std::uint64_t generation, std::uint64_t registryGeneration, std::optional<std::uint32_t> highest) {
    const std::lock_guard recordingGuard(recording);
    if (generation == 0) return;
    const IndexRangeKey key{address, use, bytes};
    if (const auto found = indexRanges.find(key); found != indexRanges.end()) {
        indexRangeRecency.erase(found->second.recent);
        indexRanges.erase(found);
    }
    while (indexRanges.size() >= IndexRangeEntries) {
        indexRanges.erase(indexRangeRecency.front());
        indexRangeRecency.pop_front();
    }
    indexRangeRecency.push_back(key);
    try {
        indexRanges.emplace(key, IndexRange{generation, registryGeneration, std::prev(indexRangeRecency.end()), highest});
    } catch (...) {
        indexRangeRecency.pop_back();
        throw;
    }
}

std::shared_ptr<BatchFaultRecord> Recorder::CallerFaultRecord(std::uint32_t chunks) {
    const std::lock_guard recordingGuard(recording);
    const auto stream = streamOfCaller();
    const auto batch = opens[stream] != nullptr ? opens[stream]->value : streams[stream].opened + 1u;
    auto& current = streams[stream].faultRecord;
    if (current == nullptr || current->Batch() != batch || !current->Admit(chunks)) {
        current = std::make_shared<BatchFaultRecord>(context, stream, batch);
        Require(current->Admit(chunks), "a build may store to more chunks than a fault record holds");
    }
    return current;
}

bool Recorder::UseRecord(const std::shared_ptr<Buffer>& device, VkDeviceSize offset, VkDeviceSize cleared, const std::shared_ptr<Buffer>& host, VkDeviceSize hostOffset, VkDeviceSize bytes) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    Require(open != nullptr, "a GPU record was used outside an open batch");
    Require(host != nullptr, "a GPU record has no host copy");
    for (const auto& copy : open->recordCopies) {
        if (copy.device == device.get() && copy.offset == offset) return false;
    }
    context.Resolved(&DeviceFunctions::cmdFillBuffer, "vkCmdFillBuffer")(open->commands, device->Handle(), offset, cleared, 0u);
    open->recordCopies.push_back({device.get(), offset, device->Handle(), host->Handle(), hostOffset, bytes});
    Keep(device);
    Keep(host);
    return true;
}

bool Recorder::RecordUsed(const Buffer* device, VkDeviceSize offset) const {
    const std::lock_guard recordingGuard(recording);
    const auto* open = callerBatch();
    if (open == nullptr) return false;
    return std::any_of(open->recordCopies.begin(), open->recordCopies.end(), [&](const Batch::RecordCopy& copy) { return copy.device == device && copy.offset == offset; });
}

bool Recorder::CallerFaultRecordCurrent(const BatchFaultRecord& record) const {
    const std::lock_guard recordingGuard(recording);
    const auto stream = streamOfCaller();
    return opens[stream] != nullptr && record.Stream() == stream && record.Batch() == opens[stream]->value;
}

BatchFaults& Recorder::CallerBatchFaults() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    ensureOpen();
    if (open->faults == nullptr) {
        open->faults = std::make_shared<BatchFaults>();
        open->completions.push_back([faults = open->faults] { faults->Complete(); });
        ++open->writeBackCompletionCount;
        writeBackCompletions.fetch_add(1, std::memory_order_acq_rel);
    }
    return *open->faults;
}

void Recorder::OnComplete(std::function<void()> action) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    ensureOpen();
    open->completions.push_back(std::move(action));
    ++open->writeBackCompletionCount;
    writeBackCompletions.fetch_add(1, std::memory_order_acq_rel);
}

bool Recorder::noteWrite(std::uint64_t address, std::size_t bytes, bool ownLabel, int value) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    CaptureTrace::Log("buffer-write batch=%llu address=%llx bytes=%zu label=%d", static_cast<unsigned long long>(submissions + 1), static_cast<unsigned long long>(address), bytes, ownLabel);
    if (bytes == 0) return false;
    ensureOpen();
    const auto end = address + bytes;
    open->writes.emplace_back(address, end);
    open->writeNotes.push_back(++writeNoteCount);
    open->writeValues.push_back(static_cast<std::int16_t>(value));
    // A poller waiting on this range learns that the open batch may now hold its producer.
    if (activeRecorder == this) writeGeneration.fetch_add(1, std::memory_order_release);
    // Consecutive dispatches write the same output buffers: a range the snapshot already contains
    // leaves the published union unchanged, so the O(N log N) rebuild is skipped.
    if (SnapshotCovers(address, end)) {
        ++snapshotCovered;
        return false;
    }
    return true;
}

void Recorder::noteWriteOn(Batch& batch, std::uint64_t address, std::size_t bytes, bool ownLabel) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (bytes == 0) return;
    if (&batch == open.get()) {
        if (!noteWrite(address, bytes, ownLabel)) return;
        publishPendingWrites();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return;
    }
    // An in-flight batch: its range joins the snapshot at once (the completion that stores it runs
    // when the batch finishes, and the hook must sync for a CPU read until then). The generation
    // moves as well, so a poller re-consults the label table for a completion label.
    batch.writes.emplace_back(address, address + bytes);
    batch.writeNotes.push_back(++writeNoteCount);
    batch.writeValues.push_back(-1);
    if (activeRecorder == this) writeGeneration.fetch_add(1, std::memory_order_release);
    if (!SnapshotCovers(address, address + bytes)) publishPendingWrites();
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::NotePendingWrite(std::uint64_t address, std::size_t bytes) {
    const std::lock_guard recordingGuard(recording);
    if (!noteWrite(address, bytes)) return;
    publishPendingWrites();
    // The note precedes this thread's vkQueueSubmit and the label another queue polls for; the
    // fence makes that order hold without relying on x86 store ordering.
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::NotePendingFill(std::uint64_t address, std::size_t bytes, std::uint8_t value) {
    const std::lock_guard recordingGuard(recording);
    if (!noteWrite(address, bytes, false, value)) return;
    publishPendingWrites();
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::NotePendingWrites(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    const std::lock_guard recordingGuard(recording);
    bool publish = false;
    for (const auto& [begin, end] : ranges) {
        if (end > begin && noteWrite(begin, static_cast<std::size_t>(end - begin))) publish = true;
    }
    if (!publish) return;
    publishPendingWrites();
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::publishPendingWrites() const {
    const std::lock_guard recordingGuard(recording);
    if (activeRecorder != this) return;
    const auto started = DrawProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& sources = publishSources;
    sources.clear();
    const auto add = [&](const Batch& batch) { sources.push_back({&batch, batch.writes, batch.writeNotes}); };
    forEachOpen(add);
    for (const auto& batch : inFlight) add(*batch);
    for (const auto* batch : finishing) add(*batch);
    auto published = publishedWrites.Publish(sources, pendingWrites.load(std::memory_order_acquire));
    pendingWrites.store(std::move(published.snapshot), std::memory_order_release);
    publishGeneration.fetch_add(1, std::memory_order_release);
    if (published.rebuilt) ++snapshotRebuilds;
    if (DrawProfiled()) snapshotRebuildMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

bool Recorder::overlaps(const Batch& batch, std::uint64_t address, std::uint64_t end) {
    for (const auto& [begin, finish] : batch.writes) {
        if (address < finish && begin < end) return true;
    }
    return false;
}

const Recorder::Batch* Recorder::openOverlapping(std::uint64_t address, std::uint64_t end) const {
    const std::lock_guard recordingGuard(recording);
    for (std::size_t stream = 0; stream < streams.size(); ++stream) {
        if (opens[stream] != nullptr && overlaps(*opens[stream], address, end)) return opens[stream].get();
    }
    return nullptr;
}

bool Recorder::PendingWriteOverlaps(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    if (bytes == 0) return false;
    const auto end = address + bytes;
    if (openOverlapping(address, end) != nullptr) return true;
    for (const auto& batch : inFlight) {
        if (overlaps(*batch, address, end)) return true;
    }
    return false;
}

bool Recorder::OpenWriteOverlaps(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    return bytes != 0 && openOverlapping(address, address + bytes) != nullptr;
}

bool Recorder::signaled(const Batch& batch) const {
    const std::lock_guard recordingGuard(recording);
    return batch.submitted && fenceStatus(batch.fence) == VK_SUCCESS;
}

bool Recorder::PendingWriteSettled(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    if (bytes == 0) return true;
    const auto end = address + bytes;
    if (openOverlapping(address, end) != nullptr) return false;
    for (const auto& batch : inFlight) {
        if (overlaps(*batch, address, end) && !signaled(*batch)) return false;
    }
    return true;
}

std::uint64_t Recorder::LastWriteNote(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    const auto* open = callerBatch();
    if (open == nullptr || open->writes.empty() || open->writeNotes.size() != open->writes.size()) return 0;
    if (open->writes.back() != std::pair<std::uint64_t, std::uint64_t>{address, address + bytes}) return 0;
    return open->writeNotes.back() == writeNoteCount ? writeNoteCount : 0;
}

std::uint64_t Recorder::NewestWriteNote(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    if (bytes == 0) return 0;
    const auto end = address + bytes;
    std::uint64_t newest = 0;
    const auto scan = [&](const Batch& batch) {
        for (std::size_t i = 0; i < batch.writes.size(); ++i) {
            if (address < batch.writes[i].second && batch.writes[i].first < end) newest = std::max(newest, batch.writeNotes[i]);
        }
    };
    forEachOpen(scan);
    for (const auto& batch : inFlight) scan(*batch);
    return newest;
}

std::vector<Recorder::PendingWrite> Recorder::PendingWritesOver(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    std::vector<PendingWrite> found;
    if (bytes == 0) return found;
    const auto end = address + bytes;
    const auto scan = [&](const Batch& batch) {
        for (std::size_t i = 0; i < batch.writes.size(); ++i) {
            if (address < batch.writes[i].second && batch.writes[i].first < end) found.push_back({batch.writes[i].first, batch.writes[i].second, batch.writeNotes[i], batch.writeValues[i]});
        }
    };
    forEachOpen(scan);
    for (const auto& batch : inFlight) scan(*batch);
    return found;
}

bool Recorder::ReadTracking() {
    return ReadTrackingEnabled();
}

void Recorder::NotePendingRead(std::uint64_t address, std::size_t bytes, ReadKind kind) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    CaptureTrace::Log("buffer-read batch=%llu address=%llx bytes=%zu kind=%d", static_cast<unsigned long long>(submissions + 1), static_cast<unsigned long long>(address), bytes, static_cast<int>(kind));
    if (bytes == 0 || !ReadTrackingEnabled()) return;
    ensureOpen();
    open->reads.push_back({address, address + bytes, kind});
    readsNoted.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::NotePendingReads(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, ReadKind kind) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (ranges.empty() || !ReadTrackingEnabled()) return;
    ensureOpen();
    for (const auto& [begin, end] : ranges) {
        CaptureTrace::Log("buffer-read batch=%llu address=%llx bytes=%llu kind=%d", static_cast<unsigned long long>(submissions + 1), static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin), static_cast<int>(kind));
        if (end > begin) open->reads.push_back({begin, end, kind});
    }
    readsNoted.fetch_add(ranges.size(), std::memory_order_relaxed);
}

const Recorder::Batch::Read* Recorder::readOverlap(const Batch& batch, std::uint64_t address, std::uint64_t end) {
    for (const auto& read : batch.reads) {
        if (address < read.end && read.begin < end) return &read;
    }
    return nullptr;
}

bool Recorder::PendingReadOverlaps(std::uint64_t address, std::size_t bytes, bool ignoreSignaled) const {
    const std::lock_guard recordingGuard(recording);
    if (bytes == 0) return false;
    readQueries.fetch_add(1, std::memory_order_relaxed);
    const auto end = address + bytes;
    for (std::size_t stream = 0; stream < streams.size(); ++stream) {
        if (opens[stream] == nullptr) continue;
        if (const auto* read = readOverlap(*opens[stream], address, end)) {
            readHits[static_cast<std::size_t>(read->kind)].fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    // Newest first: the batch most likely still running decides without a status query per batch.
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
        const auto* read = readOverlap(**it, address, end);
        if (read == nullptr) continue;
        if (ignoreSignaled && signaled(**it)) {
            readStaleIgnored.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        readHits[static_cast<std::size_t>(read->kind)].fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

std::optional<Recorder::PendingReadInfo> Recorder::DescribePendingRead(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    if (bytes == 0) return std::nullopt;
    const auto end = address + bytes;
    for (std::size_t stream = 0; stream < streams.size(); ++stream) {
        if (opens[stream] == nullptr) continue;
        if (const auto* read = readOverlap(*opens[stream], address, end)) return PendingReadInfo{submissions + 1, opens[stream]->queue, read->kind, true, false};
    }
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
        if (const auto* read = readOverlap(**it, address, end)) return PendingReadInfo{(*it)->serial, (*it)->queue, read->kind, false, signaled(**it)};
    }
    return std::nullopt;
}

Recorder::ReadStatistics Recorder::ReadCounts() {
    ReadStatistics counts{readsNoted.load(std::memory_order_relaxed), readQueries.load(std::memory_order_relaxed), readStaleIgnored.load(std::memory_order_relaxed), {}};
    for (std::size_t kind = 0; kind < ReadKinds; ++kind) counts.hits[kind] = readHits[kind].load(std::memory_order_relaxed);
    return counts;
}

std::optional<Recorder::PendingWriteInfo> Recorder::DescribePendingWrite(std::uint64_t address, std::size_t bytes) const {
    const std::lock_guard recordingGuard(recording);
    if (bytes == 0) return std::nullopt;
    const auto end = address + bytes;
    // The same choice SyncThrough makes: the open batch forces a full Sync, otherwise the newest
    // overlapping in-flight batch is the target. Of that batch's ranges the first overlapping one
    // is reported (a dispatch notes each written V# once, so it is the range the access hit).
    const auto firstOverlap = [&](const Batch& batch) -> const std::pair<std::uint64_t, std::uint64_t>* {
        for (const auto& range : batch.writes) {
            if (address < range.second && range.first < end) return &range;
        }
        return nullptr;
    };
    // With SyncThrough disabled every hit is a full Sync: the open batch (if any) is submitted and
    // everything in flight finishes, whichever batch noted the range.
    const bool syncAll = !SyncThroughEnabled();
    const bool recording = anyOpen();
    const auto allBatches = inFlight.size() + (recording ? 1 : 0);
    for (std::size_t stream = 0; stream < streams.size(); ++stream) {
        if (opens[stream] == nullptr) continue;
        if (const auto* range = firstOverlap(*opens[stream])) return PendingWriteInfo{submissions + 1, true, false, range->first, range->second, inFlight.size() + 1};
    }
    std::size_t finished = inFlight.size();
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it, --finished) {
        const auto* range = firstOverlap(**it);
        if (range == nullptr) continue;
        // In flight means submitted, so the fence is live; signaled = the GPU already ran it.
        const bool signaled = fenceStatus((*it)->fence) == VK_SUCCESS;
        if (syncAll) return PendingWriteInfo{submissions + (recording ? 1 : 0), recording, signaled, range->first, range->second, allBatches};
        return PendingWriteInfo{(*it)->serial, false, signaled, range->first, range->second, finished};
    }
    return std::nullopt;
}

bool Recorder::HasCompletions() const {
    const std::lock_guard recordingGuard(recording);
    bool open = false;
    forEachOpen([&](const Batch& batch) { open = open || !batch.completions.empty(); });
    if (open) return true;
    for (const auto& batch : inFlight) {
        if (!batch->completions.empty()) return true;
    }
    return false;
}

void Recorder::noteLabelOn(Batch& batch, std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue, bool behindCompletion) {
    const std::lock_guard recordingGuard(recording);
    std::lock_guard tableLock(labelTableMutex);
    for (std::size_t offset = 0; offset + 4 <= bytes.size(); offset += 4) {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, 4);
        const auto dword = address + offset;
        auto& writers = labels[dword];
        const LabelWriter entry{batch.stream, batch.value, queue, stamp, value, behindCompletion};
        const auto found = std::find_if(writers.begin(), writers.end(), [&](const LabelWriter& writer) { return writer.stream == batch.stream; });
        if (found != writers.end()) *found = entry;
        else writers.push_back(entry);
        batch.labelDwords.push_back(dword);
    }
    recordedLabels.store(labels.size(), std::memory_order_relaxed);
}

void Recorder::markBehindCompletion(const Batch& batch, std::uint64_t begin, std::uint64_t end) {
    const std::lock_guard recordingGuard(recording);
    std::lock_guard tableLock(labelTableMutex);
    for (auto it = labels.lower_bound(begin); it != labels.end() && it->first < end; ++it) {
        for (auto& writer : it->second) {
            if (writer.stream == batch.stream && writer.batch == batch.value) writer.behindCompletion = true;
        }
    }
}

void Recorder::NoteLabel(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    ensureOpen();
    // The table entry before the write note: the note bumps the write generation a poller
    // watches, and a poller that sees the bump then finds the label without the GPU mutex.
    const bool tabled = bytes.size() <= LabelTableBytes;
    if (tabled) {
        noteLabelOn(*open, address, bytes, stamp, queue);
    } else {
        std::lock_guard tableLock(labelTableMutex);
        labels.erase(labels.lower_bound(address), labels.lower_bound(address + bytes.size()));
        recordedLabels.store(labels.size(), std::memory_order_relaxed);
        open->wideLabels.push_back(wideLabels.emplace(address, WideLabel{address + bytes.size(), std::vector<std::byte>(bytes.begin(), bytes.end())}));
        wideLabelBytes = std::max<std::uint64_t>(wideLabelBytes, bytes.size());
    }
    startLabelDeadline(open->stream);
    noteWriteOn(*open, address, bytes.size(), true);
}

std::optional<Recorder::LabelHit> Recorder::PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, const LabelAccepts& accepts, bool* behindCompletion) const {
    // Under the GPU mutex; the table mutex is taken too so this one path serves both lock regimes.
    std::lock_guard tableLock(labelTableMutex);
    return lookupLabel(address, bytes, afterStamp, accepts, behindCompletion);
}

bool Recorder::LabelPendingAt(std::uint64_t address) const {
    std::lock_guard tableLock(labelTableMutex);
    if (labels.contains(address)) return true;
    const auto queued = queuedLabels.find(address);
    return queued != queuedLabels.end() && queued->second.queue == GuestMemory::GpuLockThreadTag();
}

Recorder::Token Recorder::LabelWriterBatches(std::uint64_t address, std::size_t bytes) const {
    Token token{};
    std::lock_guard tableLock(labelTableMutex);
    for (auto dword = address & ~std::uint64_t{3}; dword < address + bytes; dword += 4) {
        const auto found = labels.find(dword);
        if (found == labels.end()) continue;
        for (const auto& writer : found->second) token[writer.stream] = std::max(token[writer.stream], writer.batch);
    }
    return token;
}

std::size_t Recorder::PendingLabels() const {
    std::lock_guard tableLock(labelTableMutex);
    return labels.size();
}

bool Recorder::PendingLabelIn(std::uint64_t address, std::size_t bytes) const {
    std::lock_guard tableLock(labelTableMutex);
    if (bytes == 0) return false;
    const auto end = address + bytes;
    if (const auto first = labels.lower_bound(address); first != labels.end() && first->first < end) return true;
    if (wideLabelInLocked(address, bytes)) return true;
    // A queued label of another queue is unordered against the caller on hardware (nothing that
    // queue recorded could have satisfied a wait yet); it is still reported, since the table mutex
    // is held anyway and a caller deciding a CPU store wants the conservative answer.
    if (const auto first = queuedLabels.lower_bound(address); first != queuedLabels.end() && first->first < end) return true;
    return false;
}

bool Recorder::LabelStoredAtCompletionIn(std::uint64_t address, std::size_t bytes) const {
    std::lock_guard tableLock(labelTableMutex);
    if (bytes == 0) return false;
    const auto end = address + bytes;
    for (auto it = labels.lower_bound(address); it != labels.end() && it->first < end; ++it) {
        if (std::any_of(it->second.begin(), it->second.end(), [](const LabelWriter& writer) { return writer.behindCompletion; })) return true;
    }
    return false;
}

std::vector<std::pair<std::uint64_t, std::size_t>> Recorder::CompletionLabelRanges() const {
    std::lock_guard tableLock(labelTableMutex);
    std::vector<std::pair<std::uint64_t, std::size_t>> ranges;
    for (const auto& [address, writers] : labels) {
        if (std::any_of(writers.begin(), writers.end(), [](const LabelWriter& writer) { return writer.behindCompletion; })) ranges.emplace_back(address, sizeof(std::uint32_t));
    }
    return ranges;
}

bool Recorder::CompletionLabelIn(std::uint64_t address, std::size_t bytes) const {
    std::lock_guard tableLock(labelTableMutex);
    if (bytes == 0) return false;
    const auto end = address + bytes;
    for (auto it = labels.lower_bound(address); it != labels.end() && it->first < end; ++it) {
        if (std::any_of(it->second.begin(), it->second.end(), [](const LabelWriter& writer) { return writer.behindCompletion; })) return true;
    }
    if (const auto first = queuedLabels.lower_bound(address); first != queuedLabels.end() && first->first < end) return true;
    return false;
}

std::uint32_t Recorder::knownStreamOfCaller() const {
    if (streams.size() == 1) return 0;
    const auto tag = GuestMemory::GpuLockThreadTag();
    if (tag == 0 || tag >= queueStreams.size()) return 0;
    const auto slot = queueStreams[tag];
    return slot == 0 ? NoStream : slot - 1u;
}

std::optional<Recorder::LabelHit> Recorder::SelectLabelWriter(std::span<const std::vector<LabelWriter>> dwords, std::uint32_t ownStream, std::uint64_t afterStamp, const LabelAccepts& accepts) {
    if (dwords.empty()) return std::nullopt;
    std::optional<LabelHit> best;
    bool bestOwn = false;
    for (const auto& first : dwords.front()) {
        const auto stream = first.stream;
        LabelHit hit{0, first.queue, 0, stream, 0};
        bool complete = true;
        for (std::size_t index = 0; index < dwords.size() && complete; ++index) {
            const auto found = std::find_if(dwords[index].begin(), dwords[index].end(), [&](const LabelWriter& writer) { return writer.stream == stream; });
            complete = found != dwords[index].end() && found->stamp > afterStamp;
            if (!complete) break;
            hit.value |= static_cast<std::uint64_t>(found->value) << (index * 32u);
            hit.stamp = std::max(hit.stamp, found->stamp);
            hit.batch = std::max(hit.batch, found->batch);
        }
        if (!complete || !accepts(hit.value)) continue;
        const bool own = stream == ownStream;
        if (!best.has_value() || (own && !bestOwn) || (own == bestOwn && hit.stamp < best->stamp)) {
            best = hit;
            bestOwn = own;
        }
    }
    return best;
}

std::optional<Recorder::LabelHit> Recorder::lookupLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, const LabelAccepts& accepts, bool* behindCompletion) const {
    if (behindCompletion != nullptr) *behindCompletion = false;
    if ((labels.empty() && queuedLabels.empty()) || (bytes != 4 && bytes != 8) || address % 4 != 0) return std::nullopt;
    const auto tag = GuestMemory::GpuLockThreadTag();
    const auto known = knownStreamOfCaller();
    const auto own = known != NoStream ? known : static_cast<std::uint32_t>(MaxRecorderStreams);
    std::array<std::vector<LabelWriter>, 2> dwords;
    for (std::size_t index = 0; index * 4 < bytes; ++index) {
        const auto dword = address + index * 4;
        auto& writers = dwords[index];
        if (const auto found = labels.find(dword); found != labels.end()) {
            for (const auto& writer : found->second) {
                if (behindCompletion != nullptr && writer.behindCompletion) *behindCompletion = true;
                writers.push_back(writer);
            }
        }
        if (const auto queued = queuedLabels.find(dword); queued != queuedLabels.end() && queued->second.queue == tag) {
            std::erase_if(writers, [&](const LabelWriter& writer) { return writer.stream == own; });
            auto entry = queued->second;
            entry.stream = own;
            entry.batch = 0;
            writers.push_back(entry);
        }
    }
    const auto hit = SelectLabelWriter(std::span<const std::vector<LabelWriter>>(dwords).first(bytes / 4), own, afterStamp, accepts);
    if (hit.has_value() && hit->batch == 0) queuedLabelHits.fetch_add(1, std::memory_order_relaxed);
    return hit;
}

bool Recorder::unsignaled(std::uint64_t serial) const {
    const std::lock_guard recordingGuard(recording);
    for (const auto& batch : inFlight) {
        if (batch->serial == serial) return fenceStatus(batch->fence) != VK_SUCCESS;
    }
    return false;
}

Recorder::Batch* Recorder::callerNewest() {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (open != nullptr) return open.get();
    const auto stream = streamOfCaller();
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
        if ((*it)->stream == stream) return it->get();
    }
    return nullptr;
}

void Recorder::AfterCompletions(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue, bool storedOnGpu) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    if (streams.size() > 1) {
        DeclareSection({});
        beginSection();
        const auto stream = streamOfCaller();
        const auto end = address + bytes.size();
        Epoch after{};
        for (std::uint32_t other = 0; other < streams.size(); ++other) {
            if (other != stream && opens[other] != nullptr && overlaps(*opens[other], address, end)) after[other] = opens[other]->value;
        }
        for (const auto& batch : inFlight) {
            if (batch->stream != stream && overlaps(*batch, address, end)) after[batch->stream] = std::max(after[batch->stream], batch->value);
        }
        dependOn(after);
    }
    Require(open != nullptr || !inFlight.empty(), "no batch to append a completion label to");
    Batch& batch = open != nullptr ? *open : *inFlight.back();
    std::vector<std::byte> copy(bytes.begin(), bytes.end());
    // The store bypasses the flush hook: it runs inside finish() under the mutex, and the hook would
    // find this very range noted (or a later label to it) and sync re-entrantly. Stamped like every
    // driver store so a collect memoized for the packet sees it.
    static const bool always = std::getenv("APS5_LABEL_STORE_ALWAYS") != nullptr;
    const auto sequence = [&] {
        std::lock_guard ringLock(writtenBackMutex);
        return writtenBackSequence;
    }();
    batch.completions.push_back([this, address, copy = std::move(copy), sequence, storedOnGpu] {
        if (storedOnGpu && !always && !writtenBackSince(sequence, address, address + copy.size())) {
            completionStoresSkipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        completionStoresRun.fetch_add(1, std::memory_order_relaxed);
        GuestMemory::CheckRange(reinterpret_cast<const void*>(address), copy.size(), 4, true);
        GuestMemory::StoreOwnBytes(address, copy);
    });
    // Counted per batch and counted down when the batch finishes, on every exit path of finish()
    // (a label whose range became unmapped is reported there, and a lost device throws before the
    // completions run): a label that will never land must not leave the workers reaping forever.
    // One stored on the GPU already needs no reap unless a write-back overlaps it (the store above
    // is skipped otherwise), so it enters the count at that write-back (NoteWrittenBack).
    const bool behindCompletion = !(storedOnGpu && !always && !CountAllCompletionLabels());
    if (!behindCompletion) {
        batch.completionLabelRanges.push_back({address, address + bytes.size(), false});
    } else {
        ++batch.completionLabelCount;
        completionLabels.fetch_add(1, std::memory_order_acq_rel);
    }
    // The table entry before the write note: the note bumps the generation a poller watches, and a
    // poller that sees the bump then finds the label without the GPU mutex.
    noteLabelOn(batch, address, bytes, stamp, queue, behindCompletion);
    noteWriteOn(batch, address, bytes.size(), true);
    if (&batch == open.get()) startLabelDeadline(batch.stream);
}

bool Recorder::AfterRecordedWork(std::function<void()> action) {
    const std::lock_guard recordingGuard(recording);
    auto& open = callerOpen();
    Batch* newest = callerNewest();
    if (newest == nullptr) return false;
    Batch& batch = *newest;
    batch.completions.push_back(std::move(action));
    ++batch.completionLabelCount;
    completionLabels.fetch_add(1, std::memory_order_acq_rel);
    if (&batch == open.get()) startLabelDeadline(batch.stream);
    return true;
}

void Recorder::NoteWrittenBack(std::uint64_t address, std::size_t bytes) {
    Recorder* recorder = activeRecorder;
    if (recorder == nullptr || bytes == 0) return;
    GuestMemory::AssertGpuLockHeld("Recorder::NoteWrittenBack");
    const std::lock_guard recordingGuard(recorder->recording);
    const auto end = address + bytes;
    {
        std::lock_guard ringLock(recorder->writtenBackMutex);
        recorder->writtenBack.push_back({++recorder->writtenBackSequence, address, end});
        while (recorder->writtenBack.size() > 16384) recorder->writtenBack.pop_front();
    }
    if (recorder->recordedLabels.load(std::memory_order_relaxed) != 0 && recorder->PendingLabelIn(address, bytes)) writeBacksOverLabels.fetch_add(1, std::memory_order_relaxed);
    // The GPU-stored completion labels this write-back overlaps now need their completion store,
    // so they enter the pending count (once each; finish() subtracts the batch's count). Batches
    // in `finishing` run their own completions in the same finish(), so only open and in-flight
    // ones are scanned.
    const auto count = [&](Batch& batch) {
        for (auto& label : batch.completionLabelRanges) {
            if (label.counted || label.begin >= end || address >= label.end) continue;
            label.counted = true;
            ++batch.completionLabelCount;
            completionLabels.fetch_add(1, std::memory_order_acq_rel);
            completionLabelsCountedLate.fetch_add(1, std::memory_order_relaxed);
            recorder->markBehindCompletion(batch, label.begin, label.end);
        }
    };
    for (std::size_t stream = 0; stream < recorder->streams.size(); ++stream) {
        if (recorder->opens[stream] != nullptr) count(*recorder->opens[stream]);
    }
    for (auto& batch : recorder->inFlight) count(*batch);
}

bool Recorder::writtenBackSince(std::uint64_t sequence, std::uint64_t begin, std::uint64_t end) {
    const std::lock_guard recordingGuard(recording);
    std::lock_guard ringLock(writtenBackMutex);
    if (writtenBack.empty() || writtenBack.back()[0] <= sequence) return false;
    if (writtenBack.front()[0] > sequence + 1) return true;
    for (auto it = writtenBack.rbegin(); it != writtenBack.rend() && (*it)[0] > sequence; ++it) {
        if ((*it)[1] < end && begin < (*it)[2]) return true;
    }
    return false;
}

void Recorder::Submit() {
    const std::lock_guard recordingGuard(recording);
    PerformanceTimer frameTiming("Recorder.Submit");
    // The work count is cleared even when nothing is open: the driver counts a dispatch after its
    // call returns (outside the mutex), so a submit by another thread in between leaves a stale
    // count behind, and the callers that act on it would otherwise take the mutex for nothing at
    // every packet or submission end until a real batch is next submitted.
    if (activeRecorder == this) workSinceSubmit.store(0, std::memory_order_relaxed);
    dropStaleHoldKept();
    for (std::uint32_t stream = 0; stream < streams.size(); ++stream) submitStream(stream);
}

void Recorder::SubmitLabelled() {
    const std::lock_guard recordingGuard(recording);
    PerformanceTimer frameTiming("Recorder.Submit");
    dropStaleHoldKept();
    for (std::uint32_t stream = 0; stream < streams.size(); ++stream) {
        if (labelSince[stream] != NoPendingLabel) submitStream(stream);
    }
    if (activeRecorder == this && !anyOpen()) workSinceSubmit.store(0, std::memory_order_relaxed);
}

void Recorder::SubmitCaller() {
    const std::lock_guard recordingGuard(recording);
    PerformanceTimer frameTiming("Recorder.Submit");
    dropStaleHoldKept();
    submitStream(streamOfCaller());
    if (activeRecorder == this && !anyOpen()) workSinceSubmit.store(0, std::memory_order_relaxed);
}

void Recorder::submitStream(std::uint32_t stream) {
    const std::lock_guard recordingGuard(recording);
    if (opens[stream] == nullptr) return;
    if (!ownStreamRecording) GuestMemory::AssertGpuLockHeld("Recorder::Submit");
    for (std::uint32_t other = 0; other < streams.size() && opens[stream] != nullptr; ++other) {
        if (other == stream || opens[stream]->waits[other] <= streams[other].submitted || opens[other] == nullptr) continue;
        Require(!ownStreamRecording, "a batch recorded without the device lock was submitted before the batch it waits for");
        submitStream(other);
    }
    if (opens[stream] == nullptr) return;
    struct Submitting {
        std::uint32_t& slot;
        std::uint32_t previous;
        ~Submitting() { slot = previous; }
    } submitting{submittingStream, std::exchange(submittingStream, stream)};
    if (std::exchange(opens[stream]->signalPublicationOwed, false)) {
        PublishForCompletionSignal();
        if (opens[stream] == nullptr) return;
    }
    auto& open = opens[stream];
    if (!open->keyStores.empty()) recordKeyStores(false);
    if (open->renderPass.open) endOpenRenderPass();
    const bool segments = segmentMode();
    if (!segments) {
        endSamples(*open);
        if (stream == 0 && pendingSamples.size() >= SampleFoldCapacity) foldSamples(*open, 0);
    }
    const bool hostReadCovered = (open->run.open || !open->run.queued.empty()) && closeStoreRun(true);
    if (!open->recordCopies.empty()) {
        recordBarrier(open->commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        for (const auto& copy : open->recordCopies) {
            const VkBufferCopy region{copy.offset, copy.hostOffset, copy.bytes};
            context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(open->commands, copy.source, copy.host, 1, &region);
        }
        recordBarrier(open->commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        open->recordCopies.clear();
    }
    if (segments && stream == 0 && pendingSamples.size() >= SampleFoldCapacity) foldSegments();
    if (BarrierValidate()) validateBatchEnds.fetch_add(1, std::memory_order_relaxed);
    if (open->hostReadOwed && !hostReadCovered) {
        // A fence makes nothing visible to the host: a lean draw's writes (no download barrier of
        // its own) get the batch's one here.
        recordBarrier(open->commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        CountBarriers(CommandClass::Draw);
    }
    EndGpuTiming(open->batchTiming);
    if (!GpuTimingEnabled()) reportBarriers();
    if (FlipReadCheck()) {
        // A CPU write into a noted read after this collect (and so possibly before the GPU read
        // it) fails UnchangedSince at the presenter's check (VulkanDevice RetirePresents).
        for (const auto& read : open->reads) open->readGeneration = std::max(open->readGeneration, GuestMemory::CollectWrites(read.begin, read.end - read.begin));
    }
    auto batch = std::move(open);
    struct DiscardUnsubmitted {
        Recorder& recorder;
        std::unique_ptr<Batch>& batch;
        ~DiscardUnsubmitted() {
            if (batch != nullptr && !batch->submitted) recorder.discard(std::move(batch));
        }
    } discardUnsubmitted{*this, batch};
    Check(function(endCommandBuffer, "vkEndCommandBuffer")(batch->commands), "vkEndCommandBuffer recorder");
    VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &batch->commands;
    const std::uint64_t serial = submissions + 1;
    CaptureTrace::Log("submit batch=%llu reads=%zu writes=%zu", static_cast<unsigned long long>(serial), batch->reads.size(), batch->writes.size());
    auto& target = streams[stream];
    std::array<VkSemaphore, MaxRecorderStreams> waitSemaphores{};
    std::array<std::uint64_t, MaxRecorderStreams> waitValues{};
    std::array<VkPipelineStageFlags, MaxRecorderStreams> waitStages{};
    std::uint32_t waits = 0;
    for (std::uint32_t other = 0; other < streams.size(); ++other) {
        if (other == stream || batch->waits[other] <= streams[other].finished) continue;
        waitSemaphores[waits] = streams[other].timeline;
        waitValues[waits] = batch->waits[other];
        waitStages[waits] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        ++waits;
    }
    VkTimelineSemaphoreSubmitInfoKHR timelineInfo{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR};
    timelineInfo.waitSemaphoreValueCount = waits;
    timelineInfo.pWaitSemaphoreValues = waitValues.data();
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &batch->value;
    if (target.timeline != VK_NULL_HANDLE) {
        submission.pNext = &timelineInfo;
        submission.waitSemaphoreCount = waits;
        submission.pWaitSemaphores = waitSemaphores.data();
        submission.pWaitDstStageMask = waitStages.data();
        submission.signalSemaphoreCount = 1;
        submission.pSignalSemaphores = &target.timeline;
    }
    if (!ownStreamRecording) PooledView::Settle();
    const auto submitStart = std::chrono::steady_clock::now();
    const auto result = function(queueSubmit, "vkQueueSubmit")(target.queue, 1, &submission, batch->fence);
    if (result == VK_ERROR_DEVICE_LOST) {
        const auto idle = context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(context.device);
        if (idle != VK_SUCCESS && idle != VK_ERROR_DEVICE_LOST) std::terminate();
    }
    Check(result, "vkQueueSubmit recorder");
    batch->submitted = true;
    batch->serial = ++submissions;
    batch->submittedAt = std::chrono::steady_clock::now();
    target.submitted = batch->value;
    NotifyOrderWaiters();
    if (DrawProfiled()) {
        const auto us = std::chrono::duration<double, std::micro>(batch->submittedAt - submitStart).count();
        ++submitCount;
        submitUs += us;
        submitMaxUs = std::max(submitMaxUs, us);
        if (auto* frame = PerformanceContext::Current()) frame->Add(frame->Get("Recorder", "submit"), batch->submittedAt - submitStart);
    }
    {
        std::lock_guard lock(completedMutex);
        newestSubmitted = batch->serial;
        newestSubmittedAt = batch->submittedAt;
        auto& values = submittedValues[batch->serial % submittedValues.size()];
        values.serial = batch->serial;
        for (std::uint32_t other = 0; other < streams.size(); ++other) values.values[other] = streams[other].submitted;
    }
    inFlight.push_back(std::move(batch));
    inFlightKeptBytes += inFlight.back()->keptBytes;
    target.newest = inFlight.back().get();
    endLabelDeadline(stream);
}

void Recorder::startLabelDeadline(std::uint32_t stream) {
    const std::lock_guard recordingGuard(recording);
    if (activeRecorder != this || labelSince[stream] != NoPendingLabel) return;
    labelSince[stream] = std::chrono::steady_clock::now().time_since_epoch().count();
    if (pendingLabelSince.load(std::memory_order_relaxed) == NoPendingLabel) pendingLabelSince.store(labelSince[stream], std::memory_order_release);
}

void Recorder::endLabelDeadline(std::uint32_t stream) {
    const std::lock_guard recordingGuard(recording);
    if (activeRecorder != this || labelSince[stream] == NoPendingLabel) return;
    labelSince[stream] = NoPendingLabel;
    std::int64_t oldest = NoPendingLabel;
    for (std::uint32_t other = 0; other < streams.size(); ++other) {
        if (labelSince[other] != NoPendingLabel && (oldest == NoPendingLabel || labelSince[other] < oldest)) oldest = labelSince[other];
    }
    pendingLabelSince.store(oldest, std::memory_order_release);
}

std::uint64_t Recorder::SubmitAndEpoch() {
    const std::lock_guard recordingGuard(recording);
    Submit();
    return inFlight.empty() ? 0 : submissions;
}

namespace {

// One thread's registration as a timeline waiter (see `unlockedWaiters`): constructed while the
// caller still holds the mutex (or the device), so a teardown that starts after the mutex is
// released sees it and keeps the semaphore until the wait returned.
struct UnlockedWaiter {
    std::atomic<int>& count;
    explicit UnlockedWaiter(std::uint64_t recorderId) : count(WaitersOf(recorderId)) { count.fetch_add(1, std::memory_order_acq_rel); }
    ~UnlockedWaiter() { count.fetch_sub(1, std::memory_order_acq_rel); }
    UnlockedWaiter(const UnlockedWaiter&) = delete;
    UnlockedWaiter& operator=(const UnlockedWaiter&) = delete;
};

// Restores the thread's active sync-site entry on every exit path (a Check throw on a lost device
// included), so the next sync on the thread is not attributed to this one's site.
struct RestoreSyncSite {
    std::size_t site;
    ~RestoreSyncSite() { activeSyncSite = site; }
};

// The timeline wait itself, on handles copied out of the recorder: the caller holds no mutex, and
// the recorder may be torn down meanwhile (its destructor waits for the registered waiters to
// leave after completing every batch, so the handles stay valid until this returns).
struct TimelinePoints {
    std::array<VkSemaphore, MaxRecorderStreams> semaphores{};
    std::array<std::uint64_t, MaxRecorderStreams> values{};
    std::uint32_t count = 0;
    void Add(VkSemaphore semaphore, std::uint64_t value) {
        if (semaphore == VK_NULL_HANDLE || value == 0) return;
        semaphores[count] = semaphore;
        values[count] = value;
        ++count;
    }
};

VkResult WaitTimeline(VkDevice device, const TimelinePoints& points, PFN_vkWaitSemaphoresKHR waitSemaphores, std::uint64_t serial) {
    if (points.count == 0) return VK_SUCCESS;
    VkSemaphoreWaitInfoKHR wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR};
    wait.semaphoreCount = points.count;
    wait.pSemaphores = points.semaphores.data();
    wait.pValues = points.values.data();
    auto result = waitSemaphores(device, &wait, 5'000'000'000ull);
    for (int waited = 5; result == VK_TIMEOUT; waited += 5) {
        // As for the fence wait in finish(): a hung batch is reported every 5 s.
        std::fprintf(stderr, "[gpu] recorded batch %llu still running on the GPU after %d s (timeline wait)\n", static_cast<unsigned long long>(serial), waited);
        result = waitSemaphores(device, &wait, 5'000'000'000ull);
    }
    return result;
}

}

std::vector<Recorder::QueuePoint> Recorder::OtherQueuePoints() const {
    const std::lock_guard recordingGuard(recording);
    std::vector<QueuePoint> points;
    for (std::uint32_t stream = 1; stream < streams.size(); ++stream) {
        if (streams[stream].submitted > streams[stream].finished) points.push_back({streams[stream].timeline, streams[stream].submitted});
    }
    return points;
}

void Recorder::WaitSerial(std::uint64_t serial) {
    PerformanceTimer frameTiming("Recorder.WaitSerial");
    if (!HasTimeline() || serial == 0) return;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const UnlockedWaiter waiter{id};
    TimelinePoints points;
    {
        std::lock_guard lock(completedMutex);
        const auto* values = &submittedValues[serial % submittedValues.size()];
        if (values->serial < serial) values = &submittedValues[newestSubmitted % submittedValues.size()];
        for (std::uint32_t stream = 0; stream < streams.size(); ++stream) points.Add(streams[stream].timeline, values->values[stream]);
    }
    const auto result = WaitTimeline(context.device, points, context.Function<PFN_vkWaitSemaphoresKHR>("vkWaitSemaphoresKHR"), serial);
    if (profile) {
        unlockedWaits.fetch_add(1, std::memory_order_relaxed);
        unlockedWaitedUs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
    }
    Check(result, "vkWaitSemaphoresKHR recorder");
}

void Recorder::FinishUpTo(std::uint64_t serial) {
    const std::lock_guard recordingGuard(recording);
    // A drain: the wait already happened (unlocked), the fences are signaled, so the source-0 wait
    // recorded here is the bookkeeping only (a caller that did not wait first, WaitForLeases, shows
    // up in the site table by its own address).
    ++syncCounts[0];
    announcedSource = 4;
    const void* site = std::exchange(announcedSite, nullptr);
    const auto previousSite = inFlight.empty() || inFlight.front()->serial > serial ? activeSyncSite : BeginSyncSite(0, site != nullptr ? site : __builtin_return_address(0));
    while (!inFlight.empty() && inFlight.front()->serial <= serial) {
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, 0);
    }
    activeSyncSite = previousSite;
}

void Recorder::WaitToken(const Token& token) {
    const std::lock_guard recordingGuard(recording);
    PerformanceTimer frameTiming("Recorder.WaitToken");
    TimelinePoints points;
    {
        std::lock_guard lock(completedMutex);
        const auto& submitted = submittedValues[newestSubmitted % submittedValues.size()];
        for (std::uint32_t stream = 0; stream < streams.size(); ++stream) {
            if (token[stream] == 0) continue;
            Require(streams[stream].timeline != VK_NULL_HANDLE && submitted.serial == newestSubmitted && token[stream] <= submitted.values[stream], "a token wait targets work that was not submitted");
            points.Add(streams[stream].timeline, token[stream]);
        }
    }
    const UnlockedWaiter waiter{id};
    const RecordingRelease release(recording);
    Check(WaitTimeline(context.device, points, context.Function<PFN_vkWaitSemaphoresKHR>("vkWaitSemaphoresKHR"), 0), "vkWaitSemaphoresKHR recorder token");
}

bool Recorder::WaitTokenUnlocked(const Token& token) {
    RecordingHold recordingHold(recording);
    auto& mutex = GuestMemory::GpuMutex();
    Require(mutex.DepthOnThisThread() == 1, "an unlocked token wait needs the device lock held once by its caller");
    TimelinePoints points;
    {
        std::lock_guard lock(completedMutex);
        const auto& submitted = submittedValues[newestSubmitted % submittedValues.size()];
        for (std::uint32_t stream = 0; stream < streams.size(); ++stream) {
            if (token[stream] == 0 || token[stream] <= streams[stream].finished) continue;
            Require(streams[stream].timeline != VK_NULL_HANDLE && submitted.serial == newestSubmitted && token[stream] <= submitted.values[stream], "a token wait targets work that was not submitted");
            points.Add(streams[stream].timeline, token[stream]);
        }
    }
    if (points.count == 0) {
        FinishToken(token);
        return true;
    }
    const auto myId = id;
    const auto device = context.device;
    const auto waitSemaphores = context.Function<PFN_vkWaitSemaphoresKHR>("vkWaitSemaphoresKHR");
    VkResult result = VK_SUCCESS;
    std::uint32_t recordingHeld = 0;
    {
        const UnlockedWaiter waiter{myId};
        recordingHeld = recording.ReleaseAll();
        mutex.unlock();
        result = WaitTimeline(device, points, waitSemaphores, 0);
    }
    mutex.lock();
    if (!RecorderAlive(myId)) {
        recordingHold.Abandon();
        Check(result, "vkWaitSemaphoresKHR recorder token");
        return false;
    }
    recording.Reacquire(recordingHeld);
    Check(result, "vkWaitSemaphoresKHR recorder token");
    FinishToken(token);
    return true;
}

void Recorder::FinishToken(const Token& token) {
    const std::lock_guard recordingGuard(recording);
    WaitToken(token);
    Reap();
    while (!Reached(token)) {
        Require(!inFlight.empty(), "a waited token's batches are neither in flight nor finished");
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, 4);
    }
}

void Recorder::Sync() {
    const std::lock_guard recordingGuard(recording);
    PerformanceTimer frameTiming("Recorder.Sync");
    const auto source = std::exchange(announcedSource, 4);
    const void* site = std::exchange(announcedSite, nullptr);
    Submit();
    // Counted only when something is waited for: an idle recorder's Sync is free and would only
    // swamp the site table with the callers that check nothing first.
    const auto previousSite = inFlight.empty() ? activeSyncSite : BeginSyncSite(source, site != nullptr ? site : __builtin_return_address(0));
    while (!inFlight.empty()) {
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, source);
    }
    activeSyncSite = previousSite;
}

void Recorder::FlushDeferredReleases() {
    const std::lock_guard recordingGuard(recording);
    holdKept.clear();
    ReleaseDeferredKeeps();
}

void Recorder::SyncThrough(std::uint64_t address, std::size_t bytes, bool waitUnlocked) {
    const std::lock_guard recordingGuard(recording);
    if (bytes == 0) return;
    const auto end = address + bytes;
    if (completionDepth != 0 && !CompletionStoreSyncs()) {
        // A store made by a completion action (a copied buffer's write-back, GuestBufferMemory::
        // WriteBack) reached the flush hook, and a batch still in flight notes the range. Every
        // batch in flight now was recorded after the completing one (finish runs front to back),
        // so in program order this store precedes their writes: landing it before them is the
        // order the hardware gives; waiting for them first (what the hook would do) lands the
        // completing batch's bytes over the later batch's, which is wrong, and does so under the
        // hold of whoever reaped (the [hooksync] 'store' entries, up to tens of ms per reap). The
        // later batch's own ordering against this write-back is its recorder's business at record
        // time (FillBuffer and DispatchIndirect consult copiedWriters; a later copied writer's
        // write-back runs after this one in finish order; a label behind completions lands after).
        announcedSource = 4;
        announcedSite = nullptr;
        ++holdCounters.completionSyncsSkipped;
        return;
    }
    if (completionDepth != 0) ++holdCounters.completionSyncsWaited;
    const auto completionWaitStart = completionDepth != 0 && DrawProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    struct CompletionWait {
        std::chrono::steady_clock::time_point start;
        ~CompletionWait() {
            if (start != std::chrono::steady_clock::time_point{}) holdCounters.completionSyncWaitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    } completionWait{completionWaitStart};
    if (waitUnlocked) {
        // The hook's wait without the mutex: its own acquisition is the outermost on this thread.
        const auto source = announcedSource;
        const void* site = announcedSite != nullptr ? announcedSite : __builtin_return_address(0);
        if (syncThroughUnlocked(address, end, source, site)) return;
        // A completion-store wait (kill switch above) is already counted as 'waited'.
        if (completionDepth == 0) ++holdCounters.hookLockedWaits;
    }
    if (!SyncThroughEnabled()) {
        // The announced site (or this call's caller) names the sync, not this function.
        if (announcedSite == nullptr) announcedSite = __builtin_return_address(0);
        // The open batch's work has not run; without one the newest in flight is the target.
        if (anyOpen() || (!inFlight.empty() && unsignaled(inFlight.back()->serial))) ++hookRealWaits;
        Sync();
        return;
    }
    const auto source = std::exchange(announcedSource, 4);
    const void* site = std::exchange(announcedSite, nullptr);
    auto epoch = syncTargets(address, end);
    if (std::all_of(epoch.begin(), epoch.end(), [](std::uint64_t value) { return value == 0; })) return;
    closeEpoch(epoch);
    if (epochUnsignaled(epoch)) ++hookRealWaits;
    targetedSyncs.fetch_add(1, std::memory_order_relaxed);
    const RestoreSyncSite restoreSite{BeginSyncSite(source, site != nullptr ? site : __builtin_return_address(0))};
    finishEpoch(epoch, source);
    batchesLeftInFlight.fetch_add(inFlight.size(), std::memory_order_relaxed);
}

namespace {

// What the unlocked hook waits waited for, per read site (the [syncwait] line every 10 s under
// APS5_PROFILE_DRAW): the batches up to the target, how many of them the GPU had not finished when
// the wait began, and whether the target was the newest submission. APS5_TRACE_CAPSYNC=1 prints
// every such wait ([capsync] sync lines).
struct SyncWaitSite {
    std::uint64_t count = 0;
    std::uint64_t batchesToTarget = 0;
    std::uint64_t unsignaledAtStart = 0;
    std::uint64_t newest = 0;
    double waitedMs = 0;
    // The target batch's submit -> wait-end latency: what the wait's length is made of when the
    // batches ahead of it carry little GPU work (submission latency, not throughput).
    double targetAgeMs = 0;
};

struct SyncWaitStats {
    std::array<SyncWaitSite, static_cast<std::size_t>(GuestMemory::ReadSite::Count)> bySite{};
    // Batches up to the targets, by the queue that opened them: whether a wait for one queue's
    // producer is really a wait for the other queues' batches ahead of it on the one timeline.
    std::map<std::uint32_t, std::uint64_t> aheadByQueue;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

SyncWaitStats& SyncWaits() {
    static SyncWaitStats stats;
    return stats;
}

bool TraceCapSync() {
    static const bool trace = std::getenv("APS5_TRACE_CAPSYNC") != nullptr;
    return trace;
}

void ReportSyncWaits(SyncWaitStats& stats) {
    std::string report = "[syncwait] unlocked hook waits by read site (10 s; count/GPU wait, avg batches up to the target, avg of them unsignaled at the start, target was the newest submission, avg target submit-to-signal ms):";
    for (std::size_t site = 0; site < stats.bySite.size(); ++site) {
        const auto& entry = stats.bySite[site];
        if (entry.count == 0) continue;
        char text[160];
        std::snprintf(text, sizeof(text), " %s %llu/%.0fms %.1f %.1f %.0f%% %.1f", GuestMemory::ReadSiteName(static_cast<GuestMemory::ReadSite>(site)), static_cast<unsigned long long>(entry.count), entry.waitedMs, static_cast<double>(entry.batchesToTarget) / entry.count, static_cast<double>(entry.unsignaledAtStart) / entry.count, entry.newest * 100.0 / entry.count, entry.targetAgeMs / entry.count);
        report += text;
    }
    report += "; batches ahead by queue:";
    for (const auto& [queue, count] : stats.aheadByQueue) {
        char text[48];
        if (queue == 0xffffffffu) std::snprintf(text, sizeof(text), " untagged %llu", static_cast<unsigned long long>(count));
        else std::snprintf(text, sizeof(text), " 0x%x %llu", queue, static_cast<unsigned long long>(count));
        report += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", report.c_str());
    stats = SyncWaitStats{};
}

}

bool Recorder::syncThroughUnlocked(std::uint64_t address, std::uint64_t end, int source, const void* site) {
    RecordingHold recordingHold(recording);
    // Only for the outermost acquisition: a nested hook (inside a dispatch's hold) must not give
    // up the caller's mutex, and a thread that waits here holds nothing the relock could invert
    // (the hook took the mutex from a depth of 0 with the same locks held).
    if (HookLockedWait() || !HasTimeline() || GuestMemory::GpuMutex().DepthOnThisThread() != 1) return false;
    Epoch epoch{};
    if (!SyncThroughEnabled()) {
        Submit();
        for (std::uint32_t stream = 0; stream < streams.size(); ++stream) epoch[stream] = streams[stream].submitted;
    } else {
        epoch = syncTargets(address, end);
    }
    announcedSource = 4;
    announcedSite = nullptr;
    closeEpoch(epoch);
    std::uint64_t targetSerial = 0;
    for (const auto& batch : inFlight) {
        if (batch->value <= epoch[batch->stream]) targetSerial = std::max(targetSerial, batch->serial);
    }
    if (targetSerial == 0) return true;
    if (epochUnsignaled(epoch)) ++hookRealWaits;
    // Everything the wait needs is copied out: the recorder may be torn down while the mutex is
    // released (its destructor completes every batch and waits for the semaphore to be left, see
    // WaitTimeline), after which `this` must not be touched. The id re-check below relies on the
    // teardown running under the GpuMutex (the driver's drain and failure paths reset the device
    // under it; a worker dropping its last device reference outside its lock would not).
    const auto myId = id;
    const auto device = context.device;
    TimelinePoints points;
    for (std::uint32_t stream = 0; stream < streams.size(); ++stream) {
        if (epoch[stream] > streams[stream].finished) points.Add(streams[stream].timeline, epoch[stream]);
    }
    const auto waitSemaphores = context.Function<PFN_vkWaitSemaphoresKHR>("vkWaitSemaphoresKHR");
    // The site entry is taken under the mutex (the table is only touched under it; entries are
    // appended, never removed, so the index stays valid across the release).
    const RestoreSyncSite restoreSite{BeginSyncSite(source, site)};
    const bool profile = DrawProfiled();
    const bool trace = TraceCapSync();
    const auto readSite = GuestMemory::CurrentReadSite();
    const auto newestSerial = submissions;
    std::size_t batchesToTarget = 0, unsignaledAtStart = 0, targetRanges = 0;
    std::pair<std::uint64_t, std::uint64_t> hit{0, 0};
    std::chrono::steady_clock::time_point targetSubmittedAt{};
    std::map<std::uint32_t, std::uint64_t> aheadByQueue;
    if (profile || trace) {
        const auto status = context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus");
        for (const auto& batch : inFlight) {
            if (batch->value > epoch[batch->stream]) continue;
            ++batchesToTarget;
            ++aheadByQueue[batch->queue];
            if (status(context.device, batch->fence) != VK_SUCCESS) ++unsignaledAtStart;
            if (batch->serial != targetSerial) continue;
            targetSubmittedAt = batch->submittedAt;
            targetRanges = batch->writes.size();
            for (const auto& range : batch->writes) {
                if (address < range.second && range.first < end) {
                    hit = range;
                    break;
                }
            }
        }
    }
    const auto start = profile || trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto waited = start;
    auto& mutex = GuestMemory::GpuMutex();
    VkResult result = VK_SUCCESS;
    std::uint32_t recordingHeld = 0;
    {
        // Registered before the release: a teardown can only start once the mutex is free.
        const UnlockedWaiter waiter{myId};
        recordingHeld = recording.ReleaseAll();
        mutex.unlock();
        result = WaitTimeline(device, points, waitSemaphores, targetSerial);
        // Read before the relock: the GPU wait, not the wait for the mutex behind it.
        if (profile || trace) waited = std::chrono::steady_clock::now();
    }
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
    mutex.lock();
    if (RecorderAlive(myId)) recording.Reacquire(recordingHeld);
    else recordingHold.Abandon();
    const auto ms = std::chrono::duration<double, std::milli>(waited - start).count();
    if (profile) {
        // The GPU wait is charged like a locked wait, so the per-source, per-thread and per-site
        // totals of the [recorder] line keep their meaning; the unlocked count says how many were
        // not held. The relock's wait for the mutex is kept apart (the [lock] line also counts it
        // under 'hook' waits, for a cross-check).
        syncWaitedMs[source] += ms;
        CountThreadSync(source, ms);
        CountSiteWait(ms);
        ++holdCounters.hookUnlockedWaits;
        holdCounters.hookUnlockedWaitMs += ms;
        if (source >= 0 && source < 5) {
            ++holdCounters.hookUnlockedWaitsBySource[source];
            holdCounters.hookUnlockedWaitMsBySource[source] += ms;
        }
        holdCounters.hookRelockMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waited).count();
        auto& stats = SyncWaits();
        auto& siteStats = stats.bySite[static_cast<std::size_t>(readSite)];
        ++siteStats.count;
        siteStats.waitedMs += ms;
        siteStats.batchesToTarget += batchesToTarget;
        siteStats.unsignaledAtStart += unsignaledAtStart;
        if (targetSerial == newestSerial) ++siteStats.newest;
        if (targetSubmittedAt != std::chrono::steady_clock::time_point{}) siteStats.targetAgeMs += std::chrono::duration<double, std::milli>(waited - targetSubmittedAt).count();
        for (const auto& [queue, count] : aheadByQueue) stats.aheadByQueue[queue] += count;
        if (std::chrono::steady_clock::now() - stats.lastReport > std::chrono::seconds(10)) ReportSyncWaits(stats);
    }
    if (trace) {
        static std::atomic<int> lines{0};
        if (lines.fetch_add(1) < 600) {
            const auto packet = GuestMemory::CurrentPacket();
            std::fprintf(stderr, "[capsync] sync q0x%x %s %s 0x%llx+0x%llx: target %llu (newest %llu), %zu batches up to it (%zu unsignaled at the start), target noted %zu ranges, hit 0x%llx+0x%llx, GPU wait %.1f ms\n", packet.queue, PacketName(packet.opcode).c_str(), GuestMemory::ReadSiteName(readSite), static_cast<unsigned long long>(address), static_cast<unsigned long long>(end - address), static_cast<unsigned long long>(targetSerial), static_cast<unsigned long long>(newestSerial), batchesToTarget, unsignaledAtStart, targetRanges, static_cast<unsigned long long>(hit.first), static_cast<unsigned long long>(hit.second - hit.first), ms);
        }
    }
    if (!RecorderAlive(myId)) {
        // Torn down meanwhile: its destructor's Sync() ran every completion, nothing is pending.
        ++holdCounters.hookUnlockedTornDown;
        Check(result, "vkWaitSemaphoresKHR recorder");
        return true;
    }
    Check(result, "vkWaitSemaphoresKHR recorder");
    // The completions up to the target, under the mutex again; the fences are signaled, so the
    // waits inside finish() return at once (charged to the same site entry). Another thread may
    // have finished some meanwhile.
    targetedSyncs.fetch_add(1, std::memory_order_relaxed);
    finishEpoch(epoch, source);
    batchesLeftInFlight.fetch_add(inFlight.size(), std::memory_order_relaxed);
    return true;
}

bool Recorder::Reap() {
    const std::lock_guard recordingGuard(recording);
    PerformanceTimer frameTiming("Recorder.Reap");
    dropStaleHoldKept();
    const bool profile = DrawProfiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::uint64_t retired = 0;
    std::array<bool, MaxRecorderStreams> blocked{};
    std::size_t blockedStreams = 0;
    for (auto it = inFlight.begin(); it != inFlight.end() && blockedStreams < streams.size();) {
        const auto& batch = **it;
        if (blocked[batch.stream]) {
            ++it;
            continue;
        }
        bool ready = true;
        for (std::uint32_t other = 0; other < streams.size() && ready; ++other) ready = other == batch.stream || batch.waits[other] <= streams[other].finished;
        if (!ready || fenceStatus(batch.fence) != VK_SUCCESS) {
            blocked[batch.stream] = true;
            ++blockedStreams;
            ++it;
            continue;
        }
        auto finished = std::move(*it);
        inFlight.erase(it);
        finish(std::move(finished), false, 4);
        ++retired;
        it = inFlight.begin();
    }
    if (profile) {
        ++holdCounters.reaps;
        if (retired != 0) ++holdCounters.reapsWithWork;
        holdCounters.reapBatches += retired;
        holdCounters.reapMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    return inFlight.empty();
}

void Recorder::finish(std::unique_ptr<Batch> batch, bool wait, int source, bool releaseNow) {
    const std::lock_guard recordingGuard(recording);
    inFlightKeptBytes -= batch->keptBytes;
    // The batch's writes stay published until its completions ran, on every exit path.
    struct Finishing {
        Recorder& recorder;
        const Batch* batch;
        ~Finishing() {
            auto& list = recorder.finishing;
            list.erase(std::remove(list.begin(), list.end(), batch), list.end());
            // The batch's completion labels have landed, or never will (fence failure below).
            if (batch->completionLabelCount != 0) completionLabels.fetch_sub(batch->completionLabelCount, std::memory_order_acq_rel);
            if (batch->writeBackCompletionCount != 0) writeBackCompletions.fetch_sub(batch->writeBackCompletionCount, std::memory_order_acq_rel);
            // This also runs while the fence-failure throw unwinds: an allocation failure in the
            // rebuild keeps the previous snapshot, which is a superset (a stale entry only causes a
            // false hit, never a miss) rather than terminating.
            try {
                recorder.publishPendingWrites();
            } catch (...) {
            }
        }
    } finishingScope{*this, batch.get()};
    finishing.push_back(batch.get());
    if (wait) {
        // APS5_PROFILE_DRAW: how long the CPU waits for recorded GPU work, reported every 10 s.
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        static double waitedMs = 0;
        static std::uint64_t waits = 0;
        static auto lastReport = std::chrono::steady_clock::now();
        const auto waitStart = std::chrono::steady_clock::now();
        // A fence still unsignaled here makes this a real GPU wait under the mutex (finish runs
        // under it): the [lock] line's 'locked GPU waits'.
        const bool signaledAtStart = !profile || context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus")(context.device, batch->fence) == VK_SUCCESS;
        struct Report {
            bool enabled;
            int source;
            std::chrono::steady_clock::time_point start;
            const Recorder& recorder;
            bool signaledAtStart;
            ~Report() {
                if (!enabled) return;
                const auto now = std::chrono::steady_clock::now();
                const auto ms = std::chrono::duration<double, std::milli>(now - start).count();
                waitedMs += ms;
                syncWaitedMs[source] += ms;
                ++waits;
                CountThreadSync(source, ms);
                CountSiteWait(ms);
                if (!signaledAtStart) GuestMemory::NoteLockedGpuWait(ms);
                if (now - lastReport > std::chrono::seconds(10)) {
                    lastReport = now;
                    const auto& h = holdCounters;
                    AgcDriver::ProfilePrint_nid_no_patch("[recorder] %llu syncs waited %.1f s for the GPU in total (sources, count/wait: idle %llu/%.1fs, pending write %llu/%.1fs, recorded store %llu/%.1fs, address-based %llu/%.1fs, other %llu/%.1fs); hook %llu calls, %llu locked; %llu targeted syncs left %llu batches in flight; snapshot %llu rebuilds %.0f ms, %llu notes covered; %llu unlocked timeline waits %.1f s; %llu submissions, %zu label entries, %llu completion labels pending; fence waits by thread (count/wait):%s; top sync sites (source@caller syncs/batches/wait):%s; under holds (cumulative): %llu completions ran %.0f ms (kept objects released %.0f ms), pending-write syncs from completions: %llu skipped, %llu waited %.0f ms; %llu reaps (%llu with work) retired %llu batches in %.0f ms; hook waits unlocked %llu / %.0f ms GPU (pending write %llu / %.0f ms, recorded store %llu / %.0f ms) + %.0f ms relock (%llu found the recorder torn down), locked %llu; deferred releases: on the release thread %llu batches (%llu objects) in %.0f ms, inline %llu batches (%llu objects) in %.0f ms (%llu batches over the queue bound of %zu), queue max %llu batches, %llu pending\n",static_cast<unsigned long long>(waits), waitedMs / 1000, static_cast<unsigned long long>(syncCounts[0]), syncWaitedMs[0] / 1000, static_cast<unsigned long long>(syncCounts[1]), syncWaitedMs[1] / 1000, static_cast<unsigned long long>(syncCounts[2]), syncWaitedMs[2] / 1000, static_cast<unsigned long long>(syncCounts[3]), syncWaitedMs[3] / 1000, static_cast<unsigned long long>(syncCounts[4]), syncWaitedMs[4] / 1000, static_cast<unsigned long long>(hookCalls.load()), static_cast<unsigned long long>(hookLocks.load()), static_cast<unsigned long long>(targetedSyncs.load()), static_cast<unsigned long long>(batchesLeftInFlight.load()), static_cast<unsigned long long>(snapshotRebuilds), snapshotRebuildMs, static_cast<unsigned long long>(snapshotCovered), static_cast<unsigned long long>(unlockedWaits.load()), unlockedWaitedUs.load() / 1e6, static_cast<unsigned long long>(recorder.submissions), recorder.PendingLabels(), static_cast<unsigned long long>(completionLabels.load()), ThreadSyncReport().c_str(), SyncSiteReport().c_str(), static_cast<unsigned long long>(h.completions), h.completionMs, h.keptReleaseMs, static_cast<unsigned long long>(h.completionSyncsSkipped), static_cast<unsigned long long>(h.completionSyncsWaited), h.completionSyncWaitMs, static_cast<unsigned long long>(h.reaps), static_cast<unsigned long long>(h.reapsWithWork), static_cast<unsigned long long>(h.reapBatches), h.reapMs, static_cast<unsigned long long>(h.hookUnlockedWaits), h.hookUnlockedWaitMs, static_cast<unsigned long long>(h.hookUnlockedWaitsBySource[1]), h.hookUnlockedWaitMsBySource[1], static_cast<unsigned long long>(h.hookUnlockedWaitsBySource[2]), h.hookUnlockedWaitMsBySource[2], h.hookRelockMs, static_cast<unsigned long long>(h.hookUnlockedTornDown), static_cast<unsigned long long>(h.hookLockedWaits), static_cast<unsigned long long>(threadReleases.load()), static_cast<unsigned long long>(threadObjects.load()), threadReleaseUs.load() / 1000.0, static_cast<unsigned long long>(inlineReleases.load()), static_cast<unsigned long long>(inlineObjects.load()), inlineReleaseUs.load() / 1000.0, static_cast<unsigned long long>(inlineOverBound.load()), ReleaseQueueBound(), static_cast<unsigned long long>(releaseQueueMax.load()), static_cast<unsigned long long>(deferredPending.load()));
                    AgcDriver::ProfilePrint_nid_no_patch("[recorder] completion label stores: %llu run, %llu skipped (no CPU write-back overlapped them); %llu counted pending at a write-back; %llu write-backs over a tracked label; %llu write-back completions pending\n", static_cast<unsigned long long>(completionStoresRun.load()), static_cast<unsigned long long>(completionStoresSkipped.load()), static_cast<unsigned long long>(completionLabelsCountedLate.load()), static_cast<unsigned long long>(writeBacksOverLabels.load()), static_cast<unsigned long long>(writeBackCompletions.load()));
                    AgcDriver::ProfilePrint_nid_no_patch("[recorder] submits %llu, vkQueueSubmit mean %.1f us, max %.1f us\n", static_cast<unsigned long long>(submitCount), submitCount != 0 ? submitUs / static_cast<double>(submitCount) : 0.0, submitMaxUs);
                    const auto reads = Recorder::ReadCounts();
                    AgcDriver::ProfilePrint_nid_no_patch("[recorder] in-place reads: %llu noted, %llu queries, hits by reader: dispatch element %llu, gpu copy %llu, address-based %llu, indirect %llu, storage upload %llu, copy source %llu, draw input %llu; %llu hits on signaled batches ignored\n", static_cast<unsigned long long>(reads.noted), static_cast<unsigned long long>(reads.queries), static_cast<unsigned long long>(reads.hits[0]), static_cast<unsigned long long>(reads.hits[1]), static_cast<unsigned long long>(reads.hits[2]), static_cast<unsigned long long>(reads.hits[3]), static_cast<unsigned long long>(reads.hits[4]), static_cast<unsigned long long>(reads.hits[5]), static_cast<unsigned long long>(reads.hits[6]), static_cast<unsigned long long>(reads.staleIgnored));
                }
            }
        } report{profile, source, waitStart, *this, signaledAtStart};
        const auto waitFences = function(waitForFences, "vkWaitForFences");
        VkResult result = VK_SUCCESS;
        {
            const RecordingRelease release(recording);
            result = waitFences(context.device, 1, &batch->fence, VK_TRUE, 5'000'000'000ull);
            for (int waited = 5; result == VK_TIMEOUT; waited += 5) {
                std::fprintf(stderr, "[gpu] recorded batch still running on the GPU after %d s\n", waited);
                result = waitFences(context.device, 1, &batch->fence, VK_TRUE, 5'000'000'000ull);
            }
        }
        if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
            const auto idle = context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(context.device);
            Check(idle, "vkDeviceWaitIdle after recorder fence failure");
        }
        if (result != VK_SUCCESS) {
            release(*batch);
            Check(result, "vkWaitForFences recorder");
        }
    }
    streams[batch->stream].finished = std::max(streams[batch->stream].finished, batch->value);
    if (streams[batch->stream].newest == batch.get()) streams[batch->stream].newest = nullptr;
    readGpuTiming(*batch);
    readSamples(*batch);
    if (batch->serial != 0) {
        std::lock_guard ringLock(completedMutex);
        auto& entry = completed[batch->serial % completed.size()];
        entry.serial = batch->serial;
        entry.gpuStartNs = batch->gpuStartNs;
        entry.gpuEndNs = batch->gpuEndNs;
        entry.submittedAt = batch->submittedAt;
        entry.fenceSeenAt = std::chrono::steady_clock::now();
        entry.readGeneration = batch->readGeneration;
        entry.reads.clear();
        if (FlipReadCheck()) {
            for (const auto& read : batch->reads) entry.reads.emplace_back(read.begin, read.end);
        }
    }
    // Completions store GPU results to guest memory; a failing one is reported, the rest still run.
    // Timed (APS5_PROFILE_DRAW) with the release of the kept objects: both run under whichever hold
    // reaped or synced the batch. Inside them the flush hook makes no pending-write wait (SyncThrough).
    const bool profile = DrawProfiled();
    const auto completionStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    {
        struct InCompletion {
            InCompletion() { ++completionDepth; }
            ~InCompletion() { --completionDepth; }
        } inCompletion;
        for (auto& action : batch->completions) {
            try {
                action();
            } catch (const std::exception& error) {
                AgcDriver::LogCoalescedStderr_nid_no_patch("[gpu] deferred write-back failed: %s\n", error.what());
            }
        }
    }
    if (profile) holdCounters.completions += batch->completions.size();
    const auto keptStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Deferred only from inside a hold: the unlock that releases the list is this thread's own, and
    // a caller finishing batches without the mutex (a test) might never make one.
    if (releaseNow || ReleaseUnderLock() || !GuestMemory::GpuMutex().HeldByThisThread()) {
        batch->completions.clear();
        batch->kept.clear();
    } else {
        // To this thread's unlock (see ReleaseDeferredKeeps); the completions go with the objects,
        // their captures hold the same ones. The push comes first: a failed one (the list's growth)
        // destroys the batch's objects here, under the mutex, counts nothing the destructor would
        // wait for and must not escape (the rest of finish() releases the fence and command buffer).
        try {
            DeferredBatches().push_back({std::move(batch->kept), std::move(batch->completions)});
            deferredPending.fetch_add(1, std::memory_order_acq_rel);
            DeferredBytes() += batch->keptBytes;
        } catch (...) {
        }
        batch->completions.clear();
        batch->kept.clear();
    }
    if (profile) {
        const auto now = std::chrono::steady_clock::now();
        holdCounters.keptReleaseMs += std::chrono::duration<double, std::milli>(now - keptStart).count();
        holdCounters.completionMs += std::chrono::duration<double, std::milli>(now - completionStart).count();
    }
    // The batch's label entries leave the table (a later label to the same dword already replaced
    // its entry and belongs to another batch). Correctness never depended on this removal.
    removeLabels(*batch);
    release(*batch);
}

void Recorder::discard(std::unique_ptr<Batch> batch) noexcept {
    const std::lock_guard recordingGuard(recording);
    removeLabels(*batch);
    if (batch->completionLabelCount != 0) completionLabels.fetch_sub(batch->completionLabelCount, std::memory_order_acq_rel);
    if (batch->writeBackCompletionCount != 0) writeBackCompletions.fetch_sub(batch->writeBackCompletionCount, std::memory_order_acq_rel);
    endLabelDeadline(batch->stream);
    release(*batch);
    try {
        publishPendingWrites();
    } catch (...) {
    }
}

void Recorder::removeLabels(Batch& batch) {
    const std::lock_guard recordingGuard(recording);
    if (!batch.labelDwords.empty() || !batch.wideLabels.empty()) {
        std::lock_guard tableLock(labelTableMutex);
        for (const auto dword : batch.labelDwords) {
            const auto found = labels.find(dword);
            if (found == labels.end()) continue;
            std::erase_if(found->second, [&](const LabelWriter& writer) { return writer.stream == batch.stream && writer.batch == batch.value; });
            if (found->second.empty()) labels.erase(found);
        }
        for (const auto entry : batch.wideLabels) wideLabels.erase(entry);
        recordedLabels.store(labels.size(), std::memory_order_relaxed);
    }
    batch.labelDwords.clear();
    batch.wideLabels.clear();
}

void Recorder::release(Batch& batch) noexcept {
    const std::lock_guard recordingGuard(recording);
    if (batch.queries != VK_NULL_HANDLE) {
        if (!GpuTimingEnabled() && sparePools.size() < 64) {
            try {
                sparePools.push_back(batch.queries);
                batch.queries = VK_NULL_HANDLE;
            } catch (...) {
            }
        }
        if (batch.queries != VK_NULL_HANDLE) context.Function<PFN_vkDestroyQueryPool>("vkDestroyQueryPool")(context.device, batch.queries, nullptr);
    }
    batch.queries = VK_NULL_HANDLE;
    if (batch.samples != VK_NULL_HANDLE && batch.samplePool == nullptr) context.Function<PFN_vkDestroyQueryPool>("vkDestroyQueryPool")(context.device, batch.samples, nullptr);
    batch.samples = VK_NULL_HANDLE;
    batch.samplePool.reset();
    batch.segments = VK_NULL_HANDLE;
    batch.segmentPool.reset();
    // A completed (or never submitted) batch's objects are kept for reuse: the fence is signaled or
    // untouched, so resetting it cannot block, and the command buffer is no longer pending.
    auto& spare = streams[batch.stream].spare;
    if (batch.commands != VK_NULL_HANDLE && batch.fence != VK_NULL_HANDLE && spare.size() < 64 && function(resetFences, "vkResetFences")(context.device, 1, &batch.fence) == VK_SUCCESS) {
        spare.emplace_back(batch.commands, batch.fence);
        batch.commands = VK_NULL_HANDLE;
        batch.fence = VK_NULL_HANDLE;
        return;
    }
    if (batch.commands != VK_NULL_HANDLE) context.Function<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(context.device, streamPool(batch.stream), 1, &batch.commands);
    if (batch.fence != VK_NULL_HANDLE) context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, batch.fence, nullptr);
    batch.commands = VK_NULL_HANDLE;
    batch.fence = VK_NULL_HANDLE;
}

}
