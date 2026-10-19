#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/HostThread.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace AgcDriver {

namespace {

thread_local bool creationThread = false;

struct CreationQueue {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> tasks;
    std::uint64_t routed = 0;
    bool started = false;
};

CreationQueue& Queue() {
    static auto* queue = new CreationQueue();
    return *queue;
}

thread_local bool creationPoolThread = false;

struct AtUsePool {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::pair<std::function<void()>, bool>> tasks;
    std::size_t threads = 0;
    std::size_t idle = 0;
    std::uint64_t onPool = 0;
    std::uint64_t background = 0;
    std::uint64_t backgroundOutstanding = 0;
    std::condition_variable backgroundDone;
    std::uint64_t waits = 0;
    std::uint64_t waitNanoseconds = 0;
};

AtUsePool& Pool() {
    static auto* pool = new AtUsePool();
    return *pool;
}

void* RunAtUseCreations(void* argument) {
    auto& pool = *static_cast<AtUsePool*>(argument);
    creationPoolThread = true;
    std::unique_lock lock(pool.mutex);
    for (;;) {
        ++pool.idle;
        pool.ready.wait(lock, [&] { return !pool.tasks.empty(); });
        --pool.idle;
        auto [task, background] = std::move(pool.tasks.front());
        pool.tasks.pop_front();
        ++(background ? pool.background : pool.onPool);
        lock.unlock();
        if (background) {
            const CreationInBackground behind;
            task();
        } else {
            task();
        }
        task = {};
        lock.lock();
        if (background && --pool.backgroundOutstanding == 0) pool.backgroundDone.notify_all();
    }
    return nullptr;
}

struct RamSide {
    std::mutex mutex;
    std::chrono::steady_clock::time_point sampled{};
    std::size_t guestMapped = 0;
    CreationGate::MemorySource deviceLocal = nullptr;
};

RamSide& RamSideState() {
    static auto* state = new RamSide();
    return *state;
}

std::size_t GuestMappedBytes() {
    std::size_t bytes = 0;
#ifdef _WIN32
    if (!GuestArena::GuestArenaAvailable_nid_postfix()) return 0;
    GuestArena::GuestArenaForEachMapping_nid_postfix([](void* context, std::uintptr_t, std::size_t size, GuestArena::MappingChange change, std::uint32_t) {
        if (change == GuestArena::MappingChange::Mapped) *static_cast<std::size_t*>(context) += size;
    }, &bytes);
#endif
    return bytes;
}

std::size_t PrivateBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) return counters.PrivateUsage;
#endif
    return 0;
}

void* RunCreations(void* argument) {
    auto& queue = *static_cast<CreationQueue*>(argument);
    creationThread = true;
    const CreationAhead ahead;
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock lock(queue.mutex);
            queue.ready.wait(lock, [&] { return !queue.tasks.empty(); });
            task = std::move(queue.tasks.front());
            queue.tasks.pop_front();
        }
        task();
    }
    return nullptr;
}

}

std::atomic<std::uint64_t>& RecordUse::Marks() {
    static std::atomic<std::uint64_t> marks{0};
    return marks;
}

bool& CreationAhead::Flag() {
    thread_local bool ahead = false;
    return ahead;
}

bool& CreationInBackground::Flag() {
    thread_local bool background = false;
    return background;
}

std::size_t CreationThread::Words() {
    static const std::size_t words = RetainedBudgetBytes / (std::max(2u, std::thread::hardware_concurrency()) * RetainedBytesPerWord);
    return words;
}

CreationGate::Shared& CreationGate::State() {
    static auto& shared = [] () -> Shared& {
        auto* created = new Shared;
        created->memory = &RamSideBytes;
        return *created;
    }();
    return shared;
}

namespace {

std::uint64_t PhysicalMemoryBytes() {
    std::uint64_t physical = 0;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) physical = status.ullTotalPhys;
#else
    const auto pages = sysconf(_SC_PHYS_PAGES);
    const auto pageSize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && pageSize > 0) physical = static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(pageSize);
#endif
    return physical;
}

struct PipelineBudgetState {
    std::atomic<std::size_t> live{0};
    std::atomic<std::size_t> budget{0};
    std::atomic<std::uint64_t> tick{0};
    std::atomic<std::uint64_t> evicted{0};
};

PipelineBudgetState& PipelineBudgetShared() {
    static auto* state = new PipelineBudgetState;
    return *state;
}

}

std::size_t CreationGate::DefaultBudgetBytes() {
    static const std::size_t budget = [] {
        const auto physical = PhysicalMemoryBytes();
        return static_cast<std::size_t>(physical != 0 ? physical / 16u : std::uint64_t{4} << 30u);
    }();
    return budget;
}

PipelineBudget::Charge::Charge(std::size_t bytes) : bytes(bytes) {
    PipelineBudgetShared().live.fetch_add(bytes, std::memory_order_relaxed);
}

PipelineBudget::Charge& PipelineBudget::Charge::operator=(Charge&& other) noexcept {
    if (this != &other) {
        PipelineBudgetShared().live.fetch_sub(bytes, std::memory_order_relaxed);
        bytes = std::exchange(other.bytes, 0);
    }
    return *this;
}

PipelineBudget::Charge::~Charge() {
    PipelineBudgetShared().live.fetch_sub(bytes, std::memory_order_relaxed);
}

std::size_t PipelineBudget::DefaultBudgetBytes() {
    static const std::size_t budget = [] {
        const auto physical = PhysicalMemoryBytes();
        return std::max(MinimumBudgetBytes, static_cast<std::size_t>(physical != 0 ? physical / 32u : std::uint64_t{2} << 30u));
    }();
    return budget;
}

std::size_t PipelineBudget::BudgetBytes() {
    const auto set = PipelineBudgetShared().budget.load(std::memory_order_relaxed);
    return set != 0 ? set : DefaultBudgetBytes();
}

void PipelineBudget::SetBudgetBytes(std::size_t bytes) {
    PipelineBudgetShared().budget.store(bytes, std::memory_order_relaxed);
}

std::size_t PipelineBudget::LiveBytes() {
    return PipelineBudgetShared().live.load(std::memory_order_relaxed);
}

std::uint64_t PipelineBudget::Tick() {
    return PipelineBudgetShared().tick.fetch_add(1, std::memory_order_relaxed) + 1;
}

void PipelineBudget::CountEvicted() {
    PipelineBudgetShared().evicted.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t PipelineBudget::Evicted() {
    return PipelineBudgetShared().evicted.load(std::memory_order_relaxed);
}

std::size_t CreationGate::RamSideBytes() {
    auto& state = RamSideState();
    std::lock_guard lock(state.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (state.sampled == std::chrono::steady_clock::time_point{} || now - state.sampled >= std::chrono::seconds(1)) {
        state.guestMapped = GuestMappedBytes();
        state.sampled = now;
    }
    const auto total = PrivateBytes() + state.guestMapped;
    const auto deviceLocal = state.deviceLocal != nullptr ? state.deviceLocal() : 0u;
    return total > deviceLocal ? total - deviceLocal : 0u;
}

void CreationGate::UseRamSide(MemorySource deviceLocal) {
    {
        auto& state = RamSideState();
        std::lock_guard lock(state.mutex);
        state.deviceLocal = deviceLocal;
    }
    SetMemorySource(&RamSideBytes, MemoryLimitBytes);
}

bool CreationPool::Current() {
    return creationPoolThread;
}

CreationPool::Counts CreationPool::Totals() {
    auto& pool = Pool();
    std::lock_guard lock(pool.mutex);
    return {pool.onPool, pool.waits, pool.waitNanoseconds, pool.threads, pool.background};
}

void CreationPool::Later(std::function<void()> task) {
    Submit(std::move(task), true);
}

void CreationPool::WaitBackground() {
    auto& pool = Pool();
    std::unique_lock lock(pool.mutex);
    pool.backgroundDone.wait(lock, [&] { return pool.backgroundOutstanding == 0; });
}

void CreationPool::Submit(std::function<void()> task, bool background) {
    auto& pool = Pool();
    {
        std::lock_guard lock(pool.mutex);
        if (pool.idle <= pool.tasks.size()) {
            if (!HostThread::StartDetached(std::size_t{64} << 20u, &RunAtUseCreations, &pool)) throw std::runtime_error("AGC driver: a pipeline creation pool thread did not start");
            ++pool.threads;
        }
        pool.tasks.emplace_back(std::move(task), background);
        if (background) ++pool.backgroundOutstanding;
    }
    pool.ready.notify_one();
}

void CreationPool::NoteWait(std::uint64_t nanoseconds) {
    auto& pool = Pool();
    std::lock_guard lock(pool.mutex);
    ++pool.waits;
    pool.waitNanoseconds += nanoseconds;
}

bool CreationThread::Current() {
    return creationThread;
}

std::uint64_t CreationThread::Routed() {
    auto& queue = Queue();
    std::lock_guard lock(queue.mutex);
    return queue.routed;
}

void CreationThread::Submit(std::function<void()> task) {
    auto& queue = Queue();
    {
        std::lock_guard lock(queue.mutex);
        if (!queue.started) {
            if (!HostThread::StartDetached(std::size_t{64} << 20u, &RunCreations, &queue)) throw std::runtime_error("AGC driver: the pipeline creation thread did not start");
            queue.started = true;
        }
        queue.tasks.push_back(std::move(task));
        ++queue.routed;
    }
    queue.ready.notify_one();
}

}
