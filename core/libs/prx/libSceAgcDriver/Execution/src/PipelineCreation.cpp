#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "prx/libc/include/HostThread.hpp"
#include <algorithm>
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

std::size_t CreationThread::Words() {
    static const std::size_t words = RetainedBudgetBytes / (std::max(2u, std::thread::hardware_concurrency()) * RetainedBytesPerWord);
    return words;
}

CreationGate::Shared& CreationGate::State() {
    static auto& shared = *new Shared;
    return shared;
}

std::size_t CreationGate::DefaultBudgetBytes() {
    static const std::size_t budget = [] {
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
        return static_cast<std::size_t>(physical != 0 ? physical / 16u : std::uint64_t{4} << 30u);
    }();
    return budget;
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
