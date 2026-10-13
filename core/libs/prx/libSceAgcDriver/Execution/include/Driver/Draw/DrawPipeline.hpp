#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

class DrawPipeline {
public:
    using Commit = std::function<void()>;
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    enum class DrainReason : std::uint8_t { Packet, Flush, Labels, Capture, Indirect, Submission, Count };

    static DrawPipeline& Queue0();
    static std::size_t Depth();
    static bool& Active();
    static std::atomic<std::uint64_t>& EpochToken();
    static void FollowEpoch(std::uint64_t token);
    // APS5_PIPELINE_INDIRECT=1 (from upstream #1896's GPU-side part): an indirect draw whose arguments
    // the GPU reads is enqueued like a direct one instead of draining the pipeline first.
    static bool PipelineIndirect();

    void Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {});
    void Drain(DrainReason reason, std::uint32_t opcode = 0x100);
    bool Busy() const { return outstanding.load(std::memory_order_acquire) != 0; }
    bool Overlaps(std::uint64_t address, std::size_t bytes);
    std::optional<std::uint64_t> PendingLabel(std::uint64_t address, std::size_t bytes);
    // Drains when any of `ranges` (bounded by `bounds`) overlaps a write still pending in the pipeline.
    template<typename TRanges, typename TBounds>
    bool DrainIfOverlaps(const TRanges& ranges, TBounds&& bounds, DrainReason reason, std::uint32_t opcode = 0x100) {
        if (!Busy() || !pendingWrites(ranges, bounds)) return false;
        Drain(reason, opcode);
        return true;
    }

private:
    struct Item {
        Commit commit;
        std::vector<Range> writes;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> labelBytes;
        // The worker's collect epoch token when it enqueued the item: the committer follows it
        // before every item, not only draws (a fill or DMA commit collects writes too).
        std::uint64_t epoch = 0;
    };
    DrawPipeline() = default;
    void run();
    void rethrowFailure();
    void report(std::chrono::steady_clock::time_point now);
    template<typename TRanges, typename TBounds>
    bool pendingWrites(const TRanges& ranges, TBounds& bounds) {
        std::uint64_t lowest = ~std::uint64_t{0};
        std::uint64_t highest = 0;
        for (const auto& range : ranges) {
            const auto [first, end] = bounds(range);
            if (first >= end) continue;
            lowest = std::min(lowest, first);
            highest = std::max(highest, end);
        }
        if (lowest >= highest) return false;
        std::lock_guard lock(mutex);
        for (const auto& item : items) {
            for (const auto& [begin, limit] : item.writes) {
                if (begin >= highest || limit <= lowest) continue;
                for (const auto& range : ranges) {
                    const auto [first, end] = bounds(range);
                    if (first < limit && begin < end) return true;
                }
            }
        }
        return false;
    }

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Item> items;
    std::exception_ptr failure;
    std::atomic<std::size_t> outstanding{0};
    std::uint32_t idleWaiters = 0;
    bool committerWaiting = false;
    std::thread thread;
    std::uint64_t commits = 0;
    std::uint64_t commitNs = 0;
    std::uint64_t commitErrors = 0;
    std::uint64_t enqueued = 0;
    std::uint64_t fullWaitNs = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drains{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drainWaitNs{};
    std::array<std::uint64_t, 257> drainOpcodes{};
    std::uint64_t depthSum = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

}

#endif
