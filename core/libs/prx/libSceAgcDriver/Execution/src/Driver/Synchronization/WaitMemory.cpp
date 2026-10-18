#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/SubmissionRing.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <cstdlib>
#include <mutex>
#include <set>
#ifdef _WIN32
#include <windows.h>
#endif
#include <immintrin.h>
#include <algorithm>
#include <tuple>
#include <vector>

namespace AgcDriver::DriverDetail {

void PollSleep() {
#ifdef _WIN32
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (timer != nullptr) {
        LARGE_INTEGER due{};
        due.QuadPart = -2000;
        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
            WaitForSingleObject(timer, INFINITE);
            return;
        }
    }
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(200));
}

int Driver::waitTimeoutMs() {
    static const int value = [] { const char* text = std::getenv("APS5_GPU_WAIT_TIMEOUT_MS"); return text ? std::atoi(text) : 1000; }();
    return value;
}

std::vector<Driver::PendingWait>& Driver::pendingWaits() {
    struct WaitsTag {};
    return HostThreadLocal<std::vector<PendingWait>, WaitsTag>();
}

void Driver::notePendingWait(std::span<const std::uint32_t> packet, std::uint32_t queue, const Graphics::Recorder::Token& writer) {
    if (packet.size() > std::tuple_size_v<decltype(PendingWait::words)> || queue == 0) return;
    if (std::all_of(writer.begin(), writer.end(), [](std::uint64_t value) { return value == 0; })) return;
    if (Pm4::WaitSatisfiedUnchecked(packet)) return;
    auto& waits = pendingWaits();
    for (auto& wait : waits) {
        if (wait.count != packet.size() || !std::equal(packet.begin(), packet.end(), wait.words.begin())) continue;
        for (std::size_t stream = 0; stream < writer.size(); ++stream) wait.writer[stream] = std::max(wait.writer[stream], writer[stream]);
        return;
    }
    PendingWait wait{};
    std::copy(packet.begin(), packet.end(), wait.words.begin());
    wait.count = packet.size();
    wait.writer = writer;
    waits.push_back(wait);
}

void Driver::landPendingWaits(std::uint32_t queue) {
    auto& waits = pendingWaits();
    if (waits.empty()) return;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    struct LandStats {
        std::uint64_t lands = 0, waited = 0;
        double waitedMs = 0, maxMs = 0;
        std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
    };
    static thread_local LandStats stats;
    const auto localDevice = device.Load();
    for (const auto& wait : waits) {
        const auto packet = std::span<const std::uint32_t>(wait.words).first(wait.count);
        const std::uint64_t awaited = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
        const std::size_t awaitedBytes = Pm4::WaitAwaitedBytes(packet);
        const auto landed = [&] { return Pm4::WaitSatisfiedUnchecked(packet) || localDevice == nullptr || localDevice->TokenSignaled(wait.writer) || !Graphics::Recorder::SnapshotWriteOverlaps(awaited, awaitedBytes); };
        ++stats.lands;
        if (landed()) continue;
        const auto started = std::chrono::steady_clock::now();
        bool warned = false;
        while (!landed()) {
            PollSleep();
            CheckFailure();
            if (!warned && std::chrono::steady_clock::now() - started > std::chrono::milliseconds(waitTimeoutMs())) {
                warned = true;
                std::fprintf(stderr, "[gpu] queue 0x%x landing of a wait on 0x%llx timed out after %dms (function %u, value 0x%x)\n", queue, static_cast<unsigned long long>(awaited), waitTimeoutMs(), packet[1] & 7u, *reinterpret_cast<const volatile std::uint32_t*>(awaited));
            }
        }
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        ++stats.waited;
        stats.waitedMs += ms;
        stats.maxMs = std::max(stats.maxMs, ms);
    }
    waits.clear();
    if (profile && std::chrono::steady_clock::now() - stats.reported > std::chrono::seconds(10)) {
        AgcDriver::ProfilePrint_nid_no_patch("[landwait] queue 0x%x (10 s): %llu waits landed before a guest read, %llu had to wait for the GPU: %.1f ms in total, max %.2f ms\n", queue, static_cast<unsigned long long>(stats.lands), static_cast<unsigned long long>(stats.waited), stats.waitedMs, stats.maxMs);
        stats = LandStats{};
    }
}

void Driver::waitMemory(std::span<const std::uint32_t> packet, std::uint32_t queue, const PacketHistory& context, std::uint64_t received, bool heldAtSubmit) {

    auto start = std::chrono::steady_clock::now();
    auto lastDone = packetsDone.load();

    GuestMemory::TagGpuLockThread(queue);
    auto& outcomes = waitOutcomes();
    const std::uint64_t awaited = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
    const std::size_t awaitedBytes = Pm4::WaitAwaitedBytes(packet);
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    if (traceGpu) std::fprintf(stderr, "[gpu] %.1f queue 0x%x waits 0x%llx == 0x%x (now 0x%x)\n", TraceMs(), queue, static_cast<unsigned long long>(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u)), packet[4],
                               *reinterpret_cast<const volatile std::uint32_t*>(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u)));
    struct WaitTrace {
        bool enabled; std::uint32_t queue; std::uint64_t address; std::chrono::steady_clock::time_point begin;
        ~WaitTrace() {
            if (!enabled) return;
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            if (ms >= 0.5) std::fprintf(stderr, "[gpu] %.1f queue 0x%x wait on 0x%llx done after %.1f ms\n", TraceMs(), queue, static_cast<unsigned long long>(address), ms);
        }
    } waitTrace{traceGpu, queue, packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), std::chrono::steady_clock::now()};
    bool warned = false;

    struct EpochPoint {
        bool bump = true;
        ~EpochPoint() {
            if (bump) bumpEpoch(&EpochBumps::waits);
        }
    } epochPoint;
    const auto accepts = [&](std::uint64_t value) { return Pm4::WaitComparesValue(packet, value); };
    const auto orderAfterValue = [&] {
        const auto localDevice = device.Load();
        if (localDevice == nullptr || !localDevice->SeparateQueues() || !Graphics::Recorder::SnapshotWriteOverlaps(awaited, awaitedBytes)) return;
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Wait);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        localDevice->OrderAfterValue(awaited, awaitedBytes);
    };
    const auto orderAfterOtherQueue = [&](const std::optional<Graphics::Recorder::LabelHit>& hit) {
        const auto localDevice = device.Load();
        if (localDevice == nullptr) return;
        Graphics::Recorder::Token writer{};
        if (hit.has_value()) writer[hit->stream] = hit->batch;
        else writer = localDevice->LabelWriterBatches(awaited, awaitedBytes);
        notePendingWait(packet, queue, writer);
        if (!localDevice->SeparateQueues()) return;
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Wait);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (hit.has_value()) localDevice->OrderAfterBatch(hit->stream, hit->batch);
        else localDevice->OrderAfterLabel(awaited, awaitedBytes);
    };
    GuestMemory::CheckRange(reinterpret_cast<const void*>(awaited), awaitedBytes, awaitedBytes);
    if (Pm4::WaitSatisfiedUnchecked(packet)) {
        ++outcomes.atEntry;
        orderAfterValue();
        return;
    }
    if (heldAtSubmit) {
        orderAfterValue();
        return;
    }
    if (std::uint32_t writer = queue; storedSince(packet, awaited, awaitedBytes, received, &writer)) {
        if (writer != queue) orderAfterOtherQueue(std::nullopt);
        else epochPoint.bump = false;
        return;
    }

    bool behindCompletion = false;
    const auto takeLabel = [&](const std::optional<Graphics::Recorder::LabelHit>& hit, bool polling, bool unlockedHit) {
        if (!hit.has_value()) return false;
        ++(polling ? outcomes.fromRecorderPolling : outcomes.fromRecorder);
        if (unlockedHit) ++outcomes.fromRecorderUnlocked;
        if (hit->queue == queue) {
            ++outcomes.fromRecorderSameQueue;
            epochPoint.bump = false;
        } else {
            orderAfterOtherQueue(hit);
        }
        return true;
    };

    std::uint64_t seenGeneration = Graphics::Recorder::WriteGeneration();

    if (takeLabel(Graphics::Recorder::LookupLabel(awaited, awaitedBytes, received, accepts, &behindCompletion), false, true)) return;
    const bool lockedEntry = Graphics::Recorder::SnapshotWriteOverlaps(awaited, awaitedBytes);
    if (!lockedEntry) ++outcomes.entriesUnlocked;

    bool recheck = !lockedEntry;
    bool entryTryFailed = false;
    if (lockedEntry) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Wait);
        std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
        if (!gpuLock.try_lock()) {
            ++outcomes.entryTriesFailed;
            recheck = true;
            entryTryFailed = true;
        }
        if (const auto localDevice = gpuLock.owns_lock() ? device.Load() : std::shared_ptr<VulkanDevice>{}) {
            if (takeLabel(localDevice->PendingLabel(awaited, awaitedBytes, received, accepts, &behindCompletion), false, false)) return;
            if (localDevice->OpenWriteOverlaps(awaited, awaitedBytes)) localDevice->SubmitRecorded(false);
        }
    }

    static const bool pollProfile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto& poll = pollStats();

    const bool reapAll = queue == 0;
    std::chrono::steady_clock::time_point lastTry{};
    const auto pollService = [&](bool sleeping) {
        if (pollProfile) ++poll.ticks;
        const auto generation = Graphics::Recorder::WriteGeneration();
        const bool changed = generation != seenGeneration || recheck;
        const bool labelDue = labelFlushDue();
        const bool reap = sleeping && (behindCompletion || ((reapAll || queue0Dormant.load(std::memory_order_relaxed)) && completionsPending()));
        if (!changed && !labelDue && !reap) return false;

        if (!changed) {
            const auto now = std::chrono::steady_clock::now();
            if (now - lastTry < PollTryInterval) return false;
            lastTry = now;
        }
        if (changed) {
            if (takeLabel(Graphics::Recorder::LookupLabel(awaited, awaitedBytes, received, accepts, &behindCompletion), true, true)) {
                if (pollProfile) ++poll.tableHits;
                return true;
            }
        }
        if (pollProfile) ++poll.tries;
        std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::try_to_lock);
        if (!gpuLock.owns_lock()) {
            ++triesFailed[TryPoll];
            if (pollProfile) ++poll.triesFailed;
            return false;
        }

        struct TryHold {
            bool enabled;
            PollStats& poll;
            std::chrono::steady_clock::time_point start;
            ~TryHold() {
                if (!enabled) return;
                const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                poll.tryHoldUs += us;
                poll.longestTryHoldUs = std::max(poll.longestTryHoldUs, us);
            }
        } tryHold{pollProfile, poll, pollProfile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
        const auto localDevice = device.Load();
        if (localDevice == nullptr) {
            seenGeneration = generation;
            recheck = false;
            return false;
        }
        bool submitted = false;
        if (changed) {
            seenGeneration = generation;
            recheck = false;
            if (takeLabel(localDevice->PendingLabel(awaited, awaitedBytes, received, accepts, &behindCompletion), true, false)) {
                if (pollProfile) ++poll.tableHits;
                return true;
            }
            if (localDevice->OpenWriteOverlaps(awaited, awaitedBytes)) {
                localDevice->SubmitRecorded(false);
                submitted = true;
            }
        }
        if (labelDue && !submitted && Graphics::Recorder::PendingLabelSince().has_value()) {
            localDevice->SubmitLabelled(false);
            submitted = true;
        }
        if (submitted) {
            ++outcomes.pollSubmits;
            if (pollProfile) ++poll.submits;
        }
        if (reap) {
            const auto withWorkBefore = pollProfile ? Graphics::Recorder::ReapsWithWork() : 0;
            localDevice->ReapRecorded();
            ++outcomes.pollReaps;
            if (pollProfile) {
                ++poll.reaps;
                ++(behindCompletion ? poll.reapsBehindCompletion : poll.reapsOther);
                poll.reapsWithWork += Graphics::Recorder::ReapsWithWork() - withWorkBefore;
            }
        }
        return false;
    };

    if (entryTryFailed && pollService(false)) return;

    noteWaitBlocked(queue, awaited, true);
    struct BlockedWait {
        Driver& driver;
        std::uint32_t queue;
        ~BlockedWait() { driver.noteWaitBlocked(queue, 0, false); }
    } blocked{*this, queue};
    static const bool pauseSpin = std::getenv("APS5_NO_PAUSE_SPIN") == nullptr;
    const auto spinLimit = queue == 0 ? std::chrono::microseconds(1500) : std::chrono::microseconds(100);
    const auto spinStart = std::chrono::steady_clock::now();
    bool spinning = pauseSpin;
    std::uint32_t polls = 0;
    StallWatch stall(queue);
    while (!Pm4::WaitSatisfiedUnchecked(packet)) {
        if (std::uint32_t writer = queue; storedSince(packet, awaited, awaitedBytes, received, &writer)) {
            if (writer != queue) orderAfterOtherQueue(std::nullopt);
            else epochPoint.bump = false;
            return;
        }
        ++polls;
        if (spinning) {
            _mm_pause();
            if ((polls & 63u) != 0) continue;
            if (std::chrono::steady_clock::now() - spinStart > spinLimit) spinning = false;
        } else if (pauseSpin || polls >= 4000) {
            PollSleep();
        } else {
            std::this_thread::yield();
        }
        CheckFailure();
        if (pollService(!spinning)) return;
        noteAwaitingTitle(queue, awaited);
        if (const auto done = packetsDone.load(); done != lastDone || packetsInFlight.load() != 0) {
            lastDone = done;
            start = std::chrono::steady_clock::now();
            stall.Progressed();
        }
        stall.Check(((packet[0] >> 8u) & 0xffu) == 0x93u ? "WAIT_REG_MEM_64 at" : "WAIT_REG_MEM at", awaited);

        if (!warned && std::chrono::steady_clock::now() - start > std::chrono::milliseconds(waitTimeoutMs())) {
            warned = true;
            ++outcomes.timedOut;
            {
                static std::mutex reportedMutex;
                static std::set<std::uint64_t> reported;
                static std::uint64_t timeouts = 0;
                const std::lock_guard lock(reportedMutex);
                if (++timeouts % 20 == 0) std::fprintf(stderr, "[gpu] %llu GPU waits have timed out\n", static_cast<unsigned long long>(timeouts));
                if (!reported.insert(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u)).second) continue;
            }
            const bool wide = ((packet[0] >> 8u) & 0xffu) == 0x93u;
            const std::uint64_t reference = wide ? packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u) : packet[4];
            const std::uint64_t mask = wide ? packet[6] | (static_cast<std::uint64_t>(packet[7]) << 32u) : packet[5];
            const std::uint64_t current = wide ? *reinterpret_cast<const volatile std::uint64_t*>(awaited) : *reinterpret_cast<const volatile std::uint32_t*>(awaited);
            std::fprintf(stderr, "[gpu] queue 0x%x WAIT_REG_MEM%s at 0x%llx timed out after %dms (function %u ref 0x%llx mask 0x%llx value 0x%llx)\n", queue, wide ? "_64" : "",
                         static_cast<unsigned long long>(awaited), waitTimeoutMs(), packet[1] & 7u, static_cast<unsigned long long>(reference), static_cast<unsigned long long>(mask), static_cast<unsigned long long>(current));
            const auto address = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
            for (const auto& record : writeHistory()) {
                if (record.length != 0 && record.target <= address && address < record.target + std::max<std::uint64_t>(record.length, 4))
                    std::fprintf(stderr, "[gpu]   earlier write by queue 0x%x opcode 0x%x at 0x%llx+0x%llx\n", record.queue, record.opcode, static_cast<unsigned long long>(record.target), static_cast<unsigned long long>(record.length));
            }

            if (const auto reportDevice = device.Load(); reportDevice != nullptr) {
                const auto pending = Graphics::Recorder::LookupLabelValue(awaited, awaitedBytes);
                const auto since = Graphics::Recorder::PendingLabelSince();
                const auto age = since.has_value() ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *since).count() : -1.0;
                std::fprintf(stderr, "[gpu]   recorder: newest pending label %s 0x%llx, open batch writes it %d, snapshot write overlaps %d, pending completion labels %llu, write-back completions %llu, pending label age %.1f ms, write generation %llu seen %llu, queue 0 dormant %d, completions pending %d, label due %d\n",
                             pending.has_value() ? "value" : "none", static_cast<unsigned long long>(pending.value_or(0)), reportDevice->OpenWriteOverlaps(awaited, awaitedBytes) ? 1 : 0, Graphics::Recorder::SnapshotWriteOverlaps(awaited, awaitedBytes) ? 1 : 0,
                             static_cast<unsigned long long>(Graphics::Recorder::PendingCompletionLabels()), static_cast<unsigned long long>(Graphics::Recorder::PendingWriteBackCompletions()), age,
                             static_cast<unsigned long long>(Graphics::Recorder::WriteGeneration()), static_cast<unsigned long long>(seenGeneration), queue0Dormant.load(std::memory_order_relaxed) ? 1 : 0, completionsPending() ? 1 : 0, labelFlushDue() ? 1 : 0);
            }
            context.Print(stderr, "[gpu]   preceding packet %s\n");
        }
    }
    ++outcomes.polled;
    orderAfterValue();
}

}
