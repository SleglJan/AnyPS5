#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PIPELINECREATION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PIPELINECREATION_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <utility>

namespace AgcDriver {

class RecordUse {
public:
    explicit RecordUse(std::shared_ptr<std::atomic<std::uint64_t>> uses) : uses(std::move(uses)) {}
    RecordUse(const RecordUse&) = delete;
    RecordUse& operator=(const RecordUse&) = delete;

    void Mark() noexcept {
        if (order.load(std::memory_order_relaxed) != 0) return;
        std::uint64_t unused = 0;
        if (order.compare_exchange_strong(unused, uses->fetch_add(1, std::memory_order_relaxed) + 1, std::memory_order_relaxed)) Marks().fetch_add(1, std::memory_order_relaxed);
    }

    std::uint64_t Order() const noexcept { return order.load(std::memory_order_relaxed); }
    static std::atomic<std::uint64_t>& Marks();

private:
    std::shared_ptr<std::atomic<std::uint64_t>> uses;
    std::atomic<std::uint64_t> order{0};
};

class CreationAhead {
public:
    CreationAhead() : previous(Flag()) { Flag() = true; }
    ~CreationAhead() { Flag() = previous; }
    CreationAhead(const CreationAhead&) = delete;
    CreationAhead& operator=(const CreationAhead&) = delete;
    static bool Active() { return Flag(); }

private:
    static bool& Flag();

    bool previous;
};

class CreationInBackground {
public:
    CreationInBackground() : previous(Flag()) { Flag() = true; }
    ~CreationInBackground() { Flag() = previous; }
    CreationInBackground(const CreationInBackground&) = delete;
    CreationInBackground& operator=(const CreationInBackground&) = delete;
    static bool Active() { return Flag(); }
    static bool BehindFrames() { return Active() || CreationAhead::Active(); }

private:
    static bool& Flag();

    bool previous;
};

class PipelineBudget {
public:
    static constexpr std::size_t BytesPerWord = 40;
    static constexpr std::size_t MinimumPipelineBytes = std::size_t{64} << 10u;
    static constexpr std::size_t MinimumBudgetBytes = std::size_t{512} << 20u;

    class Charge {
    public:
        Charge() = default;
        explicit Charge(std::size_t bytes);
        Charge(Charge&& other) noexcept : bytes(std::exchange(other.bytes, 0)) {}
        Charge& operator=(Charge&& other) noexcept;
        ~Charge();
        std::size_t Bytes() const { return bytes; }

    private:
        std::size_t bytes = 0;
    };

    static std::size_t PipelineBytes(std::size_t words) { return std::max(MinimumPipelineBytes, words * BytesPerWord); }
    static std::size_t DefaultBudgetBytes();
    static std::size_t BudgetBytes();
    static void SetBudgetBytes(std::size_t bytes);
    static std::size_t LiveBytes();
    static bool Over() { return LiveBytes() > BudgetBytes(); }
    static std::uint64_t Tick();
    static void CountEvicted();
    static std::uint64_t Evicted();
};

class CreationThread {
public:
    static constexpr std::size_t RetainedBudgetBytes = std::size_t{1} << 30u;
    static constexpr std::size_t RetainedBytesPerWord = 567;

    static std::size_t Words();

    template<typename TCreate>
    static auto Run(std::size_t words, TCreate&& create) -> decltype(create()) {
        if (!CreationInBackground::BehindFrames() || words < Words() || Current()) return create();
        using Result = decltype(create());
        std::promise<Result> done;
        auto result = done.get_future();
        Submit([&] {
            try {
                done.set_value(create());
            } catch (...) {
                done.set_exception(std::current_exception());
            }
        });
        return result.get();
    }

    static bool Current();
    static std::uint64_t Routed();

private:
    static void Submit(std::function<void()> task);
};

class CreationPool {
public:
    struct Counts {
        std::uint64_t onPool = 0;
        std::uint64_t waits = 0;
        std::uint64_t waitNanoseconds = 0;
        std::size_t threads = 0;
        std::uint64_t background = 0;
    };

    template<typename TCreate>
    static auto AtUse(TCreate&& create) -> decltype(create()) {
        if (CreationAhead::Active() || Current()) return create();
        using Result = decltype(create());
        std::promise<Result> done;
        auto result = done.get_future();
        const auto started = std::chrono::steady_clock::now();
        Submit([&] {
            try {
                done.set_value(create());
            } catch (...) {
                done.set_exception(std::current_exception());
            }
        });
        result.wait();
        NoteWait(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
        return result.get();
    }

    static bool Current();
    static Counts Totals();
    static void Later(std::function<void()> task);
    static void WaitBackground();

private:
    static void Submit(std::function<void()> task, bool background = false);
    static void NoteWait(std::uint64_t nanoseconds);
};

class CreationGate {
public:
    static constexpr std::size_t CompilerBytesPerWord = 12288;
    static constexpr std::size_t MemoryLimitBytes = std::size_t{16} << 30u;
    using MemorySource = std::size_t (*)();

    explicit CreationGate(std::size_t words) : words(words) {
        auto& gate = State();
        std::unique_lock lock(gate.mutex);
        bool heldCounted = false;
        const auto fits = [&] {
            if (gate.inFlight == 0) return true;
            if (gate.inFlight + words > budgetWords(gate)) return false;
            if (memoryFits(gate, words)) return true;
            if (!heldCounted) {
                heldCounted = true;
                ++gate.heldByMemory;
            }
            return false;
        };
        if (CreationInBackground::BehindFrames()) {
            const auto ticket = gate.issued++;
            gate.changed.wait(lock, [&] { return ticket == gate.admitted && gate.framesWaiting == 0 && fits(); });
            ++gate.admitted;
        } else if (!fits()) {
            ++gate.framesWaiting;
            gate.changed.wait(lock, fits);
            --gate.framesWaiting;
        }
        if (gate.inFlight == 0 && !memoryFits(gate, words)) ++gate.overMemory;
        gate.inFlight += words;
        gate.peak = std::max(gate.peak, ++gate.creations);
        lock.unlock();
        gate.changed.notify_all();
    }

    ~CreationGate() {
        auto& gate = State();
        {
            std::lock_guard lock(gate.mutex);
            gate.inFlight -= words;
            --gate.creations;
        }
        gate.changed.notify_all();
    }

    CreationGate(const CreationGate&) = delete;
    CreationGate& operator=(const CreationGate&) = delete;

    static std::size_t InFlight() {
        auto& gate = State();
        std::lock_guard lock(gate.mutex);
        return gate.inFlight;
    }

    static std::size_t Creations() {
        auto& gate = State();
        std::lock_guard lock(gate.mutex);
        return gate.creations;
    }

    static std::size_t PeakCreations() {
        auto& gate = State();
        std::lock_guard lock(gate.mutex);
        return gate.peak;
    }

    static std::size_t EstimateBytes(std::size_t words) { return words * CompilerBytesPerWord; }
    static std::size_t DefaultBudgetBytes();

    static std::size_t BudgetWords() {
        auto& gate = State();
        std::lock_guard lock(gate.mutex);
        return budgetWords(gate);
    }

    static void SetBudgetBytes(std::size_t bytes) {
        auto& gate = State();
        {
            std::lock_guard lock(gate.mutex);
            gate.budgetBytes = bytes;
        }
        gate.changed.notify_all();
    }

    static void SetMemorySource(MemorySource source, std::size_t limitBytes = MemoryLimitBytes) {
        auto& gate = State();
        {
            std::lock_guard lock(gate.mutex);
            gate.memory = source;
            gate.memoryLimit = limitBytes;
        }
        gate.changed.notify_all();
    }

    static std::size_t RamSideBytes();
    static void UseRamSide(MemorySource deviceLocal);

    static std::size_t HeldByMemory() {
        auto& gate = State();
        std::lock_guard lock(gate.mutex);
        return gate.heldByMemory;
    }

    static std::size_t OverMemory() {
        auto& gate = State();
        std::lock_guard lock(gate.mutex);
        return gate.overMemory;
    }

private:
    struct Shared {
        std::mutex mutex;
        std::condition_variable changed;
        std::uint64_t issued = 0;
        std::uint64_t admitted = 0;
        std::size_t inFlight = 0;
        std::size_t creations = 0;
        std::size_t peak = 0;
        std::size_t budgetBytes = 0;
        std::size_t framesWaiting = 0;
        MemorySource memory = nullptr;
        std::size_t memoryLimit = MemoryLimitBytes;
        std::size_t heldByMemory = 0;
        std::size_t overMemory = 0;
    };

    static std::size_t budgetWords(const Shared& gate) {
        return std::max<std::size_t>(1, (gate.budgetBytes != 0 ? gate.budgetBytes : DefaultBudgetBytes()) / CompilerBytesPerWord);
    }

    static bool memoryFits(const Shared& gate, std::size_t words) {
        if (gate.memory == nullptr) return true;
        const auto used = gate.memory();
        const auto needed = EstimateBytes(gate.inFlight + words);
        return used <= gate.memoryLimit && needed <= gate.memoryLimit - used;
    }

    static Shared& State();

    std::size_t words;
};

}

#endif
