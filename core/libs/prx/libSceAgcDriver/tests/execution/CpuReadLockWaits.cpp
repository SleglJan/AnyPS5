#define SDL_MAIN_HANDLED
#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include <SDL.h>
#include <SDL_vulkan.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t WriterQueue = 0x20;
constexpr std::uint32_t LabelQueue = 0x28;
constexpr std::uint32_t SpinLanes = 64;
constexpr std::uint32_t SpinRounds = 1u << 22u;
constexpr std::uint32_t SpinAttempts = 4;
constexpr std::uint32_t VLshrRevB32 = 0x16u;
constexpr std::uint32_t VXorB32 = 0x1du;
constexpr std::uint32_t MoveLiteralToS9 = 0xbe8903ffu;
constexpr std::uint32_t DecrementS9 = 0x80898109u;
constexpr std::uint32_t CompareS9NotZero = 0xbf078009u;
constexpr std::uint32_t BranchBackWhileNotZero = 0xbf85fff7u;
constexpr double MeaningfulReadMs = 150.0;
constexpr double LockWaitLimitMs = 20.0;
constexpr double LabelLimitMs = 60.0;
constexpr auto SettleDelay = std::chrono::milliseconds(30);
constexpr std::uint64_t MarkerOffset = 1024;
constexpr std::uint64_t OwnStreamOffset = 2048;
constexpr std::uint64_t GateOffset = 2052;
constexpr std::uint64_t WaitFreeOffset = 3072;
constexpr std::uint32_t WaitEqual = 3;
constexpr auto RecordingHold = std::chrono::milliseconds(300);

std::array<Program, SpinAttempts> Spinners;
Program Storer;

void BuildPrograms() {
    using namespace Isa;
    Storer << Vop2(VLshlRevB32, 1, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 0, 1, 0) << EndProgram;
    for (std::uint32_t attempt = 0; attempt < SpinAttempts; ++attempt) {
        Spinners[attempt] << Vop2(VAddNcU32, 1, InlineInteger(1), 0) << MoveLiteralToS9 << (SpinRounds << attempt)
                          << Vop2(VLshlRevB32, 2, InlineInteger(13), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
                          << Vop2(VLshrRevB32, 2, InlineInteger(17), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
                          << Vop2(VLshlRevB32, 2, InlineInteger(5), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
                          << DecrementS9 << CompareS9NotZero << BranchBackWhileNotZero
                          << Vop2(VLshlRevB32, 3, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 1, 3, 0) << EndProgram;
    }
}

std::uint32_t Spun(std::uint32_t lane, std::uint32_t attempt) {
    std::uint32_t value = lane + 1u;
    for (std::uint32_t round = 0; round < (SpinRounds << attempt); ++round) {
        value ^= value << 13u;
        value ^= value >> 17u;
        value ^= value << 5u;
    }
    return value;
}

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

bool Recorded(std::uint64_t address) {
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr && recorder->LabelPendingAt(address);
}

double Ms(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

struct Outcome {
    double readMs = 0;
    double lockWaitMs = 0;
    double labelMs = -1;
    bool valuesSeen = false;
    bool recordedFirst = false;
    bool resident = false;
    bool pendingNoted = false;
};

Outcome ReadWhileOthersLock(std::uint64_t output, std::uint32_t attempt, PrivatePages& labels, std::uint32_t labelValue, const std::function<void(std::vector<std::uint32_t>&)>& read) {
    Outcome outcome;
    const auto marker = labels.Address(MarkerOffset);
    volatile auto* markerWord = labels.Words() + MarkerOffset / 4u;
    *markerWord = 0u;
    AgcDriver::GuestMemory::CollectWritesUncached(marker, 4u);
    Commands spin;
    spin.Dispatch(Spinners[attempt], BufferResource(output, SpinLanes * 4u), SpinLanes);
    spin.WriteData(marker, {labelValue});
    spin.Submit(WriterQueue);
    const auto recordStart = Clock::now();
    while (!Recorded(marker) && *markerWord != labelValue && Clock::now() - recordStart < std::chrono::seconds(10)) std::this_thread::yield();
    outcome.recordedFirst = *markerWord != labelValue && Recorded(marker);
    Locked([&] { outcome.resident = Residency().Manager().Slot(PageOf(output)).has_value(); outcome.pendingNoted = AgcDriver::Graphics::Recorder::Active()->PendingWriteOverlaps(output, SpinLanes * 4u); });
    std::atomic<bool> reading{false};
    std::vector<std::uint32_t> seen(SpinLanes, 0u);
    std::exception_ptr failure;
    std::jthread reader([&] {
        try {
            reading.store(true);
            const auto start = Clock::now();
            read(seen);
            outcome.readMs = Ms(Clock::now() - start);
        } catch (...) {
            failure = std::current_exception();
        }
    });
    while (!reading.load()) std::this_thread::yield();
    std::this_thread::sleep_for(SettleDelay);
    Commands label;
    label.WriteData(labels.Address(), {labelValue});
    const auto labelStart = Clock::now();
    label.Submit(LabelQueue);
    const auto lockStart = Clock::now();
    Locked([] {});
    outcome.lockWaitMs = Ms(Clock::now() - lockStart);
    while (labels.Words()[0] != labelValue && !Recorded(labels.Address()) && Clock::now() - labelStart < std::chrono::seconds(10)) std::this_thread::yield();
    if (labels.Words()[0] == labelValue || Recorded(labels.Address())) outcome.labelMs = Ms(Clock::now() - labelStart);
    reader.join();
    if (failure) std::rethrow_exception(failure);
    outcome.valuesSeen = true;
    for (std::uint32_t lane = 0; lane < SpinLanes; ++lane) outcome.valuesSeen = outcome.valuesSeen && seen[lane] == Spun(lane, attempt);
    WaitIdle();
    Settle();
    return outcome;
}

void DirectRead(std::uint64_t address, std::vector<std::uint32_t>& seen) {
    AgcDriver::GuestMemory::Read(address, std::as_writable_bytes(std::span(seen)), 4);
}

template <typename Pages>
void ReaderWaitsUnlocked(const std::string& what, Pages& pages, PrivatePages& labels, std::uint32_t& labelValue, const std::function<void(std::vector<std::uint32_t>&)>& read) {
    for (std::uint32_t attempt = 0; attempt < SpinAttempts; ++attempt) {
        for (std::uint32_t lane = 0; lane < SpinLanes; ++lane) pages.Words()[lane] = 0u;
        AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), SpinLanes * 4u);
        Settle();
        const auto outcome = ReadWhileOthersLock(pages.Address(), attempt, labels, ++labelValue, read);
        if (!outcome.recordedFirst) continue;
        Expect(outcome.valuesSeen, what + ": the CPU read returned other values than the spinning dispatch stored (it did not wait for the GPU)");
        if (outcome.readMs < MeaningfulReadMs) continue;
        std::printf("%s: read %.1f ms, lock wait %.1f ms, label recorded after %.1f ms (resident %d, noted %d)\n", what.c_str(), outcome.readMs, outcome.lockWaitMs, outcome.labelMs, outcome.resident, outcome.pendingNoted);
        Expect(outcome.lockWaitMs < LockWaitLimitMs, what + ": a thread taking the device lock waited " + std::to_string(outcome.lockWaitMs) + " ms while a CPU read waited " + std::to_string(outcome.readMs) + " ms for the GPU");
        Expect(outcome.labelMs >= 0 && outcome.labelMs < LabelLimitMs, what + ": another queue's label was recorded after " + std::to_string(outcome.labelMs) + " ms while a CPU read waited " + std::to_string(outcome.readMs) + " ms for the GPU");
        return;
    }
    Expect(false, what + ": in all " + std::to_string(SpinAttempts) + " attempts the spinning dispatch finished before it was seen recorded or before the CPU read could be observed waiting");
}

void DirtyMirrorWaitsUnlocked(PrivatePages& labels, std::uint32_t& labelValue) {
    const std::string what = "a dirty mirrored page behind a running dispatch";
    for (std::uint32_t attempt = 0; attempt < SpinAttempts; ++attempt) {
        SharedPages mirrored(1);
        PrivatePages busy(1);
        for (std::uint32_t lane = 0; lane < SpinLanes; ++lane) {
            mirrored.Words()[lane] = 0u;
            busy.Words()[lane] = 0u;
        }
        AgcDriver::GuestMemory::CollectWritesUncached(mirrored.Address(), SpinLanes * 4u);
        AgcDriver::GuestMemory::CollectWritesUncached(busy.Address(), SpinLanes * 4u);
        Settle();
        Commands store;
        store.Dispatch(Storer, BufferResource(mirrored.Address(), SpinLanes * 4u), SpinLanes);
        store.Submit(WriterQueue);
        WaitIdle();
        Settle();
        bool dirty = false;
        Locked([&] { dirty = Residency().Manager().DirtyIn(mirrored.Address(), SpinLanes * 4u); });
        if (!dirty) {
            Expect(false, what + ": the GPU's store did not stay in the page's mirror");
            return;
        }
        std::vector<std::uint32_t> stored(SpinLanes, 0u);
        double mirroredMs = 0;
        const auto outcome = ReadWhileOthersLock(busy.Address(), attempt, labels, ++labelValue, [&](std::vector<std::uint32_t>& seen) {
            const auto start = Clock::now();
            DirectRead(mirrored.Address(), stored);
            mirroredMs = Ms(Clock::now() - start);
            DirectRead(busy.Address(), seen);
        });
        bool storedSeen = true;
        for (std::uint32_t lane = 0; lane < SpinLanes; ++lane) storedSeen = storedSeen && stored[lane] == lane;
        Expect(storedSeen, what + ": the CPU read did not return the values the GPU stored into the mirror");
        if (!outcome.recordedFirst || mirroredMs < MeaningfulReadMs) continue;
        Expect(outcome.valuesSeen, what + ": the CPU read returned other values than the spinning dispatch stored (it did not wait for the GPU)");
        std::printf("%s: mirrored read %.1f ms, lock wait %.1f ms, label recorded after %.1f ms\n", what.c_str(), mirroredMs, outcome.lockWaitMs, outcome.labelMs);
        Expect(outcome.lockWaitMs < LockWaitLimitMs, what + ": a thread taking the device lock waited " + std::to_string(outcome.lockWaitMs) + " ms while the read of a dirty mirrored page waited " + std::to_string(mirroredMs) + " ms for its write-back");
        Expect(outcome.labelMs >= 0 && outcome.labelMs < LabelLimitMs, what + ": another queue's label was recorded after " + std::to_string(outcome.labelMs) + " ms while the read of a dirty mirrored page waited " + std::to_string(mirroredMs) + " ms for its write-back");
        return;
    }
    Expect(false, what + ": in all " + std::to_string(SpinAttempts) + " attempts the read of the dirty mirrored page did not wait behind the spinning dispatch");
}

void SettledWriterReadSkipsDeviceLock(PrivatePages& labels, std::uint32_t& labelValue) {
    constexpr auto HoldTime = std::chrono::milliseconds(300);
    const std::string what = "a CPU read of a mirrored page whose GPU writer and write-back have settled";
    SharedPages mirrored(1);
    for (std::uint32_t lane = 0; lane < SpinLanes; ++lane) mirrored.Words()[lane] = 0u;
    AgcDriver::GuestMemory::CollectWritesUncached(mirrored.Address(), SpinLanes * 4u);
    Settle();
    const auto marker = labels.Address(MarkerOffset);
    volatile auto* markerWord = labels.Words() + MarkerOffset / 4u;
    *markerWord = 0u;
    AgcDriver::GuestMemory::CollectWritesUncached(marker, 4u);
    const auto value = ++labelValue;
    Commands store;
    store.Dispatch(Storer, BufferResource(mirrored.Address(), SpinLanes * 4u), SpinLanes);
    store.WriteData(marker, {value});
    store.Submit(WriterQueue);
    const auto start = Clock::now();
    while (*markerWord != value && Clock::now() - start < std::chrono::seconds(10)) std::this_thread::yield();
    Expect(*markerWord == value, what + ": the store's label did not land");
    bool dirty = false;
    Locked([&] { dirty = Residency().Manager().DirtyIn(mirrored.Address(), SpinLanes * 4u); });
    if (!dirty) {
        Expect(false, what + ": the GPU's store did not stay in the page's mirror");
        return;
    }
    bool settledBefore = true;
    Locked([&] {
        AgcDriver::Graphics::WriteBackAllMirrors();
        dirty = Residency().Manager().DirtyIn(mirrored.Address(), SpinLanes * 4u);
        settledBefore = Residency().Manager().CpuReadSettled(mirrored.Address(), SpinLanes * 4u);
    });
    if (dirty || settledBefore) {
        Expect(false, what + ": after the write-back the page should be clean with its writer still recorded (dirty " + std::to_string(dirty) + ", settled " + std::to_string(settledBefore) + ")");
        return;
    }
    std::atomic<bool> holding{false};
    std::jthread holder([&] {
        Locked([&] {
            holding.store(true);
            std::this_thread::sleep_for(HoldTime);
        });
    });
    while (!holding.load()) std::this_thread::yield();
    std::vector<std::uint32_t> seen(SpinLanes, 0u);
    const auto readStart = Clock::now();
    DirectRead(mirrored.Address(), seen);
    const auto readMs = Ms(Clock::now() - readStart);
    holder.join();
    bool storedSeen = true;
    for (std::uint32_t lane = 0; lane < SpinLanes; ++lane) storedSeen = storedSeen && seen[lane] == lane;
    Expect(storedSeen, what + ": the CPU read did not return the values the GPU stored");
    bool settledAfter = false;
    Locked([&] { settledAfter = Residency().Manager().CpuReadSettled(mirrored.Address(), SpinLanes * 4u); });
    Expect(settledAfter, what + ": the read left the settled writer recorded");
    std::printf("%s: read %.1f ms while another thread held the device lock for %lld ms\n", what.c_str(), readMs, static_cast<long long>(HoldTime.count()));
    Expect(readMs < LockWaitLimitMs, what + ": the read waited " + std::to_string(readMs) + " ms for a thread holding the device lock, although it needed only residency work");
    WaitIdle();
    Settle();
}

constexpr std::uint32_t DisplayExtent = 64;
constexpr std::uint64_t DisplayBgra8 = 0x8000000000000000ull;

class Window {
public:
    Window() {
        SDL_SetMainReady();
        if (SDL_Init(SDL_INIT_VIDEO) != 0) return;
        window = SDL_CreateWindow("lock waits", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, DisplayExtent, DisplayExtent, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
        if (window == nullptr) return;
        unsigned count = 0;
        if (SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr) != SDL_TRUE) return;
        extensions.resize(count);
        if (SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data()) != SDL_TRUE) return;
        presentation = AgcDriver::PresentationWindow{window, extensions, [](void* context, VkInstance instance) {
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            Require(SDL_Vulkan_CreateSurface(static_cast<SDL_Window*>(context), instance, &surface) == SDL_TRUE, SDL_GetError());
            return surface;
        }, [](void*, std::uint32_t* width, std::uint32_t* height) {
            *width = DisplayExtent;
            *height = DisplayExtent;
        }, DisplayExtent, DisplayExtent, nullptr};
        AgcDriverAttachWindow_nid_postfix(presentation);
        attached = true;
    }

    ~Window() {
        if (attached) AgcDriverReleaseWindow_nid_postfix(window);
        if (window != nullptr) SDL_DestroyWindow(window);
        SDL_Quit();
    }

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool Attached() const { return attached; }

    void Present(const AgcDriver::DisplayBuffer* buffer) {
        std::atomic<bool> ready{false};
        const auto done = [](void* context) { static_cast<std::atomic<bool>*>(context)->store(true); };
        if (buffer != nullptr) AgcDriverPresentBuffer_nid_postfix(presentation, *buffer, done, &ready);
        else AgcDriverPresentClear_nid_postfix(presentation, true, done, &ready);
        Require(ready.load(), "the presentation did not report its completion");
    }

private:
    SDL_Window* window = nullptr;
    std::vector<const char*> extensions;
    AgcDriver::PresentationWindow presentation{};
    bool attached = false;
};

void PresenterWaitsUnlocked(Window& window, PrivatePages& labels, std::uint32_t& labelValue) {
    PrivatePages display(1);
    std::memset(display.Data(), 0, display.Bytes());
    window.Present(nullptr);
    const AgcDriver::DisplayBuffer buffer{display.Address(), DisplayBgra8, DisplayExtent, DisplayExtent, 0, 0};
    ReaderWaitsUnlocked("a presentation", display, labels, labelValue, [&](std::vector<std::uint32_t>& seen) {
        window.Present(&buffer);
        for (std::uint32_t lane = 0; lane < SpinLanes; ++lane) seen[lane] = display.Words()[lane];
    });
}

bool LandsWithin(volatile std::uint32_t* word, std::uint32_t value, Clock::duration limit) {
    const auto start = Clock::now();
    while (*word != value && Clock::now() - start < limit) std::this_thread::yield();
    return *word == value;
}

void LabelBesideOtherQueueRecording(PrivatePages& labels, std::uint32_t& labelValue) {
    const auto address = labels.Address(OwnStreamOffset);
    volatile auto* word = labels.Words() + OwnStreamOffset / 4u;
    const auto gate = labels.Address(GateOffset);
    labels.Words()[GateOffset / 4u] = 0x6a7e0001u;
    AgcDriver::GuestMemory::CollectWritesUncached(gate, 4u);
    Locked([] {
        auto* recorder = AgcDriver::Graphics::Recorder::Active();
        recorder->DeclareSection({});
        static_cast<void>(recorder->Commands());
    });
    Commands warm;
    warm.Wait(gate, WaitEqual, 0x6a7e0001u, 0xffffffffu);
    warm.WriteData(address, {++labelValue});
    warm.Submit(WriterQueue);
    Expect(LandsWithin(word, labelValue, std::chrono::seconds(10)), "a label beside another queue's recording: the first label did not land");
    WaitIdle();
    Settle();
    for (const bool barrier : {false, true}) {
        const std::string what = barrier ? "a label after another queue's barrier section" : "a label beside another queue's open batch";
        for (std::uint32_t attempt = 0; attempt < 3; ++attempt) {
            std::atomic<bool> holding{false};
            Clock::time_point released{};
            std::jthread holder([&] {
                Locked([&] {
                    auto* recorder = AgcDriver::Graphics::Recorder::Active();
                    if (!barrier) recorder->DeclareSection({});
                    static_cast<void>(recorder->Commands());
                    holding.store(true);
                    std::this_thread::sleep_for(RecordingHold);
                    released = Clock::now();
                });
            });
            while (!holding.load()) std::this_thread::yield();
            const auto value = ++labelValue;
            Commands label;
            label.Wait(gate, WaitEqual, 0x6a7e0001u, 0xffffffffu);
            label.WriteData(address, {value});
            const auto start = Clock::now();
            label.Submit(WriterQueue);
            const bool landed = LandsWithin(word, value, std::chrono::seconds(10));
            const auto landedAt = Clock::now();
            holder.join();
            const auto landedMs = Ms(landedAt - start);
            const auto holdLeftMs = Ms(released - landedAt);
            std::printf("%s (attempt %u): label landed after %.1f ms, %.1f ms before the device lock was released\n", what.c_str(), attempt, landedMs, holdLeftMs);
            Expect(landed, what + ": the label did not land within 10 s");
            if (barrier) Expect(landedAt >= released, what + ": the label landed " + std::to_string(holdLeftMs) + " ms before the barrier section's batch could be submitted");
            else Expect(landedMs < LabelLimitMs && landedAt < released, what + ": the label landed after " + std::to_string(landedMs) + " ms, behind the device lock held by another queue's recording");
            WaitIdle();
            Settle();
        }
    }
}

void WaitFreeOrderBesideRecording(PrivatePages& labels, std::uint32_t& labelValue) {
    const auto address = labels.Address(WaitFreeOffset);
    volatile auto* word = labels.Words() + WaitFreeOffset / 4u;
    const std::string what = "a wait-free submission ordered after queue 0's open batch";
    for (std::uint32_t attempt = 0; attempt < 3; ++attempt) {
        std::atomic<bool> holding{false};
        Clock::time_point released{};
        AgcDriver::Graphics::Recorder::Token awaited{};
        std::jthread holder([&] {
            Locked([&] {
                auto* recorder = AgcDriver::Graphics::Recorder::Active();
                recorder->DeclareSection({});
                static_cast<void>(recorder->Commands());
                std::uint32_t stream = 0;
                std::uint64_t value = 0;
                Require(recorder->CallerBatchValue(stream, value) && stream == 0, what + ": queue 0 has no open batch");
                awaited[0] = value;
                holding.store(true);
                std::this_thread::sleep_for(RecordingHold);
                released = Clock::now();
            });
        });
        while (!holding.load()) std::this_thread::yield();
        const auto value = ++labelValue;
        Commands label;
        label.WriteData(address, {value});
        const auto start = Clock::now();
        label.Submit(WriterQueue);
        while (*word != value && !Recorded(address) && Clock::now() - start < std::chrono::seconds(10)) std::this_thread::yield();
        const auto recordedAt = Clock::now();
        const bool landed = LandsWithin(word, value, std::chrono::seconds(10));
        const auto landedAt = Clock::now();
        const bool queue0Done = AgcDriver::Graphics::Recorder::Active()->TokenSignaled(awaited);
        holder.join();
        const auto recordedMs = Ms(recordedAt - start);
        std::printf("%s (attempt %u): recorded after %.1f ms, %.1f ms before the device lock was released; landed %.1f ms after the release\n", what.c_str(), attempt, recordedMs, Ms(released - recordedAt), Ms(landedAt - released));
        Expect(recordedMs < LabelLimitMs && recordedAt < released, what + ": recorded after " + std::to_string(recordedMs) + " ms, behind the device lock held by queue 0's recording");
        Expect(landed, what + ": the label did not land within 10 s");
        Expect(landedAt >= released && queue0Done, what + ": the label landed before queue 0's open batch was submitted and finished");
        WaitIdle();
        Settle();
    }
}

void ThreeQueueLoad(PrivatePages& labels) {
    constexpr std::uint32_t Rounds = 40;
    constexpr std::array<std::uint32_t, 3> Queues{0x0u, WriterQueue, LabelQueue};
    PrivatePages targets(static_cast<std::uint32_t>(Queues.size()));
    SharedPages shared(static_cast<std::uint32_t>(Queues.size()));
    Settle();
    std::atomic<bool> done{false};
    std::atomic<std::uint64_t> reads{0};
    std::exception_ptr failure;
    std::jthread reader([&](std::stop_token stop) {
        try {
            std::uint32_t word = 0;
            while (!done.load() && !stop.stop_requested()) {
                for (std::uint32_t index = 0; index < Queues.size(); ++index) {
                    AgcDriver::GuestMemory::Read(targets.Address(index * PageBytes), std::as_writable_bytes(std::span(&word, 1)), 4);
                    AgcDriver::GuestMemory::Read(shared.Address(index), std::as_writable_bytes(std::span(&word, 1)), 4);
                    reads.fetch_add(1);
                }
            }
        } catch (...) {
            failure = std::current_exception();
        }
    });
    const auto start = Clock::now();
    for (std::uint32_t round = 1; round <= Rounds; ++round) {
        for (std::uint32_t index = 0; index < Queues.size(); ++index) {
            Commands commands;
            commands.Dispatch(Storer, BufferResource(shared.Address(index), SpinLanes * 4u), SpinLanes);
            commands.WriteData(targets.Address(index * PageBytes), {round});
            commands.WriteData(labels.Address(64u + index * 4u), {round});
            commands.Submit(Queues[index]);
        }
    }
    bool landed = false;
    while (Clock::now() - start < std::chrono::seconds(120)) {
        landed = true;
        for (std::uint32_t index = 0; index < Queues.size(); ++index) landed = landed && labels.Words()[16u + index] == Rounds;
        if (landed) break;
        std::this_thread::yield();
    }
    done.store(true);
    reader.join();
    if (failure) std::rethrow_exception(failure);
    Expect(landed, "three-queue load: the queues' last labels did not land within 120 s (a deadlock between the CPU reads and the recording queues)");
    WaitIdle();
    Settle();
    for (std::uint32_t index = 0; index < Queues.size(); ++index) {
        std::uint32_t word = 0;
        AgcDriver::GuestMemory::Read(targets.Address(index * PageBytes), std::as_writable_bytes(std::span(&word, 1)), 4);
        Expect(word == Rounds, "three-queue load: queue " + Hex(Queues[index]) + "'s last store reads " + Hex(word));
    }
    Expect(reads.load() != 0, "three-queue load: the reader made no read");
}

}

#endif

int main() {
#ifdef _WIN32
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    try {
        BuildPrograms();
        auto window = std::make_unique<Window>();
        PrivatePages labels(1);
        std::uint32_t labelValue = 0x51000000u;
        {
            MirrorReader reader;
            PrivatePages warm(1);
            reader.Read(warm.Address(), 1);
        }
        Locked([] { Residency().Manager().SetBudget(std::uint64_t{1} << 30); });
        {
            SharedPages mirrored(1);
            ReaderWaitsUnlocked("a mirrored page", mirrored, labels, labelValue, [&](std::vector<std::uint32_t>& seen) { DirectRead(mirrored.Address(), seen); });
        }
        DirtyMirrorWaitsUnlocked(labels, labelValue);
        SettledWriterReadSkipsDeviceLock(labels, labelValue);
        {
            PrivatePages pending(1);
            ReaderWaitsUnlocked("a pending write", pending, labels, labelValue, [&](std::vector<std::uint32_t>& seen) { DirectRead(pending.Address(), seen); });
        }
        if (window->Attached()) PresenterWaitsUnlocked(*window, labels, labelValue);
        else std::printf("a presentation: skipped, no window could be created\n");
        LabelBesideOtherQueueRecording(labels, labelValue);
        WaitFreeOrderBesideRecording(labels, labelValue);
        ThreeQueueLoad(labels);
        Locked([] { Residency().Manager().SetBudget(0); });
        window.reset();
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d CPU read lock wait checks failed\n", failures);
        return 1;
    }
    std::printf("CPU read lock wait tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
