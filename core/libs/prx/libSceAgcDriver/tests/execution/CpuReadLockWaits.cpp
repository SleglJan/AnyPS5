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
        {
            PrivatePages pending(1);
            ReaderWaitsUnlocked("a pending write", pending, labels, labelValue, [&](std::vector<std::uint32_t>& seen) { DirectRead(pending.Address(), seen); });
        }
        if (window->Attached()) PresenterWaitsUnlocked(*window, labels, labelValue);
        else std::printf("a presentation: skipped, no window could be created\n");
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
