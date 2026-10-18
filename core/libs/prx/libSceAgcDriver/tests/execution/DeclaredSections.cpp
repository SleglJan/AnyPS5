#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>

#ifdef _WIN32

namespace {

using namespace NewPath;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t GraphicsQueue = 0;
constexpr std::uint32_t LabelQueue = 0x20;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t WaitEqual = 3;
constexpr std::uint32_t Open = 0x6a7e0001u;
constexpr std::uint64_t GateOffset = 0;
constexpr std::uint64_t HeldGateOffset = 4;
constexpr std::uint64_t DoneOffset = 8;
constexpr std::uint64_t LabelOffset = 1024;
constexpr auto RecordingHold = std::chrono::milliseconds(300);
constexpr auto SettleDelay = std::chrono::milliseconds(50);

Program Storer;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

double Ms(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

AgcDriver::Graphics::Recorder& ActiveRecorder() {
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    Require(recorder != nullptr, "declared sections: the device has no recorder");
    return *recorder;
}

bool LandsWithin(volatile std::uint32_t* word, std::uint32_t value, Clock::duration limit) {
    const auto start = Clock::now();
    while (*word != value && Clock::now() - start < limit) std::this_thread::yield();
    return *word == value;
}

void SetWord(PrivatePages& pages, std::uint64_t offset, std::uint32_t value) {
    pages.Words()[offset / 4u] = value;
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(offset), 4u);
}

void SubmitLabel(PrivatePages& labels, std::uint32_t value) {
    Commands label;
    label.Wait(labels.Address(GateOffset), WaitEqual, Open, 0xffffffffu);
    label.WriteData(labels.Address(LabelOffset), {value});
    label.Submit(LabelQueue);
}

void WarmLabelQueue(PrivatePages& labels, PrivatePages& stored, std::uint32_t& labelValue) {
    Commands dispatch;
    dispatch.Dispatch(Storer, BufferResource(stored.Address(), Lanes * 4u), Lanes);
    dispatch.Submit(GraphicsQueue);
    WaitIdle();
    SubmitLabel(labels, ++labelValue);
    Require(LandsWithin(labels.Words() + LabelOffset / 4u, labelValue, std::chrono::seconds(10)), "declared sections: the first label did not land");
    WaitIdle();
    Settle();
}

void LabelBesidePageTableDispatch(PrivatePages& labels, PrivatePages& stored, PrivatePages& gates, std::uint32_t& labelValue) {
    const std::string what = "a label on another queue beside queue 0's open batch holding a page-table dispatch";
    auto& recorder = ActiveRecorder();
    for (std::uint32_t attempt = 0; attempt < 3; ++attempt) {
        SetWord(gates, HeldGateOffset, 0u);
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) stored.Words()[lane] = 0u;
        AgcDriver::GuestMemory::CollectWritesUncached(stored.Address(), Lanes * 4u);
        const auto done = attempt + 1u;
        Commands held;
        held.Dispatch(Storer, BufferResource(stored.Address(), Lanes * 4u), Lanes);
        held.Wait(gates.Address(HeldGateOffset), WaitEqual, Open, 0xffffffffu);
        held.WriteData(gates.Address(DoneOffset), {done});
        held.Submit(GraphicsQueue);
        std::uint32_t stream = 0;
        std::uint64_t open = 0;
        const auto start = Clock::now();
        while (!recorder.CallerBatchValue(stream, open) && Clock::now() - start < std::chrono::seconds(10)) std::this_thread::yield();
        Require(stream == 0 && open != 0, what + ": queue 0 opened no batch for its dispatch");
        std::this_thread::sleep_for(SettleDelay);
        std::uint64_t current = 0;
        Require(recorder.CallerBatchValue(stream, current) && current == open, what + ": queue 0's batch was submitted before the label");
        const auto orderSubmits = recorder.OrderSubmits();
        const auto value = ++labelValue;
        const auto labelStart = Clock::now();
        SubmitLabel(labels, value);
        const bool landed = LandsWithin(labels.Words() + LabelOffset / 4u, value, std::chrono::seconds(10));
        const auto landedMs = Ms(Clock::now() - labelStart);
        const bool forced = recorder.OrderSubmits() != orderSubmits;
        const bool queue0Waiting = gates.Words()[DoneOffset / 4u] != done;
        std::printf("%s (attempt %u): label landed after %.1f ms; its section %s queue 0's open batch while queue 0 %s\n", what.c_str(), attempt, landedMs, forced ? "submitted" : "did not submit", queue0Waiting ? "waited" : "had finished");
        Expect(landed, what + ": the label did not land");
        Expect(queue0Waiting, what + ": queue 0 did not wait for its gate");
        Expect(!forced, what + ": the label's section submitted queue 0's open batch");
        SetWord(gates, HeldGateOffset, Open);
        Expect(LandsWithin(gates.Words() + DoneOffset / 4u, done, std::chrono::seconds(10)), what + ": queue 0 did not finish after its gate opened");
        WaitIdle();
        Settle();
        std::uint32_t wrong = 0;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            std::uint32_t word = 0;
            AgcDriver::GuestMemory::Read(stored.Address(lane * 4u), std::as_writable_bytes(std::span(&word, 1)), 4);
            wrong += word != lane ? 1u : 0u;
        }
        Expect(wrong == 0u, what + ": the dispatch stored " + std::to_string(wrong) + " wrong words");
    }
}

void OwnStreamStoreKeepsDeclaration(PrivatePages& labels, std::uint32_t& labelValue) {
    const std::string what = "an own-stream label between a lock holder's declaration and its section";
    auto& recorder = ActiveRecorder();
    const int first = 0;
    const int declaredObject = 0;
    for (std::uint32_t attempt = 0; attempt < 3; ++attempt) {
        SubmitLabel(labels, ++labelValue);
        Require(LandsWithin(labels.Words() + LabelOffset / 4u, labelValue, std::chrono::seconds(10)), what + ": the label before the holder did not land");
        WaitIdle();
        std::atomic<bool> holding{false};
        std::atomic<bool> between{false};
        std::atomic<bool> betweenLanded{false};
        Clock::time_point released{};
        std::jthread holder([&] {
            Locked([&] {
                const std::array<AgcDriver::Graphics::Recorder::SectionObject, 1> opened{{{&first, true}}};
                recorder.DeclareSection(opened);
                static_cast<void>(recorder.Commands());
                const std::array<AgcDriver::Graphics::Recorder::SectionObject, 1> objects{{{&declaredObject, true}}};
                recorder.DeclareSection(objects);
                between.store(true);
                const auto value = labelValue + 1u;
                betweenLanded.store(LandsWithin(labels.Words() + LabelOffset / 4u, value, std::chrono::seconds(2)));
                static_cast<void>(recorder.Commands());
                holding.store(true);
                std::this_thread::sleep_for(RecordingHold);
                released = Clock::now();
            });
        });
        while (!between.load()) std::this_thread::yield();
        SubmitLabel(labels, ++labelValue);
        while (!holding.load()) std::this_thread::yield();
        const auto value = ++labelValue;
        const auto start = Clock::now();
        SubmitLabel(labels, value);
        const bool landed = LandsWithin(labels.Words() + LabelOffset / 4u, value, std::chrono::seconds(10));
        const auto landedAt = Clock::now();
        holder.join();
        const auto landedMs = Ms(landedAt - start);
        std::printf("%s (attempt %u): the label in between %s; the next label landed after %.1f ms, %.1f ms before the device lock was released\n", what.c_str(), attempt, betweenLanded.load() ? "landed" : "did not land", landedMs, Ms(released - landedAt));
        Expect(betweenLanded.load(), what + ": the label in between was not recorded on its own stream while the lock was held");
        Expect(landed && landedAt < released, what + ": the next label landed after " + std::to_string(landedMs) + " ms, behind the holder's section, which became a barrier");
        WaitIdle();
        Settle();
    }
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
        using namespace Isa;
        Storer << Vop2(VLshlRevB32, 1, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 0, 1, 0) << EndProgram;
        PrivatePages labels(1);
        PrivatePages stored(1);
        PrivatePages gates(1);
        std::uint32_t labelValue = 0x53000000u;
        SetWord(labels, GateOffset, Open);
        WarmLabelQueue(labels, stored, labelValue);
        Require(ActiveRecorder().Streams() > 1, "declared sections: the device records every queue on one stream");
        LabelBesidePageTableDispatch(labels, stored, gates, labelValue);
        OwnStreamStoreKeepsDeclaration(labels, labelValue);
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d declared section checks failed\n", failures);
        return 1;
    }
    std::printf("declared section tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
