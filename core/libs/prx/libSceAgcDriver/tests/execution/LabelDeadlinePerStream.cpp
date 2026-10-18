#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t GraphicsQueue = 0;
constexpr std::uint32_t LabelQueue = 0x20;
constexpr std::uint32_t OtherQueue = 0x28;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t WaitEqual = 3;
constexpr std::uint32_t Open = 0x6a7e0001u;
constexpr std::uint64_t LabelGateOffset = 0;
constexpr std::uint64_t OtherGateOffset = 4;
constexpr std::uint64_t GraphicsGateOffset = 8;
constexpr std::uint64_t OtherDoneOffset = 12;
constexpr std::uint64_t GraphicsMarkOffset = 16;
constexpr std::uint64_t GraphicsDoneOffset = 20;
constexpr std::uint64_t LabelOffset = 1024;
constexpr auto SettleDelay = std::chrono::milliseconds(50);
constexpr auto PastDeadline = std::chrono::milliseconds(30);

Program Storer;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

AgcDriver::Graphics::Recorder& ActiveRecorder() {
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    Require(recorder != nullptr, "label deadline per stream: the device has no recorder");
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

std::vector<std::uint64_t> OpenBatches(const AgcDriver::Graphics::Recorder& recorder) {
    std::vector<std::uint64_t> values(recorder.Streams());
    for (std::uint32_t stream = 0; stream < values.size(); ++stream) values[stream] = recorder.OpenBatchValue(stream);
    return values;
}

void Warm(PrivatePages& labels, PrivatePages& gates, PrivatePages& stored, std::uint32_t& labelValue) {
    SetWord(gates, LabelGateOffset, Open);
    for (const auto queue : {GraphicsQueue, OtherQueue}) {
        Commands dispatch;
        dispatch.Dispatch(Storer, BufferResource(stored.Address(), Lanes * 4u), Lanes);
        dispatch.Submit(queue);
        WaitIdle();
    }
    Commands label;
    label.Wait(gates.Address(LabelGateOffset), WaitEqual, Open, 0xffffffffu);
    label.WriteData(labels.Address(LabelOffset), {++labelValue});
    label.Submit(LabelQueue);
    Require(LandsWithin(labels.Words() + LabelOffset / 4u, labelValue, std::chrono::seconds(10)), "label deadline per stream: the first label did not land");
    WaitIdle();
    Settle();
}

void LabelWaitLeavesOtherStreamOpen(PrivatePages& labels, PrivatePages& gates, PrivatePages& stored, std::uint32_t& labelValue) {
    const std::string what = "queue 0 waiting on queue 0x20's label beside queue 0x28's unrelated open batch";
    auto& recorder = ActiveRecorder();
    for (std::uint32_t attempt = 0; attempt < 3; ++attempt) {
        const auto done = attempt + 1u;
        const auto value = ++labelValue;
        for (const auto offset : {LabelGateOffset, OtherGateOffset, GraphicsGateOffset}) SetWord(gates, offset, 0u);

        Commands other;
        other.Dispatch(Storer, BufferResource(stored.Address(), Lanes * 4u), Lanes);
        other.Wait(gates.Address(OtherGateOffset), WaitEqual, Open, 0xffffffffu);
        other.WriteData(gates.Address(OtherDoneOffset), {done});
        other.Submit(OtherQueue);

        std::vector<std::uint64_t> before;
        const auto start = Clock::now();
        auto anyOpen = [&] {
            before = OpenBatches(recorder);
            for (std::uint32_t stream = 1; stream < before.size(); ++stream) {
                if (before[stream] != 0) return true;
            }
            return false;
        };
        while (!anyOpen() && Clock::now() - start < std::chrono::seconds(10)) std::this_thread::yield();
        Require(anyOpen(), what + ": queue 0x28 opened no batch for its dispatch");
        std::this_thread::sleep_for(SettleDelay);
        Require(OpenBatches(recorder) == before, what + ": queue 0x28's batch was submitted before the label");

        Commands graphics;
        graphics.Wait(labels.Address(LabelOffset), WaitEqual, value, 0xffffffffu);
        graphics.WriteData(gates.Address(GraphicsMarkOffset), {done});
        graphics.Wait(gates.Address(GraphicsGateOffset), WaitEqual, Open, 0xffffffffu);
        graphics.WriteData(gates.Address(GraphicsDoneOffset), {done});
        graphics.Submit(GraphicsQueue);

        Commands label;
        label.Wait(gates.Address(LabelGateOffset), WaitEqual, Open, 0xffffffffu);
        label.WriteData(labels.Address(LabelOffset), {value});
        label.Submit(LabelQueue);
        std::this_thread::sleep_for(SettleDelay);
        SetWord(gates, LabelGateOffset, Open);

        const bool landed = LandsWithin(labels.Words() + LabelOffset / 4u, value, std::chrono::seconds(10));
        const bool marked = LandsWithin(gates.Words() + GraphicsMarkOffset / 4u, done, std::chrono::seconds(10));
        std::this_thread::sleep_for(PastDeadline);
        const auto after = OpenBatches(recorder);
        std::uint32_t submitted = 0;
        for (std::uint32_t stream = 1; stream < before.size(); ++stream) submitted += before[stream] != 0 && after[stream] != before[stream] ? 1u : 0u;
        const bool otherWaiting = gates.Words()[OtherDoneOffset / 4u] != done;
        std::printf("%s (attempt %u): label %s, queue 0 %s past it; %u unrelated open batches submitted while queue 0x28 %s\n", what.c_str(), attempt, landed ? "landed" : "did not land", marked ? "went" : "did not go", submitted,
                    otherWaiting ? "waited" : "had finished");
        Expect(landed, what + ": the label did not land");
        Expect(marked, what + ": queue 0 did not pass its wait on the label");
        Expect(otherWaiting, what + ": queue 0x28 did not wait for its gate");
        Expect(submitted == 0u, what + ": the label deadline submitted queue 0x28's unrelated open batch");

        SetWord(gates, OtherGateOffset, Open);
        SetWord(gates, GraphicsGateOffset, Open);
        Expect(LandsWithin(gates.Words() + OtherDoneOffset / 4u, done, std::chrono::seconds(10)), what + ": queue 0x28 did not finish after its gate opened");
        Expect(LandsWithin(gates.Words() + GraphicsDoneOffset / 4u, done, std::chrono::seconds(10)), what + ": queue 0 did not finish after its gate opened");
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
        PrivatePages gates(1);
        PrivatePages stored(1);
        std::uint32_t labelValue = 0x55000000u;
        Warm(labels, gates, stored, labelValue);
        Require(ActiveRecorder().Streams() > 2, "label deadline per stream: the device has fewer than three recorder streams");
        LabelWaitLeavesOtherStreamOpen(labels, gates, stored, labelValue);
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d label deadline checks failed\n", failures);
        return 1;
    }
    std::printf("label deadline per stream tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
