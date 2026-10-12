#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t OutputBytes = Threads * 16u * 4u;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint8_t Fill = 0xcd;
constexpr std::uint32_t Label = 0x600d1abeu;

alignas(256) constexpr std::array<std::uint32_t, 78> LoadsCode{
    0x34020084, 0x34040086, 0xdc388000, 0x04080001, 0xbf8c3f70, 0x7e1402ff, 0x5a5a0000, 0x7e1602ff,
    0x5a5a0001, 0x7e1802ff, 0x5a5a0002, 0x7e1a02ff, 0x5a5a0003, 0x7e1c02ff, 0x5a5a0004, 0x7e1e02ff,
    0x5a5a0005, 0x7e2002ff, 0x5a5a0006, 0x7e2202ff, 0x5a5a0007, 0x7e2402ff, 0x5a5a0008, 0x7e2602ff,
    0x5a5a0009, 0x7e2802ff, 0x5a5a000a, 0x7e2a02ff, 0x5a5a000b, 0x7e2c02ff, 0x5a5a000c, 0x7e2e02ff,
    0x5a5a000d, 0x7e3002ff, 0x5a5a000e, 0x7e3202ff, 0x5a5a000f, 0x4a4008ff, 0x00000a00, 0x4a420eff,
    0x00000200, 0xd70f6a22, 0x02024008, 0x7e460209, 0x50464680, 0x4a4c0aff, 0x00000200, 0xd70f6a24,
    0x02024c08, 0x7e4a0209, 0x504a4a80, 0x36500c81, 0x7daa5080, 0xdc208800, 0x0a080020, 0xdc2487ff,
    0x0b080020, 0xdc288ffd, 0x0c080020, 0xdc2c8001, 0x0d7d0022, 0xdc308fff, 0x0e080020, 0xdc308000,
    0x18080021, 0xdc208800, 0x197d0022, 0xbf8c0070, 0xbefe04c1, 0xe0781000, 0x80010a02, 0xe0781010,
    0x80010e02, 0xe0781020, 0x80011202, 0xe0781030, 0x80011602, 0xbf810000
};

// v_lshlrev_b32 v1, 2, v0; buffer_load_dword v2, v1, s[8:11], 0 offen; s_waitcnt vmcnt(0);
// buffer_store_dword v2, v1, s[4:7], 0 offen; s_endpgm: copies a descriptor-bound buffer, one
// dword per thread, so its input is read in place through a host import.
alignas(256) constexpr std::array<std::uint32_t, 7> CopyCode{0x34020082, 0xe0301000, 0x80020201, 0xbf8c3f70, 0xe0701000, 0x80010201, 0xbf810000};

class GuestBlock {
public:
    GuestBlock(std::uint8_t fill, bool registered) : registered(registered) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "completion label order: cannot allocate a guest block");
        std::memset(block, fill, BlockBytes);
        if (registered) GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        if (registered) GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    volatile std::uint8_t* Data() const { return block; }
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

private:
    std::uint8_t* block = nullptr;
    bool registered;
};

// A dispatch storing into memory no guest allocation registers, so nothing imports it: the
// region is copied and its results reach guest memory by a CPU write-back when the batch is
// reaped.
void RecordDispatch(AgcDriver::VulkanDevice& device, const GuestBlock& input, std::uint64_t output) {
    std::vector<std::uint32_t> userData(10, 0u);
    const std::array<std::uint32_t, 4> descriptor{static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), OutputBytes, 0x31016facu};
    std::copy(descriptor.begin(), descriptor.end(), userData.begin() + 4);
    userData[8] = static_cast<std::uint32_t>(input.Address());
    userData[9] = static_cast<std::uint32_t>(input.Address() >> 32u);
    const std::span<const std::uint32_t> code(LoadsCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
}

void RecordDescriptorCopy(AgcDriver::VulkanDevice& device, std::uint64_t input, std::uint64_t output) {
    std::vector<std::uint32_t> userData(12, 0u);
    const std::array<std::uint32_t, 4> out{static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), Threads * 4u, 0x31016facu};
    const std::array<std::uint32_t, 4> in{static_cast<std::uint32_t>(input), static_cast<std::uint32_t>((input >> 32u) & 0xffffu), Threads * 4u, 0x31016facu};
    std::copy(out.begin(), out.end(), userData.begin() + 4);
    std::copy(in.begin(), in.end(), userData.begin() + 8);
    const std::span<const std::uint32_t> code(CopyCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
}

bool Stored(const GuestBlock& block, std::uint64_t address) {
    const auto* begin = block.Data() + (address - block.Address());
    for (std::uint32_t offset = 0; offset < OutputBytes; ++offset) {
        if (begin[offset] != Fill) return true;
    }
    return false;
}

std::uint32_t Word(const GuestBlock& block, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, const_cast<const std::uint8_t*>(block.Data() + offset), sizeof(value));
    return value;
}

void SetWord(const GuestBlock& block, std::size_t offset, std::uint32_t value) {
    std::memcpy(const_cast<std::uint8_t*>(block.Data() + offset), &value, sizeof(value));
}

int HoldLabel(AgcDriver::VulkanDevice& device, std::uint64_t address, std::uint32_t value, std::uint64_t stamp, bool submit = true) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const auto bytes = std::as_bytes(std::span(&value, 1));
    const int reason = device.WriteLabelOnGpu(address, bytes, stamp, 0, false, true);
    if (submit) device.SubmitRecorded(true);
    return reason;
}

// Thread 0 of LoadsCode reads its input's first four dwords, and with dword 2 odd loads the dword
// at input + dword 3 + 0x200 into the 15th dword of its output record: a label held at dword 3
// selects which of two marker words the dispatch stores.
constexpr std::size_t LabelOffset = 12;
constexpr std::size_t PollLabelOffset = 64;
constexpr std::size_t WaitLabelOffset = 128;
constexpr std::uint32_t HeldTarget = 0x100;
constexpr std::uint32_t StaleMarker = 0xaaaa0000u;
constexpr std::uint32_t FreshMarker = 0xbbbb0000u;
constexpr std::size_t MarkerDword = 14;
constexpr std::size_t BoundLabelOffset = 256;
constexpr std::uint32_t BoundTarget = 0x2468ace0u;

}

// A completion label recorded after a dispatch whose results are copied back by the CPU is held
// until that write-back: (c) a guest polling the label with plain loads never sees it before the
// results; (a) a dispatch recorded after it that reads the label's bytes in place sees the new
// value; (b) a WAIT_REG_MEM on the held label is answered by the label table at once (no wait on
// the reap), and a waiter that submits and reaps, as a queue worker does for a refused late
// label, sees it land in guest memory after the copied results.
int main() {
#ifdef _WIN32
    _putenv_s("APS5_CPU_COPIES", "1");
#else
    setenv("APS5_CPU_COPIES", "1", 1);
#endif
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const GuestBlock input(0, true);
        const GuestBlock first(Fill, false);
        const GuestBlock consumer(Fill, false);
        const GuestBlock third(Fill, false);
        const GuestBlock labels(0, true);
        SetWord(labels, 8, 1u);
        SetWord(labels, 0x200, StaleMarker);
        SetWord(labels, 0x300, FreshMarker);
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            const std::array<std::uint32_t, 4> pattern{1u, 2u, 3u, 4u};
            if (!device->FillBuffer(labels.Address() + 4096, 4096, pattern)) {
                std::puts("skipped, the device does not import guest memory");
                return VulkanTestSkipped;
            }
        }

        const auto firstOutput = first.Address() + 4u;
        RecordDispatch(*device, input, firstOutput);
        const int pollReason = HoldLabel(*device, labels.Address() + PollLabelOffset, Label, 1);
        Require(pollReason != 3, "completion label order: the label block was not imported");
        Require(pollReason == 5, "completion label order: a completion label recorded after a pending copied write-back was not kept behind it (reason " + std::to_string(pollReason) + ")");
        const int heldReason = HoldLabel(*device, labels.Address() + LabelOffset, HeldTarget, 2);
        Require(heldReason == 5, "completion label order: the second label was not kept behind the write-back (reason " + std::to_string(heldReason) + ")");
        bool seenEarly = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
        while (std::chrono::steady_clock::now() < deadline) {
            if (Word(labels, PollLabelOffset) == Label) {
                seenEarly = true;
                Require(Stored(first, firstOutput), "completion label order: the label was visible before the dispatch's copied results");
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }

        const auto consumerOutput = consumer.Address() + 4u;
        RecordDispatch(*device, labels, consumerOutput);
        device->WaitIdle();
        Require(Word(labels, PollLabelOffset) == Label && Word(labels, LabelOffset) == HeldTarget, "completion label order: a held label never landed");
        Require(Stored(first, firstOutput), "completion label order: the first dispatch's copied results never landed");
        const auto marker = Word(consumer, 4u + MarkerDword * sizeof(std::uint32_t));
        char line[200];
        std::snprintf(line, sizeof(line), "completion label order: a dispatch recorded after a held label read 0x%08x through it (0x%08x is the held value's marker, 0x%08x the stale one's)", marker, FreshMarker, StaleMarker);
        Require(marker == FreshMarker, line);

        const GuestBlock fourth(Fill, false);
        const GuestBlock bound(Fill, false);
        const auto fourthOutput = fourth.Address() + 4u;
        RecordDispatch(*device, input, fourthOutput);
        const int boundReason = HoldLabel(*device, labels.Address() + BoundLabelOffset, BoundTarget, 4, false);
        Require(boundReason == 5, "completion label order: the bound label was not kept behind the write-back (reason " + std::to_string(boundReason) + ")");
        const auto boundOutput = bound.Address() + 4u;
        RecordDescriptorCopy(*device, labels.Address() + BoundLabelOffset, boundOutput);
        device->WaitIdle();
        char boundLine[200];
        std::snprintf(boundLine, sizeof(boundLine), "completion label order: a dispatch recorded after a held label read 0x%08x through a descriptor bound in place, the label holds 0x%08x", Word(bound, 4u), BoundTarget);
        Require(Word(bound, 4u) == BoundTarget, boundLine);

        const auto thirdOutput = third.Address() + 4u;
        RecordDispatch(*device, input, thirdOutput);
        const int waitReason = HoldLabel(*device, labels.Address() + WaitLabelOffset, Label, 3);
        Require(waitReason == 5, "completion label order: the waited label was not kept behind the write-back (reason " + std::to_string(waitReason) + ")");
        AgcDriver::Graphics::Recorder::LabelRefusal refusal{};
        std::optional<AgcDriver::Graphics::Recorder::LabelHit> hit;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            hit = device->PendingLabel(labels.Address() + WaitLabelOffset, 4, 0, &refusal);
        }
        Require(hit.has_value() && hit->value == Label, "completion label order: the label table does not answer a wait on the held label with its value (refusal " + std::to_string(static_cast<int>(refusal)) + ")");
        bool landed = false;
        std::thread waiter([&] {
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < until) {
                if (Word(labels, WaitLabelOffset) == Label) {
                    landed = Stored(third, thirdOutput);
                    return;
                }
                {
                    std::unique_lock gpu(AgcDriver::GuestMemory::GpuMutex(), std::try_to_lock);
                    if (gpu.owns_lock()) {
                        device->SubmitRecorded(false);
                        device->ReapRecorded();
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        });
        waiter.join();
        Require(landed, "completion label order: a waiter submitting and reaping did not see the held label land after the copied results");
        device->WaitIdle();
        std::printf("completion label order tests passed (polled label %s before the reap)\n", seenEarly ? "seen" : "not seen");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
