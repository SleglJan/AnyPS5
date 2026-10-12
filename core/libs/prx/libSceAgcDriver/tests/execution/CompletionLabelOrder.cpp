#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
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

bool Stored(const GuestBlock& block, std::uint64_t address) {
    const auto* begin = block.Data() + (address - block.Address());
    for (std::uint32_t offset = 0; offset < OutputBytes; ++offset) {
        if (begin[offset] != Fill) return true;
    }
    return false;
}

std::uint32_t LabelValue(const GuestBlock& block) {
    std::uint32_t value = 0;
    std::memcpy(&value, const_cast<const std::uint8_t*>(block.Data() + 4), sizeof(value));
    return value;
}

}

// A completion label recorded after a dispatch whose results are copied back by the CPU must not
// be visible before them: a guest that polls the label and then reads the results with plain
// loads would read stale bytes. The poll below runs while the GPU finishes and nothing reaps the
// batch, so a label the GPU stored would be seen there without the results.
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
        const GuestBlock output(Fill, false);
        const GuestBlock labels(0, true);
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            const std::array<std::uint32_t, 4> pattern{1u, 2u, 3u, 4u};
            if (!device->FillBuffer(labels.Address() + 4096, 4096, pattern)) {
                std::puts("skipped, the device does not import guest memory");
                return VulkanTestSkipped;
            }
        }
        const auto outputAddress = output.Address() + 4u;
        RecordDispatch(*device, input, outputAddress);
        int reason = -1;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            const auto bytes = std::as_bytes(std::span(&Label, 1));
            reason = device->WriteLabelOnGpu(labels.Address() + 4u, bytes, 1, 0, true, true);
            Require(reason != 3, "completion label order: the label block was not imported");
            device->SubmitRecorded(true);
        }
        Require(reason == 5, "completion label order: a completion label recorded after a pending copied write-back was not kept behind it (reason " + std::to_string(reason) + ")");
        bool seenEarly = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
        while (std::chrono::steady_clock::now() < deadline) {
            if (LabelValue(labels) == Label) {
                seenEarly = true;
                Require(Stored(output, outputAddress), "completion label order: the label was visible before the dispatch's copied results");
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        device->WaitIdle();
        Require(LabelValue(labels) == Label, "completion label order: the label never landed");
        Require(Stored(output, outputAddress), "completion label order: the dispatch's copied results never landed");
        std::printf("completion label order tests passed (label %s before the reap)\n", seenEarly ? "seen" : "not seen");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
