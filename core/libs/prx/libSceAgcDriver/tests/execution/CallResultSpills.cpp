#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t MaxThreads = 64;
constexpr std::uint32_t Inputs = 8;
constexpr std::uint32_t Results = 8;
TestGuest::Array<std::uint32_t, MaxThreads * Inputs> Input;
TestGuest::Array<std::uint32_t, MaxThreads * Results> Output;

alignas(256) constexpr std::array<std::uint32_t, 52> Code{
    0x34020085, 0x34040085,
    0xe0301000, 0x80000401, 0xbf8c3f70, 0x4a100881,
    0xe0301004, 0x80000501, 0xbf8c3f70, 0x4a120a82,
    0xe0301008, 0x80000601, 0xe030100c, 0x80000701, 0xbf8c3f70, 0x4a140f06,
    0xe0301010, 0x80000c01, 0xbf8c3f70, 0x4a160c83,
    0xbe880383, 0x7e1a0280, 0x7e1e0280, 0x4a1c0294,
    0x7e20030d, 0xe0301000, 0x80000d0e, 0xbf8c3f70, 0x4a1e1f10, 0x341e1e81, 0x4a1e1f0d, 0x4a1c1c84,
    0x80888108, 0xbf078008, 0xbf85fff5,
    0xe0701000, 0x80010802, 0xe0701004, 0x80010902, 0xe0701008, 0x80010a02, 0xe070100c, 0x80010b02,
    0xe0701010, 0x80010c02, 0xe0701014, 0x80010f02, 0xe0701018, 0x80010d02, 0xe070101c, 0x80011002,
    0xbf810000,
};

constexpr const char* Names[Results] = {"first + 1 before the second load", "second + 2 before the third load", "third + fourth", "third + 3 after the fifth load", "fifth", "loop accumulation of the previous and the new load", "last loop load", "previous loop load"};

std::uint32_t Value(std::uint32_t tid, std::uint32_t load) {
    return ((load + 1u) << 24u) | (tid << 8u) | load;
}

std::array<std::uint32_t, Results> ExpectedFor(std::uint32_t tid) {
    std::uint32_t accumulated = 0;
    std::uint32_t previous = 0;
    std::uint32_t current = 0;
    for (std::uint32_t load = 5; load < Inputs; ++load) {
        previous = current;
        current = Value(tid, load);
        accumulated = ((accumulated + previous) << 1u) + current;
    }
    return {Value(tid, 0) + 1u, Value(tid, 1) + 2u, Value(tid, 2) + Value(tid, 3), Value(tid, 2) + 3u, Value(tid, 4), accumulated, current, previous};
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void Run(AgcDriver::VulkanDevice& device, std::uint32_t waveSize) {
    for (std::uint32_t tid = 0; tid < MaxThreads; ++tid) {
        for (std::uint32_t load = 0; load < Inputs; ++load) Input[tid * Inputs + load] = Value(tid, load);
    }
    Output.fill(0xdeadbeefu);
    TestGuest::Vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{waveSize, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < waveSize; ++tid) {
        const auto expected = ExpectedFor(tid);
        for (std::uint32_t slot = 0; slot < Results; ++slot) {
            const auto actual = Output[tid * Results + slot];
            Require(actual == expected[slot], std::string("call result spills: wave") + std::to_string(waveSize) + " lane " + std::to_string(tid) + " " + Names[slot] + " is " + Hex(actual) + ", expected " + Hex(expected[slot]));
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (device->Target().subgroupSize < 32u) {
            std::printf("skipped, subgroup size %u cannot hold a wave32\n", device->Target().subgroupSize);
            return VulkanTestSkipped;
        }
        Run(*device, 32);
        Run(*device, 64);
        std::puts("call result spill tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
