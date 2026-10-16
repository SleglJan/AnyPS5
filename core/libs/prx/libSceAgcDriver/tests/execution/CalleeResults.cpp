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
constexpr std::uint32_t Inputs = 4;
constexpr std::uint32_t Results = 4;
TestGuest::Array<std::uint32_t, MaxThreads * Inputs> Input;
TestGuest::Array<std::uint32_t, MaxThreads * Results> Output;

alignas(256) constexpr std::array<std::uint32_t, 19> Code{
    0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70,
    0x7e120304, 0xbb080007, 0x7e18030a,
    0x7e120305, 0xbb080004, 0x4a1a150c,
    0xe0781000, 0x80010a01, 0xbf810000,
    0x3a1412ff, 0xa5a5a5a5, 0x4a1612ff, 0x13579bdf, 0xbe802008, 0xbf810000,
};

constexpr const char* Names[Results] = {"second call's xor", "second call's sum", "first call's xor after the second call", "sum of both xors"};

std::uint32_t Value(std::uint32_t tid, std::uint32_t load) {
    return 0x9e3779b9u * (tid + 1u) + 0x01000193u * load;
}

std::array<std::uint32_t, Results> ExpectedFor(std::uint32_t tid) {
    const auto first = Value(tid, 0) ^ 0xa5a5a5a5u;
    const auto second = Value(tid, 1) ^ 0xa5a5a5a5u;
    return {second, Value(tid, 1) + 0x13579bdfu, first, first + second};
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

void Run(AgcDriver::VulkanDevice& device, std::uint32_t waveSize, const ShaderRecompiler::SpirvTarget& target) {
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
        target,
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
            Require(actual == expected[slot], std::string("callee results: wave") + std::to_string(waveSize) + " lane " + std::to_string(tid) + " " + Names[slot] + " is " + Hex(actual) + ", expected " + Hex(expected[slot]));
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device, 32, device->ComputeTarget(32));
        Run(*device, 64, device->Target());
        std::puts("callee result tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
