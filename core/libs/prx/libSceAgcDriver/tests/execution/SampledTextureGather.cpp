#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Size = 4;
constexpr std::uint32_t FormatBc1 = 169;
constexpr std::uint32_t Type2D = 9;

TestGuest::Array<float, Threads * 3> Input;
TestGuest::Array<float, Threads * 4> Output;
TestGuest::Array<std::uint8_t, 4096> Texels;

alignas(256) constexpr std::array<std::uint32_t, 13> Code{
    0x1614008c, 0xe03c1000, 0x8000010a, 0xbf8c3f70, 0xf0900f08, 0x00820401, 0xbf8c3f70,
    0x34160084, 0xe0781000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor() {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Texels.data()));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (FormatBc1 << 20u) | (((Size - 1u) & 3u) << 30u),
        ((Size - 1u) >> 2u) | ((Size - 1u) << 14u),
        0xfacu | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::array<float, 4> Color(std::uint32_t seed) {
    return seed == 0u ? std::array<float, 4>{1.0f, 0.0f, 0.0f, 1.0f} : std::array<float, 4>{0.0f, 0.0f, 1.0f, 1.0f};
}

void FillTexels(std::uint32_t seed) {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, FormatBc1, Size, Size, 1);
    Require(AgcDriver::Graphics::ComputeSurfaceSize(mips, 1) <= Texels.size(), "sampled texture gather: the texture does not fit its storage");
    Texels.fill(0);
    const std::uint16_t color0 = seed == 0u ? 0xf800u : 0x001fu;
    auto* block = &Texels[mips[0].tiledOffset];
    block[0] = static_cast<std::uint8_t>(color0);
    block[1] = static_cast<std::uint8_t>(color0 >> 8u);
}

void Sample(AgcDriver::VulkanDevice& device, std::uint32_t seed, const std::string& what) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Input[tid * 3u + 0u] = (static_cast<float>(tid % Size) + 0.5f) / static_cast<float>(Size);
        Input[tid * 3u + 1u] = (static_cast<float>((tid / Size) % Size) + 0.5f) / static_cast<float>(Size);
        Input[tid * 3u + 2u] = 0.0f;
    }
    TestGuest::Vector<std::uint32_t> userData(20, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(decltype(Input)::ByteSize));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(decltype(Output)::ByteSize));
    const auto texture = TextureDescriptor();
    const std::array<std::uint32_t, 4> sampler{0x92u, 0x00fff000u, 1u << 24u, 0u};
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    std::copy(texture.begin(), texture.end(), userData.begin() + 8);
    std::copy(sampler.begin(), sampler.end(), userData.begin() + 16);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(Code.data()), std::as_bytes(std::span(Code))}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(Code.data()), Code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    Output.fill(-1.0f);
    device.Dispatch(result, 1, 1, 1, reinterpret_cast<std::uintptr_t>(Code.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t channel = 0; channel < 4u; ++channel) {
            const float expected = Color(seed)[channel];
            const float value = Output[tid * 4u + channel];
            Require(std::fabs(value - expected) <= 0.002f, what + ": thread " + std::to_string(tid) + " component " + std::to_string(channel) + " sampled " + std::to_string(value) + ", expected " + std::to_string(expected));
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto before = AgcDriver::Graphics::SampledTextureSources();
        FillTexels(0u);
        Sample(*device, 0u, "a texture first sampled from guest memory");
        const auto first = AgcDriver::Graphics::SampledTextureSources();
        Require(first.snapshots == before.snapshots, "the sampled texture was read into a CPU snapshot");
        Require(first.gathers > before.gathers, "the sampled texture was not gathered from guest memory on the GPU");
        Sample(*device, 0u, "the same texture sampled again");
        FillTexels(1u);
        Sample(*device, 1u, "the texture after the CPU rewrote its texels");
        const auto rewritten = AgcDriver::Graphics::SampledTextureSources();
        Require(rewritten.snapshots == before.snapshots, "the rewritten texture was read into a CPU snapshot");
        Require(rewritten.gathers > first.gathers, "the rewritten texture was not gathered again");
        std::puts("sampled texture gather tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
