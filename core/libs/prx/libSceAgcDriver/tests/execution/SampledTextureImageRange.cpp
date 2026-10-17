#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>

#ifdef _WIN32

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Side = 64;
constexpr std::uint32_t FormatBc1 = 169;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t TexelBytes = 16384;
constexpr std::size_t BlockBytes = 8;

constexpr std::array<std::uint8_t, TexelBytes> MakeTexels() {
    std::array<std::uint8_t, TexelBytes> texels{};
    for (std::size_t offset = 0; offset < TexelBytes; offset += BlockBytes) {
        const std::uint16_t color0 = (offset / BlockBytes) % 2u == 0u ? 0xf800u : 0x001fu;
        texels[offset] = static_cast<std::uint8_t>(color0);
        texels[offset + 1] = static_cast<std::uint8_t>(color0 >> 8u);
    }
    return texels;
}

alignas(4096) const std::array<std::uint8_t, TexelBytes> ImageTexels = MakeTexels();

TestGuest::Array<float, Threads * 3> Input;
TestGuest::Array<float, Threads * 4> Output;

constexpr std::array<std::uint32_t, 13> CodeWords{
    0x1614008c, 0xe03c1000, 0x8000010a, 0xbf8c3f70, 0xf0900f08, 0x00820401, 0xbf8c3f70,
    0x34160084, 0xe0781000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000,
};

TestGuest::Array<std::uint32_t, CodeWords.size()> Code;

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = AddressOf(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor() {
    const auto address = AddressOf(ImageTexels.data());
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (FormatBc1 << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

struct Texel {
    std::uint32_t x;
    std::uint32_t y;
};

Texel SampledTexel(std::uint32_t tid) {
    return {(tid * 37u + 3u) % Side, (tid * 53u + 7u) % Side};
}

std::array<float, 4> ExpectedColor(Texel texel) {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, FormatBc1, Side, Side, 1);
    const auto offset = static_cast<std::size_t>(mips[0].tiledOffset) + (texel.y / 4u) * mips[0].pitchBytes + (texel.x / 4u) * BlockBytes;
    return (offset / BlockBytes) % 2u == 0u ? std::array<float, 4>{1.0f, 0.0f, 0.0f, 1.0f} : std::array<float, 4>{0.0f, 0.0f, 1.0f, 1.0f};
}

void Sample(AgcDriver::VulkanDevice& device, const std::string& what) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto texel = SampledTexel(tid);
        Input[tid * 3u + 0u] = (static_cast<float>(texel.x) + 0.5f) / static_cast<float>(Side);
        Input[tid * 3u + 1u] = (static_cast<float>(texel.y) + 0.5f) / static_cast<float>(Side);
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
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(Code.data()), std::as_bytes(std::span<const std::uint32_t>(Code.data(), Code.size()))}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(Code.data()), std::span<const std::uint32_t>(Code.data(), Code.size()), 0, {}},
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
        const auto texel = SampledTexel(tid);
        const auto expected = ExpectedColor(texel);
        for (std::uint32_t channel = 0; channel < 4u; ++channel) {
            const float value = Output[tid * 4u + channel];
            Require(std::fabs(value - expected[channel]) <= 0.002f, what + ": texel (" + std::to_string(texel.x) + ", " + std::to_string(texel.y) + ") component " + std::to_string(channel) + " sampled " + std::to_string(value) + ", expected " + std::to_string(expected[channel]));
        }
    }
}

void Run(AgcDriver::VulkanDevice& device) {
    Code = CodeWords;
    const auto descriptor = TextureDescriptor();
    const auto bytes = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(descriptor)).guestBytes;
    Require(bytes != 0u && bytes <= TexelBytes, "sampled texture image range: the texture does not fit its read-only words");
    MEMORY_BASIC_INFORMATION info{};
    Require(VirtualQuery(ImageTexels.data(), &info, sizeof(info)) == sizeof(info) && info.Type == MEM_IMAGE && (info.Protect & 0xffu) == PAGE_READONLY, "sampled texture image range: the texels are not read-only image pages");
    GuestAllocations::Mutation().RegisterMainImage();
    GuestAllocations::Mutation().Protect(ImageTexels.data(), TexelBytes, true, false, true, [] {});
    {
        const auto range = GuestAllocations::GuestAllocationsLookup_nid_postfix(static_cast<std::uintptr_t>(AddressOf(ImageTexels.data())), false);
        Require(range != nullptr && range->gpu && !range->writable && !range->releasable, "sampled texture image range: the texels are not one read-only GPU range of the main image");
    }

    const auto before = AgcDriver::Graphics::SampledTextureSources();
    const auto mirrorsBefore = AgcDriver::Graphics::MirrorCounters();
    Sample(device, "a texture in a read-only range of the main image");
    const auto first = AgcDriver::Graphics::SampledTextureSources();
    const auto mirrors = AgcDriver::Graphics::MirrorCounters();
    Require(first.snapshots == before.snapshots, "a texture no import serves was read into a CPU snapshot");
    Require(first.gathers == before.gathers + 1u, "a texture no import serves was not gathered on the GPU");
    Require(mirrors.rebuilds > mirrorsBefore.rebuilds && mirrors.deviceBytes == mirrors.bytes, "the texture's range was not mirrored in device memory");
    Sample(device, "the same texture sampled again");
    const auto again = AgcDriver::Graphics::SampledTextureSources();
    Require(again.snapshots == before.snapshots && again.gathers == first.gathers, "a texture whose read-only range did not change was gathered again");
    GuestAllocations::Mutation().UnregisterImage(GetModuleHandleW(nullptr));
}

}

#endif

int main() {
#ifdef _WIN32
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        std::puts("sampled texture image range tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#else
    std::printf("skipped, image ranges are Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
