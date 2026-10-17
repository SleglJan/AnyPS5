#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <string>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t FormatBc1 = 169;
constexpr std::uint32_t Format8888Unorm = 56;
constexpr std::uint32_t Format3232Uint = 62;
constexpr std::uint32_t TileLinear = 0;
constexpr std::uint32_t TileRenderTarget64KB = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint8_t KeysUncompressed = 0xff;
constexpr std::uint8_t KeysClear0000 = 0x00;
constexpr std::uint8_t KeysClear0001 = 0x40;
constexpr std::uint8_t KeysSingle = 0x10;
constexpr std::size_t KeyBlockBytes = 256;
constexpr std::size_t SurfaceBytes = 65536;

TestGuest::Array<float, Threads * 3> Input;
TestGuest::Array<float, Threads * 4> Output;

alignas(256) constexpr std::array<std::uint32_t, 13> Code{
    0x1614008c, 0xe03c1000, 0x8000010a, 0xbf8c3f70, 0xf0900f08, 0x00820401, 0xbf8c3f70,
    0x34160084, 0xe0781000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000,
};

struct Surface {
    std::uint32_t format;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t tileMode;
    bool alphaOnMsb;
    std::uint8_t* texels;
    std::uint8_t* keys;
};

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::uint8_t* GuestBytes(std::size_t bytes, std::size_t alignment) {
    return reinterpret_cast<std::uint8_t*>(TestGuest::Allocate(bytes, alignment));
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = AddressOf(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const Surface& surface) {
    const auto address = AddressOf(surface.texels);
    const auto meta = AddressOf(surface.keys) >> 8u;
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (surface.format << 20u) | (((surface.width - 1u) & 3u) << 30u),
        ((surface.width - 1u) >> 2u) | ((surface.height - 1u) << 14u),
        0xfacu | (surface.tileMode << 20u) | (Type2D << 28u),
        0u,
        0u,
        (1u << 20u) | (1u << 21u) | (surface.alphaOnMsb ? 1u << 22u : 0u) | (static_cast<std::uint32_t>(meta & 0xffu) << 24u),
        static_cast<std::uint32_t>(meta >> 8u),
    };
}

std::uint64_t GuestSurfaceBytes(const Surface& surface) {
    const auto words = TextureDescriptor(surface);
    return AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(words)).guestBytes;
}

void SetKeys(const Surface& surface, std::uint8_t keys) {
    std::memset(surface.keys, keys, static_cast<std::size_t>(GuestSurfaceBytes(surface) / KeyBlockBytes));
}

struct Texel {
    std::uint32_t x;
    std::uint32_t y;
};

Texel SampledTexel(const Surface& surface, std::uint32_t tid) {
    return {(tid * 37u + 3u) % surface.width, (tid * 53u + 7u) % surface.height};
}

void Sample(AgcDriver::VulkanDevice& device, const Surface& surface, const std::string& what, const std::function<void(std::uint32_t, std::span<const float, 4>)>& check) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto texel = SampledTexel(surface, tid);
        Input[tid * 3u + 0u] = (static_cast<float>(texel.x) + 0.5f) / static_cast<float>(surface.width);
        Input[tid * 3u + 1u] = (static_cast<float>(texel.y) + 0.5f) / static_cast<float>(surface.height);
        Input[tid * 3u + 2u] = 0.0f;
    }
    TestGuest::Vector<std::uint32_t> userData(20, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(decltype(Input)::ByteSize));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(decltype(Output)::ByteSize));
    const auto texture = TextureDescriptor(surface);
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
        const std::array<float, 4> sampled{Output[tid * 4u], Output[tid * 4u + 1u], Output[tid * 4u + 2u], Output[tid * 4u + 3u]};
        try {
            check(tid, sampled);
        } catch (const std::exception& error) {
            const auto texel = SampledTexel(surface, tid);
            throw std::runtime_error(what + ": texel (" + std::to_string(texel.x) + ", " + std::to_string(texel.y) + "): " + error.what());
        }
    }
}

void ExpectColor(std::span<const float, 4> sampled, const std::array<float, 4>& expected) {
    for (std::uint32_t channel = 0; channel < 4u; ++channel) {
        if (std::fabs(sampled[channel] - expected[channel]) > 0.002f) throw std::runtime_error("component " + std::to_string(channel) + " sampled " + std::to_string(sampled[channel]) + ", expected " + std::to_string(expected[channel]));
    }
}

struct Sources {
    std::uint64_t snapshots;
    std::uint64_t gathers;
    std::uint64_t clears;
};

Sources Taken() {
    const auto counts = AgcDriver::Graphics::SampledTextureSources();
    return {counts.snapshots, counts.gathers, counts.clears};
}

void ExpectSources(const Sources& before, bool gathered, bool cleared, const std::string& what) {
    const auto after = Taken();
    Require(after.snapshots == before.snapshots, what + ": the sampled texture was read into a CPU snapshot");
    Require((after.gathers > before.gathers) == gathered, what + (gathered ? ": the sampled texture was not gathered from guest memory on the GPU" : ": the sampled texture was gathered from guest memory"));
    Require((after.clears > before.clears) == cleared, what + (cleared ? ": the sampled texture was not cleared on the GPU" : ": the sampled texture was cleared instead of read"));
}

constexpr std::array<float, 4> Red{1.0f, 0.0f, 0.0f, 1.0f};
constexpr std::array<float, 4> Blue{0.0f, 0.0f, 1.0f, 1.0f};
constexpr std::array<float, 4> Black{0.0f, 0.0f, 0.0f, 1.0f};

void FillBc1(const Surface& surface) {
    const auto bytes = static_cast<std::size_t>(GuestSurfaceBytes(surface));
    for (std::size_t offset = 0; offset < bytes; offset += 8) {
        const std::uint16_t color0 = offset % KeyBlockBytes == 0 ? 0xf800u : 0x001fu;
        std::memset(surface.texels + offset, 0, 8);
        surface.texels[offset] = static_cast<std::uint8_t>(color0);
        surface.texels[offset + 1] = static_cast<std::uint8_t>(color0 >> 8u);
    }
}

std::size_t Bc1BlockOffset(const Surface& surface, Texel texel) {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, surface.format, surface.width, surface.height, 1);
    return static_cast<std::size_t>(mips[0].tiledOffset) + (texel.y / 4u) * mips[0].pitchBytes + (texel.x / 4u) * 8u;
}

void CompressedClears(AgcDriver::VulkanDevice& device) {
    Surface surface{FormatBc1, 128, 128, TileLinear, false, GuestBytes(SurfaceBytes, 256), GuestBytes(KeyBlockBytes, 256)};
    const auto bytes = GuestSurfaceBytes(surface);
    Require(bytes >= KeyBlockBytes && bytes % KeyBlockBytes == 0 && bytes <= SurfaceBytes, "sampled texture keys: the compressed surface does not fit its storage in whole key blocks");
    FillBc1(surface);

    SetKeys(surface, KeysClear0000);
    auto before = Taken();
    Sample(device, surface, "a compressed texture under 0000 clear keys", [](std::uint32_t, std::span<const float, 4> sampled) { ExpectColor(sampled, Black); });
    ExpectSources(before, false, true, "a compressed texture under 0000 clear keys");

    SetKeys(surface, KeysSingle);
    before = Taken();
    Sample(device, surface, "a compressed texture under single-color keys", [](std::uint32_t, std::span<const float, 4> sampled) { ExpectColor(sampled, Red); });
    ExpectSources(before, true, false, "a compressed texture under single-color keys");

    SetKeys(surface, KeysUncompressed);
    before = Taken();
    Sample(device, surface, "the compressed texture once its keys are uncompressed", [&](std::uint32_t tid, std::span<const float, 4> sampled) {
        ExpectColor(sampled, Bc1BlockOffset(surface, SampledTexel(surface, tid)) % KeyBlockBytes == 0 ? Red : Blue);
    });
    ExpectSources(before, true, false, "the compressed texture once its keys are uncompressed");
}

std::array<std::uint8_t, 4> TiledElement(std::size_t offset) {
    if (offset % KeyBlockBytes == 0) return {static_cast<std::uint8_t>(offset >> 8u), 0x40u, 0x80u, 0xffu};
    return {static_cast<std::uint8_t>(offset >> 2u), static_cast<std::uint8_t>(offset >> 10u), 0x11u, 0x00u};
}

void TiledSingleColor(AgcDriver::VulkanDevice& device) {
    constexpr std::uint32_t Side = 128;
    Surface surface{Format8888Unorm, Side, Side, TileRenderTarget64KB, true, GuestBytes(SurfaceBytes, 65536), GuestBytes(KeyBlockBytes, 256)};
    Require(GuestSurfaceBytes(surface) == SurfaceBytes, "sampled texture keys: the tiled surface is not one 64 KiB block");
    for (std::size_t offset = 0; offset < SurfaceBytes; offset += 4) {
        const auto element = TiledElement(offset);
        std::memcpy(surface.texels + offset, element.data(), element.size());
    }
    const AgcDriver::Graphics::ColorTargetLayout layout(Side, Side, AgcDriver::Graphics::ColorTileMode::RenderTarget, 4);
    SetKeys(surface, KeysSingle);
    const auto before = Taken();
    Sample(device, surface, "a tiled texture under single-color keys", [&](std::uint32_t tid, std::span<const float, 4> sampled) {
        const auto texel = SampledTexel(surface, tid);
        const auto element = TiledElement(layout.Offset(texel.x, texel.y) / KeyBlockBytes * KeyBlockBytes);
        ExpectColor(sampled, {element[0] / 255.0f, element[1] / 255.0f, element[2] / 255.0f, element[3] / 255.0f});
    });
    ExpectSources(before, true, false, "a tiled texture under single-color keys");
}

void WideClearCode(AgcDriver::VulkanDevice& device) {
    Surface surface{Format3232Uint, 32, 32, TileLinear, true, GuestBytes(SurfaceBytes, 256), GuestBytes(KeyBlockBytes, 256)};
    const auto bytes = GuestSurfaceBytes(surface);
    Require(bytes >= KeyBlockBytes && bytes % KeyBlockBytes == 0 && bytes <= SurfaceBytes, "sampled texture keys: the wide surface does not fit its storage in whole key blocks");
    std::memset(surface.texels, 0x5a, static_cast<std::size_t>(bytes));
    SetKeys(surface, KeysClear0001);
    const auto before = Taken();
    Sample(device, surface, "a 64-bit integer texture under 0001 clear keys", [](std::uint32_t, std::span<const float, 4> sampled) {
        std::array<std::uint32_t, 2> words{};
        std::memcpy(words.data(), sampled.data(), sizeof(words));
        if (words[0] != 0u || words[1] != 0xffffffffu) {
            char text[128];
            std::snprintf(text, sizeof(text), "sampled 0x%08x 0x%08x, expected 0 and 0xffffffff (alpha in the high channel)", words[0], words[1]);
            throw std::runtime_error(text);
        }
    });
    ExpectSources(before, false, true, "a 64-bit integer texture under 0001 clear keys");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        CompressedClears(*device);
        TiledSingleColor(*device);
        WideClearCode(*device);
        std::puts("sampled texture keys tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
