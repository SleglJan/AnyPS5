#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t MaxSamples = 512;
constexpr std::uint32_t UnormFormat = 30;
constexpr std::uint32_t FloatFormat = 43;
constexpr std::uint32_t RawFormat = 20;
constexpr std::uint32_t Type1D = 8;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Type3D = 10;
constexpr std::uint32_t TypeCube = 11;
constexpr std::uint32_t Type2DArray = 13;
constexpr std::uint32_t SwizzleXYZ1 = 0x3acu;
constexpr std::uint32_t SwizzleZ0X1 = 0x306u;
constexpr std::uint32_t SwizzleYZX0 = 0x135u;
constexpr std::uint32_t Wrap = 0;
constexpr std::uint32_t Mirror = 1;
constexpr std::uint32_t ClampEdge = 2;
constexpr std::uint32_t MirrorOnce = 3;
constexpr std::uint32_t ClampBorder = 6;
constexpr std::uint32_t Point = 0;
constexpr std::uint32_t Bilinear = 1;
constexpr std::uint32_t TransparentBlack = 0;
constexpr std::uint32_t OpaqueBlack = 1;
constexpr std::uint32_t OpaqueWhite = 2;
constexpr std::uint32_t One = 0x3f800000u;
constexpr std::size_t RegionBytes = 65536;
constexpr std::size_t BlockBytes = 2u << 20u;
constexpr std::uint32_t LodWidth = 64;
constexpr std::uint32_t LodHeight = 4;
constexpr std::uint32_t Float32AbgrExport = 9;

TestGuest::Array<std::uint32_t, MaxSamples * 4> Output;
TestGuest::Array<std::uint32_t, MaxSamples * 4> Input;
TestGuest::Array<std::uint8_t, RegionBytes * 16> Snapshots;
TestGuest::Array<std::uint32_t, LodWidth * LodHeight * 4> LodPixels;

const TestGuest::Array<std::array<float, 4>, 3> Triangle{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}};

alignas(256) constexpr std::array<std::uint32_t, 6> LodVertexCode{
    0xe0382000u, 0x80000005u, 0xbf8c3f70u, 0xf80008cfu, 0x03020100u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 10> LodPixelCode{
    0xbefe0a7eu, 0x10040010u, 0x7e060280u, 0xf1800308u, 0x00610402u, 0xbf8c3f70u, 0xf8001803u, 0x07060504u, 0xbf810000u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 15> SampleCode(std::uint32_t word0) {
    return {
        0x7e080218u, 0x343c0885u, 0x4a3c3d00u, 0x34063c84u, 0xe0381000u, 0x80041403u, 0xbf8c3f70u, word0,
        0x00610a14u, 0xbf8c3f70u, 0xe0781000u, 0x80000a03u, 0xbf810000u, 0xbf810000u, 0xbf810000u,
    };
}

alignas(256) constexpr auto SampleLevel1D = SampleCode(0xf0900f00u);
alignas(256) constexpr auto SampleLevel2D = SampleCode(0xf0900f08u);
alignas(256) constexpr auto SampleLevel2DArray = SampleCode(0xf0900f28u);
alignas(256) constexpr auto SampleLevel3D = SampleCode(0xf0900f10u);
alignas(256) constexpr auto SampleLevelCube = SampleCode(0xf0900f18u);
alignas(256) constexpr auto GatherX = SampleCode(0xf11c0108u);
alignas(256) constexpr auto GatherY = SampleCode(0xf11c0208u);
alignas(256) constexpr auto GatherZ = SampleCode(0xf11c0408u);
alignas(256) constexpr auto GatherW = SampleCode(0xf11c0808u);
alignas(256) constexpr auto SampleZero2D = SampleCode(0xf09c0f08u);
alignas(256) constexpr auto SampleZeroOffset2D = SampleCode(0xf0dc0708u);
constexpr std::array<std::uint32_t, 12> StoreCode(std::uint32_t word0) {
    return {
        0x7e080218u, 0x343c0885u, 0x4a3c3d00u, 0x34063c84u, 0xe0381000u, 0x80041403u, 0xbf8c3f70u, word0,
        0x00011714u, 0xbf810000u, 0xbf810000u, 0xbf810000u,
    };
}

alignas(256) constexpr auto StoreRaw1D = StoreCode(0xf0201100u);
alignas(256) constexpr auto StoreRaw2D = StoreCode(0xf0201108u);
alignas(256) constexpr auto StoreRaw3D = StoreCode(0xf0201110u);
alignas(256) constexpr auto StoreRaw2DArray = StoreCode(0xf0201128u);

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "image converted sample: cannot allocate the guest block");
        Require(AgcDriver::GuestMemory::Watched(reinterpret_cast<std::uintptr_t>(block), BlockBytes), "image converted sample: the guest block is not write-watched");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        AgcDriver::GuestMemory::CollectWritesUncached(reinterpret_cast<std::uintptr_t>(block), BlockBytes);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(block, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, BlockBytes);
#else
        munmap(block, BlockBytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, BlockBytes);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }

private:
    std::uint8_t* block = nullptr;
};

struct Shape {
    std::uint32_t type;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t depth;
    std::uint32_t levels;
};

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* data, std::uint32_t format, std::uint32_t swizzle, const Shape& shape) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    const bool deep = shape.type == Type3D || shape.type == Type2DArray || shape.type == TypeCube;
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((shape.width - 1u) & 3u) << 30u),
        ((shape.width - 1u) >> 2u) | ((shape.height - 1u) << 14u),
        swizzle | ((shape.levels - 1u) << 16u) | (shape.type << 28u),
        deep ? shape.depth - 1u : 0u,
        (shape.levels - 1u) << 4u,
        0u,
        0u,
    };
}

std::array<std::uint32_t, 4> SamplerDescriptor(std::uint32_t clamp, std::uint32_t filter, std::uint32_t border = TransparentBlack) {
    return {clamp | (clamp << 3u) | (clamp << 6u), 0xfffu << 12u, (filter << 20u) | (filter << 22u) | (1u << 26u), border << 30u};
}

void Run(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 8>& texture, const std::array<std::uint32_t, 4>& sampler, std::uint32_t samples) {
    Output.fill(0xdeadbeefu);
    TestGuest::Vector<std::uint32_t> userData(24, 0u);
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    std::copy(output.begin(), output.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    std::copy(sampler.begin(), sampler.end(), userData.begin() + 12);
    std::copy(input.begin(), input.end(), userData.begin() + 16);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, (samples + Threads - 1u) / Threads, 1, 1, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

std::uint32_t Mix(std::uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

std::uint32_t FiniteTexel(std::uint32_t format, std::uint32_t seed) {
    const auto bits = Mix(seed);
    if (format == UnormFormat) return bits;
    const auto field = [](std::uint32_t value, std::uint32_t width) {
        const auto mantissaBits = width - 5u;
        return (((value >> mantissaBits) % 31u) << mantissaBits) | (value & ((1u << mantissaBits) - 1u));
    };
    return field(bits & 0x3ffu, 10u) | (field((bits >> 10u) & 0x7ffu, 11u) << 10u) | (field(bits >> 21u, 11u) << 21u);
}

std::uint32_t UnormBits(std::uint32_t value, std::uint32_t width) {
    return std::bit_cast<std::uint32_t>(static_cast<float>(static_cast<double>(value) / static_cast<double>((1u << width) - 1u)));
}

std::uint32_t SmallFloatBits(std::uint32_t value, std::uint32_t width) {
    const auto mantissaBits = width - 5u;
    const auto exponent = value >> mantissaBits;
    const auto mantissa = value & ((1u << mantissaBits) - 1u);
    if (exponent == 31u) return 0x7f800000u | (mantissa << (23u - mantissaBits));
    if (exponent == 0u) return std::bit_cast<std::uint32_t>(std::ldexp(static_cast<float>(mantissa), -14 - static_cast<int>(mantissaBits)));
    return ((exponent + 112u) << 23u) | (mantissa << (23u - mantissaBits));
}

std::array<std::uint32_t, 3> Decode(std::uint32_t format, std::uint32_t texel) {
    if (format == UnormFormat) return {UnormBits(texel & 0x7ffu, 11u), UnormBits((texel >> 11u) & 0x7ffu, 11u), UnormBits(texel >> 22u, 10u)};
    return {SmallFloatBits(texel & 0x3ffu, 10u), SmallFloatBits((texel >> 10u) & 0x7ffu, 11u), SmallFloatBits(texel >> 21u, 11u)};
}

std::uint32_t Selected(const std::array<std::uint32_t, 3>& decoded, std::uint32_t swizzle, std::uint32_t component) {
    const auto selector = (swizzle >> (component * 3u)) & 7u;
    return selector == 0u ? 0u : selector == 1u ? One : decoded[selector - 4u];
}

bool NaN(std::uint32_t bits) {
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu) != 0u;
}

void RequireBits(std::uint32_t actual, std::uint32_t expected, const std::string& what) {
    const bool same = NaN(expected) ? NaN(actual) : actual == expected;
    Require(same, what + " is " + Hex(actual) + ", expected " + Hex(expected));
}

void RequireNear(std::uint32_t actualBits, double expected, double tolerance, const std::string& what) {
    const double actual = std::bit_cast<float>(actualBits);
    Require(std::fabs(actual - expected) <= tolerance, what + " is " + std::to_string(actual) + " (" + Hex(actualBits) + "), expected " + std::to_string(expected) + " within " + std::to_string(tolerance));
}

double FilterTolerance(std::uint32_t format, double largestTap) {
    return format == UnormFormat ? std::ldexp(1.0, -18) : largestTap * std::ldexp(1.0, -9) + std::ldexp(1.0, -24);
}

class Surface {
public:
    Surface(std::uint8_t* base, std::uint32_t format, const Shape& shape) : base(base), format(format), shape(shape), geometry(AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(Descriptor(SwizzleXYZ1)))) {}

    std::array<std::uint32_t, 8> Descriptor(std::uint32_t swizzle, std::uint32_t asFormat = 0u) const { return TextureDescriptor(base, asFormat != 0u ? asFormat : format, swizzle, shape); }
    std::uint64_t Bytes() const { return geometry.guestBytes; }
    std::uint32_t Width(std::uint32_t level) const { return std::max(shape.width >> level, 1u); }
    std::uint32_t Height(std::uint32_t level) const { return std::max(shape.height >> level, 1u); }
    std::uint32_t Depth(std::uint32_t level) const { return shape.type == Type3D ? std::max(shape.depth >> level, 1u) : 1u; }
    std::uint32_t Layers() const { return shape.type == Type2DArray || shape.type == TypeCube ? shape.depth : 1u; }

    std::uint8_t* At(std::uint32_t x, std::uint32_t y, std::uint32_t slice, std::uint32_t level) const {
        const auto& mip = geometry.mips.at(level);
        return base + geometry.GuestLayerOffset(slice) + mip.tiledOffset + static_cast<std::uint64_t>(y) * mip.pitchBytes + static_cast<std::uint64_t>(x) * 4u;
    }

    std::uint32_t Texel(std::uint32_t x, std::uint32_t y, std::uint32_t slice, std::uint32_t level) const {
        std::uint32_t texel = 0;
        std::memcpy(&texel, At(x, y, slice, level), 4u);
        return texel;
    }

    void Fill(const std::function<std::uint32_t(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)>& texel) const {
        Require(geometry.guestBytes <= RegionBytes, "image converted sample: the surface does not fit its region");
        for (std::uint32_t level = 0; level < shape.levels; ++level) {
            for (std::uint32_t slice = 0; slice < std::max(Depth(level), Layers()); ++slice) {
                for (std::uint32_t y = 0; y < Height(level); ++y) {
                    for (std::uint32_t x = 0; x < Width(level); ++x) {
                        const auto value = texel(x, y, slice, level);
                        std::memcpy(At(x, y, slice, level), &value, 4u);
                    }
                }
            }
        }
    }

    std::uint8_t* base;
    std::uint32_t format;
    Shape shape;
    AgcDriver::Graphics::SurfaceGeometry geometry;
};

std::string Name(std::uint32_t format) {
    return format == UnormFormat ? "11_11_10_UNORM" : "10_11_11_FLOAT";
}

void SetInput(std::uint32_t sample, float u, float v, float w, float lod) {
    Input[sample * 4u] = std::bit_cast<std::uint32_t>(u);
    Input[sample * 4u + 1u] = std::bit_cast<std::uint32_t>(v);
    Input[sample * 4u + 2u] = std::bit_cast<std::uint32_t>(w);
    Input[sample * 4u + 3u] = std::bit_cast<std::uint32_t>(lod);
}

void CheckPointLevels(AgcDriver::VulkanDevice& device, const Surface& surface, std::uint32_t swizzle, const std::string& what) {
    struct Point3 {
        std::uint32_t x, y, slice, level;
    };
    std::vector<Point3> points;
    for (std::uint32_t level = 0; level < surface.shape.levels; ++level) {
        for (std::uint32_t slice = 0; slice < std::max(surface.Depth(level), surface.Layers()); ++slice) {
            for (std::uint32_t y = 0; y < surface.Height(level); ++y) {
                for (std::uint32_t x = 0; x < surface.Width(level); ++x) points.push_back({x, y, slice, level});
            }
        }
    }
    const auto type = surface.shape.type;
    const auto& code = type == Type3D ? SampleLevel3D : type == Type2DArray ? SampleLevel2DArray : type == TypeCube ? SampleLevelCube : type == Type1D ? SampleLevel1D : SampleLevel2D;
    for (std::size_t first = 0; first < points.size(); first += MaxSamples) {
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(MaxSamples, points.size() - first));
        Input.fill(0u);
        for (std::uint32_t sample = 0; sample < count; ++sample) {
            const auto& point = points[first + sample];
            const float u = (static_cast<float>(point.x) + 0.5f) / static_cast<float>(surface.Width(point.level));
            const float v = (static_cast<float>(point.y) + 0.5f) / static_cast<float>(surface.Height(point.level));
            const float w = type == Type3D ? (static_cast<float>(point.slice) + 0.5f) / static_cast<float>(surface.Depth(point.level)) : static_cast<float>(point.slice);
            const auto level = static_cast<float>(point.level);
            if (type == Type1D) SetInput(sample, u, level, 0.0f, 0.0f);
            else if (type == Type2D) SetInput(sample, u, v, level, 0.0f);
            else if (type == TypeCube) SetInput(sample, 1.0f + u, 1.0f + v, w, level);
            else SetInput(sample, u, v, w, level);
        }
        Run(device, code, surface.Descriptor(swizzle), SamplerDescriptor(ClampEdge, Point), count);
        for (std::uint32_t sample = 0; sample < count; ++sample) {
            const auto& point = points[first + sample];
            const auto decoded = Decode(surface.format, surface.Texel(point.x, point.y, point.slice, point.level));
            for (std::uint32_t component = 0; component < 4u; ++component) {
                RequireBits(Output[sample * 4u + component], Selected(decoded, swizzle, component), what + ": texel (" + std::to_string(point.x) + ", " + std::to_string(point.y) + ", " + std::to_string(point.slice) + ") level " + std::to_string(point.level) + " component " + std::to_string(component));
            }
        }
    }
}

double Value(std::uint32_t bits) {
    return std::bit_cast<float>(bits);
}

std::uint32_t Clamp(std::int32_t index, std::uint32_t size) {
    return static_cast<std::uint32_t>(std::clamp(index, 0, static_cast<std::int32_t>(size) - 1));
}

void CheckBilinear(AgcDriver::VulkanDevice& device, const Surface& surface, const std::string& what) {
    constexpr std::uint32_t Samples = 128;
    const auto width = surface.Width(0);
    const auto height = surface.Height(0);
    std::array<std::array<std::int32_t, 4>, Samples> taps{};
    Input.fill(0u);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto seed = Mix(sample * 977u + 13u);
        const auto x = static_cast<std::int32_t>(seed % width);
        const auto y = static_cast<std::int32_t>((seed >> 8u) % height);
        const auto fx = static_cast<std::int32_t>((seed >> 16u) % 16u);
        const auto fy = static_cast<std::int32_t>((seed >> 20u) % 16u);
        taps[sample] = {x, y, fx, fy};
        const float u = (static_cast<float>(x) + 0.5f + static_cast<float>(fx) / 16.0f) / static_cast<float>(width);
        const float v = (static_cast<float>(y) + 0.5f + static_cast<float>(fy) / 16.0f) / static_cast<float>(height);
        SetInput(sample, u, v, 0.0f, 0.0f);
    }
    Run(device, SampleLevel2D, surface.Descriptor(SwizzleXYZ1), SamplerDescriptor(ClampEdge, Bilinear), Samples);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, fx, fy] = taps[sample];
        const double a = fx / 16.0;
        const double b = fy / 16.0;
        const auto t00 = Decode(surface.format, surface.Texel(Clamp(x, width), Clamp(y, height), 0, 0));
        const auto t10 = Decode(surface.format, surface.Texel(Clamp(x + 1, width), Clamp(y, height), 0, 0));
        const auto t01 = Decode(surface.format, surface.Texel(Clamp(x, width), Clamp(y + 1, height), 0, 0));
        const auto t11 = Decode(surface.format, surface.Texel(Clamp(x + 1, width), Clamp(y + 1, height), 0, 0));
        for (std::uint32_t component = 0; component < 3u; ++component) {
            const double expected = (1.0 - a) * (1.0 - b) * Value(t00[component]) + a * (1.0 - b) * Value(t10[component]) + (1.0 - a) * b * Value(t01[component]) + a * b * Value(t11[component]);
            const double largest = std::max({std::fabs(Value(t00[component])), std::fabs(Value(t10[component])), std::fabs(Value(t01[component])), std::fabs(Value(t11[component]))});
            RequireNear(Output[sample * 4u + component], expected, FilterTolerance(surface.format, largest), what + ": bilinear sample " + std::to_string(sample) + " at texel (" + std::to_string(x) + ", " + std::to_string(y) + ") + (" + std::to_string(fx) + ", " + std::to_string(fy) + ")/16 component " + std::to_string(component));
        }
        RequireBits(Output[sample * 4u + 3u], One, what + ": bilinear sample " + std::to_string(sample) + " component 3");
    }
}

std::uint32_t Address(std::int32_t index, std::uint32_t size, std::uint32_t clamp, bool& border) {
    const auto extent = static_cast<std::int32_t>(size);
    border = false;
    switch (clamp) {
        case Wrap: return static_cast<std::uint32_t>(((index % extent) + extent) % extent);
        case Mirror: {
            const auto period = 2 * extent;
            const auto folded = ((index % period) + period) % period;
            return static_cast<std::uint32_t>(folded < extent ? folded : period - 1 - folded);
        }
        case ClampEdge: return Clamp(index, size);
        case MirrorOnce: return Clamp(index < 0 ? -1 - index : index, size);
        default:
            border = index < 0 || index >= extent;
            return Clamp(index, size);
    }
}

void CheckAddressModes(AgcDriver::VulkanDevice& device, const Surface& surface, const std::string& what) {
    constexpr std::array<float, 8> Coordinates{-1.3f, -0.71f, -0.02f, 0.47f, 1.01f, 1.6f, 2.21f, -2.9f};
    constexpr std::array<std::array<std::uint32_t, 2>, 8> Modes{{{Wrap, TransparentBlack}, {Mirror, TransparentBlack}, {ClampEdge, TransparentBlack}, {MirrorOnce, TransparentBlack}, {ClampBorder, TransparentBlack}, {ClampBorder, OpaqueBlack}, {ClampBorder, OpaqueWhite}, {Wrap, OpaqueWhite}}};
    const auto width = surface.Width(0);
    const auto height = surface.Height(0);
    for (const auto& [clamp, borderType] : Modes) {
        Input.fill(0u);
        std::uint32_t samples = 0;
        for (const float u : Coordinates) {
            for (const float v : Coordinates) SetInput(samples++, u, v, 0.0f, 0.0f);
        }
        for (const auto swizzle : {SwizzleXYZ1, SwizzleZ0X1}) {
            Run(device, SampleLevel2D, surface.Descriptor(swizzle), SamplerDescriptor(clamp, Point, borderType), samples);
            std::uint32_t sample = 0;
            for (const float u : Coordinates) {
                for (const float v : Coordinates) {
                    bool borderX = false;
                    bool borderY = false;
                    const auto x = Address(static_cast<std::int32_t>(std::floor(u * static_cast<float>(width))), width, clamp, borderX);
                    const auto y = Address(static_cast<std::int32_t>(std::floor(v * static_cast<float>(height))), height, clamp, borderY);
                    const auto decoded = Decode(surface.format, surface.Texel(x, y, 0, 0));
                    const std::uint32_t borderColor = borderType == OpaqueWhite ? One : 0u;
                    const std::uint32_t borderAlpha = borderType == TransparentBlack ? 0u : One;
                    for (std::uint32_t component = 0; component < 4u; ++component) {
                        const auto selector = (swizzle >> (component * 3u)) & 7u;
                        const auto expected = !(borderX || borderY) || selector < 4u ? Selected(decoded, swizzle, component) : selector == 7u ? borderAlpha : borderColor;
                        RequireBits(Output[sample * 4u + component], expected, what + ": clamp " + std::to_string(clamp) + " border " + std::to_string(borderType) + " swizzle " + Hex(swizzle) + " at (" + std::to_string(u) + ", " + std::to_string(v) + ") component " + std::to_string(component));
                    }
                    ++sample;
                }
            }
        }
    }
}

void CheckGather(AgcDriver::VulkanDevice& device, const Surface& surface, const std::string& what) {
    constexpr std::uint32_t Samples = 64;
    const auto width = surface.Width(0);
    const auto height = surface.Height(0);
    std::array<std::array<std::int32_t, 2>, Samples> anchors{};
    Input.fill(0u);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto seed = Mix(sample * 131u + 7u);
        const auto x = static_cast<std::int32_t>(seed % (width + 1u)) - 1;
        const auto y = static_cast<std::int32_t>((seed >> 8u) % (height + 1u)) - 1;
        anchors[sample] = {x, y};
        const float u = (static_cast<float>(x) + 1.0f + static_cast<float>((seed >> 16u) % 15u + 1u) / 32.0f - 0.25f) / static_cast<float>(width);
        const float v = (static_cast<float>(y) + 1.0f + static_cast<float>((seed >> 20u) % 15u + 1u) / 32.0f - 0.25f) / static_cast<float>(height);
        SetInput(sample, u, v, 0.0f, 0.0f);
    }
    const std::array<std::span<const std::uint32_t>, 3> programs{GatherX, GatherY, GatherZ};
    for (std::uint32_t component = 0; component < 3u; ++component) {
        Run(device, programs[component], surface.Descriptor(SwizzleYZX0), SamplerDescriptor(ClampEdge, Bilinear), Samples);
        for (std::uint32_t sample = 0; sample < Samples; ++sample) {
            const auto [x, y] = anchors[sample];
            const std::array<std::array<std::int32_t, 2>, 4> corners{{{x, y + 1}, {x + 1, y + 1}, {x + 1, y}, {x, y}}};
            for (std::uint32_t texel = 0; texel < 4u; ++texel) {
                const auto decoded = Decode(surface.format, surface.Texel(Clamp(corners[texel][0], width), Clamp(corners[texel][1], height), 0, 0));
                RequireBits(Output[sample * 4u + texel], Selected(decoded, SwizzleYZX0, component), what + ": image_gather4_lz of component " + std::to_string(component) + " sample " + std::to_string(sample) + " texel " + std::to_string(texel));
            }
        }
    }
}

void CheckSamplerStates(AgcDriver::VulkanDevice& device, const Surface& surface, const std::string& what) {
    const auto width = surface.Width(0);
    const auto height = surface.Height(0);
    constexpr std::uint32_t Samples = 64;
    std::array<std::array<std::int32_t, 4>, Samples> taps{};
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto seed = Mix(sample * 613u + 11u);
        taps[sample] = {static_cast<std::int32_t>(seed % (width - 1u)), static_cast<std::int32_t>((seed >> 8u) % (height - 1u)), static_cast<std::int32_t>((seed >> 16u) % 15u) + 1, static_cast<std::int32_t>((seed >> 20u) % 15u) + 1};
    }
    Input.fill(0u);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, fx, fy] = taps[sample];
        SetInput(sample, (static_cast<float>(x) + 0.5f) / static_cast<float>(width), (static_cast<float>(y) + 0.5f) / static_cast<float>(height), 0.5f, 0.0f);
    }
    const auto descriptor = surface.Descriptor(SwizzleXYZ1);
    Run(device, SampleLevel2D, descriptor, {ClampEdge | (ClampEdge << 3u) | (ClampEdge << 6u), 0xfffu << 12u, 2u << 26u, 0u}, Samples);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, fx, fy] = taps[sample];
        const auto upper = Decode(surface.format, surface.Texel(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), 0, 0));
        const auto lower = Decode(surface.format, surface.Texel(static_cast<std::uint32_t>(x) >> 1u, static_cast<std::uint32_t>(y) >> 1u, 0, 1));
        for (std::uint32_t component = 0; component < 3u; ++component) {
            const double expected = 0.5 * Value(upper[component]) + 0.5 * Value(lower[component]);
            RequireNear(Output[sample * 4u + component], expected, FilterTolerance(surface.format, std::max(std::fabs(Value(upper[component])), std::fabs(Value(lower[component])))), what + ": linear mip filter at LOD 0.5, sample " + std::to_string(sample) + " component " + std::to_string(component));
        }
    }
    Input.fill(0u);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, fx, fy] = taps[sample];
        SetInput(sample, (static_cast<float>(x) + 0.5f) / static_cast<float>(width), (static_cast<float>(y) + 0.5f) / static_cast<float>(height), 0.0f, 0.0f);
    }
    Run(device, SampleLevel2D, descriptor, {ClampEdge | (ClampEdge << 3u) | (ClampEdge << 6u), 0x100u | (0xfffu << 12u), 1u << 26u, 0u}, Samples);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, fx, fy] = taps[sample];
        const auto lower = Decode(surface.format, surface.Texel(static_cast<std::uint32_t>(x) >> 1u, static_cast<std::uint32_t>(y) >> 1u, 0, 1));
        for (std::uint32_t component = 0; component < 4u; ++component) RequireBits(Output[sample * 4u + component], Selected(lower, SwizzleXYZ1, component), what + ": MIN_LOD 1 clamping LOD 0, sample " + std::to_string(sample) + " component " + std::to_string(component));
    }
    if (!device.SamplerFilterMinmax()) return;
    Input.fill(0u);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, fx, fy] = taps[sample];
        SetInput(sample, (static_cast<float>(x) + 0.5f + static_cast<float>(fx) / 16.0f) / static_cast<float>(width), (static_cast<float>(y) + 0.5f + static_cast<float>(fy) / 16.0f) / static_cast<float>(height), 0.0f, 0.0f);
    }
    for (const std::uint32_t reduction : {1u, 2u}) {
        Run(device, SampleLevel2D, descriptor, {ClampEdge | (ClampEdge << 3u) | (ClampEdge << 6u) | (reduction << 29u), 0xfffu << 12u, (Bilinear << 20u) | (Bilinear << 22u) | (1u << 26u), 0u}, Samples);
        for (std::uint32_t sample = 0; sample < Samples; ++sample) {
            const auto [x, y, fx, fy] = taps[sample];
            std::array<std::array<std::uint32_t, 3>, 4> footprint{};
            for (std::uint32_t tap = 0; tap < 4u; ++tap) footprint[tap] = Decode(surface.format, surface.Texel(static_cast<std::uint32_t>(x) + (tap & 1u), static_cast<std::uint32_t>(y) + (tap >> 1u), 0, 0));
            for (std::uint32_t component = 0; component < 3u; ++component) {
                auto expected = footprint[0][component];
                for (std::uint32_t tap = 1; tap < 4u; ++tap) {
                    const bool smaller = Value(footprint[tap][component]) < Value(expected);
                    if (reduction == 1u ? smaller : !smaller && Value(footprint[tap][component]) != Value(expected)) expected = footprint[tap][component];
                }
                RequireBits(Output[sample * 4u + component], expected, what + ": " + (reduction == 1u ? "min" : "max") + " reduction, sample " + std::to_string(sample) + " component " + std::to_string(component));
            }
        }
    }
}

void CheckLevelZeroVariants(AgcDriver::VulkanDevice& device, const Surface& surface, const std::string& what) {
    const auto width = surface.Width(0);
    const auto height = surface.Height(0);
    constexpr std::uint32_t Samples = 128;
    std::array<std::array<std::int32_t, 4>, Samples> taps{};
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto seed = Mix(sample * 389u + 5u);
        taps[sample] = {static_cast<std::int32_t>(seed % width), static_cast<std::int32_t>((seed >> 8u) % height), static_cast<std::int32_t>((seed >> 16u) % 16u) - 8, static_cast<std::int32_t>((seed >> 20u) % 16u) - 8};
    }
    Input.fill(0u);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, dx, dy] = taps[sample];
        SetInput(sample, (static_cast<float>(x) + 0.5f) / static_cast<float>(width), (static_cast<float>(y) + 0.5f) / static_cast<float>(height), 0.0f, 0.0f);
    }
    Run(device, SampleZero2D, surface.Descriptor(SwizzleXYZ1), SamplerDescriptor(ClampEdge, Point), Samples);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, dx, dy] = taps[sample];
        const auto decoded = Decode(surface.format, surface.Texel(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), 0, 0));
        for (std::uint32_t component = 0; component < 4u; ++component) RequireBits(Output[sample * 4u + component], Selected(decoded, SwizzleXYZ1, component), what + ": image_sample_lz sample " + std::to_string(sample) + " component " + std::to_string(component));
    }
    Input.fill(0u);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, dx, dy] = taps[sample];
        Input[sample * 4u] = (static_cast<std::uint32_t>(dx) & 0x3fu) | ((static_cast<std::uint32_t>(dy) & 0x3fu) << 8u);
        Input[sample * 4u + 1u] = std::bit_cast<std::uint32_t>((static_cast<float>(x) + 0.5f) / static_cast<float>(width));
        Input[sample * 4u + 2u] = std::bit_cast<std::uint32_t>((static_cast<float>(y) + 0.5f) / static_cast<float>(height));
    }
    Run(device, SampleZeroOffset2D, surface.Descriptor(SwizzleZ0X1), SamplerDescriptor(ClampEdge, Point), Samples);
    for (std::uint32_t sample = 0; sample < Samples; ++sample) {
        const auto [x, y, dx, dy] = taps[sample];
        const auto decoded = Decode(surface.format, surface.Texel(Clamp(x + dx, width), Clamp(y + dy, height), 0, 0));
        for (std::uint32_t component = 0; component < 3u; ++component) RequireBits(Output[sample * 4u + component], Selected(decoded, SwizzleZ0X1, component), what + ": image_sample_lz_o offset (" + std::to_string(dx) + ", " + std::to_string(dy) + ") sample " + std::to_string(sample) + " component " + std::to_string(component));
    }
    for (const auto swizzle : {SwizzleZ0X1, SwizzleYZX0}) {
        Run(device, GatherW, surface.Descriptor(swizzle), SamplerDescriptor(ClampEdge, Point), Samples);
        const std::uint32_t expected = ((swizzle >> 9u) & 7u) == 1u ? One : 0u;
        for (std::uint32_t sample = 0; sample < Samples; ++sample) {
            for (std::uint32_t texel = 0; texel < 4u; ++texel) RequireBits(Output[sample * 4u + texel], expected, what + ": image_gather4_lz of the constant W selector " + Hex(swizzle) + " sample " + std::to_string(sample) + " texel " + std::to_string(texel));
        }
    }
}

void DrawLod(AgcDriver::VulkanDevice& device, const Surface& surface, float scale) {
    const auto& target = device.Target();
    LodPixels.fill(0xdeadbeefu);
    TestGuest::Vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexAddress = reinterpret_cast<std::uintptr_t>(Triangle.data());
    const std::array<std::uint32_t, 4> vertexBuffer{static_cast<std::uint32_t>(vertexAddress), static_cast<std::uint32_t>((vertexAddress >> 32u) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Triangle.size()), 0x01016facu};
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(LodVertexCode.data()), std::as_bytes(std::span(LodVertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(LodVertexCode.data()), LodVertexCode, 0, {}},
        {64, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.wave32 = false;
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = Float32AbgrExport;
    pixel.targetExportMapping.fill(0xe4u);
    TestGuest::Vector<std::uint32_t> pixelUserData(17, 0u);
    const auto texture = surface.Descriptor(SwizzleXYZ1);
    const auto sampler = SamplerDescriptor(ClampEdge, Bilinear);
    std::copy(texture.begin(), texture.end(), pixelUserData.begin() + 4);
    std::copy(sampler.begin(), sampler.end(), pixelUserData.begin() + 12);
    pixelUserData[16] = std::bit_cast<std::uint32_t>(scale);
    const std::span<const std::uint32_t> pixelCode(LodPixelCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(pixelCode.data()), std::as_bytes(pixelCode)}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(pixelCode.data()), pixelCode, 0, {}},
        {64, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    fragment.useCache = false;
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, &pixelResult, PixelPushOffset(vertexPush, target)}
    }};
    AgcDriver::Graphics::ColorTarget color{reinterpret_cast<std::uintptr_t>(LodPixels.data()), {LodWidth, LodHeight}, VK_FORMAT_R32G32B32A32_SFLOAT, LodPixels.size() * 4u, 0xe4u};
    color.elementBytes = 16u;
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.color = color;
    state.colors = {color};
    state.hasColorTarget = true;
    state.renderExtent = {LodWidth, LodHeight};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(LodHeight), static_cast<float>(LodWidth), -static_cast<float>(LodHeight), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {LodWidth, LodHeight}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangle.size()), 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(LodPixels.data()), LodPixels.size() * 4u, nullptr, "test");
    device.WaitIdle();
}

void CheckLod(AgcDriver::VulkanDevice& device, const Surface& surface, const std::string& what) {
    Require(AgcDriver::Graphics::ColorTargetLayout(LodWidth, LodHeight, AgcDriver::Graphics::ColorTileMode::Linear, 16).Bytes() == LodPixels.size() * 4u, what + ": the linear LOD target is padded");
    const auto lastLevel = static_cast<double>(surface.shape.levels - 1u);
    for (const int lod : {-1, 0, 1, 2, 3, 6}) {
        DrawLod(device, surface, std::ldexp(1.0f, lod) / static_cast<float>(surface.Width(0)));
        for (std::uint32_t y = 0; y < LodHeight; ++y) {
            for (std::uint32_t x = 0; x < LodWidth; ++x) {
                const auto pixel = (y * LodWidth + x) * 4u;
                const auto where = what + ": image_get_lod in a pixel shader, " + std::to_string(lod) + " levels per pixel step, pixel (" + std::to_string(x) + ", " + std::to_string(y) + ")";
                RequireNear(LodPixels[pixel], std::clamp(static_cast<double>(lod), 0.0, lastLevel), 1.0 / 16.0, where + " clamped level of detail");
                RequireNear(LodPixels[pixel + 1u], static_cast<double>(lod), 1.0 / 16.0, where + " computed level of detail");
            }
        }
    }
}
void RequireCounts(std::uint64_t textures, std::uint64_t refreshes, const std::string& what) {
    const auto counts = AgcDriver::Graphics::Texture::TakeDecodedCopyCounts();
    Require(counts.textures == textures && counts.refreshes == refreshes, what + ": " + std::to_string(counts.textures) + " decoded copies made and " + std::to_string(counts.refreshes) + " refreshed, expected " + std::to_string(textures) + " and " + std::to_string(refreshes));
}

void StoreTexels(AgcDriver::VulkanDevice& device, const Surface& surface, std::uint32_t seed) {
    const auto width = surface.Width(0);
    const auto height = surface.Height(0);
    Input.fill(0u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto sample = y * width + x;
            Input[sample * 4u] = x;
            Input[sample * 4u + 1u] = y;
            Input[sample * 4u + 3u] = FiniteTexel(surface.format, seed + sample);
        }
    }
    Run(device, StoreRaw2D, surface.Descriptor(0xfacu, RawFormat), SamplerDescriptor(ClampEdge, Point), width * height);
}

void CheckStorageViews(AgcDriver::VulkanDevice& device, std::uint32_t format, std::uint8_t* block, const Shape& shape, std::span<const std::uint32_t> store, const std::string& what) {
    const auto name = Name(format) + " storage image, " + what;
    const Surface surface(block, format, shape);
    surface.Fill([&](std::uint32_t x, std::uint32_t y, std::uint32_t slice, std::uint32_t level) { return FiniteTexel(format, 0x8000000u | (level << 20u) | (slice << 16u) | (y << 8u) | x); });
    AgcDriver::GuestMemory::CollectWritesUncached(reinterpret_cast<std::uintptr_t>(block), static_cast<std::size_t>(surface.Bytes()));
    AgcDriver::Graphics::Texture::TakeDecodedCopyCounts();
    const auto slices = std::max(surface.Depth(0), surface.Layers());
    std::uint32_t samples = 0;
    Input.fill(0u);
    for (std::uint32_t slice = 0; slice < slices; ++slice) {
        for (std::uint32_t y = 0; y < surface.Height(0); ++y) {
            for (std::uint32_t x = 0; x < surface.Width(0); ++x) {
                Input[samples * 4u] = x;
                Input[samples * 4u + 1u] = shape.type == Type1D ? slice : y;
                Input[samples * 4u + 2u] = slice;
                Input[samples * 4u + 3u] = FiniteTexel(format, 0x9000000u + samples);
                ++samples;
            }
        }
    }
    Require(samples <= MaxSamples, name + ": too many texels to store at once");
    Run(device, store, surface.Descriptor(0xfacu, RawFormat), SamplerDescriptor(ClampEdge, Point), samples);
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(block), static_cast<std::size_t>(surface.Bytes()), nullptr, "test");
    device.WaitIdle();
    std::uint32_t sample = 0;
    for (std::uint32_t slice = 0; slice < slices; ++slice) {
        for (std::uint32_t y = 0; y < surface.Height(0); ++y) {
            for (std::uint32_t x = 0; x < surface.Width(0); ++x) {
                const auto expected = FiniteTexel(format, 0x9000000u + sample++);
                Require(surface.Texel(x, y, slice, 0) == expected, name + ": stored texel (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(slice) + ") is " + Hex(surface.Texel(x, y, slice, 0)) + ", expected " + Hex(expected));
            }
        }
    }
    CheckPointLevels(device, surface, SwizzleXYZ1, name);
    const auto counts = AgcDriver::Graphics::Texture::TakeDecodedCopyCounts();
    Require(counts.textures == 1u && counts.refreshes >= 1u, name + ": " + std::to_string(counts.textures) + " decoded copies made and " + std::to_string(counts.refreshes) + " refreshed, expected 1 made from the storage image");
}

void RequireStored(const Surface& surface, std::uint32_t seed, const std::string& what) {
    for (std::uint32_t y = 0; y < surface.Height(0); ++y) {
        for (std::uint32_t x = 0; x < surface.Width(0); ++x) {
            const auto expected = FiniteTexel(surface.format, seed + y * surface.Width(0) + x);
            Require(surface.Texel(x, y, 0, 0) == expected, what + ": guest texel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + Hex(surface.Texel(x, y, 0, 0)) + ", expected " + Hex(expected));
        }
    }
}

void CheckStoredSample(AgcDriver::VulkanDevice& device, const Surface& surface, std::uint32_t seed, const std::string& what) {
    const auto width = surface.Width(0);
    const auto height = surface.Height(0);
    Input.fill(0u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) SetInput(y * width + x, (static_cast<float>(x) + 0.5f) / static_cast<float>(width), (static_cast<float>(y) + 0.5f) / static_cast<float>(height), 0.0f, 0.0f);
    }
    Run(device, SampleLevel2D, surface.Descriptor(SwizzleXYZ1), SamplerDescriptor(ClampEdge, Point), width * height);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto sample = y * width + x;
            const auto decoded = Decode(surface.format, FiniteTexel(surface.format, seed + sample));
            for (std::uint32_t component = 0; component < 4u; ++component) RequireBits(Output[sample * 4u + component], Selected(decoded, SwizzleXYZ1, component), what + ": texel (" + std::to_string(x) + ", " + std::to_string(y) + ") component " + std::to_string(component));
        }
    }
}

void CheckSnapshotSurfaces(AgcDriver::VulkanDevice& device, std::uint32_t format, std::uint8_t* region) {
    const auto name = Name(format);
    const Surface mips(region, format, {Type2D, 16, 16, 1, 5});
    mips.Fill([&](std::uint32_t x, std::uint32_t y, std::uint32_t, std::uint32_t level) { return FiniteTexel(format, (level << 16u) | (y << 8u) | x); });
    AgcDriver::Graphics::Texture::TakeDecodedCopyCounts();
    CheckPointLevels(device, mips, SwizzleXYZ1, name + " 16x16, 5 levels, X Y Z 1");
    CheckPointLevels(device, mips, SwizzleZ0X1, name + " 16x16, 5 levels, Z 0 X 1");
    CheckBilinear(device, mips, name + " 16x16");
    CheckAddressModes(device, mips, name + " 16x16");
    CheckGather(device, mips, name + " 16x16");
    CheckLevelZeroVariants(device, mips, name + " 16x16");
    CheckSamplerStates(device, mips, name + " 16x16, 5 levels");
    CheckLod(device, mips, name + " 16x16, 5 levels");

    const Surface volume(region + RegionBytes, format, {Type3D, 8, 4, 4, 1});
    volume.Fill([&](std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t) { return FiniteTexel(format, 0x1000000u | (z << 16u) | (y << 8u) | x); });
    CheckPointLevels(device, volume, SwizzleXYZ1, name + " 3D 8x4x4");

    const Surface layers(region + 2u * RegionBytes, format, {Type2DArray, 8, 8, 3, 2});
    layers.Fill([&](std::uint32_t x, std::uint32_t y, std::uint32_t layer, std::uint32_t level) { return FiniteTexel(format, 0x2000000u | (level << 20u) | (layer << 16u) | (y << 8u) | x); });
    CheckPointLevels(device, layers, SwizzleZ0X1, name + " 2D array 8x8, 3 layers, 2 levels");

    const Surface cube(region + 5u * RegionBytes, format, {TypeCube, 4, 4, 6, 2});
    cube.Fill([&](std::uint32_t x, std::uint32_t y, std::uint32_t face, std::uint32_t level) { return FiniteTexel(format, 0xa000000u | (level << 20u) | (face << 16u) | (y << 8u) | x); });
    CheckPointLevels(device, cube, SwizzleXYZ1, name + " cube 4x4, 2 levels");

    const Surface line(region + 6u * RegionBytes, format, {Type1D, 16, 1, 1, 3});
    line.Fill([&](std::uint32_t x, std::uint32_t, std::uint32_t, std::uint32_t level) { return FiniteTexel(format, 0xb000000u | (level << 16u) | x); });
    CheckPointLevels(device, line, SwizzleYZX0, name + " 1D 16, 3 levels");

    const Surface edges(region + 3u * RegionBytes, format, {Type2D, 8, 8, 1, 1});
    edges.Fill([&](std::uint32_t x, std::uint32_t y, std::uint32_t, std::uint32_t) {
        if (format == UnormFormat) {
            constexpr std::array<std::uint32_t, 4> codes{0u, 1u, 1023u, 2047u};
            return codes[x % 4u] | (codes[(x / 4u + y) % 4u] << 11u) | ((codes[y % 4u] & 1023u) << 22u);
        }
        constexpr std::array<std::uint32_t, 8> elevens{0x000u, 0x001u, 0x03fu, 0x040u, 0x7bfu, 0x7c0u, 0x7c1u, 0x7ffu};
        constexpr std::array<std::uint32_t, 8> tens{0x000u, 0x001u, 0x01fu, 0x020u, 0x3dfu, 0x3e0u, 0x3e1u, 0x3ffu};
        return tens[x] | (elevens[y] << 10u) | (elevens[(x + y) % 8u] << 21u);
    });
    CheckPointLevels(device, edges, SwizzleXYZ1, name + " zero, denormal, largest, infinity and NaN codes");

    edges.Fill([&](std::uint32_t x, std::uint32_t y, std::uint32_t, std::uint32_t) { return FiniteTexel(format, 0x3000000u | (y << 8u) | x); });
    CheckPointLevels(device, edges, SwizzleXYZ1, name + " after the CPU rewrote the texels");
}

void RequireViewsOffRecordingThread(const AgcDriver::Graphics::ViewCreationTotals& before, const AgcDriver::Graphics::ViewCreationTotals& after, const std::string& what) {
    using AgcDriver::Graphics::ViewKind;
    const auto index = [](ViewKind kind) { return static_cast<std::size_t>(kind); };
    for (std::size_t kind = 0; kind < AgcDriver::Graphics::ViewCreationTotals::Kinds; ++kind) {
        Require(after.onCaller[kind] == before.onCaller[kind], what + ": view kind " + std::to_string(kind) + " was created on the recording thread " + std::to_string(after.onCaller[kind] - before.onCaller[kind]) + " times");
        Require(after.waits[kind] == before.waits[kind], what + ": the recording thread waited " + std::to_string(after.waits[kind] - before.waits[kind]) + " times for a view of kind " + std::to_string(kind));
    }
    for (const auto kind : {ViewKind::DefaultStorage, ViewKind::DecodeSource, ViewKind::DecodedCopy}) Require(after.onPool[index(kind)] > before.onPool[index(kind)], what + ": no view of kind " + std::to_string(index(kind)) + " was made on the view pool");
}

void CheckStorageSource(AgcDriver::VulkanDevice& device, std::uint32_t format, std::uint8_t* block) {
    const auto name = Name(format) + " storage image";
    const Surface surface(block, format, {Type2D, 16, 8, 1, 2});
    surface.Fill([&](std::uint32_t x, std::uint32_t y, std::uint32_t, std::uint32_t level) { return FiniteTexel(format, 0x4000000u | (level << 16u) | (y << 8u) | x); });
    AgcDriver::GuestMemory::CollectWritesUncached(reinterpret_cast<std::uintptr_t>(block), static_cast<std::size_t>(surface.Bytes()));
    AgcDriver::Graphics::Texture::TakeDecodedCopyCounts();
    const auto viewsBefore = AgcDriver::Graphics::ViewCreationCounts();
    StoreTexels(device, surface, 0x5000000u);
    CheckStoredSample(device, surface, 0x5000000u, name + ", first store");
    RequireCounts(1, 1, name + ", first sample of the pending results");
    RequireViewsOffRecordingThread(viewsBefore, AgcDriver::Graphics::ViewCreationCounts(), name + ", first store and sample");
    StoreTexels(device, surface, 0x6000000u);
    CheckStoredSample(device, surface, 0x6000000u, name + ", second store");
    RequireCounts(0, 1, name + ", sample after a second store");
    CheckStoredSample(device, surface, 0x6000000u, name + ", unchanged");
    RequireCounts(0, 0, name + ", sample of unchanged results");
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(block), static_cast<std::size_t>(surface.Bytes()), nullptr, "test");
    device.WaitIdle();
    RequireStored(surface, 0x6000000u, name + ", flushed results");
    CheckPointLevels(device, surface, SwizzleXYZ1, name + ", both levels after the results were stored");
    AgcDriver::Graphics::Texture::TakeDecodedCopyCounts();
    CheckPointLevels(device, surface, SwizzleXYZ1, name + ", both levels again");
    RequireCounts(0, 0, name + ", second sample of both levels");
    for (std::uint32_t y = 0; y < surface.Height(0); ++y) {
        for (std::uint32_t x = 0; x < surface.Width(0); ++x) {
            const auto texel = FiniteTexel(format, 0x7000000u + y * surface.Width(0) + x);
            std::memcpy(surface.At(x, y, 0, 0), &texel, 4u);
        }
    }
    CheckStoredSample(device, surface, 0x7000000u, name + ", after the CPU rewrote the texels");
    const auto counts = AgcDriver::Graphics::Texture::TakeDecodedCopyCounts();
    Require(counts.refreshes == 1u, name + ", after the CPU rewrote the texels: " + std::to_string(counts.refreshes) + " refreshes, expected 1");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        CheckSnapshotSurfaces(*device, UnormFormat, Snapshots.data());
        CheckSnapshotSurfaces(*device, FloatFormat, Snapshots.data() + 8u * RegionBytes);
        GuestBlock block;
        CheckStorageSource(*device, UnormFormat, block.Data());
        CheckStorageSource(*device, FloatFormat, block.Data() + RegionBytes);
        for (const auto format : {UnormFormat, FloatFormat}) {
            auto* region = block.Data() + (format == UnormFormat ? 2u : 5u) * RegionBytes;
            CheckStorageViews(*device, format, region, {Type1D, 64, 1, 1, 2}, StoreRaw1D, "1D 64, 2 levels");
            CheckStorageViews(*device, format, region + RegionBytes, {Type2DArray, 8, 8, 3, 2}, StoreRaw2DArray, "2D array 8x8, 3 layers, 2 levels");
            CheckStorageViews(*device, format, region + 2u * RegionBytes, {Type3D, 8, 4, 4, 1}, StoreRaw3D, "3D 8x4x4");
        }
        std::puts("image converted sample tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
