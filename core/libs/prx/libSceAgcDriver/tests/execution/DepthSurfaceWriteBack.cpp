#include "GuestTestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSampleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HtileEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "DepthFastClearHarness.hpp"
#include "VulkanTestDevice.hpp"
#ifndef _WIN32
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <sys/mman.h>
#endif
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using DepthFastClearHarness::Height;
using DepthFastClearHarness::PixelCode;
using DepthFastClearHarness::Pixels;
using DepthFastClearHarness::Require;
using DepthFastClearHarness::VertexCode;
using DepthFastClearHarness::Width;
using ShaderRecompiler::ShaderStage;
namespace Graphics = AgcDriver::Graphics;

constexpr std::size_t PlaneBytes = 65536;
constexpr std::uint32_t Untouched = 0x5a5a5a5au;
constexpr std::uint8_t StencilValue = 0x3cu;
constexpr std::uint32_t StoreThreads = 32;

alignas(256) constexpr std::array<std::uint32_t, 9> StoreLaneDepthCode{
    0x7e280d00u, 0x102828ffu, 0x3c800000u, 0x7e3c0300u, 0x7e3e0280u, 0xf0201108u, 0x0001141eu, 0xbf810000u, 0xbf810000u,
};

struct Shaders {
    ShaderRecompiler::RecompileResult vertex;
    ShaderRecompiler::RecompileResult pixel;
    std::uint32_t vertexPush = 0;
};

const TestGuest::Vector<std::array<float, 4>>& Triangle(bool sloped) {
    static const TestGuest::Vector<std::array<float, 4>> flat{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}};
    static const TestGuest::Vector<std::array<float, 4>> slope{{-1.0f, -1.0f, 0.0f, 1.0f}, {3.0f, -1.0f, 1.0f, 1.0f}, {-1.0f, 3.0f, 0.0f, 1.0f}};
    return sloped ? slope : flat;
}

Shaders Compile(AgcDriver::VulkanDevice& device, bool sloped) {
    const auto target = device.Target();
    Shaders shaders;
    TestGuest::Vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = DepthFastClearHarness::BufferDescriptor(Triangle(sloped).data(), 16u, static_cast<std::uint32_t>(Triangle(sloped).size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {32, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    shaders.vertex = ShaderRecompiler::Recompile(vertex);
    shaders.vertexPush = static_cast<std::uint32_t>(shaders.vertex.pushConstants.size());
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.wave32 = true;
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    TestGuest::Vector<std::uint32_t> pixelUserData(4, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {32, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(shaders.vertexPush, target)
    };
    fragment.useCache = false;
    shaders.pixel = ShaderRecompiler::Recompile(fragment);
    return shaders;
}

struct Planes {
    std::uint64_t depth = 0;
    std::uint64_t stencil = 0;
    VkFormat format = VK_FORMAT_D32_SFLOAT;
    std::uint32_t samples = 1;
    std::uint64_t htile = 0;
    bool htileStencil = false;
    std::uint32_t stencilSwizzle = static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX);
    VkExtent2D extent{Width, Height};
};

void DrawPlanes(AgcDriver::VulkanDevice& device, const Shaders& compiled, const Planes& planes, bool depthWrite, bool stencilWrite) {
    Pixels.fill(std::byte{0});
    const std::array<Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &compiled.vertex, 0},
        {ShaderStage::Fragment, &compiled.pixel, PixelPushOffset(compiled.vertexPush, device.Target())}
    }};
    Graphics::State state{};
    state.stages = {Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    if (planes.samples == 1u && planes.extent.width == Width && planes.extent.height == Height) {
        state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
        state.colors = {state.color};
        state.hasColorTarget = true;
        state.blend.colorWriteMask = 15;
        state.blends = {state.blend};
    } else {
        state.hasColorTarget = false;
    }
    state.samples = planes.samples;
    state.depth = Graphics::DepthTarget{planes.depth, planes.stencil, planes.extent, planes.format, 1.0f, 0, planes.htile, planes.htileStencil, planes.samples};
    state.depth->depthSwizzle = static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX);
    state.depth->stencilSwizzle = planes.stencilSwizzle;
    if (planes.htile != 0u) {
        state.depth->htileDepthOnly = !planes.htileStencil;
        state.depth->iterate256 = state.depth->htilePipeAligned = true;
    }
    state.depthTest = true;
    state.depthWrite = depthWrite;
    state.depthCompare = VK_COMPARE_OP_ALWAYS;
    if (stencilWrite) {
        state.stencilTest = true;
        state.stencilFront = {VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS, 0xffu, 0xffu, StencilValue};
        state.stencilBack = state.stencilFront;
    }
    state.renderExtent = planes.extent;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(planes.extent.height), static_cast<float>(planes.extent.width), -static_cast<float>(planes.extent.height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, planes.extent};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, 3u, 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
}

bool Owned(std::uint64_t address) {
    auto* residency = Graphics::ResidencyDriver::Active();
    Require(residency != nullptr, "depth write-back: the device has no residency driver");
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    return residency->Manager().OwnerOf(address >> Graphics::ResidencyPageShift) != nullptr;
}

Graphics::ResidencyMask Stale(std::uint64_t address) {
    auto* residency = Graphics::ResidencyDriver::Active();
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    return residency->Manager().Stale(address >> Graphics::ResidencyPageShift);
}

const std::array<std::uint32_t, 16>& EquationFor(std::uint32_t swizzle, std::uint32_t elementBytes, std::uint32_t samples) {
    if (samples == 1u) {
        for (const auto& equation : Graphics::kTextureSwizzleEquations) {
            if (equation.swizzleMode == swizzle && equation.elementBytes == elementBytes) return equation.bits;
        }
    }
    const auto* equation = Graphics::FindDepthSampleEquation(samples, elementBytes);
    Require(equation != nullptr && swizzle == static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX), "depth write-back: no equation for the plane");
    return equation->bits;
}

std::uint64_t GuestOffset(std::uint32_t x, std::uint32_t y, std::uint32_t sample, std::uint32_t elementBytes, std::uint32_t samples, std::uint32_t swizzle = static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX), std::uint32_t width = Width) {
    const auto& bits = EquationFor(swizzle, elementBytes, samples);
    const auto sampleBits = static_cast<std::uint32_t>(std::countr_zero(samples));
    const auto pixelBits = 16u - static_cast<std::uint32_t>(std::countr_zero(elementBytes)) - sampleBits;
    const auto widthBits = (pixelBits + ((sampleBits & 1u) == 0u ? 1u : 0u)) / 2u;
    const auto heightBits = pixelBits - widthBits;
    const auto blocksPerRow = (width + (1u << widthBits) - 1u) >> widthBits;
    const auto block = (y >> heightBits) * blocksPerRow + (x >> widthBits);
    std::uint32_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16u; ++bit) {
        const auto mask = bits[bit];
        const auto selected = (x & mask & 0xfffu) ^ ((y << 12u) & mask & 0xfff000u) ^ ((sample << 24u) & mask & 0xff000000u);
        offset |= static_cast<std::uint32_t>(std::popcount(selected) & 1) << bit;
    }
    return std::uint64_t{block} * 65536u + offset;
}

constexpr std::array<float, 4> Standard4x{0.375f, 0.875f, 0.125f, 0.625f};

float SampleX(std::uint32_t samples, std::uint32_t sample) {
    return samples == 1u ? 0.5f : Standard4x[sample];
}

std::uint32_t Unorm16(double value) {
    return static_cast<std::uint32_t>(std::lround(std::min(1.0, std::max(0.0, value)) * 65535.0));
}

struct GuestPlane {
    std::byte* data;
    std::uint64_t address;

    explicit GuestPlane(std::size_t bytes = PlaneBytes) : data(TestGuest::Allocate(bytes, PlaneBytes)), address(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data))) {
        auto* words = reinterpret_cast<volatile std::uint32_t*>(data);
        for (std::size_t word = 0; word < bytes / 4u; ++word) words[word] = Untouched;
    }
};

std::vector<std::byte> ReadPlane(std::uint64_t address) {
    std::vector<std::byte> read(PlaneBytes);
    AgcDriver::GuestMemory::Read(address, std::span(read));
    return read;
}

void CheckUntouched(const std::vector<std::byte>& read, const std::vector<bool>& element, const std::string& what) {
    std::size_t changed = 0;
    for (std::size_t byte = 0; byte < read.size(); ++byte) {
        const auto expected = static_cast<std::byte>(Untouched >> ((byte & 3u) * 8u));
        if (!element[byte] && read[byte] != expected) ++changed;
    }
    Require(changed == 0u, what + ": " + std::to_string(changed) + " bytes outside the plane's elements changed");
}

void CheckHtile(std::uint64_t address, std::uint64_t bytes, std::uint32_t expected, const std::string& what) {
    std::vector<std::uint32_t> read(static_cast<std::size_t>(bytes / 4u));
    AgcDriver::GuestMemory::Read(address, std::as_writable_bytes(std::span(read)));
    std::size_t wrong = 0;
    for (const auto word : read) wrong += word != expected ? 1u : 0u;
    Require(wrong == 0u, what + ": " + std::to_string(wrong) + " HTILE words read after the draw are not the expanded word");
}

void CheckDepth(std::uint64_t address, VkFormat format, std::uint32_t samples, bool sloped, const std::string& what) {
    const auto read = ReadPlane(address);
    const bool d16 = format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto elementBytes = d16 ? 2u : 4u;
    std::vector<bool> element(read.size(), false);
    std::set<std::uint64_t> offsets;
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            for (std::uint32_t sample = 0; sample < samples; ++sample) {
                const auto offset = GuestOffset(x, y, sample, elementBytes, samples);
                Require(offset + elementBytes <= read.size() && offsets.insert(offset).second, what + ": the guest layout maps two elements to one address or past the plane");
                for (std::uint32_t byte = 0; byte < elementBytes; ++byte) element[offset + byte] = true;
                const double expected = sloped ? (x + SampleX(samples, sample)) / 128.0 : 0.5;
                bool good = false;
                std::string actual;
                if (d16) {
                    std::uint16_t value = 0;
                    std::memcpy(&value, &read[offset], 2u);
                    good = std::abs(static_cast<int>(value) - static_cast<int>(Unorm16(expected))) <= 1;
                    actual = std::to_string(value) + ", expected " + std::to_string(Unorm16(expected));
                } else {
                    float value = 0.0f;
                    std::memcpy(&value, &read[offset], 4u);
                    good = std::abs(value - expected) <= 1.0e-5;
                    actual = std::to_string(value) + ", expected " + std::to_string(expected);
                }
                if (!good && wrong++ == 0u) first = "(" + std::to_string(x) + ", " + std::to_string(y) + ") sample " + std::to_string(sample) + " holds " + actual;
            }
        }
    }
    Require(wrong == 0u, what + ": a CPU read after the draw misses the drawn depth in " + std::to_string(wrong) + " elements, first " + first);
    CheckUntouched(read, element, what);
}

void CheckStencil(std::uint64_t address, std::uint32_t samples, const std::string& what, std::uint32_t swizzle = static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX)) {
    const auto read = ReadPlane(address);
    std::vector<bool> element(read.size(), false);
    std::size_t wrong = 0;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            for (std::uint32_t sample = 0; sample < samples; ++sample) {
                const auto offset = GuestOffset(x, y, sample, 1u, samples, swizzle);
                element[offset] = true;
                if (read[offset] != std::byte{StencilValue}) ++wrong;
            }
        }
    }
    Require(wrong == 0u, what + ": a CPU read after the draw misses the drawn stencil in " + std::to_string(wrong) + " elements");
    CheckUntouched(read, element, what);
}

void DrawnDepthReachesMemory(AgcDriver::VulkanDevice& device, const Shaders& flat) {
    GuestPlane plane;
    Require(Graphics::DepthSliceBytes({Width, Height}, 4u) == PlaneBytes, "depth write-back: the plane is not one 64 KiB block");
    const Planes planes{plane.address, 0, VK_FORMAT_D32_SFLOAT, 1u};
    DrawPlanes(device, flat, planes, true, false);
    Require(Owned(plane.address) && Stale(plane.address) != 0u, "depth write-back: the depth draw left no stale subpage in the surface's page");
    const auto read = ReadPlane(plane.address);
    const Graphics::ColorTargetLayout layout(Width, Height, Graphics::ColorTileMode::Z64KBX, 4u);
    std::vector<bool> element(PlaneBytes, false);
    std::size_t wrong = 0;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const auto offset = layout.Offset(x, y);
            for (std::uint32_t byte = 0; byte < 4u; ++byte) element[offset + byte] = true;
            float value = 0.0f;
            std::memcpy(&value, &read[offset], sizeof(value));
            if (value != 0.5f) ++wrong;
        }
    }
    Require(wrong == 0u, "depth write-back: a CPU read after the draw misses the drawn depth in " + std::to_string(wrong) + " texels of the color-layout addressing");
    CheckUntouched(read, element, "single-sample depth");
    Require(Stale(plane.address) == 0u, "depth write-back: the CPU read left the page stale");
    DrawPlanes(device, flat, planes, false, false);
    Require(Stale(plane.address) == 0u, "depth write-back: a draw that writes no depth marked the plane stale");
    std::printf("single-sample depth reaches guest memory\n");
}

void DepthAndStencilReachMemory(AgcDriver::VulkanDevice& device, const Shaders& sloped, std::uint32_t samples) {
    GuestPlane depth;
    GuestPlane stencil;
    const Planes planes{depth.address, stencil.address, VK_FORMAT_D32_SFLOAT_S8_UINT, samples};
    const auto what = std::string(samples == 1u ? "single-sample" : "multisampled") + " depth with stencil";
    DrawPlanes(device, sloped, planes, true, true);
    Require(Owned(depth.address) && Owned(stencil.address), what + ": the surface does not own both planes");
    Require(Stale(depth.address) != 0u && Stale(stencil.address) != 0u, what + ": the draw left a plane without stale subpages");
    CheckDepth(depth.address, planes.format, samples, true, what + ", depth plane");
    CheckStencil(stencil.address, samples, what + ", stencil plane");
    std::printf("%s reaches guest memory\n", what.c_str());
}

void MultisampledDepthReachesMemory(AgcDriver::VulkanDevice& device, const Shaders& sloped, VkFormat format) {
    GuestPlane depth;
    const Planes planes{depth.address, 0, format, 4u};
    const auto what = std::string("multisampled ") + (format == VK_FORMAT_D16_UNORM ? "16-bit" : "32-bit") + " depth";
    DrawPlanes(device, sloped, planes, true, false);
    Require(Owned(depth.address) && Stale(depth.address) != 0u, what + ": the draw left no stale subpage in an owned plane");
    CheckDepth(depth.address, format, 4u, true, what);
    std::printf("%s reaches guest memory\n", what.c_str());
}

void NativeMultisampledDepth(AgcDriver::VulkanDevice& device, const Shaders& sloped) {
    GuestPlane depth;
    GuestPlane htile;
    const auto htileBytes = Graphics::HtileMetadataBytes({Width, Height});
    auto* words = reinterpret_cast<volatile std::uint32_t*>(htile.data);
    for (std::size_t word = 0; word < htileBytes / 4u; ++word) words[word] = 0xfffffff0u;
    const Planes planes{depth.address, 0, VK_FORMAT_D32_SFLOAT, 4u, htile.address};
    const std::string what = "natively sampled multisampled depth";
    DrawPlanes(device, sloped, planes, true, false);
    Require(Owned(depth.address) && Stale(depth.address) != 0u, what + ": the draw left no stale subpage in an owned plane");
    Require(Graphics::HasResidentDepthGuards_nid_no_patch(), what + ": the target is not natively sampled");
    CheckHtile(htile.address, htileBytes, 0xfffc000fu, what);
    CheckDepth(depth.address, planes.format, 4u, true, what);
    DrawPlanes(device, sloped, planes, true, false);
    Require(Stale(depth.address) != 0u, what + ": the second draw left no stale subpage");
    CheckDepth(depth.address, planes.format, 4u, true, what + ", after a second draw over the published plane");
    CheckHtile(htile.address, htileBytes, 0xfffc000fu, what + ", after a second draw");
    std::printf("%s reaches guest memory with its HTILE expanded\n", what.c_str());
}

void HtileWithStencil(AgcDriver::VulkanDevice& device, const Shaders& sloped) {
    GuestPlane depth;
    GuestPlane stencil;
    GuestPlane htile;
    const auto htileBytes = Graphics::HtileMetadataBytes({Width, Height});
    auto* words = reinterpret_cast<volatile std::uint32_t*>(htile.data);
    for (std::size_t word = 0; word < htileBytes / 4u; ++word) words[word] = 0x00000105u;
    Planes planes{depth.address, stencil.address, VK_FORMAT_D32_SFLOAT_S8_UINT, 1u, htile.address};
    planes.htileStencil = true;
    const std::string what = "single-sample depth and stencil with HTILE";
    DrawPlanes(device, sloped, planes, true, true);
    Require(Owned(htile.address) && Stale(htile.address) != 0u, what + ": the HTILE is not owned and stale after the draw");
    CheckHtile(htile.address, htileBytes, 0xfffff3ffu, what);
    Require(Stale(depth.address) == 0u && Stale(stencil.address) == 0u, what + ": the HTILE read left a plane stale");
    CheckDepth(depth.address, planes.format, 1u, true, what + ", depth plane");
    CheckStencil(stencil.address, 1u, what + ", stencil plane");
    std::printf("%s reaches guest memory with its HTILE expanded\n", what.c_str());
}

void PartialHtilePublication(AgcDriver::VulkanDevice& device, const Shaders& sloped) {
    constexpr VkExtent2D Extent{512u, 256u};
    constexpr std::uint32_t Before = 0x00000105u;
    constexpr std::uint32_t Expanded = 0xfffc000fu;
    constexpr std::uint32_t TilesX = Extent.width / 8u;
    constexpr std::uint32_t TilesY = Extent.height / 8u;
    constexpr auto Swizzle = static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX);
    const auto depthBytes = static_cast<std::size_t>(Graphics::DepthSliceBytes(Extent, 4u));
    const auto htileBytes = static_cast<std::size_t>(Graphics::HtileMetadataBytes(Extent));
    GuestPlane depth(depthBytes);
    GuestPlane htile;
    auto* words = reinterpret_cast<volatile std::uint32_t*>(htile.data);
    for (std::size_t word = 0; word < htileBytes / 4u; ++word) words[word] = Before;
    Planes planes{depth.address, 0, VK_FORMAT_D32_SFLOAT, 1u, htile.address};
    planes.extent = Extent;
    const std::string what = "a partial write-back of single-sample depth with HTILE";
    constexpr std::uint64_t ReadPage = 1u;
    std::set<std::uint32_t> published;
    for (std::uint32_t y = 0; y < Extent.height; ++y) {
        for (std::uint32_t x = 0; x < Extent.width; ++x) {
            if (GuestOffset(x, y, 0u, 4u, 1u, Swizzle, Extent.width) / PlaneBytes == ReadPage) published.insert((y / 8u) * TilesX + x / 8u);
        }
    }
    Require(!published.empty() && published.size() < TilesX * TilesY, what + ": the read page holds no tiles or all of them");
    const auto expected = [&](std::uint32_t x) { return (x + 0.5) / (2.0 * Extent.width); };
    DrawPlanes(device, sloped, planes, true, false);
    Require(Owned(htile.address) && Stale(htile.address) != 0u && Stale(depth.address) != 0u, what + ": the draw left the HTILE or the plane without stale subpages");
    std::vector<std::byte> part(Graphics::ResidencySubpageBytes);
    AgcDriver::GuestMemory::Read(depth.address + ReadPage * PlaneBytes, std::span(part));
    const auto* equation = Graphics::FindHtileEquation(1u);
    std::size_t expandedWrong = 0;
    std::size_t untouchedWrong = 0;
    for (std::uint32_t tileY = 0; tileY < TilesY; ++tileY) {
        for (std::uint32_t tileX = 0; tileX < TilesX; ++tileX) {
            const auto word = words[Graphics::HtileWordOffset(*equation, Extent.width, tileX * 8u, tileY * 8u, 0u) / 4u];
            if (published.contains(tileY * TilesX + tileX)) expandedWrong += word != Expanded ? 1u : 0u;
            else untouchedWrong += word != Before ? 1u : 0u;
        }
    }
    Require(expandedWrong == 0u, what + ": " + std::to_string(expandedWrong) + " of the read page's " + std::to_string(published.size()) + " tiles kept their old HTILE word");
    Require(untouchedWrong == 0u, what + ": " + std::to_string(untouchedWrong) + " HTILE words of tiles outside the read page changed");
    std::size_t planeWrong = 0;
    for (std::uint32_t y = 0; y < Extent.height; ++y) {
        for (std::uint32_t x = 0; x < Extent.width; ++x) {
            const auto offset = GuestOffset(x, y, 0u, 4u, 1u, Swizzle, Extent.width);
            float value = 0.0f;
            std::uint32_t bits = 0;
            std::memcpy(&value, depth.data + offset, sizeof(value));
            std::memcpy(&bits, depth.data + offset, sizeof(bits));
            const bool inside = published.contains((y / 8u) * TilesX + x / 8u);
            if (inside ? std::abs(value - expected(x)) > 1.0e-5 : bits != Untouched) ++planeWrong;
        }
    }
    Require(planeWrong == 0u, what + ": " + std::to_string(planeWrong) + " depth elements are not published exactly for the read page's tiles");
    Require(Stale(htile.address) != 0u && Stale(depth.address) != 0u, what + ": the partial write-back published the whole surface");
    CheckHtile(htile.address, htileBytes, Expanded, what + ", the whole HTILE read afterwards");
    std::vector<std::byte> whole(depthBytes);
    AgcDriver::GuestMemory::Read(depth.address, std::span(whole));
    std::size_t wholeWrong = 0;
    for (std::uint32_t y = 0; y < Extent.height; ++y) {
        for (std::uint32_t x = 0; x < Extent.width; ++x) {
            float value = 0.0f;
            std::memcpy(&value, &whole[GuestOffset(x, y, 0u, 4u, 1u, Swizzle, Extent.width)], sizeof(value));
            if (std::abs(value - expected(x)) > 1.0e-5) ++wholeWrong;
        }
    }
    Require(wholeWrong == 0u, what + ": the whole plane read afterwards misses the drawn depth in " + std::to_string(wholeWrong) + " elements");
    std::printf("%s publishes %zu of %u tiles and leaves the other HTILE words untouched\n", what.c_str(), published.size(), TilesX * TilesY);
}

ShaderRecompiler::RecompileResult StoreLaneDepth(AgcDriver::VulkanDevice& device, std::uint64_t plane) {
    constexpr std::uint32_t Format32Float = 22u;
    constexpr std::uint32_t Type2D = 9u;
    TestGuest::Vector<std::uint32_t> userData(16, 0u);
    const std::array<std::uint32_t, 8> texture{
        static_cast<std::uint32_t>(plane >> 8u),
        static_cast<std::uint32_t>((plane >> 40u) & 0xffu) | (Format32Float << 20u) | (((Width - 1u) & 3u) << 30u),
        ((Width - 1u) >> 2u) | ((Height - 1u) << 14u),
        0xfacu | (static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX) << 20u) | (Type2D << 28u),
        0u, 0u, 0u, 0u,
    };
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(StoreLaneDepthCode.data()), std::as_bytes(std::span(StoreLaneDepthCode))}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{StoreThreads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(StoreLaneDepthCode.data()), StoreLaneDepthCode, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

void StorageWriteMarksPlaneStale(AgcDriver::VulkanDevice& device, const Shaders& flat) {
    GuestPlane depth;
    GuestPlane htile;
    const auto htileBytes = Graphics::HtileMetadataBytes({Width, Height});
    auto* words = reinterpret_cast<volatile std::uint32_t*>(htile.data);
    for (std::size_t word = 0; word < htileBytes / 4u; ++word) words[word] = 0x00000105u;
    const Planes planes{depth.address, 0, VK_FORMAT_D32_SFLOAT, 1u, htile.address};
    const std::string what = "a storage write into an owned depth plane with HTILE";
    DrawPlanes(device, flat, planes, true, false);
    CheckHtile(htile.address, htileBytes, 0xfffc000fu, what + ", after the draw");
    static_cast<void>(ReadPlane(depth.address));
    Require(Owned(depth.address) && Stale(depth.address) == 0u && Stale(htile.address) == 0u, what + ": the reads after the draw left the plane or the HTILE stale");
    const auto program = StoreLaneDepth(device, depth.address);
    device.Dispatch(program, 1, 1, 1, reinterpret_cast<std::uintptr_t>(StoreLaneDepthCode.data()));
    Require(Stale(depth.address) != 0u, what + ": the storage write left the surface's plane without stale subpages");
    Require(Stale(htile.address) != 0u, what + ": the storage write left the surface's HTILE without stale subpages");
    device.WaitIdle();
    const auto read = ReadPlane(depth.address);
    std::size_t wrong = 0;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            float value = 0.0f;
            std::memcpy(&value, &read[GuestOffset(x, y, 0u, 4u, 1u)], sizeof(value));
            const float expected = y == 0u && x < StoreThreads ? static_cast<float>(x) / 64.0f : 0.5f;
            if (value != expected) ++wrong;
        }
    }
    Require(wrong == 0u, what + ": a CPU read after the dispatch misses the stored or drawn depth in " + std::to_string(wrong) + " elements");
    CheckHtile(htile.address, htileBytes, 0xfffc000fu, what + ", after the storage write");
    Require(Stale(depth.address) == 0u, what + ": the CPU read left the plane stale");
    std::printf("%s marks the plane and its HTILE stale for its owner\n", what.c_str());
}

void HtileFillStoresNothing(AgcDriver::VulkanDevice& device, const Shaders& flat) {
    GuestPlane depth;
    GuestPlane htile;
    const auto htileBytes = static_cast<std::size_t>(Graphics::HtileMetadataBytes({Width, Height}));
    auto* words = reinterpret_cast<volatile std::uint32_t*>(htile.data);
    for (std::size_t word = 0; word < htileBytes / 4u; ++word) words[word] = 0x00000105u;
    const Planes planes{depth.address, 0, VK_FORMAT_D32_SFLOAT, 1u, htile.address};
    const std::string what = "a fill over the whole HTILE of a drawn depth surface";
    for (std::uint32_t frame = 0; frame < 3u; ++frame) {
        const auto at = what + ", frame " + std::to_string(frame);
        DrawPlanes(device, flat, planes, true, false);
        Require(Owned(htile.address) && Stale(htile.address) != 0u && Stale(depth.address) != 0u, at + ": the draw left the HTILE or the plane without stale subpages");
        static_cast<void>(Graphics::StorageTexture::DiscardPendingInside(htile.address, htileBytes));
        Require(!Graphics::StorageTexture::FlushPending(htile.address, htileBytes, nullptr, "buffer fill", Graphics::PublishScope::PartialUnits), at + ": the store ahead of the fill wrote the surface back although the fill overwrites every HTILE byte");
        Require(Stale(htile.address) == 0u, at + ": the HTILE the fill overwrites is still stale");
        Require(Stale(depth.address) != 0u, at + ": the fill over the HTILE published the depth plane");
        const std::array<std::uint32_t, 4> pattern{0x0000fff0u, 0x0000fff0u, 0x0000fff0u, 0x0000fff0u};
        Require(device.FillBuffer(htile.address, htileBytes, pattern), at + ": the device did not store the fill");
        device.WaitIdle();
        CheckDepth(depth.address, planes.format, 1u, false, at + ", depth plane read after the fill");
        Require(Stale(depth.address) == 0u, at + ": the CPU read left the plane stale");
        std::vector<std::uint32_t> read(htileBytes / 4u);
        AgcDriver::GuestMemory::Read(htile.address, std::as_writable_bytes(std::span(read)));
        std::size_t wrong = 0;
        for (const auto word : read) wrong += word != pattern[0] ? 1u : 0u;
        Require(wrong == 0u, at + ": " + std::to_string(wrong) + " HTILE words lost the fill's value after the plane was read");
    }
    std::printf("%s stores nothing ahead of the fill and keeps the plane's results\n", what.c_str());
}

void StencilSwizzle(AgcDriver::VulkanDevice& device, const Shaders& sloped) {
    GuestPlane depth;
    GuestPlane stencil;
    Planes planes{depth.address, stencil.address, VK_FORMAT_D32_SFLOAT_S8_UINT, 1u};
    planes.stencilSwizzle = 25u;
    const std::string what = "stencil in the 64 KiB S swizzle under Z-swizzled depth";
    DrawPlanes(device, sloped, planes, true, true);
    Require(Owned(stencil.address), what + ": the stencil plane is not owned");
    CheckDepth(depth.address, planes.format, 1u, true, what + ", depth plane");
    CheckStencil(stencil.address, 1u, what + ", stencil plane", planes.stencilSwizzle);
    std::printf("%s reaches guest memory\n", what.c_str());
}

void ConvertedDepth(AgcDriver::VulkanDevice& device, const Shaders& sloped) {
    GuestPlane depth;
    GuestPlane stencil;
    const Planes planes{depth.address, stencil.address, VK_FORMAT_D16_UNORM_S8_UINT, 1u};
    const auto host = Graphics::HostDepthFormat(AgcDriver::VulkanDeviceTestAccess::ContextOf(device), VK_FORMAT_D16_UNORM_S8_UINT);
    DrawPlanes(device, sloped, planes, true, true);
    Require(Owned(stencil.address), "converted depth: the stencil plane is not owned");
    CheckStencil(stencil.address, 1u, "converted depth, stencil plane");
    if (host == VK_FORMAT_D24_UNORM_S8_UINT) {
        Require(!Owned(depth.address), "converted depth: a 16-bit depth plane stored as 24-bit depth is owned although it cannot convert back exactly");
        std::printf("16-bit depth stored as 24-bit depth stays unowned; its stencil reaches guest memory\n");
        return;
    }
    Require(Owned(depth.address), "converted depth: the depth plane is not owned");
    CheckDepth(depth.address, planes.format, 1u, true, "converted depth, depth plane");
    std::printf("16-bit depth with stencil stored as vk format %d converts back exactly\n", static_cast<int>(host));
}

}

std::byte* RegisterAdjacent(std::size_t low, std::size_t high) {
    const auto bytes = low + high;
    std::byte* base = nullptr;
#ifdef _WIN32
    base = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536));
    if (base != nullptr) {
        GuestArena::GuestArenaCommit_nid_postfix(base, low, TestGuest::PageReadWrite, low);
        GuestArena::GuestArenaCommit_nid_postfix(base + low, high, TestGuest::PageReadWrite, high);
    }
#else
    constexpr std::uintptr_t align = 65536;
    void* mapped = mmap(nullptr, bytes + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped != MAP_FAILED) {
        const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
        const auto aligned = (begin + align - 1) & ~(align - 1);
        if (aligned != begin) munmap(mapped, aligned - begin);
        if (aligned + bytes != begin + bytes + align) munmap(reinterpret_cast<void*>(aligned + bytes), begin + align - aligned);
        base = reinterpret_cast<std::byte*>(aligned);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(base, bytes);
    }
#endif
    Require(base != nullptr, "depth write-back: the adjacent guest ranges cannot be allocated");
    auto* words = reinterpret_cast<volatile std::uint32_t*>(base);
    for (std::size_t word = 0; word < bytes / 4u; ++word) words[word] = Untouched;
    GuestAllocations::Mutation().Add(base, low, true, true, true);
    GuestAllocations::Mutation().Add(base + low, high, true, true, true);
    return base;
}

void PlaneAcrossAdjacentRanges(AgcDriver::VulkanDevice& device, const Shaders& flat) {
    constexpr std::size_t MiB = std::size_t{1} << 20u;
    constexpr VkExtent2D extent{1536u, 1408u};
    const auto planeBytes = Graphics::DepthSliceBytes(extent, 4u);
    auto* base = RegisterAdjacent(12u * MiB, 10u * MiB);
    const auto boundary = reinterpret_cast<std::uint64_t>(base) + 12u * MiB;
    const auto address = boundary - 4u * MiB;
    Require(planeBytes > 8u * MiB && address + planeBytes > boundary && address + planeBytes <= boundary + 10u * MiB, "depth write-back: the plane does not cross the two registered ranges");
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    Require(Graphics::HostImportFor(context, address, static_cast<std::size_t>(planeBytes)) == nullptr, "depth write-back: a plane across two registered ranges has a windowed import");
    Planes planes{address, 0, VK_FORMAT_D32_SFLOAT, 1u};
    planes.extent = extent;
    DrawPlanes(device, flat, planes, true, false);
    Require(Owned(address) && Stale(address) != 0u, "depth write-back: the draw across two registered ranges left no stale subpage");
    std::vector<std::byte> read(static_cast<std::size_t>(planeBytes));
    AgcDriver::GuestMemory::Read(address, std::span(read));
    std::size_t wrong = 0;
    for (std::uint32_t y = 0; y < extent.height; ++y) {
        for (std::uint32_t x = 0; x < extent.width; ++x) {
            float value = 0.0f;
            std::memcpy(&value, &read[GuestOffset(x, y, 0u, 4u, 1u, static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX), extent.width)], sizeof(value));
            if (value != 0.5f) ++wrong;
        }
    }
    Require(wrong == 0u, "depth write-back: a CPU read of a plane across two registered ranges misses the drawn depth in " + std::to_string(wrong) + " texels");
    const auto* joined = Graphics::HostImportFor(context, address, static_cast<std::size_t>(planeBytes));
    Require(joined != nullptr && joined->base == reinterpret_cast<std::uint64_t>(base) && joined->bytes == 22u * MiB, "depth write-back: the plane across two registered ranges was not written back through one joined import");
    Require(Stale(address) == 0u, "depth write-back: the CPU read left the plane across two registered ranges stale");
    std::printf("a depth plane across two adjacent registered ranges is written back through one joined import\n");
}

void PairOfAdjacentRangesJoins(AgcDriver::VulkanDevice& device) {
    constexpr std::size_t MiB = std::size_t{1} << 20u;
    auto* base = RegisterAdjacent(2u * MiB, 4u * MiB);
    const auto address = reinterpret_cast<std::uint64_t>(base);
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    const auto* joined = Graphics::HostImportSpanning(context, address + MiB, 2u * MiB);
    Require(joined != nullptr && joined->base == address && joined->bytes == 6u * MiB, "depth write-back: a range across a 2 MiB and a 4 MiB registered range was not imported as one");
    Require(Graphics::HostImportFor(context, address, 6u * MiB) == joined, "depth write-back: the 2 MiB and 4 MiB pair is not served by its joined import");
    char named[64];
    std::snprintf(named, sizeof(named), "0x%llx+0x%llx", static_cast<unsigned long long>(address + 7u * MiB), static_cast<unsigned long long>(MiB));
    const auto refusal = Graphics::HostImportRefusal(context, address + 7u * MiB, MiB);
    Require(refusal.find(named) != std::string::npos && refusal.find(": ") != std::string::npos, "depth write-back: the import refusal does not name the range: " + refusal);
    std::printf("a 2 MiB and a 4 MiB adjacent registered range join into one import\n");
}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto flat = Compile(*device, false);
        const auto sloped = Compile(*device, true);
        DrawnDepthReachesMemory(*device, flat);
        DepthAndStencilReachMemory(*device, sloped, 1u);
        MultisampledDepthReachesMemory(*device, sloped, VK_FORMAT_D32_SFLOAT);
        MultisampledDepthReachesMemory(*device, sloped, VK_FORMAT_D16_UNORM);
        DepthAndStencilReachMemory(*device, sloped, 4u);
        NativeMultisampledDepth(*device, sloped);
        HtileWithStencil(*device, sloped);
        PartialHtilePublication(*device, sloped);
        StorageWriteMarksPlaneStale(*device, flat);
        HtileFillStoresNothing(*device, flat);
        StencilSwizzle(*device, sloped);
        ConvertedDepth(*device, sloped);
        PlaneAcrossAdjacentRanges(*device, flat);
        PairOfAdjacentRangesJoins(*device);
        LibcRunShutdown_nid_postfix();
        std::puts("depth surface write-back tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
