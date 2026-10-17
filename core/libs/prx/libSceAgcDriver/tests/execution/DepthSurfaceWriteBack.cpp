#include "GuestTestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSampleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "DepthFastClearHarness.hpp"
#include "VulkanTestDevice.hpp"
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
};

void DrawPlanes(AgcDriver::VulkanDevice& device, const Shaders& compiled, const Planes& planes, bool depthWrite, bool stencilWrite) {
    Pixels.fill(std::byte{0});
    const std::array<Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &compiled.vertex, 0},
        {ShaderStage::Fragment, &compiled.pixel, PixelPushOffset(compiled.vertexPush, device.Target())}
    }};
    Graphics::State state{};
    state.stages = {Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    if (planes.samples == 1u) {
        state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
        state.colors = {state.color};
        state.hasColorTarget = true;
        state.blend.colorWriteMask = 15;
        state.blends = {state.blend};
    } else {
        state.hasColorTarget = false;
    }
    state.samples = planes.samples;
    state.depth = Graphics::DepthTarget{planes.depth, planes.stencil, {Width, Height}, planes.format, 1.0f, 0, planes.htile, false, planes.samples};
    state.depth->depthSwizzle = static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX);
    if (planes.htile != 0u) state.depth->htileDepthOnly = state.depth->iterate256 = state.depth->htilePipeAligned = true;
    state.depthTest = true;
    state.depthWrite = depthWrite;
    state.depthCompare = VK_COMPARE_OP_ALWAYS;
    if (stencilWrite) {
        state.stencilTest = true;
        state.stencilFront = {VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS, 0xffu, 0xffu, StencilValue};
        state.stencilBack = state.stencilFront;
    }
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
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

std::uint64_t GuestOffset(std::uint32_t x, std::uint32_t y, std::uint32_t sample, std::uint32_t elementBytes, std::uint32_t samples) {
    const auto* equation = Graphics::FindDepthSampleEquation(samples, elementBytes);
    Require(equation != nullptr, "depth write-back: no equation for the plane");
    const auto sampleBits = static_cast<std::uint32_t>(std::countr_zero(samples));
    const auto pixelBits = 16u - static_cast<std::uint32_t>(std::countr_zero(elementBytes)) - sampleBits;
    const auto widthBits = (pixelBits + ((sampleBits & 1u) == 0u ? 1u : 0u)) / 2u;
    const auto heightBits = pixelBits - widthBits;
    const auto blocksPerRow = (Width + (1u << widthBits) - 1u) >> widthBits;
    const auto block = (y >> heightBits) * blocksPerRow + (x >> widthBits);
    std::uint32_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16u; ++bit) {
        const auto mask = equation->bits[bit];
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

    GuestPlane() : data(TestGuest::Allocate(PlaneBytes, PlaneBytes)), address(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data))) {
        auto* words = reinterpret_cast<volatile std::uint32_t*>(data);
        for (std::size_t word = 0; word < PlaneBytes / 4u; ++word) words[word] = Untouched;
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

void CheckStencil(std::uint64_t address, std::uint32_t samples, const std::string& what) {
    const auto read = ReadPlane(address);
    std::vector<bool> element(read.size(), false);
    std::size_t wrong = 0;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            for (std::uint32_t sample = 0; sample < samples; ++sample) {
                const auto offset = GuestOffset(x, y, sample, 1u, samples);
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
    CheckDepth(depth.address, planes.format, 4u, true, what);
    bool refused = false;
    try {
        Graphics::RequireNoResidentDepthRead_nid_no_patch(htile.address, 4u);
    } catch (const std::exception& error) {
        refused = std::string(error.what()).find("sample-preserving guest publication") != std::string::npos;
    }
    Require(refused, what + ": a read of the HTILE, which is not published, was not refused");
    DrawPlanes(device, sloped, planes, true, false);
    Require(Stale(depth.address) != 0u, what + ": the second draw left no stale subpage");
    CheckDepth(depth.address, planes.format, 4u, true, what + ", after a second draw over the published plane");
    std::printf("%s reaches guest memory, its HTILE stays refused\n", what.c_str());
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
        ConvertedDepth(*device, sloped);
        LibcRunShutdown_nid_postfix();
        std::puts("depth surface write-back tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
