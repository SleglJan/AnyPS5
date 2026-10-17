#include "GuestTestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "DepthFastClearHarness.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using DepthFastClearHarness::Compile;
using DepthFastClearHarness::Height;
using DepthFastClearHarness::Pixels;
using DepthFastClearHarness::Require;
using DepthFastClearHarness::Shaders;
using DepthFastClearHarness::Triangles;
using DepthFastClearHarness::Width;
using ShaderRecompiler::ShaderStage;
namespace Graphics = AgcDriver::Graphics;

constexpr std::size_t PlaneBytes = 65536;
constexpr float Drawn = 0.5f;
constexpr std::uint32_t Untouched = 0x5a5a5a5au;

void DrawDepth(AgcDriver::VulkanDevice& device, const Shaders& compiled, std::uint64_t depth, bool depthWrite) {
    Pixels.fill(std::byte{0});
    const std::array<Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &compiled.vertex, 0},
        {ShaderStage::Fragment, &compiled.pixel, PixelPushOffset(compiled.vertexPush, device.Target())}
    }};
    Graphics::State state{};
    state.stages = {Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.depth = Graphics::DepthTarget{depth, 0, {Width, Height}, VK_FORMAT_D32_SFLOAT, 1.0f, 0, 0, false};
    state.depth->depthSwizzle = static_cast<std::uint32_t>(Graphics::ColorTileMode::Z64KBX);
    state.depthTest = true;
    state.depthWrite = depthWrite;
    state.depthCompare = VK_COMPARE_OP_ALWAYS;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangles().size()), 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
}

Graphics::ResidencyMask Stale(std::uint64_t address) {
    auto* residency = Graphics::ResidencyDriver::Active();
    Require(residency != nullptr, "depth write-back: the device has no residency driver");
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    Require(residency->Manager().OwnerOf(address >> Graphics::ResidencyPageShift) != nullptr, "depth write-back: the bound depth surface does not own its depth plane");
    return residency->Manager().Stale(address >> Graphics::ResidencyPageShift);
}

void DrawnDepthReachesMemory(AgcDriver::VulkanDevice& device, const Shaders& shaders) {
    auto* plane = TestGuest::Allocate(PlaneBytes, PlaneBytes);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(plane));
    auto* words = reinterpret_cast<volatile std::uint32_t*>(plane);
    for (std::size_t word = 0; word < PlaneBytes / 4u; ++word) words[word] = Untouched;
    Require(Graphics::DepthSliceBytes({Width, Height}, 4u) == PlaneBytes, "depth write-back: the plane is not one 64 KiB block");
    DrawDepth(device, shaders, address, true);
    Require(Stale(address) != 0u, "depth write-back: the depth draw left no stale subpage in the surface's page");
    std::vector<std::uint32_t> read(PlaneBytes / 4u);
    AgcDriver::GuestMemory::Read(address, std::as_writable_bytes(std::span(read)));
    const Graphics::ColorTargetLayout layout(Width, Height, Graphics::ColorTileMode::Z64KBX, 4u);
    std::vector<bool> texel(PlaneBytes / 4u, false);
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const auto word = layout.Offset(x, y) / 4u;
            texel[word] = true;
            float value = 0.0f;
            std::memcpy(&value, &read[word], sizeof(value));
            if (value != Drawn && wrong++ == 0u) first = "texel (" + std::to_string(x) + ", " + std::to_string(y) + ") holds " + std::to_string(value);
        }
    }
    Require(wrong == 0u, "depth write-back: a CPU read after the draw misses the drawn depth in " + std::to_string(wrong) + " texels, first " + first);
    std::size_t padding = 0;
    for (std::size_t word = 0; word < read.size(); ++word) padding += !texel[word] && read[word] != Untouched ? 1u : 0u;
    Require(padding == 0u, "depth write-back: " + std::to_string(padding) + " words outside the surface's texels changed");
    Require(Stale(address) == 0u, "depth write-back: the CPU read left the page stale");
    DrawDepth(device, shaders, address, false);
    Require(Stale(address) == 0u, "depth write-back: a draw that writes no depth marked the plane stale");
    std::printf("drawn depth reaches guest memory\n");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto shaders = Compile(*device);
        DrawnDepthReachesMemory(*device, shaders);
        LibcRunShutdown_nid_postfix();
        std::puts("depth surface write-back tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
