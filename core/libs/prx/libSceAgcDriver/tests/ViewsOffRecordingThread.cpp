#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Cover_vert_spv.h"
#include "Cover_frag_spv.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

constexpr int VulkanTestSkipped = 77;
constexpr std::uint32_t Width = 61;
constexpr std::uint32_t Height = 13;
constexpr std::uint32_t Stale = 0x5a5a5a5au;
constexpr std::uint32_t Clear = 0x11223344u;
constexpr std::uint32_t Drawn = 0xffffffffu;

alignas(4096) std::array<std::byte, 2 * 65536> SurfaceStorage{};
const std::span<std::byte, 65536> Surface{reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(SurfaceStorage.data()) + 65535u) & ~std::uintptr_t{65535u}), 65536};
alignas(4096) std::array<std::uint8_t, 4096> Cmask{};

ColorTarget target(ColorTileMode mode) {
    const ColorTargetLayout layout(Width, Height, mode, 4);
    ColorTarget color{};
    color.address = reinterpret_cast<std::uintptr_t>(Surface.data());
    color.extent = {Width, Height};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.bytes = layout.Bytes();
    color.componentMapping = 0xe4u;
    color.tileMode = mode;
    color.elementBytes = 4;
    color.clearWords = {Clear, 0};
    color.surfaceAddress = color.address;
    color.surfaceExtent = color.extent;
    color.cmaskAddress = reinterpret_cast<std::uintptr_t>(Cmask.data());
    color.cmaskBytes = CmaskLayout(Width, Height).Bytes();
    return color;
}

void prepare(const ColorTarget& color, const std::function<bool(std::uint32_t, std::uint32_t)>& cleared, std::uint8_t code = 0) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    StorageTexture::FlushPending(color.address, Surface.size(), nullptr, "test");
    std::vector<std::byte> stale(Surface.size());
    for (std::size_t offset = 0; offset < stale.size(); offset += 4) std::memcpy(stale.data() + offset, &Stale, 4);
    AgcDriver::GuestMemory::Write(color.address, stale);
    const CmaskLayout layout(Width, Height);
    std::vector<std::byte> mask(Cmask.size(), std::byte{0xff});
    for (std::uint32_t tileY = 0; tileY < layout.TilesY(); ++tileY) {
        for (std::uint32_t tileX = 0; tileX < layout.TilesX(); ++tileX) {
            if (!cleared(tileX, tileY)) continue;
            const auto nibble = layout.Nibble(tileX, tileY);
            mask[nibble / 2u] &= std::byte{static_cast<std::uint8_t>((nibble % 2u) == 0 ? 0xf0u : 0x0fu)};
            mask[nibble / 2u] |= std::byte{static_cast<std::uint8_t>((nibble % 2u) == 0 ? code : code << 4u)};
        }
    }
    AgcDriver::GuestMemory::Write(color.cmaskAddress, mask);
}

void draw(AgcDriver::VulkanDevice& device, const ColorTarget& color, VkRect2D scissor) {
    ShaderRecompiler::RecompileResult vertex;
    vertex.spirv = std::vector<std::uint32_t>(std::begin(COVER_VERT_SPV), std::end(COVER_VERT_SPV));
    ShaderRecompiler::RecompileResult fragment;
    fragment.spirv = std::vector<std::uint32_t>(std::begin(COVER_FRAG_SPV), std::end(COVER_FRAG_SPV));
    const std::array<CompiledShader, 2> shaders{{
        {ShaderRecompiler::ShaderStage::Vertex, &vertex, 0},
        {ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}
    }};
    State state{};
    state.stages = {ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.color = color;
    state.colors = {color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, 0, static_cast<float>(Width), static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = scissor;
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    device.Draw(state, {0, 3, 0, 1, 0, false}, shaders);
    device.WaitIdle();
}

void expect(AgcDriver::VulkanDevice& device, const ColorTarget& color, const std::function<std::uint32_t(std::uint32_t, std::uint32_t)>& texel, const std::string& what) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    StorageTexture::FlushPending(color.address, color.bytes, nullptr, "test");
    device.WaitIdle();
    const ColorTargetLayout layout(Width, Height, color.tileMode, 4);
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            std::uint32_t stored = 0;
            std::memcpy(&stored, Surface.data() + layout.Offset(x, y), 4);
            Require(stored == texel(x, y), what + ": texel (" + std::to_string(x) + ", " + std::to_string(y) + ") holds " + std::to_string(stored));
        }
    }
    Require(std::all_of(Cmask.begin(), Cmask.end(), [](std::uint8_t value) { return value == 0xffu; }), what + ": the CMASK was not left expanded");
}


std::string Counted(const ViewCreationTotals& before, const ViewCreationTotals& after) {
    static constexpr std::array<const char*, ViewCreationTotals::Kinds> names{"sampled", "storage", "default storage", "element", "element layer", "decode source", "decoded copy", "attachment"};
    std::string text;
    for (std::size_t kind = 0; kind < ViewCreationTotals::Kinds; ++kind) {
        const auto pool = after.onPool[kind] - before.onPool[kind];
        const auto caller = after.onCaller[kind] - before.onCaller[kind];
        const auto waits = after.waits[kind] - before.waits[kind];
        if (pool == 0 && caller == 0 && waits == 0) continue;
        text += std::string(text.empty() ? "" : ", ") + names[kind] + " " + std::to_string(pool) + " on the pool, " + std::to_string(caller) + " on the recording thread, " + std::to_string(waits) + " waits";
    }
    return text.empty() ? "no view creations" : text;
}

std::uint64_t Of(const std::array<std::uint64_t, ViewCreationTotals::Kinds>& counts, ViewKind kind) {
    return counts[static_cast<std::size_t>(kind)];
}

void requireOffRecordingThread(const ViewCreationTotals& before, const ViewCreationTotals& after, const std::string& what) {
    for (std::size_t kind = 0; kind < ViewCreationTotals::Kinds; ++kind) {
        Require(after.onCaller[kind] == before.onCaller[kind], what + ": a view was created on the recording thread (" + Counted(before, after) + ")");
        if (kind == static_cast<std::size_t>(ViewKind::Attachment)) continue;
        Require(after.waits[kind] == before.waits[kind], what + ": the recording thread waited for a view (" + Counted(before, after) + ")");
    }
    Require(Of(after.waits, ViewKind::Attachment) - Of(before.waits, ViewKind::Attachment) <= Of(after.onPool, ViewKind::Attachment) - Of(before.onPool, ViewKind::Attachment), what + ": more attachment waits than attachment views made (" + Counted(before, after) + ")");
}

void requireMade(const ViewCreationTotals& before, const ViewCreationTotals& after, ViewKind kind, const char* name, const std::string& what) {
    Require(Of(after.onPool, kind) > Of(before.onPool, kind), what + ": no " + name + " view was made on the view pool (" + Counted(before, after) + ")");
}

void newTarget(AgcDriver::VulkanDevice& device) {
    const auto color = target(ColorTileMode::RenderTarget);
    prepare(color, [](std::uint32_t, std::uint32_t) { return true; });
    const std::string what = "a draw into a new fast-cleared SW_64KB_R_X target";
    const auto before = ViewCreationCounts();
    draw(device, color, {{0, 0}, {20, Height}});
    const auto after = ViewCreationCounts();
    std::printf("%s: %s\n", what.c_str(), Counted(before, after).c_str());
    requireOffRecordingThread(before, after, what);
    requireMade(before, after, ViewKind::DefaultStorage, "default storage", what);
    requireMade(before, after, ViewKind::Element, "element", what);
    requireMade(before, after, ViewKind::Attachment, "attachment", what);
    expect(device, color, [](std::uint32_t x, std::uint32_t) { return x < 20 ? Drawn : Clear; }, what);
    const std::string again = "a second draw into the same target";
    const auto reused = ViewCreationCounts();
    draw(device, color, {{40, 0}, {Width - 40, Height}});
    const auto later = ViewCreationCounts();
    std::printf("%s: %s\n", again.c_str(), Counted(reused, later).c_str());
    requireOffRecordingThread(reused, later, again);
    Require(Of(later.waits, ViewKind::Attachment) == Of(reused.waits, ViewKind::Attachment), again + ": the recording thread waited for an attachment view made before (" + Counted(reused, later) + ")");
    expect(device, color, [](std::uint32_t x, std::uint32_t) { return x < 20 || x >= 40 ? Drawn : Clear; }, again);
}

}

int main() {
    try {
        std::unique_ptr<AgcDriver::VulkanDevice> device;
        try {
            device = std::make_unique<AgcDriver::VulkanDevice>();
        } catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::printf("skipped, no usable Vulkan device: %s\n", error.what());
            return VulkanTestSkipped;
        }
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(SurfaceStorage.data(), SurfaceStorage.size(), true, true, true);
            mutation.Add(Cmask.data(), Cmask.size(), true, true, true);
        }
        newTarget(*device);
        device->WaitIdle();
        device.reset();
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(SurfaceStorage.data());
            mutation.Remove(Cmask.data());
        }
        std::cout << "views off the recording thread tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
