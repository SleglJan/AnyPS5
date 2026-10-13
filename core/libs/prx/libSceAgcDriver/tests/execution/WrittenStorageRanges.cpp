#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdio>
#include <exception>
#include <span>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint64_t Base = 0x4000000000ull;
constexpr std::uint32_t Width = 256;
constexpr std::uint32_t Height = 128;
constexpr std::uint32_t FormatR32Float = 22;
constexpr std::uint32_t TileRenderTarget64KB = 0x1b;
constexpr std::uint32_t Type2D = 9;

std::array<std::uint32_t, 8> tsharp(std::uint64_t address) {
    std::array<std::uint32_t, 8> words{};
    const auto base = address >> 8u;
    const auto widthMinus1 = Width - 1u;
    const auto heightMinus1 = Height - 1u;
    words[0] = static_cast<std::uint32_t>(base);
    words[1] = static_cast<std::uint32_t>((base >> 32u) & 0xffu) | (FormatR32Float << 20u) | ((widthMinus1 & 0x3u) << 30u);
    words[2] = ((widthMinus1 >> 2u) & 0xfffu) | ((heightMinus1 & 0x3fffu) << 14u);
    words[3] = 4u | (5u << 3u) | (6u << 6u) | (7u << 9u) | (TileRenderTarget64KB << 20u) | (Type2D << 28u);
    return words;
}

void run() {
    const auto words = tsharp(Base);
    ShaderRecompiler::RecompileResult program{};
    ShaderRecompiler::DescriptorBinding binding{};
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageImage;
    binding.count = 2;
    binding.guestDescriptor.assign(8, 0u);
    binding.guestDescriptor.insert(binding.guestDescriptor.end(), words.begin(), words.end());
    binding.imageWritten = {true, true};
    program.bindings.push_back(binding);
    const AgcDriver::Graphics::CompiledShader stage{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    const AgcDriver::Graphics::State graphics{};
    const auto ranges = AgcDriver::DriverDetail::DrawWriteRanges(graphics, std::span<const AgcDriver::Graphics::CompiledShader>(&stage, 1), false);
    const auto bytes = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(words)).guestBytes;
    Require(bytes != 0, "the bound slot describes a surface");
    Require(ranges.size() == 1, "an unbound storage slot adds no written range and does not end the walk");
    Require(ranges[0].first == Base && ranges[0].second == Base + bytes, "the bound storage slot's range covers its surface");
    std::puts("written storage ranges skip an unbound slot: ok");
}

}

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "written storage ranges: %s\n", error.what());
        return 1;
    }
}
