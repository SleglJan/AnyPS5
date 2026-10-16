#include "DeviceTableFixture.hpp"
#include <spirv/unified1/spirv.hpp>
#include <map>
#include <set>
#include <span>

namespace {

using namespace DeviceTable;

TestGuest::Array<float, Width * Height> Texels;

float Texel(std::uint32_t x, std::uint32_t y) {
    return static_cast<float>(x) + 100.0f * static_cast<float>(y) + 0.5f;
}

const Program& SampleAcrossBranchProgram() {
    static const Program program = [] {
        using namespace Asm;
        constexpr std::uint32_t VCmpGtU32LaneBound = 0x7d8800c0u;
        constexpr std::uint32_t SCbranchVcczOver1 = 0xbf860001u;
        Program built;
        LoadSrt(built, {{28, OutputSlot}, {32, SamplerSlot}, {36, ImageSlot}});
        TexelCenter(built, 2);
        built << Mimg(ImageSampleLz, 1u, true, 4, 1, 36, 32)
              << VCmpGtU32LaneBound
              << SCbranchVcczOver1
              << Vop1(VMovB32, 6, InlineInteger(1))
              << Mimg(ImageSampleLz, 1u, true, 5, 1, 36, 32)
              << Vop1(VMovB32, 6, InlineInteger(0))
              << Vop1(VMovB32, 7, InlineInteger(0));
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

struct Lookups {
    std::uint32_t calls = 0;
    std::uint32_t shared = 0;
};

Lookups MainLookups(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    std::set<std::uint32_t> vectors;
    std::uint32_t u32 = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
        if (op == spv::OpTypeInt && words[cursor + 2] == 32u && words[cursor + 3] == 0u) u32 = words[cursor + 1];
        if (op == spv::OpTypeVector && words[cursor + 2] == u32 && words[cursor + 3] == 4u) vectors.insert(words[cursor + 1]);
    }
    Lookups lookups;
    std::string function;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) function = names[words[cursor + 2]];
        if (function != "main") continue;
        if (op == spv::OpFunctionCall && names[words[cursor + 3]] == "device_image_lookup") ++lookups.calls;
        if (op == spv::OpGroupNonUniformBroadcastFirst && vectors.contains(words[cursor + 1])) ++lookups.shared;
    }
    return lookups;
}

void Run(AgcDriver::VulkanDevice& device) {
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) Texels[y * Width + x] = Texel(x, y);
    }
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(SamplerSlot, PointClamp);
    Place(ImageSlot, Image(Texels.data(), Format32Float));
    const Expected twice{Height, 0, 2, [](std::uint32_t row, std::uint32_t x, std::uint32_t, std::uint32_t word) { return SameFloat(word, Texel(x, row)); }};

    const auto compiled = Compile(device, SampleAcrossBranchProgram(), userData);
    ExpectRouted(compiled, "uniform lookup");
    const auto lookups = MainLookups(compiled.result->spirv.Words());
    Expect(lookups.calls == 1u && lookups.shared == 1u, "uniform lookup: main calls device_image_lookup " + std::to_string(lookups.calls) + " times and broadcasts " + std::to_string(lookups.shared) + " lookup results from the first lane, expected one lookup where the wave-uniform descriptor is loaded and its result broadcast once");
    const auto missed = Run(device, compiled, Height);
    const auto read = ExpectFirstUse(missed, twice, 2, "uniform lookup: first dispatch, both samples recording their misses");
    const auto hit = Run(device, Compile(device, SampleAcrossBranchProgram(), userData), Height);
    ExpectHit(hit, twice, "uniform lookup: second dispatch, both samples through the broadcast slot");
    ExpectResolved(missed, read, hit, 1, "uniform lookup");
}

}

int main() {
    return Main("device table uniform lookup", Run);
}
