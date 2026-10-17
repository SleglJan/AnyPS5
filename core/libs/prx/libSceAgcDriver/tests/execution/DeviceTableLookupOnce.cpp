#include "DeviceTableFixture.hpp"
#include <spirv/unified1/spirv.hpp>
#include <cstdio>
#include <map>
#include <span>

namespace {

using namespace DeviceTable;

constexpr std::uint32_t Samples = 4;
constexpr std::uint32_t Nudge = 0x358637bdu;
constexpr std::size_t SampleLookupWords = 2200;

TestGuest::Array<float, Width * Height> Texels;

float Texel(std::uint32_t x, std::uint32_t y) {
    return static_cast<float>(x) + 100.0f * static_cast<float>(y) + 0.5f;
}

Program SamplesProgram(std::uint32_t samples) {
    using namespace Asm;
    Program built;
    LoadSrt(built, {{28, OutputSlot}, {32, SamplerSlot}, {36, ImageSlot}});
    TexelCenter(built, 2);
    built << Vop1(VMovB32, 6, InlineInteger(0));
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
        built << Vop2(VAddF32, 1, Literal, 1) << Nudge << Mimg(ImageSampleLz, 1u, true, 4u + (sample & 1u), 1, 36, 32)
              << Vop2(VAddF32, 6, Vgpr(4u + (sample & 1u)), 6);
    }
    built << Vop1(VMovB32, 7, InlineInteger(0));
    StoreLane(built, 2, 28);
    return built;
}

struct MainShape {
    std::uint32_t lookups = 0;
    std::uint32_t misses = 0;
    std::uint32_t mapReads = 0;
};

MainShape Shape(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        if ((words[cursor] & 0xffffu) == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
    }
    MainShape shape;
    std::string function;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) function = names[words[cursor + 2]];
        if (function != "main") continue;
        if (op == spv::OpFunctionCall && names[words[cursor + 3]] == "device_image_lookup") ++shape.lookups;
        if (op == spv::OpFunctionCall && names[words[cursor + 3]] == "device_image_record_miss") ++shape.misses;
        if ((op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain) && names[words[cursor + 3]] == "table_map") ++shape.mapReads;
    }
    return shape;
}

void Run(AgcDriver::VulkanDevice& device) {
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) Texels[y * Width + x] = Texel(x, y);
    }
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(SamplerSlot, PointClamp);
    Place(ImageSlot, Image(Texels.data(), Format32Float));
    const std::string what = std::to_string(Samples) + " samples of one device-table texture";
    const auto fewer = Compile(device, SamplesProgram(Samples - 1u), userData);
    const auto compiled = Compile(device, SamplesProgram(Samples), userData);
    ExpectRouted(compiled, what);
    const auto shape = Shape(compiled.result->spirv.Words());
    const auto fewerShape = Shape(fewer.result->spirv.Words());
    const auto added = compiled.result->spirv.Words().size() - fewer.result->spirv.Words().size();
    std::printf("lookup once: one more sample adds %zu SPIR-V words; main reads the table map %u times with %u samples, %u with %u\n", added, shape.mapReads, Samples, fewerShape.mapReads, Samples - 1u);
    Expect(shape.lookups == 1u, what + ": main calls device_image_lookup " + std::to_string(shape.lookups) + " times, expected one lookup where the descriptor is loaded");
    Expect(shape.misses == Samples, what + ": main calls device_image_record_miss " + std::to_string(shape.misses) + " times, expected one miss record per sample");
    Expect(shape.mapReads == fewerShape.mapReads, what + ": main reads the image table map " + std::to_string(shape.mapReads) + " times, " + std::to_string(fewerShape.mapReads) + " with one sample fewer, expected the lookup's operands built once at the lookup");
    Expect(added <= SampleLookupWords, what + ": one more sample adds " + std::to_string(added) + " SPIR-V words, expected at most " + std::to_string(SampleLookupWords));
    const Expected expected{Height, 0, 2, [](std::uint32_t row, std::uint32_t x, std::uint32_t, std::uint32_t word) { return SameFloat(word, Texel(x, row)); }};
    const auto missed = Run(device, compiled, Height);
    const auto read = ExpectFirstUse(missed, expected, 1, what + ": first dispatch");
    const auto hit = Run(device, Compile(device, SamplesProgram(Samples), userData), Height);
    ExpectHit(hit, expected, what + ": second dispatch");
    ExpectResolved(missed, read, hit, 1, what);
}

}

int main() {
    return Main("device table lookup once", Run);
}
