#include "DeviceTableFixture.hpp"
#include <spirv/unified1/spirv.hpp>
#include <map>

namespace {

using namespace DeviceTable;

TestGuest::Array<float, Width * Height> Texels;

float Texel(std::uint32_t x, std::uint32_t y) {
    return static_cast<float>(x) + 100.0f * static_cast<float>(y) + 0.25f;
}

const Program& SampleTwiceProgram() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        LoadSrt(built, {{28, OutputSlot}, {32, SamplerSlot}, {36, ImageSlot}});
        TexelCenter(built, 2);
        built << Mimg(ImageSampleLz, 1u, true, 4, 1, 36, 32)
              << Mimg(ImageSampleLz, 1u, true, 5, 1, 36, 32)
              << Vop1(VMovB32, 6, InlineInteger(0))
              << Vop1(VMovB32, 7, InlineInteger(0));
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

struct MainCalls {
    std::uint32_t lookups = 0;
    std::uint32_t misses = 0;
};

MainCalls CallsFromMain(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        if ((words[cursor] & 0xffffu) == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
    }
    MainCalls calls;
    std::string function;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) function = names[words[cursor + 2]];
        if (op != spv::OpFunctionCall || function != "main") continue;
        const auto& callee = names[words[cursor + 3]];
        if (callee == "device_image_lookup") ++calls.lookups;
        if (callee == "device_image_record_miss") ++calls.misses;
    }
    return calls;
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

    const auto compiled = Compile(device, SampleTwiceProgram(), userData);
    ExpectRouted(compiled, "same texture");
    const auto calls = CallsFromMain(compiled.result->spirv.Words());
    Expect(calls.lookups == 1u && calls.misses == 1u, "same texture: main calls device_image_lookup " + std::to_string(calls.lookups) + " times and device_image_record_miss " + std::to_string(calls.misses) + " times, expected one lookup for both samples of one descriptor and the miss record of the second sample");
    const auto missed = Run(device, compiled, Height);
    const auto read = ExpectFirstUse(missed, twice, 2, "same texture: first dispatch, both samples recording their misses");
    const auto hit = Run(device, Compile(device, SampleTwiceProgram(), userData), Height);
    ExpectHit(hit, twice, "same texture: second dispatch");
    ExpectResolved(missed, read, hit, 1, "same texture");
}

}

int main() {
    return Main("device table same texture", Run);
}
