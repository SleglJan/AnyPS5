#include "DeviceTableFixture.hpp"
#include <spirv/unified1/spirv.hpp>
#include <map>
#include <span>

namespace {

using namespace DeviceTable;

constexpr std::uint32_t Samples = 64;
constexpr std::uint32_t Nudge = 0x358637bdu;

TestGuest::Array<float, Width * Height> Texels;

float Texel(std::uint32_t x, std::uint32_t y) {
    return static_cast<float>(x) + 100.0f * static_cast<float>(y) + 0.25f;
}

const Program& ManySamplesProgram() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        LoadSrt(built, {{28, OutputSlot}, {32, SamplerSlot}, {36, ImageSlot}});
        TexelCenter(built, 2);
        built << Vop1(VMovB32, 6, InlineInteger(0));
        for (std::uint32_t sample = 0; sample < Samples; ++sample) {
            built << Vop2(VAddF32, 1, Literal, 1) << Nudge << Mimg(ImageSampleLz, 1u, true, 4u + (sample & 1u), 1, 36, 32)
                  << Vop2(VAddF32, 6, Vgpr(4u + (sample & 1u)), 6);
        }
        built << Vop1(VMovB32, 7, InlineInteger(0));
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

struct MainCalls {
    std::uint32_t lookups = 0;
    std::uint32_t tables = 0;
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
        if (callee.starts_with("table_image_")) ++calls.tables;
    }
    return calls;
}

std::size_t ModeSwitchArms(std::span<const std::uint32_t> words) {
    std::size_t arms = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpImageSampleExplicitLod || op == spv::OpImageSampleImplicitLod) ++arms;
    }
    return arms;
}

void Run(AgcDriver::VulkanDevice& device) {
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) Texels[y * Width + x] = Texel(x, y);
    }
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(SamplerSlot, PointClamp);
    Place(ImageSlot, Image(Texels.data(), Format32Float));
    const Expected expected{Height, 0, 2, [](std::uint32_t row, std::uint32_t x, std::uint32_t, std::uint32_t word) { return SameFloat(word, Texel(x, row)); }};
    const std::string what = std::to_string(Samples) + " samples of one device-table texture";
    const auto compiled = Compile(device, ManySamplesProgram(), userData);
    ExpectRouted(compiled, what);
    const auto words = compiled.result->spirv.Words();
    const auto calls = CallsFromMain(words);
    Expect(calls.lookups == 1u, what + ": main calls device_image_lookup " + std::to_string(calls.lookups) + " times, expected one lookup where the descriptor is loaded");
    Expect(calls.tables == Samples, what + ": main calls table image functions " + std::to_string(calls.tables) + " times, expected every sample to call the shared mode switch once the inline switches exceed the program's budget");
    const auto arms = ModeSwitchArms(words);
    Expect(arms < Samples, what + ": the module holds " + std::to_string(arms) + " image samples, expected the mode switch emitted once and shared, not once per sample");
    const auto missed = Run(device, compiled, Height);
    const auto read = ExpectFirstUse(missed, expected, 1, what + ": first dispatch");
    const auto hit = Run(device, Compile(device, ManySamplesProgram(), userData), Height);
    ExpectHit(hit, expected, what + ": second dispatch");
    ExpectResolved(missed, read, hit, 1, what);
}

}

int main() {
    return Main("device table mode switch budget", Run);
}
