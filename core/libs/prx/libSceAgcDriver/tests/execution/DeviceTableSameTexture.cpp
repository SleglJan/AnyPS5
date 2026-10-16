#include "DeviceTableFixture.hpp"
#include "DeviceImageTableAbi.hpp"
#include <spirv/unified1/spirv.hpp>
#include <span>
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

struct MainCalls {
    std::uint32_t lookups = 0;
    std::uint32_t misses = 0;
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
        if (callee == "device_image_record_miss") ++calls.misses;
        if (callee.starts_with("table_image_")) ++calls.tables;
    }
    return calls;
}

std::size_t LookupVectorLoads(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    std::set<std::uint32_t> vectors;
    std::uint32_t u32 = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
        if (op == spv::OpTypeInt && words[cursor + 2] == 32u && words[cursor + 3] == 0u) u32 = words[cursor + 1];
        if (op == spv::OpTypeVector && words[cursor + 2] == u32 && words[cursor + 3] == 4u) vectors.insert(words[cursor + 1]);
    }
    std::size_t loads = 0;
    std::string function;
    std::set<std::uint32_t> addresses;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) function = names[words[cursor + 2]];
        if (op == spv::OpConvertUToPtr) addresses.insert(words[cursor + 2]);
        if (op == spv::OpLoad && function == "device_image_lookup" && vectors.contains(words[cursor + 1]) && addresses.contains(words[cursor + 3])) ++loads;
    }
    return loads;
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
    Expect(calls.lookups == 1u && calls.misses == 2u, "same texture: main calls device_image_lookup " + std::to_string(calls.lookups) + " times and device_image_record_miss " + std::to_string(calls.misses) + " times, expected one lookup where the descriptor is loaded and each sample recording its own miss");
    const auto bucketLoads = LookupVectorLoads(compiled.result->spirv.Words());
    Expect(bucketLoads == ShaderRecompiler::DeviceImageTableAbi::BucketWords / 4u, "same texture: the device image lookup loads " + std::to_string(bucketLoads) + " 16-byte vectors per probed bucket, expected its " + std::to_string(ShaderRecompiler::DeviceImageTableAbi::BucketWords) + " words in " + std::to_string(ShaderRecompiler::DeviceImageTableAbi::BucketWords / 4u) + " vector loads");
    Expect(calls.tables == 0u, "same texture: main calls table image functions " + std::to_string(calls.tables) + " times, expected each sample to switch over its image modes where it reads");
    const auto missed = Run(device, compiled, Height);
    const auto read = ExpectFirstUse(missed, twice, 2, "same texture: first dispatch, both samples recording their misses");
    const auto hit = Run(device, Compile(device, SampleTwiceProgram(), userData), Height);
    ExpectHit(hit, twice, "same texture: second dispatch");
    ExpectResolved(missed, read, hit, 1, "same texture");

    const auto across = Compile(device, SampleAcrossBranchProgram(), userData);
    ExpectRouted(across, "same texture across a branch");
    const auto acrossCalls = CallsFromMain(across.result->spirv.Words());
    Expect(acrossCalls.lookups == 1u && acrossCalls.misses == 2u, "same texture across a branch: main calls device_image_lookup " + std::to_string(acrossCalls.lookups) + " times and device_image_record_miss " + std::to_string(acrossCalls.misses) + " times, expected one lookup where the descriptor is loaded for the samples before and after the branch and each sample recording its own miss");
    Expect(acrossCalls.tables == 0u, "same texture across a branch: main calls table image functions " + std::to_string(acrossCalls.tables) + " times, expected each sample to switch over its image modes where it reads");
    const auto acrossMissed = Run(device, across, Height);
    const auto acrossRead = ExpectFirstUse(acrossMissed, twice, 2, "same texture across a branch: first dispatch, both samples recording their misses");
    const auto acrossHit = Run(device, Compile(device, SampleAcrossBranchProgram(), userData), Height);
    ExpectHit(acrossHit, twice, "same texture across a branch: second dispatch");
    ExpectResolved(acrossMissed, acrossRead, acrossHit, 1, "same texture across a branch");
}

}

int main() {
    return Main("device table same texture", Run);
}
