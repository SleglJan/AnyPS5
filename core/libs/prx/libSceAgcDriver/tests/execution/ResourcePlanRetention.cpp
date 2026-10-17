#include "DeviceTableFixture.hpp"

namespace {

using namespace DeviceTable;

constexpr std::uint32_t DataSlot = 24;

TestGuest::Array<float, Width * Height> Texels;
TestGuest::Array<std::uint32_t, 256> Data;

Program ConditionedSample(std::uint32_t chain) {
    using namespace Asm;
    constexpr std::array<std::uint32_t, 2> LoadIndexed{0xe0301000u, 0x80060b0au};
    constexpr std::uint32_t ReadFirstLaneS20 = 0x7e28050bu;
    constexpr std::array<std::uint32_t, 2> LoadSrtAtS20{0xf4000540u, 0x28000000u};
    constexpr std::uint32_t CompareS21 = 0xbf078015u;
    constexpr std::uint32_t SkipSample = 0xbf840002u;
    Program built;
    LoadSrt(built, {{28, OutputSlot}, {32, SamplerSlot}, {36, ImageSlot}, {24, DataSlot}});
    built << Vop1(VMovB32, 10, Vgpr(0));
    for (std::uint32_t step = 0; step < chain; ++step) built << Vop2(VAddNcU32, 10, InlineInteger(1), 10);
    built << Vop2(VLshlRevB32, 10, InlineInteger(2), 10)
          << LoadIndexed << WaitAll << ReadFirstLaneS20 << LoadSrtAtS20 << WaitAll;
    TexelCenter(built, 2);
    built << Vop1(VMovB32, 4, InlineInteger(0))
          << CompareS21 << SkipSample
          << Mimg(ImageSampleLz, 1u, true, 4, 1, 36, 32)
          << Vop1(VMovB32, 5, InlineInteger(0))
          << Vop1(VMovB32, 6, InlineInteger(0))
          << Vop1(VMovB32, 7, InlineInteger(0));
    StoreLane(built, 2, 28);
    return built;
}

void Run(AgcDriver::VulkanDevice& device) {
    Srt.fill(0u);
    Data.fill(0u);
    const auto userData = SrtUserData();
    Place(SamplerSlot, PointClamp);
    Place(ImageSlot, Image(Texels.data(), Format32Float));
    Place(DataSlot, Buffer(Data.data(), static_cast<std::uint32_t>(decltype(Data)::ByteSize)));
    std::vector<std::size_t> values;
    std::vector<std::size_t> conditions;
    for (const std::uint32_t chain : {1u, 16u, 64u}) {
        const auto program = ConditionedSample(chain);
        const auto compiled = Compile(device, program, userData);
        ExpectRouted(compiled, "plan retention, chain " + std::to_string(chain));
        const auto& plan = *compiled.plan;
        std::size_t kept = 0;
        for (const auto& block : plan.controlFlow) kept += block.condition != nullptr ? 1u : 0u;
        values.push_back(plan.valueStorage.size());
        conditions.push_back(kept);
    }
    Expect(conditions.front() != 0u, "plan retention: the plan keeps no branch condition, so the test does not reach the condition's values");
    Expect(values[0] == values[1] && values[1] == values[2], "plan retention: the plan keeps " + std::to_string(values[0]) + ", " + std::to_string(values[1]) + " and " + std::to_string(values[2]) + " values for vector chains of 1, 16 and 64 steps before the scalar load its branch condition reads; it must keep only what the condition's evaluation can reach");
}

}

int main() {
    return Main("resource plan retention", Run);
}
