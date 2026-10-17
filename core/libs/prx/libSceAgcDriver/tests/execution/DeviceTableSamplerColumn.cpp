#include "DeviceTableFixture.hpp"

namespace {

using namespace DeviceTable;

constexpr std::uint32_t TableSlot = 24;
constexpr std::uint32_t RecordWords = 12;
constexpr std::uint32_t SamplerRecordWord = 0;
constexpr std::uint32_t ImageRecordWord = 4;

TestGuest::Array<float, Width * Height> Texels;
TestGuest::Array<std::uint32_t, Height * RecordWords> Records;

float Texel(std::uint32_t x, std::uint32_t y) {
    return static_cast<float>(x) * 2.0f + 1000.0f * static_cast<float>(y) + 0.75f;
}

const Program& SampleColumnProgram() {
    static const Program program = [] {
        using namespace Asm;
        constexpr std::uint32_t SMulRecordOffset = 0x9312b002u;
        constexpr std::array<std::uint32_t, 2> LoadRecordSampler{0xf424080cu, 0x24000000u | (SamplerRecordWord * 4u)};
        constexpr std::array<std::uint32_t, 2> LoadRecordImage{0xf428090cu, 0x24000000u | (ImageRecordWord * 4u)};
        Program built;
        LoadSrt(built, {{28, OutputSlot}, {24, TableSlot}});
        built << SMulRecordOffset << LoadRecordSampler << LoadRecordImage << WaitAll;
        TexelCenter(built, 2);
        built << Mimg(ImageSampleLz, 1u, true, 4, 1, 36, 32)
              << Vop1(VMovB32, 5, InlineInteger(0))
              << Vop1(VMovB32, 6, InlineInteger(0))
              << Vop1(VMovB32, 7, InlineInteger(0));
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

void Run(AgcDriver::VulkanDevice& device) {
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) Texels[y * Width + x] = Texel(x, y);
    }
    Records.fill(0u);
    const auto image = Image(Texels.data(), Format32Float);
    for (std::uint32_t record = 0; record < Height; ++record) {
        std::copy(PointClamp.begin(), PointClamp.end(), Records.begin() + record * RecordWords + SamplerRecordWord);
        std::copy(image.begin(), image.end(), Records.begin() + record * RecordWords + ImageRecordWord);
    }
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(TableSlot, Buffer(Records.data(), static_cast<std::uint32_t>(decltype(Records)::ByteSize)));
    const Expected expected{Height, 0, 1, [](std::uint32_t row, std::uint32_t x, std::uint32_t, std::uint32_t word) { return SameFloat(word, Texel(x, row)); }};

    const auto compiled = Compile(device, SampleColumnProgram(), userData);
    ExpectRouted(compiled, "sampler from a table column");
    const auto& samplers = compiled.plan->info.samplers;
    Expect(samplers.size() == 1u && samplers[0].deviceTable, "sampler from a table column: the S# read from the table is not read through the device image table");
    const auto first = Run(device, compiled, Height);
    const auto read = ExpectFirstUse(first, expected, 1, "sampler from a table column: first dispatch");
    const auto hit = Run(device, Compile(device, SampleColumnProgram(), userData), Height);
    ExpectHit(hit, expected, "sampler from a table column: second dispatch, through the S# the shader read from the table");
    ExpectResolved(first, read, hit, 1, "sampler from a table column");
}

}

int main() {
    return Main("device table sampler column", Run);
}
