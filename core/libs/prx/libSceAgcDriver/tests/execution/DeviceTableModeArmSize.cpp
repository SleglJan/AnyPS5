#include "DeviceTableFixture.hpp"
#include <cstdio>

namespace {

using namespace DeviceTable;

constexpr std::uint32_t Nudge = 0x358637bdu;
constexpr std::size_t SampleArmWords = 2450;
constexpr std::size_t StoreArmWords = 11450;

TestGuest::Array<float, Width * Height> Texels;

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

Program StoresProgram(std::uint32_t stores) {
    using namespace Asm;
    Program built;
    LoadSrt(built, {{28, OutputSlot}, {36, ImageSlot}});
    TexelIndex(built, 2);
    for (std::uint32_t component = 0; component < 4u; ++component) built << Vop1(VMovB32, 4 + component, InlineInteger(component + 1u));
    for (std::uint32_t store = 0; store < stores; ++store) {
        built << Mimg(ImageStore, 0xfu, true, 4, 1, 36) << Vop2(VAddNcU32, 1, InlineInteger(1), 1);
    }
    StoreLane(built, 2, 28);
    return built;
}

std::size_t Words(AgcDriver::VulkanDevice& device, const Program& program, std::span<const std::uint32_t> userData, const std::string& what) {
    const auto compiled = Compile(device, program, userData);
    ExpectRouted(compiled, what);
    return compiled.result->spirv.Words().size();
}

void Run(AgcDriver::VulkanDevice& device) {
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(SamplerSlot, PointClamp);
    Place(ImageSlot, Image(Texels.data(), Format32Float));
    const auto sampleArm = Words(device, SamplesProgram(3), userData, "three samples") - Words(device, SamplesProgram(2), userData, "two samples");
    const auto storeArm = Words(device, StoresProgram(3), userData, "three stores") - Words(device, StoresProgram(2), userData, "two stores");
    std::printf("mode arm size: a device-table sample adds %zu SPIR-V words, a device-table store %zu\n", sampleArm, storeArm);
    Expect(sampleArm <= SampleArmWords, "mode arm size: one more sample of a device-table texture adds " + std::to_string(sampleArm) + " SPIR-V words, expected at most " + std::to_string(SampleArmWords));
    Expect(storeArm <= StoreArmWords, "mode arm size: one more store to a device-table storage image adds " + std::to_string(storeArm) + " SPIR-V words, expected at most " + std::to_string(StoreArmWords));
}

}

int main() {
    return Main("device table mode arm size", Run);
}
