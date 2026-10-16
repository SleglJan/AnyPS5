#include "DeviceTableFixture.hpp"

namespace {

using namespace DeviceTable;

TestGuest::Array<float, Width * Height> Texels;

const Program& SampleProgram() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        LoadSrt(built, {{28, OutputSlot}, {32, SamplerSlot}, {36, ImageSlot}});
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
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(SamplerSlot, PointClamp);
    for (const std::uint32_t type : {0u, 1u}) {
        auto words = Image(Texels.data(), Format32Float);
        words[3] = (words[3] & 0x0fffffffu) | (type << 28u);
        Place(ImageSlot, words);
        const std::string what = "TYPE " + std::to_string(type) + " T# in a read-only slot";
        try {
            const auto compiled = Compile(device, SampleProgram(), userData);
            ExpectRouted(compiled, what);
            const auto& image = compiled.plan->info.images.at(0);
            Expect(image.directRuntime, what + ": the image is not marked as a direct run-time image");
            Expect(compiled.result->deviceImages.empty(), what + ": the pre-pass names " + std::to_string(compiled.result->deviceImages.size()) + " device image keys, expected none for a null image");
        } catch (const std::exception& error) {
            Expect(false, what + ": compiling failed: " + error.what());
        }
    }
}

}

int main() {
    return Main("device table null type", Run);
}
