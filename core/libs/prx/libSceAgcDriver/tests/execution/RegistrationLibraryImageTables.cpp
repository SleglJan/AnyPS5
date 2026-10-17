#include "DeviceTableFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

namespace {

using namespace DeviceTable;
using AgcDriver::Graphics::PipelineLibraryCounters;

constexpr std::size_t FragmentShaderPart = 2;
constexpr std::uint32_t UserWords = 8;

TestGuest::Array<std::uint32_t, Width * Height> Texels;

const Program& SamplePixel() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        built << Vop1(VMovB32, 0, Half)
              << Vop1(VMovB32, 1, Half)
              << Mimg(ImageSampleLz, 0xfu, true, 2, 0, 0, 4)
              << WaitAll
              << 0xf800180fu << 0x05040302u
              << EndProgram;
        return built;
    }();
    return program;
}

ShaderRecompiler::RecompileRequest Request(const AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> userData) {
    const auto code = SamplePixel().Code();
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.targetOutputMode[0] = 9u;
    pixel.targetExportMapping.fill(0xe4u);
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Fragment, SamplePixel().Address(), code, 0, {}},
        {64, 0, userData, std::nullopt, pixel, std::nullopt, {}},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.layout.pushConstantOffsetBytes = AgcDriver::Graphics::FixedPushOffset(ShaderStage::Fragment);
    return request;
}

std::uint64_t FragmentParts(const AgcDriver::VulkanDevice& device) {
    return AgcDriver::Graphics::PipelineLibraryCountersOf(device.Device()).built[FragmentShaderPart];
}

int Run(AgcDriver::VulkanDevice& device) {
    if (!device.Target().registrationModules || !device.GraphicsPipelineLibraries()) {
        std::puts("skipped, the device prebuilds no registration modules into graphics pipeline libraries");
        return VulkanTestSkipped;
    }
    const std::vector<std::uint32_t> registered(UserWords, 0u);
    const auto registration = Request(device, registered);
    const auto handle = ShaderRecompiler::PrepareShader(registration);
    Expect(handle != nullptr, "the pixel program sampling an image from its user data was not prepared");
    if (handle == nullptr) return Finish("registration library image tables");
    std::shared_ptr<const ShaderRecompiler::RecompileResult> prebuilt;
    try {
        prebuilt = ShaderRecompiler::RegistrationResult(registration, *handle);
    } catch (const std::exception& error) {
        Expect(false, std::string("the registration result of a pixel program whose image comes from its user data was refused: ") + error.what());
        return Finish("registration library image tables");
    }
    Expect(prebuilt != nullptr, "the registration result of a pixel program whose image comes from its user data is missing");
    if (prebuilt == nullptr) return Finish("registration library image tables");
    const AgcDriver::Graphics::ShaderStages stages{AgcDriver::Graphics::ShaderPath::Vertex, 0u, 32u, registration.context.waveSize, {}, {}};
    const auto pushOffset = registration.layout.pushConstantOffsetBytes;
    const auto before = FragmentParts(device);
    const std::array<AgcDriver::Graphics::CompiledShader, 1> prebuild{{{ShaderStage::Fragment, prebuilt.get(), pushOffset}}};
    Expect(device.PrebuildGraphicsLibraries(stages, prebuild), "the registration prebuild reported no library path");
    const auto afterPrebuild = FragmentParts(device);
    Expect(afterPrebuild == before + 1u, "the registration prebuild built " + std::to_string(afterPrebuild - before) + " fragment-shader library parts, expected one");

    std::vector<std::uint32_t> drawn(UserWords, 0u);
    const auto image = Image(Texels.data(), Format8_8_8_8UNorm);
    std::copy(image.begin(), image.end(), drawn.begin());
    std::copy(PointClamp.begin(), PointClamp.end(), drawn.begin() + 4);
    const auto draw = Request(device, drawn);
    std::vector<std::uint64_t> key;
    ShaderRecompiler::BuildPreparedShaderKey(draw, key);
    const auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(draw, handle, key);
    Expect(invocation.has_value(), "the draw's request does not match the program prepared at registration");
    if (!invocation.has_value()) return Finish("registration library image tables");
    const auto result = invocation->MaterializeAtRunTime();
    Expect(result != nullptr, "the draw's materialization of the registered pixel program is missing");
    if (result == nullptr) return Finish("registration library image tables");
    const std::array<AgcDriver::Graphics::CompiledShader, 1> used{{{ShaderStage::Fragment, result.get(), pushOffset}}};
    Expect(device.PrebuildGraphicsLibraries(stages, used), "the draw's stage reported no library path");
    const auto afterDraw = FragmentParts(device);
    Expect(afterDraw == afterPrebuild, "the draw's materialization built " + std::to_string(afterDraw - afterPrebuild) + " fragment-shader library parts, expected the one built at registration");
    return Finish("registration library image tables");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        return Run(*device);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
