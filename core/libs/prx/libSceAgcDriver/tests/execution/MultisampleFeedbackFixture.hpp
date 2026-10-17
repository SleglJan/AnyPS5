#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_MULTISAMPLEFEEDBACKFIXTURE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_MULTISAMPLEFEEDBACKFIXTURE_HPP

#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "Recompiler.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#ifdef _WIN32

namespace MultisampleFeedback {

using NewPath::Program;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Float32Export = 9;
constexpr std::uint32_t Attempts = 3;

inline int failures = 0;

inline void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

namespace Code {

constexpr std::array<std::uint32_t, 2> LoadDescriptorFromMemory{0xf40c0200u, 0xfa000000u};
constexpr std::array<std::uint32_t, 3> ZeroCoordinates{0x7e080280u, 0x7e0a0280u, 0x7e0c0280u};
constexpr std::uint32_t WaitScalar = 0xbf8cc07fu;
constexpr std::uint32_t WaitVector = 0xbf8c3f70u;
constexpr std::uint32_t LoadMultisampleWord0 = 0xf0001f30u;
constexpr std::uint32_t LoadFromMemoryDescriptor = 0x00020804u;
constexpr std::uint32_t LoadFromUserDescriptor = 0x00000804u;
constexpr std::array<std::uint32_t, 2> ExportColor{0xf800180fu, 0x0b0a0908u};
constexpr std::array<std::uint32_t, 2> ExportDepth{0xf8001881u, 0x00000008u};
constexpr std::uint32_t EndProgram = 0xbf810000u;

}

inline const Program& VertexProgram() {
    static const Program program = [] {
        Program built;
        for (const auto word : {0xe0382000u, 0x80000005u, 0xbf8c3f70u, 0xf80008cfu, 0x03020100u, 0xbf810000u}) built << word;
        return built;
    }();
    return program;
}

inline const Program& SamplingPixel(bool fromMemory, bool color) {
    using namespace Code;
    static std::array<Program, 4> programs{};
    auto& built = programs[(fromMemory ? 2u : 0u) + (color ? 1u : 0u)];
    if (built.size != 0u) return built;
    if (fromMemory) built << LoadDescriptorFromMemory;
    for (const auto word : ZeroCoordinates) built << word;
    if (fromMemory) built << WaitScalar;
    built << LoadMultisampleWord0 << (fromMemory ? LoadFromMemoryDescriptor : LoadFromUserDescriptor) << WaitVector;
    built << (color ? ExportColor : ExportDepth) << EndProgram;
    return built;
}

inline bool ReadGuestWord(void*, std::uint64_t address, std::uint32_t* value) {
    if (!AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(address), sizeof(*value))) return false;
    std::memcpy(value, reinterpret_cast<const void*>(address), sizeof(*value));
    return true;
}

class Triangle {
public:
    Triangle() : memory(1) {
        const std::array<std::array<float, 4>, 3> corners{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};
        std::memcpy(memory.Data(), corners.data(), sizeof(corners));
    }

    std::array<std::uint32_t, 4> Descriptor() const {
        const auto address = memory.Address();
        return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (16u << 16u), 3u, 0x01016facu};
    }

private:
    NewPath::PrivatePages memory;
};

inline void Draw(AgcDriver::VulkanDevice& device, const AgcDriver::Graphics::State& state, const Triangle& triangle, const Program& pixel, std::span<const std::uint32_t> pixelUserData, bool depthExport = false) {
    const auto target = device.Target();
    constexpr std::uint32_t waveSize = 64;
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = triangle.Descriptor();
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const auto vertexCode = VertexProgram().Code();
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{VertexProgram().Address(), std::as_bytes(vertexCode)}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, VertexProgram().Address(), vertexCode, 0, {}},
        {waveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());
    const auto codeAddress = pixel.Address();
    const auto pixelCode = pixel.Code();
    const AgcDriver::Registers context{{0x1b6, 0x8000}, {0x1b3, 0}, {0x1b4, 0}, {0x203, depthExport ? 1u : 0u}, {0x1c5, Float32Export}};
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{codeAddress, std::as_bytes(pixelCode)}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, codeAddress, pixelCode, 0, {}},
        {waveSize, 0, pixelUserData, std::nullopt, AgcDriver::Graphics::DecodePixelStageInfo(context, {}), std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    const auto handle = ShaderRecompiler::PrepareShader(fragment);
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{codeAddress, 0, 1, {pixelCode.begin(), pixelCode.end()}, {}};
    snapshot.prepared->entries.push_back({0, handle});
    fragment.shader.code = snapshot.code;
    fragment.context.pixel = AgcDriver::Graphics::DecodePixelStageInfo(context, AgcDriver::Graphics::ExportMappings(state));
    ShaderRecompiler::SrtRuntime runtime{};
    runtime.userData = pixelUserData;
    runtime.shaderBase = codeAddress;
    runtime.readMemory = &ReadGuestWord;
    const auto invocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, fragment);
    const auto pixelResult = invocation.Materialize(*invocation.Capture(runtime));
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, pixelResult.get(), PixelPushOffset(vertexPush, target)}
    }};
    const AgcDriver::Pm4::DrawParameters draw{0, 3u, 0, 1, 0, false};
    device.Draw(state, draw, shaders);
}

inline void ExpectRefusedWithin(AgcDriver::VulkanDevice& device, const std::function<void()>& draw, const std::string& refusal, const std::string& what) {
    for (std::uint32_t attempt = 0; attempt < Attempts; ++attempt) {
        try {
            draw();
            device.WaitIdle();
        } catch (const std::exception& error) {
            const std::string message = error.what();
            Expect(message.find(refusal) != std::string::npos, what + ": the draw failed for another reason: " + message);
            device.WaitIdle();
            return;
        }
    }
    Expect(false, what + ": " + std::to_string(Attempts) + " draws were recorded; the draw must be refused once the sampled texture is known (\"" + refusal + "\")");
}

inline int Main(const char* name, void (*run)(AgcDriver::VulkanDevice&)) {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        run(*device);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d %s checks failed\n", failures, name);
        return 1;
    }
    std::printf("%s tests passed\n", name);
    return 0;
}

}

#endif

#endif
