#include "DeviceTableFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"

#ifdef _WIN32

namespace {

using namespace DeviceTable;

constexpr std::uint32_t Texels = Width * Height;
constexpr std::uint64_t TextureBytes = std::uint64_t{Texels} * 4u;
constexpr std::uint32_t Corners = 3;

TestGuest::Array<std::uint32_t, Texels> Pixels;
const TestGuest::Array<std::array<float, 4>, 3> Triangle{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}};

std::uint32_t Texel(std::uint32_t seed, std::uint32_t x, std::uint32_t y) {
    return ((seed * 37u + 3u * x) & 0xffu) | (((seed * 11u + 50u * y) & 0xffu) << 8u) | (((255u - x - seed) & 0xffu) << 16u) | (((0x20u + x + 7u * y + seed) & 0xffu) << 24u);
}

const Program& PositionVertex() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        built << 0xe0382000u << 0x80000005u << WaitAll << 0xf80008cfu << 0x03020100u << EndProgram;
        return built;
    }();
    return program;
}

const Program& SamplePixel() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        built << Vop2(VMulF32, 0, Literal, 0) << OneOverWidth
              << Vop2(VMulF32, 1, Literal, 1) << OneOverHeight
              << Mimg(ImageSampleLz, 0xfu, true, 2, 0, 0, 4)
              << WaitAll
              << 0xf800180fu << 0x05040302u
              << EndProgram;
        return built;
    }();
    return program;
}

struct Shaders {
    Compiled vertex;
    Compiled pixel;
    std::uint32_t pixelPush = 0;
};

Shaders Compile(AgcDriver::VulkanDevice& device, const std::uint8_t* texels) {
    const auto target = device.Target();
    std::array<std::uint32_t, 4> vertexUserData{};
    const auto vertexAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Triangle.data()));
    vertexUserData = {static_cast<std::uint32_t>(vertexAddress), static_cast<std::uint32_t>((vertexAddress >> 32u) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Triangle.size()), 0x01016facu};
    std::array<std::uint32_t, 8> pixelUserData{};
    const auto image = Image(texels, Format8_8_8_8UNorm);
    std::copy(image.begin(), image.end(), pixelUserData.begin());
    std::copy(PointClamp.begin(), PointClamp.end(), pixelUserData.begin() + 4);
    Shaders shaders;
    shaders.vertex = CompileRequest({
        {ShaderRecompiler::ShaderStage::Vertex, PositionVertex().Address(), PositionVertex().Code(), 0, {}},
        {64, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, {}},
        target,
        {0, 0, 0, 64}
    });
    const auto vertexPush = static_cast<std::uint32_t>(shaders.vertex.result->pushConstants.size());
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    shaders.pixel = CompileRequest({
        {ShaderRecompiler::ShaderStage::Fragment, SamplePixel().Address(), SamplePixel().Code(), 0, {}},
        {64, 0, pixelUserData, std::nullopt, pixel, std::nullopt, {}},
        target,
        PixelPushLayout(vertexPush, target)
    });
    shaders.pixelPush = PixelPushOffset(vertexPush, target);
    return shaders;
}

void Draw(AgcDriver::VulkanDevice& device, const Shaders& shaders, const std::uint8_t* indices) {
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, decltype(Pixels)::ByteSize, 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const std::array<AgcDriver::Graphics::CompiledShader, 2> stages{{
        {ShaderRecompiler::ShaderStage::Vertex, shaders.vertex.result.get(), 0},
        {ShaderRecompiler::ShaderStage::Fragment, shaders.pixel.result.get(), shaders.pixelPush}
    }};
    const AgcDriver::Pm4::DrawParameters draw{static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(indices)), Corners, 4u, 1u, 0u, true};
    device.Draw(state, draw, stages);
}

void Fill(std::uint8_t* texels, std::uint8_t* indices, std::uint32_t seed) {
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const auto texel = Texel(seed, x, y);
            std::memcpy(texels + (std::size_t{y} * Width + x) * 4u, &texel, 4u);
        }
    }
    for (std::uint32_t corner = 0; corner < Corners; ++corner) std::memcpy(indices + std::size_t{corner} * 4u, &corner, 4u);
    AgcDriver::GuestMemory::BumpCollectEpoch();
}

void Check(AgcDriver::VulkanDevice& device, std::uint8_t* texels, std::uint8_t* indices, std::uint32_t seed, const std::string& what) {
    const auto shaders = Compile(device, texels);
    const auto textures = AgcDriver::Graphics::SampledTextureSources();
    const auto scans = AgcDriver::Graphics::DrawIndexScans();
    Pixels.fill(Untouched);
    NewPath::StderrCapture log;
    Draw(device, shaders, indices);
    device.WaitIdle();
    const auto text = log.Finish();
    AgcDriver::GuestMemory::FlushGpuWrites(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Pixels.data())), decltype(Pixels)::ByteSize);
    const auto after = AgcDriver::Graphics::SampledTextureSources();
    std::size_t wrong = 0;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) wrong += Pixels[y * Width + x] != Texel(seed, x, y);
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " pixels are wrong\n" + text);
    Expect(after.gathers > textures.gathers, what + ": the texture was not gathered on the GPU from the image mirror and the import");
    Expect(AgcDriver::Graphics::DrawIndexScans() == scans, what + ": the indices were copied and scanned on the CPU instead of read from the image mirror and the import");
}

constexpr std::uint64_t ImagePage = 4096;

alignas(4096) std::array<std::uint32_t, 3u * 1024u> ImageWords{1u};

void ProtectImage(std::uint8_t* page, bool writable) {
    GuestAllocations::Mutation().Protect(page, ImagePage, true, writable, true, [page, writable] {
        DWORD old = 0;
        VirtualProtect(page, ImagePage, writable ? PAGE_READWRITE : PAGE_READONLY, &old);
    });
}

void ImageCase(AgcDriver::VulkanDevice& device, std::uint32_t seed, const std::string& what) {
    auto* base = reinterpret_cast<std::uint8_t*>(ImageWords.data());
    auto* texels = base + ImagePage - TextureBytes / 2u;
    auto* indices = base + 2u * ImagePage - 4u;
    Fill(texels, indices, seed);
    ProtectImage(base, false);
    ProtectImage(base + 2u * ImagePage, false);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    Check(device, texels, indices, seed, what);
    ProtectImage(base, true);
    ProtectImage(base + 2u * ImagePage, true);
}

void Run(AgcDriver::VulkanDevice& device) {
    GuestAllocations::Mutation().RegisterMainImage();
    ImageCase(device, 5u, "a texture and indices across a read-only image page and a writable one");
    ImageCase(device, 6u, "the same draw after the CPU rewrote both");
    GuestAllocations::Mutation().UnregisterImage(GetModuleHandleW(nullptr));
}

}

#endif

int main() {
#ifdef _WIN32
    return Main("draw inputs from mirrors", Run);
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
