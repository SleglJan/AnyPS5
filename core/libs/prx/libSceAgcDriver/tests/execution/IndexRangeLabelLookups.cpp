#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"

#ifdef _WIN32

namespace {

using namespace DrawSubmit;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 64;
constexpr std::uint32_t RenderTargetAttrib3 = (0x1bu << 14u) | (1u << 24u) | (1u << 27u);
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t Red = 0xff0000ffu;
constexpr std::uint32_t Cleared = 0u;
constexpr std::uint32_t SmallIndices = 3;
constexpr std::uint32_t LargeIndices = 3 * 4096;

using Corners = std::array<std::array<float, 4>, 3>;
constexpr Corners FullScreen{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};

NewPath::PrivatePages* Target = nullptr;

Header VertexHeader;
Header PixelHeader;

const Program& BoundedVertex() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Indexed(BufferLoadDwordX4, 10, 5, 8)
              << WaitAll
              << Export(PositionTarget, 10, false)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& RedPixel() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Vop1(VMovB32, 0, InlineOne)
              << Vop1(VMovB32, 1, InlineInteger(0))
              << Vop1(VMovB32, 2, InlineInteger(0))
              << Vop1(VMovB32, 3, InlineOne)
              << Export(ColorTarget, 0, true)
              << EndProgram;
        return built;
    }();
    return program;
}

std::uint64_t DrawLookups(const NewPath::PrivatePages& vertices, const NewPath::PrivatePages& indices, std::uint32_t count, const std::string& what) {
    std::fill_n(reinterpret_cast<std::uint32_t*>(Target->Data()), Width * Height, Cleared);
    const auto address = vertices.Address();
    const std::array<std::uint32_t, VertexUserWords> descriptor{Low(address), (High(address) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(FullScreen.size()), 0x01016facu};
    const auto before = AgcDriver::Graphics::Recorder::LabelTableLookups();
    {
        Commands commands;
        Baseline(commands, {Target->Data(), Width, Height}, CenterAndPosition);
        Context(commands, 0x3b8, RenderTargetAttrib3);
        Context(commands, 0x31b, 0u);
        Programs(commands, BoundedVertex(), VertexUserWords, RedPixel(), 0u);
        Instances(commands, 1u);
        IndexType(commands, Index32);
        VertexUserData(commands, descriptor);
        DrawIndex(commands, indices.Address(), count, count);
        commands.Submit(0);
        NewPath::WaitIdle();
    }
    const auto lookups = AgcDriver::Graphics::Recorder::LabelTableLookups() - before;
    const auto pixels = ReadBack(reinterpret_cast<const std::uint32_t*>(Target->Data()), Width * Height);
    const auto wrong = static_cast<std::size_t>(std::ranges::count_if(pixels, [](std::uint32_t pixel) { return pixel != Red; }));
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(pixels.size()) + " pixels are not " + Hex(Red));
    return lookups;
}

void Run() {
    Register(VertexHeader, BoundedVertex(), VertexShaderType);
    Register(PixelHeader, RedPixel(), PixelShaderType);
    NewPath::PrivatePages target(1);
    Target = &target;
    NewPath::PrivatePages vertices(1);
    NewPath::PrivatePages indices(1);
    std::memcpy(vertices.Data(), FullScreen.data(), sizeof(FullScreen));
    for (std::uint32_t index = 0; index < LargeIndices; ++index) indices.Words()[index] = index % 3u;
    static_cast<void>(DrawLookups(vertices, indices, SmallIndices, "a warm-up draw"));
    const auto small = DrawLookups(vertices, indices, SmallIndices, "a draw of 3 indices");
    const auto large = DrawLookups(vertices, indices, LargeIndices, "a draw of " + std::to_string(LargeIndices) + " indices");
    Expect(small != 0u, "an indexed draw looked up no label for its index range: the cross-stream order over the range did not run");
    Expect(large == small, "an indexed draw over " + std::to_string(LargeIndices * 4u) + " index bytes made " + std::to_string(large) + " label table lookups, one over 12 bytes made " + std::to_string(small) + ": the range must be looked up once, not per dword");
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("index range label lookups", Run);
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
