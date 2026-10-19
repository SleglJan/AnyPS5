#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"

#ifdef _WIN32

namespace {

using namespace DrawSubmit;

constexpr std::uint32_t Width = 128;
constexpr std::uint32_t Height = 128;
constexpr std::uint32_t RenderTargetAttrib3 = (0x1bu << 14u) | (1u << 24u) | (1u << 27u);
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t Red = 0xff0000ffu;
constexpr std::uint32_t Cleared = 0u;
constexpr std::uint32_t VertexBytes = 16;
constexpr std::uint32_t Draws = 16;
constexpr std::uint32_t KeyBytes = Width * Height * 4u / 256u;
constexpr std::uint32_t DccEnable = 0x10000000u;
constexpr std::uint32_t KeysUncompressed = 0xffffffffu;
constexpr std::uint32_t KeysClear0000 = 0u;

using Corners = std::array<std::array<float, 4>, 3>;
constexpr Corners FullScreen{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};

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

std::array<std::uint32_t, VertexUserWords> VertexDescriptor(std::uint64_t address) {
    return {Low(address), (High(address) & 0xffffu) | (VertexBytes << 16u), static_cast<std::uint32_t>(FullScreen.size()), 0x01016facu};
}

struct Memory {
    NewPath::PrivatePages target{1};
    NewPath::PrivatePages keys{1};
    NewPath::PrivatePages vertices{1};
    bool dcc = false;
};

std::uint64_t Submission(const Memory& memory, std::uint32_t draws, const std::string& what) {
    const auto before = AgcDriver::Graphics::StorageTexture::RefreshesProved();
    {
        Commands commands;
        Baseline(commands, {memory.target.Data(), Width, Height}, CenterAndPosition);
        if (memory.dcc) {
            Context(commands, 0x31c, 0x28028u | DccEnable);
            Context(commands, 0x325, Low(memory.keys.Address() >> 8u));
            Context(commands, 0x3a8, Low(memory.keys.Address() >> 40u));
        }
        Context(commands, 0x3b8, RenderTargetAttrib3);
        Context(commands, 0x31b, 0u);
        Programs(commands, BoundedVertex(), VertexUserWords, RedPixel(), 0u);
        Instances(commands, 1u);
        for (std::uint32_t draw = 0; draw < draws; ++draw) {
            VertexUserData(commands, VertexDescriptor(memory.vertices.Address(0)));
            DrawAuto(commands, static_cast<std::uint32_t>(FullScreen.size()));
        }
        commands.Submit(0);
        NewPath::WaitIdle();
    }
    const auto proved = AgcDriver::Graphics::StorageTexture::RefreshesProved() - before;
    const auto pixels = ReadBack(reinterpret_cast<const std::uint32_t*>(memory.target.Data()), Width * Height);
    const auto wrong = static_cast<std::size_t>(std::ranges::count_if(pixels, [](std::uint32_t pixel) { return pixel != Red; }));
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(pixels.size()) + " pixels are not " + Hex(Red));
    std::printf("%s: %llu of %u target refreshes proved\n", what.c_str(), static_cast<unsigned long long>(proved), draws);
    return proved;
}

void Case(bool dcc) {
    const std::string what = dcc ? "DCC target" : "target";
    Memory memory;
    memory.dcc = dcc;
    std::memcpy(memory.vertices.Data(), FullScreen.data(), sizeof(FullScreen));
    std::fill_n(memory.target.Words(), Width * Height, Cleared);
    std::fill_n(memory.keys.Words(), KeyBytes / 4u, KeysUncompressed);
    static_cast<void>(Submission(memory, Draws, what + ", first submission"));
    const auto proved = Submission(memory, Draws, what + ", unchanged");
    Expect(proved + 1u >= Draws, what + ": " + std::to_string(Draws - 1u - std::min<std::uint64_t>(proved, Draws - 1u)) + " of the " + std::to_string(Draws - 1u) + " later draws into the unchanged target refreshed it in full: the target is walked per draw, not per change");
    std::fill_n(memory.target.Words(), Width * Height, Cleared);
    const auto changed = Submission(memory, Draws, what + ", written by the CPU");
    Expect(changed < Draws, what + ": the first draw after a CPU write of the target was answered from the proof");
    if (!dcc) return;
    std::fill_n(memory.keys.Words(), KeyBytes / 4u, KeysClear0000);
    const auto cleared = Submission(memory, Draws, what + ", fast-cleared keys");
    Expect(cleared < Draws, what + ": the first draw after a fast clear of the keys was answered from the proof");
    Expect(cleared + 2u >= Draws, what + ": " + std::to_string(Draws - 2u - std::min<std::uint64_t>(cleared, Draws - 2u)) + " of the " + std::to_string(Draws - 2u) + " draws after the two that decided the fast-cleared keys and proved the target refreshed it in full: the target is walked per draw, not per change");
}

void Run() {
    Register(VertexHeader, BoundedVertex(), VertexShaderType);
    Register(PixelHeader, RedPixel(), PixelShaderType);
    Cases({{"target", [] { Case(false); }}, {"DCC target", [] { Case(true); }}});
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("resident target refresh", Run);
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
