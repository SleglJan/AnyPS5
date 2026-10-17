#include "DrawSubmitFixture.hpp"

namespace {

using namespace DrawSubmit;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 32;
constexpr std::uint32_t VertexUserWords = 8;
constexpr std::uint32_t SSetpcB64 = 0x20u;
constexpr std::uint32_t SSwappcB64 = 0x21u;

TestGuest::Array<std::uint32_t, Width * Height> Pixels;
TestGuest::Array<std::uint32_t, 12> Output;
TestGuest::Array<std::array<float, 4>, 3> Triangle;
TestGuest::Array<std::uint32_t, 4> Table;
TestGuest::Array<std::uint32_t, 64> FetchSubroutine;

Header VertexHeader;
Header PixelHeader;

constexpr std::uint32_t Sop1(std::uint32_t opcode, std::uint32_t sdst, std::uint32_t ssrc0) {
    return 0xbe800000u | (sdst << 16u) | (opcode << 8u) | ssrc0;
}

const Program& CallingVertex() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Sop1(SSwappcB64, 2, 8)
              << Vop2(VLshlRevB32, 1, InlineInteger(4), 5)
              << Mubuf(BufferStoreDwordX4, 20, 1, 12)
              << Export(PositionTarget, 20, false)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& FlatPixel() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Vop1(VMovB32, 7, InlineOne)
              << Vop1(VMovB32, 8, InlineOne)
              << Vop1(VMovB32, 9, InlineOne)
              << Vop1(VMovB32, 10, InlineOne)
              << Export(ColorTarget, 7, true)
              << EndProgram;
        return built;
    }();
    return program;
}

void WriteFetchSubroutine() {
    using namespace DrawSubmit::Isa;
    Program built;
    built << Smem(SLoadDwordX4, 16, 10, 0)
          << WaitAll
          << Indexed(BufferLoadFormatXyzw, 20, 5, 16)
          << WaitAll
          << Sop1(SSetpcB64, 0, 2);
    FetchSubroutine.fill(EndProgram);
    std::copy(built.words.begin(), built.words.begin() + static_cast<std::ptrdiff_t>(built.size), FetchSubroutine.begin());
}

std::array<std::uint32_t, VertexUserWords> UserData() {
    const auto subroutine = AddressOf(FetchSubroutine.data());
    const auto table = AddressOf(Table.data());
    const auto output = DeviceTable::Buffer(Output.data(), static_cast<std::uint32_t>(decltype(Output)::ByteSize));
    return {Low(subroutine), High(subroutine), Low(table), High(table), output[0], output[1], output[2], output[3]};
}

bool Rejected(const std::string& message) {
    const auto where = "shader " + NewPath::Hex(CallingVertex().Address()) + " stage " + std::to_string(static_cast<unsigned>(ShaderRecompiler::ShaderStage::Vertex)) + ": ";
    const bool driver = message.find("calls code through its user data is not implemented") != std::string::npos;
    const bool recompiler = message.find("s_swappc_b64 fetch-shader call at program counter 0 has no parsed fetch-shader plan to inline") != std::string::npos;
    return message.find(where) != std::string::npos && (driver || recompiler);
}

void Run() {
    Triangle[0] = {-1.0f, -1.0f, 0.5f, 1.0f};
    Triangle[1] = {3.0f, -1.0f, 0.25f, 1.0f};
    Triangle[2] = {-1.0f, 3.0f, 0.75f, 1.0f};
    const auto vertices = AddressOf(Triangle.data());
    Table = {Low(vertices), (High(vertices) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Triangle.size()), 0x0104dfacu};
    WriteFetchSubroutine();
    Output.fill(DeviceTable::Untouched);
    Register(VertexHeader, CallingVertex(), VertexShaderType);
    Register(PixelHeader, FlatPixel(), PixelShaderType);
    Commands commands;
    Baseline(commands, {Pixels.data(), Width, Height}, CenterAndPosition);
    Programs(commands, CallingVertex(), VertexUserWords, FlatPixel(), 0u);
    Instances(commands, 1u);
    VertexUserData(commands, UserData());
    DrawAuto(commands, static_cast<std::uint32_t>(Triangle.size()));
    commands.Submit(0);
    try {
        NewPath::WaitIdle();
    } catch (const std::exception& error) {
        Expect(Rejected(error.what()), std::string("fetch subroutine call: the draw failed with another error: ") + error.what());
        return;
    }
    Expect(false, "fetch subroutine call: a vertex program that calls a fetch subroutine at a user data address was drawn instead of rejected");
}

}

int main() {
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    try {
        Run();
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(Rejected(error.what()), std::string("shutdown: ") + error.what());
    }
    return DeviceTable::Finish("fetch subroutine rejection");
}
