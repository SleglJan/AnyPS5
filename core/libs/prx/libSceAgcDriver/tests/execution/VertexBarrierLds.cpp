#include "DrawSubmitFixture.hpp"
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32

namespace {

using namespace DrawSubmit;
using NewPath::PrivatePages;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 64;
constexpr std::uint32_t Vertices = 96;
constexpr std::uint32_t Marks = 1024;
constexpr std::uint32_t Marked = 0x40000000u;
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t DsWriteB32 = 0x0du;
constexpr std::uint32_t DsReadB32 = 0x36u;
constexpr std::uint32_t SBarrier = 0xbf8a0000u;
constexpr std::uint32_t PassthroughRouting = 0x02002000u;
constexpr auto LabelLimit = std::chrono::seconds(10);

Header VertexHeader;
Header PixelHeader;

constexpr std::array<std::uint32_t, 2> Ds(std::uint32_t opcode, std::uint32_t address, std::uint32_t data, std::uint32_t destination) {
    return {0xd8000000u | (opcode << 18u), (destination << 24u) | (data << 8u) | address};
}

const Program& LdsVertex() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        constexpr std::uint32_t VOrB32 = 0x1cu;
        Program built;
        built << SBarrier
              << Vop2(VLshlRevB32, 4, InlineInteger(2), 5)
              << Vop2(VOrB32, 6, Literal, 5) << Marked
              << Ds(DsWriteB32, 4, 6, 0)
              << WaitAll
              << Ds(DsReadB32, 4, 0, 7)
              << WaitAll
              << Mubuf(BufferStoreDword, 7, 4, 8)
              << Vop1(VMovB32, 16, InlineInteger(0))
              << Vop1(VMovB32, 17, InlineInteger(0))
              << Vop1(VMovB32, 18, InlineInteger(0))
              << Vop1(VMovB32, 19, InlineOne)
              << WaitAll
              << Export(PositionTarget, 16, false)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& FlatPixel() {
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

void Run() {
    Register(VertexHeader, LdsVertex(), VertexShaderType);
    Register(PixelHeader, FlatPixel(), PixelShaderType);
    const std::string what = "a draw on the vertex path whose vertex program begins with s_barrier and then uses LDS and stores to memory";
    PrivatePages pixels(1);
    PrivatePages marks(1);
    PrivatePages labels(1);
    PrivatePages indices(1);
    for (std::uint32_t index = 0; index < Vertices; ++index) indices.Words()[index] = index;
    for (std::uint32_t word = 0; word < Width * Height; ++word) pixels.Words()[word] = 0u;
    for (std::uint32_t word = 0; word < Marks; ++word) marks.Words()[word] = 0u;
    for (std::uint32_t word = 0; word < 64u; ++word) labels.Words()[word] = 0u;
    AgcDriver::GuestMemory::BumpCollectEpoch();
    Commands commands;
    Baseline(commands, {pixels.Data(), Width, Height}, CenterAndPosition);
    DrawSubmit::Context(commands, 0x2d5, PassthroughRouting);
    Programs(commands, LdsVertex(), VertexUserWords, FlatPixel(), 0u);
    const auto buffer = NewPath::BufferResource(marks.Address(), Marks * 4u);
    VertexUserData(commands, buffer);
    DrawSubmit::Instances(commands, 1u);
    IndexType(commands, Index32);
    DrawIndex(commands, indices.Address(), Vertices, Vertices);
    commands.WriteData(labels.Address(), {1u});
    std::string failure;
    try {
        commands.Submit(0);
        bool landed = false;
        for (const auto deadline = Clock::now() + LabelLimit; Clock::now() < deadline && !landed;) {
            landed = labels.Words()[0] == 1u;
            if (!landed) std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        Expect(landed, what + ": the label never landed");
        NewPath::WaitIdle();
        NewPath::Settle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    AgcDriver::GuestMemory::FlushGpuWrites(marks.Address(), marks.Bytes());
    std::size_t wrong = 0;
    std::string firstWrong;
    for (std::uint32_t mark = 0; mark < Marks; ++mark) {
        const auto actual = marks.Words()[mark];
        const auto expected = mark < Vertices ? (Marked | mark) : 0u;
        if (actual == expected) continue;
        ++wrong;
        if (firstWrong.empty()) firstWrong = "vertex " + std::to_string(mark) + " marked " + Hex(actual) + ", expected " + Hex(expected);
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " vertex marks are wrong; first " + firstWrong);
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("vertex barrier lds", Run);
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
