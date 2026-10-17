#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"

#ifdef _WIN32

namespace {

using namespace DrawSubmit;
using NewPath::PageBytes;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 64;
constexpr std::uint32_t RenderTargetAttrib3 = (0x1bu << 14u) | (1u << 24u) | (1u << 27u);
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t Red = 0xff0000ffu;
constexpr std::uint32_t Cleared = 0u;
constexpr std::uint32_t VertexBytes = 16;
constexpr std::uint32_t VertexSpareWord = 64;
constexpr std::uint32_t Draws = 8;
constexpr std::uint32_t Frames = 6;
constexpr std::uint32_t UsedWrittenPages = 2;
constexpr std::array<std::uint32_t, 2> WrittenCounts{16u, 1024u};

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

AgcDriver::Graphics::ResidencyCounts Counts() {
    AgcDriver::Graphics::ResidencyCounts counts{};
    NewPath::Locked([&] { counts = NewPath::Residency().Manager().Counts(); });
    return counts;
}

std::uint32_t ResidentPages(const NewPath::SharedPages& pages, std::uint32_t count) {
    std::uint32_t resident = 0;
    NewPath::Locked([&] {
        for (std::uint32_t page = 0; page < count; ++page) resident += NewPath::Residency().Manager().Slot(NewPath::PageOf(pages.Address(page))).has_value() ? 1u : 0u;
    });
    return resident;
}

void Submission(const NewPath::SharedPages& vertices, const std::string& what) {
    std::fill_n(reinterpret_cast<std::uint32_t*>(Target->Data()), Width * Height, Cleared);
    {
        Commands commands;
        Baseline(commands, {Target->Data(), Width, Height}, CenterAndPosition);
        Context(commands, 0x3b8, RenderTargetAttrib3);
        Context(commands, 0x31b, 0u);
        Programs(commands, BoundedVertex(), VertexUserWords, RedPixel(), 0u);
        Instances(commands, 1u);
        const auto address = vertices.Address();
        for (std::uint32_t draw = 0; draw < Draws; ++draw) {
            VertexUserData(commands, std::array<std::uint32_t, VertexUserWords>{Low(address), (High(address) & 0xffffu) | (VertexBytes << 16u), static_cast<std::uint32_t>(FullScreen.size()), 0x01016facu});
            DrawAuto(commands, static_cast<std::uint32_t>(FullScreen.size()));
        }
        commands.Submit(0);
        NewPath::WaitIdle();
    }
    const auto pixels = ReadBack(reinterpret_cast<const std::uint32_t*>(Target->Data()), Width * Height);
    const auto wrong = static_cast<std::size_t>(std::ranges::count_if(pixels, [](std::uint32_t pixel) { return pixel != Red; }));
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(pixels.size()) + " pixels are not " + Hex(Red));
}

std::uint32_t Value(std::uint32_t frame, std::uint32_t page) {
    return (frame << 20u) | page;
}

struct Work {
    std::uint64_t faults = 0;
    std::uint64_t refreshed = 0;
    std::uint64_t hot = 0;
    std::uint64_t copiedInBytes = 0;
};

std::string Describe(const Work& work) {
    return std::to_string(work.faults) + " faults, " + std::to_string(work.refreshed) + " refreshes, " + std::to_string(work.hot) + " pages made hot, " + std::to_string(work.copiedInBytes) + " bytes copied in";
}

Work WrittenFrames(std::uint32_t count) {
    const std::string what = std::to_string(count) + " eagerly admitted pages written by the CPU";
    NewPath::PrivatePages target(1);
    Target = &target;
    NewPath::SharedPages vertices(1);
    std::memcpy(const_cast<std::uint32_t*>(vertices.Words()), FullScreen.data(), sizeof(FullScreen));
    NewPath::SharedPages written(count);
    for (std::uint32_t page = 0; page < count; ++page) written.Words(page)[0] = page;
    NewPath::AdmitEagerly();
    Expect(ResidentPages(written, count) == count, what + ": " + std::to_string(ResidentPages(written, count)) + " of them are resident after eager admission");
    Submission(vertices, what + ", before the writes");
    const auto beforeFirst = Counts();
    for (std::uint32_t page = 0; page < count; ++page) written.Words(page)[0] = Value(1, page);
    vertices.Words()[VertexSpareWord] = 1;
    Submission(vertices, what + ", first written frame");
    const auto afterFirst = Counts();
    Expect(afterFirst.unusedDemoted - beforeFirst.unusedDemoted == count, what + ": " + std::to_string(afterFirst.unusedDemoted - beforeFirst.unusedDemoted) + " of them left residency at their first write before any GPU use");
    Expect(ResidentPages(written, count) == 0, what + ": " + std::to_string(ResidentPages(written, count)) + " of them are still resident after their first write");
    for (std::uint32_t frame = 2; frame < Frames + 2; ++frame) {
        for (std::uint32_t page = 0; page < count; ++page) written.Words(page)[0] = Value(frame, page);
        vertices.Words()[VertexSpareWord] = frame;
        Submission(vertices, what + ", frame " + std::to_string(frame));
    }
    const auto last = Counts();
    Expect(last.unusedDemoted == afterFirst.unusedDemoted, what + ": " + std::to_string(last.unusedDemoted - afterFirst.unusedDemoted) + " more pages were admitted and left again in the later frames");
    NewPath::MirrorReader reader;
    const auto shown = std::min<std::uint32_t>(count, NewPath::MirrorReader::Pages);
    const auto lanes = reader.Read(written.Address(), shown);
    for (std::uint32_t page = 0; page < shown; ++page) {
        const auto lane = lanes[page * NewPath::SubpagesPerPage];
        Expect(lane == Value(Frames + 1, page), what + ": the GPU reads " + Hex(lane) + " from page " + std::to_string(page) + ", the CPU last wrote " + Hex(Value(Frames + 1, page)));
    }
    return {last.faults - afterFirst.faults, last.refreshed - afterFirst.refreshed, last.hot - afterFirst.hot, last.copiedInBytes - afterFirst.copiedInBytes};
}

void Case() {
    std::array<Work, WrittenCounts.size()> work{};
    for (std::size_t index = 0; index < WrittenCounts.size(); ++index) work[index] = WrittenFrames(WrittenCounts[index]);
    const auto& few = work.front();
    const auto& many = work.back();
    const auto shapes = std::to_string(WrittenCounts.front()) + " written pages: " + Describe(few) + "; " + std::to_string(WrittenCounts.back()) + ": " + Describe(many);
    Expect(few.faults == many.faults && few.refreshed == many.refreshed, "the residency work of " + std::to_string(Frames) + " frames grows with the eagerly admitted pages the CPU writes and no GPU work uses (" + shapes + ")");
    for (const auto& shape : work) Expect(shape.hot <= UsedWrittenPages && shape.copiedInBytes <= std::uint64_t{Frames} * UsedWrittenPages * PageBytes, "pages other than the target and the vertices the draws use were made hot or copied in (" + shapes + ")");
    Expect(few.refreshed >= Frames, "the page every draw reads is not refreshed after each CPU write (" + shapes + ")");
}

void Run() {
    Register(VertexHeader, BoundedVertex(), VertexShaderType);
    Register(PixelHeader, RedPixel(), PixelShaderType);
    Cases({{"eagerly admitted pages written before any GPU use", [] { Case(); }}});
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("residency unused mirror writes", Run);
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
