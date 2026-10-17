#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"

#ifdef _WIN32

namespace {

using namespace DrawSubmit;
using NewPath::PageBytes;

constexpr std::uint32_t Width = 128;
constexpr std::uint32_t Height = 128;
constexpr std::uint32_t RenderTargetAttrib3 = (0x1bu << 14u) | (1u << 24u) | (1u << 27u);
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t Red = 0xff0000ffu;
constexpr std::uint32_t Cleared = 0u;
constexpr std::uint32_t VertexBytes = 16;
constexpr std::uint32_t BaseDraws = 4;
constexpr std::array<std::uint32_t, 3> Scales{1u, 4u, 16u};
constexpr std::uint32_t MostDraws = BaseDraws * Scales.back();

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

struct Counter {
    const char* name;
    bool perDraw;
};

constexpr std::array<Counter, 20> Counters{{
    {"draw-input copies", false},
    {"draw-input gathers", false},
    {"index scans", false},
    {"render passes begun", false},
    {"render passes continued", true},
    {"label table lookups", true},
    {"pending-write notes", true},
    {"pending-write publications", true},
    {"DCC key proofs scanned", true},
    {"DCC key range proofs scanned", true},
    {"HTILE store-back reads", true},
    {"image registry stores", true},
    {"image refreshes proved", true},
    {"untracked image compares", true},
    {"single-pass image moves", true},
    {"address space builds", true},
    {"address space rebuilds", true},
    {"mirror rebuilds", true},
    {"mirror sweeps", true},
    {"lease contention waits", true},
}};

using Sample = std::array<std::uint64_t, Counters.size()>;

Sample Take() {
    using namespace AgcDriver::Graphics;
    const auto passes = DrawPasses();
    const auto proofs = KeyProofCounts();
    const auto spaces = AddressSpaceCounters();
    const auto rebuilt = spaces.rebuiltFirst + spaces.rebuiltGeneration + spaces.rebuiltEpoch + spaces.rebuiltDevice + spaces.rebuiltWaiterDrop;
    const auto mirrors = MirrorCounters();
    const auto leases = LeaseCounters();
    return {DrawInputCopies(), DrawInputGathers(), DrawIndexScans(), passes.begun, passes.continued, Recorder::LabelTableLookups(), Recorder::WriteGeneration(), Recorder::PublishGeneration(), proofs.scanned, proofs.rangeScanned, HtileStoreBackReads(), StorageTexture::RegistryStores(), StorageTexture::RefreshesProved(), StorageTexture::UntrackedCompares(), StorageTexture::SinglePassMoves(), spaces.hits + rebuilt, rebuilt, mirrors.rebuilds, mirrors.sweeps, leases.contentionWaits};
}

void Begin(Commands& commands) {
    Baseline(commands, {Target->Data(), Width, Height}, CenterAndPosition);
    Context(commands, 0x3b8, RenderTargetAttrib3);
    Context(commands, 0x31b, 0u);
    Programs(commands, BoundedVertex(), VertexUserWords, RedPixel(), 0u);
    Instances(commands, 1u);
    IndexType(commands, Index32);
}

std::array<std::uint32_t, VertexUserWords> VertexDescriptor(std::uint64_t address) {
    return {Low(address), (High(address) & 0xffffu) | (VertexBytes << 16u), static_cast<std::uint32_t>(FullScreen.size()), 0x01016facu};
}

Sample Submission(const NewPath::PrivatePages& vertices, const NewPath::PrivatePages& indices, bool indexed, bool distinct, std::uint32_t draws, const std::string& what) {
    std::fill_n(reinterpret_cast<std::uint32_t*>(Target->Data()), Width * Height, Cleared);
    const auto before = Take();
    {
        Commands commands;
        Begin(commands);
        for (std::uint32_t draw = 0; draw < draws; ++draw) {
            const auto own = distinct ? draw : 0u;
            VertexUserData(commands, VertexDescriptor(vertices.Address(own * PageBytes)));
            if (indexed) DrawIndex(commands, indices.Address(own * 3u * sizeof(std::uint32_t)), 3u, 3u);
            else DrawAuto(commands, static_cast<std::uint32_t>(FullScreen.size()));
        }
        commands.Submit(0);
        NewPath::WaitIdle();
    }
    const auto after = Take();
    const auto pixels = ReadBack(reinterpret_cast<const std::uint32_t*>(Target->Data()), Width * Height);
    const auto wrong = static_cast<std::size_t>(std::ranges::count_if(pixels, [](std::uint32_t pixel) { return pixel != Red; }));
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(pixels.size()) + " pixels are not " + Hex(Red));
    Sample delta{};
    for (std::size_t counter = 0; counter < delta.size(); ++counter) delta[counter] = after[counter] - before[counter];
    return delta;
}

void Case(bool indexed, bool distinct) {
    const std::string what = std::string(indexed ? "indexed draws" : "draws") + (distinct ? " with their own vertex page and index range" : " over one vertex and one index range");
    NewPath::PrivatePages vertices(distinct ? MostDraws : 1u);
    NewPath::PrivatePages indices(1);
    for (std::uint32_t page = 0; page < (distinct ? MostDraws : 1u); ++page) std::memcpy(vertices.Data() + page * PageBytes, FullScreen.data(), sizeof(FullScreen));
    for (std::uint32_t index = 0; index < 3u * MostDraws; ++index) indices.Words()[index] = index % 3u;
    static_cast<void>(Submission(vertices, indices, indexed, distinct, MostDraws, what + ", warm-up"));
    std::array<Sample, Scales.size()> deltas{};
    for (std::size_t scale = 0; scale < Scales.size(); ++scale) deltas[scale] = Submission(vertices, indices, indexed, distinct, BaseDraws * Scales[scale], what + ", " + std::to_string(BaseDraws * Scales[scale]) + " draws");
    for (std::size_t counter = 0; counter < Counters.size(); ++counter) {
        const auto& [name, perDraw] = Counters[counter];
        std::string counts;
        for (std::size_t scale = 0; scale < Scales.size(); ++scale) counts += (scale == 0 ? "" : ", ") + std::to_string(deltas[scale][counter]) + " for " + std::to_string(BaseDraws * Scales[scale]) + " draws";
        const auto first = deltas[0][counter];
        const auto middle = deltas[1][counter];
        const auto last = deltas[2][counter];
        const auto firstStep = static_cast<std::uint64_t>(Scales[1] - Scales[0]) * BaseDraws;
        const auto lastStep = static_cast<std::uint64_t>(Scales[2] - Scales[1]) * BaseDraws;
        const bool linear = middle >= first && last >= middle && (last - middle) * firstStep == (middle - first) * lastStep;
        Expect(linear, what + ": " + name + " do not grow linearly with the draws (" + counts + "): per-draw work depends on more than the draw");
        if (!perDraw) Expect(middle == first && last == first, what + ": " + name + " grow with the draws (" + counts + "): repeated draws over the same resources must not repeat it");
    }
}

void Run() {
    Register(VertexHeader, BoundedVertex(), VertexShaderType);
    Register(PixelHeader, RedPixel(), PixelShaderType);
    NewPath::PrivatePages target(1);
    Target = &target;
    Cases({{"draws", [] { Case(false, false); }}, {"indexed draws", [] { Case(true, false); }}, {"distinct draws", [] { Case(false, true); }}, {"distinct indexed draws", [] { Case(true, true); }}});
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("draw work constant", Run);
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
