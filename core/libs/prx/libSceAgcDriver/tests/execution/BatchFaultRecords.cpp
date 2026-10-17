#include "DrawSubmitFixture.hpp"
#include "BdaAbi.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using DeviceTable::Expect;
using NewPath::Commands;
using NewPath::Hex;
using NewPath::High;
using NewPath::Low;
using NewPath::PageBytes;
using NewPath::PrivatePages;
using NewPath::Program;
using NewPath::Require;
using NewPath::WordsPerPage;

constexpr std::uint32_t GraphicsQueue = 0;
constexpr std::uint32_t Lanes = 16;
constexpr std::uint32_t SliceWords = Lanes;
constexpr std::uint32_t Writers = 64;
constexpr std::uint32_t FaultAt = Writers / 2;
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t StoredBase = 0x6b000000u;
constexpr std::uint32_t Textures = 8;
constexpr std::uint32_t TexelWords = DeviceTable::Width * DeviceTable::Height;
constexpr std::uint32_t MaxSamples = 2 * Textures;
constexpr std::uint32_t RotationBuilds = 40;
constexpr std::uint32_t ChunksPerBuild = 64;
constexpr std::uint64_t ChunkBytes = std::uint64_t{1} << ShaderRecompiler::BdaAbi::WrittenChunkShift;
constexpr std::uint32_t TargetWidth = 128;
constexpr std::uint32_t TargetHeight = 128;
constexpr std::uint32_t RenderTargetAttrib3 = (0x1bu << 14u) | (1u << 24u) | (1u << 27u);
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t Red = 0xff0000ffu;
constexpr std::uint32_t RectList = 17;
constexpr std::uint32_t SpinLanes = 64;
constexpr std::uint32_t SpinRounds = 1u << 22u;
constexpr std::uint32_t SpinWord = 2048;
constexpr std::uint32_t LabelWord = 4096;
constexpr std::uint32_t LabelValue = 0x1abe1000u;
constexpr auto FreeLimit = std::chrono::seconds(10);

using Corners = std::array<std::array<float, 4>, 3>;
constexpr Corners FullRectangle{{{-1.0f, -1.0f, 0.5f, 1.0f}, {1.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 1.0f, 0.5f, 1.0f}}};
constexpr Corners DegenerateRectangle{{{-1.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, -1.0f, 0.5f, 1.0f}}};

TestGuest::Array<std::uint32_t, MaxSamples * DeviceTable::SrtWords> Srts;
TestGuest::Array<std::uint32_t, MaxSamples * DeviceTable::OutputWords> Results;
std::uint32_t samples = 0;
DrawSubmit::Header VertexHeader;
DrawSubmit::Header PixelHeader;

const Program& SliceStoreProgram() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        Program built;
        built << Vop2(VLshlRevB32, 2, InlineInteger(2), 0)
              << Vop2(VAddNcU32, 1, 4, 0)
              << Mubuf(BufferStoreDword, 1, 2, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& GlobalStoreProgram() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(2), 0)
              << Vop2(VAddNcU32, 2, Literal, 0) << StoredBase
              << Global(GlobalStoreDword, 0, 1, 2, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& UnalignedAtomicProgram() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        constexpr std::uint32_t BufferAtomicAdd = 0x32u;
        Program built;
        built << Vop1(VMovB32, 1, InlineInteger(1))
              << Mubuf(BufferAtomicAdd, 1, 0, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& ChunkStoreProgram() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        Program built;
        built << Vop2(VLshlRevB32, 2, InlineInteger(ShaderRecompiler::BdaAbi::WrittenChunkShift), 0)
              << Vop2(VAddNcU32, 1, 4, 0)
              << Mubuf(BufferStoreDword, 1, 2, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& TexelStoreProgram() {
    static const Program program = [] {
        using namespace DeviceTable::Asm;
        Program built;
        built << Vop2(VLshlRevB32, 2, InlineInteger(2), 0)
              << Vop2(VAddNcU32, 1, 4, 0)
              << Vop1(VCvtF32U32, 1, Vgpr(1))
              << Mubuf(BufferStoreDword, 1, 2, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& SampleProgram() {
    static const Program program = [] {
        using namespace DeviceTable::Asm;
        Program built;
        LoadSrt(built, {{28, DeviceTable::OutputSlot}, {32, DeviceTable::SamplerSlot}, {36, DeviceTable::ImageSlot}});
        TexelCenter(built, 2);
        built << Mimg(ImageSampleLz, 1u, true, 4, 1, 36, 32);
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

const Program& SpinProgram() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        constexpr std::uint32_t VLshrRevB32 = 0x16u;
        constexpr std::uint32_t VXorB32 = 0x1du;
        constexpr std::uint32_t MoveLiteralToS9 = 0xbe8903ffu;
        constexpr std::uint32_t DecrementS9 = 0x80898109u;
        constexpr std::uint32_t CompareS9NotZero = 0xbf078009u;
        constexpr std::uint32_t BranchBackWhileNotZero = 0xbf85fff7u;
        Program built;
        built << Vop2(VAddNcU32, 1, InlineInteger(1), 0) << MoveLiteralToS9 << SpinRounds
              << Vop2(VLshlRevB32, 2, InlineInteger(13), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << Vop2(VLshrRevB32, 2, InlineInteger(17), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << Vop2(VLshlRevB32, 2, InlineInteger(5), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << DecrementS9 << CompareS9NotZero << BranchBackWhileNotZero
              << Vop2(VLshlRevB32, 3, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 1, 3, 0) << EndProgram;
        return built;
    }();
    return program;
}

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

class GuestBlock {
public:
    explicit GuestBlock(std::uint32_t pages) : bytes(std::uint64_t{pages} * PageBytes) {
        base = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, PageBytes));
        Require(base != nullptr, "batch fault records: cannot reserve guest pages");
        GuestArena::GuestArenaCommit_nid_postfix(base, bytes, PAGE_READWRITE, 16384);
        GuestAllocations::Mutation().Add(base, bytes, true, true, true);
    }

    ~GuestBlock() {
        try {
            if (registered) GuestAllocations::Mutation().Remove(base);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: batch fault records: the guest block could not be removed: %s\n", error.what());
        }
        GuestArena::GuestArenaReset_nid_postfix(base, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(base, bytes);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t Address(std::uint64_t offset = 0) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base)) + offset; }
    volatile std::uint32_t* Words() const { return reinterpret_cast<volatile std::uint32_t*>(base); }
    void* Base() const { return base; }
    std::uint64_t Bytes() const { return bytes; }
    void Unregistered() { registered = false; }

private:
    std::uint8_t* base = nullptr;
    std::uint64_t bytes = 0;
    bool registered = true;
};

struct VertexBuffer {
    PrivatePages pages{1};

    explicit VertexBuffer(const Corners& corners) { std::memcpy(pages.Data(), corners.data(), sizeof(corners)); }

    std::array<std::uint32_t, VertexUserWords> Descriptor() const {
        const auto address = pages.Address();
        return {Low(address), (High(address) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(FullRectangle.size()), 0x01016facu};
    }
};

std::size_t Count(const std::string& text, const std::string& what) {
    std::size_t count = 0;
    for (auto at = text.find(what); at != std::string::npos; at = text.find(what, at + 1u)) ++count;
    return count;
}

std::string LineWith(const std::string& text, const std::string& what) {
    const auto at = text.find(what);
    if (at == std::string::npos) return {};
    const auto begin = text.rfind('\n', at);
    const auto first = begin == std::string::npos ? 0u : begin + 1u;
    return text.substr(first, text.find('\n', at) - first);
}

std::string ProgramTag(const Program& program) {
    return "(program " + Hex(program.Address()) + ")";
}

std::uint64_t ReportedAddress(const std::string& report) {
    const auto at = report.find("address=0x");
    return at != std::string::npos ? std::strtoull(report.c_str() + at + 10, nullptr, 16) : 0u;
}

std::string Submitted(Commands& commands, std::string& log) {
    std::string failure;
    NewPath::StderrCapture capture;
    try {
        commands.Submit(GraphicsQueue);
        NewPath::WaitIdle();
        NewPath::Settle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    log = capture.Finish();
    return failure;
}

std::uint32_t Seed(std::uint32_t round, std::uint32_t slot) {
    return 0x51000000u | (round << 16u) | (slot << 8u);
}

void SliceWriter(Commands& commands, std::uint64_t address, std::uint32_t slot, std::uint32_t seed) {
    const auto buffer = NewPath::BufferResource(address + std::uint64_t{slot} * SliceWords * 4u, SliceWords * 4u);
    const std::array<std::uint32_t, 5> userData{buffer[0], buffer[1], buffer[2], buffer[3], seed};
    commands.Dispatch(SliceStoreProgram(), userData, Lanes);
}

void ExpectSlices(const volatile std::uint32_t* words, std::uint32_t slots, const std::function<std::uint32_t(std::uint32_t)>& seed, const std::string& what) {
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t word = 0; word < slots * SliceWords; ++word) {
        const auto slot = word / SliceWords;
        const auto expected = seed(slot) + word % SliceWords;
        const auto actual = words[word];
        if (actual == expected) continue;
        if (wrong++ == 0u) first = "word " + std::to_string(word) + " (writer " + std::to_string(slot) + ") is " + Hex(actual) + ", expected " + Hex(expected);
    }
    Expect(wrong == 0u, what + ": the bounded writers' results: " + std::to_string(wrong) + " words differ, first " + first);
}

void ExpectWrittenBlock(std::uint64_t address, std::uint64_t generation, const std::string& what) {
    AgcDriver::GuestMemory::CollectWritesUncached(address, PageBytes);
    const std::array<std::uint64_t, 1> generations{generation};
    std::array<std::uint8_t, 1> changed{};
    std::array<std::uint8_t, 1> cpu{};
    AgcDriver::GuestMemory::ChangedBlocks(address, PageBytes, generations, changed, cpu);
    Expect(changed[0] == AgcDriver::GuestMemory::BlockWritten && cpu[0] == 0u, what + ": the 64 KiB block at " + Hex(address) + " was not marked written by the GPU when its batch completed (changed " + std::to_string(changed[0]) + ", CPU " + std::to_string(cpu[0]) + ")");
}

void FaultAmongBoundedWriters(bool bounded) {
    PrivatePages output(1);
    PrivatePages readOnly(1, false);
    PrivatePages atomics(1);
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) {
        output.Words()[word] = Untouched;
        atomics.Words()[word] = Untouched;
    }
    for (std::uint32_t word = 0; word < Lanes; ++word) readOnly.Words()[word] = Untouched;
    const std::uint32_t round = bounded ? 2u : 1u;
    const auto& faulting = bounded ? UnalignedAtomicProgram() : GlobalStoreProgram();
    Commands commands;
    for (std::uint32_t writer = 0; writer < Writers; ++writer) {
        if (writer == FaultAt && bounded) commands.Dispatch(faulting, NewPath::BufferResource(atomics.Address(2u), 64u), Lanes);
        if (writer == FaultAt && !bounded) commands.Dispatch(faulting, std::array<std::uint32_t, 2>{Low(readOnly.Address()), High(readOnly.Address())}, Lanes);
        SliceWriter(commands, output.Address(), writer, Seed(round, writer));
    }
    std::string log;
    const auto failure = Submitted(commands, log);
    const auto what = std::string(bounded ? "an unaligned buffer atomic (a bounded build sharing the batch's fault record)" : "a GLOBAL store to a page the guest maps without CPU write access (a grant fault of an unbounded build)") + " among " + std::to_string(Writers) + " bounded writers of one batch";
    Expect(failure.empty(), what + ": the fault stopped the driver: " + failure);
    Expect(Count(log, "BDA access failed") == 1u, what + ": the fault was not reported exactly once:\n" + log);
    const auto report = LineWith(log, "BDA access failed");
    Expect(report.find(bounded ? "reason=7" : "reason=2") != std::string::npos, what + ": the report does not give the " + (bounded ? "Unaligned" : "Permission") + " reason: " + report);
    Expect(report.find(ProgramTag(faulting)) != std::string::npos, what + ": the report does not name the faulting program " + ProgramTag(faulting) + ": " + report);
    Expect(Count(log, ProgramTag(SliceStoreProgram())) == 0u, what + ": a fault was attributed to the bounded writers' program " + ProgramTag(SliceStoreProgram()) + ":\n" + log);
    const auto address = ReportedAddress(report);
    const auto begin = bounded ? atomics.Address() : readOnly.Address();
    Expect(address >= begin && address < begin + 64u + 4u * Lanes, what + ": the report names " + Hex(address) + ", not an address the faulting program accessed (" + Hex(begin) + "+0x80): " + report);
    const bool sharing = report.find("builds sharing the batch's fault record") != std::string::npos;
    if (bounded) Expect(sharing, what + ": the report does not say the build shared the batch's fault record: " + report);
    else Expect(!sharing, what + ": an unbounded build's fault was reported from a shared record: " + report);
    AgcDriver::GuestMemory::FlushGpuWrites(output.Address(), output.Bytes());
    ExpectSlices(output.Words(), Writers, [round](std::uint32_t slot) { return Seed(round, slot); }, what);
    if (!bounded) {
        for (std::uint32_t word = 0; word < Lanes; ++word) Expect(readOnly.Words()[word] == Untouched, what + ": the faulting store landed at word " + std::to_string(word));
    }
}

void GrantFault() {
    FaultAmongBoundedWriters(false);
}

void SharedRecordFault() {
    FaultAmongBoundedWriters(true);
}

float CpuTexel(std::uint32_t texture, std::uint32_t texel) {
    return static_cast<float>(0x80000u + texture * 0x1000u + texel) + 0.5f;
}

std::uint32_t RowSeed(std::uint32_t texture, std::uint32_t row) {
    return 0x40000u + texture * 0x400u + row * 0x100u;
}

std::uint32_t Sample(const float* texels) {
    Require(samples < MaxSamples, "batch fault records: more samples than result slots");
    const auto dispatch = samples++;
    auto* srt = Srts.data() + std::size_t{dispatch} * DeviceTable::SrtWords;
    std::fill(srt, srt + DeviceTable::SrtWords, 0u);
    const auto output = DeviceTable::Buffer(Results.data() + std::size_t{dispatch} * DeviceTable::OutputWords, DeviceTable::OutputWords * 4u);
    std::copy(output.begin(), output.end(), srt + DeviceTable::OutputSlot);
    std::copy(DeviceTable::PointClamp.begin(), DeviceTable::PointClamp.end(), srt + DeviceTable::SamplerSlot);
    const auto image = DeviceTable::Image(texels, DeviceTable::Format32Float);
    std::copy(image.begin(), image.end(), srt + DeviceTable::ImageSlot);
    std::fill(Results.begin() + std::size_t{dispatch} * DeviceTable::OutputWords, Results.begin() + std::size_t{dispatch + 1u} * DeviceTable::OutputWords, Untouched);
    const auto address = DrawSubmit::AddressOf(srt);
    const std::array<std::uint32_t, 2> userData{Low(address), High(address)};
    Commands commands;
    commands.Dispatch(SampleProgram(), userData, DeviceTable::Lanes, DeviceTable::Height);
    commands.Submit(GraphicsQueue);
    NewPath::WaitIdle();
    return dispatch;
}

std::string SampleMismatch(std::uint32_t dispatch, const std::array<float, TexelWords>& model) {
    const auto* words = Results.data() + std::size_t{dispatch} * DeviceTable::OutputWords;
    AgcDriver::GuestMemory::FlushGpuWrites(DrawSubmit::AddressOf(words), DeviceTable::OutputWords * 4u);
    for (std::uint32_t row = 0; row < DeviceTable::Height; ++row) {
        for (std::uint32_t x = 0; x < DeviceTable::Lanes; ++x) {
            const auto word = words[(row * DeviceTable::Lanes + x) * DeviceTable::LaneWords];
            const auto expected = model[row * DeviceTable::Width + x];
            if (DeviceTable::SameFloat(word, expected)) continue;
            return "row " + std::to_string(row) + " texel " + std::to_string(x) + " sampled " + Hex(word) + ", expected " + Hex(std::bit_cast<std::uint32_t>(expected));
        }
    }
    return {};
}

void TexturesAfterGpuStores() {
    std::array<float*, Textures> texels{};
    for (auto& block : texels) block = reinterpret_cast<float*>(TestGuest::Allocate(PageBytes, PageBytes));
    std::array<std::array<float, TexelWords>, Textures> model{};
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        for (std::uint32_t texel = 0; texel < TexelWords; ++texel) {
            texels[texture][texel] = CpuTexel(texture, texel);
            model[texture][texel] = CpuTexel(texture, texel);
        }
    }
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        const auto mismatch = SampleMismatch(Sample(texels[texture]), model[texture]);
        Expect(mismatch.empty(), "texture " + std::to_string(texture) + " sampled before any GPU store: " + mismatch);
    }
    for (const auto* block : texels) AgcDriver::GuestMemory::CollectWritesUncached(DrawSubmit::AddressOf(block), PageBytes);
    const auto generation = AgcDriver::GuestMemory::TrackerGeneration();
    Commands commands;
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        for (std::uint32_t row = 0; row < DeviceTable::Height; ++row) {
            const auto buffer = NewPath::BufferResource(DrawSubmit::AddressOf(texels[texture] + row * DeviceTable::Width), DeviceTable::Width * 4u);
            const std::array<std::uint32_t, 5> userData{buffer[0], buffer[1], buffer[2], buffer[3], RowSeed(texture, row)};
            commands.Dispatch(TexelStoreProgram(), userData, DeviceTable::Width);
            for (std::uint32_t x = 0; x < DeviceTable::Width; ++x) model[texture][row * DeviceTable::Width + x] = static_cast<float>(RowSeed(texture, row) + x);
        }
    }
    std::string log;
    const auto failure = Submitted(commands, log);
    const auto what = std::to_string(Textures * DeviceTable::Height) + " dispatches of one batch storing the rows of " + std::to_string(Textures) + " textures through the page table";
    Expect(failure.empty() && Count(log, "deferred write-back failed") == 0u && Count(log, "BDA access failed") == 0u, what + ": the submission reported a failure: " + failure + "\n" + log);
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        const auto address = DrawSubmit::AddressOf(texels[texture]);
        AgcDriver::GuestMemory::FlushGpuWrites(address, PageBytes);
        ExpectWrittenBlock(address, generation, what + ", texture " + std::to_string(texture));
    }
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        const auto mismatch = SampleMismatch(Sample(texels[texture]), model[texture]);
        Expect(mismatch.empty(), what + ": texture " + std::to_string(texture) + " sampled after the batch completed: " + mismatch);
    }
}

std::uint32_t RotationSeed(std::uint32_t build) {
    return 0x7a000000u | (build << 8u);
}

void RecordRotation() {
    constexpr std::uint64_t BuildBytes = std::uint64_t{ChunksPerBuild} * ChunkBytes;
    constexpr std::uint64_t StrideBytes = BuildBytes + ChunkBytes;
    constexpr std::uint64_t SpanBytes = std::uint64_t{RotationBuilds} * StrideBytes;
    PrivatePages region(static_cast<std::uint32_t>((SpanBytes + ChunkBytes) / PageBytes));
    const auto base = (region.Address() + ChunkBytes - 1u) & ~(ChunkBytes - 1u);
    const auto what = std::to_string(RotationBuilds) + " bounded builds of one batch storing to " + std::to_string(ChunksPerBuild) + " distinct 128 KiB chunks each (" + std::to_string(RotationBuilds * ChunksPerBuild) + " chunks, more than one fault record admits), one unused chunk after each build's range";
    Expect(AgcDriver::GuestMemory::Watched(base, SpanBytes), what + ": the committed guest memory is not write-watched");
    AgcDriver::GuestMemory::CollectWritesUncached(base, SpanBytes);
    const auto generation = AgcDriver::GuestMemory::TrackerGeneration();
    Commands commands;
    for (std::uint32_t build = 0; build < RotationBuilds; ++build) {
        const auto buffer = NewPath::BufferResource(base + build * StrideBytes, static_cast<std::uint32_t>(BuildBytes));
        const std::array<std::uint32_t, 5> userData{buffer[0], buffer[1], buffer[2], buffer[3], RotationSeed(build)};
        commands.Dispatch(ChunkStoreProgram(), userData, ChunksPerBuild);
    }
    std::string log;
    const auto failure = Submitted(commands, log);
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    Expect(Count(log, "deferred write-back failed") == 0u, what + ": the batch's completion reported a failure:\n" + log);
    AgcDriver::GuestMemory::FlushGpuWrites(base, SpanBytes);
    const auto* words = reinterpret_cast<const volatile std::uint32_t*>(static_cast<std::uintptr_t>(base));
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t build = 0; build < RotationBuilds; ++build) {
        for (std::uint32_t lane = 0; lane < ChunksPerBuild; ++lane) {
            const auto word = (build * StrideBytes + lane * ChunkBytes) / 4u;
            const auto expected = RotationSeed(build) + lane;
            const auto stored = words[word];
            const auto after = words[word + 1u];
            if (stored == expected && after == 0u) continue;
            if (wrong++ == 0u) first = "build " + std::to_string(build) + " chunk " + std::to_string(lane) + " holds " + Hex(stored) + " then " + Hex(after) + ", expected " + Hex(expected) + " then 0";
        }
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " stores are wrong, first " + first);
    AgcDriver::GuestMemory::CollectWritesUncached(base, SpanBytes);
    const auto blocks = static_cast<std::size_t>(SpanBytes / PageBytes);
    std::vector<std::uint64_t> generations(blocks, generation);
    std::vector<std::uint8_t> changed(blocks);
    std::vector<std::uint8_t> cpu(blocks);
    AgcDriver::GuestMemory::ChangedBlocks(base, SpanBytes, generations, changed, cpu);
    std::size_t unmarked = 0;
    std::size_t spurious = 0;
    std::string firstUnmarked;
    std::string firstSpurious;
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto offset = block * PageBytes;
        const auto within = offset % StrideBytes;
        const auto build = std::to_string(offset / StrideBytes);
        const bool covered = within < BuildBytes;
        const bool stored = covered && within % ChunkBytes == 0u;
        if (stored && (changed[block] != AgcDriver::GuestMemory::BlockWritten || cpu[block] != 0u)) {
            if (unmarked++ == 0u) firstUnmarked = "block " + std::to_string(block) + " (build " + build + ")";
        }
        if (!covered && changed[block] != AgcDriver::GuestMemory::BlockUnchanged) {
            if (spurious++ == 0u) firstSpurious = "block " + std::to_string(block) + " (after build " + build + ")";
        }
    }
    Expect(unmarked == 0u, what + ": " + std::to_string(unmarked) + " of " + std::to_string(RotationBuilds * ChunksPerBuild) + " stored 64 KiB blocks were not marked written by the GPU at completion, first " + firstUnmarked);
    Expect(spurious == 0u, what + ": " + std::to_string(spurious) + " 64 KiB blocks outside every build's write range were marked written, first " + firstSpurious);
}

enum class Mix { Clean, UnboundedFault, InvalidRectangle };

void MixedRecords(Mix mix) {
    static const bool registered = [] {
        DrawSubmit::Register(VertexHeader, BoundedVertex(), DrawSubmit::VertexShaderType);
        DrawSubmit::Register(PixelHeader, RedPixel(), DrawSubmit::PixelShaderType);
        return true;
    }();
    static_cast<void>(registered);
    PrivatePages output(1);
    PrivatePages stored(1);
    PrivatePages readOnly(1, false);
    PrivatePages target(1);
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) {
        output.Words()[word] = Untouched;
        stored.Words()[word] = Untouched;
        target.Words()[word] = 0u;
    }
    for (std::uint32_t word = 0; word < Lanes; ++word) readOnly.Words()[word] = Untouched;
    const VertexBuffer corners(mix == Mix::InvalidRectangle ? DegenerateRectangle : FullRectangle);
    AgcDriver::GuestMemory::CollectWritesUncached(output.Address(), PageBytes);
    AgcDriver::GuestMemory::CollectWritesUncached(stored.Address(), PageBytes);
    const auto generation = AgcDriver::GuestMemory::TrackerGeneration();
    const auto round = 3u + static_cast<std::uint32_t>(mix);
    const auto destination = mix == Mix::UnboundedFault ? readOnly.Address() : stored.Address();
    Commands commands;
    SliceWriter(commands, output.Address(), 0, Seed(round, 0));
    commands.Dispatch(GlobalStoreProgram(), std::array<std::uint32_t, 2>{Low(destination), High(destination)}, Lanes);
    DrawSubmit::Baseline(commands, {target.Data(), TargetWidth, TargetHeight}, DrawSubmit::CenterAndPosition);
    DrawSubmit::Context(commands, 0x3b8, RenderTargetAttrib3);
    DrawSubmit::Context(commands, 0x31b, 0u);
    DrawSubmit::Programs(commands, BoundedVertex(), VertexUserWords, RedPixel(), 0u);
    DrawSubmit::Instances(commands, 1u);
    DrawSubmit::PrimitiveType(commands, RectList);
    DrawSubmit::VertexUserData(commands, corners.Descriptor());
    DrawSubmit::DrawAuto(commands, static_cast<std::uint32_t>(FullRectangle.size()));
    SliceWriter(commands, output.Address(), 1, Seed(round, 1));
    std::string log;
    const auto failure = Submitted(commands, log);
    const auto text = failure + "\n" + log;
    const std::string what = std::string("a bounded writer, an unbounded GLOBAL writer") + (mix == Mix::UnboundedFault ? " storing to a read-only page" : "") + ", a rect-list draw" + (mix == Mix::InvalidRectangle ? " with a degenerate rectangle" : "") + " and a second bounded writer in one batch";
    if (mix != Mix::InvalidRectangle) Expect(failure.empty(), what + ": the submission failed: " + failure);
    Expect(Count(text, "BDA access failed") == (mix == Mix::UnboundedFault ? 1u : 0u), what + ": " + std::to_string(Count(text, "BDA access failed")) + " BDA faults were reported:\n" + text);
    Expect(Count(text, "rect-list requires") == (mix == Mix::InvalidRectangle ? 1u : 0u), what + ": " + std::to_string(Count(text, "rect-list requires")) + " invalid rectangles were reported:\n" + text);
    if (mix == Mix::Clean) Expect(Count(text, "deferred write-back failed") == 0u, what + ": a failure was reported:\n" + text);
    if (mix == Mix::UnboundedFault) {
        const auto report = LineWith(text, "BDA access failed");
        Expect(report.find("reason=2") != std::string::npos && report.find(ProgramTag(GlobalStoreProgram())) != std::string::npos, what + ": the report is not a Permission fault of the GLOBAL writer " + ProgramTag(GlobalStoreProgram()) + ": " + report);
        for (std::uint32_t word = 0; word < Lanes; ++word) Expect(readOnly.Words()[word] == Untouched, what + ": the faulting store landed at word " + std::to_string(word));
    }
    AgcDriver::GuestMemory::FlushGpuWrites(output.Address(), output.Bytes());
    ExpectSlices(output.Words(), 2u, [round](std::uint32_t slot) { return Seed(round, slot); }, what);
    ExpectWrittenBlock(output.Address(), generation, what + ", the bounded writers' page");
    if (mix != Mix::UnboundedFault) {
        AgcDriver::GuestMemory::FlushGpuWrites(stored.Address(), stored.Bytes());
        std::size_t wrong = 0;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) wrong += stored.Words()[lane] != StoredBase + lane ? 1u : 0u;
        Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of the GLOBAL writer's " + std::to_string(Lanes) + " stores are wrong");
        ExpectWrittenBlock(stored.Address(), generation, what + ", the GLOBAL writer's page");
    }
    if (mix != Mix::InvalidRectangle) {
        const auto pixels = DrawSubmit::ReadBack(reinterpret_cast<const std::uint32_t*>(target.Data()), TargetWidth * TargetHeight);
        const auto wrong = static_cast<std::size_t>(std::ranges::count_if(pixels, [](std::uint32_t pixel) { return pixel != Red; }));
        Expect(wrong == 0u, what + ": the rect-list draw left " + std::to_string(wrong) + " of " + std::to_string(pixels.size()) + " pixels not red");
    }
}

void MixedClean() {
    MixedRecords(Mix::Clean);
}

void MixedUnboundedFault() {
    MixedRecords(Mix::UnboundedFault);
}

void MixedInvalidRectangle() {
    MixedRecords(Mix::InvalidRectangle);
}

bool FreedWithin(void* base, std::string& failure) {
    auto done = std::make_shared<std::promise<std::string>>();
    auto result = done->get_future();
    std::thread([base, done] {
        try {
            GuestAllocations::Mutation().Remove(base);
            done->set_value({});
        } catch (const std::exception& error) {
            done->set_value(error.what());
        }
    }).detach();
    if (result.wait_for(FreeLimit) != std::future_status::ready) return false;
    failure = result.get();
    return true;
}

void Free(GuestBlock& block, const std::string& what) {
    std::string failure;
    if (!FreedWithin(block.Base(), failure)) {
        Expect(false, what + ": the guest's free of the allocation did not return within " + std::to_string(FreeLimit.count()) + " s: a build's lease was never released");
        std::fflush(stderr);
        std::_Exit(1);
    }
    Expect(failure.empty(), what + ": the guest's free of the allocation failed: " + failure);
    if (failure.empty()) block.Unregistered();
}

bool LabelPending(std::uint64_t address) {
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr && recorder->LabelPendingAt(address);
}

void LeasesUntilCompletion() {
    {
        GuestBlock leased(1);
        for (std::uint32_t word = 0; word < WordsPerPage; ++word) leased.Words()[word] = Untouched;
        const auto label = leased.Address(LabelWord * 4u);
        Commands commands;
        commands.Dispatch(SpinProgram(), NewPath::BufferResource(leased.Address(SpinWord * 4u), SpinLanes * 4u), SpinLanes);
        for (std::uint32_t writer = 0; writer < Writers; ++writer) SliceWriter(commands, leased.Address(), writer, Seed(6, writer));
        commands.WriteData(label, {LabelValue});
        commands.Submit(GraphicsQueue);
        bool pending = false;
        for (const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60); !pending && std::chrono::steady_clock::now() < deadline;) {
            pending = LabelPending(label);
            if (!pending) std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        const std::string what = "the guest frees an allocation that " + std::to_string(Writers) + " bounded writers and a long dispatch of a recorded batch lease" + (pending ? "" : " (the batch's label was never seen pending)");
        Free(leased, what);
        Expect(!LabelPending(label), what + ": the free returned while the batch's last label was still pending: the leases were released before the batch completed");
        NewPath::WaitIdle();
    }
    {
        GuestBlock leased(1);
        for (std::uint32_t word = 0; word < WordsPerPage; ++word) leased.Words()[word] = Untouched;
        Commands commands;
        for (std::uint32_t writer = 0; writer < Writers; ++writer) SliceWriter(commands, leased.Address(), writer, Seed(7, writer));
        std::string log;
        const auto failure = Submitted(commands, log);
        const auto what = "the guest frees an allocation " + std::to_string(Writers) + " bounded writers leased, after their batch completed";
        Expect(failure.empty(), what + ": the submission failed: " + failure);
        AgcDriver::GuestMemory::FlushGpuWrites(leased.Address(), leased.Bytes());
        ExpectSlices(leased.Words(), Writers, [](std::uint32_t slot) { return Seed(7, slot); }, what);
        NewPath::StderrCapture capture;
        Free(leased, what);
        const auto freed = capture.Finish();
        Expect(Count(freed, "still pinned") == 0u, what + ": the free found the allocation still pinned:\n" + freed);
    }
}

}

#endif

int main() {
#ifdef _WIN32
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    DrawSubmit::Cases({
        {"grant fault among bounded writers", GrantFault},
        {"fault of a build sharing the batch's record", SharedRecordFault},
        {"textures after GPU stores", TexturesAfterGpuStores},
        {"fault record rotation", RecordRotation},
        {"bounded, unbounded and rect-list builds", MixedClean},
        {"unbounded fault beside a rect-list draw", MixedUnboundedFault},
        {"invalid rectangle beside writers", MixedInvalidRectangle},
        {"leases until completion", LeasesUntilCompletion},
    });
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    return DeviceTable::Finish("batch fault record");
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
