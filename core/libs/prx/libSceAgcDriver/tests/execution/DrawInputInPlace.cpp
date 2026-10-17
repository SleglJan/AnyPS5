#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#ifdef _WIN32

namespace {

using namespace DrawSubmit;
using NewPath::PrivatePages;
using NewPath::SharedPages;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 64;
constexpr std::uint32_t Indices = 96;
constexpr std::uint32_t Marks = 4096;
constexpr std::uint32_t Marked = 0x40000000u;
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Stale = 100;
constexpr std::uint32_t Fresh = 2000;
constexpr std::uint64_t IndexOffset = 0x100u;
constexpr auto LabelLimit = std::chrono::seconds(10);

Header VertexHeader;
Header PixelHeader;

constexpr std::size_t ImageIndexWords = 1024;

constexpr std::array<std::uint32_t, ImageIndexWords> MakeImageIndices() {
    std::array<std::uint32_t, ImageIndexWords> words{};
    for (auto& word : words) word = 0xffffu;
    for (std::uint32_t index = 0; index < Indices; ++index) words[IndexOffset / 4u + index] = Fresh + index;
    return words;
}

alignas(4096) const std::array<std::uint32_t, ImageIndexWords> ImageIndices = MakeImageIndices();

const Program& MarkVertex() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        constexpr std::uint32_t VOrB32 = 0x1cu;
        Program built;
        built << Vop2(VLshlRevB32, 4, InlineInteger(2), 5)
              << Vop2(VOrB32, 6, Literal, 5) << Marked
              << Mubuf(BufferStoreDword, 6, 4, 8)
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

const Program& IndexWriter() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        constexpr std::uint32_t VLshlAddU32 = 0x346u;
        Program built;
        built << std::array<std::uint32_t, 2>{0xd4000000u | (VLshlAddU32 << 16u) | 3u, 5u | (InlineInteger(6) << 9u) | (Vgpr(0) << 18u)}
              << Vop2(VLshlRevB32, 1, InlineInteger(2), 3)
              << Vop2(VAddNcU32, 2, 4, 3)
              << Mubuf(BufferStoreDword, 2, 1, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

class Scene {
public:
    Scene(std::uint8_t* indexBase, std::uint64_t indexBytes, bool writable = true) : pixels(1), marks(1), labels(1), indexBase(indexBase), indexBytes(indexBytes), writable(writable) {}

    void Reset(std::uint32_t first) {
        for (std::uint32_t word = 0; word < Width * Height; ++word) pixels.Words()[word] = 0u;
        for (std::uint32_t word = 0; word < Marks; ++word) marks.Words()[word] = 0u;
        for (std::uint32_t word = 0; word < 64u; ++word) labels.Words()[word] = 0u;
        AgcDriver::GuestMemory::BumpCollectEpoch();
        if (!writable) return;
        const std::uint32_t restart = 0xffffu;
        for (std::uint64_t byte = 0; byte < indexBytes; byte += 4u) std::memcpy(indexBase + byte, &restart, 4u);
        for (std::uint32_t index = 0; index < Indices; ++index) {
            const auto value = first + index;
            std::memcpy(indexBase + IndexOffset + std::size_t{index} * 4u, &value, 4u);
        }
        for (std::uint32_t word = 0; word < 64u; ++word) labels.Words()[word] = 0u;
        AgcDriver::GuestMemory::BumpCollectEpoch();
    }

    std::uint64_t IndexAddress() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(indexBase)) + IndexOffset; }

    void WriterDispatch(Commands& commands, std::uint32_t first) const {
        const auto buffer = NewPath::BufferResource(IndexAddress(), Indices * 4u);
        const std::array<std::uint32_t, 5> userData{buffer[0], buffer[1], buffer[2], buffer[3], first};
        commands.Dispatch(IndexWriter(), userData, Lanes, (Indices + Lanes - 1u) / Lanes);
    }

    void Draw(Commands& commands) const {
        Baseline(commands, {pixels.Data(), Width, Height}, CenterAndPosition);
        Programs(commands, MarkVertex(), VertexUserWords, FlatPixel(), 0u);
        const auto buffer = NewPath::BufferResource(marks.Address(), Marks * 4u);
        VertexUserData(commands, buffer);
        DrawSubmit::Instances(commands, 1u);
        IndexType(commands, Index32);
        DrawIndex(commands, IndexAddress(), Indices, Indices);
    }

    std::uint64_t Label() const { return labels.Address(); }

    bool AwaitLabel() const {
        for (const auto deadline = Clock::now() + LabelLimit; Clock::now() < deadline;) {
            if (labels.Words()[0] == 1u) return true;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return false;
    }

    void Check(std::uint32_t first, const std::string& what) const {
        AgcDriver::GuestMemory::FlushGpuWrites(marks.Address(), marks.Bytes());
        std::size_t wrong = 0;
        std::string firstWrong;
        for (std::uint32_t mark = 0; mark < Marks; ++mark) {
            const bool drawn = mark >= first && mark < first + Indices;
            const auto actual = marks.Words()[mark];
            const auto expected = drawn ? (Marked | mark) : 0u;
            if (actual == expected) continue;
            ++wrong;
            if (firstWrong.empty()) firstWrong = "vertex " + std::to_string(mark) + " marked " + Hex(actual) + ", expected " + Hex(expected);
        }
        Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " vertex marks are wrong; first " + firstWrong);
    }

private:
    PrivatePages pixels;
    PrivatePages marks;
    PrivatePages labels;
    std::uint8_t* indexBase;
    std::uint64_t indexBytes;
    bool writable;
};

void Settle(const std::string& what) {
    std::string failure;
    try {
        NewPath::WaitIdle();
        NewPath::Settle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    Expect(failure.empty(), what + ": the submission failed: " + failure);
}

void WrittenInSubmission(Scene& scene, const std::string& memory) {
    const std::string what = "an indexed draw over " + memory + " whose indices a dispatch earlier in its submission wrote";
    scene.Reset(Stale);
    Commands commands;
    scene.WriterDispatch(commands, Fresh);
    scene.Draw(commands);
    commands.WriteData(scene.Label(), {1u});
    commands.Submit(0);
    Expect(scene.AwaitLabel(), what + ": the label never landed");
    Settle(what);
    scene.Check(Fresh, what);
}

void WrittenByCpu(Scene& scene, const std::string& memory) {
    const std::string what = "an indexed draw over " + memory + " whose indices the CPU wrote";
    scene.Reset(Fresh);
    Commands commands;
    scene.Draw(commands);
    commands.WriteData(scene.Label(), {1u});
    commands.Submit(0);
    Expect(scene.AwaitLabel(), what + ": the label never landed");
    Settle(what);
    scene.Check(Fresh, what);
}

void InImageRange() {
    const std::string what = "an indexed draw whose indices lie in a read-only range of the main image";
    MEMORY_BASIC_INFORMATION info{};
    Expect(VirtualQuery(ImageIndices.data(), &info, sizeof(info)) == sizeof(info) && info.Type == MEM_IMAGE && (info.Protect & 0xffu) == PAGE_READONLY, what + ": the indices are not read-only image pages");
    GuestAllocations::Mutation().RegisterMainImage();
    GuestAllocations::Mutation().Protect(ImageIndices.data(), sizeof(ImageIndices), true, false, true, [] {});
    Scene scene(reinterpret_cast<std::uint8_t*>(const_cast<std::uint32_t*>(ImageIndices.data())), sizeof(ImageIndices), false);
    scene.Reset(Fresh);
    const auto mirrors = AgcDriver::Graphics::MirrorCounters();
    Commands commands;
    scene.Draw(commands);
    commands.WriteData(scene.Label(), {1u});
    commands.Submit(0);
    Expect(scene.AwaitLabel(), what + ": the label never landed");
    Settle(what);
    scene.Check(Fresh, what);
    const auto mirrored = AgcDriver::Graphics::MirrorCounters();
    Expect(mirrored.rebuilds > mirrors.rebuilds && mirrored.deviceBytes == mirrored.bytes, what + ": the range was not mirrored in device memory");
    GuestAllocations::Mutation().UnregisterImage(GetModuleHandleW(nullptr));
}

void Run() {
    Register(VertexHeader, MarkVertex(), VertexShaderType);
    Register(PixelHeader, FlatPixel(), PixelShaderType);
    {
        PrivatePages indices(1);
        Scene scene(indices.Data(), indices.Bytes());
        WrittenByCpu(scene, "private pages");
        WrittenInSubmission(scene, "private pages");
    }
    {
        SharedPages indices(1);
        Scene scene(reinterpret_cast<std::uint8_t*>(const_cast<std::uint32_t*>(indices.Words())), indices.Bytes());
        WrittenByCpu(scene, "shared pages");
        WrittenInSubmission(scene, "shared pages");
    }
    InImageRange();
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("draw input in place", Run);
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
