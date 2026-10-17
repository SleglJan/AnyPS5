#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <algorithm>
#include <chrono>
#include <span>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using DeviceTable::Expect;
using NewPath::Commands;
using NewPath::Hex;
using NewPath::PageBytes;
using NewPath::PrivatePages;
using NewPath::Program;
using NewPath::Require;
using NewPath::SharedPages;
namespace Graphics = AgcDriver::Graphics;

constexpr std::uint32_t Red = 0xff0000ffu;
constexpr std::uint32_t ElementBytes = 4;
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t CopyLanes = 64;
constexpr std::uint32_t Width = 256;
constexpr std::uint32_t Height = 256;
constexpr Graphics::ColorTileMode Mode = Graphics::ColorTileMode::RenderTarget;

struct Rect {
    std::uint32_t left;
    std::uint32_t top;
    std::uint32_t right;
    std::uint32_t bottom;
};

using Corners = std::array<std::array<float, 4>, 3>;
constexpr Corners FullScreen{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};

DrawSubmit::Header VertexHeader;
DrawSubmit::Header PixelHeader;

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

const Program& CopyProgram() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        Program built;
        built << Vop1(VMovB32, 1, 8)
              << Vop2(VLshlRevB32, 1, InlineInteger(6), 1)
              << Vop2(VAddNcU32, 1, Vgpr(0), 1)
              << Vop2(VLshlRevB32, 1, InlineInteger(2), 1)
              << Mubuf(BufferLoadDword, 2, 1, 0)
              << WaitAll
              << Mubuf(BufferStoreDword, 2, 1, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& LaneStoreProgram() {
    static const Program program = [] {
        using namespace NewPath::Isa;
        Program built;
        built << Vop1(VMovB32, 20, Vgpr(0))
              << Vop1(VMovB32, 30, Vgpr(0))
              << Vop1(VMovB32, 31, InlineInteger(0))
              << ImageStoreX(20, 30, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

struct VertexBuffer {
    PrivatePages pages{1};

    VertexBuffer() { std::memcpy(pages.Data(), FullScreen.data(), sizeof(FullScreen)); }

    std::array<std::uint32_t, VertexUserWords> Descriptor() const {
        const auto address = pages.Address();
        return {NewPath::Low(address), (NewPath::High(address) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(FullScreen.size()), 0x01016facu};
    }
};

std::uint64_t TargetBytes() {
    const auto mips = Graphics::ComputeElementMipLayout(Graphics::ColorTextureTileMode(Mode), ElementBytes, Width, Height, 1);
    return Graphics::ComputeSurfaceSize(mips, 1);
}

std::uint32_t Background(std::uint64_t word, std::uint32_t seed) {
    return (seed << 28u) | static_cast<std::uint32_t>(word & 0x0fffffffu);
}

std::string Submitted(Commands& commands) {
    std::string failure;
    NewPath::StderrCapture capture;
    try {
        commands.Submit(0);
        NewPath::WaitIdle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    const auto log = capture.Finish();
    if (log.find("failed") != std::string::npos) failure += "\n" + log;
    return failure;
}

std::vector<std::uint32_t> RawRead(const volatile std::uint32_t* words, std::uint64_t bytes) {
    std::vector<std::uint32_t> read(static_cast<std::size_t>(bytes / 4u));
    for (std::size_t word = 0; word < read.size(); ++word) read[word] = words[word];
    return read;
}

void Verify(std::span<const std::uint32_t> words, const Rect& rect, std::uint32_t seed, const std::string& what) {
    std::size_t red = 0;
    std::size_t foreign = 0;
    std::string first;
    for (std::size_t word = 0; word < words.size(); ++word) {
        if (words[word] == Red) {
            ++red;
            continue;
        }
        if (words[word] != Background(word, seed) && foreign++ == 0u) first = "byte " + Hex(word * 4u) + " holds " + Hex(words[word]) + ", expected " + Hex(Background(word, seed));
    }
    const auto area = std::uint64_t{rect.right - rect.left} * (rect.bottom - rect.top);
    Expect(red == area, what + ": " + std::to_string(red) + " texels hold the drawn color, expected " + std::to_string(area));
    Expect(foreign == 0u, what + ": " + std::to_string(foreign) + " words are neither their previous bytes nor the drawn color, first " + first);
}

void Prepare() {
    static const bool prepared = [] {
        DrawSubmit::Register(VertexHeader, BoundedVertex(), DrawSubmit::VertexShaderType);
        DrawSubmit::Register(PixelHeader, RedPixel(), DrawSubmit::PixelShaderType);
        return true;
    }();
    static_cast<void>(prepared);
}

void AppendDraw(Commands& commands, std::uint64_t base, const Rect& rect, const VertexBuffer& vertices) {
    DrawSubmit::Baseline(commands, {reinterpret_cast<const void*>(static_cast<std::uintptr_t>(base)), Width, Height}, DrawSubmit::CenterAndPosition);
    DrawSubmit::Context(commands, 0x3b0, ((Width - 1u) << 14u) | (Height - 1u));
    DrawSubmit::Context(commands, 0x3b8, (static_cast<std::uint32_t>(Mode) << 14u) | (1u << 24u) | (1u << 27u));
    DrawSubmit::Context(commands, 0x31b, 0u);
    DrawSubmit::Context(commands, 0x90, 0x80000000u | (rect.top << 16u) | rect.left);
    DrawSubmit::Context(commands, 0x91, (rect.bottom << 16u) | rect.right);
    DrawSubmit::Programs(commands, BoundedVertex(), VertexUserWords, RedPixel(), 0u);
    DrawSubmit::Instances(commands, 1u);
    DrawSubmit::VertexUserData(commands, vertices.Descriptor());
    DrawSubmit::DrawAuto(commands, static_cast<std::uint32_t>(FullScreen.size()));
}

enum class Signal { ReleaseMem, WriteData, FlushThenWriteData };

bool AwaitLabel(const volatile std::uint32_t* label, std::uint32_t value) {
    for (const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10); std::chrono::steady_clock::now() < deadline;) {
        if (*label == value) return true;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
}

template <typename Pages>
void ReadAfterSignal(Pages& image, Signal signal, std::uint32_t seed, std::uint64_t budget, const std::string& what) {
    Prepare();
    NewPath::Locked([&] { NewPath::Residency().Manager().SetBudget(budget); });
    const auto bytes = TargetBytes();
    PrivatePages labels(1);
    const VertexBuffer vertices;
    auto* words = reinterpret_cast<volatile std::uint32_t*>(static_cast<std::uintptr_t>(image.Address()));
    auto* label = reinterpret_cast<volatile std::uint32_t*>(labels.Data());
    *label = 0u;
    {
        Commands warm;
        AppendDraw(warm, image.Address(), {0, 0, 4, 4}, vertices);
        Expect(Submitted(warm).empty(), what + ": the warm-up draw failed");
    }
    std::vector<std::uint32_t> stored(static_cast<std::size_t>(bytes / 4u));
    AgcDriver::GuestMemory::Read(image.Address(), std::as_writable_bytes(std::span(stored)));
    NewPath::Settle();
    for (std::uint64_t word = 0; word < bytes / 4u; ++word) words[word] = Background(word, seed);
    const Rect drawn{30, 40, 190, 170};
    Commands commands;
    AppendDraw(commands, image.Address(), drawn, vertices);
    const auto labelAddress = labels.Address();
    if (signal == Signal::ReleaseMem) {
        commands.Append(0x49u, {0x514u, (1u << 29u), NewPath::Low(labelAddress), NewPath::High(labelAddress), seed, 0u, 0u});
    } else {
        if (signal == Signal::FlushThenWriteData) commands.Append(0x46u, {0x410u});
        commands.WriteData(labelAddress, {seed});
    }
    std::string failure;
    bool signalled = false;
    std::vector<std::uint32_t> read;
    {
        NewPath::StderrCapture capture;
        try {
            commands.Submit(0);
            signalled = AwaitLabel(label, seed);
            if (signalled) read = RawRead(words, bytes);
            NewPath::WaitIdle();
        } catch (const std::exception& error) {
            failure = error.what();
        }
        const auto log = capture.Finish();
        if (log.find("failed") != std::string::npos) failure += "\n" + log;
    }
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    Expect(signalled, what + ": the completion label never reached guest memory");
    if (signalled) Verify(read, signal == Signal::WriteData ? Rect{0, 0, 0, 0} : drawn, seed, what + ", read directly once the label was seen");
}

template <typename Pages>
void StorageImageAfterSignal(Pages& texels, Signal signal, std::uint32_t seed, const std::string& what) {
    constexpr std::uint32_t Lanes = 128;
    constexpr std::uint32_t Sentinel = 0xdeadbeefu;
    constexpr std::uint32_t DepthTile = 24u;
    NewPath::Locked([&] { NewPath::Residency().Manager().SetBudget(0); });
    PrivatePages labels(1);
    auto* label = reinterpret_cast<volatile std::uint32_t*>(labels.Data());
    *label = 0u;
    const std::array<std::uint32_t, 16>* equation = nullptr;
    for (const auto& entry : Graphics::kTextureSwizzleEquations) {
        if (entry.swizzleMode == DepthTile && entry.elementBytes == 4u) equation = &entry.bits;
    }
    Require(equation != nullptr, what + ": no address equation for the 64 KiB Z swizzle");
    std::array<std::uint32_t, Lanes> offsets{};
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        for (std::uint32_t bit = 0; bit < 16u; ++bit) offsets[lane] |= static_cast<std::uint32_t>(std::popcount((*equation)[bit] & (lane & 0xfffu)) & 1) << bit;
    }
    auto* words = reinterpret_cast<volatile std::uint32_t*>(static_cast<std::uintptr_t>(texels.Address()));
    for (std::uint64_t word = 0; word < PageBytes / 4u; ++word) words[word] = Sentinel;
    NewPath::Settle();
    std::array<std::uint32_t, 12> user{};
    auto image = NewPath::ImageResource(texels.Address(), NewPath::Format32Sint, Lanes, 1u);
    image[3] |= DepthTile << 20u;
    std::copy(image.begin(), image.end(), user.begin() + 4);
    Commands commands;
    commands.Dispatch(LaneStoreProgram(), user, Lanes);
    const auto labelAddress = labels.Address();
    if (signal == Signal::ReleaseMem) {
        commands.Append(0x49u, {0x514u, (1u << 29u), NewPath::Low(labelAddress), NewPath::High(labelAddress), seed, 0u, 0u});
    } else {
        commands.Append(0x46u, {0x407u});
        commands.WriteData(labelAddress, {seed});
    }
    std::string failure;
    bool signalled = false;
    std::vector<std::uint32_t> read;
    {
        NewPath::StderrCapture capture;
        try {
            commands.Submit(0);
            signalled = AwaitLabel(label, seed);
            if (signalled) {
                for (std::uint32_t lane = 0; lane < Lanes; ++lane) read.push_back(static_cast<std::uint32_t>(words[offsets[lane] / 4u]));
            }
            NewPath::WaitIdle();
        } catch (const std::exception& error) {
            failure = error.what();
        }
        const auto log = capture.Finish();
        if (log.find("failed") != std::string::npos) failure += "\n" + log;
    }
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    Expect(signalled, what + ": the completion label never reached guest memory");
    if (!signalled) return;
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        if (read[lane] != lane && wrong++ == 0u) first = "lane " + std::to_string(lane) + " holds " + Hex(read[lane]);
    }
    Expect(wrong == 0u, what + ", read directly once the label was seen: " + std::to_string(wrong) + " of " + std::to_string(Lanes) + " texels miss the dispatch's store, first " + first);
}

void StorageImageKeptOnViewPrivate() {
    PrivatePages texels(1);
    StorageImageAfterSignal(texels, Signal::ReleaseMem, 9u, "a depth-layout storage image (kept off the memory path) a dispatch wrote in private memory, then a RELEASE_MEM label");
}

void StorageImageKeptOnViewShared() {
    SharedPages texels(1);
    StorageImageAfterSignal(texels, Signal::ReleaseMem, 10u, "a depth-layout storage image (kept off the memory path) a dispatch wrote in a shared view that is not resident, then a RELEASE_MEM label");
}

void StorageImageAfterFlushPrivate() {
    PrivatePages texels(1);
    StorageImageAfterSignal(texels, Signal::WriteData, 11u, "a depth-layout storage image (kept off the memory path) a dispatch wrote in private memory, then a CS partial flush and a WRITE_DATA label");
}

void PrivateReleaseMem() {
    PrivatePages image(static_cast<std::uint32_t>((TargetBytes() + PageBytes - 1u) / PageBytes));
    ReadAfterSignal(image, Signal::ReleaseMem, 5u, 0u, "a render target in private memory, then a RELEASE_MEM label");
}

void PrivateWriteData() {
    PrivatePages image(static_cast<std::uint32_t>((TargetBytes() + PageBytes - 1u) / PageBytes));
    ReadAfterSignal(image, Signal::WriteData, 6u, 0u, "a render target in private memory, then a bare WRITE_DATA label, which publishes nothing");
}

void PrivateFlushWriteData() {
    PrivatePages image(static_cast<std::uint32_t>((TargetBytes() + PageBytes - 1u) / PageBytes));
    ReadAfterSignal(image, Signal::FlushThenWriteData, 12u, 0u, "a render target in private memory, then a PS partial flush and a WRITE_DATA label");
}

void SharedReleaseMem() {
    SharedPages image(static_cast<std::uint32_t>((TargetBytes() + PageBytes - 1u) / PageBytes));
    ReadAfterSignal(image, Signal::ReleaseMem, 7u, 0u, "a render target in a shared view that is not resident, then a RELEASE_MEM label");
}

void ResidentSharedReleaseMem() {
    SharedPages image(static_cast<std::uint32_t>((TargetBytes() + PageBytes - 1u) / PageBytes));
    ReadAfterSignal(image, Signal::ReleaseMem, 8u, std::uint64_t{1} << 30, "a render target in a resident shared view, guarded, then a RELEASE_MEM label");
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("residency label ordering", [] {
        DrawSubmit::Cases({
            {"private memory, RELEASE_MEM", PrivateReleaseMem},
            {"private memory, bare WRITE_DATA", PrivateWriteData},
            {"private memory, PS partial flush then WRITE_DATA", PrivateFlushWriteData},
            {"shared view, RELEASE_MEM", SharedReleaseMem},
            {"resident shared view, RELEASE_MEM", ResidentSharedReleaseMem},
            {"view-path storage image in private memory, RELEASE_MEM", StorageImageKeptOnViewPrivate},
            {"view-path storage image in a shared view, RELEASE_MEM", StorageImageKeptOnViewShared},
            {"view-path storage image in private memory, CS partial flush then WRITE_DATA", StorageImageAfterFlushPrivate},
        });
    });
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
