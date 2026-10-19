#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <algorithm>
#include <span>
#include <string>
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

void Draw(std::uint64_t base, const Rect& rect, const VertexBuffer& vertices, const std::string& what) {
    Commands commands;
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
    const auto failure = Submitted(commands);
    Expect(failure.empty(), what + ": the draw failed: " + failure);
}

std::vector<std::uint32_t> GpuRead(std::uint64_t base, std::uint64_t bytes, const PrivatePages& output, const std::string& what) {
    const auto words = static_cast<std::uint32_t>(bytes / 4u);
    const auto image = NewPath::BufferResource(base, static_cast<std::uint32_t>(bytes));
    const auto copy = NewPath::BufferResource(output.Address(), static_cast<std::uint32_t>(bytes));
    const std::array<std::uint32_t, 8> userData{image[0], image[1], image[2], image[3], copy[0], copy[1], copy[2], copy[3]};
    Commands commands;
    commands.Dispatch(CopyProgram(), userData, CopyLanes, (words + CopyLanes - 1u) / CopyLanes);
    const auto failure = Submitted(commands);
    Expect(failure.empty(), what + ": the dispatch reading the target's memory failed: " + failure);
    AgcDriver::GuestMemory::FlushGpuWrites(output.Address(), bytes);
    std::vector<std::uint32_t> read(words);
    std::memcpy(read.data(), output.Data(), bytes);
    return read;
}

std::vector<std::uint32_t> RawRead(const volatile std::uint32_t* words, std::uint64_t bytes) {
    std::vector<std::uint32_t> read(static_cast<std::size_t>(bytes / 4u));
    for (std::size_t word = 0; word < read.size(); ++word) read[word] = words[word];
    return read;
}

std::vector<std::uint32_t> DriverRead(std::uint64_t base, std::uint64_t bytes) {
    std::vector<std::uint32_t> read(static_cast<std::size_t>(bytes / 4u));
    AgcDriver::GuestMemory::Read(base, std::as_writable_bytes(std::span(read)));
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

std::uint32_t OwnedPages(std::uint64_t base, std::uint32_t pages) {
    std::uint32_t owned = 0;
    NewPath::Locked([&] {
        for (std::uint32_t page = 0; page < pages; ++page) owned += NewPath::Residency().Manager().OwnerOf(NewPath::PageOf(base) + page) != nullptr ? 1u : 0u;
    });
    return owned;
}

Graphics::ResidencyCounts Counts() {
    Graphics::ResidencyCounts counts{};
    NewPath::Locked([&] { counts = NewPath::Residency().Manager().Counts(); });
    return counts;
}

void Prepare() {
    static const bool prepared = [] {
        DrawSubmit::Register(VertexHeader, BoundedVertex(), DrawSubmit::VertexShaderType);
        DrawSubmit::Register(PixelHeader, RedPixel(), DrawSubmit::PixelShaderType);
        NewPath::Locked([] { NewPath::Residency().Manager().SetBudget(std::uint64_t{1} << 30); });
        return true;
    }();
    static_cast<void>(prepared);
}

void SharedTarget() {
    Prepare();
    const std::string what = "a render target in a shared view";
    const auto bytes = TargetBytes();
    const auto pages = static_cast<std::uint32_t>((bytes + PageBytes - 1u) / PageBytes);
    SharedPages image(pages);
    PrivatePages output(pages);
    const VertexBuffer vertices;
    const auto base = image.Address();
    for (std::uint64_t word = 0; word < bytes / 4u; ++word) image.Words()[word] = Background(word, 1);
    const Rect first{40, 30, 200, 100};
    const auto registry = Graphics::StorageTexture::RegistryStores();
    const auto before = Counts();
    Draw(base, first, vertices, what);
    const auto drawn = Counts();
    const auto owned = OwnedPages(base, pages);
    Expect(owned == pages && drawn.stalePages != 0u && drawn.retiles == before.retiles, what + ": after the draw the target owns " + std::to_string(owned) + " of its " + std::to_string(pages) + " pages, " + std::to_string(drawn.stalePages) + " pages are stale, after " + std::to_string(drawn.retiles - before.retiles) + " retiles");
    Expect(drawn.accessGuarded != 0u, what + ": no stale page of the resident shared view is access-guarded");
    Verify(RawRead(image.Words(), bytes), first, 1, what + ", read by the CPU after the draw");
    const auto read = Counts();
    Expect(read.retiles > drawn.retiles && read.stalePages == 0u, what + ": the CPU read retiled " + std::to_string(read.retiles - drawn.retiles) + " times and left " + std::to_string(read.stalePages) + " pages stale");
    for (std::uint64_t word = 0; word < bytes / 4u; ++word) image.Words()[word] = Background(word, 2);
    const Rect second{100, 150, 230, 250};
    Draw(base, second, vertices, what + ", second draw");
    Verify(GpuRead(base, bytes, output, what), second, 2, what + ", read by an address-based dispatch after the second draw");
    Verify(RawRead(image.Words(), bytes), second, 2, what + ", read by the CPU after the second draw");
    Expect(Graphics::StorageTexture::RegistryStores() == registry, what + ": " + std::to_string(Graphics::StorageTexture::RegistryStores() - registry) + " images were stored from the pending registry");
}

void PageReadThenDraw() {
    Prepare();
    const std::string what = "a render target in a shared view read on one page between two draws";
    const auto bytes = TargetBytes();
    const auto pages = static_cast<std::uint32_t>((bytes + PageBytes - 1u) / PageBytes);
    Require(pages >= 2u, what + ": the target needs at least two pages");
    SharedPages image(pages);
    const VertexBuffer vertices;
    const auto base = image.Address();
    for (std::uint64_t word = 0; word < bytes / 4u; ++word) image.Words()[word] = Background(word, 5);
    const Rect whole{0, 0, Width, Height};
    const auto before = Counts();
    Draw(base, whole, vertices, what);
    const auto drawn = Counts();
    Expect(drawn.stalePages - before.stalePages == pages, what + ": after the first draw " + std::to_string(drawn.stalePages - before.stalePages) + " of " + std::to_string(pages) + " pages are stale");
    const volatile std::uint32_t* words = image.Words();
    Expect(words[0] == Red, what + ": the CPU read of the first page saw " + Hex(words[0]));
    const auto read = Counts();
    Expect(read.stalePages > before.stalePages && read.stalePages < drawn.stalePages, what + ": the CPU read of one page left " + std::to_string(read.stalePages - before.stalePages) + " of " + std::to_string(pages) + " pages stale");
    Draw(base, whole, vertices, what + ", second draw");
    const auto redrawn = Counts();
    Expect(redrawn.retiles == read.retiles, what + ": the second draw retiled " + std::to_string(redrawn.retiles - read.retiles) + " times; only its own write-back changed the target's memory");
    Expect(redrawn.stalePages - before.stalePages == pages, what + ": after the second draw " + std::to_string(redrawn.stalePages - before.stalePages) + " of " + std::to_string(pages) + " pages are stale");
    Verify(RawRead(image.Words(), bytes), whole, 5, what + ", read by the CPU after the second draw");
}

void PrivateTarget() {
    Prepare();
    const std::string what = "a render target in private memory";
    const auto bytes = TargetBytes();
    const auto pages = static_cast<std::uint32_t>((bytes + PageBytes - 1u) / PageBytes);
    PrivatePages image(pages);
    PrivatePages output(pages);
    const VertexBuffer vertices;
    const auto base = image.Address();
    auto* words = reinterpret_cast<volatile std::uint32_t*>(static_cast<std::uintptr_t>(base));
    for (std::uint64_t word = 0; word < bytes / 4u; ++word) words[word] = Background(word, 3);
    const Rect first{10, 20, 120, 140};
    const auto registry = Graphics::StorageTexture::RegistryStores();
    const auto before = Counts();
    Draw(base, first, vertices, what);
    const auto drawn = Counts();
    const auto owned = OwnedPages(base, pages);
    Expect(owned == pages && drawn.stalePages != 0u && drawn.retiles == before.retiles, what + ": after the draw the target owns " + std::to_string(owned) + " of its " + std::to_string(pages) + " pages, " + std::to_string(drawn.stalePages) + " pages are stale, after " + std::to_string(drawn.retiles - before.retiles) + " retiles");
    Verify(DriverRead(base, bytes), first, 3, what + ", read by the driver after the draw");
    const auto read = Counts();
    Expect(read.retiles > drawn.retiles && read.stalePages < drawn.stalePages, what + ": the driver read retiled " + std::to_string(read.retiles - drawn.retiles) + " times, " + std::to_string(read.stalePages) + " pages stay stale");
    for (std::uint64_t word = 0; word < bytes / 4u; ++word) words[word] = Background(word, 4);
    const Rect second{60, 80, 250, 200};
    Draw(base, second, vertices, what + ", second draw");
    Verify(GpuRead(base, bytes, output, what), second, 4, what + ", read by an address-based dispatch after the second draw");
    Expect(Graphics::StorageTexture::RegistryStores() == registry, what + ": " + std::to_string(Graphics::StorageTexture::RegistryStores() - registry) + " images were stored from the pending registry");
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("residency owned target", [] {
        DrawSubmit::Cases({
            {"shared view", SharedTarget},
            {"private memory", PrivateTarget},
            {"one page read between draws", PageReadThenDraw},
        });
    });
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
