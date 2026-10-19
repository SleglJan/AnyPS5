#include "ResidencyRulesFixture.hpp"
#include "DrawSubmitFixture.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <span>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t Tag = 0x6bu;
constexpr std::uint32_t Pages = 3;
constexpr std::uint32_t Width = 128;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t StrideShift = 11;
constexpr std::uint32_t VCvtU32F32 = 0x07u;
constexpr std::uint32_t VXorB32 = 0x1du;
constexpr std::uint32_t SLoadDwordX2 = 0x01u;
constexpr std::uint32_t PositionX = 2;
constexpr std::uint32_t ColorUint32 = 0x28410u;
constexpr std::uint32_t LinearAttrib3 = (1u << 24u) | (1u << 27u);
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t PixelUserWords = 4;
constexpr std::uint32_t VertexBytes = 16;
constexpr std::uint32_t ManyReads = 1000;

using Corners = std::array<std::array<float, 4>, 3>;
constexpr Corners FullScreen{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};

AgcDriver::Graphics::ResidencyDriverCounts DriverCounts() {
    return Residency().Counts();
}

class GuestProgram {
public:
    GuestProgram& operator<<(std::uint32_t word) {
        code.push_back(word);
        return *this;
    }
    GuestProgram& operator<<(const std::array<std::uint32_t, 2>& pair) { return *this << pair[0] << pair[1]; }

    void Register(std::uint8_t type) {
        storage.assign(code.size() + 64u, 0u);
        std::memcpy(reinterpret_cast<void*>(Address()), code.data(), code.size() * 4u);
        header.shader.file_header = 0x34333231u;
        header.shader.version = 0x18u;
        header.shader.header_size = sizeof(DrawSubmit::Header);
        header.shader.shader_size = static_cast<std::uint32_t>(code.size() * sizeof(std::uint32_t));
        header.shader.code = reinterpret_cast<const std::uint32_t*>(Address());
        header.shader.type = type;
        header.shader.user_data = &header.userData;
        AgcDriverRegisterShader_nid_postfix(&header.shader);
    }

    std::uint64_t Address() const { return (reinterpret_cast<std::uintptr_t>(storage.data()) + 255u) & ~std::uintptr_t{255u}; }

private:
    std::vector<std::uint32_t> code;
    std::vector<std::uint32_t> storage;
    DrawSubmit::Header header;
};

GuestProgram& BoundedVertex() {
    static GuestProgram program = [] {
        using namespace DrawSubmit::Isa;
        GuestProgram built;
        built << Indexed(BufferLoadDwordX4, 10, 5, 8) << WaitAll << Export(PositionTarget, 10, false) << EndProgram;
        return built;
    }();
    return program;
}

void PixelOffset(GuestProgram& built) {
    using namespace DrawSubmit::Isa;
    built << Vop1(VCvtU32F32, 4, Vgpr(PositionX)) << Vop2(VLshlRevB32, 4, InlineInteger(StrideShift), 4);
}

void PixelExport(GuestProgram& built) {
    using namespace DrawSubmit::Isa;
    built << Vop1(VMovB32, 7, InlineInteger(0)) << Vop1(VMovB32, 8, InlineInteger(0)) << Vop1(VMovB32, 9, InlineInteger(0)) << Export(ColorTarget, 6, true) << EndProgram;
}

GuestProgram& BoundedPixel() {
    static GuestProgram program = [] {
        using namespace DrawSubmit::Isa;
        GuestProgram built;
        PixelOffset(built);
        built << Mubuf(BufferLoadDword, 6, 4, 0) << WaitAll;
        PixelExport(built);
        return built;
    }();
    return program;
}

GuestProgram& ManyReadsPixel() {
    static GuestProgram program = [] {
        using namespace DrawSubmit::Isa;
        GuestProgram built;
        PixelOffset(built);
        built << Mubuf(BufferLoadDword, 6, 4, 0) << WaitAll;
        for (std::uint32_t read = 0; read < ManyReads; ++read) built << Mubuf(BufferLoadDword, 10, 4, 0) << WaitAll << Vop2(VXorB32, 6, Vgpr(6), 10);
        for (std::uint32_t read = 0; read < ManyReads; ++read) built << Vop2(VXorB32, 6, Vgpr(6), 10);
        PixelExport(built);
        return built;
    }();
    return program;
}

GuestProgram& LoadedAddressPixel() {
    static GuestProgram program = [] {
        using namespace DrawSubmit::Isa;
        GuestProgram built;
        PixelOffset(built);
        built << Smem(SLoadDwordX2, 20, 0, 0) << WaitAll << Global(GlobalLoadDword, 6, 4, 0, 20) << WaitAll;
        PixelExport(built);
        return built;
    }();
    return program;
}

void Registered() {
    static const bool done = [] {
        BoundedVertex().Register(DrawSubmit::VertexShaderType);
        BoundedPixel().Register(DrawSubmit::PixelShaderType);
        ManyReadsPixel().Register(DrawSubmit::PixelShaderType);
        LoadedAddressPixel().Register(DrawSubmit::PixelShaderType);
        return true;
    }();
    static_cast<void>(done);
}

class DrawReader {
public:
    DrawReader() {
        std::memcpy(const_cast<std::uint32_t*>(vertices.Words()), FullScreen.data(), sizeof(FullScreen));
        AgcDriver::GuestMemory::CollectWritesUncached(vertices.Address(), vertices.Bytes());
        Use(vertices.Address());
    }

    std::vector<std::uint32_t> Read(const GuestProgram& pixel, std::span<const std::uint32_t> pixelUserData) {
        std::fill_n(reinterpret_cast<std::uint32_t*>(target.Data()), Width * Height, 0xdeadbeefu);
        AgcDriver::GuestMemory::CollectWritesUncached(target.Address(), Width * Height * 4u);
        Commands commands;
        DrawSubmit::Baseline(commands, {target.Data(), Width, Height}, DrawSubmit::CenterAndPosition);
        DrawSubmit::Context(commands, 0x31c, ColorUint32);
        DrawSubmit::Context(commands, 0x3b8, LinearAttrib3);
        DrawSubmit::Context(commands, 0x31b, 0u);
        commands.Registers(0x0c8, {Low(BoundedVertex().Address() >> 8u), Low(BoundedVertex().Address() >> 40u)});
        commands.Registers(0x08b, {VertexUserWords << 1u});
        commands.Registers(0x008, {Low(pixel.Address() >> 8u), Low(pixel.Address() >> 40u)});
        commands.Registers(0x00b, {PixelUserWords << 1u});
        DrawSubmit::Instances(commands, 1u);
        const auto address = vertices.Address();
        DrawSubmit::VertexUserData(commands, std::array<std::uint32_t, VertexUserWords>{Low(address), (High(address) & 0xffffu) | (VertexBytes << 16u), static_cast<std::uint32_t>(FullScreen.size()), 0x01016facu});
        DrawSubmit::PixelUserData(commands, pixelUserData);
        DrawSubmit::DrawAuto(commands, static_cast<std::uint32_t>(FullScreen.size()));
        commands.Submit(0);
        WaitIdle();
        return DrawSubmit::ReadBack(reinterpret_cast<const std::uint32_t*>(target.Data()), Width * Height);
    }

private:
    SharedPages vertices{1};
    PrivatePages target{1};
};

void ExpectPattern(const std::vector<std::uint32_t>& pixels, const std::string& what) {
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const std::uint64_t byte = std::uint64_t{x} << StrideShift;
            const auto page = byte / PageBytes;
            const auto wanted = page < Pages ? Pattern(Tag, page, static_cast<std::uint32_t>((byte % PageBytes) / 4u)) : 0u;
            const auto got = pixels[std::size_t{y} * Width + x];
            if (got == wanted) continue;
            Expect(false, what + ": pixel " + std::to_string(x) + "," + std::to_string(y) + " at byte " + Hex(byte) + " read " + Hex(got) + ", expected " + Hex(wanted));
            return;
        }
    }
}

std::array<std::uint32_t, PixelUserWords> Buffer(const SharedPages& pages) {
    return BufferResource(pages.Address(), static_cast<std::uint32_t>(std::uint64_t{Pages} * PageBytes));
}

void SettledDrawsBindTheSegmentStages() {
    DrawReader reader;
    SharedPages pages(Pages);
    Fill(pages.Words(), Pages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    for (std::uint32_t page = 0; page < Pages; ++page) Use(pages.Address(page));
    Settle();
    const auto user = Buffer(pages);

    const auto first = DriverCounts();
    ExpectPattern(reader.Read(BoundedPixel(), user), "pending: the first settled draw's reads across both page edges");
    const auto pending = DriverCounts();
    Expect(pending.settledDispatches == first.settledDispatches + 1u && pending.unsettledDispatches == first.unsettledDispatches, "pending: the first draw over three touching mirror pages was not judged settled (settled +" + std::to_string(pending.settledDispatches - first.settledDispatches) + ", unsettled +" + std::to_string(pending.unsettledDispatches - first.unsettledDispatches) + ")");
    Expect(pending.segmentPending == first.segmentPending + 1u && pending.segmentDispatches == first.segmentDispatches, "pending: the first settled draw before its segment stages exist did not run the walk stages counted as segment pending (pending +" + std::to_string(pending.segmentPending - first.segmentPending) + ", segment +" + std::to_string(pending.segmentDispatches - first.segmentDispatches) + ")");

    auto before = pending;
    auto after = pending;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    do {
        before = DriverCounts();
        const auto pixels = reader.Read(BoundedPixel(), user);
        after = DriverCounts();
        if (after.segmentDispatches != before.segmentDispatches) {
            ExpectPattern(pixels, "segment: the segment stages' reads across both page edges of one mirrored entry");
            break;
        }
        ExpectPattern(pixels, "segment: a settled draw's reads while its segment stages are pending");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while (std::chrono::steady_clock::now() < deadline);
    Expect(after.segmentDispatches == before.segmentDispatches + 1u && after.segmentPending == before.segmentPending, "segment: a settled draw never bound the segment stages of its programs within 60 s");
    Expect(after.segmentFaults == first.segmentFaults, "segment: the segment stages raised an out-of-segment error on reads inside their settled range");
    Settle();
}

void UnsettledDrawsKeepTheWalk() {
    DrawReader reader;
    SharedPages pages(Pages);
    PrivatePages addresses(1);
    Fill(pages.Words(), Pages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    addresses.Words()[0] = Low(pages.Address());
    addresses.Words()[1] = High(pages.Address());
    AgcDriver::GuestMemory::CollectWritesUncached(addresses.Address(), addresses.Bytes());
    for (std::uint32_t page = 0; page < Pages; ++page) Use(pages.Address(page));
    Settle();
    const std::array<std::uint32_t, PixelUserWords> user{Low(addresses.Address()), High(addresses.Address()), 0u, 0u};
    for (std::uint32_t attempt = 0; attempt < 2u; ++attempt) {
        const auto before = DriverCounts();
        const auto pixels = reader.Read(LoadedAddressPixel(), user);
        const auto after = DriverCounts();
        Expect(after.unsettledDispatches == before.unsettledDispatches + 1u && after.settledDispatches == before.settledDispatches, "unsettled: a draw that reads memory through an address it loads was not counted unsettled (attempt " + std::to_string(attempt) + ")");
        Expect(after.segmentDispatches == before.segmentDispatches && after.segmentPending == before.segmentPending && after.segmentRefused == before.segmentRefused, "unsettled: a draw that reads memory through an address it loads left the walk stages (attempt " + std::to_string(attempt) + ")");
        const auto wanted = pixels[0] == pages.Words()[0];
        Expect(wanted, "unsettled: the walk stages' read through the loaded address returned " + Hex(pixels[0]) + ", expected " + Hex(pages.Words()[0]));
    }
    Settle();
}

void ManyReadsRefusedByInlinedSize() {
    DrawReader reader;
    SharedPages pages(Pages);
    Fill(pages.Words(), Pages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    for (std::uint32_t page = 0; page < Pages; ++page) Use(pages.Address(page));
    Settle();
    const auto user = Buffer(pages);
    const auto first = DriverCounts();
    auto before = first;
    auto after = first;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    do {
        before = DriverCounts();
        static_cast<void>(reader.Read(ManyReadsPixel(), user));
        after = DriverCounts();
        Expect(after.settledDispatches == before.settledDispatches + 1u, "many reads: the draw over three touching mirror pages was not judged settled");
        if (after.segmentDispatches != before.segmentDispatches || after.segmentRefused != before.segmentRefused) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while (std::chrono::steady_clock::now() < deadline);
    const auto segment = after.segmentDispatches - first.segmentDispatches;
    const auto refused = after.segmentRefused - first.segmentRefused;
    Expect(refused == 1u && segment == 0u, "many reads: a settled draw whose pixel stage is over the inlined size bound was not kept on the walk stages counted as segment refused: size within 120 s (segment +" + std::to_string(segment) + ", refused +" + std::to_string(refused) + ", pending +" + std::to_string(after.segmentPending - first.segmentPending) + ")");
    Settle();
}

void Run(MirrorReader&) {
    Registered();
    SettledDrawsBindTheSegmentStages();
    ManyReadsRefusedByInlinedSize();
    UnsettledDrawsKeepTheWalk();
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("segment draw selection", Run);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
