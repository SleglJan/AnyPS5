#include "DeviceTableFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "StorageImageMemory.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <string>
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
namespace Graphics = AgcDriver::Graphics;
using Graphics::StorageTexture;

constexpr std::uint32_t Side = 256;
constexpr std::uint32_t Type2DArray = 13;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Left = 96;
constexpr std::uint32_t Top = 40;
constexpr std::uint32_t Rows = 48;
constexpr std::uint32_t InputSlot = 8;
constexpr std::uint32_t RectMark = 0xa0000000u;
constexpr std::uint32_t MipMark = 0xb0000000u;
constexpr std::uint32_t Standard256B = 0x01u;
constexpr std::uint32_t Standard64KB = 0x09u;
constexpr std::uint32_t RenderTarget64KB = 0x1bu;
constexpr std::uint32_t MipLevels = 9;
constexpr std::uint32_t MipGroups = 2;
constexpr std::uint32_t MipEntries = MipGroups * Lanes;
constexpr std::uint32_t CopyLanes = 64;

namespace Ops {

using namespace DeviceTable::Asm;

constexpr std::uint32_t SLoadDwordX8 = 0x03u;
constexpr std::uint32_t BufferLoadDwordX4 = 0x0eu;
constexpr std::uint32_t VOrB32 = 0x1cu;
constexpr std::uint32_t ImageStoreMip = 0x09u;

constexpr std::array<std::uint32_t, 2> Store(std::uint32_t opcode, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t resource, bool array) {
    return {0xf0000000u | (opcode << 18u) | 0x1000u | (1u << 8u) | ((array ? 5u : 1u) << 3u), ((resource / 4u) << 16u) | (vdata << 8u) | vaddr};
}

}

enum class Binding { Srt, UserData };

const char* NameOf(Binding binding) {
    return binding == Binding::Srt ? "a T# the program loads from memory" : "a T# in the dispatch's user data";
}

struct Image {
    std::uint32_t swizzle;
    std::uint32_t layers;
    std::uint32_t levels;
};

struct Surface {
    Image image;
    PrivatePages pages;
    std::uint64_t base;
    Graphics::SurfaceGeometry geometry;

    static std::uint64_t BytesOf(const Image& image);

    explicit Surface(const Image& shape) : image(shape), pages(static_cast<std::uint32_t>((BytesOf(shape) + PageBytes - 1u) / PageBytes)), base(pages.Address()) {
        geometry = Graphics::DescribeSurface(Graphics::DecodeTextureResource(Descriptor()));
        Require(geometry.guestBytes <= pages.Bytes(), "storage store units: the surface does not fit its pages");
    }

    std::array<std::uint32_t, 8> Descriptor() const { return DescriptorAt(base, image); }

    static std::array<std::uint32_t, 8> DescriptorAt(std::uint64_t address, const Image& image) {
        return {
            Low(address >> 8u),
            (Low(address >> 40u) & 0xffu) | (DeviceTable::Format32UInt << 20u) | (((Side - 1u) & 3u) << 30u),
            ((Side - 1u) >> 2u) | ((Side - 1u) << 14u) | (1u << 31u),
            0xfacu | ((image.levels - 1u) << 16u) | (image.swizzle << 20u) | ((image.layers > 1u ? Type2DArray : DeviceTable::Type2D) << 28u),
            image.layers - 1u,
            (image.levels - 1u) << 4u,
            0u,
            0u,
        };
    }

    std::uint64_t Bytes() const { return geometry.guestBytes; }
    std::uint64_t LayerBytes() const { return geometry.layerBytes; }
};

std::uint64_t Surface::BytesOf(const Image& image) {
    return Graphics::DescribeSurface(Graphics::DecodeTextureResource(DescriptorAt(PageBytes, image))).guestBytes;
}

std::uint32_t Background(std::uint64_t word, std::uint32_t seed) {
    return (seed << 28u) | static_cast<std::uint32_t>(word & 0x0fffffffu);
}

void Fill(const Surface& surface, std::uint32_t seed) {
    auto* words = reinterpret_cast<volatile std::uint32_t*>(static_cast<std::uintptr_t>(surface.base));
    for (std::uint64_t word = 0; word < surface.Bytes() / 4u; ++word) words[word] = Background(word, seed);
    AgcDriver::GuestMemory::BumpCollectEpoch();
}

Program& NextProgram() {
    static std::array<Program, 8> programs;
    static std::size_t used = 0;
    Require(used < programs.size(), "storage store units: more programs than the test keeps");
    return programs[used++];
}

const Program& RectProgram(Binding binding, bool array, std::uint32_t slice) {
    using namespace Ops;
    auto& built = NextProgram();
    const bool table = binding == Binding::Srt;
    const std::uint32_t resource = table ? 36u : 0u;
    const std::uint32_t group = table ? 2u : 8u;
    if (table) built << Smem(SLoadDwordX8, 36, 0, DeviceTable::ImageSlot * 4u) << WaitAll;
    built << Vop2(VAddNcU32, 1, Literal, 0) << Left
          << Vop1(VMovB32, 2, group)
          << Vop2(VAddNcU32, 2, Literal, 2) << Top;
    if (array) built << Vop1(VMovB32, 3, InlineInteger(slice));
    built << Vop2(VLshlRevB32, 4, InlineInteger(12), 2)
          << Vop2(VOrB32, 4, Vgpr(1), 4)
          << Vop2(VOrB32, 4, Literal, 4) << RectMark
          << Store(ImageStore, 4, 1, resource, array)
          << EndProgram;
    return built;
}

const Program& MipProgram(Binding binding) {
    using namespace Ops;
    auto& built = NextProgram();
    const bool table = binding == Binding::Srt;
    const std::uint32_t resource = table ? 36u : 0u;
    const std::uint32_t input = table ? 24u : 8u;
    const std::uint32_t group = table ? 2u : 12u;
    if (table) built << Smem(SLoadDwordX8, 36, 0, DeviceTable::ImageSlot * 4u) << Smem(SLoadDwordX4, 24, 0, InputSlot * 4u) << WaitAll;
    built << Vop1(VMovB32, 9, group)
          << Vop2(VLshlRevB32, 9, InlineInteger(6), 9)
          << Vop2(VAddNcU32, 9, Vgpr(0), 9)
          << Vop2(VLshlRevB32, 9, InlineInteger(4), 9)
          << Mubuf(BufferLoadDwordX4, 1, 9, input)
          << WaitAll
          << Store(ImageStoreMip, 4, 1, resource, false)
          << EndProgram;
    return built;
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

TestGuest::Array<std::uint32_t, DeviceTable::SrtWords> StoreSrt;

std::vector<std::uint32_t> UserData(Binding binding, const std::array<std::uint32_t, 8>& image, const std::array<std::uint32_t, 4>* input) {
    if (binding == Binding::UserData) {
        std::vector<std::uint32_t> words(image.begin(), image.end());
        if (input != nullptr) words.insert(words.end(), input->begin(), input->end());
        return words;
    }
    std::fill(StoreSrt.begin(), StoreSrt.end(), 0u);
    std::copy(image.begin(), image.end(), StoreSrt.begin() + DeviceTable::ImageSlot);
    if (input != nullptr) std::copy(input->begin(), input->end(), StoreSrt.begin() + InputSlot);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    const auto srt = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(StoreSrt.data()));
    return {Low(srt), High(srt)};
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
    if (log.find("failed") != std::string::npos || log.find("rejected") != std::string::npos) failure += "\n" + log;
    return failure;
}

std::vector<std::uint32_t> GpuRead(std::uint64_t address, std::uint64_t bytes, const PrivatePages& output, const std::string& what) {
    const auto words = static_cast<std::uint32_t>(bytes / 4u);
    const auto source = NewPath::BufferResource(address, static_cast<std::uint32_t>(bytes));
    const auto copy = NewPath::BufferResource(output.Address(), static_cast<std::uint32_t>(bytes));
    const std::array<std::uint32_t, 8> userData{source[0], source[1], source[2], source[3], copy[0], copy[1], copy[2], copy[3]};
    Commands commands;
    commands.Dispatch(CopyProgram(), userData, CopyLanes, (words + CopyLanes - 1u) / CopyLanes);
    const auto failure = Submitted(commands);
    Expect(failure.empty(), what + ": the buffer read of the image's memory failed: " + failure);
    AgcDriver::GuestMemory::FlushGpuWrites(output.Address(), bytes);
    std::vector<std::uint32_t> read(words);
    std::memcpy(read.data(), output.Data(), bytes);
    return read;
}

std::vector<std::uint32_t> CpuRead(std::uint64_t address, std::uint64_t bytes) {
    std::vector<std::uint32_t> read(static_cast<std::size_t>(bytes / 4u));
    AgcDriver::GuestMemory::Read(address, std::as_writable_bytes(std::span(read)));
    return read;
}

std::uint32_t UnitOf(const Surface& surface, std::uint32_t layer, std::uint32_t x, std::uint32_t y) {
    const auto layerUnits = static_cast<std::uint32_t>(surface.LayerBytes() / PageBytes);
    if (surface.image.swizzle == Standard256B) return layer * layerUnits + y * Side * 4u / static_cast<std::uint32_t>(PageBytes);
    return layer * layerUnits + (y / 128u) * (Side / 128u) + x / 128u;
}

std::uint32_t RectValue(std::uint32_t x, std::uint32_t y) {
    return RectMark | (y << 12u) | x;
}

void VerifyRect(std::span<const std::uint32_t> words, std::uint64_t firstWord, const Surface& surface, std::uint32_t slice, std::uint32_t seed, const std::string& what) {
    std::set<std::uint32_t> expected;
    const auto begin = firstWord * 4u;
    const auto end = begin + words.size() * 4u;
    for (std::uint32_t y = Top; y < Top + Rows; ++y) {
        for (std::uint32_t x = Left; x < Left + Lanes; ++x) {
            const auto unit = std::uint64_t{UnitOf(surface, slice, x, y)} * PageBytes;
            if (unit < end && unit + PageBytes > begin) expected.insert(RectValue(x, y));
        }
    }
    std::map<std::uint32_t, std::size_t> seen;
    std::size_t foreign = 0;
    std::size_t misplaced = 0;
    std::string first;
    for (std::size_t index = 0; index < words.size(); ++index) {
        const auto word = firstWord + index;
        const auto value = words[index];
        if (value == Background(word, seed)) continue;
        const auto unit = static_cast<std::uint32_t>(word * 4u / PageBytes);
        if (!expected.contains(value)) {
            if (foreign++ == 0u && first.empty()) first = "byte " + Hex(word * 4u) + " of unit " + std::to_string(unit) + " holds " + Hex(value) + ", expected its previous " + Hex(Background(word, seed));
            continue;
        }
        ++seen[value];
        const auto x = value & 0xfffu;
        const auto y = (value >> 12u) & 0xfffu;
        if (UnitOf(surface, slice, x, y) != unit && misplaced++ == 0u && first.empty()) first = "the store of texel (" + std::to_string(x) + ", " + std::to_string(y) + ") lies in unit " + std::to_string(unit);
    }
    std::size_t missing = 0;
    std::size_t repeated = 0;
    for (const auto value : expected) {
        const auto found = seen.find(value);
        if (found == seen.end()) {
            if (missing++ == 0u && first.empty()) first = "texel (" + std::to_string(value & 0xfffu) + ", " + std::to_string((value >> 12u) & 0xfffu) + ") was not stored";
        } else if (found->second != 1u) {
            ++repeated;
        }
    }
    Expect(foreign == 0u && misplaced == 0u && missing == 0u && repeated == 0u, what + ": " + std::to_string(foreign) + " words are neither their previous bytes nor a stored texel, " + std::to_string(misplaced) + " stored texels lie in another unit, " + std::to_string(missing) + " of " + std::to_string(expected.size()) + " stored texels are missing and " + std::to_string(repeated) + " appear twice; first " + first);
}

struct RectCase {
    const char* name;
    Binding binding;
    Image image;
    std::uint32_t slice;
    bool pendingByUnit;
};

void RunRect(const RectCase& rectCase, const Program& program, std::uint32_t seed) {
    const Surface surface(rectCase.image);
    PrivatePages output(static_cast<std::uint32_t>((surface.Bytes() + PageBytes - 1u) / PageBytes));
    Fill(surface, seed);
    const auto what = std::string(rectCase.name) + ", through " + NameOf(rectCase.binding);
    {
        const auto userData = UserData(rectCase.binding, surface.Descriptor(), nullptr);
        Commands commands;
        commands.Dispatch(program, userData, Lanes, Rows);
        const auto failure = Submitted(commands);
        Expect(failure.empty(), what + ": the store failed: " + failure);
    }
    NewPath::Settle();
    if (rectCase.pendingByUnit) {
        const bool memory = ShaderRecompiler::StorageImageMemory::SurfaceThroughMemory(surface.Descriptor(), ShaderRecompiler::StorageImageMemory::Access::Store);
        if (memory) Expect(StorageTexture::FindPending(surface.base, surface.Bytes()) == nullptr, what + ": a store through memory left an image with pending results over the surface");
        else Expect(StorageTexture::FindPending(surface.base, surface.Bytes()) != nullptr, what + ": the store left no image with pending results over the surface");
        const auto layerBytes = surface.LayerBytes();
        const auto flush = [&](std::uint64_t offset, std::uint64_t bytes) {
            bool stored = false;
            NewPath::Locked([&] { stored = StorageTexture::FlushPending(surface.base + offset, static_cast<std::size_t>(bytes), nullptr, "storage store unit test"); });
            NewPath::Settle();
            return stored;
        };
        struct Unwritten {
            std::uint64_t offset;
            std::uint64_t bytes;
            const char* name;
        };
        const std::array<Unwritten, 2> reads{{
            {std::uint64_t{rectCase.slice} * layerBytes + layerBytes / 2u, layerBytes / 2u, "the units of the stored layer below the rectangle"},
            {std::uint64_t{1u - rectCase.slice} * layerBytes, layerBytes, "the other layer"},
        }};
        for (const auto& read : reads) {
            if (!memory) Expect(!flush(read.offset, read.bytes), what + ": a flush of " + read.name + ", which the store did not write, stored the image: the store left units it did not write pending");
            VerifyRect(GpuRead(surface.base + read.offset, read.bytes, output, what), read.offset / 4u, surface, rectCase.slice, seed, what + ", " + read.name + " read alone by a buffer dispatch");
        }
        if (!memory) Expect(flush(std::uint64_t{rectCase.slice} * layerBytes, layerBytes / 2u), what + ": a flush of the units the store wrote stored nothing");
    }
    const auto all = GpuRead(surface.base, surface.Bytes(), output, what);
    VerifyRect(all, 0u, surface, rectCase.slice, seed, what + ", read by a buffer dispatch");
    VerifyRect(CpuRead(surface.base, surface.Bytes()), 0u, surface, rectCase.slice, seed, what + ", read by the CPU");
}

struct MipEntry {
    std::uint32_t mip;
    std::uint32_t x;
    std::uint32_t y;
};

std::uint32_t MipValue(const MipEntry& entry) {
    return MipMark | (entry.mip << 24u) | (entry.y << 12u) | entry.x;
}

std::vector<MipEntry> MipStores() {
    std::vector<MipEntry> entries;
    std::set<std::uint32_t> unique;
    const auto add = [&](std::uint32_t mip, std::uint32_t x, std::uint32_t y) {
        const MipEntry entry{mip, x, y};
        if (unique.insert(MipValue(entry)).second) entries.push_back(entry);
    };
    for (std::uint32_t index = 0; index < 32u; ++index) add(0u, (index * 37u + 5u) % Side, (index * 53u + 11u) % Side);
    for (std::uint32_t index = 0; index < 16u; ++index) add(1u, (index * 29u + 3u) % (Side / 2u), (index * 41u + 7u) % (Side / 2u));
    for (std::uint32_t mip = 2; mip < MipLevels; ++mip) {
        const auto side = Side >> mip;
        for (std::uint32_t index = 0; index < std::min(side * side, 10u); ++index) add(mip, (index * 7u + 1u) % side, (index * 5u + 2u) % side);
    }
    Require(entries.size() <= MipEntries, "storage store units: more mip stores than lanes");
    return entries;
}

void VerifyMips(std::span<const std::uint32_t> words, const Surface& surface, const std::vector<MipEntry>& entries, std::uint32_t seed, const std::string& what) {
    std::map<std::uint32_t, MipEntry> expected;
    for (const auto& entry : entries) expected.emplace(MipValue(entry), entry);
    std::map<std::uint32_t, std::size_t> seen;
    std::size_t foreign = 0;
    std::size_t misplaced = 0;
    std::string first;
    for (std::size_t word = 0; word < words.size(); ++word) {
        const auto value = words[word];
        if (value == Background(word, seed)) continue;
        const auto found = expected.find(value);
        if (found == expected.end()) {
            if (foreign++ == 0u && first.empty()) first = "byte " + Hex(word * 4u) + " holds " + Hex(value) + ", expected its previous " + Hex(Background(word, seed));
            continue;
        }
        ++seen[value];
        const auto& mip = surface.geometry.mips.at(found->second.mip);
        const auto begin = mip.tail ? mip.tiledOffset / PageBytes * PageBytes : mip.tiledOffset;
        const auto end = mip.tail ? begin + PageBytes : mip.tiledOffset + mip.tiledSize;
        const auto offset = std::uint64_t{word} * 4u;
        if ((offset < begin || offset >= end) && misplaced++ == 0u && first.empty()) first = "the store into mip " + std::to_string(found->second.mip) + " lies at byte " + Hex(offset) + ", outside the mip's bytes";
    }
    std::size_t missing = 0;
    for (const auto& [value, entry] : expected) {
        if (seen.contains(value)) continue;
        if (missing++ == 0u && first.empty()) first = "texel (" + std::to_string(entry.x) + ", " + std::to_string(entry.y) + ") of mip " + std::to_string(entry.mip) + " was not stored";
    }
    Expect(foreign == 0u && misplaced == 0u && missing == 0u, what + ": " + std::to_string(foreign) + " words are neither their previous bytes nor a stored texel, " + std::to_string(misplaced) + " stored texels lie outside their mip and " + std::to_string(missing) + " of " + std::to_string(expected.size()) + " stored texels are missing; first " + first);
}

void RunMips(Binding binding, const Program& program, std::uint32_t seed) {
    const Surface surface(Image{Standard64KB, 1u, MipLevels});
    PrivatePages output(static_cast<std::uint32_t>((surface.Bytes() + PageBytes - 1u) / PageBytes));
    PrivatePages input(1);
    const auto entries = MipStores();
    for (std::uint32_t lane = 0; lane < MipEntries; ++lane) {
        const auto& entry = entries[lane % entries.size()];
        input.Words()[lane * 4u + 0u] = entry.x;
        input.Words()[lane * 4u + 1u] = entry.y;
        input.Words()[lane * 4u + 2u] = entry.mip;
        input.Words()[lane * 4u + 3u] = MipValue(entry);
    }
    Fill(surface, seed);
    const auto what = std::string("stores into every mip of a 64 KiB tiled 256x256 chain of 9 mips, the mip chosen per lane, through ") + NameOf(binding);
    const auto inputBuffer = NewPath::BufferResource(input.Address(), MipEntries * 16u);
    const auto userData = UserData(binding, surface.Descriptor(), &inputBuffer);
    Commands commands;
    commands.Dispatch(program, userData, Lanes, MipGroups);
    const auto failure = Submitted(commands);
    Expect(failure.empty(), what + ": the store failed: " + failure);
    NewPath::Settle();
    VerifyMips(GpuRead(surface.base, surface.Bytes(), output, what), surface, entries, seed, what + ", read by a buffer dispatch");
    VerifyMips(CpuRead(surface.base, surface.Bytes()), surface, entries, seed, what + ", read by the CPU");
}

struct Programmed {
    RectCase rect;
    const Program* program;
};

}

#endif

int main() {
#ifdef _WIN32
    constexpr Image TiledArray{Standard64KB, 2u, 1u};
    constexpr Image RenderTargetArray{RenderTarget64KB, 2u, 1u};
    constexpr Image Tiled256B{Standard256B, 1u, 1u};
    const std::array<Programmed, 6> rects{{
        {{"a store into rows 40-87 and columns 96-159 of layer 0 of a 64 KiB tiled 256x256 array of 2 layers", Binding::Srt, TiledArray, 0u, true}, &RectProgram(Binding::Srt, true, 0u)},
        {{"a store into rows 40-87 and columns 96-159 of layer 1 of a 64 KiB render-target tiled 256x256 array of 2 layers", Binding::Srt, RenderTargetArray, 1u, true}, &RectProgram(Binding::Srt, true, 1u)},
        {{"a store into rows 40-87 and columns 96-159 of layer 1 of a 64 KiB tiled 256x256 array of 2 layers", Binding::UserData, TiledArray, 1u, true}, &RectProgram(Binding::UserData, true, 1u)},
        {{"a store into rows 40-87 and columns 96-159 of layer 0 of a 64 KiB render-target tiled 256x256 array of 2 layers", Binding::UserData, RenderTargetArray, 0u, true}, &RectProgram(Binding::UserData, true, 0u)},
        {{"a store into rows 40-87 and columns 96-159 of a 256 B tiled 256x256 image", Binding::Srt, Tiled256B, 0u, false}, &RectProgram(Binding::Srt, false, 0u)},
        {{"a store into rows 40-87 and columns 96-159 of a 256 B tiled 256x256 image", Binding::UserData, Tiled256B, 0u, false}, &RectProgram(Binding::UserData, false, 0u)},
    }};
    const std::array<std::pair<Binding, const Program*>, 2> mips{{{Binding::Srt, &MipProgram(Binding::Srt)}, {Binding::UserData, &MipProgram(Binding::UserData)}}};
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    std::uint32_t seed = 1;
    for (const auto& [rect, program] : rects) {
        try {
            RunRect(rect, *program, seed++);
        } catch (const std::exception& error) {
            Expect(false, std::string(rect.name) + ": exception: " + error.what());
        }
    }
    for (const auto& [binding, program] : mips) {
        try {
            RunMips(binding, *program, seed++);
        } catch (const std::exception& error) {
            Expect(false, std::string("mip stores through ") + NameOf(binding) + ": exception: " + error.what());
        }
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    return DeviceTable::Finish("storage store unit");
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
