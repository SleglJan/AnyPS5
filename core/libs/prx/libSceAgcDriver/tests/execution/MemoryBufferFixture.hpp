#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_MEMORYBUFFERFIXTURE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_MEMORYBUFFERFIXTURE_HPP

#include "DeviceTableFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <cstring>

namespace MemoryBuffers {

using namespace DeviceTable;

constexpr std::uint32_t TableSlot = 4;
constexpr std::uint32_t DataSlot = 16;
constexpr std::uint32_t InputSlot = 24;
constexpr std::uint32_t TableOutputSlot = 4;
constexpr std::uint32_t TableWords = 16;
constexpr std::uint32_t RowWords = Lanes * LaneWords;
constexpr std::uint32_t ImageBytes = Width * Height * 4u;
constexpr std::size_t BlockBytes = 65536;
constexpr std::array<std::uint32_t, 4> Bases{0x41000000u, 0x52000000u, 0x63000000u, 0x74000000u};

using Words = TestGuest::Array<std::uint32_t, OutputWords>;

namespace Ops {

using namespace DeviceTable::Asm;

constexpr std::uint32_t SLoadDwordX2 = 0x01u;
constexpr std::uint32_t SLoadDwordX8 = 0x03u;
constexpr std::uint32_t BufferLoadDwordX4 = 0x0eu;

constexpr std::array<std::uint32_t, 2> SmemRegisterOffset(std::uint32_t opcode, std::uint32_t sdata, std::uint32_t sbase, std::uint32_t soffset, std::uint32_t offset) {
    return {0xf4000000u | (opcode << 18u) | (sdata << 6u) | (sbase / 2u), (soffset << 25u) | (offset & 0x1fffffu)};
}

inline void LaneValues(Program& program, std::uint32_t groupSgpr) {
    program << Vop1(VMovB32, 10, groupSgpr)
            << Vop2(VLshlRevB32, 10, InlineInteger(8), 10)
            << Vop2(VAddNcU32, 10, Vgpr(0), 10);
    for (std::uint32_t component = 0; component < LaneWords; ++component) program << Vop2(VAddNcU32, 4 + component, Literal, 10) << Bases[component];
}

inline void LaneAddress(Program& program, std::uint32_t groupSgpr) {
    program << Vop1(VMovB32, 9, groupSgpr)
            << Vop2(VLshlRevB32, 8, InlineInteger(6), 9)
            << Vop2(VAddNcU32, 8, Vgpr(0), 8)
            << Vop2(VLshlRevB32, 8, InlineInteger(4), 8);
}

}

inline std::uint32_t Value(std::uint32_t row, std::uint32_t lane, std::uint32_t component) {
    return Bases[component] + (row << 8u) + lane;
}

inline std::uint32_t InputValue(std::uint32_t word) {
    return 0x0c000000u ^ (word * 0x9e3779b1u);
}

inline const Program& StoreProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        LoadSrt(built, {{28, OutputSlot}});
        LaneValues(built, 2);
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

inline const Program& CopyProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        LoadSrt(built, {{28, OutputSlot}, {32, InputSlot}});
        LaneAddress(built, 2);
        built << NewPath::Isa::Mubuf(BufferLoadDwordX4, 4, 8, 32);
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

inline const Program& WriterProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        for (std::uint32_t word = 0; word < 4u; ++word) built << Vop1(VMovB32, 4 + word, 4 + word);
        built << Vop1(VMovB32, 8, InlineInteger(OutputSlot * 4u))
              << NewPath::Isa::Mubuf(BufferStoreDwordX4, 4, 8, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

inline const Program& ImageStoreProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop2(VAndB32, 1, InlineInteger(63), 0)
              << Vop1(VMovB32, 2, 9)
              << Vop2(VLshlRevB32, 3, InlineInteger(8), 2)
              << Vop2(VAddNcU32, 3, Vgpr(1), 3)
              << Vop2(VAddNcU32, 3, 8, 3)
              << Vop1(VCvtF32U32, 4, Vgpr(3))
              << Mimg(ImageStore, 1u, false, 4, 1, 0, 0, true)
              << EndProgram;
        return built;
    }();
    return program;
}

inline std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

inline std::uint32_t* OwnBlock() {
    return reinterpret_cast<std::uint32_t*>(TestGuest::Allocate(BlockBytes, BlockBytes));
}

inline std::array<std::uint32_t, 2> Pointer(const void* data) {
    const auto address = AddressOf(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)};
}

using Ranges = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

inline bool Contains(const Ranges& ranges, std::uint64_t begin, std::uint64_t end) {
    return std::ranges::find(ranges, std::pair<std::uint64_t, std::uint64_t>{begin, end}) != ranges.end();
}

inline bool Covers(const Ranges& ranges, std::uint64_t begin, std::uint64_t end) {
    return std::ranges::any_of(ranges, [&](const std::pair<std::uint64_t, std::uint64_t>& range) { return range.first <= begin && range.second >= end; });
}

inline std::string Span(std::uint64_t begin, std::uint64_t end) {
    return "[" + NewPath::Hex(begin) + ", " + NewPath::Hex(end) + ")";
}

inline std::string RangeText(const Ranges& ranges, bool unbounded) {
    std::string text;
    for (const auto& [begin, end] : ranges) text += (text.empty() ? "" : ", ") + Span(begin, end);
    return (text.empty() ? "none" : text) + (unbounded ? " (unbounded)" : "");
}

inline std::string Writes(const ShaderRecompiler::RecompileResult& result) {
    return "write ranges " + RangeText(result.writeRanges, result.unboundedWrites);
}

inline std::string Reads(const ShaderRecompiler::RecompileResult& result) {
    return "read ranges " + RangeText(result.readRanges, result.unboundedReads);
}

inline bool RecordedWrites(std::uint64_t address, std::size_t bytes) {
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr && recorder->PendingWriteOverlaps(address, bytes);
}

inline bool PendingWrite(const void* data, std::size_t bytes) {
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr && recorder->PendingWriteOverlaps(AddressOf(data), bytes);
}

inline std::vector<std::uint32_t> ReadBack(const void* data, std::size_t words) {
    AgcDriver::GuestMemory::FlushGpuWrites(AddressOf(data), words * sizeof(std::uint32_t));
    std::vector<std::uint32_t> values(words);
    std::memcpy(values.data(), data, words * sizeof(std::uint32_t));
    return values;
}

inline void ExpectRows(std::span<const std::uint32_t> words, std::uint32_t rows, const std::function<std::uint32_t(std::uint32_t row, std::uint32_t lane, std::uint32_t component)>& expected, const std::string& what) {
    for (std::uint32_t word = 0; word < words.size(); ++word) {
        const auto row = word / RowWords;
        const auto lane = word % RowWords / LaneWords;
        const auto component = word % LaneWords;
        const auto wanted = row < rows ? expected(row, lane, component) : Untouched;
        if (words[word] == wanted) continue;
        Expect(false, what + ": row " + std::to_string(row) + " lane " + std::to_string(lane) + " component " + std::to_string(component) + " is " + Hex(words[word]) + ", expected " + Hex(wanted));
        return;
    }
}

inline void ExpectStored(std::span<const std::uint32_t> words, std::uint32_t rows, const std::string& what) {
    ExpectRows(words, rows, Value, what);
}

inline void ExpectUntouched(std::span<const std::uint32_t> words, const std::string& what) {
    ExpectRows(words, 0, Value, what);
}

constexpr std::uint32_t ImageTileStandard64KB = 0x09u;

inline std::array<std::uint32_t, 9> ImageStoreUserData(const void* texels, std::uint32_t seed) {
    std::array<std::uint32_t, 9> userData{};
    const auto image = Image(texels, Format32Float);
    std::copy(image.begin(), image.end(), userData.begin());
    userData[3] |= ImageTileStandard64KB << 20u;
    userData[8] = seed;
    return userData;
}

inline float StoredTexel(std::uint32_t seed, std::uint32_t x, std::uint32_t y) {
    return static_cast<float>(seed + x + 256u * y);
}

inline bool ImagePending(const void* texels) {
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{AddressOf(texels), AddressOf(texels) + ImageBytes}}};
    return AgcDriver::Graphics::StorageTexture::AnyPendingOverlaps(range);
}

inline std::uint32_t StandardTexelOffset(std::uint32_t x, std::uint32_t y) {
    const auto small = ((y << 4u) & 0x070u) ^ ((y << 5u) & 0x100u) ^ ((y << 6u) & 0x400u) ^ ((x << 2u) & 0x00cu) ^ ((x << 5u) & 0x080u) ^ ((x << 6u) & 0x200u) ^ ((x << 7u) & 0x800u);
    return small ^ ((x << 8u) & 0x2000u) ^ ((x << 9u) & 0x8000u) ^ ((y << 7u) & 0x1000u) ^ ((y << 8u) & 0x4000u);
}

inline void ExpectImage(const void* texels, std::uint32_t seed, const std::string& what) {
    AgcDriver::GuestMemory::FlushGpuWrites(AddressOf(texels), BlockBytes);
    const auto* bytes = static_cast<const std::byte*>(texels);
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            float texel = 0.0f;
            std::memcpy(&texel, bytes + StandardTexelOffset(x, y), sizeof(texel));
            if (texel == StoredTexel(seed, x, y)) continue;
            Expect(false, what + ": texel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + std::to_string(texel) + ", expected " + std::to_string(StoredTexel(seed, x, y)));
            return;
        }
    }
}

}

#endif
