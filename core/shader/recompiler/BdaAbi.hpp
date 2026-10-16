#ifndef CORE_SHADER_RECOMPILER_BDAABI_HPP
#define CORE_SHADER_RECOMPILER_BDAABI_HPP

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace ShaderRecompiler::BdaAbi {

inline constexpr std::uint32_t Version = 1;
inline constexpr std::uint32_t TableVersion = 4;
inline constexpr std::uint32_t Read = 1;
inline constexpr std::uint32_t Write = 2;

struct Header {
    std::uint32_t version;
    std::uint32_t count;
    std::uint32_t grants;
    std::uint32_t reserved;
    std::uint64_t pageTable;
    std::uint64_t touchTable;
};

inline constexpr std::uint32_t HeaderWords = 8;
inline constexpr std::uint32_t CountWord = 1;
inline constexpr std::uint32_t GrantsWord = 2;
inline constexpr std::uint32_t PageTableWord = 4;
inline constexpr std::uint32_t TouchTableWord = 6;
inline constexpr std::uint32_t RangeWords = 8;
inline constexpr std::uint32_t GrantWords = 4;
inline constexpr std::uint32_t PageShift = 16;
inline constexpr std::uint32_t WindowShift = 32;
inline constexpr std::uint32_t AddressBits = 47;
inline constexpr std::uint64_t PageTableEnd = std::uint64_t{1} << AddressBits;
inline constexpr std::uint32_t WindowCount = 1u << (AddressBits - WindowShift);
inline constexpr std::uint32_t WindowPages = 1u << (WindowShift - PageShift);
inline constexpr std::uint32_t SplitShift = 4;
inline constexpr std::uint32_t SplitChildren = 1u << SplitShift;
inline constexpr std::uint64_t LimitMax = (std::uint64_t{1} << 33) - 1u;
inline constexpr std::uint32_t FirstMask = 0x1ffffu;
inline constexpr std::uint32_t LimitHighShift = 17;
inline constexpr std::uint32_t PermissionShift = 18;
inline constexpr std::uint32_t KindShift = 30;

enum class PageKind : std::uint32_t { None, Segment, Split };

struct PageEntry {
    std::uint64_t target;
    std::uint32_t bounds;
    std::uint32_t limit;
};

constexpr PageKind KindOf(const PageEntry& entry) {
    return static_cast<PageKind>(entry.bounds >> KindShift);
}

constexpr std::uint32_t FirstOf(const PageEntry& entry) {
    return entry.bounds & FirstMask;
}

constexpr std::uint64_t LimitOf(const PageEntry& entry) {
    return entry.limit | (static_cast<std::uint64_t>((entry.bounds >> LimitHighShift) & 1u) << 32u);
}

constexpr std::uint32_t PermissionsOf(const PageEntry& entry) {
    return (entry.bounds >> PermissionShift) & (Read | Write);
}

constexpr PageEntry SegmentEntry(std::uint64_t target, std::uint32_t first, std::uint64_t limit, std::uint32_t permissions) {
    return {target, first | static_cast<std::uint32_t>(limit >> 32u) << LimitHighShift | permissions << PermissionShift | static_cast<std::uint32_t>(PageKind::Segment) << KindShift, static_cast<std::uint32_t>(limit)};
}

constexpr PageEntry SplitEntry(std::uint64_t children) {
    return {children, static_cast<std::uint32_t>(PageKind::Split) << KindShift, 0u};
}

struct Range {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t deviceAddress;
    std::uint32_t permissions;
    std::uint32_t reserved;
};

struct Grant {
    std::uint64_t begin;
    std::uint64_t end;
};

enum class FaultState : std::uint32_t { Empty, Writing, Ready };
enum class FaultReason : std::uint32_t { Unmapped = 1, Permission, Overflow, InvalidTable, InvalidRectangle, LoopLimit, Unaligned, InvalidDescriptor, ImageTable, UnlinkedCall };

struct Fault {
    FaultState state;
    FaultReason reason;
    std::uint64_t address;
    std::uint32_t bytes;
    std::uint32_t stage;
    std::uint32_t instruction;
    std::uint32_t reserved;
};

static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header> && sizeof(Header) == HeaderWords * sizeof(std::uint32_t));
static_assert(offsetof(Header, pageTable) == PageTableWord * sizeof(std::uint32_t));
static_assert(offsetof(Header, touchTable) == TouchTableWord * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<PageEntry> && std::is_trivially_copyable_v<PageEntry> && sizeof(PageEntry) == 16);
static_assert(PageShift % SplitShift == 0 && FirstMask >= (1u << PageShift) && LimitMax > (std::uint64_t{1} << PageShift) + 0xffffffffull);
static_assert(offsetof(Header, count) == CountWord * sizeof(std::uint32_t) && offsetof(Header, grants) == GrantsWord * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<Range> && std::is_trivially_copyable_v<Range> && sizeof(Range) == RangeWords * sizeof(std::uint32_t));
static_assert(offsetof(Range, deviceAddress) == 16 && offsetof(Range, permissions) == 24);
static_assert(std::is_standard_layout_v<Grant> && std::is_trivially_copyable_v<Grant> && sizeof(Grant) == GrantWords * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<Fault> && std::is_trivially_copyable_v<Fault> && sizeof(Fault) == 32);
static_assert(offsetof(Fault, address) == 8 && offsetof(Fault, instruction) == 24);

inline constexpr std::uint32_t WrittenPageShift = 12;
inline constexpr std::uint32_t WrittenChunkShift = 17;
inline constexpr std::uint32_t WrittenChunkPages = 1u << (WrittenChunkShift - WrittenPageShift);
inline constexpr std::uint32_t WrittenPageSlots = 2048;
inline constexpr std::uint32_t WrittenPageProbes = 8;
inline constexpr std::uint32_t WrittenImageKeyBit = 1u << 31;
inline constexpr std::uint32_t WrittenImageWholeKey = WrittenImageKeyBit | 0x7fffffffu;
inline constexpr std::uint64_t WrittenAddressLimit = std::uint64_t{WrittenImageKeyBit - 2u} << WrittenChunkShift;
inline constexpr std::uint32_t WrittenOverflowWord = sizeof(Fault) / sizeof(std::uint32_t);
inline constexpr std::uint32_t WrittenCountWord = WrittenOverflowWord + 1;
inline constexpr std::uint32_t WrittenSlotsWord = WrittenCountWord + 1;
inline constexpr std::uint32_t WrittenListWord = WrittenSlotsWord + 2 * WrittenPageSlots;
inline constexpr std::size_t FaultBufferBytes = (WrittenListWord + WrittenPageSlots) * sizeof(std::uint32_t);
static_assert((WrittenPageSlots & (WrittenPageSlots - 1)) == 0);
static_assert(WrittenChunkPages == 32);
static_assert((WrittenAddressLimit >> WrittenChunkShift) < (WrittenImageWholeKey & ~WrittenImageKeyBit) && PageTableEnd <= WrittenAddressLimit);

inline constexpr std::uint32_t WrittenChunkKey(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> WrittenChunkShift) + 1u;
}

inline constexpr std::uint32_t WrittenImageChunkKey(std::uint64_t address) {
    return WrittenChunkKey(address) | WrittenImageKeyBit;
}

template<typename TMark>
bool ForEachWrittenChunk(const std::uint32_t* words, bool images, TMark&& mark) {
    const auto count = words[WrittenCountWord];
    if (count > WrittenPageSlots) return false;
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        if (slot >= WrittenPageSlots || (words[WrittenSlotsWord + 2 * slot] & ~WrittenImageKeyBit) == 0) return false;
    }
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        const auto key = words[WrittenSlotsWord + 2 * slot];
        if (((key & WrittenImageKeyBit) != 0) != images || key == WrittenImageWholeKey) continue;
        auto mask = words[WrittenSlotsWord + 2 * slot + 1];
        const auto base = static_cast<std::uint64_t>((key & ~WrittenImageKeyBit) - 1u) << WrittenChunkShift;
        while (mask != 0) {
            const auto first = static_cast<std::uint32_t>(std::countr_zero(mask));
            const auto pages = static_cast<std::uint32_t>(std::countr_one(mask >> first));
            mark(base + (static_cast<std::uint64_t>(first) << WrittenPageShift), static_cast<std::size_t>(pages) << WrittenPageShift);
            mask = pages + first >= 32u ? 0u : mask & ~((1u << (first + pages)) - 1u);
        }
    }
    return true;
}

template<typename TMark>
bool ForEachWrittenRange(const std::uint32_t* words, TMark&& mark) {
    return ForEachWrittenChunk(words, false, std::forward<TMark>(mark));
}

template<typename TMark>
bool ForEachWrittenImageRange(const std::uint32_t* words, TMark&& mark) {
    return ForEachWrittenChunk(words, true, std::forward<TMark>(mark));
}

inline bool WrittenImagesWhole(const std::uint32_t* words) {
    const auto count = std::min(words[WrittenCountWord], WrittenPageSlots);
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        if (slot < WrittenPageSlots && words[WrittenSlotsWord + 2 * slot] == WrittenImageWholeKey) return true;
    }
    return false;
}

}

#endif
