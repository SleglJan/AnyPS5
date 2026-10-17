#ifndef CORE_SHADER_RECOMPILER_CONTINUATIONABI_HPP
#define CORE_SHADER_RECOMPILER_CONTINUATIONABI_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace ShaderRecompiler::ContinuationAbi {

inline constexpr std::uint32_t Version = 1u;
inline constexpr std::uint32_t MaxBins = 64u;

enum class Role : std::uint32_t { None, Head, Continuation };

inline constexpr std::uint32_t KeyWords = 4u;
inline constexpr std::uint32_t RecordRegisters = 8u;
inline constexpr std::uint32_t NoRecordRegister = 0xffffffffu;
inline constexpr std::uint32_t KeyIndexShift = 16u;

struct BinSite {
    std::uint32_t programCounter = 0;
    std::uint32_t callProgramCounter = 0;
    std::uint32_t keyMask = 0;
    std::array<std::uint32_t, KeyWords> keyDwords{NoRecordRegister, NoRecordRegister, NoRecordRegister, NoRecordRegister};
    std::array<std::uint32_t, RecordRegisters> recordRegisters{NoRecordRegister, NoRecordRegister, NoRecordRegister, NoRecordRegister, NoRecordRegister, NoRecordRegister, NoRecordRegister, NoRecordRegister};

    bool operator==(const BinSite&) const = default;
};

struct Shape {
    Role role = Role::None;
    std::span<const BinSite> sites;
    std::span<const std::uint32_t> scalarRegisters;
    std::span<const std::uint32_t> vectorRegisters;
    std::uint32_t ldsDwords = 0;
    std::span<const std::uint32_t> userDataScalars;
    std::uint32_t entrySite = 0;
};

inline constexpr std::uint32_t UserDataWords = 10u;
inline constexpr std::uint32_t BlockLowWord = 0u;
inline constexpr std::uint32_t BlockHighWord = 1u;
inline constexpr std::uint32_t BinWord = 2u;
inline constexpr std::uint32_t FirstGroupWord = 3u;
inline constexpr std::uint32_t GroupsXWord = 4u;
inline constexpr std::uint32_t GroupsYWord = 5u;
inline constexpr std::uint32_t KeyWord = 6u;

struct Header {
    std::uint32_t version;
    std::uint32_t bins;
    std::uint32_t rounds;
    std::uint32_t slots;
    std::uint32_t recordDwords;
    std::uint32_t countersDword;
    std::uint32_t listsDword;
    std::uint32_t recordsDword;
    std::uint32_t round;
    std::uint32_t reserved[7];
};

inline constexpr std::uint32_t HeaderDwords = 16u;
inline constexpr std::uint32_t VersionWord = 0u;
inline constexpr std::uint32_t BinsWord = 1u;
inline constexpr std::uint32_t RoundsWord = 2u;
inline constexpr std::uint32_t SlotsWord = 3u;
inline constexpr std::uint32_t RecordDwordsWord = 4u;
inline constexpr std::uint32_t CountersWord = 5u;
inline constexpr std::uint32_t ListsWord = 6u;
inline constexpr std::uint32_t RecordsWord = 7u;
inline constexpr std::uint32_t RoundWord = 8u;

struct BinEntry {
    std::uint32_t site;
    std::uint32_t keyMask;
    std::array<std::uint32_t, KeyWords> key;
    std::uint32_t reserved[2];
};

inline constexpr std::uint32_t BinEntryDwords = 8u;
inline constexpr std::uint32_t BinSiteWord = 0u;
inline constexpr std::uint32_t BinKeyMaskWord = 1u;
inline constexpr std::uint32_t BinKeyWord = 2u;
inline constexpr std::uint32_t BinTableDword = HeaderDwords;
inline constexpr std::uint32_t CounterDwords = 4u;
inline constexpr std::uint32_t CounterDispatchedWord = 3u;
inline constexpr std::uint32_t CountersDword = BinTableDword + MaxBins * BinEntryDwords;

inline constexpr std::uint32_t RecordHeaderDwords = 4u;
inline constexpr std::uint32_t RecordSiteWord = 0u;
inline constexpr std::uint32_t RecordTargetLowWord = 1u;
inline constexpr std::uint32_t RecordTargetHighWord = 2u;
inline constexpr std::uint32_t RecordRoundWord = 3u;

inline constexpr std::uint32_t SccWord = 0u;
inline constexpr std::uint32_t ExecLowWord = 1u;
inline constexpr std::uint32_t ExecHighWord = 2u;
inline constexpr std::uint32_t VccLowWord = 3u;
inline constexpr std::uint32_t VccHighWord = 4u;
inline constexpr std::uint32_t M0Word = 5u;
inline constexpr std::uint32_t FixedScalarWords = 6u;

constexpr std::uint32_t FixedScalarRecordWord(std::uint32_t fixed) {
    return RecordHeaderDwords + fixed;
}

constexpr std::uint32_t ScalarRecordWord(std::uint32_t index) {
    return RecordHeaderDwords + FixedScalarWords + index;
}

constexpr std::uint32_t VectorRecordWord(std::uint32_t scalarCount, std::uint32_t waveSize, std::uint32_t index) {
    return RecordHeaderDwords + FixedScalarWords + scalarCount + index * waveSize;
}

constexpr std::uint32_t LdsRecordWord(std::uint32_t scalarCount, std::uint32_t vectorCount, std::uint32_t waveSize) {
    return VectorRecordWord(scalarCount, waveSize, vectorCount);
}

constexpr std::uint32_t RecordDwords(std::uint32_t scalarCount, std::uint32_t vectorCount, std::uint32_t waveSize, std::uint32_t ldsDwords) {
    return (LdsRecordWord(scalarCount, vectorCount, waveSize) + ldsDwords + 3u) & ~3u;
}

struct Geometry {
    std::uint32_t bins = 0;
    std::uint32_t rounds = 0;
    std::uint32_t slots = 0;
    std::uint32_t recordDwords = 0;
};

constexpr std::uint64_t CounterDword(const Geometry& geometry, std::uint32_t round, std::uint32_t bin) {
    return CountersDword + (static_cast<std::uint64_t>(round - 1u) * geometry.bins + bin) * CounterDwords;
}

constexpr std::uint64_t ListsDword(const Geometry& geometry) {
    return CountersDword + static_cast<std::uint64_t>(geometry.rounds) * geometry.bins * CounterDwords;
}

constexpr std::uint64_t ListDword(const Geometry& geometry, std::uint32_t round, std::uint32_t bin, std::uint32_t index) {
    return ListsDword(geometry) + (static_cast<std::uint64_t>(round % 2u) * geometry.bins + bin) * geometry.slots + index;
}

constexpr std::uint64_t RecordsDword(const Geometry& geometry) {
    return (ListsDword(geometry) + 2ull * geometry.bins * geometry.slots + 3u) & ~std::uint64_t{3u};
}

constexpr std::uint64_t BlockDwords(const Geometry& geometry) {
    return RecordsDword(geometry) + static_cast<std::uint64_t>(geometry.slots) * geometry.recordDwords;
}

static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header> && sizeof(Header) == HeaderDwords * sizeof(std::uint32_t));
static_assert(offsetof(Header, bins) == BinsWord * 4u && offsetof(Header, rounds) == RoundsWord * 4u && offsetof(Header, slots) == SlotsWord * 4u);
static_assert(offsetof(Header, recordDwords) == RecordDwordsWord * 4u && offsetof(Header, countersDword) == CountersWord * 4u);
static_assert(offsetof(Header, listsDword) == ListsWord * 4u && offsetof(Header, recordsDword) == RecordsWord * 4u && offsetof(Header, round) == RoundWord * 4u);
static_assert(std::is_standard_layout_v<BinEntry> && sizeof(BinEntry) == BinEntryDwords * sizeof(std::uint32_t));
static_assert(offsetof(BinEntry, keyMask) == BinKeyMaskWord * 4u && offsetof(BinEntry, key) == BinKeyWord * 4u);
static_assert(CountersDword % CounterDwords == 0u);

}

#endif
