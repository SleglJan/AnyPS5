#ifndef CORE_SHADER_RECOMPILER_BDATOUCHABI_HPP
#define CORE_SHADER_RECOMPILER_BDATOUCHABI_HPP

#include <bit>
#include <cstddef>
#include <cstdint>

namespace ShaderRecompiler::BdaTouchAbi {

inline constexpr std::uint32_t HeaderWord = 6;
inline constexpr std::uint32_t PageShift = 16;
inline constexpr std::uint32_t ChunkShift = 21;
inline constexpr std::uint32_t ChunkPages = 1u << (ChunkShift - PageShift);
inline constexpr std::uint32_t Slots = 16384;
inline constexpr std::uint32_t Probes = 8;
inline constexpr std::uint32_t HashMultiplier = 0x9e3779b1u;
inline constexpr std::uint32_t OverflowWord = 0;
inline constexpr std::uint32_t SlotsWord = 4;
inline constexpr std::size_t TableWords = SlotsWord + 2 * Slots;
inline constexpr std::size_t TableBytes = TableWords * sizeof(std::uint32_t);

static_assert(ChunkPages == 32);
static_assert((Slots & (Slots - 1)) == 0);
static_assert((std::uint64_t{1} << 47 >> ChunkShift) < 0xffffffffull);

inline constexpr std::uint32_t ChunkKey(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> ChunkShift) + 1u;
}

inline constexpr std::uint32_t PageBit(std::uint64_t address) {
    return 1u << static_cast<std::uint32_t>((address >> PageShift) & (ChunkPages - 1u));
}

inline constexpr std::uint32_t FirstSlot(std::uint32_t key) {
    return (key * HashMultiplier) >> (32u - static_cast<std::uint32_t>(std::countr_zero(Slots)));
}

inline constexpr std::uint32_t SlotKeyWord(std::uint32_t slot) {
    return SlotsWord + 2u * slot;
}

template<typename TPage>
void ForEachTouchedPage(const std::uint32_t* words, TPage&& page) {
    for (std::uint32_t slot = 0; slot < Slots; ++slot) {
        const auto key = words[SlotKeyWord(slot)];
        auto mask = words[SlotKeyWord(slot) + 1u];
        if (key == 0 || mask == 0) continue;
        const auto first = static_cast<std::uint64_t>(key - 1u) << (ChunkShift - PageShift);
        while (mask != 0) {
            const auto bit = static_cast<std::uint32_t>(std::countr_zero(mask));
            page(first + bit);
            mask &= mask - 1u;
        }
    }
}

}

#endif
