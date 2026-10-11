#ifndef CORE_SHADER_RECOMPILER_RESULTKEY_HPP
#define CORE_SHADER_RECOMPILER_RESULTKEY_HPP

#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include <cstdint>
#include <span>

// Parts of the result memo key (Recompiler.cpp, snapshotHash) that are canonical: they hash only
// what the materialized result is made of, and the large ones cheaply.
namespace ShaderRecompiler::ResultKey {

namespace Detail {

constexpr std::uint64_t LaneMultipliers[4] = {0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0xd6e8feb86659fd93ull};

constexpr std::uint64_t Step(std::uint64_t lane, std::uint64_t value, std::uint64_t multiplier) {
    lane = (lane ^ value) * multiplier;
    return lane ^ (lane >> 29u);
}

constexpr std::uint64_t Rotate(std::uint64_t value, unsigned count) {
    return (value << count) | (value >> (64u - count));
}

}

// A 64-bit hash of a dword span: four independent multiply-xorshift lanes over dword pairs (a
// quarter of FNV-1a's serial multiply chain per dword), each step a bijection of its lane, with the
// length and the lanes folded at the end. Equal spans hash equal; the result memo trusts a 64-bit
// match as the FNV-1a key did.
inline std::uint64_t HashWords(std::span<const std::uint32_t> words, std::uint64_t seed) {
    using namespace Detail;
    std::uint64_t lanes[4] = {seed ^ LaneMultipliers[0], seed ^ LaneMultipliers[1], Rotate(seed, 21u) ^ LaneMultipliers[2], Rotate(seed, 43u) ^ LaneMultipliers[3]};
    const auto pair = [&](std::size_t at) { return static_cast<std::uint64_t>(words[at]) | (static_cast<std::uint64_t>(words[at + 1u]) << 32u); };
    std::size_t at = 0;
    for (; at + 8u <= words.size(); at += 8u) {
        lanes[0] = Step(lanes[0], pair(at), LaneMultipliers[0]);
        lanes[1] = Step(lanes[1], pair(at + 2u), LaneMultipliers[1]);
        lanes[2] = Step(lanes[2], pair(at + 4u), LaneMultipliers[2]);
        lanes[3] = Step(lanes[3], pair(at + 6u), LaneMultipliers[3]);
    }
    for (std::size_t lane = 0; at < words.size(); ++at, lane = (lane + 1u) & 3u) lanes[lane] = Step(lanes[lane], words[at] | (static_cast<std::uint64_t>(at) << 32u), LaneMultipliers[lane]);
    auto hash = Step(lanes[0], words.size(), LaneMultipliers[1]);
    hash = Step(hash, Rotate(lanes[1], 17u), LaneMultipliers[2]);
    hash = Step(hash, Rotate(lanes[2], 31u), LaneMultipliers[3]);
    hash = Step(hash, Rotate(lanes[3], 47u), LaneMultipliers[0]);
    return hash;
}

// Everything of a snapshot's tables that ResolveImageTables and the result's image-table fields
// read: the shader, each column (base, size, records, keys, fault, outside, codes, read ranges),
// the distinct words (their first dwordCount dwords) and the ranges.
inline std::uint64_t TablesHash(const ImageTableSnapshot& tables) {
    using namespace Detail;
    std::uint64_t hash = Step(0x5bd1e9955bd1e995ull, tables.shader, LaneMultipliers[0]);
    hash = Step(hash, tables.tables.size(), LaneMultipliers[1]);
    for (const auto& table : tables.tables) {
        hash = Step(hash, table.base, LaneMultipliers[2]);
        hash = Step(hash, table.size, LaneMultipliers[3]);
        hash = Step(hash, (static_cast<std::uint64_t>(table.records) << 32u) | table.keys, LaneMultipliers[0]);
        hash = Step(hash, (static_cast<std::uint64_t>(table.fault) << 1u) | (table.outside ? 1u : 0u), LaneMultipliers[1]);
        hash = Step(hash, HashWords(table.codes, hash), LaneMultipliers[2]);
        hash = Step(hash, table.readRanges.size(), LaneMultipliers[3]);
        for (const auto& [base, size] : table.readRanges) {
            hash = Step(hash, base, LaneMultipliers[0]);
            hash = Step(hash, size, LaneMultipliers[1]);
        }
    }
    hash = Step(hash, tables.words.size(), LaneMultipliers[2]);
    for (const auto& value : tables.words) {
        const auto count = value.dwordCount < value.dwords.size() ? value.dwordCount : static_cast<std::uint32_t>(value.dwords.size());
        hash = Step(hash, HashWords(std::span<const std::uint32_t>(value.dwords).first(count), value.dwordCount), LaneMultipliers[3]);
    }
    hash = Step(hash, tables.ranges.size(), LaneMultipliers[0]);
    for (const auto& [base, size] : tables.ranges) {
        hash = Step(hash, base, LaneMultipliers[1]);
        hash = Step(hash, size, LaneMultipliers[2]);
    }
    return hash;
}

// The user data a variant's materialized result reads: ShaderDataDwordsFor copies the words at the
// layout's userDataRegisters (relative to the variant's user-data base) and nothing else of it
// reaches the result (the SRT walk's reads of it are keyed through the buffers, images, samplers
// and the flat SRT). A register outside the user data mixes a marker (the materialization throws).
inline std::uint64_t ConsumedUserDataHash(std::span<const std::uint32_t> userData, std::span<const std::uint32_t> registers, std::uint32_t userDataBase) {
    using namespace Detail;
    std::uint64_t hash = Step(0x27d4eb2f165667c5ull, registers.size(), LaneMultipliers[0]);
    for (const auto reg : registers) {
        const bool inside = reg >= userDataBase && reg - userDataBase < userData.size();
        hash = Step(hash, (static_cast<std::uint64_t>(reg) << 32u) | (inside ? userData[reg - userDataBase] : 0u), LaneMultipliers[1]);
        if (!inside) hash = Step(hash, ~std::uint64_t{0}, LaneMultipliers[2]);
    }
    return hash;
}

}

#endif
