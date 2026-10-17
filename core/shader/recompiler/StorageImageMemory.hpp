#ifndef CORE_SHADER_RECOMPILER_STORAGEIMAGEMEMORY_HPP
#define CORE_SHADER_RECOMPILER_STORAGEIMAGEMEMORY_HPP

#include "IntermediateRepresentation/IrMetadata/Resources.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderInfo.hpp"
#include "StorageImagePages.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace ShaderRecompiler::StorageImageMemory {

inline constexpr std::uint32_t MemorySlot = 0xffffffffu;
inline constexpr std::uint32_t LastMemoryFormat = 127u;

enum class Access : std::uint32_t { Load, Store, Atomic32, Atomic64 };

enum class Layout : std::uint32_t { Unsupported, Linear, Thin, Thick };

struct Fields {
    std::uint64_t base = 0;
    std::uint32_t xorBits = 0;
    std::uint32_t format = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t baseLevel = 0;
    std::uint32_t lastLevel = 0;
    std::uint32_t tileMode = 0;
    std::uint32_t type = 0;
    std::uint32_t depth = 0;
    std::uint32_t baseArray = 0;
    std::uint32_t arrayPitch = 0;
    std::uint32_t maxMip = 0;
    bool metaCompress = false;
};

struct Texel {
    bool inside = false;
    std::uint64_t address = 0;
};

struct Equation {
    std::uint32_t swizzleMode;
    std::uint32_t elementBytes;
    std::array<std::uint32_t, 16> bits;
};

inline constexpr std::array<Equation, 75> Equations{{
    {24u, 1u, {0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x00000004u, 0x00004000u, 0x00000008u, 0x00010000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00040000u, 0x00000040u, 0x00080000u, 0x00000080u}},
    {24u, 2u, {0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x00000004u, 0x00004000u, 0x00000008u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00010000u, 0x00000040u, 0x00040000u, 0x00000080u}},
    {24u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x00000004u, 0x00004000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00008000u, 0x00000010u, 0x00040000u, 0x00000040u}},
    {24u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x00000004u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00004000u, 0x00000008u, 0x00010000u, 0x00000040u}},
    {24u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00004000u, 0x00000004u, 0x00008000u, 0x00000010u}},
    {25u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00008000u, 0x08010080u, 0x04080010u, 0x02020040u, 0x01040020u, 0x00040000u, 0x00000040u, 0x00080000u, 0x00000080u}},
    {25u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x08008080u, 0x04040010u, 0x02010040u, 0x01020020u, 0x00020000u, 0x00000040u, 0x00040000u, 0x00000080u}},
    {25u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x08008040u, 0x04040008u, 0x02010020u, 0x01020010u, 0x00020000u, 0x00000020u, 0x00040000u, 0x00000040u}},
    {25u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00002000u, 0x00000002u, 0x00000004u, 0x08004040u, 0x04020008u, 0x02008020u, 0x01010010u, 0x00010000u, 0x00000020u, 0x00020000u, 0x00000040u}},
    {25u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00001000u, 0x00002000u, 0x00000001u, 0x00000002u, 0x08004020u, 0x04020004u, 0x02008010u, 0x01010008u, 0x00010000u, 0x00000010u, 0x00020000u, 0x00000020u}},
    {26u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00001000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x08010080u, 0x04080010u, 0x02020040u, 0x01040020u, 0x00040000u, 0x00000040u, 0x00080000u, 0x00000080u}},
    {26u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x08008080u, 0x04040010u, 0x02010040u, 0x01020020u, 0x00020000u, 0x00000040u, 0x00040000u, 0x00000080u}},
    {26u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x08008040u, 0x04040008u, 0x02010020u, 0x01020010u, 0x00020000u, 0x00000020u, 0x00040000u, 0x00000040u}},
    {26u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00000004u, 0x00002000u, 0x08004040u, 0x04020008u, 0x02008020u, 0x01010010u, 0x00010000u, 0x00000020u, 0x00020000u, 0x00000040u}},
    {26u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x08004020u, 0x04020004u, 0x02008010u, 0x01010008u, 0x00010000u, 0x00000010u, 0x00020000u, 0x00000020u}},
    {27u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00001000u, 0x00004000u, 0x00000008u, 0x00010000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00040000u, 0x00000040u, 0x00080000u, 0x00000080u}},
    {27u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00010000u, 0x00000040u, 0x00040000u, 0x00000080u}},
    {27u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00008000u, 0x00000010u, 0x00040000u, 0x00000040u}},
    {27u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00000004u, 0x00002000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00004000u, 0x00000008u, 0x00010000u, 0x00000040u}},
    {27u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00004000u, 0x00000004u, 0x00008000u, 0x00000010u}},
    {2u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00001000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {2u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {2u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {2u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {2u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {6u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00001000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x00010000u, 0x00000010u, 0x00020000u, 0x00000020u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {6u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x00000010u, 0x00010000u, 0x00000020u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {6u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x00008000u, 0x00000008u, 0x00010000u, 0x00000010u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {6u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x00000010u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {6u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x00004000u, 0x00000004u, 0x00008000u, 0x00000008u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {10u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00001000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x00010000u, 0x00000010u, 0x00020000u, 0x00000020u, 0x00040000u, 0x00000040u, 0x00080000u, 0x00000080u}},
    {10u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x00000010u, 0x00010000u, 0x00000020u, 0x00020000u, 0x00000040u, 0x00040000u, 0x00000080u}},
    {10u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x00008000u, 0x00000008u, 0x00010000u, 0x00000010u, 0x00020000u, 0x00000020u, 0x00040000u, 0x00000040u}},
    {10u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x00000010u, 0x00010000u, 0x00000020u, 0x00020000u, 0x00000040u}},
    {10u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x00004000u, 0x00000004u, 0x00008000u, 0x00000008u, 0x00010000u, 0x00000010u, 0x00020000u, 0x00000020u}},
    {17u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00008000u, 0x00010080u, 0x00080010u, 0x00020040u, 0x00040020u, 0x00040000u, 0x00000040u, 0x00080000u, 0x00000080u}},
    {17u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x00008080u, 0x00040010u, 0x00010040u, 0x00020020u, 0x00020000u, 0x00000040u, 0x00040000u, 0x00000080u}},
    {17u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x00008040u, 0x00040008u, 0x00010020u, 0x00020010u, 0x00020000u, 0x00000020u, 0x00040000u, 0x00000040u}},
    {17u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00002000u, 0x00000002u, 0x00000004u, 0x00004040u, 0x00020008u, 0x00008020u, 0x00010010u, 0x00010000u, 0x00000020u, 0x00020000u, 0x00000040u}},
    {17u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00001000u, 0x00002000u, 0x00000001u, 0x00000002u, 0x00004020u, 0x00020004u, 0x00008010u, 0x00010008u, 0x00010000u, 0x00000010u, 0x00020000u, 0x00000020u}},
    {18u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00001000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x00010080u, 0x00080010u, 0x00020040u, 0x00040020u, 0x00040000u, 0x00000040u, 0x00080000u, 0x00000080u}},
    {18u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x00008080u, 0x00040010u, 0x00010040u, 0x00020020u, 0x00020000u, 0x00000040u, 0x00040000u, 0x00000080u}},
    {18u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x00008040u, 0x00040008u, 0x00010020u, 0x00020010u, 0x00020000u, 0x00000020u, 0x00040000u, 0x00000040u}},
    {18u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00004040u, 0x00020008u, 0x00008020u, 0x00010010u, 0x00010000u, 0x00000020u, 0x00020000u, 0x00000040u}},
    {18u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x00004020u, 0x00020004u, 0x00008010u, 0x00010008u, 0x00010000u, 0x00000010u, 0x00020000u, 0x00000020u}},
    {21u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00008000u, 0x08010080u, 0x04080010u, 0x02020040u, 0x01040020u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {21u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x08008080u, 0x04040010u, 0x02010040u, 0x01020020u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {21u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x08008040u, 0x04040008u, 0x02010020u, 0x01020010u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {21u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00002000u, 0x00000002u, 0x00000004u, 0x08004040u, 0x04020008u, 0x02008020u, 0x01010010u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {21u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00001000u, 0x00002000u, 0x00000001u, 0x00000002u, 0x08004020u, 0x04020004u, 0x02008010u, 0x01010008u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {22u, 1u, {0x00000001u, 0x00000002u, 0x00000004u, 0x00002000u, 0x00001000u, 0x00004000u, 0x00000008u, 0x00008000u, 0x08010080u, 0x04080010u, 0x02020040u, 0x01040020u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {22u, 2u, {0x00000000u, 0x00000001u, 0x00000002u, 0x00000004u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000008u, 0x08008080u, 0x04040010u, 0x02010040u, 0x01020020u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {22u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00000002u, 0x00001000u, 0x00002000u, 0x00004000u, 0x00000004u, 0x08008040u, 0x04040008u, 0x02010020u, 0x01020010u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {22u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00000004u, 0x00002000u, 0x08004040u, 0x04020008u, 0x02008020u, 0x01010010u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {22u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x00002000u, 0x08004020u, 0x04020004u, 0x02008010u, 0x01010008u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {0x105u, 1u, {0x00000001u, 0x00000002u, 0x01000000u, 0x00001000u, 0x02000000u, 0x00002000u, 0x00000004u, 0x04000000u, 0x00004000u, 0x00000008u, 0x08000000u, 0x00008000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {0x105u, 2u, {0x00000000u, 0x00000001u, 0x01000000u, 0x00001000u, 0x02000000u, 0x00002000u, 0x00000002u, 0x04000000u, 0x00004000u, 0x00000004u, 0x08000000u, 0x00008000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {0x105u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x01000000u, 0x00002000u, 0x00000002u, 0x02000000u, 0x00004000u, 0x00000004u, 0x04000000u, 0x00008000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {0x105u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x01000000u, 0x00001000u, 0x00000002u, 0x02000000u, 0x00002000u, 0x00000004u, 0x04000000u, 0x00004000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {0x105u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x01000000u, 0x00001000u, 0x00000001u, 0x02000000u, 0x00002000u, 0x00000002u, 0x04000000u, 0x00004000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {0x109u, 1u, {0x00000001u, 0x00000002u, 0x01000000u, 0x00001000u, 0x02000000u, 0x00002000u, 0x00000004u, 0x04000000u, 0x00004000u, 0x00000008u, 0x08000000u, 0x00008000u, 0x00000010u, 0x10000000u, 0x00010000u, 0x00000020u}},
    {0x109u, 2u, {0x00000000u, 0x00000001u, 0x01000000u, 0x00001000u, 0x02000000u, 0x00002000u, 0x00000002u, 0x04000000u, 0x00004000u, 0x00000004u, 0x08000000u, 0x00008000u, 0x00000008u, 0x10000000u, 0x00010000u, 0x00000010u}},
    {0x109u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x01000000u, 0x00002000u, 0x00000002u, 0x02000000u, 0x00004000u, 0x00000004u, 0x04000000u, 0x00008000u, 0x00000008u, 0x08000000u, 0x00010000u, 0x00000010u}},
    {0x109u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x01000000u, 0x00001000u, 0x00000002u, 0x02000000u, 0x00002000u, 0x00000004u, 0x04000000u, 0x00004000u, 0x00000008u, 0x08000000u, 0x00008000u, 0x00000010u}},
    {0x109u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x01000000u, 0x00001000u, 0x00000001u, 0x02000000u, 0x00002000u, 0x00000002u, 0x04000000u, 0x00004000u, 0x00000004u, 0x08000000u, 0x00008000u, 0x00000008u}},
    {0x119u, 1u, {0x00000001u, 0x00000002u, 0x01000000u, 0x00001000u, 0x02000000u, 0x00002000u, 0x00000004u, 0x04000000u, 0x40004040u, 0x20020008u, 0x08010020u, 0x10008010u, 0x00000010u, 0x10000000u, 0x00010000u, 0x00000020u}},
    {0x119u, 2u, {0x00000000u, 0x00000001u, 0x01000000u, 0x00001000u, 0x02000000u, 0x00002000u, 0x00000002u, 0x04000000u, 0x40004020u, 0x20020004u, 0x08010010u, 0x10008008u, 0x00000008u, 0x10000000u, 0x00010000u, 0x00000010u}},
    {0x119u, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x01000000u, 0x00002000u, 0x00000002u, 0x02000000u, 0x20004020u, 0x10020004u, 0x04010010u, 0x08008008u, 0x00000008u, 0x08000000u, 0x00010000u, 0x00000010u}},
    {0x119u, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x01000000u, 0x00001000u, 0x00000002u, 0x02000000u, 0x20002020u, 0x10010004u, 0x04008010u, 0x08004008u, 0x00000008u, 0x08000000u, 0x00008000u, 0x00000010u}},
    {0x119u, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x01000000u, 0x00001000u, 0x00000001u, 0x02000000u, 0x20002010u, 0x10010002u, 0x04008008u, 0x08004004u, 0x00000004u, 0x08000000u, 0x00008000u, 0x00000008u}},
    {0x11au, 1u, {0x00000001u, 0x00000002u, 0x01000000u, 0x00001000u, 0x00002000u, 0x02000000u, 0x00000004u, 0x04000000u, 0x00008008u, 0x00010010u, 0x00024040u, 0x00040020u, 0x08000000u, 0x00008000u, 0x00000010u, 0x10000000u}},
    {0x11au, 2u, {0x00000000u, 0x00000001u, 0x01000000u, 0x00001000u, 0x00000002u, 0x02000000u, 0x00002000u, 0x04000000u, 0x00008008u, 0x00010010u, 0x08020040u, 0x00044020u, 0x00000004u, 0x10000000u, 0x00008000u, 0x00000010u}},
    {0x11au, 4u, {0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x00000002u, 0x01000000u, 0x00002000u, 0x02000000u, 0x00008008u, 0x00010010u, 0x04020040u, 0x00044020u, 0x00000004u, 0x08000000u, 0x00008000u, 0x00000010u}},
    {0x11au, 8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00001000u, 0x01000000u, 0x00000002u, 0x02000000u, 0x00008008u, 0x00010010u, 0x04020040u, 0x00042020u, 0x00000004u, 0x08000000u, 0x00004000u, 0x00000008u}},
    {0x11au, 16u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x01000000u, 0x00001000u, 0x02000000u, 0x00008008u, 0x00010012u, 0x04020040u, 0x00042020u, 0x00000004u, 0x08000000u, 0x00004000u, 0x00000008u}},
}};

constexpr const Equation* FindEquation(std::uint32_t swizzleMode, std::uint32_t elementBytes) {
    for (const auto& equation : Equations) {
        if (equation.swizzleMode == swizzleMode && equation.elementBytes == elementBytes) return &equation;
    }
    return nullptr;
}

inline constexpr std::array<std::array<std::uint32_t, 8>, 5> TailLocations4KB{{
    {(32u << 16) | 0u, (16u << 16) | 32u, (0u << 16) | 48u, (0u << 16) | 32u, (16u << 16) | 16u, (16u << 16) | 0u, (0u << 16) | 16u, 0u},
    {(32u << 16) | 0u, (16u << 16) | 16u, (0u << 16) | 24u, (0u << 16) | 16u, (16u << 16) | 8u, (16u << 16) | 0u, (0u << 16) | 8u, 0u},
    {(16u << 16) | 0u, (8u << 16) | 16u, (0u << 16) | 24u, (0u << 16) | 16u, (8u << 16) | 8u, (8u << 16) | 0u, (0u << 16) | 8u, 0u},
    {(16u << 16) | 0u, (8u << 16) | 8u, (0u << 16) | 12u, (0u << 16) | 8u, (8u << 16) | 4u, (8u << 16) | 0u, (0u << 16) | 4u, 0u},
    {(8u << 16) | 0u, (4u << 16) | 8u, (0u << 16) | 12u, (0u << 16) | 8u, (4u << 16) | 4u, (4u << 16) | 0u, (0u << 16) | 4u, 0u},
}};

inline constexpr std::array<std::array<std::uint32_t, 3>, 5> ThickBlockLog2For4KB{{{4u, 4u, 4u}, {3u, 4u, 4u}, {3u, 4u, 3u}, {3u, 3u, 3u}, {2u, 3u, 3u}}};
inline constexpr std::array<std::array<std::uint32_t, 3>, 5> ThickBlockLog2For64KB{{{6u, 5u, 5u}, {5u, 5u, 5u}, {5u, 5u, 4u}, {5u, 4u, 4u}, {4u, 4u, 4u}}};
inline constexpr std::array<std::array<std::uint32_t, 2>, 5> ThickTailMicroBlock{{{8u, 4u}, {4u, 4u}, {4u, 4u}, {4u, 2u}, {2u, 2u}}};

constexpr bool MemoryFormat(std::uint32_t format) {
    return format <= LastMemoryFormat && StorageImagePages::ElementSizeCode(format) != 0u;
}

constexpr std::uint32_t ElementLog2(std::uint32_t format) {
    return StorageImagePages::ElementSizeCode(format) - 1u;
}

constexpr bool DepthSwizzle(std::uint32_t tileMode) {
    return tileMode == 4u || tileMode == 8u || tileMode == 16u || tileMode == 20u || tileMode == 24u;
}

constexpr bool ThickTile(std::uint32_t tileMode) {
    return tileMode == 5u || tileMode == 9u || tileMode == 25u || tileMode == 26u;
}

constexpr std::uint32_t ThickEquationMode(std::uint32_t tileMode) {
    return 0x100u | tileMode;
}

constexpr bool Arrayed(std::uint32_t type) {
    return type == 11u || type == 12u || type == 13u;
}

constexpr Fields Decode(const std::array<std::uint32_t, 8>& words) {
    Fields fields;
    fields.base = (static_cast<std::uint64_t>(words[0]) | static_cast<std::uint64_t>(words[1] & 0xffu) << 32u) << 8u;
    fields.format = (words[1] >> 20u) & 0x1ffu;
    fields.width = (((words[1] >> 30u) & 0x3u) | ((words[2] & 0xfffu) << 2u)) + 1u;
    fields.height = ((words[2] >> 14u) & 0x3fffu) + 1u;
    fields.baseLevel = (words[3] >> 12u) & 0xfu;
    fields.lastLevel = (words[3] >> 16u) & 0xfu;
    fields.tileMode = (words[3] >> 20u) & 0x1fu;
    fields.type = words[3] >> 28u;
    fields.depth = words[4] & 0x1fffu;
    fields.baseArray = (words[4] >> 16u) & 0x1fffu;
    fields.arrayPitch = words[5] & 0xfu;
    fields.maxMip = (words[5] >> 4u) & 0xfu;
    fields.metaCompress = ((words[6] >> 21u) & 0x1u) != 0u;
    if (StorageImagePages::XorMode(fields.tileMode)) {
        fields.xorBits = static_cast<std::uint32_t>(fields.base & (fields.tileMode == 21u || fields.tileMode == 22u ? 0xfffu : 0xffffu));
        fields.base -= fields.xorBits;
    }
    return fields;
}

inline Layout LayoutOf(const Fields& fields) {
    if (!MemoryFormat(fields.format)) return Layout::Unsupported;
    const auto bytes = 1u << ElementLog2(fields.format);
    if (fields.tileMode == 0u) return Layout::Linear;
    if (fields.type == 10u && ThickTile(fields.tileMode)) return FindEquation(ThickEquationMode(fields.tileMode), bytes) != nullptr ? Layout::Thick : Layout::Unsupported;
    if (fields.type == 10u && fields.tileMode != 24u && fields.tileMode != 27u) return Layout::Unsupported;
    return FindEquation(fields.tileMode, bytes) != nullptr ? Layout::Thin : Layout::Unsupported;
}

constexpr std::uint32_t BlockLog2(const Fields& fields, Layout layout) {
    if (layout == Layout::Linear) return 8u;
    if (layout == Layout::Thick) return fields.tileMode == 5u ? 12u : 16u;
    return StorageImagePages::BlockLog2(StorageImagePages::ClassOf(fields.tileMode));
}

constexpr bool DepthLayout(const Fields& fields) {
    if (!DepthSwizzle(fields.tileMode) || (fields.type != 9u && fields.type != 13u) || !MemoryFormat(fields.format)) return false;
    const auto element = ElementLog2(fields.format);
    return element == 1u || element == 2u;
}

inline bool SurfaceThroughMemory(const std::array<std::uint32_t, 8>& words, Access access) {
    const auto fields = Decode(words);
    if (fields.base == 0u || fields.metaCompress || fields.arrayPitch != 0u || fields.type < 8u || fields.type > 13u) return false;
    if (fields.baseLevel > fields.maxMip || fields.baseLevel > fields.lastLevel) return false;
    if (fields.type == 11u && fields.width != fields.height) return false;
    if (access != Access::Load && DepthLayout(fields)) return false;
    const auto layout = LayoutOf(fields);
    if (layout == Layout::Unsupported) return false;
    if ((fields.base & ((std::uint64_t{1} << BlockLog2(fields, layout)) - 1u)) != 0u) return false;
    if (access == Access::Atomic32) return fields.format == 20u || fields.format == 21u || fields.format == 22u;
    if (access == Access::Atomic64) return fields.format == 62u || fields.format == 63u || fields.format == 64u;
    return true;
}

inline bool RuntimeResourceThroughMemory(const ShaderInfo& info, const ImageResource& image) {
    return info.usesDma && MemoryPathRuntimeImage(image) && (image.resourceClass != ImageResourceClass::Storage || info.bdaWrites);
}

inline Access AccessOf(const ImageResource& image) {
    if (image.resourceClass == ImageResourceClass::Sampled) return Access::Load;
    if (image.atomic64) return Access::Atomic64;
    if (image.atomic) return Access::Atomic32;
    return Access::Store;
}

inline std::uint32_t EquationOffset(const Equation& equation, std::uint32_t x, std::uint32_t y, std::uint32_t slice) {
    std::uint32_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16u; ++bit) {
        const auto mask = equation.bits[bit];
        const auto selected = (x & mask & 0xfffu) ^ ((y << 12u) & mask & 0xfff000u) ^ ((slice << 24u) & mask & 0xff000000u);
        offset |= (static_cast<std::uint32_t>(std::popcount(selected)) & 1u) << bit;
    }
    return offset;
}

inline Texel Locate(const std::array<std::uint32_t, 8>& words, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t lod) {
    using StorageImagePages::AlignUp;
    using StorageImagePages::ShiftCeil;
    const auto fields = Decode(words);
    if (fields.type == 8u) y = 0u;
    const auto layout = LayoutOf(fields);
    if (layout == Layout::Unsupported) return {};
    const auto element = ElementLog2(fields.format);
    const auto mipCount = fields.maxMip + 1u;
    const auto level = fields.baseLevel + lod;
    if (lod > 15u || level > std::min(fields.lastLevel, fields.maxMip)) return {};
    if (x >= std::max(fields.width >> level, 1u) || y >= std::max(fields.height >> level, 1u)) return {};
    std::uint32_t slice = 0;
    if (Arrayed(fields.type)) {
        const auto lastLayer = fields.type == 11u && fields.baseArray == fields.depth ? fields.depth + 5u : fields.depth;
        if (z > lastLayer || fields.baseArray + z > lastLayer) return {};
        slice = fields.baseArray + z;
    } else if (fields.type == 10u) {
        if (z >= std::max((fields.depth + 1u) >> level, 1u)) return {};
        slice = z;
    }
    if (layout == Layout::Linear) {
        const auto rowElements = 256u >> element;
        std::uint64_t total = 0;
        std::uint64_t levelOffset = 0;
        std::uint64_t pitch = 0;
        for (auto current = mipCount; current-- > 0u;) {
            const auto padded = AlignUp(std::max(ShiftCeil(fields.width, current), 1u), rowElements);
            const auto rows = std::max(ShiftCeil(fields.height, current), 1u);
            if (current == level) {
                levelOffset = total;
                pitch = static_cast<std::uint64_t>(padded) << element;
            }
            total += static_cast<std::uint64_t>(padded) * rows << element;
        }
        return {true, fields.base + slice * total + levelOffset + y * pitch + (static_cast<std::uint64_t>(x) << element)};
    }
    const auto bytes = 1u << element;
    if (layout == Layout::Thick) {
        const auto blockLog2 = BlockLog2(fields, layout);
        const auto& extent = blockLog2 == 12u ? ThickBlockLog2For4KB[element] : ThickBlockLog2For64KB[element];
        const auto tailWidth = blockLog2 % 3u == 1u ? (1u << extent[0]) >> 1u : 1u << extent[0];
        const auto tailHeight = blockLog2 % 3u == 0u ? (1u << extent[1]) >> 1u : 1u << extent[1];
        const auto effective = blockLog2 - (blockLog2 - 8u) / 3u;
        const auto maxTailLevels = effective <= 11u ? 1u + (1u << (effective - 9u)) : effective - 4u;
        auto firstTail = mipCount;
        for (std::uint32_t current = 0; mipCount > 1u && current < mipCount; ++current) {
            if (ShiftCeil(fields.width, current) <= tailWidth && ShiftCeil(fields.height, current) <= tailHeight && mipCount - current <= maxTailLevels) {
                firstTail = current;
                break;
            }
        }
        std::uint64_t offset = firstTail < mipCount ? std::uint64_t{1} << blockLog2 : 0u;
        std::uint64_t levelOffset = 0;
        std::uint32_t blocksPerRow = 0;
        for (auto current = firstTail; current-- > 0u;) {
            const auto columns = AlignUp(std::max(ShiftCeil(fields.width, current), 1u), 1u << extent[0]) >> extent[0];
            const auto rows = AlignUp(std::max(ShiftCeil(fields.height, current), 1u), 1u << extent[1]) >> extent[1];
            if (current == level) {
                levelOffset = offset;
                blocksPerRow = columns;
            }
            offset += static_cast<std::uint64_t>(columns) * rows << blockLog2;
        }
        auto address = fields.base + static_cast<std::uint64_t>(z >> extent[2]) * offset;
        auto swizzleX = x;
        auto swizzleY = y;
        if (level < firstTail) {
            address += levelOffset + ((static_cast<std::uint64_t>(y >> extent[1]) * blocksPerRow + (x >> extent[0])) << blockLog2);
        } else {
            const auto slot = maxTailLevels - 1u - (level - firstTail);
            const auto slotOffset = slot > 6u ? 16u << slot : slot << 8u;
            std::uint32_t tailX = 0;
            std::uint32_t tailY = 0;
            for (std::uint32_t bit = 0; bit < 6u; ++bit) {
                tailX |= ((slotOffset >> (9u + 2u * bit)) & 1u) << bit;
                tailY |= ((slotOffset >> (8u + 2u * bit)) & 1u) << bit;
            }
            swizzleX += tailX * ThickTailMicroBlock[element][0];
            swizzleY += tailY * ThickTailMicroBlock[element][1];
        }
        const auto* equation = FindEquation(ThickEquationMode(fields.tileMode), bytes);
        return {true, address + (EquationOffset(*equation, swizzleX, swizzleY, z) ^ fields.xorBits)};
    }
    const auto blockLog2 = BlockLog2(fields, layout);
    const auto widthLog2 = (blockLog2 - element + 1u) / 2u;
    const auto heightLog2 = (blockLog2 - element) / 2u;
    auto widthLimit = (1u << widthLog2) >> 1u;
    auto heightLimit = 1u << heightLog2;
    if (fields.tileMode == 24u && element <= 1u) {
        widthLimit >>= 1u;
        if (element == 0u) heightLimit >>= 1u;
    }
    const auto maxTailLevels = blockLog2 == 12u ? 8u : 12u;
    auto firstTail = mipCount;
    if (blockLog2 != 8u && mipCount > 1u) {
        for (std::uint32_t current = 0; current < mipCount; ++current) {
            if (ShiftCeil(fields.width, current) <= widthLimit && ShiftCeil(fields.height, current) <= heightLimit && mipCount - current <= maxTailLevels) {
                firstTail = current;
                break;
            }
        }
    }
    std::uint64_t offset = firstTail < mipCount ? std::uint64_t{1} << blockLog2 : 0u;
    std::uint64_t levelOffset = 0;
    std::uint32_t blocksPerRow = 0;
    for (auto current = firstTail; current-- > 0u;) {
        const auto paddedWidth = AlignUp(std::max(ShiftCeil(fields.width, current), 1u), 1u << widthLog2);
        const auto paddedHeight = AlignUp(std::max(ShiftCeil(fields.height, current), 1u), 1u << heightLog2);
        if (current == level) {
            levelOffset = offset;
            blocksPerRow = paddedWidth >> widthLog2;
        }
        offset += static_cast<std::uint64_t>(paddedWidth) * paddedHeight << element;
    }
    auto address = fields.base + static_cast<std::uint64_t>(slice) * offset;
    auto swizzleX = x;
    auto swizzleY = y;
    if (level < firstTail) {
        address += levelOffset + ((static_cast<std::uint64_t>(y >> heightLog2) * blocksPerRow + (x >> widthLog2)) << blockLog2);
    } else {
        const auto location = blockLog2 == 12u ? TailLocations4KB[element][level - firstTail] : StorageImagePages::TailLocations64KB[element][level - firstTail];
        swizzleX += location >> 16u;
        swizzleY += location & 0xffffu;
    }
    const auto* equation = FindEquation(fields.tileMode, bytes);
    return {true, address + (EquationOffset(*equation, swizzleX, swizzleY, slice) ^ fields.xorBits)};
}

}

#endif
