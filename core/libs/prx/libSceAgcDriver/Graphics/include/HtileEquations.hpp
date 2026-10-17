#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HTILEEQUATIONS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HTILEEQUATIONS_HPP

#include <array>
#include <bit>
#include <cstdint>

namespace AgcDriver::Graphics {

struct HtileEquation {
    std::uint32_t samples;
    std::array<std::uint32_t, 16> nibbles;
};

inline constexpr HtileEquation kHtileEquations[] = {
    {1u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000008u, 0x00010000u, 0x00000040u, 0x00040000u, 0x00000080u, 0x00080000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00000100u, 0x00100000u, 0x00000200u}},
    {2u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000008u, 0x00010000u, 0x00000040u, 0x00040000u, 0x00000080u, 0x00080000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00000100u, 0x00100000u, 0x00000200u}},
    {4u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000008u, 0x00010000u, 0x00000040u, 0x00040000u, 0x00000080u, 0x00080000u, 0x08008008u, 0x04010010u, 0x02020040u, 0x01040020u, 0x00000100u, 0x00100000u, 0x00000200u}},
    {8u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000008u, 0x00010000u, 0x00000040u, 0x00040000u, 0x00000080u, 0x00080000u, 0x04008008u, 0x02010010u, 0x01020040u, 0x00040020u, 0x00000100u, 0x00100000u, 0x00000200u}},
};

inline constexpr std::uint32_t HtileBlockWidth = 1024u;
inline constexpr std::uint32_t HtileBlockHeight = 512u;
inline constexpr std::uint64_t HtileBlockBytes = 32768u;

constexpr const HtileEquation* FindHtileEquation(std::uint32_t samples) {
    for (const auto& equation : kHtileEquations) {
        if (equation.samples == samples) return &equation;
    }
    return nullptr;
}

constexpr std::uint64_t HtileWordOffset(const HtileEquation& equation, std::uint32_t width, std::uint32_t x, std::uint32_t y, std::uint32_t pipeBankXor) {
    std::uint32_t nibble = 0;
    const auto coordinates = (x & 0xfffu) | ((y & 0xfffu) << 12u);
    for (std::uint32_t bit = 0; bit < equation.nibbles.size(); ++bit) nibble |= static_cast<std::uint32_t>(std::popcount(equation.nibbles[bit] & coordinates) & 1) << bit;
    const auto pitch = (width + HtileBlockWidth - 1u) / HtileBlockWidth;
    const auto block = static_cast<std::uint64_t>(y / HtileBlockHeight) * pitch + x / HtileBlockWidth;
    const auto pipeXor = static_cast<std::uint32_t>(pipeBankXor & 0x0f00u);
    return block * HtileBlockBytes + (((nibble >> 1u) ^ pipeXor) & static_cast<std::uint32_t>(HtileBlockBytes - 1u));
}

}

#endif
