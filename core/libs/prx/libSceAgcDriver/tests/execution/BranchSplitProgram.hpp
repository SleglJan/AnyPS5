#pragma once

#include <cstdint>
#include <vector>

namespace BranchSplitProgram {

enum class Condition {
    Scc,
    Vcc,
};

constexpr std::uint32_t ArmA = 0;
constexpr std::uint32_t ArmB = 1;
constexpr std::uint32_t ArmC = 2;

inline std::uint32_t Constant(std::uint32_t arm, std::uint32_t index) {
    return (index * 7u + arm * 13u) % 63u + 1u;
}

inline void AppendArm(std::vector<std::uint32_t>& code, std::uint32_t arm, std::uint32_t operations) {
    code.push_back(0x7e040301u);
    for (std::uint32_t index = 0; index < operations; ++index) {
        const auto constant = 128u + Constant(arm, index);
        switch (index % 3u) {
        case 0: code.push_back((0x25u << 25u) | (2u << 17u) | (2u << 9u) | constant); break;
        case 1: code.push_back((0x1du << 25u) | (2u << 17u) | (2u << 9u) | 257u); break;
        default: code.push_back((0x0bu << 25u) | (2u << 17u) | (2u << 9u) | constant); break;
        }
    }
}

inline void AppendStore(std::vector<std::uint32_t>& code) {
    code.insert(code.end(), {0xe0701000u, 0x80060203u, 0xbf810000u});
}

inline std::uint32_t Expected(std::uint32_t arm, std::uint32_t operations, std::uint32_t thread) {
    std::uint32_t value = thread;
    for (std::uint32_t index = 0; index < operations; ++index) {
        const auto constant = Constant(arm, index);
        switch (index % 3u) {
        case 0: value += constant; break;
        case 1: value ^= thread; break;
        default: value = (value & 0xffffffu) * constant; break;
        }
    }
    return value;
}

inline std::uint32_t Branch(std::uint32_t op, std::size_t at, std::size_t target) {
    return 0xbf800000u | (op << 16u) | (static_cast<std::uint32_t>(static_cast<std::int64_t>(target) - static_cast<std::int64_t>(at) - 1) & 0xffffu);
}

inline std::vector<std::uint32_t> Prologue() {
    return {0xf4080600u, 0xfa000000u, 0xbf8cc07fu, 0x8f038602u, 0x4a020003u, 0x34060282u};
}

inline std::vector<std::uint32_t> Code(Condition condition, std::uint32_t operations) {
    auto code = Prologue();
    code.push_back(0xbf0d8002u);
    const auto first = code.size();
    code.push_back(0u);
    AppendArm(code, ArmA, operations);
    AppendStore(code);
    const auto second = code.size();
    if (condition == Condition::Scc) {
        code.push_back(0xbf0d8102u);
    } else {
        code.insert(code.end(), {0x7e080280u, 0x87058202u, 0x7d8a0805u});
    }
    const auto site = code.size();
    code.push_back(0u);
    AppendArm(code, ArmB, operations);
    AppendStore(code);
    const auto armC = code.size();
    AppendArm(code, ArmC, operations);
    AppendStore(code);
    code[first] = Branch(5u, first, second);
    code[site] = Branch(condition == Condition::Scc ? 5u : 7u, site, armC);
    return code;
}

inline std::uint32_t ArmOf(std::uint32_t group) {
    if ((group & 1u) == 0u) return ArmA;
    return (group & 2u) == 0u ? ArmB : ArmC;
}

}
