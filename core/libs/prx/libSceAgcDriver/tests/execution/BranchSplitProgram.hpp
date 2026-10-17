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

inline void AppendOperations(std::vector<std::uint32_t>& code, std::uint32_t arm, std::uint32_t operations) {
    for (std::uint32_t index = 0; index < operations; ++index) {
        const auto constant = 128u + Constant(arm, index);
        switch (index % 3u) {
        case 0: code.push_back((0x25u << 25u) | (2u << 17u) | (2u << 9u) | constant); break;
        case 1: code.push_back((0x1du << 25u) | (2u << 17u) | (2u << 9u) | 257u); break;
        default: code.push_back((0x0bu << 25u) | (2u << 17u) | (2u << 9u) | constant); break;
        }
    }
}

inline void AppendArm(std::vector<std::uint32_t>& code, std::uint32_t arm, std::uint32_t operations) {
    code.push_back(0x7e040301u);
    AppendOperations(code, arm, operations);
}

inline void AppendStore(std::vector<std::uint32_t>& code) {
    code.insert(code.end(), {0xe0701000u, 0x80060203u, 0xbf810000u});
}

inline std::uint32_t Apply(std::uint32_t value, std::uint32_t arm, std::uint32_t operations, std::uint32_t thread) {
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

inline std::uint32_t Expected(std::uint32_t arm, std::uint32_t operations, std::uint32_t thread) {
    return Apply(thread, arm, operations, thread);
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

constexpr std::uint32_t ArmD = 3;
constexpr std::uint32_t CalleeArm = 5;

inline std::vector<std::uint32_t> TreeCode(std::uint32_t operations) {
    auto code = Prologue();
    code.push_back(0xbf0d8002u);
    const auto root = code.size();
    code.push_back(0u);
    code.push_back(0xbf0d8102u);
    const auto left = code.size();
    code.push_back(0u);
    AppendArm(code, ArmA, operations);
    AppendStore(code);
    const auto hop = code.size();
    code.push_back(0u);
    const auto armB = code.size();
    AppendArm(code, ArmB, operations);
    AppendStore(code);
    const auto right = code.size();
    code.push_back(0xbf0d8102u);
    const auto rightSite = code.size();
    code.push_back(0u);
    AppendArm(code, ArmC, operations);
    AppendStore(code);
    const auto armD = code.size();
    AppendArm(code, ArmD, operations);
    AppendStore(code);
    code[root] = Branch(5u, root, hop);
    code[hop] = Branch(2u, hop, right);
    code[left] = Branch(5u, left, armB);
    code[rightSite] = Branch(5u, rightSite, armD);
    return code;
}

inline std::uint32_t TreeArmOf(std::uint32_t group) {
    if ((group & 1u) == 0u) return (group & 2u) == 0u ? ArmA : ArmB;
    return (group & 2u) == 0u ? ArmC : ArmD;
}

inline std::vector<std::uint32_t> CalleeCode(std::uint32_t operations) {
    std::vector<std::uint32_t> code;
    AppendOperations(code, CalleeArm, operations);
    code.push_back(0xbe80201eu);
    return code;
}

struct Caller {
    std::vector<std::uint32_t> code;
    std::vector<std::size_t> calls;
};

inline Caller CallerCode(std::uint32_t operations, bool firstArmCalls = true) {
    Caller caller;
    auto& code = caller.code;
    code = {0xf4080600u, 0xfa000000u, 0xbf8cc07fu, 0x8f058604u, 0x4a020005u, 0x34060282u, 0xbf0d8004u};
    const auto first = code.size();
    code.push_back(0u);
    AppendArm(code, ArmA, operations);
    if (firstArmCalls) {
        caller.calls.push_back(code.size());
        code.push_back(0xbe9e2102u);
    }
    AppendStore(code);
    const auto second = code.size();
    code.push_back(0xbf0d8104u);
    const auto site = code.size();
    code.push_back(0u);
    AppendArm(code, ArmB, operations);
    AppendStore(code);
    const auto armC = code.size();
    AppendArm(code, ArmC, operations);
    caller.calls.push_back(code.size());
    code.push_back(0xbe9e2102u);
    AppendStore(code);
    code[first] = Branch(5u, first, second);
    code[site] = Branch(5u, site, armC);
    return caller;
}

inline std::uint32_t CallerExpected(std::uint32_t group, std::uint32_t operations, std::uint32_t calleeOperations, bool firstArmCalls, std::uint32_t thread) {
    const auto arm = ArmOf(group);
    const auto value = Expected(arm, operations, thread);
    return arm == ArmB || (arm == ArmA && !firstArmCalls) ? value : Apply(value, CalleeArm, calleeOperations, thread);
}

}
