#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

struct Function {
    std::string name;
    bool subgroup = false;
    bool loopSubgroup = false;
    std::set<std::uint32_t> callees;
    std::set<std::uint32_t> loopCallees;
};

std::vector<std::uint32_t> Recompiled(std::span<const std::uint32_t> code) {
    const std::array<std::uint32_t, 12> userData{0x10000000u, 0x00000000u, 0x100u, 0x31016facu, 0x20000000u, 0x00000000u, 0x100u, 0x31016facu, 0x30000000u, 0x00000000u, 0x100u, 0x31016facu};
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    static constexpr std::array<std::uint32_t, 8> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityGroupNonUniform,
        spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformArithmetic};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x20000u, code, 0, {}};
    request.context.waveSize = 32;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    auto result = Recompile(request);
    if (result.spirv.empty()) throw std::runtime_error("the shader did not recompile");
    return result.spirv.Copy();
}

bool SubgroupOpcode(std::uint32_t opcode) {
    return (opcode >= spv::OpGroupNonUniformElect && opcode <= spv::OpGroupNonUniformQuadSwap) || (opcode >= spv::OpSubgroupBallotKHR && opcode <= spv::OpSubgroupReadInvocationKHR) ||
        opcode == spv::OpGroupNonUniformRotateKHR;
}

std::map<std::uint32_t, Function> Parse(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, Function> functions;
    std::map<std::uint32_t, std::string> names;
    std::uint32_t current = 0;
    std::uint32_t entry = 0;
    std::vector<std::uint32_t> openLoops;
    for (std::size_t at = 5; at < words.size();) {
        const auto opcode = words[at] & 0xffffu;
        const auto count = words[at] >> 16u;
        if (count == 0u || at + count > words.size()) throw std::runtime_error("malformed SPIR-V");
        const auto* operands = &words[at + 1u];
        if (opcode == spv::OpEntryPoint) entry = operands[1];
        if (opcode == spv::OpName) names[operands[0]] = reinterpret_cast<const char*>(&operands[1]);
        if (opcode == spv::OpFunction) {
            current = operands[1];
            functions[current].name = names[current];
            openLoops.clear();
        }
        if (opcode == spv::OpFunctionEnd) current = 0;
        if (current != 0u) {
            auto& function = functions[current];
            if (opcode == spv::OpLoopMerge) openLoops.push_back(operands[0]);
            if (opcode == spv::OpLabel) {
                for (std::size_t loop = 0; loop < openLoops.size(); ++loop) {
                    if (openLoops[loop] != operands[0]) continue;
                    openLoops.resize(loop);
                    break;
                }
            }
            if (SubgroupOpcode(opcode)) {
                function.subgroup = true;
                if (!openLoops.empty() && current != entry) function.loopSubgroup = true;
            }
            if (opcode == spv::OpFunctionCall) {
                function.callees.insert(operands[2]);
                if (!openLoops.empty()) function.loopCallees.insert(operands[2]);
            }
        }
        at += count;
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& [id, function] : functions) {
            if (function.subgroup) continue;
            for (const auto callee : function.callees) {
                const auto found = functions.find(callee);
                if (found == functions.end() || !found->second.subgroup) continue;
                function.subgroup = true;
                changed = true;
                break;
            }
        }
    }
    return functions;
}

std::vector<std::string> LoopCallsIntoSubgroupHelpers(const std::map<std::uint32_t, Function>& functions) {
    std::vector<std::string> found;
    for (const auto& [id, function] : functions) {
        for (const auto callee : function.loopCallees) {
            const auto target = functions.find(callee);
            if (target != functions.end() && target->second.subgroup) found.push_back(function.name + " -> " + target->second.name);
        }
        if (function.loopSubgroup) found.push_back(function.name + " (a loop of its own)");
    }
    return found;
}

}

int main() {
    try {
        const std::vector<std::uint32_t> code{
            0x34020084u,
            0x7e200281u,
            0xe03c1000u, 0x80000201u,
            0xe03c1010u, 0x80000501u,
            0xe03c1020u, 0x80000801u,
            0xe0301000u, 0x80010b01u,
            0xe0301004u, 0x80010c01u,
            0xf4200502u, 0xfa000000u,
            0xf4200542u, 0xfa000004u,
            0xf4200582u, 0xfa000008u,
            0xe00c1000u, 0x80001401u,
            0xe0c81040u, 0x80011001u,
            0xe0c81044u, 0x80011001u,
            0xbf8c3f70u,
            0xbf8cc07fu,
            0x7e1a0214u,
            0x7e1c0215u,
            0x7e1e0216u,
            0xe07c1000u, 0x80020201u,
            0xe07c1010u, 0x80020501u,
            0xe07c1020u, 0x80020801u,
            0xe07c1060u, 0x80020d01u,
            0xe0741030u, 0x80020b01u,
            0xe0781040u, 0x80021401u,
            0xbf810000u};
        const auto functions = Parse(Recompiled(code));
        bool components = false;
        for (const auto& [id, function] : functions) components = components || function.name.starts_with("read_bda_components");
        if (!components) throw std::runtime_error("the module has no read_bda_components function, so the loop helpers are not exercised");
        const auto offending = LoopCallsIntoSubgroupHelpers(functions);
        if (!offending.empty()) {
            std::string list;
            for (const auto& call : offending) list += (list.empty() ? "" : ", ") + call;
            throw std::runtime_error("helpers that run subgroup operations are called from a loop: " + list);
        }
    } catch (const std::exception& error) {
        std::cerr << "SpirvSubgroupLoopHelpers: " << error.what() << '\n';
        return 1;
    }
    std::puts("SpirvSubgroupLoopHelpers: no helper with a subgroup operation is called from a loop");
    return 0;
}
