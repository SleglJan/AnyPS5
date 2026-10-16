#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
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

constexpr std::uint64_t Output = 0x1100050000ull;

void Require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<std::uint32_t> Recompiled(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData) {
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    static constexpr std::array<std::uint32_t, 8> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityGroupNonUniform,
        spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformArithmetic};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x1100040000ull, code, 0, {}};
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
    const auto result = Recompile(request);
    Require(!result.spirv.empty(), "the program did not recompile");
    return result.spirv.Copy();
}

struct Function {
    std::uint32_t resultType = 0;
    std::uint32_t control = 0;
    std::size_t valueParameters = 0;
    std::size_t returnValues = 0;
    std::set<std::uint32_t> privateStores;
    std::set<std::uint32_t> privateLoads;
};

struct Module {
    std::uint32_t entry = 0;
    std::uint32_t voidType = 0;
    std::map<std::uint32_t, std::string> names;
    std::set<std::uint32_t> pointerTypes;
    std::set<std::uint32_t> privates;
    std::map<std::uint32_t, Function> functions;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> calls;
    std::map<std::uint32_t, std::uint32_t> callResults;
    std::map<std::uint32_t, std::uint32_t> spillLoads;
    std::set<std::uint32_t> extracted;
};

Module Parse(std::span<const std::uint32_t> words) {
    Module module;
    std::uint32_t function = 0;
    std::uint32_t lastCall = 0;
    std::uint32_t lastCallStore = 0;
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = words[cursor] & 0xffffu;
        if (count == 0 || cursor + count > words.size()) throw std::runtime_error("truncated SPIR-V instruction");
        const auto operand = [&](std::size_t index) { return words[cursor + 1u + index]; };
        if (op == spv::OpName) {
            module.names[operand(0)] = reinterpret_cast<const char*>(words.data() + cursor + 2u);
        } else if (op == spv::OpEntryPoint) {
            module.entry = operand(1);
        } else if (op == spv::OpTypeVoid) {
            module.voidType = operand(0);
        } else if (op == spv::OpTypePointer) {
            module.pointerTypes.insert(operand(0));
        } else if (op == spv::OpVariable && function == 0u && operand(2) == spv::StorageClassPrivate) {
            module.privates.insert(operand(1));
        } else if (op == spv::OpFunction) {
            function = operand(1);
            module.functions[function].resultType = operand(0);
            module.functions[function].control = operand(2);
        } else if (op == spv::OpFunctionEnd) {
            function = 0;
        } else if (op == spv::OpFunctionParameter && !module.pointerTypes.contains(operand(0))) {
            ++module.functions[function].valueParameters;
        } else if (op == spv::OpReturnValue) {
            ++module.functions[function].returnValues;
        } else if (op == spv::OpFunctionCall) {
            module.calls.emplace_back(function, operand(2));
            module.callResults[operand(1)] = operand(2);
            lastCall = operand(1);
        } else if (op == spv::OpStore && module.privates.contains(operand(0))) {
            module.functions[function].privateStores.insert(operand(0));
        } else if (op == spv::OpLoad && module.privates.contains(operand(2))) {
            module.functions[function].privateLoads.insert(operand(2));
        } else if (op == spv::OpCompositeExtract) {
            module.extracted.insert(operand(2));
        }
        if (op == spv::OpStore && count == 4u && operand(1) == lastCall) lastCallStore = operand(0);
        if (op == spv::OpLoad && count == 5u && lastCallStore != 0u && operand(2) == lastCallStore) module.spillLoads[operand(1)] = lastCall;
        if (op != spv::OpFunctionCall && op != spv::OpStore) lastCall = 0;
        if (op != spv::OpStore) lastCallStore = 0;
        cursor += count;
    }
    return module;
}

void CheckNestedCallResults() {
    const std::vector<std::uint32_t> code{0x34020084u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u, 0xbe8a1f00u, 0x800aff0au, 0x00000014u, 0x820b800bu, 0xbe88210au, 0xbf82000bu, 0x4a0808ffu, 0x13579bdfu,
        0xbb0c0001u, 0xbf820006u, 0x3a0a0affu, 0xa5a5a5a5u, 0x4a0c0f06u, 0x3a0e0effu, 0x5a5a5a5au, 0xbe80200cu, 0xbe802008u, 0xe0781000u, 0x80010401u, 0xbf810000u};
    const std::array<std::uint32_t, 8> userData{static_cast<std::uint32_t>(Output), static_cast<std::uint32_t>(Output >> 32u) & 0xffffu, 0x400u, 0x31016facu, static_cast<std::uint32_t>(Output + 0x1000u), static_cast<std::uint32_t>(Output >> 32u) & 0xffffu, 0x400u, 0x31016facu};
    const auto module = Parse(Recompiled(code, userData));
    std::vector<std::uint32_t> callees;
    for (const auto& [id, name] : module.names) {
        if (name.starts_with("guest_callee_") && module.functions.contains(id)) callees.push_back(id);
    }
    Require(callees.size() == 2u, "a static call inside a static callee: " + std::to_string(callees.size()) + " callee functions, expected 2");
    std::size_t parameters = 0;
    for (const auto callee : callees) {
        const auto& function = module.functions.at(callee);
        const auto& name = module.names.at(callee);
        Require((function.control & spv::FunctionControlDontInlineMask) != 0u, name + " is not DontInline");
        Require(function.resultType != module.voidType && function.returnValues != 0u, name + " does not return the registers it writes");
        parameters += function.valueParameters;
        for (const auto& [caller, called] : module.calls) {
            if (called != callee) continue;
            const auto& outer = module.functions.at(caller);
            for (const auto variable : function.privateStores) Require(!outer.privateLoads.contains(variable), name + " hands a register back to its caller through Private memory");
            for (const auto variable : outer.privateStores) Require(!function.privateLoads.contains(variable), name + " takes a register from its caller through Private memory");
        }
        bool used = false;
        for (const auto& [result, called] : module.callResults) {
            if (called == callee && module.extracted.contains(result)) used = true;
        }
        for (const auto& [loaded, call] : module.spillLoads) {
            if (module.callResults.contains(call) && module.callResults.at(call) == callee && module.extracted.contains(loaded)) used = true;
        }
        Require(used, "the caller of " + name + " does not take the registers it wrote from its result");
    }
    Require(parameters != 0u, "no callee takes the registers it reads as parameters");
}

}

int main() {
    try {
        CheckNestedCallResults();
    } catch (const std::exception& error) {
        std::cerr << "SpirvCalleeResults: " << error.what() << '\n';
        return 1;
    }
    std::puts("callee result tests passed");
    return 0;
}
