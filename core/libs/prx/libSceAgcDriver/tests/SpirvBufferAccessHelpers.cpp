#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    std::uint32_t control = 0;
    std::string name;
    std::size_t widestSwitch = 0;
    std::size_t switches = 0;
    std::size_t dynamicExtracts = 0;
    std::map<std::uint32_t, std::uint32_t> calls;
    std::size_t bufferMemoryLoads = 0;
};

struct Module {
    std::map<std::uint32_t, Function> functions;
    std::map<std::uint32_t, std::uint32_t> callsFromMain;
    std::map<std::uint32_t, std::uint32_t> branchedCallsFromMain;
    std::size_t mainBufferMemoryLoads = 0;
    std::map<std::uint32_t, std::size_t> mainBuiltinLoads;
    std::size_t mainPrivateWideLoads = 0;
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

Module Parse(std::span<const std::uint32_t> words) {
    Module module;
    std::uint32_t entry = 0;
    std::set<std::uint32_t> physicalPointerTypes;
    std::set<std::uint32_t> physicalPointers;
    std::uint32_t current = 0;
    std::uint32_t label = 0;
    std::uint32_t merge = 0;
    std::set<std::uint32_t> conditionalTargets;
    std::map<std::uint32_t, std::uint32_t> builtins;
    std::set<std::uint32_t> wideTypes;
    std::set<std::uint32_t> privateWidePointers;
    std::set<std::uint32_t> privateWide;
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = words[cursor] & 0xffffu;
        if (count == 0 || cursor + count > words.size()) throw std::runtime_error("truncated SPIR-V instruction");
        const auto* operands = words.data() + cursor + 1;
        if (op == spv::OpEntryPoint) entry = operands[1];
        if (op == spv::OpDecorate && count >= 4u && operands[1] == spv::DecorationBuiltIn) builtins[operands[0]] = operands[2];
        if (op == spv::OpName) {
            std::string name(reinterpret_cast<const char*>(operands + 1), (count - 2u) * 4u);
            module.functions[operands[0]].name = name.c_str();
        }
        if (op == spv::OpTypePointer && operands[1] == spv::StorageClassPhysicalStorageBuffer) physicalPointerTypes.insert(operands[0]);
        if (op == spv::OpTypeInt && operands[1] == 64u) wideTypes.insert(operands[0]);
        if (op == spv::OpTypePointer && operands[1] == spv::StorageClassPrivate && wideTypes.contains(operands[2])) privateWidePointers.insert(operands[0]);
        if (op == spv::OpVariable && privateWidePointers.contains(operands[0])) privateWide.insert(operands[1]);
        if ((op == spv::OpConvertUToPtr || op == spv::OpAccessChain || op == spv::OpPtrAccessChain) && physicalPointerTypes.contains(operands[0])) physicalPointers.insert(operands[1]);
        if (op == spv::OpFunction) {
            current = operands[1];
            module.functions[current].control = operands[2];
        }
        if (op == spv::OpFunctionEnd) current = 0;
        if (op == spv::OpLabel) label = operands[0];
        if (op == spv::OpSelectionMerge) merge = operands[0];
        if (op == spv::OpBranchConditional) {
            for (const auto target : {operands[1], operands[2]}) {
                if (target != merge) conditionalTargets.insert(target);
            }
        }
        if (current != 0 && op == spv::OpFunctionCall) ++module.functions[current].calls[operands[2]];
        if (current != 0 && op == spv::OpVectorExtractDynamic) ++module.functions[current].dynamicExtracts;
        if (current != 0 && (op == spv::OpLoad || op == spv::OpAtomicLoad) && physicalPointers.contains(operands[2])) ++module.functions[current].bufferMemoryLoads;
        if (current != 0 && op == spv::OpSwitch) {
            std::set<std::uint32_t> targets{operands[1]};
            for (std::uint32_t operand = 3; operand < count - 1u; operand += 2u) targets.insert(operands[operand]);
            ++module.functions[current].switches;
            auto& widest = module.functions[current].widestSwitch;
            widest = std::max(widest, targets.size());
        }
        if (current != 0 && current == entry) {
            if (op == spv::OpFunctionCall) ++module.callsFromMain[operands[2]];
            if (op == spv::OpFunctionCall && conditionalTargets.contains(label)) ++module.branchedCallsFromMain[operands[2]];
            if ((op == spv::OpLoad || op == spv::OpAtomicLoad) && physicalPointers.contains(operands[2])) ++module.mainBufferMemoryLoads;
            if (op == spv::OpLoad && builtins.contains(operands[2])) ++module.mainBuiltinLoads[builtins.at(operands[2])];
            if (op == spv::OpLoad && privateWide.contains(operands[2])) ++module.mainPrivateWideLoads;
        }
        cursor += count;
    }
    return module;
}

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint32_t> Calls(const Module& module, std::string_view prefix) {
    std::vector<std::uint32_t> calls;
    for (const auto& [id, function] : module.functions) {
        if (!function.name.starts_with(prefix)) continue;
        Expect((function.control & spv::FunctionControlDontInlineMask) != 0u, function.name + " is not DontInline");
        const auto found = module.callsFromMain.find(id);
        calls.push_back(found == module.callsFromMain.end() ? 0u : found->second);
    }
    return calls;
}

std::string Describe(const std::vector<std::uint32_t>& calls) {
    std::string text = std::to_string(calls.size()) + " helpers called";
    for (const auto count : calls) text += " " + std::to_string(count);
    return text + " times";
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
        const auto module = Parse(Recompiled(code));
        const auto loads = Calls(module, "load_buffer_");
        Expect(loads.size() == 2u && ((loads[0] == 3u && loads[1] == 2u) || (loads[0] == 2u && loads[1] == 3u)), "expected one load helper called by the three dwordx3 loads and one called by the two dword loads, found " + Describe(loads));
        const auto stores = Calls(module, "store_buffer_");
        Expect(stores.size() == 1u && stores[0] == 4u, "expected one store helper called by the four dwordx3 stores, found " + Describe(stores));
        const auto scalars = Calls(module, "load_scalar_buffer_");
        Expect(scalars.size() == 1u && scalars[0] == 3u, "expected one scalar load helper called by the three s_buffer_load_dword, found " + Describe(scalars));
        const auto atomics = Calls(module, "buffer_atomic_address_");
        Expect(atomics.size() == 1u && atomics[0] == 2u, "expected one atomic address helper called by the two buffer_atomic_add, found " + Describe(atomics));
        const auto formatted = Calls(module, "load_formatted_buffer_");
        Expect(formatted.size() == 1u && formatted[0] == 1u, "expected one formatted load helper called once, found " + Describe(formatted));
        for (const auto builtin : {spv::BuiltInSubgroupLocalInvocationId, spv::BuiltInLocalInvocationIndex}) {
            const auto found = module.mainBuiltinLoads.find(builtin);
            const auto loads = found == module.mainBuiltinLoads.end() ? 0u : found->second;
            Expect(loads <= 1u, "main loads built-in " + std::to_string(builtin) + " " + std::to_string(loads) + " times, expected once at its entry");
        }
        Expect(module.mainPrivateWideLoads == 3u, "main loads the read cache " + std::to_string(module.mainPrivateWideLoads) + " times, expected one probe of its begin, end and offset for the three consecutive s_buffer_load_dword");
        Expect(module.mainBufferMemoryLoads == 1u, "main loads buffer memory " + std::to_string(module.mainBufferMemoryLoads) + " times, expected one vector load for the three consecutive s_buffer_load_dword, through their cached translation, with the refill left to the helper");
        std::uint32_t write = 0;
        for (const auto& [id, function] : module.functions) {
            if (function.name == "write_bda") write = id;
        }
        Expect(write != 0u, "the module has no write_bda function");
        std::uint32_t segment = 0;
        for (const auto& [id, function] : module.functions) {
            if (function.name == "bda_segment") segment = id;
        }
        Expect(segment != 0u, "the module has no bda_segment function");
        const auto mainRefreshes = [&](const Module& checked) {
            const auto found = checked.callsFromMain.find(segment);
            return found == checked.callsFromMain.end() ? 0u : found->second;
        };
        Expect(mainRefreshes(module) == 0u, "main refreshes the read segment " + std::to_string(mainRefreshes(module)) + " times for its three consecutive s_buffer_load_dword, expected their helpers to refill it");
        std::size_t spanReaders = 0;
        for (const auto& [id, function] : module.functions) {
            if (!function.name.starts_with("read_bda_span")) continue;
            ++spanReaders;
            const auto found = function.calls.find(segment);
            const auto refreshes = found == function.calls.end() ? 0u : found->second;
            Expect(refreshes == 1u && function.bufferMemoryLoads != 0u, function.name + " refreshes the invocation's read segment " + std::to_string(refreshes) + " times and loads buffer memory " + std::to_string(function.bufferMemoryLoads) + " times, expected one refresh and direct loads of spans inside the cached segment");
        }
        Expect(spanReaders != 0u, "the module has no read_bda_span function");
        for (const auto& [id, function] : module.functions) {
            if (function.name.starts_with("load_scalar_buffer_")) Expect(function.bufferMemoryLoads == 3u, function.name + " loads buffer memory " + std::to_string(function.bufferMemoryLoads) + " times, expected three times: through its cached translation, through a refilled one and through the uncached read");
            if (function.name.starts_with("load_formatted_buffer_")) {
                Expect(function.widestSwitch != 0u && function.widestSwitch <= 15u, function.name + " switches to " + std::to_string(function.widestSwitch) + " labels, expected one per format layout and the invalid format (at most 15)");
                Expect(function.switches == 1u, function.name + " has " + std::to_string(function.switches) + " switches, expected only its layout switch with the formats of a layout decoded by shared functions");
                std::size_t decoders = 0;
                for (const auto& [callee, count] : function.calls) {
                    const auto found = module.functions.find(callee);
                    if (found != module.functions.end() && found->second.name.starts_with("decode_format_layout_")) ++decoders;
                }
                Expect(decoders != 0u, function.name + " calls no shared format decoder");
                Expect(function.dynamicExtracts != 0u, function.name + " selects its destination components without a dynamic extract");
            }
            if (function.name.starts_with("decode_format_layout_")) Expect((function.control & spv::FunctionControlDontInlineMask) != 0u && function.switches == 1u, function.name + " is not one DontInline switch over the formats of its layout");
            if (function.name.starts_with("store_buffer_x3")) {
                const auto found = function.calls.find(write);
                const auto writes = found == function.calls.end() ? 0u : found->second;
                Expect(writes == 3u, function.name + " calls write_bda " + std::to_string(writes) + " times, expected once per component");
            }
            const auto branched = module.branchedCallsFromMain.find(id);
            const bool access = function.name.starts_with("load_buffer_") || function.name.starts_with("store_buffer_") || function.name.starts_with("load_formatted_buffer_") || function.name.starts_with("buffer_atomic_address_");
            Expect(!access || branched == module.branchedCallsFromMain.end(), "main calls " + function.name + " from a conditional block " + std::to_string(branched == module.branchedCallsFromMain.end() ? 0u : branched->second) + " times, expected its active lanes passed to it instead");
        }
        const std::vector<std::uint32_t> wide{
            0x34020084u,
            0xf42c0502u, 0xfa000000u,
            0xbf8cc07fu,
            0x7e1a0214u, 0x7e1c0215u, 0x7e1e0216u, 0x7e200217u,
            0x7e220218u, 0x7e240219u, 0x7e26021au, 0x7e28021bu,
            0xe0781000u, 0x80020d01u,
            0xe0781010u, 0x80021101u,
            0xbf810000u};
        const auto wideModule = Parse(Recompiled(wide));
        std::uint32_t wideSegment = 0;
        for (const auto& [id, function] : wideModule.functions) {
            if (function.name == "bda_segment") wideSegment = id;
        }
        const auto wideRefreshes = wideModule.callsFromMain.contains(wideSegment) ? wideModule.callsFromMain.at(wideSegment) : 0u;
        Expect(wideSegment != 0u && wideRefreshes == 1u, "main refreshes the read segment " + std::to_string(wideRefreshes) + " times for an s_buffer_load_dwordx8, expected one refresh before the eight dwords fall back to their helpers");
        const auto wideScalars = Calls(wideModule, "load_scalar_buffer_");
        Expect(wideScalars.size() == 1u && wideScalars[0] == 8u, "expected one scalar load helper called by the eight dwords of the s_buffer_load_dwordx8 on their fallback path, found " + Describe(wideScalars));
    } catch (const std::exception& error) {
        std::cerr << "SpirvBufferAccessHelpers: " << error.what() << '\n';
        return 1;
    }
    std::puts("SpirvBufferAccessHelpers: buffer accesses call one helper per resource and shape");
    return 0;
}
