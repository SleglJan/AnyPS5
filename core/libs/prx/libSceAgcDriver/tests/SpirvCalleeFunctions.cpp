#include "Recompiler.hpp"
#include "ControlFlow/CalleeLinker.hpp"
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
#include <utility>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint64_t Srt = 0x1100010000ull;
constexpr std::uint64_t Table = 0x1100020000ull;
constexpr std::uint64_t Code = 0x1100030000ull;
constexpr std::uint64_t Output = 0x1100050000ull;

void Require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::uint32_t Low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address);
}

std::uint32_t High(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 32u);
}

struct GuestMemory {
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions;
};

bool ReadGuestMemory(void* context, std::uint64_t address, std::span<std::uint32_t> words) {
    for (const auto& [base, region] : static_cast<const GuestMemory*>(context)->regions) {
        if (address >= base && (address - base) / 4u + words.size() <= region.size()) {
            std::copy_n(region.begin() + static_cast<std::ptrdiff_t>((address - base) / 4u), words.size(), words.begin());
            return true;
        }
    }
    return false;
}

std::vector<std::uint32_t> Recompiled(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const GuestMemory* memory) {
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    static constexpr std::array<std::uint32_t, 8> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityGroupNonUniform,
        spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformArithmetic};
    std::vector<std::vector<std::byte>> bytes;
    std::vector<MemoryRegion> regions;
    if (memory != nullptr) {
        for (const auto& [base, words] : memory->regions) {
            const auto view = std::as_bytes(std::span(words));
            bytes.emplace_back(view.begin(), view.end());
        }
        for (std::size_t index = 0; index < bytes.size(); ++index) regions.push_back({memory->regions[index].first, bytes[index]});
    }
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x1100040000ull, code, 0, {}};
    request.context.waveSize = 32;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.memory = regions;
    request.context.compute = ShaderComputeStageInfo{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    LinkedCode linked;
    if (memory != nullptr) {
        linked = LinkCallees(request, CalleeMemory{const_cast<GuestMemory*>(memory), &ReadGuestMemory, &ReadGuestMemory, nullptr});
        Require(!linked.code.empty(), "the program's callees were not linked");
        request.shader.code = linked.code;
        request.shader.linkedLiterals = linked.linkedLiterals;
        request.context.linkedWords = linked.linkedWords;
    }
    const auto result = Recompile(request);
    Require(!result.spirv.empty(), "the program did not recompile");
    return result.spirv.Copy();
}

struct Module {
    std::uint32_t entry = 0;
    std::uint32_t voidType = 0;
    std::map<std::uint32_t, std::string> names;
    std::map<std::uint32_t, std::uint32_t> controls;
    std::map<std::uint32_t, std::uint32_t> resultTypes;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> calls;
    std::map<std::uint32_t, std::set<std::uint32_t>> composites;
    std::map<std::uint32_t, std::set<std::uint32_t>> returned;
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> callResults;
    std::map<std::uint32_t, std::uint32_t> pointerResults;
    std::map<std::uint32_t, std::uint32_t> lastParameters;
    std::map<std::uint32_t, std::set<std::uint32_t>> parameterStores;
    std::map<std::uint32_t, std::set<std::uint32_t>> extracted;
};

Module Parse(std::span<const std::uint32_t> words) {
    Module module;
    std::uint32_t function = 0;
    std::uint32_t lastCall = 0;
    std::uint32_t lastPointer = 0;
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
        } else if (op == spv::OpFunction) {
            function = operand(1);
            module.controls[function] = operand(2);
            module.resultTypes[function] = operand(0);
        } else if (op == spv::OpFunctionEnd) {
            function = 0;
        } else if (op == spv::OpFunctionParameter) {
            module.lastParameters[function] = operand(1);
        } else if (op == spv::OpFunctionCall) {
            module.calls.emplace_back(function, operand(2));
            module.callResults[operand(1)] = {function, operand(2)};
        } else if (op == spv::OpStore && module.lastParameters.contains(function) && operand(0) == module.lastParameters.at(function)) {
            module.parameterStores[function].insert(operand(1));
        } else if (op == spv::OpCompositeConstruct) {
            module.composites[function].insert(operand(1));
        } else if (op == spv::OpReturnValue) {
            module.returned[function].insert(operand(0));
        } else if (op == spv::OpCompositeExtract) {
            module.extracted[function].insert(operand(2));
        }
        if (op == spv::OpLoad && lastCall != 0u && operand(2) == lastPointer) module.pointerResults[operand(1)] = lastCall;
        lastCall = 0;
        if (op == spv::OpFunctionCall && count > 4u) {
            lastCall = operand(1);
            lastPointer = operand(count - 2u);
        }
        cursor += count;
    }
    return module;
}

std::vector<std::uint32_t> Callees(const Module& module) {
    std::vector<std::uint32_t> callees;
    for (const auto& [id, name] : module.names) {
        if (name.starts_with("guest_callee_") && module.controls.contains(id)) callees.push_back(id);
    }
    return callees;
}

std::vector<std::uint32_t> Callers(const Module& module, std::uint32_t callee) {
    std::vector<std::uint32_t> callers;
    for (const auto& [caller, called] : module.calls) {
        if (called == callee) callers.push_back(caller);
    }
    return callers;
}

void RequireCallees(const Module& module, std::size_t count, const std::string& what) {
    const auto callees = Callees(module);
    Require(callees.size() == count, what + ": " + std::to_string(callees.size()) + " callee functions, expected " + std::to_string(count));
    for (const auto callee : callees) {
        Require((module.controls.at(callee) & spv::FunctionControlDontInlineMask) != 0u, what + ": " + module.names.at(callee) + " is not DontInline");
        Require(Callers(module, callee).size() == 1u, what + ": " + module.names.at(callee) + " is called " + std::to_string(Callers(module, callee).size()) + " times, expected once");
    }
}

void RequireReturns(const Module& module, std::uint32_t callee, std::uint32_t caller, const std::string& what) {
    const auto returned = module.returned.find(callee);
    const auto stored = module.parameterStores.find(callee);
    const auto composites = module.composites.find(callee);
    const auto builds = [&](const std::set<std::uint32_t>& values) {
        return composites != module.composites.end() && std::ranges::any_of(values, [&](std::uint32_t value) { return composites->second.contains(value); });
    };
    const bool byValue = module.resultTypes.at(callee) != module.voidType && returned != module.returned.end() && builds(returned->second);
    const bool throughPointer = module.resultTypes.at(callee) == module.voidType && stored != module.parameterStores.end() && builds(stored->second);
    const bool returnsRegisters = byValue || throughPointer;
    const auto extracted = module.extracted.find(caller);
    bool taken = false;
    if (extracted != module.extracted.end()) {
        for (const auto& [result, call] : module.callResults) {
            if (call.first == caller && call.second == callee && extracted->second.contains(result)) taken = true;
        }
        for (const auto& [loaded, result] : module.pointerResults) {
            const auto call = module.callResults.find(result);
            if (call != module.callResults.end() && call->second.first == caller && call->second.second == callee && extracted->second.contains(loaded)) taken = true;
        }
    }
    Require(returnsRegisters && taken, what + ": " + module.names.at(callee) + " does not hand the register it wrote back to its caller");
}

void CheckTableCall() {
    const std::vector<std::uint32_t> caller{0x36020081u, 0xbe9e037eu, 0x7e280501u, 0xbe9f037eu, 0x7da40214u, 0x8f148414u, 0xf4080400u, 0xfa000000u, 0xbf8cc07fu, 0xf4280208u, 0x28000000u,
        0xbf8cc07fu, 0xbe8e0308u, 0xbe8f0309u, 0xbe8e210eu, 0x8a7e7e1fu, 0xbf89fff1u, 0xbefe031eu, 0xf4080600u, 0xfa000010u, 0x34060082u, 0xbf8cc07fu, 0xe0701000u, 0x80060203u, 0xbf810000u};
    const std::vector<std::uint32_t> first{0x4a040085u, 0xbe80200eu};
    const std::vector<std::uint32_t> second{0x4a0400c0u, 0xbe80200eu};
    std::vector<std::uint32_t> page(1024u, 0u);
    std::copy(first.begin(), first.end(), page.begin());
    std::copy(second.begin(), second.end(), page.begin() + 0x200);
    const std::vector<std::uint32_t> srt{Low(Table), High(Table) | (16u << 16u), 2u, 0x31016facu, Low(Output), High(Output) & 0xffffu, 0x100u, 0x31016facu};
    const std::vector<std::uint32_t> records{Low(Code), High(Code), 0x00c0ffeeu, 0x12u, Low(Code + 0x800u), High(Code + 0x800u), 0x0badf00du, 0x13u};
    const GuestMemory memory{{{Srt, srt}, {Table, records}, {Code, page}}};
    const std::array<std::uint32_t, 2> userData{Low(Srt), High(Srt)};
    const auto module = Parse(Recompiled(caller, userData, &memory));
    RequireCallees(module, 2u, "a table call of two targets");
    for (const auto callee : Callees(module)) {
        Require(Callers(module, callee).front() == module.entry, "a table callee is not called from the entry point's compare chain");
        RequireReturns(module, callee, module.entry, "a table callee that writes v2, which the caller stores after its loop");
    }
}

void CheckUserDataCalls() {
    const std::vector<std::uint32_t> caller{0x7e020280u, 0xbe940383u, 0xbe8e0300u, 0xbe8f0301u, 0xbe8e210eu, 0xbe8e0302u, 0xbe8f0303u, 0xbe8e210eu,
        0x80948114u, 0xbf078014u, 0xbf85fff7u, 0x34040082u, 0xe0701000u, 0x80010102u, 0xbf810000u};
    const std::vector<std::uint32_t> addFive{0x4a020285u, 0xbe80200eu};
    const std::vector<std::uint32_t> addSixtyFour{0x4a0202c0u, 0xbe80200eu};
    std::vector<std::uint32_t> page(1024u, 0u);
    std::copy(addFive.begin(), addFive.end(), page.begin());
    std::copy(addSixtyFour.begin(), addSixtyFour.end(), page.begin() + 0x200);
    const GuestMemory memory{{{Code, page}}};
    const std::array<std::uint32_t, 8> userData{Low(Code), High(Code), Low(Code + 0x800u), High(Code + 0x800u), Low(Output), High(Output) & 0xffffu, 0x100u, 0x31016facu};
    const auto module = Parse(Recompiled(caller, userData, &memory));
    RequireCallees(module, 2u, "two user-data calls");
    for (const auto callee : Callees(module)) {
        Require(Callers(module, callee).front() == module.entry, "a user-data callee is not called from the entry point");
        RequireReturns(module, callee, module.entry, "a user-data callee that writes v1, which the caller stores after the calls");
    }
}

void CheckNestedCalls() {
    const std::vector<std::uint32_t> code{0x34020084u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u, 0xbe8a1f00u, 0x800aff0au, 0x00000014u, 0x820b800bu, 0xbe88210au, 0xbf82000bu, 0x4a0808ffu, 0x13579bdfu,
        0xbb0c0001u, 0xbf820006u, 0x3a0a0affu, 0xa5a5a5a5u, 0x4a0c0f06u, 0x3a0e0effu, 0x5a5a5a5au, 0xbe80200cu, 0xbe802008u, 0xe0781000u, 0x80010401u, 0xbf810000u};
    const std::array<std::uint32_t, 8> userData{Low(Output), High(Output) & 0xffffu, 0x400u, 0x31016facu, Low(Output + 0x1000u), High(Output) & 0xffffu, 0x400u, 0x31016facu};
    const auto module = Parse(Recompiled(code, userData, nullptr));
    RequireCallees(module, 2u, "a static call inside a static callee");
    const auto callees = Callees(module);
    const auto outer = std::ranges::find_if(callees, [&](std::uint32_t callee) { return Callers(module, callee).front() == module.entry; });
    Require(outer != callees.end(), "no callee is called from the entry point");
    const auto inner = callees.front() == *outer ? callees.back() : callees.front();
    Require(Callers(module, inner).front() == *outer, "the inner callee is not called from the outer callee");
    RequireReturns(module, inner, *outer, "an inner callee that writes v5, v6 and v7, which its caller stores after returning");
    RequireReturns(module, *outer, module.entry, "an outer callee that writes v4");
}

}

int main() {
    try {
        CheckTableCall();
        CheckUserDataCalls();
        CheckNestedCalls();
    } catch (const std::exception& error) {
        std::cerr << "SpirvCalleeFunctions: " << error.what() << '\n';
        return 1;
    }
    std::puts("callee function tests passed");
    return 0;
}
