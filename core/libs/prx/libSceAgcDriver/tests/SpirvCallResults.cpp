#include "SpirvBackend/SpirvModule.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <set>
#include <spirv/unified1/spirv.hpp>
#include <vector>

using namespace ShaderRecompiler;

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", what);
        ++failures;
    }
}

void testCallResultsSpillOnlyAcrossCallsOfTheirCallee() {
    SpirvModule module(0x00010400u);
    const auto u32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto boolean = module.Type(spv::OpTypeBool);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto pairType = module.Type(spv::OpTypeStruct, u32, u32);
    const auto unary = module.Type(spv::OpTypeFunction, u32, u32);
    const auto pairing = module.Type(spv::OpTypeFunction, pairType, u32);
    const auto procedure = module.Type(spv::OpTypeFunction, voidType);
    const auto one = module.Constant(spv::OpConstant, u32, 1u);
    const auto two = module.Constant(spv::OpConstant, u32, 2u);
    const auto three = module.Constant(spv::OpConstant, u32, 3u);
    const auto hundred = module.Constant(spv::OpConstant, u32, 100u);
    const auto shared = module.AllocateId();
    const auto inlinable = module.AllocateId();
    const auto pair = module.AllocateId();
    const auto main = module.AllocateId();
    const auto define = [&](std::uint32_t id, std::uint32_t control) {
        const auto parameter = module.AllocateId();
        module.AddFunction(spv::OpFunction, u32, id, control, unary);
        module.AddFunction(spv::OpFunctionParameter, u32, parameter);
        module.AddFunction(spv::OpLabel, module.AllocateId());
        module.AddFunction(spv::OpReturnValue, parameter);
        module.AddFunction(spv::OpFunctionEnd);
    };
    define(shared, spv::FunctionControlDontInlineMask);
    define(inlinable, spv::FunctionControlMaskNone);
    module.BeginHelperFunction();
    {
        const auto parameter = module.AllocateId();
        const auto composite = module.AllocateId();
        module.AddFunction(spv::OpFunction, pairType, pair, spv::FunctionControlDontInlineMask, pairing);
        module.AddFunction(spv::OpFunctionParameter, u32, parameter);
        module.AddFunction(spv::OpLabel, module.AllocateId());
        module.AddFunction(spv::OpCompositeConstruct, pairType, composite, parameter, parameter);
        module.AddFunction(spv::OpReturnValue, composite);
        module.AddFunction(spv::OpFunctionEnd);
    }
    module.EndHelperFunction();
    const auto id = [&]() { return module.AllocateId(); };
    const auto entry = id(), header = id(), body = id(), merge = id();
    const auto a = id(), b = id(), sum = id(), c = id(), tripled = id(), d = id(), d1 = id();
    const auto i1 = id(), i2 = id(), i3 = id();
    const auto s1 = id(), e1 = id(), s2 = id(), e2 = id(), e3 = id();
    const auto carried = id(), condition = id(), r = id(), q = id(), out = id();
    const auto local = id(), w = id(), z = id(), ze = id();
    const auto privateVariable = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassPrivate, u32), spv::StorageClassPrivate);
    module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, procedure);
    module.AddFunction(spv::OpLabel, entry);
    module.AddFunction(spv::OpVariable, module.Type(spv::OpTypePointer, spv::StorageClassFunction, u32), local, spv::StorageClassFunction);
    module.AddFunction(spv::OpFunctionCall, u32, a, shared, one);
    module.AddFunction(spv::OpFunctionCall, u32, b, shared, a);
    module.AddFunction(spv::OpIAdd, u32, sum, a, b);
    module.AddFunction(spv::OpFunctionCall, u32, c, shared, sum);
    module.AddFunction(spv::OpIMul, u32, tripled, c, three);
    module.AddFunction(spv::OpFunctionCall, u32, d, shared, tripled);
    module.AddFunction(spv::OpIAdd, u32, d1, d, one);
    module.AddFunction(spv::OpFunctionCall, u32, i1, inlinable, one);
    module.AddFunction(spv::OpFunctionCall, u32, i2, inlinable, two);
    module.AddFunction(spv::OpIAdd, u32, i3, i1, i2);
    module.AddFunction(spv::OpFunctionCall, pairType, s1, pair, one);
    module.AddFunction(spv::OpCompositeExtract, u32, e1, s1, 0u);
    module.AddFunction(spv::OpFunctionCall, pairType, s2, pair, two);
    module.AddFunction(spv::OpCompositeExtract, u32, e2, s2, 1u);
    module.AddFunction(spv::OpIAdd, u32, e3, e1, e2);
    module.AddFunction(spv::OpFunctionCall, u32, w, shared, three);
    module.AddFunction(spv::OpStore, local, w);
    module.AddFunction(spv::OpFunctionCall, pairType, z, pair, three);
    module.AddFunction(spv::OpCompositeExtract, u32, ze, z, 1u);
    module.AddFunction(spv::OpStore, privateVariable, ze);
    module.AddFunction(spv::OpBranch, header);
    module.AddFunction(spv::OpLabel, header);
    module.AddFunction(spv::OpPhi, u32, carried, d1, entry, r, body);
    module.AddFunction(spv::OpULessThan, boolean, condition, carried, hundred);
    module.AddFunction(spv::OpLoopMerge, merge, body, spv::LoopControlMaskNone);
    module.AddFunction(spv::OpBranchConditional, condition, body, merge);
    module.AddFunction(spv::OpLabel, body);
    module.AddFunction(spv::OpFunctionCall, u32, r, shared, carried);
    module.AddFunction(spv::OpIAdd, u32, q, carried, r);
    module.AddFunction(spv::OpBranch, header);
    module.AddFunction(spv::OpLabel, merge);
    module.AddFunction(spv::OpIAdd, u32, out, carried, e3);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.SpillCallResults();
    module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", {});
    const auto words = module.Finalize();
    std::map<std::uint32_t, std::uint32_t> functionPointees;
    std::map<std::uint32_t, std::uint32_t> resultTypes;
    std::map<std::uint32_t, std::size_t> returnValues;
    std::map<std::uint32_t, std::size_t> parameters;
    std::map<std::uint32_t, std::uint32_t> lastParameterTypes;
    std::vector<std::vector<std::uint32_t>> mainBody;
    std::uint32_t current = 0;
    for (std::size_t offset = 5; offset < words.size(); offset += words[offset] >> spv::WordCountShift) {
        const std::vector<std::uint32_t> instruction(words.begin() + static_cast<std::ptrdiff_t>(offset), words.begin() + static_cast<std::ptrdiff_t>(offset + (words[offset] >> spv::WordCountShift)));
        const auto opcode = instruction[0] & spv::OpCodeMask;
        if (opcode == spv::OpTypePointer && instruction[2] == spv::StorageClassFunction) functionPointees[instruction[1]] = instruction[3];
        if (opcode == spv::OpFunction) {
            current = instruction[2];
            resultTypes[current] = instruction[1];
            continue;
        }
        if (current == 0u) continue;
        if (opcode == spv::OpFunctionParameter) {
            ++parameters[current];
            lastParameterTypes[current] = instruction[1];
        }
        if (opcode == spv::OpReturnValue) ++returnValues[current];
        if (current == main) mainBody.push_back(instruction);
    }
    check(resultTypes[shared] == u32 && returnValues[shared] == 1u && parameters[shared] == 2u && functionPointees[lastParameterTypes[shared]] == u32, "a DontInline function with spilled call results does not return its result by value and take a Function pointer to its result type last");
    check(resultTypes[pair] == pairType && returnValues[pair] == 1u && parameters[pair] == 2u && functionPointees[lastParameterTypes[pair]] == pairType, "a DontInline helper function with spilled call results does not return its result by value and take a Function pointer to its result type last");
    check(resultTypes[inlinable] == u32 && returnValues[inlinable] == 1u && parameters[inlinable] == 1u, "a function without DontInline changed");
    std::map<std::uint32_t, std::uint32_t> variables;
    for (std::size_t index = 1; index < mainBody.size() && (mainBody[index][0] & spv::OpCodeMask) == spv::OpVariable; ++index) {
        if (mainBody[index][2] != local && mainBody[index][3] == spv::StorageClassFunction && functionPointees.contains(mainBody[index][1])) variables[mainBody[index][2]] = functionPointees[mainBody[index][1]];
    }
    std::set<std::uint32_t> variableTypes;
    for (const auto& [variable, type] : variables) variableTypes.insert(type);
    check((mainBody.front()[0] & spv::OpCodeMask) == spv::OpLabel && variables.size() == 2u && variableTypes == std::set<std::uint32_t>{u32, pairType}, "the caller does not declare one Function variable per spilled result type at the start of its entry block");
    std::set<std::uint32_t> spilled;
    std::set<std::uint32_t> byValue;
    std::size_t anchoredCalls = 0;
    for (std::size_t index = 0; index < mainBody.size(); ++index) {
        const auto& call = mainBody[index];
        if ((call[0] & spv::OpCodeMask) != spv::OpFunctionCall || (call[3] != shared && call[3] != pair)) continue;
        const bool anchoredCall = call.size() == 6u && variables.contains(call[5]) && variables[call[5]] == call[1];
        if (anchoredCall) ++anchoredCalls;
        const bool stored = anchoredCall && index + 2u < mainBody.size() && (mainBody[index + 1u][0] & spv::OpCodeMask) == spv::OpStore && mainBody[index + 1u].size() == 4u && mainBody[index + 1u][1] == call[5] && mainBody[index + 1u][2] == call[2] && mainBody[index + 1u][3] == spv::MemoryAccessVolatileMask;
        const auto& load = mainBody[std::min(index + 2u, mainBody.size() - 1u)];
        const bool loaded = stored && (load[0] & spv::OpCodeMask) == spv::OpLoad && load.size() == 5u && load[1] == call[1] && load[3] == mainBody[index + 1u][1] && load[4] == spv::MemoryAccessVolatileMask;
        if (loaded) spilled.insert(load[2]);
        else byValue.insert(call[2]);
    }
    check(spilled == std::set<std::uint32_t>{a, s1, r, w, z}, "the results used after a later call of their callee (directly, through an extract or around a loop) or stored to Function or Private memory are not stored and loaded back right after their calls");
    check(byValue == std::set<std::uint32_t>{b, c, d, s2}, "a result used only before the next call of its callee does not stay a value");
    check(anchoredCalls == 9u, "a call of a DontInline function with spilled call results does not pass the caller's Function variable of its result type");
    std::size_t inlinableCalls = 0;
    for (const auto& instruction : mainBody) {
        if ((instruction[0] & spv::OpCodeMask) == spv::OpFunctionCall && instruction[3] == inlinable && (instruction[2] == i1 || instruction[2] == i2)) ++inlinableCalls;
    }
    check(inlinableCalls == 2u, "a call of a function without DontInline changed");
}

}

int main() {
    testCallResultsSpillOnlyAcrossCallsOfTheirCallee();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
