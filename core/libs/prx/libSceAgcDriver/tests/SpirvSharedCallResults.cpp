#include "SpirvBackend/SpirvModule.hpp"
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

void testOnlySharedCalleesReturnThroughPointers() {
    SpirvModule module(0x00010400u);
    const auto u32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto boolean = module.Type(spv::OpTypeBool);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto unary = module.Type(spv::OpTypeFunction, u32, u32);
    const auto procedure = module.Type(spv::OpTypeFunction, voidType);
    const auto one = module.Constant(spv::OpConstant, u32, 1u);
    const auto hundred = module.Constant(spv::OpConstant, u32, 100u);
    const auto id = [&]() { return module.AllocateId(); };
    const auto shared = id(), once = id(), looped = id(), twoCallers = id(), inlinable = id(), other = id(), main = id();
    const auto define = [&](std::uint32_t function, std::uint32_t control) {
        const auto parameter = id();
        module.AddFunction(spv::OpFunction, u32, function, control, unary);
        module.AddFunction(spv::OpFunctionParameter, u32, parameter);
        module.AddFunction(spv::OpLabel, id());
        module.AddFunction(spv::OpReturnValue, parameter);
        module.AddFunction(spv::OpFunctionEnd);
    };
    define(shared, spv::FunctionControlDontInlineMask);
    define(once, spv::FunctionControlDontInlineMask);
    define(looped, spv::FunctionControlDontInlineMask);
    define(inlinable, spv::FunctionControlMaskNone);
    module.BeginHelperFunction();
    define(twoCallers, spv::FunctionControlDontInlineMask);
    const auto fromOther = id();
    module.AddFunction(spv::OpFunction, voidType, other, spv::FunctionControlDontInlineMask, procedure);
    module.AddFunction(spv::OpLabel, id());
    module.AddFunction(spv::OpFunctionCall, u32, fromOther, twoCallers, one);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.EndHelperFunction();
    const auto entry = id(), header = id(), body = id(), merge = id();
    const auto a = id(), b = id(), c = id(), d = id(), i1 = id(), i2 = id(), carried = id(), condition = id(), r = id(), sum = id();
    module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, procedure);
    module.AddFunction(spv::OpLabel, entry);
    module.AddFunction(spv::OpFunctionCall, u32, a, shared, one);
    module.AddFunction(spv::OpFunctionCall, u32, b, shared, a);
    module.AddFunction(spv::OpFunctionCall, u32, c, once, b);
    module.AddFunction(spv::OpFunctionCall, u32, d, twoCallers, c);
    module.AddFunction(spv::OpFunctionCall, u32, i1, inlinable, one);
    module.AddFunction(spv::OpFunctionCall, u32, i2, inlinable, i1);
    module.AddFunction(spv::OpFunctionCall, voidType, id(), other);
    module.AddFunction(spv::OpBranch, header);
    module.AddFunction(spv::OpLabel, header);
    module.AddFunction(spv::OpPhi, u32, carried, d, entry, r, body);
    module.AddFunction(spv::OpULessThan, boolean, condition, carried, hundred);
    module.AddFunction(spv::OpLoopMerge, merge, body, spv::LoopControlMaskNone);
    module.AddFunction(spv::OpBranchConditional, condition, body, merge);
    module.AddFunction(spv::OpLabel, body);
    module.AddFunction(spv::OpFunctionCall, u32, r, looped, carried);
    module.AddFunction(spv::OpBranch, header);
    module.AddFunction(spv::OpLabel, merge);
    module.AddFunction(spv::OpIAdd, u32, sum, a, carried);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.ReturnSharedResultsThroughPointers();
    module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", {});
    const auto words = module.Finalize();
    std::map<std::uint32_t, std::uint32_t> functionPointees;
    std::map<std::uint32_t, std::uint32_t> resultTypes;
    std::map<std::uint32_t, std::size_t> parameters;
    std::map<std::uint32_t, std::size_t> returnValues;
    std::map<std::uint32_t, std::size_t> parameterStores;
    std::map<std::uint32_t, std::uint32_t> lastParameters;
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
            lastParameters[current] = instruction[2];
            lastParameterTypes[current] = instruction[1];
        }
        if (opcode == spv::OpReturnValue) ++returnValues[current];
        if (opcode == spv::OpStore && lastParameters.contains(current) && instruction[1] == lastParameters[current]) ++parameterStores[current];
        if (current == main) mainBody.push_back(instruction);
    }
    for (const auto callee : {shared, looped}) {
        check(resultTypes[callee] == voidType && returnValues[callee] == 0u && parameters[callee] == 2u && functionPointees[lastParameterTypes[callee]] == u32 && parameterStores[callee] == 1u,
            "a DontInline function called twice from one caller, or from a loop, does not return void and store its result through a Function pointer parameter");
    }
    for (const auto callee : {once, twoCallers}) {
        check(resultTypes[callee] == u32 && returnValues[callee] == 1u && parameters[callee] == 1u, "a DontInline function called once per caller outside loops no longer returns its result by value");
    }
    check(resultTypes[inlinable] == u32 && returnValues[inlinable] == 1u && parameters[inlinable] == 1u, "a function without DontInline changed");
    std::set<std::uint32_t> variables;
    for (std::size_t index = 1; index < mainBody.size() && (mainBody[index][0] & spv::OpCodeMask) == spv::OpVariable; ++index) {
        if (mainBody[index][3] == spv::StorageClassFunction && functionPointees[mainBody[index][1]] == u32) variables.insert(mainBody[index][2]);
    }
    check(variables.size() == 2u, "the caller does not declare one Function variable for each shared callee's result");
    std::set<std::uint32_t> loaded;
    std::set<std::uint32_t> byValue;
    for (std::size_t index = 0; index + 1u < mainBody.size(); ++index) {
        const auto& call = mainBody[index];
        if ((call[0] & spv::OpCodeMask) != spv::OpFunctionCall) continue;
        const auto& next = mainBody[index + 1u];
        if (call[3] == shared || call[3] == looped) {
            if (call[1] == voidType && call.size() == 6u && variables.contains(call[5]) && (next[0] & spv::OpCodeMask) == spv::OpLoad && next[1] == u32 && next[3] == call[5]) loaded.insert(next[2]);
        } else if (call[1] == u32) {
            byValue.insert(call[2]);
        }
    }
    check(loaded == std::set<std::uint32_t>{a, b, r}, "a call of a shared callee does not pass the caller's variable and load the result right after the call");
    check(byValue == std::set<std::uint32_t>{c, d, i1, i2}, "a call of a callee called once per caller outside loops does not keep its result as a value");
}

}

int main() {
    testOnlySharedCalleesReturnThroughPointers();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
