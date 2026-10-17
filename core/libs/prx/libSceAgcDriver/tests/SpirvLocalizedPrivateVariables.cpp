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

void testEachFunctionGetsItsOwnCopy() {
    SpirvModule module(0x00010400u);
    const auto u64 = module.Type(spv::OpTypeInt, 64u, 0u);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto procedure = module.Type(spv::OpTypeFunction, voidType);
    const auto privatePointer = module.Type(spv::OpTypePointer, static_cast<std::uint32_t>(spv::StorageClassPrivate), u64);
    const auto zero = module.Constant(spv::OpConstant, u64, 0u, 0u);
    const auto id = [&]() { return module.AllocateId(); };
    const auto cache = id(), kept = id();
    module.EmitGlobalVariable({spv::OpVariable, privatePointer, cache, spv::StorageClassPrivate, zero});
    module.EmitGlobalVariable({spv::OpVariable, privatePointer, kept, spv::StorageClassPrivate, zero});
    module.AddName(cache, "cache");
    const auto helper = id(), untouched = id(), main = id();
    module.BeginHelperFunction();
    module.AddFunction(spv::OpFunction, voidType, helper, spv::FunctionControlDontInlineMask, procedure);
    module.AddFunction(spv::OpLabel, id());
    const auto read = id();
    module.AddFunction(spv::OpLoad, u64, read, cache);
    module.AddFunction(spv::OpStore, cache, read);
    module.AddFunction(spv::OpStore, kept, read);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.EndHelperFunction();
    module.AddFunction(spv::OpFunction, voidType, untouched, spv::FunctionControlDontInlineMask, procedure);
    module.AddFunction(spv::OpLabel, id());
    module.AddFunction(spv::OpStore, kept, zero);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, procedure);
    module.AddFunction(spv::OpLabel, id());
    module.AddFunction(spv::OpStore, cache, zero);
    module.AddFunction(spv::OpFunctionCall, voidType, id(), helper);
    module.AddFunction(spv::OpFunctionCall, voidType, id(), untouched);
    module.AddFunction(spv::OpLoad, u64, id(), cache);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    const std::vector<std::uint32_t> variables{cache, 0u};
    module.LocalizePrivateVariables(variables);
    module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", {});
    const auto words = module.Finalize();
    std::map<std::uint32_t, std::uint32_t> functionPointees;
    std::map<std::uint32_t, std::vector<std::uint32_t>> declaredIn;
    std::map<std::uint32_t, std::set<std::uint32_t>> accessedIn;
    std::set<std::uint32_t> privateVariables;
    std::set<std::uint32_t> interfaceIds;
    bool named = false;
    bool afterLabel = true;
    std::uint32_t current = 0;
    std::size_t labels = 0;
    bool bodyStarted = false;
    for (std::size_t offset = 5; offset < words.size(); offset += words[offset] >> spv::WordCountShift) {
        const std::vector<std::uint32_t> instruction(words.begin() + static_cast<std::ptrdiff_t>(offset), words.begin() + static_cast<std::ptrdiff_t>(offset + (words[offset] >> spv::WordCountShift)));
        const auto opcode = instruction[0] & spv::OpCodeMask;
        if (opcode == spv::OpEntryPoint) interfaceIds.insert(instruction.begin() + 5, instruction.end());
        if (opcode == spv::OpName && instruction[1] == cache) named = true;
        if (opcode == spv::OpTypePointer && instruction[2] == spv::StorageClassFunction) functionPointees[instruction[1]] = instruction[3];
        if (opcode == spv::OpFunction) {
            current = instruction[2];
            labels = 0;
            bodyStarted = false;
            continue;
        }
        if (opcode == spv::OpLabel) {
            ++labels;
            continue;
        }
        if (opcode == spv::OpVariable) {
            if (instruction[3] == spv::StorageClassPrivate) privateVariables.insert(instruction[2]);
            if (current != 0u && instruction[3] == spv::StorageClassFunction) {
                afterLabel &= labels == 1u && !bodyStarted;
                if (functionPointees[instruction[1]] == u64 && instruction.size() == 5u && instruction[4] == zero) declaredIn[current].push_back(instruction[2]);
            }
            continue;
        }
        if (current != 0u) bodyStarted = true;
        if (opcode == spv::OpLoad) accessedIn[current].insert(instruction[3]);
        if (opcode == spv::OpStore) accessedIn[current].insert(instruction[1]);
        if (opcode == spv::OpFunctionEnd) current = 0;
    }
    const auto copyIn = [&](std::uint32_t function) { return declaredIn[function].empty() ? 0u : declaredIn[function][0]; };
    check(privateVariables == std::set<std::uint32_t>{kept}, "the localized Private variable is still declared, or the other one is gone");
    check(!interfaceIds.contains(cache) && interfaceIds.contains(kept), "the entry point interface still lists the localized variable or lost the other one");
    check(!named, "the name of the localized variable was kept");
    check(declaredIn[main].size() == 1u && declaredIn[helper].size() == 1u && copyIn(main) != copyIn(helper), "main and the helper do not each declare their own zero-initialized Function copy");
    check(!declaredIn.contains(untouched), "a function that does not use the variable declares a copy");
    check(afterLabel, "a copy is not declared in its function's entry block");
    check(accessedIn[main] == std::set<std::uint32_t>{copyIn(main)}, "main does not load and store through its own copy");
    check(accessedIn[helper] == std::set<std::uint32_t>{copyIn(helper), kept}, "the helper does not load and store through its own copy, or the other variable was renamed");
    check(accessedIn[untouched] == std::set<std::uint32_t>{kept}, "the other variable was renamed in a function that only uses it");
}

}

int main() {
    testEachFunctionGetsItsOwnCopy();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
