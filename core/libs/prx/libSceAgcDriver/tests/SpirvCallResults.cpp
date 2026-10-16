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

void testLocalMemoryUsersAreCounted() {
    SpirvModule module(0x00010400u);
    const auto u32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto four = module.Constant(spv::OpConstant, u32, 4u);
    const auto one = module.Constant(spv::OpConstant, u32, 1u);
    const auto array = module.Type(spv::OpTypeArray, u32, four);
    const auto functionU32 = module.Type(spv::OpTypePointer, spv::StorageClassFunction, u32);
    const auto functionArray = module.Type(spv::OpTypePointer, spv::StorageClassFunction, array);
    const auto privateU32 = module.Type(spv::OpTypePointer, spv::StorageClassPrivate, u32);
    const auto procedure = module.Type(spv::OpTypeFunction, voidType);
    const auto taking = module.Type(spv::OpTypeFunction, voidType, functionU32);
    const auto shared = module.DefineGlobalVariable(privateU32, spv::StorageClassPrivate);
    const auto own = module.DefineGlobalVariable(privateU32, spv::StorageClassPrivate);
    const auto helper = module.AllocateId();
    const auto main = module.AllocateId();
    const auto parameter = module.AllocateId();
    module.AddFunction(spv::OpFunction, voidType, helper, spv::FunctionControlDontInlineMask, taking);
    module.AddFunction(spv::OpFunctionParameter, functionU32, parameter);
    module.AddFunction(spv::OpLabel, module.AllocateId());
    module.AddFunction(spv::OpStore, shared, one);
    module.AddFunction(spv::OpStore, parameter, one);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    const auto indexed = module.AllocateId(), fixed = module.AllocateId(), passed = module.AllocateId(), spilled = module.AllocateId(), index = module.AllocateId();
    const auto dynamicElement = module.AllocateId(), fixedElement = module.AllocateId();
    module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, procedure);
    module.AddFunction(spv::OpLabel, module.AllocateId());
    module.AddFunction(spv::OpVariable, functionArray, indexed, spv::StorageClassFunction);
    module.AddFunction(spv::OpVariable, functionArray, fixed, spv::StorageClassFunction);
    module.AddFunction(spv::OpVariable, functionU32, passed, spv::StorageClassFunction);
    module.AddFunction(spv::OpVariable, functionU32, spilled, spv::StorageClassFunction);
    module.AddFunction(spv::OpLoad, u32, index, shared);
    module.AddFunction(spv::OpStore, own, index);
    module.AddFunction(spv::OpAccessChain, functionU32, dynamicElement, indexed, index);
    module.AddFunction(spv::OpStore, dynamicElement, one);
    module.AddFunction(spv::OpAccessChain, functionU32, fixedElement, fixed, one);
    module.AddFunction(spv::OpStore, fixedElement, one);
    module.AddFunction(spv::OpFunctionCall, voidType, module.AllocateId(), helper, passed);
    module.AddFunction(spv::OpStore, spilled, one, static_cast<std::uint32_t>(spv::MemoryAccessVolatileMask));
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", {});
    const auto users = CountLocalMemoryUsers(module.Finalize());
    check(users.dynamicallyIndexed == 1u, "a dynamically indexed Function array was not counted once, or a constant index was counted");
    check(users.passedToCalls == 1u, "a Function variable passed to a call was not counted once");
    check(users.volatileAccess == 1u, "a Function variable accessed as Volatile was not counted once");
    check(users.sharedPrivate == 1u, "a Private variable used by two functions was not counted once, or one used by one function was counted");
    check(users.total == 4u, "the local memory users were not counted once each");
}

}

int main() {
    testLocalMemoryUsersAreCounted();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
