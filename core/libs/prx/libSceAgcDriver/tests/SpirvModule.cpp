#include "SpirvBackend/SpirvModule.hpp"
#include "SpirvBackend/SpirvSpecialization.hpp"
#include <array>
#include <cstdio>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <string>
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

std::map<std::uint32_t, std::size_t> opcodeCounts(const std::vector<std::uint32_t>& words) {
    std::map<std::uint32_t, std::size_t> counts;
    std::size_t offset = 5;
    while (offset < words.size()) {
        const auto header = words[offset];
        const auto wordCount = header >> spv::WordCountShift;
        if (wordCount == 0 || offset + wordCount > words.size()) {
            return counts;
        }
        ++counts[header & 0xffffu];
        offset += wordCount;
    }
    return counts;
}

std::size_t declared(const SpirvModule& module, std::uint32_t opcode) {
    const auto counts = opcodeCounts(module.Finalize());
    const auto found = counts.find(opcode);
    return found == counts.end() ? 0 : found->second;
}

void testRepeatedTypesShareOneId() {
    SpirvModule module;
    const auto first = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto second = module.Type(spv::OpTypeInt, 32u, 0u);
    check(first == second, "a repeated OpTypeInt 32 0 returned a different id");
    check(declared(module, spv::OpTypeInt) == 1, "a repeated OpTypeInt 32 0 was declared more than once");
    check(module.Type(spv::OpTypeVoid) == module.Type(spv::OpTypeVoid), "a repeated OpTypeVoid returned a different id");
    check(declared(module, spv::OpTypeVoid) == 1, "OpTypeVoid was not declared exactly once");
    check(module.Type(spv::OpTypeBool) == module.Type(spv::OpTypeBool), "a repeated OpTypeBool returned a different id");
    check(module.Type(spv::OpTypeFloat, 32u) == module.Type(spv::OpTypeFloat, 32u), "a repeated OpTypeFloat 32 returned a different id");
    check(declared(module, spv::OpTypeFloat) == 1, "OpTypeFloat was not declared exactly once");
}

void testOperandShapesDoNotCollide() {
    SpirvModule module;
    const auto oneWord = module.Type(spv::OpTypeInt, 32u);
    const auto twoWords = module.Type(spv::OpTypeInt, 32u, 0u);
    check(oneWord != twoWords, "OpTypeInt with one operand collided with OpTypeInt with two");
    check(declared(module, spv::OpTypeInt) == 2, "the two OpTypeInt shapes were not both declared");
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto boolType = module.Type(spv::OpTypeBool);
    check(voidType != boolType, "OpTypeVoid collided with OpTypeBool");
    check(module.Type(spv::OpTypeInt, 32u, 0u) != module.Type(spv::OpTypeInt, 32u, 1u), "unsigned and signed OpTypeInt 32 collided");
    check(module.Type(spv::OpTypeInt, 32u, 0u) != module.Type(spv::OpTypeInt, 64u, 0u), "OpTypeInt 32 collided with OpTypeInt 64");
    check(module.Type(spv::OpTypeFloat, 32u) != module.Type(spv::OpTypeFloat, 64u), "OpTypeFloat 32 collided with OpTypeFloat 64");
}

void testOperandKindsAgree() {
    SpirvModule module;
    const auto fromUnsigned = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto fromSigned = module.Type(spv::OpTypeInt, static_cast<std::int32_t>(32), static_cast<std::int32_t>(0));
    check(fromUnsigned == fromSigned, "OpTypeInt disagreed between unsigned and signed operands");
    const auto typeU32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto fromEnum = module.Type(spv::OpTypePointer, spv::StorageClassFunction, typeU32);
    const auto fromWord = module.Type(spv::OpTypePointer, static_cast<std::uint32_t>(spv::StorageClassFunction), typeU32);
    check(fromEnum == fromWord, "OpTypePointer disagreed between enum and integer operands");
    check(fromEnum == module.Type(spv::OpTypePointer, spv::StorageClassFunction, typeU32), "a repeated enum-operand OpTypePointer returned a different id");
    check(declared(module, spv::OpTypePointer) == 1, "OpTypePointer was not declared exactly once");
}

void testDecoratedTypesStayDistinct() {
    SpirvModule module;
    const auto typeU32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto plain = module.Type(spv::OpTypeRuntimeArray, typeU32);
    const auto decorated = module.DecoratedType(spv::OpTypeRuntimeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, typeU32);
    check(plain != decorated, "a plain OpTypeRuntimeArray collided with the decorated one");
    check(decorated == module.DecoratedType(spv::OpTypeRuntimeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, typeU32), "a repeated decorated type returned a different id");
    check(declared(module, spv::OpTypeRuntimeArray) == 2, "the plain and decorated OpTypeRuntimeArray were not both declared");
}

void testConstantsShareOneId() {
    SpirvModule module;
    const auto typeU32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto typeBool = module.Type(spv::OpTypeBool);
    check(module.Constant(spv::OpConstant, typeU32, 0u) == module.Constant(spv::OpConstant, typeU32, 0u), "a repeated OpConstant 0 returned a different id");
    check(module.Constant(spv::OpConstant, typeU32, 0u) != module.Constant(spv::OpConstant, typeU32, 1u), "OpConstant 0 collided with OpConstant 1");
    check(module.Constant(spv::OpConstant, typeU32, 1000u) != module.Constant(spv::OpConstant, typeU32, 0u), "OpConstant 1000 collided with OpConstant 0");
    check(module.Constant(spv::OpConstantTrue, typeBool) == module.Constant(spv::OpConstantTrue, typeBool), "a repeated OpConstantTrue returned a different id");
    check(module.Constant(spv::OpConstantTrue, typeBool) != module.Constant(spv::OpConstantFalse, typeBool), "OpConstantTrue collided with OpConstantFalse");
    check(declared(module, spv::OpConstant) == 3, "OpConstant was not declared once per distinct value");
    check(declared(module, spv::OpConstantTrue) == 1, "OpConstantTrue was not declared exactly once");
    const auto signedId = module.Constant(spv::OpConstant, module.Type(spv::OpTypeInt, 32u, 1u), static_cast<std::uint32_t>(-1));
    check(signedId != 0, "a signed OpConstant produced a zero id");
}

void testOperandFreeConstantsShareOneId() {
    SpirvModule module;
    const auto typeBool = module.Type(spv::OpTypeBool);
    const auto typeVoid = module.Type(spv::OpTypeVoid);
    check(module.Constant(spv::OpConstantTrue, typeBool) == module.Constant(spv::OpConstantTrue, typeBool), "a repeated OpConstantTrue returned a different id");
    check(module.Constant(spv::OpConstantTrue, typeBool) != module.Constant(spv::OpConstantFalse, typeBool), "OpConstantTrue collided with OpConstantFalse");
    check(module.Constant(spv::OpConstantTrue, typeVoid) != module.Constant(spv::OpConstantTrue, typeBool), "OpConstantTrue of different types collided");
    check(declared(module, spv::OpConstantTrue) == 2, "OpConstantTrue was not declared once per type");
    check(declared(module, spv::OpConstantFalse) == 1, "OpConstantFalse was not declared exactly once");
    check(module.Type(spv::OpTypeVoid) == typeVoid, "a repeated operand-free type returned a different id");
}

void testIdsSurviveInterleavedAllocation() {
    SpirvModule module;
    const auto typeU32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto constant = module.Constant(spv::OpConstant, typeU32, 4u);
    for (int i = 0; i < 8; ++i) {
        check(module.AllocateId() != 0, "AllocateId returned zero");
        check(module.Type(spv::OpTypeInt, 32u, 0u) == typeU32, "a memoized type changed id after AllocateId");
        check(module.Constant(spv::OpConstant, typeU32, 4u) == constant, "a memoized constant changed id after AllocateId");
    }
}

void testModulesAreIndependent() {
    SpirvModule first;
    SpirvModule second;
    const auto firstId = first.Type(spv::OpTypeInt, 32u, 0u);
    const auto secondId = second.Type(spv::OpTypeInt, 32u, 0u);
    check(firstId == 1, "the first id of a fresh module was not 1");
    check(secondId == 1, "a second module did not start its ids from 1");
    first.Type(spv::OpTypeVoid);
    check(second.Type(spv::OpTypeInt, 32u, 0u) == secondId, "a second module's ids moved when the first module was used");
}

void testIdsStayUniqueBeyondCacheCapacity() {
    SpirvModule module;
    const auto typeU32 = module.Type(spv::OpTypeInt, 32u, 0u);
    std::vector<std::uint32_t> ids;
    ids.reserve(4096);
    for (std::uint32_t value = 0; value < 4096; ++value) {
        ids.push_back(module.Constant(spv::OpConstant, typeU32, value));
    }
    const std::set<std::uint32_t> distinct(ids.begin(), ids.end());
    check(distinct.size() == ids.size(), "distinct constants shared an id once the cache was overrun");
    for (std::uint32_t value = 0; value < 4096; ++value) {
        if (module.Constant(spv::OpConstant, typeU32, value) != ids[value]) {
            std::fprintf(stderr, "constant %u changed id after the cache was overrun\n", value);
            ++failures;
            break;
        }
    }
    std::uint32_t largest = 0;
    for (const auto id : distinct) {
        largest = largest < id ? id : largest;
    }
    const auto first = ids.front();
    check(largest - first + 1 == distinct.size(), "the constant ids were not contiguous, so some were never declared");
    check(first > typeU32, "a constant was numbered before the type it uses");
}

void testEmittedWordsAreStable() {
    SpirvModule module;
    const auto typeU32 = module.Type(spv::OpTypeInt, 32u, 0u);
    for (int i = 0; i < 32; ++i) {
        module.Type(spv::OpTypeVector, typeU32, 4u);
        module.Type(spv::OpTypePointer, spv::StorageClassFunction, typeU32);
        module.Constant(spv::OpConstant, typeU32, static_cast<std::uint32_t>(i % 8));
        module.AddFunction(spv::OpNop);
    }
    const auto words = module.Finalize();
    check(!words.empty(), "Finalize produced no words");
    const auto counts = opcodeCounts(words);
    check(counts.at(spv::OpTypeVector) == 1, "OpTypeVector was declared more than once");
    check(counts.at(spv::OpTypePointer) == 1, "OpTypePointer was declared more than once");
    check(counts.at(spv::OpConstant) == 8, "OpConstant was not declared once per distinct value");
    check(counts.at(spv::OpNop) == 32, "the function body lost instructions");
}

}

void testStructExtractsAreNotShuffled() {
    const auto op = [](std::uint32_t count, spv::Op code) { return (count << 16u) | static_cast<std::uint32_t>(code); };
    for (const bool vectorSource : {false, true}) {
        const std::vector<std::uint32_t> words{
            spv::MagicNumber, 0x00010300u, 0u, 20u, 0u,
            op(2, spv::OpCapability), spv::CapabilityShader,
            op(3, spv::OpMemoryModel), spv::AddressingModelLogical, spv::MemoryModelGLSL450,
            op(6, spv::OpEntryPoint), spv::ExecutionModelGLCompute, 10u, 0x6e69616du, 0u, 9u,
            op(6, spv::OpExecutionMode), 10u, spv::ExecutionModeLocalSize, 1u, 1u, 1u,
            op(2, spv::OpTypeVoid), 1u,
            op(4, spv::OpTypeInt), 2u, 32u, 0u,
            vectorSource ? op(4, spv::OpTypeVector) : op(4, spv::OpTypeStruct), 3u, 2u, 2u,
            op(4, spv::OpTypeVector), 4u, 2u, 4u,
            op(4, spv::OpTypePointer), 5u, spv::StorageClassPrivate, 4u,
            op(3, spv::OpTypeFunction), 6u, 1u,
            op(4, spv::OpConstant), 2u, 7u, 1u,
            op(4, spv::OpConstant), 2u, 8u, 2u,
            op(4, spv::OpVariable), 5u, 9u, spv::StorageClassPrivate,
            op(5, spv::OpFunction), 1u, 10u, 0u, 6u,
            op(2, spv::OpLabel), 11u,
            op(5, spv::OpCompositeConstruct), 3u, 12u, 7u, 8u,
            op(5, spv::OpCompositeExtract), 2u, 13u, 12u, 0u,
            op(5, spv::OpCompositeExtract), 2u, 14u, 12u, 1u,
            op(7, spv::OpCompositeConstruct), 4u, 15u, 13u, 14u, 13u, 13u,
            op(3, spv::OpStore), 9u, 15u,
            op(1, spv::OpReturn),
            op(1, spv::OpFunctionEnd),
        };
        const auto specialized = ShaderRecompiler::SpecializeSpirv(words);
        bool shuffled = false;
        for (std::size_t cursor = 5; cursor < specialized.size(); cursor += specialized[cursor] >> 16u) {
            if ((specialized[cursor] >> 16u) == 0u) break;
            if ((specialized[cursor] & 0xffffu) == spv::OpVectorShuffle) shuffled = true;
        }
        check(shuffled == vectorSource, vectorSource ? "extracts of one vector were not folded into a shuffle" : "extracts of a struct were folded into a vector shuffle");
    }
}

std::vector<std::uint32_t> foldedSwitchModule(std::uint32_t deadSum, std::array<std::uint32_t, 3> localSize) {
    const auto op = [](std::uint32_t count, spv::Op code) { return (count << 16u) | static_cast<std::uint32_t>(code); };
    return {
        spv::MagicNumber, 0x00010300u, 0u, 40u, 0u,
        op(2, spv::OpCapability), spv::CapabilityShader,
        op(3, spv::OpMemoryModel), spv::AddressingModelLogical, spv::MemoryModelGLSL450,
        op(6, spv::OpEntryPoint), spv::ExecutionModelGLCompute, 11u, 0x6e69616du, 0u, 10u,
        op(6, spv::OpExecutionMode), 11u, spv::ExecutionModeLocalSize, localSize[0], localSize[1], localSize[2],
        op(2, spv::OpTypeVoid), 1u,
        op(4, spv::OpTypeInt), 2u, 32u, 0u,
        op(4, spv::OpTypePointer), 4u, spv::StorageClassPrivate, 2u,
        op(3, spv::OpTypeFunction), 5u, 1u,
        op(4, spv::OpConstant), 2u, 6u, 2u,
        op(4, spv::OpConstant), 2u, 7u, 10u,
        op(4, spv::OpConstant), 2u, 8u, 20u,
        op(4, spv::OpVariable), 4u, 10u, spv::StorageClassPrivate,
        op(5, spv::OpFunction), 1u, 11u, 0u, 5u,
        op(2, spv::OpLabel), 12u,
        op(5, spv::OpIAdd), 2u, deadSum, 6u, 7u,
        op(5, spv::OpIMul), 2u, 14u, deadSum, 7u,
        op(3, spv::OpSelectionMerge), 20u, spv::SelectionControlMaskNone,
        op(7, spv::OpSwitch), 6u, 15u, 1u, 16u, 2u, 17u,
        op(2, spv::OpLabel), 15u,
        op(3, spv::OpStore), 10u, 7u,
        op(2, spv::OpBranch), 20u,
        op(2, spv::OpLabel), 16u,
        op(3, spv::OpStore), 10u, 8u,
        op(2, spv::OpBranch), 20u,
        op(2, spv::OpLabel), 17u,
        op(3, spv::OpStore), 10u, 14u,
        op(2, spv::OpBranch), 20u,
        op(2, spv::OpLabel), 20u,
        op(1, spv::OpReturn),
        op(1, spv::OpFunctionEnd),
    };
}

void checkFoldedSwitch(const std::vector<std::uint32_t>& words, std::uint32_t deadSum, const char* what) {
    const auto specialized = ShaderRecompiler::SpecializeSpirv(words);
    const auto counts = opcodeCounts(specialized);
    std::set<std::uint32_t> constants;
    bool storesFolded = false;
    for (std::size_t cursor = 5; cursor < specialized.size(); cursor += specialized[cursor] >> 16u) {
        if ((specialized[cursor] >> 16u) == 0u) break;
        const auto opcode = specialized[cursor] & 0xffffu;
        if (opcode == spv::OpConstant) constants.insert(specialized[cursor + 2u]);
        if (opcode == spv::OpStore && specialized[cursor + 2u] == 14u) storesFolded = true;
    }
    const std::string context = std::string(" (") + what + ")";
    check(!counts.contains(spv::OpUnreachable), ("blocks the folded switch left unreachable were kept" + context).c_str());
    check(counts.at(spv::OpLabel) == 1u, ("the three live blocks the folded switch leaves in a chain were not merged into one" + context).c_str());
    check(storesFolded && constants.contains(14u), ("the folded product the live case stores is missing" + context).c_str());
    check(!constants.contains(deadSum), ("a folded constant without uses was kept" + context).c_str());
}

void testFoldingLeavesNoDeadBlocksOrConstants() {
    checkFoldedSwitch(foldedSwitchModule(13u, {1u, 1u, 1u}), 13u, "no literal names a dead id");
    checkFoldedSwitch(foldedSwitchModule(32u, {15u, 16u, 1u}), 32u, "the OpTypeInt width and the LocalSize literals equal the dead ids");
}

std::vector<std::uint32_t> internedConstantsModule(std::uint32_t selector) {
    const auto op = [](std::uint32_t count, spv::Op code) { return (count << 16u) | static_cast<std::uint32_t>(code); };
    return {
        spv::MagicNumber, 0x00010300u, 0u, 50u, 0u,
        op(2, spv::OpCapability), spv::CapabilityShader,
        op(3, spv::OpMemoryModel), spv::AddressingModelLogical, spv::MemoryModelGLSL450,
        op(5, spv::OpEntryPoint), spv::ExecutionModelGLCompute, 30u, 0x6e69616du, 0u,
        op(6, spv::OpExecutionMode), 30u, spv::ExecutionModeLocalSize, 1u, 1u, 1u,
        op(3, spv::OpDecorate), 13u, spv::DecorationRelaxedPrecision,
        op(2, spv::OpTypeVoid), 1u,
        op(4, spv::OpTypeInt), 2u, 32u, 0u,
        op(4, spv::OpTypeInt), 3u, 32u, 1u,
        op(2, spv::OpTypeBool), 4u,
        op(4, spv::OpTypePointer), 5u, spv::StorageClassPrivate, 2u,
        op(4, spv::OpTypePointer), 6u, spv::StorageClassPrivate, 3u,
        op(4, spv::OpTypePointer), 7u, spv::StorageClassPrivate, 4u,
        op(3, spv::OpTypeFunction), 8u, 1u,
        op(4, spv::OpConstant), 2u, 9u, 1u,
        op(4, spv::OpConstant), 2u, 10u, 2u,
        op(4, spv::OpConstant), 2u, 11u, 3u,
        op(4, spv::OpConstant), 3u, 12u, 1u,
        op(4, spv::OpConstant), 2u, 13u, 3u,
        op(4, spv::OpConstant), 2u, 14u, selector,
        op(4, spv::OpVariable), 5u, 15u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 5u, 16u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 5u, 17u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 5u, 18u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 6u, 19u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 7u, 20u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 7u, 21u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 5u, 22u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 5u, 23u, spv::StorageClassPrivate,
        op(4, spv::OpVariable), 5u, 24u, spv::StorageClassPrivate,
        op(5, spv::OpFunction), 1u, 30u, 0u, 8u,
        op(2, spv::OpLabel), 31u,
        op(4, spv::OpLoad), 2u, 40u, 24u,
        op(5, spv::OpIAdd), 2u, 41u, 40u, 11u,
        op(3, spv::OpStore), 24u, 41u,
        op(5, spv::OpIAdd), 2u, 32u, 9u, 10u,
        op(5, spv::OpIMul), 2u, 33u, 11u, 9u,
        op(5, spv::OpISub), 2u, 34u, 11u, 10u,
        op(5, spv::OpIAdd), 2u, 35u, 10u, 10u,
        op(5, spv::OpShiftLeftLogical), 2u, 36u, 9u, 10u,
        op(5, spv::OpIMul), 3u, 37u, 12u, 12u,
        op(5, spv::OpULessThan), 4u, 38u, 9u, 10u,
        op(5, spv::OpIEqual), 4u, 39u, 14u, 14u,
        op(3, spv::OpStore), 15u, 32u,
        op(3, spv::OpStore), 16u, 33u,
        op(3, spv::OpStore), 17u, 34u,
        op(3, spv::OpStore), 18u, 35u,
        op(3, spv::OpStore), 22u, 36u,
        op(3, spv::OpStore), 19u, 37u,
        op(3, spv::OpStore), 20u, 38u,
        op(3, spv::OpStore), 21u, 39u,
        op(3, spv::OpStore), 23u, 13u,
        op(1, spv::OpReturn),
        op(1, spv::OpFunctionEnd),
    };
}

void testFoldedValuesShareConstants() {
    const auto specialized = ShaderRecompiler::SpecializeSpirv(internedConstantsModule(4u));
    struct Constant {
        std::uint32_t opcode;
        std::uint32_t type;
        std::uint32_t value;
    };
    std::map<std::uint32_t, Constant> constants;
    std::map<std::uint32_t, std::uint32_t> stored;
    std::set<std::vector<std::uint32_t>> keys;
    bool shared = true;
    for (std::size_t cursor = 5; cursor < specialized.size(); cursor += specialized[cursor] >> 16u) {
        if ((specialized[cursor] >> 16u) == 0u) break;
        const auto opcode = specialized[cursor] & 0xffffu;
        if (opcode == spv::OpConstant || opcode == spv::OpConstantTrue || opcode == spv::OpConstantFalse) {
            const auto value = opcode == spv::OpConstant ? specialized[cursor + 3u] : 0u;
            constants.emplace(specialized[cursor + 2u], Constant{opcode, specialized[cursor + 1u], value});
            if (specialized[cursor + 2u] != 13u) shared &= keys.insert({opcode, specialized[cursor + 1u], value}).second;
        }
        if (opcode == spv::OpStore) stored[specialized[cursor + 1u]] = specialized[cursor + 2u];
    }
    const auto holds = [&](std::uint32_t variable, std::uint32_t opcode, std::uint32_t type, std::uint32_t value) {
        const auto id = stored.find(variable);
        const auto constant = id == stored.end() ? constants.end() : constants.find(id->second);
        return constant != constants.end() && constant->second.opcode == opcode && constant->second.type == type && constant->second.value == value;
    };
    check(shared, "a folded value declared another constant of a type and value the module already has");
    check(constants.size() == 6u, "the specialized module does not declare exactly one constant per type and value");
    check(holds(15u, spv::OpConstant, 2u, 3u) && holds(16u, spv::OpConstant, 2u, 3u) && holds(17u, spv::OpConstant, 2u, 1u), "a store lost the folded value it writes");
    check(stored.at(15u) == 11u && stored.at(16u) == 11u, "a folded value did not reuse the module constant of its type and value");
    check(holds(18u, spv::OpConstant, 2u, 4u) && holds(22u, spv::OpConstant, 2u, 4u) && stored.at(18u) == stored.at(22u), "two folds of one new value did not share its constant");
    check(holds(19u, spv::OpConstant, 3u, 1u), "a signed folded value shares a constant of the unsigned type");
    check(holds(20u, spv::OpConstantTrue, 4u, 0u) && holds(21u, spv::OpConstantTrue, 4u, 0u), "a folded comparison lost its value");
    check(stored.at(23u) == 13u, "a store of a decorated constant was moved to an undecorated one");
    check(specialized == ShaderRecompiler::SpecializeSpirv(internedConstantsModule(5u)), "variants that differ only in a constant the folding consumes specialized to different SPIR-V");
    check(ShaderRecompiler::SpecializeSpirv(specialized) == specialized, "constant interning is not stable");
}

void testOperationsOneOperandDecidesFold() {
    constexpr std::uint32_t uintType = 2u, intType = 3u, boolType = 4u, floatType = 5u;
    constexpr std::uint32_t zero = 20u, one = 21u, allOnes = 22u, lowBits = 23u, signedZero = 24u, signedOne = 25u, trueValue = 26u, falseValue = 27u, floatZero = 28u, floatOne = 29u, otherZero = 31u;
    constexpr std::uint32_t x = 62u, y = 63u, b = 64u, f = 65u, none = 0u;
    struct Case {
        spv::Op op;
        std::uint32_t type;
        std::vector<std::uint32_t> operands;
        std::uint32_t operand;
        std::optional<std::uint32_t> constant;
    };
    const std::vector<Case> cases{
        {spv::OpIAdd, uintType, {x, zero}, x, {}},
        {spv::OpIAdd, uintType, {zero, x}, x, {}},
        {spv::OpISub, uintType, {x, zero}, x, {}},
        {spv::OpBitwiseOr, uintType, {x, zero}, x, {}},
        {spv::OpBitwiseOr, uintType, {zero, x}, x, {}},
        {spv::OpBitwiseXor, uintType, {zero, x}, x, {}},
        {spv::OpIMul, uintType, {x, one}, x, {}},
        {spv::OpIMul, uintType, {one, x}, x, {}},
        {spv::OpUDiv, uintType, {x, one}, x, {}},
        {spv::OpShiftLeftLogical, uintType, {x, zero}, x, {}},
        {spv::OpShiftRightLogical, uintType, {x, zero}, x, {}},
        {spv::OpShiftRightArithmetic, intType, {y, zero}, y, {}},
        {spv::OpSDiv, intType, {y, signedOne}, y, {}},
        {spv::OpBitwiseAnd, uintType, {x, allOnes}, x, {}},
        {spv::OpBitwiseAnd, uintType, {allOnes, x}, x, {}},
        {spv::OpLogicalAnd, boolType, {b, trueValue}, b, {}},
        {spv::OpLogicalAnd, boolType, {trueValue, b}, b, {}},
        {spv::OpLogicalOr, boolType, {b, falseValue}, b, {}},
        {spv::OpLogicalEqual, boolType, {b, trueValue}, b, {}},
        {spv::OpLogicalNotEqual, boolType, {falseValue, b}, b, {}},
        {spv::OpSelect, uintType, {b, x, x}, x, {}},
        {spv::OpSelect, floatType, {b, f, f}, f, {}},
        {spv::OpSelect, uintType, {b, zero, otherZero}, none, 0u},
        {spv::OpBitwiseAnd, uintType, {x, zero}, none, 0u},
        {spv::OpBitwiseAnd, uintType, {zero, x}, none, 0u},
        {spv::OpIMul, uintType, {zero, x}, none, 0u},
        {spv::OpBitwiseOr, uintType, {x, allOnes}, none, 0xffffffffu},
        {spv::OpLogicalAnd, boolType, {b, falseValue}, none, 0u},
        {spv::OpLogicalOr, boolType, {trueValue, b}, none, 1u},
        {spv::OpISub, uintType, {zero, x}, none, {}},
        {spv::OpUDiv, uintType, {one, x}, none, {}},
        {spv::OpShiftRightLogical, uintType, {one, x}, none, {}},
        {spv::OpBitwiseAnd, uintType, {x, lowBits}, none, {}},
        {spv::OpBitwiseOr, uintType, {x, lowBits}, none, {}},
        {spv::OpIAdd, uintType, {y, signedZero}, none, {}},
        {spv::OpFAdd, floatType, {f, floatZero}, none, {}},
        {spv::OpFMul, floatType, {f, floatOne}, none, {}},
        {spv::OpFMul, floatType, {f, floatZero}, none, {}},
        {spv::OpSelect, uintType, {b, x, zero}, none, {}},
    };
    std::vector<std::uint32_t> words{spv::MagicNumber, 0x00010300u, 0u, 300u, 0u};
    const auto emit = [&](spv::Op op, std::initializer_list<std::uint32_t> operands) {
        words.push_back((static_cast<std::uint32_t>(operands.size() + 1u) << 16u) | op);
        words.insert(words.end(), operands);
    };
    emit(spv::OpCapability, {spv::CapabilityShader});
    emit(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    emit(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, 60u, 0x6e69616du, 0u});
    emit(spv::OpExecutionMode, {60u, spv::ExecutionModeLocalSize, 1u, 1u, 1u});
    emit(spv::OpTypeVoid, {1u});
    emit(spv::OpTypeInt, {uintType, 32u, 0u});
    emit(spv::OpTypeInt, {intType, 32u, 1u});
    emit(spv::OpTypeBool, {boolType});
    emit(spv::OpTypeFloat, {floatType, 32u});
    const std::map<std::uint32_t, std::uint32_t> pointers{{uintType, 6u}, {intType, 7u}, {boolType, 8u}, {floatType, 9u}};
    for (const auto& [type, pointer] : pointers) emit(spv::OpTypePointer, {pointer, spv::StorageClassPrivate, type});
    emit(spv::OpTypeFunction, {10u, 1u});
    emit(spv::OpConstant, {uintType, zero, 0u});
    emit(spv::OpConstant, {uintType, one, 1u});
    emit(spv::OpConstant, {uintType, allOnes, 0xffffffffu});
    emit(spv::OpConstant, {uintType, lowBits, 0x7fffffffu});
    emit(spv::OpConstant, {intType, signedZero, 0u});
    emit(spv::OpConstant, {intType, signedOne, 1u});
    emit(spv::OpConstantTrue, {boolType, trueValue});
    emit(spv::OpConstantFalse, {boolType, falseValue});
    emit(spv::OpConstant, {floatType, floatZero, 0u});
    emit(spv::OpConstant, {floatType, floatOne, 0x3f800000u});
    emit(spv::OpConstant, {uintType, otherZero, 0u});
    emit(spv::OpVariable, {pointers.at(uintType), 40u, spv::StorageClassPrivate});
    emit(spv::OpVariable, {pointers.at(intType), 41u, spv::StorageClassPrivate});
    emit(spv::OpVariable, {pointers.at(boolType), 42u, spv::StorageClassPrivate});
    emit(spv::OpVariable, {pointers.at(floatType), 43u, spv::StorageClassPrivate});
    for (std::uint32_t index = 0; index < cases.size(); ++index) emit(spv::OpVariable, {pointers.at(cases[index].type), 200u + index, spv::StorageClassPrivate});
    emit(spv::OpFunction, {1u, 60u, spv::FunctionControlMaskNone, 10u});
    emit(spv::OpLabel, {61u});
    emit(spv::OpLoad, {uintType, x, 40u});
    emit(spv::OpLoad, {intType, y, 41u});
    emit(spv::OpLoad, {boolType, b, 42u});
    emit(spv::OpLoad, {floatType, f, 43u});
    for (std::uint32_t index = 0; index < cases.size(); ++index) {
        const auto& test = cases[index];
        words.push_back((static_cast<std::uint32_t>(test.operands.size() + 3u) << 16u) | test.op);
        words.insert(words.end(), {test.type, 100u + index});
        words.insert(words.end(), test.operands.begin(), test.operands.end());
        emit(spv::OpStore, {200u + index, 100u + index});
    }
    emit(spv::OpReturn, {});
    emit(spv::OpFunctionEnd, {});
    const auto specialized = ShaderRecompiler::SpecializeSpirv(words);
    std::set<std::uint32_t> producers{spv::OpLoad};
    for (const auto& test : cases) producers.insert(test.op);
    struct Constant {
        std::uint32_t type;
        std::uint32_t value;
    };
    std::map<std::uint32_t, std::uint32_t> definitions;
    std::map<std::uint32_t, Constant> constants;
    std::map<std::uint32_t, std::uint32_t> stored;
    std::size_t copies = 0;
    for (std::size_t cursor = 5; cursor < specialized.size(); cursor += specialized[cursor] >> 16u) {
        if ((specialized[cursor] >> 16u) == 0u) break;
        const auto opcode = specialized[cursor] & 0xffffu;
        if (producers.contains(opcode)) definitions[specialized[cursor + 2u]] = opcode;
        if (opcode == spv::OpConstant) constants[specialized[cursor + 2u]] = {specialized[cursor + 1u], specialized[cursor + 3u]};
        if (opcode == spv::OpConstantTrue || opcode == spv::OpConstantFalse) constants[specialized[cursor + 2u]] = {specialized[cursor + 1u], opcode == spv::OpConstantTrue ? 1u : 0u};
        if (opcode == spv::OpStore) stored[specialized[cursor + 1u]] = specialized[cursor + 2u];
        if (opcode == spv::OpCopyObject) ++copies;
    }
    for (std::uint32_t index = 0; index < cases.size(); ++index) {
        const auto& test = cases[index];
        const auto value = stored.at(200u + index);
        const auto context = " (case " + std::to_string(index) + ")";
        const auto constant = constants.find(value);
        if (test.operand != none) check(value == test.operand, ("an identity operation was not replaced by its operand" + context).c_str());
        else if (test.constant) check(constant != constants.end() && constant->second.type == test.type && constant->second.value == *test.constant, ("an operation one constant operand decides was not replaced by its value" + context).c_str());
        else check(value == 100u + index && definitions.contains(value) && definitions.at(value) == static_cast<std::uint32_t>(test.op), ("an operation that one operand does not decide for every input was folded" + context).c_str());
    }
    check(copies == 0u, "the folding left copies behind");
    check(ShaderRecompiler::SpecializeSpirv(specialized) == specialized, "folding decided operations is not stable");
}

void testSingleSuccessorChainsMerge() {
    std::vector<std::uint32_t> words{spv::MagicNumber, 0x00010300u, 0u, 100u, 0u};
    const auto emit = [&](spv::Op op, std::initializer_list<std::uint32_t> operands) {
        words.push_back((static_cast<std::uint32_t>(operands.size() + 1u) << 16u) | op);
        words.insert(words.end(), operands);
    };
    emit(spv::OpCapability, {spv::CapabilityShader});
    emit(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    emit(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, 40u, 0x6e69616du, 0u});
    emit(spv::OpExecutionMode, {40u, spv::ExecutionModeLocalSize, 1u, 1u, 1u});
    emit(spv::OpName, {16u, 0x00747865u});
    emit(spv::OpTypeVoid, {1u});
    emit(spv::OpTypeInt, {2u, 32u, 0u});
    emit(spv::OpTypeBool, {3u});
    emit(spv::OpTypePointer, {4u, spv::StorageClassPrivate, 2u});
    emit(spv::OpTypeFunction, {5u, 1u});
    emit(spv::OpConstant, {2u, 6u, 0u});
    emit(spv::OpConstant, {2u, 7u, 1u});
    emit(spv::OpConstant, {2u, 8u, 2u});
    emit(spv::OpVariable, {4u, 9u, spv::StorageClassPrivate});
    emit(spv::OpVariable, {4u, 50u, spv::StorageClassPrivate});
    emit(spv::OpFunction, {1u, 40u, spv::FunctionControlMaskNone, 5u});
    emit(spv::OpLabel, {10u});
    emit(spv::OpLoad, {2u, 60u, 50u});
    emit(spv::OpIEqual, {3u, 61u, 60u, 6u});
    emit(spv::OpBranch, {11u});
    emit(spv::OpLabel, {11u});
    emit(spv::OpStore, {9u, 7u});
    emit(spv::OpBranch, {12u});
    emit(spv::OpLabel, {12u});
    emit(spv::OpSelectionMerge, {15u, spv::SelectionControlMaskNone});
    emit(spv::OpBranchConditional, {61u, 13u, 14u});
    emit(spv::OpLabel, {13u});
    emit(spv::OpStore, {9u, 8u});
    emit(spv::OpBranch, {16u});
    emit(spv::OpLabel, {16u});
    emit(spv::OpBranch, {15u});
    emit(spv::OpLabel, {14u});
    emit(spv::OpBranch, {15u});
    emit(spv::OpLabel, {15u});
    emit(spv::OpPhi, {2u, 62u, 7u, 16u, 8u, 14u});
    emit(spv::OpStore, {9u, 62u});
    emit(spv::OpBranch, {20u});
    emit(spv::OpLabel, {20u});
    emit(spv::OpLoopMerge, {23u, 22u, spv::LoopControlMaskNone});
    emit(spv::OpBranch, {21u});
    emit(spv::OpLabel, {21u});
    emit(spv::OpLoad, {2u, 63u, 50u});
    emit(spv::OpIEqual, {3u, 64u, 63u, 7u});
    emit(spv::OpBranchConditional, {64u, 23u, 24u});
    emit(spv::OpLabel, {24u});
    emit(spv::OpBranch, {22u});
    emit(spv::OpLabel, {22u});
    emit(spv::OpBranch, {20u});
    emit(spv::OpLabel, {23u});
    emit(spv::OpBranch, {30u});
    emit(spv::OpLabel, {30u});
    emit(spv::OpSelectionMerge, {33u, spv::SelectionControlMaskNone});
    emit(spv::OpBranchConditional, {61u, 31u, 32u});
    emit(spv::OpLabel, {31u});
    emit(spv::OpReturn, {});
    emit(spv::OpLabel, {32u});
    emit(spv::OpBranch, {33u});
    emit(spv::OpLabel, {33u});
    emit(spv::OpReturn, {});
    emit(spv::OpFunctionEnd, {});
    const auto specialized = ShaderRecompiler::SpecializeSpirv(words);
    std::vector<std::uint32_t> labels;
    std::vector<std::uint32_t> parents;
    std::size_t stores = 0;
    std::size_t names = 0;
    for (std::size_t cursor = 5; cursor < specialized.size(); cursor += specialized[cursor] >> 16u) {
        const auto count = specialized[cursor] >> 16u;
        if (count == 0u) break;
        const auto opcode = specialized[cursor] & 0xffffu;
        if (opcode == spv::OpLabel) labels.push_back(specialized[cursor + 1u]);
        if (opcode == spv::OpPhi) {
            for (std::size_t operand = 4; operand < count; operand += 2u) parents.push_back(specialized[cursor + operand]);
        }
        if (opcode == spv::OpStore) ++stores;
        if (opcode == spv::OpName) ++names;
    }
    check(labels == std::vector<std::uint32_t>{10u, 13u, 14u, 15u, 20u, 21u, 24u, 22u, 23u, 31u, 32u, 33u}, "blocks whose only predecessor branches to them were not merged, or a merge block, continue target, loop header or loop body was merged");
    check(parents == std::vector<std::uint32_t>{13u, 14u}, "a phi still names a merged block as its parent");
    check(stores == 3u, "block merging lost a store");
    check(names == 0u, "the name of a merged block was kept");
    check(ShaderRecompiler::SpecializeSpirv(specialized) == specialized, "block merging is not stable");
}

int main() {
    testFoldingLeavesNoDeadBlocksOrConstants();
    testFoldedValuesShareConstants();
    testOperationsOneOperandDecidesFold();
    testSingleSuccessorChainsMerge();
    testStructExtractsAreNotShuffled();
    testRepeatedTypesShareOneId();
    testOperandShapesDoNotCollide();
    testOperandKindsAgree();
    testDecoratedTypesStayDistinct();
    testConstantsShareOneId();
    testOperandFreeConstantsShareOneId();
    testIdsSurviveInterleavedAllocation();
    testModulesAreIndependent();
    testIdsStayUniqueBeyondCacheCapacity();
    testEmittedWordsAreStable();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
