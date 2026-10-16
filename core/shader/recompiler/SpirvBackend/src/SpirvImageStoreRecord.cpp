#include "SpirvBackend/SpirvImageEmitter.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvMemory/SpirvSubgroup.hpp"
#include "BdaAbi.hpp"
#include "StorageImagePages.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <span>
#include <spirv/unified1/GLSL.std.450.h>
#include <vector>

namespace ShaderRecompiler {

namespace {

namespace Pages = StorageImagePages;

std::uint32_t ConstantAt(SpirvEmitterState& state, std::span<const std::uint32_t> values, std::uint32_t index, std::uint32_t first = 0u) {
    if (values.size() == 1u) return ConstantU32(state, values.front());
    const auto half = static_cast<std::uint32_t>(values.size() / 2u);
    const auto low = ConstantAt(state, values.first(half), index, first);
    const auto high = ConstantAt(state, values.subspan(half), index, first + half);
    return Select(state, TypeU32(state), Binary(state, spv::OpULessThan, TypeBool(state), index, ConstantU32(state, first + half)), low, high);
}

template<typename TBody>
void EmitCountedLoop(SpirvEmitterState& state, std::uint32_t counter, std::uint32_t limit, TBody&& body) {
    const auto u32 = TypeU32(state);
    state.module.AddFunction(spv::OpStore, counter, ConstantU32(state, 0u));
    const auto header = state.module.AllocateId();
    const auto loop = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, current, counter);
    const auto more = Binary(state, spv::OpULessThan, TypeBool(state), current, limit);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, more, loop, merge);
    EmitLabel(state, loop);
    body(current);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, counter, Binary(state, spv::OpIAdd, u32, current, ConstantU32(state, 1u)));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
}

void ReturnUnless(SpirvEmitterState& state, std::uint32_t condition, const std::function<void()>& before) {
    const auto stop = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, next, stop);
    EmitLabel(state, stop);
    if (before) before();
    state.module.AddFunction(spv::OpReturn);
    EmitLabel(state, next);
}

}

bool ImageStoresRecorded(const IrProgram& program, std::span<const std::uint32_t> capabilities) {
    const auto& info = program.Info();
    const bool writes = std::ranges::any_of(info.images, [](const ImageResource& image) { return image.written || image.atomic; });
    return writes && (info.usesFaultBuffer || info.usesDma) && std::ranges::find(capabilities, static_cast<std::uint32_t>(spv::CapabilityInt64)) != capabilities.end();
}

void DefineImageStoreRecord(SpirvEmitterState& state) {
    if (state.faultBufferVariable == 0u || !ImageStoresRecorded(state.program, state.supportedCapabilities)) return;
    state.module.EmitCapability(spv::CapabilityInt64);
    state.writtenChunkNoteFunction = DefineWrittenChunkNote(state);
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto formats = Pages::FormatTable();
    std::array<std::uint32_t, 60> locations{};
    for (std::uint32_t element = 0; element < 5u; ++element) std::copy(Pages::TailLocations64KB[element].begin(), Pages::TailLocations64KB[element].end(), locations.begin() + element * 12u);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto wide = [&](std::uint64_t value) { return BdaConstant(state, value); };
    const auto op = [&](spv::Op code, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, code, type, left, right); };
    const auto add = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpIAdd, u32, left, right); };
    const auto sub = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpISub, u32, left, right); };
    const auto andBits = [&](std::uint32_t value, std::uint32_t mask) { return op(spv::OpBitwiseAnd, u32, value, constant(mask)); };
    const auto shr = [&](std::uint32_t value, std::uint32_t shift) { return op(spv::OpShiftRightLogical, u32, value, shift); };
    const auto shl = [&](std::uint32_t value, std::uint32_t shift) { return op(spv::OpShiftLeftLogical, u32, value, shift); };
    const auto field = [&](std::uint32_t value, std::uint32_t offset, std::uint32_t mask) { return andBits(shr(value, constant(offset)), mask); };
    const auto equal = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpIEqual, boolean, left, right); };
    const auto is = [&](std::uint32_t value, std::uint32_t expected) { return equal(value, constant(expected)); };
    const auto less = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpULessThan, boolean, left, right); };
    const auto lessEqual = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpULessThanEqual, boolean, left, right); };
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpLogicalAnd, boolean, left, right); };
    const auto either = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpLogicalOr, boolean, left, right); };
    const auto none = [&](std::uint32_t value) { return Unary(state, spv::OpLogicalNot, boolean, value); };
    const auto pick = [&](std::uint32_t type, std::uint32_t condition, std::uint32_t whenTrue, std::uint32_t whenFalse) { return Select(state, type, condition, whenTrue, whenFalse); };
    const auto any = [&](std::uint32_t value, std::initializer_list<std::uint32_t> candidates) {
        std::uint32_t result = 0;
        for (const auto candidate : candidates) result = result == 0u ? is(value, candidate) : either(result, is(value, candidate));
        return result;
    };
    const auto widen = [&](std::uint32_t value) { return Unary(state, spv::OpUConvert, u64, value); };
    const auto umin = [&](std::uint32_t left, std::uint32_t right) { return EmitGlsl<GLSLstd450UMin, IrType::U32>(state, left, right); };
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto supports = [&](spv::Capability capability) { return std::ranges::find(state.supportedCapabilities, static_cast<std::uint32_t>(capability)) != state.supportedCapabilities.end(); };
    const bool shared = !HostSubgroupSingleLane(state) && supports(spv::CapabilityGroupNonUniformBallot) && supports(spv::CapabilityGroupNonUniformArithmetic);
    if (shared) {
        state.module.EmitCapability(spv::CapabilityGroupNonUniform);
        state.module.EmitCapability(spv::CapabilityGroupNonUniformBallot);
        state.module.EmitCapability(spv::CapabilityGroupNonUniformArithmetic);
    }
    std::uint32_t pendingVariable = 0;
    const auto call = [&](std::uint32_t key, std::uint32_t bit) {
        state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.writtenChunkNoteFunction, key, bit);
    };
    const auto note = [&](std::uint32_t key, std::uint32_t bit) {
        if (!shared) {
            call(key, bit);
            return;
        }
        const auto subgroup = constant(spv::ScopeSubgroup);
        state.module.AddFunction(spv::OpStore, pendingVariable, ConstantBool(state, true));
        const auto header = state.module.AllocateId();
        const auto loop = state.module.AllocateId();
        const auto continuation = state.module.AllocateId();
        const auto merge = state.module.AllocateId();
        state.module.AddFunction(spv::OpBranch, header);
        EmitLabel(state, header);
        const auto pending = load(boolean, pendingVariable);
        state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, pending, loop, merge);
        EmitLabel(state, loop);
        const auto first = state.module.AllocateId();
        state.module.AddFunction(spv::OpGroupNonUniformBroadcastFirst, u32, first, subgroup, key);
        EmitIfCondition(state, equal(key, first), [&] {
            const auto bits = state.module.AllocateId();
            state.module.AddFunction(spv::OpGroupNonUniformBitwiseOr, u32, bits, subgroup, spv::GroupOperationReduce, bit);
            const auto elected = state.module.AllocateId();
            state.module.AddFunction(spv::OpGroupNonUniformElect, boolean, elected, subgroup);
            EmitIfCondition(state, elected, [&] { call(first, bits); });
            state.module.AddFunction(spv::OpStore, pendingVariable, ConstantBool(state, false));
        });
        state.module.AddFunction(spv::OpBranch, continuation);
        EmitLabel(state, continuation);
        state.module.AddFunction(spv::OpBranch, header);
        EmitLabel(state, merge);
    };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "record_image_store");
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, TypeVoid(state), u32, u32, u32, u32, u32, u32, u32, u32, u32, u32));
    std::array<std::uint32_t, 10> parameters{};
    for (auto& parameter : parameters) {
        parameter = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, u32, parameter);
    }
    EmitLabel(state, state.module.AllocateId());
    const auto variable = [&](std::uint32_t type) {
        const auto id = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, type), id, spv::StorageClassFunction);
        return id;
    };
    const auto counter = variable(u32);
    const auto firstTailVariable = variable(u32);
    const auto totalVariable = variable(u64);
    const auto aboveVariable = variable(u64);
    const auto paddedVariable = variable(u32);
    pendingVariable = variable(boolean);
    const auto [w0, w1, w2, w3, w4, w5, x, y, layer, lod] = parameters;
    const auto base = op(spv::OpShiftLeftLogical, u64, op(spv::OpBitwiseOr, u64, widen(w0), op(spv::OpShiftLeftLogical, u64, widen(andBits(w1, 0xffu)), wide(32u))), wide(8u));
    const auto format = field(w1, 20u, 0x1ffu);
    const auto width = add(op(spv::OpBitwiseOr, u32, field(w1, 30u, 0x3u), shl(andBits(w2, 0xfffu), constant(2u))), constant(1u));
    const auto height = add(field(w2, 14u, 0x3fffu), constant(1u));
    const auto baseLevel = field(w3, 12u, 0xfu);
    const auto lastLevel = field(w3, 16u, 0xfu);
    const auto tileMode = field(w3, 20u, 0x1fu);
    const auto type = shr(w3, constant(28u));
    const auto depth = andBits(w4, 0x1fffu);
    const auto baseArray = field(w4, 16u, 0x1fffu);
    const auto arrayPitch = andBits(w5, 0xfu);
    const auto maxMip = field(w5, 4u, 0xfu);
    const auto formatIndex = umin(format, constant(Pages::FormatTableEntries - 1u));
    const auto sizeCode = pick(u32, less(format, constant(Pages::FormatTableEntries)), andBits(shr(ConstantAt(state, formats, shr(formatIndex, constant(3u))), shl(andBits(formatIndex, 7u), constant(2u))), 0xfu), constant(0u));
    const auto linear = is(tileMode, 0u);
    const auto block256 = any(tileMode, {1u, 2u});
    const auto block4KB = any(tileMode, {5u, 6u, 21u, 22u});
    const auto block64KB = any(tileMode, {9u, 10u, 17u, 18u, 24u, 25u, 26u, 27u});
    const auto xorMode = any(tileMode, {17u, 18u, 21u, 22u, 24u, 25u, 26u, 27u});
    const auto noDepth = both(is(depth, 0u), is(baseArray, 0u));
    const auto typed = either(either(either(both(is(type, 8u), both(is(height, 1u), noDepth)), both(is(type, 9u), noDepth)), both(is(type, 11u), equal(width, height))),
        either(both(is(type, 12u), both(is(height, 1u), any(tileMode, {0u, 24u, 27u}))), is(type, 13u)));
    const auto arrayed = any(type, {11u, 12u, 13u});
    const auto xorBits = pick(u64, xorMode, op(spv::OpBitwiseAnd, u64, base, pick(u64, any(tileMode, {21u, 22u}), wide(0xfffu), wide(0xffffu))), wide(0u));
    const auto start = op(spv::OpISub, u64, base, xorBits);
    const auto blockLog2 = pick(u32, block256, constant(8u), pick(u32, block4KB, constant(12u), constant(16u)));
    const auto blockAligned = either(linear, op(spv::OpIEqual, boolean, op(spv::OpBitwiseAnd, u64, start, op(spv::OpISub, u64, op(spv::OpShiftLeftLogical, u64, wide(1u), widen(blockLog2)), wide(1u))), wide(0u)));
    const auto level = add(baseLevel, lod);
    const auto placeable = both(both(both(op(spv::OpINotEqual, boolean, sizeCode, constant(0u)), either(either(linear, block256), either(block4KB, block64KB))), both(typed, is(arrayPitch, 0u))),
        both(both(lessEqual(baseLevel, maxMip), lessEqual(baseLevel, lastLevel)), both(both(op(spv::OpINotEqual, boolean, start, wide(0u)), blockAligned), both(lessEqual(lod, constant(15u)), lessEqual(level, umin(lastLevel, maxMip))))));
    ReturnUnless(state, placeable, [&] { note(constant(BdaAbi::WrittenImageWholeKey), constant(1u)); });
    const auto lastLayer = pick(u32, both(is(type, 11u), equal(baseArray, depth)), add(depth, constant(5u)), depth);
    const auto absoluteLayer = pick(u32, arrayed, add(baseArray, layer), constant(0u));
    const auto layerInside = either(none(arrayed), both(lessEqual(layer, lastLayer), lessEqual(absoluteLayer, lastLayer)));
    const auto levelWidth = EmitGlsl<GLSLstd450UMax, IrType::U32>(state, shr(width, level), constant(1u));
    const auto levelHeight = EmitGlsl<GLSLstd450UMax, IrType::U32>(state, shr(height, level), constant(1u));
    ReturnUnless(state, both(layerInside, both(less(x, levelWidth), less(y, levelHeight))), nullptr);
    const auto elementLog2 = sub(sizeCode, constant(1u));
    const auto mipCount = add(maxMip, constant(1u));
    const auto widthLog2 = shr(add(sub(blockLog2, elementLog2), constant(1u)), constant(1u));
    const auto heightLog2 = shr(sub(blockLog2, elementLog2), constant(1u));
    const auto rowAlignment = pick(u32, linear, shr(constant(256u), elementLog2), shl(constant(1u), widthLog2));
    const auto columnAlignment = pick(u32, linear, constant(1u), shl(constant(1u), heightLog2));
    const auto small = both(is(tileMode, 24u), lessEqual(elementLog2, constant(1u)));
    const auto widthLimit = shr(shr(shl(constant(1u), widthLog2), constant(1u)), pick(u32, small, constant(1u), constant(0u)));
    const auto heightLimit = shr(shl(constant(1u), heightLog2), pick(u32, both(small, is(elementLog2, 0u)), constant(1u), constant(0u)));
    const auto maxTailLevels = pick(u32, block4KB, constant(8u), constant(12u));
    const auto tails = both(either(block4KB, block64KB), less(constant(1u), mipCount));
    const auto shiftCeil = [&](std::uint32_t value, std::uint32_t shift) { return shr(sub(add(value, shl(constant(1u), shift)), constant(1u)), shift); };
    const auto alignUp = [&](std::uint32_t value, std::uint32_t alignment) { return op(spv::OpBitwiseAnd, u32, sub(add(value, alignment), constant(1u)), Unary(state, spv::OpNot, u32, sub(alignment, constant(1u)))); };
    state.module.AddFunction(spv::OpStore, firstTailVariable, mipCount);
    EmitCountedLoop(state, counter, mipCount, [&](std::uint32_t current) {
        const auto fits = both(both(lessEqual(shiftCeil(width, current), widthLimit), lessEqual(shiftCeil(height, current), heightLimit)), lessEqual(sub(mipCount, current), maxTailLevels));
        const auto firstTail = load(u32, firstTailVariable);
        state.module.AddFunction(spv::OpStore, firstTailVariable, pick(u32, both(both(tails, fits), equal(firstTail, mipCount)), current, firstTail));
    });
    const auto firstTail = load(u32, firstTailVariable);
    state.module.AddFunction(spv::OpStore, totalVariable, wide(0u));
    state.module.AddFunction(spv::OpStore, aboveVariable, wide(0u));
    state.module.AddFunction(spv::OpStore, paddedVariable, constant(0u));
    EmitCountedLoop(state, counter, firstTail, [&](std::uint32_t current) {
        const auto paddedWidth = alignUp(shiftCeil(width, current), rowAlignment);
        const auto paddedHeight = alignUp(shiftCeil(height, current), columnAlignment);
        const auto bytes = op(spv::OpShiftLeftLogical, u64, op(spv::OpIMul, u64, widen(paddedWidth), widen(paddedHeight)), widen(elementLog2));
        state.module.AddFunction(spv::OpStore, totalVariable, op(spv::OpIAdd, u64, load(u64, totalVariable), bytes));
        state.module.AddFunction(spv::OpStore, aboveVariable, op(spv::OpIAdd, u64, load(u64, aboveVariable), pick(u64, less(level, current), bytes, wide(0u))));
        state.module.AddFunction(spv::OpStore, paddedVariable, pick(u32, equal(current, level), paddedWidth, load(u32, paddedVariable)));
    });
    const auto tailBlock = pick(u64, less(firstTail, mipCount), op(spv::OpShiftLeftLogical, u64, wide(1u), widen(blockLog2)), wide(0u));
    const auto total = op(spv::OpIAdd, u64, tailBlock, load(u64, totalVariable));
    const auto levelOffset = op(spv::OpIAdd, u64, tailBlock, load(u64, aboveVariable));
    const auto padded = load(u32, paddedVariable);
    const auto layerStart = op(spv::OpIAdd, u64, start, op(spv::OpIMul, u64, widen(absoluteLayer), total));
    const auto linearAddress = op(spv::OpIAdd, u64, op(spv::OpIAdd, u64, layerStart, levelOffset), op(spv::OpShiftLeftLogical, u64, op(spv::OpIAdd, u64, op(spv::OpIMul, u64, widen(y), widen(padded)), widen(x)), widen(elementLog2)));
    const auto inTail = lessEqual(firstTail, level);
    const auto blockIndex = add(op(spv::OpIMul, u32, shr(y, heightLog2), shr(padded, widthLog2)), shr(x, widthLog2));
    const auto blockStart = op(spv::OpIAdd, u64, layerStart, pick(u64, inTail, wide(0u), op(spv::OpIAdd, u64, levelOffset, op(spv::OpShiftLeftLogical, u64, widen(blockIndex), widen(blockLog2)))));
    const auto tailIndex = umin(sub(level, umin(firstTail, level)), constant(11u));
    const auto location = ConstantAt(state, locations, add(op(spv::OpIMul, u32, umin(elementLog2, constant(4u)), constant(12u)), tailIndex));
    const auto swizzleX = pick(u32, inTail, add(x, shr(location, constant(16u))), x);
    const auto swizzleY = pick(u32, inTail, add(y, andBits(location, 0xffffu)), y);
    std::array<std::uint32_t, 2> groups{};
    for (std::uint32_t group = 0; group < 2u; ++group) {
        auto inputs = constant(Pages::PageBitInputs[group][4]);
        for (std::uint32_t element = 4u; element-- > 0u;) inputs = pick(u32, is(elementLog2, element), constant(Pages::PageBitInputs[group][element]), inputs);
        groups[group] = inputs;
    }
    const auto inputs = pick(u32, any(tileMode, {24u, 27u}), groups[1], groups[0]);
    auto bits = constant(0u);
    for (std::uint32_t bit = 0; bit < 4u; ++bit) {
        const auto input = field(inputs, bit * 5u, 0x1fu);
        const auto coordinate = pick(u32, op(spv::OpINotEqual, boolean, andBits(input, 0x10u), constant(0u)), swizzleY, swizzleX);
        bits = op(spv::OpBitwiseOr, u32, bits, shl(andBits(shr(coordinate, andBits(input, 0xfu)), 1u), constant(bit)));
    }
    const auto pageBits = andBits(op(spv::OpBitwiseXor, u32, bits, Unary(state, spv::OpUConvert, u32, op(spv::OpShiftRightLogical, u64, xorBits, wide(12u)))), 0xfu);
    const auto tiledAddress = op(spv::OpIAdd, u64, blockStart, pick(u64, block64KB, op(spv::OpShiftLeftLogical, u64, widen(pageBits), wide(12u)), wide(0u)));
    const auto address = pick(u64, linear, linearAddress, tiledAddress);
    const auto key = op(spv::OpBitwiseOr, u32, add(Unary(state, spv::OpUConvert, u32, op(spv::OpShiftRightLogical, u64, address, wide(BdaAbi::WrittenChunkShift))), constant(1u)), constant(BdaAbi::WrittenImageKeyBit));
    const auto page = andBits(Unary(state, spv::OpUConvert, u32, op(spv::OpShiftRightLogical, u64, address, wide(BdaAbi::WrittenPageShift))), BdaAbi::WrittenChunkPages - 1u);
    note(key, shl(constant(1u), page));
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.imageStoreRecordFunction = function;
}

}
