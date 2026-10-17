#include "SpirvBackend/SpirvImageEmitter.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvMemory/SpirvSubgroup.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "SpirvBackend/SpirvMemory/SpirvFormatConvert.hpp"
#include "BdaAbi.hpp"
#include "StorageImageMemory.hpp"
#include "StorageImagePages.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <span>
#include <spirv/unified1/GLSL.std.450.h>
#include <stdexcept>
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

namespace Memory = StorageImageMemory;

std::vector<std::uint32_t> FunctionParameters(SpirvEmitterState& state, std::uint32_t count) {
    std::vector<std::uint32_t> parameters(count);
    for (auto& parameter : parameters) {
        parameter = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, TypeU32(state), parameter);
    }
    return parameters;
}

std::uint32_t BeginFunction(SpirvEmitterState& state, const char* name, std::uint32_t result, std::uint32_t parameters) {
    const std::vector<std::uint32_t> types(parameters, TypeU32(state));
    std::vector<std::uint32_t> signature{result};
    signature.insert(signature.end(), types.begin(), types.end());
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    state.module.AddFunction(spv::OpFunction, result, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, std::span<const std::uint32_t>(signature)));
    return function;
}

std::uint32_t Phi(SpirvEmitterState& state, std::uint32_t type, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& incoming) {
    std::vector<std::uint32_t> words{spv::OpPhi, type, state.module.AllocateId()};
    for (const auto& [value, parent] : incoming) words.insert(words.end(), {value, parent});
    state.module.AddFunction(std::span<const std::uint32_t>(words));
    return words[2];
}

std::uint32_t DefineTexelFunction(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto result = TypeU32Vector(state, 4u);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto wide = [&](std::uint64_t value) { return BdaConstant(state, value); };
    const auto op = [&](spv::Op code, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, code, type, left, right); };
    const auto add = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpIAdd, u32, left, right); };
    const auto sub = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpISub, u32, left, right); };
    const auto mul = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpIMul, u32, left, right); };
    const auto andBits = [&](std::uint32_t value, std::uint32_t mask) { return op(spv::OpBitwiseAnd, u32, value, constant(mask)); };
    const auto orBits = [&](std::uint32_t left, std::uint32_t right) { return op(spv::OpBitwiseOr, u32, left, right); };
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
    const auto pick32 = [&](std::uint32_t condition, std::uint32_t whenTrue, std::uint32_t whenFalse) { return pick(u32, condition, whenTrue, whenFalse); };
    const auto any = [&](std::uint32_t value, std::initializer_list<std::uint32_t> candidates) {
        std::uint32_t found = 0;
        for (const auto candidate : candidates) found = found == 0u ? is(value, candidate) : either(found, is(value, candidate));
        return found;
    };
    const auto widen = [&](std::uint32_t value) { return Unary(state, spv::OpUConvert, u64, value); };
    const auto umin = [&](std::uint32_t left, std::uint32_t right) { return EmitGlsl<GLSLstd450UMin, IrType::U32>(state, left, right); };
    const auto umax = [&](std::uint32_t left, std::uint32_t right) { return EmitGlsl<GLSLstd450UMax, IrType::U32>(state, left, right); };
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto store = [&](std::uint32_t pointer, std::uint32_t value) { state.module.AddFunction(spv::OpStore, pointer, value); };
    const auto shiftCeil = [&](std::uint32_t value, std::uint32_t shift) { return shr(sub(add(value, shl(constant(1u), shift)), constant(1u)), shift); };
    const auto alignUp = [&](std::uint32_t value, std::uint32_t alignment) { return op(spv::OpBitwiseAnd, u32, sub(add(value, alignment), constant(1u)), Unary(state, spv::OpNot, u32, sub(alignment, constant(1u)))); };
    const auto function = BeginFunction(state, "storage_image_texel", result, 10u);
    const auto parameters = FunctionParameters(state, 10u);
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
    const auto w0 = parameters[0], w1 = parameters[1], w2 = parameters[2], w3 = parameters[3], w4 = parameters[4], w5 = parameters[5];
    const auto x = parameters[6], z = parameters[8], lod = parameters[9];
    const auto base = op(spv::OpShiftLeftLogical, u64, op(spv::OpBitwiseOr, u64, widen(w0), op(spv::OpShiftLeftLogical, u64, widen(andBits(w1, 0xffu)), wide(32u))), wide(8u));
    const auto format = field(w1, 20u, 0x1ffu);
    const auto width = add(orBits(field(w1, 30u, 0x3u), shl(andBits(w2, 0xfffu), constant(2u))), constant(1u));
    const auto height = add(field(w2, 14u, 0x3fffu), constant(1u));
    const auto baseLevel = field(w3, 12u, 0xfu);
    const auto lastLevel = field(w3, 16u, 0xfu);
    const auto tileMode = field(w3, 20u, 0x1fu);
    const auto type = shr(w3, constant(28u));
    const auto depth = andBits(w4, 0x1fffu);
    const auto baseArray = field(w4, 16u, 0x1fffu);
    const auto maxMip = field(w5, 4u, 0xfu);
    const auto y = pick32(is(type, 8u), constant(0u), parameters[7]);
    const auto formats = StorageImagePages::FormatTable();
    const auto formatIndex = umin(format, constant(Memory::LastMemoryFormat));
    const auto sizeCode = pick32(lessEqual(format, constant(Memory::LastMemoryFormat)), andBits(shr(ConstantAt(state, formats, shr(formatIndex, constant(3u))), shl(andBits(formatIndex, 7u), constant(2u))), 0xfu), constant(0u));
    const auto element = umin(sub(umax(sizeCode, constant(1u)), constant(1u)), constant(4u));
    const auto linear = is(tileMode, 0u);
    const auto volume = is(type, 10u);
    const auto thick = both(volume, any(tileMode, {5u, 9u, 25u, 26u}));
    const auto xorMode = any(tileMode, {17u, 18u, 21u, 22u, 24u, 25u, 26u, 27u});
    const auto xorBits = pick(u64, xorMode, op(spv::OpBitwiseAnd, u64, base, pick(u64, any(tileMode, {21u, 22u}), wide(0xfffu), wide(0xffffu))), wide(0u));
    const auto start = op(spv::OpISub, u64, base, xorBits);
    const auto thinBlockLog2 = pick32(any(tileMode, {1u, 2u}), constant(8u), pick32(any(tileMode, {5u, 6u, 21u, 22u}), constant(12u), constant(16u)));
    const auto thickBlockLog2 = pick32(is(tileMode, 5u), constant(12u), constant(16u));
    const auto blockLog2 = pick32(linear, constant(8u), pick32(thick, thickBlockLog2, thinBlockLog2));
    std::array<std::uint32_t, 10> thickExtents{};
    for (std::uint32_t entry = 0; entry < 5u; ++entry) {
        const auto& small = Memory::ThickBlockLog2For4KB[entry];
        const auto& large = Memory::ThickBlockLog2For64KB[entry];
        thickExtents[entry] = small[0] | small[1] << 4u | small[2] << 8u;
        thickExtents[entry + 5u] = large[0] | large[1] << 4u | large[2] << 8u;
    }
    const auto extents = ConstantAt(state, thickExtents, add(pick32(is(blockLog2, 12u), constant(0u), constant(5u)), element));
    const auto thickWidthLog2 = andBits(extents, 0xfu);
    const auto thickHeightLog2 = field(extents, 4u, 0xfu);
    const auto thickDepthLog2 = field(extents, 8u, 0xfu);
    const auto thinWidthLog2 = shr(add(sub(blockLog2, element), constant(1u)), constant(1u));
    const auto thinHeightLog2 = shr(sub(blockLog2, element), constant(1u));
    const auto columnLog2 = pick32(linear, sub(constant(8u), element), pick32(thick, thickWidthLog2, thinWidthLog2));
    const auto rowLog2 = pick32(linear, constant(0u), pick32(thick, thickHeightLog2, thinHeightLog2));
    const auto depthLog2 = pick32(thick, thickDepthLog2, constant(0u));
    const auto mipCount = add(maxMip, constant(1u));
    const auto level = add(baseLevel, lod);
    const auto shiftLevel = umin(level, constant(15u));
    const auto small = both(is(tileMode, 24u), lessEqual(element, constant(1u)));
    const auto thinWidthLimit = shr(shr(shl(constant(1u), thinWidthLog2), constant(1u)), pick32(small, constant(1u), constant(0u)));
    const auto thinHeightLimit = shr(shl(constant(1u), thinHeightLog2), pick32(both(small, is(element, 0u)), constant(1u), constant(0u)));
    const auto thickWidthLimit = shr(shl(constant(1u), thickWidthLog2), pick32(is(blockLog2, 16u), constant(1u), constant(0u)));
    const auto thickHeightLimit = shr(shl(constant(1u), thickHeightLog2), pick32(is(blockLog2, 12u), constant(1u), constant(0u)));
    const auto widthLimit = pick32(thick, thickWidthLimit, thinWidthLimit);
    const auto heightLimit = pick32(thick, thickHeightLimit, thinHeightLimit);
    const auto maxTailLevels = pick32(thick, pick32(is(blockLog2, 12u), constant(5u), constant(10u)), pick32(is(blockLog2, 12u), constant(8u), constant(12u)));
    const auto tails = both(both(none(linear), none(is(blockLog2, 8u))), less(constant(1u), mipCount));
    store(firstTailVariable, mipCount);
    EmitCountedLoop(state, counter, mipCount, [&](std::uint32_t current) {
        const auto fits = both(both(lessEqual(shiftCeil(width, current), widthLimit), lessEqual(shiftCeil(height, current), heightLimit)), lessEqual(sub(mipCount, current), maxTailLevels));
        const auto firstTail = load(u32, firstTailVariable);
        store(firstTailVariable, pick32(both(both(tails, fits), equal(firstTail, mipCount)), current, firstTail));
    });
    const auto firstTail = load(u32, firstTailVariable);
    store(totalVariable, wide(0u));
    store(aboveVariable, wide(0u));
    store(paddedVariable, constant(0u));
    EmitCountedLoop(state, counter, firstTail, [&](std::uint32_t current) {
        const auto paddedWidth = alignUp(shiftCeil(width, current), shl(constant(1u), columnLog2));
        const auto paddedHeight = alignUp(shiftCeil(height, current), shl(constant(1u), rowLog2));
        const auto bytes = op(spv::OpShiftLeftLogical, u64, op(spv::OpIMul, u64, widen(paddedWidth), widen(paddedHeight)), widen(add(element, depthLog2)));
        store(totalVariable, op(spv::OpIAdd, u64, load(u64, totalVariable), bytes));
        store(aboveVariable, op(spv::OpIAdd, u64, load(u64, aboveVariable), pick(u64, less(level, current), bytes, wide(0u))));
        store(paddedVariable, pick32(equal(current, level), paddedWidth, load(u32, paddedVariable)));
    });
    const auto tailBlock = pick(u64, less(firstTail, mipCount), op(spv::OpShiftLeftLogical, u64, wide(1u), widen(blockLog2)), wide(0u));
    const auto total = op(spv::OpIAdd, u64, tailBlock, load(u64, totalVariable));
    const auto levelOffset = op(spv::OpIAdd, u64, tailBlock, load(u64, aboveVariable));
    const auto padded = load(u32, paddedVariable);
    const auto inTail = both(none(linear), lessEqual(firstTail, level));
    const auto arrayed = any(type, {11u, 12u, 13u});
    const auto lastLayer = pick32(both(is(type, 11u), equal(baseArray, depth)), add(depth, constant(5u)), depth);
    const auto slice = pick32(arrayed, add(baseArray, z), pick32(volume, z, constant(0u)));
    const auto levelWidth = umax(shr(width, shiftLevel), constant(1u));
    const auto levelHeight = umax(shr(height, shiftLevel), constant(1u));
    const auto levelDepth = umax(shr(add(depth, constant(1u)), shiftLevel), constant(1u));
    auto inside = both(both(lessEqual(lod, constant(15u)), lessEqual(level, umin(lastLevel, maxMip))), both(less(x, levelWidth), less(y, levelHeight)));
    inside = both(inside, either(none(arrayed), both(lessEqual(z, lastLayer), lessEqual(slice, lastLayer))));
    inside = both(inside, either(none(volume), less(z, levelDepth)));
    inside = both(inside, none(is(sizeCode, 0u)));
    const auto sliceIndex = pick32(thick, shr(z, depthLog2), slice);
    const auto sliceStart = op(spv::OpIAdd, u64, start, op(spv::OpIMul, u64, widen(sliceIndex), total));
    const auto linearAddress = op(spv::OpIAdd, u64, op(spv::OpIAdd, u64, sliceStart, levelOffset), op(spv::OpShiftLeftLogical, u64, op(spv::OpIAdd, u64, op(spv::OpIMul, u64, widen(y), widen(padded)), widen(x)), widen(element)));
    const auto blockIndex = add(mul(shr(y, rowLog2), shr(padded, columnLog2)), shr(x, columnLog2));
    const auto blockStart = op(spv::OpIAdd, u64, sliceStart, pick(u64, inTail, wide(0u), op(spv::OpIAdd, u64, levelOffset, op(spv::OpShiftLeftLogical, u64, widen(blockIndex), widen(blockLog2)))));
    std::array<std::uint32_t, 40> tails4KB{};
    for (std::uint32_t entry = 0; entry < 5u; ++entry) std::copy(Memory::TailLocations4KB[entry].begin(), Memory::TailLocations4KB[entry].end(), tails4KB.begin() + entry * 8u);
    std::array<std::uint32_t, 60> tails64KB{};
    for (std::uint32_t entry = 0; entry < 5u; ++entry) std::copy(StorageImagePages::TailLocations64KB[entry].begin(), StorageImagePages::TailLocations64KB[entry].end(), tails64KB.begin() + entry * 12u);
    std::array<std::uint32_t, 5> microBlocks{};
    for (std::uint32_t entry = 0; entry < 5u; ++entry) microBlocks[entry] = Memory::ThickTailMicroBlock[entry][0] << 16u | Memory::ThickTailMicroBlock[entry][1];
    const auto location = EmitValueOrZeroIfCondition(state, inTail, [&] {
        const auto index = sub(level, firstTail);
        const auto thin4KB = ConstantAt(state, tails4KB, add(mul(element, constant(8u)), umin(index, constant(7u))));
        const auto thin64KB = ConstantAt(state, tails64KB, add(mul(element, constant(12u)), umin(index, constant(11u))));
        const auto slot = umin(sub(sub(maxTailLevels, constant(1u)), umin(index, sub(maxTailLevels, constant(1u)))), constant(15u));
        const auto slotOffset = pick32(less(constant(6u), slot), shl(constant(16u), slot), shl(slot, constant(8u)));
        auto tailX = constant(0u);
        auto tailY = constant(0u);
        for (std::uint32_t bit = 0; bit < 6u; ++bit) {
            tailX = orBits(tailX, shl(andBits(shr(slotOffset, constant(9u + 2u * bit)), 1u), constant(bit)));
            tailY = orBits(tailY, shl(andBits(shr(slotOffset, constant(8u + 2u * bit)), 1u), constant(bit)));
        }
        const auto micro = ConstantAt(state, microBlocks, element);
        const auto thickLocation = orBits(shl(mul(tailX, shr(micro, constant(16u))), constant(16u)), mul(tailY, andBits(micro, 0xffffu)));
        return pick32(thick, thickLocation, pick32(is(blockLog2, 12u), thin4KB, thin64KB));
    });
    const auto swizzleX = add(x, shr(location, constant(16u)));
    const auto swizzleY = add(y, andBits(location, 0xffffu));
    const auto equationSlice = pick32(thick, z, slice);
    const auto key = orBits(shl(orBits(tileMode, pick32(thick, constant(0x100u), constant(0u))), constant(3u)), element);
    const auto merge = state.module.AllocateId();
    const auto fallback = state.module.AllocateId();
    std::vector<std::uint32_t> labels(Memory::Equations.size());
    std::vector<std::uint32_t> branch{spv::OpSwitch, key, fallback};
    for (std::size_t entry = 0; entry < Memory::Equations.size(); ++entry) {
        labels[entry] = state.module.AllocateId();
        const auto& equation = Memory::Equations[entry];
        branch.insert(branch.end(), {equation.swizzleMode << 3u | static_cast<std::uint32_t>(std::countr_zero(equation.elementBytes)), labels[entry]});
    }
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(std::span<const std::uint32_t>(branch));
    std::array<std::vector<std::pair<std::uint32_t, std::uint32_t>>, 17> incoming{};
    for (std::size_t entry = 0; entry < Memory::Equations.size(); ++entry) {
        EmitLabel(state, labels[entry]);
        for (std::uint32_t bit = 0; bit < 16u; ++bit) incoming[bit].emplace_back(constant(Memory::Equations[entry].bits[bit]), labels[entry]);
        incoming[16].emplace_back(ConstantBool(state, true), labels[entry]);
        state.module.AddFunction(spv::OpBranch, merge);
    }
    EmitLabel(state, fallback);
    for (std::uint32_t bit = 0; bit < 16u; ++bit) incoming[bit].emplace_back(constant(0u), fallback);
    incoming[16].emplace_back(ConstantBool(state, false), fallback);
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    std::array<std::uint32_t, 16> masks{};
    for (std::uint32_t bit = 0; bit < 16u; ++bit) masks[bit] = Phi(state, u32, incoming[bit]);
    const auto found = Phi(state, boolean, incoming[16]);
    const auto packed = orBits(orBits(andBits(swizzleX, 0xfffu), shl(andBits(swizzleY, 0xfffu), constant(12u))), shl(andBits(equationSlice, 0xffu), constant(24u)));
    auto offset = constant(0u);
    for (std::uint32_t bit = 0; bit < 16u; ++bit) {
        const auto count = Unary(state, spv::OpBitCount, u32, op(spv::OpBitwiseAnd, u32, packed, masks[bit]));
        offset = orBits(offset, shl(andBits(count, 1u), constant(bit)));
    }
    const auto tiledAddress = op(spv::OpIAdd, u64, blockStart, widen(op(spv::OpBitwiseXor, u32, offset, Unary(state, spv::OpUConvert, u32, xorBits))));
    const auto address = pick(u64, linear, linearAddress, tiledAddress);
    inside = both(inside, either(linear, found));
    const auto low = Unary(state, spv::OpUConvert, u32, address);
    const auto high = Unary(state, spv::OpUConvert, u32, op(spv::OpShiftRightLogical, u64, address, wide(32u)));
    const auto texel = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, result, texel, low, high, element, pick32(inside, constant(1u), constant(0u)));
    state.module.AddFunction(spv::OpReturnValue, texel);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

struct ComponentPlace {
    std::uint32_t word;
    std::uint32_t offset;
};

ComponentPlace PlaceOf(const SpirvBufferFormatInfo& info, std::uint32_t component) {
    if (info.packedBitfield) return {info.componentBitOffset[component] / 32u, info.componentBitOffset[component] % 32u};
    const auto byte = GetFormatComponentByteOffset(info, component);
    return {byte / 4u, byte % 4u * 8u};
}

std::vector<std::uint32_t> MemoryFormats() {
    std::vector<std::uint32_t> formats;
    for (std::uint32_t format = 1; format <= Memory::LastMemoryFormat; ++format) {
        if (!Memory::MemoryFormat(format)) continue;
        const auto info = GetFormatInfo(static_cast<IrBufferFormat>(format));
        if (info.type == SpirvFormatComponentType::Unknown || info.componentCount == 0u || info.byteSize != 1u << Memory::ElementLog2(format)) throw std::runtime_error("storage image memory format " + std::to_string(format) + " has no codec layout of its element size");
        formats.push_back(format);
    }
    return formats;
}

template<typename TBody>
std::vector<std::uint32_t> FormatSwitch(SpirvEmitterState& state, std::uint32_t format, std::uint32_t results, TBody&& body) {
    const auto u32 = TypeU32(state);
    const auto formats = MemoryFormats();
    const auto merge = state.module.AllocateId();
    const auto fallback = state.module.AllocateId();
    std::vector<std::uint32_t> labels(formats.size());
    std::vector<std::uint32_t> branch{spv::OpSwitch, format, fallback};
    for (std::size_t index = 0; index < formats.size(); ++index) {
        labels[index] = state.module.AllocateId();
        branch.insert(branch.end(), {formats[index], labels[index]});
    }
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(std::span<const std::uint32_t>(branch));
    std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> incoming(results);
    for (std::size_t index = 0; index < formats.size(); ++index) {
        EmitLabel(state, labels[index]);
        const auto values = body(GetFormatInfo(static_cast<IrBufferFormat>(formats[index])));
        const auto exit = state.module.AllocateId();
        state.module.AddFunction(spv::OpBranch, exit);
        EmitLabel(state, exit);
        for (std::uint32_t value = 0; value < results; ++value) incoming[value].emplace_back(values.at(value), exit);
        state.module.AddFunction(spv::OpBranch, merge);
    }
    EmitLabel(state, fallback);
    for (std::uint32_t value = 0; value < results; ++value) incoming[value].emplace_back(ConstantU32(state, 0u), fallback);
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    std::vector<std::uint32_t> merged(results);
    for (std::uint32_t value = 0; value < results; ++value) merged[value] = Phi(state, u32, incoming[value]);
    return merged;
}

std::uint32_t DefineDecodeFunction(SpirvEmitterState& state, std::uint32_t resultType) {
    const auto u32 = TypeU32(state);
    const auto function = BeginFunction(state, "storage_image_decode", resultType, 5u);
    const auto parameters = FunctionParameters(state, 5u);
    EmitLabel(state, state.module.AllocateId());
    const std::array<std::uint32_t, 4> words{parameters[1], parameters[2], parameters[3], parameters[4]};
    const auto values = FormatSwitch(state, parameters[0], 5u, [&](const SpirvBufferFormatInfo& info) {
        const bool integer = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint;
        const auto one = ConstantU32(state, integer ? 1u : 0x3f800000u);
        std::vector<std::uint32_t> components{ConstantU32(state, 0u), ConstantU32(state, 0u), ConstantU32(state, 0u), one, one};
        for (std::uint32_t component = 0; component < info.componentCount; ++component) {
            const auto place = PlaceOf(info, component);
            const auto bits = info.componentBits[component];
            auto raw = words[place.word];
            if (bits != 32u) {
                const bool signedComponent = IsSignedFormatComponent(info.type);
                const auto valueType = signedComponent ? TypeI32(state) : u32;
                const auto extracted = state.module.AllocateId();
                state.module.AddFunction(signedComponent ? spv::OpBitFieldSExtract : spv::OpBitFieldUExtract, valueType, extracted, signedComponent ? Unary(state, spv::OpBitcast, valueType, raw) : raw, ConstantU32(state, place.offset), ConstantU32(state, bits));
                raw = signedComponent ? Unary(state, spv::OpBitcast, u32, extracted) : extracted;
            }
            components[component] = NormalizeFormatComponent(state, info, component, raw);
        }
        return components;
    });
    const auto decoded = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, resultType, decoded, values[0], values[1], values[2], values[3], values[4]);
    state.module.AddFunction(spv::OpReturnValue, decoded);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

std::uint32_t DefineEncodeFunction(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto result = TypeU32Vector(state, 4u);
    const auto function = BeginFunction(state, "storage_image_encode", result, 6u);
    const auto parameters = FunctionParameters(state, 6u);
    EmitLabel(state, state.module.AllocateId());
    const auto d16 = Binary(state, spv::OpINotEqual, TypeBool(state), parameters[1], ConstantU32(state, 0u));
    const std::array<std::uint32_t, 4> values{parameters[2], parameters[3], parameters[4], parameters[5]};
    const auto words = FormatSwitch(state, parameters[0], 4u, [&](const SpirvBufferFormatInfo& info) {
        std::vector<std::uint32_t> packed{ConstantU32(state, 0u), ConstantU32(state, 0u), ConstantU32(state, 0u), ConstantU32(state, 0u)};
        for (std::uint32_t component = 0; component < info.componentCount; ++component) {
            const auto place = PlaceOf(info, component);
            const auto bits = info.componentBits[component];
            const auto encoded = Select(state, u32, d16, EmitD16StoreComponent(state, info, component, values[component]), EmitFormatStoreComponent(state, info, component, values[component]));
            const auto masked = bits == 32u ? encoded : Binary(state, spv::OpBitwiseAnd, u32, encoded, ConstantU32(state, (1u << bits) - 1u));
            const auto shifted = place.offset == 0u ? masked : Binary(state, spv::OpShiftLeftLogical, u32, masked, ConstantU32(state, place.offset));
            packed[place.word] = Binary(state, spv::OpBitwiseOr, u32, packed[place.word], shifted);
        }
        return packed;
    });
    const auto encoded = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, result, encoded, words[0], words[1], words[2], words[3]);
    state.module.AddFunction(spv::OpReturnValue, encoded);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void DefineStorageImageMemory(SpirvEmitterState& state) {
    auto& functions = state.storageImageMemory;
    functions = {};
    const auto& info = state.program.Info();
    bool loads = false;
    bool writes = false;
    for (const auto& image : info.images) {
        if (!Memory::ResourceThroughMemory(info, image) && !Memory::RuntimeResourceThroughMemory(info, image)) continue;
        loads = loads || image.resourceClass == ImageResourceClass::Sampled;
        writes = writes || image.resourceClass == ImageResourceClass::Storage;
    }
    if (!loads && !writes) return;
    if (std::ranges::find(state.supportedCapabilities, static_cast<std::uint32_t>(spv::CapabilityInt64)) == state.supportedCapabilities.end()) throw std::runtime_error("storage images through memory need 64-bit integers");
    state.module.EmitCapability(spv::CapabilityInt64);
    functions.texel = DefineTexelFunction(state);
    if (loads) {
        const std::array<std::uint32_t, 5> members{TypeU32(state), TypeU32(state), TypeU32(state), TypeU32(state), TypeU32(state)};
        functions.decodeType = state.module.Type(spv::OpTypeStruct, std::span<const std::uint32_t>(members));
        functions.decode = DefineDecodeFunction(state, functions.decodeType);
    }
    if (writes) functions.encode = DefineEncodeFunction(state);
}

}

const StorageImageMemoryFunctions* StorageImageMemoryFunctionsOf(const SpirvEmitterState& state) {
    return &state.storageImageMemory;
}

bool ImageStoresRecorded(const IrProgram& program, std::span<const std::uint32_t> capabilities) {
    const auto& info = program.Info();
    const bool writes = std::ranges::any_of(info.images, [](const ImageResource& image) { return image.written || image.atomic; });
    return writes && (info.usesFaultBuffer || info.usesDma) && std::ranges::find(capabilities, static_cast<std::uint32_t>(spv::CapabilityInt64)) != capabilities.end();
}

void DefineImageStoreRecord(SpirvEmitterState& state) {
    DefineStorageImageMemory(state);
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
