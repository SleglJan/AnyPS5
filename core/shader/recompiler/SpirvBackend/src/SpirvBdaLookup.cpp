#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "BdaTouchAbi.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>

namespace ShaderRecompiler {

namespace {

std::uint32_t TranslationType(SpirvEmitterState& state) {
    return state.module.Type(spv::OpTypeStruct, TypeScalarU64(state), TypeU32(state));
}

void ReturnIf(SpirvEmitterState& state, std::uint32_t condition) {
    const auto taken = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, taken, next);
    EmitLabel(state, taken);
    state.module.AddFunction(spv::OpReturn);
    EmitLabel(state, next);
}

void ReturnValueIf(SpirvEmitterState& state, std::uint32_t condition, std::uint32_t value) {
    const auto taken = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, taken, next);
    EmitLabel(state, taken);
    state.module.AddFunction(spv::OpReturnValue, value);
    EmitLabel(state, next);
}

std::uint32_t DefineBdaIndexVariable(SpirvEmitterState& state) {
    const auto id = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, TypeU32(state)), id, spv::StorageClassFunction);
    return id;
}

std::uint32_t EmitBdaBisection(SpirvEmitterState& state, std::uint32_t low, std::uint32_t high, std::uint32_t address, std::uint32_t first, std::uint32_t count, std::uint32_t stride) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto load = [&](std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, pointer);
        return value;
    };
    state.module.AddFunction(spv::OpStore, low, ConstantU32(state, 0));
    state.module.AddFunction(spv::OpStore, high, count);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto lower = load(low);
    const auto upper = load(high);
    const auto search = binary(spv::OpULessThan, boolean, lower, upper);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, search, body, merge);
    EmitLabel(state, body);
    const auto midpoint = binary(spv::OpIAdd, u32, lower, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, upper, lower), ConstantU32(state, 1)));
    const auto probe = binary(spv::OpIAdd, u32, first, binary(spv::OpIMul, u32, midpoint, ConstantU32(state, stride)));
    const auto before = binary(spv::OpULessThan, boolean, address, BdaLoadAddress(state, probe));
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, ConstantU32(state, 1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpStore, low, newLow);
    state.module.AddFunction(spv::OpStore, high, newHigh);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    return load(low);
}

std::uint32_t DefineBdaWriteGranted(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "bda_write_granted");
    state.module.AddFunction(spv::OpFunction, boolean, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, boolean, u64, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto end = state.module.AllocateId();
    const auto first = state.module.AllocateId();
    const auto count = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u64, end);
    state.module.AddFunction(spv::OpFunctionParameter, u32, first);
    state.module.AddFunction(spv::OpFunctionParameter, u32, count);
    EmitLabel(state, state.module.AllocateId());
    const auto low = DefineBdaIndexVariable(state);
    const auto high = DefineBdaIndexVariable(state);
    const auto index = EmitBdaBisection(state, low, high, address, first, count, BdaAbi::GrantWords);
    ReturnValueIf(state, Binary(state, spv::OpIEqual, boolean, index, ConstantU32(state, 0)), ConstantBool(state, false));
    const auto grant = Binary(state, spv::OpIAdd, u32, first, Binary(state, spv::OpIMul, u32, Binary(state, spv::OpISub, u32, index, ConstantU32(state, 1)), ConstantU32(state, BdaAbi::GrantWords)));
    state.module.AddFunction(spv::OpReturnValue, Binary(state, spv::OpULessThanEqual, boolean, end, BdaLoadAddress(state, Binary(state, spv::OpIAdd, u32, grant, ConstantU32(state, 2)))));
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

std::uint32_t DefineBdaTouch(SpirvEmitterState& state) {
    namespace Touch = BdaTouchAbi;
    static_assert(Touch::HeaderWord == BdaAbi::TouchTableWord);
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_touch");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, voidType, u64));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    EmitLabel(state, state.module.AllocateId());
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), done, spv::StorageClassFunction);
    const auto counter = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), counter, spv::StorageClassFunction);
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    ReturnIf(state, binary(spv::OpULessThan, boolean, length, constant(BdaAbi::HeaderWords)));
    const auto table = binary(spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, BdaLoadWord(state, constant(BdaAbi::TouchTableWord))), binary(spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, BdaLoadWord(state, constant(BdaAbi::TouchTableWord + 1u))), BdaConstant(state, 32u)));
    ReturnIf(state, binary(spv::OpIEqual, boolean, table, BdaConstant(state, 0u)));
    const auto word = [&](std::uint32_t index) {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, binary(spv::OpIAdd, u64, table, binary(spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, index), BdaConstant(state, 2u))));
        return pointer;
    };
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    const auto key = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, Touch::ChunkShift))), constant(1u));
    const auto bit = binary(spv::OpShiftLeftLogical, u32, constant(1u), binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, Touch::PageShift))), constant(Touch::ChunkPages - 1u)));
    const auto home = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, key, constant(Touch::HashMultiplier)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(Touch::Slots))));
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, counter, constant(0u));
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto probe = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, probe, counter);
    const auto found = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, found, done);
    const auto searching = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, found), binary(spv::OpULessThan, boolean, probe, constant(Touch::Probes)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto slot = binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, home, probe), constant(Touch::Slots - 1u));
    const auto keyWord = binary(spv::OpIAdd, u32, constant(Touch::SlotsWord), binary(spv::OpShiftLeftLogical, u32, slot, constant(1u)));
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, u32, current, word(keyWord), scope, relaxed);
    const auto same = binary(spv::OpIEqual, boolean, current, key);
    state.module.AddFunction(spv::OpStore, done, same);
    EmitIfCondition(state, binary(spv::OpIEqual, boolean, current, constant(0u)), [&] {
        const auto previous = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, word(keyWord), scope, relaxed, relaxed, key, constant(0u));
        state.module.AddFunction(spv::OpStore, done, binary(spv::OpLogicalOr, boolean, binary(spv::OpIEqual, boolean, previous, constant(0u)), binary(spv::OpIEqual, boolean, previous, key)));
    });
    const auto holds = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, holds, done);
    EmitIfCondition(state, holds, [&] {
        const auto maskWord = binary(spv::OpIAdd, u32, keyWord, constant(1u));
        const auto mask = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicLoad, u32, mask, word(maskWord), scope, relaxed);
        EmitIfCondition(state, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, mask, bit), constant(0u)), [&] {
            state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), word(maskWord), scope, relaxed, bit);
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, counter, binary(spv::OpIAdd, u32, probe, constant(1u)));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, word(constant(Touch::OverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void NoteBdaTouch(SpirvEmitterState& state, std::uint32_t touch, std::uint32_t address) {
    state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), touch, address);
}

std::uint32_t DefineBdaTranslate(SpirvEmitterState& state, bool writes, std::uint32_t touch) {
    const auto grantedFunction = writes ? DefineBdaWriteGranted(state) : 0u;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto entryType = TypeU32Vector(state, 4);
    const auto result = TranslationType(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto widen = [&](std::uint32_t value) { return Unary(state, spv::OpUConvert, u64, value); };
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto loadDevice = [&](std::uint32_t type, std::uint32_t address, std::uint32_t alignment) {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, type), pointer, address);
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer, spv::MemoryAccessAlignedMask, alignment);
        return value;
    };
    const auto component = [&](std::uint32_t vector, std::uint32_t index) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, value, vector, index);
        return value;
    };
    const auto pair = [&](std::uint32_t low, std::uint32_t high) { return binary(spv::OpBitwiseOr, u64, widen(low), binary(spv::OpShiftLeftLogical, u64, widen(high), BdaConstant(state, 32u))); };
    const auto translation = [&](std::uint32_t device, std::uint32_t reason) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, result, value, device, reason);
        return value;
    };
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) {
        ReturnValueIf(state, condition, state.module.Constant(spv::OpConstantComposite, result, BdaConstant(state, 0u), constant(static_cast<std::uint32_t>(reason))));
    };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "bda_translate");
    state.module.AddFunction(spv::OpFunction, result, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, result, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto permission = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, permission);
    EmitLabel(state, state.module.AllocateId());
    const auto variable = [&](std::uint32_t type) {
        const auto id = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, type), id, spv::StorageClassFunction);
        return id;
    };
    const auto low = variable(u32);
    const auto high = variable(u32);
    const auto entry = variable(entryType);
    const auto shift = variable(u32);
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    fail(binary(spv::OpULessThan, boolean, length, constant(BdaAbi::HeaderWords)), BdaAbi::FaultReason::InvalidTable);
    std::array<std::uint32_t, BdaAbi::HeaderWords> fields{};
    for (std::uint32_t word = 0; word < BdaAbi::HeaderWords; ++word) fields[word] = BdaLoadWord(state, constant(word));
    const auto differs = [&](std::uint32_t value, std::uint32_t expected) { return binary(spv::OpINotEqual, boolean, value, expected); };
    const auto either = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalOr, boolean, left, right); };
    const auto count = fields[BdaAbi::CountWord];
    const auto grants = fields[BdaAbi::GrantsWord];
    const auto available = binary(spv::OpISub, u32, length, constant(BdaAbi::HeaderWords));
    const auto reserved = fields[3];
    const auto layout = either(differs(fields[0], constant(BdaAbi::TableVersion)), differs(reserved, constant(0)));
    const auto rangeWords = binary(spv::OpIMul, u32, count, constant(BdaAbi::RangeWords));
    const auto exceeds = either(binary(spv::OpUGreaterThan, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))), binary(spv::OpUGreaterThan, boolean, grants, binary(spv::OpShiftRightLogical, u32, available, constant(2))));
    const auto sized = either(exceeds, differs(binary(spv::OpIAdd, u32, rangeWords, binary(spv::OpIMul, u32, grants, constant(BdaAbi::GrantWords))), available));
    fail(either(layout, sized), BdaAbi::FaultReason::InvalidTable);
    const auto end = binary(spv::OpIAdd, u64, address, widen(bytes));
    fail(binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow);
    const auto grantsFirst = binary(spv::OpIAdd, u32, constant(BdaAbi::HeaderWords), rangeWords);
    const auto permitted = [&](std::uint32_t permissions, std::uint32_t when) {
        const auto own = binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, permission), permission);
        if (!writes) return own;
        const auto readable = binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, binary(spv::OpBitwiseOr, u32, permissions, constant(BdaAbi::Write)), permission), permission);
        auto ask = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, own), readable);
        if (when != 0u) ask = binary(spv::OpLogicalAnd, boolean, when, ask);
        return EmitValueOrDefaultIfCondition(state, ask, boolean, own, [&] {
            const auto granted = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, boolean, granted, grantedFunction, address, end, grantsFirst, grants);
            return granted;
        });
    };
    const auto pages = pair(fields[BdaAbi::PageTableWord], fields[BdaAbi::PageTableWord + 1u]);
    const auto walk = binary(spv::OpLogicalAnd, boolean, binary(spv::OpINotEqual, boolean, pages, BdaConstant(state, 0u)), binary(spv::OpULessThan, boolean, address, BdaConstant(state, BdaAbi::PageTableEnd)));
    EmitIfCondition(state, walk, [&] {
        const auto window = binary(spv::OpShiftLeftLogical, u64, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WindowShift)), BdaConstant(state, 3u));
        const auto leaf = loadDevice(u64, binary(spv::OpIAdd, u64, pages, window), sizeof(std::uint64_t));
        EmitIfCondition(state, binary(spv::OpINotEqual, boolean, leaf, BdaConstant(state, 0u)), [&] {
            const auto page = binary(spv::OpShiftLeftLogical, u64, binary(spv::OpBitwiseAnd, u64, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::PageShift)), BdaConstant(state, BdaAbi::WindowPages - 1u)), BdaConstant(state, 4u));
            state.module.AddFunction(spv::OpStore, entry, loadDevice(entryType, binary(spv::OpIAdd, u64, leaf, page), sizeof(BdaAbi::PageEntry)));
            state.module.AddFunction(spv::OpStore, shift, constant(BdaAbi::PageShift));
            const auto header = state.module.AllocateId();
            const auto body = state.module.AllocateId();
            const auto continuation = state.module.AllocateId();
            const auto merge = state.module.AllocateId();
            state.module.AddFunction(spv::OpBranch, header);
            EmitLabel(state, header);
            const auto current = load(entryType, entry);
            const auto level = load(u32, shift);
            const auto split = binary(spv::OpIEqual, boolean, binary(spv::OpShiftRightLogical, u32, component(current, 2u), constant(BdaAbi::KindShift)), constant(static_cast<std::uint32_t>(BdaAbi::PageKind::Split)));
            const auto descend = binary(spv::OpLogicalAnd, boolean, split, binary(spv::OpINotEqual, boolean, level, constant(0)));
            state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
            state.module.AddFunction(spv::OpBranchConditional, descend, body, merge);
            EmitLabel(state, body);
            const auto next = binary(spv::OpISub, u32, level, constant(BdaAbi::SplitShift));
            const auto child = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, widen(next))), constant(BdaAbi::SplitChildren - 1u));
            const auto children = binary(spv::OpIAdd, u64, leaf, pair(component(current, 0u), component(current, 1u)));
            state.module.AddFunction(spv::OpStore, entry, loadDevice(entryType, binary(spv::OpIAdd, u64, children, widen(binary(spv::OpShiftLeftLogical, u32, child, constant(4u)))), sizeof(BdaAbi::PageEntry)));
            state.module.AddFunction(spv::OpStore, shift, next);
            state.module.AddFunction(spv::OpBranch, continuation);
            EmitLabel(state, continuation);
            state.module.AddFunction(spv::OpBranch, header);
            EmitLabel(state, merge);
            const auto found = load(entryType, entry);
            const auto bounds = component(found, 2u);
            const auto kind = binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::KindShift));
            fail(binary(spv::OpUGreaterThan, boolean, kind, constant(static_cast<std::uint32_t>(BdaAbi::PageKind::Segment))), BdaAbi::FaultReason::InvalidTable);
            const auto mask = binary(spv::OpISub, u32, binary(spv::OpShiftLeftLogical, u32, constant(1u), load(u32, shift)), constant(1u));
            const auto offset = widen(binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, address), mask));
            const auto first = widen(binary(spv::OpBitwiseAnd, u32, bounds, constant(BdaAbi::FirstMask)));
            const auto limit = pair(component(found, 3u), binary(spv::OpBitwiseAnd, u32, binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::LimitHighShift)), constant(1u)));
            const auto inside = binary(spv::OpLogicalAnd, boolean, binary(spv::OpUGreaterThanEqual, boolean, offset, first), binary(spv::OpULessThan, boolean, offset, limit));
            const auto segment = binary(spv::OpIEqual, boolean, kind, constant(static_cast<std::uint32_t>(BdaAbi::PageKind::Segment)));
            const auto held = binary(spv::OpLogicalAnd, boolean, segment, inside);
            fail(binary(spv::OpLogicalAnd, boolean, held, binary(spv::OpUGreaterThan, boolean, binary(spv::OpIAdd, u64, offset, widen(bytes)), limit)), BdaAbi::FaultReason::Unmapped);
            const auto allowed = permitted(binary(spv::OpBitwiseAnd, u32, binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::PermissionShift)), constant(BdaAbi::Read | BdaAbi::Write)), held);
            fail(binary(spv::OpLogicalAnd, boolean, held, Unary(state, spv::OpLogicalNot, boolean, allowed)), BdaAbi::FaultReason::Permission);
            EmitIfCondition(state, held, [&] { NoteBdaTouch(state, touch, address); });
            ReturnValueIf(state, held, translation(binary(spv::OpIAdd, u64, address, pair(component(found, 0u), component(found, 1u))), constant(0)));
        });
    });
    const auto index = EmitBdaBisection(state, low, high, address, constant(BdaAbi::HeaderWords), count, BdaAbi::RangeWords);
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(BdaAbi::HeaderWords), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(BdaAbi::RangeWords)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpUGreaterThanEqual, boolean, begin, finish), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped);
    fail(Unary(state, spv::OpLogicalNot, boolean, permitted(permissions, 0u)), BdaAbi::FaultReason::Permission);
    fail(binary(spv::OpIEqual, boolean, base, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable);
    const auto device = binary(spv::OpIAdd, u64, base, binary(spv::OpISub, u64, address, begin));
    fail(binary(spv::OpULessThan, boolean, device, base), BdaAbi::FaultReason::Overflow);
    const auto deviceEnd = binary(spv::OpIAdd, u64, device, widen(bytes));
    fail(binary(spv::OpULessThanEqual, boolean, deviceEnd, device), BdaAbi::FaultReason::Overflow);
    NoteBdaTouch(state, touch, address);
    state.module.AddFunction(spv::OpReturnValue, translation(device, constant(0)));
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

std::uint32_t SegmentType(SpirvEmitterState& state) {
    const auto u64 = TypeScalarU64(state);
    return state.module.Type(spv::OpTypeStruct, u64, u64, u64);
}

std::uint32_t DefineBdaSegment(SpirvEmitterState& state, std::uint32_t touch) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto entryType = TypeU32Vector(state, 4);
    const auto result = SegmentType(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto widen = [&](std::uint32_t value) { return Unary(state, spv::OpUConvert, u64, value); };
    const auto either = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalOr, boolean, left, right); };
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalAnd, boolean, left, right); };
    const auto differs = [&](std::uint32_t value, std::uint32_t expected) { return binary(spv::OpINotEqual, boolean, value, expected); };
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto loadDevice = [&](std::uint32_t type, std::uint32_t address, std::uint32_t alignment) {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, type), pointer, address);
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer, spv::MemoryAccessAlignedMask, alignment);
        return value;
    };
    const auto component = [&](std::uint32_t vector, std::uint32_t index) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, value, vector, index);
        return value;
    };
    const auto pair = [&](std::uint32_t low, std::uint32_t high) { return binary(spv::OpBitwiseOr, u64, widen(low), binary(spv::OpShiftLeftLogical, u64, widen(high), BdaConstant(state, 32u))); };
    const auto segment = [&](std::uint32_t begin, std::uint32_t end, std::uint32_t delta) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, result, value, begin, end, delta);
        return value;
    };
    const auto zero = BdaConstant(state, 0u);
    const auto empty = state.module.Constant(spv::OpConstantComposite, result, zero, zero, zero);
    const auto none = [&](std::uint32_t condition) { ReturnValueIf(state, condition, empty); };
    const auto readable = [&](std::uint32_t permissions) { return binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(BdaAbi::Read)), constant(0u)); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "bda_segment");
    state.module.AddFunction(spv::OpFunction, result, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, result, u64));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    EmitLabel(state, state.module.AllocateId());
    const auto variable = [&](std::uint32_t type) {
        const auto id = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, type), id, spv::StorageClassFunction);
        return id;
    };
    const auto low = variable(u32);
    const auto high = variable(u32);
    const auto entry = variable(entryType);
    const auto shift = variable(u32);
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    none(binary(spv::OpULessThan, boolean, length, constant(BdaAbi::HeaderWords)));
    std::array<std::uint32_t, BdaAbi::HeaderWords> fields{};
    for (std::uint32_t word = 0; word < BdaAbi::HeaderWords; ++word) fields[word] = BdaLoadWord(state, constant(word));
    const auto count = fields[BdaAbi::CountWord];
    const auto grants = fields[BdaAbi::GrantsWord];
    const auto available = binary(spv::OpISub, u32, length, constant(BdaAbi::HeaderWords));
    const auto reserved = fields[3];
    const auto layout = either(differs(fields[0], constant(BdaAbi::TableVersion)), differs(reserved, constant(0)));
    const auto rangeWords = binary(spv::OpIMul, u32, count, constant(BdaAbi::RangeWords));
    const auto exceeds = either(binary(spv::OpUGreaterThan, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))), binary(spv::OpUGreaterThan, boolean, grants, binary(spv::OpShiftRightLogical, u32, available, constant(2))));
    const auto sized = either(exceeds, differs(binary(spv::OpIAdd, u32, rangeWords, binary(spv::OpIMul, u32, grants, constant(BdaAbi::GrantWords))), available));
    none(either(layout, sized));
    const auto pages = pair(fields[BdaAbi::PageTableWord], fields[BdaAbi::PageTableWord + 1u]);
    const auto walk = both(binary(spv::OpINotEqual, boolean, pages, zero), binary(spv::OpULessThan, boolean, address, BdaConstant(state, BdaAbi::PageTableEnd)));
    EmitIfCondition(state, walk, [&] {
        const auto window = binary(spv::OpShiftLeftLogical, u64, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WindowShift)), BdaConstant(state, 3u));
        const auto leaf = loadDevice(u64, binary(spv::OpIAdd, u64, pages, window), sizeof(std::uint64_t));
        EmitIfCondition(state, binary(spv::OpINotEqual, boolean, leaf, zero), [&] {
            const auto page = binary(spv::OpShiftLeftLogical, u64, binary(spv::OpBitwiseAnd, u64, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::PageShift)), BdaConstant(state, BdaAbi::WindowPages - 1u)), BdaConstant(state, 4u));
            state.module.AddFunction(spv::OpStore, entry, loadDevice(entryType, binary(spv::OpIAdd, u64, leaf, page), sizeof(BdaAbi::PageEntry)));
            state.module.AddFunction(spv::OpStore, shift, constant(BdaAbi::PageShift));
            const auto header = state.module.AllocateId();
            const auto body = state.module.AllocateId();
            const auto continuation = state.module.AllocateId();
            const auto merge = state.module.AllocateId();
            state.module.AddFunction(spv::OpBranch, header);
            EmitLabel(state, header);
            const auto current = load(entryType, entry);
            const auto level = load(u32, shift);
            const auto split = binary(spv::OpIEqual, boolean, binary(spv::OpShiftRightLogical, u32, component(current, 2u), constant(BdaAbi::KindShift)), constant(static_cast<std::uint32_t>(BdaAbi::PageKind::Split)));
            const auto descend = both(split, binary(spv::OpINotEqual, boolean, level, constant(0)));
            state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
            state.module.AddFunction(spv::OpBranchConditional, descend, body, merge);
            EmitLabel(state, body);
            const auto next = binary(spv::OpISub, u32, level, constant(BdaAbi::SplitShift));
            const auto child = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, widen(next))), constant(BdaAbi::SplitChildren - 1u));
            const auto children = binary(spv::OpIAdd, u64, leaf, pair(component(current, 0u), component(current, 1u)));
            state.module.AddFunction(spv::OpStore, entry, loadDevice(entryType, binary(spv::OpIAdd, u64, children, widen(binary(spv::OpShiftLeftLogical, u32, child, constant(4u)))), sizeof(BdaAbi::PageEntry)));
            state.module.AddFunction(spv::OpStore, shift, next);
            state.module.AddFunction(spv::OpBranch, continuation);
            EmitLabel(state, continuation);
            state.module.AddFunction(spv::OpBranch, header);
            EmitLabel(state, merge);
            const auto found = load(entryType, entry);
            const auto bounds = component(found, 2u);
            const auto kind = binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::KindShift));
            none(binary(spv::OpUGreaterThan, boolean, kind, constant(static_cast<std::uint32_t>(BdaAbi::PageKind::Segment))));
            const auto mask = binary(spv::OpISub, u32, binary(spv::OpShiftLeftLogical, u32, constant(1u), load(u32, shift)), constant(1u));
            const auto offset = widen(binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, address), mask));
            const auto first = widen(binary(spv::OpBitwiseAnd, u32, bounds, constant(BdaAbi::FirstMask)));
            const auto limit = pair(component(found, 3u), binary(spv::OpBitwiseAnd, u32, binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::LimitHighShift)), constant(1u)));
            const auto inside = both(binary(spv::OpUGreaterThanEqual, boolean, offset, first), binary(spv::OpULessThan, boolean, offset, limit));
            const auto held = both(binary(spv::OpIEqual, boolean, kind, constant(static_cast<std::uint32_t>(BdaAbi::PageKind::Segment))), inside);
            const auto permissions = binary(spv::OpBitwiseAnd, u32, binary(spv::OpShiftRightLogical, u32, bounds, constant(BdaAbi::PermissionShift)), constant(BdaAbi::Read | BdaAbi::Write));
            none(both(held, Unary(state, spv::OpLogicalNot, boolean, readable(permissions))));
            const auto start = binary(spv::OpISub, u64, address, offset);
            EmitIfCondition(state, held, [&] { NoteBdaTouch(state, touch, address); });
            ReturnValueIf(state, held, segment(binary(spv::OpIAdd, u64, start, first), binary(spv::OpIAdd, u64, start, limit), pair(component(found, 0u), component(found, 1u))));
        });
    });
    const auto index = EmitBdaBisection(state, low, high, address, constant(BdaAbi::HeaderWords), count, BdaAbi::RangeWords);
    none(binary(spv::OpIEqual, boolean, index, constant(0)));
    const auto selected = binary(spv::OpIAdd, u32, constant(BdaAbi::HeaderWords), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(BdaAbi::RangeWords)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    none(differs(BdaLoadWord(state, at(7)), constant(0)));
    none(binary(spv::OpUGreaterThanEqual, boolean, begin, finish));
    none(binary(spv::OpUGreaterThanEqual, boolean, address, finish));
    none(Unary(state, spv::OpLogicalNot, boolean, readable(BdaLoadWord(state, at(6)))));
    none(binary(spv::OpIEqual, boolean, base, zero));
    NoteBdaTouch(state, touch, address);
    state.module.AddFunction(spv::OpReturnValue, segment(begin, finish, binary(spv::OpISub, u64, base, begin)));
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    if (recordFaults && state.bdaFaultFunction == 0) throw std::runtime_error("BDA fault function is missing");
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    state.module.AddFunction(spv::OpFunction, u64, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, u64, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto translated = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TranslationType(state), translated, state.bdaTranslateFunction, address, bytes, ConstantU32(state, permission));
    const auto device = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u64, device, translated, 0u);
    if (recordFaults) {
        const auto reason = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, reason, translated, 1u);
        EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), reason, ConstantU32(state, 0u)), [&] {
            state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, reason, address, bytes, instruction, BdaFaultStage(state), ConstantU32(state, 0u));
        });
    }
    state.module.AddFunction(spv::OpReturnValue, device);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

static void EmitWrittenChunkNote(SpirvEmitterState& state, std::uint32_t key, std::uint32_t bit, std::uint32_t done, std::uint32_t counter) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, counter, constant(0u));
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, key, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto probe = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, probe, counter);
    const auto found = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, found, done);
    const auto searching = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, found), binary(spv::OpULessThan, boolean, probe, constant(BdaAbi::WrittenPageProbes)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto slot = binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, probe), constant(BdaAbi::WrittenPageSlots - 1u));
    const auto keyWord = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpShiftLeftLogical, u32, slot, constant(1u)));
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, u32, current, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed);
    const auto same = binary(spv::OpIEqual, boolean, current, key);
    state.module.AddFunction(spv::OpStore, done, same);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, same), [&] {
        const auto previous = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed, relaxed, key, constant(0u));
        const auto claimed = binary(spv::OpIEqual, boolean, previous, constant(0u));
        state.module.AddFunction(spv::OpStore, done, binary(spv::OpLogicalOr, boolean, claimed, binary(spv::OpIEqual, boolean, previous, key)));
        EmitIfCondition(state, claimed, [&] {
            const auto index = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicIAdd, u32, index, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenCountWord)), scope, relaxed, constant(1u));
            state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenListWord), index)), scope, relaxed, slot);
        });
    });
    const auto holds = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, holds, done);
    EmitIfCondition(state, holds, [&] {
        const auto maskWord = binary(spv::OpIAdd, u32, keyWord, constant(1u));
        const auto mask = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicLoad, u32, mask, BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed);
        EmitIfCondition(state, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, mask, bit), constant(0u)), [&] {
            state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed, bit);
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, counter, binary(spv::OpIAdd, u32, probe, constant(1u)));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
}

static std::array<std::uint32_t, 2> DeclareWrittenChunkNoteVariables(SpirvEmitterState& state) {
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, TypeBool(state)), done, spv::StorageClassFunction);
    const auto counter = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, TypeU32(state)), counter, spv::StorageClassFunction);
    return {done, counter};
}

static std::uint32_t DefineBdaNoteWrite(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_write");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, voidType, u64));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    EmitLabel(state, state.module.AllocateId());
    const auto [done, counter] = DeclareWrittenChunkNoteVariables(state);
    const auto key = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenChunkShift))), constant(1));
    const auto page = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(BdaAbi::WrittenChunkPages - 1u));
    EmitWrittenChunkNote(state, key, binary(spv::OpShiftLeftLogical, u32, constant(1u), page), done, counter);
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

std::uint32_t DefineWrittenChunkNote(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_written_chunk");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, voidType, u32, u32));
    const auto key = state.module.AllocateId();
    const auto bit = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u32, key);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bit);
    EmitLabel(state, state.module.AllocateId());
    const auto [done, counter] = DeclareWrittenChunkNoteVariables(state);
    EmitWrittenChunkNote(state, key, bit, done, counter);
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void DeclareBdaReadCache(SpirvEmitterState& state) {
    state.bdaReadCache = {};
    if (state.bdaSegmentFunction == 0u) return;
    const auto pointer = TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state));
    const auto zero = BdaConstant(state, 0u);
    for (auto& variable : state.bdaReadCache) {
        variable = state.module.AllocateId();
        state.module.EmitGlobalVariable({spv::OpVariable, pointer, variable, spv::StorageClassPrivate, zero});
    }
}

bool BdaReadCacheAvailable(const SpirvEmitterState& state) {
    return state.bdaReadCache[0] != 0u;
}

void DeclareBdaVectorCache(SpirvEmitterState& state) {
    state.bdaVectorCache = {};
    if (state.bdaSegmentFunction == 0u) return;
    const auto pointer = TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state));
    const auto zero = BdaConstant(state, 0u);
    for (auto& variable : state.bdaVectorCache) {
        variable = state.module.AllocateId();
        state.module.EmitGlobalVariable({spv::OpVariable, pointer, variable, spv::StorageClassPrivate, zero});
    }
}

bool BdaVectorCacheAvailable(const SpirvEmitterState& state) {
    return state.bdaVectorCache[0] != 0u;
}

BdaCacheProbe ProbeBdaSegment(SpirvEmitterState& state, const BdaSegmentValues& segment, std::uint32_t address, std::uint32_t bytes) {
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto end = Binary(state, spv::OpIAdd, u64, address, bytes);
    const auto device = Binary(state, spv::OpIAdd, u64, address, segment.delta);
    const auto inside = both(Binary(state, spv::OpULessThanEqual, boolean, segment.begin, address), Binary(state, spv::OpULessThanEqual, boolean, end, segment.end));
    const auto aligned = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u64, device, BdaConstant(state, 3u)), BdaConstant(state, 0u));
    return {both(both(Binary(state, spv::OpULessThan, boolean, address, end), inside), aligned), device};
}

BdaCacheProbe ProbeBdaVectorCache(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes) {
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeScalarU64(state), value, variable);
        return value;
    };
    const auto begin = load(state.bdaVectorCache[0]);
    const auto end = load(state.bdaVectorCache[1]);
    return ProbeBdaSegment(state, {begin, end, load(state.bdaVectorCache[2])}, address, bytes);
}

void RefreshBdaVectorCache(SpirvEmitterState& state, std::uint32_t address) {
    const auto segment = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, SegmentType(state), segment, state.bdaSegmentFunction, address);
    for (std::uint32_t field = 0; field < 3u; ++field) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeScalarU64(state), value, segment, field);
        state.module.AddFunction(spv::OpStore, state.bdaVectorCache[field], value);
    }
}

bool WaveUniformDescriptor(SpirvEmitterState& state, const std::vector<const IrValue*>& words) {
    const auto& program = state.program;
    if (program.Resources().stage != IrShaderStage::Compute || state.laneCount != 1u || state.hostSubgroupSize < 2u || state.hostSubgroupSize > program.WaveSize()) return false;
    if (std::find(state.supportedCapabilities.begin(), state.supportedCapabilities.end(), static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformBallot)) == state.supportedCapabilities.end()) return false;
    bool anyLaneValue = false;
    for (const auto* word : words) {
        const auto* resolved = word->Resolve();
        if (resolved == nullptr) return false;
        if (resolved->HasImmediate()) continue;
        if (!state.sharedLaneValues.contains(resolved)) return false;
        anyLaneValue = true;
    }
    return anyLaneValue;
}

std::uint32_t BroadcastFirstLane(SpirvEmitterState& state, std::uint32_t type, std::uint32_t value) {
    state.module.EmitCapability(spv::CapabilityGroupNonUniform);
    state.module.EmitCapability(spv::CapabilityGroupNonUniformBallot);
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpGroupNonUniformBroadcastFirst, type, result, ConstantU32(state, spv::ScopeSubgroup), value);
    return result;
}

namespace {

constexpr std::uint64_t FormatCheckTag = ~std::uint64_t{1};

bool BufferLoadOpcode(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::LoadBufferU8:
    case IrOpcode::LoadBufferU16:
    case IrOpcode::LoadBufferU32:
    case IrOpcode::LoadBufferU32x2:
    case IrOpcode::LoadBufferU32x3:
    case IrOpcode::LoadBufferU32x4:
        return true;
    default:
        return false;
    }
}

const IrValue* RuntimeFormatHandle(const SpirvEmitterState& state, const IrValue& inst) {
    if (!state.bdaStopsInvocations || !BufferLoadOpcode(inst.Opcode()) || inst.ArgumentCount() < 2u) return nullptr;
    const auto& infos = state.program.Resources().memoryInfo;
    const auto index = inst.Flags<MemoryFlags>().index;
    if (index >= infos.size()) return nullptr;
    const auto& memory = infos[index];
    if (!memory.gpuDescriptor || !memory.formatted || memory.typed || memory.kind != ResourceKind::Buffer) return nullptr;
    const auto* handle = inst.Argument(0)->Resolve();
    if (handle == nullptr || handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) return nullptr;
    return handle->Argument(3)->Resolve()->HasImmediate() ? nullptr : handle;
}

}

std::vector<const IrValue*> InvalidFormatInputs(const SpirvEmitterState& state, const IrValue& inst) {
    const auto* handle = RuntimeFormatHandle(state, inst);
    if (handle == nullptr) return {};
    return {handle->Argument(0), handle->Argument(1), handle->Argument(3), inst.Argument(inst.ArgumentCount() - 1u)};
}

void EmitInvalidFormatStop(SpirvValueEmitContext& ctx, const IrValue& inst, const IrValue& anchor) {
    auto& state = ctx.state;
    const auto* handle = RuntimeFormatHandle(state, inst);
    if (handle == nullptr || state.module.InHelperFunction() || anchor.Parent() != inst.Parent() || inst.Parent() == nullptr) return;
    const auto inputs = InvalidFormatInputs(state, inst);
    for (const auto* input : inputs) {
        const auto* resolved = input->Resolve();
        if (!resolved->HasImmediate() && !ctx.definitions.contains(resolved)) return;
    }
    bool between = false;
    for (const IrValue* other : inst.Parent()->Instructions()) {
        if (other == &anchor) {
            between = true;
            continue;
        }
        if (other == &inst) break;
        if (between && IrOpcodeHasSideEffects(other->Opcode())) return;
    }
    if (!between) return;
    std::vector<std::uint64_t> key{FormatCheckTag, state.laneHalf};
    for (const auto* input : inputs) {
        const auto* resolved = input->Resolve();
        key.push_back(resolved->HasImmediate() ? 0u : reinterpret_cast<std::uintptr_t>(resolved));
        key.push_back(resolved->HasImmediate() ? resolved->ImmediateU32() : 0u);
    }
    if (state.deviceImageLookups.contains(key)) return;
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto format = Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpShiftRightLogical, u32, ctx.Def(inputs[2]), ConstantU32(state, 12u)), ConstantU32(state, 0x7fu));
    const auto unsupported = Binary(state, spv::OpLogicalOr, boolean, Binary(state, spv::OpIEqual, boolean, format, ConstantU32(state, 0u)), Binary(state, spv::OpUGreaterThan, boolean, format, ConstantU32(state, static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float))));
    const auto stops = Binary(state, spv::OpLogicalAnd, boolean, ctx.Def(inputs[3]), unsupported);
    EmitIfCondition(state, stops, [&] {
        const auto high = Binary(state, spv::OpBitwiseAnd, u32, ctx.Def(inputs[1]), ConstantU32(state, 0xffffu));
        const auto u64 = TypeScalarU64(state);
        const auto base = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, ctx.Def(inputs[0])), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, high), BdaConstant(state, 32u)));
        RecordBdaFault(state, base, ConstantU32(state, 0u), BdaInstructionPc(state, inst), BdaAbi::FaultReason::InvalidDescriptor);
    });
    StopBdaInvocationIf(state, stops);
    state.deviceImageLookups.emplace(std::move(key), 0u);
}

namespace {

constexpr std::uint64_t ResolvedBufferTag = ~std::uint64_t{0};

bool RawDwordBufferRead(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::LoadBufferU32:
    case IrOpcode::LoadBufferU32x2:
    case IrOpcode::LoadBufferU32x3:
    case IrOpcode::LoadBufferU32x4:
        return true;
    default:
        return false;
    }
}

const IrValue* ResolvedBufferHandle(const SpirvEmitterState& state, const IrValue& inst) {
    const bool raw = RawDwordBufferRead(inst.Opcode());
    if ((!raw && inst.Opcode() != IrOpcode::ReadConstBuffer) || inst.ArgumentCount() == 0u) return nullptr;
    const auto* handle = inst.Argument(0)->Resolve();
    if (handle == nullptr || handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) return nullptr;
    const auto& infos = state.program.Resources().memoryInfo;
    const auto index = inst.Flags<MemoryFlags>().index;
    if (index >= infos.size()) return nullptr;
    const auto& memory = infos[index];
    if (!memory.gpuDescriptor || memory.planningOnly) return nullptr;
    if (raw) return memory.kind == ResourceKind::Buffer && (memory.formatted || memory.dataBits == 32u) ? handle : nullptr;
    return memory.kind == ResourceKind::ScalarBuffer ? handle : nullptr;
}

std::vector<std::uint64_t> ResolvedBufferKey(const SpirvEmitterState& state, const IrValue& handle) {
    std::vector<std::uint64_t> key{ResolvedBufferTag, state.laneHalf};
    for (std::uint32_t word = 0; word < 2u; ++word) {
        const auto* resolved = handle.Argument(word)->Resolve();
        key.push_back(resolved->HasImmediate() ? 0u : reinterpret_cast<std::uintptr_t>(resolved));
        key.push_back(resolved->HasImmediate() ? resolved->ImmediateU32() : 0u);
    }
    return key;
}

std::uint32_t SplitU64(SpirvEmitterState& state, std::uint32_t value, bool high) {
    const auto u64 = TypeScalarU64(state);
    return Unary(state, spv::OpUConvert, TypeU32(state), high ? Binary(state, spv::OpShiftRightLogical, u64, value, BdaConstant(state, 32u)) : value);
}

std::uint32_t JoinU64(SpirvEmitterState& state, std::uint32_t low, std::uint32_t high) {
    const auto u64 = TypeScalarU64(state);
    return Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, low), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, high), BdaConstant(state, 32u)));
}

std::uint32_t Extract(SpirvEmitterState& state, std::uint32_t type, std::uint32_t composite, std::uint32_t index) {
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, type, value, composite, index);
    return value;
}

}

std::vector<const IrValue*> ResolvedBufferInputs(const SpirvEmitterState& state, const IrValue& inst) {
    const auto* handle = ResolvedBufferHandle(state, inst);
    if (handle == nullptr || !state.program.Info().usesDma || BdaByteReadsForced()) return {};
    return {handle->Argument(0), handle->Argument(1)};
}

void EmitResolvedBufferSegment(SpirvValueEmitContext& ctx, const IrValue& inst, const IrValue& anchor) {
    auto& state = ctx.state;
    const auto* handle = ResolvedBufferHandle(state, inst);
    if (handle == nullptr || state.bdaSegmentFunction == 0u) return;
    std::vector<const IrValue*> words{handle->Argument(0), handle->Argument(1)};
    if (std::ranges::none_of(words, [&](const IrValue* word) { return word->Resolve() == &anchor; })) return;
    for (const auto* word : words) {
        const auto* resolved = word->Resolve();
        if (!resolved->HasImmediate() && !ctx.definitions.contains(resolved)) return;
    }
    auto key = ResolvedBufferKey(state, *handle);
    if (state.deviceImageLookups.contains(key)) return;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto type = SegmentType(state);
    const auto high = Binary(state, spv::OpBitwiseAnd, u32, ctx.Def(words[1]), ConstantU32(state, 0xffffu));
    auto segment = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, type, segment, state.bdaSegmentFunction, JoinU64(state, ctx.Def(words[0]), high));
    if (WaveUniformDescriptor(state, words)) {
        std::array<std::uint32_t, 3> fields{};
        for (std::uint32_t field = 0; field < 3u; ++field) fields[field] = Extract(state, u64, segment, field);
        const auto bounds = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), bounds, SplitU64(state, fields[0], false), SplitU64(state, fields[0], true), SplitU64(state, fields[1], false), SplitU64(state, fields[1], true));
        const auto delta = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 2u), delta, SplitU64(state, fields[2], false), SplitU64(state, fields[2], true));
        const auto sharedBounds = BroadcastFirstLane(state, TypeU32Vector(state, 4u), bounds);
        const auto sharedDelta = BroadcastFirstLane(state, TypeU32Vector(state, 2u), delta);
        const auto part = [&](std::uint32_t vector, std::uint32_t index) { return Extract(state, u32, vector, index); };
        segment = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, type, segment, JoinU64(state, part(sharedBounds, 0u), part(sharedBounds, 1u)), JoinU64(state, part(sharedBounds, 2u), part(sharedBounds, 3u)), JoinU64(state, part(sharedDelta, 0u), part(sharedDelta, 1u)));
    }
    state.deviceImageLookups.emplace(std::move(key), segment);
}

std::optional<BdaSegmentValues> ResolvedBufferSegment(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    const auto* handle = ResolvedBufferHandle(state, inst);
    if (handle == nullptr || state.module.InHelperFunction()) return std::nullopt;
    const auto found = state.deviceImageLookups.find(ResolvedBufferKey(state, *handle));
    if (found == state.deviceImageLookups.end()) return std::nullopt;
    const auto u64 = TypeScalarU64(state);
    return BdaSegmentValues{Extract(state, u64, found->second, 0u), Extract(state, u64, found->second, 1u), Extract(state, u64, found->second, 2u)};
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (state.faultBufferVariable != 0) DefineBdaFaultFunction(state);
    if (!state.program.Info().usesDma) return;
    const auto touch = DefineBdaTouch(state);
    state.bdaTranslateFunction = DefineBdaTranslate(state, state.program.Info().bdaWrites, touch);
    if (!BdaByteReadsForced()) state.bdaSegmentFunction = DefineBdaSegment(state, touch);
    DeclareBdaVectorCache(state);
    DeclareBdaReadCache(state);
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    DefineBdaSpanReadFunctions(state);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        if (!BdaByteReadsForced()) state.bdaWriteProbeFunction = DefineBdaLookup(state, "probe_bda_write_pointer", false, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
        DefineBdaByteWriteFunctions(state);
        DefineBdaWriteFunctions(state);
    }
}

}
