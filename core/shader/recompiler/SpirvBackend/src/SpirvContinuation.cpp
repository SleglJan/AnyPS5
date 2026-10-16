#include "SpirvBackend/SpirvContinuation.hpp"
#include "BdaAbi.hpp"
#include "ContinuationAbi.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvEmitterInstructions.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>

namespace ShaderRecompiler {

namespace {

constexpr std::uint32_t NoBin = 0xffffffffu;

class Block {
public:
    Block(SpirvEmitterState& state, std::uint32_t low, std::uint32_t high) : state(state) {
        const auto u64 = TypeScalarU64(state);
        address = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, low), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, high), BdaConstant(state, 32u)));
    }

    std::uint32_t Pointer(std::uint32_t dword) const {
        const auto u64 = TypeScalarU64(state);
        const auto offset = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, dword), BdaConstant(state, 2u));
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, TypeU32(state)), pointer, Binary(state, spv::OpIAdd, u64, address, offset));
        return pointer;
    }

    std::uint32_t Load(std::uint32_t dword) const {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), value, Pointer(dword), spv::MemoryAccessAlignedMask, 4u);
        return value;
    }

    void Store(std::uint32_t dword, std::uint32_t value) const {
        state.module.AddFunction(spv::OpStore, Pointer(dword), value, spv::MemoryAccessAlignedMask, 4u);
    }

    std::uint32_t Header(std::uint32_t word) const {
        return Load(ConstantU32(state, word));
    }

    std::uint32_t Record(std::uint32_t slot, std::uint32_t word) const {
        const auto start = EmitAddU32(state, Header(ContinuationAbi::RecordsWord), EmitBinaryU32(state, spv::OpIMul, slot, Header(ContinuationAbi::RecordDwordsWord)));
        return EmitAddU32(state, start, word);
    }

private:
    SpirvEmitterState& state;
    std::uint32_t address = 0;
};

template<typename TBody>
std::uint32_t EmitCountedLoop(SpirvEmitterState& state, std::uint32_t first, std::uint32_t step, std::uint32_t end, std::uint32_t initial, TBody&& body) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto entry = state.currentLabel;
    const auto header = state.module.AllocateId();
    const auto bodyLabel = state.module.AllocateId();
    const auto continueLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto index = state.module.AllocateId();
    const auto carried = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    const auto updated = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    state.module.AddFunction(spv::OpPhi, u32, index, first, entry, next, continueLabel);
    state.module.AddFunction(spv::OpPhi, u32, carried, initial, entry, updated, continueLabel);
    const auto more = Binary(state, spv::OpLogicalAnd, boolean, Binary(state, spv::OpULessThan, boolean, index, end), Binary(state, spv::OpIEqual, boolean, carried, initial));
    state.module.AddFunction(spv::OpLoopMerge, merge, continueLabel, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, more, bodyLabel, merge);
    EmitLabel(state, bodyLabel);
    const auto result = body(index, carried);
    state.module.AddFunction(spv::OpCopyObject, u32, updated, result);
    state.module.AddFunction(spv::OpBranch, continueLabel);
    EmitLabel(state, continueLabel);
    state.module.AddFunction(spv::OpIAdd, u32, next, index, step);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    return carried;
}

std::uint32_t SearchBin(SpirvEmitterState& state, const Block& block, std::uint32_t site, const std::array<std::uint32_t, ContinuationAbi::KeyWords>& key) {
    const auto boolean = TypeBool(state);
    const auto bins = block.Header(ContinuationAbi::BinsWord);
    return EmitCountedLoop(state, ConstantU32(state, 0u), ConstantU32(state, 1u), bins, ConstantU32(state, NoBin), [&](std::uint32_t index, std::uint32_t found) {
        const auto entry = EmitAddU32(state, ConstantU32(state, ContinuationAbi::BinTableDword), EmitBinaryU32(state, spv::OpIMul, index, ConstantU32(state, ContinuationAbi::BinEntryDwords)));
        const auto word = [&](std::uint32_t offset) { return block.Load(EmitAddU32(state, entry, ConstantU32(state, offset))); };
        const auto mask = word(ContinuationAbi::BinKeyMaskWord);
        auto match = Binary(state, spv::OpIEqual, boolean, word(ContinuationAbi::BinSiteWord), site);
        for (std::uint32_t keyWord = 0; keyWord < ContinuationAbi::KeyWords; ++keyWord) {
            const auto ignored = Binary(state, spv::OpIEqual, boolean, EmitBinaryU32(state, spv::OpBitwiseAnd, mask, ConstantU32(state, 1u << keyWord)), ConstantU32(state, 0u));
            const auto equal = Binary(state, spv::OpIEqual, boolean, word(ContinuationAbi::BinKeyWord + keyWord), key[keyWord]);
            match = Binary(state, spv::OpLogicalAnd, boolean, match, Binary(state, spv::OpLogicalOr, boolean, ignored, equal));
        }
        return Select(state, TypeU32(state), match, index, found);
    });
}

void EmitLdsBarrier(SpirvEmitterState& state) {
    state.module.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup), ConstantU32(state, spv::ScopeWorkgroup), ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | LdsMemorySemantics(state)));
}

void EmitLdsCopy(SpirvValueEmitContext& ctx, const IrValue& inst, bool spill) {
    auto& state = ctx.state;
    const Block block(state, ctx.Arg(inst, 0), ctx.Arg(inst, 1));
    const auto slot = ctx.Arg(inst, 2);
    const auto lane = ctx.Arg(inst, 3);
    const auto ldsWord = ctx.Arg(inst, 4);
    const IrValue* count = inst.Argument(5)->Resolve();
    if (!count->HasImmediate()) ctx.Fail(inst, "copies an LDS size that is not a constant");
    if (count->ImmediateU32() > LdsDwordCount(state)) ctx.Fail(inst, "copies more LDS than the program allocates");
    if (spill && ctx.half == 0u) EmitLdsBarrier(state);
    MemoryInfo memory;
    memory.kind = ResourceKind::Lds;
    const auto access = PrepareMemoryResourceAccess(state, memory);
    const auto base = block.Record(slot, ldsWord);
    static_cast<void>(EmitCountedLoop(state, lane, ConstantU32(state, state.program.WaveSize()), ConstantU32(state, count->ImmediateU32()), ConstantU32(state, 0u), [&](std::uint32_t index, std::uint32_t carried) {
        const auto pointer = EmitMemoryElementPointer(state, access, index);
        const auto word = EmitAddU32(state, base, index);
        if (spill) {
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
            block.Store(word, value);
        } else {
            state.module.AddFunction(spv::OpStore, pointer, block.Load(word));
        }
        return carried;
    }));
    if (!spill && ctx.half + 1u == state.laneCount) EmitLdsBarrier(state);
}

}

void EmitContinuationSpill(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    const Block block(state, ctx.Arg(inst, 0), ctx.Arg(inst, 1));
    const auto store = [&] { block.Store(block.Record(ctx.Arg(inst, 2), ctx.Arg(inst, 3)), ctx.Arg(inst, 4)); };
    const IrValue* condition = inst.Argument(5)->Resolve();
    if (condition->HasImmediate() && condition->ImmediateBool()) {
        store();
        return;
    }
    EmitIfCondition(state, ctx.Arg(inst, 5), store);
}

void EmitContinuationSpillLds(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitLdsCopy(ctx, inst, true);
}

void EmitTableCallBin(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    if (state.faultBufferVariable == 0) ctx.Fail(inst, "bins a table call without a fault buffer");
    const auto boolean = TypeBool(state);
    const auto site = ctx.Arg(inst, 0);
    const auto programCounter = ctx.Arg(inst, 1);
    const std::array<std::uint32_t, ContinuationAbi::KeyWords> key{ctx.Arg(inst, 2), ctx.Arg(inst, 3), ctx.Arg(inst, 4), ctx.Arg(inst, 5)};
    const auto targetLow = key[0];
    const auto targetHigh = key[1];
    const Block block(state, ctx.Arg(inst, 6), ctx.Arg(inst, 7));
    const auto slot = ctx.Arg(inst, 8);
    const auto lane = ctx.Arg(inst, 9);
    const auto bin = SearchBin(state, block, site, key);
    const auto round = block.Header(ContinuationAbi::RoundWord);
    const auto next = EmitAddU32(state, round, ConstantU32(state, 1u));
    const auto rounds = block.Header(ContinuationAbi::RoundsWord);
    const auto known = Binary(state, spv::OpINotEqual, boolean, bin, ConstantU32(state, NoBin));
    const auto inRange = Binary(state, spv::OpULessThanEqual, boolean, next, rounds);
    const auto leader = Binary(state, spv::OpIEqual, boolean, lane, ConstantU32(state, 0u));
    const auto both = [&](std::uint32_t first, std::uint32_t second) { return Binary(state, spv::OpLogicalAnd, boolean, first, second); };
    const auto none = [&](std::uint32_t value) { return Unary(state, spv::OpLogicalNot, boolean, value); };
    EmitIfCondition(state, leader, [&] {
        const auto record = block.Record(slot, ConstantU32(state, 0u));
        const auto at = [&](std::uint32_t word) { return EmitAddU32(state, record, ConstantU32(state, word)); };
        block.Store(at(ContinuationAbi::RecordSiteWord), site);
        block.Store(at(ContinuationAbi::RecordTargetLowWord), targetLow);
        block.Store(at(ContinuationAbi::RecordTargetHighWord), targetHigh);
        block.Store(at(ContinuationAbi::RecordRoundWord), next);
    });
    EmitIfCondition(state, both(leader, none(known)), [&] {
        RecordBdaFaultWords(state, targetLow, targetHigh, ConstantU32(state, 0u), programCounter, BdaAbi::FaultReason::UnlinkedCall);
    });
    EmitIfCondition(state, both(leader, both(known, none(inRange))), [&] {
        RecordBdaFaultWords(state, targetLow, targetHigh, rounds, programCounter, BdaAbi::FaultReason::Overflow);
    });
    EmitIfCondition(state, both(leader, both(known, inRange)), [&] {
        const auto bins = block.Header(ContinuationAbi::BinsWord);
        const auto counter = EmitAddU32(state, block.Header(ContinuationAbi::CountersWord), EmitBinaryU32(state, spv::OpIMul, EmitAddU32(state, EmitBinaryU32(state, spv::OpIMul, round, bins), bin), ConstantU32(state, ContinuationAbi::CounterDwords)));
        const auto index = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicIAdd, TypeU32(state), index, block.Pointer(counter), ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, 1u));
        const auto parity = EmitBinaryU32(state, spv::OpBitwiseAnd, next, ConstantU32(state, 1u));
        const auto list = EmitBinaryU32(state, spv::OpIMul, EmitAddU32(state, EmitBinaryU32(state, spv::OpIMul, parity, bins), bin), block.Header(ContinuationAbi::SlotsWord));
        block.Store(EmitAddU32(state, block.Header(ContinuationAbi::ListsWord), EmitAddU32(state, list, index)), slot);
    });
}

std::uint32_t EmitContinuationSlot(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    const Block block(state, ctx.Arg(inst, 0), ctx.Arg(inst, 1));
    const auto parity = EmitBinaryU32(state, spv::OpBitwiseAnd, block.Header(ContinuationAbi::RoundWord), ConstantU32(state, 1u));
    const auto list = EmitBinaryU32(state, spv::OpIMul, EmitAddU32(state, EmitBinaryU32(state, spv::OpIMul, parity, block.Header(ContinuationAbi::BinsWord)), ctx.Arg(inst, 2)), block.Header(ContinuationAbi::SlotsWord));
    return block.Load(EmitAddU32(state, block.Header(ContinuationAbi::ListsWord), EmitAddU32(state, list, ctx.Arg(inst, 3))));
}

std::uint32_t EmitContinuationLoad(SpirvValueEmitContext& ctx, const IrValue& inst) {
    const Block block(ctx.state, ctx.Arg(inst, 0), ctx.Arg(inst, 1));
    return block.Load(block.Record(ctx.Arg(inst, 2), ctx.Arg(inst, 3)));
}

void EmitContinuationRestoreLds(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitLdsCopy(ctx, inst, false);
}

}
