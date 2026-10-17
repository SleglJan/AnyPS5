#include "SpirvBackend/SpirvAluEmitter.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvEmitterInstructions.hpp"
#include <algorithm>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ShaderRecompiler {

void EmitAluValue(SpirvModule& module, const IrValue& value, std::uint32_t resultId) {
    throw std::runtime_error("EmitAluValue not implemented");
}

void EmitAluValue(SpirvValueEmitContext& context, const IrValue& value, std::uint32_t resultId) {
    throw std::runtime_error("EmitAluValue not implemented");
}

void EmitGetThreadBitScalarRegister(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetThreadBitScalarRegister(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetScalarMaskTag(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetScalarMaskTag(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetScalarRegister(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetScalarRegister(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetVectorRegister(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetVectorRegister(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

namespace {

std::uint32_t VectorWindowElement(SpirvEmitterState& state, std::uint32_t index) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassFunction, TypeU32(state)), pointer, state.vectorWindowVariable.at(state.laneHalf), index);
    return pointer;
}

bool IndexedWindowInSlots(const SpirvEmitterState& state) {
    return state.vectorWindowIndexed != 0u && state.vectorWindowSlots.at(state.laneHalf).size() == state.program.Metadata().vectorWindowSize;
}

std::uint32_t WindowSlot(const SpirvEmitterState& state, std::uint32_t entry) {
    return state.vectorWindowSlots.at(state.laneHalf).at(IndexedWindowInSlots(state) ? entry : entry - state.vectorWindowIndexed);
}

constexpr std::uint32_t WindowSegmentShift = 3u;

std::uint32_t AccessWindowSlots(SpirvEmitterState& state, std::uint32_t index, std::uint32_t first, std::uint32_t end, const std::function<std::uint32_t(std::uint32_t)>& access) {
    const auto entry = state.currentLabel;
    if (end - first > (1u << WindowSegmentShift)) {
        const auto segment = Binary(state, spv::OpShiftRightLogical, TypeU32(state), index, ConstantU32(state, WindowSegmentShift));
        const auto segmentsMerge = state.module.AllocateId();
        const auto firstSegment = first >> WindowSegmentShift;
        const auto lastSegment = (end - 1u) >> WindowSegmentShift;
        std::vector<std::uint32_t> labels(lastSegment - firstSegment + 1u);
        std::vector<std::uint32_t> words{spv::OpSwitch, segment, segmentsMerge};
        for (auto at = firstSegment; at <= lastSegment; ++at) {
            labels[at - firstSegment] = state.module.AllocateId();
            words.push_back(at);
            words.push_back(labels[at - firstSegment]);
        }
        state.module.AddFunction(spv::OpSelectionMerge, segmentsMerge, spv::SelectionControlMaskNone);
        state.module.AddFunction(std::span<const std::uint32_t>(words));
        std::vector<std::uint32_t> incoming;
        for (auto at = firstSegment; at <= lastSegment; ++at) {
            EmitLabel(state, labels[at - firstSegment]);
            const auto from = std::max(first, at << WindowSegmentShift);
            const auto to = std::min(end, (at + 1u) << WindowSegmentShift);
            incoming.push_back(AccessWindowSlots(state, index, from, to, access));
            incoming.push_back(state.currentLabel);
            state.module.AddFunction(spv::OpBranch, segmentsMerge);
        }
        incoming.push_back(ConstantU32(state, 0u));
        incoming.push_back(entry);
        EmitLabel(state, segmentsMerge);
        const auto value = state.module.AllocateId();
        std::vector<std::uint32_t> phi{spv::OpPhi, TypeU32(state), value};
        phi.insert(phi.end(), incoming.begin(), incoming.end());
        state.module.AddFunction(std::span<const std::uint32_t>(phi));
        return value;
    }
    const auto slotsMerge = state.module.AllocateId();
    std::vector<std::uint32_t> labels(end - first);
    std::vector<std::uint32_t> words{spv::OpSwitch, index, slotsMerge};
    for (std::uint32_t slot = first; slot < end; ++slot) {
        labels[slot - first] = state.module.AllocateId();
        words.push_back(slot);
        words.push_back(labels[slot - first]);
    }
    state.module.AddFunction(spv::OpSelectionMerge, slotsMerge, spv::SelectionControlMaskNone);
    state.module.AddFunction(std::span<const std::uint32_t>(words));
    std::vector<std::uint32_t> incoming;
    for (std::uint32_t slot = first; slot < end; ++slot) {
        EmitLabel(state, labels[slot - first]);
        incoming.push_back(access(WindowSlot(state, slot)));
        incoming.push_back(labels[slot - first]);
        state.module.AddFunction(spv::OpBranch, slotsMerge);
    }
    incoming.push_back(ConstantU32(state, 0u));
    incoming.push_back(entry);
    EmitLabel(state, slotsMerge);
    const auto value = state.module.AllocateId();
    std::vector<std::uint32_t> phi{spv::OpPhi, TypeU32(state), value};
    phi.insert(phi.end(), incoming.begin(), incoming.end());
    state.module.AddFunction(std::span<const std::uint32_t>(phi));
    return value;
}

std::uint32_t AccessNarrowedWindow(SpirvEmitterState& state, std::uint32_t index, const std::function<std::uint32_t(std::uint32_t)>& access) {
    const auto window = state.program.Metadata().vectorWindowSize;
    const auto arrayLabel = state.module.AllocateId();
    const auto slotsLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto inArray = Binary(state, spv::OpULessThan, TypeBool(state), index, ConstantU32(state, state.vectorWindowIndexed));
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, inArray, arrayLabel, slotsLabel);
    EmitLabel(state, arrayLabel);
    const auto fromArray = IndexedWindowInSlots(state) ? AccessWindowSlots(state, index, 0u, state.vectorWindowIndexed, access) : access(VectorWindowElement(state, index));
    const auto arrayExit = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, slotsLabel);
    const auto fromSlots = AccessWindowSlots(state, index, state.vectorWindowIndexed, window, access);
    const auto slotsExit = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, TypeU32(state), value, fromArray, arrayExit, fromSlots, slotsExit);
    return value;
}

std::uint32_t DirectWindowElement(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    const IrValue* index = inst.Argument(0)->Resolve();
    if (state.vectorWindowIndexed == 0u) return VectorWindowElement(state, ctx.Arg(inst, 0u));
    if (!index->HasImmediate()) return 0u;
    if (index->ImmediateU32() < state.vectorWindowIndexed && !IndexedWindowInSlots(state)) return VectorWindowElement(state, ctx.Arg(inst, 0u));
    return WindowSlot(state, index->ImmediateU32());
}

}

void EmitLoadVectorWindow(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    const auto load = [&state](std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
        return value;
    };
    const auto pointer = DirectWindowElement(ctx, inst);
    ctx.Define(inst, pointer != 0u ? load(pointer) : AccessNarrowedWindow(state, ctx.Arg(inst, 0u), load));
}

void EmitStoreVectorWindow(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    const auto value = ctx.Arg(inst, 1u);
    const auto store = [&state, value](std::uint32_t pointer) {
        state.module.AddFunction(spv::OpStore, pointer, value);
        return ConstantU32(state, 0u);
    };
    if (const auto pointer = DirectWindowElement(ctx, inst); pointer != 0u) (void)store(pointer);
    else (void)AccessNarrowedWindow(state, ctx.Arg(inst, 0u), store);
}

void EmitGetGotoVariable(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetGotoVariable(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetScc(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetScc(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetExec(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetExec(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetExecLo(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetExecLo(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetExecHi(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetExecHi(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetVcc(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetVcc(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetVccLo(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetVccLo(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetVccHi(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetVccHi(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitGetM0(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

void EmitSetM0(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

}
