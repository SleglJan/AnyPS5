#include "SpirvBackend/SpirvFlowEmitter.hpp"
#include "Optimization/FloatIdioms.hpp"
#include "PipelineSpecialization.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvEmitterInstructions.hpp"
#include "SpirvBackend/SpirvImageEmitter.hpp"
#include "SpirvBackend/SpirvMemory/SpirvSubgroup.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <span>
#include <string>
#include <spirv/unified1/spirv.hpp>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ShaderRecompiler {
namespace {

void EmitKillIfBoolFalse(SpirvEmitterState& state, std::uint32_t active) {
    const auto killLabel = state.module.AllocateId();
    const auto mergeLabel = state.module.AllocateId();
    const auto inactive = state.module.AllocateId();
    state.module.AddFunction(spv::OpLogicalNot, TypeBool(state), inactive, active);
    state.module.AddFunction(spv::OpSelectionMerge, mergeLabel, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, inactive, killLabel, mergeLabel);
    EmitLabel(state, killLabel);
    state.module.AddFunction(spv::OpKill);
    EmitLabel(state, mergeLabel);
}

void EmitKillIfPixelValidMaskInactive(SpirvEmitterState& state) {
    if (state.pixelValidMaskVariable == 0) {
        return;
    }
    const auto maskValue = state.module.AllocateId();
    const auto active = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), maskValue, state.pixelValidMaskVariable);
    state.module.AddFunction(spv::OpINotEqual, TypeBool(state), active, maskValue, ConstantU32(state, 0u));
    auto keep = active;
    if (const auto helperVariable = InputVariableForKind(state, StageInputKind::HelperInvocation); helperVariable != 0u) {
        const auto helper = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeBool(state), helper, helperVariable);
        keep = Binary(state, spv::OpLogicalOr, TypeBool(state), active, helper);
    }
    EmitKillIfBoolFalse(state, keep);
}

// Block metadata is paired with BlockOrder by position (as IrProgram validation does); terminator
// targets name control-flow ids, which are not IR block ids.
const IrBlock* TargetBlock(const IrProgram& program, std::uint32_t id) {
    const auto& blocks = program.BlockOrder();
    const auto& blockInfo = program.Metadata().blockInfo;
    if (blocks.size() != blockInfo.size()) {
        throw std::runtime_error("SPIR-V control flow block metadata is inconsistent");
    }
    for (std::size_t index = 0; index < blockInfo.size(); index++) {
        if (blockInfo[index].id == id) {
            if (blocks[index] == nullptr) {
                throw std::runtime_error("SPIR-V control flow target block is null");
            }
            return blocks[index];
        }
    }
    throw std::runtime_error("SPIR-V control flow target block is missing");
}

const BlockInfo* BlockInfoFor(const IrProgram& program, const IrBlock* block) {
    const auto& blocks = program.BlockOrder();
    const auto& blockInfo = program.Metadata().blockInfo;
    if (blocks.size() != blockInfo.size()) {
        throw std::runtime_error("SPIR-V control flow block metadata is inconsistent");
    }
    if (block == nullptr) {
        throw std::runtime_error("SPIR-V control flow block is null");
    }
    for (std::size_t index = 0; index < blocks.size(); index++) {
        if (blocks[index] == block) {
            return &blockInfo[index];
        }
    }
    throw std::runtime_error("SPIR-V control flow block has no metadata");
}

void EmitReturnTerminator(SpirvValueEmitContext& ctx) {
    auto& state = ctx.state;
    if (state.loopGuardLimit != 0 && state.faultBufferVariable != 0) {
        const auto pc = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), pc, state.loopGuardPc);
        EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), pc, ConstantU32(state, 0u)), [&] {
            const auto hash = state.program.Resources().shaderHash;
            RecordBdaFaultWords(state, ConstantU32(state, static_cast<std::uint32_t>(hash)), ConstantU32(state, static_cast<std::uint32_t>(hash >> 32u)), ConstantU32(state, state.loopGuardLimit), EmitBinaryU32(state, spv::OpISub, pc, ConstantU32(state, 1u)), BdaAbi::FaultReason::LoopLimit);
        });
    }
    if (state.calleeStopValue != 0u) {
        state.module.AddFunction(spv::OpReturnValue, state.calleeStopValue);
        return;
    }
    if (OrderedPixelShader(state)) {
        state.module.AddFunction(spv::OpEndInvocationInterlockEXT);
    }
    EmitKillIfPixelValidMaskInactive(state);
    if (state.stencilReplay) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeI32(state), value, state.stencilRefVariable);
        const auto reference = Unary(state, spv::OpBitcast, TypeU32(state), value);
        const auto masked = Binary(state, spv::OpBitwiseAnd, TypeU32(state), reference, ConstantU32(state, 0xffu));
        const auto offset = state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::StencilReferenceOffset, state.program.Metadata().bindings.stencilReferenceDword);
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer, state.pushConstantVariable, ConstantU32(state, 0u), offset);
        const auto expected = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), expected, pointer);
        EmitKillIfBoolFalse(state, Binary(state, spv::OpIEqual, TypeBool(state), masked, expected));
    }
    state.module.AddFunction(spv::OpReturn);
}

// The loop guard (see SpirvEmitterState::loopGuardLimit) at a branch that may leave a loop: counts the
// evaluation and, past the limit, takes the exit and remembers the block's PC (plus one, 0 = none).
std::uint32_t GuardLoopExit(SpirvEmitterState& state, const BlockInfo& info, std::uint32_t condition, bool exitWhenTrue) {
    const auto visits = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), visits, state.loopGuardVisits);
    const auto next = EmitBinaryU32(state, spv::OpIAdd, visits, ConstantU32(state, 1u));
    state.module.AddFunction(spv::OpStore, state.loopGuardVisits, next);
    const auto tripped = Binary(state, spv::OpUGreaterThan, TypeBool(state), next, ConstantU32(state, state.loopGuardLimit));
    const auto previous = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), previous, state.loopGuardPc);
    const auto first = Binary(state, spv::OpLogicalAnd, TypeBool(state), tripped, Binary(state, spv::OpIEqual, TypeBool(state), previous, ConstantU32(state, 0u)));
    const auto pc = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelect, TypeU32(state), pc, first, ConstantU32(state, info.startPc + 1u), previous);
    state.module.AddFunction(spv::OpStore, state.loopGuardPc, pc);
    if (exitWhenTrue) return Binary(state, spv::OpLogicalOr, TypeBool(state), condition, tripped);
    const auto stay = state.module.AllocateId();
    state.module.AddFunction(spv::OpLogicalNot, TypeBool(state), stay, tripped);
    return Binary(state, spv::OpLogicalAnd, TypeBool(state), condition, stay);
}

bool IsLoopMerge(const IrProgram& program, std::uint32_t block) {
    for (const auto& info : program.Metadata().blockInfo) {
        if (info.terminator.loopHeader && info.terminator.mergeBlock == block) return true;
    }
    return false;
}

bool IsContinueTarget(const IrProgram& program, std::uint32_t block) {
    for (const auto& info : program.Metadata().blockInfo) {
        if (info.terminator.loopHeader && info.terminator.continueBlock == block) return true;
    }
    return false;
}

std::uint32_t EmitWaveAny(SpirvEmitterState& state, std::uint32_t predicate) {
    const auto ballot = state.module.AllocateId();
    EmitSubgroupBallot(state, ballot, predicate);
    const auto wave = EmitWaveBallot(state, ballot);
    const auto low = state.module.AllocateId();
    const auto high = state.module.AllocateId();
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, wave, 0u);
    state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, wave, 1u);
    state.module.AddFunction(spv::OpINotEqual, TypeBool(state), result, EmitBinaryU32(state, spv::OpBitwiseOr, low, high), ConstantU32(state, 0u));
    return result;
}

std::uint32_t EmitWaveMaskBranchCondition(SpirvValueEmitContext& ctx, const BlockInfo& info) {
    auto& state = ctx.state;
    const auto kind = info.terminator.condition;
    if (kind == BranchCondition::ExecNonZero || kind == BranchCondition::VccNonZero) {
        return EmitWaveAny(state, ctx.Def(info.condition));
    }
    const auto clear = state.module.AllocateId();
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpLogicalNot, TypeBool(state), clear, ctx.Def(info.condition));
    state.module.AddFunction(spv::OpLogicalNot, TypeBool(state), result, EmitWaveAny(state, clear));
    return result;
}

std::uint32_t EmitBranchCondition(SpirvValueEmitContext& ctx, const BlockInfo& info) {
    const auto condition = info.terminator.condition;
    if (ctx.otherHalf == nullptr && IsWaveMaskBranch(condition)) {
        return EmitWaveMaskBranchCondition(ctx, info);
    }
    if (ctx.otherHalf == nullptr || condition == BranchCondition::ScalarInstruction || condition == BranchCondition::GotoVariable || condition == BranchCondition::IndirectTarget) {
        return ctx.Def(info.condition);
    }
    auto& state = ctx.state;
    const auto ballot = ctx.Ballot(info.condition);
    const auto low = state.module.AllocateId();
    const auto high = state.module.AllocateId();
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0u);
    state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1u);
    const auto kind = info.terminator.condition;
    const bool zero = kind == BranchCondition::ExecZero || kind == BranchCondition::VccZero || kind == BranchCondition::SccZero;
    const auto combined = EmitBinaryU32(state, zero ? spv::OpBitwiseAnd : spv::OpBitwiseOr, low, high);
    state.module.AddFunction(zero ? spv::OpIEqual : spv::OpINotEqual, TypeBool(state), result, combined, ConstantU32(state, zero ? ~0u : 0u));
    return result;
}

void EmitStructuredTerminator(SpirvValueEmitContext& ctx, const IrProgram& program, const BlockInfo& info) {
    auto& state = ctx.state;
    const Terminator& term = info.terminator;
    const auto emitMerge = [&]() {
        if (term.loopHeader) {
            const IrBlock* merge = TargetBlock(program, term.mergeBlock);
            const IrBlock* cont = TargetBlock(program, term.continueBlock);
            if (merge != nullptr && cont != nullptr) {
                state.module.AddFunction(spv::OpLoopMerge, ctx.Label(merge), ctx.Label(cont), spv::LoopControlMaskNone);
            }
        } else if (term.kind == TerminatorKind::ConditionalBranch && term.mergeBlock != InvalidControlFlowId) {
            if (const IrBlock* merge = TargetBlock(program, term.mergeBlock); merge != nullptr) {
                state.module.AddFunction(spv::OpSelectionMerge, ctx.Label(merge), spv::SelectionControlMaskNone);
            }
        }
    };
    switch (term.kind) {
        case TerminatorKind::Branch: {
            const IrBlock* target = TargetBlock(program, term.trueBlock);
            if (target == nullptr) {
                EmitReturnTerminator(ctx);
                return;
            }
            emitMerge();
            state.module.AddFunction(spv::OpBranch, ctx.Label(target));
            return;
        }
        case TerminatorKind::ConditionalBranch: {
            const IrBlock* trueBlock = TargetBlock(program, term.trueBlock);
            const IrBlock* falseBlock = TargetBlock(program, term.falseBlock);
            if (trueBlock == nullptr || falseBlock == nullptr || info.condition == nullptr) {
                EmitReturnTerminator(ctx);
                return;
            }
            if (const IrValue* resolved = info.condition->Resolve(); !resolved->HasImmediate() && !ctx.definitions.contains(resolved)) {
                const std::string where = resolved->Parent() == nullptr ? std::string("no block") : "block " + std::to_string(resolved->Parent()->Id());
                const std::string message = "branch condition " + std::string(IrOpcodeName(resolved->Opcode())) + " of block " + std::to_string(info.id) + " (defined in " + where + ") was not emitted before the branch";
                ctx.Fail(message.c_str());
            }
            if (ctx.Label(trueBlock) == ctx.Label(falseBlock)) {
                if (term.loopHeader) emitMerge();
                state.module.AddFunction(spv::OpBranch, ctx.Label(trueBlock));
                return;
            }
            auto condition = EmitBranchCondition(ctx, info);
            if (state.loopGuardLimit != 0 && term.trueBlock != term.falseBlock) {
                if (IsLoopMerge(program, term.trueBlock)) condition = GuardLoopExit(state, info, condition, true);
                else if (IsLoopMerge(program, term.falseBlock)) condition = GuardLoopExit(state, info, condition, false);
            }
            static const bool debug = std::getenv("APS5_SRT_DEBUG") != nullptr;
            if (debug) std::fprintf(stderr, "[spirv] block %u branches on %%%u = %s (kind %d)\n", static_cast<unsigned>(info.id), condition, std::string(IrOpcodeName(info.condition->Resolve()->Opcode())).c_str(), static_cast<int>(info.terminator.condition));
            emitMerge();
            state.module.AddFunction(spv::OpBranchConditional, condition, ctx.Label(trueBlock), ctx.Label(falseBlock));
            return;
        }
        default:
            EmitReturnTerminator(ctx);
            return;
    }
}

template<typename TArgument>
decltype(auto) DispatchArgument(SpirvValueEmitContext& ctx, const IrValue& inst, std::size_t index) {
    if constexpr (std::is_same_v<TArgument, const IrValue&>) {
        return inst;
    } else if constexpr (std::is_same_v<TArgument, const IrValue*>) {
        return static_cast<const IrValue*>(inst.Argument(index));
    } else if constexpr (std::is_same_v<TArgument, ScalarReg>) {
        return static_cast<ScalarReg>(inst.Argument(index)->Register().index);
    } else {
        static_assert(std::is_same_v<TArgument, std::uint32_t>);
        return ctx.Def(inst.Argument(index));
    }
}

template<typename TContext, typename TReturn, typename... TArguments>
void Invoke(TReturn (*emit)(TContext&, TArguments...), SpirvValueEmitContext& ctx, const IrValue& inst) {
    static_assert(std::is_same_v<TContext, SpirvValueEmitContext> || std::is_same_v<TContext, SpirvEmitterState>);
    auto& context = [&]() -> TContext& {
        if constexpr (std::is_same_v<TContext, SpirvEmitterState>) {
            return ctx.state;
        } else {
            return ctx;
        }
    }();
    constexpr bool hasInst = (std::is_same_v<TArguments, const IrValue&> || ...);
    [&]<std::size_t... TIndices>(std::index_sequence<TIndices...>) {
        static_assert(((!std::is_same_v<TArguments, const IrValue&> || TIndices == 0) && ...));
        const auto call = [&] {
            return emit(context, DispatchArgument<TArguments>(ctx, inst, TIndices - (hasInst && TIndices != 0 ? 1u : 0u))...);
        };
        if constexpr (std::is_void_v<TReturn>) {
            call();
        } else {
            static_assert(std::is_same_v<TReturn, std::uint32_t>);
            ctx.Define(inst, call());
        }
    }(std::index_sequence_for<TArguments...>{});
}

const IrValue* FloatNanTestSource(const SpirvValueEmitContext& ctx, const IrValue& test) {
    const IrValue* bits = NanTestedF32Source(test);
    if (bits == nullptr) return nullptr;
    bits = bits->Resolve();
    if (bits->Opcode() == IrOpcode::BitCastU32F32 && bits->ArgumentCount() == 1u) {
        const IrValue* value = bits->Argument(0)->Resolve();
        if (value->Type() == IrType::F32 && !value->HasImmediate()) return value;
    }
    for (const IrValue* view : bits->Uses()) {
        if (view->Opcode() == IrOpcode::BitCastF32U32 && view->Parent() == test.Parent() && ctx.definitions.contains(view)) return view;
    }
    return nullptr;
}

bool OnlyFloatNanTests(const SpirvValueEmitContext& ctx, const IrValue& mask) {
    if (MaskedOperand(mask, IrOpcode::BitwiseAnd32, 0x7fffffffu) == nullptr || !mask.HasUses()) return false;
    for (const IrValue* use : mask.Uses()) {
        if (use->ArgumentCount() == 0u || use->Argument(0)->Resolve() != &mask || FloatNanTestSource(ctx, *use) == nullptr) return false;
    }
    return true;
}

void EmitDirectInstruction(SpirvValueEmitContext& ctx, const IrValue& inst) {
    switch (inst.Opcode()) {
        case IrOpcode::Identity: return Invoke(EmitIdentity, ctx, inst);
        case IrOpcode::Phi: return Invoke(EmitPhi, ctx, inst);
        case IrOpcode::IAdd32: return Invoke(EmitIAdd32, ctx, inst);
        case IrOpcode::ISub32: return Invoke(EmitISub32, ctx, inst);
        case IrOpcode::IMul32: return Invoke(EmitIMul32, ctx, inst);
        case IrOpcode::Unreachable: return Invoke(EmitUnreachable, ctx, inst);
        case IrOpcode::Barrier: return Invoke(EmitBarrier, ctx, inst);
        case IrOpcode::ReadFirstLane: return Invoke(EmitReadFirstLane, ctx, inst);
        case IrOpcode::ReadLane: return Invoke(EmitReadLane, ctx, inst);
        case IrOpcode::Ballot: return Invoke(EmitBallot, ctx, inst);
        case IrOpcode::Void: return Invoke(EmitVoid, ctx, inst);
        case IrOpcode::Reference: return Invoke(EmitReference, ctx, inst);
        case IrOpcode::ReferenceU32: return Invoke(EmitReferenceU32, ctx, inst);
        case IrOpcode::UnlinkedCall: return Invoke(EmitUnlinkedCall, ctx, inst);
        case IrOpcode::GetUserData: return Invoke(EmitGetUserData, ctx, inst);
        case IrOpcode::GetLinkedWord: return Invoke(EmitVoid, ctx, inst);
        case IrOpcode::GetShaderBase: return Invoke(EmitGetShaderBase, ctx, inst);
        case IrOpcode::ShaderClock: return Invoke(EmitShaderClock, ctx, inst);
        case IrOpcode::RealtimeClock: return Invoke(EmitRealtimeClock, ctx, inst);
        case IrOpcode::MeshDrawParameter: return Invoke(EmitMeshDrawParameter, ctx, inst);
        case IrOpcode::MeshArgument: return Invoke(EmitMeshArgument, ctx, inst);
        case IrOpcode::MeshRestartStart: return Invoke(EmitMeshRestartStart, ctx, inst);
        case IrOpcode::MeshAllocate: return Invoke(EmitMeshAllocate, ctx, inst);
        case IrOpcode::TessellationBase: return Invoke(EmitTessellationBase, ctx, inst);
        case IrOpcode::GetTessellationAttribute: return Invoke(EmitGetTessellationAttribute, ctx, inst);
        case IrOpcode::SetTessellationAttribute: return Invoke(EmitSetTessellationAttribute, ctx, inst);
        case IrOpcode::GetBuiltin: return Invoke(EmitGetBuiltin, ctx, inst);
        case IrOpcode::GetThreadBitScalarRegister: return Invoke(EmitGetThreadBitScalarRegister, ctx, inst);
        case IrOpcode::SetThreadBitScalarRegister: return Invoke(EmitSetThreadBitScalarRegister, ctx, inst);
        case IrOpcode::GetScalarMaskTag: return Invoke(EmitGetScalarMaskTag, ctx, inst);
        case IrOpcode::SetScalarMaskTag: return Invoke(EmitSetScalarMaskTag, ctx, inst);
        case IrOpcode::GetScalarRegister: return Invoke(EmitGetScalarRegister, ctx, inst);
        case IrOpcode::SetScalarRegister: return Invoke(EmitSetScalarRegister, ctx, inst);
        case IrOpcode::GetVectorRegister: return Invoke(EmitGetVectorRegister, ctx, inst);
        case IrOpcode::SetVectorRegister: return Invoke(EmitSetVectorRegister, ctx, inst);
        case IrOpcode::LoadVectorWindow: return Invoke(EmitLoadVectorWindow, ctx, inst);
        case IrOpcode::StoreVectorWindow: return Invoke(EmitStoreVectorWindow, ctx, inst);
        case IrOpcode::GetGotoVariable: return Invoke(EmitGetGotoVariable, ctx, inst);
        case IrOpcode::SetGotoVariable: return Invoke(EmitSetGotoVariable, ctx, inst);
        case IrOpcode::GetScc: return Invoke(EmitGetScc, ctx, inst);
        case IrOpcode::SetScc: return Invoke(EmitSetScc, ctx, inst);
        case IrOpcode::GetExec: return Invoke(EmitGetExec, ctx, inst);
        case IrOpcode::SetExec: return Invoke(EmitSetExec, ctx, inst);
        case IrOpcode::GetExecLo: return Invoke(EmitGetExecLo, ctx, inst);
        case IrOpcode::SetExecLo: return Invoke(EmitSetExecLo, ctx, inst);
        case IrOpcode::GetExecHi: return Invoke(EmitGetExecHi, ctx, inst);
        case IrOpcode::SetExecHi: return Invoke(EmitSetExecHi, ctx, inst);
        case IrOpcode::GetVcc: return Invoke(EmitGetVcc, ctx, inst);
        case IrOpcode::SetVcc: return Invoke(EmitSetVcc, ctx, inst);
        case IrOpcode::GetVccLo: return Invoke(EmitGetVccLo, ctx, inst);
        case IrOpcode::SetVccLo: return Invoke(EmitSetVccLo, ctx, inst);
        case IrOpcode::GetVccHi: return Invoke(EmitGetVccHi, ctx, inst);
        case IrOpcode::SetVccHi: return Invoke(EmitSetVccHi, ctx, inst);
        case IrOpcode::GetM0: return Invoke(EmitGetM0, ctx, inst);
        case IrOpcode::SetM0: return Invoke(EmitSetM0, ctx, inst);
        case IrOpcode::UndefU1: return Invoke(EmitUndefU1, ctx, inst);
        case IrOpcode::UndefU8: return Invoke(EmitUndefU8, ctx, inst);
        case IrOpcode::UndefU16: return Invoke(EmitUndefU16, ctx, inst);
        case IrOpcode::UndefU32: return Invoke(EmitUndefU32, ctx, inst);
        case IrOpcode::UndefU64: return Invoke(EmitUndefU64, ctx, inst);
        case IrOpcode::BitCastU16F16: return Invoke(EmitBitCastU16F16, ctx, inst);
        case IrOpcode::BitCastF16U16: return Invoke(EmitBitCastF16U16, ctx, inst);
        case IrOpcode::BitCastU32F32: return Invoke(EmitBitCastU32F32, ctx, inst);
        case IrOpcode::BitCastF32U32: return Invoke(EmitBitCastF32U32, ctx, inst);
        case IrOpcode::ConvertU16U32: return Invoke(EmitConvertU16U32, ctx, inst);
        case IrOpcode::ConvertU32U16: return Invoke(EmitConvertU32U16, ctx, inst);
        case IrOpcode::ConvertU8U32: return Invoke(EmitConvertU8U32, ctx, inst);
        case IrOpcode::ConvertU32U8: return Invoke(EmitConvertU32U8, ctx, inst);
        case IrOpcode::ConvertF32F16: return Invoke(EmitConvertF32F16, ctx, inst);
        case IrOpcode::ConvertF16F32: return Invoke(EmitConvertF16F32, ctx, inst);
        case IrOpcode::ConvertS32F32: return Invoke(EmitConvertS32F32, ctx, inst);
        case IrOpcode::ConvertU32F32: return Invoke(EmitConvertU32F32, ctx, inst);
        case IrOpcode::ConvertF32S32: return Invoke(EmitConvertF32S32, ctx, inst);
        case IrOpcode::ConvertF32U32: return Invoke(EmitConvertF32U32, ctx, inst);
        case IrOpcode::CompositeConstructU64: return Invoke(EmitCompositeConstructU64, ctx, inst);
        case IrOpcode::CompositeConstructU32x2: return Invoke(EmitCompositeConstructU32x2, ctx, inst);
        case IrOpcode::CompositeConstructU32x3: return Invoke(EmitCompositeConstructU32x3, ctx, inst);
        case IrOpcode::CompositeConstructF32x2: return Invoke(EmitCompositeConstructF32x2, ctx, inst);
        case IrOpcode::CompositeConstructU32x4: return Invoke(EmitCompositeConstructU32x4, ctx, inst);
        case IrOpcode::CompositeExtractU64: return Invoke(EmitCompositeExtractU64, ctx, inst);
        case IrOpcode::CompositeExtractU32x2: return Invoke(EmitCompositeExtractU32x2, ctx, inst);
        case IrOpcode::CompositeExtractU32x3: return Invoke(EmitCompositeExtractU32x3, ctx, inst);
        case IrOpcode::CompositeExtractU32x4: return Invoke(EmitCompositeExtractU32x4, ctx, inst);
        case IrOpcode::PackHalf2x16: return Invoke(EmitPackHalf2x16, ctx, inst);
        case IrOpcode::PackFloat2x16Rtz: return Invoke(EmitPackFloat2x16Rtz, ctx, inst);
        case IrOpcode::FPAdd16x2: return Invoke(EmitFPAdd16x2, ctx, inst);
        case IrOpcode::FPMul16x2: return Invoke(EmitFPMul16x2, ctx, inst);
        case IrOpcode::FPAbs32: return Invoke(EmitFPAbs32, ctx, inst);
        case IrOpcode::FPNeg32: return Invoke(EmitFPNeg32, ctx, inst);
        case IrOpcode::FPSaturate32: return Invoke(EmitFPSaturate32, ctx, inst);
        case IrOpcode::BitFieldInsert: return Invoke(EmitBitFieldInsert, ctx, inst);
        case IrOpcode::BitFieldUExtract: return Invoke(EmitBitFieldUExtract, ctx, inst);
        case IrOpcode::BitFieldSExtract: return Invoke(EmitBitFieldSExtract, ctx, inst);
        case IrOpcode::DppMoveU32: return Invoke(EmitDppMoveU32, ctx, inst);
        case IrOpcode::DppUpdateU32: return Invoke(EmitDppUpdateU32, ctx, inst);
        case IrOpcode::WqmU64: return Invoke(EmitWqmU64, ctx, inst);
        case IrOpcode::SelectU1: return Invoke(EmitSelectU1, ctx, inst);
        case IrOpcode::SelectF32: return Invoke(EmitSelectF32, ctx, inst);
        case IrOpcode::IAdd64: return Invoke(EmitIAdd64, ctx, inst);
        case IrOpcode::IAddCarry32: return Invoke(EmitIAddCarry32, ctx, inst);
        case IrOpcode::ISub64: return Invoke(EmitISub64, ctx, inst);
        case IrOpcode::IMul64: return Invoke(EmitIMul64, ctx, inst);
        case IrOpcode::UDiv32: return Invoke(EmitUDiv32, ctx, inst);
        case IrOpcode::SMulHi: return Invoke(EmitSMulHi, ctx, inst);
        case IrOpcode::UMulHi: return Invoke(EmitUMulHi, ctx, inst);
        case IrOpcode::IAbs32: return Invoke(EmitIAbs32, ctx, inst);
        case IrOpcode::ShiftLeftLogical32: return Invoke(EmitShiftLeftLogical32, ctx, inst);
        case IrOpcode::ShiftLeftLogical64: return Invoke(EmitShiftLeftLogical64, ctx, inst);
        case IrOpcode::ShiftRightLogical32: return Invoke(EmitShiftRightLogical32, ctx, inst);
        case IrOpcode::ShiftRightLogical64: return Invoke(EmitShiftRightLogical64, ctx, inst);
        case IrOpcode::ShiftRightArithmetic32: return Invoke(EmitShiftRightArithmetic32, ctx, inst);
        case IrOpcode::ShiftRightArithmetic64: return Invoke(EmitShiftRightArithmetic64, ctx, inst);
        case IrOpcode::BitwiseAnd32:
            if (OnlyFloatNanTests(ctx, inst)) return;
            return Invoke(EmitBitwiseAnd32, ctx, inst);
        case IrOpcode::BitwiseAnd64: return Invoke(EmitBitwiseAnd64, ctx, inst);
        case IrOpcode::BitwiseOr32: return Invoke(EmitBitwiseOr32, ctx, inst);
        case IrOpcode::BitwiseXor32: return Invoke(EmitBitwiseXor32, ctx, inst);
        case IrOpcode::BitwiseNot32: return Invoke(EmitBitwiseNot32, ctx, inst);
        case IrOpcode::BitReverse32: return Invoke(EmitBitReverse32, ctx, inst);
        case IrOpcode::BitCount32: return Invoke(EmitBitCount32, ctx, inst);
        case IrOpcode::BitCount64: return Invoke(EmitBitCount64, ctx, inst);
        case IrOpcode::FindUMsb32: return Invoke(EmitFindUMsb32, ctx, inst);
        case IrOpcode::FindUMsb64: return Invoke(EmitFindUMsb64, ctx, inst);
        case IrOpcode::FindILsb32: return Invoke(EmitFindILsb32, ctx, inst);
        case IrOpcode::SMin32: return Invoke(EmitSMin32, ctx, inst);
        case IrOpcode::UMin32: return Invoke(EmitUMin32, ctx, inst);
        case IrOpcode::SMax32: return Invoke(EmitSMax32, ctx, inst);
        case IrOpcode::UMax32: return Invoke(EmitUMax32, ctx, inst);
        case IrOpcode::SMinTri32: return Invoke(EmitSMinTri32, ctx, inst);
        case IrOpcode::UMinTri32: return Invoke(EmitUMinTri32, ctx, inst);
        case IrOpcode::SMaxTri32: return Invoke(EmitSMaxTri32, ctx, inst);
        case IrOpcode::UMaxTri32: return Invoke(EmitUMaxTri32, ctx, inst);
        case IrOpcode::SMedTri32: return Invoke(EmitSMedTri32, ctx, inst);
        case IrOpcode::UMedTri32: return Invoke(EmitUMedTri32, ctx, inst);
        case IrOpcode::SLessThan32: return Invoke(EmitSLessThan32, ctx, inst);
        case IrOpcode::SLessThan64: return Invoke(EmitSLessThan64, ctx, inst);
        case IrOpcode::ULessThan32: return Invoke(EmitULessThan32, ctx, inst);
        case IrOpcode::ULessThan64: return Invoke(EmitULessThan64, ctx, inst);
        case IrOpcode::IEqual32: return Invoke(EmitIEqual32, ctx, inst);
        case IrOpcode::IEqual64: return Invoke(EmitIEqual64, ctx, inst);
        case IrOpcode::SLessThanEqual32: return Invoke(EmitSLessThanEqual32, ctx, inst);
        case IrOpcode::ULessThanEqual32: return Invoke(EmitULessThanEqual32, ctx, inst);
        case IrOpcode::SGreaterThan32: return Invoke(EmitSGreaterThan32, ctx, inst);
        case IrOpcode::UGreaterThan32:
            if (const IrValue* value = FloatNanTestSource(ctx, inst)) {
                ctx.Define(inst, EmitNative<spv::OpIsNan, IrType::U1>(ctx.state, ctx.Def(value)));
                return;
            }
            return Invoke(EmitUGreaterThan32, ctx, inst);
        case IrOpcode::UGreaterThan64: return Invoke(EmitUGreaterThan64, ctx, inst);
        case IrOpcode::INotEqual32: return Invoke(EmitINotEqual32, ctx, inst);
        case IrOpcode::INotEqual64: return Invoke(EmitINotEqual64, ctx, inst);
        case IrOpcode::SGreaterThanEqual32: return Invoke(EmitSGreaterThanEqual32, ctx, inst);
        case IrOpcode::UGreaterThanEqual32: return Invoke(EmitUGreaterThanEqual32, ctx, inst);
        case IrOpcode::LogicalOr: return Invoke(EmitLogicalOr, ctx, inst);
        case IrOpcode::LogicalAnd: return Invoke(EmitLogicalAnd, ctx, inst);
        case IrOpcode::LogicalXor: return Invoke(EmitLogicalXor, ctx, inst);
        case IrOpcode::LogicalNot: return Invoke(EmitLogicalNot, ctx, inst);
        case IrOpcode::FPOrdEqual32: return Invoke(EmitFPOrdEqual32, ctx, inst);
        case IrOpcode::FPUnordEqual32: return Invoke(EmitFPUnordEqual32, ctx, inst);
        case IrOpcode::FPOrdNotEqual32: return Invoke(EmitFPOrdNotEqual32, ctx, inst);
        case IrOpcode::FPUnordNotEqual32: return Invoke(EmitFPUnordNotEqual32, ctx, inst);
        case IrOpcode::FPOrdLessThan32: return Invoke(EmitFPOrdLessThan32, ctx, inst);
        case IrOpcode::FPUnordLessThan32: return Invoke(EmitFPUnordLessThan32, ctx, inst);
        case IrOpcode::FPOrdGreaterThan32: return Invoke(EmitFPOrdGreaterThan32, ctx, inst);
        case IrOpcode::FPUnordGreaterThan32: return Invoke(EmitFPUnordGreaterThan32, ctx, inst);
        case IrOpcode::FPOrdLessThanEqual32: return Invoke(EmitFPOrdLessThanEqual32, ctx, inst);
        case IrOpcode::FPUnordLessThanEqual32: return Invoke(EmitFPUnordLessThanEqual32, ctx, inst);
        case IrOpcode::FPOrdGreaterThanEqual32: return Invoke(EmitFPOrdGreaterThanEqual32, ctx, inst);
        case IrOpcode::FPUnordGreaterThanEqual32: return Invoke(EmitFPUnordGreaterThanEqual32, ctx, inst);
        case IrOpcode::FPIsNan32: return Invoke(EmitFPIsNan32, ctx, inst);
        case IrOpcode::FPCmpClass32: return Invoke(EmitFPCmpClass32, ctx, inst);
        case IrOpcode::FPAdd32: return Invoke(EmitFPAdd32, ctx, inst);
        case IrOpcode::FPSub32: return Invoke(EmitFPSub32, ctx, inst);
        case IrOpcode::FPFma32: return Invoke(EmitFPFma32, ctx, inst);
        case IrOpcode::FPMad32: return Invoke(EmitFPMad32, ctx, inst);
        case IrOpcode::FPMul32: return Invoke(EmitFPMul32, ctx, inst);
        case IrOpcode::FPMin32: return Invoke(EmitFPMin32, ctx, inst);
        case IrOpcode::FPMax32: return Invoke(EmitFPMax32, ctx, inst);
        case IrOpcode::FPMinTri32: return Invoke(EmitFPMinTri32, ctx, inst);
        case IrOpcode::FPMaxTri32: return Invoke(EmitFPMaxTri32, ctx, inst);
        case IrOpcode::FPMedTri32: return Invoke(EmitFPMedTri32, ctx, inst);
        case IrOpcode::FPRecip32: return Invoke(EmitFPRecip32, ctx, inst);
        case IrOpcode::FPRecipIFlag32: return Invoke(EmitFPRecipIFlag32, ctx, inst);
        case IrOpcode::FPRecipSqrt32: return Invoke(EmitFPRecipSqrt32, ctx, inst);
        case IrOpcode::FPSqrt: return Invoke(EmitFPSqrt, ctx, inst);
        case IrOpcode::FPSin: return Invoke(EmitFPSin, ctx, inst);
        case IrOpcode::FPCos: return Invoke(EmitFPCos, ctx, inst);
        case IrOpcode::FPExp2: return Invoke(EmitFPExp2, ctx, inst);
        case IrOpcode::FPLog2: return Invoke(EmitFPLog2, ctx, inst);
        case IrOpcode::FPRoundEven32: return Invoke(EmitFPRoundEven32, ctx, inst);
        case IrOpcode::FPFloor32: return Invoke(EmitFPFloor32, ctx, inst);
        case IrOpcode::FPCeil32: return Invoke(EmitFPCeil32, ctx, inst);
        case IrOpcode::FPTrunc32: return Invoke(EmitFPTrunc32, ctx, inst);
        case IrOpcode::FPFract32: return Invoke(EmitFPFract32, ctx, inst);
        case IrOpcode::FPAdd64: return Invoke(EmitFPAdd64, ctx, inst);
        case IrOpcode::FPMul64: return Invoke(EmitFPMul64, ctx, inst);
        case IrOpcode::FPFma64: return Invoke(EmitFPFma64, ctx, inst);
        case IrOpcode::FPFmaScale64: return Invoke(EmitFPFmaScale64, ctx, inst);
        case IrOpcode::FPMin64: return Invoke(EmitFPMin64, ctx, inst);
        case IrOpcode::FPMax64: return Invoke(EmitFPMax64, ctx, inst);
        case IrOpcode::FPSaturate64: return Invoke(EmitFPSaturate64, ctx, inst);
        case IrOpcode::FPLdexp64: return Invoke(EmitFPLdexp64, ctx, inst);
        case IrOpcode::FPRoundEven64: return Invoke(EmitFPRoundEven64, ctx, inst);
        case IrOpcode::FPFloor64: return Invoke(EmitFPFloor64, ctx, inst);
        case IrOpcode::FPCeil64: return Invoke(EmitFPCeil64, ctx, inst);
        case IrOpcode::FPTrunc64: return Invoke(EmitFPTrunc64, ctx, inst);
        case IrOpcode::FPFract64: return Invoke(EmitFPFract64, ctx, inst);
        case IrOpcode::FPFrexpMant64: return Invoke(EmitFPFrexpMant64, ctx, inst);
        case IrOpcode::FPFrexpExp64: return Invoke(EmitFPFrexpExp64, ctx, inst);
        case IrOpcode::FPRcp64: return Invoke(EmitFPRcp64, ctx, inst);
        case IrOpcode::FPRsq64: return Invoke(EmitFPRsq64, ctx, inst);
        case IrOpcode::FPSqrt64: return Invoke(EmitFPSqrt64, ctx, inst);
        case IrOpcode::FPTrigPreop64: return Invoke(EmitFPTrigPreop64, ctx, inst);
        case IrOpcode::FPDot2F32F16: return Invoke(EmitFPDot2F32F16, ctx, inst);
        case IrOpcode::FPInterpolateF32: return Invoke(EmitFPInterpolateF32, ctx, inst);
        case IrOpcode::FPInterpolateF16: return Invoke(EmitFPInterpolateF16, ctx, inst);
        case IrOpcode::ConvertF32F64: return Invoke(EmitConvertF32F64, ctx, inst);
        case IrOpcode::ConvertF64F32: return Invoke(EmitConvertF64F32, ctx, inst);
        case IrOpcode::ConvertF64S32: return Invoke(EmitConvertF64S32, ctx, inst);
        case IrOpcode::ConvertF64U32: return Invoke(EmitConvertF64U32, ctx, inst);
        case IrOpcode::ConvertS32F64: return Invoke(EmitConvertS32F64, ctx, inst);
        case IrOpcode::ConvertU32F64: return Invoke(EmitConvertU32F64, ctx, inst);
        case IrOpcode::LaneId: return Invoke(EmitLaneId, ctx, inst);
        case IrOpcode::WriteLane: return Invoke(EmitWriteLane, ctx, inst);
        case IrOpcode::Permlane16U32: return Invoke(EmitPermlane16U32, ctx, inst);
        case IrOpcode::BpermuteU32: return Invoke(EmitBpermuteU32, ctx, inst);
        case IrOpcode::PermuteU32: return Invoke(EmitPermuteU32, ctx, inst);
        case IrOpcode::GetSrtResource: return Invoke(EmitGetSrtResource, ctx, inst);
        case IrOpcode::GetBufferResource: return Invoke(EmitGetBufferResource, ctx, inst);
        case IrOpcode::GetAddressResource: return Invoke(EmitGetAddressResource, ctx, inst);
        case IrOpcode::GetScratchResource: return Invoke(EmitGetScratchResource, ctx, inst);
        case IrOpcode::GetImageResource: return Invoke(EmitGetImageResource, ctx, inst);
        case IrOpcode::GetSamplerResource: return Invoke(EmitGetSamplerResource, ctx, inst);
        case IrOpcode::MakeImageAddress: return Invoke(EmitMakeImageAddress, ctx, inst);
        case IrOpcode::ReadConst: return Invoke(EmitReadConst, ctx, inst);
        case IrOpcode::ReadConstBuffer: return Invoke(EmitReadConstBuffer, ctx, inst);
        case IrOpcode::LoadAddressU8: return Invoke(EmitLoadAddressU8, ctx, inst);
        case IrOpcode::LoadAddressU16: return Invoke(EmitLoadAddressU16, ctx, inst);
        case IrOpcode::LoadAddressU32: return Invoke(EmitLoadAddressU32, ctx, inst);
        case IrOpcode::LoadAddressU32x2: return Invoke(EmitLoadAddressU32x2, ctx, inst);
        case IrOpcode::LoadAddressU32x3: return Invoke(EmitLoadAddressU32x3, ctx, inst);
        case IrOpcode::LoadAddressU32x4: return Invoke(EmitLoadAddressU32x4, ctx, inst);
        case IrOpcode::StoreAddressU8: return Invoke(EmitStoreAddressU8, ctx, inst);
        case IrOpcode::StoreAddressU16: return Invoke(EmitStoreAddressU16, ctx, inst);
        case IrOpcode::StoreAddressU32: return Invoke(EmitStoreAddressU32, ctx, inst);
        case IrOpcode::StoreAddressU32x2: return Invoke(EmitStoreAddressU32x2, ctx, inst);
        case IrOpcode::StoreAddressU32x3: return Invoke(EmitStoreAddressU32x3, ctx, inst);
        case IrOpcode::StoreAddressU32x4: return Invoke(EmitStoreAddressU32x4, ctx, inst);
        case IrOpcode::AddressAtomicSwap32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicCmpSwap32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicIAdd32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicISub32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicSMin32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicUMin32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicSMax32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicUMax32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicAnd32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicOr32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicXor32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicInc32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicDec32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicUSubSat32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicSwap64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicCmpSwap64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicIAdd64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicISub64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicSMin64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicUMin64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicSMax64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicUMax64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicAnd64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicOr64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicXor64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicFCmpSwap32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicFMin32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicFMax32: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicFCmpSwap64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicFMin64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicFMax64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicInc64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::AddressAtomicDec64: return Invoke(EmitAddressAtomic, ctx, inst);
        case IrOpcode::LoadBufferU8: return Invoke(EmitLoadBufferU8, ctx, inst);
        case IrOpcode::LoadBufferU16: return Invoke(EmitLoadBufferU16, ctx, inst);
        case IrOpcode::LoadBufferU32: return Invoke(EmitLoadBufferU32, ctx, inst);
        case IrOpcode::LoadBufferU32x2: return Invoke(EmitLoadBufferU32x2, ctx, inst);
        case IrOpcode::LoadBufferU32x3: return Invoke(EmitLoadBufferU32x3, ctx, inst);
        case IrOpcode::LoadBufferU32x4: return Invoke(EmitLoadBufferU32x4, ctx, inst);
        case IrOpcode::StoreBufferU8: return Invoke(EmitStoreBufferU8, ctx, inst);
        case IrOpcode::StoreBufferU16: return Invoke(EmitStoreBufferU16, ctx, inst);
        case IrOpcode::StoreBufferU32: return Invoke(EmitStoreBufferU32, ctx, inst);
        case IrOpcode::StoreBufferU32x2: return Invoke(EmitStoreBufferU32x2, ctx, inst);
        case IrOpcode::StoreBufferU32x3: return Invoke(EmitStoreBufferU32x3, ctx, inst);
        case IrOpcode::StoreBufferU32x4: return Invoke(EmitStoreBufferU32x4, ctx, inst);
        case IrOpcode::BufferAtomicSwap32: return Invoke(EmitBufferAtomicSwap32, ctx, inst);
        case IrOpcode::BufferAtomicCmpSwap32: return Invoke(EmitBufferAtomicCmpSwap32, ctx, inst);
        case IrOpcode::BufferAtomicSwap64: return Invoke(EmitBufferAtomicSwap64, ctx, inst);
        case IrOpcode::BufferAtomicIAdd32: return Invoke(EmitBufferAtomicIAdd32, ctx, inst);
        case IrOpcode::BufferAtomicISub32: return Invoke(EmitBufferAtomicISub32, ctx, inst);
        case IrOpcode::BufferAtomicSMin32: return Invoke(EmitBufferAtomicSMin32, ctx, inst);
        case IrOpcode::BufferAtomicUMin32: return Invoke(EmitBufferAtomicUMin32, ctx, inst);
        case IrOpcode::BufferAtomicSMax32: return Invoke(EmitBufferAtomicSMax32, ctx, inst);
        case IrOpcode::BufferAtomicUMax32: return Invoke(EmitBufferAtomicUMax32, ctx, inst);
        case IrOpcode::BufferAtomicAnd32: return Invoke(EmitBufferAtomicAnd32, ctx, inst);
        case IrOpcode::BufferAtomicOr32: return Invoke(EmitBufferAtomicOr32, ctx, inst);
        case IrOpcode::BufferAtomicOr64: return Invoke(EmitBufferAtomicOr64, ctx, inst);
        case IrOpcode::BufferAtomicInc32: return Invoke(EmitBufferAtomicInc32, ctx, inst);
        case IrOpcode::BufferAtomicDec32: return Invoke(EmitBufferAtomicDec32, ctx, inst);
        case IrOpcode::BufferAtomicUSubSat32: return Invoke(EmitBufferAtomicUSubSat32, ctx, inst);
        case IrOpcode::BufferAtomicIAdd64: return Invoke(EmitBufferAtomicIAdd64, ctx, inst);
        case IrOpcode::BufferAtomicISub64: return Invoke(EmitBufferAtomicISub64, ctx, inst);
        case IrOpcode::BufferAtomicSMin64: return Invoke(EmitBufferAtomicSMin64, ctx, inst);
        case IrOpcode::BufferAtomicUMin64: return Invoke(EmitBufferAtomicUMin64, ctx, inst);
        case IrOpcode::BufferAtomicSMax64: return Invoke(EmitBufferAtomicSMax64, ctx, inst);
        case IrOpcode::BufferAtomicUMax64: return Invoke(EmitBufferAtomicUMax64, ctx, inst);
        case IrOpcode::BufferAtomicAnd64: return Invoke(EmitBufferAtomicAnd64, ctx, inst);
        case IrOpcode::BufferAtomicXor64: return Invoke(EmitBufferAtomicXor64, ctx, inst);
        case IrOpcode::BufferAtomicCmpSwap64: return Invoke(EmitBufferAtomicCmpSwap64, ctx, inst);
        case IrOpcode::BufferAtomicFCmpSwap32: return Invoke(EmitBufferAtomicFCmpSwap32, ctx, inst);
        case IrOpcode::BufferAtomicFCmpSwap64: return Invoke(EmitBufferAtomicFCmpSwap64, ctx, inst);
        case IrOpcode::BufferAtomicFMin64: return Invoke(EmitBufferAtomicFMin64, ctx, inst);
        case IrOpcode::BufferAtomicFMax64: return Invoke(EmitBufferAtomicFMax64, ctx, inst);
        case IrOpcode::BufferAtomicInc64: return Invoke(EmitBufferAtomicInc64, ctx, inst);
        case IrOpcode::BufferAtomicDec64: return Invoke(EmitBufferAtomicDec64, ctx, inst);
        case IrOpcode::BufferAtomicXor32: return Invoke(EmitBufferAtomicXor32, ctx, inst);
        case IrOpcode::BufferAtomicFMin32: return Invoke(EmitBufferAtomicFMin32, ctx, inst);
        case IrOpcode::BufferAtomicFMax32: return Invoke(EmitBufferAtomicFMax32, ctx, inst);
        case IrOpcode::LoadSharedU8: return Invoke(EmitLoadSharedU8, ctx, inst);
        case IrOpcode::LoadSharedU16: return Invoke(EmitLoadSharedU16, ctx, inst);
        case IrOpcode::LoadSharedU32: return Invoke(EmitLoadSharedU32, ctx, inst);
        case IrOpcode::LoadSharedU32x2: return Invoke(EmitLoadSharedU32x2, ctx, inst);
        case IrOpcode::LoadSharedU32x3: return Invoke(EmitLoadSharedU32x3, ctx, inst);
        case IrOpcode::LoadSharedU32x4: return Invoke(EmitLoadSharedU32x4, ctx, inst);
        case IrOpcode::WriteSharedU8: return Invoke(EmitWriteSharedU8, ctx, inst);
        case IrOpcode::WriteSharedU16: return Invoke(EmitWriteSharedU16, ctx, inst);
        case IrOpcode::WriteSharedU32: return Invoke(EmitWriteSharedU32, ctx, inst);
        case IrOpcode::WriteSharedU32x2: return Invoke(EmitWriteSharedU32x2, ctx, inst);
        case IrOpcode::WriteSharedU32x3: return Invoke(EmitWriteSharedU32x3, ctx, inst);
        case IrOpcode::WriteSharedU32x4: return Invoke(EmitWriteSharedU32x4, ctx, inst);
        case IrOpcode::SharedAtomicFMin32: return Invoke(EmitSharedAtomicFMin32, ctx, inst);
        case IrOpcode::SharedAtomicFMax32: return Invoke(EmitSharedAtomicFMax32, ctx, inst);
        case IrOpcode::SharedAtomicSwap32: return Invoke(EmitSharedAtomicSwap32, ctx, inst);
        case IrOpcode::SharedAtomicIAdd32: return Invoke(EmitSharedAtomicIAdd32, ctx, inst);
        case IrOpcode::SharedAtomicISub32: return Invoke(EmitSharedAtomicISub32, ctx, inst);
        case IrOpcode::SharedAtomicInc32: return Invoke(EmitSharedAtomicInc32, ctx, inst);
        case IrOpcode::SharedAtomicDec32: return Invoke(EmitSharedAtomicDec32, ctx, inst);
        case IrOpcode::SharedAtomicSMin32: return Invoke(EmitSharedAtomicSMin32, ctx, inst);
        case IrOpcode::SharedAtomicUMin32: return Invoke(EmitSharedAtomicUMin32, ctx, inst);
        case IrOpcode::SharedAtomicSMax32: return Invoke(EmitSharedAtomicSMax32, ctx, inst);
        case IrOpcode::SharedAtomicUMax32: return Invoke(EmitSharedAtomicUMax32, ctx, inst);
        case IrOpcode::SharedAtomicAnd32: return Invoke(EmitSharedAtomicAnd32, ctx, inst);
        case IrOpcode::SharedAtomicOr32: return Invoke(EmitSharedAtomicOr32, ctx, inst);
        case IrOpcode::SharedAtomicXor32: return Invoke(EmitSharedAtomicXor32, ctx, inst);
        case IrOpcode::SharedAtomicRsub32: return Invoke(EmitSharedAtomicRsub32, ctx, inst);
        case IrOpcode::SharedAtomicFAdd32: return Invoke(EmitSharedAtomicFAdd32, ctx, inst);
        case IrOpcode::SharedAtomicCmpst32: return Invoke(EmitSharedAtomicCmpst32, ctx, inst);
        case IrOpcode::SharedAtomicCmpstF32: return Invoke(EmitSharedAtomicCmpstF32, ctx, inst);
        case IrOpcode::SharedAtomicMskor32: return Invoke(EmitSharedAtomicMskor32, ctx, inst);
        case IrOpcode::SharedAtomicWrap32: return Invoke(EmitSharedAtomicWrap32, ctx, inst);
        case IrOpcode::SharedAtomicSwap64: return Invoke(EmitSharedAtomicSwap64, ctx, inst);
        case IrOpcode::SharedAtomicIAdd64: return Invoke(EmitSharedAtomicIAdd64, ctx, inst);
        case IrOpcode::SharedAtomicISub64: return Invoke(EmitSharedAtomicISub64, ctx, inst);
        case IrOpcode::SharedAtomicRsub64: return Invoke(EmitSharedAtomicRsub64, ctx, inst);
        case IrOpcode::SharedAtomicInc64: return Invoke(EmitSharedAtomicInc64, ctx, inst);
        case IrOpcode::SharedAtomicDec64: return Invoke(EmitSharedAtomicDec64, ctx, inst);
        case IrOpcode::SharedAtomicSMin64: return Invoke(EmitSharedAtomicSMin64, ctx, inst);
        case IrOpcode::SharedAtomicUMin64: return Invoke(EmitSharedAtomicUMin64, ctx, inst);
        case IrOpcode::SharedAtomicSMax64: return Invoke(EmitSharedAtomicSMax64, ctx, inst);
        case IrOpcode::SharedAtomicUMax64: return Invoke(EmitSharedAtomicUMax64, ctx, inst);
        case IrOpcode::SharedAtomicAnd64: return Invoke(EmitSharedAtomicAnd64, ctx, inst);
        case IrOpcode::SharedAtomicOr64: return Invoke(EmitSharedAtomicOr64, ctx, inst);
        case IrOpcode::SharedAtomicXor64: return Invoke(EmitSharedAtomicXor64, ctx, inst);
        case IrOpcode::SharedAtomicFMin64: return Invoke(EmitSharedAtomicFMin64, ctx, inst);
        case IrOpcode::SharedAtomicFMax64: return Invoke(EmitSharedAtomicFMax64, ctx, inst);
        case IrOpcode::SharedAtomicCmpst64: return Invoke(EmitSharedAtomicCmpst64, ctx, inst);
        case IrOpcode::SharedAtomicCmpstF64: return Invoke(EmitSharedAtomicCmpstF64, ctx, inst);
        case IrOpcode::SharedAtomicMskor64: return Invoke(EmitSharedAtomicMskor64, ctx, inst);
        case IrOpcode::DataAppend: return Invoke(EmitDataAppend, ctx, inst);
        case IrOpcode::DataConsume: return Invoke(EmitDataConsume, ctx, inst);
        case IrOpcode::DataOrderedCount: return Invoke(EmitDataOrderedCount, ctx, inst);
        case IrOpcode::SwizzleU32: return Invoke(EmitSwizzleU32, ctx, inst);
        case IrOpcode::ImageQueryDimensions: return Invoke(EmitImageQueryDimensions, ctx, inst);
        case IrOpcode::ImageQueryLod: return Invoke(EmitImageQueryLod, ctx, inst);
        case IrOpcode::ImageRead: return Invoke(EmitImageRead, ctx, inst);
        case IrOpcode::ImageBvhIntersectRay: return Invoke(EmitImageBvhIntersectRay, ctx, inst);
        case IrOpcode::ImageWrite: return Invoke(EmitImageWrite, ctx, inst);
        case IrOpcode::ImageSampleRaw: return Invoke(EmitImageSampleRaw, ctx, inst);
        case IrOpcode::ImageGatherRaw: return Invoke(EmitImageGatherRaw, ctx, inst);
        case IrOpcode::ImageAtomicSwap32: return Invoke(EmitImageAtomicSwap32, ctx, inst);
        case IrOpcode::ImageAtomicIAdd32: return Invoke(EmitImageAtomicIAdd32, ctx, inst);
        case IrOpcode::ImageAtomicUMin32: return Invoke(EmitImageAtomicUMin32, ctx, inst);
        case IrOpcode::ImageAtomicUMax32: return Invoke(EmitImageAtomicUMax32, ctx, inst);
        case IrOpcode::ImageAtomicAnd32: return Invoke(EmitImageAtomicAnd32, ctx, inst);
        case IrOpcode::ImageAtomicOr32: return Invoke(EmitImageAtomicOr32, ctx, inst);
        case IrOpcode::ImageAtomicXor32: return Invoke(EmitImageAtomicXor32, ctx, inst);
        case IrOpcode::ImageAtomicCmpSwap32: return Invoke(EmitImageAtomicCmpSwap32, ctx, inst);
        case IrOpcode::ImageAtomicISub32: return Invoke(EmitImageAtomicISub32, ctx, inst);
        case IrOpcode::ImageAtomicSMin32: return Invoke(EmitImageAtomicSMin32, ctx, inst);
        case IrOpcode::ImageAtomicSMax32: return Invoke(EmitImageAtomicSMax32, ctx, inst);
        case IrOpcode::ImageAtomicInc32: return Invoke(EmitImageAtomicInc32, ctx, inst);
        case IrOpcode::ImageAtomicDec32: return Invoke(EmitImageAtomicDec32, ctx, inst);
        case IrOpcode::ImageAtomicFCmpSwap32: return Invoke(EmitImageAtomicFCmpSwap32, ctx, inst);
        case IrOpcode::ImageAtomicFMin32: return Invoke(EmitImageAtomicFMin32, ctx, inst);
        case IrOpcode::ImageAtomicFMax32: return Invoke(EmitImageAtomicFMax32, ctx, inst);
        case IrOpcode::ImageAtomicSwap64: return Invoke(EmitImageAtomicSwap64, ctx, inst);
        case IrOpcode::ImageAtomicIAdd64: return Invoke(EmitImageAtomicIAdd64, ctx, inst);
        case IrOpcode::ImageAtomicISub64: return Invoke(EmitImageAtomicISub64, ctx, inst);
        case IrOpcode::ImageAtomicUMin64: return Invoke(EmitImageAtomicUMin64, ctx, inst);
        case IrOpcode::ImageAtomicUMax64: return Invoke(EmitImageAtomicUMax64, ctx, inst);
        case IrOpcode::ImageAtomicSMin64: return Invoke(EmitImageAtomicSMin64, ctx, inst);
        case IrOpcode::ImageAtomicSMax64: return Invoke(EmitImageAtomicSMax64, ctx, inst);
        case IrOpcode::ImageAtomicAnd64: return Invoke(EmitImageAtomicAnd64, ctx, inst);
        case IrOpcode::ImageAtomicOr64: return Invoke(EmitImageAtomicOr64, ctx, inst);
        case IrOpcode::ImageAtomicXor64: return Invoke(EmitImageAtomicXor64, ctx, inst);
        case IrOpcode::ImageAtomicCmpSwap64: return Invoke(EmitImageAtomicCmpSwap64, ctx, inst);
        case IrOpcode::GetAttribute: return Invoke(EmitGetAttribute, ctx, inst);
        case IrOpcode::GetInterpolationParameter: return Invoke(EmitGetInterpolationParameter, ctx, inst);
        case IrOpcode::GetInterpolationParameterF16: return Invoke(EmitGetInterpolationParameterF16, ctx, inst);
        case IrOpcode::SetAttribute: return Invoke(EmitSetAttribute, ctx, inst);
        case IrOpcode::ControlNop: return Invoke(EmitControlNop, ctx, inst);
        case IrOpcode::Waitcnt: return Invoke(EmitWaitcnt, ctx, inst);
        case IrOpcode::Sendmsg: return Invoke(EmitSendmsg, ctx, inst);
        case IrOpcode::TtraceData: return Invoke(EmitTtraceData, ctx, inst);
        case IrOpcode::InstPrefetch: return Invoke(EmitInstPrefetch, ctx, inst);
        case IrOpcode::SelectU32: return Invoke(EmitSelectU32, ctx, inst);
        default: ctx.Fail(inst, "has no direct SPIR-V emitter");
    }
}

void EmitStructuredInstruction(SpirvValueEmitContext& ctx, StructuredFunctionState& functionState, const IrValue& inst) {
    if (inst.Opcode() == IrOpcode::Phi) {
        const auto type = TypeId(ctx.state, inst.Type());
        if (type == 0 || inst.ArgumentCount() == 0) {
            ctx.Fail(inst, "has no native SPIR-V representation");
        }
        for (std::size_t index = 0; index < inst.ArgumentCount(); index++) {
            const IrBlock* predecessor = inst.PhiBlock(index);
            if (predecessor == nullptr || !ctx.state.labels.contains(predecessor)) {
                ctx.Fail(inst, "has a predecessor outside the structured function");
            }
        }
        functionState.deferredPhis.push_back({ctx.state.module.AddDeferredPhi(type, ctx.Result(inst), inst.ArgumentCount()), &inst, ctx.half});
        return;
    }
    EmitDirectInstruction(ctx, inst);
}

void EmitStructuredBlock(SpirvValueEmitContext& ctx, StructuredFunctionState& functionState, const IrBlock* block) {
    auto& state = ctx.state;
    state.currentBlock = block;
    EmitLabel(state, ctx.Label(block));
    bool emittedNonPhi = false;
    // Wave LDS ordering (see WaveLdsScope): a barrier separates an LDS write from the next LDS access
    // and a read from the next write. The block may be entered right after a write.
    bool ldsWritten = true;
    bool ldsRead = false;
    std::vector<const IrValue*> phiAnchors;
    for (const IrValue* inst : block->Instructions()) {
        if (inst->Opcode() == IrOpcode::Phi) {
            if (emittedNonPhi) {
                ctx.Fail(*inst, "appears after a non-Phi instruction");
            }
        } else {
            emittedNonPhi = true;
            for (const IrValue* anchor : phiAnchors) EmitAnchoredDeviceImageLookups(ctx, *anchor);
            phiAnchors.clear();
        }
        if (state.waveLdsScope != 0) {
            const auto access = SharedAccessOf(inst->Opcode());
            if (inst->Opcode() == IrOpcode::Barrier) {
                ldsWritten = false;
                ldsRead = false;
            } else if (access != SharedAccess::None) {
                const bool writes = access != SharedAccess::Read;
                if (ldsWritten || (writes && ldsRead)) {
                    state.module.AddFunction(spv::OpControlBarrier, ConstantU32(state, state.waveLdsScope), ConstantU32(state, spv::ScopeWorkgroup), ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | LdsMemorySemantics(state)));
                    ldsWritten = false;
                    ldsRead = false;
                }
                ldsWritten |= writes;
                ldsRead |= access != SharedAccess::Write;
            }
        }
        const bool shared = state.laneCount == 2u && inst->Type() != IrType::Void && !IrOpcodeHasSideEffects(inst->Opcode()) && state.sharedLaneValues.contains(inst);
        for (std::uint32_t half = 0; half < state.laneCount; half++) {
            if (half != 0 && ctx.otherHalf == nullptr) {
                ctx.Fail(*inst, "requires a second lane context");
            }
            if (half != 0 && shared) {
                if (const auto found = ctx.definitions.find(inst); found != ctx.definitions.end()) {
                    ctx.otherHalf->Define(*inst, found->second);
                    continue;
                }
            }
            SpirvValueEmitContext& lane = half == 0 ? ctx : *ctx.otherHalf;
            state.laneHalf = half;
            if (half == 0 || (inst->Opcode() != IrOpcode::Barrier && inst->Opcode() != IrOpcode::MeshAllocate)) {
                EmitStructuredInstruction(lane, functionState, *inst);
            }
        }
        state.laneHalf = 0;
        if (state.laneCount == 1u && state.deviceImageAnchors.contains(inst)) {
            if (inst->Opcode() == IrOpcode::Phi) phiAnchors.push_back(inst);
            else EmitAnchoredDeviceImageLookups(ctx, *inst);
        }
    }
    for (const IrValue* anchor : phiAnchors) EmitAnchoredDeviceImageLookups(ctx, *anchor);
}

void PatchStructuredPhis(SpirvValueEmitContext& ctx, StructuredFunctionState& functionState) {
    for (const SpirvDeferredPhiPatch& deferred : functionState.deferredPhis) {
        SpirvValueEmitContext& lane = deferred.half == 0 ? ctx : *ctx.otherHalf;
        for (std::size_t index = 0; index < deferred.instruction->ArgumentCount(); index++) {
            const IrBlock* predecessor = deferred.instruction->PhiBlock(index);
            const auto found = functionState.blockExitLabels.find(predecessor);
            if (found == functionState.blockExitLabels.end()) {
                ctx.Fail(*deferred.instruction, "has a predecessor that was not emitted");
            }
            ctx.state.module.PatchDeferredPhi(deferred.phi, index, lane.Def(deferred.instruction->Argument(index)), found->second);
        }
    }
}

bool CommunicatesThroughMemory(IrOpcode opcode) {
    const auto buffer = BufferAccessOf(opcode);
    const auto address = AddressOpcodeInfoOf(opcode).access;
    const auto image = ImageOpcodeInfoOf(opcode).access;
    return SharedAccessOf(opcode) != SharedAccess::None || buffer == BufferAccess::Write || buffer == BufferAccess::Atomic || address == AddressAccess::Write || address == AddressAccess::Atomic || image == ImageAccess::Write || image == ImageAccess::Atomic;
}

struct FunctionLayout {
    std::vector<std::vector<std::size_t>> blocks;
    std::vector<std::uint32_t> callee;
    std::vector<std::size_t> callBlock;
    std::vector<std::size_t> exitBlock;
    std::vector<const IrBlock*> entry;
    std::vector<const IrBlock*> continuation;
};

struct CalleeValue {
    const IrValue* value = nullptr;
    std::uint32_t half = 0;
};

struct CalleeExit {
    std::vector<CalleeValue> outputs;
    bool status = false;
    std::uint32_t type = 0;
};

struct CalleeSignature {
    std::uint32_t id = 0;
    std::uint32_t type = 0;
    std::vector<CalleeValue> inputs;
    std::vector<std::uint32_t> pointers;
    CalleeExit exit;
};

struct CalleeStateScope {
    explicit CalleeStateScope(SpirvEmitterState& state)
        : state(state), currentLabel(state.currentLabel), currentBlock(state.currentBlock), continueTarget(state.continueTarget), bdaStopsInvocations(state.bdaStopsInvocations),
          bdaStopValue(state.bdaStopValue), bdaPcOverride(state.bdaPcOverride), calleeStopValue(state.calleeStopValue), memoryByteOffsets(state.memoryByteOffsets),
          bufferAddresses(state.bufferAddresses), bufferDwords(state.bufferDwords), ldsBufferBase(state.ldsBufferBase), gdsLength(state.gdsLength),
          scratchVariable(state.scratchVariable), vectorWindowVariable(state.vectorWindowVariable), ldsVariable(state.ldsVariable), laneHalf(state.laneHalf),
          entryInvocationId(state.entryInvocationId), entryInvocationIndex(state.entryInvocationIndex), deviceImageLookups(std::move(state.deviceImageLookups)) {
        state.deviceImageLookups.clear();
    }
    CalleeStateScope(const CalleeStateScope&) = delete;
    CalleeStateScope& operator=(const CalleeStateScope&) = delete;
    ~CalleeStateScope() {
        state.currentLabel = currentLabel;
        state.currentBlock = currentBlock;
        state.continueTarget = continueTarget;
        state.bdaStopsInvocations = bdaStopsInvocations;
        state.bdaStopValue = bdaStopValue;
        state.bdaPcOverride = bdaPcOverride;
        state.calleeStopValue = calleeStopValue;
        state.memoryByteOffsets = std::move(memoryByteOffsets);
        state.bufferAddresses = std::move(bufferAddresses);
        state.bufferDwords = std::move(bufferDwords);
        state.ldsBufferBase = ldsBufferBase;
        state.gdsLength = gdsLength;
        state.scratchVariable = scratchVariable;
        state.vectorWindowVariable = vectorWindowVariable;
        state.ldsVariable = ldsVariable;
        state.laneHalf = laneHalf;
        state.entryInvocationId = entryInvocationId;
        state.entryInvocationIndex = entryInvocationIndex;
        state.deviceImageLookups = std::move(deviceImageLookups);
    }

    SpirvEmitterState& state;
    std::uint32_t currentLabel;
    const IrBlock* currentBlock;
    bool continueTarget;
    bool bdaStopsInvocations;
    std::uint32_t bdaStopValue;
    std::uint32_t bdaPcOverride;
    std::uint32_t calleeStopValue;
    std::vector<std::uint32_t> memoryByteOffsets;
    std::vector<std::uint32_t> bufferAddresses;
    std::vector<std::uint32_t> bufferDwords;
    std::uint32_t ldsBufferBase;
    std::uint32_t gdsLength;
    std::array<std::uint32_t, 2> scratchVariable;
    std::array<std::uint32_t, 2> vectorWindowVariable;
    std::uint32_t ldsVariable;
    std::uint32_t laneHalf;
    std::uint32_t entryInvocationId;
    std::uint32_t entryInvocationIndex;
    std::map<std::vector<std::uint64_t>, std::uint32_t> deviceImageLookups;
};

FunctionLayout PlanFunctions(SpirvValueEmitContext& ctx, const IrProgram& program) {
    auto& state = ctx.state;
    const auto& blocks = program.BlockOrder();
    const auto& infos = program.Metadata().blockInfo;
    const auto& functions = program.Metadata().callFunctions;
    if (blocks.size() != infos.size()) ctx.Fail("SPIR-V control flow block metadata is inconsistent");
    const auto count = functions.size() + 1u;
    FunctionLayout layout;
    layout.blocks.resize(count);
    layout.callee.assign(blocks.size(), 0u);
    layout.callBlock.assign(count, 0u);
    layout.exitBlock.assign(count, 0u);
    layout.entry.assign(count, nullptr);
    layout.continuation.assign(count, nullptr);
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        if (blocks[index] == nullptr) ctx.Fail("structured control flow contains a null block");
        if (infos[index].function >= count) ctx.Fail("a block belongs to a callee function the program does not have");
        layout.blocks[infos[index].function].push_back(index);
    }
    if (functions.empty()) return layout;
    if (program.Resources().stage != IrShaderStage::Compute) ctx.Fail("guest callees become functions only in compute programs");
    std::unordered_map<std::uint32_t, std::size_t> positions;
    std::uint32_t lastId = 0;
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        positions.emplace(infos[index].id, index);
        lastId = std::max(lastId, blocks[index]->Id());
    }
    state.blockFunctions.assign(lastId + 1u, 0u);
    for (std::size_t index = 0; index < blocks.size(); ++index) state.blockFunctions[blocks[index]->Id()] = infos[index].function;
    const auto position = [&](std::uint32_t id) {
        const auto found = positions.find(id);
        if (found == positions.end()) ctx.Fail("a callee function names a block the program does not have");
        return found->second;
    };
    state.functionParents.assign(count, 0u);
    for (std::uint32_t function = 1u; function < count; ++function) {
        const auto& call = functions[function - 1u];
        if (call.parent >= count || call.parent == function) ctx.Fail("a callee function names a parent the program does not have");
        state.functionParents[function] = call.parent;
        const auto callIndex = position(call.callBlock);
        const auto exitIndex = position(call.exitBlock);
        const auto& callInfo = infos[callIndex];
        const auto& exitInfo = infos[exitIndex];
        if (callInfo.function != call.parent || exitInfo.function != function || callInfo.terminator.kind != TerminatorKind::Branch || exitInfo.terminator.kind != TerminatorKind::Branch ||
            exitInfo.terminator.trueBlock != call.continuationBlock || infos[position(callInfo.terminator.trueBlock)].function != function || infos[position(call.continuationBlock)].function != call.parent) {
            ctx.Fail("a callee function does not match its call and return blocks");
        }
        if (layout.callee[callIndex] != 0u) ctx.Fail("a block calls two callee functions");
        layout.callee[callIndex] = function;
        layout.callBlock[function] = callIndex;
        layout.exitBlock[function] = exitIndex;
        layout.entry[function] = TargetBlock(program, callInfo.terminator.trueBlock);
        layout.continuation[function] = blocks[position(call.continuationBlock)];
    }
    return layout;
}

std::vector<const IrValue*> CalleeLiveOuts(SpirvEmitterState& state, const IrProgram& program, std::uint32_t function) {
    const auto inside = [&](const IrBlock* block) {
        return block != nullptr && block->Id() < state.blockFunctions.size() && EnclosesFunction(state, function, state.blockFunctions[block->Id()]);
    };
    const auto usedOutside = [&](const auto& self, const IrValue* value) -> bool {
        for (const IrValue* user : value->Uses()) {
            if (user->IsIdentity()) {
                if (self(self, user)) return true;
            } else if (user->Parent() != nullptr && !inside(user->Parent())) {
                return true;
            }
        }
        return false;
    };
    std::vector<const IrValue*> values;
    std::unordered_set<const IrValue*> added;
    const auto add = [&](const auto& self, const IrValue* value) -> void {
        if (value == nullptr) return;
        const IrValue* resolved = value->Resolve();
        if (resolved->HasImmediate() || !inside(resolved->Parent()) || !added.insert(resolved).second) return;
        if (TypeId(state, resolved->Type()) == 0u) {
            for (const IrValue* argument : resolved->Arguments()) self(self, argument);
            return;
        }
        values.push_back(resolved);
    };
    for (const IrBlock* block : program.BlockOrder()) {
        if (!inside(block)) continue;
        for (const IrValue* value : block->Instructions()) {
            if (value->HasUses() && usedOutside(usedOutside, value)) add(add, value);
        }
    }
    return values;
}

bool CalleeStops(const SpirvEmitterState& state, const IrProgram& program, const FunctionLayout& layout, std::uint32_t function) {
    if (state.programStopsInvocations) return true;
    const auto& infos = program.Metadata().blockInfo;
    for (std::size_t index = 0; index < infos.size(); ++index) {
        const auto owner = infos[index].function;
        if (!EnclosesFunction(state, function, owner) || layout.callee[index] != 0u || (owner != 0u && index == layout.exitBlock[owner])) continue;
        const auto kind = infos[index].terminator.kind;
        if (kind != TerminatorKind::Branch && kind != TerminatorKind::ConditionalBranch) return true;
    }
    return false;
}

void EmitFunctionBlocks(SpirvValueEmitContext& context, StructuredFunctionState& functionState, const IrProgram& program, const FunctionLayout& layout, std::uint32_t function, const CalleeExit* calleeExit);

void EmitCalleeExit(SpirvValueEmitContext& context, const CalleeExit& calleeExit) {
    auto& state = context.state;
    if (calleeExit.outputs.empty()) {
        if (calleeExit.status) state.module.AddFunction(spv::OpReturnValue, ConstantU32(state, 1u));
        else state.module.AddFunction(spv::OpReturn);
        return;
    }
    const auto result = state.module.AllocateId();
    std::vector<std::uint32_t> construct{spv::OpCompositeConstruct, calleeExit.type, result};
    if (calleeExit.status) construct.push_back(ConstantU32(state, 1u));
    for (const auto& output : calleeExit.outputs) construct.push_back((output.half == 0u ? context : *context.otherHalf).Def(output.value));
    state.module.AddFunction(std::span<const std::uint32_t>(construct));
    state.module.AddFunction(spv::OpReturnValue, result);
}

CalleeSignature EmitCalleeFunction(SpirvValueEmitContext& caller, const IrProgram& program, const FunctionLayout& layout, std::uint32_t function) {
    auto& state = caller.state;
    CalleeSignature signature;
    for (const IrValue* value : CalleeLiveOuts(state, program, function)) {
        for (std::uint32_t half = 0; half < state.laneCount; ++half) signature.exit.outputs.push_back({value, half});
    }
    signature.exit.status = CalleeStops(state, program, layout, function);
    std::vector<std::uint32_t> members;
    if (signature.exit.status) members.push_back(TypeU32(state));
    for (const auto& output : signature.exit.outputs) members.push_back(TypeId(state, output.value->Type()));
    if (signature.exit.outputs.empty()) signature.type = signature.exit.status ? TypeU32(state) : TypeVoid(state);
    else signature.type = state.module.Type(spv::OpTypeStruct, std::span<const std::uint32_t>(members));
    signature.exit.type = signature.type;
    struct Pointer {
        std::uint32_t* slot;
        std::uint32_t type;
    };
    std::vector<Pointer> pointers;
    if (state.requirements.functionScratch) {
        for (std::uint32_t half = 0; half < state.laneCount; ++half) pointers.push_back({&state.scratchVariable.at(half), TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.Info().scratchDwords)});
    }
    if (const auto window = state.program.Metadata().vectorWindowSize; window != 0u) {
        const auto array = state.module.Type(spv::OpTypeArray, TypeU32(state), ConstantU32(state, window));
        for (std::uint32_t half = 0; half < state.laneCount; ++half) pointers.push_back({&state.vectorWindowVariable.at(half), TypePointer(state, spv::StorageClassFunction, array)});
    }
    if (state.requirements.functionLds) pointers.push_back({&state.ldsVariable, TypeU32ArrayPointer(state, spv::StorageClassFunction, LdsDwordCount(state))});
    signature.id = state.module.AllocateId();
    const CalleeStateScope scope(state);
    std::vector<std::uint32_t> pointerParameters;
    for (auto& pointer : pointers) {
        signature.pointers.push_back(*pointer.slot);
        *pointer.slot = state.module.AllocateId();
        pointerParameters.push_back(*pointer.slot);
    }
    const auto& blocks = program.BlockOrder();
    const auto start = state.module.FunctionWordCount();
    EmitLabel(state, state.module.AllocateId());
    SpirvValueEmitContext low(state);
    SpirvValueEmitContext high(state);
    std::vector<SpirvValueEmitContext*> lanes{&low};
    if (state.laneCount == 2u) {
        low.otherHalf = &high;
        high.otherHalf = &low;
        high.half = 1u;
        lanes.push_back(&high);
    }
    const bool swizzles = std::ranges::any_of(layout.blocks[function], [&](std::size_t index) {
        return std::ranges::any_of(blocks[index]->Instructions(), [](const IrValue* inst) { return inst->Opcode() == IrOpcode::SwizzleU32; });
    });
    for (auto* lane : lanes) {
        lane->functionScope = true;
        lane->function = function;
        if (swizzles) {
            lane->scratchU32Variable = state.module.AllocateId();
            state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, TypeU32(state)), lane->scratchU32Variable, spv::StorageClassFunction);
        }
    }
    EmitMemoryOffsets(state);
    EmitEntryInvocationIds(state);
    EmitLdsBufferBase(state);
    if (state.gdsVariable != 0u) {
        state.gdsLength = state.module.AllocateId();
        state.module.AddFunction(spv::OpArrayLength, TypeU32(state), state.gdsLength, state.gdsVariable, 0u);
    }
    state.calleeStopValue = !signature.exit.status ? 0u : signature.exit.outputs.empty() ? ConstantU32(state, 0u) : state.module.Constant(spv::OpConstantNull, signature.type);
    state.bdaStopValue = state.calleeStopValue;
    state.bdaPcOverride = 0u;
    StructuredFunctionState functionState;
    functionState.blockExitLabels.emplace(blocks[layout.callBlock[function]], state.currentLabel);
    state.module.AddFunction(spv::OpBranch, low.Label(layout.entry[function]));
    EmitFunctionBlocks(low, functionState, program, layout, function, &signature.exit);
    PatchStructuredPhis(low, functionState);
    const auto words = state.module.TakeFunctionWords(start);
    std::vector<std::uint32_t> types{signature.type};
    std::vector<std::uint32_t> ids;
    for (std::size_t index = 0; index < pointers.size(); ++index) {
        types.push_back(pointers[index].type);
        ids.push_back(pointerParameters[index]);
    }
    for (const auto* lane : lanes) {
        for (const auto& [value, id] : lane->inputs) {
            signature.inputs.push_back({value, lane->half});
            types.push_back(TypeId(state, value->Type()));
            ids.push_back(id);
        }
    }
    state.module.BeginHelperFunction();
    state.module.AddFunction(spv::OpFunction, signature.type, signature.id, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, std::span<const std::uint32_t>(types)));
    for (std::size_t index = 0; index < ids.size(); ++index) state.module.AddFunction(spv::OpFunctionParameter, types[index + 1u], ids[index]);
    state.module.AppendFunctionWords(words);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.EndHelperFunction();
    char name[32];
    std::snprintf(name, sizeof(name), "guest_callee_%x", BlockInfoFor(program, layout.entry[function])->startPc);
    state.module.AddName(signature.id, name);
    return signature;
}

void EmitGuestCall(SpirvValueEmitContext& caller, StructuredFunctionState& functionState, const IrProgram& program, const FunctionLayout& layout, std::uint32_t function, const BlockInfo& info) {
    auto& state = caller.state;
    if (info.terminator.loopHeader) {
        state.module.AddFunction(spv::OpLoopMerge, caller.Label(TargetBlock(program, info.terminator.mergeBlock)), caller.Label(TargetBlock(program, info.terminator.continueBlock)), spv::LoopControlMaskNone);
    }
    const auto callLabel = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, callLabel);
    EmitLabel(state, callLabel);
    const auto signature = EmitCalleeFunction(caller, program, layout, function);
    const auto lane = [&](std::uint32_t half) -> SpirvValueEmitContext& { return half == 0u ? caller : *caller.otherHalf; };
    const auto result = state.module.AllocateId();
    std::vector<std::uint32_t> call{spv::OpFunctionCall, signature.type, result, signature.id};
    call.insert(call.end(), signature.pointers.begin(), signature.pointers.end());
    for (const auto& input : signature.inputs) call.push_back(lane(input.half).Def(input.value));
    state.module.AddFunction(std::span<const std::uint32_t>(call));
    const bool composite = !signature.exit.outputs.empty();
    std::uint32_t member = signature.exit.status ? 1u : 0u;
    for (const auto& output : signature.exit.outputs) {
        const auto returned = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeId(state, output.value->Type()), returned, result, member++);
        lane(output.half).Define(*output.value, returned);
    }
    if (signature.exit.status) {
        std::uint32_t status = result;
        if (composite) {
            status = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), status, result, 0u);
        }
        const auto stopped = Binary(state, spv::OpIEqual, TypeBool(state), status, ConstantU32(state, 0u));
        const auto stop = state.module.AllocateId();
        const auto next = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, stopped, stop, next);
        EmitLabel(state, stop);
        if (state.calleeStopValue != 0u) state.module.AddFunction(spv::OpReturnValue, state.calleeStopValue);
        else state.module.AddFunction(spv::OpReturn);
        EmitLabel(state, next);
    }
    functionState.blockExitLabels.emplace(program.BlockOrder()[layout.exitBlock[function]], state.currentLabel);
    state.module.AddFunction(spv::OpBranch, caller.Label(layout.continuation[function]));
}

void EmitFunctionBlocks(SpirvValueEmitContext& context, StructuredFunctionState& functionState, const IrProgram& program, const FunctionLayout& layout, std::uint32_t function, const CalleeExit* calleeExit) {
    auto& state = context.state;
    const auto& blocks = program.BlockOrder();
    const auto& infos = program.Metadata().blockInfo;
    for (const auto index : layout.blocks[function]) {
        const IrBlock* block = blocks[index];
        const BlockInfo& info = infos[index];
        state.continueTarget = IsContinueTarget(program, info.id);
        state.bdaStopsInvocations = state.programStopsInvocations && !state.continueTarget;
        EmitStructuredBlock(context, functionState, block);
        functionState.blockExitLabels.emplace(block, state.currentLabel);
        if (layout.callee[index] != 0u) {
            EmitGuestCall(context, functionState, program, layout, layout.callee[index], info);
        } else if (calleeExit != nullptr && index == layout.exitBlock[function]) {
            EmitCalleeExit(context, *calleeExit);
        } else {
            EmitStructuredTerminator(context, program, info);
        }
        state.bdaStopsInvocations = state.programStopsInvocations;
        state.continueTarget = false;
    }
}

}

void EmitControlFlow(SpirvModule& module, const IrProgram& program) {
    throw std::runtime_error("EmitControlFlow not implemented");
}

void EmitControlFlow(SpirvValueEmitContext& context, StructuredFunctionState& functionState, const IrProgram& program) {
    auto& state = context.state;
    const std::vector<IrBlock*>& blocks = program.BlockOrder();
    if (blocks.empty() || blocks.front() == nullptr) {
        context.Fail("structured control flow requires at least one block");
    }
    if (state.loopGuardLimit != 0) {
        const auto pointer = TypePointer(state, spv::StorageClassPrivate, TypeU32(state));
        if (state.loopGuardVisits == 0) state.loopGuardVisits = state.module.DefineGlobalVariable(pointer, spv::StorageClassPrivate);
        if (state.loopGuardPc == 0) state.loopGuardPc = state.module.DefineGlobalVariable(pointer, spv::StorageClassPrivate);
        state.module.AddFunction(spv::OpStore, state.loopGuardVisits, ConstantU32(state, 0u));
        state.module.AddFunction(spv::OpStore, state.loopGuardPc, ConstantU32(state, 0u));
    }
    DefineWaveExchange(state);
    state.module.AddFunction(spv::OpBranch, context.Label(blocks.front()));
    static const bool debug = std::getenv("APS5_SRT_DEBUG") != nullptr;
    if (debug) {
        const auto& infos = program.Metadata().blockInfo;
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            const auto* byId = BlockInfoFor(program, blocks[index]);
            std::fprintf(stderr, "[spirv] order %zu: IR block %u, positional info id %u, id-matched info %s (kind %d true %u false %u)\n", index, blocks[index]->Id(),
                         index < infos.size() ? infos[index].id : 0xffffffffu, byId ? std::to_string(byId->id).c_str() : "none",
                         byId ? static_cast<int>(byId->terminator.kind) : -1, byId ? byId->terminator.trueBlock : 0u, byId ? byId->terminator.falseBlock : 0u);
        }
    }
    const auto layout = PlanFunctions(context, program);
    if (!state.functionParents.empty()) {
        context.functionScope = true;
        if (context.otherHalf != nullptr) context.otherHalf->functionScope = true;
    }
    state.programStopsInvocations = state.bdaStopsInvocations;
    EmitFunctionBlocks(context, functionState, program, layout, 0u, nullptr);
    PatchStructuredPhis(context, functionState);
}

void EmitVoid(SpirvValueEmitContext&) {
}

void EmitBarrier(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    if (state.program.Resources().stage == IrShaderStage::Vertex) {
        for (const auto* block : state.program.BlockOrder()) {
            for (const auto* instruction : block->Instructions()) {
                if (CommunicatesThroughMemory(instruction->Opcode())) ctx.Fail(inst, "s_barrier in a vertex program that accesses LDS or GDS or writes memory is not implemented");
            }
        }
        return;
    }
    const bool tessellation = state.program.Resources().stage == IrShaderStage::TessellationControl;
    const auto memoryScope = tessellation ? spv::ScopeInvocation : spv::ScopeWorkgroup;
    const std::uint32_t semantics = tessellation ? static_cast<std::uint32_t>(spv::MemorySemanticsMaskNone) : spv::MemorySemanticsAcquireReleaseMask | LdsMemorySemantics(state);
    state.module.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup), ConstantU32(state, memoryScope), ConstantU32(state, semantics));
}

void EmitUnreachable(SpirvValueEmitContext& ctx, const IrValue& inst) {
    ctx.Fail(inst, "must be lowered before SPIR-V emission");
}

void EmitReference(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitReferenceU32(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitUnlinkedCall(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t low, std::uint32_t high, std::uint32_t programCounter) {
    auto& state = ctx.state;
    if (state.faultBufferVariable == 0) ctx.Fail(inst, "calls an unlinked target without a fault buffer");
    RecordBdaFaultWords(state, low, high, ConstantU32(state, 0u), programCounter, BdaAbi::FaultReason::UnlinkedCall);
    StopBdaInvocationIf(state, ConstantBool(state, true));
}

void EmitControlNop(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitWaitcnt(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitSendmsg(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitTtraceData(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitInstPrefetch(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitPhi(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitUnreachable(ctx, inst);
}

}
