#include "Optimization/SrtWalker/SrtPlanBuilder.hpp"
#include "Optimization/SrtWalker/SrtDiagnostics.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"
#include "Optimization/SrtWalker/SrtRuntimeValidator.hpp"

#include "IntermediateRepresentation/IrBuilder.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace ShaderRecompiler::Detail {

void PlanBuilder::Run() {
    _program.Resources().srtReads.clear();
    _program.Metadata().dynamicReads.clear();
    for (auto& block : _program.Blocks()) {
        for (IrValue* inst : block->Instructions()) {
            const auto op = inst->Opcode();
            if (op == IrOpcode::LoadAddressU32 || op == IrOpcode::ReadConstBuffer) {
                const auto flags = inst->Flags<MemoryFlags>();
                if (flags.index < _program.Resources().memoryInfo.size()) {
                    const auto kind = _program.Resources().memoryInfo[flags.index].kind;
                    const bool crosswired = (op == IrOpcode::LoadAddressU32 && kind == ResourceKind::ScalarBuffer) || (op == IrOpcode::ReadConstBuffer && kind == ResourceKind::ScalarAddress);
                    if (crosswired) {
                        Fail(_program.Resources(), flags.pc, std::string(IrOpcodeName(op)) + " has incompatible scalar memory metadata");
                    }
                }
            }
            if (IsDescriptorHandle(inst->Opcode())) {
                for (std::size_t index = 0; index < inst->ArgumentCount(); index++) {
                    Collect(inst->Argument(index), 0);
                }
            }
        }
    }
    for (auto& block : _program.Blocks()) {
        for (IrValue* inst : block->Instructions()) {
            if (inst->Opcode() == IrOpcode::LoadAddressU32 && IsRawRead(_program.Resources(), *inst) && inst->Argument(1)->Resolve()->HasImmediate() && RuntimeValidator(_program.Resources(), RuntimeValueType::Any).Run(inst)) {
                Collect(inst, inst->Flags<MemoryFlags>().pc);
            }
        }
    }
    PatchReads();
}

void PlanBuilder::Collect(IrValue* raw, std::uint32_t usePc) {
    struct Frame {
        IrValue* inst;
        std::size_t nextArgument = 0;
    };
    std::vector<Frame> pending;
    std::unordered_map<IrValue*, std::size_t> visiting;
    const auto enter = [&](IrValue* candidate) {
        IrValue* inst = candidate->Resolve();
        if (inst->Opcode() == IrOpcode::Void) return;
        const auto cycle = visiting.find(inst);
        if (cycle != visiting.end()) {
            const auto first = pending.begin() + cycle->second;
            if (std::any_of(first, pending.end(), [](const Frame& frame) { return frame.inst->Opcode() == IrOpcode::Phi; })) return;
            Fail(_program.Resources(), usePc, "cyclic typed planning value " + std::string(IrOpcodeName(inst->Opcode())) + " without a phi");
        }
        if (_visited.contains(inst)) return;
        visiting.emplace(inst, pending.size());
        pending.push_back({inst});
    };
    const auto collectRead = [&](IrValue* inst) {
        if (!IsRawRead(_program.Resources(), *inst)) return;
        IrValue* offset = inst->Argument(1)->Resolve();
        const auto foldable = offset->HasImmediate() && offset->Type() == IrType::U32 && RuntimeValidator(_program.Resources(), RuntimeValueType::Any).Run(inst);
        if (!foldable) {
            if (std::find(_program.Metadata().dynamicReads.begin(), _program.Metadata().dynamicReads.end(), inst) == _program.Metadata().dynamicReads.end()) {
                _program.Metadata().dynamicReads.push_back(inst);
            }
            return;
        }
        for (std::uint32_t slot = 0; slot < _program.Resources().srtReads.size(); slot++) {
            if (EquivalentValue(_program.Resources(), inst, _program.Resources().srtReads[slot].value)) {
                _patches.push_back({inst, slot, false});
                return;
            }
        }
        const auto slot = static_cast<std::uint32_t>(_program.Resources().srtReads.size());
        _program.Resources().srtReads.push_back({inst, slot});
        _patches.push_back({inst, slot, true});
    };
    enter(raw);
    while (!pending.empty()) {
        auto& frame = pending.back();
        if (frame.nextArgument < frame.inst->ArgumentCount()) {
            auto* argument = frame.inst->Argument(frame.nextArgument++);
            enter(argument);
            continue;
        }
        auto* inst = frame.inst;
        pending.pop_back();
        visiting.erase(inst);
        _visited.insert(inst);
        collectRead(inst);
    }
}

void PlanBuilder::PatchReads() {
    IrBuilder builder(_program);
    for (const auto& patch : _patches) {
        IrBlock* block = patch.inst->Parent();
        if (block == nullptr) {
            throw std::runtime_error("SrtWalker::BuildPlan patched value has no parent block");
        }
        builder.SetInsertionPoint(*block);
        IrValue& resource = builder.Emit(IrOpcode::GetSrtResource, IrType::SrtResource, {});
        block->RemoveInstruction(&resource);
        block->InsertInstructionBefore(patch.inst, &resource);
        IrValue& slotConstant = builder.Constant(patch.slot);
        IrValue& flat = builder.Emit(IrOpcode::ReadConst, IrType::U32, {&resource, &slotConstant});
        block->RemoveInstruction(&flat);
        block->InsertInstructionBefore(patch.inst, &flat);
        for (const auto& use : std::vector<IrUse>(patch.inst->OperandUses())) {
            use.user->ReplaceArgument(use.operand, &flat);
        }
        for (auto& info : _program.Metadata().blockInfo) {
            if (info.condition != nullptr && info.condition->Resolve() == patch.inst) {
                info.condition = &flat;
            }
            if (info.indirectTarget != nullptr && info.indirectTarget->Resolve() == patch.inst) {
                info.indirectTarget = &flat;
            }
        }
        if (patch.keep) {
            const auto memory = patch.inst->Flags<MemoryFlags>().index;
            if (memory < _program.Resources().memoryInfo.size()) {
                _program.Resources().memoryInfo[memory].planningOnly = true;
            }
            const auto& irVal = builder.Emit(IrOpcode::ReferenceU32, IrType::Void, {patch.inst});
        }
    }
}

}
