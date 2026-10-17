#include "Optimization/DenormalFlushEliminator.hpp"
#include "Optimization/DominatorTree.hpp"
#include "Optimization/FloatIdioms.hpp"
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

namespace {

bool isImmediate(const IrValue& value, std::uint32_t bits) {
    return value.HasImmediate() && value.ImmediateU32() == bits;
}

const IrValue* bitsOf(const IrValue& value) {
    const IrValue* current = value.Resolve();
    while (current->Opcode() == IrOpcode::BitCastU32F32 || current->Opcode() == IrOpcode::BitCastF32U32) current = current->Argument(0)->Resolve();
    return current;
}

bool comparesMagnitude(const IrValue& condition, IrOpcode opcode, std::uint32_t bound, const IrValue* source) {
    const IrValue& compare = *condition.Resolve();
    if (compare.Opcode() != opcode || compare.ArgumentCount() != 2u || !isImmediate(*compare.Argument(1)->Resolve(), bound)) return false;
    return MaskedOperand(*compare.Argument(0)->Resolve(), IrOpcode::BitwiseAnd32, 0x7fffffffu) == source;
}

bool offsetOf(const IrValue& value, const IrValue* source, std::uint32_t step) {
    const IrValue& offset = *value.Resolve();
    if (offset.Opcode() == IrOpcode::IAdd32) return MaskedOperand(offset, IrOpcode::IAdd32, step) == source;
    return offset.Opcode() == IrOpcode::ISub32 && offset.Argument(0)->Resolve() == source && isImmediate(*offset.Argument(1)->Resolve(), 0u - step);
}

bool outputModifierResult(const IrValue& value) {
    if (value.Opcode() != IrOpcode::SelectU32 || value.ArgumentCount() != 3u) return false;
    const IrValue* bits = value.Argument(1)->Resolve();
    if (!comparesMagnitude(*value.Argument(0), IrOpcode::UGreaterThan32, 0x7f7fffffu, bits)) return false;
    const IrValue& small = *value.Argument(2)->Resolve();
    if (small.Opcode() != IrOpcode::SelectU32 || !isImmediate(*small.Argument(1)->Resolve(), 0u) || !comparesMagnitude(*small.Argument(0), IrOpcode::ULessThan32, 0x00800000u, bits)) return false;
    const IrValue& scaled = *small.Argument(2)->Resolve();
    if (scaled.Opcode() != IrOpcode::SelectU32) return false;
    if (comparesMagnitude(*scaled.Argument(0), IrOpcode::ULessThan32, 0x01000000u, bits)) {
        return MaskedOperand(*scaled.Argument(1)->Resolve(), IrOpcode::BitwiseAnd32, 0x80000000u) == bits && offsetOf(*scaled.Argument(2), bits, 0u - 0x00800000u);
    }
    for (const std::uint32_t step : {0x00800000u, 0x01000000u}) {
        if (!comparesMagnitude(*scaled.Argument(0), IrOpcode::ULessThan32, 0x7f800000u - step, bits) || !offsetOf(*scaled.Argument(1), bits, step)) continue;
        const IrValue* sign = MaskedOperand(*scaled.Argument(2)->Resolve(), IrOpcode::BitwiseOr32, 0x7f800000u);
        return sign != nullptr && MaskedOperand(*sign, IrOpcode::BitwiseAnd32, 0x80000000u) == bits;
    }
    return false;
}

const IrValue* exponentAllOnesSource(const IrValue& condition) {
    const IrValue& test = *condition.Resolve();
    if (const IrValue* source = NanTestedF32Source(test)) return source;
    if (const IrValue* exponent = MaskedOperand(test, IrOpcode::IEqual32, 0x7f800000u)) return MaskedOperand(*exponent, IrOpcode::BitwiseAnd32, 0x7fc00000u);
    if (test.Opcode() != IrOpcode::LogicalAnd) return nullptr;
    for (std::size_t index = 0; index < 2u; ++index) {
        if (const IrValue* source = exponentAllOnesSource(*test.Argument(index))) return source;
    }
    return nullptr;
}

const IrValue* nanPassingOtherArm(const IrValue& value) {
    if ((value.Opcode() != IrOpcode::SelectU32 && value.Opcode() != IrOpcode::SelectF32) || value.ArgumentCount() != 3u) return nullptr;
    const IrValue* source = exponentAllOnesSource(*value.Argument(0));
    if (source == nullptr) return nullptr;
    const IrValue* sourceBits = bitsOf(*source);
    const IrValue* passed = bitsOf(*value.Argument(1));
    if (passed != sourceBits && QuietedF32Source(*passed) != sourceBits) return nullptr;
    return value.Argument(2);
}

const IrValue* tinyProductOtherArm(const IrValue& value) {
    if ((value.Opcode() != IrOpcode::SelectU32 && value.Opcode() != IrOpcode::SelectF32) || value.ArgumentCount() != 3u) return nullptr;
    const IrValue& product = *bitsOf(*value.Argument(2));
    if (product.Opcode() != IrOpcode::FPMul32 || product.ArgumentCount() != 2u) return nullptr;
    const IrValue& tiny = *value.Argument(0)->Resolve();
    if (tiny.Opcode() != IrOpcode::ULessThan32 || tiny.ArgumentCount() != 2u || !isImmediate(*tiny.Argument(1)->Resolve(), 0x207fffffu)) return nullptr;
    const IrValue& below = *tiny.Argument(0)->Resolve();
    if (below.Opcode() != IrOpcode::ISub32 || below.ArgumentCount() != 2u || !isImmediate(*below.Argument(1)->Resolve(), 1u)) return nullptr;
    const IrValue* magnitude = MaskedOperand(*below.Argument(0)->Resolve(), IrOpcode::BitwiseAnd32, 0x7fffffffu);
    if (magnitude == nullptr) return nullptr;
    const IrValue& scaled = *bitsOf(*magnitude);
    if (scaled.Opcode() != IrOpcode::FPMul32 || scaled.ArgumentCount() != 2u) return nullptr;
    const IrValue& lifted = *scaled.Argument(0)->Resolve();
    if (lifted.Opcode() != IrOpcode::FPMul32 || lifted.ArgumentCount() != 2u || !isImmediate(*lifted.Argument(1)->Resolve(), 0x5f800000u)) return nullptr;
    if (lifted.Argument(0)->Resolve() != product.Argument(0)->Resolve() || scaled.Argument(1)->Resolve() != product.Argument(1)->Resolve()) return nullptr;
    return value.Argument(1);
}

class DenormalFreeValues {
public:
    explicit DenormalFreeValues(const IrProgram& program) {
        std::vector<const IrValue*> values;
        for (const IrBlock* block : program.BlockOrder()) {
            for (const IrValue* inst : block->Instructions()) {
                values.push_back(inst);
                known.emplace(inst, true);
            }
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (const IrValue* value : values) {
                auto& slot = known[value];
                if (!slot || compute(*value)) continue;
                slot = false;
                changed = true;
            }
        }
    }

    bool Contains(const IrValue* value) {
        value = value->Resolve();
        if (const auto found = known.find(value); found != known.end()) return found->second;
        known.emplace(value, false);
        const bool result = compute(*value);
        known[value] = result;
        return result;
    }

private:
    bool compute(const IrValue& value) {
        if (value.HasImmediate()) {
            const std::uint32_t bits = value.ImmediateU32();
            return (bits & 0x7f800000u) != 0u || (bits & 0x7fffffffu) == 0u;
        }
        if (FlushedF32Source(value) != nullptr || MaskedOperand(value, IrOpcode::BitwiseAnd32, 0x80000000u) != nullptr || outputModifierResult(value)) return true;
        if (const IrValue* other = nanPassingOtherArm(value)) return Contains(other);
        if (const IrValue* sign = tinyProductOtherArm(value)) return Contains(sign);
        switch (value.Opcode()) {
            case IrOpcode::ConvertIToF32:
            case IrOpcode::ConvertUToF32:
            case IrOpcode::ConvertF32S32:
            case IrOpcode::ConvertF32U32:
            case IrOpcode::ConvertF32F16:
            case IrOpcode::FPFloor32:
            case IrOpcode::FPCeil32:
            case IrOpcode::FPTrunc32:
            case IrOpcode::FPRoundEven32:
                return true;
            case IrOpcode::FPFract32:
                return Contains(value.Argument(0));
            case IrOpcode::FPMin32:
            case IrOpcode::FPMax32:
            case IrOpcode::FPMinTri32:
            case IrOpcode::FPMaxTri32:
            case IrOpcode::FPMedTri32:
                for (std::size_t index = 0; index < value.ArgumentCount(); ++index) {
                    if (!Contains(value.Argument(index))) return false;
                }
                return true;
            case IrOpcode::BitCastU32F32:
            case IrOpcode::BitCastF32U32:
            case IrOpcode::FPAbs32:
            case IrOpcode::FPNeg32:
            case IrOpcode::FPSaturate32:
                return Contains(value.Argument(0));
            case IrOpcode::BitwiseXor32:
                if (const IrValue* operand = MaskedOperand(value, IrOpcode::BitwiseXor32, 0x80000000u)) return Contains(operand);
                return false;
            case IrOpcode::BitwiseAnd32:
                if (const IrValue* operand = MaskedOperand(value, IrOpcode::BitwiseAnd32, 0x7fffffffu)) return Contains(operand);
                return false;
            case IrOpcode::SelectU32:
            case IrOpcode::SelectF32:
                return Contains(value.Argument(1)) && Contains(value.Argument(2));
            case IrOpcode::Phi:
                for (std::size_t index = 0; index < value.ArgumentCount(); ++index) {
                    if (!Contains(value.Argument(index))) return false;
                }
                return true;
            default:
                return false;
        }
    }

    std::unordered_map<const IrValue*, bool> known;
};

}

namespace {

std::uint32_t distributeFlushes(IrProgram& program, DenormalFreeValues& denormalFree) {
    const DominatorTree tree(program);
    std::unordered_map<const IrValue*, IrValue*> flushed;
    std::vector<const IrValue*> added;
    std::vector<std::pair<std::uint32_t, std::size_t>> scopes;
    std::uint32_t distributed = 0;
    const auto available = [&](IrValue* value) -> IrValue* {
        if (denormalFree.Contains(value)) return value;
        const auto found = flushed.find(value->Resolve());
        return found == flushed.end() ? nullptr : found->second;
    };
    for (const auto& [block, depth] : tree.Preorder()) {
        while (!scopes.empty() && scopes.back().first >= depth) {
            for (std::size_t index = scopes.back().second; index < added.size(); ++index) flushed.erase(added[index]);
            added.resize(scopes.back().second);
            scopes.pop_back();
        }
        scopes.emplace_back(depth, added.size());
        const std::vector<IrValue*> instructions(block->Instructions().begin(), block->Instructions().end());
        for (IrValue* inst : instructions) {
            if (!inst->HasUses()) continue;
            IrValue* source = FlushedF32Source(*inst);
            if (source == nullptr) continue;
            IrValue* replacement = inst;
            if (source->Opcode() == IrOpcode::SelectU32 && source->ArgumentCount() == 3u) {
                IrValue* first = available(source->Argument(1));
                IrValue* second = available(source->Argument(2));
                if (first != nullptr && second != nullptr) {
                    auto& select = program.CreateValue(IrOpcode::SelectU32, IrType::U32);
                    select.AddArgument(source->Argument(0));
                    select.AddArgument(first);
                    select.AddArgument(second);
                    block->InsertInstructionBefore(inst, &select);
                    inst->ReplaceAllUsesWith(&select);
                    replacement = &select;
                    ++distributed;
                }
            }
            if (flushed.emplace(source, replacement).second) added.push_back(source);
        }
    }
    return distributed;
}

}

DenormalFlushEliminationStats DenormalFlushEliminator::Eliminate(IrProgram& program) const {
    DenormalFlushEliminationStats stats;
    DenormalFreeValues denormalFree(program);
    for (auto* block : program.BlockOrder()) {
        for (auto* inst : block->Instructions()) {
            if (!inst->HasUses()) continue;
            IrValue* source = FlushedF32Source(*inst);
            if (source == nullptr || !denormalFree.Contains(source)) continue;
            inst->ReplaceAllUsesWith(source);
            ++stats.removedFlushes;
        }
    }
    stats.removedFlushes += distributeFlushes(program, denormalFree);
    return stats;
}

}
