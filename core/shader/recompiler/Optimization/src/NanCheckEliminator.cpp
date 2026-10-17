#include "Optimization/NanCheckEliminator.hpp"
#include "Optimization/FloatIdioms.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler {

namespace {

constexpr std::uint8_t MayBeNan = 1u;
constexpr std::uint8_t MayBeInfinite = 2u;
constexpr std::uint8_t MayBeZero = 4u;
constexpr std::uint8_t AnyValue = MayBeNan | MayBeInfinite | MayBeZero;

const IrValue* bitsOf(const IrValue& value) {
    const IrValue* current = value.Resolve();
    while (current->Opcode() == IrOpcode::BitCastU32F32 || current->Opcode() == IrOpcode::BitCastF32U32) current = current->Argument(0)->Resolve();
    return current;
}

const IrValue* testedSource(const IrValue& condition) {
    if (const IrValue* source = NanTestedF32Source(condition)) return source;
    if (condition.Opcode() == IrOpcode::FPIsNan32) return condition.Argument(0)->Resolve();
    return nullptr;
}

std::uint8_t ofBits(std::uint32_t bits) {
    if ((bits & 0x7f800000u) == 0x7f800000u) return (bits & 0x007fffffu) != 0u ? MayBeNan : MayBeInfinite;
    return (bits & 0x7f800000u) == 0u ? MayBeZero : 0u;
}

bool invalidProduct(std::uint8_t lhs, std::uint8_t rhs) {
    return ((lhs & MayBeZero) != 0u && (rhs & MayBeInfinite) != 0u) || ((lhs & MayBeInfinite) != 0u && (rhs & MayBeZero) != 0u);
}

bool invalidSum(std::uint8_t lhs, std::uint8_t rhs, std::uint8_t addend) {
    return invalidProduct(lhs, rhs) || (((lhs | rhs) & MayBeInfinite) != 0u && (addend & MayBeInfinite) != 0u);
}

std::uint8_t arithmeticResult(bool nan) {
    return static_cast<std::uint8_t>(MayBeInfinite | MayBeZero | (nan ? MayBeNan : 0u));
}

class FloatClasses {
public:
    explicit FloatClasses(const IrProgram& program) {
        std::uint32_t count = 0;
        for (const IrBlock* block : program.BlockOrder()) {
            for (const IrValue* inst : block->Instructions()) count = std::max(count, inst->Id() + 1u);
        }
        classes.assign(count, Unseen);
        for (const IrBlock* block : program.BlockOrder()) {
            for (const IrValue* inst : block->Instructions()) classes[inst->Id()] = 0u;
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (const IrBlock* block : program.BlockOrder()) {
                for (const IrValue* inst : block->Instructions()) {
                    std::uint8_t& slot = classes[inst->Id()];
                    const std::uint8_t grown = static_cast<std::uint8_t>(slot | compute(*inst));
                    if (grown == slot) continue;
                    slot = grown;
                    changed = true;
                }
            }
        }
    }

    std::uint8_t Of(const IrValue* value) const {
        value = value->Resolve();
        if (value->HasImmediate()) return value->Type() == IrType::Bool ? AnyValue : ofBits(value->ImmediateU32());
        return value->Id() < classes.size() && classes[value->Id()] != Unseen ? classes[value->Id()] : AnyValue;
    }

    bool CannotBeNan(const IrValue* value) const {
        return (Of(value) & MayBeNan) == 0u;
    }

    bool NeverTrue(const IrValue& condition, std::uint32_t depth = 0u) const {
        const IrValue& test = *condition.Resolve();
        if (test.HasImmediate()) return test.Type() == IrType::Bool && !test.ImmediateBool();
        if (const IrValue* source = testedSource(test)) return CannotBeNan(source);
        if (depth > 8u) return false;
        if (const IrValue* magnitude = MaskedOperand(test, IrOpcode::IEqual32, 0x7f800000u)) {
            const IrValue* source = MaskedOperand(*magnitude, IrOpcode::BitwiseAnd32, 0x7fffffffu);
            return source != nullptr && (Of(source) & MayBeInfinite) == 0u;
        }
        if (const IrValue* magnitude = MaskedOperand(test, IrOpcode::IEqual32, 0u)) {
            const IrValue* source = MaskedOperand(*magnitude, IrOpcode::BitwiseAnd32, 0x7fffffffu);
            return source != nullptr && (Of(source) & MayBeZero) == 0u;
        }
        switch (test.Opcode()) {
            case IrOpcode::LogicalAnd: return NeverTrue(*test.Argument(0), depth + 1u) || NeverTrue(*test.Argument(1), depth + 1u);
            case IrOpcode::LogicalOr: return NeverTrue(*test.Argument(0), depth + 1u) && NeverTrue(*test.Argument(1), depth + 1u);
            case IrOpcode::LogicalNot: return AlwaysTrue(*test.Argument(0), depth + 1u);
            default: return false;
        }
    }

    bool AlwaysTrue(const IrValue& condition, std::uint32_t depth = 0u) const {
        const IrValue& test = *condition.Resolve();
        if (test.HasImmediate()) return test.Type() == IrType::Bool && test.ImmediateBool();
        if (depth > 8u) return false;
        switch (test.Opcode()) {
            case IrOpcode::LogicalAnd: return AlwaysTrue(*test.Argument(0), depth + 1u) && AlwaysTrue(*test.Argument(1), depth + 1u);
            case IrOpcode::LogicalOr: return AlwaysTrue(*test.Argument(0), depth + 1u) || AlwaysTrue(*test.Argument(1), depth + 1u);
            case IrOpcode::LogicalNot: return NeverTrue(*test.Argument(0), depth + 1u);
            default: return false;
        }
    }

    const IrValue* FiresWhenNan(const IrValue& condition) const {
        const IrValue& test = *condition.Resolve();
        if (const IrValue* source = NanTestedF32Source(test)) return source;
        if (test.Opcode() != IrOpcode::LogicalAnd) return nullptr;
        if (AlwaysTrue(*test.Argument(1))) return FiresWhenNan(*test.Argument(0));
        if (AlwaysTrue(*test.Argument(0))) return FiresWhenNan(*test.Argument(1));
        return nullptr;
    }

private:
    std::uint8_t allOf(const IrValue& value) const {
        std::uint8_t result = 0u;
        for (std::size_t index = 0; index < value.ArgumentCount(); ++index) result |= Of(value.Argument(index));
        return result;
    }

    std::uint8_t compute(const IrValue& value) const {
        if (const IrValue* source = FlushedF32Source(value)) return Of(source);
        if (MaskedOperand(value, IrOpcode::BitwiseAnd32, 0x80000000u) != nullptr) return MayBeZero;
        if (const IrValue* operand = MaskedOperand(value, IrOpcode::BitwiseAnd32, 0x7fffffffu)) return Of(operand);
        if (const IrValue* operand = MaskedOperand(value, IrOpcode::BitwiseXor32, 0x80000000u)) return Of(operand);
        switch (value.Opcode()) {
            case IrOpcode::BitCastU32F32:
            case IrOpcode::BitCastF32U32:
            case IrOpcode::FPAbs32:
            case IrOpcode::FPNeg32:
                return Of(value.Argument(0));
            case IrOpcode::ConvertIToF32:
            case IrOpcode::ConvertUToF32:
            case IrOpcode::ConvertF32S32:
            case IrOpcode::ConvertF32U32:
                return MayBeZero;
            case IrOpcode::FPFloor32:
            case IrOpcode::FPCeil32:
            case IrOpcode::FPTrunc32:
            case IrOpcode::FPRoundEven32:
                return static_cast<std::uint8_t>((Of(value.Argument(0)) & (MayBeNan | MayBeInfinite)) | MayBeZero);
            case IrOpcode::FPFract32:
                return (Of(value.Argument(0)) & (MayBeNan | MayBeInfinite)) != 0u ? MayBeNan | MayBeZero : MayBeZero;
            case IrOpcode::FPSaturate32:
                return static_cast<std::uint8_t>((Of(value.Argument(0)) & MayBeNan) | MayBeZero);
            case IrOpcode::FPMin32:
            case IrOpcode::FPMax32:
            case IrOpcode::FPMinTri32:
            case IrOpcode::FPMaxTri32:
            case IrOpcode::FPMedTri32:
            case IrOpcode::Phi:
                return allOf(value);
            case IrOpcode::SelectU32:
            case IrOpcode::SelectF32:
                if (NeverTrue(*value.Argument(0))) return Of(value.Argument(2));
                return Of(value.Argument(1)) | Of(value.Argument(2));
            case IrOpcode::FPAdd32:
            case IrOpcode::FPSub32: {
                const std::uint8_t lhs = Of(value.Argument(0));
                const std::uint8_t rhs = Of(value.Argument(1));
                return arithmeticResult(((lhs | rhs) & MayBeNan) != 0u || (lhs & rhs & MayBeInfinite) != 0u);
            }
            case IrOpcode::FPMul32: {
                const std::uint8_t lhs = Of(value.Argument(0));
                const std::uint8_t rhs = Of(value.Argument(1));
                return arithmeticResult(((lhs | rhs) & MayBeNan) != 0u || invalidProduct(lhs, rhs));
            }
            case IrOpcode::FPMad32:
            case IrOpcode::FPFma32: {
                const std::uint8_t lhs = Of(value.Argument(0));
                const std::uint8_t rhs = Of(value.Argument(1));
                const std::uint8_t addend = Of(value.Argument(2));
                return arithmeticResult(((lhs | rhs | addend) & MayBeNan) != 0u || invalidSum(lhs, rhs, addend));
            }
            default:
                return AnyValue;
        }
    }

    static constexpr std::uint8_t Unseen = 0xffu;
    std::vector<std::uint8_t> classes;
};

bool sourcesCovered(const FloatClasses& classes, const IrValue& value, const std::unordered_set<const IrValue*>& tested, std::uint32_t depth) {
    const IrValue& current = *value.Resolve();
    if (classes.CannotBeNan(&current)) return true;
    if (depth > 8u) return false;
    const auto covered = [&](const IrValue* operand) { return classes.CannotBeNan(operand) || tested.contains(MagnitudeSource(*bitsOf(*operand))); };
    switch (current.Opcode()) {
        case IrOpcode::BitCastU32F32:
        case IrOpcode::BitCastF32U32:
            return sourcesCovered(classes, *current.Argument(0), tested, depth + 1u);
        case IrOpcode::SelectU32:
        case IrOpcode::SelectF32:
            return sourcesCovered(classes, *current.Argument(1), tested, depth + 1u) && sourcesCovered(classes, *current.Argument(2), tested, depth + 1u);
        case IrOpcode::FPAdd32:
        case IrOpcode::FPSub32:
            return (classes.Of(current.Argument(0)) & classes.Of(current.Argument(1)) & MayBeInfinite) == 0u && covered(current.Argument(0)) && covered(current.Argument(1));
        case IrOpcode::FPMul32:
            return !invalidProduct(classes.Of(current.Argument(0)), classes.Of(current.Argument(1))) && covered(current.Argument(0)) && covered(current.Argument(1));
        case IrOpcode::FPMad32:
        case IrOpcode::FPFma32:
            return !invalidSum(classes.Of(current.Argument(0)), classes.Of(current.Argument(1)), classes.Of(current.Argument(2))) && covered(current.Argument(0)) && covered(current.Argument(1)) && covered(current.Argument(2));
        default:
            return false;
    }
}

bool isDefaultNanCheck(const IrValue& value) {
    if (value.Opcode() != IrOpcode::SelectU32 || value.ArgumentCount() != 3u) return false;
    const IrValue* nan = value.Argument(1)->Resolve();
    return nan->HasImmediate() && nan->ImmediateU32() == 0xffc00000u && NanTestedF32Source(*value.Argument(0)->Resolve()) == value.Argument(2)->Resolve();
}

const IrValue* soleLiveUser(const IrValue& value) {
    const IrValue* sole = nullptr;
    for (const IrValue* user : value.Uses()) {
        if (!user->HasUses() && !user->MayHaveSideEffects()) continue;
        if (sole != nullptr) return nullptr;
        sole = user;
    }
    return sole;
}

bool defaultNanUnreachable(const FloatClasses& classes, const IrValue& check) {
    std::unordered_set<const IrValue*> tested;
    const IrValue* current = &check;
    while (const IrValue* user = soleLiveUser(*current)) {
        if ((user->Opcode() != IrOpcode::SelectU32 && user->Opcode() != IrOpcode::SelectF32) || user->ArgumentCount() != 3u) break;
        if (user->Argument(2)->Resolve() != current || user->Argument(1)->Resolve() == current) break;
        if (const IrValue* source = classes.FiresWhenNan(*user->Argument(0))) tested.insert(MagnitudeSource(*source));
        current = user;
    }
    return sourcesCovered(classes, *check.Argument(2), tested, 0u);
}

}

NanCheckEliminationStats NanCheckEliminator::Eliminate(IrProgram& program) const {
    NanCheckEliminationStats stats;
    const FloatClasses classes(program);
    for (auto* block : program.BlockOrder()) {
        for (auto* inst : block->Instructions()) {
            if (!inst->HasUses() || (inst->Opcode() != IrOpcode::SelectU32 && inst->Opcode() != IrOpcode::SelectF32) || inst->ArgumentCount() != 3u) continue;
            if (!classes.NeverTrue(*inst->Argument(0))) continue;
            inst->ReplaceAllUsesWith(inst->Argument(2));
            ++stats.removedSelects;
        }
    }
    std::vector<IrValue*> checks;
    for (auto* block : program.BlockOrder()) {
        for (auto* inst : block->Instructions()) {
            if (inst->HasUses() && isDefaultNanCheck(*inst)) checks.push_back(inst);
        }
    }
    for (IrValue* check : checks) {
        if (!defaultNanUnreachable(classes, *check)) continue;
        check->ReplaceAllUsesWith(check->Argument(2));
        ++stats.removedDefaultNans;
    }
    return stats;
}

}
