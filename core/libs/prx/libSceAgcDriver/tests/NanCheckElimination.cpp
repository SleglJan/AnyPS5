#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/NanCheckEliminator.hpp"
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

using namespace ShaderRecompiler;

namespace {

struct Builder {
    IrProgram program;
    IrBlock* block = nullptr;

    Builder() {
        block = &newBlock();
        program.SetEntryBlock(*block);
    }

    IrBlock& newBlock() {
        auto& created = program.CreateBlock();
        program.BlockOrder().push_back(&created);
        return created;
    }

    IrValue& constant(std::uint32_t value) {
        auto& created = program.CreateValue(IrOpcode::Void, IrType::U32);
        created.SetImmediateU32(value);
        return created;
    }

    IrValue& emit(IrOpcode opcode, IrType type, std::initializer_list<IrValue*> arguments) {
        auto& created = program.CreateValue(opcode, type);
        for (auto* argument : arguments) created.AddArgument(argument);
        block->AppendInstruction(&created);
        return created;
    }

    IrValue& unknown() {
        return emit(IrOpcode::LaneId, IrType::U32, {});
    }

    IrValue& integer() {
        return emit(IrOpcode::BitCastU32F32, IrType::U32, {&emit(IrOpcode::ConvertF32U32, IrType::F32, {&unknown()})});
    }

    IrValue& asFloat(IrValue& bits) {
        return emit(IrOpcode::BitCastF32U32, IrType::F32, {&bits});
    }

    IrValue& magnitude(IrValue& bits) {
        return emit(IrOpcode::BitwiseAnd32, IrType::U32, {&bits, &constant(0x7fffffffu)});
    }

    IrValue& isNan(IrValue& bits) {
        return emit(IrOpcode::UGreaterThan32, IrType::Bool, {&magnitude(bits), &constant(0x7f800000u)});
    }

    IrValue& select(IrValue& condition, IrValue& whenTrue, IrValue& whenFalse) {
        return emit(IrOpcode::SelectU32, IrType::U32, {&condition, &whenTrue, &whenFalse});
    }

    IrValue& invalidProduct(IrValue& lhs, IrValue& rhs) {
        const auto infiniteTimesZero = [&](IrValue& infinite, IrValue& zero) -> IrValue& {
            return emit(IrOpcode::LogicalAnd, IrType::Bool, {&emit(IrOpcode::IEqual32, IrType::Bool, {&magnitude(infinite), &constant(0x7f800000u)}), &emit(IrOpcode::IEqual32, IrType::Bool, {&magnitude(zero), &constant(0u)})});
        };
        return emit(IrOpcode::LogicalOr, IrType::Bool, {&infiniteTimesZero(lhs, rhs), &infiniteTimesZero(rhs, lhs)});
    }

    struct Result {
        IrValue* check = nullptr;
        IrValue* value = nullptr;
    };

    Result nanResult(IrOpcode opcode, std::initializer_list<IrValue*> sources) {
        std::vector<IrValue*> floats;
        for (IrValue* source : sources) floats.push_back(&asFloat(*source));
        IrValue* operation = nullptr;
        if (floats.size() == 2u) {
            operation = &emit(opcode, IrType::F32, {floats[0], floats[1]});
        } else {
            operation = &emit(opcode, IrType::F32, {floats[0], floats[1], floats[2]});
        }
        auto& bits = emit(IrOpcode::BitCastU32F32, IrType::U32, {operation});
        auto& check = select(isNan(bits), constant(0xffc00000u), bits);
        IrValue* value = &check;
        std::vector<IrValue*> ordered(sources);
        for (std::size_t index = ordered.size(); index-- > 0u;) {
            IrValue* condition = &isNan(*ordered[index]);
            if (ordered.size() == 3u && index == 2u) condition = &emit(IrOpcode::LogicalAnd, IrType::Bool, {condition, &emit(IrOpcode::LogicalNot, IrType::Bool, {&invalidProduct(*ordered[0], *ordered[1])})});
            value = &select(*condition, *ordered[index], *value);
        }
        (void)emit(IrOpcode::ReferenceU32, IrType::Void, {value});
        return {&check, value};
    }

    void keep(IrValue& value) {
        (void)emit(IrOpcode::ReferenceU32, IrType::Void, {&value});
    }

    NanCheckEliminationStats eliminate() {
        return NanCheckEliminator{}.Eliminate(program);
    }
};

bool expect(const char* name, bool condition) {
    if (!condition) std::fprintf(stderr, "%s\n", name);
    return condition;
}

bool run(const char* name, const std::function<bool()>& test) {
    try {
        return expect(name, test());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: %s\n", name, error.what());
        return false;
    }
}

}

int main() {
    bool passed = true;

    passed &= run("a NaN test of a value that cannot be NaN selects the other arm, of any other value it stays", [] {
        Builder b;
        auto& integer = b.integer();
        auto& rounded = b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPFloor32, IrType::F32, {&b.asFloat(integer)})});
        auto& unknown = b.unknown();
        auto& fromInteger = b.select(b.isNan(integer), integer, b.unknown());
        auto& fromRounded = b.select(b.emit(IrOpcode::FPIsNan32, IrType::Bool, {&b.asFloat(rounded)}), rounded, b.unknown());
        auto& fromUnknown = b.select(b.isNan(unknown), unknown, b.unknown());
        b.keep(fromInteger);
        b.keep(fromRounded);
        b.keep(fromUnknown);
        return b.eliminate().removedSelects == 2u && !fromInteger.HasUses() && !fromRounded.HasUses() && fromUnknown.HasUses();
    });

    passed &= run("an infinity or zero test of a value that cannot be one selects the other arm", [] {
        Builder b;
        auto& choice = b.emit(IrOpcode::IEqual32, IrType::Bool, {&b.unknown(), &b.unknown()});
        auto& normal = b.select(choice, b.constant(0x3fc00000u), b.constant(0xc1200000u));
        auto& infinite = b.select(b.emit(IrOpcode::IEqual32, IrType::Bool, {&b.magnitude(b.integer()), &b.constant(0x7f800000u)}), b.unknown(), b.unknown());
        auto& zero = b.select(b.emit(IrOpcode::IEqual32, IrType::Bool, {&b.magnitude(normal), &b.constant(0u)}), b.unknown(), b.unknown());
        auto& integerZero = b.select(b.emit(IrOpcode::IEqual32, IrType::Bool, {&b.magnitude(b.integer()), &b.constant(0u)}), b.unknown(), b.unknown());
        b.keep(normal);
        b.keep(infinite);
        b.keep(zero);
        b.keep(integerZero);
        return b.eliminate().removedSelects == 2u && normal.HasUses() && !infinite.HasUses() && !zero.HasUses() && integerZero.HasUses();
    });

    passed &= run("an operation that cannot make a NaN of its operands loses its default NaN", [] {
        Builder b;
        const auto checked = [&](IrOpcode opcode, std::initializer_list<IrValue*> sources) { return b.nanResult(opcode, sources).check; };
        IrValue* plusOne = checked(IrOpcode::FPAdd32, {&b.unknown(), &b.constant(0x3f800000u)});
        IrValue* minusInteger = checked(IrOpcode::FPSub32, {&b.unknown(), &b.integer()});
        IrValue* half = checked(IrOpcode::FPMul32, {&b.constant(0x3f000000u), &b.unknown()});
        IrValue* fused = checked(IrOpcode::FPFma32, {&b.unknown(), &b.constant(0x40000000u), &b.constant(0x3f800000u)});
        IrValue* fusedAddend = checked(IrOpcode::FPFma32, {&b.integer(), &b.constant(0x40000000u), &b.unknown()});
        IrValue* fusedInfinities = checked(IrOpcode::FPFma32, {&b.unknown(), &b.constant(0x40000000u), &b.unknown()});
        IrValue* sum = checked(IrOpcode::FPAdd32, {&b.unknown(), &b.unknown()});
        IrValue* zeroTimes = checked(IrOpcode::FPMul32, {&b.constant(0u), &b.unknown()});
        IrValue* infinityTimes = checked(IrOpcode::FPMul32, {&b.unknown(), &b.constant(0x7f800000u)});
        IrValue* product = checked(IrOpcode::FPFma32, {&b.integer(), &b.unknown(), &b.constant(0x3f800000u)});
        const auto stats = b.eliminate();
        return stats.removedDefaultNans == 5u && !plusOne->HasUses() && !minusInteger->HasUses() && !half->HasUses() && !fused->HasUses() && !fusedAddend->HasUses() &&
            fusedInfinities->HasUses() && sum->HasUses() && zeroTimes->HasUses() && infinityTimes->HasUses() && product->HasUses();
    });

    passed &= run("a multiply-add that cannot make a NaN of its operands loses its default NaN and keeps its class", [] {
        Builder b;
        const auto checked = [&](std::initializer_list<IrValue*> sources) { return b.nanResult(IrOpcode::FPMad32, sources).check; };
        IrValue* finite = checked({&b.unknown(), &b.constant(0x40000000u), &b.constant(0x3f800000u)});
        IrValue* integerAddend = checked({&b.integer(), &b.constant(0x40000000u), &b.unknown()});
        IrValue* infinities = checked({&b.unknown(), &b.constant(0x40000000u), &b.unknown()});
        IrValue* integerProduct = checked({&b.integer(), &b.unknown(), &b.constant(0x3f800000u)});
        auto& mad = b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPMad32, IrType::F32, {&b.asFloat(b.integer()), &b.asFloat(b.constant(0x40000000u)), &b.asFloat(b.constant(0x3f800000u))})});
        auto& tested = b.select(b.isNan(mad), mad, b.unknown());
        auto& open = b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPMad32, IrType::F32, {&b.asFloat(b.unknown()), &b.asFloat(b.integer()), &b.asFloat(b.integer())})});
        auto& kept = b.select(b.isNan(open), open, b.unknown());
        b.keep(tested);
        b.keep(kept);
        const auto stats = b.eliminate();
        return stats.removedDefaultNans == 2u && !finite->HasUses() && !integerAddend->HasUses() && infinities->HasUses() && integerProduct->HasUses() && !tested.HasUses() && kept.HasUses();
    });

    passed &= run("a default NaN stays when an operand that can be NaN is not tested after it", [] {
        Builder b;
        auto& source = b.unknown();
        auto& bits = b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPAdd32, IrType::F32, {&b.asFloat(source), &b.asFloat(b.constant(0x3f800000u))})});
        auto& check = b.select(b.isNan(bits), b.constant(0xffc00000u), bits);
        auto& other = b.select(b.isNan(b.unknown()), b.unknown(), check);
        b.keep(other);
        auto& shared = b.unknown();
        auto& sharedBits = b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPAdd32, IrType::F32, {&b.asFloat(shared), &b.asFloat(b.constant(0x3f800000u))})});
        auto& sharedCheck = b.select(b.isNan(sharedBits), b.constant(0xffc00000u), sharedBits);
        b.keep(b.select(b.isNan(shared), shared, sharedCheck));
        b.keep(sharedCheck);
        return b.eliminate().removedDefaultNans == 0u && check.HasUses() && sharedCheck.HasUses();
    });

    passed &= run("values that cannot be NaN stay so through phis, a loop that adds an infinity does not", [] {
        const auto loop = [](bool finiteStep) {
            Builder b;
            IrBlock& entry = *b.block;
            IrBlock& body = b.newBlock();
            IrBlock& exit = b.newBlock();
            entry.AddBranch(&body);
            body.AddBranch(&body);
            body.AddBranch(&exit);
            auto& initial = b.integer();
            b.block = &body;
            auto& carried = b.program.CreateValue(IrOpcode::Phi, IrType::U32);
            body.AppendInstruction(&carried);
            auto& step = b.constant(finiteStep ? 0x3f800000u : 0x7f800000u);
            auto& next = b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPSub32, IrType::F32, {&b.asFloat(carried), &b.asFloat(step)})});
            carried.AddPhiOperand(&entry, &initial);
            carried.AddPhiOperand(&body, &next);
            b.block = &exit;
            auto& tested = b.select(b.isNan(carried), carried, b.unknown());
            b.keep(tested);
            (void)b.eliminate();
            return !tested.HasUses();
        };
        return loop(true) && !loop(false);
    });

    return passed ? 0 : 1;
}
