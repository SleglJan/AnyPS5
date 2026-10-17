#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/DeadCodeEliminator.hpp"
#include "Optimization/FloatSelectRetyper.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>

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

    IrValue& phi() {
        auto& created = program.CreateValue(IrOpcode::Phi, IrType::U32);
        block->AppendInstruction(&created);
        return created;
    }

    IrValue& unknown() {
        return emit(IrOpcode::LaneId, IrType::U32, {});
    }

    IrValue& condition() {
        return emit(IrOpcode::IEqual32, IrType::Bool, {&unknown(), &constant(0u)});
    }

    IrValue& real() {
        auto& value = view(unknown());
        return emit(IrOpcode::FMul32, IrType::F32, {&value, &value});
    }

    IrValue& bits(IrValue& value) {
        return emit(IrOpcode::BitCastU32F32, IrType::U32, {&value});
    }

    IrValue& view(IrValue& value) {
        return emit(IrOpcode::BitCastF32U32, IrType::F32, {&value});
    }

    IrValue& useReal(IrValue& value) {
        auto& sum = emit(IrOpcode::FAdd32, IrType::F32, {&value, &value});
        keep(bits(sum));
        return sum;
    }

    void keep(IrValue& value) {
        (void)emit(IrOpcode::ReferenceU32, IrType::Void, {&value});
    }

    FloatSelectRetypeStats retype() {
        const auto stats = FloatSelectRetyper{}.Retype(program);
        DeadCodeEliminator{}.Eliminate(program);
        return stats;
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

    passed &= run("a select of floats read as a float is a float select", [] {
        Builder b;
        auto& first = b.real();
        auto& second = b.real();
        auto& select = b.emit(IrOpcode::SelectU32, IrType::U32, {&b.condition(), &b.bits(first), &b.bits(second)});
        auto& sum = b.useReal(b.view(select));
        const auto stats = b.retype();
        IrValue* retyped = sum.Argument(0);
        return stats.retypedValues == 1u && retyped->Opcode() == IrOpcode::SelectF32 && retyped->Type() == IrType::F32 && retyped->Argument(1) == &first &&
            retyped->Argument(2) == &second && sum.Argument(1) == retyped && select.Parent() == nullptr;
    });

    passed &= run("a loop-carried float under a select chain is a float phi and the integer cycle goes", [] {
        Builder b;
        IrBlock& entry = *b.block;
        IrBlock& header = b.newBlock();
        IrBlock& exit = b.newBlock();
        entry.AddBranch(&header);
        header.AddBranch(&header);
        header.AddBranch(&exit);
        b.block = &header;
        auto& carried = b.phi();
        auto& active = b.condition();
        auto& product = b.real();
        auto& merged = b.emit(IrOpcode::SelectU32, IrType::U32, {&active, &b.bits(product), &carried});
        auto& sum = b.useReal(b.view(merged));
        carried.AddPhiOperand(&entry, &b.constant(0x3f800000u));
        carried.AddPhiOperand(&header, &merged);
        const auto stats = b.retype();
        IrValue* select = sum.Argument(0);
        IrValue* loop = select->Argument(2);
        return stats.retypedValues == 2u && select->Opcode() == IrOpcode::SelectF32 && select->Argument(1) == &product && loop->IsPhi() && loop->Type() == IrType::F32 &&
            loop->Argument(0)->HasImmediate() && loop->Argument(0)->ImmediateF32() == 1.0f && loop->Argument(1) == select && carried.Parent() == nullptr &&
            merged.Parent() == nullptr;
    });

    passed &= run("an integer reader of a float select reads it through one bit cast", [] {
        Builder b;
        auto& first = b.real();
        auto& second = b.real();
        auto& select = b.emit(IrOpcode::SelectU32, IrType::U32, {&b.condition(), &b.bits(first), &b.bits(second)});
        auto& sum = b.useReal(b.view(select));
        auto& masked = b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&select, &b.constant(0x7fffffffu)});
        b.keep(masked);
        const auto stats = b.retype();
        IrValue* retyped = sum.Argument(0);
        IrValue* back = masked.Argument(0);
        return stats.retypedValues == 1u && retyped->Opcode() == IrOpcode::SelectF32 && back->Opcode() == IrOpcode::BitCastU32F32 && back->Argument(0) == retyped;
    });

    passed &= run("a select of integers read once as a float stays an integer select", [] {
        Builder b;
        auto& first = b.unknown();
        auto& second = b.unknown();
        auto& select = b.emit(IrOpcode::SelectU32, IrType::U32, {&b.condition(), &first, &second});
        auto& view = b.view(select);
        b.useReal(view);
        return !b.retype().Changed() && view.Argument(0) == &select && select.Parent() != nullptr;
    });

    passed &= run("a select used as a select condition source keeps its integer form for that use", [] {
        Builder b;
        auto& first = b.real();
        auto& second = b.real();
        auto& select = b.emit(IrOpcode::SelectU32, IrType::U32, {&b.condition(), &b.bits(first), &b.bits(second)});
        b.useReal(b.view(select));
        auto& test = b.emit(IrOpcode::IEqual32, IrType::Bool, {&select, &b.constant(0u)});
        auto& chosen = b.emit(IrOpcode::SelectU32, IrType::U32, {&test, &b.unknown(), &b.unknown()});
        b.keep(chosen);
        b.retype();
        IrValue* compared = test.Argument(0);
        return compared->Type() == IrType::U32 && compared->Opcode() == IrOpcode::BitCastU32F32 && compared->Argument(0)->Opcode() == IrOpcode::SelectF32 &&
            chosen.Opcode() == IrOpcode::SelectU32;
    });

    passed &= run("a select with a denormal or NaN constant stays an integer select", [] {
        Builder b;
        auto& denormal = b.emit(IrOpcode::SelectU32, IrType::U32, {&b.condition(), &b.bits(b.real()), &b.constant(0x00000001u)});
        auto& nan = b.emit(IrOpcode::SelectU32, IrType::U32, {&b.condition(), &b.bits(b.real()), &b.constant(0x7fc00001u)});
        auto& normal = b.emit(IrOpcode::SelectU32, IrType::U32, {&b.condition(), &b.bits(b.real()), &b.constant(0x80000000u)});
        auto& denormalView = b.view(denormal);
        auto& nanView = b.view(nan);
        auto& normalSum = b.useReal(b.view(normal));
        b.useReal(denormalView);
        b.useReal(nanView);
        const auto stats = b.retype();
        return stats.retypedValues == 1u && denormalView.Argument(0) == &denormal && nanView.Argument(0) == &nan && normalSum.Argument(0)->Opcode() == IrOpcode::SelectF32 &&
            normalSum.Argument(0)->Argument(2)->ImmediateF32() == -0.0f && std::signbit(normalSum.Argument(0)->Argument(2)->ImmediateF32());
    });

    return passed ? 0 : 1;
}
