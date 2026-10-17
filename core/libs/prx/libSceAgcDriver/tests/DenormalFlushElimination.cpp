#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/DenormalFlushEliminator.hpp"
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>
#include <utility>

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

    IrValue& flush(IrValue& bits) {
        auto& exponent = emit(IrOpcode::BitwiseAnd32, IrType::U32, {&bits, &constant(0x7f800000u)});
        auto& denormal = emit(IrOpcode::IEqual32, IrType::Bool, {&exponent, &constant(0u)});
        auto& sign = emit(IrOpcode::BitwiseAnd32, IrType::U32, {&bits, &constant(0x80000000u)});
        return emit(IrOpcode::SelectU32, IrType::U32, {&denormal, &sign, &bits});
    }

    IrValue& phi(IrBlock& block, std::initializer_list<std::pair<IrBlock*, IrValue*>> incoming) {
        auto& created = program.CreateValue(IrOpcode::Phi, IrType::U32);
        block.AppendInstruction(&created);
        for (const auto& [predecessor, value] : incoming) created.AddPhiOperand(predecessor, value);
        return created;
    }

    void keep(IrValue& value) {
        (void)emit(IrOpcode::ReferenceU32, IrType::Void, {&value});
    }

    std::uint32_t eliminate() {
        return DenormalFlushEliminator{}.Eliminate(program).removedFlushes;
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

    passed &= run("a flush of a flushed value is removed and the first flush stays", [] {
        Builder b;
        auto& first = b.flush(b.unknown());
        auto& second = b.flush(first);
        b.keep(second);
        return b.eliminate() == 1u && !second.HasUses() && first.HasUses();
    });

    passed &= run("a flush of an unknown value stays", [] {
        Builder b;
        auto& flushed = b.flush(b.unknown());
        b.keep(flushed);
        return b.eliminate() == 0u && flushed.HasUses();
    });

    passed &= run("a sign change or absolute value of a flushed value needs no flush", [] {
        Builder b;
        auto& first = b.flush(b.unknown());
        auto& negated = b.flush(b.emit(IrOpcode::BitwiseXor32, IrType::U32, {&first, &b.constant(0x80000000u)}));
        auto& absolute = b.flush(b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&b.constant(0x7fffffffu), &first}));
        b.keep(negated);
        b.keep(absolute);
        return b.eliminate() == 2u && !negated.HasUses() && !absolute.HasUses();
    });

    passed &= run("a select of flushed values needs no flush, one with an unknown arm does", [] {
        Builder b;
        auto& condition = b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.unknown(), &b.constant(16u)});
        auto& both = b.flush(b.emit(IrOpcode::SelectU32, IrType::U32, {&condition, &b.flush(b.unknown()), &b.flush(b.unknown())}));
        auto& mixed = b.flush(b.emit(IrOpcode::SelectU32, IrType::U32, {&condition, &b.flush(b.unknown()), &b.unknown()}));
        b.keep(both);
        b.keep(mixed);
        return b.eliminate() == 1u && !both.HasUses() && mixed.HasUses();
    });

    passed &= run("normal and zero constants need no flush, a denormal constant does", [] {
        Builder b;
        auto& one = b.flush(b.constant(0x3f800000u));
        auto& negativeZero = b.flush(b.constant(0x80000000u));
        auto& denormal = b.flush(b.constant(0x00000001u));
        b.keep(one);
        b.keep(negativeZero);
        b.keep(denormal);
        return b.eliminate() == 2u && !one.HasUses() && !negativeZero.HasUses() && denormal.HasUses();
    });

    passed &= run("an integer or f16 converted to f32 needs no flush", [] {
        Builder b;
        auto& integer = b.flush(b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::ConvertF32U32, IrType::F32, {&b.unknown()})}));
        auto& half = b.flush(b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::ConvertF32F16, IrType::F32, {&b.emit(IrOpcode::BitCastF16U16, IrType::F16, {&b.emit(IrOpcode::ConvertU16U32, IrType::U16, {&b.unknown()})})})}));
        b.keep(integer);
        b.keep(half);
        return b.eliminate() == 2u && !integer.HasUses() && !half.HasUses();
    });

    passed &= run("a clamped flushed value needs no flush, an arithmetic result does", [] {
        Builder b;
        auto& first = b.emit(IrOpcode::BitCastF32U32, IrType::F32, {&b.flush(b.unknown())});
        auto& clamped = b.flush(b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPSaturate32, IrType::F32, {&first})}));
        auto& product = b.flush(b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPMul32, IrType::F32, {&first, &first})}));
        b.keep(clamped);
        b.keep(product);
        return b.eliminate() == 1u && !clamped.HasUses() && product.HasUses();
    });

    passed &= run("a min, max, min3, max3 or med3 of flushed values needs no flush, one with an unknown operand does", [] {
        Builder b;
        const auto value = [&](IrValue& bits) -> IrValue& { return b.emit(IrOpcode::BitCastF32U32, IrType::F32, {&bits}); };
        const auto flushedResult = [&](IrOpcode opcode, std::initializer_list<IrValue*> arguments) -> IrValue& {
            return b.flush(b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(opcode, IrType::F32, arguments)}));
        };
        auto& x = value(b.flush(b.unknown()));
        auto& y = value(b.flush(b.unknown()));
        auto& z = value(b.flush(b.unknown()));
        auto& raw = value(b.unknown());
        IrValue* removed[] = {
            &flushedResult(IrOpcode::FPMin32, {&x, &y}),
            &flushedResult(IrOpcode::FPMax32, {&y, &x}),
            &flushedResult(IrOpcode::FPMinTri32, {&x, &y, &z}),
            &flushedResult(IrOpcode::FPMaxTri32, {&z, &y, &x}),
            &flushedResult(IrOpcode::FPMedTri32, {&x, &z, &y}),
        };
        auto& mixedPair = flushedResult(IrOpcode::FPMax32, {&x, &raw});
        auto& mixedMedian = flushedResult(IrOpcode::FPMedTri32, {&x, &y, &raw});
        for (auto* flushed : removed) b.keep(*flushed);
        b.keep(mixedPair);
        b.keep(mixedMedian);
        const bool counted = b.eliminate() == 5u;
        bool kept = false;
        for (auto* flushed : removed) kept |= flushed->HasUses();
        return counted && !kept && mixedPair.HasUses() && mixedMedian.HasUses();
    });

    passed &= run("a floor or ceil needs no flush, even of an unknown value", [] {
        Builder b;
        auto& raw = b.emit(IrOpcode::BitCastF32U32, IrType::F32, {&b.unknown()});
        auto& floored = b.flush(b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPFloor32, IrType::F32, {&raw})}));
        auto& ceiled = b.flush(b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPCeil32, IrType::F32, {&raw})}));
        b.keep(floored);
        b.keep(ceiled);
        return b.eliminate() == 2u && !floored.HasUses() && !ceiled.HasUses();
    });

    passed &= run("a max carried around a loop of flushed values needs no flush, one with an unknown step keeps it", [] {
        const auto loop = [](bool flushedStep) {
            Builder b;
            IrBlock& head = *b.block;
            IrBlock& body = b.newBlock();
            IrBlock& exit = b.newBlock();
            head.AddBranch(&body);
            body.AddBranch(&body);
            body.AddBranch(&exit);
            auto& initial = b.flush(b.unknown());
            b.block = &body;
            auto& carried = b.program.CreateValue(IrOpcode::Phi, IrType::U32);
            body.AppendInstruction(&carried);
            auto& stepBits = flushedStep ? b.flush(b.unknown()) : b.unknown();
            auto& step = b.emit(IrOpcode::BitCastF32U32, IrType::F32, {&stepBits});
            auto& maximum = b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::FPMax32, IrType::F32, {&b.emit(IrOpcode::BitCastF32U32, IrType::F32, {&carried}), &step})});
            carried.AddPhiOperand(&head, &initial);
            carried.AddPhiOperand(&body, &maximum);
            auto& flushed = b.flush(maximum);
            b.block = &exit;
            b.keep(flushed);
            const auto removed = b.eliminate();
            return flushedStep ? removed == 1u && !flushed.HasUses() : removed == 0u && flushed.HasUses();
        };
        return loop(true) && loop(false);
    });

    passed &= run("values that only feed each other around a loop and take flushed values need no flush, a denormal constant keeps it", [] {
        const auto loop = [](std::uint32_t initialBits) {
            Builder b;
            IrBlock& head = *b.block;
            IrBlock& body = b.newBlock();
            IrBlock& exit = b.newBlock();
            head.AddBranch(&body);
            body.AddBranch(&body);
            body.AddBranch(&exit);
            b.block = &body;
            auto& first = b.program.CreateValue(IrOpcode::Phi, IrType::U32);
            auto& second = b.program.CreateValue(IrOpcode::Phi, IrType::U32);
            body.AppendInstruction(&first);
            body.AppendInstruction(&second);
            auto& converted = b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(IrOpcode::ConvertF32U32, IrType::F32, {&b.unknown()})});
            auto& active = b.emit(IrOpcode::IEqual32, IrType::Bool, {&b.unknown(), &b.constant(0u)});
            auto& chosen = b.emit(IrOpcode::SelectU32, IrType::U32, {&active, &converted, &second});
            first.AddPhiOperand(&head, &b.constant(initialBits));
            first.AddPhiOperand(&body, &chosen);
            second.AddPhiOperand(&head, &b.constant(0u));
            second.AddPhiOperand(&body, &first);
            auto& flushed = b.flush(chosen);
            b.block = &exit;
            b.keep(flushed);
            b.eliminate();
            return !flushed.HasUses();
        };
        return loop(0x3f800000u) && !loop(0x00000001u);
    });

    passed &= run("a phi of flushed values needs no flush", [] {
        Builder b;
        IrBlock& head = *b.block;
        IrBlock& body = b.newBlock();
        IrBlock& merge = b.newBlock();
        head.AddBranch(&body);
        head.AddBranch(&merge);
        body.AddBranch(&merge);
        auto& fromHead = b.flush(b.unknown());
        b.block = &body;
        auto& fromBody = b.flush(b.unknown());
        b.block = &merge;
        auto& merged = b.flush(b.phi(merge, {{&head, &fromHead}, {&body, &fromBody}}));
        b.keep(merged);
        return b.eliminate() == 1u && !merged.HasUses();
    });

    passed &= run("a loop phi needs no flush when every value it takes is flushed, and keeps it otherwise", [] {
        const auto loop = [](bool flushedBackEdge) {
            Builder b;
            IrBlock& head = *b.block;
            IrBlock& body = b.newBlock();
            IrBlock& exit = b.newBlock();
            head.AddBranch(&body);
            body.AddBranch(&body);
            body.AddBranch(&exit);
            auto& initial = b.flush(b.unknown());
            b.block = &body;
            auto& carried = b.program.CreateValue(IrOpcode::Phi, IrType::U32);
            body.AppendInstruction(&carried);
            auto& next = b.flush(carried);
            auto& backEdge = flushedBackEdge ? next : b.emit(IrOpcode::BitwiseXor32, IrType::U32, {&b.unknown(), &next});
            carried.AddPhiOperand(&head, &initial);
            carried.AddPhiOperand(&body, &backEdge);
            b.block = &exit;
            b.keep(next);
            b.eliminate();
            return !next.HasUses();
        };
        return loop(true) && !loop(false);
    });

    passed &= run("an output modifier's result needs no flush, a look-alike with another step does", [] {
        const auto modified = [](Builder& b, std::uint32_t omod, std::uint32_t step) -> IrValue& {
            auto& bits = b.unknown();
            auto& magnitude = b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&bits, &b.constant(0x7fffffffu)});
            auto& sign = b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&bits, &b.constant(0x80000000u)});
            const auto below = [&](std::uint32_t bound) -> IrValue& { return b.emit(IrOpcode::ULessThan32, IrType::Bool, {&magnitude, &b.constant(bound)}); };
            IrValue* scaled = nullptr;
            if (omod == 3u) {
                scaled = &b.emit(IrOpcode::SelectU32, IrType::U32, {&below(0x01000000u), &sign, &b.emit(IrOpcode::ISub32, IrType::U32, {&bits, &b.constant(step)})});
            } else {
                auto& infinity = b.emit(IrOpcode::BitwiseOr32, IrType::U32, {&sign, &b.constant(0x7f800000u)});
                scaled = &b.emit(IrOpcode::SelectU32, IrType::U32, {&below(0x7f800000u - step), &b.emit(IrOpcode::IAdd32, IrType::U32, {&bits, &b.constant(step)}), &infinity});
            }
            auto& small = b.emit(IrOpcode::SelectU32, IrType::U32, {&below(0x00800000u), &b.constant(0u), scaled});
            auto& special = b.emit(IrOpcode::UGreaterThan32, IrType::Bool, {&magnitude, &b.constant(0x7f7fffffu)});
            return b.flush(b.emit(IrOpcode::SelectU32, IrType::U32, {&special, &bits, &small}));
        };
        Builder b;
        auto& twice = modified(b, 1u, 0x00800000u);
        auto& fourTimes = modified(b, 2u, 0x01000000u);
        auto& half = modified(b, 3u, 0x00800000u);
        auto& lookAlike = modified(b, 1u, 0x00400000u);
        for (IrValue* flushed : {&twice, &fourTimes, &half, &lookAlike}) b.keep(*flushed);
        return b.eliminate() == 3u && !twice.HasUses() && !fourTimes.HasUses() && !half.HasUses() && lookAlike.HasUses();
    });

    passed &= run("rounded results need no flush, a fraction or a maximum only of flushed operands", [] {
        Builder b;
        const auto asFloat = [&](IrValue& bits) -> IrValue& { return b.emit(IrOpcode::BitCastF32U32, IrType::F32, {&bits}); };
        const auto ofFloat = [&](IrOpcode opcode, std::initializer_list<IrValue*> arguments) -> IrValue& {
            return b.flush(b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&b.emit(opcode, IrType::F32, arguments)}));
        };
        auto& floor = ofFloat(IrOpcode::FPFloor32, {&asFloat(b.unknown())});
        auto& ceil = ofFloat(IrOpcode::FPCeil32, {&asFloat(b.unknown())});
        auto& trunc = ofFloat(IrOpcode::FPTrunc32, {&asFloat(b.unknown())});
        auto& even = ofFloat(IrOpcode::FPRoundEven32, {&asFloat(b.unknown())});
        auto& fraction = ofFloat(IrOpcode::FPFract32, {&asFloat(b.flush(b.unknown()))});
        auto& rawFraction = ofFloat(IrOpcode::FPFract32, {&asFloat(b.unknown())});
        auto& maximum = ofFloat(IrOpcode::FPMax32, {&asFloat(b.flush(b.unknown())), &asFloat(b.flush(b.unknown()))});
        auto& median = ofFloat(IrOpcode::FPMedTri32, {&asFloat(b.flush(b.unknown())), &asFloat(b.flush(b.unknown())), &asFloat(b.flush(b.unknown()))});
        auto& rawMinimum = ofFloat(IrOpcode::FPMin32, {&asFloat(b.flush(b.unknown())), &asFloat(b.unknown())});
        for (IrValue* flushed : {&floor, &ceil, &trunc, &even, &fraction, &rawFraction, &maximum, &median, &rawMinimum}) b.keep(*flushed);
        return b.eliminate() == 7u && !floor.HasUses() && !ceil.HasUses() && !trunc.HasUses() && !even.HasUses() && !fraction.HasUses() && !maximum.HasUses() && !median.HasUses() && rawFraction.HasUses() && rawMinimum.HasUses();
    });

    passed &= run("a select that passes a NaN through needs no flush when its other arm needs none", [] {
        Builder b;
        const auto isNan = [&](IrValue& bits) -> IrValue& {
            return b.emit(IrOpcode::UGreaterThan32, IrType::Bool, {&b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&bits, &b.constant(0x7fffffffu)}), &b.constant(0x7f800000u)});
        };
        const auto signaling = [&](IrValue& bits) -> IrValue& {
            auto& exponent = b.emit(IrOpcode::IEqual32, IrType::Bool, {&b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&bits, &b.constant(0x7fc00000u)}), &b.constant(0x7f800000u)});
            auto& payload = b.emit(IrOpcode::INotEqual32, IrType::Bool, {&b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&bits, &b.constant(0x003fffffu)}), &b.constant(0u)});
            return b.emit(IrOpcode::LogicalAnd, IrType::Bool, {&exponent, &payload});
        };
        const auto quiet = [&](IrValue& bits) -> IrValue& { return b.emit(IrOpcode::BitwiseOr32, IrType::U32, {&bits, &b.constant(0x00400000u)}); };
        const auto select = [&](IrValue& condition, IrValue& passed, IrValue& other) -> IrValue& { return b.flush(b.emit(IrOpcode::SelectU32, IrType::U32, {&condition, &passed, &other})); };
        auto& source = b.unknown();
        auto& passes = select(isNan(source), source, b.flush(b.unknown()));
        auto& quiets = select(isNan(source), quiet(source), b.flush(b.unknown()));
        auto& signals = select(signaling(source), quiet(source), b.flush(b.unknown()));
        auto& otherValue = select(isNan(source), b.unknown(), b.flush(b.unknown()));
        auto& unknownArm = select(isNan(source), source, b.unknown());
        for (IrValue* flushed : {&passes, &quiets, &signals, &otherValue, &unknownArm}) b.keep(*flushed);
        return b.eliminate() == 3u && !passes.HasUses() && !quiets.HasUses() && !signals.HasUses() && otherValue.HasUses() && unknownArm.HasUses();
    });

    passed &= run("a product whose tiny results are replaced by their sign needs no flush, look-alikes do", [] {
        Builder b;
        const auto asFloat = [&](IrValue& bits) -> IrValue& { return b.emit(IrOpcode::BitCastF32U32, IrType::F32, {&bits}); };
        const auto asBits = [&](IrValue& value) -> IrValue& { return b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&value}); };
        const auto product = [&](std::uint32_t scale, std::uint32_t bound, bool sameFactor, bool zeroAddend) -> IrValue& {
            auto& lhsBits = b.flush(b.unknown());
            auto& rhsBits = b.flush(b.unknown());
            auto& lhs = asFloat(lhsBits);
            auto& rhs = asFloat(rhsBits);
            auto& other = asFloat(b.flush(b.unknown()));
            auto& exact = b.emit(IrOpcode::FPMul32, IrType::F32, {&lhs, &rhs});
            auto& lifted = b.emit(IrOpcode::FPMul32, IrType::F32, {&lhs, &b.constant(scale)});
            auto& scaled = b.emit(IrOpcode::FPMul32, IrType::F32, {&lifted, sameFactor ? &rhs : &other});
            auto& magnitude = b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&asBits(scaled), &b.constant(0x7fffffffu)});
            IrValue* tiny = &b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.emit(IrOpcode::ISub32, IrType::U32, {&magnitude, &b.constant(1u)}), &b.constant(bound)});
            if (zeroAddend) tiny = &b.emit(IrOpcode::LogicalAnd, IrType::Bool, {tiny, &b.emit(IrOpcode::IEqual32, IrType::Bool, {&b.unknown(), &b.constant(0u)})});
            auto& sign = b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&b.emit(IrOpcode::BitwiseXor32, IrType::U32, {&lhsBits, &rhsBits}), &b.constant(0x80000000u)});
            return b.flush(asBits(b.emit(IrOpcode::SelectF32, IrType::F32, {tiny, &asFloat(sign), &exact})));
        };
        auto& flushedProduct = product(0x5f800000u, 0x207fffffu, true, false);
        auto& otherScale = product(0x5f000000u, 0x207fffffu, true, false);
        auto& otherBound = product(0x5f800000u, 0x203fffffu, true, false);
        auto& otherFactor = product(0x5f800000u, 0x207fffffu, false, false);
        auto& withAddend = product(0x5f800000u, 0x207fffffu, true, true);
        for (IrValue* flushed : {&flushedProduct, &otherScale, &otherBound, &otherFactor, &withAddend}) b.keep(*flushed);
        return b.eliminate() == 1u && !flushedProduct.HasUses() && otherScale.HasUses() && otherBound.HasUses() && otherFactor.HasUses() && withAddend.HasUses();
    });

    passed &= run("a product by a constant whose scaled factor was folded needs no flush, a wrong folded factor keeps it", [] {
        Builder b;
        const auto asFloat = [&](IrValue& bits) -> IrValue& { return b.emit(IrOpcode::BitCastF32U32, IrType::F32, {&bits}); };
        const auto asBits = [&](IrValue& value) -> IrValue& { return b.emit(IrOpcode::BitCastU32F32, IrType::U32, {&value}); };
        const auto product = [&](std::uint32_t folded) -> IrValue& {
            auto& factor = b.constant(0x3b800000u);
            auto& rhsBits = b.flush(b.unknown());
            auto& rhs = asFloat(rhsBits);
            auto& exact = b.emit(IrOpcode::FPMul32, IrType::F32, {&factor, &rhs});
            auto& scaled = b.emit(IrOpcode::FPMul32, IrType::F32, {&b.constant(folded), &rhs});
            auto& magnitude = b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&asBits(scaled), &b.constant(0x7fffffffu)});
            auto& tiny = b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.emit(IrOpcode::ISub32, IrType::U32, {&magnitude, &b.constant(1u)}), &b.constant(0x207fffffu)});
            auto& sign = b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&b.emit(IrOpcode::BitwiseXor32, IrType::U32, {&factor, &rhsBits}), &b.constant(0x80000000u)});
            return b.flush(asBits(b.emit(IrOpcode::SelectF32, IrType::F32, {&tiny, &asFloat(sign), &exact})));
        };
        auto& exactFactor = product(0x5b800000u);
        auto& wrongFactor = product(0x5b000000u);
        b.keep(exactFactor);
        b.keep(wrongFactor);
        return b.eliminate() == 1u && !exactFactor.HasUses() && wrongFactor.HasUses();
    });

    passed &= run("a flush of a select is distributed over its arms when each arm is denormal-free or already flushed", [] {
        Builder b;
        auto& condition = b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.unknown(), &b.constant(16u)});
        auto& earlier = b.unknown();
        auto& earlierFlushed = b.flush(earlier);
        auto& fresh = b.flush(b.unknown());
        auto& distributed = b.flush(b.emit(IrOpcode::SelectU32, IrType::U32, {&condition, &fresh, &earlier}));
        auto& unflushedArm = b.flush(b.emit(IrOpcode::SelectU32, IrType::U32, {&condition, &fresh, &b.unknown()}));
        b.keep(earlierFlushed);
        b.keep(distributed);
        b.keep(unflushedArm);
        if (b.eliminate() != 1u || distributed.HasUses() || !unflushedArm.HasUses()) return false;
        for (const IrValue* use : earlierFlushed.Uses()) {
            if (use->Opcode() == IrOpcode::SelectU32 && use->Argument(0)->Resolve() == &condition && use->Argument(1)->Resolve() == &fresh && use->Argument(2)->Resolve() == &earlierFlushed) return true;
        }
        return false;
    });

    passed &= run("the sign bit of a value needs no flush", [] {
        Builder b;
        auto& sign = b.flush(b.emit(IrOpcode::BitwiseAnd32, IrType::U32, {&b.unknown(), &b.constant(0x80000000u)}));
        b.keep(sign);
        return b.eliminate() == 1u && !sign.HasUses();
    });

    return passed ? 0 : 1;
}
