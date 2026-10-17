#include "IntermediateRepresentation/IrBuilder.hpp"
#include "Optimization/ConstantFolder.hpp"
#include "Optimization/DeadCodeEliminator.hpp"
#include <bit>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ShaderRecompiler;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", what.c_str());
        ++failures;
    }
}

struct Case {
    std::uint32_t lhs;
    std::uint32_t rhs;
    bool folds;
    std::uint32_t product;
    const char* what;
};

void testConstantProducts() {
    const std::vector<Case> cases{
        {0x3b800000u, 0x5f800000u, true, 0x5b800000u, "1/256 times 2^64"},
        {0x3fc00000u, 0xc0000000u, true, 0xc0400000u, "1.5 times -2"},
        {0x00000000u, 0x40a00000u, true, 0x00000000u, "zero times 5"},
        {0x80000000u, 0x40a00000u, true, 0x80000000u, "minus zero times 5"},
        {0x0d800000u, 0x30800000u, false, 0u, "a product that would be denormal"},
        {0x0d800000u, 0x0d800000u, false, 0u, "a product that would round to zero"},
        {0x71800000u, 0x71800000u, false, 0u, "a product that would overflow"},
        {0x00000001u, 0x40000000u, false, 0u, "a denormal operand"},
        {0x7fc00000u, 0x40000000u, false, 0u, "a NaN operand"},
        {0x7f800000u, 0x40000000u, false, 0u, "an infinite operand"},
    };
    for (const auto& item : cases) {
        IrProgram program;
        auto& block = program.CreateBlock();
        program.BlockOrder().push_back(&block);
        program.SetEntryBlock(block);
        BlockInfo info;
        info.terminator.kind = TerminatorKind::Return;
        program.Metadata().blockInfo.push_back(info);
        IrBuilder builder(program);
        builder.SetInsertionPoint(block);
        auto& product = builder.Emit(IrOpcode::FPMul32, IrType::F32, {&builder.ConstantF32(std::bit_cast<float>(item.lhs)), &builder.ConstantF32(std::bit_cast<float>(item.rhs))});
        auto& reference = builder.Emit(IrOpcode::ReferenceU32, IrType::Void, {&builder.Emit(IrOpcode::BitCastU32F32, IrType::U32, {&product})});
        ConstantFolder{}.Fold(program);
        DeadCodeEliminator{}.RemoveIdentities(program);
        DeadCodeEliminator{}.Eliminate(program);
        const IrValue* result = reference.Argument(0)->Resolve();
        if (item.folds) {
            check(result->HasImmediate() && result->ImmediateU32() == item.product, std::string(item.what) + ": the product of two constants was not folded to its exact value");
        } else {
            const bool kept = !result->HasImmediate() && result->Opcode() == IrOpcode::BitCastU32F32 && result->Argument(0)->Resolve()->Opcode() == IrOpcode::FPMul32;
            check(kept, std::string(item.what) + ": a product the host might compute differently was folded");
        }
    }
}

}

int main() {
    testConstantProducts();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
