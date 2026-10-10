#include "IntermediateRepresentation/IrBuilder.hpp"
#include "Optimization/SrtWalker.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace ShaderRecompiler;

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckDeep() {
    IrProgram program;
    auto& block = program.CreateBlock();
    IrBuilder builder(program);
    builder.SetInsertionPoint(block);
    auto* value = &builder.Emit(IrOpcode::LaneId, IrType::U32, {});
    auto& one = builder.Constant(1u);
    for (std::uint32_t depth = 0; depth < 65536u; ++depth) {
        value = &builder.Emit(IrOpcode::IAdd32, IrType::U32, {value, &one});
    }
    for (std::uint32_t depth = 0; depth < 256u; ++depth) {
        value = &builder.Emit(IrOpcode::IAdd32, IrType::U32, {value, value});
    }
    auto& handle = builder.Emit(IrOpcode::GetBufferResource, IrType::BufferResource, {value, value, &one, &one});
    SrtWalker{}.BuildPlan(program);
    Require(program.Resources().srtPlanComplete, "deep SRT planning did not finish");
    Require(program.Resources().srtReads.empty() && program.Metadata().dynamicReads.empty(), "deep SRT planning invented reads");
    Require(handle.Argument(0) == value && handle.Argument(1) == value, "deep SRT planning changed the descriptor dependency");
}

void CheckCycle(bool phiCycle) {
    IrProgram program;
    auto& block = program.CreateBlock();
    IrBuilder builder(program);
    builder.SetInsertionPoint(block);
    auto& one = builder.Constant(1u);
    auto& cycle = phiCycle ? builder.Emit(IrOpcode::Phi, IrType::U32, {}) : builder.Emit(IrOpcode::IAdd32, IrType::U32, {&one, &one});
    if (phiCycle) {
        cycle.AddPhiOperand(&block, &one);
        cycle.AddPhiOperand(&block, &cycle);
    } else {
        cycle.ReplaceArgument(0, &cycle);
    }
    static_cast<void>(builder.Emit(IrOpcode::GetBufferResource, IrType::BufferResource, {&cycle, &one, &one, &one}));
    try {
        SrtWalker{}.BuildPlan(program);
    } catch (const std::runtime_error& error) {
        if (!phiCycle && std::string(error.what()).find("without a phi") != std::string::npos) return;
        throw;
    }
    Require(phiCycle && program.Resources().srtPlanComplete, "SRT planning accepted a cycle without a phi");
}

void CheckDynamicOrder() {
    IrProgram program;
    auto& block = program.CreateBlock();
    IrBuilder builder(program);
    builder.SetInsertionPoint(block);
    program.Resources().memoryInfo.push_back({});
    program.Resources().memoryInfo.back().kind = ResourceKind::ScalarAddress;
    auto& zero = builder.Constant(0u);
    auto& offset = builder.Emit(IrOpcode::LaneId, IrType::U32, {});
    auto& active = builder.ConstantBool(true);
    auto& address = builder.Emit(IrOpcode::GetAddressResource, IrType::AddressResource, {&zero, &zero});
    auto& first = builder.Emit(IrOpcode::LoadAddressU32, IrType::U32, {&address, &offset, &zero, &active});
    first.SetFlags(MemoryFlags{0u, 4u});
    auto& dependent = builder.Emit(IrOpcode::GetAddressResource, IrType::AddressResource, {&first, &zero});
    auto& second = builder.Emit(IrOpcode::LoadAddressU32, IrType::U32, {&dependent, &offset, &zero, &active});
    second.SetFlags(MemoryFlags{0u, 8u});
    static_cast<void>(builder.Emit(IrOpcode::GetBufferResource, IrType::BufferResource, {&second, &first, &second, &zero}));
    SrtWalker{}.BuildPlan(program);
    const auto& reads = program.Metadata().dynamicReads;
    Require(reads.size() == 2u && reads[0] == &first && reads[1] == &second, "SRT planning changed dependency order or duplicated a shared read");
    Require(program.Resources().srtReads.empty(), "SRT planning flattened a dynamic offset");
}

void CheckStaticOrder() {
    IrProgram program;
    auto& block = program.CreateBlock();
    IrBuilder builder(program);
    builder.SetInsertionPoint(block);
    program.Resources().memoryInfo.resize(3u);
    for (auto& memory : program.Resources().memoryInfo) memory.kind = ResourceKind::ScalarAddress;
    auto& zero = builder.Constant(0u);
    auto& base = builder.Constant(0x1000u);
    auto& active = builder.ConstantBool(true);
    auto& address = builder.Emit(IrOpcode::GetAddressResource, IrType::AddressResource, {&base, &zero});
    auto& first = builder.Emit(IrOpcode::LoadAddressU32, IrType::U32, {&address, &zero, &zero, &active});
    first.SetFlags(MemoryFlags{0u, 4u});
    auto& dependent = builder.Emit(IrOpcode::GetAddressResource, IrType::AddressResource, {&first, &zero});
    auto& second = builder.Emit(IrOpcode::LoadAddressU32, IrType::U32, {&dependent, &zero, &zero, &active});
    second.SetFlags(MemoryFlags{1u, 8u});
    auto& duplicate = builder.Emit(IrOpcode::LoadAddressU32, IrType::U32, {&address, &zero, &zero, &active});
    duplicate.SetFlags(MemoryFlags{2u, 12u});
    auto& handle = builder.Emit(IrOpcode::GetBufferResource, IrType::BufferResource, {&second, &first, &duplicate, &zero});
    SrtWalker{}.BuildPlan(program);
    const auto& reads = program.Resources().srtReads;
    Require(reads.size() == 2u && reads[0].value == &first && reads[1].value == &second, "SRT planning changed flattened dependency order or duplicated an equivalent read");
    for (std::uint32_t index = 0; index < 3u; ++index) {
        const auto* value = handle.Argument(index)->Resolve();
        Require(value->Opcode() == IrOpcode::ReadConst && value->Argument(1)->ImmediateU32() == (index == 0u ? 1u : 0u), "SRT planning patched a descriptor with the wrong flat slot");
    }
    SrtRuntime runtime;
    runtime.readMemory = +[](void*, std::uint64_t address, std::uint32_t* value) {
        if (address == 0x1000u) { *value = 0x2000u; return true; }
        if (address == 0x2000u) { *value = 0xabcdef01u; return true; }
        return false;
    };
    std::vector<std::uint32_t> flat;
    SrtWalker{}.Walk(program.Resources(), runtime, flat);
    Require(flat.size() == 2u && flat[0] == 0x2000u && flat[1] == 0xabcdef01u, "SRT planning changed dependent memory reads");
}

}

int main(int argc, char** argv) {
    try {
        const auto mode = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
        if (mode != "--small") CheckDeep();
        if (mode != "--deep") {
            CheckCycle(true);
            CheckCycle(false);
            CheckDynamicOrder();
            CheckStaticOrder();
        }
        std::puts("SRT dependency planning tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
