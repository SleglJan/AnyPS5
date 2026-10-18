#include "BdaShader.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvImageEmitter.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "IntermediateRepresentation/IrBuilder.hpp"
#include <stdexcept>
#include <string>
#if ANYPS5_ENABLE_SPIRV_TOOLS
#include "SpirvBackend/SpirvOptimizer.hpp"
#endif

namespace {

std::vector<std::uint32_t> Structured(std::vector<std::uint32_t> words) {
    for (std::size_t at = 5; at < words.size();) {
        const auto opcode = words[at] & 0xffffu;
        const auto count = words[at] >> 16u;
        if (count == 0u || at + count > words.size()) throw std::runtime_error("a BDA test module has a malformed instruction");
        if (opcode == spv::OpLoopMerge || opcode == spv::OpSelectionMerge) {
            const auto next = at + count < words.size() ? words[at + count] & 0xffffu : 0u;
            const bool branch = opcode == spv::OpLoopMerge ? next == spv::OpBranch || next == spv::OpBranchConditional : next == spv::OpBranchConditional || next == spv::OpSwitch;
            if (!branch) throw std::runtime_error("a BDA test module has a merge instruction that does not immediately precede its branch, at word " + std::to_string(at));
        }
        at += count;
    }
    return words;
}

template<typename TEmit>
std::vector<std::uint32_t> MakeShader(std::uint64_t address, bool coherent, bool stops, const TEmit& emit, std::uint32_t extracted = 0, bool writes = false) {
    using namespace ShaderRecompiler;
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesDma = true;
    program.Info().bdaWrites = writes;
    if (coherent) {
        MemoryInfo memory;
        memory.coherent = true;
        program.Resources().memoryInfo = {memory};
    }
    SpirvEmitterState state(program, {});
    EmitBaseHeader(state.module, program);
    const auto define = [&](std::uint32_t binding) {
        const auto variable = state.module.DefineGlobalVariable(TypeStorageBufferPointer(state), spv::StorageClassStorageBuffer);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0u);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
        return variable;
    };
    state.bdaPagetableVariable = define(0);
    state.faultBufferVariable = define(1);
    const auto output = define(2);
    std::vector<std::uint32_t> interfaceVariables;
    if (!stops) {
        interfaceVariables.push_back(state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassInput, TypeU32(state)), spv::StorageClassInput));
        state.module.AddAnnotation(spv::OpDecorate, interfaceVariables[0], spv::DecorationBuiltIn, spv::BuiltInLocalInvocationIndex);
    }
    state.bdaStopsInvocations = stops;
    DefineGetBdaPointer(state);
    const auto main = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), main, spv::FunctionControlMaskNone, TypeFunction(state));
    EmitLabel(state, state.module.AllocateId());
    SpirvValueEmitContext ctx(state);
    auto& instruction = writes ? program.CreateValue(IrOpcode::StoreAddressU32, IrType::Void) : extracted != 0u ? program.CreateValue(IrOpcode::LoadAddressU32x4, IrType::U32x4) : program.CreateValue(IrOpcode::LoadAddressU32, IrType::U32);
    IrBuilder ir(program);
    for (std::uint32_t dword = 0; dword < 4u; ++dword) {
        if ((extracted & (1u << dword)) == 0u) continue;
        auto& extract = program.CreateValue(IrOpcode::CompositeExtractU32x4, IrType::U32);
        extract.AddArgument(&instruction);
        extract.AddArgument(&ir.Constant(dword));
    }
    MemoryFlags flags{};
    flags.pc = 0x1234;
    instruction.SetFlags(flags);
    auto base = BdaConstant(state, address);
    if (!stops) {
        const auto index = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), index, interfaceVariables[0]);
        base = Binary(state, spv::OpIAdd, TypeScalarU64(state), base, Unary(state, spv::OpUConvert, TypeScalarU64(state), index));
    }
    const auto values = emit(ctx, instruction, base);
    for (std::uint32_t index = 0; index < values.size(); ++index) state.module.AddFunction(spv::OpStore, BdaWord(state, output, ConstantU32(state, index)), values[index]);
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.AddExecutionMode(main, spv::ExecutionModeLocalSize, 1u, 1u, 1u);
    state.module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", interfaceVariables);
    return Structured(state.module.Finalize());
}

}

std::vector<std::uint32_t> MakeBdaTranslateTestShader(std::uint32_t count) {
    using namespace ShaderRecompiler;
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesDma = true;
    program.Info().bdaWrites = true;
    SpirvEmitterState state(program, {});
    EmitBaseHeader(state.module, program);
    const auto define = [&](std::uint32_t binding) {
        const auto variable = state.module.DefineGlobalVariable(TypeStorageBufferPointer(state), spv::StorageClassStorageBuffer);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0u);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
        return variable;
    };
    state.bdaPagetableVariable = define(0);
    state.faultBufferVariable = define(1);
    const auto io = define(2);
    const auto invocation = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassInput, TypeU32Vector(state, 3)), spv::StorageClassInput);
    state.module.AddAnnotation(spv::OpDecorate, invocation, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
    DefineGetBdaPointer(state);
    const auto main = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), main, spv::FunctionControlMaskNone, TypeFunction(state));
    EmitLabel(state, state.module.AllocateId());
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto ids = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32Vector(state, 3), ids, invocation);
    const auto index = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u32, index, ids, 0u);
    EmitIfCondition(state, Binary(state, spv::OpULessThan, TypeBool(state), index, ConstantU32(state, count)), [&] {
        const auto word = [&](std::uint32_t at) { return Binary(state, spv::OpIAdd, u32, Binary(state, spv::OpShiftLeftLogical, u32, index, ConstantU32(state, 2u)), ConstantU32(state, at)); };
        const auto load = [&](std::uint32_t at) {
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, u32, value, BdaWord(state, io, word(at)));
            return value;
        };
        const auto address = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, load(0u)), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, load(1u)), BdaConstant(state, 32u)));
        const auto translated = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeStruct, u64, u32), translated, state.bdaTranslateFunction, address, load(2u), load(3u));
        const auto device = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, device, translated, 0u);
        const auto reason = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, reason, translated, 1u);
        const auto results = count * 4u;
        state.module.AddFunction(spv::OpStore, BdaWord(state, io, word(results)), Unary(state, spv::OpUConvert, u32, device));
        state.module.AddFunction(spv::OpStore, BdaWord(state, io, word(results + 1u)), Unary(state, spv::OpUConvert, u32, Binary(state, spv::OpShiftRightLogical, u64, device, BdaConstant(state, 32u))));
        state.module.AddFunction(spv::OpStore, BdaWord(state, io, word(results + 2u)), reason);
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.AddExecutionMode(main, spv::ExecutionModeLocalSize, 64u, 1u, 1u);
    state.module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", std::vector<std::uint32_t>{invocation});
    return Structured(state.module.Finalize());
}

std::vector<std::uint32_t> MakeBdaTestShader(std::uint64_t address, std::uint32_t bits, std::int64_t offset) {
    using namespace ShaderRecompiler;
    return MakeShader(address, false, true, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        if (offset != 0) base = AddBdaAddress(ctx, instruction, base, BdaConstant(ctx.state, offset < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(offset) : static_cast<std::uint64_t>(offset)), offset < 0);
        return std::vector<std::uint32_t>{EmitBdaRead(ctx, instruction, base, bits)};
    });
}

std::vector<std::uint32_t> MakeBdaSpanReadTestShader(std::uint64_t address, std::uint32_t offset, std::uint32_t extracted, bool coherent, bool stops) {
    using namespace ShaderRecompiler;
    return MakeShader(address, coherent, stops, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        const auto values = EmitBdaDwordReads(ctx, instruction, base, offset, 4u);
        return std::vector<std::uint32_t>(values.begin(), values.end());
    }, extracted);
}

std::vector<std::uint32_t> MakeBdaCachedReadTestShader(const std::vector<BdaCachedRead>& reads) {
    using namespace ShaderRecompiler;
    return MakeShader(0u, false, false, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t) {
        auto& state = ctx.state;
        const auto vector = TypeU32Vector(state, 4);
        std::vector<std::uint32_t> values;
        for (const auto& read : reads) {
            const auto label = state.currentLabel;
            state.module.BeginHelperFunction();
            const auto helper = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunction, vector, helper, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, vector));
            EmitLabel(state, state.module.AllocateId());
            const auto loaded = EmitBdaDwordReads(ctx, instruction, BdaConstant(state, read.address), 0u, read.dwords, true);
            std::array<std::uint32_t, 4> parts{};
            for (std::uint32_t dword = 0; dword < 4u; ++dword) parts[dword] = dword < read.dwords ? loaded[dword] : ConstantU32(state, 0u);
            const auto composite = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, vector, composite, parts[0], parts[1], parts[2], parts[3]);
            state.module.AddFunction(spv::OpReturnValue, composite);
            state.module.AddFunction(spv::OpFunctionEnd);
            state.module.EndHelperFunction();
            state.currentLabel = label;
            const auto called = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, vector, called, helper);
            for (std::uint32_t dword = 0; dword < read.dwords; ++dword) {
                values.push_back(state.module.AllocateId());
                state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), values.back(), called, dword);
            }
        }
        return values;
    });
}

std::vector<std::uint32_t> MakeBdaDwordWriteTestShader(std::uint64_t address, std::uint32_t dwords, const std::uint32_t* values) {
    using namespace ShaderRecompiler;
    return MakeShader(address, false, true, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        auto& state = ctx.state;
        std::vector<std::uint32_t> parts;
        for (std::uint32_t dword = 0; dword < dwords; ++dword) parts.push_back(ConstantU32(state, values[dword]));
        const auto composite = state.module.AllocateId();
        if (dwords == 2u) state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 2u), composite, parts[0], parts[1]);
        else if (dwords == 3u) state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 3u), composite, parts[0], parts[1], parts[2]);
        else state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), composite, parts[0], parts[1], parts[2], parts[3]);
        EmitBdaDwordWrites(ctx, instruction, base, 0u, dwords, composite);
        return std::vector<std::uint32_t>{ConstantU32(state, 1u)};
    }, 0u, true);
}

std::vector<std::uint32_t> MakeBdaDwordReadTestShader(std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops) {
    using namespace ShaderRecompiler;
    return MakeShader(address, coherent, stops, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        const auto values = EmitBdaDwordReads(ctx, instruction, base, 0u, dwords, true);
        return std::vector<std::uint32_t>(values.begin(), values.begin() + dwords);
    });
}

std::vector<std::uint32_t> MakeBdaStoreTestShader(std::uint64_t address, std::uint32_t value, std::uint32_t bits, bool coherent) {
    using namespace ShaderRecompiler;
    auto shader = MakeShader(address, coherent, true, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        EmitBdaStore(ctx, instruction, base, ConstantU32(ctx.state, value), bits);
        return std::vector<std::uint32_t>{ConstantU32(ctx.state, 1u)};
    }, 0u, true);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    shader = ValidateAndOptimizeSpirv(shader, VK_API_VERSION_1_1, shader[1], false, false);
#endif
    return shader;
}

std::vector<std::uint32_t> MakeBdaLaneStoreTestShader(std::uint64_t address, std::uint64_t stride, std::uint64_t active, BdaLaneStore store) {
    using namespace ShaderRecompiler;
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesDma = true;
    program.Info().bdaWrites = true;
    SpirvEmitterState state(program, {});
    EmitBaseHeader(state.module, program);
    const auto define = [&](std::uint32_t binding) {
        const auto variable = state.module.DefineGlobalVariable(TypeStorageBufferPointer(state), spv::StorageClassStorageBuffer);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0u);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
        return variable;
    };
    state.bdaPagetableVariable = define(0);
    state.faultBufferVariable = define(1);
    define(2);
    const auto invocation = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassInput, TypeU32Vector(state, 3)), spv::StorageClassInput);
    state.module.AddAnnotation(spv::OpDecorate, invocation, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
    state.bdaStopsInvocations = false;
    DefineGetBdaPointer(state);
    const auto main = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), main, spv::FunctionControlMaskNone, TypeFunction(state));
    EmitLabel(state, state.module.AllocateId());
    SpirvValueEmitContext ctx(state);
    auto& instruction = program.CreateValue(IrOpcode::StoreAddressU32, IrType::Void);
    MemoryFlags flags{};
    flags.pc = 0x1234;
    instruction.SetFlags(flags);
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto ids = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32Vector(state, 3), ids, invocation);
    const auto lane = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u32, lane, ids, 0u);
    const auto wide = Unary(state, spv::OpUConvert, u64, lane);
    const auto enabled = Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, u64, Binary(state, spv::OpShiftRightLogical, u64, BdaConstant(state, active), wide), BdaConstant(state, 1u)), BdaConstant(state, 0u));
    EmitIfCondition(state, enabled, [&] {
        const auto target = Binary(state, spv::OpIAdd, u64, BdaConstant(state, address), Binary(state, spv::OpIMul, u64, wide, BdaConstant(state, stride)));
        const auto value = ConstantU32(state, BdaLaneStoreValue);
        switch (store) {
            case BdaLaneStore::Byte: EmitBdaStore(ctx, instruction, target, value, 8u); break;
            case BdaLaneStore::Short: EmitBdaStore(ctx, instruction, target, value, 16u); break;
            case BdaLaneStore::Dword: EmitBdaStore(ctx, instruction, target, value, 32u); break;
            case BdaLaneStore::RuntimeDword: EmitBdaWrite(ctx, instruction, target, value, 32u, ConstantBool(state, true)); break;
            case BdaLaneStore::Dwordx4: {
                const auto vector = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), vector, value, value, value, value);
                EmitBdaDwordWrites(ctx, instruction, target, 0u, 4u, vector);
                break;
            }
            case BdaLaneStore::Atomic:
                EmitBdaAtomic(ctx, instruction, target, 4u, [&](std::uint32_t pointer) {
                    const auto old = state.module.AllocateId();
                    state.module.AddFunction(spv::OpAtomicIAdd, u32, old, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), value);
                    return old;
                });
                break;
        }
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.AddExecutionMode(main, spv::ExecutionModeLocalSize, 64u, 1u, 1u);
    state.module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", std::vector<std::uint32_t>{invocation});
    auto shader = Structured(state.module.Finalize());
#if ANYPS5_ENABLE_SPIRV_TOOLS
    shader = ValidateAndOptimizeSpirv(shader, VK_API_VERSION_1_1, shader[1], false, false);
#endif
    return shader;
}

std::vector<std::uint32_t> MakeImageStoreRecordTestShader(const std::array<std::uint32_t, 8>& words, const ImageStoreRect& rect, bool subgroups) {
    using namespace ShaderRecompiler;
    static constexpr std::array<std::uint32_t, 3> capabilities{spv::CapabilityInt64, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformArithmetic};
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesFaultBuffer = true;
    ImageResource image;
    image.written = true;
    program.Info().images = {image};
    SpirvEmitterState state(program, {});
    state.supportedCapabilities = std::span<const std::uint32_t>(capabilities.data(), subgroups ? capabilities.size() : 1u);
    state.hostSubgroupSize = 32u;
    EmitBaseHeader(state.module, program);
    const auto define = [&](std::uint32_t binding) {
        const auto variable = state.module.DefineGlobalVariable(TypeStorageBufferPointer(state), spv::StorageClassStorageBuffer);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0u);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
        return variable;
    };
    define(0);
    state.faultBufferVariable = define(1);
    define(2);
    const auto invocation = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassInput, TypeU32Vector(state, 3)), spv::StorageClassInput);
    state.module.AddAnnotation(spv::OpDecorate, invocation, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
    DefineImageStoreRecord(state);
    if (state.imageStoreRecordFunction == 0u) throw std::runtime_error("the image store record function was not defined");
    const auto main = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), main, spv::FunctionControlMaskNone, TypeFunction(state));
    state.entryBody = true;
    EmitLabel(state, state.module.AllocateId());
    const auto u32 = TypeU32(state);
    const auto ids = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32Vector(state, 3), ids, invocation);
    const auto index = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u32, index, ids, 0u);
    const auto column = Binary(state, spv::OpUMod, u32, index, ConstantU32(state, rect.width));
    const auto row = Binary(state, spv::OpUDiv, u32, index, ConstantU32(state, rect.width));
    const auto inside = Binary(state, spv::OpULessThan, TypeBool(state), index, ConstantU32(state, rect.width * rect.height));
    EmitIfCondition(state, inside, [&] {
        const auto x = Binary(state, spv::OpIAdd, u32, column, ConstantU32(state, rect.x));
        const auto y = Binary(state, spv::OpIAdd, u32, row, ConstantU32(state, rect.y));
        const auto noted = state.module.AllocateId();
        std::vector<std::uint32_t> call{spv::OpFunctionCall, TypeU32Vector(state, 2u), noted, state.imageStoreRecordFunction};
        for (std::uint32_t word = 0; word < 6u; ++word) call.push_back(ConstantU32(state, words[word]));
        call.insert(call.end(), {x, y, ConstantU32(state, rect.layer), ConstantU32(state, rect.lod)});
        state.module.AddFunction(call);
        EmitImageStoreNote(state, noted);
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.AddExecutionMode(main, spv::ExecutionModeLocalSize, 64u, 1u, 1u);
    state.module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", std::vector<std::uint32_t>{invocation});
    auto shader = Structured(state.module.Finalize());
#if ANYPS5_ENABLE_SPIRV_TOOLS
    shader = ValidateAndOptimizeSpirv(shader, VK_API_VERSION_1_1, shader[1], false, false);
#endif
    return shader;
}

std::vector<std::uint32_t> MakeBdaWriteTestShader(std::uint64_t address, std::uint32_t value, std::uint32_t bits, bool coherent, bool enabled) {
    using namespace ShaderRecompiler;
    auto shader = MakeShader(address, coherent, true, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        EmitBdaWrite(ctx, instruction, base, ConstantU32(ctx.state, value), bits, ConstantBool(ctx.state, enabled));
        return std::vector<std::uint32_t>{ConstantU32(ctx.state, 1u)};
    }, 0u, true);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    shader = ValidateAndOptimizeSpirv(shader, VK_API_VERSION_1_1, shader[1], false, false);
#endif
    return shader;
}
