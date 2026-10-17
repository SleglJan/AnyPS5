#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvMemory/SpirvSubgroup.hpp"
#include "BdaTouchAbi.hpp"
#include <spirv/unified1/GLSL.std.450.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

void EmitBdaOverflowCheck(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction) {
    const auto overflow = Binary(state, spv::OpUGreaterThan, TypeBool(state), address, BdaConstant(state, std::numeric_limits<std::uint64_t>::max() - bytes));
    EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, address, ConstantU32(state, bytes), instruction, BdaAbi::FaultReason::Overflow); });
    StopBdaInvocationIf(state, overflow);
}

bool BdaCoherent(const SpirvEmitterState& state, const IrValue& inst) {
    const auto index = inst.Flags<MemoryFlags>().index;
    const auto& memory = state.program.Resources().memoryInfo;
    if (index >= memory.size()) return false;
    if (memory[index].coherent) return true;
    const auto access = BufferAccessOf(inst.Opcode());
    if (access == BufferAccess::None) return false;
    const bool store = access == BufferAccess::Write || access == BufferAccess::Atomic;
    return state.requirements.coherentBuffers && (store || !state.requirements.coherentBufferAlias);
}

std::uint32_t BdaAccessMask(bool coherent) {
    return spv::MemoryAccessAlignedMask | (coherent ? spv::MemoryAccessVolatileMask : 0u);
}

std::uint32_t BdaAccessMask(const SpirvEmitterState& state, const IrValue& inst) {
    return BdaAccessMask(BdaCoherent(state, inst));
}

std::uint32_t LoadBdaDevice(SpirvEmitterState& state, const IrValue& inst, std::uint32_t device) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, device);
    const auto value = state.module.AllocateId();
    if (BdaCoherent(state, inst)) state.module.AddFunction(spv::OpAtomicLoad, TypeU32(state), value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
    else state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, BdaAccessMask(false), 4u);
    return value;
}

std::uint32_t EmitBdaByte(SpirvEmitterState& state, std::uint32_t guest, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaPointerFunction, guest, ConstantU32(state, 1u), instruction);
    if (state.bdaStopsInvocations) {
        StopBdaInvocationIf(state, Binary(state, spv::OpIEqual, TypeBool(state), physical, BdaConstant(state, 0u)));
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, physical);
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
        return Unary(state, spv::OpUConvert, TypeU32(state), loaded);
    }
    // Unmapped bytes read as zero; the fault is recorded and every invocation reaches the
    // program's barriers.
    const auto mapped = Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u));
    const auto before = state.currentLabel;
    const auto loadLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, mapped, loadLabel, merge);
    EmitLabel(state, loadLabel);
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, physical);
    const auto loaded = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
    const auto widened = Unary(state, spv::OpUConvert, TypeU32(state), loaded);
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, TypeU32(state), value, widened, loadLabel, ConstantU32(state, 0u), before);
    return value;
}

std::uint32_t EmitBdaByteLoop(SpirvEmitterState& state, std::uint32_t address, std::uint32_t count, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto u32 = TypeU32(state);
    const auto entry = state.currentLabel;
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto byte = state.module.AllocateId();
    const auto result = state.module.AllocateId();
    const auto nextByte = state.module.AllocateId();
    const auto nextResult = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    state.module.AddFunction(spv::OpPhi, u32, byte, ConstantU32(state, 0u), entry, nextByte, continuation);
    state.module.AddFunction(spv::OpPhi, u32, result, ConstantU32(state, 0u), entry, nextResult, continuation);
    const auto more = Binary(state, spv::OpULessThan, TypeBool(state), byte, count);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, more, body, merge);
    EmitLabel(state, body);
    const auto guest = Binary(state, spv::OpIAdd, TypeScalarU64(state), address, Unary(state, spv::OpUConvert, TypeScalarU64(state), byte));
    const auto value = EmitBdaByte(state, guest, instruction, accessMask);
    state.module.AddFunction(spv::OpBitwiseOr, u32, nextResult, result, Binary(state, spv::OpShiftLeftLogical, u32, value, Binary(state, spv::OpShiftLeftLogical, u32, byte, ConstantU32(state, 3u))));
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpIAdd, u32, nextByte, byte, ConstantU32(state, 1u));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    return result;
}

std::uint32_t EmitBdaBytes(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t accessMask) {
    if (bytes == 1u) return EmitBdaByte(state, address, instruction, accessMask);
    return EmitBdaByteLoop(state, address, ConstantU32(state, bytes), instruction, accessMask);
}

// The dwords of a wide load that the program extracts (all of them when the composite is used
// whole). Dead-code elimination has dropped the unused extracts, as it dropped unused dword loads
// when every dword was its own instruction.
std::uint32_t UsedBdaDwords(const IrValue& inst, std::uint32_t dwords) {
    const auto all = (1u << dwords) - 1u;
    if (dwords == 1u) return all;
    std::uint32_t used = 0;
    for (const IrValue* user : inst.Uses()) {
        const auto opcode = user->Opcode();
        if (opcode != IrOpcode::CompositeExtractU32x2 && opcode != IrOpcode::CompositeExtractU32x3 && opcode != IrOpcode::CompositeExtractU32x4) return all;
        const IrValue* index = user->ArgumentCount() > 1u ? user->Argument(1) : nullptr;
        if (index == nullptr || !index->HasImmediate() || index->ImmediateU32() >= dwords) return all;
        used |= 1u << index->ImmediateU32();
    }
    return used;
}

void EmitBdaStoreAt(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t pointerType, std::uint32_t value, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, address, ConstantU32(state, bytes), instruction);
    EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), [&] {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, pointerType, pointer, physical);
        state.module.AddFunction(spv::OpStore, pointer, value, accessMask, bytes);
        state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, address);
    });
}

std::uint32_t BdaAddressUnaligned(SpirvEmitterState& state, std::uint32_t address) {
    return Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address, BdaConstant(state, 3u)), BdaConstant(state, 0u));
}

std::uint32_t AddBdaSignedOffset(SpirvEmitterState& state, std::uint32_t address, std::uint32_t offset, std::uint32_t instruction) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto negative = Binary(state, spv::OpSLessThan, boolean, offset, ConstantU32(state, 0u));
    const auto magnitude = Unary(state, spv::OpUConvert, u64, Select(state, u32, negative, Binary(state, spv::OpISub, u32, ConstantU32(state, 0u), offset), offset));
    const auto sum = Binary(state, spv::OpIAdd, u64, address, magnitude);
    const auto result = Select(state, u64, negative, Binary(state, spv::OpISub, u64, address, magnitude), sum);
    const auto overflow = Select(state, boolean, negative, Binary(state, spv::OpUGreaterThan, boolean, magnitude, address), Binary(state, spv::OpULessThan, boolean, sum, address));
    EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, address, ConstantU32(state, 0u), instruction, BdaAbi::FaultReason::Overflow); });
    StopBdaInvocationIf(state, overflow);
    return result;
}

std::uint32_t BdaSpanReadResultType(SpirvEmitterState& state) {
    const auto values = TypeU32Vector(state, 4);
    return state.bdaStopsInvocations ? state.module.Type(spv::OpTypeStruct, values, TypeU32(state)) : values;
}

std::uint32_t DefineBdaSpanLoadFunction(SpirvEmitterState& state, bool coherent) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto values = TypeU32Vector(state, 4);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, std::string("load_bda_span") + (coherent ? "_coherent" : ""));
    state.module.AddFunction(spv::OpFunction, values, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, values, u64, u32, u32));
    const auto physical = state.module.AllocateId();
    const auto firstBytes = state.module.AllocateId();
    const auto used = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, physical);
    state.module.AddFunction(spv::OpFunctionParameter, u32, firstBytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, used);
    EmitLabel(state, state.module.AllocateId());
    std::array<std::uint32_t, 4> loaded{};
    for (std::uint32_t dword = 0; dword < 4u; ++dword) {
        const auto isUsed = Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, u32, used, ConstantU32(state, 1u << dword)), ConstantU32(state, 0u));
        loaded[dword] = EmitValueOrZeroIfCondition(state, isUsed, [&] {
            const auto delta = Unary(state, spv::OpUConvert, u64, Binary(state, spv::OpISub, u32, ConstantU32(state, dword * 4u), firstBytes));
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, Binary(state, spv::OpIAdd, u64, physical, delta));
            const auto value = state.module.AllocateId();
            if (coherent) state.module.AddFunction(spv::OpAtomicLoad, u32, value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
            else state.module.AddFunction(spv::OpLoad, u32, value, pointer, BdaAccessMask(coherent), 4u);
            return value;
        });
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, values, result, loaded[0], loaded[1], loaded[2], loaded[3]);
    state.module.AddFunction(spv::OpReturnValue, result);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

std::uint32_t DefineBdaSpanReadFunction(SpirvEmitterState& state, bool coherent, std::uint32_t loader) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto values = TypeU32Vector(state, 4);
    const auto zero = ConstantU32(state, 0u);
    const bool stops = state.bdaStopsInvocations;
    const auto result = BdaSpanReadResultType(state);
    const auto reader = state.bdaDwordReadFunctions[stops][coherent];
    const auto function = state.module.AllocateId();
    state.module.AddName(function, std::string("read_bda_span") + (stops ? "_stop" : "") + (coherent ? "_coherent" : ""));
    state.module.AddFunction(spv::OpFunction, result, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, result, u64, u32, u32, u32));
    const auto address = state.module.AllocateId();
    const auto offset = state.module.AllocateId();
    const auto used = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, offset);
    state.module.AddFunction(spv::OpFunctionParameter, u32, used);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto first = EmitGlsl<GLSLstd450FindILsb, IrType::U32>(state, used);
    const auto last = EmitGlsl<GLSLstd450FindUMsb, IrType::U32>(state, used);
    const auto firstBytes = Binary(state, spv::OpShiftLeftLogical, u32, first, ConstantU32(state, 2u));
    const auto firstAddress = AddBdaSignedOffset(state, address, Binary(state, spv::OpIAdd, u32, offset, firstBytes), instruction);
    const auto span = Binary(state, spv::OpISub, u32, Binary(state, spv::OpShiftLeftLogical, u32, Binary(state, spv::OpIAdd, u32, last, ConstantU32(state, 1u)), ConstantU32(state, 2u)), firstBytes);
    if (BdaVectorCacheAvailable(state)) {
        const auto bytes = Unary(state, spv::OpUConvert, u64, span);
        const auto returnIfCached = [&] {
            const auto cached = ProbeBdaVectorCache(state, firstAddress, bytes);
            const auto hit = state.module.AllocateId();
            const auto miss = state.module.AllocateId();
            state.module.AddFunction(spv::OpSelectionMerge, miss, spv::SelectionControlMaskNone);
            state.module.AddFunction(spv::OpBranchConditional, cached.hit, hit, miss);
            EmitLabel(state, hit);
            NoteBdaSegmentRead(state, firstAddress, bytes);
            std::array<std::uint32_t, 4> parts{};
            for (std::uint32_t dword = 0; dword < 4u; ++dword) {
                const auto wanted = Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, used, ConstantU32(state, 1u << dword)), zero);
                parts[dword] = EmitValueOrZeroIfCondition(state, wanted, [&] {
                    const auto delta = Unary(state, spv::OpUConvert, u64, Binary(state, spv::OpISub, u32, ConstantU32(state, dword * 4u), firstBytes));
                    const auto pointer = state.module.AllocateId();
                    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, Binary(state, spv::OpIAdd, u64, cached.device, delta));
                    const auto value = state.module.AllocateId();
                    if (coherent) state.module.AddFunction(spv::OpAtomicLoad, u32, value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
                    else state.module.AddFunction(spv::OpLoad, u32, value, pointer, BdaAccessMask(coherent), 4u);
                    return value;
                });
            }
            auto direct = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, values, direct, parts[0], parts[1], parts[2], parts[3]);
            if (stops) {
                const auto wrapped = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeConstruct, result, wrapped, direct, zero);
                direct = wrapped;
            }
            state.module.AddFunction(spv::OpReturnValue, direct);
            EmitLabel(state, miss);
        };
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, ProbeBdaVectorCache(state, firstAddress, bytes).hit), [&] { RefreshBdaVectorCache(state, firstAddress); });
        returnIfCached();
    }
    const auto isUsed = [&](std::uint32_t dword) {
        return Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, used, ConstantU32(state, 1u << dword)), zero);
    };
    const auto composite = [&](const std::array<std::uint32_t, 4>& parts) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, values, value, parts[0], parts[1], parts[2], parts[3]);
        return value;
    };
    const auto dwords = [&] {
        const auto entry = state.currentLabel;
        const auto header = state.module.AllocateId();
        const auto body = state.module.AllocateId();
        const auto continuation = state.module.AllocateId();
        const auto merge = state.module.AllocateId();
        const auto dword = state.module.AllocateId();
        const auto gathered = state.module.AllocateId();
        const auto nextDword = state.module.AllocateId();
        const auto nextGathered = state.module.AllocateId();
        const auto zeros = state.module.Constant(spv::OpConstantComposite, values, zero, zero, zero, zero);
        state.module.AddFunction(spv::OpBranch, header);
        EmitLabel(state, header);
        state.module.AddFunction(spv::OpPhi, u32, dword, zero, entry, nextDword, continuation);
        state.module.AddFunction(spv::OpPhi, values, gathered, zeros, entry, nextGathered, continuation);
        const auto more = Binary(state, spv::OpULessThan, boolean, dword, ConstantU32(state, 4u));
        state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, more, body, merge);
        EmitLabel(state, body);
        const auto used32 = Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, used, Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1u), dword)), zero);
        const auto value = EmitValueOrZeroIfCondition(state, used32, [&] {
            const auto isFirst = Binary(state, spv::OpIEqual, boolean, first, dword);
            const auto dwordAddress = EmitValueIfElse(state, isFirst, u64, [&] { return firstAddress; }, [&] {
                return AddBdaSignedOffset(state, address, Binary(state, spv::OpIAdd, u32, offset, Binary(state, spv::OpShiftLeftLogical, u32, dword, ConstantU32(state, 2u))), instruction);
            });
            const auto pair = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 2), pair, reader, dwordAddress, instruction);
            if (stops) {
                const auto stopped = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeExtract, u32, stopped, pair, 1u);
                StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, boolean, stopped, zero));
            }
            const auto loaded = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, u32, loaded, pair, 0u);
            return loaded;
        });
        state.module.AddFunction(spv::OpVectorInsertDynamic, values, nextGathered, gathered, value, dword);
        state.module.AddFunction(spv::OpBranch, continuation);
        EmitLabel(state, continuation);
        state.module.AddFunction(spv::OpIAdd, u32, nextDword, dword, ConstantU32(state, 1u));
        state.module.AddFunction(spv::OpBranch, header);
        EmitLabel(state, merge);
        return gathered;
    };
    std::uint32_t read = 0;
    if (BdaVectorCacheAvailable(state)) {
        read = dwords();
    } else {
        const auto physical = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, u64, physical, state.bdaProbeFunction, firstAddress, span, instruction);
        const auto mapped = Binary(state, spv::OpINotEqual, boolean, physical, BdaConstant(state, 0u));
        const auto wide = Binary(state, spv::OpLogicalAnd, boolean, mapped, Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, physical)));
        read = EmitValueIfElse(state, wide, values, [&] {
            const auto loaded = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, values, loaded, loader, physical, firstBytes, used);
            return loaded;
        }, dwords);
    }
    auto returned = read;
    if (stops) {
        returned = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, result, returned, read, zero);
    }
    state.module.AddFunction(spv::OpReturnValue, returned);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

void DefineBdaDwordReadFunctions(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto pair = TypeU32Vector(state, 2);
    const auto type = state.module.Type(spv::OpTypeFunction, pair, u64, u32);
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    const bool stops = state.bdaStopsInvocations;
    if (stops) state.bdaStopValue = state.module.Constant(spv::OpConstantComposite, pair, ConstantU32(state, 0u), ConstantU32(state, 1u));
    for (const bool stop : {false, true}) {
        for (const bool coherent : {false, true}) {
            if ((stop && !stops) || (coherent && !coherentAccesses)) continue;
            const auto function = state.module.AllocateId();
            state.module.AddName(function, std::string("read_bda_dword_bytes") + (stop ? "_stop" : "") + (coherent ? "_coherent" : ""));
            state.module.AddFunction(spv::OpFunction, pair, function, spv::FunctionControlDontInlineMask, type);
            const auto address = state.module.AllocateId();
            const auto instruction = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, u64, address);
            state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
            EmitLabel(state, state.module.AllocateId());
            state.bdaStopsInvocations = stop;
            EmitBdaOverflowCheck(state, address, 4u, instruction);
            std::uint32_t value = 0;
            if (coherent && state.bdaProbeFunction != 0) {
                const auto u64t = TypeScalarU64(state);
                const auto physical = state.module.AllocateId();
                state.module.AddFunction(spv::OpFunctionCall, u64t, physical, state.bdaProbeFunction, address, ConstantU32(state, 4u), instruction);
                const auto usable = Binary(state, spv::OpLogicalAnd, TypeBool(state), Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), Binary(state, spv::OpIEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, u64t, physical, BdaConstant(state, 3u)), BdaConstant(state, 0u)));
                const auto atomicLabel = state.module.AllocateId();
                const auto byteLabel = state.module.AllocateId();
                const auto byteExit = state.module.AllocateId();
                const auto merge = state.module.AllocateId();
                state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
                state.module.AddFunction(spv::OpBranchConditional, usable, atomicLabel, byteLabel);
                EmitLabel(state, atomicLabel);
                const auto pointer = state.module.AllocateId();
                state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, physical);
                const auto atomicValue = state.module.AllocateId();
                state.module.AddFunction(spv::OpAtomicLoad, u32, atomicValue, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
                state.module.AddFunction(spv::OpBranch, merge);
                EmitLabel(state, byteLabel);
                const auto byteValue = EmitBdaBytes(state, address, 4u, instruction, BdaAccessMask(coherent));
                state.module.AddFunction(spv::OpBranch, byteExit);
                EmitLabel(state, byteExit);
                state.module.AddFunction(spv::OpBranch, merge);
                EmitLabel(state, merge);
                value = state.module.AllocateId();
                state.module.AddFunction(spv::OpPhi, u32, value, atomicValue, atomicLabel, byteValue, byteExit);
            } else {
                value = EmitBdaBytes(state, address, 4u, instruction, BdaAccessMask(coherent));
            }
            const auto result = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, pair, result, value, ConstantU32(state, 0u));
            state.module.AddFunction(spv::OpReturnValue, result);
            state.module.AddFunction(spv::OpFunctionEnd);
            state.bdaDwordReadFunctions[stop][coherent] = function;
        }
    }
    state.bdaStopsInvocations = stops;
    state.bdaStopValue = 0;
}

void DefineBdaSpanReadFunctions(SpirvEmitterState& state) {
    if (state.bdaProbeFunction == 0) return;
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    const bool stops = state.bdaStopsInvocations;
    const auto zero = ConstantU32(state, 0u);
    std::array<std::uint32_t, 2> loaders{};
    for (const bool coherent : {false, true}) {
        if (!coherent || coherentAccesses) loaders[coherent] = DefineBdaSpanLoadFunction(state, coherent);
    }
    for (const bool stop : {false, true}) {
        if (stop && !stops) continue;
        state.bdaStopsInvocations = stop;
        state.bdaStopValue = stop ? state.module.Constant(spv::OpConstantComposite, BdaSpanReadResultType(state), state.module.Constant(spv::OpConstantComposite, TypeU32Vector(state, 4), zero, zero, zero, zero), ConstantU32(state, 1u)) : 0u;
        for (const bool coherent : {false, true}) {
            if (coherent && !coherentAccesses) continue;
            state.bdaSpanReadFunctions[stop][coherent] = DefineBdaSpanReadFunction(state, coherent, loaders[coherent]);
        }
    }
    state.bdaStopsInvocations = stops;
    state.bdaStopValue = 0;
}

// APS5_BDA_BYTE_READS=1 restores one lookup and one byte load per byte for every read.
bool BdaByteReadsForced() {
    static const bool forced = std::getenv("APS5_BDA_BYTE_READS") != nullptr;
    return forced;
}

std::uint32_t BdaInstructionPc(SpirvEmitterState& state, const IrValue& inst) {
    return state.bdaPcOverride != 0u ? state.bdaPcOverride : ConstantU32(state, inst.Flags<MemoryFlags>().pc);
}

std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract) {
    auto& state = ctx.state;
    const auto result = Binary(state, subtract ? spv::OpISub : spv::OpIAdd, TypeScalarU64(state), address, offset);
    const auto overflow = subtract ? Binary(state, spv::OpUGreaterThan, TypeBool(state), offset, address) : Binary(state, spv::OpULessThan, TypeBool(state), result, address);
    EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, address, ConstantU32(state, 0u), BdaInstructionPc(state, inst), BdaAbi::FaultReason::Overflow); });
    StopBdaInvocationIf(state, overflow);
    return result;
}

std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate) {
    if (immediate == 0) {
        return address;
    }
    const auto magnitude = immediate < 0 ? -static_cast<std::int64_t>(immediate) : static_cast<std::int64_t>(immediate);
    return AddBdaAddress(ctx, inst, address, BdaConstant(ctx.state, static_cast<std::uint64_t>(magnitude)), immediate < 0);
}

void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target) {
    if (program.Info().usesFaultBuffer && target.bdaAbiVersion != BdaAbi::Version) throw std::runtime_error("the fault buffer needs the BDA fault ABI, which the target lacks");
    if (!program.Info().usesDma) return;
    if (target.bdaAbiVersion != BdaAbi::Version) throw std::runtime_error("unsupported BDA ABI version");
    for (const auto capability : {spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess}) {
        if (std::find(target.supportedCapabilities.begin(), target.supportedCapabilities.end(), static_cast<std::uint32_t>(capability)) == target.supportedCapabilities.end()) throw std::runtime_error("BDA requires unsupported SPIR-V capability " + std::to_string(capability));
    }
    for (const auto extension : {"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"}) {
        if (std::find(target.supportedExtensions.begin(), target.supportedExtensions.end(), extension) == target.supportedExtensions.end()) throw std::runtime_error(std::string("BDA requires unsupported extension ") + extension);
    }
}

bool BdaInvocationsMayStop(const IrProgram& program) {
    const auto stage = program.Resources().stage;
    if (stage == IrShaderStage::Mesh || stage == IrShaderStage::TessellationControl) return false;
    for (const auto* block : program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            if (instruction->Opcode() == IrOpcode::Barrier) return false;
        }
    }
    return true;
}

std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits) {
    auto& state = ctx.state;
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA read width");
    if (state.bdaPointerFunction == 0) ctx.Fail(inst, "BDA lookup function is missing");
    if (bits == 32u) return EmitBdaDwordReads(ctx, inst, address, 0u, 1u)[0];
    const auto instruction = BdaInstructionPc(state, inst);
    EmitBdaOverflowCheck(state, address, bits / 8u, instruction);
    return EmitBdaBytes(state, address, bits / 8u, instruction, BdaAccessMask(state, inst));
}

namespace {

constexpr std::uint32_t ScalarGroupRefreshDwords = 8u;

bool BdaSegmentsShared(SpirvEmitterState& state) {
    const auto& capabilities = state.supportedCapabilities;
    const bool ballot = std::find(capabilities.begin(), capabilities.end(), static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformBallot)) != capabilities.end();
    if (HostSubgroupSingleLane(state) || !ballot) return false;
    state.module.EmitCapability(spv::CapabilityGroupNonUniform);
    state.module.EmitCapability(spv::CapabilityGroupNonUniformBallot);
    return true;
}

std::uint32_t BroadcastFirstU64(SpirvEmitterState& state, std::uint32_t value) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto first = [&](std::uint32_t word) {
        const auto result = state.module.AllocateId();
        state.module.AddFunction(spv::OpGroupNonUniformBroadcastFirst, u32, result, ConstantU32(state, spv::ScopeSubgroup), word);
        return result;
    };
    const auto low = first(Unary(state, spv::OpUConvert, u32, value));
    const auto high = first(Unary(state, spv::OpUConvert, u32, Binary(state, spv::OpShiftRightLogical, u64, value, BdaConstant(state, 32u))));
    return Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, low), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, high), BdaConstant(state, 32u)));
}

std::uint32_t RefreshBdaReadCache(SpirvEmitterState& state, std::uint32_t address, std::uint32_t end, std::uint32_t enabled, bool elect) {
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto& cache = state.bdaReadCache;
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, value, variable);
        return value;
    };
    const auto segmentType = state.module.Type(spv::OpTypeStruct, u64, u64, u64);
    const auto translate = [&] {
        const auto segment = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, segmentType, segment, state.bdaSegmentFunction, address);
        return segment;
    };
    const bool shared = elect && BdaSegmentsShared(state);
    std::uint32_t segment = 0;
    if (shared) {
        const auto elected = state.module.AllocateId();
        EmitSubgroupElect(state, elected);
        const auto zero = BdaConstant(state, 0u);
        segment = EmitValueOrDefaultIfCondition(state, elected, segmentType, state.module.Constant(spv::OpConstantComposite, segmentType, zero, zero, zero), translate);
    } else {
        segment = translate();
    }
    std::array<std::uint32_t, 3> fields{};
    for (std::uint32_t field = 0; field < 3u; ++field) {
        fields[field] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, fields[field], segment, field);
        if (shared) fields[field] = BroadcastFirstU64(state, fields[field]);
    }
    const auto aligned = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u64, Binary(state, spv::OpIAdd, u64, address, fields[2]), BdaConstant(state, 3u)), BdaConstant(state, 0u));
    const auto within = both(Binary(state, spv::OpULessThanEqual, boolean, fields[0], address), Binary(state, spv::OpULessThanEqual, boolean, end, fields[1]));
    const auto covers = both(both(enabled, both(Binary(state, spv::OpULessThan, boolean, address, end), aligned)), within);
    for (std::uint32_t field = 0; field < 3u; ++field) state.module.AddFunction(spv::OpStore, cache[field], Select(state, u64, covers, fields[field], load(cache[field])));
    return covers;
}

}

std::uint32_t DefineBdaNotePages(SpirvEmitterState& state, std::uint32_t touch) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_pages");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, voidType, u32, u32, u32));
    const auto first = state.module.AllocateId();
    const auto last = state.module.AllocateId();
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u32, first);
    state.module.AddFunction(spv::OpFunctionParameter, u32, last);
    state.module.AddFunction(spv::OpFunctionParameter, u32, noted);
    EmitLabel(state, state.module.AllocateId());
    const auto note = [&](std::uint32_t page) {
        const auto address = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, page), BdaConstant(state, BdaTouchAbi::PageShift));
        state.module.AddFunction(spv::OpFunctionCall, voidType, state.module.AllocateId(), touch, address);
    };
    const auto moved = Binary(state, spv::OpLogicalOr, boolean, Binary(state, spv::OpINotEqual, boolean, first, noted), Binary(state, spv::OpINotEqual, boolean, last, noted));
    EmitIfCondition(state, moved, [&] {
        if (BdaSegmentsShared(state)) {
            const auto leader = state.module.AllocateId();
            state.module.AddFunction(spv::OpGroupNonUniformBroadcastFirst, u32, leader, ConstantU32(state, spv::ScopeSubgroup), first);
            const auto elected = state.module.AllocateId();
            EmitSubgroupElect(state, elected);
            EmitIfCondition(state, Binary(state, spv::OpLogicalOr, boolean, Binary(state, spv::OpINotEqual, boolean, first, leader), elected), [&] { note(first); });
        } else {
            note(first);
        }
        EmitIfCondition(state, Binary(state, spv::OpINotEqual, boolean, last, first), [&] { note(last); });
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void NoteBdaSegmentRead(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes) {
    if (state.bdaNotePagesFunction == 0u || state.bdaNotedPage == 0u) return;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto page = [&](std::uint32_t value) { return Unary(state, spv::OpUConvert, u32, Binary(state, spv::OpShiftRightLogical, u64, value, BdaConstant(state, BdaTouchAbi::PageShift))); };
    const auto first = page(address);
    const auto last = page(Binary(state, spv::OpISub, u64, Binary(state, spv::OpIAdd, u64, address, bytes), BdaConstant(state, 1u)));
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, noted, state.bdaNotedPage);
    state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNotePagesFunction, first, last, noted);
    state.module.AddFunction(spv::OpStore, state.bdaNotedPage, last);
}

std::uint32_t EmitBdaCachedDword(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t enabled, const std::function<std::uint32_t()>& uncached, bool refill) {
    auto& state = ctx.state;
    if (!BdaReadCacheAvailable(state)) ctx.Fail(inst, "BDA read cache is missing");
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto& cache = state.bdaReadCache;
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, value, variable);
        return value;
    };
    const auto end = Binary(state, spv::OpIAdd, u64, address, BdaConstant(state, 4u));
    const auto within = [&](std::uint32_t begin, std::uint32_t limit) { return both(Binary(state, spv::OpULessThanEqual, boolean, begin, address), Binary(state, spv::OpULessThanEqual, boolean, end, limit)); };
    const auto fast = [&] {
        NoteBdaSegmentRead(state, address, BdaConstant(state, 4u));
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, Binary(state, spv::OpIAdd, u64, address, load(cache[2])));
        const auto value = state.module.AllocateId();
        if (BdaCoherent(state, inst)) state.module.AddFunction(spv::OpAtomicLoad, u32, value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
        else state.module.AddFunction(spv::OpLoad, u32, value, pointer, BdaAccessMask(false), 4u);
        return value;
    };
    const auto& segment = state.bdaAccessSegment;
    if (segment[0] != 0u) {
        const auto resolved = ProbeBdaSegment(state, {segment[0], segment[1], segment[2]}, address, BdaConstant(state, 4u));
        const auto direct = [&] {
            NoteBdaSegmentRead(state, address, BdaConstant(state, 4u));
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, resolved.device);
            const auto value = state.module.AllocateId();
            if (BdaCoherent(state, inst)) state.module.AddFunction(spv::OpAtomicLoad, u32, value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
            else state.module.AddFunction(spv::OpLoad, u32, value, pointer, BdaAccessMask(false), 4u);
            return value;
        };
        return EmitValueIfElse(state, both(enabled, resolved.hit), u32, direct, uncached);
    }
    const auto hit = both(enabled, within(load(cache[0]), load(cache[1])));
    if (!refill) return EmitValueIfElse(state, hit, u32, fast, uncached);
    return EmitValueIfElse(state, hit, u32, fast, [&] { return EmitValueIfElse(state, RefreshBdaReadCache(state, address, end, enabled, true), u32, fast, uncached); });
}

std::uint32_t EmitBdaCachedDwords(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t enabled, std::uint32_t dwords, std::uint32_t type, const std::function<std::uint32_t()>& miss) {
    auto& state = ctx.state;
    if (!BdaReadCacheAvailable(state)) ctx.Fail(inst, "BDA read cache is missing");
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto& cache = state.bdaReadCache;
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, value, variable);
        return value;
    };
    const auto end = Binary(state, spv::OpIAdd, u64, address, BdaConstant(state, dwords * 4u));
    const auto within = both(Binary(state, spv::OpULessThanEqual, boolean, load(cache[0]), address), Binary(state, spv::OpULessThanEqual, boolean, end, load(cache[1])));
    const auto span = [&] {
        NoteBdaSegmentRead(state, address, BdaConstant(state, dwords * 4u));
        const auto device = Binary(state, spv::OpIAdd, u64, address, load(cache[2]));
        std::vector<std::uint32_t> construct{spv::OpCompositeConstruct, type, state.module.AllocateId()};
        const bool coherent = BdaCoherent(state, inst);
        for (std::uint32_t dword = 0; dword < dwords;) {
            const auto width = coherent ? 1u : std::min(4u, dwords - dword);
            const auto element = dword == 0u ? device : Binary(state, spv::OpIAdd, u64, device, BdaConstant(state, dword * 4u));
            const auto pointer = state.module.AllocateId();
            const auto loaded = state.module.AllocateId();
            if (width == 1u) {
                state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, element);
                if (coherent) state.module.AddFunction(spv::OpAtomicLoad, u32, loaded, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
                else state.module.AddFunction(spv::OpLoad, u32, loaded, pointer, BdaAccessMask(false), 4u);
                construct.push_back(loaded);
            } else {
                const auto vector = TypeU32Vector(state, width);
                state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, vector), pointer, element);
                state.module.AddFunction(spv::OpLoad, vector, loaded, pointer, BdaAccessMask(false), 4u);
                for (std::uint32_t component = 0; component < width; ++component) {
                    const auto value = state.module.AllocateId();
                    state.module.AddFunction(spv::OpCompositeExtract, u32, value, loaded, component);
                    construct.push_back(value);
                }
            }
            dword += width;
        }
        state.module.AddFunction(construct);
        return construct[2];
    };
    if (dwords < ScalarGroupRefreshDwords) return EmitValueIfElse(state, both(enabled, within), type, span, miss);
    return EmitValueIfElse(state, both(enabled, within), type, span, [&] { return EmitValueIfElse(state, RefreshBdaReadCache(state, address, end, enabled, false), type, span, miss); });
}

void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits, std::uint32_t enabled) {
    auto& state = ctx.state;
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA write width");
    const auto function = state.bdaWriteFunctions[BdaCoherent(state, inst)];
    if (function == 0) ctx.Fail(inst, "BDA write function is missing");
    state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), function, address, value, ConstantU32(state, bits / 8u), BdaInstructionPc(state, inst), enabled);
}

void DefineBdaByteWriteFunctions(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
    const auto type = state.module.Type(spv::OpTypeFunction, voidType, u64, u32, u32, u32);
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    for (const bool coherent : {false, true}) {
        if (coherent && !coherentAccesses) continue;
        const auto function = state.module.AllocateId();
        state.module.AddName(function, std::string("write_bda_bytes") + (coherent ? "_coherent" : ""));
        state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlDontInlineMask, type);
        const auto address = state.module.AllocateId();
        const auto value = state.module.AllocateId();
        const auto bytes = state.module.AllocateId();
        const auto instruction = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, u64, address);
        state.module.AddFunction(spv::OpFunctionParameter, u32, value);
        state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
        state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
        EmitLabel(state, state.module.AllocateId());
        const auto counter = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), counter, spv::StorageClassFunction);
        state.module.AddFunction(spv::OpStore, counter, ConstantU32(state, 0u));
        const auto header = state.module.AllocateId();
        const auto body = state.module.AllocateId();
        const auto continuation = state.module.AllocateId();
        const auto merge = state.module.AllocateId();
        state.module.AddFunction(spv::OpBranch, header);
        EmitLabel(state, header);
        const auto byte = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, byte, counter);
        const auto pending = Binary(state, spv::OpULessThan, TypeBool(state), byte, bytes);
        state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, pending, body, merge);
        EmitLabel(state, body);
        const auto guest = Binary(state, spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, byte));
        const auto shifted = Binary(state, spv::OpShiftRightLogical, u32, value, Binary(state, spv::OpShiftLeftLogical, u32, byte, ConstantU32(state, 3u)));
        EmitBdaStoreAt(state, guest, 1u, bytePointer, Unary(state, spv::OpUConvert, byteType, shifted), instruction, BdaAccessMask(coherent));
        state.module.AddFunction(spv::OpBranch, continuation);
        EmitLabel(state, continuation);
        state.module.AddFunction(spv::OpStore, counter, Binary(state, spv::OpIAdd, u32, byte, ConstantU32(state, 1u)));
        state.module.AddFunction(spv::OpBranch, header);
        EmitLabel(state, merge);
        state.module.AddFunction(spv::OpReturn);
        state.module.AddFunction(spv::OpFunctionEnd);
        state.bdaByteWriteFunctions[coherent] = function;
    }
}

void DefineBdaWriteFunctions(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto type = state.module.Type(spv::OpTypeFunction, voidType, u64, u32, u32, u32, boolean);
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    for (const bool coherent : {false, true}) {
        if (coherent && !coherentAccesses) continue;
        const auto function = state.module.AllocateId();
        state.module.AddName(function, std::string("write_bda") + (coherent ? "_coherent" : ""));
        state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlDontInlineMask, type);
        const auto address = state.module.AllocateId();
        const auto value = state.module.AllocateId();
        const auto bytes = state.module.AllocateId();
        const auto instruction = state.module.AllocateId();
        const auto enabled = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, u64, address);
        state.module.AddFunction(spv::OpFunctionParameter, u32, value);
        state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
        state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
        state.module.AddFunction(spv::OpFunctionParameter, boolean, enabled);
        EmitLabel(state, state.module.AllocateId());
        EmitIfCondition(state, enabled, [&] {
            const auto dword = Binary(state, spv::OpLogicalAnd, boolean, Binary(state, spv::OpIEqual, boolean, bytes, ConstantU32(state, 4u)), Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, address)));
            EmitIfCondition(state, dword, [&] { EmitBdaStoreAt(state, address, 4u, TypePhysicalU32Pointer(state), value, instruction, BdaAccessMask(coherent)); });
            EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, dword), [&] {
                const auto limit = Binary(state, spv::OpISub, u64, BdaConstant(state, std::numeric_limits<std::uint64_t>::max()), Unary(state, spv::OpUConvert, u64, bytes));
                EmitIfCondition(state, Binary(state, spv::OpUGreaterThan, boolean, address, limit), [&] { RecordBdaFault(state, address, bytes, instruction, BdaAbi::FaultReason::Overflow); });
                state.module.AddFunction(spv::OpFunctionCall, voidType, state.module.AllocateId(), state.bdaByteWriteFunctions[coherent], address, value, bytes, instruction);
            });
        });
        state.module.AddFunction(spv::OpReturn);
        state.module.AddFunction(spv::OpFunctionEnd);
        state.bdaWriteFunctions[coherent] = function;
    }
}

namespace {

std::uint32_t DefineBdaComponentReadFunction(SpirvEmitterState& state, bool coherent) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto values = TypeU32Vector(state, 4);
    const auto addressVector = state.module.Type(spv::OpTypeVector, u64, 4u);
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
    const auto translation = state.module.Type(spv::OpTypeStruct, u64, u32);
    const auto result = BdaSpanReadResultType(state);
    const auto accessMask = BdaAccessMask(coherent);
    const bool stops = state.bdaStopsInvocations;
    const auto zero = ConstantU32(state, 0u);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto equal = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpIEqual, boolean, left, right); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, std::string("read_bda_components") + (stops ? "_stop" : "") + (coherent ? "_coherent" : ""));
    state.module.AddFunction(spv::OpFunction, result, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, result, addressVector, values, u32));
    const auto addresses = state.module.AllocateId();
    const auto sizes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, addressVector, addresses);
    state.module.AddFunction(spv::OpFunctionParameter, values, sizes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto variable = [&](std::uint32_t type) {
        const auto id = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, type), id, spv::StorageClassFunction);
        return id;
    };
    const auto cursor = variable(u32);
    const auto bytewise = variable(boolean);
    const auto gathered = variable(values);
    state.module.AddFunction(spv::OpStore, cursor, zero);
    state.module.AddFunction(spv::OpStore, bytewise, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, gathered, state.module.Constant(spv::OpConstantComposite, values, zero, zero, zero, zero));
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto position = load(u32, cursor);
    const auto byteMode = load(boolean, bytewise);
    const auto accumulated = load(values, gathered);
    const auto pending = binary(spv::OpULessThan, boolean, position, constant(16u));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, pending, body, merge);
    EmitLabel(state, body);
    const auto component = binary(spv::OpShiftRightLogical, u32, position, constant(2u));
    const auto within = binary(spv::OpBitwiseAnd, u32, position, constant(3u));
    const auto size = state.module.AllocateId();
    state.module.AddFunction(spv::OpVectorExtractDynamic, u32, size, sizes, component);
    const auto active = binary(spv::OpULessThan, boolean, within, size);
    state.module.AddFunction(spv::OpStore, cursor, binary(spv::OpShiftLeftLogical, u32, binary(spv::OpIAdd, u32, component, constant(1u)), constant(2u)));
    state.module.AddFunction(spv::OpStore, bytewise, ConstantBool(state, false));
    const auto nextByte = [&] {
        const auto next = binary(spv::OpIAdd, u32, position, constant(1u));
        state.module.AddFunction(spv::OpStore, cursor, next);
        state.module.AddFunction(spv::OpStore, bytewise, binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, next, constant(3u)), zero));
    };
    const auto insert = [&](std::uint32_t value) {
        const auto updated = state.module.AllocateId();
        state.module.AddFunction(spv::OpVectorInsertDynamic, values, updated, accumulated, value, component);
        state.module.AddFunction(spv::OpStore, gathered, updated);
    };
    const auto byteAt = [&](std::uint32_t at) {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, at);
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
        return Unary(state, spv::OpUConvert, u32, loaded);
    };
    EmitIfCondition(state, active, [&] {
        const auto base = state.module.AllocateId();
        state.module.AddFunction(spv::OpVectorExtractDynamic, u64, base, addresses, component);
        const auto guest = binary(spv::OpIAdd, u64, base, Unary(state, spv::OpUConvert, u64, within));
        const auto single = binary(spv::OpLogicalOr, boolean, byteMode, binary(spv::OpINotEqual, boolean, within, zero));
        const auto whole = Unary(state, spv::OpLogicalNot, boolean, single);
        EmitIfCondition(state, whole, [&] {
            const auto overflow = binary(spv::OpUGreaterThan, boolean, guest, binary(spv::OpISub, u64, BdaConstant(state, std::numeric_limits<std::uint64_t>::max()), Unary(state, spv::OpUConvert, u64, size)));
            EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, guest, size, instruction, BdaAbi::FaultReason::Overflow); });
            StopBdaInvocationIf(state, overflow);
        });
        const auto want = Select(state, u32, single, constant(1u), size);
        const auto translate = [&] {
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, translation, value, state.bdaTranslateFunction, guest, want, constant(BdaAbi::Read));
            return value;
        };
        auto translated = 0u;
        if (BdaVectorCacheAvailable(state)) {
            const auto bytes = Unary(state, spv::OpUConvert, u64, want);
            EmitIfCondition(state, binary(spv::OpLogicalAnd, boolean, whole, Unary(state, spv::OpLogicalNot, boolean, ProbeBdaVectorCache(state, guest, bytes).hit)), [&] { RefreshBdaVectorCache(state, guest); });
            const auto cached = ProbeBdaVectorCache(state, guest, bytes);
            translated = EmitValueIfElse(state, binary(spv::OpLogicalAnd, boolean, whole, cached.hit), translation, [&] {
                NoteBdaSegmentRead(state, guest, bytes);
                const auto value = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeConstruct, translation, value, cached.device, zero);
                return value;
            }, translate);
        } else {
            translated = translate();
        }
        const auto device = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, device, translated, 0u);
        const auto reason = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, reason, translated, 1u);
        const auto mapped = equal(reason, zero);
        EmitIfCondition(state, mapped, [&] {
            EmitIfCondition(state, whole, [&] {
                const auto dword = binary(spv::OpLogicalAnd, boolean, equal(size, constant(4u)), Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, device)));
                insert(EmitValueIfElse(state, dword, u32, [&] {
                    const auto pointer = state.module.AllocateId();
                    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, device);
                    const auto value = state.module.AllocateId();
                    if (coherent) state.module.AddFunction(spv::OpAtomicLoad, u32, value, pointer, ConstantU32(state, spv::ScopeDevice), zero);
                    else state.module.AddFunction(spv::OpLoad, u32, value, pointer, accessMask, 4u);
                    return value;
                }, [&] {
                    auto value = byteAt(device);
                    for (std::uint32_t byte = 1u; byte < 4u; ++byte) {
                        const auto part = EmitValueOrZeroIfCondition(state, binary(spv::OpULessThan, boolean, constant(byte), size), [&] { return byteAt(binary(spv::OpIAdd, u64, device, BdaConstant(state, byte))); });
                        value = binary(spv::OpBitwiseOr, u32, value, binary(spv::OpShiftLeftLogical, u32, part, constant(byte * 8u)));
                    }
                    return value;
                }));
            });
            EmitIfCondition(state, single, [&] {
                const auto part = state.module.AllocateId();
                state.module.AddFunction(spv::OpVectorExtractDynamic, u32, part, accumulated, component);
                insert(binary(spv::OpBitwiseOr, u32, part, binary(spv::OpShiftLeftLogical, u32, byteAt(device), binary(spv::OpShiftLeftLogical, u32, within, constant(3u)))));
                nextByte();
            });
        });
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, mapped), [&] {
            const auto again = binary(spv::OpLogicalAnd, boolean, whole, binary(spv::OpUGreaterThan, boolean, size, constant(1u)));
            EmitIfCondition(state, again, [&] {
                state.module.AddFunction(spv::OpStore, cursor, position);
                state.module.AddFunction(spv::OpStore, bytewise, ConstantBool(state, true));
            });
            EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, again), [&] {
                RecordBdaFaultReason(state, guest, constant(1u), instruction, reason);
                StopBdaInvocationIf(state, ConstantBool(state, true));
                nextByte();
            });
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    auto returned = load(values, gathered);
    if (stops) {
        const auto composite = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, result, composite, returned, zero);
        returned = composite;
    }
    state.module.AddFunction(spv::OpReturnValue, returned);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

void DefineBdaComponentReadFunctions(SpirvEmitterState& state) {
    if (state.bdaTranslateFunction == 0 || state.bdaFaultFunction == 0) return;
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    const bool stops = state.bdaStopsInvocations;
    const auto zero = ConstantU32(state, 0u);
    for (const bool stop : {false, true}) {
        if (stop && !stops) continue;
        state.bdaStopsInvocations = stop;
        state.bdaStopValue = stop ? state.module.Constant(spv::OpConstantComposite, BdaSpanReadResultType(state), state.module.Constant(spv::OpConstantComposite, TypeU32Vector(state, 4), zero, zero, zero, zero), ConstantU32(state, 1u)) : 0u;
        for (const bool coherent : {false, true}) {
            if (coherent && !coherentAccesses) continue;
            state.bdaComponentReadFunctions[stop][coherent] = DefineBdaComponentReadFunction(state, coherent);
        }
    }
    state.bdaStopsInvocations = stops;
    state.bdaStopValue = 0;
}

bool BdaComponentReadsAvailable(const SpirvEmitterState& state, const IrValue& inst) {
    return state.bdaComponentReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)] != 0;
}

std::array<std::uint32_t, 4> EmitBdaComponentReads(SpirvValueEmitContext& ctx, const IrValue& inst, const std::array<std::uint32_t, 4>& addresses, const std::array<std::uint32_t, 4>& sizes) {
    auto& state = ctx.state;
    const auto function = state.bdaComponentReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)];
    if (function == 0) ctx.Fail(inst, "BDA component read function is missing");
    const auto u32 = TypeU32(state);
    const auto addressVector = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, state.module.Type(spv::OpTypeVector, TypeScalarU64(state), 4u), addressVector, addresses[0], addresses[1], addresses[2], addresses[3]);
    const auto sizeVector = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), sizeVector, sizes[0], sizes[1], sizes[2], sizes[3]);
    const auto read = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, BdaSpanReadResultType(state), read, function, addressVector, sizeVector, BdaInstructionPc(state, inst));
    if (state.bdaStopsInvocations) {
        const auto stopped = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, stopped, read, 1u);
        StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, TypeBool(state), stopped, ConstantU32(state, 0u)));
    }
    std::array<std::uint32_t, 4> values{};
    for (std::uint32_t component = 0; component < 4u; ++component) {
        values[component] = state.module.AllocateId();
        if (state.bdaStopsInvocations) state.module.AddFunction(spv::OpCompositeExtract, u32, values[component], read, 0u, component);
        else state.module.AddFunction(spv::OpCompositeExtract, u32, values[component], read, component);
    }
    return values;
}

void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits) {
    auto& state = ctx.state;
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA store width");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = BdaInstructionPc(state, inst);
    const auto storeBytes = [&] {
        const auto function = state.bdaByteWriteFunctions[BdaCoherent(state, inst)];
        if (function == 0) ctx.Fail(inst, "BDA byte write function is missing");
        state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), function, address, value, ConstantU32(state, bits / 8u), instruction);
    };
    if (bits != 32u) {
        storeBytes();
        return;
    }
    const auto unaligned = BdaAddressUnaligned(state, address);
    EmitIfCondition(state, unaligned, storeBytes);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), [&] {
        EmitBdaStoreAt(state, address, 4u, TypePhysicalU32Pointer(state), value, instruction, BdaAccessMask(state, inst));
    });
}

void EmitBdaDwordWrites(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, std::uint32_t value) {
    auto& state = ctx.state;
    if (dwords < 2u || dwords > 4u) ctx.Fail(inst, "unsupported BDA dword store count");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    std::array<std::uint32_t, 4> values{};
    std::array<std::uint32_t, 4> addresses{};
    for (std::uint32_t dword = 0; dword < dwords; ++dword) {
        values[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, values[dword], value, dword);
        addresses[dword] = AddBdaImmediate(ctx, inst, address, static_cast<std::int32_t>(offset + dword * 4u));
    }
    const auto storeDwords = [&] {
        for (std::uint32_t dword = 0; dword < dwords; ++dword) EmitBdaStore(ctx, inst, addresses[dword], values[dword], 32u);
    };
    if (state.bdaWriteProbeFunction == 0) {
        storeDwords();
        return;
    }
    const auto instruction = BdaInstructionPc(state, inst);
    const auto accessMask = BdaAccessMask(state, inst);
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, u64, physical, state.bdaWriteProbeFunction, addresses[0], ConstantU32(state, dwords * 4u), instruction);
    const auto mapped = Binary(state, spv::OpINotEqual, boolean, physical, BdaConstant(state, 0u));
    const auto aligned = Binary(state, spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, addresses[0])), Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, physical)));
    const auto wide = Binary(state, spv::OpLogicalAnd, boolean, mapped, aligned);
    const auto wideLabel = state.module.AllocateId();
    const auto dwordLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, wide, wideLabel, dwordLabel);
    EmitLabel(state, wideLabel);
    const auto vectorBytes = dwords == 3u ? 4u : dwords * 4u;
    const auto separate = [&] {
        for (std::uint32_t dword = 0; dword < dwords; ++dword) {
            const auto element = dword == 0u ? physical : Binary(state, spv::OpIAdd, u64, physical, BdaConstant(state, dword * 4u));
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, element);
            state.module.AddFunction(spv::OpStore, pointer, values[dword], accessMask, 4u);
        }
    };
    if (dwords == 3u) {
        separate();
    } else {
        const auto vectorAligned = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u64, physical, BdaConstant(state, vectorBytes - 1u)), BdaConstant(state, 0u));
        const auto vectorType = TypeU32Vector(state, dwords);
        const auto vector = state.module.AllocateId();
        if (dwords == 2u) state.module.AddFunction(spv::OpCompositeConstruct, vectorType, vector, values[0], values[1]);
        else state.module.AddFunction(spv::OpCompositeConstruct, vectorType, vector, values[0], values[1], values[2], values[3]);
        EmitIfCondition(state, vectorAligned, [&] {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, vectorType), pointer, physical);
            state.module.AddFunction(spv::OpStore, pointer, vector, accessMask, vectorBytes);
        });
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, vectorAligned), separate);
    }
    const auto last = addresses[dwords - 1u];
    state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, addresses[0]);
    const auto pageShift = BdaConstant(state, BdaAbi::WrittenPageShift);
    const auto crosses = Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpShiftRightLogical, u64, addresses[0], pageShift), Binary(state, spv::OpShiftRightLogical, u64, last, pageShift));
    EmitIfCondition(state, crosses, [&] {
        state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, last);
    });
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, dwordLabel);
    storeDwords();
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
}

std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation) {
    auto& state = ctx.state;
    if (bytes != 4u && bytes != 8u) ctx.Fail(inst, "unsupported BDA atomic width");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = BdaInstructionPc(state, inst);
    const auto type = bytes == 8u ? TypeScalarU64(state) : TypeU32(state);
    const auto zero = bytes == 8u ? BdaConstant(state, 0u) : ConstantU32(state, 0u);
    const auto unaligned = Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address, BdaConstant(state, bytes - 1u)), BdaConstant(state, 0u));
    EmitIfCondition(state, unaligned, [&] { RecordBdaFault(state, address, ConstantU32(state, bytes), instruction, BdaAbi::FaultReason::Unaligned); });
    return EmitValueOrDefaultIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), type, zero, [&] {
        const auto physical = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, address, ConstantU32(state, bytes), instruction);
        return EmitValueOrDefaultIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), type, zero, [&] {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, type), pointer, physical);
            const auto old = operation(pointer);
            state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, address);
            return old;
        });
    });
}

namespace {

bool ScalarAddressLoad(const SpirvEmitterState& state, const IrValue& inst) {
    switch (inst.Opcode()) {
    case IrOpcode::LoadAddressU32:
    case IrOpcode::LoadAddressU32x2:
    case IrOpcode::LoadAddressU32x3:
    case IrOpcode::LoadAddressU32x4: {
        const auto& memory = state.program.Resources().memoryInfo;
        const auto index = inst.Flags<MemoryFlags>().index;
        return index < memory.size() && memory[index].kind == ResourceKind::ScalarAddress;
    }
    default:
        return false;
    }
}

}

// One probe of the span from the first to the last extracted dword replaces 4 lookups per dword (a
// Bink DC pass spent ~450 dependent table levels per iteration on them). The span is loaded
// dword-wise from the returned device address when it lies in one range and that address is
// 4-aligned; a span crossing ranges (the table is byte-granular, see tests/BdaExecution.cpp), an
// unaligned mirror or a wrapping span falls back to the byte path. That path repeats, per extracted
// dword in order, the address, overflow check and byte lookups of a separate dword load, so every
// fault is recorded exactly as one; the probe records none, and a span it accepts cannot fault.
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword, std::uint32_t usedDwords) {
    auto& state = ctx.state;
    if (dwords == 0u || dwords > 4u) ctx.Fail(inst, "unsupported BDA read width");
    if (state.bdaPointerFunction == 0) ctx.Fail(inst, "BDA lookup function is missing");
    const auto instruction = BdaInstructionPc(state, inst);
    const auto used = usedDwords != 0u ? usedDwords : everyDword ? (1u << dwords) - 1u : UsedBdaDwords(inst, dwords);
    const auto isUsed = [&](std::uint32_t dword) { return (used & (1u << dword)) != 0u; };
    std::array<std::uint32_t, 4> values{};
    std::uint32_t first = dwords;
    std::uint32_t last = 0;
    for (std::uint32_t dword = 0; dword < dwords; ++dword) {
        if (!isUsed(dword)) {
            values[dword] = ConstantU32(state, 0u);
            continue;
        }
        first = std::min(first, dword);
        last = dword;
    }
    if (used == 0u) return values;
    std::array<std::uint32_t, 4> addresses{};
    const auto dwordAddress = [&](std::uint32_t dword) { return AddBdaImmediate(ctx, inst, address, static_cast<std::int32_t>(offset + dword * 4u)); };
    const auto readBytes = [&](std::array<std::uint32_t, 4>& into) {
        const auto function = state.bdaDwordReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)];
        if (function == 0) ctx.Fail(inst, "BDA dword read function is missing");
        for (std::uint32_t dword = first; dword <= last; ++dword) {
            if (!isUsed(dword)) continue;
            if (dword != first) addresses[dword] = dwordAddress(dword);
            const auto read = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 2), read, function, addresses[dword], instruction);
            into[dword] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), into[dword], read, 0u);
            if (!state.bdaStopsInvocations) continue;
            const auto stopped = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), stopped, read, 1u);
            StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, TypeBool(state), stopped, ConstantU32(state, 0u)));
        }
    };
    if (state.bdaProbeFunction == 0) {
        addresses[first] = dwordAddress(first);
        readBytes(values);
        return values;
    }
    const auto function = state.bdaSpanReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)];
    if (function == 0) ctx.Fail(inst, "BDA span read function is missing");
    const auto vector = TypeU32Vector(state, 4);
    const auto span = [&] {
        const auto read = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, BdaSpanReadResultType(state), read, function, address, ConstantU32(state, offset), ConstantU32(state, used), instruction);
        if (!state.bdaStopsInvocations) return read;
        const auto stopped = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), stopped, read, 1u);
        StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, TypeBool(state), stopped, ConstantU32(state, 0u)));
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, vector, loaded, read, 0u);
        return loaded;
    };
    auto read = 0u;
    if (state.bdaAccessSegment[0] != 0u) {
        const auto u64 = TypeScalarU64(state);
        const auto boolean = TypeBool(state);
        const auto start = static_cast<std::int32_t>(offset + first * 4u);
        const auto magnitude = BdaConstant(state, start < 0 ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(start)) : static_cast<std::uint64_t>(start));
        const auto firstAddress = Binary(state, start < 0 ? spv::OpISub : spv::OpIAdd, u64, address, magnitude);
        const auto unwrapped = Binary(state, spv::OpUGreaterThanEqual, boolean, start < 0 ? address : firstAddress, start < 0 ? magnitude : address);
        const auto& segment = state.bdaAccessSegment;
        const auto resolved = ProbeBdaSegment(state, {segment[0], segment[1], segment[2]}, firstAddress, BdaConstant(state, (last - first + 1u) * 4u));
        read = EmitValueIfElse(state, Binary(state, spv::OpLogicalAnd, boolean, unwrapped, resolved.hit), vector, [&] {
            NoteBdaSegmentRead(state, firstAddress, BdaConstant(state, (last - first + 1u) * 4u));
            std::array<std::uint32_t, 4> loaded{};
            for (std::uint32_t dword = 0; dword < 4u; ++dword) {
                loaded[dword] = ConstantU32(state, 0u);
                if (dword < first || dword > last || !isUsed(dword)) continue;
                loaded[dword] = LoadBdaDevice(state, inst, dword == first ? resolved.device : Binary(state, spv::OpIAdd, u64, resolved.device, BdaConstant(state, (dword - first) * 4u)));
            }
            const auto composite = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, vector, composite, loaded[0], loaded[1], loaded[2], loaded[3]);
            return composite;
        }, span);
    } else if (BdaVectorCacheAvailable(state) && (state.module.InHelperFunction() || ScalarAddressLoad(state, inst))) {
        const auto u64 = TypeScalarU64(state);
        const auto boolean = TypeBool(state);
        const auto start = static_cast<std::int32_t>(offset + first * 4u);
        const auto magnitude = BdaConstant(state, start < 0 ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(start)) : static_cast<std::uint64_t>(start));
        const auto firstAddress = Binary(state, start < 0 ? spv::OpISub : spv::OpIAdd, u64, address, magnitude);
        const auto unwrapped = Binary(state, spv::OpUGreaterThanEqual, boolean, start < 0 ? address : firstAddress, start < 0 ? magnitude : address);
        const auto bytes = BdaConstant(state, (last - first + 1u) * 4u);
        const auto missed = Unary(state, spv::OpLogicalNot, boolean, ProbeBdaVectorCache(state, firstAddress, bytes).hit);
        EmitIfCondition(state, Binary(state, spv::OpLogicalAnd, boolean, unwrapped, missed), [&] { RefreshBdaVectorCache(state, firstAddress); });
        const auto cached = ProbeBdaVectorCache(state, firstAddress, bytes);
        read = EmitValueIfElse(state, Binary(state, spv::OpLogicalAnd, boolean, unwrapped, cached.hit), vector, [&] {
            NoteBdaSegmentRead(state, firstAddress, bytes);
            std::array<std::uint32_t, 4> loaded{};
            for (std::uint32_t dword = 0; dword < 4u; ++dword) {
                loaded[dword] = ConstantU32(state, 0u);
                if (dword < first || dword > last || !isUsed(dword)) continue;
                loaded[dword] = LoadBdaDevice(state, inst, dword == first ? cached.device : Binary(state, spv::OpIAdd, u64, cached.device, BdaConstant(state, (dword - first) * 4u)));
            }
            const auto composite = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, vector, composite, loaded[0], loaded[1], loaded[2], loaded[3]);
            return composite;
        }, span);
    } else {
        read = span();
    }
    for (std::uint32_t dword = first; dword <= last; ++dword) {
        if (!isUsed(dword)) continue;
        values[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), values[dword], read, dword);
    }
    return values;
}

}
