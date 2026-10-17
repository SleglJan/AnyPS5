#include "SpirvBackend/SpirvRuntimeBuffer.hpp"
#include "SpirvBackend/SpirvEmitterInstructions.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "SpirvBackend/SpirvMemory/SpirvSubgroup.hpp"
#include "SpirvBackend/SpirvMemory/SpirvInputOutput.hpp"
#include "PipelineSpecialization.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ShaderRecompiler {
namespace {

struct RuntimeDescriptor {
    std::uint32_t base;
    std::uint32_t records;
    std::uint32_t stride;
    std::uint32_t word3;
    std::uint32_t swizzle;
    std::uint32_t index;
    std::uint32_t valid;
    std::uint32_t offset = 0;
    std::uint32_t soffset = 0;
    std::uint32_t address = 0;
    std::uint32_t byteOffset = 0;
    std::uint32_t dwords = 0;
};

struct BufferValues {
    std::array<std::uint32_t, 4> words{};
    std::uint32_t index = 0;
    std::uint32_t lane = 0;
    std::uint32_t offset = 0;
    std::uint32_t soffset = 0;
    std::uint32_t immediate = 0;
    std::uint32_t address = 0;
    std::uint32_t byteOffset = 0;
    std::uint32_t dwords = 0;
};

struct RuntimeAddress {
    std::uint32_t guest;
    std::uint32_t inBounds;
    std::uint32_t range;
};

std::uint32_t field(SpirvEmitterState& state, std::uint32_t word, std::uint32_t offset, std::uint32_t count) {
    return EmitBitFieldUExtract(state, word, ConstantU32(state, offset), ConstantU32(state, count));
}

std::uint32_t both(SpirvEmitterState& state, std::uint32_t left, std::uint32_t right) {
    return Binary(state, spv::OpLogicalAnd, TypeBool(state), left, right);
}

std::uint32_t equal(SpirvEmitterState& state, std::uint32_t left, std::uint32_t right) {
    return Binary(state, spv::OpIEqual, TypeBool(state), left, right);
}

std::uint32_t nonzero(SpirvEmitterState& state, std::uint32_t value) {
    return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0u));
}

void faultIf(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t condition, std::uint32_t address, std::uint32_t bytes, BdaAbi::FaultReason reason) {
    auto& state = context.state;
    EmitIfCondition(state, condition, [&] { RecordBdaFault(state, address, ConstantU32(state, bytes), BdaInstructionPc(state, instruction), reason); });
}

void recordInvalidBufferDescriptor(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t condition, const std::array<std::uint32_t, 4>& words) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    EmitIfCondition(state, condition, [&] {
        const auto u64 = TypeScalarU64(state);
        const auto address = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, words[0]), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, words[1]), BdaConstant(state, 32u)));
        RecordBdaDescriptorFault(state, address, words[2], BdaInstructionPc(state, instruction), memory.resource, words[3]);
    });
}

bool runtimeWords(const SpirvEmitterState& state, const MemoryInfo& memory) {
    return memory.gpuDescriptor || state.registrationModules;
}

bool runtimeFormatWord(const SpirvEmitterState& state, const MemoryInfo& memory) {
    return memory.gpuDescriptor || state.registrationModules;
}

bool computedBase(const IrValue& instruction, const MemoryInfo& memory) {
    return memory.gpuDescriptor || BufferAccessOf(instruction.Opcode()) == BufferAccess::Atomic;
}

std::uint32_t runtimeWordMask(const SpirvEmitterState& state, const IrValue& instruction, const MemoryInfo& memory) {
    return 4u | (runtimeWords(state, memory) || computedBase(instruction, memory) ? 3u : 0u) | (runtimeFormatWord(state, memory) ? 8u : 0u);
}

std::uint32_t specializedWord(SpirvEmitterState& state, const MemoryInfo& memory, std::uint32_t word) {
    return state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::BufferBase + memory.resource * PipelineSpecialization::BufferWords + word, 0u);
}

std::uint32_t formatWord(SpirvEmitterState& state, const MemoryInfo& memory, const BufferValues& values) {
    return runtimeFormatWord(state, memory) ? values.words[3] : specializedWord(state, memory, 1u);
}

bool laneIndexed(const SpirvEmitterState& state, const MemoryInfo& memory) {
    return memory.kind == ResourceKind::Buffer && state.program.Resources().stage == IrShaderStage::Compute;
}

std::uint32_t laneIndex(SpirvEmitterState& state, std::uint32_t word3, std::uint32_t index, std::uint32_t lane) {
    const auto u32 = TypeU32(state);
    return Binary(state, spv::OpIAdd, u32, index, Select(state, u32, nonzero(state, field(state, word3, 23u, 1u)), lane, ConstantU32(state, 0u)));
}

BufferValues instructionValues(SpirvValueEmitContext& context, const IrValue& instruction, bool scalar) {
    auto& state = context.state;
    const auto* handle = instruction.Argument(0)->Resolve();
    if (handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) context.Fail(instruction, "requires a runtime four-dword V#");
    const auto& memory = context.Memory(instruction);
    BufferValues values;
    const auto used = runtimeWordMask(state, instruction, memory);
    for (std::uint32_t word = 0; word < 4u; ++word) {
        if ((used & (1u << word)) != 0u) values.words[word] = context.Arg(*handle, word);
    }
    if (memory.kind == ResourceKind::Buffer) {
        values.index = context.Arg(instruction, 1u);
        values.offset = Binary(state, spv::OpIAdd, TypeU32(state), context.Arg(instruction, 2u), ConstantU32(state, memory.offset));
        values.soffset = context.Arg(instruction, 3u);
    }
    if (scalar) {
        values.offset = context.Arg(instruction, 1u);
        values.immediate = ConstantU32(state, memory.offset);
    }
    if (laneIndexed(state, memory)) {
        values.lane = state.laneCount == 1u && state.hostSubgroupSize < state.program.WaveSize() ?
            Binary(state, spv::OpBitwiseAnd, TypeU32(state), EmitLocalInvocationIndex(state), ConstantU32(state, state.program.WaveSize() - 1u)) : EmitSubgroupLocalInvocationId(state);
    }
    if (!memory.gpuDescriptor) {
        const auto element = ResourceForDescriptor(state, DescriptorBindingKind::Buffers, memory.resource);
        values.address = state.bufferAddresses.at(element);
        values.byteOffset = state.memoryByteOffsets.at(element);
        values.dwords = state.bufferDwords.at(element);
    }
    return values;
}

RuntimeDescriptor descriptor(SpirvValueEmitContext& context, const IrValue& instruction, const BufferValues& values) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const bool runtime = runtimeWords(state, memory);
    const auto word1 = runtime ? values.words[1] : specializedWord(state, memory, 0u);
    const auto word3 = formatWord(state, memory, values);
    auto base = ConstantU32(state, 0u);
    if (computedBase(instruction, memory)) {
        const auto u64 = TypeScalarU64(state);
        const auto high = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, field(state, values.words[1], 0u, 16u)), BdaConstant(state, 32u));
        base = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, values.words[0]), high);
    }
    const bool scalar = memory.kind == ResourceKind::ScalarBuffer;
    auto invalid = scalar ? ConstantBool(state, false) : nonzero(state, field(state, word3, 30u, 2u));
    auto index = ConstantU32(state, 0u);
    if (memory.kind == ResourceKind::Buffer) {
        index = values.index;
        if (laneIndexed(state, memory)) {
            index = laneIndex(state, word3, index, values.lane);
        } else {
            invalid = Binary(state, spv::OpLogicalOr, TypeBool(state), invalid, nonzero(state, field(state, word3, 23u, 1u)));
        }
    }
    if (memory.gpuDescriptor && !scalar) recordInvalidBufferDescriptor(context, instruction, invalid, {values.words[0], word1, values.words[2], word3});
    const auto stride = field(state, word1, 16u, 14u);
    const auto present = runtime ? both(state, nonzero(state, values.words[2]), Binary(state, spv::OpLogicalOr, TypeBool(state), nonzero(state, values.words[0]), nonzero(state, field(state, word1, 0u, 16u)))) : nonzero(state, specializedWord(state, memory, 2u));
    const auto valid = memory.gpuDescriptor ? both(state, Unary(state, spv::OpLogicalNot, TypeBool(state), invalid), Binary(state, spv::OpINotEqual, TypeBool(state), base, BdaConstant(state, 0u))) : present;
    return {base, values.words[2], stride, word3, both(state, nonzero(state, stride), nonzero(state, field(state, word1, 31u, 1u))), index, valid, values.offset, values.soffset, values.address, values.byteOffset, values.dwords};
}

std::uint32_t usable(SpirvEmitterState& state, const RuntimeDescriptor& resource) {
    return both(state, resource.valid, nonzero(state, field(state, resource.word3, 12u, 7u)));
}

RuntimeAddress address(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t componentOffset, std::uint32_t bytes) {
    auto& state = context.state;
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto add = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpIAdd, u32, left, right); };
    const auto mul = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpIMul, u32, left, right); };
    const auto less = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpULessThan, boolean, left, right); };
    const auto fits = [&](std::uint32_t offset, std::uint32_t size) {
        return both(state, Binary(state, spv::OpUGreaterThanEqual, boolean, size, constant(bytes)), Binary(state, spv::OpULessThanEqual, boolean, offset, Binary(state, spv::OpISub, u32, size, constant(bytes))));
    };
    const auto offset = componentOffset != 0u ? add(resource.offset, constant(componentOffset)) : resource.offset;
    const auto soffset = resource.soffset;
    const auto indexShift = add(field(state, resource.word3, 21u, 2u), constant(3u));
    const auto indices = Binary(state, spv::OpShiftLeftLogical, u32, constant(1u), indexShift);
    const auto indexMsb = Binary(state, spv::OpShiftRightLogical, u32, resource.index, indexShift);
    const auto indexLsb = Binary(state, spv::OpBitwiseAnd, u32, resource.index, Binary(state, spv::OpISub, u32, indices, constant(1u)));
    const auto offsetMsb = Binary(state, spv::OpBitwiseAnd, u32, offset, constant(~3u));
    const auto offsetLsb = Binary(state, spv::OpBitwiseAnd, u32, offset, constant(3u));
    const auto swizzled = add(mul(add(mul(indexMsb, resource.stride), offsetMsb), indices), add(Binary(state, spv::OpShiftLeftLogical, u32, indexLsb, constant(2u)), offsetLsb));
    const auto linear = add(mul(resource.index, resource.stride), offset);
    const auto byte = add(Select(state, u32, resource.swizzle, swizzled, linear), soffset);
    const auto mode = field(state, resource.word3, 28u, 2u);
    const auto indexInBounds = less(resource.index, resource.records);
    const auto structured = both(state, indexInBounds, less(offset, resource.stride));
    const auto zeroOffset = soffset == constant(0u);
    const auto rawRecords = zeroOffset ? resource.records : Binary(state, spv::OpISub, u32, resource.records, soffset);
    const auto raw = both(state, Binary(state, spv::OpULessThanEqual, boolean, soffset, resource.records), Select(state, boolean, resource.swizzle, both(state, zeroOffset ? indexInBounds : less(resource.index, rawRecords), fits(offset, resource.stride)), fits(offset, rawRecords)));
    auto inBounds = Select(state, boolean, equal(state, mode, constant(0u)), structured, indexInBounds);
    inBounds = Select(state, boolean, less(mode, constant(2u)), inBounds, Select(state, boolean, equal(state, mode, constant(2u)), nonzero(state, resource.records), raw));
    const auto range = inBounds;
    inBounds = both(state, usable(state, resource), range);
    if (!context.Memory(instruction).gpuDescriptor) return {byte, inBounds, range};
    return {Binary(state, spv::OpIAdd, TypeScalarU64(state), resource.base, Unary(state, spv::OpUConvert, TypeScalarU64(state), byte)), inBounds, range};
}

MemoryResourceAccess directAccess(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, bool store, bool wide = false) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    MemoryResourceAccess access;
    access.kind = memory.kind;
    access.objectPointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalBlockPointer(state, wide), access.objectPointer, resource.address);
    access.byteOffset = resource.byteOffset;
    access.alignment = wide ? 8u : 4u;
    access.memoryAccess = spv::MemoryAccessAlignedMask;
    access.coherent = memory.coherent || (state.requirements.coherentBuffers && (store || !state.requirements.coherentBufferAlias));
    access.length = wide ? Binary(state, spv::OpShiftRightLogical, TypeU32(state), resource.dwords, ConstantU32(state, 1u)) : resource.dwords;
    return access;
}

std::uint32_t baseMisalignment(SpirvValueEmitContext& context, const IrValue& instruction) {
    const auto& memory = context.Memory(instruction);
    if (memory.kind == ResourceKind::ScalarBuffer) return ConstantU32(context.state, 0u);
    auto& state = context.state;
    return state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::BufferBase + memory.resource * PipelineSpecialization::BufferWords + 3u, 0u);
}

std::uint32_t directBaseWord(SpirvEmitterState& state, const MemoryResourceAccess& access) {
    return Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.byteOffset, ConstantU32(state, 2u));
}

std::uint32_t directWordIndex(SpirvEmitterState& state, std::uint32_t baseWord, std::uint32_t address) {
    const auto relative = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2u));
    return EmitAddU32(state, relative, baseWord);
}

struct DirectWords {
    std::uint32_t index;
    std::uint32_t next;
    std::uint32_t inRange;
};

DirectWords directWords(SpirvEmitterState& state, const MemoryResourceAccess& access, std::uint32_t baseWord, std::uint32_t unaligned, std::uint32_t address) {
    const auto index = directWordIndex(state, baseWord, address);
    const auto next = EmitAddU32(state, index, ConstantU32(state, 1u));
    const auto last = Select(state, TypeU32(state), unaligned, next, index);
    return {index, next, EmitMemoryElementInBounds(state, access, last)};
}

std::uint32_t directWordLoad(SpirvEmitterState& state, const MemoryResourceAccess& access, std::uint32_t word) {
    const auto value = state.module.AllocateId();
    const auto pointer = EmitMemoryElementPointer(state, access, word);
    if (access.coherent) state.module.AddFunction(spv::OpAtomicLoad, TypeU32(state), value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone));
    else state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, access.memoryAccess, access.alignment);
    return value;
}

std::uint32_t directMisalignmentShift(SpirvEmitterState& state, std::uint32_t misalignment) {
    return Binary(state, spv::OpShiftLeftLogical, TypeU32(state), misalignment, ConstantU32(state, 3u));
}

std::uint32_t directHighPart(SpirvEmitterState& state, const MemoryResourceAccess& access, std::uint32_t misalignment, std::uint32_t next) {
    const auto remaining = Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 32u), directMisalignmentShift(state, misalignment));
    return Binary(state, spv::OpShiftLeftLogical, TypeU32(state), directWordLoad(state, access, next), remaining);
}

std::uint32_t directValue(SpirvEmitterState& state, std::uint32_t unaligned, std::uint32_t misalignment, std::uint32_t low, std::uint32_t high, std::uint32_t address, std::uint32_t bits) {
    const auto shifted = Binary(state, spv::OpShiftRightLogical, TypeU32(state), low, directMisalignmentShift(state, misalignment));
    auto value = Select(state, TypeU32(state), unaligned, EmitOrU32(state, shifted, high), low);
    if (bits == 32u) return value;
    const auto subwordShift = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), EmitAndConstant(state, address, 3u), ConstantU32(state, 3u));
    value = Binary(state, spv::OpShiftRightLogical, TypeU32(state), value, subwordShift);
    return EmitAndConstant(state, value, (1u << bits) - 1u);
}

std::uint32_t readBuffer(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t address, std::uint32_t bits) {
    if (context.Memory(instruction).gpuDescriptor) return EmitBdaRead(context, instruction, address, bits);
    auto& state = context.state;
    const auto access = directAccess(context, instruction, resource, false);
    const auto misalignment = baseMisalignment(context, instruction);
    const auto unaligned = nonzero(state, misalignment);
    const auto words = directWords(state, access, directBaseWord(state, access), unaligned, address);
    return EmitValueOrZeroIfCondition(state, words.inRange, [&] {
        const auto low = directWordLoad(state, access, words.index);
        const auto high = EmitValueOrZeroIfCondition(state, unaligned, [&] { return directHighPart(state, access, misalignment, words.next); });
        return directValue(state, unaligned, misalignment, low, high, address, bits);
    });
}

std::uint32_t mergeWriter(SpirvEmitterState& state, std::uint32_t word, std::uint32_t byte) {
    const auto& capabilities = state.supportedCapabilities;
    const bool ballot = std::find(capabilities.begin(), capabilities.end(), static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformBallot)) != capabilities.end();
    if (state.program.Resources().stage != IrShaderStage::Compute || HostSubgroupSingleLane(state) || !ballot) return ConstantBool(state, true);
    state.module.EmitCapability(spv::CapabilityGroupNonUniform);
    state.module.EmitCapability(spv::CapabilityGroupNonUniformBallot);
    const auto first = [&](std::uint32_t value) {
        const auto result = state.module.AllocateId();
        state.module.AddFunction(spv::OpGroupNonUniformBroadcastFirst, TypeU32(state), result, ConstantU32(state, spv::ScopeSubgroup), value);
        return result;
    };
    const auto shared = both(state, equal(state, word, first(word)), equal(state, byte, first(byte)));
    const auto elected = state.module.AllocateId();
    EmitSubgroupElect(state, elected);
    return Binary(state, spv::OpLogicalOr, TypeBool(state), Unary(state, spv::OpLogicalNot, TypeBool(state), shared), elected);
}

bool byteStores(SpirvEmitterState& state) {
    if (state.program.Resources().stage != IrShaderStage::Compute) return false;
    if (std::find(state.supportedCapabilities.begin(), state.supportedCapabilities.end(), static_cast<std::uint32_t>(spv::CapabilityStorageBuffer8BitAccess)) == state.supportedCapabilities.end()) return false;
    state.module.EmitCapability(spv::CapabilityStorageBuffer8BitAccess);
    state.module.EmitExtension("SPV_KHR_8bit_storage");
    return true;
}

void writeBuffer(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t address, std::uint32_t value, std::uint32_t bits = 32u) {
    auto& state = context.state;
    if (context.Memory(instruction).gpuDescriptor) {
        EmitBdaWrite(context, instruction, address, value, bits, ConstantBool(state, true));
        return;
    }
    const auto access = directAccess(context, instruction, resource, true);
    const auto u32 = TypeU32(state);
    const auto index = directWordIndex(state, directBaseWord(state, access), address);
    const auto misalignment = baseMisalignment(context, instruction);
    const auto unaligned = nonzero(state, misalignment);
    const auto next = EmitAddU32(state, index, ConstantU32(state, 1u));
    const auto last = Select(state, u32, unaligned, next, index);
    if (bits != 32u && !access.coherent && byteStores(state)) {
        const auto block = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU8BlockPointer(state), block, resource.address);
        EmitIfCondition(state, EmitMemoryElementInBounds(state, access, last), [&] {
            const auto first = EmitAddU32(state, EmitAddU32(state, Binary(state, spv::OpShiftLeftLogical, u32, index, ConstantU32(state, 2u)), misalignment), EmitAndConstant(state, address, 3u));
            for (std::uint32_t byte = 0; byte < bits / 8u; ++byte) {
                const auto pointer = state.module.AllocateId();
                state.module.AddFunction(spv::OpAccessChain, TypePhysicalU8Pointer(state), pointer, block, ConstantU32(state, 0u), EmitAddU32(state, first, ConstantU32(state, byte)));
                const auto shifted = byte == 0u ? value : Binary(state, spv::OpShiftRightLogical, u32, value, ConstantU32(state, byte * 8u));
                const auto stored = Unary(state, spv::OpUConvert, TypeU8(state), shifted);
                state.module.AddFunction(spv::OpStore, pointer, stored, spv::MemoryAccessAlignedMask, 1u);
            }
        });
        return;
    }
    EmitIfCondition(state, EmitMemoryElementInBounds(state, access, last), [&] {
        auto mask = ConstantU32(state, bits == 32u ? 0xffffffffu : (1u << bits) - 1u);
        auto masked = Binary(state, spv::OpBitwiseAnd, u32, value, mask);
        if (bits != 32u) {
            const auto subwordShift = Binary(state, spv::OpShiftLeftLogical, u32, EmitAndConstant(state, address, 3u), ConstantU32(state, 3u));
            mask = Binary(state, spv::OpShiftLeftLogical, u32, mask, subwordShift);
            masked = Binary(state, spv::OpShiftLeftLogical, u32, masked, subwordShift);
        }
        const auto update = [&](std::uint32_t word, std::uint32_t wordMask, std::uint32_t replacement) {
            AtomicUpdate(state, EmitMemoryElementPointer(state, access, word), access.kind, [&](std::uint32_t old) {
                return EmitOrU32(state, Binary(state, spv::OpBitwiseAnd, u32, old, Unary(state, spv::OpNot, u32, wordMask)), replacement);
            });
        };
        if (bits == 32u) {
            EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), [&] {
                const auto pointer = EmitMemoryElementPointer(state, access, index);
                if (access.coherent) state.module.AddFunction(spv::OpAtomicStore, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), value);
                else state.module.AddFunction(spv::OpStore, pointer, value, access.memoryAccess, access.alignment);
            });
        }
        EmitIfCondition(state, bits == 32u ? unaligned : ConstantBool(state, true), [&] {
            EmitIfCondition(state, mergeWriter(state, index, EmitAndConstant(state, address, 3u)), [&] {
                const auto shift = Binary(state, spv::OpShiftLeftLogical, u32, misalignment, ConstantU32(state, 3u));
                update(index, Binary(state, spv::OpShiftLeftLogical, u32, mask, shift), Binary(state, spv::OpShiftLeftLogical, u32, masked, shift));
                EmitIfCondition(state, unaligned, [&] {
                    const auto remaining = Binary(state, spv::OpISub, u32, ConstantU32(state, 32u), shift);
                    update(next, Binary(state, spv::OpShiftRightLogical, u32, mask, remaining), Binary(state, spv::OpShiftRightLogical, u32, masked, remaining));
                });
            });
        });
    });
}

std::uint32_t composite(SpirvEmitterState& state, std::uint32_t components, const std::array<std::uint32_t, 4>& values) {
    if (components == 1u) return values[0];
    const auto result = state.module.AllocateId();
    std::vector<std::uint32_t> words{spv::OpCompositeConstruct, TypeU32Composite(state, components), result};
    words.insert(words.end(), values.begin(), values.begin() + components);
    state.module.AddFunction(words);
    return result;
}

std::uint32_t storeData(SpirvValueEmitContext& context, const IrValue& instruction) {
    return context.Arg(instruction, instruction.ArgumentCount() - 2u);
}

std::array<std::uint32_t, 4> storeValues(SpirvEmitterState& state, std::uint32_t data, std::uint32_t components) {
    std::array<std::uint32_t, 4> result{};
    for (std::uint32_t component = 0; component < components; ++component) {
        result[component] = data;
        if (components != 1u) {
            result[component] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), result[component], data, component);
        }
    }
    return result;
}

std::uint32_t directLoad(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t components, std::uint32_t active, std::uint32_t type, std::uint32_t zero, const BufferValues& operands) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    const auto resource = descriptor(context, instruction, operands);
    const auto access = directAccess(context, instruction, resource, false);
    const auto misalignment = baseMisalignment(context, instruction);
    const auto unaligned = nonzero(state, misalignment);
    const auto baseWord = directBaseWord(state, access);
    std::array<std::uint32_t, 4> guests{};
    std::array<DirectWords, 4> words{};
    std::array<std::uint32_t, 4> valid{};
    std::uint32_t any = 0;
    for (std::uint32_t component = 0; component < components; ++component) {
        const auto location = address(context, instruction, resource, component * 4u, memory.dataBits / 8u);
        guests[component] = location.guest;
        words[component] = directWords(state, access, baseWord, unaligned, location.guest);
        valid[component] = both(state, location.range, words[component].inRange);
        any = component == 0u ? valid[component] : Binary(state, spv::OpLogicalOr, TypeBool(state), any, valid[component]);
    }
    return EmitValueOrDefaultIfCondition(state, both(state, active, both(state, usable(state, resource), any)), type, zero, [&] {
        const auto whenValid = [&](std::uint32_t component, std::uint32_t value) { return components == 1u ? value : Select(state, u32, valid[component], value, ConstantU32(state, 0u)); };
        std::array<std::uint32_t, 4> lows{};
        for (std::uint32_t component = 0; component < components; ++component) lows[component] = directWordLoad(state, access, whenValid(component, words[component].index));
        const auto highs = EmitValuesOrDefaultIfCondition(state, unaligned, u32, ConstantU32(state, 0u), components, [&] {
            std::array<std::uint32_t, 4> parts{};
            for (std::uint32_t component = 0; component < components; ++component) parts[component] = directHighPart(state, access, misalignment, whenValid(component, words[component].next));
            return parts;
        });
        std::array<std::uint32_t, 4> values{};
        for (std::uint32_t component = 0; component < components; ++component) values[component] = whenValid(component, directValue(state, unaligned, misalignment, lows[component], highs[component], guests[component], memory.dataBits));
        return composite(state, components, values);
    });
}

struct FormattedLayout {
    std::array<std::uint32_t, 4> selectors{};
    std::array<std::uint32_t, 4> selected{};
    std::array<std::uint32_t, 4> required{};
    std::array<RuntimeAddress, 4> addresses{};
    std::uint32_t inBounds = 0;
};

FormattedLayout formattedLayout(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, const SpirvBufferFormatInfo& info, std::uint32_t components, bool store) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    FormattedLayout layout;
    layout.inBounds = ConstantBool(state, true);
    for (std::uint32_t output = 0; output < components; ++output) {
        layout.selectors[output] = memory.typed ? ConstantU32(state, output < info.componentCount ? 4u + output : 0u) : field(state, resource.word3, output * 3u, 3u);
        layout.selected[output] = Binary(state, spv::OpUMod, u32, Binary(state, spv::OpISub, u32, layout.selectors[output], ConstantU32(state, 4u)), ConstantU32(state, info.componentCount));
    }
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        layout.required[component] = ConstantBool(state, store);
        if (!store) {
            for (std::uint32_t output = 0; output < components; ++output) {
                const auto fromMemory = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), layout.selectors[output], ConstantU32(state, 4u));
                layout.required[component] = Binary(state, spv::OpLogicalOr, TypeBool(state), layout.required[component], both(state, fromMemory, equal(state, layout.selected[output], ConstantU32(state, component))));
            }
        }
        layout.addresses[component] = address(context, instruction, resource, GetFormatComponentByteOffset(info, component), info.packedBitfield ? 4u : info.componentBits[component] / 8u);
        layout.inBounds = both(state, layout.inBounds, Binary(state, spv::OpLogicalOr, TypeBool(state), Unary(state, spv::OpLogicalNot, TypeBool(state), layout.required[component]), layout.addresses[component].inBounds));
    }
    return layout;
}

std::uint32_t decodeFormatted(SpirvEmitterState& state, const MemoryInfo& memory, const SpirvBufferFormatInfo& info, std::uint32_t component, std::uint32_t raw) {
    const auto u32 = TypeU32(state);
    const auto bits = info.componentBits[component];
    if (bits != 32u) {
        const auto signedType = IsSignedFormatComponent(info.type);
        const auto type = signedType ? TypeI32(state) : u32;
        const auto input = signedType ? Unary(state, spv::OpBitcast, type, raw) : raw;
        raw = state.module.AllocateId();
        state.module.AddFunction(signedType ? spv::OpBitFieldSExtract : spv::OpBitFieldUExtract, type, raw, input, ConstantU32(state, info.packedBitfield ? info.componentBitOffset[component] : 0u), ConstantU32(state, bits));
        if (signedType) raw = Unary(state, spv::OpBitcast, u32, raw);
    }
    return memory.d16 ? EmitD16FormatComponent(state, info, component, raw) : NormalizeFormatComponent(state, info, component, raw);
}

std::uint32_t encodeFormatted(SpirvEmitterState& state, const MemoryInfo& memory, const SpirvBufferFormatInfo& info, std::uint32_t component, std::uint32_t components, const std::array<std::uint32_t, 4>& values) {
    return component >= components ? ConstantU32(state, 0u) : memory.d16 ? EmitD16StoreComponent(state, info, component, values[component]) : EmitFormatStoreComponent(state, info, component, values[component]);
}

std::uint32_t formattedOne(SpirvEmitterState& state, const MemoryInfo& memory, const SpirvBufferFormatInfo& info) {
    const bool integer = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint;
    return ConstantU32(state, integer ? 1u : memory.d16 ? 0x3c00u : 0x3f800000u);
}

std::uint32_t selectFormatted(SpirvEmitterState& state, std::uint32_t components, const SpirvBufferFormatInfo& info, const FormattedLayout& layout, const std::array<std::uint32_t, 4>& values, std::uint32_t one) {
    const auto u32 = TypeU32(state);
    std::array<std::uint32_t, 4> outputs{};
    std::uint32_t vector = 0;
    if (info.componentCount > 2u) {
        std::vector<std::uint32_t> construct{spv::OpCompositeConstruct, TypeU32Vector(state, info.componentCount), state.module.AllocateId()};
        construct.insert(construct.end(), values.begin(), values.begin() + info.componentCount);
        state.module.AddFunction(construct);
        vector = construct[2];
    }
    for (std::uint32_t output = 0; output < components; ++output) {
        auto value = values[0];
        if (vector != 0u) {
            value = state.module.AllocateId();
            state.module.AddFunction(spv::OpVectorExtractDynamic, u32, value, vector, layout.selected[output]);
        }
        for (std::uint32_t component = 1; vector == 0u && component < info.componentCount; ++component) value = Select(state, u32, equal(state, layout.selected[output], ConstantU32(state, component)), values[component], value);
        value = Select(state, u32, equal(state, layout.selectors[output], ConstantU32(state, 1u)), one, value);
        outputs[output] = Select(state, u32, equal(state, layout.selectors[output], ConstantU32(state, 0u)), ConstantU32(state, 0u), value);
    }
    return composite(state, components, outputs);
}

void writeFormatted(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, const SpirvBufferFormatInfo& info, const FormattedLayout& layout, const std::array<std::uint32_t, 4>& encoded) {
    auto& state = context.state;
    const auto u32 = TypeU32(state);
    auto packed = ConstantU32(state, 0u);
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        if (info.packedBitfield) {
            packed = EmitOrU32(state, packed, Binary(state, spv::OpShiftLeftLogical, u32, EmitAndConstant(state, encoded[component], (1u << info.componentBits[component]) - 1u), ConstantU32(state, info.componentBitOffset[component])));
        } else {
            writeBuffer(context, instruction, resource, layout.addresses[component].guest, encoded[component], info.componentBits[component]);
        }
    }
    if (info.packedBitfield) writeBuffer(context, instruction, resource, layout.addresses[0].guest, packed);
}

std::uint32_t formatted(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, const SpirvBufferFormatInfo& info, std::uint32_t components, bool store, std::uint32_t data) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto layout = formattedLayout(context, instruction, resource, info, components, store);
    if (store) {
        const auto values = storeValues(state, data, components);
        EmitIfCondition(state, layout.inBounds, [&] {
            std::array<std::uint32_t, 4> encoded{};
            for (std::uint32_t component = 0; component < info.componentCount; ++component) encoded[component] = encodeFormatted(state, memory, info, component, components, values);
            writeFormatted(context, instruction, resource, info, layout, encoded);
        });
        return ConstantU32(state, 0u);
    }
    std::array<std::uint32_t, 4> values{};
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        values[component] = EmitValueOrZeroIfCondition(state, both(state, layout.inBounds, layout.required[component]), [&] {
            const auto raw = readBuffer(context, instruction, resource, layout.addresses[component].guest, info.packedBitfield ? 32u : info.componentBits[component]);
            return decodeFormatted(state, memory, info, component, raw);
        });
    }
    return selectFormatted(state, components, info, layout, values, formattedOne(state, memory, info));
}

template<typename TBody>
std::vector<std::uint32_t> switchValues(SpirvEmitterState& state, std::uint32_t selector, const std::vector<std::vector<std::uint32_t>>& cases, bool firstIsDefault, const std::vector<std::uint32_t>& types, TBody&& body) {
    if (firstIsDefault && cases.size() == 1u) return body(0u);
    const auto merge = state.module.AllocateId();
    std::vector<std::uint32_t> labels(cases.size() + (firstIsDefault ? 0u : 1u));
    for (auto& label : labels) label = state.module.AllocateId();
    std::vector<std::uint32_t> branch{spv::OpSwitch, selector, firstIsDefault ? labels.front() : labels.back()};
    for (std::size_t index = firstIsDefault ? 1u : 0u; index < cases.size(); ++index) {
        for (const auto literal : cases[index]) branch.insert(branch.end(), {literal, labels[index]});
    }
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(branch);
    std::vector<std::vector<std::uint32_t>> phis;
    for (const auto type : types) phis.push_back({spv::OpPhi, type, state.module.AllocateId()});
    for (std::size_t index = 0; index < labels.size(); ++index) {
        EmitLabel(state, labels[index]);
        const auto values = body(index);
        const auto exit = state.module.AllocateId();
        state.module.AddFunction(spv::OpBranch, exit);
        EmitLabel(state, exit);
        state.module.AddFunction(spv::OpBranch, merge);
        for (std::size_t value = 0; value < phis.size(); ++value) phis[value].insert(phis[value].end(), {values.at(value), exit});
    }
    EmitLabel(state, merge);
    std::vector<std::uint32_t> results;
    for (const auto& phi : phis) {
        state.module.AddFunction(phi);
        results.push_back(phi[2]);
    }
    return results;
}

std::vector<std::uint32_t> formatLayoutKey(const SpirvBufferFormatInfo& info) {
    std::vector<std::uint32_t> key{info.componentCount, info.packedBitfield ? 1u : 0u};
    for (std::uint32_t component = 0; component < info.componentCount; ++component) key.insert(key.end(), {info.componentBits[component], info.componentBitOffset[component], GetFormatComponentByteOffset(info, component)});
    return key;
}

std::vector<std::vector<std::uint32_t>> formatLayouts() {
    constexpr std::uint32_t count = static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float);
    std::vector<std::vector<std::uint32_t>> layouts;
    std::map<std::vector<std::uint32_t>, std::size_t> byKey;
    for (std::uint32_t format = 1; format <= count; ++format) {
        const auto found = byKey.emplace(formatLayoutKey(GetFormatInfo(static_cast<IrBufferFormat>(format))), layouts.size()).first;
        if (found->second == layouts.size()) layouts.emplace_back();
        layouts[found->second].push_back(format);
    }
    return layouts;
}

const std::vector<std::vector<std::uint32_t>>& sharedFormatLayouts() {
    static const auto layouts = formatLayouts();
    return layouts;
}

std::vector<std::vector<std::uint32_t>> formatMembers(std::size_t layout) {
    std::vector<std::vector<std::uint32_t>> cases;
    for (const auto member : sharedFormatLayouts()[layout]) cases.push_back({member});
    return cases;
}

std::uint32_t codecKey(std::size_t layout, bool d16, std::uint32_t components) {
    return static_cast<std::uint32_t>(layout) << 4u | (d16 ? 8u : 0u) | components;
}

std::uint32_t codecType(SpirvEmitterState& state, std::uint32_t results) {
    const std::vector<std::uint32_t> members(results, TypeU32(state));
    return state.module.Type(spv::OpTypeStruct, std::span<const std::uint32_t>(members));
}

std::vector<std::uint32_t> callCodec(SpirvEmitterState& state, std::map<std::uint32_t, std::uint32_t>& functions, std::uint32_t key, std::uint32_t format, std::span<const std::uint32_t> arguments, std::uint32_t results) {
    auto& function = functions[key];
    if (function == 0u) function = state.module.AllocateId();
    std::vector<std::uint32_t> call{spv::OpFunctionCall, codecType(state, results), state.module.AllocateId(), function, format};
    call.insert(call.end(), arguments.begin(), arguments.end());
    state.module.AddFunction(call);
    std::vector<std::uint32_t> values(results);
    for (std::uint32_t index = 0; index < results; ++index) {
        values[index] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), values[index], call[2], index);
    }
    return values;
}

std::uint32_t formatSwitch(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    const auto format = field(state, resource.word3, 12u, 7u);
    const auto& layouts = sharedFormatLayouts();
    const auto type = store || components == 1u ? u32 : TypeU32Composite(state, components);
    const auto zero = store || components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components);
    const auto members = [&](std::size_t layout) { return formatMembers(layout); };
    const auto result = switchValues(state, format, layouts, false, store ? std::vector<std::uint32_t>{} : std::vector<std::uint32_t>{type}, [&](std::size_t index) -> std::vector<std::uint32_t> {
        if (index == layouts.size()) {
            if (memory.gpuDescriptor) RecordBdaFault(state, resource.base, ConstantU32(state, 0u), BdaInstructionPc(state, instruction), BdaAbi::FaultReason::InvalidDescriptor);
            return {zero};
        }
        const auto info = GetFormatInfo(static_cast<IrBufferFormat>(layouts[index].front()));
        const auto layout = formattedLayout(context, instruction, resource, info, components, store);
        const auto memberInfo = [&](std::size_t member) { return GetFormatInfo(static_cast<IrBufferFormat>(layouts[index][member])); };
        if (store) {
            const auto values = storeValues(state, data, components);
            EmitIfCondition(state, layout.inBounds, [&] {
                const auto encoded = layouts[index].size() > 1u ? callCodec(state, state.formatEncoders, codecKey(index, memory.d16, components), format, std::span<const std::uint32_t>(values.data(), components), info.componentCount) : switchValues(state, format, members(index), true, std::vector<std::uint32_t>(info.componentCount, u32), [&](std::size_t member) {
                    std::vector<std::uint32_t> parts;
                    for (std::uint32_t component = 0; component < info.componentCount; ++component) parts.push_back(encodeFormatted(state, memory, memberInfo(member), component, components, values));
                    return parts;
                });
                std::array<std::uint32_t, 4> parts{};
                std::copy(encoded.begin(), encoded.end(), parts.begin());
                writeFormatted(context, instruction, resource, info, layout, parts);
            });
            return {};
        }
        std::array<std::uint32_t, 4> read{};
        std::array<std::uint32_t, 4> raw{};
        for (std::uint32_t component = 0; component < info.componentCount; ++component) {
            read[component] = both(state, layout.inBounds, layout.required[component]);
            raw[component] = EmitValueOrZeroIfCondition(state, read[component], [&] { return readBuffer(context, instruction, resource, layout.addresses[component].guest, info.packedBitfield ? 32u : info.componentBits[component]); });
        }
        auto decoded = layouts[index].size() > 1u ? callCodec(state, state.formatDecoders, codecKey(index, memory.d16, 0u), format, std::span<const std::uint32_t>(raw.data(), info.componentCount), info.componentCount + 1u) : switchValues(state, format, members(index), true, std::vector<std::uint32_t>(info.componentCount + 1u, u32), [&](std::size_t member) {
            std::vector<std::uint32_t> parts;
            for (std::uint32_t component = 0; component < info.componentCount; ++component) parts.push_back(decodeFormatted(state, memory, memberInfo(member), component, raw[component]));
            parts.push_back(formattedOne(state, memory, memberInfo(member)));
            return parts;
        });
        for (std::uint32_t component = 0; component < info.componentCount; ++component) decoded[component] = Select(state, u32, read[component], decoded[component], ConstantU32(state, 0u));
        std::array<std::uint32_t, 4> values{};
        std::copy(decoded.begin(), decoded.begin() + info.componentCount, values.begin());
        return {selectFormatted(state, components, info, layout, values, decoded[info.componentCount])};
    });
    return store ? ConstantU32(state, 0u) : result.front();
}

std::uint32_t formattedDispatch(SpirvValueEmitContext& context, const IrValue& instruction, const RuntimeDescriptor& resource, std::uint32_t components, bool store, std::uint32_t data) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    if (memory.typed) {
        const auto info = GetFormatInfo(DecodeTBufferFormat(memory.dataFormat, memory.numberFormat));
        if (info.type == SpirvFormatComponentType::Unknown) context.Fail(instruction, "typed buffer instruction has an invalid format");
        return formatted(context, instruction, resource, info, components, store, data);
    }
    if (!store && memory.gpuDescriptor) {
        auto invalid = ConstantBool(state, false);
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto selector = field(state, resource.word3, component * 3u, 3u);
            invalid = Binary(state, spv::OpLogicalOr, TypeBool(state), invalid, Binary(state, spv::OpLogicalOr, TypeBool(state), equal(state, selector, ConstantU32(state, 2u)), equal(state, selector, ConstantU32(state, 3u))));
        }
        faultIf(context, instruction, invalid, resource.base, 0u, BdaAbi::FaultReason::InvalidDescriptor);
    }
    const auto* word = instruction.Argument(0)->Resolve()->Argument(3)->Resolve();
    if (memory.gpuDescriptor && word->HasImmediate()) {
        const auto format = static_cast<IrBufferFormat>((word->ImmediateU32() >> 12u) & 0x7fu);
        if (format == IrBufferFormat::Invalid) context.Fail(instruction, "formatted buffer instruction has an invalid constant format");
        return formatted(context, instruction, resource, GetFormatInfo(format), components, store, data);
    }
    return formatSwitch(context, instruction, resource, components, store, data);
}

bool runtimeFormatted(const IrValue& instruction, const MemoryInfo& memory) {
    return memory.formatted && !memory.typed && !(memory.gpuDescriptor && instruction.Argument(0)->Resolve()->Argument(3)->Resolve()->HasImmediate());
}

std::uint32_t formattedAccess(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t components, bool store, std::uint32_t active, const BufferValues& operands, std::uint32_t data) {
    auto& state = context.state;
    if (store) {
        EmitIfCondition(state, active, [&] {
            const auto resource = descriptor(context, instruction, operands);
            EmitIfCondition(state, resource.valid, [&] { static_cast<void>(formattedDispatch(context, instruction, resource, components, true, data)); });
        });
        return 0u;
    }
    const auto type = components == 1u ? TypeU32(state) : TypeU32Composite(state, components);
    const auto zero = components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components);
    return EmitValueOrDefaultIfCondition(state, active, type, zero, [&] {
        const auto resource = descriptor(context, instruction, operands);
        return EmitValueOrDefaultIfCondition(state, resource.valid, type, zero, [&] { return formattedDispatch(context, instruction, resource, components, false, 0u); });
    });
}

std::uint32_t rawLoad(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t components, std::uint32_t active, const BufferValues& operands) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto type = components == 1u ? TypeU32(state) : TypeU32Composite(state, components);
    const auto zero = components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components);
    const bool wideAtomic = components == 2u && memory.coherent && state.requirements.bufferInt64Atomics;
    if (!memory.gpuDescriptor && !wideAtomic) return directLoad(context, instruction, components, active, type, zero, operands);
    return EmitValueOrDefaultIfCondition(state, active, type, zero, [&] {
        const auto resource = descriptor(context, instruction, operands);
        const auto readComponents = [&] {
            std::array<std::uint32_t, 4> values{};
            for (std::uint32_t component = 0; component < components; ++component) {
                const auto access = address(context, instruction, resource, component * 4u, memory.dataBits / 8u);
                values[component] = EmitValueOrZeroIfCondition(state, access.inBounds, [&] { return readBuffer(context, instruction, resource, access.guest, memory.dataBits); });
            }
            return composite(state, components, values);
        };
        const auto pair = [&](std::uint32_t value) {
            const auto high = Binary(state, spv::OpShiftRightLogical, TypeScalarU64(state), value, BdaConstant(state, 32u));
            return composite(state, 2u, {Unary(state, spv::OpUConvert, TypeU32(state), value), Unary(state, spv::OpUConvert, TypeU32(state), high), 0u, 0u});
        };
        if (!wideAtomic || (memory.gpuDescriptor && state.bdaProbeFunction == 0u)) return readComponents();
        const auto runtime = address(context, instruction, resource, 0u, 8u);
        if (memory.gpuDescriptor) {
            const auto u64 = TypeScalarU64(state);
            const auto guestAligned = Binary(state, spv::OpIEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, u64, runtime.guest, BdaConstant(state, 7u)), BdaConstant(state, 0u));
            const auto physical = EmitValueOrDefaultIfCondition(state, both(state, runtime.inBounds, guestAligned), u64, BdaConstant(state, 0u), [&] {
                const auto probed = state.module.AllocateId();
                state.module.AddFunction(spv::OpFunctionCall, u64, probed, state.bdaProbeFunction, runtime.guest, ConstantU32(state, 8u), BdaInstructionPc(state, instruction));
                return probed;
            });
            const auto mapped = both(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), Binary(state, spv::OpIEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, u64, physical, BdaConstant(state, 7u)), BdaConstant(state, 0u)));
            return EmitValueIfElse(state, mapped, type, [&] {
                const auto pointer = state.module.AllocateId();
                state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU64Pointer(state), pointer, physical);
                const auto value = state.module.AllocateId();
                state.module.AddFunction(spv::OpAtomicLoad, u64, value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone));
                return pair(value);
            }, readComponents);
        }
        const auto wide = directAccess(context, instruction, resource, false, true);
        const auto byte = EmitAddU32(state, runtime.guest, wide.byteOffset);
        const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte, ConstantU32(state, 3u));
        const auto aligned = both(state, equal(state, baseMisalignment(context, instruction), ConstantU32(state, 0u)), equal(state, EmitAndConstant(state, byte, 7u), ConstantU32(state, 0u)));
        const auto inBounds = both(state, runtime.inBounds, EmitMemoryElementInBounds(state, wide, index));
        return EmitValueIfElse(state, both(state, aligned, inBounds), type, [&] {
            const auto pointer = EmitStorageBufferElementPointer(state, wide, index, TypePhysicalU64Pointer(state));
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicLoad, TypeScalarU64(state), value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone));
            return pair(value);
        }, readComponents);
    });
}

void rawStore(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t components, std::uint32_t active, const BufferValues& operands, std::uint32_t data) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    EmitIfCondition(state, active, [&] {
        const auto resource = descriptor(context, instruction, operands);
        const auto values = storeValues(state, data, components);
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto access = address(context, instruction, resource, component * 4u, memory.dataBits / 8u);
            if (memory.gpuDescriptor) EmitBdaWrite(context, instruction, access.guest, values[component], memory.dataBits, access.inBounds);
            else EmitIfCondition(state, access.inBounds, [&] { writeBuffer(context, instruction, resource, access.guest, values[component], memory.dataBits); });
        }
    });
}

struct ScalarAccess {
    RuntimeDescriptor resource;
    std::uint32_t byte;
    std::uint32_t offset;
    std::uint32_t inBounds;
};

ScalarAccess scalarAccess(SpirvValueEmitContext& context, const IrValue& instruction, const BufferValues& operands, std::uint32_t bytes = 4u) {
    auto& state = context.state;
    const auto resource = descriptor(context, instruction, operands);
    const auto u64 = TypeScalarU64(state);
    const auto size = Select(state, u64, equal(state, resource.stride, ConstantU32(state, 0u)), Unary(state, spv::OpUConvert, u64, resource.records), Binary(state, spv::OpIMul, u64, Unary(state, spv::OpUConvert, u64, resource.stride), Unary(state, spv::OpUConvert, u64, resource.records)));
    const auto address = EmitAddU32(state, operands.offset, operands.immediate);
    const auto noCarry = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address, operands.offset);
    const auto byte = EmitAndConstant(state, address, ~3u);
    const auto offset = Unary(state, spv::OpUConvert, u64, byte);
    const auto inBounds = both(state, noCarry, both(state, resource.valid, Binary(state, spv::OpULessThanEqual, TypeBool(state), Binary(state, spv::OpIAdd, u64, offset, BdaConstant(state, bytes)), size)));
    return {resource, byte, offset, inBounds};
}

std::uint32_t scalarGuestAddress(SpirvEmitterState& state, const ScalarAccess& access) {
    const auto u64 = TypeScalarU64(state);
    return Binary(state, spv::OpIAdd, u64, Binary(state, spv::OpBitwiseAnd, u64, access.resource.base, BdaConstant(state, ~std::uint64_t{3u})), access.offset);
}

std::uint32_t scalarLoad(SpirvValueEmitContext& context, const IrValue& instruction, const BufferValues& operands) {
    auto& state = context.state;
    const auto [resource, byte, offset, inBounds] = scalarAccess(context, instruction, operands);
    if (!context.Memory(instruction).gpuDescriptor) {
        const auto access = directAccess(context, instruction, resource, false);
        const auto index = directWordIndex(state, directBaseWord(state, access), byte);
        return EmitValueOrZeroIfCondition(state, both(state, inBounds, EmitMemoryElementInBounds(state, access, index)), [&] { return directWordLoad(state, access, index); });
    }
    const auto guest = scalarGuestAddress(state, {resource, byte, offset, inBounds});
    const auto uncached = [&] { return EmitValueOrZeroIfCondition(state, inBounds, [&] { return readBuffer(context, instruction, resource, guest, 32u); }); };
    return BdaReadCacheAvailable(state) ? EmitBdaCachedDword(context, instruction, guest, inBounds, uncached) : uncached();
}

std::uint32_t atomicTargetType(SpirvEmitterState& state, const MemoryInfo& memory) {
    const auto u64 = TypeScalarU64(state);
    return memory.gpuDescriptor ? state.module.Type(spv::OpTypeStruct, u64, u64) : u64;
}

std::uint32_t noAtomicTarget(SpirvEmitterState& state, const MemoryInfo& memory) {
    const auto zero = BdaConstant(state, 0u);
    return memory.gpuDescriptor ? state.module.Constant(spv::OpConstantComposite, atomicTargetType(state, memory), zero, zero) : zero;
}

std::uint32_t atomicTarget(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t bits, std::uint32_t active, const BufferValues& operands) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto u64 = TypeScalarU64(state);
    const auto type = atomicTargetType(state, memory);
    const auto none = noAtomicTarget(state, memory);
    return EmitValueOrDefaultIfCondition(state, active, type, none, [&] {
        const auto resource = descriptor(context, instruction, operands);
        const auto access = address(context, instruction, resource, 0u, bits / 8u);
        return EmitValueOrDefaultIfCondition(state, access.inBounds, type, none, [&] {
            if (!memory.gpuDescriptor) {
                const auto storage = directAccess(context, instruction, resource, true, bits == 64u);
                const auto byte = EmitAddU32(state, access.guest, storage.byteOffset);
                const auto unaligned = nonzero(state, EmitAndConstant(state, byte, bits / 8u - 1u));
                const auto guest = Binary(state, spv::OpIAdd, u64, resource.base, Unary(state, spv::OpUConvert, u64, access.guest));
                EmitIfCondition(state, unaligned, [&] {
                    const auto high = Unary(state, spv::OpUConvert, TypeU32(state), Binary(state, spv::OpShiftRightLogical, u64, guest, BdaConstant(state, 32u)));
                    RecordBdaFaultWords(state, Unary(state, spv::OpUConvert, TypeU32(state), guest), high, ConstantU32(state, bits / 8u), BdaInstructionPc(state, instruction), BdaAbi::FaultReason::Unaligned);
                });
                const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte, ConstantU32(state, bits == 64u ? 3u : 2u));
                return EmitValueOrDefaultIfCondition(state, both(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), EmitMemoryElementInBounds(state, storage, index)), u64, none, [&] {
                    const auto pointer = EmitStorageBufferElementPointer(state, storage, index, bits == 64u ? TypePhysicalU64Pointer(state) : TypePhysicalU32Pointer(state));
                    return Unary(state, spv::OpConvertPtrToU, u64, pointer);
                });
            }
            const auto unaligned = Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, u64, access.guest, BdaConstant(state, bits / 8u - 1u)), BdaConstant(state, 0u));
            faultIf(context, instruction, unaligned, access.guest, bits / 8u, BdaAbi::FaultReason::Unaligned);
            const auto physical = EmitValueOrDefaultIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), u64, BdaConstant(state, 0u), [&] {
                const auto mapped = state.module.AllocateId();
                state.module.AddFunction(spv::OpFunctionCall, u64, mapped, state.bdaAtomicPointerFunction, access.guest, ConstantU32(state, bits / 8u), BdaInstructionPc(state, instruction));
                return mapped;
            });
            const auto target = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, type, target, physical, access.guest);
            return target;
        });
    });
}

struct HelperArgument {
    std::uint32_t type;
    std::uint32_t value;
};

template<typename TBody>
std::uint32_t callAccessHelper(SpirvEmitterState& state, const std::vector<std::uint32_t>& key, const std::string& name, std::uint32_t resultType, const std::vector<HelperArgument>& arguments, TBody&& body) {
    auto function = state.bufferAccessFunctions.find(key);
    if (function == state.bufferAccessFunctions.end()) {
        const auto id = state.module.AllocateId();
        std::vector<std::uint32_t> types{resultType};
        for (const auto& argument : arguments) types.push_back(argument.type);
        struct Restore {
            SpirvEmitterState& state;
            std::uint32_t label;
            std::uint32_t pcOverride;
            ~Restore() {
                state.module.EndHelperFunction();
                state.currentLabel = label;
                state.bdaPcOverride = pcOverride;
            }
        };
        state.module.BeginHelperFunction();
        const Restore restore{state, state.currentLabel, state.bdaPcOverride};
        state.module.AddName(id, name);
        state.module.AddFunction(spv::OpFunction, resultType, id, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, std::span<const std::uint32_t>(types)));
        std::vector<std::uint32_t> parameters;
        for (const auto& argument : arguments) {
            parameters.push_back(state.module.AllocateId());
            state.module.AddFunction(spv::OpFunctionParameter, argument.type, parameters.back());
        }
        EmitLabel(state, state.module.AllocateId());
        state.bdaPcOverride = 0u;
        const std::uint32_t value = body(std::span<const std::uint32_t>(parameters));
        if (value == 0u) state.module.AddFunction(spv::OpReturn);
        else state.module.AddFunction(spv::OpReturnValue, value);
        state.module.AddFunction(spv::OpFunctionEnd);
        function = state.bufferAccessFunctions.emplace(key, id).first;
    }
    std::vector<std::uint32_t> call{spv::OpFunctionCall, resultType, state.module.AllocateId(), function->second};
    for (const auto& argument : arguments) call.push_back(argument.value);
    state.module.AddFunction(call);
    return call[2];
}

enum class AccessKind : std::uint32_t { RawLoad = 1, RawStore, ScalarLoad, FormattedLoad, FormattedStore, Atomic };

struct AccessRequest {
    AccessKind kind = AccessKind::RawLoad;
    std::uint32_t components = 0;
    std::vector<std::uint32_t> extra;
};

std::uint32_t loadComponents(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::LoadBufferU8:
    case IrOpcode::LoadBufferU16:
    case IrOpcode::LoadBufferU32:
        return 1u;
    case IrOpcode::LoadBufferU32x2:
        return 2u;
    case IrOpcode::LoadBufferU32x3:
        return 3u;
    case IrOpcode::LoadBufferU32x4:
        return 4u;
    default:
        return 0u;
    }
}

std::uint32_t storeComponents(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::StoreBufferU8:
    case IrOpcode::StoreBufferU16:
    case IrOpcode::StoreBufferU32:
        return 1u;
    case IrOpcode::StoreBufferU32x2:
        return 2u;
    case IrOpcode::StoreBufferU32x3:
        return 3u;
    case IrOpcode::StoreBufferU32x4:
        return 4u;
    default:
        return 0u;
    }
}

bool fourWordHandle(const IrValue& instruction) {
    const auto* handle = instruction.ArgumentCount() == 0u ? nullptr : instruction.Argument(0)->Resolve();
    return handle != nullptr && handle->Opcode() == IrOpcode::GetBufferResource && handle->ArgumentCount() == 4u;
}

std::optional<AccessRequest> accessRequest(const IrValue& instruction, const MemoryInfo& memory) {
    const auto opcode = instruction.Opcode();
    const std::vector<std::uint32_t> format{memory.d16 ? 1u : 0u, memory.typed ? 1u : 0u, memory.typed ? memory.dataFormat : 0u, memory.typed ? memory.numberFormat : 0u};
    if (opcode == IrOpcode::ReadConstBuffer) {
        if (memory.planningOnly || memory.kind != ResourceKind::ScalarBuffer) return std::nullopt;
        return AccessRequest{AccessKind::ScalarLoad, 1u, {}};
    }
    if (const auto components = loadComponents(opcode); components != 0u) return AccessRequest{memory.formatted ? AccessKind::FormattedLoad : AccessKind::RawLoad, components, memory.formatted ? format : std::vector<std::uint32_t>{}};
    if (const auto components = storeComponents(opcode); components != 0u) return AccessRequest{memory.formatted ? AccessKind::FormattedStore : AccessKind::RawStore, components, memory.formatted ? format : std::vector<std::uint32_t>{}};
    if (BufferAccessOf(opcode) == BufferAccess::Atomic && memory.kind == ResourceKind::Buffer) return AccessRequest{AccessKind::Atomic, 1u, {instruction.Type() == IrType::U64 ? 64u : 32u}};
    return std::nullopt;
}

const IrValue* constantWord(const IrValue& instruction, std::uint32_t word) {
    const auto* value = instruction.Argument(0)->Resolve()->Argument(word)->Resolve();
    return value->HasImmediate() ? value : nullptr;
}

struct AccessShape {
    std::vector<std::uint32_t> key;
    std::uint32_t parameterWords = 0;
    bool laneParameter = false;
    bool constantDescriptor = false;
};

AccessShape accessShape(const SpirvEmitterState& state, const IrValue& instruction, const MemoryInfo& memory, const AccessRequest& request) {
    AccessShape shape;
    const auto mask = runtimeWordMask(state, instruction, memory);
    shape.parameterWords = mask;
    std::vector<std::uint32_t> constants;
    for (const std::uint32_t word : {1u, 3u}) {
        const auto* value = (mask & (1u << word)) != 0u ? constantWord(instruction, word) : nullptr;
        if (value == nullptr) continue;
        shape.parameterWords &= ~(1u << word);
        constants.insert(constants.end(), {word, value->ImmediateU32()});
    }
    shape.constantDescriptor = memory.gpuDescriptor && constantWord(instruction, 0u) != nullptr && constantWord(instruction, 2u) != nullptr && shape.parameterWords == 5u;
    shape.laneParameter = laneIndexed(state, memory) && (shape.parameterWords & 8u) != 0u;
    shape.key = {static_cast<std::uint32_t>(request.kind), memory.gpuDescriptor ? 0xffffffffu : memory.resource, request.components, memory.dataBits, memory.coherent ? 1u : 0u, static_cast<std::uint32_t>(memory.kind), mask, shape.parameterWords, shape.laneParameter ? 1u : 0u};
    shape.key.insert(shape.key.end(), constants.begin(), constants.end());
    shape.key.insert(shape.key.end(), request.extra.begin(), request.extra.end());
    return shape;
}

std::uint32_t accessEmissions(SpirvEmitterState& state, const std::vector<std::uint32_t>& key) {
    if (!state.bufferAccessUsesCounted) {
        state.bufferAccessUsesCounted = true;
        const auto& memoryInfo = state.program.Resources().memoryInfo;
        for (const auto* block : state.program.BlockOrder()) {
            for (const auto* instruction : block->Instructions()) {
                const auto opcode = instruction->Opcode();
                if (loadComponents(opcode) == 0u && storeComponents(opcode) == 0u && opcode != IrOpcode::ReadConstBuffer && BufferAccessOf(opcode) != BufferAccess::Atomic) continue;
                if (!fourWordHandle(*instruction)) continue;
                const auto& memory = memoryInfo.at(instruction->Flags<MemoryFlags>().index);
                const auto request = accessRequest(*instruction, memory);
                if (!request) continue;
                const auto shape = accessShape(state, *instruction, memory, *request);
                if (!shape.constantDescriptor) state.bufferAccessUses[shape.key] += state.laneCount;
            }
        }
    }
    const auto found = state.bufferAccessUses.find(key);
    return found == state.bufferAccessUses.end() ? 0u : found->second;
}

struct HelperValues {
    BufferValues values;
    std::size_t next = 0;
};

std::vector<HelperArgument> accessArguments(SpirvValueEmitContext& context, const IrValue& instruction, const AccessShape& shape, AccessKind kind, const BufferValues& operands) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto u32 = TypeU32(state);
    std::vector<HelperArgument> arguments;
    if (kind == AccessKind::ScalarLoad) {
        arguments = {{u32, operands.offset}, {u32, operands.immediate}};
    } else {
        const auto index = laneIndexed(state, memory) && !shape.laneParameter ? laneIndex(state, formatWord(state, memory, operands), operands.index, operands.lane) : operands.index;
        arguments = {{u32, index}, {u32, operands.offset}, {u32, operands.soffset}};
        if (shape.laneParameter) arguments.push_back({u32, operands.lane});
    }
    for (std::uint32_t word = 0; word < 4u; ++word) {
        if ((shape.parameterWords & (1u << word)) != 0u) arguments.push_back({u32, operands.words[word]});
    }
    if (memory.gpuDescriptor) {
        arguments.push_back({u32, BdaInstructionPc(state, instruction)});
    } else {
        arguments.insert(arguments.end(), {{TypeScalarU64(state), operands.address}, {u32, operands.byteOffset}, {u32, operands.dwords}});
    }
    return arguments;
}

HelperValues helperValues(SpirvEmitterState& state, const AccessShape& shape, AccessKind kind, const BufferValues& operands, bool gpuDescriptor, std::span<const std::uint32_t> parameters) {
    HelperValues helper;
    auto& values = helper.values;
    std::size_t next = 0;
    if (kind == AccessKind::ScalarLoad) {
        values.offset = parameters[next++];
        values.immediate = parameters[next++];
    } else {
        values.index = parameters[next++];
        values.offset = parameters[next++];
        values.soffset = parameters[next++];
        values.lane = shape.laneParameter ? parameters[next++] : ConstantU32(state, 0u);
    }
    for (std::uint32_t word = 0; word < 4u; ++word) {
        if ((shape.parameterWords & (1u << word)) != 0u) values.words[word] = parameters[next++];
        else values.words[word] = operands.words[word];
    }
    if (gpuDescriptor) {
        state.bdaPcOverride = parameters[next++];
    } else {
        values.address = parameters[next++];
        values.byteOffset = parameters[next++];
        values.dwords = parameters[next++];
    }
    helper.next = next;
    return helper;
}

std::string helperName(const AccessRequest& request, const MemoryInfo& memory) {
    static constexpr std::array<const char*, 6> kinds{"load_buffer", "store_buffer", "load_scalar_buffer", "load_formatted_buffer", "store_formatted_buffer", "buffer_atomic_address"};
    return std::string(kinds.at(static_cast<std::uint32_t>(request.kind) - 1u)) + "_x" + std::to_string(request.components) + "_b" + std::to_string(memory.dataBits) + (memory.gpuDescriptor ? std::string("_gpu") : "_r" + std::to_string(memory.resource)) + (memory.coherent ? "_coherent" : "");
}

template<typename TBody>
std::uint32_t emitAccess(SpirvValueEmitContext& context, const IrValue& instruction, const AccessRequest& request, std::uint32_t active, std::uint32_t valueType, std::uint32_t zero, const std::vector<HelperArgument>& extra, bool outOfLine, TBody&& body, bool inlineOnly = false, bool missPath = false) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    struct Faults {
        SpirvEmitterState& state;
        bool stops;
        ~Faults() { state.bdaStopsInvocations = stops; }
    };
    const Faults faults{state, state.bdaStopsInvocations};
    state.bdaStopsInvocations = false;
    struct Segment {
        SpirvEmitterState& state;
        std::array<std::uint32_t, 3> saved;
        ~Segment() { state.bdaAccessSegment = saved; }
    };
    const Segment segmentScope{state, state.bdaAccessSegment};
    std::array<std::uint32_t, 3> segment{};
    if (!missPath && (request.kind == AccessKind::RawLoad || request.kind == AccessKind::ScalarLoad || request.kind == AccessKind::FormattedLoad)) {
        if (const auto resolved = ResolvedBufferSegment(context, instruction)) segment = {resolved->begin, resolved->end, resolved->delta};
    }
    const auto operands = instructionValues(context, instruction, request.kind == AccessKind::ScalarLoad);
    auto shape = accessShape(state, instruction, memory, request);
    const bool segmentParameters = segment[0] != 0u && (request.kind == AccessKind::RawLoad || request.kind == AccessKind::FormattedLoad);
    std::vector<std::uint32_t> values;
    for (const auto& argument : extra) values.push_back(argument.value);
    if (inlineOnly || shape.constantDescriptor || (!outOfLine && accessEmissions(state, shape.key) < 2u)) {
        state.bdaAccessSegment = segment;
        return body(operands, active, std::span<const std::uint32_t>(values));
    }
    if (segmentParameters) shape.key.push_back(0x5e9u);
    const bool gated = request.kind != AccessKind::ScalarLoad;
    const auto call = [&] {
        auto arguments = accessArguments(context, instruction, shape, request.kind, operands);
        if (gated) arguments.push_back({TypeBool(state), active});
        if (segmentParameters) {
            for (const auto value : segment) arguments.push_back({TypeScalarU64(state), value});
        }
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        state.bdaAccessSegment = {};
        const auto result = callAccessHelper(state, shape.key, helperName(request, memory), valueType == 0u ? TypeVoid(state) : valueType, arguments, [&](std::span<const std::uint32_t> parameters) {
            const auto helper = helperValues(state, shape, request.kind, operands, memory.gpuDescriptor, parameters);
            const auto enabled = gated ? parameters[helper.next] : ConstantBool(state, true);
            auto next = helper.next + (gated ? 1u : 0u);
            if (segmentParameters) {
                state.bdaAccessSegment = {parameters[next], parameters[next + 1u], parameters[next + 2u]};
                next += 3u;
            }
            const auto value = body(helper.values, enabled, parameters.subspan(next));
            state.bdaAccessSegment = {};
            return value;
        });
        return valueType == 0u ? 0u : result;
    };
    if (!missPath && request.kind == AccessKind::ScalarLoad && memory.gpuDescriptor && BdaReadCacheAvailable(state)) {
        const auto access = scalarAccess(context, instruction, operands);
        state.bdaAccessSegment = segment;
        return EmitBdaCachedDword(context, instruction, scalarGuestAddress(state, access), access.inBounds, call, false);
    }
    return call();
}

}

std::uint32_t EmitRuntimeBufferLoad(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t components) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto type = components == 1u ? TypeU32(state) : TypeU32Composite(state, components);
    const auto zero = components == 1u ? ConstantU32(state, 0u) : ConstantU32CompositeZero(state, components);
    const auto active = context.Arg(instruction, instruction.ArgumentCount() - 1u);
    const AccessRequest request = *accessRequest(instruction, memory);
    if (request.components != components) context.Fail(instruction, "buffer load component count does not match its opcode");
    return emitAccess(context, instruction, request, active, type, zero, {}, runtimeFormatted(instruction, memory), [&](const BufferValues& values, std::uint32_t enabled, std::span<const std::uint32_t>) {
        return memory.formatted ? formattedAccess(context, instruction, components, false, enabled, values, 0u) : rawLoad(context, instruction, components, enabled, values);
    });
}

void EmitRuntimeBufferStore(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t components) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    const auto active = context.Arg(instruction, instruction.ArgumentCount() - 1u);
    const AccessRequest request = *accessRequest(instruction, memory);
    if (request.components != components) context.Fail(instruction, "buffer store component count does not match its opcode");
    const auto dataType = TypeId(state, instruction.Argument(instruction.ArgumentCount() - 2u)->Type());
    if (dataType == 0u) context.Fail(instruction, "buffer store data has no SPIR-V type");
    static_cast<void>(emitAccess(context, instruction, request, active, 0u, 0u, {{dataType, storeData(context, instruction)}}, runtimeFormatted(instruction, memory), [&](const BufferValues& values, std::uint32_t enabled, std::span<const std::uint32_t> data) {
        if (memory.formatted) return formattedAccess(context, instruction, components, true, enabled, values, data[0]);
        rawStore(context, instruction, components, enabled, values, data[0]);
        return 0u;
    }, false));
}

void DefineFormatCodecs(SpirvEmitterState& state) {
    const auto& layouts = sharedFormatLayouts();
    const auto u32 = TypeU32(state);
    const auto define = [&](std::uint32_t function, const std::string& name, std::uint32_t parameters, std::uint32_t results, const auto& body) {
        const auto type = codecType(state, results);
        std::vector<std::uint32_t> signature(parameters + 2u, u32);
        signature[0] = type;
        state.module.AddName(function, name);
        state.module.AddFunction(spv::OpFunction, type, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, std::span<const std::uint32_t>(signature)));
        std::vector<std::uint32_t> ids(parameters + 1u);
        for (auto& id : ids) {
            id = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, u32, id);
        }
        EmitLabel(state, state.module.AllocateId());
        const std::vector<std::uint32_t> values = body(ids[0], std::span<const std::uint32_t>(ids).subspan(1u));
        std::vector<std::uint32_t> construct{spv::OpCompositeConstruct, type, state.module.AllocateId()};
        construct.insert(construct.end(), values.begin(), values.end());
        state.module.AddFunction(construct);
        state.module.AddFunction(spv::OpReturnValue, construct[2]);
        state.module.AddFunction(spv::OpFunctionEnd);
    };
    for (const auto& [key, function] : state.formatDecoders) {
        const auto layout = key >> 4u;
        MemoryInfo memory{};
        memory.d16 = (key & 8u) != 0u;
        const auto info = GetFormatInfo(static_cast<IrBufferFormat>(layouts[layout].front()));
        define(function, "decode_format_layout_" + std::to_string(layout) + (memory.d16 ? "_d16" : ""), info.componentCount, info.componentCount + 1u, [&](std::uint32_t format, std::span<const std::uint32_t> raw) {
            return switchValues(state, format, formatMembers(layout), true, std::vector<std::uint32_t>(info.componentCount + 1u, u32), [&](std::size_t member) {
                const auto memberInfo = GetFormatInfo(static_cast<IrBufferFormat>(layouts[layout][member]));
                std::vector<std::uint32_t> parts;
                for (std::uint32_t component = 0; component < info.componentCount; ++component) parts.push_back(decodeFormatted(state, memory, memberInfo, component, raw[component]));
                parts.push_back(formattedOne(state, memory, memberInfo));
                return parts;
            });
        });
    }
    for (const auto& [key, function] : state.formatEncoders) {
        const auto layout = key >> 4u;
        const auto components = key & 7u;
        MemoryInfo memory{};
        memory.d16 = (key & 8u) != 0u;
        const auto info = GetFormatInfo(static_cast<IrBufferFormat>(layouts[layout].front()));
        define(function, "encode_format_layout_" + std::to_string(layout) + "_x" + std::to_string(components) + (memory.d16 ? "_d16" : ""), components, info.componentCount, [&](std::uint32_t format, std::span<const std::uint32_t> data) {
            std::array<std::uint32_t, 4> values{};
            std::copy(data.begin(), data.end(), values.begin());
            return switchValues(state, format, formatMembers(layout), true, std::vector<std::uint32_t>(info.componentCount, u32), [&](std::size_t member) {
                const auto memberInfo = GetFormatInfo(static_cast<IrBufferFormat>(layouts[layout][member]));
                std::vector<std::uint32_t> parts;
                for (std::uint32_t component = 0; component < info.componentCount; ++component) parts.push_back(encodeFormatted(state, memory, memberInfo, component, components, values));
                return parts;
            });
        });
    }
}

namespace {

bool groupableScalarBufferLoad(SpirvValueEmitContext& context, const IrValue& instruction) {
    if (instruction.Opcode() != IrOpcode::ReadConstBuffer || instruction.ArgumentCount() < 2u || !fourWordHandle(instruction)) return false;
    const auto& memory = context.Memory(instruction);
    return memory.kind == ResourceKind::ScalarBuffer && memory.gpuDescriptor && !memory.planningOnly;
}

void scanScalarBufferGroups(SpirvValueEmitContext& context, const IrBlock& block) {
    auto& groups = context.state.scalarBufferGroups;
    if (!groups.scanned.insert(&block).second) return;
    std::vector<const IrValue*> run;
    const auto flush = [&] {
        if (run.size() > 1u) groups.leaders[run.front()] = run;
        run.clear();
    };
    const auto same = [](const IrValue* left, const IrValue* right) {
        left = left->Resolve();
        right = right->Resolve();
        return left == right || (left->HasImmediate() && right->HasImmediate() && left->Type() == right->Type() && left->ImmediateU32() == right->ImmediateU32());
    };
    const auto sameHandle = [&](const IrValue& left, const IrValue& right) {
        const auto* a = left.Argument(0)->Resolve();
        const auto* b = right.Argument(0)->Resolve();
        if (a == b) return true;
        for (std::size_t word = 0; word < 4u; ++word) {
            if (!same(a->Argument(word), b->Argument(word))) return false;
        }
        return true;
    };
    const auto joins = [&](const IrValue& instruction) {
        const auto& first = *run.front();
        return sameHandle(instruction, first) && same(instruction.Argument(1), first.Argument(1)) && context.Memory(instruction).offset == context.Memory(first).offset + static_cast<std::uint32_t>(run.size()) * 4u;
    };
    for (const IrValue* instruction : block.Instructions()) {
        if (instruction->Opcode() == IrOpcode::ReferenceU32 || instruction->Opcode() == IrOpcode::GetBufferResource) continue;
        if (!groupableScalarBufferLoad(context, *instruction)) {
            flush();
            continue;
        }
        if (!run.empty() && (run.size() == 16u || !joins(*instruction))) flush();
        run.push_back(instruction);
    }
    flush();
}

std::uint32_t scalarBufferLoad(SpirvValueEmitContext& context, const IrValue& instruction, bool missPath) {
    auto& state = context.state;
    return emitAccess(context, instruction, AccessRequest{AccessKind::ScalarLoad, 1u, {}}, 0u, TypeU32(state), ConstantU32(state, 0u), {}, false, [&](const BufferValues& values, std::uint32_t, std::span<const std::uint32_t>) {
        return scalarLoad(context, instruction, values);
    }, false, missPath);
}

}

std::uint32_t EmitRuntimeScalarBufferLoad(SpirvValueEmitContext& context, const IrValue& instruction) {
    auto& state = context.state;
    if (std::ranges::find(state.supportedCapabilities, spv::CapabilityInt64) == state.supportedCapabilities.end()) context.Fail(instruction, "scalar buffer bounds require shaderInt64");
    state.module.EmitCapability(spv::CapabilityInt64);
    auto& groups = state.scalarBufferGroups;
    if (const auto defined = groups.defined.find(&instruction); defined != groups.defined.end()) {
        const auto value = defined->second;
        groups.defined.erase(defined);
        return value;
    }
    if (groupableScalarBufferLoad(context, instruction) && BdaReadCacheAvailable(state) && instruction.Parent() != nullptr) {
        scanScalarBufferGroups(context, *instruction.Parent());
        if (const auto leader = groups.leaders.find(&instruction); leader != groups.leaders.end()) {
            const auto members = std::move(leader->second);
            groups.leaders.erase(leader);
            const auto count = static_cast<std::uint32_t>(members.size());
            const auto u32 = TypeU32(state);
            const auto type = state.module.Type(spv::OpTypeArray, u32, ConstantU32(state, count));
            const auto span = scalarAccess(context, instruction, instructionValues(context, instruction, true), count * 4u);
            const auto values = EmitBdaCachedDwords(context, instruction, scalarGuestAddress(state, span), span.inBounds, count, type, [&] {
                std::vector<std::uint32_t> construct{spv::OpCompositeConstruct, type, state.module.AllocateId()};
                for (const auto* member : members) construct.push_back(scalarBufferLoad(context, *member, true));
                state.module.AddFunction(construct);
                return construct[2];
            });
            std::uint32_t result = 0;
            for (std::uint32_t index = 0; index < count; ++index) {
                const auto value = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeExtract, u32, value, values, index);
                if (index == 0u) result = value;
                else groups.defined[members[index]] = value;
            }
            return result;
        }
    }
    return scalarBufferLoad(context, instruction, false);
}

std::uint32_t EmitRuntimeBufferAtomic(SpirvValueEmitContext& context, const IrValue& instruction, std::uint32_t bits, std::uint32_t active, const std::function<std::uint32_t(std::uint32_t)>& operation) {
    auto& state = context.state;
    const auto& memory = context.Memory(instruction);
    if (bits != 32u && bits != 64u) context.Fail(instruction, "unsupported runtime buffer atomic width");
    if (memory.gpuDescriptor && (state.bdaAtomicPointerFunction == 0u || state.bdaNoteWriteFunction == 0u)) context.Fail(instruction, "BDA atomic functions are missing");
    if (!memory.gpuDescriptor) {
        if (std::ranges::find(state.supportedCapabilities, spv::CapabilityInt64) == state.supportedCapabilities.end()) context.Fail(instruction, "buffer atomic fault addresses require shaderInt64");
        state.module.EmitCapability(spv::CapabilityInt64);
    }
    const auto type = bits == 64u ? TypeU64(state) : TypeU32(state);
    const auto scalar = bits == 64u ? TypeScalarU64(state) : TypeU32(state);
    const auto zero = bits == 64u ? ConstantU64(state, 0u) : ConstantU32(state, 0u);
    const auto u64 = TypeScalarU64(state);
    const auto target = emitAccess(context, instruction, AccessRequest{AccessKind::Atomic, 1u, {bits}}, active, atomicTargetType(state, memory), noAtomicTarget(state, memory), {}, false, [&](const BufferValues& values, std::uint32_t enabled, std::span<const std::uint32_t>) {
        return atomicTarget(context, instruction, bits, enabled, values);
    });
    auto physical = target;
    auto guest = 0u;
    if (memory.gpuDescriptor) {
        physical = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, physical, target, 0u);
        guest = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, guest, target, 1u);
    }
    return EmitValueOrDefaultIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), type, zero, [&] {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, scalar), pointer, physical);
        const auto result = operation(pointer);
        if (memory.gpuDescriptor) state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, guest);
        return result;
    });
}

}
