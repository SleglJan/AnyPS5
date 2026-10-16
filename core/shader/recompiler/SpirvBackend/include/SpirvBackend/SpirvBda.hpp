#ifndef CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP
#define CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP

#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvEmitter.hpp"
#include "BdaAbi.hpp"
#include <optional>
#include <vector>
#include <array>
#include <functional>
#include <span>

namespace ShaderRecompiler {

std::uint32_t BdaConstant(SpirvEmitterState& state, std::uint64_t value);
std::uint32_t BdaWord(SpirvEmitterState& state, std::uint32_t variable, std::uint32_t index);
std::uint32_t BdaLoadWord(SpirvEmitterState& state, std::uint32_t index);
std::uint32_t BdaLoadAddress(SpirvEmitterState& state, std::uint32_t index);
void DefineBdaFaultFunction(SpirvEmitterState& state);
void DeclareBdaReadCache(SpirvEmitterState& state);
void InitializeBdaTable(SpirvEmitterState& state);
bool BdaReadCacheAvailable(const SpirvEmitterState& state);
void DeclareBdaVectorCache(SpirvEmitterState& state);
bool BdaVectorCacheAvailable(const SpirvEmitterState& state);
struct BdaCacheProbe {
    std::uint32_t hit;
    std::uint32_t device;
};
BdaCacheProbe ProbeBdaVectorCache(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes);
void RefreshBdaVectorCache(SpirvEmitterState& state, std::uint32_t address);
std::uint32_t DefineWrittenChunkNote(SpirvEmitterState& state);
void DefineBdaDwordReadFunctions(SpirvEmitterState& state);
void DefineBdaSpanReadFunctions(SpirvEmitterState& state);
void DefineBdaComponentReadFunctions(SpirvEmitterState& state);
bool BdaComponentReadsAvailable(const SpirvEmitterState& state, const IrValue& inst);
std::array<std::uint32_t, 4> EmitBdaComponentReads(SpirvValueEmitContext& ctx, const IrValue& inst, const std::array<std::uint32_t, 4>& addresses, const std::array<std::uint32_t, 4>& sizes);
void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void RecordBdaFaultReason(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t reason);
void RecordBdaFaultWords(SpirvEmitterState& state, std::uint32_t addressLow, std::uint32_t addressHigh, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
std::uint32_t BdaFaultStage(SpirvEmitterState& state);
void RecordBdaDescriptorFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t resource, std::uint32_t reserved);
void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target);
// Whether a faulting BDA access may end its invocation. Programs with workgroup barriers must keep
// every invocation running, so their faulting reads return zero instead.
bool BdaInvocationsMayStop(const IrProgram& program);
void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition);
std::uint32_t BdaInstructionPc(SpirvEmitterState& state, const IrValue& inst);
std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits);
std::uint32_t EmitBdaCachedDword(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t enabled, const std::function<std::uint32_t()>& uncached, bool refill = true);
std::uint32_t EmitBdaCachedDwords(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t enabled, std::uint32_t dwords, std::uint32_t type, const std::function<std::uint32_t()>& miss);
void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits, std::uint32_t enabled);
void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits);
void DefineBdaByteWriteFunctions(SpirvEmitterState& state);
void DefineBdaWriteFunctions(SpirvEmitterState& state);
void EmitBdaDwordWrites(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, std::uint32_t value);
std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation);
// Reads the dwords of a 1-4 dword load at address + offset that the program extracts, with one
// table lookup for the whole span; the per-byte lookups of EmitBdaRead remain the fallback (and
// the only path under APS5_BDA_BYTE_READS=1). Dwords the program never extracts are neither read
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword = false, std::uint32_t usedDwords = 0u);
bool BdaByteReadsForced();
std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract);
std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate);
bool WaveUniformDescriptor(SpirvEmitterState& state, const std::vector<const IrValue*>& words);
std::uint32_t BroadcastFirstLane(SpirvEmitterState& state, std::uint32_t type, std::uint32_t value);
std::vector<const IrValue*> InvalidFormatInputs(const SpirvEmitterState& state, const IrValue& inst);
void EmitInvalidFormatStop(SpirvValueEmitContext& ctx, const IrValue& inst, const IrValue& anchor);
struct BdaSegmentValues {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    std::uint32_t delta = 0;
};
BdaCacheProbe ProbeBdaSegment(SpirvEmitterState& state, const BdaSegmentValues& segment, std::uint32_t address, std::uint32_t bytes);
std::vector<const IrValue*> ResolvedBufferInputs(const SpirvEmitterState& state, const IrValue& inst);
void EmitResolvedBufferSegment(SpirvValueEmitContext& ctx, const IrValue& inst, const IrValue& anchor);
std::optional<BdaSegmentValues> ResolvedBufferSegment(SpirvValueEmitContext& ctx, const IrValue& inst);

}

#endif
