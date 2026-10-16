#ifndef CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVCONTINUATION_HPP
#define CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVCONTINUATION_HPP

#include "SpirvBackend/SpirvEmitterState.hpp"
#include <cstdint>

namespace ShaderRecompiler {

void EmitContinuationSpill(SpirvValueEmitContext& ctx, const IrValue& inst);
void EmitContinuationSpillLds(SpirvValueEmitContext& ctx, const IrValue& inst);
void EmitTableCallBin(SpirvValueEmitContext& ctx, const IrValue& inst);
std::uint32_t EmitContinuationSlot(SpirvValueEmitContext& ctx, const IrValue& inst);
std::uint32_t EmitContinuationLoad(SpirvValueEmitContext& ctx, const IrValue& inst);
void EmitContinuationRestoreLds(SpirvValueEmitContext& ctx, const IrValue& inst);

}

#endif
