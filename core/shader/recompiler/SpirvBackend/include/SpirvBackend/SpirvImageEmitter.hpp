#ifndef CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVIMAGEEMITTER_HPP
#define CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVIMAGEEMITTER_HPP

#include "IntermediateRepresentation/IrValue.hpp"
#include "SpirvBackend/SpirvModule.hpp"
#include <cstdint>

#include "SpirvBackend/SpirvEmitterState.hpp"
#include <span>

namespace ShaderRecompiler {

void EmitImageOperation(SpirvModule& module, const IrValue& value, std::uint32_t resultId);

void EmitImageOperation(SpirvValueEmitContext& context, const IrValue& value, std::uint32_t resultId);

void DefineDeviceImageLookup(SpirvEmitterState& state);

void PlanDeviceImageAnchors(SpirvEmitterState& state);

void EmitAnchoredDeviceImageLookups(SpirvValueEmitContext& ctx, const IrValue& anchor);

bool ImageStoresRecorded(const IrProgram& program, std::span<const std::uint32_t> capabilities);

void DefineImageStoreRecord(SpirvEmitterState& state);

}

#endif
