#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_FLOATIDIOMS_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_FLOATIDIOMS_HPP

#include "IntermediateRepresentation/IrValue.hpp"
#include <cstdint>
#include <optional>

namespace ShaderRecompiler {

[[nodiscard]] IrValue* MaskedOperand(const IrValue& value, IrOpcode opcode, std::uint32_t bits);
[[nodiscard]] IrValue* FlushedF32Source(const IrValue& value);
[[nodiscard]] IrValue* NanTestedF32Source(const IrValue& value);
[[nodiscard]] IrValue* QuietedF32Source(const IrValue& value);
[[nodiscard]] IrValue* MagnitudeSource(const IrValue& value);
[[nodiscard]] std::optional<std::uint32_t> ExactF32Product(std::uint32_t lhs, std::uint32_t rhs);

}

#endif
