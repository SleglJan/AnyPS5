#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_FLOATSELECTRETYPER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_FLOATSELECTRETYPER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <cstdint>

namespace ShaderRecompiler {

struct FloatSelectRetypeStats {
    std::uint32_t retypedValues = 0;

    [[nodiscard]] bool Changed() const { return retypedValues != 0u; }
};

class FloatSelectRetyper {
public:
    [[nodiscard]] FloatSelectRetypeStats Retype(IrProgram& program) const;
};

}

#endif
