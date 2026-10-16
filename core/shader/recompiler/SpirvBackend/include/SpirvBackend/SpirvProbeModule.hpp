#ifndef CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVPROBEMODULE_HPP
#define CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVPROBEMODULE_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ShaderRecompiler {

std::optional<std::vector<std::uint32_t>> DeviceImageProbeModule(std::span<const std::uint32_t> words);

}

#endif
