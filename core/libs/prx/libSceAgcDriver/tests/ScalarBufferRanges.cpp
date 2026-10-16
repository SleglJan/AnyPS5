#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint64_t InputBase = 0x10000000u;
constexpr std::uint32_t InputBytes = 0x100u;

bool ReadsInput(std::uint32_t word3) {
    const std::array<std::uint32_t, 8> userData{static_cast<std::uint32_t>(InputBase), 0u, InputBytes, word3, 0x20000000u, 0u, 0x100u, 0x31016facu};
    static constexpr std::array<std::uint32_t, 8> code{0x34020082u, 0xf4200200u, 0xfa000004u, 0xbf8cc07fu, 0x7e040208u, 0xe0701000u, 0x80010201u, 0xbf810000u};
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    static constexpr std::array<std::uint32_t, 8> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityGroupNonUniform,
        spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformArithmetic};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x20000u, code, 0, {}};
    request.context.waveSize = 32;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    const auto result = Recompile(request);
    if (result.unboundedReads) return true;
    for (const auto& [begin, end] : result.readRanges) {
        if (begin <= InputBase + 4u && InputBase + 8u <= end) return true;
    }
    return false;
}

}

int main() {
    try {
        for (const std::uint32_t word3 : {0u, 0x31016facu, 0x91b00f2eu, 0xffffffffu}) {
            char text[16];
            std::snprintf(text, sizeof(text), "0x%08x", word3);
            if (!ReadsInput(word3)) throw std::runtime_error(std::string("an s_buffer_load_dword through a V# whose word 3 is ") + text + " has no read range over the dword it reads");
        }
    } catch (const std::exception& error) {
        std::cerr << "ScalarBufferRanges: " << error.what() << '\n';
        return 1;
    }
    std::puts("ScalarBufferRanges: scalar buffer reads are bounded whatever the V#'s unused fields hold");
    return 0;
}
