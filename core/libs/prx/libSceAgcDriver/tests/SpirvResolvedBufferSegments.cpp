#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

std::vector<std::uint32_t> Recompiled(std::span<const std::uint32_t> code) {
    const std::array<std::uint32_t, 12> userData{0x10000000u, 0x00000000u, 0x100u, 0x31016facu, 0x20000000u, 0x00000000u, 0x100u, 0x31016facu, 0x30000000u, 0x00000000u, 0x100u, 0x31016facu};
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
    auto result = Recompile(request);
    if (result.spirv.empty()) throw std::runtime_error("the shader did not recompile");
    return result.spirv.Copy();
}

std::uint32_t MainCalls(std::span<const std::uint32_t> words, std::string_view callee) {
    std::map<std::uint32_t, std::string> names;
    std::uint32_t entry = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
        if (op == spv::OpEntryPoint) entry = words[cursor + 2];
    }
    std::uint32_t calls = 0;
    std::uint32_t function = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) function = words[cursor + 2];
        if (op == spv::OpFunctionCall && function == entry && names[words[cursor + 3]] == callee) ++calls;
    }
    return calls;
}

std::uint32_t SegmentFieldReads(std::span<const std::uint32_t> words) {
    std::uint32_t entry = 0;
    std::uint32_t u64 = 0;
    std::uint32_t segmentType = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        const auto count = words[cursor] >> 16u;
        if (op == spv::OpEntryPoint) entry = words[cursor + 2];
        if (op == spv::OpTypeInt && words[cursor + 2] == 64u && words[cursor + 3] == 0u) u64 = words[cursor + 1];
        if (op == spv::OpTypeStruct && count == 5u && words[cursor + 2] == u64 && words[cursor + 3] == u64 && words[cursor + 4] == u64) segmentType = words[cursor + 1];
    }
    std::map<std::uint32_t, std::uint32_t> types;
    std::uint32_t reads = 0;
    std::uint32_t function = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) function = words[cursor + 2];
        if (function != entry) continue;
        if (op == spv::OpFunctionCall || op == spv::OpCompositeConstruct || op == spv::OpLoad) types[words[cursor + 2]] = words[cursor + 1];
        if (op == spv::OpCompositeExtract && segmentType != 0u && types[words[cursor + 3]] == segmentType) ++reads;
    }
    return reads;
}

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}

int main() {
    try {
        const std::vector<std::uint32_t> twoBuffers{
            0x34020082u,
            0xe0301000u, 0x80000201u,
            0xe0301000u, 0x80020301u,
            0xe0301004u, 0x80000401u,
            0xbf8c3f70u,
            0xe0701000u, 0x80010201u,
            0xe0701004u, 0x80010301u,
            0xe0701008u, 0x80010401u,
            0xbf810000u};
        const auto module = Recompiled(twoBuffers);
        const auto calls = MainCalls(module, "bda_segment");
        Expect(calls == 2u, "main translates " + std::to_string(calls) + " V# bases for three buffer_load_dword through two V#s, expected one translation of each V# where it is loaded");
        const auto reads = SegmentFieldReads(module);
        Expect(reads == 3u * calls, "main reads " + std::to_string(reads) + " fields of segment records for " + std::to_string(calls) + " segments, expected the three fields of each once");
    } catch (const std::exception& error) {
        std::cerr << "SpirvResolvedBufferSegments: " << error.what() << '\n';
        return 1;
    }
    std::puts("SpirvResolvedBufferSegments: each run-time V# is translated once where it is loaded and its fields are read once");
    return 0;
}
