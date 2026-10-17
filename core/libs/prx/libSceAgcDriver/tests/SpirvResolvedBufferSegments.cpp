#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

std::vector<std::uint32_t> Recompiled(std::span<const std::uint32_t> code, std::uint32_t subgroupSize = 32u) {
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
    request.target.subgroupSize = subgroupSize;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    auto result = Recompile(request);
    if (result.spirv.empty()) throw std::runtime_error("the shader did not recompile");
    return result.spirv.Copy();
}

struct Translations {
    std::uint32_t total = 0;
    std::uint32_t elected = 0;
};

Translations MainTranslations(std::span<const std::uint32_t> words, std::string_view callee) {
    std::map<std::uint32_t, std::string> names;
    std::uint32_t entry = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
        if (op == spv::OpEntryPoint) entry = words[cursor + 2];
    }
    std::set<std::uint32_t> elected;
    Translations translations;
    std::uint32_t function = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        const auto count = words[cursor] >> 16u;
        if (op == spv::OpFunction) function = words[cursor + 2];
        if (function != entry) continue;
        if (op == spv::OpGroupNonUniformElect) elected.insert(words[cursor + 2]);
        if (op != spv::OpFunctionCall || names[words[cursor + 3]] != callee) continue;
        ++translations.total;
        if (std::any_of(words.begin() + static_cast<std::ptrdiff_t>(cursor) + 4, words.begin() + static_cast<std::ptrdiff_t>(cursor + count), [&](std::uint32_t argument) { return elected.contains(argument); })) ++translations.elected;
    }
    return translations;
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
        if (op == spv::OpFunctionCall || op == spv::OpCompositeConstruct || op == spv::OpLoad || op == spv::OpPhi) types[words[cursor + 2]] = words[cursor + 1];
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
        const auto translations = MainTranslations(module, "bda_segment");
        const auto calls = translations.total;
        Expect(calls == 2u, "main translates " + std::to_string(calls) + " V# bases for three buffer_load_dword through two V#s, expected one translation of each V# where it is loaded");
        const auto reads = SegmentFieldReads(module);
        Expect(reads == 3u * calls, "main reads " + std::to_string(reads) + " fields of segment records for " + std::to_string(calls) + " segments, expected the three fields of each once");
        const auto elected = translations.elected;
        Expect(elected == calls, "one active lane translates " + std::to_string(elected) + " of " + std::to_string(calls) + " wave-uniform V# bases, expected every lookup of a wave-uniform V# in one elected lane and broadcast");
        const auto perLane = Recompiled(twoBuffers, 64u);
        const auto perLaneTranslations = MainTranslations(perLane, "bda_segment");
        const auto perLaneCalls = perLaneTranslations.total;
        Expect(perLaneCalls == 2u, "a host subgroup of two waves translates " + std::to_string(perLaneCalls) + " V# bases, expected one translation of each V#");
        const auto perLaneElected = perLaneTranslations.elected;
        Expect(perLaneElected == 0u, "a host subgroup of two waves translates " + std::to_string(perLaneElected) + " V# bases in an elected lane, expected every lane to translate its own wave's V#");
    } catch (const std::exception& error) {
        std::cerr << "SpirvResolvedBufferSegments: " << error.what() << '\n';
        return 1;
    }
    std::puts("SpirvResolvedBufferSegments: each run-time V# is translated once where it is loaded, its fields are read once, and a wave-uniform V# is translated by one elected lane");
    return 0;
}
