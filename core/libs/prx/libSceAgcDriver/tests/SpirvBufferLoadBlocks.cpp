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
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint32_t LoadDword = 0xe0301000u;
constexpr std::uint32_t LoadDwordX2 = 0xe0341000u;
constexpr std::uint32_t LoadDwordX3 = 0xe03c1000u;
constexpr std::uint32_t LoadDwordX4 = 0xe0381000u;
constexpr std::uint32_t StoreDword = 0xe0701000u;
constexpr std::uint32_t StoreDwordX2 = 0xe0741000u;
constexpr std::uint32_t StoreDwordX3 = 0xe07c1000u;
constexpr std::uint32_t StoreDwordX4 = 0xe0781000u;
constexpr std::uint32_t SBufferLoadDword = 0xf4200200u;
constexpr std::uint32_t AlignedBase = 0x10000000u;
constexpr std::uint32_t UnalignedBase = 0x10000001u;

struct Added {
    std::size_t blocks;
    std::size_t guards;
};

std::vector<std::uint32_t> Recompiled(std::span<const std::uint32_t> code, std::uint32_t base) {
    const std::array<std::uint32_t, 8> userData{base, 0x00000000u, 0x100u, 0x31016facu, 0x20000000u, 0x00000000u, 0x100u, 0x31016facu};
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

Added Count(std::span<const std::uint32_t> words) {
    Added added{0, 0};
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        if (count == 0 || cursor + count > words.size()) throw std::runtime_error("truncated SPIR-V instruction");
        if ((words[cursor] & 0xffffu) == spv::OpLabel) ++added.blocks;
        if ((words[cursor] & 0xffffu) == spv::OpBranchConditional) ++added.guards;
        cursor += count;
    }
    return added;
}

Added Delta(const std::vector<std::uint32_t>& loaded, const std::vector<std::uint32_t>& constant, std::uint32_t base) {
    const auto with = Count(Recompiled(loaded, base));
    const auto without = Count(Recompiled(constant, base));
    if (with.blocks < without.blocks || with.guards < without.guards) throw std::runtime_error("the load removed blocks");
    return {with.blocks - without.blocks, with.guards - without.guards};
}

Added LoadAdded(std::uint32_t load, std::uint32_t store, std::uint32_t components, std::uint32_t base) {
    std::vector<std::uint32_t> loaded{0x34020082u, load, 0x80000201u, 0xbf8c3f70u};
    std::vector<std::uint32_t> constant{0x34020082u};
    for (std::uint32_t component = 0; component < components; ++component) constant.push_back(0x7e000280u | ((2u + component) << 17u));
    for (auto* program : {&loaded, &constant}) program->insert(program->end(), {store, 0x80010201u, 0xbf810000u});
    return Delta(loaded, constant, base);
}

Added ScalarLoadAdded() {
    const std::vector<std::uint32_t> loaded{0x34020082u, SBufferLoadDword, 0xfa000000u, 0xbf8cc07fu, 0x7e040208u, StoreDword, 0x80010201u, 0xbf810000u};
    const std::vector<std::uint32_t> constant{0x34020082u, 0x7e040280u, StoreDword, 0x80010201u, 0xbf810000u};
    return Delta(loaded, constant, AlignedBase);
}

Added RuntimeLoadBudget(std::uint32_t dwords) {
    return {4u + 5u * dwords, 2u + 2u * dwords};
}

void Expect(const char* name, const Added& added, const Added& limit) {
    if (added.blocks > limit.blocks) throw std::runtime_error(std::string(name) + " adds " + std::to_string(added.blocks) + " blocks, expected at most " + std::to_string(limit.blocks));
    if (added.guards > limit.guards) throw std::runtime_error(std::string(name) + " adds " + std::to_string(added.guards) + " conditional branches, expected at most " + std::to_string(limit.guards));
}

}

int main() {
    try {
        Expect("buffer_load_dword", LoadAdded(LoadDword, StoreDword, 1u, AlignedBase), RuntimeLoadBudget(1u));
        Expect("buffer_load_dwordx2", LoadAdded(LoadDwordX2, StoreDwordX2, 2u, AlignedBase), RuntimeLoadBudget(2u));
        Expect("buffer_load_dwordx3", LoadAdded(LoadDwordX3, StoreDwordX3, 3u, AlignedBase), RuntimeLoadBudget(3u));
        Expect("buffer_load_dwordx4", LoadAdded(LoadDwordX4, StoreDwordX4, 4u, AlignedBase), RuntimeLoadBudget(4u));
        Expect("buffer_load_dwordx3 with an unaligned base", LoadAdded(LoadDwordX3, StoreDwordX3, 3u, UnalignedBase), RuntimeLoadBudget(3u));
        Expect("s_buffer_load_dword", ScalarLoadAdded(), {12u, 5u});
    } catch (const std::exception& error) {
        std::cerr << "SpirvBufferLoadBlocks: " << error.what() << '\n';
        return 1;
    }
    std::puts("SpirvBufferLoadBlocks: buffer loads stay within their block budget");
    return 0;
}
