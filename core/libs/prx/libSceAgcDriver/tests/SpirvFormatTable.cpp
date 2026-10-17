#include "Recompiler.hpp"
#include "RuntimeAbi.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "SpirvBackend/SpirvModule.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
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

RecompileResult Recompiled(std::span<const std::uint32_t> code) {
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
    return result;
}

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::size_t FormatTables(const RecompileResult& result) {
    return static_cast<std::size_t>(std::ranges::count_if(result.bindings, [](const DescriptorBinding& binding) { return binding.role == DescriptorRole::FormatTable; }));
}

}

int main() {
    try {
        const std::vector<std::uint32_t> formatted{
            0x34020084u,
            0xe00c1000u, 0x80001401u,
            0xbf8c3f70u,
            0xe0781040u, 0x80021401u,
            0xbf810000u};
        const auto result = Recompiled(formatted);
        const auto local = CountLocalMemoryUsers(result.spirv.Words());
        Expect(local.dynamicallyIndexed == 0u, "a run-time formatted load leaves " + std::to_string(local.dynamicallyIndexed) + " dynamically indexed Private or Function variables, expected the format table read from constant memory");
        Expect(FormatTables(result) == 1u, "a run-time formatted load binds " + std::to_string(FormatTables(result)) + " format tables, expected one");
        const auto table = std::ranges::find_if(result.bindings, [](const DescriptorBinding& binding) { return binding.role == DescriptorRole::FormatTable; });
        Expect(table->kind == DescriptorKind::StorageBuffer && table->readOnly && table->count == 1u && table->guestDescriptor.empty() && table->binding == RuntimeAbi::BindingNumber(RuntimeAbi::Stage::Main, RuntimeAbi::Binding::FormatTable), "the format table is not one read-only storage buffer at its runtime ABI binding without guest words");

        const std::vector<std::uint32_t> constantFormat{
            0x34020084u,
            0xbe8303ffu, 0x31016facu,
            0xe00c1000u, 0x80001401u,
            0xbf8c3f70u,
            0xe0781040u, 0x80021401u,
            0xbf810000u};
        const auto plain = Recompiled(constantFormat);
        Expect(FormatTables(plain) == 0u, "a formatted load through a V# with a constant format binds the format table");

        const auto words = RuntimeFormatTable();
        const auto entry = [&](IrBufferFormat format) { return std::span<const std::uint32_t>(words).subspan(static_cast<std::uint32_t>(format) * RuntimeAbi::FormatTableEntryWords, RuntimeAbi::FormatTableEntryWords); };
        const auto invalid = entry(IrBufferFormat::Invalid);
        Expect(invalid[0] == 0u && invalid[1] == 0x20202020u && invalid[2] == 0u && invalid[3] == 0u, "the invalid format's entry is not marked invalid with 32-bit component widths");
        const auto single = entry(IrBufferFormat::Format32Float);
        Expect(single[0] == (1u | 1u << 1u | 5u << 5u) && single[1] == 0x20202020u && single[2] == 0u && single[3] == 0u, "the 32-bit float format's entry does not describe one float component of 32 bits at byte 0");
        const auto last = entry(IrBufferFormat::Format32_32_32_32Float);
        Expect(last[0] == (1u | 4u << 1u | 5u << 5u) && last[1] == 0x20202020u && last[3] == 0x0c080400u, "the four-component 32-bit float format's entry does not describe four float components at bytes 0, 4, 8 and 12");
        for (std::uint32_t format = static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float) + 1u; format < RuntimeAbi::FormatTableEntries; ++format) {
            Expect(words[format * RuntimeAbi::FormatTableEntryWords] == 0u, "format " + std::to_string(format) + " past the last buffer format is marked valid");
        }
    } catch (const std::exception& error) {
        std::cerr << "SpirvFormatTable: " << error.what() << '\n';
        return 1;
    }
    std::puts("SpirvFormatTable: run-time buffer formats are read from the device's read-only format table");
    return 0;
}
