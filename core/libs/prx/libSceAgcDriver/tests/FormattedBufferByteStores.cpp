#include "BdaAbi.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdio>
#include <exception>
#include <map>
#include <set>
#include <span>
#include <spirv/unified1/spirv.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

SpirvTarget Target(bool byteAccess) {
    static constexpr std::array<std::uint32_t, 3> withBytes{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess,
    };
    static constexpr std::array<std::uint32_t, 2> withoutBytes{spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses};
    static constexpr std::array<std::string_view, 4> extensions{
        "SPV_EXT_descriptor_indexing", "SPV_KHR_8bit_storage", "SPV_KHR_float_controls", "SPV_KHR_physical_storage_buffer",
    };
    const auto capabilities = byteAccess ? std::span<const std::uint32_t>(withBytes) : std::span<const std::uint32_t>(withoutBytes);
    return {0x00401000u, 0x00010300u, 32, BdaAbi::Version, capabilities, extensions, false,
            {1024, 1024, 64}, 1024, 32768, {}, {}};
}

struct Counts {
    std::size_t storeCompareExchanges = 0;
    std::map<std::string, std::size_t> byteStores;
    std::map<std::string, std::size_t> volatileByteStores;
    std::map<std::string, std::size_t> byteWriteCalls;
    std::map<std::string, std::set<std::string>> calls;
    bool byteCapability = false;
};

Counts Count(std::span<const std::uint32_t> words) {
    Check(words.size() >= 5 && words[0] == spv::MagicNumber, "formatted buffer byte stores: missing SPIR-V module");
    Counts counts;
    std::map<std::uint32_t, std::string> names;
    std::set<std::uint32_t> byteTypes;
    std::set<std::uint32_t> bytePointerTypes;
    std::set<std::uint32_t> bytePointers;
    std::set<std::uint32_t> wordTypes;
    std::set<std::uint32_t> ones;
    std::string function;
    for (std::size_t offset = 5; offset < words.size();) {
        const auto count = words[offset] >> spv::WordCountShift;
        Check(count != 0 && count <= words.size() - offset, "formatted buffer byte stores: malformed SPIR-V instruction");
        const auto opcode = words[offset] & spv::OpCodeMask;
        const auto* operands = words.data() + offset + 1;
        if (opcode == spv::OpName) names[operands[0]] = reinterpret_cast<const char*>(operands + 1);
        if (opcode == spv::OpCapability && operands[0] == spv::CapabilityStorageBuffer8BitAccess) counts.byteCapability = true;
        if (opcode == spv::OpTypeInt && operands[1] == 8u) byteTypes.insert(operands[0]);
        if (opcode == spv::OpTypeInt && operands[1] == 32u) wordTypes.insert(operands[0]);
        if (opcode == spv::OpConstant && count == 4u && wordTypes.contains(operands[0]) && operands[2] == 1u) ones.insert(operands[1]);
        if (opcode == spv::OpTypePointer && operands[1] == spv::StorageClassPhysicalStorageBuffer && byteTypes.contains(operands[2])) bytePointerTypes.insert(operands[0]);
        if (opcode == spv::OpFunction) function = names[operands[1]];
        if ((opcode == spv::OpConvertUToPtr || opcode == spv::OpAccessChain) && bytePointerTypes.contains(operands[0])) bytePointers.insert(operands[1]);
        if (opcode == spv::OpAtomicCompareExchange && function != "record_bda_fault" && function != "note_bda_write" && function != "note_bda_touch") ++counts.storeCompareExchanges;
        if (opcode == spv::OpStore && bytePointers.contains(operands[0])) {
            ++counts.byteStores[function];
            if (count > 3 && (operands[2] & spv::MemoryAccessVolatileMask) != 0u) ++counts.volatileByteStores[function];
        }
        if (opcode == spv::OpFunctionCall) {
            const auto callee = names[operands[2]];
            counts.calls[function].insert(callee);
            if ((callee == "write_bda" || callee == "write_bda_coherent") && count > 6u && ones.contains(operands[5])) ++counts.byteWriteCalls[callee];
        }
        offset += count;
    }
    return counts;
}

std::vector<std::uint32_t> Recompiled(bool byteAccess, bool coherent) {
    std::vector<std::uint32_t> code{0xe0000000u, 0x80020a00u, 0xbf8c3f70u, 0xe0100000u | (coherent ? 0x4000u : 0u), 0x80020a00u};
    code.push_back(0xbf810000u);
    const std::span<const std::uint32_t> words(code);
    const std::array<MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(words)}}};
    const std::array<std::uint32_t, 12> userData{0u, 0u, 0u, 0u, 0x10000000u, 0u, 0x1000u, 0x31016facu, 0x20000000u, 0u, 0x1000u, 0x31038facu};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), words, 0, {}};
    request.target = Target(byteAccess);
    request.layout = {0, 0, 0, 128};
    request.context.waveSize = 32;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    request.context.memory = memory;
    request.useCache = false;
    return Recompile(request).spirv.Copy();
}

}

int main() {
    try {
        {
            const std::string stage = "compute";
            auto bytes = Count(Recompiled(true, false));
            Check(bytes.storeCompareExchanges == 0, "formatted buffer byte stores: a " + stage + " store patches its dword with " + std::to_string(bytes.storeCompareExchanges) + " compare-exchange loops");
            Check(bytes.byteCapability && bytes.byteWriteCalls["write_bda"] >= 4, "formatted buffer byte stores: the 8-bit components of a " + stage + " store make " + std::to_string(bytes.byteWriteCalls["write_bda"]) + " one-byte writes, expected at least 4");
            Check(bytes.calls["write_bda"].contains("write_bda_bytes") && bytes.byteStores["write_bda_bytes"] >= 1, "formatted buffer byte stores: a one-byte write of a " + stage + " store does not reach a store through a physical byte pointer");
            auto coherent = Count(Recompiled(true, true));
            Check(coherent.storeCompareExchanges == 0 && coherent.byteWriteCalls["write_bda_coherent"] >= 4, "formatted buffer byte stores: the 8-bit components of a glc " + stage + " store make " + std::to_string(coherent.byteWriteCalls["write_bda_coherent"]) + " coherent one-byte writes (" + std::to_string(coherent.storeCompareExchanges) + " compare-exchange loops), expected at least 4");
            Check(coherent.calls["write_bda_coherent"].contains("write_bda_bytes_coherent") && coherent.volatileByteStores["write_bda_bytes_coherent"] >= 1, "formatted buffer byte stores: a glc " + stage + " store does not store its bytes as volatile byte stores");
        }
        bool rejected = false;
        try {
            static_cast<void>(Recompiled(false, false));
        } catch (const std::exception& error) {
            rejected = std::string(error.what()).find("capability 4448") != std::string::npos;
        }
        Check(rejected, "formatted buffer byte stores: a store through a run-time V# recompiled for a target without StorageBuffer8BitAccess");
        std::puts("formatted buffer byte stores tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
