#include "Recompiler.hpp"
#include "BdaAbi.hpp"
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

std::vector<std::uint32_t> Recompiled(std::span<const std::uint32_t> code, bool segments) {
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
    request.target.subgroupSize = 32u;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.bdaSegments = segments;
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    auto result = Recompile(request);
    if (result.spirv.empty()) throw std::runtime_error("the shader did not recompile");
    return result.spirv.Copy();
}

struct Shape {
    std::map<std::string, std::uint32_t> mainCalls;
    std::uint32_t dontInline = 0;
    std::uint32_t outsideSegmentRecords = 0;
};

Shape ShapeOf(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    std::map<std::uint32_t, std::uint32_t> constants;
    std::uint32_t entry = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
        if (op == spv::OpEntryPoint) entry = words[cursor + 2];
        if (op == spv::OpConstant && (words[cursor] >> 16u) == 4u) constants[words[cursor + 2]] = words[cursor + 3];
    }
    Shape shape;
    std::uint32_t function = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) {
            function = words[cursor + 2];
            if ((words[cursor + 3] & spv::FunctionControlDontInlineMask) != 0u) ++shape.dontInline;
        }
        if (op != spv::OpFunctionCall) continue;
        const auto& callee = names[words[cursor + 3]];
        if (function == entry || names[function].starts_with("load_")) ++shape.mainCalls[callee];
        if (callee == "record_bda_fault" && (words[cursor] >> 16u) > 4u) {
            const auto reason = constants.find(words[cursor + 4]);
            if (reason != constants.end() && reason->second == static_cast<std::uint32_t>(BdaAbi::FaultReason::OutsideSegment)) ++shape.outsideSegmentRecords;
        }
    }
    return shape;
}

std::uint32_t Calls(const Shape& shape, std::string_view callee) {
    const auto found = shape.mainCalls.find(std::string(callee));
    return found == shape.mainCalls.end() ? 0u : found->second;
}

std::uint32_t WalkCalls(const Shape& shape) {
    std::uint32_t calls = 0;
    for (const auto& [callee, count] : shape.mainCalls) {
        if (callee == "get_bda_pointer" || callee == "probe_bda_pointer" || callee == "bda_translate" || callee.starts_with("read_bda")) calls += count;
    }
    return calls;
}

std::string Listed(const Shape& shape) {
    std::string text;
    for (const auto& [callee, count] : shape.mainCalls) text += " " + callee + " x" + std::to_string(count);
    return text;
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
        const auto walk = ShapeOf(Recompiled(twoBuffers, false));
        const auto segment = ShapeOf(Recompiled(twoBuffers, true));
        Expect(walk.dontInline != 0u, "the walk program marks no helper DontInline; the test no longer tells the two programs apart");
        Expect(WalkCalls(walk) != 0u, "the walk program's main and its load helpers call no page-table walk on a segment miss; the test no longer tells the two programs apart; main and its load helpers call" + Listed(walk));
        Expect(Calls(segment, "bda_segment") == 2u, "the segment program translates " + std::to_string(Calls(segment, "bda_segment")) + " V# bases, expected one per V#");
        Expect(WalkCalls(segment) == 0u, "the segment program's main and its load helpers call the page-table walk " + std::to_string(WalkCalls(segment)) + " times, expected none: a read outside its V#'s settled range is a fault, not a walk; main and its load helpers call" + Listed(segment));
        Expect(segment.outsideSegmentRecords != 0u, "the segment program records no out-of-segment fault, expected a read outside the settled range to record one");
        Expect(Calls(segment, "flush_bda_fault") != 0u, "the segment program never publishes its deferred fault record before the invocation ends");
        Expect(segment.dontInline == 0u, "the segment program keeps " + std::to_string(segment.dontInline) + " helpers out of line, expected every helper inline");
    } catch (const std::exception& error) {
        std::cerr << "SpirvSegmentPrograms: " << error.what() << '\n';
        return 1;
    }
    std::puts("SpirvSegmentPrograms: the segment program reads each V# through its settled range inline, records a read outside it as a fault and never walks the page table");
    return 0;
}
