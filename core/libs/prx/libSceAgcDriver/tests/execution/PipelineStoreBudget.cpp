#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "Recompiler.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::PipelineBudget;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Programs = 40;
constexpr std::uint32_t Kept = 8;
constexpr std::uint32_t ProgramWords = 64;
constexpr std::uint32_t Untouched = 0xcafebabeu;

TestGuest::Array<std::uint32_t, Programs * ProgramWords> Code;
TestGuest::Array<std::uint32_t, Threads> Input;
TestGuest::Array<std::uint32_t, Threads> Output;

std::array<std::uint32_t, 4> Descriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::span<const std::uint32_t> Program(std::uint32_t index) {
    return {Code.data() + index * ProgramWords, 8};
}

struct Compiled {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::uint64_t address = 0;
};

Compiled Compile(AgcDriver::VulkanDevice& device, std::uint32_t index) {
    const auto code = Program(index);
    const auto source = Descriptor(Input.data(), decltype(Input)::ByteSize);
    const auto target = Descriptor(Output.data(), decltype(Output)::ByteSize);
    const std::array<std::uint32_t, 8> userData{source[0], source[1], source[2], source[3], target[0], target[1], target[2], target[3]};
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    const ShaderRecompiler::RecompileRequest request{{ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}}, {Threads, 0, userData, compute, std::nullopt, std::nullopt, memory}, device.ComputeTarget(Threads), {0, 0, 0, 128}};
    const auto handle = ShaderRecompiler::PrepareShader(request);
    ShaderRecompiler::SrtRuntime runtime{};
    runtime.userData = userData;
    const auto capture = ShaderRecompiler::CaptureResources(request, runtime, *handle);
    return {ShaderRecompiler::MaterializeShader(request, *capture, *handle), reinterpret_cast<std::uintptr_t>(code.data())};
}

void Expect(std::uint32_t added, const std::string& what) {
    for (std::uint32_t lane = 0; lane < Threads; ++lane) Require(Output[lane] == Input[lane] + added, what + ": lane " + std::to_string(lane) + " is " + std::to_string(Output[lane]) + ", expected " + std::to_string(Input[lane] + added));
}

void Flood(AgcDriver::VulkanDevice& device, const std::vector<Compiled>& programs, std::size_t budget, const std::string& what) {
    for (std::uint32_t index = 1; index < Programs; ++index) {
        static_cast<void>(device.PrepareComputePipeline(*programs[index].result, programs[index].address));
        Require(PipelineBudget::LiveBytes() <= budget, what + ": " + std::to_string(PipelineBudget::LiveBytes()) + " live pipeline bytes after " + std::to_string(index) + " programs, over the budget of " + std::to_string(budget));
    }
}

void Run(AgcDriver::VulkanDevice& device) {
    std::vector<Compiled> programs;
    std::size_t largest = 0;
    for (std::uint32_t index = 0; index < Programs; ++index) {
        programs.push_back(Compile(device, index));
        largest = std::max(largest, PipelineBudget::PipelineBytes(programs.back().result->spirv.size()));
    }
    const auto budget = PipelineBudget::LiveBytes() + Kept * largest;
    PipelineBudget::SetBudgetBytes(budget);

    Output.fill(Untouched);
    device.Dispatch(*programs[0].result, 1, 1, 1, programs[0].address);
    const auto evictedBefore = PipelineBudget::Evicted();
    Flood(device, programs, budget, "a store fed more pipelines than its budget while one is in a pending batch");
    Require(PipelineBudget::Evicted() - evictedBefore >= Programs - 1u - Kept, "the store evicted " + std::to_string(PipelineBudget::Evicted() - evictedBefore) + " pipelines, expected at least " + std::to_string(Programs - 1u - Kept));
    Require(!device.PrepareComputePipeline(*programs[0].result, programs[0].address), "the pipeline of a pending batch was evicted from the store");
    device.WaitIdle();
    Expect(1u, "the pending dispatch whose pipeline the store kept");

    Flood(device, programs, budget, "a store fed more pipelines than its budget after the batch retired");
    const auto feedbackBefore = AgcDriver::Graphics::PipelineCreationFeedbackTotals();
    Require(device.PrepareComputePipeline(*programs[0].result, programs[0].address), "the least recently used pipeline, no longer held by a batch, was not evicted");
    const auto feedbackAfter = AgcDriver::Graphics::PipelineCreationFeedbackTotals();
    constexpr auto compute = static_cast<std::size_t>(AgcDriver::Graphics::PipelineCreationKind::Compute);
    Require(feedbackAfter.unreported[compute] == feedbackBefore.unreported[compute], "the driver reported no creation feedback for the recreated compute pipeline");
    Require(feedbackAfter.hits[compute] == feedbackBefore.hits[compute] + 1u && feedbackAfter.misses[compute] == feedbackBefore.misses[compute], "the pipeline recreated after its eviction missed the pipeline cache (" + std::to_string(feedbackAfter.hits[compute] - feedbackBefore.hits[compute]) + " hits, " + std::to_string(feedbackAfter.misses[compute] - feedbackBefore.misses[compute]) + " misses)");
    Output.fill(Untouched);
    device.Dispatch(*programs[0].result, 1, 1, 1, programs[0].address);
    device.WaitIdle();
    Expect(1u, "a dispatch through a pipeline recreated after its eviction");
    for (const std::uint32_t index : {Programs - 1u, Programs - 2u}) {
        Output.fill(Untouched);
        device.Dispatch(*programs[index].result, 1, 1, 1, programs[index].address);
        device.WaitIdle();
        Expect(index + 1u, "a dispatch through a pipeline the store kept");
    }
    Require(PipelineBudget::LiveBytes() <= budget, "the store is over its budget at the end");
}

}

int main() {
    try {
        for (std::uint32_t lane = 0; lane < Threads; ++lane) Input[lane] = lane * 2654435761u + 17u;
        for (std::uint32_t index = 0; index < Programs; ++index) {
            const std::array<std::uint32_t, 8> words{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a040400u | (0x81u + index), 0xe0701000u, 0x80010201u, 0xbf810000u};
            std::copy(words.begin(), words.end(), Code.begin() + index * ProgramWords);
        }
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        std::puts("pipeline store budget tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
