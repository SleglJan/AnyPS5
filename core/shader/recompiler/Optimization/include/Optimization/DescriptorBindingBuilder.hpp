#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DESCRIPTORBINDINGBUILDER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DESCRIPTORBINDINGBUILDER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Recompiler.hpp"
#include <cstdint>

namespace ShaderRecompiler {

bool ReducesBetweenTexels(std::uint32_t word0, std::uint32_t filter);
DescriptorImageShape ImageShapeForResource(const ImageResource& image);
std::uint32_t PointFilteredSamplerWord(std::uint32_t word0, std::uint32_t filter);
struct PreparedDescriptorBinding {
    DescriptorBinding descriptor;
    std::vector<std::uint32_t> resources;
    std::vector<std::uint32_t> samplerFilterElements;
    std::vector<bool> inactive;
    std::vector<std::vector<std::uint32_t>> tableSamplerElements;
};

struct DescriptorBindingPlan {
    std::vector<PreparedDescriptorBinding> bindings;
    std::vector<PipelineSpecializationConstant> specialization;
    std::vector<std::uint32_t> runtimeState;
    bool storageData = false;
};

class DescriptorBindingBuilder {
public:
    DescriptorBindingPlan Prepare(const IrBindingLayout& layout, const ShaderInfo& info, IrShaderStage stage, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exportMappings = {}, bool registrationModules = false, bool fragmentDescriptorSet = false) const;
    DescriptorBindingPlan Select(const DescriptorBindingPlan& plan, std::span<const std::uint32_t> bindings, bool samplerHeap, bool imageHeap) const;
    void Populate(BindingAllocationResult& allocation, const CompiledBindingLayout& compiled, const DescriptorBindingPlan& plan, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, const std::vector<std::uint32_t>* tableMap = nullptr) const;
    void Populate(BindingAllocationResult& allocation, const IrProgram& program, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings = {}) const;
    void Populate(BindingAllocationResult& allocation, const ShaderInfo& info, IrShaderStage stage, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings = {}) const;
};

}

#endif
