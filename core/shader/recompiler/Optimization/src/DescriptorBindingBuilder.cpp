#include "Optimization/DescriptorBindingBuilder.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ShaderRecompiler {

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}


void MergeSamplerElements(std::vector<std::uint32_t>& into, std::span<const std::uint32_t> from) {
    into.insert(into.end(), from.begin(), from.end());
    std::ranges::sort(into);
    into.erase(std::unique(into.begin(), into.end()), into.end());
}

DescriptorKind PhysicalKindFor(DescriptorBindingKind kind) {
    if (kind == DescriptorBindingKind::Samplers || kind == DescriptorBindingKind::SamplerTable) {
        return DescriptorKind::SamplerSlot;
    }
    if (kind == DescriptorBindingKind::ImageTable) {
        return DescriptorKind::SampledImage;
    }
    const ImageResourceClass imageClass = ImageBindingResourceClass(kind);
    if (imageClass == ImageResourceClass::Sampled) {
        return DescriptorKind::SampledImage;
    }
    if (imageClass == ImageResourceClass::Storage) {
        return DescriptorKind::StorageImage;
    }
    return DescriptorKind::StorageBuffer;
}

DescriptorRole RoleFor(DescriptorBindingKind kind) {
    if (kind == DescriptorBindingKind::Samplers) {
        return DescriptorRole::GuestSamplers;
    }
    if (kind == DescriptorBindingKind::Gds) {
        return DescriptorRole::Gds;
    }
    if (kind == DescriptorBindingKind::BdaPagetable) {
        return DescriptorRole::BdaPagetable;
    }
    if (kind == DescriptorBindingKind::FaultBuffer) {
        return DescriptorRole::FaultBuffer;
    }
    if (kind == DescriptorBindingKind::FlattenedSrt) {
        return DescriptorRole::FlattenedSrt;
    }
    if (kind == DescriptorBindingKind::ShaderData) {
        return DescriptorRole::ShaderData;
    }
    if (kind == DescriptorBindingKind::ImageTable) {
        return DescriptorRole::ImageTable;
    }
    if (kind == DescriptorBindingKind::SamplerTable) {
        return DescriptorRole::SamplerTable;
    }
    if (kind == DescriptorBindingKind::ImageTableMap) {
        return DescriptorRole::ImageTableMap;
    }
    fail("DescriptorBindingBuilder::Populate binding kind has no descriptor role");
}

std::vector<std::uint32_t> GuestSamplersDescriptor(const std::vector<std::uint32_t>& resources, const ResourceSnapshot& snapshot) {
    std::vector<std::uint32_t> result;
    std::uint32_t dwordCount = 0;
    for (std::size_t i = 0; i < resources.size(); i++) {
        const std::uint32_t r = resources[i];
        if (r >= snapshot.samplers.size()) {
            fail("DescriptorBindingBuilder::Populate guest sampler index is out of range");
        }
        const DescriptorValue& value = snapshot.samplers[r];
        if (value.dwordCount == 0u) {
            fail("DescriptorBindingBuilder::Populate guest sampler descriptor is empty");
        }
        if (i == 0u) {
            dwordCount = value.dwordCount;
        } else if (value.dwordCount != dwordCount) {
            fail("DescriptorBindingBuilder::Populate guest sampler descriptors have inconsistent widths");
        }
        for (std::uint32_t dword = 0; dword < value.dwordCount; dword++) {
            result.push_back(value.dwords[dword]);
        }
    }
    return result;
}

constexpr std::uint32_t ForceUnnormalizedBit = 1u << 15u;

[[noreturn]] void failUnnormalized(const char* reason) {
    fail(std::string("DescriptorBindingBuilder::Populate unnormalized guest sampler ") + reason + ", which is not implemented");
}

struct UnnormalizedProof {
    std::vector<bool> samplers;
};

UnnormalizedProof ProveUnnormalized(const ShaderInfo& info, const ResourceSnapshot& snapshot) {
    UnnormalizedProof proof{std::vector<bool>(info.samplers.size())};
    for (std::uint32_t r = 0; r < info.samplers.size(); r++) {
        if (snapshot.samplers.at(r).dwordCount != 4u) fail("sampler descriptor must contain four dwords");
        const auto& sampler = info.samplers[r];
        if (sampler.table != NoTable || sampler.deviceTable) continue;
        if ((snapshot.samplers[r].dwords[0] & ForceUnnormalizedBit) == 0u) {
            if (sampler.foldTexelOffsetsOnly) {
                fail("DescriptorBindingBuilder::Populate normalized guest sampler is used with a texel offset that is not a constant, which image sampling takes only with VK_KHR_maintenance8 and shaderImageGatherExtended");
            }
            continue;
        }
        if (const auto unsupported = UnnormalizedUnsupportedUses(sampler)) {
            failUnnormalized(UnnormalizedUseReason(unsupported));
        }
        if (sampler.depthCompare) {
            failUnnormalized("is used with depth comparison");
        }
        if (std::ranges::any_of(info.sampledPairs, [&](const SampledResourcePair& pair) { return pair.sampler == r; })) {
            failUnnormalized("samples an image selected at run time");
        }
        proof.samplers[r] = true;
    }
    return proof;
}

std::vector<std::uint32_t> ShaderDataDwordsFor(const IrBindingLayout& layout, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint32_t> runtimeState) {
    std::vector<std::uint32_t> result(layout.ShaderDataDwords(), 0u);
    for (std::size_t i = 0; i < layout.userDataRegisters.size(); ++i) {
        const auto reg = layout.userDataRegisters[i];
        if (reg < userDataBase || reg - userDataBase >= snapshot.userData.size()) fail("shader user-data register is out of range");
        result[i] = snapshot.userData[reg - userDataBase];
    }
    if (layout.dispatchThreadLimit) {
        if (partialThreads == std::array<std::uint32_t, 3>{}) fail("partial-group shader has no dispatch size");
        std::copy(partialThreads.begin(), partialThreads.end(), result.begin() + layout.DispatchThreadLimitDword());
    }
    if (runtimeState.size() != layout.runtimeStateDwords) fail("runtime state does not match its shader data layout");
    std::copy(runtimeState.begin(), runtimeState.end(), result.begin() + layout.RuntimeStateDword());
    return result;
}

}

DescriptorImageShape ImageShapeForResource(const ImageResource& image) {
    const RdnaImageDimensionInfo& info = RdnaImageDimensionInfoFor(image.dimension);
    if (info.multisampled != 0u) {
        if (info.arrayed != 0u) {
            fail("DescriptorBindingBuilder::Populate multisampled array image resources have no descriptor image shape");
        }
        return DescriptorImageShape::Image2DMultisample;
    }
    if (info.spirvDimension == spv::Dim1D) {
        return info.arrayed != 0u ? DescriptorImageShape::Image1DArray : DescriptorImageShape::Image1D;
    }
    if (info.spirvDimension == spv::Dim3D) {
        return DescriptorImageShape::Image3D;
    }
    if (info.spirvDimension == spv::Dim2D) {
        if (image.cube && info.arrayed == 0u) {
            fail("DescriptorBindingBuilder::Populate cube image resource is not arrayed");
        }
        return info.arrayed != 0u ? DescriptorImageShape::Image2DArray : DescriptorImageShape::Image2D;
    }
    fail("DescriptorBindingBuilder::Populate image resource dimension has no descriptor image shape");
}

bool ReducesBetweenTexels(std::uint32_t word0, std::uint32_t filter) {
    return ((word0 >> 29u) & 3u) != 0u && (((filter >> 20u) & 0xfu) != 0u || ((filter >> 26u) & 3u) == 2u);
}

std::uint32_t PointFilteredSamplerWord(std::uint32_t word0, std::uint32_t filter) {
    if (ReducesBetweenTexels(word0, filter)) {
        fail("DescriptorBindingBuilder: a min or max reduction sampler that filters between texels or mip levels samples an image that needs point filtering (sint, converted or depth-bits format), which is not implemented");
    }
    const bool mipmapped = ((filter >> 26u) & 3u) != 0u;
    return (filter & ~(0xffu << 20u)) | (1u << 24u) | (mipmapped ? 1u << 26u : 0u);
}

void DescriptorBindingBuilder::Populate(BindingAllocationResult& allocation, const IrProgram& program, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings) const {
    Populate(allocation, program.Info(), program.Resources().stage, program.Resources().userDataBase, snapshot, partialThreads, exportMappings);
}

void DescriptorBindingBuilder::Populate(BindingAllocationResult& allocation, const ShaderInfo& info, IrShaderStage stage, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings) const {
    auto plan = Prepare(allocation.layout, info, stage, snapshot, exportMappings);
    const auto tables = ResourceMaterializer::ResolveImageTables(info, snapshot);
    Populate(allocation, allocation, plan, userDataBase, snapshot, partialThreads, &tables);
    allocation.specialization = std::move(plan.specialization);
}

DescriptorBindingPlan DescriptorBindingBuilder::Prepare(const IrBindingLayout& layout, const ShaderInfo& info, IrShaderStage stage, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exportMappings, bool registrationModules, bool fragmentDescriptorSet) const {
    if (stage == IrShaderStage::Pixel && exportMappings.size() != 8u) fail("fragment runtime export mappings are missing");
    if (info.images.size() > RuntimeAbi::ImageCapacity) fail("runtime image count exceeds the static capacity");
    DescriptorBindingPlan plan;
    const auto unnormalized = ProveUnnormalized(info, snapshot);
    for (std::uint32_t index = 0; index < info.samplers.size(); ++index) {
        if (FoldsTexelOffsets(info.samplers[index])) plan.specialization.push_back({PipelineSpecialization::FoldOffset(index), unnormalized.samplers[index] ? 1u : 0u});
        if (SamplesBaseLevelIfUnnormalized(info.samplers[index])) plan.specialization.push_back({PipelineSpecialization::UnnormalizedBaseLevel(index), unnormalized.samplers[index] ? 1u : 0u});
    }
    for (std::uint32_t target = 0; target < exportMappings.size() && !registrationModules; ++target) {
        for (std::uint32_t component = 0; component < 4u; ++component) plan.specialization.push_back({PipelineSpecialization::ExportBase + target * 4u + component, (exportMappings[target] >> (component * 2u)) & 3u});
    }
    std::vector<bool> pairedSamplers(info.samplers.size(), false);
    for (const auto& pair : info.sampledPairs) {
        if (!info.samplers.at(pair.sampler).deviceTable) pairedSamplers.at(pair.sampler) = true;
    }
    plan.bindings.reserve(layout.descriptors.size());
    for (const auto& logical : layout.descriptors) {
        if (ImageBindingResourceClass(logical.kind) != ImageResourceClass::None) fail("an image table resource was placed in a typed heap");
        const bool samplerHeap = logical.kind == DescriptorBindingKind::Samplers;
        IrDescriptorBinding compact{logical.kind, logical.resources};
        std::vector<bool> inactive;
        for (const auto resource : logical.resources) inactive.push_back(samplerHeap && !pairedSamplers.at(resource));
        if (samplerHeap && compact.resources.empty()) continue;
        PreparedDescriptorBinding entry{};
        auto& physical = entry.descriptor;
        physical.descriptorSet = fragmentDescriptorSet && stage == IrShaderStage::Pixel ? RuntimeAbi::FragmentDescriptorSet : RuntimeAbi::DescriptorSet;
        physical.binding = NativeBinding(stage, compact.kind);
        physical.count = compact.resources.empty() ? 1u : static_cast<std::uint32_t>(compact.resources.size());
        physical.kind = PhysicalKindFor(compact.kind);
        physical.role = RoleFor(compact.kind);
        if (physical.role == DescriptorRole::GuestSamplers) {
            for (std::uint32_t element = 0; element < compact.resources.size(); ++element) {
                const auto resource = compact.resources[element];
                bool compare = false;
                for (const auto& pair : info.sampledPairs) if (pair.sampler == resource) compare |= info.images.at(pair.image).depthCompare;
                physical.samplerDepthCompare.push_back(compare);
                physical.samplerUnnormalized.push_back(unnormalized.samplers.at(resource));
                physical.samplerIntegerBorder.push_back(false);
                physical.samplerUsed.push_back(!inactive[element]);
                const auto& words = snapshot.samplers.at(resource).dwords;
                if ((element & 1u) != 0u && !inactive[element] && !ReducesBetweenTexels(words[0], words[2])) entry.samplerFilterElements.push_back(element);
            }
        } else if (physical.role == DescriptorRole::ShaderData && layout.UsesPushData()) {
            fail("DescriptorBindingBuilder::Populate shader-data binding must not exist when push data is used");
        }
        plan.storageData |= physical.role == DescriptorRole::ShaderData;
        entry.resources = std::move(compact.resources);
        if (samplerHeap) entry.inactive = std::move(inactive);
        plan.bindings.push_back(std::move(entry));
    }
    if (layout.runtimeStateDwords != 0u) {
        plan.runtimeState.assign(layout.runtimeStateDwords, 0u);
        for (std::uint32_t target = 0; target < exportMappings.size(); ++target) plan.runtimeState.at(RuntimeAbi::StateExportMappings + target / 4u) |= static_cast<std::uint32_t>(exportMappings[target]) << (target % 4u * 8u);
    }
    std::vector<std::vector<std::uint32_t>> samplerElements(info.samplers.size());
    for (const auto& binding : plan.bindings) {
        if (binding.descriptor.role != DescriptorRole::GuestSamplers) continue;
        for (std::uint32_t element = 0; element < binding.resources.size(); ++element) {
            if (binding.inactive.empty() || !binding.inactive[element]) samplerElements.at(binding.resources[element]).push_back(element);
        }
    }
    for (auto& binding : plan.bindings) {
        if (binding.descriptor.role == DescriptorRole::ImageTable) binding.tableSamplerElements = samplerElements;
    }
    return plan;
}

DescriptorBindingPlan DescriptorBindingBuilder::Select(const DescriptorBindingPlan& plan, std::span<const std::uint32_t> bindings, bool samplerHeap, bool imageHeap) const {
    DescriptorBindingPlan selected;
    std::set<std::uint32_t> remaining(bindings.begin(), bindings.end());
    selected.bindings.reserve(remaining.size());
    for (const auto& binding : plan.bindings) {
        const bool declared = remaining.erase(binding.descriptor.binding) != 0u;
        const bool kept = binding.descriptor.kind == DescriptorKind::SamplerSlot ? samplerHeap : binding.descriptor.role == DescriptorRole::ImageTable ? imageHeap : declared || binding.descriptor.kind == DescriptorKind::DeviceAddress;
        if (!kept) continue;
        selected.storageData |= binding.descriptor.role == DescriptorRole::ShaderData;
        selected.bindings.push_back(binding);
    }
    if (!remaining.empty()) fail("specialized module bindings disagree with the prepared plan");
    const bool hasSamplers = std::any_of(selected.bindings.begin(), selected.bindings.end(), [](const auto& binding) { return binding.descriptor.role == DescriptorRole::GuestSamplers; });
    if (!hasSamplers) {
        for (auto& binding : selected.bindings) {
            for (auto& samplers : binding.descriptor.imageSamplers) samplers.clear();
            binding.tableSamplerElements.clear();
        }
    }
    selected.runtimeState = plan.runtimeState;
    return selected;
}

void DescriptorBindingBuilder::Populate(BindingAllocationResult& allocation, const CompiledBindingLayout& compiled, const DescriptorBindingPlan& plan, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, const ResolvedImageTables* tables) const {
    const auto& layout = compiled.layout;
    const bool needsShaderData = layout.UsesPushData() || plan.storageData;
    auto shaderData = needsShaderData ? ShaderDataDwordsFor(layout, userDataBase, snapshot, partialThreads, plan.runtimeState) : std::vector<std::uint32_t>{};
    allocation.shaderDataUserWords.clear();
    if (needsShaderData) {
        for (const auto reg : layout.userDataRegisters) allocation.shaderDataUserWords.push_back(reg - userDataBase);
    }
    std::vector<DescriptorBinding> bindings;
    bindings.reserve(plan.bindings.size());
    for (const auto& entry : plan.bindings) {
        auto physical = entry.descriptor;
        switch (physical.role) {
        case DescriptorRole::GuestSamplers:
            physical.guestDescriptor = GuestSamplersDescriptor(entry.resources, snapshot);
            for (const auto element : entry.samplerFilterElements) {
                auto& filter = physical.guestDescriptor.at(element * 4u + 2u);
                filter = PointFilteredSamplerWord(physical.guestDescriptor.at(element * 4u), filter);
            }
            break;
        case DescriptorRole::ImageTable:
            if (tables == nullptr) fail("DescriptorBindingBuilder::Populate image table elements were not resolved");
            physical.guestDescriptor.reserve(tables->elements.size() * 8u);
            for (const auto& element : tables->elements) {
                physical.guestDescriptor.insert(physical.guestDescriptor.end(), element.words.begin(), element.words.end());
                ImageResource shape;
                shape.dimension = element.dimension;
                physical.imageShapes.push_back(ImageShapeForResource(shape));
                physical.imageDepthCompare.push_back(element.depthCompare);
                physical.imageDecodedCopy.push_back(element.decodedCopy);
                physical.imageWritten.push_back(false);
                physical.imageAtomic.push_back(false);
                std::vector<std::uint32_t> samplers;
                for (const auto sampler : element.samplers) {
                    if (sampler < entry.tableSamplerElements.size()) MergeSamplerElements(samplers, entry.tableSamplerElements[sampler]);
                }
                physical.imageSamplers.push_back(std::move(samplers));
            }
            physical.count = static_cast<std::uint32_t>(tables->elements.size());
            physical.tableSamplerElements = entry.tableSamplerElements;
            break;
        case DescriptorRole::SamplerTable:
            if (tables == nullptr) fail("DescriptorBindingBuilder::Populate sampler table elements were not resolved");
            physical.guestDescriptor.reserve(tables->samplers.size() * 4u);
            for (const auto& element : tables->samplers) {
                const auto first = physical.guestDescriptor.size();
                physical.guestDescriptor.insert(physical.guestDescriptor.end(), element.words.begin(), element.words.end());
                if (element.point) physical.guestDescriptor[first + 2u] = PointFilteredSamplerWord(physical.guestDescriptor[first], physical.guestDescriptor[first + 2u]);
                physical.samplerDepthCompare.push_back(element.compare);
            }
            physical.count = static_cast<std::uint32_t>(tables->samplers.size());
            break;
        case DescriptorRole::ImageTableMap:
            if (tables == nullptr || tables->map.empty()) fail("DescriptorBindingBuilder::Populate image table map was not resolved");
            physical.guestDescriptor = tables->map;
            break;
        case DescriptorRole::FlattenedSrt:
            if (snapshot.flattenedSrt.empty()) {
                fail("DescriptorBindingBuilder::Populate flattened SRT snapshot is empty");
            }
            physical.guestDescriptor = snapshot.flattenedSrt;
            break;
        case DescriptorRole::ShaderData:
            physical.guestDescriptor = std::move(shaderData);
            break;
        case DescriptorRole::Gds:
        case DescriptorRole::BdaPagetable:
        case DescriptorRole::FaultBuffer:
            break;
        }

        if (physical.role == DescriptorRole::GuestSamplers) {
            if (physical.count == 0u || physical.guestDescriptor.size() % physical.count != 0u) {
                fail("DescriptorBindingBuilder::Populate guest descriptor size is not a multiple of the binding count");
            }
        }

        bindings.push_back(std::move(physical));
    }

    allocation.bindings = std::move(bindings);
    allocation.pushConstants.clear();
    if (layout.UsesPushData()) {
        allocation.pushConstants.resize(static_cast<std::size_t>(shaderData.size()) * sizeof(std::uint32_t));
        std::memcpy(allocation.pushConstants.data(), shaderData.data(), allocation.pushConstants.size());
    }
}

}
