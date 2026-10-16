#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineDescription.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DescriptorHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineSpecialization.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <list>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Graphics {

void LogPipelineStatistics_nid_no_patch(const Context& context, VkPipeline pipeline) {
    if (!context.pipelineExecutableInfo) return;
    const auto getProperties = context.Function<PFN_vkGetPipelineExecutablePropertiesKHR>("vkGetPipelineExecutablePropertiesKHR");
    const auto getStatistics = context.Function<PFN_vkGetPipelineExecutableStatisticsKHR>("vkGetPipelineExecutableStatisticsKHR");
    VkPipelineInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pipelineInfo.pipeline = pipeline;
    std::uint32_t count = 0;
    Check(getProperties(context.device, &pipelineInfo, &count, nullptr), "vkGetPipelineExecutablePropertiesKHR count");
    std::vector<VkPipelineExecutablePropertiesKHR> properties(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    Check(getProperties(context.device, &pipelineInfo, &count, properties.data()), "vkGetPipelineExecutablePropertiesKHR");
    properties.resize(count);
    for (std::uint32_t index = 0; index < properties.size(); ++index) {
        VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        executable.pipeline = pipeline;
        executable.executableIndex = index;
        count = 0;
        Check(getStatistics(context.device, &executable, &count, nullptr), "vkGetPipelineExecutableStatisticsKHR count");
        std::vector<VkPipelineExecutableStatisticKHR> statistics(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        Check(getStatistics(context.device, &executable, &count, statistics.data()), "vkGetPipelineExecutableStatisticsKHR");
        statistics.resize(count);
        const auto& property = properties[index];
        std::fprintf(stderr, "[pipeline-stats] executable=%u stages=0x%x subgroup=%u name=%s\n", index, property.stages, property.subgroupSize, property.name);
        for (const auto& statistic : statistics) {
            std::fprintf(stderr, "[pipeline-stats] executable=%u %s=", index, statistic.name);
            switch (statistic.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                std::fprintf(stderr, "%s", statistic.value.b32 ? "true" : "false");
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                std::fprintf(stderr, "%lld", static_cast<long long>(statistic.value.i64));
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                std::fprintf(stderr, "%llu", static_cast<unsigned long long>(statistic.value.u64));
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                std::fprintf(stderr, "%.17g", statistic.value.f64);
                break;
            default:
                throw std::runtime_error("unsupported pipeline statistic format");
            }
            std::fprintf(stderr, " (%s)\n", statistic.description);
        }
    }
}

bool HasStencil(VkFormat format) {
    return format == VK_FORMAT_D16_UNORM_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_S8_UINT;
}

Framebuffer::Framebuffer(const Context& context, VkRenderPass renderPass, std::span<const VkImageView> targets, VkExtent2D extent, std::uint32_t layers) : context(context) {
    // Cached objects outlive their device's teardown; they must not keep its buffer pool alive past it.
    this->context.bufferPool.reset();
    Require(extent.width != 0 && extent.height != 0 && extent.width <= context.limits.maxFramebufferWidth && extent.height <= context.limits.maxFramebufferHeight, "framebuffer extent exceeds device limits");
    Require(layers != 0 && layers <= context.limits.maxFramebufferLayers, "framebuffer layers exceed device limits");
    views.assign(targets.begin(), targets.end());
    if (renderPass == VK_NULL_HANDLE) return;
    VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = static_cast<std::uint32_t>(targets.size());
    framebufferInfo.pAttachments = targets.empty() ? nullptr : targets.data();
    framebufferInfo.width = extent.width;
    framebufferInfo.height = extent.height;
    framebufferInfo.layers = layers;
    Check(context.Function<PFN_vkCreateFramebuffer>("vkCreateFramebuffer")(context.device, &framebufferInfo, nullptr, &framebuffer), "vkCreateFramebuffer");
}

Framebuffer::~Framebuffer() {
    if (framebuffer) context.Function<PFN_vkDestroyFramebuffer>("vkDestroyFramebuffer")(context.device, framebuffer, nullptr);
}

void ValidateDepthBounds(const Context& context, const State& state) {
    Require(!state.depthBoundsTest || context.depthRangeUnrestricted || (state.minDepthBounds >= 0.0f && state.minDepthBounds <= 1.0f && state.maxDepthBounds >= 0.0f && state.maxDepthBounds <= 1.0f), "depth bounds outside [0, 1] require VK_EXT_depth_range_unrestricted");
}

void ValidateProvokingVertex(const Context& context, const State& state, std::span<const CompiledShader> shaders) {
    Require(state.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT || state.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT, "invalid provoking vertex mode");
    if (state.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT) {
        for (const auto& shader : shaders) {
            if (shader.stage != ShaderRecompiler::ShaderStage::Fragment || shader.program == nullptr) continue;
            Require(std::none_of(shader.program->fragmentParameters.begin(), shader.program->fragmentParameters.end(), [](const ShaderRecompiler::FragmentParameter& parameter) { return parameter.flatP10; }), "v_interp_mov p10 of a flat input with the first provoking vertex is not measured");
        }
        return;
    }
    Require(context.provokingVertexLast, "last provoking vertex requires VK_EXT_provoking_vertex with provokingVertexLast enabled");
    Require(!state.rectList && state.stages.path == ShaderPath::Vertex && !state.stages.mesh && !state.stages.tessellation, "last provoking vertex is unsupported for generated primitive pipelines");
    Require(state.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, "last provoking vertex is unsupported for this primitive topology");
    for (const auto& shader : shaders) {
        if (shader.stage != ShaderRecompiler::ShaderStage::Fragment) continue;
        Require(shader.program != nullptr, "missing compiled fragment shader");
        Require(std::none_of(shader.program->fragmentParameters.begin(), shader.program->fragmentParameters.end(), [](const ShaderRecompiler::FragmentParameter& parameter) { return parameter.flat && parameter.perVertex; }), "last provoking vertex with explicit per-vertex flat interpolation is unsupported");
        Require(state.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || std::none_of(shader.program->fragmentParameters.begin(), shader.program->fragmentParameters.end(), [](const ShaderRecompiler::FragmentParameter& parameter) { return parameter.vertexOrder; }), "last provoking vertex with explicit per-vertex triangle-strip interpolation is unsupported");
    }
}

bool ValidateSampleLocations(const Context& context, const State& state) {
    const bool extensionSupportsSamples = (context.sampleLocationSampleCounts & state.samples) != 0;
    Require(!state.customSampleLocations || extensionSupportsSamples, "custom sample locations require VK_EXT_sample_locations support for the draw's sample count");
    Require(!state.sampleLocationGrid || (context.sampleLocationGridSampleCounts & state.samples) != 0, "sample locations varying between pixels require a 2x2 sample location grid for the draw's sample count");
    const bool enabled = extensionSupportsSamples && (!state.depth || SampleLocationsCompatibleDepth(context, HostDepthFormat(context, state.depth->format), state.depth->samples));
    Require(!state.customSampleLocations || enabled, "custom sample locations require a compatible depth format and sample count");
    return enabled;
}

void ValidateViewport(const Context& context, const VkViewport& viewport) {
    Require(std::isfinite(viewport.minDepth) && std::isfinite(viewport.maxDepth), "non-finite viewport depth range");
    Require(context.depthRangeUnrestricted || (viewport.minDepth >= 0 && viewport.minDepth <= 1 && viewport.maxDepth >= 0 && viewport.maxDepth <= 1), "viewport depth outside [0, 1] requires VK_EXT_depth_range_unrestricted");
    Require(std::isfinite(viewport.x) && std::isfinite(viewport.y) && std::isfinite(viewport.width) && std::isfinite(viewport.height), "viewport arithmetic overflow");
    Require(viewport.width <= context.limits.maxViewportDimensions[0] && std::abs(viewport.height) <= context.limits.maxViewportDimensions[1], "viewport dimensions exceed device limits");
    Require(viewport.x >= context.limits.viewportBoundsRange[0] && viewport.x + viewport.width <= context.limits.viewportBoundsRange[1], "viewport X exceeds device bounds");
    Require(std::min(viewport.y, viewport.y + viewport.height) >= context.limits.viewportBoundsRange[0] && std::max(viewport.y, viewport.y + viewport.height) <= context.limits.viewportBoundsRange[1], "viewport Y exceeds device bounds");
}

bool ShadersWriteMemory(std::span<const CompiledShader> shaders) {
    const auto any = [](const std::vector<bool>& written, std::uint32_t count) { return written.size() < count || std::ranges::any_of(written, [](bool element) { return element; }); };
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) continue;
        for (const auto& binding : shader.program->bindings) {
            switch (binding.role) {
            case ShaderRecompiler::DescriptorRole::GuestImages:
                if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && any(binding.imageWritten, binding.count)) return true;
                break;
            case ShaderRecompiler::DescriptorRole::Gds:
                return true;
            case ShaderRecompiler::DescriptorRole::BdaPagetable:
                if (shader.program->bdaWrites) return true;
                break;
            case ShaderRecompiler::DescriptorRole::GuestSamplers:
            case ShaderRecompiler::DescriptorRole::FaultBuffer:
            case ShaderRecompiler::DescriptorRole::FlattenedSrt:
            case ShaderRecompiler::DescriptorRole::ShaderData:
                break;
            }
        }
    }
    return false;
}

void ValidateSampleMask(const Context& context, const State& state, std::span<const CompiledShader> shaders) {
    if (!PartialSampleMask(state)) return;
    for (const auto& shader : shaders) {
        if (shader.stage != ShaderRecompiler::ShaderStage::Fragment) continue;
        Require(shader.program != nullptr, "missing compiled fragment shader");
        if (UniformSampleMask(state)) {
            Require(context.earlyFragmentSampleMaskTestBeforeSampleCounting || !DeclaresEarlyFragmentTests(*shader.program), "a sample mask with early fragment tests needs VkPhysicalDeviceMaintenance5Properties::earlyFragmentSampleMaskTestBeforeSampleCounting, which the device lacks");
            continue;
        }
        const bool tested = state.depth.has_value() && (state.depthTest || state.stencilTest || state.depthBoundsTest);
        Require(!state.pixelEarlyZ || !tested || !ShadersWriteMemory(std::span(&shader, 1)), "a sample mask that differs between the pixels of a quad, from a pixel program with early depth or stencil tests that writes memory, is unsupported: the masked pixels are removed in the shader, after the tests");
    }
}

namespace {

thread_local bool prebuilding = false;
thread_local bool prebuilt = false;

VkShaderStageFlags pushStagesOf(const Context& context) {
    return PreRasterizationPushStages(context.meshShader, context.tessellationShader, context.geometryShader);
}

template<typename TValue>
void appendKey(std::vector<std::byte>& key, const TValue& value) {
    static_assert(std::is_trivially_copyable_v<TValue>);
    const auto bytes = std::as_bytes(std::span(&value, 1));
    key.insert(key.end(), bytes.begin(), bytes.end());
}

VkShaderStageFlags HeapStages(std::span<const CompiledShader> shaders) {
    VkShaderStageFlags stages = 0;
    for (const auto& shader : shaders) stages |= VulkanStage(shader.stage);
    return stages;
}

bool libraryKeys(PipelineLibraryKeys& keys, const Context& context, const State& state, const VertexInputLayout& input, const PipelineSetLayouts& resources, std::span<const CompiledShader> shaders) {
    using Stage = ShaderRecompiler::ShaderStage;
    if (!context.graphicsPipelineLibrary || context.pipelineExecutableInfo || state.depthBiasPerFace || state.rectList || SplitsAlternateAlpha(state)) return false;
    keys = {};
    if (resources.split) keys.preRasterizationSets = keys.fragmentSets = 1u << ShaderRecompiler::RuntimeAbi::HeapDescriptorSet;
    for (const auto& shader : shaders) {
        const auto identity = shader.program->Identity();
        if (shader.stage == Stage::Geometry || identity.Empty()) return false;
        auto& key = shader.stage == Stage::Fragment ? keys.fragmentShader : keys.preRasterization;
        appendKey(key, shader.stage);
        appendKey(key, identity.low);
        appendKey(key, identity.high);
        appendKey(key, shader.pushConstantOffset);
        if (resources.split) {
            auto& sets = shader.stage == Stage::Fragment ? keys.fragmentSets : keys.preRasterizationSets;
            for (const auto& binding : shader.program->bindings) {
                Require(binding.descriptorSet < 2u, "split pipeline library descriptor set exceeds two slots");
                if (binding.kind != ShaderRecompiler::DescriptorKind::SamplerSlot) sets |= 1u << binding.descriptorSet;
            }
        }
    }
    for (const auto& binding : input.bindings) {
        appendKey(keys.vertexInput, binding.binding);
        appendKey(keys.vertexInput, binding.stride);
        appendKey(keys.vertexInput, binding.inputRate);
    }
    appendKey(keys.vertexInput, input.attributes.size());
    for (const auto& attribute : input.attributes) {
        appendKey(keys.vertexInput, attribute.location);
        appendKey(keys.vertexInput, attribute.binding);
        appendKey(keys.vertexInput, attribute.format);
        appendKey(keys.vertexInput, attribute.offset);
    }
    appendKey(keys.vertexInput, state.topology);
    appendKey(keys.vertexInput, state.primitiveRestart);
    appendKey(keys.preRasterization, state.provokingVertexMode);
    appendKey(keys.preRasterization, state.negativeOneToOne);
    appendKey(keys.preRasterization, state.depthClamp && context.depthClamp);
    appendKey(keys.preRasterization, state.conservativeRasterization);
    appendKey(keys.preRasterization, state.depth.has_value() && state.depthBias);
    appendKey(keys.preRasterization, state.stages.tessellation.has_value());
    if (state.stages.tessellation) appendKey(keys.preRasterization, state.stages.tessellation->inputControlPoints);
    appendKey(keys.fragmentShader, state.samples);
    appendKey(keys.fragmentOutput, state.samples);
    appendKey(keys.fragmentShader, state.perSampleShading);
    appendKey(keys.fragmentOutput, state.perSampleShading);
    const auto sampleLocationsEnabled = ValidateSampleLocations(context, state);
    appendKey(keys.fragmentShader, sampleLocationsEnabled);
    appendKey(keys.fragmentOutput, sampleLocationsEnabled);
    appendKey(keys.fragmentShader, state.customSampleLocations);
    appendKey(keys.fragmentOutput, state.customSampleLocations);
    if (state.customSampleLocations) {
        appendKey(keys.fragmentShader, state.sampleLocationGrid);
        appendKey(keys.fragmentOutput, state.sampleLocationGrid);
        const auto grid = SampleLocationGrid(state);
        for (std::uint32_t pixel = 0; pixel < grid.width * grid.height; ++pixel) {
            for (std::uint32_t sample = 0; sample < state.samples; ++sample) {
                appendKey(keys.fragmentShader, state.sampleLocations[pixel * 8u + sample]);
                appendKey(keys.fragmentOutput, state.sampleLocations[pixel * 8u + sample]);
            }
        }
    }
    for (const auto& blend : state.blends) appendKey(keys.fragmentOutput, blend);
    for (const auto value : state.blendConstants) appendKey(keys.fragmentOutput, value);
    appendKey(keys.renderPass, state.blends.size());
    for (const auto& color : state.colors) {
        appendKey(keys.renderPass, color.format);
        appendKey(keys.renderPass, color.exportIndex);
    }
    appendKey(keys.renderPass, state.depth.has_value());
    if (state.depth) appendKey(keys.renderPass, state.depth->format);
    appendKey(keys.fragmentShader, HostSampleMask(state));
    appendKey(keys.fragmentOutput, HostSampleMask(state));
    if (!resources.split) {
        appendKey(keys.layout, pushStagesOf(context));
        appendKey(keys.layout, HeapStages(shaders));
        for (const auto word : resources.mainKey) appendKey(keys.layout, word);
        keys.preRasterizationLayout = keys.fragmentLayout = keys.layout;
        return true;
    }
    const auto splitKey = [&](std::vector<std::byte>& key, std::uint32_t sets) {
        appendKey(key, ~0u);
        appendKey(key, sets);
        appendKey(key, pushStagesOf(context));
        appendKey(key, HeapStages(shaders));
        if ((sets & 1u) != 0u) {
            appendKey(key, resources.mainKey.size());
            for (const auto word : resources.mainKey) appendKey(key, word);
        }
        if ((sets & 2u) != 0u) {
            appendKey(key, resources.fragmentKey.size());
            for (const auto word : resources.fragmentKey) appendKey(key, word);
        }
    };
    splitKey(keys.layout, 7u);
    splitKey(keys.preRasterizationLayout, keys.preRasterizationSets);
    splitKey(keys.fragmentLayout, keys.fragmentSets);
    return true;
}

}

Pipeline::Pipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) : Pipeline(context, state, vertexInput, resources.SetLayouts(), shaders, attachmentLayout) {}

Pipeline::Pipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const PipelineSetLayouts& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) : context(context), _modules(shaders.size()), attachments(state.colors.size() + (state.depth ? 1u : 0u)), colorAttachments(state.colors.size()), depthBounds(state.depth.has_value() && state.depthBoundsTest), depthBias(state.depth.has_value() && state.depthBias), dynamicRendering(context.graphicsPipelineLibrary) {
    PerformanceTimer timing("Vulkan.GraphicsPipeline");
    sampleLocationsEnabled = ValidateSampleLocations(context, state);
    // A cached pipeline may outlive its device's teardown (see ClearCachedPipelines); it must not keep
    // the buffer pool, which is reset with the device, alive past it.
    this->context.bufferPool.reset();
    Require(state.blends.size() == (state.colors.empty() ? 0u : state.colors.back().exportIndex + 1u) && state.colors.size() <= state.blends.size(), "blend states do not match decoded color state");
    Require(state.blends.size() <= context.limits.maxColorAttachments, "color targets exceed device attachment limits");
    Require(state.hasColorTarget || (context.limits.framebufferNoAttachmentsSampleCounts & static_cast<VkSampleCountFlagBits>(state.samples)) != 0, "device does not support rendering without attachments at the draw's sample count");
    Require(state.samples == 1 || !state.depth.has_value() || (context.limits.framebufferDepthSampleCounts & static_cast<VkSampleCountFlagBits>(state.samples)) != 0, "device does not support depth targets with the draw's sample count");
    Require(!depthBounds || context.depthBounds, "device does not support the depth bounds test");
    Require(!depthBias || state.depthBiasClamp == 0.0f || context.depthBiasClamp, "device does not support depth bias clamping");
    Require(!state.dualSourceBlend || (context.dualSrcBlend && state.blends.size() <= context.limits.maxFragmentDualSrcAttachments), "device does not support dual-source blending into this many attachments");
    Require(state.reversedBlend == 0 || (context.dualSrcBlend && state.blends.size() <= context.limits.maxFragmentDualSrcAttachments), "blending into a color target with a reversed component order through a second source color needs dual-source blending, which the device does not support");
    Require(!state.negativeOneToOne || context.depthClipControl, "negative-one-to-one depth clipping requires VK_EXT_depth_clip_control with depthClipControl enabled");
    Require(!state.perSampleShading || context.sampleRateShading, "per-sample pixel shading requires the sampleRateShading device feature");
    Require(state.conservativeRasterization == VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT || context.conservativeRasterization, "conservative rasterization requires VK_EXT_conservative_rasterization with at most 1/256 pixel of overestimation and degenerate triangles rasterized");
    ValidateProvokingVertex(context, state, shaders);
    if (state.rectList) Require(context.tessellationShader && context.limits.maxTessellationPatchSize >= 4, "rect-list requires tessellation with four output control points");
    if (std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.stage == ShaderRecompiler::ShaderStage::Geometry; })) Require(context.geometryShader, "device does not support geometry shaders");
    if (state.stages.tessellation) {
        Require(context.tessellationShader, "device does not support tessellation shaders");
        Require(state.stages.tessellation->inputControlPoints <= context.limits.maxTessellationPatchSize && state.stages.tessellation->outputControlPoints <= context.limits.maxTessellationPatchSize, "tessellation patch exceeds device limits");
    }
    if (state.stages.mesh) {
        Require(context.meshShader, "device does not support VK_EXT_mesh_shader");
        const auto& mesh = *state.stages.mesh;
        const auto invocations = state.stages.vertexWaveSize == 64u && context.subgroup.subgroupSize == 32u ? mesh.threadsPerGroup / 2u : mesh.threadsPerGroup;
        Require(invocations <= context.meshLimits.maxMeshWorkGroupInvocations && invocations <= context.meshLimits.maxMeshWorkGroupSize[0], "mesh workgroup exceeds device limits");
        Require(mesh.maxVertices <= context.meshLimits.maxMeshOutputVertices && mesh.maxPrimitives <= context.meshLimits.maxMeshOutputPrimitives && static_cast<std::uint64_t>(mesh.ldsSizeDwords) * 4 <= context.meshLimits.maxMeshSharedMemorySize, "mesh output or LDS exceeds device limits");
    }
    const auto pushStages = PushConstantStages(shaders);
    Require(pushStages == 0 || context.limits.maxPushConstantsSize >= PushBlockBytes(dynamicRendering), "graphics push constant range exceeds device limit");
    if (dynamicRendering) {
        for (const auto& shader : shaders) Require(shader.program->pushConstants.empty() || (shader.pushConstantOffset >= PipelinePushSlotBytes) == (shader.stage == ShaderRecompiler::ShaderStage::Fragment), "a stage's push constants lie outside its push constant slot");
    }
    try {
        std::vector<VkPipelineShaderStageCreateInfo> stages(shaders.size());
        std::vector<PipelineSpecialization> specializations;
        specializations.reserve(shaders.size());
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT meshSubgroup{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT, nullptr, 32u};
        for (std::uint32_t i = 0; i < shaders.size(); ++i) {
            const auto& shader = *shaders[i].program;
            VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            module.codeSize = shader.spirv.size() * sizeof(std::uint32_t);
            module.pCode = shader.spirv.data();
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &module, nullptr, &_modules[i]), "vkCreateShaderModule graphics");
            const auto stage = VulkanStage(shaders[i].stage);
            stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[i].stage = stage;
            stages[i].module = _modules[i];
            stages[i].pName = "main";
            if (shaders[i].stage == ShaderRecompiler::ShaderStage::Mesh && context.meshWave32 && shader.hostSubgroupSize == 32u) stages[i].pNext = &meshSubgroup;
            specializations.emplace_back(shader);
            stages[i].pSpecializationInfo = specializations.back().Info();
        }
        if (std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && !shader.program->stencilReferences.empty(); })) {
            for (std::size_t i = 0; i < shaders.size(); ++i) {
                if (shaders[i].stage != ShaderRecompiler::ShaderStage::Vertex) continue;
                Require(stages[i].flags == 0u && stages[i].pNext == nullptr, "stencil replay vertex stage allows varying subgroup size");
            }
        }
        // A descriptor set layout with the same bindings as this one is compatible with the pipeline
        // layout, so later draws bind their own ShaderResources' set under it.
        heapSet = context.descriptorHeap->Set(HeapStages(shaders));
        const std::array<VkDescriptorSetLayout, 3> setLayouts{resources.main, resources.split ? resources.fragment : context.descriptorHeap->EmptyLayout(), context.descriptorHeap->Layout(HeapStages(shaders))};
        const VkPushConstantRange push{pushStages, 0, PipelinePushSlotBytes};
        const std::array<VkPushConstantRange, 2> slots{{{pushStagesOf(context), 0, PipelinePushSlotBytes}, {VK_SHADER_STAGE_FRAGMENT_BIT, PipelinePushSlotBytes, PipelinePushSlotBytes}}};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.flags = resources.split ? VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT : 0;
        layoutInfo.setLayoutCount = static_cast<std::uint32_t>(setLayouts.size());
        layoutInfo.pSetLayouts = setLayouts.data();
        layoutInfo.pushConstantRangeCount = dynamicRendering ? static_cast<std::uint32_t>(slots.size()) : pushStages != 0 ? 1 : 0;
        layoutInfo.pPushConstantRanges = dynamicRendering ? slots.data() : pushStages != 0 ? &push : nullptr;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout graphics");
        std::vector<VkAttachmentDescription> colors;
        std::vector<VkAttachmentReference> references(state.blends.size(), VkAttachmentReference{VK_ATTACHMENT_UNUSED, attachmentLayout});
        for (std::uint32_t index = 0; index < state.colors.size(); ++index) {
            VkAttachmentDescription color{};
            color.format = state.colors[index].format;
            color.samples = static_cast<VkSampleCountFlagBits>(state.samples);
            color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            color.initialLayout = attachmentLayout;
            color.finalLayout = attachmentLayout;
            colors.push_back(color);
            references.at(state.colors[index].exportIndex) = {index, attachmentLayout};
        }
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<std::uint32_t>(references.size());
        subpass.pColorAttachments = references.empty() ? nullptr : references.data();
        const VkAttachmentReference depthReference{static_cast<std::uint32_t>(colors.size()), VK_IMAGE_LAYOUT_GENERAL};
        if (state.depth) {
            VkAttachmentDescription depth{};
            depth.format = HostDepthFormat(context, state.depth->format);
            depth.samples = static_cast<VkSampleCountFlagBits>(state.samples);
            depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
            depth.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
            colors.push_back(depth);
            subpass.pDepthStencilAttachment = &depthReference;
        }
        std::vector<VkFormat> renderingColors(state.blends.size(), VK_FORMAT_UNDEFINED);
        for (const auto& color : state.colors) renderingColors.at(color.exportIndex) = color.format;
        VkPipelineRenderingCreateInfoKHR rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
        rendering.colorAttachmentCount = static_cast<std::uint32_t>(renderingColors.size());
        rendering.pColorAttachmentFormats = renderingColors.empty() ? nullptr : renderingColors.data();
        if (state.depth) {
            rendering.depthAttachmentFormat = HostDepthFormat(context, state.depth->format);
            if (HasStencil(state.depth->format)) rendering.stencilAttachmentFormat = rendering.depthAttachmentFormat;
        }
        if (dynamicRendering) {
            depthFormat = state.depth ? HostDepthFormat(context, state.depth->format) : VK_FORMAT_UNDEFINED;
            colorSlots.assign(state.blends.size(), ~0u);
            for (std::uint32_t index = 0; index < state.colors.size(); ++index) colorSlots.at(state.colors[index].exportIndex) = index;
            this->attachmentLayout = attachmentLayout;
        }
        VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        passInfo.attachmentCount = static_cast<std::uint32_t>(colors.size());
        passInfo.pAttachments = colors.empty() ? nullptr : colors.data();
        passInfo.subpassCount = 1;
        passInfo.pSubpasses = &subpass;
        if (!dynamicRendering) Check(context.Function<PFN_vkCreateRenderPass>("vkCreateRenderPass")(context.device, &passInfo, nullptr, &renderPass), "vkCreateRenderPass");
        VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        input.vertexBindingDescriptionCount = static_cast<std::uint32_t>(vertexInput.bindings.size());
        input.pVertexBindingDescriptions = vertexInput.bindings.data();
        input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(vertexInput.attributes.size());
        input.pVertexAttributeDescriptions = vertexInput.attributes.data();
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = state.topology;
        const bool listTopology = state.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        assembly.primitiveRestartEnable = state.primitiveRestart && (!listTopology || context.primitiveListRestart) ? VK_TRUE : VK_FALSE;
        // Viewport and scissor are set per draw (Begin), so they do not multiply pipelines; the depth
        // clip control stays baked in.
        VkPipelineViewportStateCreateInfo viewports{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        VkPipelineViewportDepthClipControlCreateInfoEXT depthClip{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
        depthClip.negativeOneToOne = state.negativeOneToOne;
        if (state.negativeOneToOne) viewports.pNext = &depthClip;
        viewports.viewportCount = 1;
        viewports.scissorCount = 1;
        PipelineLibraryKeys keys;
        libraries = libraryKeys(keys, context, state, vertexInput, resources, shaders);
        std::vector<VkDynamicState> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        if (state.stencilReplayReference) dynamicStates.push_back(VK_DYNAMIC_STATE_STENCIL_REFERENCE);
        if (depthBounds) dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
        if (depthBias) dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
        if (libraries) dynamicStates.assign(PipelineLibraryDynamicStates().begin(), PipelineLibraryDynamicStates().end());
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.depthClampEnable = state.depthClamp && context.depthClamp ? VK_TRUE : VK_FALSE;
        Require(!state.depthBiasPerFace || (depthBias && state.cullMode == VK_CULL_MODE_NONE), "per-face depth bias needs both faces rasterized");
        raster.cullMode = state.depthBiasPerFace ? VK_CULL_MODE_BACK_BIT : state.cullMode;
        raster.frontFace = state.frontFace;
        raster.depthBiasEnable = depthBias;
        raster.lineWidth = 1;
        VkPipelineRasterizationConservativeStateCreateInfoEXT conservative{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT};
        conservative.conservativeRasterizationMode = state.conservativeRasterization;
        if (state.conservativeRasterization != VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT) raster.pNext = &conservative;
        VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT};
        provoking.provokingVertexMode = state.provokingVertexMode;
        if (context.provokingVertexLast) {
            provoking.pNext = raster.pNext;
            raster.pNext = &provoking;
        }
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        samples.rasterizationSamples = static_cast<VkSampleCountFlagBits>(state.samples);
        samples.sampleShadingEnable = state.perSampleShading ? VK_TRUE : VK_FALSE;
        samples.minSampleShading = state.perSampleShading ? 1.0f : 0.0f;
        const auto locations = SampleLocations(state);
        VkPipelineSampleLocationsStateCreateInfoEXT sampleLocations{VK_STRUCTURE_TYPE_PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO_EXT};
        sampleLocations.sampleLocationsEnable = VK_TRUE;
        const auto grid = SampleLocationGrid(state);
        sampleLocations.sampleLocationsInfo = {VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT, nullptr, samples.rasterizationSamples, grid, state.samples * grid.width * grid.height, locations.data()};
        if (sampleLocationsEnabled) samples.pNext = &sampleLocations;
        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = state.depthTest;
        depthStencil.depthWriteEnable = state.depthWrite;
        depthStencil.depthCompareOp = state.depthCompare;
        depthStencil.depthBoundsTestEnable = depthBounds;
        depthStencil.stencilTestEnable = state.stencilTest;
        depthStencil.front = state.stencilFront;
        depthStencil.back = state.stencilBack;
        const VkSampleMask sampleMask = HostSampleMask(state);
        if (PartialSampleMask(state) && UniformSampleMask(state)) samples.pSampleMask = &sampleMask;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        const bool alternateAlpha = SplitsAlternateAlpha(state);
        std::vector<VkPipelineColorBlendAttachmentState> firstChannels(state.blends.begin(), state.blends.end());
        std::vector<VkPipelineColorBlendAttachmentState> secondChannels(state.blends.begin(), state.blends.end());
        if (alternateAlpha) {
            const auto keepsStencil = [](const VkStencilOpState& face) { return face.writeMask == 0 || (face.failOp == VK_STENCIL_OP_KEEP && face.passOp == VK_STENCIL_OP_KEEP && face.depthFailOp == VK_STENCIL_OP_KEEP); };
            Require(!(state.depth && (state.depthWrite || (state.stencilTest && (!keepsStencil(state.stencilFront) || !keepsStencil(state.stencilBack))))) && !state.depthBiasPerFace, "blending into an 8_8 color target with the alternate component order and a separate alpha blend from a draw that writes depth or stencil is not implemented");
            Require(!ShadersWriteMemory(shaders), "blending into an 8_8 color target with the alternate component order and a separate alpha blend from a draw whose shaders write memory is not implemented");
            for (std::size_t index = 0; index < secondChannels.size(); ++index) {
                const bool split = index < state.colors.size() && state.colors[index].componentMapping == 0xecu && state.blends[index].blendEnable;
                if (!split) {
                    secondChannels[index].colorWriteMask = 0;
                    continue;
                }
                firstChannels[index].colorWriteMask &= ~VK_COLOR_COMPONENT_G_BIT;
                secondChannels[index].colorWriteMask &= VK_COLOR_COMPONENT_G_BIT;
                secondChannels[index].srcColorBlendFactor = state.blends[index].srcAlphaBlendFactor;
                secondChannels[index].dstColorBlendFactor = state.blends[index].dstAlphaBlendFactor;
                secondChannels[index].colorBlendOp = state.blends[index].alphaBlendOp;
            }
        }
        blend.attachmentCount = static_cast<std::uint32_t>(firstChannels.size());
        blend.pAttachments = firstChannels.empty() ? nullptr : firstChannels.data();
        std::copy(state.blendConstants.begin(), state.blendConstants.end(), blend.blendConstants);
        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.flags = context.pipelineExecutableInfo ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        VkPipelineTessellationStateCreateInfo tessellation{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
        if (state.rectList || state.stages.tessellation) {
            tessellation.patchControlPoints = state.rectList ? 3u : state.stages.tessellation->inputControlPoints;
            pipelineInfo.pTessellationState = &tessellation;
        }
        pipelineInfo.pVertexInputState = state.stages.mesh ? nullptr : &input;
        pipelineInfo.pInputAssemblyState = state.stages.mesh ? nullptr : &assembly;
        pipelineInfo.pViewportState = &viewports;
        pipelineInfo.pRasterizationState = &raster;
        pipelineInfo.pMultisampleState = &samples;
        pipelineInfo.pDepthStencilState = state.depth ? &depthStencil : nullptr;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = layout;
        pipelineInfo.renderPass = renderPass;
        if (dynamicRendering) pipelineInfo.pNext = &rendering;
        timing.Mark("modules_and_state");
        if (prebuilding && !libraries) {
            for (const auto module : _modules) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            _modules.clear();
            return;
        }
        std::vector<std::size_t> stageWords;
        stageWords.reserve(shaders.size());
        for (const auto& shader : shaders) stageWords.push_back(shader.program->spirv.size());
        if (prebuilding) {
            static_cast<void>(LinkPipelineFromLibraries(context, pipelineInfo, rendering, layoutInfo, keys, &optimized, true, _modules, stageWords));
            for (const auto module : _modules) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            _modules.clear();
            prebuilt = true;
            return;
        }
        if (libraries) pipeline = LinkPipelineFromLibraries(context, pipelineInfo, rendering, layoutInfo, keys, &optimized, false, _modules, stageWords);
        else Check(CreateGraphicsPipeline(context, pipelineInfo, &pipeline), "vkCreateGraphicsPipelines");
        if (alternateAlpha) {
            blend.pAttachments = secondChannels.data();
            Check(CreateGraphicsPipeline(context, pipelineInfo, &secondChannel), "vkCreateGraphicsPipelines");
        }
        if (state.depthBiasPerFace) {
            raster.cullMode = VK_CULL_MODE_FRONT_BIT;
            Check(CreateGraphicsPipeline(context, pipelineInfo, &backFaces), "vkCreateGraphicsPipelines");
        }
        for (const auto module : _modules) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        _modules.clear();
        timing.Mark("create");
        LogPipelineStatistics_nid_no_patch(context, pipeline);
    } catch (...) {
        release();
        throw;
    }
}

bool Pipeline::Prebuild(const Context& context, const State& state, const VertexInputLayout& vertexInput, const PipelineSetLayouts& resources, std::span<const CompiledShader> shaders) {
    if (std::ranges::any_of(shaders, [](const CompiledShader& shader) { return !shader.program->stencilReferences.empty(); })) return false;
    PipelineLibraryKeys keys;
    if (!libraryKeys(keys, context, state, vertexInput, resources, shaders)) return false;
    const auto depthFormat = state.depth ? HostDepthFormat(context, state.depth->format) : VK_FORMAT_UNDEFINED;
    if (PipelineLibraryShadersBuilt(context, keys, depthFormat, state.depth && HasStencil(state.depth->format) ? depthFormat : VK_FORMAT_UNDEFINED)) return true;
    struct Scope {
        Scope() { prebuilding = true; prebuilt = false; }
        ~Scope() { prebuilding = false; }
    } scope;
    const Pipeline pipeline(context, state, vertexInput, resources, shaders);
    return prebuilt;
}

Pipeline::~Pipeline() {
    release();
}

void Pipeline::release() noexcept {
    framebuffers.clear();
    if (optimized != nullptr) optimized->Release();
    optimized.reset();
    if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
    if (backFaces) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, backFaces, nullptr);
    if (secondChannel) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, secondChannel, nullptr);
    if (renderPass) context.Function<PFN_vkDestroyRenderPass>("vkDestroyRenderPass")(context.device, renderPass, nullptr);
    if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
    for (auto module : _modules) {
        if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
    }
    pipeline = VK_NULL_HANDLE;
    backFaces = VK_NULL_HANDLE;
    secondChannel = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    _modules.clear();
}

void Pipeline::Abandon() noexcept {
    if (optimized != nullptr) optimized->released.store(true);
    optimized.reset();
    for (auto& entry : framebuffers) entry.framebuffer->Abandon();
    framebuffers.clear();
    pipeline = VK_NULL_HANDLE;
    backFaces = VK_NULL_HANDLE;
    secondChannel = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    _modules.clear();
}

void Framebuffer::Abandon() noexcept {
    framebuffer = VK_NULL_HANDLE;
}

VkPipelineLayout Pipeline::Layout() const {
    return layout;
}

void Pipeline::BindHeap(VkCommandBuffer commands) const {
    context.descriptorHeap->Bind(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, heapSet);
}

std::shared_ptr<Framebuffer> Pipeline::AcquireFramebuffer(std::span<const VkImageView> targets, std::span<const std::shared_ptr<StorageTexture>> owners, VkExtent2D extent, std::uint32_t layers, std::span<const std::uint8_t> persistent) {
    Require(targets.size() == attachments && owners.size() == colorAttachments, "render targets do not match the pipeline's attachments");
    bool resident = true;
    for (std::size_t i = 0; i < owners.size(); ++i) resident = resident && (owners[i] != nullptr || (i < persistent.size() && persistent[i] != 0));
    if (resident) {
        // Entries whose views are gone can never match again and go as soon as no recorded draw holds
        // them (Kept keeps its framebuffer until the batch completes).
        std::erase_if(framebuffers, [](const CachedFramebuffer& entry) {
            if (entry.framebuffer.use_count() != 1) return false;
            for (std::size_t i = 0; i < entry.owners.size(); ++i) {
                if (entry.owned[i] && entry.owners[i].expired()) return true;
            }
            return false;
        });
        for (auto it = framebuffers.begin(); it != framebuffers.end(); ++it) {
            if (it->extent.width != extent.width || it->extent.height != extent.height || it->layers != layers || !std::equal(it->views.begin(), it->views.end(), targets.begin(), targets.end())) continue;
            // View handles are recycled once a StorageTexture is destroyed, so the owners must be the
            // very objects the views were made for.
            bool same = true;
            for (std::size_t i = 0; i < owners.size() && same; ++i) same = it->owned[i] == (owners[i] != nullptr) && it->owners[i].lock().get() == owners[i].get();
            if (!same) continue;
            std::rotate(it, std::next(it), framebuffers.end());
            return framebuffers.back().framebuffer;
        }
    }
    auto framebuffer = std::make_shared<Framebuffer>(context, renderPass, targets, extent, layers);
    if (!resident) return framebuffer;
    // Beyond the bound the least recently used unreferenced entry goes.
    constexpr std::size_t bound = 8;
    while (framebuffers.size() >= bound) {
        const auto victim = std::find_if(framebuffers.begin(), framebuffers.end(), [](const CachedFramebuffer& entry) { return entry.framebuffer.use_count() == 1; });
        if (victim == framebuffers.end()) break;
        framebuffers.erase(victim);
    }
    CachedFramebuffer entry;
    entry.views.assign(targets.begin(), targets.end());
    entry.owners.assign(owners.begin(), owners.end());
    for (const auto& owner : owners) entry.owned.push_back(owner != nullptr);
    entry.extent = extent;
    entry.layers = layers;
    entry.framebuffer = framebuffer;
    framebuffers.push_back(std::move(entry));
    return framebuffer;
}

void Pipeline::Begin(VkCommandBuffer commands, const Framebuffer& framebuffer, VkExtent2D extent, const State& state) const {
    if (dynamicRendering) {
        const auto views = framebuffer.Views();
        Require(views.size() == attachments, "render targets do not match the pipeline's attachments");
        std::vector<VkRenderingAttachmentInfoKHR> colors(colorSlots.size(), VkRenderingAttachmentInfoKHR{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR});
        for (std::size_t slot = 0; slot < colorSlots.size(); ++slot) {
            if (colorSlots[slot] == ~0u) continue;
            colors[slot].imageView = views[colorSlots[slot]];
            colors[slot].imageLayout = attachmentLayout;
            colors[slot].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            colors[slot].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }
        VkRenderingAttachmentInfoKHR depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
        if (depthFormat != VK_FORMAT_UNDEFINED) {
            depth.imageView = views.back();
            depth.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }
        VkRenderingInfoKHR rendering{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
        rendering.renderArea = {{0, 0}, extent};
        rendering.layerCount = state.renderLayers;
        rendering.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
        rendering.pColorAttachments = colors.empty() ? nullptr : colors.data();
        if (depthFormat != VK_FORMAT_UNDEFINED) rendering.pDepthAttachment = &depth;
        if (depthFormat != VK_FORMAT_UNDEFINED && HasStencil(depthFormat)) rendering.pStencilAttachment = &depth;
        context.Resolved(&DeviceFunctions::cmdBeginRendering, "vkCmdBeginRenderingKHR")(commands, &rendering);
        Continue(commands, state);
        return;
    }
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = renderPass;
    begin.framebuffer = framebuffer.Handle();
    begin.renderArea = {{0, 0}, extent};
    const auto locations = SampleLocations(state);
    const auto grid = SampleLocationGrid(state);
    const VkSampleLocationsInfoEXT info{VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT, nullptr, static_cast<VkSampleCountFlagBits>(state.samples), grid, state.samples * grid.width * grid.height, locations.data()};
    const VkAttachmentSampleLocationsEXT initial{static_cast<std::uint32_t>(state.colors.size()), info};
    const VkSubpassSampleLocationsEXT post{0, info};
    VkRenderPassSampleLocationsBeginInfoEXT sampleLocations{VK_STRUCTURE_TYPE_RENDER_PASS_SAMPLE_LOCATIONS_BEGIN_INFO_EXT};
    sampleLocations.attachmentInitialSampleLocationsCount = state.depth ? 1 : 0;
    sampleLocations.pAttachmentInitialSampleLocations = state.depth ? &initial : nullptr;
    sampleLocations.postSubpassSampleLocationsCount = 1;
    sampleLocations.pPostSubpassSampleLocations = &post;
    if (sampleLocationsEnabled) begin.pNext = &sampleLocations;
    context.Resolved(&DeviceFunctions::cmdBeginRenderPass, "vkCmdBeginRenderPass")(commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
    Continue(commands, state);
}

void Pipeline::Continue(VkCommandBuffer commands, const State& state) const {
    const auto best = optimized != nullptr ? optimized->handle.load() : VK_NULL_HANDLE;
    context.Resolved(&DeviceFunctions::cmdBindPipeline, "vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, best != VK_NULL_HANDLE ? best : pipeline);
    context.Resolved(&DeviceFunctions::cmdSetViewport, "vkCmdSetViewport")(commands, 0, 1, &state.viewport);
    context.Resolved(&DeviceFunctions::cmdSetScissor, "vkCmdSetScissor")(commands, 0, 1, &state.scissor);
    if (state.stencilReplayReference) context.Resolved(&DeviceFunctions::cmdSetStencilReference, "vkCmdSetStencilReference")(commands, VK_STENCIL_FACE_FRONT_AND_BACK, *state.stencilReplayReference);
    if (libraries) {
        setLibraryState(commands, state);
        return;
    }
    if (depthBias) context.Resolved(&DeviceFunctions::cmdSetDepthBias, "vkCmdSetDepthBias")(commands, HostDepthBiasConstant(context, state.depth->format, state.depthBiasConstant), state.depthBiasClamp, state.depthBiasSlope);
    if (!depthBounds) return;
    context.Resolved(&DeviceFunctions::cmdSetDepthBounds, "vkCmdSetDepthBounds")(commands, state.minDepthBounds, state.maxDepthBounds);
}

void Pipeline::setLibraryState(VkCommandBuffer commands, const State& state) const {
    const bool depth = state.depth.has_value();
    context.Resolved(&DeviceFunctions::cmdSetCullMode, "vkCmdSetCullModeEXT")(commands, state.cullMode);
    context.Resolved(&DeviceFunctions::cmdSetFrontFace, "vkCmdSetFrontFaceEXT")(commands, state.frontFace);
    context.Resolved(&DeviceFunctions::cmdSetDepthBias, "vkCmdSetDepthBias")(commands, depthBias ? HostDepthBiasConstant(context, state.depth->format, state.depthBiasConstant) : 0.0f, depthBias ? state.depthBiasClamp : 0.0f, depthBias ? state.depthBiasSlope : 0.0f);
    context.Resolved(&DeviceFunctions::cmdSetDepthBounds, "vkCmdSetDepthBounds")(commands, depthBounds ? state.minDepthBounds : 0.0f, depthBounds ? state.maxDepthBounds : 1.0f);
    context.Resolved(&DeviceFunctions::cmdSetDepthTestEnable, "vkCmdSetDepthTestEnableEXT")(commands, depth && state.depthTest);
    context.Resolved(&DeviceFunctions::cmdSetDepthWriteEnable, "vkCmdSetDepthWriteEnableEXT")(commands, depth && state.depthWrite);
    context.Resolved(&DeviceFunctions::cmdSetDepthCompareOp, "vkCmdSetDepthCompareOpEXT")(commands, depth ? state.depthCompare : VK_COMPARE_OP_ALWAYS);
    context.Resolved(&DeviceFunctions::cmdSetDepthBoundsTestEnable, "vkCmdSetDepthBoundsTestEnableEXT")(commands, depthBounds);
    context.Resolved(&DeviceFunctions::cmdSetStencilTestEnable, "vkCmdSetStencilTestEnableEXT")(commands, depth && state.stencilTest);
    const auto stencil = [&](VkStencilFaceFlags face, const VkStencilOpState& op) {
        context.Resolved(&DeviceFunctions::cmdSetStencilOp, "vkCmdSetStencilOpEXT")(commands, face, op.failOp, op.passOp, op.depthFailOp, op.compareOp);
        context.Resolved(&DeviceFunctions::cmdSetStencilCompareMask, "vkCmdSetStencilCompareMask")(commands, face, op.compareMask);
        context.Resolved(&DeviceFunctions::cmdSetStencilWriteMask, "vkCmdSetStencilWriteMask")(commands, face, op.writeMask);
        context.Resolved(&DeviceFunctions::cmdSetStencilReference, "vkCmdSetStencilReference")(commands, face, op.reference);
    };
    stencil(VK_STENCIL_FACE_FRONT_BIT, depth ? state.stencilFront : VkStencilOpState{});
    stencil(VK_STENCIL_FACE_BACK_BIT, depth ? state.stencilBack : VkStencilOpState{});
}

void Pipeline::ContinueSecondChannel(VkCommandBuffer commands) const {
    Require(secondChannel != VK_NULL_HANDLE, "the pipeline blends every channel in one pass");
    context.Resolved(&DeviceFunctions::cmdBindPipeline, "vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, secondChannel);
}

void Pipeline::ContinueBackFaces(VkCommandBuffer commands, const State& state) const {
    Require(backFaces != VK_NULL_HANDLE, "the pipeline draws both faces in one pass");
    context.Resolved(&DeviceFunctions::cmdBindPipeline, "vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, backFaces);
    context.Resolved(&DeviceFunctions::cmdSetViewport, "vkCmdSetViewport")(commands, 0, 1, &state.viewport);
    context.Resolved(&DeviceFunctions::cmdSetScissor, "vkCmdSetScissor")(commands, 0, 1, &state.scissor);
    context.Resolved(&DeviceFunctions::cmdSetDepthBias, "vkCmdSetDepthBias")(commands, HostDepthBiasConstant(context, state.depth->format, state.backDepthBiasConstant), state.depthBiasClamp, state.backDepthBiasSlope);
    if (depthBounds) context.Resolved(&DeviceFunctions::cmdSetDepthBounds, "vkCmdSetDepthBounds")(commands, state.minDepthBounds, state.maxDepthBounds);
}

void Pipeline::PushConstants(VkCommandBuffer commands, VkShaderStageFlags stages, std::span<const std::byte, PipelinePushConstantBytes> bytes) const {
    if (stages == 0) return;
    if (dynamicRendering) {
        context.Resolved(&DeviceFunctions::cmdPushConstants, "vkCmdPushConstants")(commands, layout, pushStagesOf(context), 0, PipelinePushSlotBytes, bytes.data());
        context.Resolved(&DeviceFunctions::cmdPushConstants, "vkCmdPushConstants")(commands, layout, VK_SHADER_STAGE_FRAGMENT_BIT, PipelinePushSlotBytes, PipelinePushSlotBytes, bytes.data() + PipelinePushSlotBytes);
        return;
    }
    context.Resolved(&DeviceFunctions::cmdPushConstants, "vkCmdPushConstants")(commands, layout, stages, 0, PipelinePushSlotBytes, bytes.data());
}

namespace {

template<typename TValue>
void append(std::vector<std::byte>& key, const TValue& value) {
    static_assert(std::is_trivially_copyable_v<TValue>);
    const auto bytes = std::as_bytes(std::span(&value, 1));
    key.insert(key.end(), bytes.begin(), bytes.end());
}

// Everything the Pipeline objects are built from, or empty when a stage's result has no module
// (nothing else may share its pipeline). The stage identities and the state fields are the same on
// every run (CanonicalPipelineState keeps exactly the fields read here), which lets the journal's
// replay insert an entry a later draw finds; a generated stage (rect-list pair, geometry) is keyed
// by its identity like the others, so a replayed pipeline built from another device's generated
// module is missed rather than used.
void pipelineKey(std::vector<std::byte>& key, const Context& context, const State& state, const VertexInputLayout& input, const PipelineSetLayouts& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) {
    key.clear();
    append(key, context.device);
    append(key, attachmentLayout);
    append(key, shaders.size());
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        const auto identity = shader.program->Identity();
        if (identity.Empty()) {
            key.clear();
            return;
        }
        append(key, shader.stage);
        append(key, identity.low);
        append(key, identity.high);
        // Where the stage's push constants sit in the block (AssemblePushConstants).
        append(key, shader.pushConstantOffset);
    }
    append(key, PushConstantStages(shaders));
    append(key, input.bindings.size());
    for (const auto& binding : input.bindings) {
        append(key, binding.binding);
        append(key, binding.stride);
        append(key, binding.inputRate);
    }
    append(key, input.attributes.size());
    for (const auto& attribute : input.attributes) {
        append(key, attribute.location);
        append(key, attribute.binding);
        append(key, attribute.format);
        append(key, attribute.offset);
    }
    append(key, resources.mainKey.size());
    for (const auto word : resources.mainKey) append(key, word);
    append(key, resources.fragmentKey.size());
    for (const auto word : resources.fragmentKey) append(key, word);
    append(key, state.hasColorTarget);
    append(key, state.rectList);
    append(key, state.topology);
    append(key, state.primitiveRestart);
    append(key, state.cullMode);
    append(key, state.frontFace);
    append(key, state.provokingVertexMode);
    append(key, state.negativeOneToOne);
    append(key, state.depthClamp && context.depthClamp);
    append(key, state.conservativeRasterization);
    append(key, state.blends.size());
    for (const auto& blend : state.blends) append(key, blend);
    for (const auto value : state.blendConstants) append(key, value);
    append(key, state.samples);
    append(key, state.perSampleShading);
    append(key, state.customSampleLocations);
    if (state.customSampleLocations) {
        append(key, state.sampleLocationGrid);
        append(key, state.sampleLocations);
    }
    append(key, state.colors.size());
    for (const auto& color : state.colors) {
        append(key, color.format);
        append(key, color.componentMapping);
    }
    if (state.blends.size() != state.colors.size()) {
        for (const auto& color : state.colors) append(key, color.exportIndex);
    }
    append(key, state.depth.has_value());
    if (state.depth) {
        append(key, state.depth->format);
        append(key, state.depth->samples);
        append(key, state.depthTest);
        append(key, state.depthWrite);
        append(key, state.depthCompare);
        append(key, state.depthBoundsTest);
        append(key, state.depthBias);
        append(key, state.depthBias && state.depthBiasClamp != 0.0f);
        append(key, state.depthBiasPerFace);
        append(key, state.stencilTest);
        auto front = state.stencilFront;
        auto back = state.stencilBack;
        if (state.stencilReplayReference) front.reference = back.reference = 0u;
        append(key, state.stencilReplayReference.has_value());
        append(key, front);
        append(key, back);
    }
    append(key, state.stages.mesh.has_value());
    if (state.stages.mesh) {
        const auto& mesh = *state.stages.mesh;
        append(key, mesh.inputPrimitive);
        append(key, mesh.primitivesPerGroup);
        append(key, mesh.verticesPerGroup);
        append(key, mesh.maxVertices);
        append(key, mesh.maxPrimitives);
        append(key, mesh.threadsPerGroup);
        append(key, mesh.ldsSizeDwords);
        append(key, mesh.provokingVertex);
        append(key, mesh.outputPrimitiveSize);
    }
    append(key, state.stages.tessellation.has_value());
    if (state.stages.tessellation) {
        const auto& tessellation = *state.stages.tessellation;
        append(key, tessellation.inputControlPoints);
        append(key, tessellation.outputControlPoints);
        append(key, tessellation.domain);
        append(key, tessellation.partitioning);
        append(key, tessellation.outputTopology);
        append(key, tessellation.patchesPerGroup);
        append(key, tessellation.primitiveGroupSize);
        append(key, tessellation.breakAtInstance);
    }
    append(key, HostSampleMask(state));
}

std::uint64_t hashKey(const std::vector<std::byte>& key) {
    return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(key.data()), key.size()));
}

struct PipelineStore {
    struct Entry {
        VkDevice device;
        // The device's buffer pool at insertion: it is made and reset with the device, so it tells
        // the device instance apart from a later one the loader gave the same handle value.
        std::weak_ptr<BufferPool> pool;
        std::uint64_t hash;
        std::vector<std::byte> key;
        std::shared_ptr<Pipeline> pipeline;
        std::shared_ptr<RecordUse> use;
    };
    struct Building {
        std::vector<std::byte> key;
        std::shared_future<std::shared_ptr<Pipeline>> pipeline;
    };
    std::mutex mutex;
    // Least recently used first.
    std::list<Entry> entries;
    std::unordered_map<std::uint64_t, std::list<Entry>::iterator> index;
    std::unordered_map<std::uint64_t, Building> building;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t uncached = 0;
    std::uint64_t evicted = 0;
    PipelineStoreCounters totals;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

// Never destroyed: the pipelines belong to a device that may already be gone when statics die, and
// the device's teardown (ClearCachedPipelines) is the place to destroy them.
PipelineStore& Pipelines() {
    static auto* store = new PipelineStore();
    return *store;
}

// Whether the entry's objects belong to the device the context names. A context without a buffer
// pool (tests) is identified by the handle alone.
bool alive(const PipelineStore::Entry& entry, const Context& context) {
    if (entry.device != context.device) return false;
    return context.bufferPool == nullptr || entry.pool.lock() == context.bufferPool;
}

// Drops an entry of a device that is gone: its objects went with the device, so they are forgotten,
// not destroyed.
std::list<PipelineStore::Entry>::iterator abandon(PipelineStore& store, std::list<PipelineStore::Entry>::iterator it) {
    it->pipeline->Abandon();
    store.index.erase(it->hash);
    return store.entries.erase(it);
}

void reportPipelines(PipelineStore& store) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - store.lastReport < std::chrono::seconds(10)) return;
    store.lastReport = now;
    const auto lookups = store.hits + store.misses + store.uncached;
    AgcDriver::ProfilePrint_nid_no_patch( "[pipecache] %llu lookups over 10 s: %llu hits (%.0f%%), %llu misses, %llu private (no variant id), %llu evicted, %zu cached\n", static_cast<unsigned long long>(lookups), static_cast<unsigned long long>(store.hits), lookups != 0 ? 100.0 * static_cast<double>(store.hits) / static_cast<double>(lookups) : 0.0, static_cast<unsigned long long>(store.misses), static_cast<unsigned long long>(store.uncached), static_cast<unsigned long long>(store.evicted), store.entries.size());
    store.hits = store.misses = store.uncached = store.evicted = 0;
}

}

std::shared_ptr<Pipeline> CachedPipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) {
    return CachedPipeline(context, state, vertexInput, resources.SetLayouts(), shaders, attachmentLayout);
}

namespace {

std::shared_ptr<Pipeline> cachedPipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const PipelineSetLayouts& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout, std::shared_ptr<RecordUse> replayed) {
    static const bool disabled = std::getenv("APS5_NO_PIPELINE_CACHE") != nullptr;
    if (disabled) return std::make_shared<Pipeline>(context, state, vertexInput, resources, shaders, attachmentLayout);
    auto& store = Pipelines();
    struct PipelineKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::byte>, PipelineKeyStorage>();
    pipelineKey(key, context, state, vertexInput, resources, shaders, attachmentLayout);
    const auto hash = key.empty() ? 0u : hashKey(key);
    std::shared_future<std::shared_ptr<Pipeline>> other;
    std::promise<std::shared_ptr<Pipeline>> building;
    bool shared = !key.empty();
    {
        std::lock_guard lock(store.mutex);
        reportPipelines(store);
        if (!shared) ++store.uncached;
        if (const auto found = shared ? store.index.find(hash) : store.index.end(); found != store.index.end()) {
            const auto it = found->second;
            if (it->key == key) {
                if (alive(*it, context)) {
                    ++store.hits;
                    store.entries.splice(store.entries.end(), store.entries, it);
                    if (replayed == nullptr && it->use != nullptr) it->use->Mark();
                    return it->pipeline;
                }
                abandon(store, it);
            } else {
                ++store.uncached;
                shared = false;
            }
        }
        if (shared) {
            if (const auto pending = store.building.find(hash); pending != store.building.end()) {
                if (pending->second.key == key) other = pending->second.pipeline;
                else {
                    ++store.uncached;
                    shared = false;
                }
            } else {
                ++store.misses;
                for (auto it = store.entries.begin(); it != store.entries.end();) {
                    it = alive(*it, context) ? std::next(it) : abandon(store, it);
                }
                store.building.emplace(hash, PipelineStore::Building{key, building.get_future().share()});
            }
        }
        ++store.misses;
        ++store.totals.misses;
    }
    if (!shared) return std::make_shared<Pipeline>(context, state, vertexInput, resources, shaders, attachmentLayout);
    if (other.valid()) {
        auto pipeline = other.get();
        if (replayed == nullptr) {
            std::lock_guard lock(store.mutex);
            if (const auto found = store.index.find(hash); found != store.index.end() && found->second->pipeline == pipeline && found->second->use != nullptr) found->second->use->Mark();
        }
        return pipeline;
    }
    const auto ownKey = key;
    std::shared_ptr<Pipeline> pipeline;
    auto use = replayed;
    try {
        pipeline = std::make_shared<Pipeline>(context, state, vertexInput, resources, shaders, attachmentLayout);
        if (use == nullptr && !CreationAhead::Active() && context.pipelineRecorder != nullptr) use = context.pipelineRecorder->Created(context, state, vertexInput, resources, shaders, attachmentLayout);
    } catch (...) {
        {
            std::lock_guard lock(store.mutex);
            store.building.erase(hash);
        }
        building.set_exception(std::current_exception());
        throw;
    }
    {
        std::lock_guard lock(store.mutex);
        store.building.erase(hash);
        store.entries.push_back({context.device, context.bufferPool, hash, ownKey, pipeline, use});
        store.index[hash] = std::prev(store.entries.end());
        constexpr std::size_t bound = 1024;
        while (store.entries.size() > bound) {
            const auto victim = std::find_if(store.entries.begin(), store.entries.end(), [](const PipelineStore::Entry& entry) { return entry.pipeline.use_count() == 1; });
            if (victim == store.entries.end()) break;
            store.index.erase(victim->hash);
            store.entries.erase(victim);
            ++store.evicted;
        }
    }
    building.set_value(pipeline);
    return pipeline;
}

}

std::shared_ptr<Pipeline> CachedPipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const PipelineSetLayouts& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) {
    return cachedPipeline(context, state, vertexInput, resources, shaders, attachmentLayout, nullptr);
}

std::shared_ptr<Pipeline> PrepareCachedPipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const PipelineSetLayouts& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout, std::shared_ptr<RecordUse> use) {
    Require(use != nullptr, "a pipeline prepared ahead needs its manifest record");
    return cachedPipeline(context, state, vertexInput, resources, shaders, attachmentLayout, std::move(use));
}

void ClearCachedPipelines(VkDevice device) {
    auto& store = Pipelines();
    std::lock_guard lock(store.mutex);
    for (auto it = store.entries.begin(); it != store.entries.end();) {
        if (it->device != device) {
            ++it;
            continue;
        }
        store.index.erase(it->hash);
        it = store.entries.erase(it);
    }
    ClearPipelineLibraries(device);
}

namespace {

constexpr std::uint32_t DescriptionVersion = 2;

class DescriptionWriter {
public:
    explicit DescriptionWriter(std::vector<std::byte>& out) : out(out) {}

    template<typename TValue>
    void Value(const TValue& value) {
        static_assert(std::is_trivially_copyable_v<TValue>);
        const auto bytes = std::as_bytes(std::span(&value, 1));
        out.insert(out.end(), bytes.begin(), bytes.end());
    }

    void Flag(bool value) { Value(static_cast<std::uint8_t>(value ? 1u : 0u)); }

    template<typename TValue>
    void Optional(const std::optional<TValue>& value) {
        Flag(value.has_value());
        if (value) Value(*value);
    }

    template<typename TValue, typename TWrite>
    void List(const std::vector<TValue>& values, TWrite&& write) {
        Value(static_cast<std::uint32_t>(values.size()));
        for (const auto& value : values) write(value);
    }

private:
    std::vector<std::byte>& out;
};

class DescriptionReader {
public:
    explicit DescriptionReader(std::span<const std::byte> bytes) : bytes(bytes) {}

    template<typename TValue>
    bool Value(TValue& value) {
        static_assert(std::is_trivially_copyable_v<TValue>);
        if (bytes.size() - offset < sizeof(TValue)) return false;
        std::memcpy(&value, bytes.data() + offset, sizeof(TValue));
        offset += sizeof(TValue);
        return true;
    }

    bool Flag(bool& value) {
        std::uint8_t raw = 0;
        if (!Value(raw) || raw > 1u) return false;
        value = raw != 0u;
        return true;
    }

    template<typename TValue>
    bool Optional(std::optional<TValue>& value) {
        bool present = false;
        if (!Flag(present)) return false;
        if (!present) {
            value.reset();
            return true;
        }
        TValue read{};
        if (!Value(read)) return false;
        value = read;
        return true;
    }

    template<typename TValue, typename TRead>
    bool List(std::vector<TValue>& values, std::size_t minimumBytes, TRead&& read) {
        std::uint32_t count = 0;
        if (!Value(count) || count > (bytes.size() - offset) / minimumBytes) return false;
        values.resize(count);
        for (auto& value : values) {
            if (!read(value)) return false;
        }
        return true;
    }

    bool Done() const { return offset == bytes.size(); }

private:
    std::span<const std::byte> bytes;
    std::size_t offset = 0;
};

}

std::vector<std::byte> EncodePipelineDescription(const State& state, const VertexInputLayout& vertexInput, VkImageLayout attachmentLayout) {
    std::vector<std::byte> out;
    DescriptionWriter writer(out);
    writer.Value(DescriptionVersion);
    writer.Value(state.stages.path);
    writer.Value(state.stages.registerValue);
    writer.Value(state.stages.vertexWaveSize);
    writer.Value(state.stages.fragmentWaveSize);
    writer.Optional(state.stages.mesh);
    writer.Optional(state.stages.tessellation);
    writer.Value(state.stages.minTessellationLevel);
    writer.Value(state.stages.maxTessellationLevel);
    writer.Flag(state.depth.has_value());
    if (state.depth) {
        writer.Value(state.depth->format);
        writer.Value(state.depth->samples);
    }
    writer.Flag(state.depthTest);
    writer.Flag(state.depthWrite);
    writer.Value(state.depthCompare);
    writer.Flag(state.depthBoundsTest);
    writer.Value(state.minDepthBounds);
    writer.Value(state.maxDepthBounds);
    writer.Flag(state.depthBias);
    writer.Value(state.depthBiasConstant);
    writer.Value(state.depthBiasSlope);
    writer.Value(state.depthBiasClamp);
    writer.Flag(state.depthBiasPerFace);
    writer.Value(state.backDepthBiasConstant);
    writer.Value(state.backDepthBiasSlope);
    writer.Flag(state.stencilTest);
    writer.Value(state.stencilFront);
    writer.Value(state.stencilBack);
    writer.List(state.colors, [&](const ColorTarget& color) {
        writer.Value(color.format);
        writer.Value(color.componentMapping);
        writer.Value(color.exportIndex);
        writer.Value(color.samples);
        writer.Value(color.packing);
        writer.Flag(color.uintExport);
    });
    writer.List(state.blends, [&](const VkPipelineColorBlendAttachmentState& blend) { writer.Value(blend); });
    writer.Flag(state.hasColorTarget);
    writer.Flag(state.rectList);
    writer.Flag(state.dualSourceBlend);
    writer.Value(state.renderLayers);
    writer.Value(state.layerExports);
    writer.Value(state.topology);
    writer.Flag(state.primitiveRestart);
    writer.Flag(state.negativeOneToOne);
    writer.Flag(state.depthClamp);
    writer.Flag(state.alphaToCoverage);
    writer.Value(state.conservativeRasterization);
    writer.Value(state.cullMode);
    writer.Value(state.frontFace);
    writer.Value(state.provokingVertexMode);
    writer.Value(state.blendConstants);
    writer.Value(state.samples);
    writer.Value(state.sampleLocations);
    writer.Flag(state.customSampleLocations);
    writer.Value(state.paClVsOutCntl);
    writer.Optional(state.stencilReplayReference);
    writer.Value(state.reversedBlend);
    writer.Flag(state.ditheredAlphaToCoverage);
    writer.Flag(state.perSampleShading);
    writer.Flag(state.sampleLocationGrid);
    writer.Value(state.quadSampleMask);
    writer.Flag(state.pixelEarlyZ);
    writer.List(vertexInput.bindings, [&](const VkVertexInputBindingDescription& binding) { writer.Value(binding); });
    writer.List(vertexInput.attributes, [&](const VkVertexInputAttributeDescription& attribute) { writer.Value(attribute); });
    writer.Value(attachmentLayout);
    return out;
}

std::optional<PipelineDescription> DecodePipelineDescription(std::span<const std::byte> bytes) {
    DescriptionReader reader(bytes);
    PipelineDescription result;
    auto& state = result.state;
    std::uint32_t version = 0;
    if (!reader.Value(version) || version != DescriptionVersion) return std::nullopt;
    bool depth = false;
    bool ok = reader.Value(state.stages.path) && reader.Value(state.stages.registerValue) && reader.Value(state.stages.vertexWaveSize) && reader.Value(state.stages.fragmentWaveSize)
        && reader.Optional(state.stages.mesh) && reader.Optional(state.stages.tessellation) && reader.Value(state.stages.minTessellationLevel) && reader.Value(state.stages.maxTessellationLevel)
        && reader.Flag(depth);
    if (ok && depth) {
        DepthTarget target{};
        ok = reader.Value(target.format) && reader.Value(target.samples);
        state.depth = target;
    }
    ok = ok && reader.Flag(state.depthTest) && reader.Flag(state.depthWrite) && reader.Value(state.depthCompare) && reader.Flag(state.depthBoundsTest) && reader.Value(state.minDepthBounds) && reader.Value(state.maxDepthBounds)
        && reader.Flag(state.depthBias) && reader.Value(state.depthBiasConstant) && reader.Value(state.depthBiasSlope) && reader.Value(state.depthBiasClamp) && reader.Flag(state.depthBiasPerFace)
        && reader.Value(state.backDepthBiasConstant) && reader.Value(state.backDepthBiasSlope) && reader.Flag(state.stencilTest) && reader.Value(state.stencilFront) && reader.Value(state.stencilBack)
        && reader.List(state.colors, 15u, [&](ColorTarget& color) {
               color = ColorTarget{};
               return reader.Value(color.format) && reader.Value(color.componentMapping) && reader.Value(color.exportIndex) && reader.Value(color.samples) && reader.Value(color.packing) && reader.Flag(color.uintExport);
           })
        && reader.List(state.blends, sizeof(VkPipelineColorBlendAttachmentState), [&](VkPipelineColorBlendAttachmentState& blend) { return reader.Value(blend); })
        && reader.Flag(state.hasColorTarget) && reader.Flag(state.rectList) && reader.Flag(state.dualSourceBlend) && reader.Value(state.renderLayers) && reader.Value(state.layerExports)
        && reader.Value(state.topology) && reader.Flag(state.primitiveRestart) && reader.Flag(state.negativeOneToOne) && reader.Flag(state.depthClamp) && reader.Flag(state.alphaToCoverage)
        && reader.Value(state.conservativeRasterization) && reader.Value(state.cullMode) && reader.Value(state.frontFace) && reader.Value(state.provokingVertexMode) && reader.Value(state.blendConstants)
        && reader.Value(state.samples) && reader.Value(state.sampleLocations) && reader.Flag(state.customSampleLocations) && reader.Value(state.paClVsOutCntl)
        && reader.Optional(state.stencilReplayReference) && reader.Value(state.reversedBlend) && reader.Flag(state.ditheredAlphaToCoverage) && reader.Flag(state.perSampleShading)
        && reader.Flag(state.sampleLocationGrid) && reader.Value(state.quadSampleMask) && reader.Flag(state.pixelEarlyZ)
        && reader.List(result.vertexInput.bindings, sizeof(VkVertexInputBindingDescription), [&](VkVertexInputBindingDescription& binding) { return reader.Value(binding); })
        && reader.List(result.vertexInput.attributes, sizeof(VkVertexInputAttributeDescription), [&](VkVertexInputAttributeDescription& attribute) { return reader.Value(attribute); })
        && reader.Value(result.attachmentLayout);
    if (!ok || !reader.Done()) return std::nullopt;
    return result;
}

bool SamePipelineKey(const Context& context, const PipelineSetLayouts& resources, std::span<const CompiledShader> shaders, const PipelineDescription& left, const PipelineDescription& right) {
    std::vector<std::byte> leftKey;
    std::vector<std::byte> rightKey;
    pipelineKey(leftKey, context, left.state, left.vertexInput, resources, shaders, left.attachmentLayout);
    pipelineKey(rightKey, context, right.state, right.vertexInput, resources, shaders, right.attachmentLayout);
    return !leftKey.empty() && leftKey == rightKey;
}

}
