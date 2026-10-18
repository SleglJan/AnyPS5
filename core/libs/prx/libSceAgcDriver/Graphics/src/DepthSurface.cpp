#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSampleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HtileEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/shaders/DepthPublish00_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/DepthPublish01_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/DepthPublish10_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/DepthPublish11_spv.h"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyMirrors.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <deque>
#include <limits>
#include <shared_mutex>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

std::uint64_t depthWrites = 0;

Recorder* RecorderFor(const Context& context) {
    auto* recorder = Recorder::Active();
    return recorder != nullptr && recorder->Device() == context.device ? recorder : nullptr;
}

bool storageFits(const DepthTarget& target, const GuestTextureResource& descriptor) {
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto format = ResolveTextureFormat(descriptor.format);
    const bool sized = d16 ? (format == VK_FORMAT_R16_UINT || format == VK_FORMAT_R16_UNORM || format == VK_FORMAT_R16_SINT || format == VK_FORMAT_R16_SNORM || format == VK_FORMAT_R16_SFLOAT) : (format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT);
    return sized && descriptor.width == target.extent.width && descriptor.height == target.extent.height && descriptor.mipCount == 1 && (descriptor.dimension == TextureDimension::k2D || descriptor.dimension == TextureDimension::k2DArray);
}

struct DepthGuard {
    VkDevice device;
    std::uint64_t address;
    std::size_t bytes;
    std::shared_ptr<std::atomic<bool>> active;
    std::vector<GuestMemory::UnchangedQuery> live;
    std::shared_ptr<std::atomic<bool>> published;
};

std::shared_mutex& depthGuardMutex() {
    static std::shared_mutex mutex;
    return mutex;
}

std::vector<DepthGuard>& depthGuards() {
    static std::vector<DepthGuard> ranges;
    return ranges;
}

std::atomic<bool>& anyDepthGuard() {
    static std::atomic<bool> any{false};
    return any;
}

struct DepthSamplingState {
    std::array<GuestMemory::UnchangedQuery, 2> initial{};
    std::string refusal;
};

bool htileProvesClear(std::uint32_t word, bool depthOnly, float clearDepth) {
    if ((word & 15u) == 15u) return false;
    const auto depth = depthOnly ? word : word & 0xfffff00fu;
    if (depth == 0u) return clearDepth == 0.0f;
    if (depth == (depthOnly ? 0xfffffff0u : 0xfffc0000u)) return clearDepth == 1.0f;
    return false;
}

DepthSamplingState prepareDepthSampling(const Context& context, const DepthTarget& target) {
    DepthSamplingState state;
    const auto refuse = [&](const char* reason) {
        state.refusal = reason;
        return state;
    };
    if (target.samples != 2u && target.samples != 4u && target.samples != 8u) return refuse("unsupported native depth sample count");
    if (target.format != VK_FORMAT_D16_UNORM && target.format != VK_FORMAT_D32_SFLOAT) return refuse("native multisample stencil or depth format import is not implemented");
    if (target.depthSwizzle != 24u || target.firstSlice != 0u || target.lastSlice != 0u) return refuse("native multisample depth arrays or non-Z_X layout are not implemented");
    if ((target.pipeBankXor & ~0xcf00u) != 0u) return refuse("unsupported native multisample depth XOR");
    if (target.address == 0u || (target.address & 65535u) != 0u) return refuse("native multisample depth base is not aligned");
    if (target.stencilAddress != 0u || target.htileStencil) return refuse("native multisample stencil import is not implemented");
    if (target.htileAddress == 0u || (target.htileAddress & 255u) != 0u || !target.iterate256 || !target.htilePipeAligned || target.htileVrsEncoding != 0u) return refuse("native multisample depth initial import requires pipe-aligned ITERATE_256 HTILE without VRS");
    if (target.clearDepth != 0.0f && target.clearDepth != 1.0f) return refuse("native multisample depth initial clear value is not implemented");
    state.initial = {{{target.address, static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(target.extent, target.format == VK_FORMAT_D16_UNORM ? 2u : 4u, target.samples)), 0u}, {target.htileAddress, static_cast<std::size_t>(HtileMetadataBytes(target.extent)), 0u}}};
    for (const auto& range : state.initial) {
        if (range.address > std::numeric_limits<std::uint64_t>::max() - range.bytes) return refuse("native depth import range overflows");
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(range.address), range.bytes) || !GuestMemory::Watched(range.address, range.bytes)) return refuse("native multisample depth import requires watched accessible guest memory");
    }
    const auto& depth = state.initial[0];
    const auto& htile = state.initial[1];
    if (depth.address + depth.bytes > htile.address && htile.address + htile.bytes > depth.address) return refuse("native depth and HTILE initial ranges overlap");
    for (auto& range : state.initial) {
        if (auto* recorder = RecorderFor(context)) recorder->SyncThrough(range.address, range.bytes);
        StorageTexture::FlushPending(range.address, range.bytes);
        GuestMemory::FlushGpuWrites(range.address, range.bytes);
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(range.address), range.bytes) || !GuestMemory::Watched(range.address, range.bytes)) return refuse("native multisample depth import requires watched accessible guest memory");
        range.generation = GuestMemory::CollectWritesUncached(range.address, range.bytes);
        if (range.generation == 0u) return refuse("native multisample depth import requires tracked generations");
    }
    std::vector<std::uint32_t> metadata(htile.bytes / sizeof(std::uint32_t));
    GuestMemory::Read(target.htileAddress, std::as_writable_bytes(std::span(metadata)), 256u);
    for (const auto word : metadata) {
        if (!htileProvesClear(word, target.htileDepthOnly, target.clearDepth)) return refuse("native multisample depth initial HTILE does not prove its clear value");
    }
    for (const auto& range : state.initial) GuestMemory::CollectWritesUncached(range.address, range.bytes);
    if (!GuestMemory::UnchangedSinceAll(state.initial)) return refuse("guest writes raced native multisample depth import");
    return state;
}

struct DepthPublishPush {
    std::array<std::uint32_t, 16> equation{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t elementBytes = 0;
    std::uint32_t blocksPerRow = 0;
    std::uint32_t blockWidthLog2 = 0;
    std::uint32_t blockHeightLog2 = 0;
    std::uint32_t pipeBankXor = 0;
    std::uint32_t slice = 0;
};

static_assert(sizeof(DepthPublishPush) == 96);

std::pair<std::uint32_t, std::uint32_t> DepthBlockLog2(std::uint32_t elementBytes, std::uint32_t samples) {
    const auto sampleBits = static_cast<std::uint32_t>(std::countr_zero(samples));
    const auto pixelBits = 16u - static_cast<std::uint32_t>(std::countr_zero(elementBytes)) - sampleBits;
    const auto widthBits = (pixelBits + ((sampleBits & 1u) == 0u ? 1u : 0u)) / 2u;
    return {widthBits, pixelBits - widthBits};
}

std::uint32_t EquationOffset(const std::array<std::uint32_t, 16>& bits, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    const auto coordinates = (x & 0xfffu) | ((y & 0xfffu) << 12u) | ((z & 0xffu) << 24u);
    std::uint32_t offset = 0;
    for (std::uint32_t bit = 0; bit < bits.size(); ++bit) offset |= static_cast<std::uint32_t>(std::popcount(bits[bit] & coordinates) & 1) << bit;
    return offset;
}

using ByteRuns = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

void MergeRuns(ByteRuns& runs) {
    std::sort(runs.begin(), runs.end());
    ByteRuns merged;
    for (const auto& run : runs) {
        if (!merged.empty() && run.first <= merged.back().second) merged.back().second = std::max(merged.back().second, run.second);
        else merged.push_back(run);
    }
    runs.swap(merged);
}

constexpr bool SameAsTextureEquation(std::uint32_t elementBytes) {
    const auto* depth = FindDepthSampleEquation(1u, elementBytes);
    if (depth == nullptr) return false;
    for (const auto& texture : kTextureSwizzleEquations) {
        if (texture.swizzleMode == 24u && texture.elementBytes == elementBytes) return texture.bits == depth->bits;
    }
    return false;
}

static_assert(SameAsTextureEquation(1u) && SameAsTextureEquation(2u) && SameAsTextureEquation(4u));

class DepthPublisher {
public:
    explicit DepthPublisher(const Context& context) : context(context) {
        try {
            const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
                {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            }};
            VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout depth publication");
            const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DepthPublishPush)};
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &setLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &push;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout depth publication");
            VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            samplerInfo.magFilter = VK_FILTER_NEAREST;
            samplerInfo.minFilter = VK_FILTER_NEAREST;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            Check(context.Function<PFN_vkCreateSampler>("vkCreateSampler")(context.device, &samplerInfo, nullptr, &sampler), "vkCreateSampler depth publication");
            const std::array<std::span<const std::uint32_t>, 4> code{{std::span(DEPTH_PUBLISH_00_SPV), std::span(DEPTH_PUBLISH_01_SPV), std::span(DEPTH_PUBLISH_10_SPV), std::span(DEPTH_PUBLISH_11_SPV)}};
            for (std::size_t variant = 0; variant < code.size(); ++variant) {
                VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                moduleInfo.codeSize = code[variant].size_bytes();
                moduleInfo.pCode = code[variant].data();
                Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &modules[variant]), "vkCreateShaderModule depth publication");
                VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                createInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                createInfo.stage.module = modules[variant];
                createInfo.stage.pName = "main";
                createInfo.layout = pipelineLayout;
                Check(CreateComputePipeline(context, createInfo, &pipelines[variant]), "vkCreateComputePipelines depth publication");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~DepthPublisher() { release(); }
    DepthPublisher(const DepthPublisher&) = delete;
    DepthPublisher& operator=(const DepthPublisher&) = delete;

    VkDescriptorSet Allocate(VkImageView view, VkBuffer tiled, VkDeviceSize bytes) {
        constexpr std::uint32_t setsPerPool = 64;
        if (pools.empty() || used == setsPerPool) {
            const std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, setsPerPool}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, setsPerPool}}};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = setsPerPool;
            poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
            poolInfo.pPoolSizes = sizes.data();
            VkDescriptorPool pool = VK_NULL_HANDLE;
            Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool depth publication");
            pools.push_back(pool);
            used = 0;
        }
        VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = pools.back();
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &setLayout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocateInfo, &set), "vkAllocateDescriptorSets depth publication");
        ++used;
        const VkDescriptorImageInfo imageInfo{sampler, view, VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorBufferInfo bufferInfo{tiled, 0, bytes};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (std::uint32_t index = 0; index < writes.size(); ++index) {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = set;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = index == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            if (index == 0) writes[index].pImageInfo = &imageInfo;
            else writes[index].pBufferInfo = &bufferInfo;
        }
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return set;
    }

    void Dispatch(VkCommandBuffer commands, bool stencil, bool multisampled, VkDescriptorSet set, const DepthPublishPush& push, std::uint32_t samples) {
        const auto pipeline = pipelines[(stencil ? 2u : 0u) + (multisampled ? 1u : 0u)];
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, (push.width + 7u) / 8u, (push.height + 7u) / 8u, samples);
    }

private:
    void release() noexcept {
        for (auto& pipeline : pipelines) {
            if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        for (auto& module : modules) {
            if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            module = VK_NULL_HANDLE;
        }
        for (const auto pool : pools) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        pools.clear();
        if (sampler) context.Function<PFN_vkDestroySampler>("vkDestroySampler")(context.device, sampler, nullptr);
        if (pipelineLayout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, pipelineLayout, nullptr);
        if (setLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
        sampler = VK_NULL_HANDLE;
        pipelineLayout = VK_NULL_HANDLE;
        setLayout = VK_NULL_HANDLE;
    }

    Context context;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    std::array<VkShaderModule, 4> modules{};
    std::array<VkPipeline, 4> pipelines{};
    std::vector<VkDescriptorPool> pools;
    std::uint32_t used = 0;
};

std::map<VkDevice, std::unique_ptr<DepthPublisher>>& publishers() {
    static auto* map = new std::map<VkDevice, std::unique_ptr<DepthPublisher>>();
    return *map;
}

DepthPublisher& PublisherFor(const Context& context) {
    auto& entry = publishers()[context.device];
    if (entry == nullptr) entry = std::make_unique<DepthPublisher>(context);
    return *entry;
}

class DepthSurface final : public IResidencyOwner {
public:
    DepthSurface(const Context& context, const DepthTarget& target, DepthSamplingState sampling = {}) : context(context), target(target), hostFormat(HostDepthFormat(context, target.format)), sampling(std::move(sampling)) {
        this->context.bufferPool.reset();
        VkFormatProperties properties{};
        context.formatProperties(context.physical, hostFormat, &properties);
        Require((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0, "depth/stencil format " + std::to_string(hostFormat) + " cannot be an attachment on this device");
        Require(target.samples != 1u || (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0, "depth/stencil format " + std::to_string(hostFormat) + " cannot be sampled on this device");
        Require(target.extent.width <= context.limits.maxFramebufferWidth && target.extent.height <= context.limits.maxFramebufferHeight, "depth target exceeds framebuffer limits");
        const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (target.stencilAddress != 0 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
        try {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            if (SampleLocationsCompatibleDepth(context, hostFormat, target.samples)) info.flags = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = hostFormat;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = static_cast<VkSampleCountFlagBits>(target.samples);
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            sampleable = target.samples == 1;
            if (target.samples != 1) {
                Require(context.imageFormatProperties != nullptr, "native multisample depth requires image format queries");
                VkImageFormatProperties supported{};
                const auto result = context.imageFormatProperties(context.physical, hostFormat, info.imageType, info.tiling, info.usage | VK_IMAGE_USAGE_SAMPLED_BIT, info.flags, &supported);
                if (result != VK_ERROR_FORMAT_NOT_SUPPORTED) Check(result, "vkGetPhysicalDeviceImageFormatProperties sampled depth");
                sampleable = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0 && result == VK_SUCCESS && (supported.sampleCounts & info.samples) != 0 && supported.maxMipLevels >= 1u && supported.maxArrayLayers >= 1u && supported.maxExtent.depth >= 1u && info.extent.width <= supported.maxExtent.width && info.extent.height <= supported.maxExtent.height;
            }
            if (sampleable) info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage depth");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory depth target");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory depth");
            if (target.samples != 1u) owned = std::make_shared<OwnedImage>(this->context, image, memory);
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = hostFormat;
            viewInfo.subresourceRange = {aspects, 0, 1, 0, 1};
            Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView depth");
            if (target.samples != 1u && sampleable && this->sampling.refusal.empty()) CheckCurrent();
            auto* recorder = RecorderFor(context);
            if (recorder != nullptr && owned != nullptr) recorder->Keep(owned);
            std::unique_ptr<CommandBatch> batch;
            if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
            if (recorder != nullptr) recorder->DeclareSection({{image, true}, {view, true}});
            const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
            VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = image;
            toGeneral.subresourceRange = viewInfo.subresourceRange;
            const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
            barrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
            const VkClearDepthStencilValue clear{target.clearDepth, target.clearStencil};
            context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &toGeneral.subresourceRange);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            if (batch) batch->SubmitAndWait();
            else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        } catch (...) {
            release();
            throw;
        }
    }

    ~DepthSurface() override {
        if (residencyOwned) {
            if (auto* residency = ResidencyDriver::Active()) residency->Manager().Forget(this);
        }
        release();
    }
    DepthSurface(const DepthSurface&) = delete;
    DepthSurface& operator=(const DepthSurface&) = delete;

    void CheckCurrent() {
        if (target.samples == 1u) return;
        const auto clearLease = clearGeneration == 0u ? GuestAllocations::Lease{} : GuestAllocations::GuestAllocationsAcquireAll_nid_postfix();
        Require(active->load(std::memory_order_acquire), "the native multisample depth owner was released");
        Require(sampling.refusal.empty(), sampling.refusal);
        for (const auto& range : sampling.initial) {
            Require(GuestMemory::Accessible(reinterpret_cast<const void*>(range.address), range.bytes) && GuestMemory::Watched(range.address, range.bytes), "native multisample depth requires watched accessible guest memory");
            if (const auto* recorder = RecorderFor(context)) Require(!recorder->PendingWriteOverlaps(range.address, range.bytes), "external GPU writes over native multisample depth are not implemented");
            GuestMemory::CollectWritesUncached(range.address, range.bytes);
        }
        if (clearGeneration != 0u) {
            const auto& depth = sampling.initial[0];
            const auto& metadata = sampling.initial[1];
            Require(GuestMemory::UnchangedSince(depth.address, depth.bytes, depth.generation), "guest writes over native multisample depth are not implemented");
            Require(GuestMemory::UnchangedSinceCollected(metadata.address, metadata.bytes, clearGeneration), "CPU writes raced native multisample depth clear");
            Require(GuestMemory::StoredAllSince_nid_no_patch(metadata.address, metadata.bytes, clearGeneration), "native multisample depth clear has no complete tracked HTILE store");
            Require(clearDepth == 0.0f || clearDepth == 1.0f, "native multisample depth clear value is not implemented");
            auto current = sampling.initial;
            current[1].generation = GuestMemory::CollectWritesUncached(metadata.address, metadata.bytes);
            std::vector<std::uint32_t> words(metadata.bytes / sizeof(std::uint32_t));
            std::memcpy(words.data(), reinterpret_cast<const void*>(metadata.address), metadata.bytes);
            for (const auto word : words) Require(htileProvesClear(word, target.htileDepthOnly, clearDepth), "native multisample depth completed HTILE does not prove its clear value");
            for (const auto& range : current) GuestMemory::CollectWritesUncached(range.address, range.bytes);
            Require(GuestMemory::UnchangedSinceAll(current), "guest writes raced native multisample depth clear");
            Require(GuestMemory::UnchangedSinceCollected(metadata.address, metadata.bytes, clearGeneration), "CPU writes raced native multisample depth clear");
            RecordClear(VK_IMAGE_ASPECT_DEPTH_BIT);
            sampling.initial = current;
            clearGeneration = 0u;
            pendingClear = 0u;
        }
        Require(GuestMemory::UnchangedSinceAll(sampling.initial), "guest writes over native multisample depth are not implemented");
    }

    void CheckResource(const GuestTextureResource& resource) const {
        if (target.samples == 1u) return;
        Require(resource.baseAddress == target.address && resource.width == target.extent.width && resource.height == target.extent.height && resource.samples == target.samples && resource.format == (target.format == VK_FORMAT_D16_UNORM ? 7u : 22u), "native multisample depth sampled identity or format mismatch");
        Require(resource.dimension == TextureDimension::k2DMsaa && resource.tileMode == TextureTileMode::kZ64KBX && resource.depthOrLastArray == 0u && resource.baseArray == 0u && resource.baseLevel == 0u && resource.lastLevel == 0u && resource.mipCount == 1u && resource.allocatedMipCount == 1u && resource.minLod == 0u, "native multisample depth sampled array, mip or layout is not implemented");
        Require(resource.pipeBankXor == target.pipeBankXor && resource.dccAddress == target.htileAddress && resource.iterate256 == target.iterate256 && resource.dccPipeAligned == target.htilePipeAligned && !resource.dccAlphaOnMsb, "native multisample depth sampled metadata or XOR mismatch");
    }

    void ApplyFastClear() {
        if (samplingActive) {
            CheckCurrent();
            Require(pendingClear == 0u, "native multisample depth fast-clear transition is not implemented");
        }
        const auto aspects = pendingClear & (VK_IMAGE_ASPECT_DEPTH_BIT | (target.stencilAddress != 0 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u));
        pendingClear = 0;
        if (aspects == 0) return;
        RecordClear(aspects);
    }

    void RecordClear(VkImageAspectFlags aspects) {
        written = ++depthWrites;
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        if (recorder != nullptr) recorder->DeclareSection({{image, true}, {view, true}});
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        const VkImageSubresourceRange range{aspects, 0, 1, 0, 1};
        constexpr VkAccessFlags access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, access, VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearDepthStencilValue clear{clearDepth, clearStencil};
        context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, access);
        if (batch) batch->SubmitAndWait();
        else {
            Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
            recorder->Keep(owned);
        }
        MarkStale(aspects);
    }

    static constexpr std::size_t DepthPlane = 0;
    static constexpr std::size_t StencilPlane = 1;

    std::uint32_t planeElementBytes(std::size_t plane) const {
        if (plane == StencilPlane) return 1u;
        return target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT ? 2u : 4u;
    }

    std::uint64_t planeAddress(std::size_t plane) const {
        return plane == StencilPlane ? target.stencilAddress : target.address;
    }

    std::uint64_t planeBytes(std::size_t plane) const {
        const auto elementBytes = planeElementBytes(plane);
        return target.samples == 1u ? DepthSliceBytes(target.extent, elementBytes) : MultisampledDepthBytes_nid_no_patch(target.extent, elementBytes, target.samples);
    }

    std::uint32_t planeSwizzle(std::size_t plane) const {
        return plane == StencilPlane ? target.stencilSwizzle : target.depthSwizzle;
    }

    const std::array<std::uint32_t, 16>* planeEquation(std::size_t plane) const {
        const auto swizzle = planeSwizzle(plane);
        const auto elementBytes = planeElementBytes(plane);
        if (target.samples == 1u) {
            if (swizzle < static_cast<std::uint32_t>(ColorTileMode::Z64KBX) || swizzle > static_cast<std::uint32_t>(ColorTileMode::RenderTarget)) return nullptr;
            for (const auto& equation : kTextureSwizzleEquations) {
                if (equation.swizzleMode == swizzle && equation.elementBytes == elementBytes) return &equation.bits;
            }
            return nullptr;
        }
        if (swizzle != static_cast<std::uint32_t>(ColorTileMode::Z64KBX)) return nullptr;
        const auto* equation = FindDepthSampleEquation(target.samples, elementBytes);
        return equation != nullptr ? &equation->bits : nullptr;
    }

    std::string PlaneRefusal(std::size_t plane) const {
        if (planeAddress(plane) == 0u) return "the plane has no guest memory";
        if (planeEquation(plane) == nullptr) return "no 64 KiB address equation for its swizzle mode and sample count";
        if (target.samples != 1u && (target.firstSlice != 0u || target.lastSlice != 0u)) return "a multisampled array slice";
        if (target.samples != 1u && !sampleable) return "the device cannot sample the multisampled plane";
        if (plane == DepthPlane && hostFormat != target.format) {
            if (hostFormat == VK_FORMAT_D24_UNORM_S8_UINT) return "16-bit depth stored as 24-bit depth rounds twice: a value next to a 16-bit rounding boundary converts back one step off";
            if (hostFormat != VK_FORMAT_D32_SFLOAT_S8_UINT) return "the device stores the depth plane in a format with no exact conversion back";
        }
        return {};
    }

    std::uint64_t htileBytes() const {
        return HtileMetadataBytes(target.extent);
    }

    std::uint32_t ExpandedHtileWord() const {
        return target.htileStencil ? 0xfffff3ffu : 0xfffc000fu;
    }

    void OwnPlanes() {
        auto* residency = ResidencyDriver::Active();
        if (residency == nullptr || residency->Device() != context.device || retired) return;
        auto& manager = residency->Manager();
        if (residencyOwned && ownedSerial == manager.OwnershipSerial()) return;
        std::array<bool, 2> imported{};
        for (std::size_t index = 0; samplingActive && index < imported.size(); ++index) {
            const auto& range = sampling.initial[index];
            imported[index] = GuestMemory::CollectWritesUncached(range.address, range.bytes) != 0u && GuestMemory::UnchangedSince(range.address, range.bytes, range.generation);
        }
        for (std::size_t plane = 0; plane < published.size(); ++plane) {
            if (!published[plane].owned && !PlaneRefusal(plane).empty()) continue;
            manager.Own(planeAddress(plane), planeBytes(plane), this);
            published[plane].owned = true;
            residencyOwned = true;
        }
        const bool htile = target.htileAddress != 0u && published[DepthPlane].owned && (!target.htileStencil || published[StencilPlane].owned);
        if (htile) {
            manager.Own(target.htileAddress, htileBytes(), this);
            if (!htileOwned) htileGeneration = GuestMemory::CollectWritesUncached(target.htileAddress, static_cast<std::size_t>(htileBytes()));
            htileOwned = true;
        }
        for (std::size_t index = 0; samplingActive && index < imported.size(); ++index) {
            auto& range = sampling.initial[index];
            if (imported[index]) range.generation = GuestMemory::CollectWritesUncached(range.address, range.bytes);
        }
        depthPublished->store(published[DepthPlane].owned, std::memory_order_release);
        htilePublished->store(htileOwned, std::memory_order_release);
        ownedSerial = manager.OwnershipSerial();
    }

    void NoteHtileBaseline() {
        auto* residency = ResidencyDriver::Active();
        if (!htileOwned || residency == nullptr || residency->Manager().StaleIn(target.htileAddress, htileBytes())) return;
        htileGeneration = GuestMemory::CollectWritesUncached(target.htileAddress, static_cast<std::size_t>(htileBytes()));
    }

    void DisownPlanes() {
        if (!residencyOwned) return;
        if (auto* residency = ResidencyDriver::Active()) residency->Manager().Disown(this);
        for (auto& plane : published) plane.owned = false;
        htileOwned = false;
        depthPublished->store(false, std::memory_order_release);
        htilePublished->store(false, std::memory_order_release);
        residencyOwned = false;
    }

    void MarkStale(VkImageAspectFlags aspects) {
        auto* residency = ResidencyDriver::Active();
        if (!residencyOwned || residency == nullptr) return;
        std::unique_lock<GuestMemory::GpuMutexType> gpu(GuestMemory::GpuMutex(), std::defer_lock);
        if (!GuestMemory::GpuMutex().HeldByThisThread()) gpu.lock();
        bool marked = false;
        if ((aspects & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 && published[DepthPlane].owned) {
            residency->Manager().MarkStale(planeAddress(DepthPlane), planeBytes(DepthPlane), this);
            marked = true;
        }
        if ((aspects & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 && published[StencilPlane].owned) {
            residency->Manager().MarkStale(planeAddress(StencilPlane), planeBytes(StencilPlane), this);
            marked = marked || target.htileStencil;
        }
        if (marked && htileOwned) residency->Manager().MarkStale(target.htileAddress, htileBytes(), this);
    }

    bool DiscardInside(std::uint64_t address, std::uint64_t bytes) {
        auto* residency = ResidencyDriver::Active();
        if (!residencyOwned || residency == nullptr || residency->Device() != context.device) return false;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (std::size_t plane = 0; plane < published.size(); ++plane) {
            if (published[plane].owned) ranges.emplace_back(planeAddress(plane), planeAddress(plane) + planeBytes(plane));
        }
        if (htileOwned) ranges.emplace_back(target.htileAddress, target.htileAddress + htileBytes());
        const auto end = address + bytes;
        const auto dead = [&](std::uint64_t subpage) {
            const auto stop = subpage + ResidencySubpageBytes;
            return std::all_of(ranges.begin(), ranges.end(), [&](const auto& range) {
                const auto from = std::max(subpage, range.first);
                const auto to = std::min(stop, range.second);
                return from >= to || (from >= address && to <= end);
            });
        };
        auto& manager = residency->Manager();
        bool discarded = false;
        for (const auto& [begin, stop] : ranges) {
            const auto from = std::max(begin, address);
            const auto to = std::min(stop, end);
            for (auto subpage = from & ~std::uint64_t{ResidencySubpageBytes - 1u}; subpage < to; subpage += ResidencySubpageBytes) {
                const auto bit = static_cast<ResidencyMask>(1u << ((subpage >> ResidencySubpageShift) & ((1u << (ResidencyPageShift - ResidencySubpageShift)) - 1u)));
                if ((manager.Stale(subpage >> ResidencyPageShift) & bit) == 0 || !dead(subpage)) continue;
                manager.ClearStale(subpage, ResidencySubpageBytes, this);
                discarded = true;
            }
        }
        return discarded;
    }

    bool PlaneOwned(VkImageAspectFlagBits aspect) const {
        return published[aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? StencilPlane : DepthPlane].owned;
    }

    void preparePlane(std::size_t plane) {
        auto& state = published[plane];
        if (state.view == VK_NULL_HANDLE) {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = hostFormat;
            viewInfo.subresourceRange = {static_cast<VkImageAspectFlags>(plane == StencilPlane ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT), 0, 1, 0, 1};
            Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &state.view), "vkCreateImageView depth publication");
        }
        if (state.tiled == nullptr) state.tiled = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(planeBytes(plane)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (state.set == VK_NULL_HANDLE) state.set = PublisherFor(context).Allocate(state.view, state.tiled->Handle(), state.tiled->Size());
    }

    struct PlaneGeometry {
        const std::array<std::uint32_t, 16>* equation = nullptr;
        std::uint32_t elementBytes = 0;
        std::uint32_t widthLog2 = 0;
        std::uint32_t heightLog2 = 0;
        std::uint32_t blocksPerRow = 0;
        std::uint32_t sliceOffset = 0;
        std::vector<std::uint32_t> inverse;
    };

    struct TileLinks {
        std::uint32_t tilesX = 0;
        std::uint32_t tilesY = 0;
        std::vector<std::uint8_t> tiles;
        std::size_t count = 0;
        bool htileAsked = false;
        std::array<ByteRuns, 2> planes;
        std::array<std::vector<std::uint8_t>, 2> fullBlocks;
        ByteRuns htile;
        std::vector<std::uint8_t> fullHtileBlocks;
        std::array<std::vector<std::uint64_t>, 2> planeSubpages;
        std::vector<std::uint64_t> htileSubpages;

        void Link(std::uint32_t x, std::uint32_t y) {
            auto& tile = tiles[static_cast<std::size_t>(y) * tilesX + x];
            if (tile == 0u) ++count;
            tile = 1u;
        }

        bool Linked(std::uint32_t x, std::uint32_t y) const {
            return tiles[static_cast<std::size_t>(y) * tilesX + x] != 0u;
        }
    };

    const PlaneGeometry& planeGeometry(std::size_t plane) {
        auto& entry = geometry[plane];
        if (!entry.inverse.empty()) return entry;
        entry.equation = planeEquation(plane);
        Require(entry.equation != nullptr, "depth surface publication: the plane has no 64 KiB address equation");
        entry.elementBytes = planeElementBytes(plane);
        const auto [widthLog2, heightLog2] = DepthBlockLog2(entry.elementBytes, target.samples);
        entry.widthLog2 = widthLog2;
        entry.heightLog2 = heightLog2;
        entry.blocksPerRow = (target.extent.width + (1u << widthLog2) - 1u) >> widthLog2;
        entry.sliceOffset = target.samples == 1u ? EquationOffset(*entry.equation, 0u, 0u, target.firstSlice) : 0u;
        std::vector<std::uint32_t> inverse(65536u / entry.elementBytes, ~0u);
        for (std::uint32_t sample = 0; sample < target.samples; ++sample) {
            for (std::uint32_t y = 0; y < (1u << heightLog2); ++y) {
                for (std::uint32_t x = 0; x < (1u << widthLog2); ++x) {
                    const auto offset = EquationOffset(*entry.equation, x, y, target.samples == 1u ? 0u : sample);
                    auto& slot = inverse[offset / entry.elementBytes];
                    Require(offset % entry.elementBytes == 0u && slot == ~0u, "depth surface publication: the plane's address equation maps two elements to one address");
                    slot = x | (y << 12u) | (sample << 24u);
                }
            }
        }
        entry.inverse = std::move(inverse);
        return entry;
    }

    std::uint64_t elementOffset(const PlaneGeometry& plane, std::uint32_t x, std::uint32_t y, std::uint32_t sample) const {
        const auto block = static_cast<std::uint64_t>(y >> plane.heightLog2) * plane.blocksPerRow + (x >> plane.widthLog2);
        return block * 65536u + ((EquationOffset(*plane.equation, x, y, target.samples == 1u ? target.firstSlice : sample) ^ target.pipeBankXor) & 65535u);
    }

    std::optional<std::pair<std::uint32_t, std::uint32_t>> pixelAt(const PlaneGeometry& plane, std::uint64_t offset) const {
        const auto block = offset >> 16u;
        const auto x0 = static_cast<std::uint32_t>(block % plane.blocksPerRow) << plane.widthLog2;
        const auto y0 = static_cast<std::uint32_t>(block / plane.blocksPerRow) << plane.heightLog2;
        const auto local = (static_cast<std::uint32_t>(offset & 65535u) ^ target.pipeBankXor ^ EquationOffset(*plane.equation, x0, y0, 0u) ^ plane.sliceOffset) & 65535u;
        const auto packed = plane.inverse[local / plane.elementBytes];
        const auto x = x0 + (packed & 0xfffu);
        const auto y = y0 + ((packed >> 12u) & 0xfffu);
        if (x >= target.extent.width || y >= target.extent.height) return std::nullopt;
        return std::pair{x, y};
    }

    const std::vector<std::uint32_t>& htileTable() {
        if (!htileInverse.empty()) return htileInverse;
        const auto* equation = FindHtileEquation(target.samples);
        Require(equation != nullptr, "depth surface publication: no HTILE address equation for the sample count");
        std::vector<std::uint32_t> inverse(HtileBlockBytes / 4u, ~0u);
        for (std::uint32_t y = 0; y < HtileBlockHeight / 8u; ++y) {
            for (std::uint32_t x = 0; x < HtileBlockWidth / 8u; ++x) {
                const auto offset = HtileWordOffset(*equation, HtileBlockWidth, x * 8u, y * 8u, 0u);
                auto& slot = inverse[offset / 4u];
                Require(offset % 4u == 0u && slot == ~0u, "depth surface publication: the HTILE address equation maps two tiles to one word");
                slot = x | (y << 12u);
            }
        }
        htileInverse = std::move(inverse);
        return htileInverse;
    }

    std::uint32_t htilePitch() const {
        return (target.extent.width + HtileBlockWidth - 1u) / HtileBlockWidth;
    }

    std::optional<std::pair<std::uint32_t, std::uint32_t>> tileAtHtile(std::uint64_t offset) const {
        const auto block = offset / HtileBlockBytes;
        const auto local = static_cast<std::uint32_t>((offset % HtileBlockBytes) ^ (target.pipeBankXor & 0x0f00u));
        const auto packed = htileInverse[local / 4u];
        const auto x = static_cast<std::uint32_t>(block % htilePitch()) * (HtileBlockWidth / 8u) + (packed & 0xfffu);
        const auto y = static_cast<std::uint32_t>(block / htilePitch()) * (HtileBlockHeight / 8u) + ((packed >> 12u) & 0xfffu);
        if (x * 8u >= target.extent.width || y * 8u >= target.extent.height) return std::nullopt;
        return std::pair{x, y};
    }

    TileLinks linkTiles(std::span<const ResidencyRange> ranges) {
        TileLinks links;
        links.tilesX = (target.extent.width + 7u) / 8u;
        links.tilesY = (target.extent.height + 7u) / 8u;
        links.tiles.assign(static_cast<std::size_t>(links.tilesX) * links.tilesY, 0u);
        const auto linkRect = [&](std::uint32_t x0, std::uint32_t y0, std::uint32_t width, std::uint32_t height) {
            for (std::uint32_t y = y0 / 8u; y < std::min(links.tilesY, (y0 + height) / 8u); ++y) {
                for (std::uint32_t x = x0 / 8u; x < std::min(links.tilesX, (x0 + width) / 8u); ++x) links.Link(x, y);
            }
        };
        for (std::size_t plane = 0; plane < published.size(); ++plane) {
            if (!published[plane].owned) continue;
            const auto& shape = planeGeometry(plane);
            const auto base = planeAddress(plane);
            const auto end = base + planeBytes(plane);
            for (const auto& range : ranges) {
                const auto from = std::max(range.address, base);
                const auto to = std::min(range.address + range.bytes, end);
                if (from >= to) continue;
                for (auto offset = (from - base) / shape.elementBytes * shape.elementBytes; offset < to - base;) {
                    const auto block = offset >> 16u;
                    const auto blockEnd = (block + 1u) << 16u;
                    if ((offset & 65535u) == 0u && blockEnd <= to - base) {
                        linkRect(static_cast<std::uint32_t>(block % shape.blocksPerRow) << shape.widthLog2, static_cast<std::uint32_t>(block / shape.blocksPerRow) << shape.heightLog2, 1u << shape.widthLog2, 1u << shape.heightLog2);
                        offset = blockEnd;
                        continue;
                    }
                    if (const auto pixel = pixelAt(shape, offset)) links.Link(pixel->first / 8u, pixel->second / 8u);
                    offset += shape.elementBytes;
                }
            }
        }
        const auto htileBase = target.htileAddress;
        const auto htileEnd = htileBase + htileBytes();
        for (const auto& range : ranges) {
            const auto from = std::max(range.address, htileBase);
            const auto to = std::min(range.address + range.bytes, htileEnd);
            if (from >= to) continue;
            htileTable();
            links.htileAsked = true;
            links.htile.emplace_back(htileBase + ((from - htileBase) & ~std::uint64_t{3}), htileBase + ((to - htileBase + 3u) & ~std::uint64_t{3}));
            for (auto offset = (from - htileBase) & ~std::uint64_t{3}; offset < to - htileBase;) {
                const auto block = offset / HtileBlockBytes;
                const auto blockEnd = (block + 1u) * HtileBlockBytes;
                if (offset % HtileBlockBytes == 0u && blockEnd <= to - htileBase) {
                    linkRect(static_cast<std::uint32_t>(block % htilePitch()) * HtileBlockWidth, static_cast<std::uint32_t>(block / htilePitch()) * HtileBlockHeight, HtileBlockWidth, HtileBlockHeight);
                    offset = blockEnd;
                    continue;
                }
                if (const auto tile = tileAtHtile(offset)) links.Link(tile->first, tile->second);
                offset += 4u;
            }
        }
        if (links.count == 0u) {
            MergeRuns(links.htile);
            return links;
        }
        for (std::size_t plane = 0; plane < published.size(); ++plane) {
            if (!published[plane].owned) continue;
            const auto& shape = planeGeometry(plane);
            const auto base = planeAddress(plane);
            const auto blockRows = (target.extent.height + (1u << shape.heightLog2) - 1u) >> shape.heightLog2;
            auto& full = links.fullBlocks[plane];
            full.assign(static_cast<std::size_t>(shape.blocksPerRow) * blockRows, 0u);
            std::vector<std::uint64_t> elements;
            for (std::uint32_t blockY = 0; blockY < blockRows; ++blockY) {
                for (std::uint32_t blockX = 0; blockX < shape.blocksPerRow; ++blockX) {
                    const auto firstX = (blockX << shape.widthLog2) / 8u;
                    const auto firstY = (blockY << shape.heightLog2) / 8u;
                    const auto lastX = std::min(links.tilesX, ((blockX + 1u) << shape.widthLog2) / 8u);
                    const auto lastY = std::min(links.tilesY, ((blockY + 1u) << shape.heightLog2) / 8u);
                    std::size_t inside = 0;
                    std::size_t linked = 0;
                    for (auto tileY = firstY; tileY < lastY; ++tileY) {
                        for (auto tileX = firstX; tileX < lastX; ++tileX) {
                            ++inside;
                            linked += links.Linked(tileX, tileY) ? 1u : 0u;
                        }
                    }
                    if (linked == 0u) continue;
                    const auto block = static_cast<std::uint64_t>(blockY) * shape.blocksPerRow + blockX;
                    if (linked == inside) {
                        full[block] = 1u;
                        links.planes[plane].emplace_back(base + block * 65536u, base + (block + 1u) * 65536u);
                        continue;
                    }
                    for (auto tileY = firstY; tileY < lastY; ++tileY) {
                        for (auto tileX = firstX; tileX < lastX; ++tileX) {
                            if (!links.Linked(tileX, tileY)) continue;
                            for (auto y = tileY * 8u; y < std::min(target.extent.height, tileY * 8u + 8u); ++y) {
                                for (auto x = tileX * 8u; x < std::min(target.extent.width, tileX * 8u + 8u); ++x) {
                                    for (std::uint32_t sample = 0; sample < target.samples; ++sample) elements.push_back(elementOffset(shape, x, y, sample));
                                }
                            }
                        }
                    }
                }
            }
            std::sort(elements.begin(), elements.end());
            elements.erase(std::unique(elements.begin(), elements.end()), elements.end());
            for (const auto offset : elements) links.planes[plane].emplace_back(base + offset, base + offset + shape.elementBytes);
            MergeRuns(links.planes[plane]);
            ByteRuns covered;
            std::uint64_t checked = ~std::uint64_t{0};
            for (const auto& [from, to] : links.planes[plane]) {
                for (auto subpage = (from - base) / ResidencySubpageBytes; subpage * ResidencySubpageBytes < to - base; ++subpage) {
                    if (subpage == checked) continue;
                    checked = subpage;
                    if (!planeSubpageCovered(links, plane, subpage * ResidencySubpageBytes)) continue;
                    links.planeSubpages[plane].push_back(base + subpage * ResidencySubpageBytes);
                    covered.emplace_back(base + subpage * ResidencySubpageBytes, base + (subpage + 1u) * ResidencySubpageBytes);
                }
            }
            links.planes[plane].insert(links.planes[plane].end(), covered.begin(), covered.end());
            MergeRuns(links.planes[plane]);
        }
        htileTable();
        const auto htileRows = (target.extent.height + HtileBlockHeight - 1u) / HtileBlockHeight;
        links.fullHtileBlocks.assign(static_cast<std::size_t>(htilePitch()) * htileRows, 0u);
        for (std::uint32_t blockY = 0; blockY < htileRows; ++blockY) {
            for (std::uint32_t blockX = 0; blockX < htilePitch(); ++blockX) {
                const auto firstX = blockX * (HtileBlockWidth / 8u);
                const auto firstY = blockY * (HtileBlockHeight / 8u);
                const auto lastX = std::min(links.tilesX, firstX + HtileBlockWidth / 8u);
                const auto lastY = std::min(links.tilesY, firstY + HtileBlockHeight / 8u);
                std::size_t inside = 0;
                std::size_t linked = 0;
                for (auto tileY = firstY; tileY < lastY; ++tileY) {
                    for (auto tileX = firstX; tileX < lastX; ++tileX) {
                        ++inside;
                        linked += links.Linked(tileX, tileY) ? 1u : 0u;
                    }
                }
                if (linked == 0u) continue;
                const auto block = static_cast<std::uint64_t>(blockY) * htilePitch() + blockX;
                if (linked == inside) {
                    links.fullHtileBlocks[block] = 1u;
                    links.htile.emplace_back(htileBase + block * HtileBlockBytes, htileBase + (block + 1u) * HtileBlockBytes);
                    continue;
                }
                for (auto tileY = firstY; tileY < lastY; ++tileY) {
                    for (auto tileX = firstX; tileX < lastX; ++tileX) {
                        if (!links.Linked(tileX, tileY)) continue;
                        const auto offset = HtileWordOffset(*FindHtileEquation(target.samples), target.extent.width, tileX * 8u, tileY * 8u, target.pipeBankXor);
                        links.htile.emplace_back(htileBase + offset, htileBase + offset + 4u);
                    }
                }
            }
        }
        MergeRuns(links.htile);
        ByteRuns covered;
        std::uint64_t checked = ~std::uint64_t{0};
        for (const auto& [from, to] : links.htile) {
            for (auto subpage = (from - htileBase) / ResidencySubpageBytes; subpage * ResidencySubpageBytes < to - htileBase; ++subpage) {
                if (subpage == checked) continue;
                checked = subpage;
                if (!htileSubpageCovered(links, subpage * ResidencySubpageBytes)) continue;
                links.htileSubpages.push_back(htileBase + subpage * ResidencySubpageBytes);
                covered.emplace_back(htileBase + subpage * ResidencySubpageBytes, htileBase + (subpage + 1u) * ResidencySubpageBytes);
            }
        }
        links.htile.insert(links.htile.end(), covered.begin(), covered.end());
        MergeRuns(links.htile);
        return links;
    }

    bool planeSubpageCovered(const TileLinks& links, std::size_t plane, std::uint64_t offset) const {
        const auto& shape = geometry[plane];
        const auto block = offset >> 16u;
        if (block < links.fullBlocks[plane].size() && links.fullBlocks[plane][block] != 0u) return true;
        for (auto at = offset; at < offset + ResidencySubpageBytes; at += shape.elementBytes) {
            const auto pixel = pixelAt(shape, at);
            if (pixel && !links.Linked(pixel->first / 8u, pixel->second / 8u)) return false;
        }
        return true;
    }

    bool htileSubpageCovered(const TileLinks& links, std::uint64_t offset) const {
        const auto block = offset / HtileBlockBytes;
        if (block < links.fullHtileBlocks.size() && links.fullHtileBlocks[block] != 0u) return true;
        for (auto at = offset; at < offset + ResidencySubpageBytes; at += 4u) {
            const auto tile = tileAtHtile(at);
            if (tile && !links.Linked(tile->first, tile->second)) return false;
        }
        return true;
    }

    void Retile(std::span<const ResidencyRange> ranges) override {
        auto* residency = ResidencyDriver::Active();
        auto* recorder = RecorderFor(context);
        Require(residency != nullptr && recorder != nullptr, "depth surface write-back needs the residency driver and an active recorder");
        TileLinks links;
        if (htileOwned) links = linkTiles(ranges);
        struct PlaneRanges {
            std::size_t plane;
            std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        };
        std::vector<PlaneRanges> work;
        for (std::size_t plane = 0; plane < published.size(); ++plane) {
            if (!published[plane].owned) continue;
            const auto begin = planeAddress(plane);
            const auto end = begin + planeBytes(plane);
            PlaneRanges entry{plane, {}};
            for (const auto& range : ranges) {
                const auto from = std::max(range.address, begin);
                const auto to = std::min(range.address + range.bytes, end);
                if (from < to) entry.ranges.emplace_back(from, to);
            }
            entry.ranges.insert(entry.ranges.end(), links.planes[plane].begin(), links.planes[plane].end());
            MergeRuns(entry.ranges);
            if (entry.ranges.empty()) continue;
            preparePlane(plane);
            work.push_back(std::move(entry));
        }
        if (work.empty() && !links.htileAsked) return;
        auto& publisher = PublisherFor(context);
        const auto collectedUnchanged = [](const GuestMemory::UnchangedQuery& range) {
            return GuestMemory::CollectWritesUncached(range.address, range.bytes) != 0u && GuestMemory::UnchangedSince(range.address, range.bytes, range.generation);
        };
        const bool overwritten = target.samples == 1u && OverwrittenInMemory();
        const bool sampledUnchanged = samplingActive && collectedUnchanged(sampling.initial[0]);
        const GuestMemory::UnchangedQuery htileRange{target.htileAddress, static_cast<std::size_t>(htileBytes()), htileGeneration};
        const bool expandHtile = (links.count != 0u || links.htileAsked) && (samplingActive ? clearGeneration == 0u && collectedUnchanged(sampling.initial[1]) : collectedUnchanged(htileRange));
        recorder->DeclareSection({{image, true}, {view, true}, residency->SectionAccess(false)});
        const auto commands = recorder->Commands();
        const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
        const auto copy = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
        const auto copyPieces = [&](std::size_t plane, const std::vector<MirrorPiece>& pieces, bool toGuest) {
            const auto base = planeAddress(plane);
            const auto tiled = published[plane].tiled->Handle();
            for (const auto& piece : pieces) {
                const auto length = piece.end - piece.begin;
                if (piece.buffer != VK_NULL_HANDLE) {
                    const VkBufferCopy part = toGuest ? VkBufferCopy{piece.begin - base, piece.offset, length} : VkBufferCopy{piece.offset, piece.begin - base, length};
                    copy(commands, toGuest ? tiled : piece.buffer, toGuest ? piece.buffer : tiled, 1, &part);
                    continue;
                }
                const auto* import = HostImportFor(context, piece.begin, static_cast<std::size_t>(length));
                Require(import != nullptr, "depth surface write-back: the plane has no host import");
                const VkBufferCopy part = toGuest ? VkBufferCopy{piece.begin - base, piece.begin - import->base, length} : VkBufferCopy{piece.begin - import->base, piece.begin - base, length};
                copy(commands, toGuest ? tiled : import->buffer, toGuest ? import->buffer : tiled, 1, &part);
            }
        };
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        std::deque<std::pair<MirrorSpan, std::vector<MirrorPiece>>> sources;
        for (const auto& entry : work) {
            for (const auto& [from, to] : entry.ranges) {
                auto& [span, pieces] = sources.emplace_back(std::piecewise_construct, std::forward_as_tuple(residency, from, to, ResidencyAccess::Read), std::forward_as_tuple());
                pieces = span.Pieces(from, to);
                copyPieces(entry.plane, pieces, false);
            }
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        for (const auto& entry : work) {
            const auto elementBytes = planeElementBytes(entry.plane);
            const auto blockLog2 = DepthBlockLog2(elementBytes, target.samples);
            DepthPublishPush push{};
            push.equation = *planeEquation(entry.plane);
            push.width = target.extent.width;
            push.height = target.extent.height;
            push.elementBytes = elementBytes;
            push.blocksPerRow = (target.extent.width + (1u << blockLog2.first) - 1u) >> blockLog2.first;
            push.blockWidthLog2 = blockLog2.first;
            push.blockHeightLog2 = blockLog2.second;
            push.pipeBankXor = target.pipeBankXor;
            push.slice = target.samples == 1u ? target.firstSlice : 0u;
            publisher.Dispatch(commands, entry.plane == StencilPlane, target.samples != 1u, published[entry.plane].set, push, target.samples);
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        std::deque<std::pair<MirrorSpan, std::vector<MirrorPiece>>> spans;
        std::uint64_t moved = 0;
        for (const auto& entry : work) {
            for (const auto& [from, to] : entry.ranges) {
                auto& [span, pieces] = spans.emplace_back(std::piecewise_construct, std::forward_as_tuple(residency, from, to, ResidencyAccess::Write), std::forward_as_tuple());
                pieces = span.Pieces(from, to);
                copyPieces(entry.plane, pieces, true);
                moved += to - from;
            }
        }
        if (expandHtile) {
            const auto fill = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
            for (const auto& [from, to] : links.htile) {
                auto& [span, pieces] = spans.emplace_back(std::piecewise_construct, std::forward_as_tuple(residency, from, to, ResidencyAccess::Write), std::forward_as_tuple());
                pieces = span.Pieces(from, to);
                for (const auto& piece : pieces) {
                    const auto length = piece.end - piece.begin;
                    if (piece.buffer != VK_NULL_HANDLE) {
                        fill(commands, piece.buffer, piece.offset, length, ExpandedHtileWord());
                        continue;
                    }
                    const auto* import = HostImportFor(context, piece.begin, static_cast<std::size_t>(length));
                    Require(import != nullptr, "depth surface write-back: the HTILE has no host import");
                    fill(commands, import->buffer, piece.begin - import->base, length, ExpandedHtileWord());
                }
                moved += to - from;
            }
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::Copy, 4);
        for (const auto& entry : work) recorder->Keep(published[entry.plane].tiled);
        if (owned != nullptr) recorder->Keep(owned);
        recorder->EndGpuTiming(timing, moved);
        for (const auto& [span, pieces] : spans) {
            span.NoteWritten(pieces);
            for (const auto& piece : pieces) GuestMemory::MarkWritten(piece.begin, static_cast<std::size_t>(piece.end - piece.begin));
        }
        if (target.samples == 1u && !overwritten) NoteMemory();
        if (sampledUnchanged) sampling.initial[0].generation = GuestMemory::CollectWritesUncached(sampling.initial[0].address, sampling.initial[0].bytes);
        if (expandHtile) {
            htileGeneration = GuestMemory::CollectWritesUncached(target.htileAddress, static_cast<std::size_t>(htileBytes()));
            if (samplingActive) sampling.initial[1].generation = htileGeneration;
        }
        for (const auto& subpages : links.planeSubpages) {
            for (const auto subpage : subpages) residency->Manager().ClearStale(subpage, ResidencySubpageBytes, this);
        }
        if (expandHtile) {
            for (const auto subpage : links.htileSubpages) residency->Manager().ClearStale(subpage, ResidencySubpageBytes, this);
        }
    }

    bool UnormStencilView(const GuestTextureResource& resource) const {
        return target.samples == 1 && target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress && ResolveTextureFormat(resource.format) == VK_FORMAT_R8_UNORM;
    }

    bool SampledAccepts(std::span<const std::uint32_t> words, const GuestTextureResource& resource) const {
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        const bool oneSlice = target.samples != 1 ? resource.dimension == TextureDimension::k2DMsaa : resource.dimension == TextureDimension::k2D || (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray == 0);
        return (ResolveTextureFormat(resource.format) == expected || depthBits || UnormStencilView(resource)) && oneSlice && resource.width == target.extent.width && resource.height == target.extent.height && resource.baseLevel == 0 && resource.lastLevel == 0 && resource.baseArray == 0;
    }

    std::shared_ptr<Texture> Sampled(std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, bool drawn = false) {
        CheckCurrent();
        CheckResource(resource);
        std::array<std::uint32_t, 12> key{};
        std::copy_n(words.begin(), std::min<std::size_t>(words.size(), 8), key.begin());
        key[8] = components.r;
        key[9] = components.g;
        key[10] = components.b;
        key[11] = components.a;
        if (const auto found = textures.find(key); found != textures.end()) return found->second;
        const bool multisampled = target.samples != 1;
        Require(!multisampled || sampleable, "the device cannot sample the native depth target's format and sample count");
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const auto format = ResolveTextureFormat(resource.format);
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        if (multisampled) {
            Require(!stencil && !depthBits && format == expected, "native multisampled depth sampling needs a numeric depth-plane format");
            Require(resource.samples == target.samples, "the sampled depth descriptor's sample count differs from its native target");
            Require(hostFormat == target.format, "sampling a native multisampled depth plane that the device stores in another depth format is not implemented");
        }
        if (stencil && UnormStencilView(resource) && SampledAccepts(words, resource)) {
            return unormStencil(key, components, resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, drawn);
        }
        if (!SampledAccepts(words, resource)) {
            char text[448];
            std::snprintf(text, sizeof(text), "AGC graphics: sampling the %s plane of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), tile mode %u, dimension %d, levels %u-%u, slice %u is not implemented (T# %08x %08x %08x %08x %08x %08x %08x %08x)",
                          stencil ? "stencil" : "depth", static_cast<unsigned long long>(target.address), target.extent.width, target.extent.height, static_cast<int>(target.format), resource.width, resource.height, resource.format, static_cast<int>(format),
                          static_cast<unsigned>(resource.tileMode), static_cast<int>(resource.dimension), resource.baseLevel, resource.lastLevel, resource.baseArray, key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]);
            throw std::runtime_error(text);
        }
        const auto viewType = resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        auto texture = std::make_shared<Texture>(context, image, hostFormat, stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, components, viewType, 0, owned);
        if (multisampled) {
            Require(samplingActive, "native multisample depth sampling requires its creation-time owner");
            texture->MarkRefreshedPerUse();
        }
        textures.emplace(key, texture);
        return texture;
    }

    void CopyAspectFrom(const DepthSurface& source, VkImageAspectFlagBits aspect) {
        pendingClear &= ~static_cast<VkImageAspectFlags>(aspect);
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        if (recorder != nullptr) recorder->DeclareSection({{image, true}, {view, true}, {source.image, true}, {source.view, true}});
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        constexpr VkAccessFlags all = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, all, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy region{};
        region.srcSubresource = {static_cast<VkImageAspectFlags>(aspect), 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.extent = {target.extent.width, target.extent.height, 1};
        context.Function<PFN_vkCmdCopyImage>("vkCmdCopyImage")(commands, source.image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, all);
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
    }

    bool Holds(const Texture* texture) const {
        const auto held = [&](const auto& entry) { return entry.second.get() == texture; };
        return std::any_of(textures.begin(), textures.end(), held) || std::any_of(stencilCopy.textures.begin(), stencilCopy.textures.end(), held);
    }

    std::uint64_t written = ++depthWrites;
    bool retired = false;

    bool Overlaps(std::uint64_t address, std::uint64_t bytes) const {
        const auto end = address + bytes;
        if (address < target.address + depthBytes() && target.address < end) return true;
        return target.stencilAddress != 0 && address < target.stencilAddress + stencilBytes() && target.stencilAddress < end;
    }

    void NoteMemory() {
        depthMemory = GuestMemory::CollectWrites(target.address, static_cast<std::size_t>(depthBytes()));
        stencilMemory = target.stencilAddress != 0 ? GuestMemory::CollectWrites(target.stencilAddress, static_cast<std::size_t>(stencilBytes())) : 0;
    }

    bool OverwrittenInMemory() const {
        GuestMemory::CollectWrites(target.address, static_cast<std::size_t>(depthBytes()));
        if (GuestMemory::WrittenSince(target.address, static_cast<std::size_t>(depthBytes()), depthMemory)) return true;
        if (target.stencilAddress == 0) return false;
        GuestMemory::CollectWrites(target.stencilAddress, static_cast<std::size_t>(stencilBytes()));
        return GuestMemory::WrittenSince(target.stencilAddress, static_cast<std::size_t>(stencilBytes()), stencilMemory);
    }

    VkImageAspectFlags pendingClear = 0;
    float clearDepth = 0.0f;
    std::uint8_t clearStencil = 0;

    bool StorageViewMatches(std::span<const std::uint32_t> words, const GuestTextureResource& resource) const {
        if (resource.width != target.extent.width || resource.height != target.extent.height) return false;
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        return ResolveTextureFormat(resource.format) == expected || depthBits;
    }

    void RequireStorable(const GuestTextureResource& resource, std::uint32_t mip) const {
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        Require(target.samples == 1, "storage image access to a multisampled depth surface is not implemented");
        if (!stencil && hostFormat != target.format) {
            char text[224];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the depth plane of depth surface 0x%llx (vk format %d) stored as vk format %d is not implemented", static_cast<unsigned long long>(target.address), static_cast<int>(target.format), static_cast<int>(hostFormat));
            throw std::runtime_error(text);
        }
        if (mip != 0 || resource.baseLevel != 0) {
            char text[224];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image access to level %u of the %s plane of depth surface 0x%llx is not implemented", mip, stencil ? "stencil" : "depth", static_cast<unsigned long long>(target.address));
            throw std::runtime_error(text);
        }
    }

    std::shared_ptr<StorageTexture> Storage(std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip) {
        if (!StorageViewMatches(words, resource)) return nullptr;
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const bool oneLayer = resource.dimension == TextureDimension::k2DArray && resource.baseArray == 0 && resource.depthOrLastArray == 0;
        if (resource.dimension != TextureDimension::k2D && !oneLayer) {
            char text[256];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image access to depth/stencil surface 0x%llx is not implemented (storage=%ux%u guest format=%u mip=%u dimension=%d tile=%d)", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, mip, static_cast<int>(resource.dimension), static_cast<int>(resource.tileMode));
            throw std::runtime_error(text);
        }
        RequireStorable(resource, mip);
        ApplyFastClear();
        auto& plane = stencil ? stencilPlane : depthPlane;
        if (plane.storage == nullptr || plane.storage->Descriptor().format != resource.format || plane.storage->Descriptor().tileMode != resource.tileMode || plane.storage->Descriptor().baseAddress != resource.baseAddress || plane.storage->Descriptor().dimension != resource.dimension) {
            std::erase_if(plane.replaced, [](const auto& old) { return old.expired(); });
            if (plane.storage != nullptr) plane.replaced.push_back(plane.storage);
            plane.storage = StorageTexture::Create(context, *context.detiler, resource, 0);
        }
        copyPlane(plane, plane.storage, stencil, true);
        return plane.storage;
    }

    VkImageAspectFlags StoreBack(const StorageTexture& candidate) {
        for (const bool stencil : {false, true}) {
            auto& plane = stencil ? stencilPlane : depthPlane;
            if (plane.storage.get() == &candidate) {
                copyPlane(plane, plane.storage, stencil, false);
                written = ++depthWrites;
                return stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
            }
            if (std::none_of(plane.replaced.begin(), plane.replaced.end(), [&](const auto& old) { return old.lock().get() == &candidate; })) continue;
            char text[288];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image writes to the %s plane of depth surface 0x%llx through a view that another view of the plane (guest format %u, tile mode %d) replaced are not implemented", stencil ? "stencil" : "depth", static_cast<unsigned long long>(target.address), plane.storage->Descriptor().format, static_cast<int>(plane.storage->Descriptor().tileMode));
            throw std::runtime_error(text);
        }
        return 0u;
    }

    VkImageAspectFlags StoreBackCovering(const StorageTexture& candidate) {
        for (const bool stencil : {false, true}) {
            if (!CoversPlane(candidate, stencil)) continue;
            auto& plane = stencil ? stencilPlane : depthPlane;
            const auto source = std::const_pointer_cast<StorageTexture>(candidate.shared_from_this());
            copyPlane(plane, source, stencil, false);
            written = ++depthWrites;
            return stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
        }
        return 0u;
    }

    void TakeStorage() {
        if (target.samples != 1 || hostFormat != target.format) return;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto bytes = DepthSliceBytes(target.extent, d16 ? 2u : 4u);
        auto storage = StorageTexture::FindPending(target.address, bytes);
        if (storage == nullptr) storage = StorageTexture::FindLive(target.address, bytes);
        if (storage == nullptr || !storageFits(target, storage->Descriptor())) return;
        storage->Refresh();
        if (depthPlane.staging == nullptr) depthPlane.staging = std::make_unique<DeviceBuffer>(context, static_cast<std::size_t>(target.extent.width) * target.extent.height * (d16 ? 2u : 4u), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        if (recorder != nullptr) recorder->DeclareSection({{image, true}, {view, true}, {storage->Image(), true}});
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        if (recorder != nullptr) recorder->Keep(storage);
        VkBufferImageCopy depthRegion{0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
        VkBufferImageCopy colorRegion = depthRegion;
        colorRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, storage->Image(), VK_IMAGE_LAYOUT_GENERAL, depthPlane.staging->Handle(), 1, &colorRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, depthPlane.staging->Handle(), image, VK_IMAGE_LAYOUT_GENERAL, 1, &depthRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
    }

    const Context context;
    const DepthTarget target;
    const VkFormat hostFormat;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::shared_ptr<std::atomic<bool>> active = std::make_shared<std::atomic<bool>>(true);
    bool sampleable = true;
    bool samplingActive = false;
    DepthSamplingState sampling;
    std::uint64_t clearGeneration = 0u;
    std::shared_ptr<OwnedImage> owned;
    bool residencyOwned = false;
    std::uint64_t ownedSerial = 0;
    struct PublishedPlane {
        bool owned = false;
        VkImageView view = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::shared_ptr<DeviceBuffer> tiled;
    };
    std::array<PublishedPlane, 2> published{};
    std::shared_ptr<std::atomic<bool>> depthPublished = std::make_shared<std::atomic<bool>>(false);
    bool htileOwned = false;
    std::uint64_t htileGeneration = 0;
    std::shared_ptr<std::atomic<bool>> htilePublished = std::make_shared<std::atomic<bool>>(false);
    std::array<PlaneGeometry, 2> geometry{};
    std::vector<std::uint32_t> htileInverse;

private:
    struct PlaneStorage {
        std::shared_ptr<StorageTexture> storage;
        std::vector<std::weak_ptr<StorageTexture>> replaced;
        std::unique_ptr<DeviceBuffer> staging;
    };
    struct StencilCopy {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::unique_ptr<DeviceBuffer> staging;
        std::map<std::array<std::uint32_t, 12>, std::shared_ptr<Texture>> textures;
        std::uint64_t written = 0;
        bool filled = false;
    };
    std::map<std::array<std::uint32_t, 12>, std::shared_ptr<Texture>> textures;
    PlaneStorage depthPlane;
    PlaneStorage stencilPlane;
    std::uint64_t depthMemory = 0;
    std::uint64_t stencilMemory = 0;
    StencilCopy stencilCopy;

    std::shared_ptr<Texture> unormStencil(const std::array<std::uint32_t, 12>& key, VkComponentMapping components, VkImageViewType viewType, bool drawn) {
        if (stencilCopy.image == VK_NULL_HANDLE) {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = VK_FORMAT_R8_UNORM;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            try {
                Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &stencilCopy.image), "vkCreateImage stencil copy");
                VkMemoryRequirements requirements{};
                context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, stencilCopy.image, &requirements);
                VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &stencilCopy.memory), "vkAllocateMemory stencil copy");
                Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, stencilCopy.image, stencilCopy.memory, 0), "vkBindImageMemory stencil copy");
                stencilCopy.staging = std::make_unique<DeviceBuffer>(context, static_cast<std::size_t>(target.extent.width) * target.extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            } catch (...) {
                releaseStencilCopy();
                throw;
            }
        }
        if (drawn || !stencilCopy.filled || stencilCopy.written != written) {
            copyStencil();
            stencilCopy.written = drawn ? 0 : written;
        }
        auto& texture = stencilCopy.textures[key];
        if (texture == nullptr) {
            texture = std::make_shared<Texture>(context, stencilCopy.image, VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, components, viewType);
            texture->MarkRefreshedPerUse();
        }
        return texture;
    }

    void copyStencil() {
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        if (recorder != nullptr) recorder->DeclareSection({{image, true}, {view, true}, {stencilCopy.image, true}});
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{0, 0, 0, {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, VK_IMAGE_LAYOUT_GENERAL, stencilCopy.staging->Handle(), 1, &region);
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = stencilCopy.filled ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = stencilCopy.image;
        toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkBufferMemoryBarrier staged{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, stencilCopy.staging->Handle(), 0, VK_WHOLE_SIZE};
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &staged, 1, &toTransfer);
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, stencilCopy.staging->Handle(), stencilCopy.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier toGeneral = toTransfer;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        stencilCopy.filled = true;
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
    }

    void releaseStencilCopy() noexcept {
        stencilCopy.textures.clear();
        stencilCopy.staging.reset();
        if (stencilCopy.image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, stencilCopy.image, nullptr);
        if (stencilCopy.memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, stencilCopy.memory, nullptr);
        stencilCopy = {};
    }

    std::uint64_t depthBytes() const {
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        return DepthSliceBytes(target.extent, d16 ? 2u : 4u);
    }

    std::uint64_t stencilBytes() const {
        return DepthSliceBytes(target.extent, 1u);
    }

    bool CoversPlane(const StorageTexture& storage, bool stencil) const {
        const auto& resource = storage.Descriptor();
        const auto address = stencil ? target.stencilAddress : target.address;
        if (address == 0 || resource.baseAddress != address || resource.width != target.extent.width || resource.height != target.extent.height || resource.baseLevel != 0 || resource.mipCount != 1) return false;
        if (resource.dimension != TextureDimension::k2D && (resource.dimension != TextureDimension::k2DArray || resource.baseArray != 0 || resource.depthOrLastArray != 0)) return false;
        if (target.samples != 1 || (!stencil && hostFormat != target.format)) return false;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        return ResolveTextureFormat(resource.format) == (stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT);
    }

    void copyPlane(PlaneStorage& plane, const std::shared_ptr<StorageTexture>& storage, bool stencil, bool toStorage) {
        auto* recorder = Recorder::Active();
        Require(recorder != nullptr, "depth surface storage copies need an active recorder");
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto bytes = static_cast<std::size_t>(target.extent.width) * target.extent.height * (stencil ? 1u : d16 ? 2u : 4u);
        if (plane.staging == nullptr) plane.staging = std::make_unique<DeviceBuffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        recorder->DeclareSection({{image, true}, {view, true}, {storage->Image(), true}});
        const auto commands = recorder->Commands();
        recorder->Keep(storage);
        const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
        VkBufferImageCopy planeRegion{0, 0, 0, {stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
        VkBufferImageCopy colorRegion = planeRegion;
        colorRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        const auto sourceImage = toStorage ? image : storage->Image();
        const auto destinationImage = toStorage ? storage->Image() : image;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, sourceImage, VK_IMAGE_LAYOUT_GENERAL, plane.staging->Handle(), 1, toStorage ? &planeRegion : &colorRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, plane.staging->Handle(), destinationImage, VK_IMAGE_LAYOUT_GENERAL, 1, toStorage ? &colorRegion : &planeRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
        recorder->EndGpuTiming(timing, bytes);
    }

    void release() noexcept {
        depthPlane = {};
        stencilPlane = {};
        releaseStencilCopy();
        textures.clear();
        for (auto& plane : published) {
            if (plane.view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, plane.view, nullptr);
            plane.view = VK_NULL_HANDLE;
            plane.set = VK_NULL_HANDLE;
            plane.tiled.reset();
        }
        if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
        if (owned != nullptr) owned.reset();
        else {
            if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
            if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        }
        view = VK_NULL_HANDLE;
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
};

struct DepthPlaneSlice {
    VkImage image = VK_NULL_HANDLE;
    std::uint64_t written = 0;
    VkImageView view = VK_NULL_HANDLE;
};

class DepthPlaneCopy {
public:
    DepthPlaneCopy(const Context& context, VkExtent2D extent, VkFormat format, std::uint32_t layers) : context(context), extent(extent), format(format), layers(layers), guestLayers(layers), surfaceLayers(layers) {
        this->context.bufferPool.reset();
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.flags = layers % 6u == 0 && extent.width == extent.height ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {extent.width, extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = layers;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        try {
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage depth plane copy");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory depth plane copy");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory depth plane copy");
            stagingSize = static_cast<std::size_t>(sliceBytes()) * layers;
            staging = std::make_unique<DeviceBuffer>(context, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        } catch (...) {
            release();
            throw;
        }
    }
    ~DepthPlaneCopy() { release(); }
    DepthPlaneCopy(const DepthPlaneCopy&) = delete;
    DepthPlaneCopy& operator=(const DepthPlaneCopy&) = delete;

    std::shared_ptr<Texture> Refresh(std::span<const DepthPlaneSlice> slices, const GuestTextureResource& resource, VkComponentMapping components, VkImageViewType viewType, std::uint32_t baseLayer) {
        Require(slices.size() == layers, "depth plane copy slices do not match its layers");
        const auto geometry = DescribeSurface(resource);
        const bool guestFill = std::any_of(slices.begin(), slices.end(), [](const auto& slice) { return slice.image == VK_NULL_HANDLE; });
        if (guestFill && (geometry.layers != layers || geometry.sliceLinearBytes < sliceBytes() || geometry.mips.empty())) {
            char text[320];
            std::snprintf(text, sizeof(text), "AGC graphics: depth plane copy geometry does not match its layers (copy %ux%u, %u layers, %llu bytes a slice; texture %ux%u guest format %u, dimension %d, tile mode %u, %u layers, %llu bytes a slice, %zu mips)", extent.width, extent.height, layers, static_cast<unsigned long long>(sliceBytes()), resource.width, resource.height, resource.format, static_cast<int>(resource.dimension), resource.tileMode, geometry.layers, static_cast<unsigned long long>(geometry.sliceLinearBytes), geometry.mips.size());
            throw std::runtime_error(text);
        }
        const auto stagingBytes = static_cast<std::size_t>(geometry.sliceLinearBytes) * layers;
        if (stagingBytes > stagingSize) {
            retiredStaging.push_back(std::move(staging));
            staging = std::make_unique<DeviceBuffer>(context, stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            stagingSize = stagingBytes;
        }
        std::vector<std::shared_ptr<Buffer>> uploads;
        std::vector<std::uint32_t> uploadLayers;
        std::vector<std::uint32_t> copiedLayers;
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            auto& held = guestLayers[layer];
            auto& surface = surfaceLayers[layer];
            if (slices[layer].image != VK_NULL_HANDLE) {
                held = {};
                if (slices[layer].written != 0 && surface.image == slices[layer].image && surface.written == slices[layer].written) continue;
                surface = slices[layer];
                copiedLayers.push_back(layer);
                continue;
            }
            surface = {};
            const GuestLayer current{resource.baseAddress + geometry.GuestLayerOffset(layer), geometry.layerBytes, resource.tileMode, geometry.thick, GuestMemory::CollectWrites(resource.baseAddress + geometry.GuestLayerOffset(layer), static_cast<std::size_t>(geometry.layerBytes))};
            if (held.generation != 0 && held.address == current.address && held.bytes == current.bytes && held.tileMode == current.tileMode && held.thick == current.thick && !Recorder::SnapshotWriteOverlaps(current.address, static_cast<std::size_t>(current.bytes)) && GuestMemory::UnchangedSince(current.address, static_cast<std::size_t>(current.bytes), held.generation)) {
                held.generation = current.generation;
                continue;
            }
            auto upload = std::make_shared<Buffer>(context, static_cast<std::size_t>(geometry.layerBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            GuestMemory::ReadCommitted(current.address, upload->Bytes().first(static_cast<std::size_t>(geometry.layerBytes)));
            uploads.push_back(std::move(upload));
            uploadLayers.push_back(layer);
            held = current;
        }
        if (!copiedLayers.empty() || !uploads.empty()) record(slices, resource, geometry, copiedLayers, uploads, uploadLayers);
        const std::array<std::uint32_t, 6> key{static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a), static_cast<std::uint32_t>(viewType), baseLayer};
        auto& texture = textures[key];
        if (texture == nullptr) {
            texture = std::make_shared<Texture>(context, image, format, aspect(), components, viewType, baseLayer);
            texture->MarkRefreshedPerUse();
        }
        return texture;
    }

    const Context context;
    const VkExtent2D extent;
    const VkFormat format;
    const std::uint32_t layers;

private:
    struct GuestLayer {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        TextureTileMode tileMode = TextureTileMode::kLinear;
        bool thick = false;
        std::uint64_t generation = 0;
    };

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::unique_ptr<DeviceBuffer> staging;
    std::size_t stagingSize = 0;
    std::vector<std::unique_ptr<DeviceBuffer>> retiredStaging;
    std::map<std::array<std::uint32_t, 6>, std::shared_ptr<Texture>> textures;
    std::vector<GuestLayer> guestLayers;
    std::vector<DepthPlaneSlice> surfaceLayers;
    bool filled = false;

    void record(std::span<const DepthPlaneSlice> slices, const GuestTextureResource& resource, const SurfaceGeometry& geometry, std::span<const std::uint32_t> copiedLayers, std::span<const std::shared_ptr<Buffer>> uploads, std::span<const std::uint32_t> uploadLayers) {
        if (!uploads.empty()) {
            Require(context.detiler != nullptr, "depth plane copy requires a texture detiler");
            context.detiler->BeginBatch();
        }
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        std::vector<Recorder::SectionObject> objects{{image, true}};
        for (const auto& slice : slices) {
            if (slice.image == VK_NULL_HANDLE) continue;
            objects.push_back({slice.image, true});
            objects.push_back({slice.view, true});
        }
        if (recorder != nullptr) recorder->DeclareSection(objects);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        std::vector<VkBufferImageCopy> regions;
        const auto& mip = geometry.mips.front();
        const auto region = [&](std::uint32_t layer, VkImageAspectFlags aspectMask, std::uint32_t imageLayer) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = geometry.LinearLayerOffset(layer) + mip.linearOffset;
            copy.bufferRowLength = mip.pitchBytes / (format == VK_FORMAT_D32_SFLOAT ? 4u : 2u);
            copy.imageSubresource = {aspectMask, 0, imageLayer, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            return copy;
        };
        for (const auto layer : copiedLayers) {
            const auto copy = region(layer, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, slices[layer].image, VK_IMAGE_LAYOUT_GENERAL, staging->Handle(), 1, &copy);
            regions.push_back(region(layer, aspect(), layer));
        }
        if (!uploads.empty()) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            for (std::size_t upload = 0; upload < uploads.size(); ++upload) {
                const auto layer = uploadLayers[upload];
                context.detiler->Dispatch(commands, resource.tileMode, format == VK_FORMAT_D32_SFLOAT ? 4u : 2u, uploads[upload]->Handle(), 0, staging->Handle(), geometry.LinearLayerOffset(layer), geometry.mips.front(), false, layer, geometry.thick);
                regions.push_back(region(layer, aspect(), layer));
            }
            if (recorder != nullptr) {
                for (const auto& upload : uploads) recorder->Keep(upload, upload->Bytes().size());
            }
        }
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = filled ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = image;
        toTransfer.subresourceRange = {aspect(), 0, 1, 0, layers};
        const VkBufferMemoryBarrier staged{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, staging->Handle(), 0, VK_WHOLE_SIZE};
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &staged, 1, &toTransfer);
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, staging->Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
        VkImageMemoryBarrier toGeneral = toTransfer;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        filled = true;
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Draw, 3);
    }

    VkDeviceSize sliceBytes() const {
        return static_cast<VkDeviceSize>(extent.width) * extent.height * (format == VK_FORMAT_D32_SFLOAT ? 4u : 2u);
    }

    VkImageAspectFlags aspect() const {
        return format == VK_FORMAT_R16_UINT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    void release() noexcept {
        textures.clear();
        staging.reset();
        if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
        if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
};

std::map<std::tuple<VkDevice, std::uint64_t, std::uint32_t, std::uint32_t, VkFormat, std::uint32_t>, std::unique_ptr<DepthPlaneCopy>>& planeCopies() {
    static auto* copies = new std::map<std::tuple<VkDevice, std::uint64_t, std::uint32_t, std::uint32_t, VkFormat, std::uint32_t>, std::unique_ptr<DepthPlaneCopy>>();
    return *copies;
}

struct LayeredSlice {
    DepthSurface* surface;
    std::uint32_t layer;
};

class LayeredDepthStorage {
public:
    LayeredDepthStorage(const Context& context, const GuestTextureResource& resource, const DepthTarget& base, std::uint32_t bytesPerTexel, std::uint32_t firstLayer, std::uint32_t layers) : context(context), base(base), extent(base.extent), bytesPerTexel(bytesPerTexel), firstLayer(firstLayer), layers(layers) {
        const auto bytes = DescribeSurface(resource).guestBytes;
        StorageTexture::FlushPending(resource.baseAddress, static_cast<std::size_t>(bytes), nullptr, "storage image creation", PublishScope::None);
        storage = StorageTexture::Create(context, *context.detiler, resource, 0);
        Require(storage->ImageLayers() == layers, "a layered depth storage view's image does not hold its layers");
        staging = std::make_unique<DeviceBuffer>(context, static_cast<std::size_t>(sliceBytes()) * layers, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    void Copy(std::span<const LayeredSlice> slices, bool toStorage) {
        if (slices.empty()) return;
        auto* recorder = Recorder::Active();
        Require(recorder != nullptr, "depth surface storage copies need an active recorder");
        std::vector<Recorder::SectionObject> objects{{storage->Image(), true}};
        for (const auto& slice : slices) {
            objects.push_back({slice.surface->image, true});
            objects.push_back({slice.surface->view, true});
        }
        recorder->DeclareSection(objects);
        const auto commands = recorder->Commands();
        recorder->Keep(storage);
        const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
        std::vector<VkBufferImageCopy> storageRegions;
        const auto region = [&](std::uint32_t layer, VkImageAspectFlags aspect, std::uint32_t imageLayer) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = sliceBytes() * layer;
            copy.imageSubresource = {aspect, 0, imageLayer, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            return copy;
        };
        for (const auto& slice : slices) storageRegions.push_back(region(slice.layer, VK_IMAGE_ASPECT_COLOR_BIT, slice.layer));
        const auto copyToBuffer = context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
        const auto copyToImage = context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        if (toStorage) {
            for (const auto& slice : slices) {
                const auto copy = region(slice.layer, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
                copyToBuffer(commands, slice.surface->image, VK_IMAGE_LAYOUT_GENERAL, staging->Handle(), 1, &copy);
            }
        } else {
            copyToBuffer(commands, storage->Image(), VK_IMAGE_LAYOUT_GENERAL, staging->Handle(), static_cast<std::uint32_t>(storageRegions.size()), storageRegions.data());
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        if (toStorage) {
            copyToImage(commands, staging->Handle(), storage->Image(), VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(storageRegions.size()), storageRegions.data());
        } else {
            for (const auto& slice : slices) {
                const auto copy = region(slice.layer, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
                copyToImage(commands, staging->Handle(), slice.surface->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
            }
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
        recorder->EndGpuTiming(timing, sliceBytes() * slices.size());
    }

    const Context context;
    const DepthTarget base;
    const VkExtent2D extent;
    const std::uint32_t bytesPerTexel;
    const std::uint32_t firstLayer;
    const std::uint32_t layers;
    std::shared_ptr<StorageTexture> storage;

private:
    std::unique_ptr<DeviceBuffer> staging;

    VkDeviceSize sliceBytes() const {
        return static_cast<VkDeviceSize>(extent.width) * extent.height * bytesPerTexel;
    }
};

std::map<std::tuple<VkDevice, std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, int, int, std::uint32_t>, std::unique_ptr<LayeredDepthStorage>>& layeredStorages() {
    static auto* storages = new std::map<std::tuple<VkDevice, std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, int, int, std::uint32_t>, std::unique_ptr<LayeredDepthStorage>>();
    return *storages;
}

bool sameSurface(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && a.samples == b.samples && (a.samples == 1u || (a.htileAddress == b.htileAddress && a.htileStencil == b.htileStencil && a.depthSwizzle == b.depthSwizzle && a.pipeBankXor == b.pipeBankXor && a.firstSlice == b.firstSlice && a.lastSlice == b.lastSlice && a.iterate256 == b.iterate256 && a.htilePipeAligned == b.htilePipeAligned && a.htileVrsEncoding == b.htileVrsEncoding && a.htileDepthOnly == b.htileDepthOnly));
}

std::mutex& surfacesMutex() {
    static std::mutex mutex;
    return mutex;
}

struct HostDepthFormats {
    std::mutex mutex;
    std::map<std::pair<VkDevice, VkPhysicalDevice>, VkFormat> chosen;
};

HostDepthFormats& hostDepthFormats() {
    static HostDepthFormats formats;
    return formats;
}

std::vector<std::unique_ptr<DepthSurface>>& surfaces() {
    static auto* list = new std::vector<std::unique_ptr<DepthSurface>>();
    return *list;
}

struct PlaneOwner {
    DepthSurface* surface = nullptr;
    std::uint64_t serial = 0;
};

using PlaneOwners = std::map<std::pair<VkDevice, std::uint64_t>, PlaneOwner>;

std::uint64_t planeBinds = 0;

PlaneOwners& depthOwners() {
    static auto* owners = new PlaneOwners();
    return *owners;
}

PlaneOwners& stencilOwners() {
    static auto* owners = new PlaneOwners();
    return *owners;
}

void takePlane(DepthSurface& surface, PlaneOwners& owners, std::uint64_t address, VkImageAspectFlagBits aspect) {
    auto& entry = owners[{surface.context.device, address}];
    auto* owner = entry.surface;
    if (owner != nullptr && owner->active->load(std::memory_order_acquire) && owner != &surface && owner->target.format == surface.target.format && owner->target.samples == surface.target.samples && owner->target.extent.width == surface.target.extent.width && owner->target.extent.height == surface.target.extent.height) {
        if ((owner->pendingClear & aspect) != 0) surface.pendingClear |= aspect;
        else surface.CopyAspectFrom(*owner, aspect);
    }
    entry = {&surface, ++planeBinds};
}

void takePlanes(DepthSurface& surface) {
    takePlane(surface, depthOwners(), surface.target.address, VK_IMAGE_ASPECT_DEPTH_BIT);
    if (surface.target.stencilAddress != 0) takePlane(surface, stencilOwners(), surface.target.stencilAddress, VK_IMAGE_ASPECT_STENCIL_BIT);
}

DepthSurface* depthSlice(VkDevice device, std::uint64_t address, const DepthTarget& base) {
    if (const auto found = depthOwners().find({device, address}); found != depthOwners().end()) {
        const auto& target = found->second.surface->target;
        if (found->second.surface->active->load(std::memory_order_acquire) && target.extent.width == base.extent.width && target.extent.height == base.extent.height && target.format == base.format) return found->second.surface;
    }
    const auto& list = surfaces();
    const auto slice = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->active->load(std::memory_order_acquire) && surface->context.device == device && !surface->retired && surface->target.address == address && surface->target.extent.width == base.extent.width && surface->target.extent.height == base.extent.height && surface->target.format == base.format; });
    return slice == list.rend() ? nullptr : slice->get();
}

DepthSurface* planeSurface(VkDevice device, std::uint64_t address) {
    const auto depth = depthOwners().find({device, address});
    const auto stencil = stencilOwners().find({device, address});
    const bool hasDepth = depth != depthOwners().end() && depth->second.surface->active->load(std::memory_order_acquire);
    const bool hasStencil = stencil != stencilOwners().end() && stencil->second.surface->active->load(std::memory_order_acquire);
    if (hasDepth && (!hasStencil || depth->second.serial > stencil->second.serial)) return depth->second.surface;
    if (hasStencil) return stencil->second.surface;
    const auto& list = surfaces();
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) {
        return surface->active->load(std::memory_order_acquire) && surface->context.device == device && !surface->retired && (surface->target.address == address || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == address));
    });
    return found == list.rend() ? nullptr : found->get();
}

void retire(DepthSurface& surface) {
    surface.DisownPlanes();
    surface.retired = true;
    for (auto* owners : {&depthOwners(), &stencilOwners()}) {
        std::erase_if(*owners, [&](const auto& entry) { return entry.second.surface == &surface; });
    }
}

std::map<std::uint64_t, std::size_t>& clearedHtiles() {
    static auto* cleared = new std::map<std::uint64_t, std::size_t>();
    return *cleared;
}

bool takeClearedHtile(const DepthTarget& target) {
    if (target.htileAddress == 0) return false;
    auto& cleared = clearedHtiles();
    const auto found = std::find_if(cleared.begin(), cleared.end(), [&](const auto& entry) { return HtileFillCovers(target.htileAddress, target.extent, entry.first, entry.second); });
    if (found == cleared.end()) return false;
    cleared.erase(found);
    return true;
}

const DepthSurface* drawnDepth = nullptr;

std::atomic<std::uint64_t> surfacesSerial{0};

std::uint32_t expandedHtileWord(std::uint32_t word, bool stencilInHtile, VkImageAspectFlags cleared) {
    if (!stencilInHtile) return (cleared & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 ? 0xfffc000fu : word;
    if ((cleared & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) word = (word & 0x3f0u) | 0xfffff00fu;
    if ((cleared & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) word |= 0x3f0u;
    return word;
}

std::mutex& htileReadsMutex() {
    static std::mutex mutex;
    return mutex;
}

std::atomic<std::uint64_t> htileStoreBackReads{0};

struct HtileRead {
    std::uint64_t generation = 0;
    std::uint64_t note = 0;
};

std::map<std::uint64_t, HtileRead>& htileReads() {
    static auto* reads = new std::map<std::uint64_t, HtileRead>();
    return *reads;
}

struct HtileFill {
    std::size_t bytes = 0;
    std::uint32_t pattern = 0;
    std::uint64_t note = 0;
};

std::map<std::uint64_t, HtileFill>& htileFills() {
    static auto* fills = new std::map<std::uint64_t, HtileFill>();
    return *fills;
}

std::optional<Recorder::PendingWrite> newestPendingWrite(const Recorder& recorder, std::uint64_t address, std::size_t bytes) {
    std::optional<Recorder::PendingWrite> newest;
    for (const auto& write : recorder.PendingWritesOver(address, bytes)) {
        if (!newest || write.note > newest->note) newest = write;
    }
    return newest;
}

void noteHtileFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern, bool covers) {
    std::uint64_t note = 0;
    auto* recorder = Recorder::Active();
    if (covers && recorder != nullptr) {
        const auto newest = newestPendingWrite(*recorder, address, bytes);
        if (newest && newest->begin <= address && newest->end >= address + bytes) note = newest->note;
    }
    std::lock_guard lock(htileReadsMutex());
    std::erase_if(htileFills(), [&](const auto& entry) { return entry.first < address + bytes && address < entry.first + entry.second.bytes; });
    if (note != 0) htileFills()[address] = {bytes, pattern, note};
}

void forgetHtileReads(std::uint64_t address, std::size_t bytes) {
    std::lock_guard lock(htileReadsMutex());
    auto& reads = htileReads();
    reads.erase(reads.lower_bound(address), reads.lower_bound(address + bytes));
}

bool RoutesToNativeDepthOwner(const DepthTarget& target) {
    if (target.samples != 1u) return true;
    if (target.htileAddress == 0 || !anyDepthGuard().load(std::memory_order_acquire)) return false;
    const auto bytes = static_cast<std::uint64_t>(HtileMetadataBytes(target.extent));
    std::shared_lock lock(depthGuardMutex());
    return std::ranges::any_of(depthGuards(), [&](const auto& range) { return range.active->load(std::memory_order_acquire) && target.htileAddress < range.address + range.bytes && range.address < target.htileAddress + bytes; });
}

std::optional<VkImageAspectFlags> storeBackFilledHtile(const Context& context, Recorder& recorder, const DepthTarget& target, std::size_t bytes, std::uint32_t pattern, std::uint64_t note) {
    const VkImageAspectFlags cleared = (VK_IMAGE_ASPECT_DEPTH_BIT | (target.htileStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u)) & HtileFillClears(pattern, target.htileStencil);
    if (cleared != 0) {
        const auto* import = HostImportFor(context, target.htileAddress, bytes);
        if (import == nullptr) return std::nullopt;
        StorageTexture::FlushPending(target.htileAddress, bytes, nullptr, "HTILE store-back");
        PrepareImportWrite(target.htileAddress, bytes);
        recorder.FlushStoresOverlapping(target.htileAddress, bytes);
        recorder.FlushKeyStoresOverlapping(target.htileAddress, bytes);
        recorder.DeclareSection({});
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Resolved(&DeviceFunctions::cmdFillBuffer, "vkCmdFillBuffer")(commands, import->buffer, target.htileAddress - import->base, bytes, expandedHtileWord(pattern, target.htileStencil, cleared));
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::Fill, 2);
        recorder.NotePendingWrite(target.htileAddress, bytes);
        GuestMemory::MarkWritten(target.htileAddress, bytes);
        NoteImportWritten(target.htileAddress, bytes);
        const auto written = newestPendingWrite(recorder, target.htileAddress, bytes);
        note = written ? written->note : 0u;
    }
    const auto generation = GuestMemory::CollectWrites(target.htileAddress, bytes);
    std::lock_guard lock(htileReadsMutex());
    htileReads()[target.htileAddress] = {generation, note};
    return cleared;
}

VkImageAspectFlags storeBackClearedHtile(const Context& context, const DepthTarget& target) {
    if (target.htileAddress == 0) return 0;
    const auto bytes = static_cast<std::size_t>(HtileMetadataBytes(target.extent));
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(target.htileAddress), bytes, true)) return 0;
    const auto collected = GuestMemory::CollectWrites(target.htileAddress, bytes);
    auto* recorder = RecorderFor(context);
    {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        const bool pending = recorder != nullptr && recorder->PendingWriteOverlaps(target.htileAddress, bytes);
        const auto newest = pending ? newestPendingWrite(*recorder, target.htileAddress, bytes) : std::nullopt;
        std::optional<std::uint32_t> filled;
        {
            std::lock_guard lock(htileReadsMutex());
            const auto found = htileReads().find(target.htileAddress);
            if (found != htileReads().end() && found->second.generation != 0 && (!pending || (newest && found->second.note == newest->note)) && GuestMemory::UnchangedSince(target.htileAddress, bytes, found->second.generation)) return 0;
            if (newest) {
                const auto fill = std::ranges::find_if(htileFills(), [&](const auto& entry) { return entry.second.note == newest->note && HtileFillCovers(target.htileAddress, target.extent, entry.first, entry.second.bytes); });
                if (fill != htileFills().end()) filled = fill->second.pattern;
            }
        }
        if (filled) {
            if (const auto cleared = storeBackFilledHtile(context, *recorder, target, bytes, *filled, newest->note)) return *cleared;
        }
    }
    htileStoreBackReads.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::uint32_t> words(bytes / 4u);
    GuestMemory::Read(target.htileAddress, std::as_writable_bytes(std::span(words)), 4);
    const bool uniform = std::all_of(words.begin(), words.end(), [&](std::uint32_t word) { return word == words.front(); });
    VkImageAspectFlags cleared = target.firstSlice == 0 || uniform ? VK_IMAGE_ASPECT_DEPTH_BIT | (target.htileStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u) : 0u;
    for (const auto word : words) cleared &= HtileFillClears(word, target.htileStencil);
    auto generation = collected;
    if (cleared != 0) {
        for (auto& word : words) word = expandedHtileWord(word, target.htileStencil, cleared);
        GuestMemory::Write(target.htileAddress, std::as_bytes(std::span(words)), 4);
        generation = GuestMemory::CollectWrites(target.htileAddress, bytes);
    }
    std::lock_guard lock(htileReadsMutex());
    htileReads()[target.htileAddress] = {generation, 0};
    return cleared;
}
}

std::uint64_t HtileStoreBackReads() {
    return htileStoreBackReads.load(std::memory_order_relaxed);
}

std::uint64_t DepthSurfacesSerial() {
    return surfacesSerial.load(std::memory_order_acquire);
}

std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel) {
    const std::uint32_t blockWidth = bytesPerTexel == 4 ? 128u : 256u;
    const std::uint32_t blockHeight = bytesPerTexel == 1 ? 256u : 128u;
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1) / blockHeight * blockHeight);
    return width * height * bytesPerTexel;
}

std::uint64_t MultisampledDepthBytes_nid_no_patch(VkExtent2D extent, std::uint32_t bytesPerTexel, std::uint32_t samples) {
    Require(extent.width != 0u && extent.width <= 16384u && extent.height != 0u && extent.height <= 16384u, "invalid multisample depth extent");
    Require((bytesPerTexel == 1u || bytesPerTexel == 2u || bytesPerTexel == 4u) && (samples == 2u || samples == 4u || samples == 8u), "unsupported multisample depth footprint");
    const auto sampleBits = std::countr_zero(samples);
    const auto pixelBits = 16u - std::countr_zero(bytesPerTexel) - sampleBits;
    const auto widthBits = (pixelBits + ((sampleBits & 1u) == 0u ? 1u : 0u)) / 2u;
    const auto blockWidth = 1u << widthBits;
    const auto blockHeight = 1u << (pixelBits - widthBits);
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1u) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1u) / blockHeight * blockHeight);
    return width * height * bytesPerTexel * samples;
}

void RequireNoResidentDepthRead_nid_no_patch(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0u || !anyDepthGuard().load(std::memory_order_acquire)) return;
    Require(address <= std::numeric_limits<std::uint64_t>::max() - bytes, "native depth read range overflows");
    std::shared_lock lock(depthGuardMutex());
    for (const auto& range : depthGuards()) {
        if (range.published != nullptr && range.published->load(std::memory_order_acquire)) continue;
        Require(address >= range.address + range.bytes || range.address >= address + bytes, "sample-preserving guest publication of native multisample depth is not implemented");
    }
}

bool HasResidentDepthGuards_nid_no_patch() {
    return anyDepthGuard().load(std::memory_order_acquire);
}

void RetireUnmappedDepth_nid_no_patch(std::uintptr_t address, std::size_t bytes) {
    if (bytes == 0u || !HasResidentDepthGuards_nid_no_patch()) return;
    Require(address <= std::numeric_limits<std::uint64_t>::max() - bytes, "native depth release range overflows");
    const auto end = address + bytes;
    std::unique_lock lock(depthGuardMutex());
    for (auto& range : depthGuards()) {
        std::vector<GuestMemory::UnchangedQuery> remaining;
        remaining.reserve(range.live.size() + 1u);
        for (const auto& part : range.live) {
            const auto finish = part.address + part.bytes;
            const auto first = std::max<std::uint64_t>(address, part.address);
            const auto last = std::min<std::uint64_t>(end, finish);
            if (first >= last) remaining.push_back(part);
            else {
                if (part.address < first) remaining.push_back({part.address, static_cast<std::size_t>(first - part.address), 0u});
                if (last < finish) remaining.push_back({last, static_cast<std::size_t>(finish - last), 0u});
            }
        }
        range.live = std::move(remaining);
    }
    for (const auto& range : depthGuards()) {
        if (std::ranges::all_of(depthGuards(), [&](const auto& part) { return part.active != range.active || part.live.empty(); })) range.active->store(false, std::memory_order_release);
    }
    std::erase_if(depthGuards(), [](const auto& range) { return !range.active->load(std::memory_order_acquire); });
    anyDepthGuard().store(!depthGuards().empty(), std::memory_order_release);
}

bool SampleLocationsCompatibleDepth(const Context& context, VkFormat format, std::uint32_t samples) {
    if (samples <= 1 || (context.sampleLocationSampleCounts & samples) == 0) return false;
    VkImageFormatProperties properties{};
    const auto result = context.imageFormatProperties(context.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT, &properties);
    if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) return false;
    Check(result, "vkGetPhysicalDeviceImageFormatProperties sample locations depth");
    return (properties.sampleCounts & samples) != 0;
}

VkFormat HostDepthFormat(const Context& context, VkFormat format) {
    if (format != VK_FORMAT_D16_UNORM_S8_UINT) return format;
    auto& formats = hostDepthFormats();
    std::lock_guard lock(formats.mutex);
    const std::pair key{context.device, context.physical};
    if (const auto found = formats.chosen.find(key); found != formats.chosen.end()) return found->second;
    constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    auto host = format;
    for (const auto candidate : {VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
        VkFormatProperties properties{};
        context.formatProperties(context.physical, candidate, &properties);
        if ((properties.optimalTilingFeatures & required) != required) continue;
        host = candidate;
        break;
    }
    formats.chosen.emplace(key, host);
    return host;
}

float HostDepthBiasConstant(const Context& context, VkFormat format, float constant) {
    const auto host = HostDepthFormat(context, format);
    if (constant == 0.0f || host == format) return constant;
    Require(host == VK_FORMAT_D24_UNORM_S8_UINT, "a constant depth bias on a 16-bit depth target with stencil that the device stores as 32-bit float depth is not implemented");
    return constant * 256.0f;
}

VkImageView DepthSurfaceView(const Context& context, const DepthTarget& target) {
    const auto htileCleared = RoutesToNativeDepthOwner(target) ? VkImageAspectFlags{0} : storeBackClearedHtile(context, target);
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::unique_lock lock(surfacesMutex());
    const bool cleared = takeClearedHtile(target);
    DepthSurface* surface = nullptr;
    for (const auto& candidate : surfaces()) {
        if (candidate->active->load(std::memory_order_acquire) && candidate->context.device == context.device && sameSurface(candidate->target, target)) {
            surface = candidate.get();
            break;
        }
    }
    const bool created = surface == nullptr;
    GuestAllocations::Lease creationLease;
    if (created) {
        for (const auto& surface : surfaces()) {
            if (!surface->active->load(std::memory_order_acquire) || !surface->samplingActive || surface->context.device != context.device) continue;
            const auto bytes = target.samples == 1u ? DepthSliceBytes(target.extent, target.format == VK_FORMAT_D16_UNORM ? 2u : 4u) : MultisampledDepthBytes_nid_no_patch(target.extent, target.format == VK_FORMAT_D16_UNORM ? 2u : 4u, target.samples);
            Require(target.address <= std::numeric_limits<std::uint64_t>::max() - bytes && target.htileAddress <= std::numeric_limits<std::uint64_t>::max() - HtileMetadataBytes(target.extent), "native depth attachment range overflows");
            for (const auto& range : surface->sampling.initial) {
                Require(target.address >= range.address + range.bytes || range.address >= target.address + bytes, "native multisample depth attachment layout or metadata changed");
                if (target.stencilAddress != 0u) {
                    const auto stencilBytes = target.samples == 1u ? DepthSliceBytes(target.extent, 1u) : MultisampledDepthBytes_nid_no_patch(target.extent, 1u, target.samples);
                    Require(target.stencilAddress <= std::numeric_limits<std::uint64_t>::max() - stencilBytes && (target.stencilAddress >= range.address + range.bytes || range.address >= target.stencilAddress + stencilBytes), "native multisample stencil attachment aliases a resident depth surface");
                }
                if (target.htileAddress != 0u) Require(target.htileAddress >= range.address + range.bytes || range.address >= target.htileAddress + HtileMetadataBytes(target.extent), "native multisample depth attachment layout or metadata changed");
            }
        }
        lock.unlock();
        auto sampling = target.samples == 1u ? DepthSamplingState{} : prepareDepthSampling(context, target);
        creationLease = target.samples == 1u || !sampling.refusal.empty() ? GuestAllocations::Lease{} : GuestAllocations::GuestAllocationsAcquireAll_nid_postfix();
        lock.lock();
        surfacesSerial.fetch_add(1, std::memory_order_release);
        surfaces().push_back(std::make_unique<DepthSurface>(context, target, std::move(sampling)));
        auto& fresh = *surfaces().back();
        if (target.samples != 1u && fresh.sampleable && fresh.sampling.refusal.empty()) {
            std::unique_lock guardLock(depthGuardMutex());
            for (std::size_t index = 0; index < fresh.sampling.initial.size(); ++index) {
                const auto& range = fresh.sampling.initial[index];
                depthGuards().push_back({context.device, range.address, range.bytes, fresh.active, {{range.address, range.bytes, 0u}}, index == 0 ? fresh.depthPublished : fresh.htilePublished});
            }
            anyDepthGuard().store(true, std::memory_order_release);
            InstallResidentImageReadGuard_nid_no_patch();
            fresh.samplingActive = true;
        }
        surface = surfaces().back().get();
    }
    surface->retired = false;
    const auto owner = depthOwners().find({context.device, target.address});
    const bool inherits = owner != depthOwners().end() && owner->second.surface->active->load(std::memory_order_acquire) && !owner->second.surface->retired;
    surface->clearDepth = target.clearDepth;
    surface->clearStencil = target.clearStencil;
    takePlanes(*surface);
    surface->OwnPlanes();
    surface->NoteHtileBaseline();
    if (created && !inherits && target.htileAddress == 0) surface->TakeStorage();
    if (cleared && (!created || inherits)) {
        if (target.samples != 1u) surface->sampling.refusal = "native multisample depth fast-clear transition is not implemented";
        surface->pendingClear |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    surface->pendingClear |= htileCleared;
    surface->ApplyFastClear();
    if (!created || inherits) surface->written = ++depthWrites;
    surface->NoteMemory();
    drawnDepth = surface;
    return surface->view;
}

std::uint64_t HtileDepthClearAddress(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads) {
    static constexpr std::array<std::uint32_t, 17> htileMaskKernel{0xd7460000u, 0x04010c06u, 0x34000084u, 0xdc388000u, 0x04020000u, 0xbf8c3f70u, 0xd7710008u, 0x00120805u, 0xd7710009u, 0x00120a05u, 0xd771000au, 0x00120c05u, 0xd771000bu, 0x00120e05u, 0xdc788000u, 0x00020800u, 0xbf810000u};
    if (userData.size() < 6 || numThreads[0] != 64 || numThreads[1] != 1 || numThreads[2] != 1) return 0;
    if (code.size() < htileMaskKernel.size() || !std::equal(htileMaskKernel.begin(), htileMaskKernel.end(), code.begin())) return 0;
    if ((userData[5] & 0xfu) != 0 || (userData[4] & 0xfu) != 0) return 0;
    return userData[2] | (static_cast<std::uint64_t>(userData[3]) << 32u);
}

std::optional<MemoryValueFill> MatchMemoryValueFill(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads, const std::array<std::uint32_t, 3>& groups) {
    static constexpr std::array<std::uint32_t, 9> kernel{0xd7460000u, 0x04010c08u, 0xf4201a82u, 0xfa000000u, 0xbf8cc07fu, 0x7e02026au, 0xe0102000u, 0x80000100u, 0xbf810000u};
    if (userData.size() < 8 || numThreads != std::array<std::uint32_t, 3>{64u, 1u, 1u} || groups[1] != 1u || groups[2] != 1u) return std::nullopt;
    if (code.size() < kernel.size() || !std::equal(kernel.begin(), kernel.end(), code.begin())) return std::nullopt;
    const auto stride = (userData[1] >> 16u) & 0x3fffu;
    const bool swizzled = ((userData[1] >> 31u) & 1u) != 0;
    const bool addTid = ((userData[3] >> 23u) & 1u) != 0;
    const auto type = userData[3] >> 30u;
    const auto format = (userData[3] >> 12u) & 0x7fu;
    const auto valueStride = (userData[5] >> 16u) & 0x3fffu;
    const auto valueBytes = valueStride == 0 ? static_cast<std::uint64_t>(userData[6]) : static_cast<std::uint64_t>(userData[6]) * valueStride;
    if (type != 0 || stride != 4 || swizzled || addTid || format < 20u || format > 22u || valueBytes < 4u) return std::nullopt;
    const auto dwords = std::min<std::uint64_t>(static_cast<std::uint64_t>(groups[0]) * 64u, userData[2]);
    const auto address = userData[0] | (static_cast<std::uint64_t>(userData[1] & 0xffffu) << 32u);
    const auto valueAddress = userData[4] | (static_cast<std::uint64_t>(userData[5] & 0xffffu) << 32u);
    if (dwords == 0 || valueAddress == 0 || (valueAddress & 3u) != 0) return std::nullopt;
    return MemoryValueFill{address, static_cast<std::size_t>(dwords * 4u), valueAddress};
}

void NoteMemoryValueFill(const MemoryValueFill& fill) {
    std::uint32_t value = 0;
    GuestMemory::Read(fill.valueAddress, std::as_writable_bytes(std::span(&value, 1)), 4);
    NoteDepthMetadataFill(fill.address, fill.bytes, value);
}

std::size_t HtileDepthClearBytes_nid_no_patch(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, std::span<const std::uint32_t> packet, std::uint64_t indirectArguments) {
    if (userData.size() != 6u || packet.size() < 5u || indirectArguments != 0u || compute.PartialGroups() || !compute.groupIdEnable[0] || compute.threadIdComponentCount == 0u || compute.threadIdComponentCount > 3u) return 0u;
    if (packet[1] == 0u || packet[1] > 0x400000u || packet[2] != 1u || packet[3] != 1u || (packet[4] & 0x23u) != 1u) return 0u;
    const auto address = HtileDepthClearAddress(code, userData, compute.numThreads);
    if (address == 0u || (address & 15u) != 0u) return 0u;
    const auto bytes = static_cast<std::uint64_t>(packet[1]) * 1024u;
    return address <= std::numeric_limits<std::uint64_t>::max() - bytes ? static_cast<std::size_t>(bytes) : 0u;
}

void NoteHtileDepthClear(std::uint64_t htileAddress, std::size_t bytes, std::size_t nativeBytes) {
    std::lock_guard lock(surfacesMutex());
    forgetHtileReads(htileAddress, bytes);
    bool existing = false;
    for (const auto& surface : surfaces()) {
        if (!surface->active->load(std::memory_order_acquire)) continue;
        if (surface->samplingActive && surface->target.htileAddress == htileAddress) {
            const auto& metadata = surface->sampling.initial[1];
            if (nativeBytes != bytes || nativeBytes < metadata.bytes || !GuestMemory::Accessible(reinterpret_cast<const void*>(metadata.address), metadata.bytes) || !GuestMemory::Watched(metadata.address, metadata.bytes)) surface->sampling.refusal = "native multisample depth incomplete or unproven fast-clear transition is not implemented";
            else if (surface->clearGeneration != 0u) surface->sampling.refusal = "overlapping native multisample depth clear transitions are not implemented";
            else surface->clearGeneration = GuestMemory::CollectWritesUncached(metadata.address, metadata.bytes);
            existing = true;
            continue;
        }
        if (!HtileFillCovers(surface->target.htileAddress, surface->target.extent, htileAddress, bytes)) continue;
        if (surface->target.samples != 1u) surface->sampling.refusal = "native multisample depth fast-clear transition is not implemented";
        surface->pendingClear |= VK_IMAGE_ASPECT_DEPTH_BIT;
        existing = true;
    }
    if (!existing) clearedHtiles()[htileAddress] = bytes;
}

void ClearDepthSurfaces(VkDevice device) {
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    drawnDepth = nullptr;
    surfacesSerial.fetch_add(1, std::memory_order_release);
    std::erase_if(depthOwners(), [&](const auto& entry) { return entry.first.first == device; });
    std::erase_if(stencilOwners(), [&](const auto& entry) { return entry.first.first == device; });
    std::erase_if(surfaces(), [&](const auto& surface) { return surface->context.device == device; });
    std::erase_if(planeCopies(), [&](const auto& entry) { return std::get<0>(entry.first) == device; });
    std::erase_if(layeredStorages(), [&](const auto& entry) { return std::get<0>(entry.first) == device; });
    publishers().erase(device);
    std::unique_lock guardLock(depthGuardMutex());
    std::erase_if(depthGuards(), [&](const auto& range) { return range.device == device; });
    anyDepthGuard().store(!depthGuards().empty(), std::memory_order_release);
    UpdateResidentImageUnmapObserver_nid_no_patch();
    guardLock.unlock();
    auto& formats = hostDepthFormats();
    std::lock_guard formatLock(formats.mutex);
    std::erase_if(formats.chosen, [&](const auto& entry) { return entry.first.first == device; });
}

std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components) {
    const auto held = [&](const auto& surface) {
        return surface->active->load(std::memory_order_acquire) && surface->context.device == context.device && !surface->retired && (surface->target.address == resource.baseAddress || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == resource.baseAddress));
    };
    {
        std::lock_guard lock(surfacesMutex());
        if (std::none_of(surfaces().begin(), surfaces().end(), held)) return nullptr;
    }
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    auto* found = planeSurface(context.device, resource.baseAddress);
    if (found == nullptr) return nullptr;
    const auto& base = found->target;
    const bool stencil = base.stencilAddress != 0 && resource.baseAddress == base.stencilAddress;
    if (stencil && BytesPerElement(resource.format) != 1u) return nullptr;
    if (base.samples != 1u || resource.dimension == TextureDimension::k2DMsaa) {
        Require(base.samples > 1u, "multisample reader overlaps a single-sample depth attachment");
        auto* recorder = RecorderFor(context);
        Require(recorder != nullptr, "a native multisampled depth target is sampled only by recorded draws and dispatches");
        found->CheckCurrent();
        found->CheckResource(resource);
        Require(found->pendingClear == 0u, "native multisample depth fast-clear transition is not implemented");
        recorder->DeclareSection({{found->image, true}, {found->view, true}});
        RecordMemoryBarrier(context, recorder->Commands(), VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
        return found->Sampled(words, resource, components);
    }
    if (resource.width != base.extent.width || resource.height != base.extent.height) return nullptr;
    const bool d16 = base.format == VK_FORMAT_D16_UNORM || base.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto format = ResolveTextureFormat(resource.format);
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    const bool integer = d16 && format == VK_FORMAT_R16_UINT && !depthBits;
    if (!stencil && format != (d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT) && !depthBits && !integer) return nullptr;
    const bool cube = resource.dimension == TextureDimension::kCube;
    const bool layered = cube || (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray != 0);
    const bool sampled = stencil || (!layered && !integer);
    const auto copyFormat = integer ? VK_FORMAT_R16_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool planes = (format == copyFormat || depthBits) && resource.width == base.extent.width && resource.height == base.extent.height && resource.baseLevel == 0 && resource.lastLevel == 0 && !(resource.baseArray != 0 && (cube || !layered || resource.baseArray > resource.depthOrLastArray)) && !(cube && resource.depthOrLastArray != 0 && resource.depthOrLastArray != 5) && (resource.dimension == TextureDimension::k2D || resource.dimension == TextureDimension::k2DArray || cube);
    if (!(sampled ? found->SampledAccepts(words, resource) : planes) && found->OverwrittenInMemory()) {
        retire(*found);
        return nullptr;
    }
    found->ApplyFastClear();
    if (sampled) return found->Sampled(words, resource, components, found == drawnDepth);
    Require(found->hostFormat == base.format, "sampling several slices or an integer view of a depth plane that the device stores in another depth format is not implemented");
    const auto layers = cube ? 6u : layered ? resource.depthOrLastArray + 1u : 1u;
    if (!planes) {
        char text[320];
        std::snprintf(text, sizeof(text), "AGC graphics: sampling the depth planes of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), dimension %d, %u slices, levels %u-%u, base slice %u is not implemented", static_cast<unsigned long long>(base.address), base.extent.width, base.extent.height, static_cast<int>(base.format), resource.width, resource.height, resource.format, static_cast<int>(format), static_cast<int>(resource.dimension), layers, resource.baseLevel, resource.lastLevel, resource.baseArray);
        throw std::runtime_error(text);
    }
    std::vector<DepthPlaneSlice> slices;
    const auto geometry = DescribeSurface(resource);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const auto address = base.address + geometry.GuestLayerOffset(layer);
        auto* slice = depthSlice(context.device, address, base);
        if (slice != nullptr) slice->ApplyFastClear();
        if (slice == nullptr) slices.push_back({});
        else slices.push_back({slice->image, slice == drawnDepth ? 0u : slice->written, slice->view});
    }
    const auto imageFormat = integer ? VK_FORMAT_R16_UINT : d16 ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
    auto& copy = planeCopies()[{context.device, base.address, base.extent.width, base.extent.height, imageFormat, layers}];
    if (copy == nullptr) {
        copy = std::make_unique<DepthPlaneCopy>(context, base.extent, imageFormat, layers);
    }
    return copy->Refresh(slices, resource, components, cube || resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, layered && !cube ? resource.baseArray : 0u);
}

bool DepthSurfaceHolds(const Context& context, const GuestTextureResource& resource, const Texture* texture) {
    if ((resource.dimension != TextureDimension::k2D && resource.dimension != TextureDimension::k2DMsaa) || resource.baseArray != 0) return false;
    std::lock_guard gpu(GuestMemory::GpuMutex());
    const auto address = resource.baseAddress;
    std::lock_guard lock(surfacesMutex());
    auto* found = planeSurface(context.device, address);
    if (found == nullptr || !found->Holds(texture)) return false;
    found->CheckCurrent();
    found->CheckResource(resource);
    return true;
}

void RequireNoNativeDepthFeedback_nid_no_patch(const Context& context, const DepthTarget& target, std::span<const std::shared_ptr<Texture>> sampled) {
    if (target.samples == 1u || sampled.empty()) return;
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    for (const auto& surface : surfaces()) {
        if (!surface->active->load(std::memory_order_acquire)) continue;
        if (surface->context.device != context.device || !sameSurface(surface->target, target)) continue;
        Require(std::none_of(sampled.begin(), sampled.end(), [&](const auto& texture) { return texture != nullptr && surface->Holds(texture.get()); }), "sampling a native multisampled depth target the draw attaches is not implemented");
    }
}

VkImageAspectFlags HtileFillClears(std::uint32_t pattern, bool stencilInHtile) {
    VkImageAspectFlags cleared = (pattern & 0xfu) == 0 ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u;
    if (stencilInHtile && ((pattern >> 8u) & 0x3u) == 0) cleared |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return cleared;
}

std::uint64_t HtileMetadataBytes(VkExtent2D extent) {
    const auto blocks = static_cast<std::uint64_t>((extent.width + 1023u) / 1024u) * ((extent.height + 511u) / 512u);
    return blocks * 32768u;
}

bool HtileFillCovers(std::uint64_t htile, VkExtent2D extent, std::uint64_t address, std::size_t bytes) {
    if (htile == 0 || htile < address || htile >= address + bytes) return false;
    return address + bytes - htile >= HtileMetadataBytes(extent);
}

std::size_t DiscardDepthInside(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) return 0;
    std::unique_lock<GuestMemory::GpuMutexType> gpu(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) gpu.lock();
    std::lock_guard lock(surfacesMutex());
    std::size_t discarded = 0;
    for (const auto& surface : surfaces()) {
        if (surface->active->load(std::memory_order_acquire) && surface->DiscardInside(address, bytes)) ++discarded;
    }
    return discarded;
}

void NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    forgetHtileReads(address, bytes);
    bool covers = false;
    for (const auto& surface : surfaces()) {
        if (!surface->active->load(std::memory_order_acquire)) continue;
        const auto& target = surface->target;
        if (target.samples != 1u && target.htileAddress != 0u && address < target.htileAddress + HtileMetadataBytes(target.extent) && target.htileAddress < address + bytes) surface->sampling.refusal = "native multisample depth metadata fill transition is not implemented";
        if (!HtileFillCovers(target.htileAddress, target.extent, address, bytes)) continue;
        covers = true;
        const VkImageAspectFlags written = VK_IMAGE_ASPECT_DEPTH_BIT | (target.htileStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
        surface->pendingClear = (surface->pendingClear & ~written) | HtileFillClears(pattern, target.htileStencil);
    }
    noteHtileFill(address, bytes, pattern, covers);
}

bool DepthSurfaceAt(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource) {
    std::lock_guard lock(surfacesMutex());
    const auto* found = planeSurface(context.device, resource.baseAddress);
    if (found == nullptr) return false;
    const auto& target = found->target;
    if (resource.width != target.extent.width || resource.height != target.extent.height) return false;
    const bool stencil = target.stencilAddress != 0 && target.stencilAddress == resource.baseAddress;
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    return ResolveTextureFormat(resource.format) == expected || depthBits;
}

namespace {

std::vector<LayeredSlice> layeredSlices(const Context& context, const LayeredDepthStorage& storage) {
    const auto& base = storage.base;
    std::vector<LayeredSlice> slices;
    const auto& list = surfaces();
    for (std::uint32_t layer = storage.firstLayer; layer < storage.layers; ++layer) {
        const auto address = base.address + static_cast<std::uint64_t>(layer) * DepthSliceBytes(base.extent, storage.bytesPerTexel);
        const auto slice = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->context.device == context.device && !surface->retired && surface->target.address == address && surface->target.extent.width == base.extent.width && surface->target.extent.height == base.extent.height && surface->target.format == base.format && surface->target.samples == 1; });
        if (slice != list.rend()) slices.push_back({slice->get(), layer});
    }
    return slices;
}

std::shared_ptr<StorageTexture> layeredDepthStorage(const Context& context, DepthSurface& surface, const GuestTextureResource& resource, std::uint32_t mip) {
    const auto& target = surface.target;
    const auto layers = resource.depthOrLastArray + 1u;
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto bytesPerTexel = d16 ? 2u : 4u;
    surface.RequireStorable(resource, mip);
    const auto geometry = DescribeSurface(resource);
    if (layers > 1 && geometry.GuestLayerOffset(1) != DepthSliceBytes(target.extent, bytesPerTexel)) {
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the depth planes of depth surface 0x%llx with a layer pitch of 0x%llx bytes, not the depth slice size 0x%llx, is not implemented", static_cast<unsigned long long>(target.address), static_cast<unsigned long long>(geometry.GuestLayerOffset(1)), static_cast<unsigned long long>(DepthSliceBytes(target.extent, bytesPerTexel)));
        throw std::runtime_error(text);
    }
    auto& entry = layeredStorages()[{context.device, resource.baseAddress, resource.width, resource.height, layers, resource.format, static_cast<int>(resource.tileMode), static_cast<int>(resource.dimension), resource.baseArray}];
    if (entry == nullptr) entry = std::make_unique<LayeredDepthStorage>(context, resource, target, bytesPerTexel, resource.baseArray, layers);
    else static_cast<void>(entry->storage->Refresh());
    const auto slices = layeredSlices(context, *entry);
    for (const auto& slice : slices) slice.surface->ApplyFastClear();
    entry->Copy(slices, true);
    return entry->storage;
}

}

std::shared_ptr<StorageTexture> DepthSurfaceStorage(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip) {
    std::lock_guard lock(surfacesMutex());
    const auto& list = surfaces();
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) {
        return surface->active->load(std::memory_order_acquire) && surface->context.device == context.device && !surface->retired && (surface->target.address == resource.baseAddress || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == resource.baseAddress));
    });
    if (found == list.rend()) return nullptr;
    auto& surface = **found;
    const bool stencil = surface.target.stencilAddress != 0 && surface.target.stencilAddress == resource.baseAddress;
    if (stencil || resource.dimension != TextureDimension::k2DArray || (resource.baseArray == 0 && resource.depthOrLastArray == 0)) return surface.Storage(words, resource, mip);
    if (!surface.StorageViewMatches(words, resource)) return nullptr;
    return layeredDepthStorage(context, surface, resource, mip);
}

bool DepthSurfaceStoreBack(const Context& context, const StorageTexture& storage) {
    std::unique_lock<GuestMemory::GpuMutexType> gpu(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) gpu.lock();
    std::vector<std::pair<DepthSurface*, VkImageAspectFlags>> written;
    {
        std::lock_guard lock(surfacesMutex());
        const auto store = [&]() {
            for (auto& [key, entry] : layeredStorages()) {
                if (std::get<0>(key) != context.device || entry->storage.get() != &storage) continue;
                const auto slices = layeredSlices(context, *entry);
                entry->Copy(slices, false);
                for (const auto& slice : slices) {
                    slice.surface->written = ++depthWrites;
                    written.emplace_back(slice.surface, VK_IMAGE_ASPECT_DEPTH_BIT);
                }
                return true;
            }
            for (const auto& surface : surfaces()) {
                if (!surface->active->load(std::memory_order_acquire) || surface->context.device != context.device) continue;
                if (const auto aspect = surface->StoreBack(storage); aspect != 0u) {
                    written.emplace_back(surface.get(), aspect);
                    return true;
                }
            }
            for (auto surface = surfaces().rbegin(); surface != surfaces().rend(); ++surface) {
                if ((*surface)->context.device != context.device) continue;
                if (const auto aspect = (*surface)->StoreBackCovering(storage); aspect != 0u) {
                    written.emplace_back(surface->get(), aspect);
                    return true;
                }
            }
            return false;
        };
        if (!store()) return false;
    }
    for (const auto& [surface, aspect] : written) surface->MarkStale(aspect);
    return true;
}

bool DepthSurfaceAt(std::uint64_t address) {
    std::lock_guard lock(surfacesMutex());
    return std::any_of(surfaces().begin(), surfaces().end(), [&](const auto& surface) { return surface->active->load(std::memory_order_acquire) && !surface->retired && (surface->target.address == address || surface->target.stencilAddress == address); });
}

void RetireDepthSurfaces(VkDevice device, std::uint64_t address, std::uint64_t bytes) {
    const auto retiring = [&](const DepthSurface& surface) { return surface.context.device == device && !surface.retired && surface.Overlaps(address, bytes); };
    {
        std::lock_guard lock(surfacesMutex());
        if (std::none_of(surfaces().begin(), surfaces().end(), [&](const auto& surface) { return retiring(*surface) && surface->residencyOwned; })) {
            for (const auto& surface : surfaces()) {
                if (retiring(*surface)) retire(*surface);
            }
            return;
        }
    }
    std::unique_lock<GuestMemory::GpuMutexType> gpu(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) gpu.lock();
    std::lock_guard lock(surfacesMutex());
    for (const auto& surface : surfaces()) {
        if (retiring(*surface)) retire(*surface);
    }
}

void NoteDepthWritten(const Context& context, const DepthTarget& target, bool depth, bool stencil) {
    std::unique_lock<GuestMemory::GpuMutexType> gpu(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) gpu.lock();
    DepthSurface* found = nullptr;
    {
        std::lock_guard lock(surfacesMutex());
        for (const auto& candidate : surfaces()) {
            if (candidate->active->load(std::memory_order_acquire) && candidate->context.device == context.device && sameSurface(candidate->target, target)) {
                found = candidate.get();
                break;
            }
        }
    }
    if (found != nullptr) found->MarkStale((depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u) | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u));
}

}
