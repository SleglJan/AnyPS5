#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/shaders/TextureDetile_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/CmaskClear_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/DccKeyClear_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TextureDetileImage_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TexelDecode_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/SurfaceFill_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/RestartStarts_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TexelDecodeImage10_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TexelDecodeImage11_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TexelDecodeImage20_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TexelDecodeImage21_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TexelDecodeImage30_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TexelDecodeImage31_spv.h"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

VkResult CreateImageWritingLayout(const Context& context, std::span<const VkDescriptorSetLayoutBinding> bindings, VkDescriptorSetLayout* layout) {
    std::vector<VkDescriptorBindingFlags> flags(bindings.size(), 0);
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (bindings[index].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) flags[index] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlags.bindingCount = static_cast<std::uint32_t>(flags.size());
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.pNext = &bindingFlags;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    return context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, layout);
}

struct Push {
    std::uint32_t srcBase;
    std::uint32_t dstBase;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pitchBytes;
    std::uint32_t blocksPerRow;
    std::uint32_t tail;
    std::uint32_t tailX;
    std::uint32_t tailY;
    std::uint32_t elementBytes;
    std::uint32_t slice;
    std::uint32_t rangeBegin;
    std::uint32_t rangeEnd;
    std::uint32_t tiledBase;
    std::uint32_t linearBase;
    std::uint32_t columnBegin;
    std::uint32_t rowBegin;
    std::uint32_t pipeBankXor;
};

std::uint32_t BlockBytesFor(TextureTileMode tileMode) {
    switch (tileMode) {
        case TextureTileMode::kLinear: return 0u;
        case TextureTileMode::kStandard256B:
        case TextureTileMode::kD256B: return 256u;
        case TextureTileMode::kStandard4KB:
        case TextureTileMode::kD4KB:
        case TextureTileMode::kS4KBX:
        case TextureTileMode::kD4KBX: return 4096u;
        case TextureTileMode::kStandard64KB:
        case TextureTileMode::kD64KB:
        case TextureTileMode::kS64KBT:
        case TextureTileMode::kD64KBT:
        case TextureTileMode::kZ64KBX:
        case TextureTileMode::kS64KBX:
        case TextureTileMode::kD64KBX:
        case TextureTileMode::kR64KBX: return 65536u;
    }
    throw std::runtime_error("AGC graphics: TextureDetiler encountered an unknown tile mode");
}

std::uint32_t PipelineKey(TextureTileMode tileMode, std::uint32_t elementBytes, bool retile, bool thick, bool image) {
    return (image ? 1u << 18 : 0u) | (thick ? 1u << 17 : 0u) | (retile ? 1u << 16 : 0u) | (static_cast<std::uint32_t>(tileMode) << 8) | elementBytes;
}

}

TextureDetiler::TextureDetiler(const Context& context) : context(context) {
    try {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &descriptorLayout), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(Push);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(TEXTURE_DETILE_SPV);
        moduleInfo.pCode = TEXTURE_DETILE_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        Check(CreateImageWritingLayout(context, bindings, &imageDescriptorLayout), "vkCreateDescriptorSetLayout");
        pipelineLayoutInfo.pSetLayouts = &imageDescriptorLayout;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &imagePipelineLayout), "vkCreatePipelineLayout");
    } catch (...) {
        release();
        throw;
    }
}

TextureDetiler::~TextureDetiler() {
    release();
}

void TextureDetiler::release() noexcept {
    if (restartPipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, restartPipeline, nullptr);
    if (restartModule) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, restartModule, nullptr);
    if (fillPipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, fillPipeline, nullptr);
    if (fillModule) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, fillModule, nullptr);
    for (const auto decodePipeline : decodePipelines) {
        if (decodePipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, decodePipeline, nullptr);
    }
    if (decodeModule) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, decodeModule, nullptr);
    for (const auto decodePipeline : decodeImagePipelines) {
        if (decodePipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, decodePipeline, nullptr);
    }
    for (const auto decodeImageModule : decodeImageModules) {
        if (decodeImageModule) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, decodeImageModule, nullptr);
    }
    destroyRing(decodeImageDescriptorSets);
    if (decodeImagePipelineLayout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, decodeImagePipelineLayout, nullptr);
    if (decodeImageDescriptorLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, decodeImageDescriptorLayout, nullptr);
    destroyRing(cmaskDescriptorSets);
    if (cmaskPipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, cmaskPipeline, nullptr);
    if (cmaskPipelineLayout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, cmaskPipelineLayout, nullptr);
    if (cmaskDescriptorLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, cmaskDescriptorLayout, nullptr);
    cmaskErrors.reset();
    keyClearErrors.reset();
    destroyRing(keyClearDescriptorSets);
    if (keyClearPipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, keyClearPipeline, nullptr);
    if (keyClearPipelineLayout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, keyClearPipelineLayout, nullptr);
    if (keyClearDescriptorLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, keyClearDescriptorLayout, nullptr);
    destroyRing(descriptorSets);
    destroyRing(imageDescriptorSets);
    if (imageModule) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, imageModule, nullptr);
    if (imagePipelineLayout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, imagePipelineLayout, nullptr);
    if (imageDescriptorLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, imageDescriptorLayout, nullptr);
    for (const auto& entry : pipelines) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, entry.second, nullptr);
    if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
    if (pipelineLayout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, pipelineLayout, nullptr);
    if (descriptorLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, descriptorLayout, nullptr);
}

VkFormat TextureDetiler::ImageElementFormat(std::uint32_t elementBytes) {
    switch (elementBytes) {
        case 1: return VK_FORMAT_R8_UINT;
        case 2: return VK_FORMAT_R16_UINT;
        case 4: return VK_FORMAT_R32_UINT;
        case 8: return VK_FORMAT_R32G32_UINT;
        case 16: return VK_FORMAT_R32G32B32A32_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

VkPipeline TextureDetiler::pipeline(TextureTileMode tileMode, std::uint32_t elementBytes, bool retile, bool thick, bool image) {
    Require(std::has_single_bit(elementBytes) && elementBytes <= 16u, "unsupported element size for texture detiling");
    const auto key = PipelineKey(tileMode, elementBytes, retile, thick, image);
    for (const auto& entry : pipelines) {
        if (entry.first == key) return entry.second;
    }
    // Constants 0-3 select element size, block size, addressing family and direction; 4-19 carry the
    // per-bit XOR equation for the equation family (2); 20-21 override the block extent in elements.
    std::array<std::uint32_t, 22> values{elementBytes, BlockBytesFor(tileMode), tileMode == TextureTileMode::kLinear ? 0u : 1u, retile ? 1u : 0u};
    if (thick) {
        const auto thickMode = tileMode == TextureTileMode::kStandard4KB ? 0x105u : tileMode == TextureTileMode::kStandard64KB ? 0x109u : tileMode == TextureTileMode::kS64KBX ? 0x119u : tileMode == TextureTileMode::kD64KBX ? 0x11au : 0u;
        Require(thickMode != 0, "3D textures are only detiled thick from SW_4KB_S, SW_64KB_S, SW_64KB_S_X or SW_64KB_D_X");
        const auto* equation = FindTextureSwizzleEquation(thickMode, elementBytes);
        Require(equation != nullptr, "no thick swizzle equation for the element size");
        values[2] = 2u;
        std::copy(equation->bits.begin(), equation->bits.end(), values.begin() + 4);
        const auto extent = ThickBlockExtent(tileMode, elementBytes);
        values[20] = extent[0];
        values[21] = extent[1];
    } else if (const auto mode = EquationSwizzleMode(tileMode); mode != 0) {
        const auto* equation = FindTextureSwizzleEquation(mode, elementBytes);
        if (equation == nullptr) Require(false, "no swizzle equation for tile mode " + std::to_string(mode) + " at " + std::to_string(elementBytes) + " bytes per element");
        values[2] = 2u;
        std::copy(equation->bits.begin(), equation->bits.end(), values.begin() + 4);
    }
    if (image && imageModule == VK_NULL_HANDLE) {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(TEXTURE_DETILE_IMAGE_SPV);
        moduleInfo.pCode = TEXTURE_DETILE_IMAGE_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &imageModule), "vkCreateShaderModule");
    }
    std::array<VkSpecializationMapEntry, 22> entries{};
    for (std::uint32_t index = 0; index < entries.size(); ++index) entries[index] = {index, index * 4u, 4};
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = static_cast<std::uint32_t>(entries.size());
    specialization.pMapEntries = entries.data();
    specialization.dataSize = sizeof(values);
    specialization.pData = values.data();
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = image ? imageModule : module;
    stage.pName = "main";
    stage.pSpecializationInfo = &specialization;
    VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    createInfo.stage = stage;
    createInfo.layout = image ? imagePipelineLayout : pipelineLayout;
    VkPipeline result = VK_NULL_HANDLE;
    Check(CreateComputePipeline(context, createInfo, &result), "vkCreateComputePipelines");
    pipelines.emplace_back(key, result);
    return result;
}

void TextureDetiler::checkWindow(const TileMipLayout& layout, const DetileWindow& window, std::uint32_t& columnEnd, std::uint32_t& rowEnd) const {
    Require(layout.width != 0 && layout.height != 0, "texture detiling requires a non-empty mip layout");
    Require(layout.tiledSize != 0 && layout.linearSize != 0, "texture detiling requires a non-empty mip layout");
    columnEnd = window.columnEnd != 0 ? std::min(window.columnEnd, layout.width) : layout.width;
    rowEnd = window.rowEnd != 0 ? std::min(window.rowEnd, layout.height) : layout.height;
    if (!(window.columnBegin < columnEnd && window.rowBegin < rowEnd)) Require(false, "texture detiling window lies outside the mip: columns " + std::to_string(window.columnBegin) + ".." + std::to_string(columnEnd) + ", rows " + std::to_string(window.rowBegin) + ".." + std::to_string(rowEnd) + " of " + std::to_string(layout.width) + "x" + std::to_string(layout.height));
}

void TextureDetiler::Dispatch(VkCommandBuffer commands, TextureTileMode tileMode, std::uint32_t elementBytes, VkBuffer source, std::uint64_t sourceOffset, VkBuffer destination, std::uint64_t destinationOffset, const TileMipLayout& layout, bool retile, std::uint32_t slice, bool thick, const DetileWindow& window) {
    Require(commands != VK_NULL_HANDLE, "texture detiling requires an active command buffer");
    Require(window.rangeBegin < window.rangeEnd && window.tiledBase <= window.rangeBegin, "texture detiling window is empty");
    Require(source != VK_NULL_HANDLE && destination != VK_NULL_HANDLE, "texture detiling requires source and destination buffers");
    std::uint32_t columnEnd = 0, rowEnd = 0;
    checkWindow(layout, window, columnEnd, rowEnd);
    const auto target = pipeline(tileMode, elementBytes, retile, thick);
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    const auto sourceDescriptorOffset = sourceOffset - sourceOffset % alignment;
    const auto destinationDescriptorOffset = destinationOffset - destinationOffset % alignment;
    const auto sourceBase = sourceOffset - sourceDescriptorOffset;
    const auto destinationBase = destinationOffset - destinationDescriptorOffset;
    Require(sourceBase <= UINT32_MAX && destinationBase <= UINT32_MAX, "texture detiling buffer offset exceeds addressable range");
    // A window covers the mip's bytes from its base to the range's end (tiled) or the last row the
    // range can touch (linear).
    const std::uint64_t tiledBytes = window.rangeEnd == 0xffffffffu ? layout.tiledSize : std::min<std::uint64_t>(layout.tiledSize, window.rangeEnd) - window.tiledBase;
    const std::uint64_t linearBytes = window.linearBytes != 0 ? window.linearBytes : layout.linearSize - std::min<std::uint64_t>(window.linearBase, layout.linearSize);
    const auto sourceRange = (sourceBase + (retile ? linearBytes : tiledBytes) + 3) / 4 * 4;
    const auto destinationRange = (destinationBase + (retile ? tiledBytes : linearBytes) + 3) / 4 * 4;
    Require(sourceRange <= context.limits.maxStorageBufferRange && destinationRange <= context.limits.maxStorageBufferRange, "texture detiling buffer range exceeds device limits");
    const auto set = allocateSet();
    const VkDescriptorBufferInfo sourceInfo{source, sourceDescriptorOffset, sourceRange};
    const VkDescriptorBufferInfo destinationInfo{destination, destinationDescriptorOffset, destinationRange};
    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &sourceInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &destinationInfo;
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, target);
    context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
    Push push{};
    push.srcBase = static_cast<std::uint32_t>(sourceBase);
    push.dstBase = static_cast<std::uint32_t>(destinationBase);
    push.width = layout.width;
    push.height = layout.height;
    push.pitchBytes = layout.pitchBytes;
    push.blocksPerRow = layout.blocksPerRow;
    push.tail = layout.tail ? 1u : 0u;
    push.tailX = layout.tailX;
    push.tailY = layout.tailY;
    push.elementBytes = elementBytes;
    push.slice = slice;
    push.rangeBegin = window.rangeBegin;
    push.rangeEnd = window.rangeEnd;
    push.tiledBase = window.tiledBase;
    push.linearBase = window.linearBase;
    push.columnBegin = window.columnBegin;
    push.rowBegin = window.rowBegin;
    push.pipeBankXor = window.pipeBankXor;
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
    const auto groupsX = (columnEnd - window.columnBegin + 7u) / 8u;
    const auto groupsY = (rowEnd - window.rowBegin + 7u) / 8u;
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groupsX, groupsY, 1);
}

void TextureDetiler::DispatchDecode(VkCommandBuffer commands, std::uint32_t guestFormat, VkBuffer source, VkBuffer destination, std::uint64_t texels) {
    Require(commands != VK_NULL_HANDLE && source != VK_NULL_HANDLE && destination != VK_NULL_HANDLE, "a texel decode needs a command buffer and its source and destination buffers");
    const auto outputBytes = DecodedCopyElementBytes(guestFormat);
    const auto variant = outputBytes == 16u ? 1u : 0u;
    if (decodeModule == VK_NULL_HANDLE) {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(TEXEL_DECODE_SPV);
        moduleInfo.pCode = TEXEL_DECODE_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &decodeModule), "vkCreateShaderModule texel decode");
    }
    auto& target = decodePipelines[variant];
    if (target == VK_NULL_HANDLE) {
        const VkSpecializationMapEntry entry{0, 0, sizeof(std::uint32_t)};
        const std::uint32_t value = variant;
        const VkSpecializationInfo specialization{1, &entry, sizeof(value), &value};
        VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        createInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        createInfo.stage.module = decodeModule;
        createInfo.stage.pName = "main";
        createInfo.stage.pSpecializationInfo = &specialization;
        createInfo.layout = pipelineLayout;
        Check(CreateComputePipeline(context, createInfo, &target), "vkCreateComputePipelines texel decode");
    }
    const std::uint64_t step = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 16u) / 4u;
    const std::uint64_t chunk = context.limits.maxStorageBufferRange / outputBytes / step * step;
    Require(chunk != 0u, "the device's storage buffer range is too small for a texel decode");
    for (std::uint64_t first = 0; first < texels; first += chunk) {
        const auto count = std::min(chunk, texels - first);
        const auto set = allocateSet();
        const VkDescriptorBufferInfo sourceInfo{source, first * 4u, count * 4u};
        const VkDescriptorBufferInfo destinationInfo{destination, first * outputBytes, count * outputBytes};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (std::uint32_t index = 0; index < writes.size(); ++index) {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = set;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = index == 0u ? &sourceInfo : &destinationInfo;
        }
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, target);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        const auto pushed = static_cast<std::uint32_t>(count);
        context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushed), &pushed);
        const auto groups = (count + 255u) / 256u;
        const auto groupsX = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, 65535u));
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groupsX, static_cast<std::uint32_t>((groups + groupsX - 1u) / groupsX), 1);
    }
}

void TextureDetiler::DispatchSurfaceFill(VkCommandBuffer commands, VkBuffer surface, std::uint64_t bytes, std::uint32_t elementBytes, bool expandSingle, const std::array<std::uint32_t, 4>& pattern) {
    Require(commands != VK_NULL_HANDLE && surface != VK_NULL_HANDLE, "a surface fill needs a command buffer and the surface's buffer");
    Require(bytes % 16u == 0u, "a surface fill covers whole 16-byte groups");
    Require(!expandSingle || (std::has_single_bit(elementBytes) && elementBytes <= 16u), "a single-color block expansion needs a power-of-two element of at most 16 bytes");
    if (fillModule == VK_NULL_HANDLE) {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(SURFACE_FILL_SPV);
        moduleInfo.pCode = SURFACE_FILL_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &fillModule), "vkCreateShaderModule surface fill");
    }
    if (fillPipeline == VK_NULL_HANDLE) {
        VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        createInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        createInfo.stage.module = fillModule;
        createInfo.stage.pName = "main";
        createInfo.layout = pipelineLayout;
        Check(CreateComputePipeline(context, createInfo, &fillPipeline), "vkCreateComputePipelines surface fill");
    }
    struct FillPush {
        std::uint32_t words;
        std::uint32_t elementBytes;
        std::uint32_t expand;
        std::uint32_t unused;
        std::array<std::uint32_t, 4> pattern;
    };
    constexpr std::uint64_t BlockBytes = 256u;
    const std::uint64_t step = std::lcm<std::uint64_t>(std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4u), BlockBytes);
    const std::uint64_t chunk = context.limits.maxStorageBufferRange / step * step;
    Require(chunk != 0u, "the device's storage buffer range is too small for a surface fill");
    for (std::uint64_t first = 0; first < bytes; first += chunk) {
        const auto count = std::min(chunk, bytes - first);
        const auto set = allocateSet();
        const VkDescriptorBufferInfo surfaceInfo{surface, first, count};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = 1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &surfaceInfo;
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, fillPipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        const FillPush push{static_cast<std::uint32_t>(count / 4u), elementBytes, expandSingle ? 1u : 0u, 0u, pattern};
        context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        const auto groups = (count + BlockBytes - 1u) / BlockBytes;
        const auto groupsX = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, 65535u));
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groupsX, static_cast<std::uint32_t>((groups + groupsX - 1u) / groupsX), 1);
    }
}

void TextureDetiler::DispatchRestartStarts(VkCommandBuffer commands, VkBuffer indices, std::uint64_t indexOffset, std::uint32_t count, std::uint32_t indexBytes, VkBuffer starts, std::uint64_t startsOffset) {
    Require(commands != VK_NULL_HANDLE && indices != VK_NULL_HANDLE && starts != VK_NULL_HANDLE, "a restart table pass needs a command buffer, the indices and the table's buffer");
    Require((indexBytes == 2u || indexBytes == 4u) && indexOffset % indexBytes == 0u && startsOffset % 4u == 0u && count != 0u, "a restart table pass reads aligned 16- or 32-bit indices into an aligned table");
    if (restartModule == VK_NULL_HANDLE) {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(RESTART_STARTS_SPV);
        moduleInfo.pCode = RESTART_STARTS_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &restartModule), "vkCreateShaderModule restart table");
    }
    if (restartPipeline == VK_NULL_HANDLE) {
        VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        createInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        createInfo.stage.module = restartModule;
        createInfo.stage.pName = "main";
        createInfo.layout = pipelineLayout;
        Check(CreateComputePipeline(context, createInfo, &restartPipeline), "vkCreateComputePipelines restart table");
    }
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    const auto indexDescriptorOffset = indexOffset - indexOffset % alignment;
    const auto startsDescriptorOffset = startsOffset - startsOffset % alignment;
    const auto indexBase = indexOffset - indexDescriptorOffset;
    const auto startsBase = startsOffset - startsDescriptorOffset;
    const auto indexRange = (indexBase + std::uint64_t{count} * indexBytes + 3u) / 4u * 4u;
    const auto startsRange = startsBase + std::uint64_t{count} * 4u;
    Require(indexRange <= context.limits.maxStorageBufferRange && startsRange <= context.limits.maxStorageBufferRange, "a restart table pass exceeds the device's storage buffer range");
    const auto set = allocateSet();
    const VkDescriptorBufferInfo indexInfo{indices, indexDescriptorOffset, indexRange};
    const VkDescriptorBufferInfo startsInfo{starts, startsDescriptorOffset, startsRange};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (std::uint32_t index = 0; index < writes.size(); ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = index == 0u ? &indexInfo : &startsInfo;
    }
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, restartPipeline);
    context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
    const std::array<std::uint32_t, 4> push{count, indexBytes, static_cast<std::uint32_t>(indexBase), static_cast<std::uint32_t>(startsBase / 4u)};
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, 1, 1, 1);
}

void TextureDetiler::DispatchDecodeImage(VkCommandBuffer commands, std::uint32_t guestFormat, VkImageViewType viewType, const PooledView& source, const PooledView& destination, VkExtent3D extent) {
    Require(commands != VK_NULL_HANDLE && source.Started() && destination.Started(), "a texel decode needs a command buffer and its source and destination views");
    const auto dimension = viewType == VK_IMAGE_VIEW_TYPE_1D_ARRAY ? 0u : viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? 1u : viewType == VK_IMAGE_VIEW_TYPE_3D ? 2u : 3u;
    Require(dimension < 3u, "a texel decode between images takes 1D array, 2D array or 3D views");
    const auto variant = dimension * 2u + (DecodedCopyElementBytes(guestFormat) == 16u ? 1u : 0u);
    if (decodeImageDescriptorLayout == VK_NULL_HANDLE) {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        Check(CreateImageWritingLayout(context, bindings, &decodeImageDescriptorLayout), "vkCreateDescriptorSetLayout texel decode");
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &decodeImageDescriptorLayout;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &decodeImagePipelineLayout), "vkCreatePipelineLayout texel decode");
    }
    auto& target = decodeImagePipelines[variant];
    if (target == VK_NULL_HANDLE) {
        constexpr std::array<std::span<const std::uint32_t>, 6> codes{TEXEL_DECODE_IMAGE_10_SPV, TEXEL_DECODE_IMAGE_11_SPV, TEXEL_DECODE_IMAGE_20_SPV, TEXEL_DECODE_IMAGE_21_SPV, TEXEL_DECODE_IMAGE_30_SPV, TEXEL_DECODE_IMAGE_31_SPV};
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = codes[variant].size_bytes();
        moduleInfo.pCode = codes[variant].data();
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &decodeImageModules[variant]), "vkCreateShaderModule texel decode");
        VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        createInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        createInfo.stage.module = decodeImageModules[variant];
        createInfo.stage.pName = "main";
        createInfo.layout = decodeImagePipelineLayout;
        Check(CreateComputePipeline(context, createInfo, &target), "vkCreateComputePipelines texel decode");
    }
    const auto set = allocateDecodeImageSet();
    source.WriteSetWhenCreated(set, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
    destination.WriteSetWhenCreated(set, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, target);
    context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, decodeImagePipelineLayout, 0, 1, &set, 0, nullptr);
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, (extent.width + 7u) / 8u, (extent.height + 7u) / 8u, extent.depth);
}

std::uint32_t TextureDetiler::KeyClearErrors() {
    if (keyClearErrors == nullptr) return 0;
    std::uint32_t codes = 0;
    std::memcpy(&codes, keyClearErrors->Bytes().data(), sizeof(codes));
    return codes;
}

std::uint32_t TextureDetiler::CmaskErrors() {
    if (cmaskErrors == nullptr) return 0;
    std::uint32_t codes = 0;
    std::memcpy(&codes, cmaskErrors->Bytes().data(), sizeof(codes));
    return codes;
}

void TextureDetiler::DispatchCmaskClear(VkCommandBuffer commands, VkBuffer cmask, std::uint64_t cmaskOffset, std::size_t cmaskBytes, const PooledView& view, std::uint32_t width, std::uint32_t height, std::uint32_t elementBytes, const std::array<std::uint32_t, 2>& clearWords, bool write) {
    Require(commands != VK_NULL_HANDLE && cmask != VK_NULL_HANDLE && view.Started(), "a CMASK clear needs a command buffer, the CMASK buffer and the target view");
    Require(elementBytes == 1 || elementBytes == 2 || elementBytes == 4 || elementBytes == 8, "a CMASK clear of this element size is not modeled");
    Require(cmaskOffset % 4 == 0 && cmaskBytes % 4 == 0 && cmaskBytes != 0, "a CMASK clear needs a word-aligned CMASK");
    if (cmaskPipeline == VK_NULL_HANDLE) {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = index == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        Check(CreateImageWritingLayout(context, bindings, &cmaskDescriptorLayout), "vkCreateDescriptorSetLayout CMASK clear");
        const VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, 7 * sizeof(std::uint32_t)};
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &cmaskDescriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &cmaskPipelineLayout), "vkCreatePipelineLayout CMASK clear");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(CMASK_CLEAR_SPV);
        moduleInfo.pCode = CMASK_CLEAR_SPV;
        VkShaderModule cmaskModule = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &cmaskModule), "vkCreateShaderModule CMASK clear");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = cmaskModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = cmaskPipelineLayout;
        const auto created = CreateComputePipeline(context, pipelineInfo, &cmaskPipeline);
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, cmaskModule, nullptr);
        Check(created, "vkCreateComputePipelines CMASK clear");
        cmaskErrors = std::make_unique<Buffer>(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memset(cmaskErrors->Bytes().data(), 0, 4);
    }
    constexpr std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 * 2}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64}}};
    const auto set = allocateFrom(cmaskDescriptorSets, cmaskDescriptorLayout, sizes, "vkCreateDescriptorPool CMASK clear", "vkAllocateDescriptorSets CMASK clear", "vkResetDescriptorPool CMASK clear", true);
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    const auto descriptorOffset = cmaskOffset - cmaskOffset % alignment;
    const auto base = cmaskOffset - descriptorOffset;
    const VkDescriptorBufferInfo cmaskInfo{cmask, descriptorOffset, base + cmaskBytes};
    const VkDescriptorBufferInfo errorInfo{cmaskErrors->Handle(), 0, 4};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (std::uint32_t index = 0; index < writes.size(); ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].pBufferInfo = &cmaskInfo;
    writes[1].pBufferInfo = &errorInfo;
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    view.WriteSetWhenCreated(set, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
    const auto barrier = [&](VkPipelineStageFlags source, VkPipelineStageFlags destination, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) {
        VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory.srcAccessMask = sourceAccess;
        memory.dstAccessMask = destinationAccess;
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, source, destination, 0, 1, &memory, 0, nullptr, 0, nullptr);
    };
    constexpr VkAccessFlags anyWrite = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    constexpr VkAccessFlags anyAccess = anyWrite | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, anyWrite, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, cmaskPipeline);
    context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, cmaskPipelineLayout, 0, 1, &set, 0, nullptr);
    const std::uint32_t mask = elementBytes >= 4 ? 0xffffffffu : (1u << (elementBytes * 8u)) - 1u;
    const std::array<std::uint32_t, 7> push{static_cast<std::uint32_t>(base), width, height, (width + 1023u) / 1024u, clearWords[0] & mask, elementBytes == 8 ? clearWords[1] : 0u, write ? 1u : 0u};
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, cmaskPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, anyAccess);
    context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, cmask, cmaskOffset, cmaskBytes, 0xffffffffu);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, anyAccess);
}

void TextureDetiler::DispatchDccKeyClear(VkCommandBuffer commands, VkBuffer keys, std::uint64_t keysOffset, std::size_t keyCount, VkBuffer decision, const PooledView& view, std::uint32_t width, std::uint32_t height, std::uint32_t enabled, const std::array<std::uint32_t, 10>& clearWords) {
    Require(commands != VK_NULL_HANDLE && keys != VK_NULL_HANDLE && decision != VK_NULL_HANDLE && view.Started(), "a DCC key clear needs a command buffer, the key buffer, its decision buffer and the target view");
    Require(keyCount != 0 && keyCount <= UINT32_MAX && width != 0 && height != 0, "a DCC key clear needs keys and a target extent");
    if (keyClearPipeline == VK_NULL_HANDLE) {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = index == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        Check(CreateImageWritingLayout(context, bindings, &keyClearDescriptorLayout), "vkCreateDescriptorSetLayout DCC key clear");
        const VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 * sizeof(std::uint32_t)};
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &keyClearDescriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &keyClearPipelineLayout), "vkCreatePipelineLayout DCC key clear");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(DCC_KEY_CLEAR_SPV);
        moduleInfo.pCode = DCC_KEY_CLEAR_SPV;
        VkShaderModule keyClearModule = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &keyClearModule), "vkCreateShaderModule DCC key clear");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = keyClearModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = keyClearPipelineLayout;
        const auto created = CreateComputePipeline(context, pipelineInfo, &keyClearPipeline);
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, keyClearModule, nullptr);
        Check(created, "vkCreateComputePipelines DCC key clear");
        keyClearErrors = std::make_unique<Buffer>(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memset(keyClearErrors->Bytes().data(), 0, 4);
    }
    constexpr std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 * 3}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64}}};
    const auto set = allocateFrom(keyClearDescriptorSets, keyClearDescriptorLayout, sizes, "vkCreateDescriptorPool DCC key clear", "vkAllocateDescriptorSets DCC key clear", "vkResetDescriptorPool DCC key clear", true);
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    const auto descriptorOffset = keysOffset - keysOffset % alignment;
    const auto base = keysOffset - descriptorOffset;
    const VkDescriptorBufferInfo keysInfo{keys, descriptorOffset, (base + keyCount + 3u) & ~VkDeviceSize{3}};
    const VkDescriptorBufferInfo decisionInfo{decision, 0, 8 * sizeof(std::uint32_t)};
    const VkDescriptorBufferInfo errorInfo{keyClearErrors->Handle(), 0, 4};
    std::array<VkWriteDescriptorSet, 3> writes{};
    constexpr std::array<std::uint32_t, 3> bufferBindings{0u, 1u, 3u};
    for (std::uint32_t index = 0; index < writes.size(); ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = set;
        writes[index].dstBinding = bufferBindings[index];
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].pBufferInfo = &keysInfo;
    writes[1].pBufferInfo = &decisionInfo;
    writes[2].pBufferInfo = &errorInfo;
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    view.WriteSetWhenCreated(set, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
    const auto barrier = [&](VkPipelineStageFlags source, VkPipelineStageFlags destination, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) {
        VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory.srcAccessMask = sourceAccess;
        memory.dstAccessMask = destinationAccess;
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, source, destination, 0, 1, &memory, 0, nullptr, 0, nullptr);
    };
    constexpr VkAccessFlags anyWrite = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    constexpr VkAccessFlags anyAccess = anyWrite | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    std::array<std::uint32_t, 16> push{static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(keyCount), width, height, 0u, enabled};
    std::copy(clearWords.begin(), clearWords.end(), push.begin() + 6);
    barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, anyWrite, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, keyClearPipeline);
    context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, keyClearPipelineLayout, 0, 1, &set, 0, nullptr);
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, keyClearPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, 1, 1, 1);
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    push[4] = 1u;
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, keyClearPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
    context.Function<PFN_vkCmdDispatchIndirect>("vkCmdDispatchIndirect")(commands, decision, 0);
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, anyAccess);
}

void TextureDetiler::DispatchImage(VkCommandBuffer commands, TextureTileMode tileMode, std::uint32_t elementBytes, VkBuffer tiled, std::uint64_t tiledOffset, const PooledView& view, const TileMipLayout& layout, bool retile, std::uint32_t slice, const DetileWindow& window) {
    Require(commands != VK_NULL_HANDLE, "texture detiling requires an active command buffer");
    Require(window.rangeBegin < window.rangeEnd && window.tiledBase <= window.rangeBegin, "texture detiling window is empty");
    Require(tiled != VK_NULL_HANDLE && view.Started(), "texture detiling requires a tiled buffer and an image view");
    Require(ImageElementFormat(elementBytes) != VK_FORMAT_UNDEFINED, "unsupported element size for texture detiling through an image");
    std::uint32_t columnEnd = 0, rowEnd = 0;
    checkWindow(layout, window, columnEnd, rowEnd);
    const auto target = pipeline(tileMode, elementBytes, retile, false, true);
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    const auto descriptorOffset = tiledOffset - tiledOffset % alignment;
    const auto base = tiledOffset - descriptorOffset;
    Require(base <= UINT32_MAX, "texture detiling buffer offset exceeds addressable range");
    const std::uint64_t tiledBytes = window.rangeEnd == 0xffffffffu ? layout.tiledSize : std::min<std::uint64_t>(layout.tiledSize, window.rangeEnd) - window.tiledBase;
    const auto range = (base + tiledBytes + 3) / 4 * 4;
    Require(range <= context.limits.maxStorageBufferRange, "texture detiling buffer range exceeds device limits");
    const auto set = allocateSet(true);
    const VkDescriptorBufferInfo bufferInfo{tiled, descriptorOffset, range};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
    view.WriteSetWhenCreated(set, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, target);
    context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, imagePipelineLayout, 0, 1, &set, 0, nullptr);
    Push push{};
    push.srcBase = static_cast<std::uint32_t>(retile ? 0 : base);
    push.dstBase = static_cast<std::uint32_t>(retile ? base : 0);
    push.width = layout.width;
    push.height = layout.height;
    push.pitchBytes = layout.pitchBytes;
    push.blocksPerRow = layout.blocksPerRow;
    push.tail = layout.tail ? 1u : 0u;
    push.tailX = layout.tailX;
    push.tailY = layout.tailY;
    push.elementBytes = elementBytes;
    push.slice = slice;
    push.rangeBegin = window.rangeBegin;
    push.rangeEnd = window.rangeEnd;
    push.tiledBase = window.tiledBase;
    push.linearBase = 0;
    push.columnBegin = window.columnBegin;
    push.rowBegin = window.rowBegin;
    push.pipeBankXor = window.pipeBankXor;
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, imagePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
    const auto groupsX = (columnEnd - window.columnBegin + 7u) / 8u;
    const auto groupsY = (rowEnd - window.rowBegin + 7u) / 8u;
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groupsX, groupsY, 1);
}

}
