#include "prx/libSceAgcDriver/Graphics/include/DescriptorHeap.hpp"
#include "RuntimeAbi.hpp"
#include <algorithm>
#include <string>

namespace AgcDriver::Graphics {

namespace {

constexpr std::array<const char*, 3> ArrayNames{"sampler", "sampled image", "storage image"};
constexpr std::uint32_t SampledImageSlots = 1u << 18u;
constexpr std::uint32_t StorageImageSlots = 1u << 16u;
constexpr std::uint32_t ReservedStageResources = 1u << 12u;

}

DescriptorHeap::Capacities DescriptorHeap::DeviceCapacities(const Context& context) {
    const auto& limits = context.descriptorIndexingLimits;
    const auto samplers = std::min({limits.maxDescriptorSetUpdateAfterBindSamplers, limits.maxPerStageDescriptorUpdateAfterBindSamplers, context.limits.maxSamplerAllocationCount});
    const auto storage = std::min({limits.maxDescriptorSetUpdateAfterBindStorageImages, limits.maxPerStageDescriptorUpdateAfterBindStorageImages, StorageImageSlots});
    const auto resources = limits.maxPerStageUpdateAfterBindResources > storage + ReservedStageResources ? limits.maxPerStageUpdateAfterBindResources - storage - ReservedStageResources : 0u;
    const auto sampled = std::min({limits.maxDescriptorSetUpdateAfterBindSampledImages, limits.maxPerStageDescriptorUpdateAfterBindSampledImages, SampledImageSlots, resources});
    return {samplers, sampled, storage};
}

DescriptorHeap::DescriptorHeap(const Context& context, Capacities capacities) : context(context) {
    Require(capacities.samplers > 1u && capacities.sampledImages > 1u && capacities.storageImages > 1u, "the descriptor heap needs room for its null descriptors and one guest descriptor of each type");
    Require(context.nullDescriptors, "the descriptor heap's null image descriptors need VK_EXT_robustness2 nullDescriptor");
    arrays[static_cast<std::size_t>(Array::Samplers)].capacity = capacities.samplers;
    arrays[static_cast<std::size_t>(Array::Samplers)].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    arrays[static_cast<std::size_t>(Array::SampledImages)].capacity = capacities.sampledImages;
    arrays[static_cast<std::size_t>(Array::SampledImages)].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    arrays[static_cast<std::size_t>(Array::StorageImages)].capacity = capacities.storageImages;
    arrays[static_cast<std::size_t>(Array::StorageImages)].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    try {
        const VkDescriptorSetLayoutCreateInfo emptyInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &emptyInfo, nullptr, &empty), "vkCreateDescriptorSetLayout empty");
        std::array<VkDescriptorPoolSize, static_cast<std::size_t>(Array::Count)> sizes{};
        for (std::size_t index = 0; index < arrays.size(); ++index) sizes[index] = {arrays[index].type, arrays[index].capacity * MaxViews};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        poolInfo.maxSets = MaxViews;
        poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        poolInfo.pPoolSizes = sizes.data();
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool descriptor heap");
        VkSamplerCreateInfo nullInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        nullInfo.magFilter = VK_FILTER_NEAREST;
        nullInfo.minFilter = VK_FILTER_NEAREST;
        nullInfo.addressModeU = nullInfo.addressModeV = nullInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        nullInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        Check(context.Function<PFN_vkCreateSampler>("vkCreateSampler")(context.device, &nullInfo, nullptr, &nullSampler), "vkCreateSampler descriptor heap null sampler");
        arrays[static_cast<std::size_t>(Array::Samplers)].entries.push_back({nullSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
        arrays[static_cast<std::size_t>(Array::SampledImages)].entries.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        arrays[static_cast<std::size_t>(Array::StorageImages)].entries.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL});
    } catch (...) {
        release();
        throw;
    }
}

DescriptorHeap::~DescriptorHeap() {
    release();
}

void DescriptorHeap::release() noexcept {
    if (pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    for (const auto& item : views) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, item.layout, nullptr);
    if (empty != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, empty, nullptr);
    if (nullSampler != VK_NULL_HANDLE) context.Function<PFN_vkDestroySampler>("vkDestroySampler")(context.device, nullSampler, nullptr);
    pool = VK_NULL_HANDLE;
    views.clear();
    empty = VK_NULL_HANDLE;
    nullSampler = VK_NULL_HANDLE;
}

void DescriptorHeap::write(const View& target, Array array, std::uint32_t first, std::uint32_t count) {
    const auto& slots = arrays[static_cast<std::size_t>(array)];
    VkWriteDescriptorSet update{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    update.dstSet = target.set;
    update.dstBinding = static_cast<std::uint32_t>(array);
    update.dstArrayElement = first;
    update.descriptorCount = count;
    update.descriptorType = slots.type;
    update.pImageInfo = slots.entries.data() + first;
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &update, 0, nullptr);
}

const DescriptorHeap::View& DescriptorHeap::view(VkShaderStageFlags stages) {
    for (const auto& item : views) {
        if (item.stages == stages) return item;
    }
    Require(views.size() < MaxViews, "the descriptor heap has a view for every one of " + std::to_string(MaxViews) + " stage combinations");
    View created{stages, VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDescriptorSetLayoutBinding, static_cast<std::size_t>(Array::Count)> bindings{};
    std::array<VkDescriptorBindingFlags, static_cast<std::size_t>(Array::Count)> flags{};
    for (std::size_t index = 0; index < arrays.size(); ++index) {
        bindings[index] = {static_cast<std::uint32_t>(index), arrays[index].type, arrays[index].capacity, stages, nullptr};
        flags[index] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlags.bindingCount = static_cast<std::uint32_t>(flags.size());
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bindingFlags};
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &created.layout), "vkCreateDescriptorSetLayout descriptor heap");
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &created.layout;
    if (const auto result = context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &created.set); result != VK_SUCCESS) {
        context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, created.layout, nullptr);
        Check(result, "vkAllocateDescriptorSets descriptor heap");
    }
    const auto& added = views.emplace_back(created);
    for (std::size_t index = 0; index < arrays.size(); ++index) write(added, static_cast<Array>(index), 0, static_cast<std::uint32_t>(arrays[index].entries.size()));
    return added;
}

VkDescriptorSetLayout DescriptorHeap::Layout(VkShaderStageFlags stages) {
    std::lock_guard lock(mutex);
    return view(stages).layout;
}

VkDescriptorSet DescriptorHeap::Set(VkShaderStageFlags stages) {
    std::lock_guard lock(mutex);
    return view(stages).set;
}

std::uint32_t DescriptorHeap::allocate(Array array, const VkDescriptorImageInfo& entry) {
    std::lock_guard lock(mutex);
    auto& slots = arrays[static_cast<std::size_t>(array)];
    std::uint32_t slot = static_cast<std::uint32_t>(slots.entries.size());
    if (!slots.free.empty()) {
        slot = slots.free.back();
        slots.free.pop_back();
        slots.entries[slot] = entry;
    } else {
        Require(slot < slots.capacity, std::string("the descriptor heap has no free ") + ArrayNames[static_cast<std::size_t>(array)] + " slot: " + std::to_string(slots.capacity) + " are live");
        slots.entries.push_back(entry);
    }
    for (const auto& item : views) write(item, array, slot, 1);
    return slot;
}

void DescriptorHeap::release(Array array, std::uint32_t slot) noexcept {
    std::lock_guard lock(mutex);
    auto& slots = arrays[static_cast<std::size_t>(array)];
    slots.entries[slot] = slots.entries[0];
    for (const auto& item : views) write(item, array, slot, 1);
    slots.free.push_back(slot);
}

std::uint32_t DescriptorHeap::AllocateSampler(VkSampler sampler) {
    return allocate(Array::Samplers, {sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
}

void DescriptorHeap::ReleaseSampler(std::uint32_t slot) noexcept {
    release(Array::Samplers, slot);
}

std::uint32_t DescriptorHeap::AllocateImage(Array array, VkImageView view, VkImageLayout layout) {
    Require(array != Array::Samplers && view != VK_NULL_HANDLE, "an image slot needs an image view");
    return allocate(array, {VK_NULL_HANDLE, view, layout});
}

std::uint32_t DescriptorHeap::AllocateImageRun(Array array, std::span<const VkImageView> imageViews, VkImageLayout layout) {
    Require(array != Array::Samplers && !imageViews.empty(), "an image slot run needs image views");
    std::lock_guard lock(mutex);
    auto& slots = arrays[static_cast<std::size_t>(array)];
    const auto count = static_cast<std::uint32_t>(imageViews.size());
    std::vector<std::uint32_t> free(slots.free.begin(), slots.free.end());
    std::sort(free.begin(), free.end());
    auto first = static_cast<std::uint32_t>(slots.entries.size());
    bool reused = false;
    for (std::size_t index = 0; index + count <= free.size(); ++index) {
        if (free[index + count - 1u] - free[index] != count - 1u) continue;
        first = free[index];
        reused = true;
        break;
    }
    if (reused) {
        std::erase_if(slots.free, [&](std::uint32_t slot) { return slot >= first && slot < first + count; });
    } else {
        Require(first + count <= slots.capacity, std::string("the descriptor heap has no run of ") + std::to_string(count) + " free " + ArrayNames[static_cast<std::size_t>(array)] + " slots: " + std::to_string(slots.capacity) + " are live");
        slots.entries.resize(first + count);
    }
    for (std::uint32_t index = 0; index < count; ++index) slots.entries[first + index] = {VK_NULL_HANDLE, imageViews[index], layout};
    for (const auto& item : views) write(item, array, first, count);
    return first;
}

std::uint32_t DescriptorHeap::ReserveImage(Array array, VkImageLayout layout) {
    Require(array != Array::Samplers, "a reserved image slot is not a sampler slot");
    return allocate(array, {VK_NULL_HANDLE, VK_NULL_HANDLE, layout});
}

void DescriptorHeap::WriteImage(Array array, std::uint32_t slot, VkImageView view, VkImageLayout layout) {
    Require(array != Array::Samplers && view != VK_NULL_HANDLE, "an image slot needs an image view");
    std::lock_guard lock(mutex);
    auto& slots = arrays[static_cast<std::size_t>(array)];
    Require(slot != 0u && slot < slots.entries.size(), std::string("a written ") + ArrayNames[static_cast<std::size_t>(array)] + " slot was never reserved");
    slots.entries[slot] = {VK_NULL_HANDLE, view, layout};
    for (const auto& item : views) write(item, array, slot, 1);
}

void DescriptorHeap::ReleaseImage(Array array, std::uint32_t slot) noexcept {
    release(array, slot);
}

void DescriptorHeap::Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout pipelineLayout, VkDescriptorSet set) const {
    context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, pipelineLayout, ShaderRecompiler::RuntimeAbi::HeapDescriptorSet, 1, &set, 0, nullptr);
}

}
