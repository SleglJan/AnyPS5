#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>

namespace AgcDriver::Graphics {

namespace {

bool RecordedValuesFinished(const Recorder& recorder, const std::array<std::uint64_t, MaxRecorderStreams>& values) {
    for (std::uint32_t stream = 0; stream < values.size(); ++stream) {
        if (values[stream] != 0 && !recorder.BatchValueFinished(stream, values[stream])) return false;
    }
    return true;
}

}

void TextureDetiler::BeginBatch() {
    GuestMemory::AssertGpuLockHeld("TextureDetiler::BeginBatch");
    if (const auto* recorder = Recorder::Active(); recorder != nullptr && !recorder->Idle()) return;
    resetRing(descriptorSets, "vkResetDescriptorPool texture detiler");
    resetRing(cmaskDescriptorSets, "vkResetDescriptorPool CMASK clear");
    resetRing(keyClearDescriptorSets, "vkResetDescriptorPool DCC key clear");
    resetRing(imageDescriptorSets, "vkResetDescriptorPool texture detiler");
    resetRing(decodeImageDescriptorSets, "vkResetDescriptorPool texel decode");
}

void TextureDetiler::resetRing(DescriptorRing& ring, const char* resetWhat) {
    for (const auto pool : ring.pools) Check(context.Function<PFN_vkResetDescriptorPool>("vkResetDescriptorPool")(context.device, pool, 0), resetWhat);
    std::fill(ring.values.begin(), ring.values.end(), std::array<std::uint64_t, MaxRecorderStreams>{});
    std::fill(ring.unrecorded.begin(), ring.unrecorded.end(), std::uint8_t{0});
    ring.current = 0;
    ring.used = 0;
}

void TextureDetiler::destroyRing(DescriptorRing& ring) noexcept {
    for (const auto pool : ring.pools) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    ring = {};
}

VkDescriptorSet TextureDetiler::allocateFrom(DescriptorRing& ring, VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, const char* createWhat, const char* allocateWhat, const char* resetWhat) {
    constexpr std::uint32_t setsPerPool = 64;
    const auto* recorder = Recorder::Active();
    if (ring.pools.empty() || ring.used == setsPerPool) {
        auto next = ring.pools.size();
        for (std::size_t step = 1; recorder != nullptr && step <= ring.pools.size(); ++step) {
            const auto index = (ring.current + step) % ring.pools.size();
            if (ring.unrecorded[index] == 0 && RecordedValuesFinished(*recorder, ring.values[index])) {
                next = index;
                break;
            }
        }
        if (next < ring.pools.size()) {
            Check(context.Function<PFN_vkResetDescriptorPool>("vkResetDescriptorPool")(context.device, ring.pools[next], 0), resetWhat);
            ring.values[next] = {};
        } else {
            VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            info.maxSets = setsPerPool;
            info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
            info.pPoolSizes = sizes.data();
            ring.pools.reserve(ring.pools.size() + 1);
            ring.values.reserve(ring.values.size() + 1);
            ring.unrecorded.reserve(ring.unrecorded.size() + 1);
            VkDescriptorPool pool = VK_NULL_HANDLE;
            Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &info, nullptr, &pool), createWhat);
            ring.pools.push_back(pool);
            ring.values.emplace_back();
            ring.unrecorded.push_back(0);
        }
        ring.current = next;
        ring.used = 0;
    }
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = ring.pools[ring.current];
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), allocateWhat);
    ++ring.used;
    std::uint32_t stream = 0;
    std::uint64_t value = 0;
    if (recorder != nullptr && recorder->CallerBatchValue(stream, value)) {
        auto& recorded = ring.values[ring.current][stream];
        recorded = std::max(recorded, value);
        ring.unrecorded[ring.current] = 0;
    } else {
        ring.unrecorded[ring.current] = 1;
    }
    return set;
}

VkDescriptorSet TextureDetiler::allocateDecodeImageSet() {
    constexpr std::array<VkDescriptorPoolSize, 1> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64 * 2}}};
    return allocateFrom(decodeImageDescriptorSets, decodeImageDescriptorLayout, sizes, "vkCreateDescriptorPool texel decode", "vkAllocateDescriptorSets texel decode", "vkResetDescriptorPool texel decode");
}

VkDescriptorSet TextureDetiler::allocateSet(bool image) {
    if (image) {
        constexpr std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64}}};
        return allocateFrom(imageDescriptorSets, imageDescriptorLayout, sizes, "vkCreateDescriptorPool texture detiler", "vkAllocateDescriptorSets texture detiler", "vkResetDescriptorPool texture detiler");
    }
    constexpr std::array<VkDescriptorPoolSize, 1> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 * 2}}};
    return allocateFrom(descriptorSets, descriptorLayout, sizes, "vkCreateDescriptorPool texture detiler", "vkAllocateDescriptorSets texture detiler", "vkResetDescriptorPool texture detiler");
}

}
