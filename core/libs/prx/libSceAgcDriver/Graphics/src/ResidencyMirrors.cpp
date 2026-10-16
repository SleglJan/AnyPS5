#include "prx/libSceAgcDriver/Graphics/include/ResidencyMirrors.hpp"
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace AgcDriver::Graphics {

MirrorChunk::MirrorChunk(const Context& context, VkDeviceSize bytes) : context(context), bytes(bytes) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer");
    try {
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const auto result = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory);
        if (result != VK_SUCCESS)
            throw std::runtime_error("residency: a mirror chunk of " + std::to_string(requirements.size >> 20) + " MiB was refused in device-local memory type " + std::to_string(allocation.memoryTypeIndex) + " (VkResult " + std::to_string(static_cast<int>(result)) + ")");
        Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory");
        VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addressInfo.buffer = buffer;
        address = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &addressInfo);
        Require(address != 0, "vkGetBufferDeviceAddressKHR returned a null address for a mirror chunk");
    } catch (...) {
        if (memory != VK_NULL_HANDLE) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        throw;
    }
}

MirrorChunk::~MirrorChunk() {
    context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

ResidencyMirrors::ResidencyMirrors(const Context& context, std::uint32_t chunkSlots)
    : context(context), chunkSlots(chunkSlots),
      copyBuffer(context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")),
      pipelineBarrier(context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")) {
    Require(context.bufferDeviceAddress, "residency mirrors need buffer device addresses");
}

std::uint64_t ResidencyMirrors::Allocate(std::uint32_t chunk) {
    if (chunk >= chunks.size()) chunks.resize(chunk + 1);
    Require(chunks[chunk] == nullptr, "residency: mirror chunk " + std::to_string(chunk) + " allocated twice");
    chunks[chunk] = std::make_shared<MirrorChunk>(context, VkDeviceSize{chunkSlots} << ResidencyPageShift);
    return chunks[chunk]->address;
}

void ResidencyMirrors::Release(std::uint32_t chunk) {
    Require(chunk < chunks.size() && chunks[chunk] != nullptr, "residency: mirror chunk " + std::to_string(chunk) + " released while not allocated");
    chunks[chunk].reset();
}

std::shared_ptr<MirrorChunk> ResidencyMirrors::Chunk(std::uint32_t chunk) const {
    Require(chunk < chunks.size() && chunks[chunk] != nullptr, "residency: mirror chunk " + std::to_string(chunk) + " is not allocated");
    return chunks[chunk];
}

std::uint64_t ResidencyMirrors::DeviceBytes() const {
    std::uint64_t bytes = 0;
    for (const auto& chunk : chunks) {
        if (chunk != nullptr) bytes += chunk->bytes;
    }
    return bytes;
}

void ResidencyMirrors::RecordCopyIn(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import) const {
    record(commands, copies, import, true);
}

void ResidencyMirrors::RecordWriteBack(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import) const {
    record(commands, copies, import, false);
}

void ResidencyMirrors::record(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import, bool intoMirror) const {
    if (copies.empty()) return;
    std::map<std::pair<VkBuffer, VkBuffer>, std::vector<VkBufferCopy>> regions;
    for (const auto& copy : copies) {
        Require(copy.offset + copy.bytes <= ResidencyPageBytes && copy.mirror.slot < chunkSlots, "residency: a copy region past its page or chunk");
        const auto location = import(copy.page);
        Require(location.buffer != VK_NULL_HANDLE, "residency: page " + std::to_string(copy.page) + " has no import to copy from or to");
        const auto mirror = Chunk(copy.mirror.chunk);
        const VkDeviceSize mirrorOffset = (VkDeviceSize{copy.mirror.slot} << ResidencyPageShift) + copy.offset;
        const VkDeviceSize importOffset = location.offset + copy.offset;
        if (intoMirror) regions[{location.buffer, mirror->buffer}].push_back({importOffset, mirrorOffset, copy.bytes});
        else regions[{mirror->buffer, location.buffer}].push_back({mirrorOffset, importOffset, copy.bytes});
    }
    const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    pipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    for (const auto& [buffers, list] : regions) copyBuffer(commands, buffers.first, buffers.second, static_cast<std::uint32_t>(list.size()), list.data());
    const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
    pipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
}

}
