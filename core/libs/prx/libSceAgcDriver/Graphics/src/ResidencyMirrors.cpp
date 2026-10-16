#include "prx/libSceAgcDriver/Graphics/include/ResidencyMirrors.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include <algorithm>
#include <map>
#include <mutex>
#include <set>
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

MirrorSpan::MirrorSpan(ResidencyDriver* residency, std::uint64_t begin, std::uint64_t end, ResidencyAccess access) {
    if (residency == nullptr || end <= begin) return;
    this->residency = residency;
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    auto& manager = residency->Manager();
    first = begin >> ResidencyPageShift;
    const auto last = (end - 1) >> ResidencyPageShift;
    pages.resize(static_cast<std::size_t>(last - first + 1));
    for (auto index = first; index <= last;) {
        if (!manager.Slot(index)) {
            ++index;
            continue;
        }
        auto next = index + 1;
        while (next <= last && manager.Slot(next)) ++next;
        manager.Use(index << ResidencyPageShift, (next - index) << ResidencyPageShift, access);
        index = next;
    }
    auto* recorder = Recorder::Active();
    std::set<std::uint32_t> kept;
    for (std::size_t k = 0; k < pages.size(); ++k) {
        const auto index = first + k;
        const auto slot = access == ResidencyAccess::Write ? manager.WriteSlot(index) : manager.Slot(index);
        if (!slot) continue;
        const auto chunk = residency->Mirrors().Chunk(slot->chunk);
        pages[k] = {chunk->buffer, VkDeviceSize{slot->slot} << ResidencyPageShift};
        if (recorder != nullptr && kept.insert(slot->chunk).second) recorder->Keep(chunk);
    }
    for (const auto& piece : Pieces(begin, end)) {
        if (piece.buffer == VK_NULL_HANDLE) manager.PrepareImportWrite(piece.begin, piece.end - piece.begin);
    }
}

const MirrorSpan::Page* MirrorSpan::page(std::uint64_t index) const {
    if (residency == nullptr || index < first || index - first >= pages.size()) return nullptr;
    return &pages[static_cast<std::size_t>(index - first)];
}

std::vector<MirrorPiece> MirrorSpan::Pieces(std::uint64_t begin, std::uint64_t end) const {
    std::vector<MirrorPiece> pieces;
    for (auto at = begin; at < end;) {
        const auto index = at >> ResidencyPageShift;
        const auto pageEnd = std::min(end, (index + 1) << ResidencyPageShift);
        const auto* entry = page(index);
        if (entry != nullptr && entry->buffer != VK_NULL_HANDLE) pieces.push_back({at, pageEnd, entry->buffer, entry->offset + (at & (ResidencyPageBytes - 1))});
        else if (!pieces.empty() && pieces.back().buffer == VK_NULL_HANDLE && pieces.back().end == at) pieces.back().end = pageEnd;
        else pieces.push_back({at, pageEnd, VK_NULL_HANDLE, 0});
        at = pageEnd;
    }
    return pieces;
}

bool MirrorSpan::Mirrored(std::uint64_t begin, std::uint64_t end) const {
    if (end <= begin) return false;
    for (auto index = begin >> ResidencyPageShift; index <= (end - 1) >> ResidencyPageShift; ++index) {
        const auto* entry = page(index);
        if (entry != nullptr && entry->buffer != VK_NULL_HANDLE) return true;
    }
    return false;
}

void MirrorSpan::NoteWritten(std::span<const MirrorPiece> pieces) const {
    if (residency == nullptr) return;
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    for (const auto& piece : pieces) {
        if (piece.end <= piece.begin) continue;
        if (piece.buffer != VK_NULL_HANDLE) residency->NoteDriverWrite(piece.begin, piece.end - piece.begin);
        else residency->RefreshImportWritten(piece.begin, piece.end - piece.begin);
    }
}

bool MirrorsDirty(std::uint64_t begin, std::uint64_t end) {
    auto* residency = ResidencyDriver::Active();
    if (residency == nullptr || end <= begin) return false;
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    auto& manager = residency->Manager();
    if (!manager.AnyDirty() || !manager.AnyResident(begin, end - begin)) return false;
    for (auto index = begin >> ResidencyPageShift; index <= (end - 1) >> ResidencyPageShift; ++index) {
        if (manager.Dirty(index) != 0) return true;
    }
    return false;
}

}
