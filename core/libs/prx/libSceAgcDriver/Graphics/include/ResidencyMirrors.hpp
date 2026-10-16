#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYMIRRORS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESIDENCYMIRRORS_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

struct MirrorChunk {
    MirrorChunk(const Context& context, VkDeviceSize bytes);
    MirrorChunk(const MirrorChunk&) = delete;
    MirrorChunk& operator=(const MirrorChunk&) = delete;
    ~MirrorChunk();

    Context context;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkDeviceSize bytes = 0;
};

struct ResidencyImportLocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
};

using ResidencyImportLocator = std::function<ResidencyImportLocation(std::uint64_t page)>;

class ResidencyMirrors {
public:
    ResidencyMirrors(const Context& context, std::uint32_t chunkSlots);

    std::uint64_t Allocate(std::uint32_t chunk);
    void Release(std::uint32_t chunk);
    std::shared_ptr<MirrorChunk> Chunk(std::uint32_t chunk) const;
    void RecordCopyIn(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import) const;
    void RecordWriteBack(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import) const;
    std::uint64_t DeviceBytes() const;

private:
    void record(VkCommandBuffer commands, std::span<const ResidencyCopy> copies, const ResidencyImportLocator& import, bool intoMirror) const;

    Context context;
    std::uint32_t chunkSlots;
    std::vector<std::shared_ptr<MirrorChunk>> chunks;
    PFN_vkCmdCopyBuffer copyBuffer = nullptr;
    PFN_vkCmdPipelineBarrier pipelineBarrier = nullptr;
};

class ResidencyDriver;

struct MirrorPiece {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
};

class MirrorSpan {
public:
    MirrorSpan(ResidencyDriver* residency, std::uint64_t begin, std::uint64_t end, ResidencyAccess access);

    std::vector<MirrorPiece> Pieces(std::uint64_t begin, std::uint64_t end) const;
    bool Mirrored(std::uint64_t begin, std::uint64_t end) const;
    void NoteWritten(std::span<const MirrorPiece> pieces) const;

private:
    struct Page {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
    };

    const Page* page(std::uint64_t index) const;

    ResidencyDriver* residency = nullptr;
    std::uint64_t first = 0;
    std::vector<Page> pages;
};

bool MirrorsDirty(std::uint64_t begin, std::uint64_t end);

}

#endif
