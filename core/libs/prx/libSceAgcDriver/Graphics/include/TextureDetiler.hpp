#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_TEXTUREDETILER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_TEXTUREDETILER_HPP

#include <array>
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

    class Buffer;

    // The part of a mip one Dispatch moves: only elements whose tiled offset (relative to the mip's
    // tiled base; the linear offset of a linear surface) lies in [rangeBegin, rangeEnd), with the
    // tiled buffer holding the mip from tiledBase and the linear buffer holding `linearBytes` of
    // it from linearBase (a window of whole rows: linearBase is a row's start, every element in
    // the range lies inside the window; 0 for the rest of the mip). The element rectangle
    // [columnBegin, columnEnd) x [rowBegin, rowEnd) (an end of 0: the mip's edge) bounds the
    // elements the range can hold: only its workgroups are dispatched, the range check stays the
    // guard. The default moves the whole mip out of whole buffers.
    struct DetileWindow {
        std::uint32_t rangeBegin = 0;
        std::uint32_t rangeEnd = 0xffffffffu;
        std::uint32_t tiledBase = 0;
        std::uint32_t linearBase = 0;
        std::uint64_t linearBytes = 0;
        std::uint32_t columnBegin = 0;
        std::uint32_t columnEnd = 0;
        std::uint32_t rowBegin = 0;
        std::uint32_t rowEnd = 0;
        std::uint32_t pipeBankXor = 0;
    };

    class PooledView;

    class TextureDetiler {
    public:
        explicit TextureDetiler(const Context& context);
        ~TextureDetiler();
        TextureDetiler(const TextureDetiler&) = delete;
        TextureDetiler& operator=(const TextureDetiler&) = delete;

        // Detiles `source` (tiled) into `destination` (linear), or with `retile` writes linear `source`
        // into tiled `destination`; offsets always refer to the respective buffers, and `window`
        // restricts the move to part of the mip (a partial upload or write-back, StorageTexture),
        // leaving the other elements of both buffers alone.
        void Dispatch(VkCommandBuffer commands, TextureTileMode tileMode, std::uint32_t elementBytes, VkBuffer source, std::uint64_t sourceOffset, VkBuffer destination, std::uint64_t destinationOffset, const TileMipLayout& layout, bool retile = false, std::uint32_t slice = 0, bool thick = false, const DetileWindow& window = {});
        void DispatchImage(VkCommandBuffer commands, TextureTileMode tileMode, std::uint32_t elementBytes, VkBuffer tiled, std::uint64_t tiledOffset, const PooledView& view, const TileMipLayout& layout, bool retile = false, std::uint32_t slice = 0, const DetileWindow& window = {});
        static VkFormat ImageElementFormat(std::uint32_t elementBytes);
        // Recycles the descriptor sets of the previous batch; call before recording a new command batch.
        void BeginBatch();
        void DispatchCmaskClear(VkCommandBuffer commands, VkBuffer cmask, std::uint64_t cmaskOffset, std::size_t cmaskBytes, const PooledView& view, std::uint32_t width, std::uint32_t height, std::uint32_t elementBytes, const std::array<std::uint32_t, 2>& clearWords, bool write);
        std::uint32_t CmaskErrors();
        std::uint32_t KeyClearErrors();
        void DispatchDccKeyClear(VkCommandBuffer commands, VkBuffer keys, std::uint64_t keysOffset, std::size_t keyCount, VkBuffer decision, const PooledView& view, std::uint32_t width, std::uint32_t height, std::uint32_t enabled, const std::array<std::uint32_t, 10>& clearWords);
        void DispatchDecode(VkCommandBuffer commands, std::uint32_t guestFormat, VkBuffer source, VkBuffer destination, std::uint64_t texels);
        void DispatchSurfaceFill(VkCommandBuffer commands, VkBuffer surface, std::uint64_t bytes, std::uint32_t elementBytes, bool expandSingle, const std::array<std::uint32_t, 4>& pattern);
        void DispatchRestartStarts(VkCommandBuffer commands, VkBuffer indices, std::uint64_t indexOffset, std::uint32_t count, std::uint32_t indexBytes, VkBuffer starts, std::uint64_t startsOffset);
        void DispatchDecodeImage(VkCommandBuffer commands, std::uint32_t guestFormat, VkImageViewType viewType, const PooledView& source, const PooledView& destination, VkExtent3D extent);

    private:
        struct DescriptorRing {
            std::vector<VkDescriptorPool> pools;
            std::vector<std::array<std::uint64_t, MaxRecorderStreams>> values;
            std::vector<std::uint8_t> unrecorded;
            std::size_t current = 0;
            std::uint32_t used = 0;
        };
        VkDescriptorSet allocateFrom(DescriptorRing& ring, VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, const char* createWhat, const char* allocateWhat, const char* resetWhat, bool updateAfterBind = false);
        void resetRing(DescriptorRing& ring, const char* resetWhat);
        void destroyRing(DescriptorRing& ring) noexcept;
        VkDescriptorSet allocateDecodeImageSet();
        VkPipeline pipeline(TextureTileMode tileMode, std::uint32_t elementBytes, bool retile, bool thick, bool image = false);
        void release() noexcept;
        VkDescriptorSet allocateSet(bool image = false);
        void checkWindow(const TileMipLayout& layout, const DetileWindow& window, std::uint32_t& columnEnd, std::uint32_t& rowEnd) const;

        const Context context;
        VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkShaderModule module = VK_NULL_HANDLE;
        std::vector<std::pair<std::uint32_t, VkPipeline>> pipelines;
        DescriptorRing descriptorSets;
        VkDescriptorSetLayout cmaskDescriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout cmaskPipelineLayout = VK_NULL_HANDLE;
        VkPipeline cmaskPipeline = VK_NULL_HANDLE;
        DescriptorRing cmaskDescriptorSets;
        std::unique_ptr<Buffer> cmaskErrors;
        VkDescriptorSetLayout keyClearDescriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout keyClearPipelineLayout = VK_NULL_HANDLE;
        VkPipeline keyClearPipeline = VK_NULL_HANDLE;
        DescriptorRing keyClearDescriptorSets;
        std::unique_ptr<Buffer> keyClearErrors;
        VkDescriptorSetLayout imageDescriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout imagePipelineLayout = VK_NULL_HANDLE;
        VkShaderModule imageModule = VK_NULL_HANDLE;
        DescriptorRing imageDescriptorSets;
        VkShaderModule decodeModule = VK_NULL_HANDLE;
        std::array<VkPipeline, 2> decodePipelines{};
        VkShaderModule fillModule = VK_NULL_HANDLE;
        VkPipeline fillPipeline = VK_NULL_HANDLE;
        VkShaderModule restartModule = VK_NULL_HANDLE;
        VkPipeline restartPipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout decodeImageDescriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout decodeImagePipelineLayout = VK_NULL_HANDLE;
        std::array<VkShaderModule, 6> decodeImageModules{};
        std::array<VkPipeline, 6> decodeImagePipelines{};
        DescriptorRing decodeImageDescriptorSets;
    };

}

#endif
