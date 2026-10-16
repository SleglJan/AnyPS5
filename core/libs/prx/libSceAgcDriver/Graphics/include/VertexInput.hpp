#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_VERTEXINPUT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_VERTEXINPUT_HPP

#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <algorithm>
#include <atomic>
#include <list>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

struct VertexInputLayout {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
};

inline std::optional<std::uint32_t> HighestDrawIndex(std::span<const std::byte> indices, std::uint32_t indexSize, bool skipRestart) {
    Require(indexSize == 2 || indexSize == 4, "unsupported index size");
    const auto restartIndex = indexSize == 2 ? 0xffffu : 0xffffffffu;
    std::optional<std::uint32_t> highest;
    for (std::size_t offset = 0; offset + indexSize <= indices.size(); offset += indexSize) {
        std::uint32_t index = 0;
        if (indexSize == 2) {
            std::uint16_t value = 0;
            std::memcpy(&value, indices.data() + offset, sizeof(value));
            index = value;
        } else {
            std::memcpy(&index, indices.data() + offset, sizeof(index));
        }
        if (skipRestart && index == restartIndex) continue;
        highest = std::max(highest.value_or(0u), index);
    }
    return highest;
}

inline std::vector<std::uint32_t> MeshRestartTable(std::span<const std::byte> indices, std::uint32_t indexSize) {
    Require(indexSize == 2 || indexSize == 4, "unsupported index size");
    const auto restartIndex = indexSize == 2 ? 0xffffu : 0xffffffffu;
    std::vector<std::uint32_t> starts(indices.size() / indexSize);
    std::uint32_t stripStart = 0;
    bool restarted = false;
    for (std::size_t position = 0; position < starts.size(); ++position) {
        std::uint32_t index = 0;
        if (indexSize == 2) {
            std::uint16_t value = 0;
            std::memcpy(&value, indices.data() + position * 2u, sizeof(value));
            index = value;
        } else {
            std::memcpy(&index, indices.data() + position * 4u, sizeof(index));
        }
        if (index == restartIndex) {
            restarted = true;
            stripStart = static_cast<std::uint32_t>(position) + 1u;
        }
        starts[position] = stripStart;
    }
    if (!restarted) starts.clear();
    return starts;
}

}

#endif
