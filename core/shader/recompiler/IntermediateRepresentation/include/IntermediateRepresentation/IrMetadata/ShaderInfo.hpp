#ifndef CORE_SHADER_RECOMPILIER_INTERMEDIATEREPRESENTATION_INCLUDE_INTERMEDIATEREPRESENTATION_IRMETADATA_SHADERINFO_HPP
#define CORE_SHADER_RECOMPILIER_INTERMEDIATEREPRESENTATION_INCLUDE_INTERMEDIATEREPRESENTATION_IRMETADATA_SHADERINFO_HPP

#include "IntermediateRepresentation/IrMetadata/Resources.hpp"
#include "IntermediateRepresentation/IrMetadata/StageIO.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace ShaderRecompiler {

struct BufferStoreSite {
    std::uint32_t source = 0;
    std::uint32_t soffset = 0;
    bool dynamicSoffset = false;
    bool scalar = false;
    std::uint32_t offset = 0;
    bool dynamicOffset = true;
    bool fromMemory = false;
    bool address = false;
    std::uint32_t bytes = 0;
    std::uint32_t index = 0;
    bool dynamicIndex = false;

    bool operator==(const BufferStoreSite& other) const = default;
};

struct TransferKernelSite {
    enum class Kind : std::uint8_t { Fill, Copy };
    Kind kind = Kind::Fill;
    std::uint32_t components = 0;
    std::uint32_t destination = 0;
    std::uint32_t source = 0;
    std::array<std::uint32_t, 4> values{};
    std::uint32_t valueRegisters = 0;
    std::uint32_t valueMemory = 0;
    std::uint32_t valueDescriptor = 0;

    bool operator==(const TransferKernelSite& other) const = default;
};

struct LinearStoreSite {
    std::uint32_t components = 0;
    std::uint32_t elementBytes = 0;
    bool descriptorFormat = false;
    std::uint32_t groupThreads = 0;
    std::uint32_t groupIdBits = 32;

    bool operator==(const LinearStoreSite& other) const = default;
};

struct ShaderInfo {
    std::uint32_t scratchDwords = 0;
    std::uint32_t sharedMemoryBytes = 0;

    std::vector<ImageResource> images;
    std::vector<std::vector<ImageResource>> runtimeImageModes;
    std::vector<SamplerResource> samplers;
    std::vector<SampledResourcePair> sampledPairs;
    std::vector<DeviceImageUse> deviceImageUses;
    std::vector<StageInput> inputs;
    std::vector<StageOutput> outputs;
    std::array<std::uint8_t, 32> vertexFetchComponents {};
    std::int32_t vertexOffsetSgpr = -1;
    std::int32_t instanceOffsetSgpr = -1;
    bool vertexOffsetShared = false;
    bool instanceOffsetShared = false;
    bool vertexOffsetConflict = false;
    bool instanceOffsetConflict = false;
    bool hasBitwiseXor = false;
    bool usesDma = false;
    bool bdaWrites = false;
    bool usesFaultBuffer = false;
    std::vector<BufferStoreSite> bufferStores;
    bool unboundedStores = false;
    std::vector<BufferStoreSite> bufferLoads;
    bool unboundedLoads = false;
    bool dispatchThreadLimit = false;
    bool dispatchThreadLimitInMemory = false;
    bool rasterPrimitive = false;
    bool layerExported = false;
    std::optional<LinearStoreSite> linearStore;
    std::optional<TransferKernelSite> transferKernel;

    bool operator==(const ShaderInfo& other) const = default;
};

[[nodiscard]] inline std::uint32_t WorkgroupMemoryStrideDwords(const ShaderInfo& info) {
    return info.sharedMemoryBytes == 0u ? 0u : info.sharedMemoryBytes / 4u + 1u;
}

}

#endif
