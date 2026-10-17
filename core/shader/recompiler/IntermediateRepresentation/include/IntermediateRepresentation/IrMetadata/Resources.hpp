#ifndef CORE_SHADER_RECOMPILIER_INTERMEDIATEREPRESENTATION_INCLUDE_INTERMEDIATEREPRESENTATION_IRMETADATA_RESOURCES_HPP
#define CORE_SHADER_RECOMPILIER_INTERMEDIATEREPRESENTATION_INCLUDE_INTERMEDIATEREPRESENTATION_IRMETADATA_RESOURCES_HPP

#include "IntermediateRepresentation/IrMetadata/BufferFormat.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include "IntermediateRepresentation/IrOpcode.hpp"
#include "RdnaDecoder/RdnaInstruction.hpp"
#include <cstdint>
#include <limits>
#include <vector>

namespace ShaderRecompiler {

enum class ImageMipMode { None, DynamicStorage };

namespace EmulatedCompare {
inline constexpr std::uint32_t NativeOffsetUnsupported = 1u << 29u;
inline constexpr std::uint32_t Unsupported = 1u << 31u;
inline constexpr std::uint32_t RequiresSingleLevel = 1u << 30u;
inline constexpr std::uint32_t Enabled = 1u << 0u;
inline constexpr std::uint32_t FunctionShift = 1u;
inline constexpr std::uint32_t Linear = 1u << 4u;
inline constexpr std::uint32_t ClampXShift = 5u;
inline constexpr std::uint32_t ClampYShift = 7u;
inline constexpr std::uint32_t ReferenceShift = 9u;
inline constexpr std::uint32_t SingleLevel = 1u << 11u;
inline constexpr std::uint32_t BorderWhite = 1u << 12u;
inline constexpr std::uint32_t AddressWrap = 0u;
inline constexpr std::uint32_t AddressEdge = 1u;
inline constexpr std::uint32_t AddressBorder = 2u;
inline constexpr std::uint32_t ReferenceFloat = 0u;
inline constexpr std::uint32_t ReferenceUnorm = 1u;
inline constexpr std::uint32_t ReferenceSnorm = 2u;
inline constexpr std::uint32_t ReferenceZero = 3u;
[[nodiscard]] inline std::uint32_t Function(std::uint32_t state) { return (state >> FunctionShift) & 0x7u; }
[[nodiscard]] inline std::uint32_t AddressX(std::uint32_t state) { return (state >> ClampXShift) & 0x3u; }
[[nodiscard]] inline std::uint32_t AddressY(std::uint32_t state) { return (state >> ClampYShift) & 0x3u; }
[[nodiscard]] inline std::uint32_t Reference(std::uint32_t state) { return (state >> ReferenceShift) & 0x3u; }
}

inline constexpr std::uint32_t NoTable = std::numeric_limits<std::uint32_t>::max();

enum class TableOperation : std::uint8_t { Supported, Unsupported };

struct ImageResource {
    std::uint32_t source = 0;
    std::uint32_t firstUsePc = 0;
    ImageResourceClass resourceClass = ImageResourceClass::None;
    IrTextureNumericClass numericClass = IrTextureNumericClass::Unsupported;
    RdnaImageDimension dimension = RdnaImageDimension::Unknown;
    ImageMipMode mipMode = ImageMipMode::None;
    std::uint32_t mipCount = 1;
    IrBufferFormat conversionFormat = IrBufferFormat::Invalid;
    std::uint32_t shaderSwizzle = ShaderImageIdentitySwizzle;
    bool read = false;
    bool written = false;
    bool atomic = false;
    bool atomic64 = false;
    bool depthCompare = false;
    bool cube = false;
    bool r128 = false;
    bool srgbDecode = false;
    bool srgbDecodeCompatible = true;
    std::uint32_t srgbDecodeFormats = 0u;
    bool depthBits = false;
    bool depthUnorm16 = false;
    bool packed = false;
    bool fmaskCompatible = true;
    bool depthBitsCompatible = true;
    bool constantSwizzle = false;
    bool constantSwizzleCompatible = true;
    bool cubeFaceCompatible = true;
    bool flatVolumeCompatible = true;
    bool singleMipPlaneBiasCompatible = false;
    bool flatLineCompatible = true;
    bool flatVolumeLineCompatible = true;
    bool foldTexelOffsetsOnly = false;
    std::uint32_t byElements = 0;
    std::uint32_t byComponents = 0;
    IrBufferFormat packedFormat = IrBufferFormat::Invalid;
    std::uint32_t emulatedCompare = 0;
    std::uint32_t table = NoTable;
    TableOperation tableOperation = TableOperation::Supported;
    bool deviceTable = false;
    bool directRuntime = false;

    bool operator==(const ImageResource& other) const = default;
};

inline bool MemoryPathImageShape(const ImageResource& image) {
    if (image.r128 || image.packed || image.byElements != 0u) return false;
    switch (image.dimension) {
        case RdnaImageDimension::Dim1D:
        case RdnaImageDimension::Dim1DArray:
        case RdnaImageDimension::Dim2D:
        case RdnaImageDimension::Dim2DArray:
        case RdnaImageDimension::Dim3D:
            break;
        default:
            return false;
    }
    if (image.resourceClass == ImageResourceClass::Storage) return image.written || image.atomic;
    if (image.resourceClass == ImageResourceClass::Sampled) return image.read && !image.written && !image.atomic && image.fmaskCompatible && image.srgbDecodeCompatible && !image.depthCompare && image.emulatedCompare == 0u;
    return false;
}

inline bool MemoryPathImage(const ImageResource& image) {
    return image.table == NoTable && !image.deviceTable && MemoryPathImageShape(image);
}

inline bool MemoryPathRuntimeImage(const ImageResource& image) {
    return image.table != NoTable && image.deviceTable && image.directRuntime && image.tableOperation == TableOperation::Supported && MemoryPathImageShape(image);
}

enum SamplerUse : std::uint16_t {
    SamplerUseExplicitLod = 1u << 0u,
    SamplerUseImplicitLod = 1u << 1u,
    SamplerUseGradient = 1u << 2u,
    SamplerUseOffset = 1u << 3u,
    SamplerUseCompare = 1u << 4u,
    SamplerUseGather = 1u << 5u,
    SamplerUseQueryLod = 1u << 6u,
    SamplerUseAdjust = 1u << 7u,
    SamplerUseNonZeroLod = 1u << 8u,
};

inline bool ImageSampleExplicitLod(std::uint32_t flags, IrShaderStage stage) {
    return (flags & (RdnaImageSampleFlagDerivative | RdnaImageSampleFlagLod | RdnaImageSampleFlagLevelZero)) != 0u || stage != IrShaderStage::Pixel;
}

struct SamplerResource {
    static constexpr std::uint32_t NoCopy = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t source = 0;
    std::uint32_t firstUsePc = 0;
    std::uint32_t copyOf = NoCopy;
    bool forcePointFiltering = false;
    bool depthCompare = false;
    std::uint16_t uses = 0;
    bool foldTexelOffsetsOnly = false;
    std::uint32_t table = NoTable;
    bool deviceTable = false;

    bool operator==(const SamplerResource& other) const = default;
};

inline bool FoldsTexelOffsets(const SamplerResource& sampler) {
    return sampler.table == NoTable && (sampler.uses & SamplerUseOffset) != 0u && (sampler.uses & ~static_cast<std::uint32_t>(SamplerUseExplicitLod | SamplerUseOffset | SamplerUseNonZeroLod)) == 0u && !sampler.depthCompare;
}

inline bool SamplesBaseLevelIfUnnormalized(const SamplerResource& sampler) {
    return (sampler.uses & (SamplerUseImplicitLod | SamplerUseGradient)) != 0u && (sampler.uses & ~static_cast<std::uint32_t>(SamplerUseExplicitLod | SamplerUseImplicitLod | SamplerUseGradient | SamplerUseNonZeroLod)) == 0u && !sampler.depthCompare;
}

inline std::uint32_t UnnormalizedUnsupportedUses(const SamplerResource& sampler) {
    const std::uint32_t unsupported = sampler.uses & ~static_cast<std::uint32_t>(SamplerUseExplicitLod | SamplerUseOffset | SamplerUseNonZeroLod);
    return unsupported != 0u && !SamplesBaseLevelIfUnnormalized(sampler) ? unsupported : 0u;
}

inline const char* UnnormalizedUseReason(std::uint32_t uses) {
    switch (uses & (~uses + 1u)) {
        case SamplerUseImplicitLod: return "is used by an implicit-LOD sample";
        case SamplerUseGradient: return "is used by a sample with derivatives";
        case SamplerUseCompare: return "is used with depth comparison";
        case SamplerUseGather: return "is used by a gather";
        case SamplerUseQueryLod: return "is used by image_get_lod";
        default: return "is used by an image_sample_*_a variant";
    }
}

inline constexpr std::uint32_t NoSampler = std::numeric_limits<std::uint32_t>::max();

struct DeviceImageUse {
    std::uint32_t image = 0;
    std::uint32_t sampler = NoSampler;
    std::uint32_t firstUsePc = 0;
    bool levelZero = false;

    bool operator==(const DeviceImageUse& other) const = default;
};

struct SampledResourcePair {
    std::uint32_t image = 0;
    std::uint32_t sampler = 0;
    std::uint32_t firstUsePc = 0;

    bool operator==(const SampledResourcePair& other) const = default;
};

}

#endif
