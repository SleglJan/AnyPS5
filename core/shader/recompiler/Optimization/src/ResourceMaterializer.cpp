#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/SrtWalker/SrtEvaluator.hpp"
#include "Optimization/SrtWalker/SrtFlatSlotClasses.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "RdnaDecoder/RdnaImageOpDecoder.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "ImageTableAbi.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

namespace {

using ImageTableAbi::PoisonReason;

std::atomic<std::uint64_t> specializationNanoseconds{0};

bool MaterializeProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

struct DecodedImage {
    IrTextureNumericClass numericClass = IrTextureNumericClass::Unsupported;
    RdnaImageDimension dimension = RdnaImageDimension::Unknown;
    std::uint32_t mipCount = 1;
    IrBufferFormat conversionFormat = IrBufferFormat::Invalid;
    std::uint32_t shaderSwizzle = ShaderImageIdentitySwizzle;
    bool cube = false;
    bool fmask = false;
    bool depthBits = false;
    bool depthUnorm16 = false;
    IrBufferFormat packedFormat = IrBufferFormat::Invalid;
    bool srgbDecode = false;
};

ShaderBufferResource decodeBufferDescriptor(const DescriptorValue& value) {
    if (value.dwordCount != 4u) {
        throw std::runtime_error("buffer descriptor has an invalid width");
    }
    ShaderBufferResource result;
    for (std::uint32_t i = 0; i < 4u; i++) {
        result.fields[i] = value.dwords[i];
    }
    return result;
}

bool nullImageDescriptor(const DescriptorValue& descriptor) {
    return descriptor.dwords[0] == 0u && (descriptor.dwords[1] & 0xffu) == 0u;
}

ImageType rawImageType(const DescriptorValue& descriptor) {
    return static_cast<ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
}

bool nullImageDescriptor(const ImageResource& image, const DescriptorValue& descriptor) {
    if (nullImageDescriptor(descriptor)) return true;
    return static_cast<std::uint32_t>(rawImageType(descriptor)) <= 1u && image.directRuntime && image.read && !image.written && !image.atomic;
}

IrBufferFormat rawImageFormat(const DescriptorValue& descriptor) {
    return static_cast<IrBufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
}

std::uint32_t descriptorImageSwizzle(const DescriptorValue& descriptor) {
    return descriptor.dwords[3] & 0xfffu;
}

bool descriptorIsCube(const DescriptorValue& descriptor) {
    return rawImageType(descriptor) == ImageType::Cube;
}

RdnaImageDimension descriptorDimension(const DescriptorValue& descriptor, RdnaImageDimension requested) {
    const bool wantArray = requested == RdnaImageDimension::Dim1DArray || requested == RdnaImageDimension::Dim2DArray || requested == RdnaImageDimension::Dim2DMsaaArray;
    switch (rawImageType(descriptor)) {
        case ImageType::Color1D:
            return RdnaImageDimension::Dim1D;
        case ImageType::Color1DArray:
            return wantArray ? RdnaImageDimension::Dim1DArray : RdnaImageDimension::Dim1D;
        case ImageType::Color3D:
            return RdnaImageDimension::Dim3D;
        case ImageType::Cube:
            return RdnaImageDimension::Dim2DArray;
        case ImageType::Color2DArray:
            return wantArray || requested == RdnaImageDimension::Dim3D ? RdnaImageDimension::Dim2DArray : RdnaImageDimension::Dim2D;
        case ImageType::Color2DMsaaArray:
            return wantArray ? RdnaImageDimension::Dim2DMsaaArray : RdnaImageDimension::Dim2DMsaa;
        case ImageType::Color2D:
            return RdnaImageDimension::Dim2D;
        case ImageType::Color2DMsaa:
            return RdnaImageDimension::Dim2DMsaa;
        default:
            throw std::runtime_error("image descriptor has an unsupported type");
    }
}

bool validImageDescriptor(const DescriptorValue& descriptor, bool r128) {
    const auto type = rawImageType(descriptor);
    const auto format = rawImageFormat(descriptor);
    if (type < ImageType::Color1D || format == IrBufferFormat::Invalid) {
        return false;
    }
    if (r128 && type != ImageType::Color1D && type != ImageType::Color2D && type != ImageType::Color2DMsaa) {
        return false;
    }
    if (type == ImageType::Color2DMsaa || type == ImageType::Color2DMsaaArray) {
        const auto baseLevel = (descriptor.dwords[3] >> 12u) & 0xfu;
        const auto fragments = (descriptor.dwords[3] >> 16u) & 0xfu;
        const auto maxMip = (descriptor.dwords[5] >> 4u) & 0xfu;
        return baseLevel == 0u && fragments >= 1u && fragments <= 3u && (r128 || maxMip == fragments);
    }
    return true;
}

std::uint32_t storageMipCount(const ImageResource& base, const DescriptorValue& descriptor) {
    if (base.mipMode != ImageMipMode::DynamicStorage || nullImageDescriptor(base, descriptor)) {
        return 1u;
    }
    const auto mipBase = (descriptor.dwords[3] >> 12u) & 0xfu;
    const auto mipLast = (descriptor.dwords[3] >> 16u) & 0xfu;
    return mipBase <= mipLast ? mipLast - mipBase + 1u : 0u;
}

struct ImageDecode {
    DecodedImage decoded;
    std::optional<PoisonReason> reason;
    std::string detail;
};

ImageDecode rejectImage(PoisonReason reason, std::string detail) {
    ImageDecode result;
    result.reason = reason;
    result.detail = std::move(detail);
    return result;
}

PoisonReason invalidImageReason(const DescriptorValue& descriptor) {
    const auto type = rawImageType(descriptor);
    if (type < ImageType::Color1D) {
        return PoisonReason::NotImage;
    }
    if (rawImageFormat(descriptor) == IrBufferFormat::Invalid) {
        return PoisonReason::InvalidFormat;
    }
    if (type == ImageType::Color2DMsaa || type == ImageType::Color2DMsaaArray) {
        return PoisonReason::Multisampled;
    }
    return PoisonReason::Dimension;
}

ImageDecode inspectImageDescriptor(const DescriptorValue& descriptor, const ImageResource& base, std::uint32_t srgbDecodeFormats) {
    ImageDecode result;
    auto& decoded = result.decoded;
    decoded.mipCount = storageMipCount(base, descriptor);
    if (decoded.mipCount == 0u) {
        return rejectImage(PoisonReason::InvalidFormat, "storage image descriptor has an invalid mip range");
    }
    if (nullImageDescriptor(base, descriptor)) {
        decoded.numericClass = base.atomic ? IrTextureNumericClass::Uint : IrTextureNumericClass::Float;
        decoded.dimension = RdnaImageDimension::Dim2D;
        decoded.cube = false;
        return result;
    }
    if (base.resourceClass == ImageResourceClass::None || (base.atomic && base.resourceClass != ImageResourceClass::Storage)) {
        throw std::runtime_error("image resource has an invalid class");
    }
    if (!validImageDescriptor(descriptor, base.r128)) {
        return rejectImage(invalidImageReason(descriptor), "image descriptor is invalid");
    }
    decoded.dimension = descriptorDimension(descriptor, base.dimension);
    decoded.cube = descriptorIsCube(descriptor);
    const auto format = rawImageFormat(descriptor);
    if (base.atomic64 && format != IrBufferFormat::Format32_32UInt && format != IrBufferFormat::Format32_32SInt && format != IrBufferFormat::Format32_32Float) {
        return rejectImage(PoisonReason::InvalidFormat, "64-bit atomic image descriptor uses an unsupported format " + std::to_string(static_cast<std::uint32_t>(format)));
    }
    if (base.atomic && !base.atomic64 && format != IrBufferFormat::Format32UInt && format != IrBufferFormat::Format32SInt && format != IrBufferFormat::Format32Float) {
        return rejectImage(PoisonReason::InvalidFormat, "atomic image descriptor uses an unsupported format " + std::to_string(static_cast<std::uint32_t>(format)));
    }
    const bool storage = base.resourceClass == ImageResourceClass::Storage;
    const bool table = base.table != NoTable;
    const bool cpuTable = table && !base.deviceTable;
    decoded.fmask = IsFmaskTextureFormat(format);
    if (decoded.fmask && !base.fmaskCompatible) return rejectImage(PoisonReason::Fmask, "FMASK requires a direct 32-bit image load");
    if (decoded.fmask && (storage || base.depthCompare || table)) {
        return rejectImage(PoisonReason::Fmask, "FMASK requires a direct sampled image load");
    }
    if (base.packed) {
        if (cpuTable || (!storage && descriptorImageSwizzle(descriptor) != ShaderImageIdentitySwizzle)) {
            return rejectImage(PoisonReason::InvalidFormat, "packed image access requires a direct image, with identity swizzle when sampled");
        }
        decoded.packedFormat = format;
    }
    if (base.byElements != 0u) {
        const bool eightBit = format == IrBufferFormat::Format8UNorm || format == IrBufferFormat::Format8SNorm || format == IrBufferFormat::Format8UInt || format == IrBufferFormat::Format8SInt;
        const bool sixteenBit = format == IrBufferFormat::Format16UNorm || format == IrBufferFormat::Format16SNorm || format == IrBufferFormat::Format16UInt || format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format16Float;
        const bool eightBitPair = format == IrBufferFormat::Format8_8UNorm || format == IrBufferFormat::Format8_8SNorm || format == IrBufferFormat::Format8_8UInt || format == IrBufferFormat::Format8_8SInt;
        const bool packedEight = format == IrBufferFormat::Format8UNorm || format == IrBufferFormat::Format8UInt || format == IrBufferFormat::Format8SInt;
        const bool packedSixteen = format == IrBufferFormat::Format16UNorm || format == IrBufferFormat::Format16UInt || format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format8_8UNorm || format == IrBufferFormat::Format8_8UInt || format == IrBufferFormat::Format8_8SInt;
        const bool measured = base.packed ? packedEight || (base.byElements == 2u && packedSixteen)
                                          : base.byElements == 4u ? base.byComponents == 1u && eightBit : base.byElements == 2u && (base.byComponents == 1u ? eightBit || sixteenBit : base.byComponents == 2u && eightBitPair);
        if (!measured || descriptorImageSwizzle(descriptor) != ShaderImageIdentitySwizzle || rawImageType(descriptor) != ImageType::Color2D || cpuTable) {
            return rejectImage(PoisonReason::InvalidFormat, base.packed ? "MIMG PCK2/PCK4 requires a direct, identity-swizzled 2D UNORM, UINT or SINT image whose elements fill one dword: R8, R16 or RG8 for PCK2, R8 for PCK4"
                                                                     : "MIMG BY2/BY4 requires a direct, identity-swizzled 2D image whose elements fill one dword: R8, R16 or RG8 for BY2, R8 for BY4");
        }
    }
    const bool decodedCopy = !storage && !base.srgbDecodeCompatible && SampledThroughDecodedCopy(format);
    decoded.conversionFormat = !decodedCopy && RemapTextureFormat(format) != format ? format : IrBufferFormat::Invalid;
    if (format == IrBufferFormat::Format11_11_10UNorm || format == IrBufferFormat::Format10_11_11Float) {
        const bool floating = format == IrBufferFormat::Format10_11_11Float;
        if (base.depthCompare) return rejectImage(PoisonReason::Conversion, floating ? "compares against a converted float image, which is not implemented" : "compares against a converted unorm image, which is not implemented");
        if (!base.depthBitsCompatible) return rejectImage(PoisonReason::Conversion, floating ? "reads or writes a converted float image with 16-bit data, which is not implemented" : "reads or writes a converted unorm image with 16-bit data, which is not implemented");
        for (std::uint32_t component = 0; component < 4u; ++component) {
            if (((descriptorImageSwizzle(descriptor) >> (component * 3u)) & 7u) == 7u) return rejectImage(PoisonReason::Conversion, "selects a channel the converted image format does not have");
        }
    }
    decoded.srgbDecode = !storage && !base.depthCompare && (srgbDecodeFormats & SrgbDecodeBit(format)) != 0u;
    if (decoded.srgbDecode && !base.srgbDecodeCompatible) {
        return rejectImage(PoisonReason::SrgbUnsupported, "samples or gathers an sRGB image the device cannot sample, which is not implemented");
    }
    if (storage || decoded.conversionFormat != IrBufferFormat::Invalid) {
        decoded.shaderSwizzle = descriptorImageSwizzle(descriptor);
    }
    const bool wideSint = format == IrBufferFormat::Format32SInt || format == IrBufferFormat::Format32_32SInt || format == IrBufferFormat::Format32_32_32_32SInt;
    const bool narrowSint = format == IrBufferFormat::Format16SInt || format == IrBufferFormat::Format8_8SInt || format == IrBufferFormat::Format16_16SInt || format == IrBufferFormat::Format8_8_8_8SInt || format == IrBufferFormat::Format16_16_16_16SInt;
    const bool rawSintStorage = storage && (wideSint || (narrowSint && !base.packed)) && base.written && !base.read && !base.atomic;
    decoded.numericClass = base.atomic ? IrTextureNumericClass::Uint : decodedCopy ? IrTextureNumericClass::Float : SampledTextureNumericClass(format);
    if (!storage && !base.depthCompare && IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3])) {
        decoded.depthBits = true;
        if (!base.depthBitsCompatible) return rejectImage(PoisonReason::DepthBits16, "runtime image reads depth bits as unsupported 16-bit results");
        decoded.depthUnorm16 = DepthBitsTextureWidth(descriptor.dwords[1], descriptor.dwords[3]) == 16u;
        decoded.numericClass = IrTextureNumericClass::Float;
        decoded.shaderSwizzle = descriptorImageSwizzle(descriptor);
    }
    if (storage) {
        if ((!rawSintStorage && decoded.numericClass == IrTextureNumericClass::Sint) || decoded.numericClass == IrTextureNumericClass::Unsupported) {
            return rejectImage(PoisonReason::NumericClass, "storage image descriptor uses an unsupported format");
        }
        if (rawSintStorage) {
            decoded.numericClass = IrTextureNumericClass::Uint;
        }
        if ((rawSintStorage || (base.atomic && format == IrBufferFormat::Format32SInt)) && !base.packed) {
            decoded.conversionFormat = format;
        }
    } else if (decoded.numericClass == IrTextureNumericClass::Unsupported || (base.depthCompare && decoded.numericClass != IrTextureNumericClass::Float)) {
        return rejectImage(PoisonReason::NumericClass, "sampled image descriptor uses an unsupported format");
    }
    return result;
}

DecodedImage decodeImageDescriptor(const DescriptorValue& descriptor, const ImageResource& base, std::uint32_t srgbDecodeFormats) {
    auto result = inspectImageDescriptor(descriptor, base, srgbDecodeFormats);
    if (result.reason.has_value()) {
        throw std::runtime_error(result.detail);
    }
    return result.decoded;
}

std::string hexText(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(value));
    return text;
}

enum class RecordState : std::uint8_t { OutsideDomain, Read, Unmapped };

void materializeSnapshot(const IrResourcePlan& plan, const SrtRuntime& runtime, SrtWalker& walker, ResourceSnapshot& snapshot, std::vector<std::uint8_t>& activeSources) {
    snapshot = ResourceSnapshot{};
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        const auto words = plan.uniformFill.fill.words;
        if (words == 0u || words > plan.uniformFill.values.size()) {
            throw std::runtime_error("uniform fill plan has an invalid word count");
        }
        std::array<std::uint32_t, 4> stored{};
        walker.EvaluateUniformValues(plan, std::span(plan.uniformFill.values).first(words), runtime, std::span(stored).first(words));
        for (std::uint32_t i = 1; i < words; i++) {
            if (stored[i] != stored[0]) {
                throw std::runtime_error("uniform fill values diverge at runtime");
            }
        }
        snapshot.uniformFill = plan.uniformFill.fill;
        snapshot.uniformFill.value = stored[0];
    }
    if (runtime.userData.size() < plan.userDataCount) {
        throw std::runtime_error("runtime user data is smaller than the shader user data count");
    }
    snapshot.userData.assign(runtime.userData.begin(), runtime.userData.begin() + plan.userDataCount);

    std::vector<DescriptorValue> values;
    walker.EvaluateRuntimeSources(plan, plan.materializationSources, runtime, values, snapshot.flattenedSrt, plan.cleanFlatSlots, activeSources, &snapshot.srtPoison, &snapshot.nullRootReads);

    std::size_t cursor = 0;

    snapshot.images.resize(plan.info.images.size());
    for (std::uint32_t i = 0; i < plan.info.images.size(); i++) {
        const auto& image = plan.info.images[i];
        if (image.source >= plan.descriptorSources.size()) {
            throw std::runtime_error("image resource references an unknown descriptor source");
        }
        if (image.table == NoTable) {
            throw std::runtime_error("an image outside the image tables is not implemented");
        }
        snapshot.images[i].dwordCount = 8u;
    }

    snapshot.samplers.resize(plan.info.samplers.size());
    for (std::uint32_t i = 0; i < plan.info.samplers.size(); i++) {
        if (plan.info.samplers[i].deviceTable) {
            snapshot.samplers[i].dwordCount = 4u;
            continue;
        }
        if (cursor >= values.size()) {
            throw std::runtime_error("materialization sources are missing sampler descriptors");
        }
        snapshot.samplers[i] = values[cursor];
        cursor++;
    }

    snapshot.stores.resize(plan.info.bufferStores.size());
    for (std::uint32_t i = 0; i < plan.info.bufferStores.size(); i++) {
        if (plan.info.bufferStores[i].fromMemory) continue;
        if (cursor >= values.size()) {
            throw std::runtime_error("materialization sources are missing stored buffer descriptors");
        }
        snapshot.stores[i] = values[cursor];
        cursor++;
    }

    snapshot.loads.resize(plan.info.bufferLoads.size());
    for (std::uint32_t i = 0; i < plan.info.bufferLoads.size(); i++) {
        if (plan.info.bufferLoads[i].fromMemory) continue;
        if (cursor >= values.size()) {
            throw std::runtime_error("materialization sources are missing loaded buffer descriptors");
        }
        snapshot.loads[i] = values[cursor];
        cursor++;
    }
}

std::uint32_t colorCompareReference(IrBufferFormat format) {
    switch (format) {
    case IrBufferFormat::Format16_16UNorm:
    case IrBufferFormat::Format11_11_10UNorm: case IrBufferFormat::Format10_11_11UNorm: case IrBufferFormat::Format2_10_10_10UNorm:
    case IrBufferFormat::Format10_10_10_2UNorm: case IrBufferFormat::Format16_16_16_16UNorm:
        return EmulatedCompare::ReferenceUnorm;
    case IrBufferFormat::Format8UNorm: case IrBufferFormat::Format8_8UNorm: case IrBufferFormat::Format8_8_8_8UNorm:
    case IrBufferFormat::Format8Srgb: case IrBufferFormat::Format8_8Srgb: case IrBufferFormat::Format8_8_8_8Srgb:
        return EmulatedCompare::ReferenceZero;
    case IrBufferFormat::Format8SNorm: case IrBufferFormat::Format16SNorm: case IrBufferFormat::Format8_8SNorm: case IrBufferFormat::Format16_16SNorm:
    case IrBufferFormat::Format11_11_10SNorm: case IrBufferFormat::Format10_11_11SNorm: case IrBufferFormat::Format2_10_10_10SNorm:
    case IrBufferFormat::Format10_10_10_2SNorm: case IrBufferFormat::Format8_8_8_8SNorm: case IrBufferFormat::Format16_16_16_16SNorm:
        return EmulatedCompare::ReferenceSnorm;
    case IrBufferFormat::Format16Float: case IrBufferFormat::Format16_16Float: case IrBufferFormat::Format11_11_10Float:
    case IrBufferFormat::Format10_11_11Float: case IrBufferFormat::Format32_32Float: case IrBufferFormat::Format16_16_16_16Float:
        return EmulatedCompare::ReferenceFloat;
    default:
        throw std::runtime_error("comparison sampling of a color texture is implemented only for float, unorm and snorm formats (format " + std::to_string(static_cast<std::uint32_t>(format)) + ")");
    }
}

bool nativeCompare(const ImageResource& image, const DescriptorValue& descriptor) {
    if (!image.depthCompare || descriptor.dwordCount != 8u || nullImageDescriptor(image, descriptor)) return true;
    const auto format = rawImageFormat(descriptor);
    return format == IrBufferFormat::Format32Float || format == IrBufferFormat::Format16UNorm || IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3]);
}

std::uint32_t colorCompareState(const ImageResource& image, const DescriptorValue& descriptor, std::span<const std::array<std::uint32_t, 4>> samplers) {
    if (nativeCompare(image, descriptor)) return 0u;
    const auto format = rawImageFormat(descriptor);
    if ((image.emulatedCompare & EmulatedCompare::Unsupported) != 0u) throw std::runtime_error("unsupported color comparison image instructions");
    if ((image.emulatedCompare & EmulatedCompare::RequiresSingleLevel) != 0u && ((descriptor.dwords[3] >> 12u) & 0xfu) != ((descriptor.dwords[3] >> 16u) & 0xfu)) throw std::runtime_error("color comparison requires a single mip level");
    const auto reference = colorCompareReference(format);
    const auto type = rawImageType(descriptor);
    if (type != ImageType::Color2D && type != ImageType::Color2DArray && type != ImageType::Cube) throw std::runtime_error("comparison sampling of a color texture is implemented only for 2D, 2D array and cube views");
    bool redChannel = false;
    bool compareChannel = false;
    for (std::uint32_t channel = 0u; channel < 4u; ++channel) {
        const auto select = (descriptorImageSwizzle(descriptor) >> (channel * 3u)) & 0x7u;
        if (select == 2u || select == 3u) throw std::runtime_error("comparison sampling of a color texture is implemented only when every view channel is a constant or a color channel");
        redChannel = redChannel || select == 4u;
        compareChannel = compareChannel || select >= 4u;
    }
    if (compareChannel && !redChannel) throw std::runtime_error("comparison sampling of a color texture is implemented only when a view channel selects red");
    std::optional<std::uint32_t> samplerState;
    for (const auto& words : samplers) {
        const auto clampX = words[0] & 0x7u;
        const auto clampY = (words[0] >> 3u) & 0x7u;
        const auto function = (words[0] >> 12u) & 0x7u;
        const bool unnormalized = ((words[0] >> 15u) & 0x1u) != 0u;
        if (((words[0] >> 29u) & 0x3u) != 0u) throw std::runtime_error("comparison sampling of a color texture through a min or max reduction sampler is not implemented");
        const auto magFilter = (words[2] >> 20u) & 0x3u;
        const auto minFilter = (words[2] >> 22u) & 0x3u;
        const bool constantZero = reference == EmulatedCompare::ReferenceZero;
        if (type == ImageType::Cube && magFilter == 1u && ((words[0] >> 28u) & 1u) == 0u && !constantZero) {
            throw std::runtime_error("bilinear cube comparison requires DISABLE_CUBE_WRAP to avoid seamless face filtering");
        }
        const auto addressMode = [](std::uint32_t clamp) {
            if (clamp == 0u) return EmulatedCompare::AddressWrap;
            if (clamp == 2u) return EmulatedCompare::AddressEdge;
            if (clamp == 6u) return EmulatedCompare::AddressBorder;
            throw std::runtime_error("comparison sampling of a color texture is implemented only with wrap, clamp-to-edge or clamp-to-border addressing");
        };
        const auto addressX = addressMode(clampX);
        const auto addressY = addressMode(clampY);
        const auto borderType = (words[3] >> 30u) & 0x3u;
        const bool border = addressX == EmulatedCompare::AddressBorder || addressY == EmulatedCompare::AddressBorder;
        if (border && borderType == 3u) throw std::runtime_error("comparison sampling of a color texture with a border color table is not implemented");
        if (!constantZero && (magFilter != minFilter || magFilter > 1u)) throw std::runtime_error("comparison sampling of a color texture is implemented only with equal point or bilinear minification and magnification filters");
        if (unnormalized) throw std::runtime_error("comparison sampling of a color texture does not implement unnormalized coordinates");
        const auto state = EmulatedCompare::Enabled | (function << EmulatedCompare::FunctionShift) | (magFilter == 1u ? EmulatedCompare::Linear : 0u)
            | (addressX << EmulatedCompare::ClampXShift) | (addressY << EmulatedCompare::ClampYShift) | (border && borderType == 2u ? EmulatedCompare::BorderWhite : 0u);
        if (samplerState.has_value() && *samplerState != state) throw std::runtime_error("comparison sampling of a color texture through samplers that disagree is not implemented");
        samplerState = state;
    }
    if (!samplerState.has_value()) throw std::runtime_error("comparison sampling of a color texture has no paired sampler");
    const bool singleLevel = ((descriptor.dwords[3] >> 12u) & 0xfu) == ((descriptor.dwords[3] >> 16u) & 0xfu);
    return *samplerState | (reference << EmulatedCompare::ReferenceShift) | (singleLevel ? EmulatedCompare::SingleLevel : 0u);
}

void materializeSrtGuards(const IrResourcePlan& plan, ResourceSnapshot& snapshot) {
    if (snapshot.flattenedSrt.size() != plan.srtReads.size()) throw std::runtime_error("runtime SRT size differs from the static interface");
    const auto& guarded = plan.guardedSrtSlots;
    const auto flags = snapshot.flattenedSrt.size();
    snapshot.flattenedSrt.resize(flags + guarded.size(), 0u);
    for (std::uint32_t record = 0u; record < snapshot.srtPoison.size(); ++record) {
        const auto& poison = snapshot.srtPoison[record];
        const auto found = std::lower_bound(guarded.begin(), guarded.end(), poison.slot);
        if (found == guarded.end() || *found != poison.slot) throw std::runtime_error("SRT read at flat offset " + std::to_string(poison.slot) + " reads inaccessible memory but has no guard");
        snapshot.flattenedSrt[flags + static_cast<std::size_t>(found - guarded.begin())] = record + 1u;
        snapshot.flattenedSrt.insert(snapshot.flattenedSrt.end(), {poison.pc, static_cast<std::uint32_t>(poison.address), static_cast<std::uint32_t>(poison.address >> 32u)});
    }
}

}

std::vector<std::uint32_t> ResourceMaterializer::ImageTableMap(const ShaderInfo& info) {
    if (std::ranges::none_of(info.images, [](const ImageResource& image) { return image.table != NoTable; })) return {};
    std::vector<std::uint32_t> map(ImageTableAbi::HeaderWords + std::max<std::size_t>(info.samplers.size(), 1u), 0u);
    map[ImageTableAbi::HeaderVersion] = ImageTableAbi::Version;
    return map;
}

namespace {

ImageResource deviceInterfaceImage(const ImageResource& image) {
    auto normalized = image;
    normalized.source = 0u;
    normalized.firstUsePc = 0u;
    if (normalized.table != NoTable) normalized.table = 0u;
    return normalized;
}

std::uint64_t mixImageInterface(std::uint64_t hash, const ImageResource& image) {
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ull;
    };
    mix(static_cast<std::uint64_t>(image.resourceClass));
    mix(static_cast<std::uint64_t>(image.numericClass));
    mix(static_cast<std::uint64_t>(image.dimension));
    mix(static_cast<std::uint64_t>(image.mipMode));
    mix(image.mipCount);
    mix(static_cast<std::uint64_t>(image.conversionFormat));
    mix(image.shaderSwizzle);
    mix((image.read ? 1u : 0u) | (image.written ? 2u : 0u) | (image.atomic ? 4u : 0u) | (image.atomic64 ? 8u : 0u) | (image.depthCompare ? 16u : 0u) | (image.cube ? 32u : 0u) | (image.r128 ? 64u : 0u) | (image.srgbDecode ? 128u : 0u));
    mix((image.srgbDecodeCompatible ? 1u : 0u) | (image.depthBits ? 2u : 0u) | (image.depthUnorm16 ? 4u : 0u) | (image.packed ? 8u : 0u) | (image.fmaskCompatible ? 16u : 0u) | (image.depthBitsCompatible ? 32u : 0u) | (image.constantSwizzle ? 64u : 0u) | (image.constantSwizzleCompatible ? 128u : 0u));
    mix((image.flatVolumeCompatible ? 1u : 0u) | (image.flatLineCompatible ? 2u : 0u) | (image.foldTexelOffsetsOnly ? 4u : 0u) | (image.table != NoTable ? 8u : 0u) | (image.deviceTable ? 16u : 0u) | (image.cubeFaceCompatible ? 32u : 0u) | (image.singleMipPlaneBiasCompatible ? 64u : 0u) | (image.flatVolumeLineCompatible ? 128u : 0u) | (image.directRuntime ? 256u : 0u));
    mix(image.srgbDecodeFormats);
    mix(image.byElements);
    mix(image.byComponents);
    mix(static_cast<std::uint64_t>(image.packedFormat));
    mix(image.emulatedCompare);
    mix(static_cast<std::uint64_t>(image.tableOperation));
    return hash;
}

struct DeviceImageInterfaceRegistry {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<const DeviceImageInterface>> interfaces;
};

DeviceImageInterfaceRegistry& deviceImageInterfaces() {
    static DeviceImageInterfaceRegistry registry;
    return registry;
}

}

std::uint64_t ResourceMaterializer::DeviceImageInterfaceId(const ImageResource& image, std::span<const ImageResource> modes) {
    auto hash = mixImageInterface(0xcbf29ce484222325ull, deviceInterfaceImage(image));
    hash = (hash ^ modes.size()) * 0x100000001b3ull;
    for (const auto& mode : modes) hash = mixImageInterface(hash, deviceInterfaceImage(mode));
    return hash;
}

void ResourceMaterializer::RegisterDeviceImageInterfaces(const ShaderInfo& info) {
    for (std::uint32_t resource = 0; resource < info.images.size(); ++resource) {
        const auto& image = info.images[resource];
        if (!image.deviceTable) continue;
        if (resource >= info.runtimeImageModes.size()) throw std::runtime_error("device image interface: the runtime image modes are missing");
        const auto& modes = info.runtimeImageModes[resource];
        const auto id = DeviceImageInterfaceId(image, modes);
        const auto same = [&](const DeviceImageInterface& entry) {
            if (entry.image != deviceInterfaceImage(image) || entry.modes.size() != modes.size()) return false;
            for (std::size_t index = 0; index < modes.size(); ++index) {
                if (entry.modes[index] != deviceInterfaceImage(modes[index])) return false;
            }
            return true;
        };
        auto& registry = deviceImageInterfaces();
        std::lock_guard lock(registry.mutex);
        if (const auto found = registry.interfaces.find(id); found != registry.interfaces.end()) {
            if (!same(*found->second)) throw std::runtime_error("device image interface: two image interfaces share the id 0x" + hexText(id));
            continue;
        }
        auto entry = std::make_shared<DeviceImageInterface>();
        entry->id = id;
        entry->image = deviceInterfaceImage(image);
        for (const auto& mode : modes) entry->modes.push_back(deviceInterfaceImage(mode));
        registry.interfaces.emplace(id, std::move(entry));
    }
}

std::shared_ptr<const DeviceImageInterface> ResourceMaterializer::FindDeviceImageInterface(std::uint64_t id) {
    auto& registry = deviceImageInterfaces();
    std::lock_guard lock(registry.mutex);
    const auto found = registry.interfaces.find(id);
    return found != registry.interfaces.end() ? found->second : nullptr;
}

void ResourceMaterializer::EvaluateMemoryBuffers(const IrResourcePlan& plan, const SrtRuntime& runtime, std::vector<DescriptorValue>& stores, std::vector<DescriptorValue>& loads) {
    evaluateBufferSites(plan, runtime, stores, loads, true);
}

void ResourceMaterializer::EvaluateBufferSites(const IrResourcePlan& plan, const SrtRuntime& runtime, std::vector<DescriptorValue>& stores, std::vector<DescriptorValue>& loads) {
    evaluateBufferSites(plan, runtime, stores, loads, false);
}

void ResourceMaterializer::evaluateBufferSites(const IrResourcePlan& plan, const SrtRuntime& runtime, std::vector<DescriptorValue>& stores, std::vector<DescriptorValue>& loads, bool memoryOnly) {
    Detail::Evaluator evaluator(plan, runtime);
    const auto evaluate = [&](const std::vector<BufferStoreSite>& sites, std::vector<DescriptorValue>& values) {
        values.resize(sites.size());
        for (std::size_t index = 0; index < sites.size(); ++index) {
            if (memoryOnly && !sites[index].fromMemory) continue;
            const auto& descriptor = plan.descriptorSources.at(sites[index].source);
            DescriptorValue value;
            value.dwordCount = descriptor.dwordCount;
            bool evaluated = sites[index].address ? descriptor.dwordCount == 2u || descriptor.dwordCount == 3u : descriptor.dwordCount == 4u || (descriptor.dwordCount == 5u && sites[index].dynamicSoffset);
            for (std::uint32_t word = 0; evaluated && word < descriptor.dwordCount; ++word) evaluated = descriptor.dwords[word] != nullptr && evaluator.Evaluate(descriptor.dwords[word], value.dwords[word]);
            values[index] = evaluated ? value : DescriptorValue{};
        }
    };
    evaluate(plan.info.bufferStores, stores);
    evaluate(plan.info.bufferLoads, loads);
}

std::vector<DeviceImageTableAbi::Expectation> ResourceMaterializer::ExpectedDeviceImages(const IrResourcePlan& plan, const SrtRuntime& runtime, bool* complete) {
    namespace Abi = DeviceImageTableAbi;
    std::vector<Abi::Expectation> expected;
    if (complete != nullptr) *complete = true;
    if (plan.info.deviceImageUses.empty()) return expected;
    Detail::Evaluator evaluator(plan, runtime);
    std::vector<std::uint8_t> active(plan.info.images.size(), 1u);
    std::vector<std::vector<std::uint32_t>> taken(plan.controlFlow.size());
    std::vector<std::uint8_t> visited(plan.controlFlow.size());
    if (!plan.controlFlow.empty()) {
        for (const auto& block : plan.controlFlow) {
            for (const auto image : block.images) active.at(image) = 0u;
        }
        std::vector<std::uint32_t> pending{0};
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (visited.at(index) != 0u) continue;
            visited[index] = 1u;
            const auto& block = plan.controlFlow[index];
            for (const auto image : block.images) active[image] = 1u;
            std::uint32_t condition = 0;
            if (block.condition != nullptr && block.successors.size() == 2u && evaluator.Evaluate(block.condition, condition)) {
                taken[index] = {block.successors[condition != 0u ? 0u : 1u]};
            } else {
                taken[index] = block.successors;
            }
            pending.insert(pending.end(), taken[index].begin(), taken[index].end());
        }
    }
    const auto avoidable = [&](std::uint32_t avoided) {
        std::vector<std::uint8_t> seen(plan.controlFlow.size());
        std::vector<std::uint32_t> pending{0};
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (index == avoided || seen.at(index) != 0u) continue;
            seen[index] = 1u;
            if (taken[index].empty()) return true;
            pending.insert(pending.end(), taken[index].begin(), taken[index].end());
        }
        return false;
    };
    const auto executed = [&](std::uint32_t image) {
        if (plan.controlFlow.empty()) return true;
        for (std::uint32_t index = 0; index < plan.controlFlow.size(); ++index) {
            if (visited[index] != 0u && std::ranges::find(plan.controlFlow[index].images, image) != plan.controlFlow[index].images.end() && !avoidable(index)) return true;
        }
        return false;
    };
    using Words = std::vector<IrValue*>;
    using Values = std::vector<std::uint32_t>;
    const auto unwrapped = [](IrValue* word) {
        IrValue* value = word->Resolve();
        while (value->Opcode() == IrOpcode::ReadFirstLane) value = value->Argument(0)->Resolve();
        return value;
    };
    const auto joining = [](const IrValue* value) { return value->IsPhi() || (value->Opcode() == IrOpcode::SelectU32 && !value->Argument(0)->Resolve()->HasImmediate()); };
    const auto incoming = [&](const Words& words) {
        Words start;
        for (auto* word : words) start.push_back(unwrapped(word));
        std::vector<Words> sets{start};
        for (std::uint32_t round = 0; round < 4u && sets.size() <= 16u; ++round) {
            std::vector<Words> next;
            bool expanded = false;
            for (const auto& set : sets) {
                const auto found = std::ranges::find_if(set, joining);
                if (found == set.end()) {
                    next.push_back(set);
                    continue;
                }
                expanded = true;
                const IrValue* join = *found;
                if (!join->IsPhi()) {
                    const IrValue* condition = join->Argument(0)->Resolve();
                    for (const std::size_t operand : {1u, 2u}) {
                        auto candidate = set;
                        for (auto*& word : candidate) {
                            if (word->Opcode() == IrOpcode::SelectU32 && word->Argument(0)->Resolve() == condition) word = unwrapped(word->Argument(operand));
                        }
                        next.push_back(candidate);
                    }
                    continue;
                }
                for (std::size_t edge = 0; edge < join->ArgumentCount(); ++edge) {
                    auto candidate = set;
                    for (auto*& word : candidate) {
                        if (!word->IsPhi() || word->ArgumentCount() != join->ArgumentCount()) continue;
                        bool joined = true;
                        for (std::size_t index = 0; joined && index < join->ArgumentCount(); ++index) joined = word->PhiBlock(index) == join->PhiBlock(index);
                        if (joined) word = unwrapped(word->Argument(edge));
                    }
                    next.push_back(candidate);
                }
            }
            sets = std::move(next);
            if (!expanded) break;
        }
        return sets;
    };
    const auto wordsOf = [&](std::uint32_t source, std::uint32_t count, Words& words) {
        const auto& descriptor = plan.descriptorSources.at(source);
        if (descriptor.dwordCount != count) return false;
        words.assign(descriptor.dwords.begin(), descriptor.dwords.begin() + count);
        return std::ranges::none_of(words, [](const IrValue* word) { return word == nullptr; });
    };
    const auto evaluated = [&](const Words& words, Values& values) {
        values.assign(words.size(), 0u);
        for (std::size_t word = 0; word < words.size(); ++word) {
            if (!evaluator.Evaluate(words[word], values[word])) return false;
        }
        return true;
    };
    const auto valuesOf = [&](const Words& words, bool& joined) {
        std::vector<Values> values(1);
        joined = false;
        if (evaluated(words, values.front())) return values;
        values.clear();
        joined = true;
        for (const auto& candidate : incoming(words)) {
            Values value;
            if (evaluated(candidate, value) && std::ranges::find(values, value) == values.end()) values.push_back(std::move(value));
        }
        return values;
    };
    const auto add = [&](const Abi::Expectation& expectation) {
        const auto same = std::ranges::find_if(expected, [&](const Abi::Expectation& held) { return held.key == expectation.key && held.pc == expectation.pc && held.image == expectation.image && held.mapSampler == expectation.mapSampler; });
        if (same == expected.end()) expected.push_back(expectation);
        else same->candidate = same->candidate && expectation.candidate;
    };
    expected.reserve(plan.info.deviceImageUses.size());
    for (const auto& use : plan.info.deviceImageUses) {
        if (active.at(use.image) == 0u) continue;
        const auto& image = plan.info.images.at(use.image);
        Words imageWords;
        const bool deviceSampler = use.sampler != NoSampler && plan.info.samplers.at(use.sampler).deviceTable;
        Words samplerWords;
        if (!wordsOf(image.source, Abi::DescriptorWords, imageWords) || (deviceSampler && !wordsOf(plan.info.samplers.at(use.sampler).source, Abi::SamplerWords, samplerWords))) {
            if (complete != nullptr) *complete = false;
            continue;
        }
        bool joinedImage = false;
        bool joinedSampler = false;
        const auto samplers = deviceSampler ? valuesOf(samplerWords, joinedSampler) : std::vector<Values>{};
        const auto images = valuesOf(imageWords, joinedImage);
        if (complete != nullptr && (images.empty() || (deviceSampler && samplers.empty()))) *complete = false;
        for (const auto& value : images) {
            Abi::Expectation expectation;
            expectation.pc = use.firstUsePc;
            expectation.image = use.image;
            expectation.candidate = joinedImage;
            auto& key = expectation.key;
            std::copy_n(value.begin(), Abi::DescriptorWords, key.begin());
            if (Abi::NullDescriptor(key[0], key[1])) continue;
            DescriptorValue descriptor;
            descriptor.dwordCount = Abi::DescriptorWords;
            std::copy_n(key.begin(), Abi::DescriptorWords, descriptor.dwords.begin());
            if (nullImageDescriptor(image, descriptor)) continue;
            if (!validImageDescriptor(descriptor, image.r128)) {
                if (!joinedImage && executed(use.image)) throw std::runtime_error("runtime image descriptor is invalid");
                continue;
            }
            if (!deviceSampler) {
                if (use.sampler != NoSampler) expectation.mapSampler = use.sampler;
                add(expectation);
                continue;
            }
            for (const auto& sampler : samplers) {
                auto sampled = expectation;
                sampled.candidate = joinedImage || joinedSampler;
                std::copy_n(sampler.begin(), Abi::SamplerWords, sampled.key.begin() + Abi::SamplerWord);
                const bool exactAtBase = use.levelZero && (sampled.key[Abi::SamplerWord] & Abi::SamplerUnnormalized) != 0u;
                sampled.key[Abi::SamplerClassWord] = Abi::RuntimeSampler | (exactAtBase ? Abi::SamplerExactAtBase : 0u);
                add(sampled);
            }
        }
    }
    return expected;
}

std::vector<ImageResource> ResourceMaterializer::RuntimeImageModes(const ImageResource& image) {
    if (image.tableOperation == TableOperation::Unsupported) return {};
    std::vector<ImageResource> modes;
    const bool storage = image.resourceClass == ImageResourceClass::Storage;
    const auto append = [&](IrTextureNumericClass numeric, IrBufferFormat conversion, IrBufferFormat packed, bool depth, bool unorm16) {
        auto mode = image;
        mode.numericClass = numeric;
        mode.conversionFormat = conversion;
        mode.packedFormat = packed;
        mode.depthBits = depth;
        mode.depthUnorm16 = unorm16;
        mode.cube = false;
        mode.mipCount = mode.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u;
        mode.shaderSwizzle = ShaderImageIdentitySwizzle;
        if (conversion == IrBufferFormat::Format11_11_10UNorm || conversion == IrBufferFormat::Format10_11_11Float) mode.shaderSwizzle = 0x2acu;
        modes.push_back(mode);
        const bool volumeLoad = (image.dimension == RdnaImageDimension::Dim2D || image.dimension == RdnaImageDimension::Dim2DArray) && image.fmaskCompatible;
        const bool volumeStore = image.dimension == RdnaImageDimension::Dim2DArray && storage && image.written && !image.read && !image.atomic && image.mipMode != ImageMipMode::DynamicStorage;
        const bool volumePlaneBias = image.dimension == RdnaImageDimension::Dim2D && image.singleMipPlaneBiasCompatible && !storage && conversion == IrBufferFormat::Invalid;
        if ((volumeLoad || volumeStore || volumePlaneBias) && !depth && packed == IrBufferFormat::Invalid && image.byElements == 0u) {
            auto volume = mode;
            volume.dimension = RdnaImageDimension::Dim3D;
            modes.push_back(volume);
        }
        if (image.dimension == RdnaImageDimension::Dim3D && image.flatVolumeCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            for (const auto dimension : {RdnaImageDimension::Dim2D, RdnaImageDimension::Dim2DArray}) {
                auto plane = mode;
                plane.dimension = dimension;
                modes.push_back(plane);
            }
        }
        if (image.dimension == RdnaImageDimension::Dim3D && image.flatVolumeLineCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto line = mode;
            line.dimension = RdnaImageDimension::Dim1D;
            modes.push_back(line);
        }
        if (image.dimension == RdnaImageDimension::Dim2D && storage && image.written && !image.read && !image.atomic && image.mipMode != ImageMipMode::DynamicStorage && !depth && packed == IrBufferFormat::Invalid && image.byElements == 0u) {
            auto line = mode;
            line.dimension = RdnaImageDimension::Dim1D;
            modes.push_back(line);
        }
        if (image.dimension == RdnaImageDimension::Dim2D && image.cubeFaceCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto face = mode;
            face.dimension = RdnaImageDimension::Dim2DArray;
            face.cube = true;
            modes.push_back(face);
        }
        if ((image.dimension == RdnaImageDimension::Dim2D || image.dimension == RdnaImageDimension::Dim2DArray) && image.flatLineCompatible && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto line = mode;
            line.dimension = RdnaImageDimension::Dim1D;
            modes.push_back(line);
        }
        if (image.dimension == RdnaImageDimension::Dim1D && !storage && !depth && conversion == IrBufferFormat::Invalid && packed == IrBufferFormat::Invalid) {
            auto plane = mode;
            plane.dimension = RdnaImageDimension::Dim2D;
            modes.push_back(plane);
        }
        if (image.dimension == RdnaImageDimension::Dim1DArray || image.dimension == RdnaImageDimension::Dim2DArray || image.dimension == RdnaImageDimension::Dim2DMsaaArray) {
            auto plain = mode;
            plain.dimension = image.dimension == RdnaImageDimension::Dim1DArray ? RdnaImageDimension::Dim1D : image.dimension == RdnaImageDimension::Dim2DArray ? RdnaImageDimension::Dim2D : RdnaImageDimension::Dim2DMsaa;
            modes.push_back(plain);
        }
        if (image.dimension == RdnaImageDimension::Dim2DArray) {
            mode.cube = true;
            modes.push_back(mode);
        }
    };
    if (image.atomic) {
        append(IrTextureNumericClass::Uint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
    } else if (image.packed) {
        for (std::uint32_t value = 1u; value <= 77u; ++value) {
            const auto format = static_cast<IrBufferFormat>(value);
            const auto info = GetFormatInfo(format);
            if (info.packedBitfield || info.componentCount == 0u || (storage && info.byteSize == 12u)) continue;
            if (image.byElements != 0u && info.byteSize * 8u * image.byElements > 32u) continue;
            const auto numeric = storage && (format == IrBufferFormat::Format32SInt || format == IrBufferFormat::Format32_32SInt || format == IrBufferFormat::Format32_32_32_32SInt) ? IrTextureNumericClass::Uint : SampledTextureNumericClass(format);
            bool supported = numeric != IrTextureNumericClass::Unsupported && (!storage || numeric != IrTextureNumericClass::Sint);
            for (std::uint32_t component = 0u; component < info.componentCount; ++component) {
                const auto bits = info.componentBits[component];
                const bool exactRead = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint || (info.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (info.type == SpirvFormatComponentType::Float && bits == 32u);
                const bool exactWrite = info.type == SpirvFormatComponentType::Uint || (bits == 32u && (info.type == SpirvFormatComponentType::Sint || info.type == SpirvFormatComponentType::Float));
                supported &= storage ? exactWrite : exactRead;
            }
            if (supported) append(numeric, IrBufferFormat::Invalid, format, false, false);
        }
    } else {
        if ((image.emulatedCompare & EmulatedCompare::NativeOffsetUnsupported) == 0u) append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
        if (!image.depthCompare) {
            append(IrTextureNumericClass::Uint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
            if (!storage) append(IrTextureNumericClass::Sint, IrBufferFormat::Invalid, IrBufferFormat::Invalid, false, false);
            append(IrTextureNumericClass::Uint, IrBufferFormat::Format11_11_10UInt, IrBufferFormat::Invalid, false, false);
            if (image.srgbDecodeCompatible && image.depthBitsCompatible) {
                append(IrTextureNumericClass::Uint, IrBufferFormat::Format11_11_10UNorm, IrBufferFormat::Invalid, false, false);
                append(IrTextureNumericClass::Uint, IrBufferFormat::Format10_11_11Float, IrBufferFormat::Invalid, false, false);
            }
            if (!storage) {
                if (image.depthBitsCompatible) {
                    append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, true, false);
                    append(IrTextureNumericClass::Float, IrBufferFormat::Invalid, IrBufferFormat::Invalid, true, true);
                }
            }
        }
    }
    if (storage && !image.packed && !image.atomic64 && ((image.written && !image.read) || image.atomic)) {
        constexpr std::array formats{IrBufferFormat::Format32SInt, IrBufferFormat::Format32_32SInt, IrBufferFormat::Format32_32_32_32SInt, IrBufferFormat::Format16SInt, IrBufferFormat::Format8_8SInt, IrBufferFormat::Format16_16SInt, IrBufferFormat::Format8_8_8_8SInt, IrBufferFormat::Format16_16_16_16SInt};
        for (const auto format : formats) {
            if (image.atomic && format != IrBufferFormat::Format32SInt) continue;
            append(IrTextureNumericClass::Uint, format, IrBufferFormat::Invalid, false, false);
        }
    }
    if (image.depthCompare && image.deviceTable && (image.emulatedCompare & EmulatedCompare::Unsupported) == 0u && (image.dimension == RdnaImageDimension::Dim2D || image.dimension == RdnaImageDimension::Dim2DArray)) {
        auto mode = image;
        mode.numericClass = IrTextureNumericClass::Float;
        mode.depthCompare = false;
        mode.cube = false;
        mode.emulatedCompare |= EmulatedCompare::Enabled;
        mode.conversionFormat = IrBufferFormat::Invalid;
        mode.packedFormat = IrBufferFormat::Invalid;
        mode.shaderSwizzle = ShaderImageIdentitySwizzle;
        modes.push_back(mode);
        if (image.dimension == RdnaImageDimension::Dim2DArray) {
            auto plain = mode;
            plain.dimension = RdnaImageDimension::Dim2D;
            modes.push_back(plain);
            mode.cube = true;
            modes.push_back(mode);
        }
    }
    if (image.srgbDecodeFormats != 0u && image.srgbDecodeCompatible && !image.depthCompare && !storage && !image.packed) {
        const auto count = modes.size();
        for (std::size_t index = 0; index < count; ++index) {
            const auto& base = modes[index];
            if (base.numericClass != IrTextureNumericClass::Float || base.conversionFormat != IrBufferFormat::Invalid || base.packedFormat != IrBufferFormat::Invalid || base.depthBits) continue;
            auto mode = base;
            mode.srgbDecode = true;
            modes.push_back(mode);
        }
    }
    if (image.constantSwizzleCompatible && !storage && !image.depthCompare && !image.packed && image.deviceTable) {
        for (const auto numeric : {IrTextureNumericClass::Float, IrTextureNumericClass::Uint, IrTextureNumericClass::Sint}) {
            auto mode = image;
            mode.numericClass = numeric;
            mode.conversionFormat = IrBufferFormat::Invalid;
            mode.packedFormat = IrBufferFormat::Invalid;
            mode.depthBits = false;
            mode.depthUnorm16 = false;
            mode.cube = false;
            mode.mipCount = 1u;
            mode.shaderSwizzle = ShaderImageIdentitySwizzle;
            mode.constantSwizzle = true;
            modes.push_back(mode);
        }
    }
    if (image.foldTexelOffsetsOnly && std::ranges::any_of(modes, [](const ImageResource& mode) { return UnnormalizedSampleMismatch(mode) == nullptr; })) {
        std::erase_if(modes, [](const ImageResource& mode) { return UnnormalizedSampleMismatch(mode) != nullptr; });
    }
    return modes;
}

const char* ResourceMaterializer::VolumePlaneBiasSamplerMismatch(const ImageResource& image, const ImageResource& mode, std::span<const std::uint32_t, 4> sampler) {
    if (!image.singleMipPlaneBiasCompatible || image.dimension != RdnaImageDimension::Dim2D || mode.dimension != RdnaImageDimension::Dim3D) return nullptr;
    constexpr std::uint32_t forceUnnormalized = 1u << 15u;
    if ((sampler[0] & forceUnnormalized) != 0u || ((sampler[0] >> 29u) & 3u) > 2u || ((sampler[2] >> 20u) & 0xfu) != 0u || ((sampler[2] >> 26u) & 3u) > 1u) return "2D biased sampling of a one-slice volume requires a normalized point sampler";
    return nullptr;
}

const char* ResourceMaterializer::UnnormalizedSampleMismatch(const ImageResource& image) {
    if (image.constantSwizzle) return nullptr;
    if ((image.dimension != RdnaImageDimension::Dim1D && image.dimension != RdnaImageDimension::Dim2D) || image.cube) return "samples a 1D-array, 2D-array, 3D, cube or multisampled image";
    if (image.depthCompare) return "is used with depth comparison";
    if (image.conversionFormat != IrBufferFormat::Invalid || image.packed) return "samples an image that needs a format conversion or packed access";
    return nullptr;
}

bool ResourceMaterializer::BindsDecodedCopy(const ImageResource& mode, const DescriptorValue& descriptor) {
    return mode.resourceClass == ImageResourceClass::Sampled && mode.numericClass == IrTextureNumericClass::Float && mode.conversionFormat == IrBufferFormat::Invalid && !mode.constantSwizzle && descriptor.dwordCount == 8u && !nullImageDescriptor(mode, descriptor) && SampledThroughDecodedCopy(rawImageFormat(descriptor));
}

RuntimeImageModeMatch ResourceMaterializer::TryRuntimeImageMode(const ImageResource& image, const DescriptorValue& descriptor, std::span<const ImageResource> modes) {
    RuntimeImageModeMatch result;
    const auto reject = [&](PoisonReason reason, std::string detail) {
        result.reason = reason;
        result.detail = std::move(detail);
        return result;
    };
    if (descriptor.dwordCount != 8u) throw std::runtime_error("runtime image descriptor must contain eight dwords");
    if (modes.empty()) throw std::runtime_error("prepared runtime image modes are missing");
    if (nullImageDescriptor(image, descriptor)) {
        result.mode = 0u;
        return result;
    }
    const auto inspected = inspectImageDescriptor(descriptor, image, image.srgbDecodeFormats);
    if (inspected.reason.has_value()) return reject(*inspected.reason, inspected.detail);
    const auto& decoded = inspected.decoded;
    const auto format = rawImageFormat(descriptor);
    const bool emulated = image.depthCompare && format != IrBufferFormat::Format32Float && format != IrBufferFormat::Format16UNorm && !IsDepthBitsTexture(descriptor.dwords[1], descriptor.dwords[3]);
    if (!emulated && (image.emulatedCompare & EmulatedCompare::NativeOffsetUnsupported) != 0u) return reject(PoisonReason::UnsupportedOperation, "native comparison with a nonconstant texel offset requires VK_KHR_maintenance8 and shaderImageGatherExtended");
    if (image.packed && decoded.packedFormat != IrBufferFormat::Invalid) {
        const auto packedFormat = GetFormatInfo(decoded.packedFormat);
        if (packedFormat.packedBitfield) return reject(PoisonReason::InvalidFormat, "runtime packed image accesses a bitfield format");
        const bool storage = image.resourceClass == ImageResourceClass::Storage;
        if (storage && packedFormat.byteSize == 12u) return reject(PoisonReason::InvalidFormat, "runtime packed image uses a format the hardware does not write");
        for (std::uint32_t component = 0u; component < packedFormat.componentCount; ++component) {
            const auto bits = packedFormat.componentBits[component];
            const bool exact = storage ? packedFormat.type == SpirvFormatComponentType::Uint || (bits == 32u && (packedFormat.type == SpirvFormatComponentType::Sint || packedFormat.type == SpirvFormatComponentType::Float)) : packedFormat.type == SpirvFormatComponentType::Uint || packedFormat.type == SpirvFormatComponentType::Sint || (packedFormat.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (packedFormat.type == SpirvFormatComponentType::Float && bits == 32u);
            if (!exact) return reject(PoisonReason::InvalidFormat, storage ? "runtime packed image bits are not reproducible through the view" : "runtime packed image bits are not recoverable from the view");
        }
    }
    if (decoded.mipCount > (image.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u)) return reject(PoisonReason::InvalidFormat, "runtime storage image mip capacity exceeded");
    if ((descriptorImageSwizzle(descriptor) & 06666u) == 0u && !decoded.fmask && !decoded.depthBits && decoded.conversionFormat == IrBufferFormat::Invalid && !decoded.srgbDecode) {
        for (std::uint32_t index = 0u; index < modes.size(); ++index) {
            if (modes[index].constantSwizzle && modes[index].numericClass == decoded.numericClass) {
                result.mode = index;
                return result;
            }
        }
    }
    if (image.singleMipPlaneBiasCompatible && image.dimension == RdnaImageDimension::Dim2D && decoded.dimension == RdnaImageDimension::Dim3D) {
        if ((descriptor.dwords[4] & 0x1fffu) != 0u || (descriptor.dwords[3] & 0xff000u) != 0u || (descriptor.dwords[5] & 0xf0u) != 0u) throw std::runtime_error("2D biased sampling of a volume requires one slice and mip level zero only");
    }
    for (std::uint32_t index = 0u; index < modes.size(); ++index) {
        const auto& mode = modes[index];
        if (mode.constantSwizzle) continue;
        if (((mode.emulatedCompare & EmulatedCompare::Enabled) != 0u) != emulated) continue;
        if (decoded.fmask) {
            if (mode.packedFormat == IrBufferFormat::Fmask8_S2_F1) {
                result.mode = index;
                return result;
            }
            continue;
        }
        if (mode.numericClass == decoded.numericClass && mode.dimension == decoded.dimension && mode.conversionFormat == decoded.conversionFormat && mode.packedFormat == decoded.packedFormat && mode.cube == decoded.cube && mode.depthBits == decoded.depthBits && mode.depthUnorm16 == decoded.depthUnorm16 && mode.srgbDecode == decoded.srgbDecode) {
            result.mode = index;
            return result;
        }
    }
    if (image.dimension == RdnaImageDimension::Dim1D && decoded.dimension != RdnaImageDimension::Dim1D) return reject(PoisonReason::Dimension, "image address has too few coordinate components");
    if (image.foldTexelOffsetsOnly) return reject(PoisonReason::UnsupportedOperation, "image sample has a texel offset that is not a constant, which image sampling takes only with VK_KHR_maintenance8 and shaderImageGatherExtended; without them it takes only images an unnormalized S# can sample");
    const auto describe = [](IrTextureNumericClass numeric, RdnaImageDimension dimension, IrBufferFormat conversion, IrBufferFormat packed, bool cube, bool depthBits, bool depthUnorm16, bool srgb) {
        return "{class " + std::to_string(static_cast<int>(numeric)) + ", dim " + std::to_string(static_cast<int>(dimension)) + ", conversion " + std::to_string(static_cast<int>(conversion)) + ", packed " + std::to_string(static_cast<int>(packed)) + ", cube " + std::to_string(cube) + ", depthBits " + std::to_string(depthBits) + ", unorm16 " + std::to_string(depthUnorm16) + ", srgb " + std::to_string(srgb) + "}";
    };
    std::string message = "image descriptor is incompatible with the static runtime image interface: descriptor format " + std::to_string(static_cast<int>(format)) + " decodes to " + describe(decoded.numericClass, decoded.dimension, decoded.conversionFormat, decoded.packedFormat, decoded.cube, decoded.depthBits, decoded.depthUnorm16, decoded.srgbDecode) + (emulated ? " (emulated compare)" : "") + "; modes:";
    for (const auto& mode : modes) message += " " + describe(mode.numericClass, mode.dimension, mode.conversionFormat, mode.packedFormat, mode.cube, mode.depthBits, mode.depthUnorm16, mode.srgbDecode);
    char words[96];
    std::snprintf(words, sizeof(words), "; T# %08x %08x %08x %08x %08x %08x %08x %08x", descriptor.dwords[0], descriptor.dwords[1], descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4], descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]);
    const auto reason = emulated ? PoisonReason::ColorCompare
        : decoded.fmask ? PoisonReason::Fmask
        : decoded.depthBits ? PoisonReason::DepthBits16
        : decoded.srgbDecode ? PoisonReason::SrgbUnsupported
        : decoded.conversionFormat != IrBufferFormat::Invalid ? PoisonReason::Conversion
        : decoded.dimension == RdnaImageDimension::Dim2DMsaa || decoded.dimension == RdnaImageDimension::Dim2DMsaaArray ? PoisonReason::Multisampled
        : std::none_of(modes.begin(), modes.end(), [&](const ImageResource& mode) { return mode.dimension == decoded.dimension && mode.cube == decoded.cube; }) ? PoisonReason::Dimension
        : PoisonReason::NumericClass;
    if (reason == PoisonReason::SrgbUnsupported) return reject(reason, "samples or gathers an sRGB image the device cannot sample, which is not implemented: " + message + words);
    return reject(reason, message + words);
}

std::uint32_t ResourceMaterializer::ColorCompareState(const ImageResource& image, const DescriptorValue& descriptor, std::span<const std::uint32_t, 4> sampler) {
    const std::array<std::array<std::uint32_t, 4>, 1> samplers{{{sampler[0], sampler[1], sampler[2], sampler[3]}}};
    return colorCompareState(image, descriptor, samplers);
}

void ResourceMaterializer::ApplyStaticInterface(IrProgram& program, bool nativeSampleOffsets) const {
    auto& resources = program.Resources();
    if (!resources.resourceTrackingComplete || !resources.srtPlanComplete) throw std::runtime_error("static resource interface requires a completed resource plan");
    auto images = resources.info.images;
    if (!nativeSampleOffsets) {
        for (const auto& block : program.Blocks()) {
            for (const auto* inst : block->Instructions()) {
                if (inst->Opcode() != IrOpcode::ImageSampleRaw) continue;
                const auto& memory = resources.memoryInfo.at(inst->Flags<MemoryFlags>().index);
                if ((memory.imageSampleFlags & RdnaImageSampleFlagOffset) == 0u) continue;
                const auto* address = inst->Argument(2)->Resolve();
                const auto component = GetRdnaImageAddressComponentLayout(memory.imageSampleFlags, 0u);
                const auto argument = component.bitOffset / 32u;
                if (component.bitWidth == 32u && argument < address->ArgumentCount() && address->Argument(argument)->Resolve()->HasImmediate()) continue;
                if ((memory.imageSampleFlags & RdnaImageSampleFlagCompare) != 0u) {
                    images.at(memory.resource).emulatedCompare |= EmulatedCompare::NativeOffsetUnsupported;
                } else {
                    images.at(memory.resource).foldTexelOffsetsOnly = true;
                    resources.info.samplers.at(memory.sampler).foldTexelOffsetsOnly = true;
                }
            }
        }
    }
    bool tables = false;
    for (auto& image : images) {
        image.srgbDecodeFormats = resources.srgbDecodeFormats;
        image.numericClass = image.atomic ? IrTextureNumericClass::Uint : IrTextureNumericClass::Float;
        image.mipCount = image.mipMode == ImageMipMode::DynamicStorage ? RuntimeAbi::StorageMipSlots : 1u;
        tables = true;
        const auto modes = RuntimeImageModes(image);
        if (modes.empty() || modes.size() > ImageTableAbi::MaxModes) image.tableOperation = TableOperation::Unsupported;
    }
    for (const auto& pair : resources.info.sampledPairs) {
        if (pair.image >= images.size() || pair.sampler >= resources.info.samplers.size()) throw std::runtime_error("static sampled pair is out of range");
        auto& sampler = resources.info.samplers[pair.sampler];
        sampler.depthCompare = sampler.depthCompare || images[pair.image].depthCompare;
    }
    resources.info.images = std::move(images);
    resources.guardedSrtSlots = Detail::ComputeGuardedFlatSlots(resources);
    resources.srtGuardOffset = static_cast<std::uint32_t>(resources.srtReads.size());
    const bool storedImages = std::ranges::any_of(resources.info.images, [](const ImageResource& image) { return image.written || image.atomic; });
    if (tables || storedImages || !resources.guardedSrtSlots.empty()) resources.info.usesFaultBuffer = true;
    PrepareImageModes(resources.info);
}

void ResourceMaterializer::PrepareImageModes(ShaderInfo& info) {
    info.runtimeImageModes.clear();
    info.runtimeImageModes.reserve(info.images.size());
    for (const auto& image : info.images) info.runtimeImageModes.push_back(RuntimeImageModes(image));
}

namespace {

enum class Evaluability : std::uint8_t { Maybe, Never, Open };

Evaluability classifyEvaluability(const IrResourcePlan& plan, IrValue* value, std::array<IrValue*, 4>& operands, std::size_t& count) {
    count = 0;
    if (value->HasImmediate()) {
        switch (value->Type()) {
            case IrType::Bool:
            case IrType::U8:
            case IrType::U16:
            case IrType::U32:
            case IrType::U64:
            case IrType::F32: return Evaluability::Maybe;
            default: return Evaluability::Never;
        }
    }
    if (value->Opcode() == IrOpcode::Void || Detail::Evaluator::Reads(plan, *value) == Detail::EvaluatorReads::Nothing) return Evaluability::Never;
    const auto strict = [&](std::size_t arity) {
        if (value->ArgumentCount() < arity) return Evaluability::Maybe;
        for (std::size_t index = 0; index < arity; ++index) {
            if (value->Argument(index) == nullptr) return Evaluability::Maybe;
            operands[count++] = value->Argument(index)->Resolve();
        }
        return Evaluability::Open;
    };
    switch (value->Opcode()) {
        case IrOpcode::UndefU1:
        case IrOpcode::UndefU8:
        case IrOpcode::UndefU16:
        case IrOpcode::UndefU32:
        case IrOpcode::UndefU64: return Evaluability::Never;
        case IrOpcode::Phi: {
            IrValue* invariant = ResolveInvariantPhi(plan, value);
            if (invariant == nullptr) return Evaluability::Never;
            operands[count++] = invariant->Resolve();
            return Evaluability::Open;
        }
        case IrOpcode::BitCastU32F32:
        case IrOpcode::BitCastF32U32:
        case IrOpcode::ConvertF32U32:
        case IrOpcode::ConvertU32F32:
        case IrOpcode::FPTrunc32:
        case IrOpcode::FPIsNan32:
        case IrOpcode::BitwiseNot32:
        case IrOpcode::LogicalNot: return strict(1);
        case IrOpcode::CompositeConstructU64:
        case IrOpcode::IAdd32:
        case IrOpcode::IAdd64:
        case IrOpcode::ISub32:
        case IrOpcode::ISub64:
        case IrOpcode::IMul32:
        case IrOpcode::IMul64:
        case IrOpcode::SMin32:
        case IrOpcode::UMin32:
        case IrOpcode::FPMul32:
        case IrOpcode::FPOrdLessThanEqual32:
        case IrOpcode::FPOrdGreaterThanEqual32:
        case IrOpcode::BitwiseAnd32:
        case IrOpcode::BitwiseAnd64:
        case IrOpcode::BitwiseOr32:
        case IrOpcode::BitwiseXor32:
        case IrOpcode::ShiftLeftLogical32:
        case IrOpcode::ShiftLeftLogical64:
        case IrOpcode::ShiftRightLogical32:
        case IrOpcode::ShiftRightLogical64:
        case IrOpcode::ShiftRightArithmetic32:
        case IrOpcode::ShiftRightArithmetic64:
        case IrOpcode::IEqual32:
        case IrOpcode::INotEqual32:
        case IrOpcode::SLessThan32:
        case IrOpcode::ULessThan32:
        case IrOpcode::UGreaterThan32:
        case IrOpcode::LogicalAnd:
        case IrOpcode::LogicalOr:
        case IrOpcode::LogicalXor: return strict(2);
        case IrOpcode::BitFieldUExtract:
        case IrOpcode::BitFieldSExtract:
        case IrOpcode::SelectU32:
        case IrOpcode::SelectU1:
        case IrOpcode::SelectF32: return strict(3);
        case IrOpcode::BitFieldInsert: return strict(4);
        default: return Evaluability::Maybe;
    }
}

bool neverEvaluable(const IrResourcePlan& plan, IrValue* root, std::unordered_map<const IrValue*, Evaluability>& known) {
    root = root->Resolve();
    std::vector<std::pair<IrValue*, bool>> pending{{root, false}};
    std::array<IrValue*, 4> operands{};
    std::size_t count = 0;
    while (!pending.empty()) {
        IrValue* value = pending.back().first;
        const bool expanded = pending.back().second;
        const auto found = known.find(value);
        if (found != known.end() && found->second != Evaluability::Open) {
            pending.pop_back();
            continue;
        }
        const auto kind = classifyEvaluability(plan, value, operands, count);
        if (kind != Evaluability::Open) {
            known[value] = kind;
            pending.pop_back();
            continue;
        }
        if (!expanded) {
            pending.back().second = true;
            known[value] = Evaluability::Open;
            for (std::size_t index = 0; index < count; ++index) {
                if (!known.contains(operands[index])) pending.emplace_back(operands[index], false);
            }
            continue;
        }
        bool never = false;
        for (std::size_t index = 0; index < count && !never; ++index) {
            const auto operand = known.find(operands[index]);
            never = operand != known.end() && operand->second == Evaluability::Never;
        }
        known[value] = never ? Evaluability::Never : Evaluability::Maybe;
        pending.pop_back();
    }
    return known.at(root) == Evaluability::Never;
}

void dropUnevaluableConditions(IrResourcePlan& plan) {
    std::unordered_map<const IrValue*, Evaluability> known;
    for (auto& block : plan.controlFlow) {
        if (block.condition != nullptr && neverEvaluable(plan, block.condition, known)) block.condition = nullptr;
    }
}

void ownPlanValues(IrResourcePlan& plan) {
    std::vector<IrValue**> roots;
    for (auto& source : plan.descriptorSources) {
        for (auto& dword : source.dwords) roots.push_back(&dword);
    }
    for (auto& read : plan.srtReads) roots.push_back(&read.value);
    for (auto& value : plan.uniformFill.values) roots.push_back(&value);
    const auto closures = roots.size();
    for (auto& block : plan.controlFlow) roots.push_back(&block.condition);

    std::unordered_map<const IrValue*, IrValue*> clones;
    std::unordered_set<const IrValue*> closed;
    std::unordered_set<const IrValue*> opened;
    std::vector<const IrValue*> order;
    std::vector<std::pair<const IrValue*, bool>> pending;
    for (std::size_t index = 0; index < roots.size(); ++index) {
        if (*roots[index] != nullptr) pending.emplace_back(*roots[index], index < closures);
    }
    while (!pending.empty()) {
        const auto [value, closure] = pending.back();
        pending.pop_back();
        if (clones.emplace(value, nullptr).second) order.push_back(value);
        const auto reads = closure ? Detail::EvaluatorReads::Closure : Detail::Evaluator::Reads(plan, *value);
        if (reads == Detail::EvaluatorReads::Nothing || closed.contains(value)) continue;
        if (closure) {
            closed.insert(value);
        } else if (!opened.insert(value).second) {
            continue;
        }
        for (const auto* argument : value->Arguments()) {
            if (argument != nullptr) pending.emplace_back(argument, closure);
        }
        if (!closure && reads == Detail::EvaluatorReads::Closure && value->ArgumentCount() != 0u && value->Argument(0) != nullptr) {
            const auto* handle = value->Argument(0)->Resolve();
            if (opened.insert(handle).second) {
                for (const auto* argument : handle->Arguments()) {
                    if (argument != nullptr) pending.emplace_back(argument, false);
                }
            }
        }
    }
    for (const auto* value : order) {
        auto clone = std::make_unique<IrValue>(value->Opcode(), value->Type(), value->Id());
        clone->SetFlags(value->Flags<std::uint64_t>());
        if (value->HasImmediate()) clone->SetImmediateU64(value->ImmediateU64());
        clone->SetRegister(value->Register());
        clones[value] = clone.get();
        plan.valueStorage.push_back(std::move(clone));
    }
    std::unordered_map<const IrBlock*, IrBlock*> blocks;
    const auto blockFor = [&](const IrBlock* block) {
        auto& clone = blocks[block];
        if (clone == nullptr) {
            plan.blockStorage.push_back(std::make_unique<IrBlock>(block->Id()));
            clone = plan.blockStorage.back().get();
        }
        return clone;
    };
    for (const auto* value : order) {
        if (!closed.contains(value) && !opened.contains(value)) continue;
        std::vector<IrValue*> operands;
        std::vector<IrBlock*> predecessors;
        operands.reserve(value->ArgumentCount());
        if (value->IsPhi()) predecessors.reserve(value->ArgumentCount());
        for (std::size_t index = 0; index < value->ArgumentCount(); index++) {
            const auto* argument = value->Argument(index);
            operands.push_back(argument == nullptr ? nullptr : clones.at(argument));
            if (value->IsPhi()) predecessors.push_back(blockFor(value->PhiBlock(index)));
        }
        clones.at(value)->SetOperandsWithoutUses(std::move(operands), std::move(predecessors));
    }
    for (auto* root : roots) {
        if (*root != nullptr) *root = clones.at(*root);
    }
}

}

IrResourcePlan ResourceMaterializer::ExtractPlan(const IrProgram& program) const {
    const IrResourcePlan& source = program.Resources();
    if (!source.resourceTrackingComplete || !source.srtPlanComplete) {
        throw std::runtime_error("ResourceMaterializer::ExtractPlan requires a completed resource and SRT plan");
    }
    IrResourcePlan plan;
    plan.stage = source.stage;
    plan.shaderHash = source.shaderHash;
    plan.userDataBase = source.userDataBase;
    plan.userDataCount = source.userDataCount;
    plan.srgbDecodeFormats = source.srgbDecodeFormats;
    plan.workgroupThreads = source.workgroupThreads;
    plan.memoryInfo = source.memoryInfo;
    plan.descriptorSources = source.descriptorSources;
    constexpr SrtWalker walker;
    const std::function<bool(IrValue*, std::uint32_t)> walkableWord = [&](IrValue* dword, std::uint32_t joins) {
        if (dword == nullptr || dword->Resolve()->Type() != IrType::U32) return false;
        if (walker.ValidateRuntimeValue(source, dword)) return true;
        const IrValue* value = dword->Resolve();
        if (joins == 0u) return false;
        if (value->Opcode() == IrOpcode::ReadFirstLane) return walkableWord(value->Argument(0), joins - 1u);
        if (value->Opcode() == IrOpcode::SelectU32) return walkableWord(value->Argument(1), joins - 1u) && walkableWord(value->Argument(2), joins - 1u);
        if (!value->IsPhi()) return false;
        for (std::size_t index = 0; index < value->ArgumentCount(); ++index) {
            if (!walkableWord(value->Argument(index), joins - 1u)) return false;
        }
        return true;
    };
    const auto walkable = [&](const DescriptorSource& descriptor) {
        return std::all_of(descriptor.dwords.begin(), descriptor.dwords.begin() + descriptor.dwordCount, [&](IrValue* dword) { return walkableWord(dword, 4u); });
    };
    for (auto& descriptor : plan.descriptorSources) {
        if (descriptor.runtime && !walkable(descriptor)) descriptor.dwords.fill(nullptr);
    }
    plan.controlFlow = source.controlFlow;
    plan.srtReads = source.srtReads;
    plan.cleanFlatSlots = source.cleanFlatSlots;
    plan.srtPlanComplete = source.srtPlanComplete;
    plan.resourceTrackingComplete = source.resourceTrackingComplete;
    plan.info = source.info;
    PrepareImageModes(plan.info);
    plan.uniformFill = source.uniformFill;
    const auto addSource = [&plan](std::uint32_t index) {
        if (index >= plan.descriptorSources.size()) {
            throw std::runtime_error("ResourceMaterializer::ExtractPlan resource references an unknown descriptor source");
        }
        plan.materializationSources.push_back(index);
    };
    for (const auto& image : plan.info.images) {
        if (image.source >= plan.descriptorSources.size()) {
            throw std::runtime_error("ResourceMaterializer::ExtractPlan image references an unknown descriptor source");
        }
        if (image.table == NoTable) {
            throw std::runtime_error("ResourceMaterializer::ExtractPlan an image outside the image tables is not implemented");
        }
    }
    for (const auto& sampler : plan.info.samplers) {
        if (!sampler.deviceTable) addSource(sampler.source);
    }
    for (const auto& store : plan.info.bufferStores) {
        if (!store.fromMemory) addSource(store.source);
    }
    for (const auto& load : plan.info.bufferLoads) {
        if (!load.fromMemory) addSource(load.source);
    }
    dropUnevaluableConditions(plan);
    plan.pureFlatSlots = Detail::ComputePureFlatSlots(plan);
    plan.guardedSrtSlots = Detail::ComputeGuardedFlatSlots(plan);
    ownPlanValues(plan);
    return plan;
}

void ResourceMaterializer::Materialize(const IrResourcePlan& program, const SrtRuntime& runtime, ResourceSnapshot& snapshot) const {
    const IrResourcePlan& plan = program;
    if (!plan.resourceTrackingComplete) {
        throw std::runtime_error("ResourceMaterializer::Materialize requires a completed resource plan");
    }
    SrtWalker walker;
    ResourceSnapshot nextSnapshot;
    std::vector<std::uint8_t> activeSources;
    materializeSnapshot(plan, runtime, walker, nextSnapshot, activeSources);
    const auto started = MaterializeProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    materializeSrtGuards(plan, nextSnapshot);
    if (MaterializeProfiled()) specializationNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    snapshot = std::move(nextSnapshot);
}

bool ResourceMaterializer::ReadsResourcesAtRunTime(const IrResourcePlan& plan) {
    if (!plan.resourceTrackingComplete || !plan.srtReads.empty() || plan.uniformFill.fill.kind != UniformFillKind::None) return false;
    const auto constant = [&](std::uint32_t source) {
        const auto& descriptor = plan.descriptorSources.at(source);
        for (std::uint32_t word = 0; word < descriptor.dwordCount; ++word) {
            if (descriptor.dwords[word] == nullptr || !descriptor.dwords[word]->Resolve()->HasImmediate()) return false;
        }
        return true;
    };
    for (const auto& sampler : plan.info.samplers) {
        if (!sampler.deviceTable && !constant(sampler.source)) return false;
    }
    return true;
}

void ResourceMaterializer::RunTimeSnapshot(const IrResourcePlan& plan, const SrtRuntime& runtime, ResourceSnapshot& snapshot) {
    if (!ReadsResourcesAtRunTime(plan)) throw std::runtime_error("ResourceMaterializer::RunTimeSnapshot requires a plan that reads its resources at run time");
    if (runtime.userData.size() < plan.userDataCount) throw std::runtime_error("runtime user data is smaller than the shader user data count");
    ResourceSnapshot next;
    next.userData.assign(runtime.userData.begin(), runtime.userData.begin() + plan.userDataCount);
    Detail::Evaluator evaluator(plan, runtime);
    const auto evaluate = [&](std::uint32_t source, std::uint32_t words, DescriptorValue& value) {
        const auto& descriptor = plan.descriptorSources.at(source);
        if (descriptor.dwordCount != words) throw std::runtime_error("a constant descriptor has an invalid width");
        value.dwordCount = words;
        for (std::uint32_t word = 0; word < words; ++word) {
            if (!evaluator.Evaluate(descriptor.dwords[word], value.dwords[word])) throw std::runtime_error("a constant descriptor word does not evaluate");
        }
    };
    next.images.resize(plan.info.images.size());
    for (std::uint32_t i = 0; i < plan.info.images.size(); ++i) {
        if (!plan.info.images[i].deviceTable) throw std::runtime_error("ResourceMaterializer::RunTimeSnapshot requires every image in the device image table");
        next.images[i].dwordCount = 8u;
    }
    next.samplers.resize(plan.info.samplers.size());
    for (std::uint32_t i = 0; i < plan.info.samplers.size(); ++i) {
        next.samplers[i].dwordCount = 4u;
        if (!plan.info.samplers[i].deviceTable) evaluate(plan.info.samplers[i].source, 4u, next.samplers[i]);
    }
    evaluateBufferSites(plan, runtime, next.stores, next.loads, false);
    next.deviceImagesComplete = false;
    materializeSrtGuards(plan, next);
    snapshot = std::move(next);
}

std::uint64_t ResourceMaterializer::SpecializationNanoseconds() {
    return specializationNanoseconds.load(std::memory_order_relaxed);
}

}
