#include "SpirvBackend/SpirvMemory/SpirvDescriptors.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "PipelineSpecialization.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler
{
namespace {

constexpr RdnaImageDimensionInfo ImageDimensions[] = {
    {RdnaImageDimension::Dim1D, spv::Dim1D, 1, 1, 0, 0},
    {RdnaImageDimension::Dim2D, spv::Dim2D, 2, 2, 0, 0},
    {RdnaImageDimension::Dim3D, spv::Dim3D, 3, 3, 0, 0},
    //{RdnaImageDimension::DimCube, spv::DimCube, 3, 2, 0, 0},
    {RdnaImageDimension::Dim1DArray, spv::Dim1D, 2, 1, 1, 0},
    {RdnaImageDimension::Dim2DArray, spv::Dim2D, 3, 2, 1, 0},
    {RdnaImageDimension::Dim2DMsaa, spv::Dim2D, 2, 2, 0, 1},
    {RdnaImageDimension::Dim2DMsaaArray, spv::Dim2D, 3, 2, 1, 1},
};

}

[[noreturn]] void ExitDescriptorBindingFailure(const SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource, const char* reason) {
    throw std::runtime_error("shader binding resolution failed during SPIR-V emit: hash=" + std::to_string(state.program.Resources().shaderHash) + " stage=" + std::to_string(static_cast<std::uint32_t>(StageOf(state))) + " resource=" + std::to_string(resource) + " binding_kind=" + std::to_string(static_cast<std::uint32_t>(kind)) + " reason=" + reason);
}

std::uint32_t ResourceForDescriptor(const SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource) {
    for (const auto& descriptor : state.program.Metadata().bindings.descriptors) {
        if (descriptor.kind != kind) {
            continue;
        }
        const auto found = std::find(descriptor.resources.begin(), descriptor.resources.end(), resource);
        if (found == descriptor.resources.end()) {
            ExitDescriptorBindingFailure(state, kind, resource, "resource is absent from descriptor group");
        }
        return static_cast<std::uint32_t>(found - descriptor.resources.begin());
    }
    ExitDescriptorBindingFailure(state, kind, resource, "descriptor group was not allocated");
}

void DecorateTableNonUniform(SpirvEmitterState& state, std::uint32_t id) {
    if (!state.tableIndexNonUniform) return;
    state.module.EmitCapability(spv::CapabilityShaderNonUniform);
    state.module.EmitCapability(spv::CapabilitySampledImageArrayNonUniformIndexing);
    state.module.AddAnnotation(spv::OpDecorate, id, spv::DecorationNonUniform);
}

bool TableModeEmitted(const ImageResource& mode) {
    const auto& info = RdnaImageDimensionInfoFor(mode.dimension);
    return (info.multisampled == 0u || (mode.deviceTable && info.arrayed == 0u)) && mode.packedFormat != IrBufferFormat::Fmask8_S2_F1;
}

bool TablePointSampled(const ImageResource& image) {
    return image.numericClass == IrTextureNumericClass::Uint || image.numericClass == IrTextureNumericClass::Sint || image.conversionFormat != IrBufferFormat::Invalid || image.depthBits;
}

std::uint32_t ImageScalarType(SpirvEmitterState& state, IrTextureNumericClass numericClass) {
    switch (numericClass) {
    case IrTextureNumericClass::Float: return TypeF32(state);
    case IrTextureNumericClass::Uint: return TypeU32(state);
    case IrTextureNumericClass::Sint: return TypeI32(state);
    case IrTextureNumericClass::Unsupported: break;
    }
    FailEmit("invalid image numeric class");
}

std::uint32_t ImageVectorType(SpirvEmitterState& state, IrTextureNumericClass numericClass, std::uint32_t components) {
    return state.module.Type(spv::OpTypeVector, ImageScalarType(state, numericClass), components);
}

std::uint32_t ImageType(SpirvEmitterState& state, const ImageResource& image) {
    std::uint32_t sampled = 0;
    std::uint32_t format = spv::ImageFormatUnknown;
    if (image.resourceClass == ImageResourceClass::Sampled) {
        if (image.atomic) {
            FailEmit("sampled image cannot be atomic");
        }
        sampled = 1;
    } else if (image.resourceClass == ImageResourceClass::Storage) {
        if (image.numericClass == IrTextureNumericClass::Sint || image.numericClass == IrTextureNumericClass::Unsupported) {
            FailEmit("storage image numeric class is not supported");
        }
        sampled = 2;
        if (image.atomic) {
            if (image.numericClass != IrTextureNumericClass::Uint) {
                FailEmit("atomic storage image must be uint");
            }
            format = image.atomic64 ? spv::ImageFormatR64ui : spv::ImageFormatR32ui;
        }
    } else {
        FailEmit("invalid image resource class");
    }
    const auto& info = RdnaImageDimensionInfoFor(image.dimension);
    if (info.multisampled != 0u) {
        if (sampled == 2u) {
            if (std::ranges::find(state.supportedCapabilities, spv::CapabilityStorageImageMultisample) == state.supportedCapabilities.end()) FailEmit("storage image multisampling is unavailable on the target device");
            state.module.EmitCapability(spv::CapabilityStorageImageMultisample);
        }
        if (info.arrayed != 0u) state.module.EmitCapability(spv::CapabilityImageMSArray);
    }
    const auto scalar = image.atomic64 ? TypeScalarU64(state) : ImageScalarType(state, image.numericClass);
    return state.module.Type(spv::OpTypeImage, scalar, info.spirvDimension, image.depthCompare ? 1u : 0u, info.arrayed, info.multisampled, sampled, format);
}

std::uint32_t ImageViewSizeType(SpirvEmitterState& state, RdnaImageDimension dimension) {
    switch (RdnaImageDimensionInfoFor(dimension).coordinateComponents) {
    case 1u: return TypeU32(state);
    case 2u: return TypeU32Vector(state, 2);
    case 3u: return TypeU32Vector(state, 3);
    default: FailEmit("image view size component count is invalid");
    }
}

std::uint32_t LoadSampledImageDescriptor(SpirvEmitterState& state, std::uint32_t resource) {
    const auto& imageResource = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (imageResource.resourceClass != ImageResourceClass::Sampled) {
        FailEmit("sampled image descriptor requested for a non-sampled image");
    }
    const auto kind = DescriptorBindingForImage(imageResource);
    const auto imageType = ImageType(state, imageResource);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageType);
    const auto found = std::find_if(state.tableImageVariables.begin(), state.tableImageVariables.end(), [&](const SpirvTableImageVariable& variable) { return variable.kind == kind; });
    if (imageResource.table == NoTable || found == state.tableImageVariables.end() || !state.tableAccess.active) ExitDescriptorBindingFailure(state, kind, resource, "image table array was not emitted for the mode");
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, pointerType, pointer, found->variable, state.tableAccess.element);
    DecorateTableNonUniform(state, pointer);
    const auto image = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, imageType, image, pointer);
    DecorateTableNonUniform(state, image);
    return image;
}

std::uint32_t LoadSamplerDescriptor(SpirvEmitterState& state, std::uint32_t sampler) {
    const auto& samplerResource = state.program.Info().samplers.at(sampler);
    const auto& image = state.runtimeImage;
    const bool table = image != nullptr && image->table != NoTable;
    const bool point = image != nullptr && (table ? TablePointSampled(*image) : (image->numericClass == IrTextureNumericClass::Sint || image->conversionFormat != IrBufferFormat::Invalid || image->depthBits));
    const auto samplerType = state.module.Type(spv::OpTypeSampler);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, samplerType);
    if (state.heapSamplerVariable == 0u) ExitDescriptorBindingFailure(state, DescriptorBindingKind::Samplers, sampler, "sampler heap was not emitted");
    if (samplerResource.deviceTable) {
        if (!state.tableAccess.active) ExitDescriptorBindingFailure(state, DescriptorBindingKind::Samplers, sampler, "device sampler access was not emitted");
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpAccessChain, pointerType, pointer, state.heapSamplerVariable, state.tableAccess.samplerElement);
        DecorateTableNonUniform(state, pointer);
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, samplerType, loaded, pointer);
        DecorateTableNonUniform(state, loaded);
        return loaded;
    }
    const auto element = ResourceForDescriptor(state, DescriptorBindingKind::Samplers, sampler) + (point ? 1u : 0u);
    const auto slot = EmitShaderDataDwordLoad(state, state.program.Metadata().bindings.SamplerSlotDword() + element);
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, pointerType, pointer, state.heapSamplerVariable, slot);
    const auto samplerId = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, samplerType, samplerId, pointer);
    return samplerId;
}

std::uint32_t MakeSampledImage(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t sampler) {
    const auto& imageResource = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    const auto image = LoadSampledImageDescriptor(state, resource);
    const auto samplerId = LoadSamplerDescriptor(state, sampler);
    const auto sampledImage = state.module.AllocateId();
    const auto sampledType = state.module.Type(spv::OpTypeSampledImage, ImageType(state, imageResource));
    state.module.AddFunction(spv::OpSampledImage, sampledType, sampledImage, image, samplerId);
    if (imageResource.table != NoTable) DecorateTableNonUniform(state, sampledImage);
    return sampledImage;
}

std::uint32_t StorageImageDescriptorPointer(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t tableElement) {
    const auto& image = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (image.resourceClass != ImageResourceClass::Storage) {
        FailEmit("storage image descriptor requested for a non-storage image");
    }
    const auto kind = DescriptorBindingForImage(image);
    const auto found = std::find_if(state.tableImageVariables.begin(), state.tableImageVariables.end(), [&](const SpirvTableImageVariable& variable) { return variable.kind == kind; });
    if (image.table == NoTable || found == state.tableImageVariables.end() || !state.tableAccess.active) ExitDescriptorBindingFailure(state, kind, resource, "storage image table array was not emitted");
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, ImageType(state, image)), pointer, found->variable, tableElement != 0u ? tableElement : state.tableAccess.element);
    if (state.tableIndexNonUniform) {
        state.module.EmitCapability(spv::CapabilityShaderNonUniform);
        state.module.EmitCapability(spv::CapabilityStorageImageArrayNonUniformIndexing);
        state.module.AddAnnotation(spv::OpDecorate, pointer, spv::DecorationNonUniform);
    }
    return pointer;
}

void EmitStorageImageWrite(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t mipLod, std::uint32_t coord, std::uint32_t texel, std::uint32_t sample) {
    const auto& image = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (image.resourceClass != ImageResourceClass::Storage) {
        FailEmit("storage image write requested for a non-storage image");
    }
    if (!image.atomic) {
        state.module.EmitCapability(spv::CapabilityStorageImageWriteWithoutFormat);
    }
    const auto kind = DescriptorBindingForImage(image);
    const auto imageType = ImageType(state, image);
    if (image.mipMode == ImageMipMode::DynamicStorage) {
        if (RdnaImageDimensionInfoFor(image.dimension).multisampled != 0u) FailEmit("multisampled storage image requires a sample and cannot have mip levels");
        const auto u32 = TypeU32(state);
        const auto field = [&](std::uint32_t shift) { return Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpShiftRightLogical, u32, state.tableAccess.dword3, ConstantU32(state, shift)), ConstantU32(state, 0xfu)); };
        const auto base = field(12u);
        const auto last = field(16u);
        const auto inRange = Binary(state, spv::OpLogicalAnd, TypeBool(state), Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), last, base), Binary(state, spv::OpULessThanEqual, TypeBool(state), mipLod, Binary(state, spv::OpISub, u32, last, base)));
        EmitIfCondition(state, inRange, [&] {
            const auto element = Binary(state, spv::OpIAdd, u32, state.tableAccess.element, mipLod);
            const auto descriptor = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, imageType, descriptor, StorageImageDescriptorPointer(state, resource, element));
            state.module.AddFunction(spv::OpImageWrite, descriptor, coord, texel);
        });
        return;
    }
    const auto descriptor = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, imageType, descriptor, StorageImageDescriptorPointer(state, resource, 0u));
    if (RdnaImageDimensionInfoFor(image.dimension).multisampled != 0u) state.module.AddFunction(spv::OpImageWrite, descriptor, coord, texel, spv::ImageOperandsSampleMask, sample);
    else state.module.AddFunction(spv::OpImageWrite, descriptor, coord, texel);
}

const RdnaImageDimensionInfo& RdnaImageDimensionInfoFor(RdnaImageDimension dimension) {
    for (const auto& info : ImageDimensions) {
        if (info.dimension == dimension) {
            return info;
        }
    }
    FailEmit("image dimension " + std::to_string(static_cast<std::uint32_t>(dimension)) + " is invalid");
}

}
