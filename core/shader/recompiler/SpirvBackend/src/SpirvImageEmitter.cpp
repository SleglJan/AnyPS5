#include "SpirvBackend/SpirvImageEmitter.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "PipelineSpecialization.hpp"
#include "SpirvBackend/SpirvAnalysis.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "SpirvBackend/SpirvEmitterInstructions.hpp"
#include "SpirvBackend/SpirvMemory/SpirvDescriptors.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "RdnaDecoder/RdnaImageOpDecoder.hpp"
#include "DeviceImageTableAbi.hpp"
#include "ImageTableAbi.hpp"
#include "StorageImageMemory.hpp"
#include <span>
#include <initializer_list>
#include <algorithm>
#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ShaderRecompiler {
namespace {

struct ImageEmitAccess {
    const IrValue& inst;
    const MemoryInfo& mem;
    const ImageResource& image;
    const IrValue& address;
};

struct SampleSetup {
    const RdnaImageDimensionInfo& dimensionInfo;
    ImageSampleLayout layout;
    IrTextureNumericClass numericClass;
    bool dref;
    std::uint32_t coord;
    std::uint32_t texelCoord = 0;
};

std::uint32_t RuntimeImageSwizzle(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t component) {
    if (state.program.Info().images.at(resource).table == NoTable || !state.tableAccess.active) FailEmit("image table swizzle requested outside a table access");
    return Binary(state, spv::OpBitwiseAnd, TypeU32(state), Binary(state, spv::OpShiftRightLogical, TypeU32(state), state.tableAccess.dword3, ConstantU32(state, component * 3u)), ConstantU32(state, 7u));
}

std::uint32_t RuntimeSwizzledComponent(SpirvEmitterState& state, std::uint32_t selector, const std::array<std::uint32_t, 4>& components) {
    const auto u32 = TypeU32(state);
    const auto constant = Select(state, u32, Binary(state, spv::OpIEqual, TypeBool(state), selector, ConstantU32(state, 1u)), ConstantU32(state, 1u), ConstantU32(state, 0u));
    const auto vector = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), vector, components[0], components[1], components[2], components[3]);
    const auto picked = state.module.AllocateId();
    state.module.AddFunction(spv::OpVectorExtractDynamic, u32, picked, vector, Binary(state, spv::OpBitwiseAnd, u32, selector, ConstantU32(state, 3u)));
    return Select(state, u32, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), selector, ConstantU32(state, 4u)), picked, constant);
}

std::uint32_t EffectiveDmask(const MemoryInfo& mem) {
    return mem.dmask != 0u ? mem.dmask : 1u;
}

std::uint32_t DmaskComponentIndex(std::uint32_t dmask, std::uint32_t component) {
    std::uint32_t index = 0;
    for (std::uint32_t i = 0; i < component; i++) {
        index += (dmask >> i) & 1u;
    }
    return index;
}

std::uint32_t DmaskComponent(std::uint32_t dmask, std::uint32_t index) {
    for (std::uint32_t component = 0; component < 4u; component++) {
        if (((dmask >> component) & 1u) != 0u && index-- == 0u) {
            return component;
        }
    }
    throw std::runtime_error("image dmask has fewer components than requested");
}

std::uint32_t ImageGatherComponent(std::uint32_t dmask) {
    switch (dmask) {
        case 0x1u:
            return 0;
        case 0x2u:
            return 1;
        case 0x4u:
            return 2;
        case 0x8u:
            return 3;
        default:
            throw std::runtime_error("image gather dmask must select exactly one component");
    }
}

bool HasFlag(const MemoryInfo& mem, std::uint32_t flag) {
    return (mem.imageSampleFlags & flag) != 0u;
}

const RdnaImageDimensionInfo& AddressDimension(const ImageEmitAccess& access) {
    return RdnaImageDimensionInfoFor(access.mem.imageDimension);
}

std::uint32_t ZeroF32(SpirvEmitterState& state) {
    return ConstantF32(state, 0);
}

std::uint32_t F32BitsToU32(SpirvValueEmitContext& ctx, std::uint32_t value) {
    return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

std::uint32_t AddressU32(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t component) {
    const auto layout = GetRdnaImageAddressComponentLayout(access.mem.imageSampleFlags, component);
    const auto packed = layout.bitOffset / 32u;
    if (packed >= access.address.ArgumentCount()) {
        ctx.Fail(access.inst, "has an image address component outside of the address operands");
    }
    auto value = ctx.Def(access.address.Argument(packed));
    if (layout.bitWidth == 16u) {
        if ((layout.bitOffset & 31u) != 0u) {
            value = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), value, ConstantU32(ctx.state, 16));
        }
        value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), value, ConstantU32(ctx.state, 0xffffu));
    }
    return value;
}

std::uint32_t AddressF32(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t component) {
    const auto value = AddressU32(ctx, access, component);
    return GetRdnaImageAddressComponentLayout(access.mem.imageSampleFlags, component).bitWidth == 16u ? EmitF16BitsToF32(ctx.state, value) : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), value);
}

ImageSampleLayout Layout(const MemoryInfo& mem) {
    ImageSampleLayout layout;
    std::uint32_t cursor = 0;
    const auto& info = RdnaImageDimensionInfoFor(mem.imageDimension);
    if (HasFlag(mem, RdnaImageSampleFlagOffset)) {
        layout.offset = cursor++;
    }
    if (HasFlag(mem, RdnaImageSampleFlagBias)) {
        layout.bias = cursor++;
    }
    if (HasFlag(mem, RdnaImageSampleFlagCompare)) {
        layout.dref = cursor++;
    }
    if (HasFlag(mem, RdnaImageSampleFlagDerivative)) {
        layout.gradX = cursor;
        cursor += info.spatialComponents;
        layout.gradY = cursor;
        cursor += info.spatialComponents;
    }
    layout.coord = cursor;
    cursor += info.coordinateComponents;
    if (HasFlag(mem, RdnaImageSampleFlagLod)) {
        layout.lod = cursor++;
    }
    if (HasFlag(mem, RdnaImageSampleFlagLodClamp)) {
        layout.clamp = cursor++;
    }
    return layout;
}

std::uint32_t CubeAxis(SpirvEmitterState& state, std::uint32_t value) {
    return Binary(state, spv::OpFSub, TypeF32(state), value, ConstantF32(state, 0x3f800000u));
}

std::uint32_t CubeLayer(SpirvEmitterState& state, std::uint32_t value) {
    const auto guest = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertFToU, TypeU32(state), guest, value);
    const auto padding = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), Binary(state, spv::OpShiftRightLogical, TypeU32(state), guest, ConstantU32(state, 3)), ConstantU32(state, 1));
    const auto host = Binary(state, spv::OpISub, TypeU32(state), guest, padding);
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToF, TypeF32(state), result, host);
    return result;
}

std::uint32_t CoordF32(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t first, std::uint32_t components, std::uint32_t encoded, bool gradient = false) {
    const bool cube = access.image.cube;
    const auto present = std::min(encoded, components);
    const bool cubeFace = cube && access.mem.imageDimension == RdnaImageDimension::Dim2D;
    if (first == NoImageComponent || (encoded < components && ((cube && !cubeFace) || encoded == 0u)) || access.mem.imageAddressComponents < first + present) {
        ctx.Fail(access.inst, "has an image address with too few coordinate components");
    }
    const auto component = [&](std::uint32_t index) { return index < present ? AddressF32(ctx, access, first + index) : ZeroF32(ctx.state); };
    auto x = component(0u);
    if (components == 1u) {
        return x;
    }
    auto y = component(1u);
    if (cube && !gradient) {
        x = CubeAxis(ctx.state, x);
        y = CubeAxis(ctx.state, y);
    }
    const auto result = ctx.state.module.AllocateId();
    if (components == 3u) {
        auto z = component(2u);
        if (cube) {
            z = CubeLayer(ctx.state, z);
        }
        ctx.state.module.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 3), result, x, y, z);
    } else {
        ctx.state.module.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 2), result, x, y);
    }
    return result;
}

std::uint32_t CoordU32(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    const auto components = RdnaImageDimensionInfoFor(access.image.dimension).coordinateComponents;
    const auto encoded = std::min(components, AddressDimension(access).coordinateComponents);
    if (access.mem.imageAddressComponents < encoded) {
        ctx.Fail(access.inst, "has an image address with too few coordinate components");
    }
    const auto component = [&](std::uint32_t index) { return index < encoded ? AddressU32(ctx, access, index) : ConstantU32(ctx.state, 0); };
    const auto x = component(0);
    if (components == 1u) {
        return x;
    }
    const auto y = component(1);
    const auto result = ctx.state.module.AllocateId();
    if (components == 3u) {
        const auto z = component(2);
        ctx.state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 3), result, x, y, z);
    } else {
        ctx.state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 2), result, x, y);
    }
    return result;
}

std::uint32_t LodU32(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    if (!access.mem.imageHasMip) {
        return ConstantU32(ctx.state, 0);
    }
    const auto component = AddressDimension(access).coordinateComponents;
    if (access.mem.imageAddressComponents <= component) {
        ctx.Fail(access.inst, "has an image address without the mip level component");
    }
    return AddressU32(ctx, access, component);
}

bool VolumeSliceAddress(const ImageEmitAccess& access) {
    return access.mem.imageDimension == RdnaImageDimension::Dim2DArray && access.image.dimension == RdnaImageDimension::Dim3D;
}

std::uint32_t VolumeSliceInRange(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t image, std::optional<std::uint32_t> lod) {
    auto& state = ctx.state;
    state.module.EmitCapability(spv::CapabilityImageQuery);
    const auto size = state.module.AllocateId();
    if (lod.has_value()) {
        state.module.AddFunction(spv::OpImageQuerySizeLod, TypeU32Vector(state, 3), size, image, *lod);
    } else {
        state.module.AddFunction(spv::OpImageQuerySize, TypeU32Vector(state, 3), size, image);
    }
    const auto depth = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), depth, size, 2u);
    return Binary(state, spv::OpULessThan, TypeBool(state), AddressU32(ctx, access, 2u), depth);
}

std::uint32_t DrefValueF32(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const ImageSampleLayout& layout) {
    if (layout.dref == NoImageComponent || access.mem.imageAddressComponents <= layout.dref) {
        ctx.Fail(access.inst, "has no depth reference in the image address");
    }
    return AddressF32(ctx, access, layout.dref);
}

std::uint32_t SampledComponentBits(SpirvValueEmitContext& ctx, std::uint32_t value, IrTextureNumericClass numericClass) {
    if (numericClass == IrTextureNumericClass::Uint) {
        return value;
    }
    return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

std::uint32_t SampledComponentZero(SpirvEmitterState& state, IrTextureNumericClass numericClass) {
    switch (numericClass) {
        case IrTextureNumericClass::Float:
            return ZeroF32(state);
        case IrTextureNumericClass::Uint:
            return ConstantU32(state, 0);
        case IrTextureNumericClass::Sint:
            return ConstantI32(state, 0);
        case IrTextureNumericClass::Unsupported:
            break;
    }
    throw std::runtime_error("invalid sampled image numeric class");
}

std::uint32_t ResultVector(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t value, IrTextureNumericClass numericClass, bool dref, bool gather) {
    auto& state = ctx.state;
    const auto& mem = access.mem;
    auto valueClass = numericClass;
    if (dref) {
        valueClass = IrTextureNumericClass::Float;
    }
    const bool integer = valueClass == IrTextureNumericClass::Uint || valueClass == IrTextureNumericClass::Sint;
    if (mem.dataBits == 16u && access.image.depthBits) {
        ctx.Fail(access.inst, "reads the bits of a depth plane as 16-bit results");
    }
    if (mem.dataBits == 16u) {
        if (mem.dataDwords > 4u) {
            ctx.Fail(access.inst, "has more than four packed image result dwords");
        }
        std::uint32_t packed[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0)};
        const auto dmask = EffectiveDmask(mem);
        const auto scalar = [&](std::uint32_t index) -> std::uint32_t {
            if (dref) {
                return value;
            }
            const auto component = gather ? index : DmaskComponent(dmask, index);
            const auto result = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, ImageScalarType(state, valueClass), result, value, component);
            return result;
        };
        for (std::uint32_t word = 0; word < mem.dataDwords; word++) {
            const auto lowIndex = word * 2u;
            const auto highIndex = lowIndex + 1u;
            const auto low = scalar(lowIndex);
            const auto high = highIndex < mem.componentCount ? scalar(highIndex) : SampledComponentZero(state, valueClass);
            if (integer) {
                const auto lowBits = SampledComponentBits(ctx, low, valueClass);
                const auto highBits = SampledComponentBits(ctx, high, valueClass);
                const auto mask = ConstantU32(state, 0xffffu);
                packed[word] = Binary(state, spv::OpBitwiseOr, TypeU32(state), Binary(state, spv::OpBitwiseAnd, TypeU32(state), lowBits, mask), Binary(state, spv::OpShiftLeftLogical, TypeU32(state), Binary(state, spv::OpBitwiseAnd, TypeU32(state), highBits, mask), ConstantU32(state, 16u)));
            } else {
                const auto pair = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, low, high);
                packed[word] = EmitPackHalf2x16(state, pair);
            }
        }
        const auto result = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, packed[0], packed[1], packed[2], packed[3]);
        return result;
    }
    std::uint32_t component[4] = {};
    for (std::uint32_t index = 0; index < 4u; index++) {
        if (dref) {
            component[index] = F32BitsToU32(ctx, value);
            continue;
        }
        const auto scalar = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, ImageScalarType(state, valueClass), scalar, value, index);
        if (access.image.depthUnorm16) {
            const auto scaled = Binary(state, spv::OpFMul, TypeF32(state), scalar, ConstantF32Value(state, 65535.0f));
            component[index] = Unary(state, spv::OpConvertFToU, TypeU32(state), Binary(state, spv::OpFAdd, TypeF32(state), scaled, ConstantF32Value(state, 0.5f)));
        } else {
            component[index] = SampledComponentBits(ctx, scalar, valueClass);
        }
        if (access.image.depthBits && !gather) {
            const auto selector = RuntimeImageSwizzle(state, mem.resource, index);
            const auto one = Binary(state, spv::OpLogicalOr, TypeBool(state), Binary(state, spv::OpIEqual, TypeBool(state), selector, ConstantU32(state, 1u)), Binary(state, spv::OpIEqual, TypeBool(state), selector, ConstantU32(state, 7u)));
            component[index] = Select(state, TypeU32(state), Binary(state, spv::OpIEqual, TypeBool(state), selector, ConstantU32(state, 4u)), component[index], Select(state, TypeU32(state), one, ConstantU32(state, 1u), ConstantU32(state, 0u)));
        }
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, component[0], component[1], component[2], component[3]);
    return result;
}

std::uint32_t DecodeSrgbTexel(SpirvEmitterState& state, std::uint32_t color) {
    const auto u32 = TypeU32(state);
    const auto f32 = TypeF32(state);
    std::uint32_t components[4] = {};
    for (std::uint32_t index = 0; index < 4u; index++) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeF32(state), value, color, index);
        const auto scaled = Binary(state, spv::OpFAdd, TypeF32(state), Binary(state, spv::OpFMul, TypeF32(state), value, ConstantF32Value(state, 255.0f)), ConstantF32Value(state, 0.5f));
        const auto code = state.module.AllocateId();
        state.module.AddFunction(spv::OpExtInst, TypeU32(state), code, GlslStd450(state), GLSLstd450UMin, Unary(state, spv::OpConvertFToU, TypeU32(state), scaled), ConstantU32(state, 255u));
        const auto encoded = Binary(state, spv::OpFDiv, f32, Unary(state, spv::OpConvertUToF, f32, code), ConstantF32Value(state, 255.0f));
        const auto low = Binary(state, spv::OpFDiv, f32, encoded, ConstantF32Value(state, 12.92f));
        const auto base = Binary(state, spv::OpFDiv, f32, Binary(state, spv::OpFAdd, f32, encoded, ConstantF32Value(state, 0.055f)), ConstantF32Value(state, 1.055f));
        const auto high = state.module.AllocateId();
        state.module.AddFunction(spv::OpExtInst, f32, high, GlslStd450(state), GLSLstd450Pow, base, ConstantF32Value(state, 2.4f));
        const auto linear = Select(state, f32, Binary(state, spv::OpULessThanEqual, TypeBool(state), code, ConstantU32(state, 10u)), low, high);
        const auto rounded = Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpIAdd, u32, Unary(state, spv::OpBitcast, u32, linear), ConstantU32(state, 0x8000u)), ConstantU32(state, 0xffff0000u));
        components[index] = Unary(state, spv::OpBitcast, f32, rounded);
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), result, components[0], components[1], components[2], components[3]);
    return result;
}

std::uint32_t QueryDimensions(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const auto dimension = access.image.dimension;
    const auto& info = RdnaImageDimensionInfoFor(dimension);
    const auto image = LoadSampledImageDescriptor(state, access.mem.resource);
    const auto size = state.module.AllocateId();
    if (info.multisampled != 0u) {
        state.module.AddFunction(spv::OpImageQuerySize, ImageViewSizeType(state, dimension), size, image);
    } else {
        state.module.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, dimension), size, image, AddressU32(ctx, access, 0));
    }
    const auto components = info.coordinateComponents;
    std::uint32_t result[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0)};
    if (components == 1u) {
        result[0] = size;
    } else {
        for (std::uint32_t index = 0; index < components; index++) {
            result[index] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), result[index], size, index);
        }
    }
    if (access.mem.imageDimension == RdnaImageDimension::Dim3D && dimension == RdnaImageDimension::Dim2D) {
        result[2] = ConstantU32(state, 1);
    }
    if (info.multisampled == 0u) {
        result[3] = state.module.AllocateId();
        state.module.AddFunction(spv::OpImageQueryLevels, TypeU32(state), result[3], image);
    }
    const auto vector = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), vector, result[0], result[1], result[2], result[3]);
    return vector;
}

std::uint32_t PackedOffset(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const ImageSampleLayout& layout) {
    auto& state = ctx.state;
    if (layout.offset == NoImageComponent || access.mem.imageAddressComponents <= layout.offset) {
        ctx.Fail(access.inst, "has no texel offset in the image address");
    }
    const auto components = RdnaImageDimensionInfoFor(access.image.dimension).spatialComponents;
    const auto packed = Unary(state, spv::OpBitcast, TypeI32(state), AddressU32(ctx, access, layout.offset));
    std::uint32_t values[3] = {};
    for (std::uint32_t index = 0; index < components; index++) {
        values[index] = state.module.AllocateId();
        state.module.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), values[index], packed, ConstantU32(state, index * 8u), ConstantU32(state, 6));
    }
    if (components == 1u) {
        return values[0];
    }
    const auto result = state.module.AllocateId();
    if (components == 3u) {
        state.module.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(state, 3), result, values[0], values[1], values[2]);
    } else {
        state.module.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(state, 2), result, values[0], values[1]);
    }
    return result;
}

std::uint32_t FoldedOffsetCoord(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const SampleSetup& setup) {
    auto& state = ctx.state;
    const auto components = setup.dimensionInfo.spatialComponents;
    const auto type = components == 1u ? TypeF32(state) : TypeF32Vector(state, components);
    const auto sum = Binary(state, spv::OpFAdd, type, setup.coord, Unary(state, spv::OpConvertSToF, type, PackedOffset(ctx, access, setup.layout)));
    state.module.AddAnnotation(spv::OpDecorate, sum, spv::DecorationNoContraction);
    return sum;
}

std::uint32_t HorizontalOffsets(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const auto components = RdnaImageDimensionInfoFor(access.image.dimension).spatialComponents;
    if (components != 1u && components != 2u) {
        ctx.Fail(access.inst, "has horizontal gather offsets for an unsupported image dimension");
    }
    const auto count = ConstantU32(state, 4);
    const auto elementType = components == 1u ? TypeI32(state) : TypeI32Vector(state, 2);
    const auto arrayType = state.module.Type(spv::OpTypeArray, elementType, count);
    std::uint32_t offsets[4] = {};
    for (std::uint32_t index = 0; index < 4u; index++) {
        const auto x = ConstantI32(state, static_cast<std::int32_t>(index) - 1);
        if (components == 1u) {
            offsets[index] = x;
        } else {
            offsets[index] = state.module.Constant(spv::OpConstantComposite, TypeI32Vector(state, 2), x, ConstantI32(state, 0));
        }
    }
    return state.module.Constant(spv::OpConstantComposite, arrayType, offsets[0], offsets[1], offsets[2], offsets[3]);
}

SpirvBufferFormatInfo ImageConversionFormat(const ImageResource& image) {
    const auto format = image.conversionFormat;
    if (format == IrBufferFormat::Invalid) {
        return {};
    }
    const auto info = GetFormatInfo(format);
    if (SampledTextureNumericClass(format) != IrTextureNumericClass::Uint || RemapTextureFormat(format) == format || (info.type != SpirvFormatComponentType::Uint && info.type != SpirvFormatComponentType::Unorm && info.type != SpirvFormatComponentType::Float) || !info.packedBitfield || info.byteSize != sizeof(std::uint32_t) || info.componentCount == 0u || info.componentCount > 4u) {
        throw std::runtime_error("image conversion format is not a packed 32-bit unsigned integer, unorm or float format");
    }
    return info;
}

void RequireConvertedUnormAccess(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const SpirvBufferFormatInfo& info) {
    if (access.mem.dataBits == 16u) {
        ctx.Fail(access.inst, info.type == SpirvFormatComponentType::Float ? "reads or writes a converted float image with 16-bit data, which is not implemented" : "reads or writes a converted unorm image with 16-bit data, which is not implemented");
    }
    for (std::uint32_t component = 0; component < 4u; component++) {
        const auto selector = (access.image.shaderSwizzle >> (component * 3u)) & 7u;
        if (selector >= 4u && selector - 4u >= info.componentCount) {
            ctx.Fail(access.inst, "selects a channel the converted image format does not have");
        }
    }
}

std::uint32_t UnpackImageTexel(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t texel) {
    auto& state = ctx.state;
    if (access.image.srgbDecode) {
        return DecodeSrgbTexel(state, texel);
    }
    const auto info = ImageConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) {
        return texel;
    }
    const bool unorm = info.type == SpirvFormatComponentType::Unorm || info.type == SpirvFormatComponentType::Float;
    if (unorm) {
        RequireConvertedUnormAccess(ctx, access, info);
    }
    const auto packed = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), packed, texel, 0u);
    std::array<std::uint32_t, 4> components{ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0)};
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        components[component] = state.module.AllocateId();
        state.module.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), components[component], packed, ConstantU32(state, info.componentBitOffset[component]), ConstantU32(state, info.componentBits[component]));
        if (unorm) {
            components[component] = NormalizeFormatComponent(state, info, component, components[component]);
        }
    }
    for (std::uint32_t component = info.componentCount; component < 4u; component++) {
        components[component] = components[component % info.componentCount];
    }
    std::uint32_t selected[4] = {};
    for (std::uint32_t component = 0; component < 4u; component++) {
        const auto selector = RuntimeImageSwizzle(state, access.mem.resource, component);
        const auto value = RuntimeSwizzledComponent(state, selector, components);
        selected[component] = Select(state, TypeU32(state), Binary(state, spv::OpIEqual, TypeBool(state), selector, ConstantU32(state, 1u)), ConstantU32(state, FormattedConstantBits(info, SpirvFormattedSourceKind::One)), value);
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, selected[0], selected[1], selected[2], selected[3]);
    return result;
}

SpirvBufferFormatInfo PackedTexelFormat(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    const auto format = access.image.packedFormat;
    if (format == IrBufferFormat::Invalid || (access.image.table != NoTable && !access.image.deviceTable) || access.image.depthBits || access.image.conversionFormat != IrBufferFormat::Invalid) {
        ctx.Fail(access.inst, "accesses packed texels of an image without a packed format");
    }
    const auto info = GetFormatInfo(format);
    if (info.packedBitfield) {
        ctx.Fail(access.inst, "accesses packed texels of a bitfield format");
    }
    return info;
}

std::uint32_t PackedImageTexel(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t texel) {
    auto& state = ctx.state;
    const auto info = PackedTexelFormat(ctx, access);
    std::uint32_t texelBits = 0;
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        const auto bits = info.componentBits[component];
        const bool exact = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint || (info.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (info.type == SpirvFormatComponentType::Float && bits == 32u);
        if (!exact) {
            ctx.Fail(access.inst, "reads packed texels of a format whose bits are not recoverable from the view");
        }
        texelBits += bits;
    }
    const auto numericClass = access.image.numericClass;
    std::uint32_t words[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0)};
    std::uint32_t position = 0;
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        const auto bits = info.componentBits[component];
        const auto scalar = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, ImageScalarType(state, numericClass), scalar, texel, component);
        std::uint32_t value;
        if (info.type == SpirvFormatComponentType::Unorm) {
            const auto scaled = Binary(state, spv::OpFMul, TypeF32(state), scalar, ConstantF32Value(state, static_cast<float>((1u << bits) - 1u)));
            value = Unary(state, spv::OpConvertFToU, TypeU32(state), Binary(state, spv::OpFAdd, TypeF32(state), scaled, ConstantF32Value(state, 0.5f)));
        } else {
            value = SampledComponentBits(ctx, scalar, numericClass);
        }
        if (bits < 32u) {
            value = Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, ConstantU32(state, (1u << bits) - 1u));
        }
        const auto word = position / 32u;
        if (position % 32u != 0u) {
            value = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), value, ConstantU32(state, position % 32u));
        }
        words[word] = Binary(state, spv::OpBitwiseOr, TypeU32(state), words[word], value);
        position += bits;
    }
    if (access.mem.dataSigned && texelBits < 32u) {
        const auto shift = ConstantU32(state, 32u - texelBits);
        words[0] = Binary(state, spv::OpShiftRightArithmetic, TypeU32(state), Binary(state, spv::OpShiftLeftLogical, TypeU32(state), words[0], shift), shift);
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, words[0], words[1], words[2], words[3]);
    return result;
}

std::uint32_t UnpackImageGather(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t gathered) {
    auto& state = ctx.state;
    const auto info = ImageConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) {
        return gathered;
    }
    const auto component = ImageGatherComponent(EffectiveDmask(access.mem));
    const auto selector = RuntimeImageSwizzle(state, access.mem.resource, component);
    std::uint32_t values[4] = {};
    for (std::uint32_t lane = 0; lane < 4u; lane++) {
        const auto packed = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), packed, gathered, lane);
        std::array<std::uint32_t, 4> components{};
        for (std::uint32_t index = 0u; index < components.size(); ++index) {
            const auto physical = index % info.componentCount;
            components[index] = state.module.AllocateId();
            state.module.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), components[index], packed, ConstantU32(state, info.componentBitOffset[physical]), ConstantU32(state, info.componentBits[physical]));
        }
        values[lane] = RuntimeSwizzledComponent(state, selector, components);
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, values[0], values[1], values[2], values[3]);
    return result;
}

std::uint32_t EmitOneDimensionalGatherLz(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const SampleSetup& setup) {
    auto& state = ctx.state;
    const auto numericClass = access.image.numericClass;
    const bool arrayed = access.image.dimension == RdnaImageDimension::Dim1DArray;
    state.module.EmitCapability(spv::CapabilityImageQuery);
    const auto image = LoadSampledImageDescriptor(state, access.mem.resource);
    auto width = state.module.AllocateId();
    state.module.AddFunction(spv::OpImageQuerySizeLod, arrayed ? TypeU32Vector(state, 2) : TypeU32(state), width, image, ConstantU32(state, 0));
    auto coord = setup.coord;
    std::uint32_t layer = 0;
    if (arrayed) {
        const auto size = width;
        width = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), width, size, 0u);
        coord = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeF32(state), coord, setup.coord, 0u);
        layer = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeF32(state), layer, setup.coord, 1u);
    }
    const auto widthF32 = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToF, TypeF32(state), widthF32, width);
    auto left = state.module.AllocateId();
    state.module.AddFunction(spv::OpExtInst, TypeF32(state), left, GlslStd450(state), GLSLstd450Floor, Binary(state, spv::OpFSub, TypeF32(state), Binary(state, spv::OpFMul, TypeF32(state), coord, widthF32), ConstantF32(state, 0x3f000000u)));
    if (setup.layout.offset != NoImageComponent) {
        left = Binary(state, spv::OpFAdd, TypeF32(state), left, Unary(state, spv::OpConvertSToF, TypeF32(state), PackedOffset(ctx, access, setup.layout)));
    }
    const auto sampled = MakeSampledImage(state, access.mem.resource, access.mem.sampler);
    const auto vectorType = ImageVectorType(state, numericClass, 4);
    const auto scalarType = ImageScalarType(state, numericClass);
    const auto component = ImageConversionFormat(access.image).format == IrBufferFormat::Invalid ? ImageGatherComponent(EffectiveDmask(access.mem)) : 0u;
    std::uint32_t values[2] = {};
    for (std::uint32_t index = 0; index < 2u; index++) {
        auto sampleCoord = Binary(state, spv::OpFDiv, TypeF32(state), Binary(state, spv::OpFAdd, TypeF32(state), left, ConstantF32(state, index == 0u ? 0x3f000000u : 0x3fc00000u)), widthF32);
        if (arrayed) {
            const auto pair = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, sampleCoord, layer);
            sampleCoord = pair;
        }
        const auto texel = state.module.AllocateId();
        state.module.AddFunction(spv::OpImageSampleExplicitLod, vectorType, texel, sampled, sampleCoord, spv::ImageOperandsLodMask, ZeroF32(state));
        values[index] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, scalarType, values[index], texel, component);
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, vectorType, result, values[0], values[1], values[1], values[0]);
    return result;
}

std::uint32_t PackImageTexel(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t texel) {
    auto& state = ctx.state;
    const auto info = ImageConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) {
        return texel;
    }
    const bool unorm = info.type == SpirvFormatComponentType::Unorm || info.type == SpirvFormatComponentType::Float;
    if (unorm) {
        RequireConvertedUnormAccess(ctx, access, info);
    }
    auto packed = ConstantU32(state, 0u);
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, texel, component);
        auto clamped = value;
        if (unorm) {
            clamped = EmitFormatStoreComponent(state, info, component, value);
        } else {
            const auto maximum = ConstantU32(state, info.componentBits[component] == 32u ? UINT32_MAX : (1u << info.componentBits[component]) - 1u);
            const auto within = Binary(state, spv::OpULessThan, TypeBool(state), value, maximum);
            clamped = Select(state, TypeU32(state), within, value, maximum);
        }
        const auto shifted = info.componentBitOffset[component] == 0u ? clamped : Binary(state, spv::OpShiftLeftLogical, TypeU32(state), clamped, ConstantU32(state, info.componentBitOffset[component]));
        packed = Binary(state, spv::OpBitwiseOr, TypeU32(state), packed, shifted);
    }
    const auto zero = ConstantU32(state, 0u);
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, packed, zero, zero, zero);
    return result;
}

SpirvBufferFormatInfo SintStorageFormat(const ImageResource& image) {
    const auto format = image.conversionFormat;
    if (image.resourceClass != ImageResourceClass::Storage || format == IrBufferFormat::Invalid || SampledTextureNumericClass(format) != IrTextureNumericClass::Sint) {
        return {};
    }
    const auto info = GetFormatInfo(format);
    if (image.numericClass != IrTextureNumericClass::Uint || info.type != SpirvFormatComponentType::Sint || info.packedBitfield || info.componentCount == 0u || info.byteSize == 12u) {
        throw std::runtime_error("storage image conversion format is not a SINT format written through a UINT view");
    }
    return info;
}

std::uint32_t StoreTexel(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t data, bool integer) {
    auto& state = ctx.state;
    const auto& mem = access.mem;
    const auto sint = SintStorageFormat(access.image);
    std::uint32_t values[4] = {};
    const auto dmask = EffectiveDmask(mem);
    for (std::uint32_t component = 0; component < 4u; component++) {
        auto raw = ConstantU32(state, 0);
        for (std::uint32_t reverse = 4u; reverse != 0u; --reverse) {
            const auto source = reverse - 1u;
            const auto packedIndex = DmaskComponentIndex(dmask, source);
            auto candidate = ConstantU32(state, 0u);
            if (((dmask >> source) & 1u) != 0u) {
                candidate = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), candidate, data, mem.dataBits == 16u ? packedIndex / 2u : packedIndex);
            }
            if (mem.dataBits == 16u && ((dmask >> source) & 1u) != 0u) {
                if ((packedIndex & 1u) != 0u) {
                    candidate = Binary(state, spv::OpShiftRightLogical, TypeU32(state), candidate, ConstantU32(state, 16u));
                }
                if (sint.format != IrBufferFormat::Invalid) {
                    const auto extended = state.module.AllocateId();
                    state.module.AddFunction(spv::OpBitFieldSExtract, TypeU32(state), extended, candidate, ConstantU32(state, 0u), ConstantU32(state, 16u));
                    candidate = extended;
                } else {
                    candidate = Binary(state, spv::OpBitwiseAnd, TypeU32(state), candidate, ConstantU32(state, 0xffffu));
                }
            }
            raw = Select(state, TypeU32(state), Binary(state, spv::OpIEqual, TypeBool(state), RuntimeImageSwizzle(state, mem.resource, source), ConstantU32(state, component + 4u)), candidate, raw);
        }
        values[component] = integer ? raw : mem.dataBits == 16u ? EmitF16BitsToF32(state, raw) : Unary(state, spv::OpBitcast, TypeF32(state), raw);
        if (component < sint.componentCount) {
            values[component] = EmitFormatStoreComponent(state, sint, component, values[component]);
        }
    }
    const auto texel = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, integer ? TypeU32Vector(state, 4) : TypeF32Vector(state, 4), texel, values[0], values[1], values[2], values[3]);
    return sint.format != IrBufferFormat::Invalid ? texel : PackImageTexel(ctx, access, texel);
}

std::uint32_t PackedStoreTexel(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t data) {
    auto& state = ctx.state;
    const auto info = PackedTexelFormat(ctx, access);
    const auto numericClass = access.image.numericClass;
    if (info.byteSize == 12u) {
        ctx.Fail(access.inst, "stores packed texels of a format the hardware does not write");
    }
    std::uint32_t words[4] = {};
    for (std::uint32_t word = 0; word < 4u; word++) {
        if (((access.mem.dmask >> word) & 1u) == 0u) {
            words[word] = ConstantU32(state, 0u);
            continue;
        }
        words[word] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), words[word], data, DmaskComponentIndex(access.mem.dmask, word));
    }
    std::uint32_t values[4] = {};
    for (std::uint32_t component = 0; component < 4u; component++) {
        if (component >= info.componentCount) {
            values[component] = numericClass == IrTextureNumericClass::Float ? ZeroF32(state) : ConstantU32(state, 0u);
            continue;
        }
        const auto bits = info.componentBits[component];
        const auto offset = info.componentBitOffset[component];
        const bool exact = (info.type == SpirvFormatComponentType::Uint && numericClass == IrTextureNumericClass::Uint) || (bits == 32u && ((info.type == SpirvFormatComponentType::Sint && numericClass == IrTextureNumericClass::Uint) || (info.type == SpirvFormatComponentType::Float && numericClass == IrTextureNumericClass::Float)));
        if (!exact || offset % 32u + bits > 32u) {
            ctx.Fail(access.inst, "stores packed texels of a format whose bits are not reproducible through the view");
        }
        auto value = words[offset / 32u];
        if (offset % 32u != 0u) {
            value = Binary(state, spv::OpShiftRightLogical, TypeU32(state), value, ConstantU32(state, offset % 32u));
        }
        if (bits < 32u) {
            value = Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, ConstantU32(state, (1u << bits) - 1u));
        }
        values[component] = numericClass == IrTextureNumericClass::Float ? Unary(state, spv::OpBitcast, TypeF32(state), value) : value;
    }
    const auto texel = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, numericClass == IrTextureNumericClass::Float ? TypeF32Vector(state, 4) : TypeU32Vector(state, 4), texel, values[0], values[1], values[2], values[3]);
    return texel;
}

std::uint32_t ImageAtomicOpcode(IrOpcode opcode) {
    switch (opcode) {
        case IrOpcode::ImageAtomicSwap32:
        case IrOpcode::ImageAtomicSwap64:
            return spv::OpAtomicExchange;
        case IrOpcode::ImageAtomicIAdd32:
        case IrOpcode::ImageAtomicIAdd64:
            return spv::OpAtomicIAdd;
        case IrOpcode::ImageAtomicUMin32:
        case IrOpcode::ImageAtomicUMin64:
            return spv::OpAtomicUMin;
        case IrOpcode::ImageAtomicUMax32:
        case IrOpcode::ImageAtomicUMax64:
            return spv::OpAtomicUMax;
        case IrOpcode::ImageAtomicAnd32:
        case IrOpcode::ImageAtomicAnd64:
            return spv::OpAtomicAnd;
        case IrOpcode::ImageAtomicOr32:
        case IrOpcode::ImageAtomicOr64:
            return spv::OpAtomicOr;
        case IrOpcode::ImageAtomicXor32:
        case IrOpcode::ImageAtomicXor64:
            return spv::OpAtomicXor;
        case IrOpcode::ImageAtomicISub32:
        case IrOpcode::ImageAtomicISub64:
            return spv::OpAtomicISub;
        case IrOpcode::ImageAtomicSMin32:
        case IrOpcode::ImageAtomicSMin64:
            return spv::OpAtomicSMin;
        case IrOpcode::ImageAtomicSMax32:
        case IrOpcode::ImageAtomicSMax64:
            return spv::OpAtomicSMax;
        default:
            throw std::runtime_error("opcode is not an image atomic");
    }
}

const ImageResource& ImageResourceOf(const SpirvEmitterState& state, const MemoryInfo& mem) {
    return state.program.Resources().info.images.at(mem.resource);
}

void EmitQueryDimensionsOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    ctx.state.module.EmitCapability(spv::CapabilityImageQuery);
    ctx.Define(access.inst, QueryDimensions(ctx, access));
}

void EmitQueryLodOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    state.module.EmitCapability(spv::CapabilityImageQuery);
    const auto sampled = MakeSampledImage(state, access.mem.resource, access.mem.sampler);
    const auto coord = CoordF32(ctx, access, 0, RdnaImageDimensionInfoFor(access.image.dimension).spatialComponents, AddressDimension(access).spatialComponents);
    const auto lod = state.module.AllocateId();
    state.module.AddFunction(spv::OpImageQueryLod, TypeF32Vector(state, 2), lod, sampled, coord);
    std::uint32_t values[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0)};
    for (std::uint32_t index = 0; index < 2u; index++) {
        const auto component = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, lod, index);
        values[index] = F32BitsToU32(ctx, component);
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, values[0], values[1], values[2], values[3]);
    ctx.Define(access.inst, result);
}

const StorageImageMemoryFunctions* MemoryFunctions(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    if (!StorageImageMemory::ResourceThroughMemory(ctx.state.program.Info(), ImageResourceOf(ctx.state, access.mem))) return nullptr;
    const auto* functions = StorageImageMemoryFunctionsOf(ctx.state);
    if (functions == nullptr || functions->texel == 0u) ctx.Fail(access.inst, "accesses an image through memory without the texel helper");
    return functions;
}

std::uint32_t MemorySlotSelected(SpirvEmitterState& state, std::uint32_t resource) {
    const auto& layout = state.program.Metadata().bindings;
    const auto slot = EmitShaderDataDwordLoad(state, layout.ImageSlotDword() + layout.imageSlotBases.at(resource));
    return Binary(state, spv::OpIEqual, TypeBool(state), slot, ConstantU32(state, StorageImageMemory::MemorySlot));
}

struct MemoryTexel {
    std::uint32_t address;
    std::uint32_t element;
    std::uint32_t inside;
    std::array<std::uint32_t, 6> words;
};

std::uint32_t MemoryField(SpirvEmitterState& state, std::uint32_t word, std::uint32_t offset, std::uint32_t mask) {
    return Binary(state, spv::OpBitwiseAnd, TypeU32(state), Binary(state, spv::OpShiftRightLogical, TypeU32(state), word, ConstantU32(state, offset)), ConstantU32(state, mask));
}

std::uint32_t MemoryExtract(SpirvEmitterState& state, std::uint32_t composite, std::uint32_t index) {
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, composite, index);
    return value;
}

std::uint32_t MemoryVector(SpirvEmitterState& state, const std::array<std::uint32_t, 4>& words) {
    const auto vector = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), vector, words[0], words[1], words[2], words[3]);
    return vector;
}

MemoryTexel EmitMemoryTexel(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const StorageImageMemoryFunctions& functions) {
    auto& state = ctx.state;
    const auto* handle = access.inst.Argument(0)->Resolve();
    if (handle->ArgumentCount() != 8u) ctx.Fail(access.inst, "accesses an image through memory without its eight descriptor words");
    std::array<std::uint32_t, 6> words{};
    for (std::uint32_t word = 0; word < words.size(); ++word) words[word] = ctx.Def(handle->Argument(word));
    const auto& info = AddressDimension(access);
    if (access.mem.imageAddressComponents < info.coordinateComponents) ctx.Fail(access.inst, "has an image address with too few coordinate components");
    const auto zero = ConstantU32(state, 0u);
    const auto x = AddressU32(ctx, access, 0u);
    const auto y = info.spatialComponents >= 2u ? AddressU32(ctx, access, 1u) : zero;
    const auto z = info.coordinateComponents >= 3u || info.arrayed != 0u ? AddressU32(ctx, access, info.coordinateComponents - 1u) : zero;
    const auto lod = LodU32(ctx, access);
    const auto texel = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 4u), texel, functions.texel, words[0], words[1], words[2], words[3], words[4], words[5], x, y, z, lod);
    const auto u64 = TypeScalarU64(state);
    const auto low = Unary(state, spv::OpUConvert, u64, MemoryExtract(state, texel, 0u));
    const auto high = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, MemoryExtract(state, texel, 1u)), BdaConstant(state, 32u));
    const auto inside = Binary(state, spv::OpINotEqual, TypeBool(state), MemoryExtract(state, texel, 3u), zero);
    return {Binary(state, spv::OpBitwiseOr, u64, low, high), MemoryExtract(state, texel, 2u), inside, words};
}

std::uint32_t EmitMemoryElementRead(SpirvValueEmitContext& ctx, const IrValue& inst, const MemoryTexel& texel) {
    auto& state = ctx.state;
    const auto u32 = TypeU32(state);
    const auto vector = TypeU32Vector(state, 4u);
    const auto zero = ConstantU32(state, 0u);
    const auto element = [&](std::uint32_t log2) { return Binary(state, spv::OpIEqual, TypeBool(state), texel.element, ConstantU32(state, log2)); };
    return EmitValueIfElse(state, element(4u), vector, [&] {
        const auto words = EmitBdaDwordReads(ctx, inst, texel.address, 0u, 4u, true);
        return MemoryVector(state, words);
    }, [&] {
        return EmitValueIfElse(state, element(3u), vector, [&] {
            const auto words = EmitBdaDwordReads(ctx, inst, texel.address, 0u, 2u, true);
            return MemoryVector(state, {words[0], words[1], zero, zero});
        }, [&] {
            const auto word = EmitValueIfElse(state, element(2u), u32, [&] { return EmitBdaDwordReads(ctx, inst, texel.address, 0u, 1u, true)[0]; }, [&] {
                return EmitValueIfElse(state, element(1u), u32, [&] { return EmitBdaRead(ctx, inst, texel.address, 16u); }, [&] { return EmitBdaRead(ctx, inst, texel.address, 8u); });
            });
            return MemoryVector(state, {word, zero, zero, zero});
        });
    });
}

std::uint32_t EmitMemoryRead(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const StorageImageMemoryFunctions& functions, std::uint32_t condition) {
    auto& state = ctx.state;
    if (functions.decode == 0u) ctx.Fail(access.inst, "loads an image through memory without the texel decoder");
    const auto u32 = TypeU32(state);
    const auto vector = TypeU32Vector(state, 4u);
    const auto zero = ConstantU32CompositeZero(state, 4u);
    return EmitValueOrDefaultIfCondition(state, condition, vector, zero, [&] {
        const auto texel = EmitMemoryTexel(ctx, access, functions);
        return EmitValueOrDefaultIfCondition(state, texel.inside, vector, zero, [&] {
            const auto raw = EmitMemoryElementRead(ctx, access.inst, texel);
            const auto decoded = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, functions.decodeType, decoded, functions.decode, MemoryField(state, texel.words[1], 20u, 0x1ffu), MemoryExtract(state, raw, 0u), MemoryExtract(state, raw, 1u), MemoryExtract(state, raw, 2u), MemoryExtract(state, raw, 3u));
            const std::array<std::uint32_t, 4> components{MemoryExtract(state, decoded, 0u), MemoryExtract(state, decoded, 1u), MemoryExtract(state, decoded, 2u), MemoryExtract(state, decoded, 3u)};
            const auto one = MemoryExtract(state, decoded, 4u);
            const auto channels = MemoryVector(state, components);
            const auto numericClass = IrTextureNumericClass::Uint;
            std::array<std::uint32_t, 4> color{};
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const auto selector = MemoryField(state, texel.words[3], component * 3u, 7u);
                const auto picked = state.module.AllocateId();
                state.module.AddFunction(spv::OpVectorExtractDynamic, u32, picked, channels, Binary(state, spv::OpBitwiseAnd, u32, selector, ConstantU32(state, 3u)));
                const auto constant = Select(state, u32, Binary(state, spv::OpIEqual, TypeBool(state), selector, ConstantU32(state, 1u)), one, ConstantU32(state, 0u));
                const auto value = Select(state, u32, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), selector, ConstantU32(state, 4u)), picked, constant);
                color[component] = value;
            }
            const auto texelColor = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, ImageVectorType(state, numericClass, 4u), texelColor, color[0], color[1], color[2], color[3]);
            auto plain = access.image;
            plain.numericClass = numericClass;
            plain.depthBits = false;
            plain.depthUnorm16 = false;
            const ImageEmitAccess plainAccess{access.inst, access.mem, plain, access.address};
            return ResultVector(ctx, plainAccess, texelColor, numericClass, false, false);
        });
    });
}

std::array<std::uint32_t, 4> MemoryStoreComponents(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t data, std::uint32_t word3) {
    auto& state = ctx.state;
    const auto& mem = access.mem;
    const auto dmask = EffectiveDmask(mem);
    std::array<std::uint32_t, 4> values{};
    for (std::uint32_t component = 0; component < 4u; component++) {
        auto raw = ConstantU32(state, 0u);
        for (std::uint32_t reverse = 4u; reverse != 0u; --reverse) {
            const auto source = reverse - 1u;
            const auto packedIndex = DmaskComponentIndex(dmask, source);
            auto candidate = ConstantU32(state, 0u);
            if (((dmask >> source) & 1u) != 0u) {
                candidate = MemoryExtract(state, data, mem.dataBits == 16u ? packedIndex / 2u : packedIndex);
                if (mem.dataBits == 16u) {
                    if ((packedIndex & 1u) != 0u) candidate = Binary(state, spv::OpShiftRightLogical, TypeU32(state), candidate, ConstantU32(state, 16u));
                    candidate = Binary(state, spv::OpBitwiseAnd, TypeU32(state), candidate, ConstantU32(state, 0xffffu));
                }
            }
            raw = Select(state, TypeU32(state), Binary(state, spv::OpIEqual, TypeBool(state), MemoryField(state, word3, source * 3u, 7u), ConstantU32(state, component + 4u)), candidate, raw);
        }
        values[component] = raw;
    }
    return values;
}

void EmitMemoryWrite(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const StorageImageMemoryFunctions& functions) {
    auto& state = ctx.state;
    if (functions.encode == 0u) ctx.Fail(access.inst, "stores to an image through memory without the texel encoder");
    EmitIfCondition(state, ctx.Arg(access.inst, 3), [&] {
        const auto texel = EmitMemoryTexel(ctx, access, functions);
        EmitIfCondition(state, texel.inside, [&] {
            const auto values = MemoryStoreComponents(ctx, access, ctx.Arg(access.inst, 2), texel.words[3]);
            const auto encoded = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 4u), encoded, functions.encode, MemoryField(state, texel.words[1], 20u, 0x1ffu), ConstantU32(state, access.mem.dataBits == 16u ? 1u : 0u), values[0], values[1], values[2], values[3]);
            const auto element = [&](std::uint32_t log2) { return Binary(state, spv::OpIEqual, TypeBool(state), texel.element, ConstantU32(state, log2)); };
            const auto enabled = ConstantBool(state, true);
            EmitIfCondition(state, element(0u), [&] { EmitBdaWrite(ctx, access.inst, texel.address, MemoryExtract(state, encoded, 0u), 8u, enabled); });
            EmitIfCondition(state, element(1u), [&] { EmitBdaWrite(ctx, access.inst, texel.address, MemoryExtract(state, encoded, 0u), 16u, enabled); });
            EmitIfCondition(state, element(2u), [&] { EmitBdaWrite(ctx, access.inst, texel.address, MemoryExtract(state, encoded, 0u), 32u, enabled); });
            EmitIfCondition(state, element(3u), [&] { EmitBdaDwordWrites(ctx, access.inst, texel.address, 0u, 2u, encoded); });
            EmitIfCondition(state, element(4u), [&] { EmitBdaDwordWrites(ctx, access.inst, texel.address, 0u, 4u, encoded); });
        });
    });
}

std::uint32_t EmitMemoryAtomic(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const StorageImageMemoryFunctions& functions) {
    auto& state = ctx.state;
    const auto opcode = access.inst.Opcode();
    const auto condition = ctx.Arg(access.inst, access.inst.ArgumentCount() - 1u);
    const auto device = ConstantU32(state, spv::ScopeDevice);
    const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
    if (IsImageAtomic64Opcode(opcode)) {
        return EmitValueOrDefaultIfCondition(state, condition, TypeU64(state), ConstantU64(state, 0u), [&] {
            const auto texel = EmitMemoryTexel(ctx, access, functions);
            return EmitValueOrDefaultIfCondition(state, texel.inside, TypeU64(state), ConstantU64(state, 0u), [&] {
                const auto scalar = TypeScalarU64(state);
                const auto old = EmitBdaAtomic(ctx, access.inst, texel.address, 8u, [&](std::uint32_t pointer) {
                    const auto value = Unary(state, spv::OpBitcast, scalar, ctx.Arg(access.inst, 2));
                    const auto previous = state.module.AllocateId();
                    if (opcode == IrOpcode::ImageAtomicCmpSwap64) {
                        state.module.AddFunction(spv::OpAtomicCompareExchange, scalar, previous, pointer, device, relaxed, relaxed, value, Unary(state, spv::OpBitcast, scalar, ctx.Arg(access.inst, 3)));
                    } else {
                        state.module.AddFunction(ImageAtomicOpcode(opcode), scalar, previous, pointer, device, relaxed, value);
                    }
                    EmitDeviceAtomicMemoryBarrier(state);
                    return previous;
                });
                return Unary(state, spv::OpBitcast, TypeU64(state), old);
            });
        });
    }
    const auto value = ctx.Arg(access.inst, 2);
    return EmitValueOrZeroIfCondition(state, condition, [&] {
        const auto texel = EmitMemoryTexel(ctx, access, functions);
        return EmitValueOrZeroIfCondition(state, texel.inside, [&] {
            return EmitBdaAtomic(ctx, access.inst, texel.address, 4u, [&](std::uint32_t pointer) {
                if (opcode == IrOpcode::ImageAtomicInc32 || opcode == IrOpcode::ImageAtomicDec32) {
                    return AtomicUpdate(state, pointer, ResourceKind::Buffer, [&](std::uint32_t current) {
                        return opcode == IrOpcode::ImageAtomicInc32 ? AtomicIncrement(state, current, value) : AtomicDecrement(state, current, value);
                    });
                }
                if (opcode == IrOpcode::ImageAtomicFMin32 || opcode == IrOpcode::ImageAtomicFMax32) {
                    return AtomicUpdate(state, pointer, ResourceKind::Buffer, [&](std::uint32_t current) {
                        return AtomicFloatMinMax(state, current, value, opcode == IrOpcode::ImageAtomicFMax32);
                    });
                }
                if (opcode == IrOpcode::ImageAtomicFCmpSwap32) {
                    const auto comparator = ctx.Arg(access.inst, 3);
                    return AtomicUpdate(state, pointer, ResourceKind::Buffer, [&](std::uint32_t current) {
                        return AtomicFloatCompareSwap(state, current, value, comparator);
                    });
                }
                const auto old = state.module.AllocateId();
                if (opcode == IrOpcode::ImageAtomicCmpSwap32) {
                    state.module.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), old, pointer, device, relaxed, relaxed, value, ctx.Arg(access.inst, 3));
                } else {
                    state.module.AddFunction(ImageAtomicOpcode(opcode), TypeU32(state), old, pointer, device, relaxed, value);
                }
                EmitDeviceAtomicMemoryBarrier(state);
                return old;
            });
        });
    });
}

void EmitByReadOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const auto elements = access.mem.imageByElements;
    const bool packed = access.mem.imagePacked;
    const auto components = packed ? 1u : access.mem.dataDwords / elements;
    if (access.image.dimension != RdnaImageDimension::Dim2D || components == 0u || (packed ? access.mem.dataDwords != 1u : components * elements != access.mem.dataDwords) || access.mem.dataBits != 32u) {
        ctx.Fail(access.inst, "is a MIMG BY2/BY4 load outside the measured 2D, 32-bit data subset");
    }
    state.module.EmitCapability(spv::CapabilityImageQuery);
    const auto condition = ctx.Arg(access.inst, 2);
    ctx.Define(access.inst, EmitValueOrDefaultIfCondition(state, condition, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4), [&]() {
        const auto image = LoadSampledImageDescriptor(state, access.mem.resource);
        const auto x = AddressU32(ctx, access, 0);
        const auto y = AddressU32(ctx, access, 1);
        const auto lod = LodU32(ctx, access);
        const auto levels = state.module.AllocateId();
        state.module.AddFunction(spv::OpImageQueryLevels, TypeU32(state), levels, image);
        const auto lodInside = Binary(state, spv::OpULessThan, TypeBool(state), lod, levels);
        const auto queryLod = Select(state, TypeU32(state), lodInside, lod, ConstantU32(state, 0));
        const auto size = state.module.AllocateId();
        state.module.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, access.image.dimension), size, image, ConstantU32(state, 0));
        const auto levelExtent = [&](std::uint32_t axis) {
            const auto base = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), base, size, axis);
            const auto shifted = Binary(state, spv::OpShiftRightLogical, TypeU32(state), base, queryLod);
            return Select(state, TypeU32(state), Binary(state, spv::OpUGreaterThan, TypeBool(state), shifted, ConstantU32(state, 1)), shifted, ConstantU32(state, 1));
        };
        const auto width = levelExtent(0u);
        const auto height = levelExtent(1u);
        auto inside = Binary(state, spv::OpLogicalAnd, TypeBool(state), lodInside, Binary(state, spv::OpULessThan, TypeBool(state), x, width));
        inside = Binary(state, spv::OpLogicalAnd, TypeBool(state), inside, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), Binary(state, spv::OpISub, TypeU32(state), width, x), ConstantU32(state, elements)));
        inside = Binary(state, spv::OpLogicalAnd, TypeBool(state), inside, Binary(state, spv::OpULessThan, TypeBool(state), y, height));
        const auto first = Select(state, TypeU32(state), inside, Binary(state, spv::OpBitwiseAnd, TypeU32(state), x, ConstantU32(state, ~(elements - 1u))), ConstantU32(state, 0));
        const auto row = Select(state, TypeU32(state), inside, y, ConstantU32(state, 0));
        MemoryInfo texelMemory = access.mem;
        texelMemory.dmask = (1u << components) - 1u;
        texelMemory.dataDwords = components;
        texelMemory.componentCount = components;
        const ImageEmitAccess texelAccess{access.inst, texelMemory, access.image, access.address};
        std::uint32_t words[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0)};
        for (std::uint32_t element = 0; element < elements; ++element) {
            const auto column = Binary(state, spv::OpIAdd, TypeU32(state), first, ConstantU32(state, element));
            const auto coord = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 2), coord, column, row);
            const auto color = state.module.AllocateId();
            state.module.AddFunction(spv::OpImageFetch, ImageVectorType(state, access.image.numericClass, 4), color, image, coord, spv::ImageOperandsLodMask, queryLod);
            if (packed) {
                const auto info = PackedTexelFormat(ctx, access);
                std::uint32_t texelBits = 0;
                for (std::uint32_t component = 0; component < info.componentCount; ++component) texelBits += info.componentBits[component];
                if (texelBits * elements > 32u) ctx.Fail(access.inst, "is a MIMG PCK2/PCK4 load whose elements do not fit one dword");
                const auto raw = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), raw, PackedImageTexel(ctx, access, color), 0u);
                const auto bits = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw, ConstantU32(state, texelBits == 32u ? 0xffffffffu : (1u << texelBits) - 1u));
                const auto shifted = element == 0u ? bits : Binary(state, spv::OpShiftLeftLogical, TypeU32(state), bits, ConstantU32(state, element * texelBits));
                words[0] = Binary(state, spv::OpBitwiseOr, TypeU32(state), words[0], Select(state, TypeU32(state), inside, shifted, ConstantU32(state, 0)));
                continue;
            }
            const auto texel = ResultVector(ctx, texelAccess, UnpackImageTexel(ctx, texelAccess, color), access.image.numericClass, false, false);
            for (std::uint32_t component = 0; component < components; ++component) {
                const auto value = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, texel, component);
                words[element * components + component] = Select(state, TypeU32(state), inside, value, ConstantU32(state, 0));
            }
        }
        const auto result = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, words[0], words[1], words[2], words[3]);
        return result;
    }));
}

std::uint32_t EmitViewRead(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, std::uint32_t condition) {
    auto& state = ctx.state;
    const auto& dimensionInfo = RdnaImageDimensionInfoFor(access.image.dimension);
    const auto numericClass = access.image.numericClass;
    if (VolumeSliceAddress(access)) {
        condition = Binary(state, spv::OpLogicalAnd, TypeBool(state), condition, VolumeSliceInRange(ctx, access, LoadSampledImageDescriptor(state, access.mem.resource), LodU32(ctx, access)));
    }
    return EmitValueOrDefaultIfCondition(state, condition, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4), [&]() {
        const auto descriptor = LoadSampledImageDescriptor(state, access.mem.resource);
        const auto color = state.module.AllocateId();
        const auto coord = CoordU32(ctx, access);
        if (dimensionInfo.multisampled != 0u) {
            const auto& addressInfo = AddressDimension(access);
            if (addressInfo.multisampled == 0u || access.mem.imageAddressComponents <= addressInfo.coordinateComponents) {
                ctx.Fail(access.inst, "has no sample index in the image address");
            }
            state.module.AddFunction(spv::OpImageFetch, ImageVectorType(state, numericClass, 4), color, descriptor, coord, spv::ImageOperandsSampleMask, AddressU32(ctx, access, addressInfo.coordinateComponents));
        } else {
            state.module.AddFunction(spv::OpImageFetch, ImageVectorType(state, numericClass, 4), color, descriptor, coord, spv::ImageOperandsLodMask, LodU32(ctx, access));
        }
        if (access.mem.imagePacked) {
            return PackedImageTexel(ctx, access, color);
        }
        return ResultVector(ctx, access, UnpackImageTexel(ctx, access, color), numericClass, false, false);
    });
}

void EmitReadOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    if (access.mem.imageByElements != 0u) {
        EmitByReadOp(ctx, access);
        return;
    }
    auto& state = ctx.state;
    auto condition = ctx.Arg(access.inst, 2);
    if (access.mem.imageDimension == RdnaImageDimension::Dim3D && (access.image.dimension == RdnaImageDimension::Dim2D || access.image.dimension == RdnaImageDimension::Dim1D)) {
        condition = Binary(state, spv::OpLogicalAnd, TypeBool(state), condition, Binary(state, spv::OpIEqual, TypeBool(state), AddressU32(ctx, access, 2u), ConstantU32(state, 0u)));
    }
    const auto* memory = MemoryFunctions(ctx, access);
    if (memory == nullptr) {
        ctx.Define(access.inst, EmitViewRead(ctx, access, condition));
        return;
    }
    ctx.Define(access.inst, EmitValueIfElse(state, MemorySlotSelected(state, access.mem.resource), TypeU32Vector(state, 4u), [&] { return EmitMemoryRead(ctx, access, *memory, condition); }, [&] { return EmitViewRead(ctx, access, condition); }));
}

std::uint32_t ImageSampleIndex(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    if (RdnaImageDimensionInfoFor(access.image.dimension).multisampled == 0u) return ConstantU32(ctx.state, 0u);
    const auto& addressInfo = AddressDimension(access);
    if (addressInfo.multisampled == 0u || access.mem.imageAddressComponents <= addressInfo.coordinateComponents) ctx.Fail(access.inst, "has no sample index in the image address");
    return AddressU32(ctx, access, addressInfo.coordinateComponents);
}

void EmitViewWrite(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const bool uintImage = access.image.numericClass == IrTextureNumericClass::Uint;
    auto condition = ctx.Arg(access.inst, 3);
    if (VolumeSliceAddress(access)) {
        if (access.image.mipMode == ImageMipMode::DynamicStorage) ctx.Fail(access.inst, "stores a 2D-array address to a 3D image with dynamic mips, which is not implemented");
        const auto image = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, ImageType(state, access.image), image, StorageImageDescriptorPointer(state, access.mem.resource, 0u));
        condition = Binary(state, spv::OpLogicalAnd, TypeBool(state), condition, VolumeSliceInRange(ctx, access, image, std::nullopt));
    }
    EmitIfCondition(state, condition, [&]() {
        const auto mipLod = access.image.mipMode == ImageMipMode::DynamicStorage ? LodU32(ctx, access) : 0u;
        const auto coord = CoordU32(ctx, access);
        const auto texel = access.mem.imagePacked ? PackedStoreTexel(ctx, access, ctx.Arg(access.inst, 2)) : StoreTexel(ctx, access, ctx.Arg(access.inst, 2), uintImage);
        EmitStorageImageWrite(state, access.mem.resource, mipLod, coord, texel, ImageSampleIndex(ctx, access));
    });
}

void EmitWriteOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const auto* memory = MemoryFunctions(ctx, access);
    if (memory == nullptr) {
        EmitViewWrite(ctx, access);
        return;
    }
    const auto selected = MemorySlotSelected(state, access.mem.resource);
    EmitIfCondition(state, selected, [&] { EmitMemoryWrite(ctx, access, *memory); });
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), selected), [&] { EmitViewWrite(ctx, access); });
}

std::uint32_t EmitViewAtomic(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const auto opcode = access.inst.Opcode();
    const auto condition = ctx.Arg(access.inst, access.inst.ArgumentCount() - 1u);
    if (IsImageAtomic64Opcode(opcode)) {
        return EmitValueOrDefaultIfCondition(state, condition, TypeU64(state), ConstantU64(state, 0u), [&]() {
            const auto scalar = TypeScalarU64(state);
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpImageTexelPointer, TypePointer(state, spv::StorageClassImage, scalar), pointer, StorageImageDescriptorPointer(state, access.mem.resource, 0u), CoordU32(ctx, access), ConstantU32(state, 0));
            const auto value = Unary(state, spv::OpBitcast, scalar, ctx.Arg(access.inst, 2));
            const auto old = state.module.AllocateId();
            if (opcode == IrOpcode::ImageAtomicCmpSwap64) {
                const auto comparator = Unary(state, spv::OpBitcast, scalar, ctx.Arg(access.inst, 3));
                state.module.AddFunction(spv::OpAtomicCompareExchange, scalar, old, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, spv::MemorySemanticsMaskNone), value, comparator);
            } else {
                state.module.AddFunction(ImageAtomicOpcode(opcode), scalar, old, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), value);
            }
            EmitDeviceAtomicMemoryBarrier(state);
            return Unary(state, spv::OpBitcast, TypeU64(state), old);
        });
    }
    const auto value = ctx.Arg(access.inst, 2);
    return EmitValueOrZeroIfCondition(state, condition, [&]() {
        const auto pointer = state.module.AllocateId();
        const auto pointerType = TypePointer(state, spv::StorageClassImage, TypeU32(state));
        state.module.AddFunction(spv::OpImageTexelPointer, pointerType, pointer, StorageImageDescriptorPointer(state, access.mem.resource, 0u), CoordU32(ctx, access), ImageSampleIndex(ctx, access));
        if (opcode == IrOpcode::ImageAtomicInc32 || opcode == IrOpcode::ImageAtomicDec32) {
            return AtomicUpdate(state, pointer, ResourceKind::Image, [&](std::uint32_t current) {
                return opcode == IrOpcode::ImageAtomicInc32 ? AtomicIncrement(state, current, value) : AtomicDecrement(state, current, value);
            });
        }
        if (opcode == IrOpcode::ImageAtomicFMin32 || opcode == IrOpcode::ImageAtomicFMax32) {
            return AtomicUpdate(state, pointer, ResourceKind::Image, [&](std::uint32_t current) {
                return AtomicFloatMinMax(state, current, value, opcode == IrOpcode::ImageAtomicFMax32);
            });
        }
        if (opcode == IrOpcode::ImageAtomicFCmpSwap32) {
            const auto comparator = ctx.Arg(access.inst, 3);
            return AtomicUpdate(state, pointer, ResourceKind::Image, [&](std::uint32_t current) {
                return AtomicFloatCompareSwap(state, current, value, comparator);
            });
        }
        const auto old = state.module.AllocateId();
        if (opcode == IrOpcode::ImageAtomicCmpSwap32) {
            state.module.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), old, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, spv::MemorySemanticsMaskNone), value, ctx.Arg(access.inst, 3));
        } else {
            state.module.AddFunction(ImageAtomicOpcode(opcode), TypeU32(state), old, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), value);
        }
        EmitDeviceAtomicMemoryBarrier(state);
        return old;
    });
}

void EmitAtomicOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const auto* memory = MemoryFunctions(ctx, access);
    if (memory == nullptr) {
        ctx.Define(access.inst, EmitViewAtomic(ctx, access));
        return;
    }
    const auto type = IsImageAtomic64Opcode(access.inst.Opcode()) ? TypeU64(state) : TypeU32(state);
    ctx.Define(access.inst, EmitValueIfElse(state, MemorySlotSelected(state, access.mem.resource), type, [&] { return EmitMemoryAtomic(ctx, access, *memory); }, [&] { return EmitViewAtomic(ctx, access); }));
}

SampleSetup MakeSampleSetup(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    const auto& dimensionInfo = RdnaImageDimensionInfoFor(access.image.dimension);
    const auto layout = Layout(access.mem);
    const bool dref = HasFlag(access.mem, RdnaImageSampleFlagCompare);
    if (dref && access.image.conversionFormat != IrBufferFormat::Invalid) {
        ctx.Fail(access.inst, "uses depth comparison with a packed integer image");
    }
    const auto coord = CoordF32(ctx, access, layout.coord, dimensionInfo.coordinateComponents, AddressDimension(access).coordinateComponents);
    const bool unnormalized = ctx.state.tableAccess.active && ctx.state.tableAccess.unnormalized != 0u;
    return {dimensionInfo, layout, access.image.numericClass, dref, coord, unnormalized ? coord : 0u};
}

void EmitPackedHorizontalGatherOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const auto& mem = access.mem;
    const auto dimension = access.image.dimension;
    if (dimension != RdnaImageDimension::Dim1D && dimension != RdnaImageDimension::Dim2D) {
        ctx.Fail(access.inst, "is a packed horizontal gather of an image dimension whose row and slice rules are not measured");
    }
    if (Layout(mem).clamp != NoImageComponent || HasFlag(mem, RdnaImageSampleFlagCompare)) {
        ctx.Fail(access.inst, "is a packed horizontal gather with depth comparison or an LOD clamp, which is not implemented");
    }
    if (AddressDimension(access).coordinateComponents < RdnaImageDimensionInfoFor(dimension).coordinateComponents) {
        ctx.Fail(access.inst, "has an image address with too few coordinate components");
    }
    const auto info = PackedTexelFormat(ctx, access);
    std::uint32_t texelBits = 0;
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        texelBits += info.componentBits[component];
    }
    std::uint32_t streamWords[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0)};
    std::uint32_t rowInside = 0;
    const bool twoDimensional = dimension == RdnaImageDimension::Dim2D;
    if (texelBits <= 32u) {
        state.module.EmitCapability(spv::CapabilityImageQuery);
        const auto image = LoadSampledImageDescriptor(state, mem.resource);
        const auto size = state.module.AllocateId();
        state.module.AddFunction(spv::OpImageQuerySizeLod, twoDimensional ? TypeI32Vector(state, 2) : TypeI32(state), size, image, ConstantU32(state, 0));
        const auto extent = [&](std::uint32_t axis) {
            if (!twoDimensional) {
                return size;
            }
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeI32(state), value, size, axis);
            return value;
        };
        const auto texelFloor = [&](std::uint32_t address, std::uint32_t length, bool centre) {
            auto scaled = Binary(state, spv::OpFMul, TypeF32(state), AddressF32(ctx, access, address), Unary(state, spv::OpConvertSToF, TypeF32(state), length));
            state.module.AddAnnotation(spv::OpDecorate, scaled, spv::DecorationNoContraction);
            if (centre) {
                scaled = Binary(state, spv::OpFSub, TypeF32(state), scaled, ConstantF32(state, 0x3f000000u));
            }
            const auto floored = state.module.AllocateId();
            state.module.AddFunction(spv::OpExtInst, TypeF32(state), floored, GlslStd450(state), GLSLstd450Floor, scaled);
            return Unary(state, spv::OpConvertFToS, TypeI32(state), floored);
        };
        const auto clampInto = [&](std::uint32_t value, std::uint32_t length) {
            const auto last = Binary(state, spv::OpISub, TypeI32(state), length, ConstantI32(state, 1));
            const auto low = Select(state, TypeI32(state), Binary(state, spv::OpSLessThan, TypeBool(state), value, ConstantI32(state, 0)), ConstantI32(state, 0), value);
            return Select(state, TypeI32(state), Binary(state, spv::OpSLessThan, TypeBool(state), last, low), last, low);
        };
        const auto width = extent(0u);
        const auto anchor = texelFloor(0u, width, true);
        std::uint32_t row = 0;
        if (twoDimensional) {
            const auto height = extent(1u);
            const auto rawRow = texelFloor(1u, height, false);
            rowInside = Binary(state, spv::OpSLessThan, TypeBool(state), rawRow, height);
            row = clampInto(rawRow, height);
        }
        for (std::uint32_t element = 0; element < 4u; element++) {
            const auto column = clampInto(Binary(state, spv::OpIAdd, TypeI32(state), anchor, ConstantI32(state, static_cast<std::int32_t>(element) - 1)), width);
            std::uint32_t coord = column;
            if (twoDimensional) {
                coord = state.module.AllocateId();
                state.module.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(state, 2), coord, column, row);
            }
            const auto color = state.module.AllocateId();
            state.module.AddFunction(spv::OpImageFetch, ImageVectorType(state, access.image.numericClass, 4), color, image, coord, spv::ImageOperandsLodMask, ConstantI32(state, 0));
            const auto packed = PackedImageTexel(ctx, access, color);
            const auto bits = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), bits, packed, 0u);
            const auto position = element * texelBits;
            const auto word = position / 32u;
            const auto shift = position % 32u;
            streamWords[word] = Binary(state, spv::OpBitwiseOr, TypeU32(state), streamWords[word], shift == 0u ? bits : Binary(state, spv::OpShiftLeftLogical, TypeU32(state), bits, ConstantU32(state, shift)));
            if (shift != 0u && shift + texelBits > 32u) {
                streamWords[word + 1u] = Binary(state, spv::OpBitwiseOr, TypeU32(state), streamWords[word + 1u], Binary(state, spv::OpShiftRightLogical, TypeU32(state), bits, ConstantU32(state, 32u - shift)));
            }
        }
    }
    std::uint32_t selected[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0)};
    std::uint32_t next = 0;
    for (std::uint32_t bit = 0; bit < 4u; bit++) {
        if (((mem.dmask >> bit) & 1u) == 0u) {
            continue;
        }
        if (texelBits <= 32u) {
            selected[next] = rowInside != 0u ? Select(state, TypeU32(state), rowInside, streamWords[bit], ConstantU32(state, 0)) : streamWords[bit];
        }
        next++;
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, selected[0], selected[1], selected[2], selected[3]);
    ctx.Define(access.inst, result);
}

void EmitGatherOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const SampleSetup& setup) {
    auto& state = ctx.state;
    const auto& mem = access.mem;
    const auto dimension = access.image.dimension;
    if (setup.layout.clamp != NoImageComponent) {
        ctx.Fail(access.inst, "is a gather with an LOD clamp, which is not implemented");
    }
    if (setup.dref && (access.image.emulatedCompare & EmulatedCompare::Enabled) != 0u) {
        ctx.Fail(access.inst, "is a comparison gather of a color texture, which is not implemented");
    }
    if (dimension == RdnaImageDimension::Dim1D || dimension == RdnaImageDimension::Dim1DArray) {
        if (setup.dref || !HasFlag(mem, RdnaImageSampleFlagLevelZero) || HasFlag(mem, RdnaImageSampleFlagGatherHorizontal)) {
            ctx.Fail(access.inst, "has an unsupported 1D gather variant");
        }
        const auto sample = EmitOneDimensionalGatherLz(ctx, access, setup);
        ctx.Define(access.inst, ResultVector(ctx, access, UnpackImageGather(ctx, access, sample), setup.numericClass, false, true));
        return;
    }
    const auto sampled = MakeSampledImage(state, mem.resource, mem.sampler);
    const auto sample = state.module.AllocateId();
    std::vector<std::uint32_t> words;
    if (setup.dref) {
        const auto drefValue = DrefValueF32(ctx, access, setup.layout);
        words = {spv::OpImageDrefGather, TypeF32Vector(state, 4), sample, sampled, setup.coord, drefValue};
    } else {
        std::uint32_t component = 0;
        if (ImageConversionFormat(access.image).format == IrBufferFormat::Invalid) {
            component = ImageGatherComponent(EffectiveDmask(mem));
        }
        words = {spv::OpImageGather, ImageVectorType(state, setup.numericClass, 4), sample, sampled, setup.coord, ConstantU32(state, component)};
    }
    if (HasFlag(mem, RdnaImageSampleFlagGatherHorizontal)) {
        words.push_back(spv::ImageOperandsConstOffsetsMask);
        words.push_back(HorizontalOffsets(ctx, access));
    } else if (setup.layout.offset != NoImageComponent) {
        words.push_back(spv::ImageOperandsOffsetMask);
        words.push_back(PackedOffset(ctx, access, setup.layout));
    }
    state.module.AddFunction(words);
    auto resultNumericClass = setup.numericClass;
    if (setup.dref) {
        resultNumericClass = IrTextureNumericClass::Float;
    }
    // Debug aid: APS5_GATHER_CONST=<n> makes every non-depth gather return that integer (or its
    // float value) in all four texels, to separate a wrong gather result from wrong math after it.
    static const char* gatherConstText = std::getenv("APS5_GATHER_CONST");
    auto gathered = sample;
    if (gatherConstText != nullptr && !setup.dref) {
        const auto value = static_cast<std::uint32_t>(std::strtoul(gatherConstText, nullptr, 0));
        const bool integer = resultNumericClass == IrTextureNumericClass::Uint || resultNumericClass == IrTextureNumericClass::Sint;
        const auto component = integer ? ConstantU32(state, value) : ConstantF32(state, std::bit_cast<std::uint32_t>(static_cast<float>(value)));
        gathered = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, ImageVectorType(state, resultNumericClass, 4), gathered, component, component, component, component);
    }
    ctx.Define(access.inst, ResultVector(ctx, access, UnpackImageGather(ctx, access, gathered), resultNumericClass, false, true));
}

void EmitEmulatedCompareSample(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const SampleSetup& setup) {
    auto& state = ctx.state;
    const auto& mem = access.mem;
    const auto& image = access.image;
    if (!state.tableAccess.active || state.tableAccess.compare == 0u) ctx.Fail(access.inst, "compares against a color texture outside the device image table");
    const auto compare = state.tableAccess.compare;
    const auto parameter = [&](std::uint32_t word) {
        constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 6> fields{{{EmulatedCompare::FunctionShift, 7u}, {4u, 1u}, {EmulatedCompare::ClampXShift, 3u}, {EmulatedCompare::ClampYShift, 3u}, {EmulatedCompare::ReferenceShift, 3u}, {12u, 1u}}};
        return EmitAndConstant(state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), compare, ConstantU32(state, fields.at(word).first)), fields.at(word).second);
    };
    const auto equal = [&](std::uint32_t value, std::uint32_t literal) { return Binary(state, spv::OpIEqual, TypeBool(state), value, ConstantU32(state, literal)); };
    if (HasFlag(mem, RdnaImageSampleFlagDerivative) || HasFlag(mem, RdnaImageSampleFlagLod)) ctx.Fail(access.inst, "compares against a color texture with gradients or an explicit LOD, which is not implemented");
    const bool arrayed = image.dimension == RdnaImageDimension::Dim2DArray;
    if (image.dimension != RdnaImageDimension::Dim2D && !arrayed) ctx.Fail(access.inst, "compares against a color texture that is not a 2D or 2D array view, which is not implemented");
    const auto f32 = TypeF32(state);
    const auto i32 = TypeI32(state);
    const auto u32 = TypeU32(state);
    const auto ext = [&](std::uint32_t type, std::uint32_t op, std::initializer_list<std::uint32_t> args) {
        const auto value = state.module.AllocateId();
        std::vector<std::uint32_t> words{spv::OpExtInst, type, value, GlslStd450(state), op};
        words.insert(words.end(), args.begin(), args.end());
        state.module.AddFunction(std::span<const std::uint32_t>(words));
        return value;
    };
    const auto extract = [&](std::uint32_t type, std::uint32_t composite, std::uint32_t index) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, type, value, composite, index);
        return value;
    };
    const auto f32Constant = [&](float value) { return ConstantF32(state, std::bit_cast<std::uint32_t>(value)); };
    const auto descriptor = LoadSampledImageDescriptor(state, mem.resource);
    state.module.EmitCapability(spv::CapabilityImageQuery);
    const auto size = state.module.AllocateId();
    state.module.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, image.dimension), size, descriptor, ConstantU32(state, 0u));
    const auto width = Unary(state, spv::OpBitcast, i32, extract(u32, size, 0u));
    const auto height = Unary(state, spv::OpBitcast, i32, extract(u32, size, 1u));
    auto layer = ConstantU32(state, 0u);
    if (arrayed) {
        const auto layers = Unary(state, spv::OpBitcast, i32, extract(u32, size, 2u));
        const auto rounded = Unary(state, spv::OpConvertFToS, i32, ext(f32, GLSLstd450Floor, {Binary(state, spv::OpFAdd, f32, extract(f32, setup.coord, 2u), f32Constant(0.5f))}));
        layer = Unary(state, spv::OpBitcast, u32, ext(i32, GLSLstd450SClamp, {rounded, ConstantI32(state, 0), Binary(state, spv::OpISub, i32, layers, ConstantI32(state, 1))}));
    }
    auto reference = DrefValueF32(ctx, access, setup.layout);
    reference = Select(state, f32, equal(parameter(4u), EmulatedCompare::ReferenceUnorm), ext(f32, GLSLstd450FClamp, {reference, f32Constant(0.0f), f32Constant(1.0f)}), reference);
    reference = Select(state, f32, equal(parameter(4u), EmulatedCompare::ReferenceSnorm), ext(f32, GLSLstd450FClamp, {reference, f32Constant(-1.0f), f32Constant(1.0f)}), reference);
    const auto address = [&](std::uint32_t index, std::uint32_t extent, std::uint32_t mode) {
        const auto remainder = Binary(state, spv::OpSRem, i32, index, extent);
        const auto negative = Binary(state, spv::OpSLessThan, TypeBool(state), remainder, ConstantI32(state, 0));
        const auto correction = Select(state, i32, negative, extent, ConstantI32(state, 0));
        const auto wrapped = Binary(state, spv::OpIAdd, i32, remainder, correction);
        const auto clamped = ext(i32, GLSLstd450SClamp, {index, ConstantI32(state, 0), Binary(state, spv::OpISub, i32, extent, ConstantI32(state, 1))});
        return Select(state, i32, equal(mode, EmulatedCompare::AddressWrap), wrapped, clamped);
    };
    const auto inside = [&](std::uint32_t index, std::uint32_t extent) {
        const auto notBelow = Binary(state, spv::OpSGreaterThanEqual, TypeBool(state), index, ConstantI32(state, 0));
        const auto below = Binary(state, spv::OpSLessThan, TypeBool(state), index, extent);
        return Binary(state, spv::OpLogicalAnd, TypeBool(state), notBelow, below);
    };
    const auto one = f32Constant(1.0f);
    const auto zero = f32Constant(0.0f);
    const auto borderRed = Select(state, f32, equal(parameter(5u), 1u), one, zero);
    const auto compareTexel = [&](std::uint32_t x, std::uint32_t y) {
        const auto modeX = parameter(2u);
        const auto modeY = parameter(3u);
        const auto ux = Unary(state, spv::OpBitcast, u32, address(x, width, modeX));
        const auto uy = Unary(state, spv::OpBitcast, u32, address(y, height, modeY));
        const auto coord = state.module.AllocateId();
        if (arrayed) state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 3), coord, ux, uy, layer);
        else state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 2), coord, ux, uy);
        const auto texel = state.module.AllocateId();
        state.module.AddFunction(spv::OpImageFetch, ImageVectorType(state, IrTextureNumericClass::Float, 4), texel, descriptor, coord, spv::ImageOperandsLodMask, ConstantU32(state, 0u));
        auto red = extract(f32, texel, 3u);
        for (std::uint32_t channel = 3u; channel-- > 0u;) red = Select(state, f32, equal(RuntimeImageSwizzle(state, mem.resource, channel), 4u), extract(f32, texel, channel), red);
        red = Select(state, f32, equal(modeX, EmulatedCompare::AddressBorder), Select(state, f32, inside(x, width), red, borderRed), red);
        red = Select(state, f32, equal(modeY, EmulatedCompare::AddressBorder), Select(state, f32, inside(y, height), red, borderRed), red);
        constexpr std::array<spv::Op, 6> operations{spv::OpFOrdLessThan, spv::OpFOrdEqual, spv::OpFOrdLessThanEqual, spv::OpFOrdGreaterThan, spv::OpFOrdNotEqual, spv::OpFOrdGreaterThanEqual};
        auto result = Select(state, f32, equal(parameter(0u), 7u), one, zero);
        for (std::uint32_t index = 0; index < operations.size(); ++index) {
            const auto compared = Select(state, f32, Binary(state, operations[index], TypeBool(state), reference, red), one, zero);
            result = Select(state, f32, equal(parameter(0u), index + 1u), compared, result);
        }
        result = Select(state, f32, equal(parameter(4u), EmulatedCompare::ReferenceZero), zero, result);
        return result;
    };
    const auto scaledU = Binary(state, spv::OpFMul, f32, extract(f32, setup.coord, 0u), Unary(state, spv::OpConvertSToF, f32, width));
    const auto scaledV = Binary(state, spv::OpFMul, f32, extract(f32, setup.coord, 1u), Unary(state, spv::OpConvertSToF, f32, height));
    auto offsetX = ConstantI32(state, 0);
    auto offsetY = ConstantI32(state, 0);
    if (setup.layout.offset != NoImageComponent) {
        const auto offset = PackedOffset(ctx, access, setup.layout);
        offsetX = extract(i32, offset, 0u);
        offsetY = extract(i32, offset, 1u);
    }
    const auto texelIndex = [&](std::uint32_t coordinate, std::uint32_t offset) {
        return Binary(state, spv::OpIAdd, i32, Unary(state, spv::OpConvertFToS, i32, ext(f32, GLSLstd450Floor, {coordinate})), offset);
    };
    const auto result = EmitValueIfElse(state, equal(parameter(1u), 0u), f32, [&] {
        return compareTexel(texelIndex(scaledU, offsetX), texelIndex(scaledV, offsetY));
    }, [&] {
        const auto half = f32Constant(0.5f);
        const auto centreU = Binary(state, spv::OpFSub, f32, scaledU, half);
        const auto centreV = Binary(state, spv::OpFSub, f32, scaledV, half);
        const auto floorU = ext(f32, GLSLstd450Floor, {centreU});
        const auto floorV = ext(f32, GLSLstd450Floor, {centreV});
        const auto weightU = Binary(state, spv::OpFSub, f32, centreU, floorU);
        const auto weightV = Binary(state, spv::OpFSub, f32, centreV, floorV);
        const auto x0 = Binary(state, spv::OpIAdd, i32, Unary(state, spv::OpConvertFToS, i32, floorU), offsetX);
        const auto y0 = Binary(state, spv::OpIAdd, i32, Unary(state, spv::OpConvertFToS, i32, floorV), offsetY);
        const auto x1 = Binary(state, spv::OpIAdd, i32, x0, ConstantI32(state, 1));
        const auto y1 = Binary(state, spv::OpIAdd, i32, y0, ConstantI32(state, 1));
        const auto top = ext(f32, GLSLstd450FMix, {compareTexel(x0, y0), compareTexel(x1, y0), weightU});
        const auto bottom = ext(f32, GLSLstd450FMix, {compareTexel(x0, y1), compareTexel(x1, y1), weightU});
        return ext(f32, GLSLstd450FMix, {top, bottom, weightV});
    });
    const auto channelResult = [&](std::uint32_t channel) {
        const auto selector = RuntimeImageSwizzle(state, mem.resource, channel);
        return Select(state, f32, equal(selector, 0u), zero, Select(state, f32, equal(selector, 1u), one, result));
    };
    std::uint32_t channels[4] = {};
    for (std::uint32_t channel = 0; channel < 4u; ++channel) channels[channel] = channelResult(channel);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), value, channels[0], channels[1], channels[2], channels[3]);
    ctx.Define(access.inst, ResultVector(ctx, access, value, IrTextureNumericClass::Float, false, false));
}

void EmitSampleOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access, const SampleSetup& setup) {
    auto& state = ctx.state;
    const auto& mem = access.mem;
    const auto& image = access.image;
    if (setup.dref && (image.emulatedCompare & EmulatedCompare::Enabled) != 0u) {
        EmitEmulatedCompareSample(ctx, access, setup);
        return;
    }
    const bool explicitLod = ImageSampleExplicitLod(mem.imageSampleFlags, state.program.Resources().stage);
    std::uint32_t opcode = spv::OpImageSampleImplicitLod;
    if (explicitLod) {
        opcode = setup.dref ? spv::OpImageSampleDrefExplicitLod : spv::OpImageSampleExplicitLod;
    } else if (setup.dref) {
        opcode = spv::OpImageSampleDrefImplicitLod;
    }
    std::uint32_t resultType = ImageVectorType(state, setup.numericClass, 4);
    std::uint32_t drefValue = 0;
    if (setup.dref) {
        resultType = TypeF32(state);
        drefValue = DrefValueF32(ctx, access, setup.layout);
    }
    std::uint32_t operandMask = 0;
    std::vector<std::uint32_t> operands;
    if (HasFlag(mem, RdnaImageSampleFlagDerivative)) {
        operandMask |= spv::ImageOperandsGradMask;
        operands.push_back(CoordF32(ctx, access, setup.layout.gradX, setup.dimensionInfo.spatialComponents, AddressDimension(access).spatialComponents, true));
        operands.push_back(CoordF32(ctx, access, setup.layout.gradY, setup.dimensionInfo.spatialComponents, AddressDimension(access).spatialComponents, true));
    } else if (explicitLod) {
        operandMask |= spv::ImageOperandsLodMask;
        operands.push_back(HasFlag(mem, RdnaImageSampleFlagLod) ? AddressF32(ctx, access, setup.layout.lod) : ZeroF32(state));
    } else if (setup.layout.bias != NoImageComponent) {
        operandMask |= spv::ImageOperandsBiasMask;
        operands.push_back(AddressF32(ctx, access, setup.layout.bias));
    }
    if (setup.layout.clamp != NoImageComponent) {
        const auto& capabilities = state.supportedCapabilities;
        if (std::find(capabilities.begin(), capabilities.end(), static_cast<std::uint32_t>(spv::CapabilityMinLod)) == capabilities.end()) ctx.Fail(access.inst, "clamps its LOD, which needs the device's shaderResourceMinLod");
        const auto clamp = AddressF32(ctx, access, setup.layout.clamp);
        if ((operandMask & spv::ImageOperandsLodMask) != 0u) {
            const auto clamped = state.module.AllocateId();
            state.module.AddFunction(spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state), GLSLstd450FMax, operands.back(), clamp);
            operands.back() = clamped;
        } else {
            state.module.EmitCapability(spv::CapabilityMinLod);
            operandMask |= spv::ImageOperandsMinLodMask;
            operands.push_back(clamp);
        }
    }
    const auto constantOffset = [&]() -> const IrValue* {
        const auto component = GetRdnaImageAddressComponentLayout(mem.imageSampleFlags, setup.layout.offset);
        const auto argument = component.bitOffset / 32u;
        if (component.bitWidth != 32u || argument >= access.address.ArgumentCount()) return nullptr;
        const auto* value = access.address.Argument(argument)->Resolve();
        return value->HasImmediate() ? value : nullptr;
    };
    const bool hasOffset = setup.layout.offset != NoImageComponent;
    const bool foldOffset = hasOffset && FoldsTexelOffsets(state.program.Info().samplers.at(mem.sampler)) && ResourceMaterializer::UnnormalizedSampleMismatch(image) == nullptr;
    const auto foldedMask = operandMask;
    const auto foldedOperands = operands;
    const bool gatherExtended = std::find(state.supportedCapabilities.begin(), state.supportedCapabilities.end(), static_cast<std::uint32_t>(spv::CapabilityImageGatherExtended)) != state.supportedCapabilities.end();
    const bool offsetOperand = hasOffset && (constantOffset() != nullptr || (state.nonConstantImageOffsets && gatherExtended));
    const bool runtimeUnnormalized = setup.texelCoord != 0u && !setup.dref;
    if (hasOffset && !offsetOperand && !foldOffset && !runtimeUnnormalized) {
        ctx.Fail(access.inst, "has a texel offset that is not a constant, which image sampling takes only with VK_KHR_maintenance8 and shaderImageGatherExtended");
    }
    if (offsetOperand && constantOffset() == nullptr) {
        state.module.EmitCapability(spv::CapabilityImageGatherExtended);
        operandMask |= spv::ImageOperandsOffsetMask;
        operands.insert(operands.end() - ((operandMask & spv::ImageOperandsMinLodMask) != 0u ? 1 : 0), PackedOffset(ctx, access, setup.layout));
    } else if (offsetOperand) {
        const auto bits = constantOffset()->ImmediateU32();
        std::array<std::uint32_t, 3> values{};
        for (std::uint32_t index = 0; index < setup.dimensionInfo.spatialComponents; index++) {
            const auto field = (bits >> (index * 8u)) & 0x3fu;
            values[index] = ConstantI32(state, static_cast<std::int32_t>(field ^ 0x20u) - 0x20);
        }
        const auto count = setup.dimensionInfo.spatialComponents;
        const auto offset = count == 1u ? values[0] : count == 2u
            ? state.module.Constant(spv::OpConstantComposite, TypeI32Vector(state, 2), values[0], values[1])
            : state.module.Constant(spv::OpConstantComposite, TypeI32Vector(state, 3), values[0], values[1], values[2]);
        operandMask |= spv::ImageOperandsConstOffsetMask;
        operands.insert(operands.end() - ((operandMask & spv::ImageOperandsMinLodMask) != 0u ? 1 : 0), offset);

    }
    const auto emitSample = [&](std::uint32_t coord, std::uint32_t mask, const std::vector<std::uint32_t>& arguments, std::uint32_t sampleOpcode) {
        const auto sampled = MakeSampledImage(state, mem.resource, mem.sampler);
        const auto sample = state.module.AllocateId();
        std::vector<std::uint32_t> words = {sampleOpcode, resultType, sample, sampled, coord};
        if (setup.dref) {
            words.push_back(drefValue);
        }
        if (mask != 0u) {
            words.push_back(mask);
            words.insert(words.end(), arguments.begin(), arguments.end());
        }
        state.module.AddFunction(words);
        return sample;
    };
    std::uint32_t sample = 0;
    const bool baseLevelIfUnnormalized = !setup.dref && SamplesBaseLevelIfUnnormalized(state.program.Info().samplers.at(mem.sampler));
    if (runtimeUnnormalized) {
        const auto unnormalized = Binary(state, spv::OpINotEqual, TypeBool(state), state.tableAccess.unnormalized, ConstantU32(state, 0u));
        const bool fold = hasOffset && setup.dimensionInfo.coordinateComponents == setup.dimensionInfo.spatialComponents;
        sample = EmitValueIfElse(state, unnormalized, resultType, [&] {
            return emitSample(fold ? FoldedOffsetCoord(ctx, access, setup) : setup.coord, spv::ImageOperandsLodMask, {ZeroF32(state)}, spv::OpImageSampleExplicitLod);
        }, [&] {
            return hasOffset && !offsetOperand ? state.module.Constant(spv::OpConstantNull, resultType) : emitSample(setup.coord, operandMask, operands, opcode);
        });
    } else if (baseLevelIfUnnormalized) {
        const auto selector = state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::UnnormalizedBaseLevel(mem.sampler), 0u);
        const auto unnormalized = Binary(state, spv::OpINotEqual, TypeBool(state), selector, ConstantU32(state, 0u));
        sample = EmitValueIfElse(state, unnormalized, resultType, [&] {
            return emitSample(setup.coord, spv::ImageOperandsLodMask, {ZeroF32(state)}, spv::OpImageSampleExplicitLod);
        }, [&] {
            return emitSample(setup.coord, operandMask, operands, opcode);
        });
    } else if (!foldOffset) {
        sample = emitSample(setup.coord, operandMask, operands, opcode);
    } else if (!offsetOperand) {
        sample = emitSample(FoldedOffsetCoord(ctx, access, setup), foldedMask, foldedOperands, opcode);
    } else {
        const auto selector = state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::FoldOffset(mem.sampler), 0u);
        const auto folded = Binary(state, spv::OpINotEqual, TypeBool(state), selector, ConstantU32(state, 0u));
        const auto foldLabel = state.module.AllocateId();
        const auto operandLabel = state.module.AllocateId();
        const auto merge = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, folded, foldLabel, operandLabel);
        EmitLabel(state, foldLabel);
        const auto foldSample = emitSample(FoldedOffsetCoord(ctx, access, setup), foldedMask, foldedOperands, opcode);
        const auto foldEnd = state.currentLabel;
        state.module.AddFunction(spv::OpBranch, merge);
        EmitLabel(state, operandLabel);
        const auto operandSample = emitSample(setup.coord, operandMask, operands, opcode);
        const auto operandEnd = state.currentLabel;
        state.module.AddFunction(spv::OpBranch, merge);
        EmitLabel(state, merge);
        sample = state.module.AllocateId();
        state.module.AddFunction(spv::OpPhi, resultType, sample, foldSample, foldEnd, operandSample, operandEnd);
    }
    const auto result = setup.dref ? sample : UnpackImageTexel(ctx, access, sample);
    ctx.Define(access.inst, ResultVector(ctx, access, result, setup.numericClass, setup.dref, false));
}

void EmitConstantSwizzleSample(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    auto& state = ctx.state;
    const auto numericClass = access.image.numericClass;
    const bool gather = access.inst.Opcode() == IrOpcode::ImageGatherRaw;
    const auto scalarType = ImageScalarType(state, numericClass);
    const auto one = numericClass == IrTextureNumericClass::Float ? ConstantF32(state, 0x3f800000u) : numericClass == IrTextureNumericClass::Sint ? ConstantI32(state, 1) : ConstantU32(state, 1u);
    const auto zero = SampledComponentZero(state, numericClass);
    std::uint32_t channels[4] = {};
    for (std::uint32_t channel = 0; channel < 4u; channel++) {
        const auto selector = RuntimeImageSwizzle(state, access.mem.resource, gather ? ImageGatherComponent(EffectiveDmask(access.mem)) : channel);
        channels[channel] = Select(state, scalarType, Binary(state, spv::OpIEqual, TypeBool(state), selector, ConstantU32(state, 1u)), one, zero);
    }
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, ImageVectorType(state, numericClass, 4), value, channels[0], channels[1], channels[2], channels[3]);
    ctx.Define(access.inst, ResultVector(ctx, access, value, numericClass, false, gather));
}

void EmitSamplingOp(SpirvValueEmitContext& ctx, const ImageEmitAccess& access) {
    if (access.image.constantSwizzle) {
        EmitConstantSwizzleSample(ctx, access);
        return;
    }
    if (access.image.srgbDecode) {
        ctx.Fail(access.inst, "samples or gathers an sRGB image the device cannot sample, which is not implemented");
    }
    if (access.mem.imagePacked) {
        EmitPackedHorizontalGatherOp(ctx, access);
        return;
    }
    const auto setup = MakeSampleSetup(ctx, access);
    if (access.inst.Opcode() == IrOpcode::ImageGatherRaw) {
        EmitGatherOp(ctx, access, setup);
    } else {
        EmitSampleOp(ctx, access, setup);
    }
}

constexpr std::uint32_t InvalidTableSelector = 0xffffffffu;

struct ScalarBufferDword {
    std::uint32_t byte = 0;
    std::uint32_t inRange = 0;
};

ScalarBufferDword EmitScalarBufferDword(SpirvEmitterState& state, std::uint32_t offset, std::uint32_t immediate, std::uint32_t sizeLow, std::uint32_t sizeHigh) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto sum = state.module.AllocateId();
    state.module.AddFunction(spv::OpIAddCarry, TypeU32Pair(state), sum, offset, immediate);
    const auto address = state.module.AllocateId();
    const auto carry = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u32, address, sum, 0u);
    state.module.AddFunction(spv::OpCompositeExtract, u32, carry, sum, 1u);
    ScalarBufferDword dword;
    dword.byte = Binary(state, spv::OpBitwiseAnd, u32, address, ConstantU32(state, ~3u));
    const auto below = Binary(state, spv::OpLogicalOr, boolean, Binary(state, spv::OpINotEqual, boolean, sizeHigh, ConstantU32(state, 0u)), Binary(state, spv::OpULessThan, boolean, dword.byte, Binary(state, spv::OpBitwiseAnd, u32, sizeLow, ConstantU32(state, ~3u))));
    dword.inRange = Binary(state, spv::OpLogicalAnd, boolean, Binary(state, spv::OpIEqual, boolean, carry, ConstantU32(state, 0u)), below);
    return dword;
}

}

std::uint32_t LoadTableMap(SpirvEmitterState& state, std::uint32_t index) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer, state.tableMapVariable, ConstantU32(state, 0u), index);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
    return value;
}

namespace {

std::uint32_t TableMapAt(SpirvEmitterState& state, std::uint32_t base, std::uint32_t offset) {
    return LoadTableMap(state, offset == 0u ? base : Binary(state, spv::OpIAdd, TypeU32(state), base, ConstantU32(state, offset)));
}

struct TableWord {
    std::uint32_t code = 0;
    std::uint32_t addressLow = 0;
    std::uint32_t addressHigh = 0;
    std::uint32_t poisoned = 0;
};

TableWord EmitTableWord(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t source, std::uint32_t table, std::size_t argument) {
    auto& state = ctx.state;
    const IrValue* handle = inst.Argument(argument)->Resolve();
    const auto& sources = state.program.Resources().descriptorSources;
    if (handle->ArgumentCount() == 0u || source >= sources.size() || !sources[source].tableColumn.has_value()) ctx.Fail(inst, "has an image table handle without its column");
    const auto& column = *sources[source].tableColumn;
    const auto offset = ctx.Def(handle->Argument(0));
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto header = ImageTableAbi::TableHeader(table);
    const auto field = [&](std::uint32_t index) { return LoadTableMap(state, ConstantU32(state, header + index)); };
    const auto mapStart = field(ImageTableAbi::TableMapStart);
    const auto keys = field(ImageTableAbi::TableKeyCount);
    const auto sizeLow = field(ImageTableAbi::TableSizeLow);
    const auto sizeHigh = field(ImageTableAbi::TableSizeHigh);
    const auto baseLow = field(ImageTableAbi::TableBaseLow);
    const auto baseHigh = field(ImageTableAbi::TableBaseHigh);
    const auto outsideCode = field(ImageTableAbi::TableOutsideCode);
    const auto tableFault = field(ImageTableAbi::TableFault);
    const auto relative = Binary(state, spv::OpISub, u32, offset, ConstantU32(state, column.addend));
    const auto record = Binary(state, spv::OpUDiv, u32, relative, ConstantU32(state, column.stride));
    const auto remainder = Binary(state, spv::OpISub, u32, relative, Binary(state, spv::OpIMul, u32, record, ConstantU32(state, column.stride)));
    const auto aligned = Binary(state, spv::OpIEqual, boolean, remainder, ConstantU32(state, 0u));
    const auto inMap = Binary(state, spv::OpLogicalAnd, boolean, aligned, Binary(state, spv::OpULessThan, boolean, record, keys));
    const auto mapped = LoadTableMap(state, Select(state, u32, inMap, Binary(state, spv::OpIAdd, u32, mapStart, record), ConstantU32(state, 0u)));
    if (column.address) {
        TableWord word;
        word.code = Select(state, u32, Binary(state, spv::OpINotEqual, boolean, tableFault, ConstantU32(state, 0u)), tableFault, Select(state, u32, inMap, mapped, ConstantU32(state, ImageTableAbi::PoisonCode(ImageTableAbi::OutsideSnapshotPoison))));
        const auto immediate = column.offset & ~3u;
        const auto first = state.module.AllocateId();
        state.module.AddFunction(spv::OpIAddCarry, TypeU32Pair(state), first, baseLow, Binary(state, spv::OpBitwiseAnd, u32, offset, ConstantU32(state, ~3u)));
        const auto firstLow = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, firstLow, first, 0u);
        const auto firstCarry = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, firstCarry, first, 1u);
        const auto second = state.module.AllocateId();
        state.module.AddFunction(spv::OpIAddCarry, TypeU32Pair(state), second, firstLow, ConstantU32(state, immediate));
        word.addressLow = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, word.addressLow, second, 0u);
        const auto secondCarry = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, secondCarry, second, 1u);
        const auto extension = ConstantU32(state, (immediate & 0x80000000u) != 0u ? 0xffffffffu : 0u);
        word.addressHigh = Binary(state, spv::OpIAdd, u32, Binary(state, spv::OpIAdd, u32, baseHigh, extension), Binary(state, spv::OpIAdd, u32, firstCarry, secondCarry));
        word.poisoned = Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, word.code, ConstantU32(state, ImageTableAbi::PoisonFlag)), ConstantU32(state, 0u));
        return word;
    }
    const auto dword = EmitScalarBufferDword(state, offset, ConstantU32(state, column.offset), sizeLow, sizeHigh);
    const auto outside = Select(state, u32, dword.inRange, ConstantU32(state, ImageTableAbi::PoisonCode(ImageTableAbi::OutsideSnapshotPoison)), outsideCode);
    TableWord word;
    word.code = Select(state, u32, Binary(state, spv::OpINotEqual, boolean, tableFault, ConstantU32(state, 0u)), tableFault, Select(state, u32, inMap, mapped, outside));
    const auto low = state.module.AllocateId();
    state.module.AddFunction(spv::OpIAddCarry, TypeU32Pair(state), low, baseLow, dword.byte);
    const auto carry = state.module.AllocateId();
    word.addressLow = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u32, word.addressLow, low, 0u);
    state.module.AddFunction(spv::OpCompositeExtract, u32, carry, low, 1u);
    word.addressHigh = Binary(state, spv::OpIAdd, u32, baseHigh, carry);
    word.poisoned = Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, word.code, ConstantU32(state, ImageTableAbi::PoisonFlag)), ConstantU32(state, 0u));
    return word;
}

void EmitTableFault(SpirvValueEmitContext& ctx, const IrValue& inst, const TableWord& word, std::uint32_t poison, std::uint32_t condition) {
    auto& state = ctx.state;
    EmitIfCondition(state, condition, [&] {
        RecordBdaFaultWords(state, word.addressLow, word.addressHigh, Binary(state, spv::OpBitwiseAnd, TypeU32(state), poison, ConstantU32(state, ~ImageTableAbi::PoisonFlag)), ConstantU32(state, inst.Flags<MemoryFlags>().pc), BdaAbi::FaultReason::ImageTable);
    });
}

std::uint32_t ZeroResult(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    switch (inst.Type()) {
        case IrType::U32: return ConstantU32(state, 0u);
        case IrType::U64: return ConstantU64(state, 0u);
        case IrType::U32x4: return ConstantU32CompositeZero(state, 4u);
        default: break;
    }
    ctx.Fail(inst, "has no zero result for an image table access");
}

struct TableAccessResult {
    std::uint32_t valid = 0;
    std::uint32_t flags = 0;
    bool unsupported = false;
};

TableAccessResult EmitTableAccess(SpirvValueEmitContext& ctx, const IrValue& inst, const MemoryInfo& memory, const ImageResource& base, bool tableSampler) {
    auto& state = ctx.state;
    if (state.tableMapVariable == 0 || state.faultBufferVariable == 0) ctx.Fail(inst, "reads an image table without its map or fault buffer");
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto negate = [&](std::uint32_t value) { return Unary(state, spv::OpLogicalNot, boolean, value); };
    const auto flagged = [&](std::uint32_t value) { return Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, value, ConstantU32(state, ImageTableAbi::PoisonFlag)), ConstantU32(state, 0u)); };
    const auto exec = ctx.Arg(inst, inst.ArgumentCount() - 1u);
    TableAccessResult result;
    result.valid = ConstantBool(state, true);
    auto live = ConstantBool(state, true);
    state.tableAccess = {};
    if (base.table != NoTable) {
        const auto word = EmitTableWord(ctx, inst, base.source, base.table, 0u);
        EmitTableFault(ctx, inst, word, word.code, both(exec, word.poisoned));
        if (base.tableOperation == TableOperation::Unsupported || state.program.Info().runtimeImageModes.at(memory.resource).empty()) {
            if (inst.Type() != IrType::Void) ctx.Define(inst, ZeroResult(ctx, inst));
            result.unsupported = true;
            return result;
        }
        const auto selected = both(Binary(state, spv::OpINotEqual, boolean, word.code, ConstantU32(state, ImageTableAbi::NullCode)), negate(word.poisoned));
        const auto record = Select(state, u32, selected, word.code, ConstantU32(state, 0u));
        const auto flags = TableMapAt(state, record, ImageTableAbi::ImageRecordFlags);
        const auto dword3 = TableMapAt(state, record, ImageTableAbi::ImageRecordDword3);
        const auto element = TableMapAt(state, record, ImageTableAbi::ImageRecordElement);
        const auto rejected = both(selected, flagged(flags));
        EmitTableFault(ctx, inst, word, flags, both(exec, rejected));
        const auto inBounds = Binary(state, spv::OpULessThan, boolean, element, LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderElementCount)));
        const auto outside = both(both(selected, negate(rejected)), negate(inBounds));
        EmitTableFault(ctx, inst, word, ConstantU32(state, ImageTableAbi::OutsideSnapshotPoison), both(exec, outside));
        const auto checked = both(both(selected, negate(rejected)), inBounds);
        const auto faults = LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderElementFaults));
        const auto driverFault = LoadTableMap(state, Select(state, u32, checked, Binary(state, spv::OpIAdd, u32, faults, element), ConstantU32(state, 0u)));
        const auto driverRejected = both(checked, flagged(driverFault));
        EmitTableFault(ctx, inst, word, driverFault, both(exec, driverRejected));
        result.valid = both(checked, negate(driverRejected));
        result.flags = flags;
        live = selected;
        const auto slot = LoadTableMap(state, Select(state, u32, checked, Binary(state, spv::OpIAdd, u32, LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderImageSlots)), element), ConstantU32(state, 0u)));
        state.tableAccess.element = Select(state, u32, checked, slot, ConstantU32(state, 0u));
        DecorateTableNonUniform(state, state.tableAccess.element);
        state.tableAccess.dword3 = dword3;
    }
    if (tableSampler) {
        const auto& sampler = state.program.Info().samplers.at(memory.sampler);
        const auto word = EmitTableWord(ctx, inst, sampler.source, sampler.table, 1u);
        EmitTableFault(ctx, inst, word, word.code, both(both(exec, live), word.poisoned));
        const auto selected = negate(word.poisoned);
        const auto record = Select(state, u32, selected, word.code, ConstantU32(state, 0u));
        const auto flags = TableMapAt(state, record, ImageTableAbi::SamplerRecordFlags);
        const auto element = TableMapAt(state, record, ImageTableAbi::SamplerRecordElement);
        const auto rejected = both(selected, flagged(flags));
        EmitTableFault(ctx, inst, word, flags, both(both(exec, live), rejected));
        const auto end = Binary(state, spv::OpIAdd, u32, element, ConstantU32(state, ImageTableAbi::SamplerEntryElements));
        const auto inBounds = Binary(state, spv::OpULessThanEqual, boolean, end, LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderSamplerCount)));
        const auto outside = both(both(selected, negate(rejected)), negate(inBounds));
        EmitTableFault(ctx, inst, word, ConstantU32(state, ImageTableAbi::OutsideSnapshotPoison), both(both(exec, live), outside));
        const auto checked = both(both(selected, negate(rejected)), inBounds);
        const auto entry = Binary(state, spv::OpShiftRightLogical, u32, element, ConstantU32(state, 2u));
        const auto driverFault = LoadTableMap(state, Select(state, u32, checked, Binary(state, spv::OpIAdd, u32, LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderSamplerFaults)), entry), ConstantU32(state, 0u)));
        const auto driverRejected = both(checked, flagged(driverFault));
        EmitTableFault(ctx, inst, word, driverFault, both(both(exec, live), driverRejected));
        result.valid = both(result.valid, both(checked, negate(driverRejected)));
        state.tableAccess.samplerElement = Binary(state, spv::OpIAdd, u32, LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderSamplerSlots)), element);
        DecorateTableNonUniform(state, state.tableAccess.samplerElement);
    }
    state.tableAccess.active = true;
    return result;
}

}

void EmitImageOperation(SpirvModule& module, const IrValue& value, std::uint32_t resultId) {
    throw std::logic_error("EmitImageOperation(SpirvModule&) is not used: image instructions are emitted per opcode through EmitImage");
}

void EmitImageOperation(SpirvValueEmitContext& context, const IrValue& value, std::uint32_t resultId) {
    throw std::logic_error("EmitImageOperation(SpirvValueEmitContext&) is not used: image instructions are emitted per opcode through EmitImage");
}

void EmitImageMode(SpirvValueEmitContext& ctx, const IrValue& inst, const ImageResource& image) {
    const auto irOpcode = inst.Opcode();
    const auto imageInfo = ImageOpcodeInfoOf(irOpcode);
    const auto& mem = ctx.Memory(inst);
    ctx.ResourceIndex(inst.Argument(0), IrOpcode::GetImageResource);
    const auto* address = ctx.ImageAddress(inst.Argument(imageInfo.needsSampler ? 2 : 1));
    if (address == nullptr) {
        ctx.Fail(inst, "has no image address");
    }
    const ImageEmitAccess access{inst, mem, image, *address};
    if (imageInfo.access == ImageAccess::Atomic) {
        EmitAtomicOp(ctx, access);
        return;
    }
    switch (irOpcode) {
        case IrOpcode::ImageQueryDimensions:
            EmitQueryDimensionsOp(ctx, access);
            return;
        case IrOpcode::ImageQueryLod:
            EmitQueryLodOp(ctx, access);
            return;
        case IrOpcode::ImageRead:
            EmitReadOp(ctx, access);
            return;
        case IrOpcode::ImageWrite:
            EmitWriteOp(ctx, access);
            return;
        case IrOpcode::ImageSampleRaw:
        case IrOpcode::ImageGatherRaw:
            EmitSamplingOp(ctx, access);
            return;
        default:
            ctx.Fail(inst, "has no image SPIR-V emitter");
    }
}

namespace {

void EmitModeCase(SpirvValueEmitContext& ctx, const IrValue& inst, const MemoryInfo& memory, const ImageResource& mode) {
    auto& state = ctx.state;
    state.runtimeImage = &mode;
    if (mode.packedFormat == IrBufferFormat::Fmask8_S2_F1) {
        if (inst.Opcode() == IrOpcode::ImageRead && memory.dataBits == 32u) {
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), value, ConstantU32(state, 0x76543210u), ConstantU32(state, 0xfedcba98u), ConstantU32(state, 0u), ConstantU32(state, 0u));
            const auto active = ctx.Arg(inst, 2u);
            const auto mapped = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, TypeBoolVector(state, 4u), mapped, active, active, active, active);
            ctx.Define(inst, Select(state, TypeU32Vector(state, 4u), mapped, value, ConstantU32CompositeZero(state, 4u)));
        } else {
            ctx.Fail(inst, "FMASK requires a 32-bit image read");
        }
    } else if (mode.depthBits && memory.dataBits == 16u) {
        ctx.Fail(inst, "depth bits require a 32-bit image result");
    } else {
        EmitImageMode(ctx, inst, mode);
    }
}

std::uint32_t EmitModeSwitch(SpirvValueEmitContext& ctx, const IrValue& inst, const MemoryInfo& memory, const std::vector<ImageResource>& modes, std::uint32_t selector, std::uint32_t resultId) {
    auto& state = ctx.state;
    const bool returnsValue = inst.Opcode() != IrOpcode::ImageWrite;
    const auto merge = state.module.AllocateId();
    const auto invalid = state.module.AllocateId();
    std::vector<std::uint32_t> labels(modes.size());
    std::vector<std::uint32_t> words{spv::OpSwitch, selector, invalid};
    for (std::uint32_t index = 0u; index < modes.size(); ++index) {
        if (!TableModeEmitted(modes[index])) continue;
        labels[index] = state.module.AllocateId();
        words.insert(words.end(), {index, labels[index]});
    }
    std::vector<std::uint32_t> incoming;
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(words);
    for (std::uint32_t index = 0u; index < modes.size(); ++index) {
        if (labels[index] == 0u) continue;
        EmitLabel(state, labels[index]);
        EmitModeCase(ctx, inst, memory, modes[index]);
        if (returnsValue) incoming.insert(incoming.end(), {ctx.Def(&inst), state.currentLabel});
        ctx.definitions.erase(&inst);
        state.module.AddFunction(spv::OpBranch, merge);
    }
    state.runtimeImage = nullptr;
    EmitLabel(state, invalid);
    if (returnsValue) incoming.insert(incoming.end(), {ZeroResult(ctx, inst), invalid});
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    if (returnsValue) {
        std::vector<std::uint32_t> phi{spv::OpPhi, TypeId(state, inst.Type()), resultId};
        phi.insert(phi.end(), incoming.begin(), incoming.end());
        state.module.AddFunction(phi);
    }
    return merge;
}

constexpr std::size_t InlineModeSwitchArms = 512u;

bool InlineModeSwitchesFit(const SpirvEmitterState& state) {
    const auto& infos = state.program.Resources().memoryInfo;
    const auto& images = state.program.Resources().info.images;
    std::size_t arms = 0;
    for (const IrBlock* block : state.program.BlockOrder()) {
        for (const IrValue* other : block->Instructions()) {
            if (ImageOpcodeInfoOf(other->Opcode()).access == ImageAccess::None) continue;
            const auto index = other->Flags<MemoryFlags>().index;
            if (index >= infos.size() || infos[index].kind != ResourceKind::Image || infos[index].resource >= images.size()) continue;
            const auto& image = images[infos[index].resource];
            if (image.table == NoTable && !image.deviceTable) continue;
            arms += image.table != NoTable ? state.program.Info().runtimeImageModes.at(infos[index].resource).size() : ResourceMaterializer::RuntimeImageModes(image).size();
            if (arms > InlineModeSwitchArms) return false;
        }
    }
    return true;
}

bool SingleTableImageUse(const SpirvEmitterState& state, const IrValue& inst, const MemoryInfo& memory) {
    const auto& infos = state.program.Resources().memoryInfo;
    std::size_t uses = 0;
    for (const IrBlock* block : state.program.BlockOrder()) {
        for (const IrValue* other : block->Instructions()) {
            if (other->Opcode() != inst.Opcode()) continue;
            const auto index = other->Flags<MemoryFlags>().index;
            if (index < infos.size() && infos[index] == memory && ++uses > 1u) return false;
        }
    }
    return uses == 1u;
}

void EmitTableImageCall(SpirvValueEmitContext& ctx, const IrValue& inst, const MemoryInfo& memory, const std::vector<ImageResource>& modes, std::uint32_t selector, bool tableSampler, bool atAccess) {
    auto& state = ctx.state;
    const bool returnsValue = inst.Opcode() != IrOpcode::ImageWrite;
    if (state.laneCount == 1u && !state.module.InHelperFunction() && (atAccess || SingleTableImageUse(state, inst, memory)) && InlineModeSwitchesFit(state)) {
        const auto result = returnsValue ? state.module.AllocateId() : 0u;
        EmitModeSwitch(ctx, inst, memory, modes, selector, result);
        if (returnsValue) ctx.Define(inst, result);
        return;
    }
    const auto resultType = returnsValue ? TypeId(state, inst.Type()) : TypeVoid(state);
    const std::size_t addressIndex = ImageOpcodeInfoOf(inst.Opcode()).needsSampler ? 2u : 1u;
    const auto* address = ctx.ImageAddress(inst.Argument(addressIndex));
    std::vector<const IrValue*> slots(address->Arguments().begin(), address->Arguments().end());
    for (std::size_t index = addressIndex + 1u; index < inst.ArgumentCount(); ++index) slots.push_back(inst.Argument(index));
    std::vector<std::uint32_t> operands{static_cast<std::uint32_t>(inst.Type())};
    std::vector<const IrValue*> values;
    for (const auto* slot : slots) {
        const auto* value = slot->Resolve();
        if (value->HasImmediate()) {
            operands.insert(operands.end(), {0u, ctx.Def(value)});
            continue;
        }
        const auto found = std::find(values.begin(), values.end(), value);
        operands.insert(operands.end(), {1u, static_cast<std::uint32_t>(found - values.begin())});
        if (found == values.end()) values.push_back(value);
    }
    const std::array<std::uint32_t, 6> access{selector, state.tableAccess.element, state.tableAccess.dword3, state.tableAccess.samplerElement, state.tableAccess.unnormalized, state.tableAccess.compare};
    auto function = std::find_if(state.tableImageFunctions.begin(), state.tableImageFunctions.end(), [&](const SpirvTableImageFunction& candidate) {
        return candidate.opcode == inst.Opcode() && candidate.memory == memory && candidate.operands == operands;
    });
    if (function == state.tableImageFunctions.end()) {
        struct Restore {
            SpirvEmitterState& state;
            std::uint32_t label;
            SpirvTableAccess tableAccess;
            ~Restore() {
                state.module.EndHelperFunction();
                state.currentLabel = label;
                state.tableAccess = tableAccess;
                state.runtimeImage = nullptr;
            }
        };
        const auto id = state.module.AllocateId();
        state.module.BeginHelperFunction();
        const Restore restore{state, state.currentLabel, state.tableAccess};
        const auto start = state.module.FunctionWordCount();
        const auto annotationStart = state.module.AnnotationWordCount();
        std::array<std::uint32_t, 6> parameters{};
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            if ((index != 3u || tableSampler) && (index != 4u || access[4] != 0u) && (index != 5u || access[5] != 0u)) parameters[index] = state.module.AllocateId();
        }
        EmitLabel(state, state.module.AllocateId());
        state.tableAccess = {parameters[1], parameters[2], parameters[3], true, parameters[4], parameters[5]};
        SpirvValueEmitContext body(state);
        const auto result = returnsValue ? state.module.AllocateId() : 0u;
        EmitModeSwitch(body, inst, memory, modes, parameters[0], result);
        if (returnsValue) state.module.AddFunction(spv::OpReturnValue, result);
        else state.module.AddFunction(spv::OpReturn);
        const auto words = state.module.TakeFunctionWords(start);
        const auto referenced = [&](std::uint32_t value) {
            for (std::size_t cursor = 0; cursor < words.size(); cursor += words[cursor] >> spv::WordCountShift) {
                for (std::size_t word = cursor + 1u; word < cursor + (words[cursor] >> spv::WordCountShift); ++word) {
                    if (words[word] == value) return true;
                }
            }
            return false;
        };
        SpirvTableImageFunction defined{inst.Opcode(), memory, operands, id, {}};
        std::vector<std::uint32_t> types{resultType};
        std::vector<std::uint32_t> ids;
        for (std::uint32_t index = 0; index < parameters.size(); ++index) {
            if (parameters[index] == 0u || !referenced(parameters[index])) continue;
            defined.arguments.push_back(index);
            types.push_back(TypeU32(state));
            ids.push_back(parameters[index]);
        }
        std::size_t passed = 0;
        for (std::uint32_t index = 0; index < values.size(); ++index) {
            const auto found = body.definitions.find(values[index]);
            if (found == body.definitions.end()) continue;
            ++passed;
            defined.arguments.push_back(static_cast<std::uint32_t>(access.size()) + index);
            types.push_back(TypeId(state, values[index]->Type()));
            ids.push_back(found->second);
            if (types.back() == 0u) ctx.Fail(inst, "passes an image table operand of a type without a SPIR-V type");
        }
        if (passed != body.definitions.size()) ctx.Fail(inst, "reads an operand its image table function does not take");
        const auto appended = state.module.TakeAnnotationWords(annotationStart);
        const auto locals = SpirvFunctionLocalIds(ids, words);
        defined.body = {resultType, ids, std::vector<std::uint32_t>(types.begin() + 1, types.end()), words, {}};
        std::vector<std::uint32_t> shared;
        for (std::size_t cursor = 0; cursor < appended.size();) {
            const auto count = appended[cursor] >> spv::WordCountShift;
            if (count < 2u || count > appended.size() - cursor) ctx.Fail(inst, "emitted a malformed annotation in its image table function");
            auto& section = locals.contains(appended[cursor + 1u]) ? defined.body.annotations : shared;
            section.insert(section.end(), appended.begin() + static_cast<std::ptrdiff_t>(cursor), appended.begin() + static_cast<std::ptrdiff_t>(cursor + count));
            cursor += count;
        }
        state.module.AppendAnnotationWords(shared);
        const auto same = std::find_if(state.tableImageFunctions.begin(), state.tableImageFunctions.end(), [&](const SpirvTableImageFunction& candidate) {
            return candidate.arguments == defined.arguments && !candidate.body.words.empty() && SameSpirvFunctionBody(candidate.body, defined.body, TypeU32(state));
        });
        if (same != state.tableImageFunctions.end()) {
            defined.function = same->function;
            defined.body = {};
        } else {
            state.module.AppendAnnotationWords(defined.body.annotations);
            state.module.AddName(id, "table_image_" + std::to_string(memory.resource));
            state.module.AddFunction(spv::OpFunction, resultType, id, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, std::span<const std::uint32_t>(types)));
            for (std::size_t index = 0; index < ids.size(); ++index) state.module.AddFunction(spv::OpFunctionParameter, types[index + 1u], ids[index]);
            state.module.AppendFunctionWords(words);
            state.module.AddFunction(spv::OpFunctionEnd);
            for (const auto index : {1u, 3u}) {
                if (std::find(defined.arguments.begin(), defined.arguments.end(), index) != defined.arguments.end()) DecorateTableNonUniform(state, parameters[index]);
            }
        }
        function = state.tableImageFunctions.insert(state.tableImageFunctions.end(), std::move(defined));
    }
    std::vector<std::uint32_t> call{spv::OpFunctionCall, resultType, returnsValue ? ctx.Result(inst) : state.module.AllocateId(), function->function};
    for (const auto index : function->arguments) call.push_back(index < access.size() ? access[index] : ctx.Def(values[index - access.size()]));
    state.module.AddFunction(call);
}

std::uint32_t DeviceAddress(SpirvEmitterState& state, std::uint32_t low, std::uint32_t high) {
    const auto u64 = TypeScalarU64(state);
    return Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, low), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, high), BdaConstant(state, 32u)));
}

std::uint32_t DeviceWordPointer(SpirvEmitterState& state, std::uint32_t base, std::uint32_t index) {
    const auto u64 = TypeScalarU64(state);
    const auto offset = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, index), BdaConstant(state, 2u));
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, Binary(state, spv::OpIAdd, u64, base, offset));
    return pointer;
}

std::uint32_t LoadDeviceWord(SpirvEmitterState& state, std::uint32_t base, std::uint32_t index) {
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, DeviceWordPointer(state, base, index), spv::MemoryAccessAlignedMask, 4u);
    return value;
}

void StoreDeviceWord(SpirvEmitterState& state, std::uint32_t base, std::uint32_t index, std::uint32_t value) {
    state.module.AddFunction(spv::OpStore, DeviceWordPointer(state, base, index), value, spv::MemoryAccessAlignedMask, 4u);
}

std::uint32_t RotateLeftU32(SpirvEmitterState& state, std::uint32_t value, std::uint32_t bits) {
    const auto u32 = TypeU32(state);
    return Binary(state, spv::OpBitwiseOr, u32, Binary(state, spv::OpShiftLeftLogical, u32, value, ConstantU32(state, bits)), Binary(state, spv::OpShiftRightLogical, u32, value, ConstantU32(state, 32u - bits)));
}

std::uint32_t DeviceKeyHash(SpirvEmitterState& state, std::span<const std::uint32_t> key) {
    namespace Abi = DeviceImageTableAbi;
    const auto u32 = TypeU32(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    auto hash = constant(Abi::HashSeed);
    for (const auto word : key) {
        const auto mixed = Binary(state, spv::OpIMul, u32, RotateLeftU32(state, Binary(state, spv::OpIMul, u32, word, constant(Abi::HashC1)), 15u), constant(Abi::HashC2));
        hash = Binary(state, spv::OpIAdd, u32, Binary(state, spv::OpIMul, u32, RotateLeftU32(state, Binary(state, spv::OpBitwiseXor, u32, hash, mixed), 13u), constant(5u)), constant(Abi::HashRound));
    }
    const auto shiftXor = [&](std::uint32_t value, std::uint32_t bits) { return Binary(state, spv::OpBitwiseXor, u32, value, Binary(state, spv::OpShiftRightLogical, u32, value, constant(bits))); };
    hash = Binary(state, spv::OpBitwiseXor, u32, hash, constant(Abi::KeyWords * 4u));
    hash = Binary(state, spv::OpIMul, u32, shiftXor(hash, 16u), constant(Abi::HashMix1));
    hash = Binary(state, spv::OpIMul, u32, shiftXor(hash, 13u), constant(Abi::HashMix2));
    return shiftXor(hash, 16u);
}

std::uint32_t CompositeU32x4(SpirvEmitterState& state, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t w) {
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), value, x, y, z, w);
    return value;
}

struct DeviceImagePlan {
    const MemoryInfo* memory = nullptr;
    const ImageResource* base = nullptr;
    std::vector<ImageResource> modes;
    const IrValue* handle = nullptr;
    const IrValue* samplerHandle = nullptr;
    bool needsSampler = false;
    bool deviceSampler = false;
    bool levelZero = false;
    std::uint64_t id = 0;
};

std::optional<DeviceImagePlan> PlanDeviceImage(const SpirvEmitterState& state, const IrValue& inst) {
    namespace Abi = DeviceImageTableAbi;
    const auto info = ImageOpcodeInfoOf(inst.Opcode());
    if (info.access == ImageAccess::None || inst.ArgumentCount() == 0u) return std::nullopt;
    const auto& infos = state.program.Resources().memoryInfo;
    const auto index = inst.Flags<MemoryFlags>().index;
    if (index >= infos.size()) return std::nullopt;
    DeviceImagePlan plan;
    plan.memory = &infos[index];
    plan.base = &ImageResourceOf(state, *plan.memory);
    if (!plan.base->deviceTable) return std::nullopt;
    plan.modes = state.program.Info().runtimeImageModes.at(plan.memory->resource);
    if (plan.base->tableOperation == TableOperation::Unsupported || plan.modes.empty()) return std::nullopt;
    plan.handle = inst.Argument(0)->Resolve();
    if (plan.handle == nullptr || plan.handle->ArgumentCount() != Abi::DescriptorWords) return std::nullopt;
    plan.needsSampler = info.needsSampler;
    if (info.needsSampler) {
        const auto& samplers = state.program.Info().samplers;
        if (plan.memory->sampler >= samplers.size() || samplers[plan.memory->sampler].table != NoTable) return std::nullopt;
        if (samplers[plan.memory->sampler].deviceTable) {
            plan.samplerHandle = inst.ArgumentCount() > 1u ? inst.Argument(1)->Resolve() : nullptr;
            if (plan.samplerHandle == nullptr || plan.samplerHandle->ArgumentCount() != Abi::SamplerWords) return std::nullopt;
            plan.deviceSampler = true;
            plan.levelZero = HasFlag(*plan.memory, RdnaImageSampleFlagLevelZero);
        }
    }
    plan.id = ResourceMaterializer::DeviceImageInterfaceId(*plan.base, plan.modes);
    return plan;
}

std::vector<std::uint64_t> DeviceImageLookupKey(const SpirvEmitterState& state, const DeviceImagePlan& plan, const IrBlock* block) {
    std::vector<std::uint64_t> key{reinterpret_cast<std::uintptr_t>(block), state.laneHalf, plan.id, plan.needsSampler ? 1u : 0u, plan.needsSampler ? plan.memory->sampler : 0u, plan.levelZero ? 1u : 0u, plan.deviceSampler ? 1u : 0u};
    const auto word = [&](const IrValue* value) {
        const auto* resolved = value->Resolve();
        key.push_back(resolved->HasImmediate() ? 0u : reinterpret_cast<std::uintptr_t>(resolved));
        key.push_back(resolved->HasImmediate() ? resolved->ImmediateU32() : 0u);
    };
    for (const auto* argument : plan.handle->Arguments()) word(argument);
    if (plan.deviceSampler) {
        for (const auto* argument : plan.samplerHandle->Arguments()) word(argument);
    }
    return key;
}

struct DeviceImageArguments {
    std::array<std::uint32_t, DeviceImageTableAbi::DescriptorWords> words{};
    std::array<std::uint32_t, DeviceImageTableAbi::SamplerWords> samplerWords{};
    std::uint32_t header = 0;
    std::uint32_t samplerClass = 0;
    std::uint32_t unnormalized = 0;
};

void CompleteDeviceImageArguments(SpirvEmitterState& state, const DeviceImagePlan& plan, DeviceImageArguments& arguments) {
    namespace Abi = DeviceImageTableAbi;
    const auto u32 = TypeU32(state);
    const auto zero = ConstantU32(state, 0u);
    arguments.header = DeviceAddress(state, LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderDeviceLow)), LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderDeviceHigh)));
    arguments.samplerClass = zero;
    if (plan.deviceSampler) {
        arguments.samplerClass = ConstantU32(state, Abi::RuntimeSampler);
        if (plan.levelZero) arguments.samplerClass = Select(state, u32, Binary(state, spv::OpINotEqual, TypeBool(state), arguments.unnormalized, zero), ConstantU32(state, Abi::RuntimeSampler | Abi::SamplerExactAtBase), arguments.samplerClass);
    } else if (plan.needsSampler) {
        arguments.samplerClass = LoadTableMap(state, Binary(state, spv::OpIAdd, u32, LoadTableMap(state, ConstantU32(state, ImageTableAbi::HeaderSamplerClasses)), ConstantU32(state, plan.memory->sampler)));
    }
}

DeviceImageArguments EmitDeviceImageArguments(SpirvValueEmitContext& ctx, const DeviceImagePlan& plan, bool complete) {
    namespace Abi = DeviceImageTableAbi;
    auto& state = ctx.state;
    const auto u32 = TypeU32(state);
    const auto zero = ConstantU32(state, 0u);
    DeviceImageArguments arguments;
    for (std::uint32_t word = 0; word < arguments.words.size(); ++word) arguments.words[word] = ctx.Def(plan.handle->Argument(word));
    arguments.samplerWords.fill(zero);
    if (plan.deviceSampler) {
        for (std::uint32_t word = 0; word < arguments.samplerWords.size(); ++word) arguments.samplerWords[word] = ctx.Def(plan.samplerHandle->Argument(word));
        arguments.unnormalized = Binary(state, spv::OpBitwiseAnd, u32, arguments.samplerWords[0], ConstantU32(state, Abi::SamplerUnnormalized));
    }
    if (complete) CompleteDeviceImageArguments(state, plan, arguments);
    return arguments;
}

std::array<std::uint32_t, 4> DeviceImageLookupOperands(SpirvEmitterState& state, const DeviceImagePlan& plan, const DeviceImageArguments& arguments, std::uint32_t pc) {
    const auto& words = arguments.words;
    const auto& samplerWords = arguments.samplerWords;
    return std::array{CompositeU32x4(state, words[0], words[1], words[2], words[3]), CompositeU32x4(state, words[4], words[5], words[6], words[7]),
        CompositeU32x4(state, ConstantU32(state, static_cast<std::uint32_t>(plan.id)), ConstantU32(state, static_cast<std::uint32_t>(plan.id >> 32u)), arguments.samplerClass, pc),
        CompositeU32x4(state, samplerWords[0], samplerWords[1], samplerWords[2], samplerWords[3])};
}

std::vector<const IrValue*> DeviceImageWords(const DeviceImagePlan& plan) {
    std::vector<const IrValue*> words(plan.handle->Arguments().begin(), plan.handle->Arguments().end());
    if (plan.deviceSampler) words.insert(words.end(), plan.samplerHandle->Arguments().begin(), plan.samplerHandle->Arguments().end());
    return words;
}

std::vector<std::vector<const IrValue*>> DescriptorInputGroups(const SpirvEmitterState& state, const IrValue& inst) {
    const auto plan = PlanDeviceImage(state, inst);
    if (plan) return {DeviceImageWords(*plan)};
    std::vector<std::vector<const IrValue*>> groups;
    for (auto inputs : {InvalidFormatInputs(state, inst), ResolvedBufferInputs(state, inst)}) {
        if (!inputs.empty()) groups.push_back(std::move(inputs));
    }
    return groups;
}

bool DeviceImageInputsDefined(const SpirvValueEmitContext& ctx, const DeviceImagePlan& plan) {
    const auto defined = [&](const IrValue* value) {
        const auto* resolved = value->Resolve();
        return resolved->HasImmediate() || ctx.definitions.contains(resolved);
    };
    if (!std::ranges::all_of(plan.handle->Arguments(), defined)) return false;
    return !plan.deviceSampler || std::ranges::all_of(plan.samplerHandle->Arguments(), defined);
}

void EmitDeviceTableImage(SpirvValueEmitContext& ctx, const IrValue& inst, const MemoryInfo& memory, const ImageResource& base, const std::vector<ImageResource>& modes, bool tableSampler) {
    namespace Abi = DeviceImageTableAbi;
    auto& state = ctx.state;
    if (state.tableMapVariable == 0 || state.faultBufferVariable == 0 || state.deviceImageLookupFunction == 0) ctx.Fail(inst, "reads a device-table image without its map, fault buffer or lookup function");
    if (tableSampler) ctx.Fail(inst, "samples a device-table image through a sampler table");
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto zero = ConstantU32(state, 0u);
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto equal = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpIEqual, boolean, left, right); };
    const auto exec = ctx.Arg(inst, inst.ArgumentCount() - 1u);
    const auto pc = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    if (base.tableOperation == TableOperation::Unsupported || modes.empty()) {
        const auto fault = LoadTableMap(state, ConstantU32(state, ImageTableAbi::TableHeader(base.table) + ImageTableAbi::TableFault));
        EmitIfCondition(state, both(exec, Binary(state, spv::OpINotEqual, boolean, fault, zero)), [&] {
            RecordBdaFaultWords(state, zero, zero, Binary(state, spv::OpBitwiseAnd, u32, fault, ConstantU32(state, ~ImageTableAbi::PoisonFlag)), pc, BdaAbi::FaultReason::ImageTable);
        });
        if (inst.Type() != IrType::Void) ctx.Define(inst, ZeroResult(ctx, inst));
        return;
    }
    if (inst.Argument(0)->Resolve()->ArgumentCount() != Abi::DescriptorWords) ctx.Fail(inst, "has a device-table image handle without its eight descriptor dwords");
    const auto plan = PlanDeviceImage(state, inst);
    if (!plan) ctx.Fail(inst, "has a device-table sampler handle without its four descriptor dwords");
    auto arguments = EmitDeviceImageArguments(ctx, *plan, false);
    const auto& words = arguments.words;
    const auto null = both(equal(words[0], zero), equal(Binary(state, spv::OpBitwiseAnd, u32, words[1], ConstantU32(state, 0xffu)), zero));
    const auto record = both(exec, Unary(state, spv::OpLogicalNot, boolean, null));
    const auto missed = [&](std::uint32_t lookup) {
        const auto missing = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, missing, lookup, 3u);
        EmitIfCondition(state, both(record, equal(missing, zero)), [&] {
            CompleteDeviceImageArguments(state, *plan, arguments);
            const auto passed = DeviceImageLookupOperands(state, *plan, arguments, pc);
            state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.deviceImageMissFunction, arguments.header, passed[0], passed[1], passed[2], passed[3]);
        });
    };
    std::uint32_t result = 0;
    const auto anchored = state.deviceImageLookups.find(DeviceImageLookupKey(state, *plan, nullptr));
    auto blockKey = DeviceImageLookupKey(state, *plan, inst.Parent());
    const auto earlier = state.module.InHelperFunction() ? state.deviceImageLookups.end() : state.deviceImageLookups.find(blockKey);
    if (anchored != state.deviceImageLookups.end()) {
        result = anchored->second;
        missed(result);
    } else if (earlier != state.deviceImageLookups.end()) {
        result = earlier->second;
        missed(result);
    } else {
        CompleteDeviceImageArguments(state, *plan, arguments);
        const auto passed = DeviceImageLookupOperands(state, *plan, arguments, pc);
        result = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 4u), result, state.deviceImageLookupFunction, arguments.header, passed[0], passed[1], passed[2], passed[3], record);
        if (!state.module.InHelperFunction()) state.deviceImageLookups.emplace(std::move(blockKey), result);
    }
    const auto extract = [&](std::uint32_t index) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, value, result, index);
        return value;
    };
    const auto slot = extract(0u);
    const auto mode = extract(1u);
    const auto fault = extract(2u);
    const auto found = extract(3u);
    const auto hit = Binary(state, spv::OpINotEqual, boolean, found, zero);
    const auto faulted = Binary(state, spv::OpINotEqual, boolean, fault, zero);
    EmitIfCondition(state, both(both(record, hit), faulted), [&] {
        RecordBdaFaultWords(state, zero, zero, Binary(state, spv::OpBitwiseOr, u32, ConstantU32(state, Abi::FaultFlag), Binary(state, spv::OpISub, u32, fault, ConstantU32(state, 1u))), pc, BdaAbi::FaultReason::ImageTable);
    });
    const auto valid = both(both(Unary(state, spv::OpLogicalNot, boolean, null), hit), Unary(state, spv::OpLogicalNot, boolean, faulted));
    const auto viewAccess = [&] {
        state.tableAccess = {};
        state.tableAccess.element = Select(state, u32, valid, slot, zero);
        DecorateTableNonUniform(state, state.tableAccess.element);
        state.tableAccess.dword3 = words[3];
        if (plan->deviceSampler) {
            state.tableAccess.samplerElement = Select(state, u32, valid, Binary(state, spv::OpISub, u32, found, ConstantU32(state, 1u)), zero);
            DecorateTableNonUniform(state, state.tableAccess.samplerElement);
            state.tableAccess.unnormalized = arguments.unnormalized;
        }
        state.tableAccess.active = true;
        auto selector = mode;
        if (std::ranges::any_of(modes, [](const ImageResource& candidate) { return (candidate.emulatedCompare & EmulatedCompare::Enabled) != 0u; })) {
            state.tableAccess.compare = Binary(state, spv::OpShiftRightLogical, u32, mode, ConstantU32(state, Abi::ModeCompareShift));
            selector = Binary(state, spv::OpBitwiseAnd, u32, mode, ConstantU32(state, (1u << Abi::ModeCompareShift) - 1u));
        }
        EmitTableImageCall(ctx, inst, memory, modes, Select(state, u32, valid, selector, ConstantU32(state, InvalidTableSelector)), plan->deviceSampler, true);
        state.tableAccess = {};
    };
    if (!StorageImageMemory::RuntimeResourceThroughMemory(state.program.Info(), base)) {
        viewAccess();
        return;
    }
    const auto* functions = StorageImageMemoryFunctionsOf(state);
    if (functions == nullptr || functions->texel == 0u) ctx.Fail(inst, "accesses a device-table image through memory without the texel helper");
    const auto* address = ctx.ImageAddress(inst.Argument(ImageOpcodeInfoOf(inst.Opcode()).needsSampler ? 2u : 1u));
    if (address == nullptr) ctx.Fail(inst, "has no image address");
    const ImageEmitAccess access{inst, memory, base, *address};
    const auto selected = both(valid, equal(slot, ConstantU32(state, StorageImageMemory::MemorySlot)));
    if (inst.Opcode() == IrOpcode::ImageWrite) {
        EmitIfCondition(state, selected, [&] { EmitMemoryWrite(ctx, access, *functions); });
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, selected), viewAccess);
        return;
    }
    const bool atomic = ImageOpcodeInfoOf(inst.Opcode()).access == ImageAccess::Atomic;
    if (!atomic && inst.Opcode() != IrOpcode::ImageRead) {
        viewAccess();
        return;
    }
    ctx.definitions.erase(&inst);
    const auto value = EmitValueIfElse(state, selected, TypeId(state, inst.Type()), [&] {
        return atomic ? EmitMemoryAtomic(ctx, access, *functions) : EmitMemoryRead(ctx, access, *functions, exec);
    }, [&] {
        viewAccess();
        const auto defined = ctx.definitions.find(&inst);
        if (defined == ctx.definitions.end()) ctx.Fail(inst, "left its device-table image access without a result");
        const auto result = defined->second;
        ctx.definitions.erase(defined);
        return result;
    });
    ctx.definitions.emplace(&inst, value);
}

}

namespace {

void EmitImageStoreRecord(SpirvValueEmitContext& ctx, const IrValue& inst, const MemoryInfo& memory, const ImageResource& base) {
    auto& state = ctx.state;
    const auto info = ImageOpcodeInfoOf(inst.Opcode());
    if (state.imageStoreRecordFunction == 0u || (inst.Opcode() != IrOpcode::ImageWrite && info.access != ImageAccess::Atomic)) return;
    auto active = ctx.Arg(inst, inst.ArgumentCount() - 1u);
    if (StorageImageMemory::ResourceThroughMemory(state.program.Info(), base)) active = Binary(state, spv::OpLogicalAnd, TypeBool(state), active, Unary(state, spv::OpLogicalNot, TypeBool(state), MemorySlotSelected(state, memory.resource)));
    const auto* handle = inst.Argument(0)->Resolve();
    const auto* address = ctx.ImageAddress(inst.Argument(info.needsSampler ? 2u : 1u));
    const auto dimension = memory.imageDimension;
    const bool placed = handle->ArgumentCount() == 8u && address != nullptr && (dimension == RdnaImageDimension::Dim1D || dimension == RdnaImageDimension::Dim1DArray || dimension == RdnaImageDimension::Dim2D || dimension == RdnaImageDimension::Dim2DArray);
    if (!placed) {
        EmitIfCondition(state, active, [&] {
            state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.writtenChunkNoteFunction, ConstantU32(state, BdaAbi::WrittenImageWholeKey), ConstantU32(state, 1u));
        });
        return;
    }
    const ImageEmitAccess access{inst, memory, base, *address};
    const auto zero = ConstantU32(state, 0u);
    const auto components = RdnaImageDimensionInfoFor(dimension).coordinateComponents;
    const auto x = AddressU32(ctx, access, 0u);
    const bool rows = dimension == RdnaImageDimension::Dim2D || dimension == RdnaImageDimension::Dim2DArray;
    const auto y = rows ? AddressU32(ctx, access, 1u) : zero;
    const auto layer = dimension == RdnaImageDimension::Dim1DArray ? AddressU32(ctx, access, 1u) : dimension == RdnaImageDimension::Dim2DArray ? AddressU32(ctx, access, 2u) : zero;
    const auto lod = memory.imageHasMip ? AddressU32(ctx, access, components) : zero;
    std::array<std::uint32_t, 6> words{};
    for (std::uint32_t word = 0; word < words.size(); ++word) words[word] = ctx.Def(handle->Argument(word));
    EmitIfCondition(state, active, [&] {
        state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.imageStoreRecordFunction, words[0], words[1], words[2], words[3], words[4], words[5], x, y, layer, lod);
    });
}

}

void PlanDeviceImageAnchors(SpirvEmitterState& state) {
    state.deviceImageAnchors.clear();
    std::unordered_map<const IrValue*, std::pair<std::size_t, std::size_t>> positions;
    const auto& blocks = state.program.BlockOrder();
    for (std::size_t block = 0; block < blocks.size(); ++block) {
        std::size_t index = 0;
        for (const IrValue* inst : blocks[block]->Instructions()) positions.emplace(inst, std::pair{block, index++});
    }
    for (const IrBlock* block : blocks) {
        for (const IrValue* inst : block->Instructions()) {
            for (const auto& inputs : DescriptorInputGroups(state, *inst)) {
                const IrValue* anchor = nullptr;
                std::pair<std::size_t, std::size_t> latest{};
                bool placed = true;
                const auto consider = [&](const IrValue* argument) {
                    const auto* resolved = argument->Resolve();
                    if (resolved->HasImmediate()) return;
                    const auto found = positions.find(resolved);
                    if (found == positions.end()) {
                        placed = false;
                        return;
                    }
                    if (anchor == nullptr || found->second > latest) {
                        anchor = resolved;
                        latest = found->second;
                    }
                };
                for (const auto* argument : inputs) consider(argument);
                if (!placed || anchor == nullptr) continue;
                auto& anchored = state.deviceImageAnchors[anchor];
                if (std::ranges::find(anchored, inst) == anchored.end()) anchored.push_back(inst);
            }
        }
    }
    if (!state.deviceImageAnchors.empty() && state.program.Resources().stage == IrShaderStage::Compute && state.sharedLaneValues.empty()) state.sharedLaneValues = WaveUniformValues(state.program);
}

void EmitAnchoredDeviceImageLookups(SpirvValueEmitContext& ctx, const IrValue& anchor) {
    auto& state = ctx.state;
    const auto anchored = state.deviceImageAnchors.find(&anchor);
    if (anchored == state.deviceImageAnchors.end()) return;
    for (const IrValue* inst : anchored->second) {
        const auto plan = PlanDeviceImage(state, *inst);
        if (!plan) {
            EmitInvalidFormatStop(ctx, *inst, anchor);
            EmitResolvedBufferSegment(ctx, *inst, anchor);
            continue;
        }
        if (state.deviceImageLookupFunction == 0u || state.tableMapVariable == 0u || !DeviceImageInputsDefined(ctx, *plan)) continue;
        auto key = DeviceImageLookupKey(state, *plan, nullptr);
        if (state.deviceImageLookups.contains(key)) continue;
        const auto arguments = EmitDeviceImageArguments(ctx, *plan, true);
        const auto passed = DeviceImageLookupOperands(state, *plan, arguments, ConstantU32(state, inst->Flags<MemoryFlags>().pc));
        auto result = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 4u), result, state.deviceImageLookupFunction, arguments.header, passed[0], passed[1], passed[2], passed[3], ConstantBool(state, false));
        if (WaveUniformDescriptor(state, DeviceImageWords(*plan))) result = BroadcastFirstLane(state, TypeU32Vector(state, 4u), result);
        state.deviceImageLookups.emplace(std::move(key), result);
    }
}

void DefineDeviceImageLookup(SpirvEmitterState& state) {
    namespace Abi = DeviceImageTableAbi;
    const auto& images = state.program.Info().images;
    if (std::none_of(images.begin(), images.end(), [](const ImageResource& image) { return image.deviceTable; })) return;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto uvec4 = TypeU32Vector(state, 4u);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto extract = [&](std::uint32_t vector, std::uint32_t index) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, value, vector, index);
        return value;
    };
    const auto load = [&](std::uint32_t variable, std::uint32_t type) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, variable);
        return value;
    };
    const auto appendMiss = [&](std::uint32_t header, const std::array<std::uint32_t, Abi::KeyWords>& key, std::uint32_t pc) {
        const auto ring = DeviceAddress(state, LoadDeviceWord(state, header, constant(Abi::HeaderRingLow)), LoadDeviceWord(state, header, constant(Abi::HeaderRingHigh)));
        const auto capacity = LoadDeviceWord(state, header, constant(Abi::HeaderRingCapacity));
        const auto index = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicIAdd, u32, index, DeviceWordPointer(state, ring, constant(Abi::RingCountWord)), constant(spv::ScopeDevice), constant(spv::MemorySemanticsMaskNone), constant(1u));
        EmitIfCondition(state, binary(spv::OpULessThan, boolean, index, capacity), [&] {
            const auto recordFirst = binary(spv::OpIAdd, u32, constant(Abi::RingRecordsWord), binary(spv::OpIMul, u32, index, constant(Abi::RecordWords)));
            for (std::uint32_t word = 0; word < Abi::KeyWords; ++word) StoreDeviceWord(state, ring, binary(spv::OpIAdd, u32, recordFirst, constant(word)), key[word]);
            StoreDeviceWord(state, ring, binary(spv::OpIAdd, u32, recordFirst, constant(Abi::RecordPcWord)), pc);
        });
    };
    const auto keyOf = [&](std::uint32_t low, std::uint32_t high, std::uint32_t extra, std::uint32_t sampler) {
        std::array<std::uint32_t, Abi::KeyWords> key{};
        for (std::uint32_t word = 0; word < 4u; ++word) {
            key[word] = extract(low, word);
            key[word + 4u] = extract(high, word);
        }
        for (std::uint32_t word = 0; word < 3u; ++word) key[Abi::InterfaceWord + word] = extract(extra, word);
        for (std::uint32_t word = 0; word < Abi::SamplerWords; ++word) key[Abi::SamplerWord + word] = extract(sampler, word);
        return key;
    };
    {
        const auto missFunction = state.module.AllocateId();
        state.module.AddName(missFunction, "device_image_record_miss");
        state.module.AddFunction(spv::OpFunction, TypeVoid(state), missFunction, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, TypeVoid(state), u64, uvec4, uvec4, uvec4, uvec4));
        std::array<std::uint32_t, 5> parameters{};
        for (std::uint32_t index = 0; index < parameters.size(); ++index) {
            parameters[index] = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, index == 0u ? u64 : uvec4, parameters[index]);
        }
        EmitLabel(state, state.module.AllocateId());
        appendMiss(parameters[0], keyOf(parameters[1], parameters[2], parameters[3], parameters[4]), extract(parameters[3], 3u));
        state.module.AddFunction(spv::OpReturn);
        state.module.AddFunction(spv::OpFunctionEnd);
        state.deviceImageMissFunction = missFunction;
    }
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "device_image_lookup");
    state.module.AddFunction(spv::OpFunction, uvec4, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, uvec4, u64, uvec4, uvec4, uvec4, uvec4, boolean));
    const auto header = state.module.AllocateId();
    const auto low = state.module.AllocateId();
    const auto high = state.module.AllocateId();
    const auto extra = state.module.AllocateId();
    const auto sampler = state.module.AllocateId();
    const auto record = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, header);
    state.module.AddFunction(spv::OpFunctionParameter, uvec4, low);
    state.module.AddFunction(spv::OpFunctionParameter, uvec4, high);
    state.module.AddFunction(spv::OpFunctionParameter, uvec4, extra);
    state.module.AddFunction(spv::OpFunctionParameter, uvec4, sampler);
    state.module.AddFunction(spv::OpFunctionParameter, boolean, record);
    EmitLabel(state, state.module.AllocateId());
    const auto stepVariable = state.module.AllocateId();
    const auto doneVariable = state.module.AllocateId();
    const auto resultVariable = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), stepVariable, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), doneVariable, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, uvec4), resultVariable, spv::StorageClassFunction);
    const auto key = keyOf(low, high, extra, sampler);
    const auto pc = extract(extra, 3u);
    const auto hash = DeviceKeyHash(state, key);
    const auto table = DeviceAddress(state, LoadDeviceWord(state, header, constant(Abi::HeaderTableLow)), LoadDeviceWord(state, header, constant(Abi::HeaderTableHigh)));
    const auto mask = LoadDeviceWord(state, header, constant(Abi::HeaderMask));
    const auto home = binary(spv::OpBitwiseAnd, u32, hash, mask);
    state.module.AddFunction(spv::OpStore, stepVariable, constant(0u));
    state.module.AddFunction(spv::OpStore, doneVariable, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, resultVariable, ConstantU32CompositeZero(state, 4u));
    const auto loopHeader = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, loopHeader);
    EmitLabel(state, loopHeader);
    const auto step = load(stepVariable, u32);
    const auto done = load(doneVariable, boolean);
    const auto searching = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, done), binary(spv::OpULessThan, boolean, step, constant(Abi::ProbeSteps)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto first = binary(spv::OpIMul, u32, binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, home, step), mask), constant(Abi::BucketWords));
    static_assert(Abi::BucketWords % 4u == 0u && Abi::BucketBytes % 16u == 0u);
    std::array<std::uint32_t, Abi::BucketWords> bucket{};
    for (std::uint32_t quad = 0; quad < Abi::BucketWords / 4u; ++quad) {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, uvec4), pointer, binary(spv::OpIAdd, u64, table, binary(spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, binary(spv::OpIAdd, u32, first, constant(quad * 4u))), BdaConstant(state, 2u))));
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, uvec4, loaded, pointer, spv::MemoryAccessAlignedMask, 16u);
        for (std::uint32_t word = 0; word < 4u; ++word) bucket[quad * 4u + word] = extract(loaded, word);
    }
    const auto at = [&](std::uint32_t word) { return bucket[word]; };
    const auto live = binary(spv::OpIEqual, boolean, at(Abi::StateWord), constant(Abi::LiveState));
    auto same = live;
    for (std::uint32_t word = 0; word < Abi::KeyWords; ++word) same = binary(spv::OpLogicalAnd, boolean, same, binary(spv::OpIEqual, boolean, at(word), key[word]));
    const auto found = CompositeU32x4(state, at(Abi::SlotWord), at(Abi::ModeWord), at(Abi::FaultWord), binary(spv::OpIAdd, u32, at(Abi::SamplerSlotWord), constant(1u)));
    const auto previous = load(resultVariable, uvec4);
    state.module.AddFunction(spv::OpStore, resultVariable, Select(state, uvec4, same, found, previous));
    state.module.AddFunction(spv::OpStore, doneVariable, binary(spv::OpLogicalOr, boolean, Unary(state, spv::OpLogicalNot, boolean, live), same));
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, stepVariable, binary(spv::OpIAdd, u32, step, constant(1u)));
    state.module.AddFunction(spv::OpBranch, loopHeader);
    EmitLabel(state, merge);
    const auto result = load(resultVariable, uvec4);
    const auto missed = binary(spv::OpIEqual, boolean, extract(result, 3u), constant(0u));
    EmitIfCondition(state, binary(spv::OpLogicalAnd, boolean, record, missed), [&] { appendMiss(header, key, pc); });
    state.module.AddFunction(spv::OpReturnValue, result);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.deviceImageLookupFunction = function;
}

void EmitImage(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    const auto& memory = ctx.Memory(inst);
    const auto& base = ImageResourceOf(state, memory);
    if (base.table == NoTable) ctx.Fail(inst, "accesses an image outside the image tables");
    EmitImageStoreRecord(ctx, inst, memory, base);
    const auto imageInfo = ImageOpcodeInfoOf(inst.Opcode());
    const auto& samplers = state.program.Info().samplers;
    const bool tableSampler = imageInfo.needsSampler && memory.sampler < samplers.size() && samplers[memory.sampler].table != NoTable;
    const auto& modes = state.program.Info().runtimeImageModes.at(memory.resource);
    if (base.deviceTable) {
        EmitDeviceTableImage(ctx, inst, memory, base, modes, tableSampler);
        return;
    }
    const auto access = EmitTableAccess(ctx, inst, memory, base, tableSampler);
    if (!access.unsupported) {
        const auto mode = Binary(state, spv::OpBitwiseAnd, TypeU32(state), Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.flags, ConstantU32(state, ImageTableAbi::ModeShift)), ConstantU32(state, ImageTableAbi::ModeMask));
        EmitTableImageCall(ctx, inst, memory, modes, Select(state, TypeU32(state), access.valid, mode, ConstantU32(state, InvalidTableSelector)), tableSampler, false);
    }
    state.tableAccess = {};
}

void EmitGetImageResource(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitGetSamplerResource(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitMakeImageAddress(SpirvValueEmitContext& context) {
    EmitVoid(context);
}

void EmitImageQueryDimensions(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageQueryLod(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageRead(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageWrite(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageSampleRaw(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageGatherRaw(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicSwap32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicIAdd32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicUMin32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicUMax32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicAnd32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicOr32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicXor32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicCmpSwap32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicISub32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicSMin32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicSMax32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicInc32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicDec32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicFCmpSwap32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicFMin32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicFMax32(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicSwap64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicIAdd64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicISub64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicUMin64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicUMax64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicSMin64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicSMax64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicAnd64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicOr64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicXor64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageAtomicCmpSwap64(SpirvValueEmitContext& ctx, const IrValue& inst) {
    EmitImage(ctx, inst);
}

void EmitImageBvhIntersectRay(SpirvValueEmitContext& ctx, const IrValue& inst) {
    auto& state = ctx.state;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto f32 = TypeF32(state);
    const auto boolean = TypeBool(state);
    const auto result4 = TypeU32Composite(state, 4u);
    const auto invalid = ConstantU32(state, 0xffffffffu);
    const auto uint = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto real = [&](std::uint32_t bits) { return ConstantF32(state, bits); };
    const auto op = [&](spv::Op opcode, std::uint32_t type, std::uint32_t lhs, std::uint32_t rhs) { return Binary(state, opcode, type, lhs, rhs); };
    const auto compose = [&](const std::array<std::uint32_t, 4>& values) {
        const auto id = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, result4, id, values[0], values[1], values[2], values[3]);
        return id;
    };
    const auto asFloat = [&](std::uint32_t value) { return Unary(state, spv::OpBitcast, f32, value); };
    const auto asUint = [&](std::uint32_t value) { return Unary(state, spv::OpBitcast, u32, value); };
    const auto selectF = [&](std::uint32_t condition, std::uint32_t whenTrue, std::uint32_t whenFalse) { return Select(state, f32, condition, whenTrue, whenFalse); };
    const auto selectU = [&](std::uint32_t condition, std::uint32_t whenTrue, std::uint32_t whenFalse) { return Select(state, u32, condition, whenTrue, whenFalse); };
    const auto both = [&](std::uint32_t lhs, std::uint32_t rhs) { return op(spv::OpLogicalAnd, boolean, lhs, rhs); };
    const auto either = [&](std::uint32_t lhs, std::uint32_t rhs) { return op(spv::OpLogicalOr, boolean, lhs, rhs); };
    const auto isNan = [&](std::uint32_t value) { return Unary(state, spv::OpIsNan, boolean, value); };
    const auto widen = [&](std::uint32_t value) { return Unary(state, spv::OpUConvert, u64, value); };
    const auto pair = [&](std::uint32_t low, std::uint32_t high) { return op(spv::OpBitwiseOr, u64, widen(low), op(spv::OpShiftLeftLogical, u64, widen(high), BdaConstant(state, 32u))); };

    const auto query = [&] {
        const auto descriptor = ctx.Arg(inst, 0);
        std::array<std::uint32_t, 4> word{};
        for (std::uint32_t i = 0; i < 4u; ++i) {
            word[i] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, u32, word[i], descriptor, i);
        }
        const IrValue* address = ctx.ImageAddress(inst.Argument(1));
        const auto component = [&](std::uint32_t index) { return ctx.Arg(*address, index); };
        const bool a16 = (ctx.Memory(inst).imageSampleFlags & RdnaImageSampleFlagA16) != 0u;
        const bool node64 = ctx.Memory(inst).addressIsFull;
        const std::uint32_t ray = node64 ? 2u : 1u;

        const auto nodeLow = component(0);
        const auto node = node64 ? pair(nodeLow, component(1)) : widen(nodeLow);
        const auto type = op(spv::OpBitwiseAnd, u32, nodeLow, uint(7u));
        const auto nodeIndex = op(spv::OpShiftRightLogical, u64, node, BdaConstant(state, 3u));
        const auto baseUnits = pair(word[0], op(spv::OpBitwiseAnd, u32, word[1], uint(0xffu)));
        const auto lastNode = pair(word[2], op(spv::OpBitwiseAnd, u32, word[3], uint(0x3ffu)));
        const auto boxGrow = op(spv::OpBitwiseAnd, u32, op(spv::OpShiftRightLogical, u32, word[1], uint(23u)), uint(0xffu));
        const auto boxSort = op(spv::OpINotEqual, boolean, op(spv::OpBitwiseAnd, u32, word[1], uint(0x80000000u)), uint(0u));
        const auto barycentrics = op(spv::OpINotEqual, boolean, op(spv::OpBitwiseAnd, u32, word[3], uint(1u << 24u)), uint(0u));
        const auto nodeAddress = op(spv::OpIAdd, u64, op(spv::OpShiftLeftLogical, u64, baseUnits, BdaConstant(state, 8u)), op(spv::OpShiftLeftLogical, u64, op(spv::OpBitwiseAnd, u64, node, BdaConstant(state, ~std::uint64_t{7u})), BdaConstant(state, 3u)));
        const auto within = [&](spv::Op compare) {
            const auto inRange = op(compare, boolean, nodeIndex, lastNode);
            return node64 ? inRange : both(op(spv::OpINotEqual, boolean, baseUnits, BdaConstant(state, 0u)), inRange);
        };
        const auto valid = within(spv::OpULessThanEqual);
        const auto isTriangle = both(valid, op(spv::OpULessThanEqual, boolean, type, uint(1u)));
        const auto isBox16 = both(valid, op(spv::OpIEqual, boolean, type, uint(4u)));
        const auto isBox32 = both(within(spv::OpULessThan), op(spv::OpIEqual, boolean, type, uint(5u)));

        const auto extent = asFloat(component(ray));
        std::array<std::uint32_t, 3> origin{};
        std::array<std::uint32_t, 3> direction{};
        std::array<std::uint32_t, 3> inverse{};
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            origin[axis] = asFloat(component(ray + 1u + axis));
            if (!a16) {
                direction[axis] = asFloat(component(ray + 4u + axis));
                inverse[axis] = asFloat(component(ray + 7u + axis));
                continue;
            }
            const auto half = [&](std::uint32_t index) {
                const auto packed = component(ray + 4u + index / 2u);
                return EmitF16BitsToF32(state, index % 2u == 0u ? op(spv::OpBitwiseAnd, u32, packed, uint(0xffffu)) : op(spv::OpShiftRightLogical, u32, packed, uint(16u)));
            };
            direction[axis] = half(axis);
            inverse[axis] = half(3u + axis);
        }

        const auto nodeDwords = [&](std::uint32_t count) {
            std::vector<std::uint32_t> dwords;
            if (state.bdaProbeFunction == 0) {
                for (std::uint32_t first = 0; first < count; first += 4u) {
                    const auto read = EmitBdaDwordReads(ctx, inst, nodeAddress, first * 4u, 4u, true);
                    dwords.insert(dwords.end(), read.begin(), read.end());
                }
                return dwords;
            }
            const auto pc = uint(inst.Flags<MemoryFlags>().pc);
            const auto bytes = uint(count * 4u);
            const auto physical = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, u64, physical, state.bdaProbeFunction, nodeAddress, bytes, pc);
            const auto mapped = both(op(spv::OpINotEqual, boolean, physical, BdaConstant(state, 0u)), op(spv::OpIEqual, boolean, op(spv::OpBitwiseAnd, u64, physical, BdaConstant(state, 3u)), BdaConstant(state, 0u)));
            EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, mapped), [&] { RecordBdaFault(state, nodeAddress, bytes, pc, BdaAbi::FaultReason::Unmapped); });
            for (std::uint32_t first = 0; first < count; first += 4u) {
                const auto quad = EmitValueOrDefaultIfCondition(state, mapped, result4, ConstantU32CompositeZero(state, 4u), [&] {
                    std::array<std::uint32_t, 4> values{};
                    for (std::uint32_t i = 0; i < 4u; ++i) {
                        const auto pointer = state.module.AllocateId();
                        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, op(spv::OpIAdd, u64, physical, BdaConstant(state, (first + i) * 4u)));
                        values[i] = state.module.AllocateId();
                        state.module.AddFunction(spv::OpLoad, u32, values[i], pointer, spv::MemoryAccessAlignedMask, 4u);
                    }
                    return compose(values);
                });
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    const auto value = state.module.AllocateId();
                    state.module.AddFunction(spv::OpCompositeExtract, u32, value, quad, i);
                    dwords.push_back(value);
                }
            }
            return dwords;
        };
        const auto sub = [&](const std::array<std::uint32_t, 3>& a, const std::array<std::uint32_t, 3>& b) {
            return std::array<std::uint32_t, 3>{op(spv::OpFSub, f32, a[0], b[0]), op(spv::OpFSub, f32, a[1], b[1]), op(spv::OpFSub, f32, a[2], b[2])};
        };
        const auto cross = [&](const std::array<std::uint32_t, 3>& a, const std::array<std::uint32_t, 3>& b) {
            const auto term = [&](std::uint32_t i, std::uint32_t j) { return op(spv::OpFSub, f32, op(spv::OpFMul, f32, a[i], b[j]), op(spv::OpFMul, f32, a[j], b[i])); };
            return std::array<std::uint32_t, 3>{term(1, 2), term(2, 0), term(0, 1)};
        };
        const auto dot = [&](const std::array<std::uint32_t, 3>& a, const std::array<std::uint32_t, 3>& b) {
            return op(spv::OpFAdd, f32, op(spv::OpFAdd, f32, op(spv::OpFMul, f32, a[0], b[0]), op(spv::OpFMul, f32, a[1], b[1])), op(spv::OpFMul, f32, a[2], b[2]));
        };

        const auto triangle = [&] {
            const auto d = nodeDwords(16u);
            const auto second = op(spv::OpIEqual, boolean, type, uint(1u));
            std::array<std::uint32_t, 3> v1{};
            std::array<std::uint32_t, 3> v2{};
            std::array<std::uint32_t, 3> v3{};
            for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                v1[axis] = selectF(second, asFloat(d[3u + axis]), asFloat(d[axis]));
                v2[axis] = selectF(second, asFloat(d[9u + axis]), asFloat(d[3u + axis]));
                v3[axis] = asFloat(d[6u + axis]);
            }
            const auto e1 = sub(v2, v1);
            const auto e2 = sub(v3, v1);
            const auto e3 = sub(origin, v1);
            const auto s1 = cross(direction, e2);
            const auto s2 = cross(e3, e1);
            auto tNum = dot(e2, s2);
            auto tDenom = dot(s1, e1);
            const auto iNum = dot(e3, s1);
            const auto jNum = dot(direction, s2);
            const auto t = op(spv::OpFDiv, f32, tNum, tDenom);
            const auto u = op(spv::OpFDiv, f32, iNum, tDenom);
            const auto v = op(spv::OpFDiv, f32, jNum, tDenom);
            const auto zero = real(0u);
            const auto one = real(0x3f800000u);
            const auto missed = either(either(either(op(spv::OpFOrdLessThan, boolean, u, zero), op(spv::OpFOrdGreaterThan, boolean, u, one)), either(op(spv::OpFOrdLessThan, boolean, v, zero), op(spv::OpFOrdGreaterThan, boolean, op(spv::OpFAdd, f32, u, v), one))), op(spv::OpFOrdLessThan, boolean, t, zero));
            tNum = selectF(missed, real(0x7f800000u), tNum);
            tDenom = selectF(missed, one, tDenom);
            const auto triangleId = d[15];
            const auto shift = op(spv::OpShiftLeftLogical, u32, type, uint(3u));
            const auto barycentric = [&](std::uint32_t sourceShift) {
                const auto source = op(spv::OpBitwiseAnd, u32, op(spv::OpShiftRightLogical, u32, triangleId, op(spv::OpIAdd, u32, shift, uint(sourceShift))), uint(3u));
                const auto rest = op(spv::OpFSub, f32, op(spv::OpFSub, f32, tDenom, iNum), jNum);
                return selectF(op(spv::OpIEqual, boolean, source, uint(1u)), iNum, selectF(op(spv::OpIEqual, boolean, source, uint(2u)), jNum, rest));
            };
            const auto hit = Unary(state, spv::OpLogicalNot, boolean, missed);
            return compose({asUint(tNum), asUint(tDenom), selectU(barycentrics, asUint(barycentric(0u)), triangleId), selectU(barycentrics, asUint(barycentric(2u)), selectU(hit, uint(1u), uint(0u)))});
        };

        const auto boxes = [&](bool fp16) {
            const auto d = nodeDwords(fp16 ? 16u : 28u);
            const auto bound = [&](std::uint32_t child, std::uint32_t index) {
                const auto position = child * 6u + index;
                if (!fp16) {
                    return asFloat(d[4u + position]);
                }
                const auto packed = d[4u + position / 2u];
                return EmitF16BitsToF32(state, position % 2u == 0u ? op(spv::OpBitwiseAnd, u32, packed, uint(0xffffu)) : op(spv::OpShiftRightLogical, u32, packed, uint(16u)));
            };
            const auto nanMax = [&](std::uint32_t a, std::uint32_t b) { return selectF(either(isNan(a), op(spv::OpFOrdGreaterThan, boolean, a, b)), a, b); };
            const auto nanMin = [&](std::uint32_t a, std::uint32_t b) { return selectF(either(isNan(a), op(spv::OpFOrdLessThan, boolean, a, b)), a, b); };
            const auto zero = real(0u);
            const auto growFactor = op(spv::OpFAdd, f32, real(0x3f800000u), op(spv::OpFMul, f32, Unary(state, spv::OpConvertUToF, f32, boxGrow), real(0x33800000u)));
            std::array<std::uint32_t, 4> children{};
            std::array<std::uint32_t, 4> keys{};
            for (std::uint32_t child = 0; child < 4u; ++child) {
                std::uint32_t enter = 0;
                std::uint32_t leave = 0;
                for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                    const auto planeMin = op(spv::OpFMul, f32, op(spv::OpFSub, f32, bound(child, axis), origin[axis]), inverse[axis]);
                    const auto planeMax = op(spv::OpFMul, f32, op(spv::OpFSub, f32, bound(child, axis + 3u), origin[axis]), inverse[axis]);
                    const auto positive = op(spv::OpFOrdGreaterThanEqual, boolean, inverse[axis], zero);
                    const auto near = selectF(positive, planeMin, planeMax);
                    const auto far = selectF(positive, planeMax, planeMin);
                    enter = axis == 0u ? near : nanMax(enter, near);
                    leave = axis == 0u ? far : nanMin(leave, far);
                }
                const auto nan = either(isNan(enter), isNan(leave));
                const auto minT = selectF(nan, real(0x7f800000u), selectF(op(spv::OpFOrdGreaterThan, boolean, enter, zero), enter, zero));
                const auto maxT = selectF(nan, real(0xff800000u), selectF(op(spv::OpFOrdLessThan, boolean, leave, extent), leave, extent));
                const auto hit = op(spv::OpFOrdLessThanEqual, boolean, minT, op(spv::OpFMul, f32, maxT, growFactor));
                children[child] = selectU(hit, d[child], invalid);
                keys[child] = minT;
            }
            auto sorted = children;
            for (const auto [a, b] : std::array<std::pair<std::uint32_t, std::uint32_t>, 5>{{{0, 2}, {1, 3}, {0, 1}, {2, 3}, {1, 2}}}) {
                const auto bValid = op(spv::OpINotEqual, boolean, sorted[b], invalid);
                const auto aInvalid = op(spv::OpIEqual, boolean, sorted[a], invalid);
                const auto swap = either(both(bValid, op(spv::OpFOrdLessThan, boolean, keys[b], keys[a])), aInvalid);
                const auto sortedA = selectU(swap, sorted[b], sorted[a]);
                const auto sortedB = selectU(swap, sorted[a], sorted[b]);
                const auto keyA = selectF(swap, keys[b], keys[a]);
                const auto keyB = selectF(swap, keys[a], keys[b]);
                sorted[a] = sortedA;
                sorted[b] = sortedB;
                keys[a] = keyA;
                keys[b] = keyB;
            }
            for (std::uint32_t child = 0; child < 4u; ++child) {
                children[child] = selectU(boxSort, sorted[child], children[child]);
            }
            return compose(children);
        };

        const auto none = compose({invalid, invalid, invalid, invalid});
        const auto tested = EmitValueOrDefaultIfCondition(state, isTriangle, result4, none, triangle);
        const auto box16 = EmitValueOrDefaultIfCondition(state, isBox16, result4, tested, [&] { return boxes(true); });
        return EmitValueOrDefaultIfCondition(state, isBox32, result4, box16, [&] { return boxes(false); });
    };
    const auto inactive = compose({invalid, invalid, invalid, invalid});
    ctx.Define(inst, EmitValueOrDefaultIfCondition(state, ctx.Arg(inst, 2), result4, inactive, query));
}

}
