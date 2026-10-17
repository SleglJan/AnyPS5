#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "StorageImageMemory.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32

namespace {

using namespace DrawSubmit;
using NewPath::PrivatePages;
using NewPath::Require;
namespace Graphics = AgcDriver::Graphics;
namespace Memory = ShaderRecompiler::StorageImageMemory;

constexpr std::uint32_t TypeLine = 8;
constexpr std::uint32_t TypePlane = 9;
constexpr std::uint32_t TypeVolume = 10;
constexpr std::uint32_t TypeCube = 11;
constexpr std::uint32_t TypeLineArray = 12;
constexpr std::uint32_t TypePlaneArray = 13;

struct Shape {
    const char* name;
    std::uint32_t format;
    std::uint32_t tileMode;
    std::uint32_t type;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t depth;
    std::uint32_t levels;
    std::uint32_t xorBits = 0;
    std::uint32_t baseLevel = 0;
    std::uint32_t lastLevel = 0xffu;
};

std::array<std::uint32_t, 8> Descriptor(std::uint64_t address, const Shape& shape) {
    const auto base = address + shape.xorBits;
    const auto last = std::min(shape.lastLevel, shape.levels - 1u);
    return {
        Low(base >> 8u),
        (Low(base >> 40u) & 0xffu) | (shape.format << 20u) | (((shape.width - 1u) & 3u) << 30u),
        ((shape.width - 1u) >> 2u) | ((shape.height - 1u) << 14u) | (1u << 31u),
        0xfacu | (shape.baseLevel << 12u) | (last << 16u) | (shape.tileMode << 20u) | (shape.type << 28u),
        shape.depth - 1u,
        (shape.levels - 1u) << 4u,
        0u,
        0u,
    };
}

std::uint32_t ThickEquationMode(Graphics::TextureTileMode mode) {
    switch (mode) {
        case Graphics::TextureTileMode::kStandard4KB: return 0x105u;
        case Graphics::TextureTileMode::kStandard64KB: return 0x109u;
        case Graphics::TextureTileMode::kS64KBX: return 0x119u;
        case Graphics::TextureTileMode::kD64KBX: return 0x11au;
        default: return 0u;
    }
}

std::optional<std::uint64_t> DriverTexelAddress(const std::array<std::uint32_t, 8>& words, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t lod) {
    const auto descriptor = Graphics::DecodeTextureResource(words);
    const auto geometry = Graphics::DescribeSurface(descriptor);
    const auto bytes = Graphics::BytesPerElement(descriptor.format);
    const auto level = descriptor.baseLevel + lod;
    if (level > descriptor.lastLevel || level >= geometry.mips.size()) return std::nullopt;
    const auto& mip = geometry.mips[level];
    if (x >= mip.width || y >= mip.height) return std::nullopt;
    const bool arrayed = descriptor.dimension == Graphics::TextureDimension::k2DArray || descriptor.dimension == Graphics::TextureDimension::k1DArray || descriptor.dimension == Graphics::TextureDimension::kCube;
    const bool volume = descriptor.dimension == Graphics::TextureDimension::k3D;
    std::uint32_t layer = 0;
    if (arrayed) {
        layer = descriptor.baseArray + z;
        if (layer >= geometry.layers) return std::nullopt;
    } else if (volume) {
        if (!geometry.HasLayer(level, z)) return std::nullopt;
        layer = z;
    }
    const auto start = descriptor.baseAddress + geometry.GuestLayerOffset(layer) + mip.tiledOffset;
    if (descriptor.tileMode == Graphics::TextureTileMode::kLinear) return start + static_cast<std::uint64_t>(y) * mip.pitchBytes + static_cast<std::uint64_t>(x) * bytes;
    auto swizzleX = x;
    auto swizzleY = y;
    auto blockStart = start;
    std::uint32_t blockBytes = 0;
    if (geometry.thick) {
        const auto extent = Graphics::ThickBlockExtent(descriptor.tileMode, bytes);
        blockBytes = descriptor.tileMode == Graphics::TextureTileMode::kStandard4KB ? 4096u : 65536u;
        if (!mip.tail) blockStart += (static_cast<std::uint64_t>(y / extent[1]) * mip.blocksPerRow + x / extent[0]) * blockBytes;
    } else {
        const auto block = Graphics::ThinBlockLayout(descriptor.tileMode, bytes);
        blockBytes = block[0];
        if (!mip.tail) blockStart += (static_cast<std::uint64_t>(y / block[2]) * mip.blocksPerRow + x / block[1]) * block[0];
    }
    if (mip.tail) {
        swizzleX += mip.tailX;
        swizzleY += mip.tailY;
    }
    const auto mode = geometry.thick ? ThickEquationMode(descriptor.tileMode) : Graphics::EquationSwizzleMode(descriptor.tileMode);
    const auto* equation = Graphics::FindTextureSwizzleEquation(mode, bytes);
    Require(equation != nullptr, "storage image memory: the driver has no equation for a tested surface");
    std::uint32_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16u; ++bit) {
        const auto mask = equation->bits[bit];
        const auto selected = (swizzleX & mask & 0xfffu) ^ ((swizzleY << 12u) & mask & 0xfff000u) ^ ((layer << 24u) & mask & 0xff000000u);
        offset |= (static_cast<std::uint32_t>(std::popcount(selected)) & 1u) << bit;
    }
    return blockStart + ((offset ^ descriptor.pipeBankXor) & (blockBytes - 1u));
}

std::uint32_t LayerCount(const Shape& shape) {
    return shape.type == TypePlaneArray || shape.type == TypeLineArray || shape.type == TypeCube || shape.type == TypeVolume ? shape.depth : 1u;
}

std::vector<std::uint32_t> Samples(std::uint32_t extent) {
    std::vector<std::uint32_t> values{0u, 1u, extent / 3u, extent / 2u, extent - 1u, extent, extent + 7u};
    for (std::uint32_t value = 2u; value < extent && value < 70u; value += 9u) values.push_back(value);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

void Reference() {
    const std::array<Shape, 21> shapes{{
        {"linear 2D 32-bit", 20u, 0u, TypePlane, 300u, 77u, 1u, 5u},
        {"linear 1D array 8-bit", 1u, 0u, TypeLineArray, 200u, 1u, 4u, 1u},
        {"linear 3D 16-bit", 13u, 0u, TypeVolume, 40u, 30u, 7u, 3u},
        {"256B D 32-bit", 56u, 2u, TypePlane, 70u, 50u, 1u, 1u},
        {"4KB D 64-bit", 62u, 6u, TypePlane, 130u, 90u, 1u, 4u},
        {"4KB S_X 16-bit with a pipe-bank XOR", 7u, 21u, TypePlane, 100u, 100u, 1u, 7u, 0x700u},
        {"4KB D_X 64-bit", 64u, 22u, TypePlane, 120u, 80u, 1u, 3u},
        {"64KB D 32-bit", 20u, 10u, TypePlane, 200u, 120u, 1u, 8u},
        {"64KB S_T 16-bit", 7u, 17u, TypePlane, 300u, 300u, 1u, 2u},
        {"64KB D_T 16-bit", 13u, 18u, TypePlane, 300u, 260u, 1u, 3u},
        {"64KB Z_X 8-bit with a pipe-bank XOR", 1u, 24u, TypePlane, 200u, 100u, 1u, 8u, 0x3000u},
        {"64KB Z_X 16-bit 2D array", 7u, 24u, TypePlaneArray, 96u, 80u, 3u, 5u},
        {"64KB S_X 32-bit with a pipe-bank XOR", 56u, 25u, TypePlane, 300u, 150u, 1u, 9u, 0x6100u},
        {"64KB D_X 128-bit", 77u, 26u, TypePlane, 64u, 64u, 1u, 7u},
        {"64KB R_X 128-bit with a pipe-bank XOR", 77u, 27u, TypePlane, 96u, 40u, 1u, 1u, 0xa000u},
        {"64KB R_X 3D 32-bit", 20u, 27u, TypeVolume, 64u, 64u, 6u, 4u},
        {"3D 4KB S 32-bit", 20u, 5u, TypeVolume, 40u, 40u, 20u, 6u},
        {"3D 64KB S 8-bit", 1u, 9u, TypeVolume, 300u, 200u, 9u, 9u},
        {"3D 64KB S_X 16-bit with a pipe-bank XOR", 7u, 25u, TypeVolume, 64u, 64u, 64u, 7u, 0x5000u},
        {"3D 64KB D_X 64-bit", 62u, 26u, TypeVolume, 33u, 17u, 40u, 6u},
        {"cube 64KB Z_X 32-bit", 20u, 24u, TypeCube, 64u, 64u, 6u, 4u},
    }};
    constexpr std::uint64_t base = 0x2340000000ull;
    for (const auto& shape : shapes) {
        const auto words = Descriptor(base, shape);
        const std::string what = std::string("reference, ") + shape.name;
        Expect(Memory::SurfaceThroughMemory(words, Memory::Access::Load), what + ": the surface does not take the memory path");
        const auto elementBytes = Graphics::BytesPerElement(shape.format);
        const bool depthLayout = (shape.tileMode == 4u || shape.tileMode == 8u || shape.tileMode == 16u || shape.tileMode == 20u || shape.tileMode == 24u) && (shape.type == TypePlane || shape.type == TypePlaneArray) && (elementBytes == 2u || elementBytes == 4u);
        Expect(Memory::SurfaceThroughMemory(words, Memory::Access::Store) != depthLayout, what + (depthLayout ? ": the surface, which a depth target can take over, takes stores through memory" : ": the surface does not take stores through memory"));
        std::size_t checked = 0;
        std::size_t wrong = 0;
        std::string first;
        for (std::uint32_t lod = 0; lod < shape.levels + 1u; ++lod) {
            const auto width = std::max(shape.width >> std::min(lod, 31u), 1u);
            const auto height = std::max(shape.height >> std::min(lod, 31u), 1u);
            const auto layers = shape.type == TypeVolume ? std::max(shape.depth >> std::min(lod, 31u), 1u) : LayerCount(shape);
            for (const auto z : Samples(layers)) {
                for (const auto y : Samples(height)) {
                    for (const auto x : Samples(width)) {
                        const auto memory = Memory::Locate(words, x, y, z, lod);
                        const auto driver = DriverTexelAddress(words, x, y, z, lod);
                        ++checked;
                        if (memory.inside == driver.has_value() && (!memory.inside || memory.address == *driver)) continue;
                        if (wrong++ != 0u) continue;
                        first = "level " + std::to_string(lod) + " texel (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + "): " + (memory.inside ? Hex(memory.address) : std::string("outside")) + " against the driver's " + (driver ? Hex(*driver) : std::string("outside"));
                    }
                }
            }
        }
        Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(checked) + " texels differ from the driver's layout, first " + first);
    }
}

void Predicates() {
    constexpr std::uint64_t base = 0x2350000000ull;
    auto words = Descriptor(base, {"plain", 20u, 27u, TypePlane, 64u, 64u, 1u, 1u});
    Expect(Memory::SurfaceThroughMemory(words, Memory::Access::Atomic32), "predicates: a 32_UINT 64KB R_X surface takes 32-bit atomics through memory");
    Expect(!Memory::SurfaceThroughMemory(words, Memory::Access::Atomic64), "predicates: a 32-bit surface takes 64-bit atomics through memory");
    words[6] |= 1u << 21u;
    Expect(!Memory::SurfaceThroughMemory(words, Memory::Access::Store), "predicates: a DCC-compressed surface takes the memory path");
    Expect(!Memory::SurfaceThroughMemory(Descriptor(base, {"4KB Z", 20u, 4u, TypePlane, 64u, 64u, 1u, 1u}), Memory::Access::Store), "predicates: a tile mode without an equation takes the memory path");
    Expect(!Memory::SurfaceThroughMemory(Descriptor(base, {"thin 64KB S", 20u, 9u, TypePlane, 64u, 64u, 1u, 1u}), Memory::Access::Store), "predicates: a thin standard swizzle takes the memory path");
    Expect(!Memory::SurfaceThroughMemory(Descriptor(base, {"96-bit", 74u, 27u, TypePlane, 64u, 64u, 1u, 1u}), Memory::Access::Store), "predicates: a 96-bit format takes the memory path");
    Expect(!Memory::SurfaceThroughMemory(Descriptor(base, {"sRGB", 130u, 27u, TypePlane, 64u, 64u, 1u, 1u}), Memory::Access::Load), "predicates: an sRGB format takes the memory path");
    Expect(Memory::SurfaceThroughMemory(Descriptor(base, {"64-bit atomics", 63u, 27u, TypePlane, 64u, 64u, 1u, 1u}), Memory::Access::Atomic64), "predicates: a 32_32_SINT surface does not take 64-bit atomics through memory");
    const auto line = Descriptor(base, {"1D", 20u, 0u, TypeLine, 200u, 1u, 1u, 1u});
    const auto atRow = Memory::Locate(line, 7u, 5u, 0u, 0u);
    const auto atZero = Memory::Locate(line, 7u, 0u, 0u, 0u);
    Expect(atZero.inside && atRow.inside && atRow.address == atZero.address, "predicates: a 1D texel address through memory depends on y");
    const auto depth = Descriptor(base, {"64KB Z_X 32_FLOAT", 22u, 24u, TypePlane, 64u, 64u, 1u, 1u});
    Expect(!Memory::SurfaceThroughMemory(depth, Memory::Access::Store) && !Memory::SurfaceThroughMemory(depth, Memory::Access::Atomic32), "predicates: a 32-bit 64KB Z_X 2D surface, which a depth target can take over, takes stores or atomics through memory");
    Expect(Memory::SurfaceThroughMemory(depth, Memory::Access::Load), "predicates: a 32-bit 64KB Z_X 2D surface does not take loads through memory");
    Expect(!Memory::SurfaceThroughMemory(Descriptor(base, {"4KB Z_X 16_UNORM array", 7u, 20u, TypePlaneArray, 64u, 64u, 2u, 1u}), Memory::Access::Store), "predicates: a 16-bit 4KB Z_X 2D array, which a depth target can take over, takes stores through memory");
    Expect(Memory::SurfaceThroughMemory(Descriptor(base, {"64KB Z_X 8_UINT", 5u, 24u, TypePlane, 64u, 64u, 1u, 1u}), Memory::Access::Store), "predicates: an 8-bit 64KB Z_X surface does not take stores through memory");
    Expect(Memory::SurfaceThroughMemory(Descriptor(base, {"cube 64KB Z_X 32_UINT", 20u, 24u, TypeCube, 64u, 64u, 6u, 1u}), Memory::Access::Store), "predicates: a 32-bit 64KB Z_X cube does not take stores through memory");
    std::size_t equations = 0;
    for (const auto& driver : Graphics::kTextureSwizzleEquations) {
        ++equations;
        const auto* copy = Memory::FindEquation(driver.swizzleMode, driver.elementBytes);
        Expect(copy != nullptr && copy->bits == driver.bits, "predicates: the memory path's equation of swizzle mode " + Hex(driver.swizzleMode) + " at " + std::to_string(driver.elementBytes) + " bytes differs from the driver's");
    }
    Expect(equations == Memory::Equations.size(), "predicates: the memory path holds " + std::to_string(Memory::Equations.size()) + " equations, the driver " + std::to_string(equations));
    for (std::uint32_t format = 0; format <= Memory::LastMemoryFormat; ++format) {
        if (!Memory::MemoryFormat(format)) continue;
        const auto info = ShaderRecompiler::GetFormatInfo(static_cast<ShaderRecompiler::IrBufferFormat>(format));
        const auto bytes = 1u << Memory::ElementLog2(format);
        Expect(info.type != ShaderRecompiler::SpirvFormatComponentType::Unknown && info.componentCount != 0u && info.byteSize == bytes, "predicates: memory format " + std::to_string(format) + " has no codec layout of " + std::to_string(bytes) + " bytes");
        Expect(Graphics::BytesPerElement(format) == bytes, "predicates: memory format " + std::to_string(format) + " is " + std::to_string(Graphics::BytesPerElement(format)) + " bytes to the driver, " + std::to_string(bytes) + " to the memory path");
    }
}

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t UserWords = 13;
constexpr std::uint32_t OutputSgpr = 8;
constexpr std::uint32_t SliceSgpr = 12;
constexpr std::uint32_t GroupSgpr = 13;
constexpr std::uint32_t ImageLoad = 0x00u;
constexpr std::uint32_t ImageStore = 0x08u;
constexpr std::uint32_t BufferStoreDwordX4 = 0x1eu;
constexpr std::uint32_t Background = 0xc5u;

std::uint32_t DimOf(std::uint32_t type) {
    return type == TypeVolume ? 2u : type == TypePlaneArray ? 5u : 1u;
}

constexpr std::array<std::uint32_t, 2> ImageOp(std::uint32_t opcode, std::uint32_t dim) {
    return {0xf0000000u | (opcode << 18u) | 0x1000u | (0xfu << 8u) | (dim << 3u), (4u << 8u) | 1u};
}

struct Programs {
    Program store;
    Program load;
};

Programs& ProgramsFor(std::uint32_t dim) {
    static std::array<std::unique_ptr<Programs>, 6> built;
    auto& entry = built.at(dim);
    if (entry != nullptr) return *entry;
    entry = std::make_unique<Programs>();
    using namespace Isa;
    const auto coordinates = [](Program& program) {
        program << Vop1(VMovB32, 1, Vgpr(0))
                << Vop1(VMovB32, 2, GroupSgpr)
                << Vop1(VMovB32, 3, SliceSgpr);
    };
    coordinates(entry->store);
    entry->store << Vop2(VLshlRevB32, 10, InlineInteger(1), 2)
                 << Vop2(VAddNcU32, 4, Vgpr(1), 10)
                 << Vop2(VLshlRevB32, 11, InlineInteger(2), 3)
                 << Vop2(VAddNcU32, 4, Vgpr(11), 4)
                 << Vop2(VAddNcU32, 5, InlineInteger(5), 4)
                 << Vop2(VAddNcU32, 6, InlineInteger(10), 4)
                 << Vop2(VAddNcU32, 7, InlineInteger(15), 4)
                 << ImageOp(ImageStore, dim)
                 << EndProgram;
    coordinates(entry->load);
    entry->load << ImageOp(ImageLoad, dim)
                << WaitAll
                << Vop2(VLshlRevB32, 8, InlineInteger(6), 2)
                << Vop2(VAddNcU32, 8, Vgpr(1), 8)
                << Vop2(VLshlRevB32, 8, InlineInteger(4), 8)
                << Mubuf(BufferStoreDwordX4, 4, 8, OutputSgpr)
                << EndProgram;
    return *entry;
}

std::uint32_t Component(std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t component) {
    return x + 2u * y + 4u * z + 5u * component;
}

struct UintFormat {
    std::uint32_t components;
    std::uint32_t bits;
};

UintFormat UintLayout(std::uint32_t format) {
    switch (format) {
        case 5u: return {1u, 8u};
        case 11u: return {1u, 16u};
        case 20u: return {1u, 32u};
        case 27u: return {2u, 16u};
        case 60u: return {4u, 8u};
        case 62u: return {2u, 32u};
        case 75u: return {4u, 32u};
        default: throw std::runtime_error("storage image memory: an untested UINT format");
    }
}

std::string Submit(Commands& commands) {
    std::string failure;
    NewPath::StderrCapture capture;
    try {
        commands.Submit(0);
        NewPath::WaitIdle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    const auto log = capture.Finish();
    if (log.find("failed") != std::string::npos || log.find("is rejected") != std::string::npos) failure += "\n" + log;
    return failure;
}

void RoundTrip(const Shape& shape, std::uint32_t lod, std::vector<std::unique_ptr<PrivatePages>>& kept) {
    const std::string what = std::string("stores and loads, ") + shape.name;
    const auto layout = UintLayout(shape.format);
    const auto probe = Descriptor(NewPath::PageBytes, shape);
    const auto bytes = Graphics::DescribeSurface(Graphics::DecodeTextureResource(probe)).guestBytes + shape.xorBits;
    kept.push_back(std::make_unique<PrivatePages>(static_cast<std::uint32_t>((bytes + NewPath::PageBytes - 1u) / NewPath::PageBytes)));
    auto& surface = *kept.back();
    kept.push_back(std::make_unique<PrivatePages>(1u));
    auto& output = *kept.back();
    std::memset(surface.Data(), static_cast<int>(Background), static_cast<std::size_t>(surface.Bytes()));
    AgcDriver::GuestMemory::BumpCollectEpoch();
    const auto words = Descriptor(surface.Address(), shape);
    Expect(Memory::SurfaceThroughMemory(words, Memory::Access::Store), what + ": the surface does not take the memory path");
    const auto level = shape.baseLevel + lod;
    const auto width = std::max(shape.width >> level, 1u);
    const auto height = std::max(shape.height >> level, 1u);
    const auto slices = shape.type == TypeVolume ? std::max(shape.depth >> level, 1u) : LayerCount(shape);
    const bool layered = shape.type == TypeVolume || shape.type == TypePlaneArray;
    const auto dispatched = layered ? slices + 1u : slices;
    Require(width <= Lanes, "storage image memory: a tested level is wider than a wave");
    const auto dim = DimOf(shape.type);
    auto& programs = ProgramsFor(dim);
    const auto userData = [&](std::uint32_t slice) {
        std::array<std::uint32_t, UserWords> data{};
        std::copy(words.begin(), words.end(), data.begin());
        const auto buffer = NewPath::BufferResource(output.Address(), static_cast<std::uint32_t>(output.Bytes()));
        std::copy(buffer.begin(), buffer.end(), data.begin() + OutputSgpr);
        data[SliceSgpr] = slice;
        return data;
    };
    {
        Commands commands;
        for (std::uint32_t slice = 0; slice < dispatched; ++slice) commands.Dispatch(programs.store, userData(slice), Lanes, height + 1u);
        const auto failure = Submit(commands);
        Expect(failure.empty(), what + ": the store submission failed: " + failure);
    }
    std::vector<std::uint8_t> expected(static_cast<std::size_t>(surface.Bytes()), static_cast<std::uint8_t>(Background));
    for (std::uint32_t slice = 0; slice < slices; ++slice) {
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                const auto address = DriverTexelAddress(words, x, y, slice, lod);
                Require(address.has_value() && *address >= surface.Address(), "storage image memory: a stored texel is outside the driver's layout");
                auto* texel = expected.data() + (*address - surface.Address());
                for (std::uint32_t component = 0; component < layout.components; ++component) {
                    const auto value = Component(x, y, slice, component);
                    std::memcpy(texel + component * layout.bits / 8u, &value, layout.bits / 8u);
                }
            }
        }
    }
    std::vector<std::uint8_t> stored(expected.size());
    AgcDriver::GuestMemory::Read(surface.Address(), std::as_writable_bytes(std::span(stored)));
    std::size_t wrong = 0;
    std::string first;
    for (std::size_t index = 0; index < stored.size(); ++index) {
        if (stored[index] == expected[index] || wrong++ != 0u) continue;
        first = "byte " + Hex(static_cast<std::uint32_t>(index)) + " is " + Hex(stored[index]) + ", expected " + Hex(expected[index]);
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " bytes of the surface differ after the stores, first " + first);
    for (std::uint32_t slice = 0; slice < dispatched; ++slice) {
        std::memset(output.Data(), 0xee, static_cast<std::size_t>(output.Bytes()));
        AgcDriver::GuestMemory::BumpCollectEpoch();
        Commands commands;
        commands.Dispatch(programs.load, userData(slice), Lanes, height + 1u);
        const auto failure = Submit(commands);
        Expect(failure.empty(), what + ": the load submission failed: " + failure);
        const auto loaded = ReadBack(reinterpret_cast<const std::uint32_t*>(output.Data()), static_cast<std::size_t>((height + 1u) * Lanes * 4u));
        std::size_t mismatched = 0;
        std::string firstLoad;
        for (std::uint32_t y = 0; y < height + 1u; ++y) {
            for (std::uint32_t x = 0; x < Lanes; ++x) {
                if (x >= width || y >= height || slice >= slices) continue;
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    const auto expectedValue = component < layout.components ? Component(x, y, slice, component) : component == 3u ? 1u : 0u;
                    const auto value = loaded[(y * Lanes + x) * 4u + component];
                    if (value == expectedValue || mismatched++ != 0u) continue;
                    firstLoad = "texel (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(slice) + ") component " + std::to_string(component) + " loaded " + Hex(value) + ", expected " + Hex(expectedValue);
                }
            }
        }
        Expect(mismatched == 0u, what + ": slice " + std::to_string(slice) + ": " + std::to_string(mismatched) + " loaded components are wrong, first " + firstLoad);
    }
}

void RoundTrips() {
    const std::array<Shape, 10> shapes{{
        {"linear 2D 8_8_8_8_UINT", 60u, 0u, TypePlane, 64u, 8u, 1u, 1u},
        {"64KB R_X 32_UINT with a pipe-bank XOR", 20u, 27u, TypePlane, 48u, 16u, 1u, 1u, 0x2000u},
        {"64KB R_X 16_16_UINT, level 1", 27u, 27u, TypePlane, 128u, 32u, 1u, 3u, 0u, 1u, 1u},
        {"64KB Z_X 8_UINT, level 1", 5u, 24u, TypePlane, 128u, 32u, 1u, 3u, 0u, 1u, 1u},
        {"4KB S_X 8_UINT with a pipe-bank XOR", 5u, 21u, TypePlane, 40u, 20u, 1u, 1u, 0x300u},
        {"64KB D 32_32_UINT", 62u, 10u, TypePlane, 32u, 32u, 1u, 1u},
        {"256B D 32_32_32_32_UINT", 75u, 2u, TypePlane, 16u, 16u, 1u, 1u},
        {"64KB S_X 32_UINT 2D array", 20u, 25u, TypePlaneArray, 32u, 32u, 3u, 1u},
        {"3D 64KB S_X 32_UINT", 20u, 25u, TypeVolume, 32u, 32u, 8u, 1u},
        {"3D 4KB S 16_UINT", 11u, 5u, TypeVolume, 16u, 16u, 8u, 1u},
    }};
    std::vector<std::unique_ptr<PrivatePages>> kept;
    for (const auto& shape : shapes) {
        try {
            RoundTrip(shape, 0u, kept);
        } catch (const std::exception& error) {
            Expect(false, std::string("stores and loads, ") + shape.name + ": exception: " + error.what());
        }
    }
}

void Run() {
    Cases({
        {"reference", Reference},
        {"predicates", Predicates},
        {"stores and loads", RoundTrips},
    });
}

}

int main() {
    return DrawSubmit::Main("storage image memory", Run);
}

#else

int main() {
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
}

#endif
