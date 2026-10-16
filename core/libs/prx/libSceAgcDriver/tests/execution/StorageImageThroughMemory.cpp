#include "DeviceTableFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <bit>

namespace {

using namespace DeviceTable;
using namespace AgcDriver::Graphics;

constexpr std::uint32_t TextureSlot = 20;
constexpr std::uint32_t Added = 1000u;
constexpr std::uint32_t Sentinel = 0x5eed5eedu;
constexpr std::uint32_t ImageAtomicAdd = 0x11u;
constexpr std::uint32_t SLoadDwordX8 = 0x03u;
constexpr std::size_t SurfaceBytes = 2u * 65536u;

std::uint8_t* const Surface = reinterpret_cast<std::uint8_t*>(TestGuest::Allocate(SurfaceBytes, 65536u));

std::array<std::uint32_t, 8> TiledImage(std::uint32_t tileMode, std::uint32_t pipeBankXor) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Surface)) + pipeBankXor;
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((Width - 1u) & 3u) << 30u),
        ((Width - 1u) >> 2u) | ((Height - 1u) << 14u) | (1u << 31u),
        0xfacu | (tileMode << 20u) | (Type2D << 28u),
        0u, 0u, 0u, 0u,
    };
}

std::uint64_t EquationOffset(const TextureSwizzleEquation& equation, std::uint32_t x, std::uint32_t y) {
    std::uint64_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16; ++bit) {
        const auto mask = equation.bits[bit];
        const auto selected = (x & (mask & 0xfffu)) ^ ((y << 12) & (mask & 0xfff000u));
        offset |= static_cast<std::uint64_t>(std::popcount(selected) & 1u) << bit;
    }
    return offset;
}

std::uint32_t Stored(std::uint32_t seed, std::uint32_t x, std::uint32_t y) {
    return seed + x + 256u * y;
}

const Program& StoreProgram() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        built << Vop2(VAndB32, 1, InlineInteger(63), 0)
              << Vop1(VMovB32, 2, 9)
              << Vop2(VLshlRevB32, 3, InlineInteger(8), 2)
              << Vop2(VAddNcU32, 3, Vgpr(1), 3)
              << Vop2(VAddNcU32, 3, 8, 3)
              << Mimg(ImageStore, 1u, false, 3, 1, 0, 0, true)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& AtomicProgram() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        built << Vop2(VAndB32, 1, InlineInteger(63), 0)
              << Vop1(VMovB32, 2, 9)
              << Vop1(VMovB32, 3, Literal) << Added
              << Mimg(ImageAtomicAdd, 1u, false, 3, 1, 0, 0, true)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& LoadProgram() {
    static const Program program = [] {
        using namespace Asm;
        Program built;
        built << Smem(SLoadDwordX4, 28, 0, OutputSlot * 4u)
              << Smem(SLoadDwordX8, 36, 0, TextureSlot * 4u)
              << WaitAll;
        TexelIndex(built, 2);
        built << Vop1(VMovB32, 5, InlineInteger(0))
              << Vop1(VMovB32, 6, InlineInteger(0))
              << Vop1(VMovB32, 7, InlineInteger(0))
              << Mimg(ImageLoad, 1u, false, 4, 1, 36, 0, true);
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

void Check(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> loadUserData, std::uint32_t tileMode, std::uint32_t pipeBankXor, std::uint32_t seed, const std::string& what) {
    for (std::size_t word = 0; word < SurfaceBytes / 4u; ++word) reinterpret_cast<std::uint32_t*>(Surface)[word] = Sentinel;
    const auto image = TiledImage(tileMode, pipeBankXor);
    const auto descriptor = DecodeTextureResource(image);
    const auto geometry = DescribeSurface(descriptor);
    Require(geometry.guestBytes + pipeBankXor <= SurfaceBytes, what + ": the surface does not fit its allocation");
    const auto* equation = FindTextureSwizzleEquation(XorSwizzleMode(descriptor.tileMode), 4u);
    Require(equation != nullptr, what + ": the swizzle equation is missing");
    std::copy(image.begin(), image.end(), Srt.begin() + TextureSlot);
    std::array<std::uint32_t, 9> storeUserData{};
    std::copy(image.begin(), image.end(), storeUserData.begin());
    storeUserData[8] = seed;
    const auto store = Compile(device, StoreProgram(), storeUserData);
    const auto atomic = Compile(device, AtomicProgram(), storeUserData);
    const auto load = Compile(device, LoadProgram(), loadUserData);
    Batch batch(device);
    batch.Dispatch(store, Height);
    batch.Dispatch(atomic, Height);
    batch.Dispatch(load, Height);
    const auto outcome = batch.Finish();
    ExpectClean(outcome, what);
    const Expected loaded{Height, 0, 1, [seed](std::uint32_t row, std::uint32_t x, std::uint32_t, std::uint32_t word) { return word == Stored(seed, x, row) + Added; }};
    std::string mismatch;
    Expect(Classify(outcome, loaded, mismatch) == Read::Expected, what + ": the image load after the store and the atomic: " + mismatch);
    AgcDriver::GuestMemory::FlushGpuWrites(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Surface)), SurfaceBytes);
    const auto& mip = geometry.mips.at(0);
    const auto layout = ThinBlockLayout(descriptor.tileMode, 4u);
    std::vector<bool> texel(SurfaceBytes / 4u, false);
    std::uint32_t wrong = 0;
    std::string first;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            auto offset = mip.tiledOffset;
            if (mip.tail) offset += EquationOffset(*equation, x + mip.tailX, y + mip.tailY) ^ pipeBankXor;
            else offset += (static_cast<std::uint64_t>(y / layout[2]) * mip.blocksPerRow + x / layout[1]) * layout[0] + (EquationOffset(*equation, x, y) ^ pipeBankXor);
            texel[offset / 4u] = true;
            std::uint32_t value = 0;
            std::memcpy(&value, Surface + offset, 4u);
            if (value == Stored(seed, x, y) + Added) continue;
            if (wrong++ == 0u) first = "texel (" + std::to_string(x) + ", " + std::to_string(y) + ") at byte " + std::to_string(offset) + " is " + Hex(value);
        }
    }
    Expect(wrong == 0u, what + ": guest memory holds " + std::to_string(wrong) + " wrong texels at the tiling equation's addresses, first " + first);
    std::uint32_t stray = 0;
    for (std::size_t word = 0; word < SurfaceBytes / 4u; ++word) {
        if (texel[word]) continue;
        stray += reinterpret_cast<const std::uint32_t*>(Surface)[word] != Sentinel ? 1u : 0u;
    }
    Expect(stray == 0u, what + ": " + std::to_string(stray) + " words outside the surface's texels changed");
}

void Run(AgcDriver::VulkanDevice& device) {
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Check(device, userData, 0x19u, 0x0000u, 0x100u, "SW_64KB_S_X");
    Check(device, userData, 0x1bu, 0x0000u, 0x200u, "SW_64KB_R_X");
    Check(device, userData, 0x19u, 0x5600u, 0x300u, "SW_64KB_S_X with pipe/bank XOR 0x5600");
    Check(device, userData, 0x16u, 0x0a00u, 0x400u, "SW_4KB_D_X with pipe/bank XOR 0xa00");
}

}

int main() {
    return Main("storage image through memory", Run);
}
