#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include <bit>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using namespace AgcDriver::Graphics;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t Texels = Width * Height;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileS64KBX = 0x19u;
constexpr std::uint32_t Rounds = 40;
constexpr std::uint32_t TexturesPerRound = 4;
constexpr std::uint32_t Textures = Rounds * TexturesPerRound;
constexpr std::uint32_t Queues = 2;
constexpr std::uint32_t Untouched = 0xcafef00du;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

namespace Ops {

using namespace Isa;

constexpr std::uint32_t VAndB32 = 0x1bu;
constexpr std::uint32_t ImageLoad = 0x00u;

constexpr std::array<std::uint32_t, 2> Mimg(std::uint32_t opcode, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t resource) {
    return {0xf0000000u | (opcode << 18u) | 0x1000u | (1u << 8u) | (1u << 3u), ((resource / 4u) << 16u) | (vdata << 8u) | vaddr};
}

}

const Program& LoadProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop2(VAndB32, 1, InlineInteger(63), 0)
              << Vop1(VMovB32, 2, 12)
              << Vop2(VLshlRevB32, 3, InlineInteger(6), 2)
              << Vop2(VAddNcU32, 3, Vgpr(1), 3)
              << Vop2(VLshlRevB32, 3, InlineInteger(2), 3)
              << Mimg(ImageLoad, 4, 1, 0)
              << WaitAll
              << Mubuf(BufferStoreDword, 4, 3, 8)
              << EndProgram;
        return built;
    }();
    return program;
}

std::array<std::uint32_t, 8> TiledImage(std::uint64_t address) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((Width - 1u) & 3u) << 30u),
        ((Width - 1u) >> 2u) | ((Height - 1u) << 14u),
        IdentitySwizzle | (TileS64KBX << 20u) | (Image2D << 28u),
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

std::uint32_t Texel(std::uint32_t texture, std::uint32_t x, std::uint32_t y) {
    return (texture << 16u) | (y << 8u) | x;
}

void FillTexture(PrivatePages& pages, std::uint32_t texture) {
    const auto image = TiledImage(pages.Address());
    const auto descriptor = DecodeTextureResource(image);
    const auto geometry = DescribeSurface(descriptor);
    Require(geometry.guestBytes <= pages.Bytes(), "detiler pools: the texture does not fit its pages");
    const auto* equation = FindTextureSwizzleEquation(XorSwizzleMode(descriptor.tileMode), 4u);
    Require(equation != nullptr, "detiler pools: the swizzle equation is missing");
    const auto& mip = geometry.mips.at(0);
    const auto layout = ThinBlockLayout(descriptor.tileMode, 4u);
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            auto offset = mip.tiledOffset;
            if (mip.tail) offset += EquationOffset(*equation, x + mip.tailX, y + mip.tailY);
            else offset += (static_cast<std::uint64_t>(y / layout[2]) * mip.blocksPerRow + x / layout[1]) * layout[0] + EquationOffset(*equation, x, y);
            const auto value = Texel(texture, x, y);
            std::memcpy(pages.Data() + offset, &value, 4u);
        }
    }
}

void Run() {
    std::vector<std::unique_ptr<PrivatePages>> textures;
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        textures.push_back(std::make_unique<PrivatePages>(1));
        FillTexture(*textures.back(), texture);
    }
    PrivatePages output(static_cast<std::uint32_t>((std::uint64_t{Textures} * Texels * 4u + PageBytes - 1u) / PageBytes));
    for (std::uint64_t word = 0; word < output.Bytes() / 4u; ++word) output.Words()[word] = Untouched;
    std::string failure;
    StderrCapture capture;
    try {
        for (std::uint32_t round = 0; round < Rounds; ++round) {
            Commands commands;
            for (std::uint32_t slot = 0; slot < TexturesPerRound; ++slot) {
                const auto texture = round * TexturesPerRound + slot;
                std::array<std::uint32_t, 12> userData{};
                const auto image = TiledImage(textures[texture]->Address());
                const auto buffer = BufferResource(output.Address(std::uint64_t{texture} * Texels * 4u), Texels * 4u);
                std::copy(image.begin(), image.end(), userData.begin());
                std::copy(buffer.begin(), buffer.end(), userData.begin() + 8);
                commands.Dispatch(LoadProgram(), userData, Width, Height);
            }
            commands.Submit(round % Queues);
        }
        WaitIdle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    const auto log = capture.Finish();
    Expect(failure.empty() && log.find("failed") == std::string::npos, "the submissions on two queues failed: " + failure + "\n" + log);
    AgcDriver::GuestMemory::FlushGpuWrites(output.Address(), output.Bytes());
    std::uint32_t wrongTextures = 0;
    std::string first;
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        for (std::uint32_t texel = 0; texel < Texels; ++texel) {
            const auto read = output.Words()[std::uint64_t{texture} * Texels + texel];
            const auto wanted = Texel(texture, texel % Width, texel / Width);
            if (read == wanted) continue;
            if (wrongTextures++ == 0u) first = "texture " + std::to_string(texture) + " texel " + std::to_string(texel) + " read " + Hex(read) + ", expected " + Hex(wanted);
            break;
        }
    }
    Expect(wrongTextures == 0u, std::to_string(wrongTextures) + " of " + std::to_string(Textures) + " tiled textures read wrong texels across " + std::to_string(Rounds) + " submissions alternating between two queues, first " + first);
}

}

#endif

int main() {
#ifdef _WIN32
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    try {
        Run();
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d detiler pool checks failed\n", failures);
        return 1;
    }
    std::printf("detiler pools across queues tests passed\n");
    return 0;
#else
    std::printf("skipped, guest pages are Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
