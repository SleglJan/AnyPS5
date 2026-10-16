#include "DrawSubmitFixture.hpp"
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32

namespace {

using namespace DrawSubmit;
using NewPath::PrivatePages;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t Texels = Width * Height;
constexpr std::uint32_t Format32Sint = 21;
constexpr std::uint32_t Format32x4Sint = 76;
constexpr std::uint32_t StoreSurfaces = 6;
constexpr std::uint32_t AtomicSurfaces = 2;
constexpr std::uint32_t UserWords = 9;
constexpr std::uint32_t SeedSgpr = 8;
constexpr std::uint32_t GroupSgpr = 9;
constexpr std::uint32_t ImageStore = 0x08u;
constexpr std::uint32_t ImageAtomicAdd = 0x11u;
constexpr std::uint32_t VOrB32 = 0x1cu;
constexpr std::array<std::uint32_t, 4> ComponentOffsets{0u, 0x1000u, 0x2000u, 0x30000000u};

constexpr std::array<std::uint32_t, 2> ImageOp(std::uint32_t opcode, std::uint32_t dmask, std::uint32_t vdata) {
    return DeviceTable::Asm::Mimg(opcode, dmask, false, vdata, 1u, 0u, 0u, true);
}

struct Surface {
    std::uint32_t format;
    std::uint32_t elementWords;
    std::unique_ptr<PrivatePages> pages;

    Surface(std::uint32_t format, std::uint32_t elementWords) : format(format), elementWords(elementWords), pages(std::make_unique<PrivatePages>(1u)) {}

    std::array<std::uint32_t, 8> Descriptor() const {
        const auto address = pages->Address();
        return {
            Low(address >> 8u),
            (Low(address >> 40u) & 0xffu) | (format << 20u) | (((Width - 1u) & 3u) << 30u),
            ((Width - 1u) >> 2u) | ((Height - 1u) << 14u) | (1u << 31u),
            0xfacu | (DeviceTable::Type2D << 28u),
            0u,
            0u,
            0u,
            0u,
        };
    }

    std::array<std::uint32_t, UserWords> UserData(std::uint32_t seed) const {
        const auto descriptor = Descriptor();
        std::array<std::uint32_t, UserWords> words{};
        std::copy(descriptor.begin(), descriptor.end(), words.begin());
        words[SeedSgpr] = seed;
        return words;
    }

    std::vector<std::uint32_t> Read() const {
        return ReadBack(reinterpret_cast<const std::uint32_t*>(pages->Data()), Texels * elementWords);
    }
};

const Program& StoreProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop1(VMovB32, 1, Vgpr(0))
              << Vop1(VMovB32, 2, GroupSgpr)
              << Vop2(VLshlRevB32, 4, InlineInteger(6), 2)
              << Vop2(VOrB32, 4, Vgpr(1), 4)
              << Vop2(VAddNcU32, 4, SeedSgpr, 4);
        for (std::uint32_t component = 1; component < 4u; ++component) built << Vop2(VAddNcU32, 4u + component, Literal, 4) << ComponentOffsets[component];
        built << ImageOp(ImageStore, 0xfu, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& AtomicProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop1(VMovB32, 1, Vgpr(0))
              << Vop1(VMovB32, 2, GroupSgpr)
              << Vop2(VLshlRevB32, 4, InlineInteger(6), 2)
              << Vop2(VOrB32, 4, Vgpr(1), 4)
              << Vop2(VAddNcU32, 4, InlineInteger(1), 4)
              << ImageOp(ImageAtomicAdd, 0x1u, 4)
              << EndProgram;
        return built;
    }();
    return program;
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

std::uint32_t Stored(std::uint32_t seed, std::uint32_t texel, std::uint32_t component) {
    return texel + seed + ComponentOffsets[component];
}

void ExpectStores(const Surface& surface, std::uint32_t seed, const std::string& what) {
    const auto words = surface.Read();
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t texel = 0; texel < Texels; ++texel) {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto word = words[texel * 4u + component];
            const auto expected = Stored(seed, texel, component);
            if (word == expected || wrong++ != 0u) continue;
            first = "texel (" + std::to_string(texel % Width) + ", " + std::to_string(texel / Width) + ") component " + std::to_string(component) + " is " + Hex(word) + ", expected " + Hex(expected);
        }
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(Texels * 4u) + " stored components are wrong, first " + first);
}

std::uint32_t AtomicStart(std::uint32_t surface, std::uint32_t texel) {
    return 0x01000000u * (surface + 1u) + 7u * texel;
}

void ExpectSums(const Surface& surface, std::uint32_t index, std::uint32_t adds, const std::string& what) {
    const auto words = surface.Read();
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t texel = 0; texel < Texels; ++texel) {
        const auto expected = AtomicStart(index, texel) + adds * (texel + 1u);
        if (words[texel] == expected || wrong++ != 0u) continue;
        first = "texel (" + std::to_string(texel % Width) + ", " + std::to_string(texel / Width) + ") is " + Hex(words[texel]) + ", expected " + Hex(expected);
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(Texels) + " sums are wrong, first " + first);
}

void Stores() {
    std::vector<Surface> surfaces;
    for (std::uint32_t index = 0; index < StoreSurfaces; ++index) surfaces.emplace_back(Format32x4Sint, 4u);
    for (std::uint32_t round = 0; round < 2u; ++round) {
        const std::string what = round == 0u ? "stores, first use: " + std::to_string(StoreSurfaces) + " dispatches in one submission, each storing into a 32_32_32_32_SINT surface through its UINT view, a view not made before" : "stores, reuse: the same dispatches in a later submission";
        Commands commands;
        for (std::uint32_t index = 0; index < StoreSurfaces; ++index) {
            const auto userData = surfaces[index].UserData(0x00100000u * (round + 1u) + 0x10000u * index);
            commands.Dispatch(StoreProgram(), userData, Width, Height);
        }
        const auto failure = Submit(commands);
        Expect(failure.empty(), what + ": the submission failed: " + failure);
        for (std::uint32_t index = 0; index < StoreSurfaces; ++index) ExpectStores(surfaces[index], 0x00100000u * (round + 1u) + 0x10000u * index, what + ": surface " + std::to_string(index));
    }
}

void Atomics() {
    std::vector<Surface> surfaces;
    for (std::uint32_t index = 0; index < AtomicSurfaces; ++index) {
        surfaces.emplace_back(Format32Sint, 1u);
        auto* words = surfaces.back().pages->Words();
        for (std::uint32_t texel = 0; texel < Texels; ++texel) words[texel] = AtomicStart(index, texel);
    }
    AgcDriver::GuestMemory::BumpCollectEpoch();
    std::uint32_t adds = 0;
    for (std::uint32_t round = 0; round < 2u; ++round) {
        const std::string what = round == 0u ? "atomics, first use: two dispatches per 32_SINT surface in one submission add through its 32-bit UINT atomic view, a view not made before" : "atomics, reuse: one more dispatch per surface in a later submission";
        const std::uint32_t dispatches = round == 0u ? 2u : 1u;
        Commands commands;
        for (std::uint32_t repeat = 0; repeat < dispatches; ++repeat) {
            for (std::uint32_t index = 0; index < AtomicSurfaces; ++index) commands.Dispatch(AtomicProgram(), surfaces[index].UserData(0u), Width, Height);
        }
        const auto failure = Submit(commands);
        Expect(failure.empty(), what + ": the submission failed: " + failure);
        adds += dispatches;
        for (std::uint32_t index = 0; index < AtomicSurfaces; ++index) ExpectSums(surfaces[index], index, adds, what + ": surface " + std::to_string(index));
    }
}

void Run() {
    Cases({
        {"stores", Stores},
        {"atomics", Atomics},
    });
}

}

int main() {
    return DrawSubmit::Main("storage view first use", Run);
}

#else

int main() {
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
}

#endif
