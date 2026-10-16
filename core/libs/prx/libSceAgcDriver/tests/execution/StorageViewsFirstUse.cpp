#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <exception>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t Images = 12;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Added = 0x10000u;
constexpr std::uint32_t Sentinel = 0xfeedfaceu;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

namespace Ops {

using namespace Isa;

constexpr std::uint32_t VAndB32 = 0x1bu;
constexpr std::uint32_t ImageStore = 0x08u;
constexpr std::uint32_t ImageAtomicAdd = 0x11u;

constexpr std::array<std::uint32_t, 2> Mimg(std::uint32_t opcode, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t resource) {
    return {0xf0000000u | (opcode << 18u) | 0x1000u | (1u << 8u) | (1u << 3u), ((resource / 4u) << 16u) | (vdata << 8u) | vaddr};
}

}

const Program& StoreProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop2(VAndB32, 1, InlineInteger(63), 0)
              << Vop1(VMovB32, 2, 9)
              << Vop2(VLshlRevB32, 3, InlineInteger(8), 2)
              << Vop2(VAddNcU32, 3, Vgpr(1), 3)
              << Vop2(VAddNcU32, 3, 8, 3)
              << Mimg(ImageStore, 3, 1, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& AtomicProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop2(VAndB32, 1, InlineInteger(63), 0)
              << Vop1(VMovB32, 2, 9)
              << Vop1(VMovB32, 3, Literal) << Added
              << Mimg(ImageAtomicAdd, 3, 1, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

std::array<std::uint32_t, 8> Image(std::uint64_t address) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((Width - 1u) & 3u) << 30u),
        ((Width - 1u) >> 2u) | ((Height - 1u) << 14u),
        IdentitySwizzle | (Image2D << 28u),
        0u, 0u, 0u, 0u,
    };
}

std::uint32_t Seed(std::uint32_t round, std::uint32_t image) {
    return (round + 1u) * 0x1000000u + image * 0x10000u;
}

void Round(std::vector<std::unique_ptr<PrivatePages>>& surfaces, std::uint32_t round, bool atomics) {
    const auto what = "round " + std::to_string(round) + (atomics ? ", stores and atomics" : ", stores");
    const auto layout = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format32UInt, Width, Height, 1u).at(0);
    for (auto& surface : surfaces) {
        for (std::uint64_t word = 0; word < surface->Bytes() / 4u; ++word) surface->Words()[word] = Sentinel;
    }
    Commands commands;
    for (std::uint32_t image = 0; image < Images; ++image) {
        std::array<std::uint32_t, 9> userData{};
        const auto descriptor = Image(surfaces[image]->Address());
        std::copy(descriptor.begin(), descriptor.end(), userData.begin());
        userData[8] = Seed(round, image);
        commands.Dispatch(StoreProgram(), userData, Width, Height);
        if (atomics) {
            commands.Barrier();
            commands.Dispatch(AtomicProgram(), userData, Width, Height);
        }
    }
    std::string failure;
    StderrCapture capture;
    try {
        commands.Submit(0);
        WaitIdle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    const auto log = capture.Finish();
    Expect(failure.empty() && log.find("failed") == std::string::npos, what + ": the submission failed: " + failure + "\n" + log);
    for (std::uint32_t image = 0; image < Images; ++image) {
        AgcDriver::GuestMemory::FlushGpuWrites(surfaces[image]->Address(), surfaces[image]->Bytes());
        std::uint32_t wrong = 0;
        std::string first;
        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                std::uint32_t value = 0;
                std::memcpy(&value, surfaces[image]->Data() + layout.tiledOffset + std::uint64_t{y} * layout.pitchBytes + x * 4u, 4u);
                const auto wanted = Seed(round, image) + x + 256u * y + (atomics ? Added : 0u);
                if (value == wanted) continue;
                if (wrong++ == 0u) first = "texel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + Hex(value) + ", expected " + Hex(wanted);
            }
        }
        Expect(wrong == 0u, what + ": image " + std::to_string(image) + ", first used in the submission, holds " + std::to_string(wrong) + " wrong texels, first " + first);
    }
}

void Run() {
    std::vector<std::unique_ptr<PrivatePages>> surfaces;
    for (std::uint32_t image = 0; image < Images; ++image) surfaces.push_back(std::make_unique<PrivatePages>(1));
    Round(surfaces, 0, false);
    std::vector<std::unique_ptr<PrivatePages>> fresh;
    for (std::uint32_t image = 0; image < Images; ++image) fresh.push_back(std::make_unique<PrivatePages>(1));
    Round(fresh, 1, true);
    Round(surfaces, 2, true);
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
        std::fprintf(stderr, "%d storage views first use checks failed\n", failures);
        return 1;
    }
    std::printf("storage views first use tests passed\n");
    return 0;
#else
    std::printf("skipped, guest pages are Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
