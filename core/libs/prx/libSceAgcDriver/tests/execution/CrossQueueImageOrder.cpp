#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <exception>
#include <string>

#ifdef _WIN32

namespace {

using namespace NewPath;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t Texels = Width * Height;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Rounds = 6;
constexpr std::uint32_t WaitEqual = 3;
constexpr std::uint32_t Untouched = 0xdeadbeefu;

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
constexpr std::uint32_t ImageStore = 0x08u;
constexpr std::array<std::uint32_t, 2> LoadDescriptor{0xf40c0300u, 0xfa000000u};
constexpr std::uint32_t WaitScalar = 0xbf8cc07fu;

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

const Program& ReadProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << LoadDescriptor
              << Vop2(VAndB32, 1, InlineInteger(63), 0)
              << Vop1(VMovB32, 2, 8)
              << Vop2(VLshlRevB32, 3, InlineInteger(6), 2)
              << Vop2(VAddNcU32, 3, Vgpr(1), 3)
              << Vop2(VLshlRevB32, 3, InlineInteger(2), 3)
              << WaitScalar
              << Mimg(ImageLoad, 4, 1, 12)
              << WaitAll
              << Mubuf(BufferStoreDword, 4, 3, 4)
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

std::uint32_t Seed(std::uint32_t round) {
    return (round + 1u) << 20u;
}

void Run() {
    PrivatePages texels(1);
    PrivatePages descriptorWords(1);
    PrivatePages output(1);
    PrivatePages labels(1);
    const auto image = Image(texels.Address());
    std::memcpy(descriptorWords.Data(), image.data(), sizeof(image));
    std::array<std::uint32_t, 9> storeUserData{};
    std::copy(image.begin(), image.end(), storeUserData.begin());
    const auto outputBuffer = BufferResource(output.Address(), Texels * 4u);
    const std::array<std::uint32_t, 8> readUserData{Low(descriptorWords.Address()), High(descriptorWords.Address()), 0u, 0u, outputBuffer[0], outputBuffer[1], outputBuffer[2], outputBuffer[3]};
    const auto read = [&](std::uint32_t queue, std::optional<std::uint32_t> label) {
        Commands commands;
        if (label) commands.Wait(labels.Address(), WaitEqual, *label, 0xffffffffu);
        commands.Dispatch(ReadProgram(), readUserData, Width, Height);
        commands.Submit(queue);
    };
    for (std::uint32_t texel = 0; texel < Texels; ++texel) texels.Words()[texel] = 0x0f000000u + texel;
    for (std::uint32_t warm = 0; warm < 2u; ++warm) {
        read(1, std::nullopt);
        WaitIdle();
    }
    for (std::uint32_t round = 0; round < Rounds; ++round) {
        const auto what = "round " + std::to_string(round);
        for (std::uint32_t word = 0; word < Texels; ++word) output.Words()[word] = Untouched;
        AgcDriver::GuestMemory::CollectWritesUncached(output.Address(), output.Bytes());
        storeUserData[8] = Seed(round);
        std::string failure;
        StderrCapture capture;
        try {
            Commands writer;
            writer.Dispatch(StoreProgram(), storeUserData, Width, Height);
            writer.Barrier();
            writer.WriteData(labels.Address(), {round + 1u});
            writer.Submit(0);
            read(1, round + 1u);
            WaitIdle();
        } catch (const std::exception& error) {
            failure = error.what();
        }
        const auto log = capture.Finish();
        Expect(failure.empty() && log.find("failed") == std::string::npos, what + ": the submissions failed: " + failure + "\n" + log);
        AgcDriver::GuestMemory::FlushGpuWrites(output.Address(), output.Bytes());
        std::uint32_t wrong = 0;
        std::string first;
        for (std::uint32_t texel = 0; texel < Texels; ++texel) {
            const auto wanted = Seed(round) + (texel % Width) + 256u * (texel / Width);
            const auto loaded = output.Words()[texel];
            if (loaded == wanted) continue;
            if (wrong++ == 0u) first = "texel " + std::to_string(texel) + " read " + Hex(loaded) + ", expected " + Hex(wanted);
        }
        Expect(wrong == 0u, what + ": a dispatch on queue 1 waiting for the label queue 0 writes after its image store read " + std::to_string(wrong) + " texels the store had not written, first " + first);
    }
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
        std::fprintf(stderr, "%d cross-queue image order checks failed\n", failures);
        return 1;
    }
    std::printf("cross-queue image order tests passed\n");
    return 0;
#else
    std::printf("skipped, guest pages are Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
