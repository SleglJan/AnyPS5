#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "VulkanTestDevice.hpp"
#include <exception>
#include <string>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t BufferPages = MirrorReader::Pages;
constexpr std::uint32_t ElementBytes = 16;
constexpr std::uint32_t Elements = static_cast<std::uint32_t>(BufferPages * PageBytes / ElementBytes);
constexpr std::uint32_t Groups = Elements / Lanes;
constexpr std::uint32_t WarmRounds = 4;
constexpr std::uint32_t Rounds = 16;
constexpr std::uint32_t AllRounds = WarmRounds + Rounds;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

namespace Ops {

using namespace Isa;

constexpr std::uint32_t VMadU32U24 = 0x143u;
constexpr std::uint32_t BufferStoreDwordX4 = 0x1eu;

constexpr std::array<std::uint32_t, 2> Vop3(std::uint32_t opcode, std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1, std::uint32_t src2) {
    return {0xd4000000u | (opcode << 16u) | vdst, src0 | (src1 << 9u) | (src2 << 18u)};
}

constexpr std::array<std::uint32_t, 2> Indexed(std::uint32_t opcode, std::uint32_t vdata, std::uint32_t vindex, std::uint32_t resource) {
    return {0xe0000000u | (opcode << 18u) | 0x2000u, (ZeroOffset << 24u) | ((resource / 4u) << 16u) | (vdata << 8u) | vindex};
}

}

const Program& FillProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop3(VMadU32U24, 4, 8, InlineInteger(64), Vgpr(0))
              << Smem(SLoadDwordX4, 12, 4, 0)
              << WaitAll
              << Vop1(VMovB32, 0, 12)
              << Vop1(VMovB32, 1, 13)
              << Vop1(VMovB32, 2, 14)
              << Vop1(VMovB32, 3, 15)
              << Indexed(BufferStoreDwordX4, 0, 4, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

std::uint32_t ValueWord(std::uint32_t round, std::uint32_t word) {
    return 0x80000000u | (round << 8u) | word;
}

std::string Submitted(Commands& commands) {
    std::string failure;
    StderrCapture capture;
    try {
        commands.Submit(0);
        WaitIdle();
        Settle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    const auto log = capture.Finish();
    if (log.find("failed") != std::string::npos) failure += "\n" + log;
    return failure;
}

struct Counters {
    std::uint64_t tables = 0;
    std::uint64_t admitted = 0;
    std::uint64_t demoted = 0;

    static Counters Now() {
        Counters counters;
        counters.tables = AgcDriver::Graphics::BdaPageTable::Totals().tables;
        Locked([&] {
            const auto counts = Residency().Manager().Counts();
            counters.admitted = counts.admitted;
            counters.demoted = counts.demoted;
        });
        return counters;
    }
};

template<typename TBuffer>
void SteadyState(TBuffer& buffer, const std::string& memory) {
    PrivatePages values(1);
    MirrorReader reader;
    for (std::uint64_t word = 0; word < buffer.Bytes() / 4u; ++word) buffer.Words()[word] = 0x5e000000u | static_cast<std::uint32_t>(word & 0x00ffffffu);
    for (std::uint32_t round = 0; round < AllRounds; ++round) {
        for (std::uint32_t word = 0; word < 4u; ++word) values.Words()[round * 4u + word] = ValueWord(round, word);
    }
    AgcDriver::GuestMemory::BumpCollectEpoch();
    const std::string shape = "a dispatch storing a 16-byte value into every element of a " + std::to_string(BufferPages) + "-page buffer in " + memory + ", then a dispatch reading one word from every 4 KiB of the buffer, each in its own submission";
    Counters before;
    for (std::uint32_t round = 0; round < AllRounds; ++round) {
        if (round == WarmRounds) before = Counters::Now();
        const std::string what = shape + ", round " + std::to_string(round);
        Commands writes;
        const auto resource = BufferResource(buffer.Address(), Elements, ElementBytes);
        const auto value = values.Address(std::uint64_t{round} * 16u);
        const std::array<std::uint32_t, 8> userData{resource[0], resource[1], resource[2], resource[3], Low(value), High(value), 0u, 0u};
        writes.Dispatch(FillProgram(), userData, Lanes, Groups);
        const auto written = Submitted(writes);
        Expect(written.empty(), what + ": the storing submission failed: " + written);
        Commands reads;
        reader.Record(reads, buffer.Address(), BufferPages);
        const auto read = Submitted(reads);
        Expect(read.empty(), what + ": the reading submission failed: " + read);
        const auto lanes = reader.Collect();
        std::size_t wrong = 0;
        std::string first;
        for (std::uint32_t lane = 0; lane < MirrorReader::Lanes; ++lane) {
            if (lanes[lane] == ValueWord(round, 0) || wrong++ != 0u) continue;
            first = "the lane reading byte " + Hex(std::uint64_t{lane} * 4096u) + " read " + Hex(lanes[lane]) + ", expected " + Hex(ValueWord(round, 0));
        }
        Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(MirrorReader::Lanes) + " reading lanes read the wrong word, first " + first);
    }
    const auto after = Counters::Now();
    Expect(after.tables == before.tables, shape + ": the " + std::to_string(Rounds) + " rounds after warm-up built " + std::to_string(after.tables - before.tables) + " page tables over an unchanged address space, while the residency admitted " + std::to_string(after.admitted - before.admitted) + " pages and demoted " + std::to_string(after.demoted - before.demoted));
    AgcDriver::GuestMemory::FlushGpuWrites(buffer.Address(), buffer.Bytes());
    std::size_t wrong = 0;
    std::string first;
    for (std::uint64_t word = 0; word < buffer.Bytes() / 4u; ++word) {
        const auto expected = ValueWord(AllRounds - 1u, static_cast<std::uint32_t>(word % 4u));
        const auto actual = buffer.Words()[word];
        if (actual == expected || wrong++ != 0u) continue;
        first = "byte " + Hex(word * 4u) + " holds " + Hex(actual) + ", expected " + Hex(expected);
    }
    Expect(wrong == 0u, shape + ": after the last round " + std::to_string(wrong) + " words of the buffer differ, first " + first);
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
        {
            MirrorReader reader;
            PrivatePages warm(1);
            reader.Read(warm.Address(), 1);
        }
        {
            PrivatePages buffer(BufferPages);
            SteadyState(buffer, "private memory");
        }
        {
            SharedPages buffer(BufferPages);
            SteadyState(buffer, "a shared mapping");
        }
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d page table steady state checks failed\n", failures);
        return 1;
    }
    std::printf("page table steady state tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
