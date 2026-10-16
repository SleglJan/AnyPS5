#include "MirrorChecks.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "VulkanTestDevice.hpp"
#include <exception>
#include <string>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using MirrorChecks::Kind;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Groups = 16;
constexpr std::uint32_t Elements = Groups * Lanes;
constexpr std::uint32_t Records = Elements + 64;
constexpr std::uint32_t ElementBytes = 16;
constexpr std::uint32_t Ranges = 100;
constexpr std::uint64_t RangeStride = 0x8000;
constexpr std::uint64_t RangeOffset = 0x40;
constexpr std::uint32_t AreaPages = static_cast<std::uint32_t>((Ranges * RangeStride + PageBytes - 1u) / PageBytes);

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

std::uint32_t Previous(std::uint64_t word) {
    return 0x5e000000u | static_cast<std::uint32_t>(word & 0x00ffffffu);
}

std::uint32_t ValueWord(std::uint32_t round, std::uint32_t range, std::uint32_t word) {
    return 0x80000000u | (round << 24u) | (range << 8u) | word;
}

class Area {
public:
    explicit Area(Kind kind) : pages(kind, AreaPages), values(static_cast<std::uint32_t>((std::uint64_t{Ranges} * 2u * 16u + PageBytes - 1u) / PageBytes)) {
        expected.resize(static_cast<std::size_t>(pages.Bytes() / 4u));
        for (std::size_t word = 0; word < expected.size(); ++word) {
            pages.Words()[word] = Previous(word);
            expected[word] = Previous(word);
        }
        for (std::uint32_t round = 0; round < 2u; ++round) {
            for (std::uint32_t range = 0; range < Ranges; ++range) {
                for (std::uint32_t word = 0; word < 4u; ++word) values.Words()[(round * Ranges + range) * 4u + word] = ValueWord(round, range, word);
            }
        }
        AgcDriver::GuestMemory::BumpCollectEpoch();
        Settle();
    }

    Kind MemoryKind() const { return pages.MemoryKind(); }

    void ExpectHeld(const AgcDriver::Graphics::ResidencyCounts& before, const std::string& what) const {
        std::vector<MirrorChecks::Range> stored;
        for (std::uint32_t range = 0; range < Ranges; ++range) {
            const auto base = pages.Address(range * RangeStride + RangeOffset);
            stored.emplace_back(base, base + std::uint64_t{Elements} * ElementBytes);
        }
        const auto failure = MirrorChecks::StoresHeld(MemoryKind(), {pages.Address(), pages.Address() + pages.Bytes()}, stored, stored, before);
        Expect(failure.empty(), what + ": " + failure);
    }

    void Dispatch(Commands& commands, std::uint32_t round, std::uint32_t range) {
        const auto base = pages.Address(range * RangeStride + RangeOffset);
        const auto buffer = BufferResource(base, Records, ElementBytes);
        const auto value = values.Address(std::uint64_t{round * Ranges + range} * 16u);
        const std::array<std::uint32_t, 8> userData{buffer[0], buffer[1], buffer[2], buffer[3], Low(value), High(value), 0u, 0u};
        commands.Dispatch(FillProgram(), userData, Lanes, Groups);
        const auto first = static_cast<std::size_t>((range * RangeStride + RangeOffset) / 4u);
        for (std::uint32_t element = 0; element < Elements; ++element) {
            for (std::uint32_t word = 0; word < 4u; ++word) expected[first + element * 4u + word] = ValueWord(round, range, word);
        }
    }

    void Check(const std::string& what) const {
        AgcDriver::GuestMemory::FlushGpuWrites(pages.Address(), pages.Bytes());
        std::size_t wrong = 0;
        std::string firstWrong;
        for (std::size_t word = 0; word < expected.size(); ++word) {
            const auto actual = pages.Words()[word];
            if (actual == expected[word]) continue;
            if (wrong++ == 0u) firstWrong = "byte " + Hex(word * 4u) + " of the area (range " + std::to_string(word * 4u / RangeStride) + ") holds " + Hex(actual) + ", expected " + Hex(expected[word]);
        }
        Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " words differ, first " + firstWrong);
    }

private:
    MirrorChecks::Pages pages;
    PrivatePages values;
    std::vector<std::uint32_t> expected;
};

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
    AgcDriver::Graphics::ResidencyCounts residency;

    static Counters Now() { return {AgcDriver::Graphics::BdaPageTable::Totals().tables, MirrorChecks::Counts()}; }
};

void ExpectNoTables(const Area& area, const Counters& before, const std::string& what) {
    const auto after = Counters::Now();
    area.ExpectHeld(before.residency, what);
    Expect(after.tables == before.tables, what + ": the dispatches built " + std::to_string(after.tables - before.tables) + " page tables over an unchanged address space");
}

void Tables(Kind kind) {
    Area area(kind);
    {
        Commands warm;
        for (std::uint32_t range = 0; range < Ranges; ++range) area.Dispatch(warm, 0u, range);
        const auto failure = Submitted(warm);
        Expect(failure.empty(), "linear store tables: the warm-up submission failed: " + failure);
        area.Check("linear store tables: the warm-up submission");
    }
    const std::string shape = std::to_string(Ranges) + " linear-store dispatches of " + std::to_string(Groups) + " groups, each into its own range of " + MirrorChecks::NameOf(kind);
    {
        const auto before = Counters::Now();
        Commands commands;
        for (std::uint32_t range = 0; range < Ranges; ++range) area.Dispatch(commands, 1u, range);
        const auto failure = Submitted(commands);
        const std::string what = shape + ", in one submission";
        Expect(failure.empty(), what + ": the submission failed: " + failure);
        ExpectNoTables(area, before, what);
        area.Check(what);
    }
    {
        const auto before = Counters::Now();
        std::string failure;
        for (std::uint32_t range = 0; range < Ranges && failure.empty(); ++range) {
            Commands commands;
            area.Dispatch(commands, range % 2u, range);
            failure = Submitted(commands);
        }
        const std::string what = shape + ", each in its own submission";
        Expect(failure.empty(), what + ": a submission failed: " + failure);
        ExpectNoTables(area, before, what);
        area.Check(what);
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
        MirrorChecks::PrepareResidency();
    } catch (const std::exception& error) {
        Expect(false, std::string("residency budget: exception: ") + error.what());
    }
    for (const auto kind : {Kind::Shared, Kind::Private}) {
        try {
            Tables(kind);
        } catch (const std::exception& error) {
            Expect(false, std::string(MirrorChecks::NameOf(kind)) + ": exception: " + error.what());
        }
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d linear store table checks failed\n", failures);
        return 1;
    }
    std::printf("linear store table tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
