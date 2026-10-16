#include "MirrorChecks.hpp"
#include "VulkanTestDevice.hpp"
#include <exception>
#include <string>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using MirrorChecks::Kind;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t ElementBytes = 4;
constexpr std::uint32_t Tail = 100;
constexpr std::uint64_t Guard = 0x10000u - 0x40u;
constexpr std::uint32_t Chained = 100;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

namespace Ops {

using namespace Isa;

constexpr std::uint32_t VLshlAddU32 = 0x346u;
constexpr std::uint32_t VXorB32 = 0x1du;

constexpr std::array<std::uint32_t, 2> Vop3(std::uint32_t opcode, std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1, std::uint32_t src2) {
    return {0xd4000000u | (opcode << 16u) | vdst, src0 | (src1 << 9u) | (src2 << 18u)};
}

constexpr std::array<std::uint32_t, 2> Indexed(std::uint32_t opcode, std::uint32_t vdata, std::uint32_t vindex, std::uint32_t resource) {
    return {0xe0000000u | (opcode << 18u) | 0x2000u, (ZeroOffset << 24u) | ((resource / 4u) << 16u) | (vdata << 8u) | vindex};
}

}

const Program& IncrementProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop3(VLshlAddU32, 4, 8, InlineInteger(6), Vgpr(0))
              << Indexed(BufferLoadDword, 5, 4, 0)
              << WaitAll
              << Vop2(VAddNcU32, 5, InlineInteger(1), 5)
              << Indexed(BufferStoreDword, 5, 4, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& NeighbourProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop3(VLshlAddU32, 4, 8, InlineInteger(6), Vgpr(0))
              << Vop2(VXorB32, 6, InlineInteger(1), 4)
              << Indexed(BufferLoadDword, 5, 6, 0)
              << WaitAll
              << Vop2(VAddNcU32, 5, InlineInteger(1), 5)
              << Indexed(BufferStoreDword, 5, 4, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

std::uint32_t Previous(std::uint64_t word) {
    return 0x2b000000u | static_cast<std::uint32_t>((word * 2654435761u) & 0x00ffffffu);
}

class Target {
public:
    Target(Kind kind, std::uint32_t groups) : groups(groups), records(groups * Lanes + Tail), pages(kind, static_cast<std::uint32_t>((Guard + std::uint64_t{records} * ElementBytes + Guard + PageBytes - 1u) / PageBytes)) {
        expected.resize(static_cast<std::size_t>(pages.Bytes() / 4u));
        for (std::size_t word = 0; word < expected.size(); ++word) {
            pages.Words()[word] = Previous(word);
            expected[word] = Previous(word);
        }
        AgcDriver::GuestMemory::BumpCollectEpoch();
        Settle();
    }

    void ExpectHeld(const AgcDriver::Graphics::ResidencyCounts& before, const std::string& what) const {
        const auto base = pages.Address(Guard);
        const std::array<MirrorChecks::Range, 1> stored{{{base, base + std::uint64_t{groups} * Lanes * ElementBytes}}};
        const std::array<MirrorChecks::Range, 1> range{{{base, base + std::uint64_t{records} * ElementBytes}}};
        const auto failure = MirrorChecks::StoresHeld(pages.MemoryKind(), {pages.Address(), pages.Address() + pages.Bytes()}, stored, range, before);
        Expect(failure.empty(), what + ": " + failure);
    }

    void Dispatch(Commands& commands, const Program& program) {
        const auto buffer = BufferResource(pages.Address(Guard), records, ElementBytes);
        const std::array<std::uint32_t, 8> userData{buffer[0], buffer[1], buffer[2], buffer[3], 0u, 0u, 0u, 0u};
        commands.Dispatch(program, userData, Lanes, groups);
        const auto first = static_cast<std::size_t>(Guard / 4u);
        const std::vector<std::uint32_t> before(expected.begin() + static_cast<std::ptrdiff_t>(first), expected.begin() + static_cast<std::ptrdiff_t>(first + groups * Lanes));
        for (std::uint32_t element = 0; element < groups * Lanes; ++element) {
            const auto read = &program == &NeighbourProgram() ? element ^ 1u : element;
            expected[first + element] = before[read] + 1u;
        }
    }

    void Check(const std::string& what) const {
        AgcDriver::GuestMemory::FlushGpuWrites(pages.Address(), pages.Bytes());
        std::size_t wrong = 0;
        std::string firstWrong;
        for (std::size_t word = 0; word < expected.size(); ++word) {
            const auto actual = pages.Words()[word];
            if (actual == expected[word]) continue;
            if (wrong++ == 0u) {
                const auto element = static_cast<std::int64_t>(word) - static_cast<std::int64_t>(Guard / 4u);
                firstWrong = "element " + std::to_string(element) + " holds " + Hex(actual) + ", expected " + Hex(expected[word]);
            }
        }
        Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " words differ, first " + firstWrong);
    }

    std::uint32_t Records() const { return records; }

private:
    std::uint32_t groups;
    std::uint32_t records;
    MirrorChecks::Pages pages;
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

void Run(Target& target, const Program& program, std::uint32_t dispatches, const std::string& what) {
    const auto before = MirrorChecks::Counts();
    Commands commands;
    for (std::uint32_t dispatch = 0; dispatch < dispatches; ++dispatch) target.Dispatch(commands, program);
    const auto failure = Submitted(commands);
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    target.ExpectHeld(before, what + ", the stores and the reads of every dispatch through the one copy");
    target.Check(what);
}

void Reads(Kind kind, std::uint32_t groups) {
    Target target(kind, groups);
    const std::string shape = std::to_string(groups) + " groups of 64 lanes storing one dword per global thread id through a V# of " + std::to_string(target.Records()) + " records of " + MirrorChecks::NameOf(kind);
    Run(target, IncrementProgram(), 1u, shape + ", each lane storing the dword it loaded from its own element plus one");
    Run(target, IncrementProgram(), Chained, shape + ", " + std::to_string(Chained) + " such dispatches in one submission");
    Run(target, NeighbourProgram(), 1u, shape + ", each lane storing the dword it loaded from its neighbour's element plus one");
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
        for (const auto groups : {16u, 512u}) {
            try {
                Reads(kind, groups);
            } catch (const std::exception& error) {
                Expect(false, std::to_string(groups) + " groups of " + MirrorChecks::NameOf(kind) + ": exception: " + error.what());
            }
        }
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d linear store read checks failed\n", failures);
        return 1;
    }
    std::printf("linear store read tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
