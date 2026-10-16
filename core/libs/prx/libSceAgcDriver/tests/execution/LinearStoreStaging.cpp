#include "MirrorChecks.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using MirrorChecks::Kind;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t ElementBytes = 16;
constexpr std::uint64_t Guard = 0x10000u - 0x40u;
constexpr std::uint32_t SpinLanes = 64;
constexpr std::uint32_t SpinLabelWord = 1024;
constexpr std::uint32_t StoreLabelWord = 2048;
constexpr std::uint32_t CalibrationRounds = 1u << 20u;
constexpr std::uint64_t MaxRounds = std::uint64_t{1} << 31u;
constexpr auto Busy = std::chrono::milliseconds(600);
constexpr auto CalibrationTime = std::chrono::milliseconds(50);
constexpr std::uint32_t Repeats = 1000;

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

constexpr std::array<std::uint32_t, 2> Indexed(std::uint32_t opcode, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t resource) {
    return {0xe0000000u | (opcode << 18u) | 0x2000u, (ZeroOffset << 24u) | ((resource / 4u) << 16u) | (vdata << 8u) | vaddr};
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

const Program& SpinProgram() {
    static const Program program = [] {
        using namespace Isa;
        constexpr std::uint32_t VLshrRevB32 = 0x16u;
        constexpr std::uint32_t VXorB32 = 0x1du;
        constexpr std::uint32_t MoveS4ToS9 = 0xbe890304u;
        constexpr std::uint32_t DecrementS9 = 0x80898109u;
        constexpr std::uint32_t CompareS9NotZero = 0xbf078009u;
        constexpr std::uint32_t BranchBackWhileNotZero = 0xbf85fff7u;
        Program built;
        built << Vop2(VAddNcU32, 1, InlineInteger(1), 0) << MoveS4ToS9
              << Vop2(VLshlRevB32, 2, InlineInteger(13), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << Vop2(VLshrRevB32, 2, InlineInteger(17), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << Vop2(VLshlRevB32, 2, InlineInteger(5), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
              << DecrementS9 << CompareS9NotZero << BranchBackWhileNotZero
              << Vop2(VLshlRevB32, 3, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 1, 3, 0) << EndProgram;
        return built;
    }();
    return program;
}

std::uint32_t Previous(std::uint64_t word) {
    return 0x3c000000u | static_cast<std::uint32_t>(word & 0x00ffffffu);
}

std::uint32_t Written(std::uint64_t word) {
    return 0xc5000000u | static_cast<std::uint32_t>(word & 0x00ffffffu);
}

struct Value {
    std::array<std::uint32_t, 4> words;
};

constexpr Value First{{0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u}};
constexpr Value Second{{0x55555555u, 0x66666666u, 0x77777777u, 0x88888888u}};

class Target {
public:
    Target(Kind kind, std::uint32_t records) : records(records), pages(kind, static_cast<std::uint32_t>((Guard + std::uint64_t{records} * ElementBytes + Guard + PageBytes - 1u) / PageBytes)), values(1) {
        for (std::uint64_t word = 0; word < pages.Bytes() / 4u; ++word) pages.Words()[word] = Previous(word);
        for (std::uint32_t word = 0; word < 4u; ++word) {
            values.Words()[word] = First.words[word];
            values.Words()[4u + word] = Second.words[word];
        }
        expected.resize(static_cast<std::size_t>(pages.Bytes() / 4u));
        for (std::size_t word = 0; word < expected.size(); ++word) expected[word] = Previous(word);
        AgcDriver::GuestMemory::BumpCollectEpoch();
        Settle();
    }

    std::uint64_t Base() const { return pages.Address(Guard); }
    std::uint32_t Records() const { return records; }
    Kind MemoryKind() const { return pages.MemoryKind(); }

    void ExpectHeld(std::uint64_t elements, const AgcDriver::Graphics::ResidencyCounts& before, const std::string& what) const {
        const std::array<MirrorChecks::Range, 1> stored{{{Base(), Base() + elements * ElementBytes}}};
        const auto failure = MirrorChecks::StoresHeld(MemoryKind(), {pages.Address(), pages.Address() + pages.Bytes()}, stored, stored, before);
        Expect(failure.empty(), what + ": " + failure);
    }

    void Dispatch(Commands& commands, std::uint32_t groups, const Value& value) {
        const auto buffer = BufferResource(Base(), records, ElementBytes);
        const auto source = values.Address(&value == &First ? 0u : 16u);
        const std::array<std::uint32_t, 8> userData{buffer[0], buffer[1], buffer[2], buffer[3], Low(source), High(source), 0u, 0u};
        commands.Dispatch(FillProgram(), userData, Lanes, groups);
        for (std::uint64_t element = 0; element < std::uint64_t{groups} * Lanes; ++element) {
            for (std::uint32_t word = 0; word < 4u; ++word) expected[static_cast<std::size_t>((Guard + element * ElementBytes) / 4u + word)] = value.words[word];
        }
    }

    void CpuWrite(std::uint64_t firstElement, std::uint64_t elements) {
        for (std::uint64_t word = (Guard + firstElement * ElementBytes) / 4u; word < (Guard + (firstElement + elements) * ElementBytes) / 4u; ++word) {
            pages.Words()[word] = Written(word);
            expected[static_cast<std::size_t>(word)] = Written(word);
        }
    }

    void Check(const std::string& what) const {
        AgcDriver::GuestMemory::FlushGpuWrites(pages.Address(), pages.Bytes());
        std::size_t wrong = 0;
        std::string first;
        for (std::size_t word = 0; word < expected.size(); ++word) {
            const auto actual = pages.Words()[word];
            if (actual == expected[word]) continue;
            if (wrong++ == 0u) {
                const auto offset = static_cast<std::int64_t>(word * 4u) - static_cast<std::int64_t>(Guard);
                first = "byte " + std::to_string(offset) + " of the V# range (element " + std::to_string(offset / static_cast<std::int64_t>(ElementBytes)) + ") holds " + Hex(actual) + ", expected " + Hex(expected[word]);
            }
        }
        Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " words differ, first " + first);
    }

    PrivatePages& Scratch() { return scratch; }

private:
    std::uint32_t records;
    MirrorChecks::Pages pages;
    PrivatePages values;
    PrivatePages scratch{1};
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

void Spin(Commands& commands, const PrivatePages& scratch, std::uint32_t rounds) {
    const auto buffer = BufferResource(scratch.Address(), SpinLanes * 4u);
    const std::array<std::uint32_t, 5> userData{buffer[0], buffer[1], buffer[2], buffer[3], rounds};
    commands.Dispatch(SpinProgram(), userData, SpinLanes);
}

std::uint32_t Calibrate() {
    PrivatePages scratch(1);
    for (std::uint64_t measured = 0;; measured = measured == 0u ? CalibrationRounds : measured * 4u) {
        Commands commands;
        Spin(commands, scratch, measured == 0u ? 1u : static_cast<std::uint32_t>(measured));
        const auto start = Clock::now();
        commands.Submit(0);
        WaitIdle();
        Settle();
        const auto took = Clock::now() - start;
        if (measured == 0u || (took < CalibrationTime && measured < MaxRounds / 4u)) continue;
        const auto microseconds = std::max<long long>(std::chrono::duration_cast<std::chrono::microseconds>(took).count(), 1);
        const auto rounds = measured * static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Busy).count()) / static_cast<std::uint64_t>(microseconds);
        return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(rounds, CalibrationRounds, MaxRounds));
    }
}

bool LabelPending(std::uint64_t address) {
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr && recorder->LabelPendingAt(address);
}

void Exact(Kind kind, std::uint32_t groups, std::uint32_t fewer, std::uint32_t tail) {
    Target target(kind, groups * Lanes + tail);
    const std::string shape = std::to_string(groups) + " groups of 64 lanes storing 16-byte elements by global thread id through a V# of " + std::to_string(target.Records()) + " records into " + MirrorChecks::NameOf(kind);
    {
        const auto before = MirrorChecks::Counts();
        Commands commands;
        target.Dispatch(commands, groups, First);
        const auto failure = Submitted(commands);
        Expect(failure.empty(), shape + ": the dispatch failed: " + failure);
        target.ExpectHeld(std::uint64_t{groups} * Lanes, before, shape);
        target.Check(shape);
    }
    {
        const auto before = MirrorChecks::Counts();
        Commands commands;
        target.Dispatch(commands, fewer, Second);
        const auto failure = Submitted(commands);
        const std::string what = shape + ", then " + std::to_string(fewer) + " groups storing another value";
        Expect(failure.empty(), what + ": the dispatch failed: " + failure);
        target.ExpectHeld(std::uint64_t{fewer} * Lanes, before, what);
        target.Check(what);
    }
    {
        const auto before = MirrorChecks::Counts();
        Commands commands;
        for (std::uint32_t repeat = 0; repeat < Repeats; ++repeat) target.Dispatch(commands, groups, repeat % 2u == 0u ? First : Second);
        const auto failure = Submitted(commands);
        const std::string what = shape + ", then " + std::to_string(Repeats) + " such dispatches in one submission storing alternating values";
        Expect(failure.empty(), what + ": the submission failed: " + failure);
        target.ExpectHeld(std::uint64_t{groups} * Lanes, before, what);
        target.Check(what);
    }
}

void TailWrittenBehindSpin(Kind kind, std::uint32_t rounds, std::uint32_t groups, std::uint32_t tail) {
    Target target(kind, groups * Lanes + tail);
    auto& scratch = target.Scratch();
    scratch.Words()[SpinLabelWord] = 0u;
    scratch.Words()[StoreLabelWord] = 0u;
    const auto spinLabel = scratch.Address(SpinLabelWord * 4u);
    const auto storeLabel = scratch.Address(StoreLabelWord * 4u);
    const std::string what = std::to_string(groups) + " groups storing 16-byte elements by global thread id through a V# of " + std::to_string(target.Records()) + " records of " + MirrorChecks::NameOf(kind) + " behind a GPU spin, while the CPU writes the " + std::to_string(tail) + " records past them";
    const std::uint64_t stored = std::uint64_t{groups} * Lanes;
    const auto storedEnd = target.Base() + stored * ElementBytes;
    const auto boundary = std::min<std::uint64_t>(stored + tail, stored + (((storedEnd + SubpageBytes - 1u) & ~(SubpageBytes - 1u)) - storedEnd) / ElementBytes);
    const auto before = MirrorChecks::Counts();
    Commands commands;
    Spin(commands, scratch, rounds);
    commands.WriteData(spinLabel, {1u});
    target.Dispatch(commands, groups, First);
    commands.WriteData(storeLabel, {1u});
    std::string failure;
    bool recorded = false;
    bool running = false;
    {
        StderrCapture capture;
        try {
            commands.Submit(0);
            for (const auto deadline = Clock::now() + std::chrono::seconds(10); !recorded && Clock::now() < deadline;) {
                if (LabelPending(storeLabel) || scratch.Words()[StoreLabelWord] == 1u) {
                    recorded = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            target.CpuWrite(boundary, stored + tail - boundary);
            running = scratch.Words()[SpinLabelWord] == 0u;
            if (kind == Kind::Private) target.CpuWrite(stored, boundary - stored);
            WaitIdle();
            Settle();
        } catch (const std::exception& error) {
            failure = error.what();
        }
        const auto log = capture.Finish();
        if (log.find("failed") != std::string::npos) failure += "\n" + log;
    }
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    Expect(recorded, what + ": the dispatch's label was never recorded");
    Expect(running, what + ": the spin had finished when the CPU wrote the records past the stores' last 4 KiB subpage, so the writes did not meet a running batch");
    target.ExpectHeld(stored, before, what);
    if (kind == Kind::Shared) target.CpuWrite(stored, boundary - stored);
    target.Check(what);
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
    const std::array<std::pair<const char*, void (*)(Kind)>, 4> cases{{
        {"exact stores of 16 groups", [](Kind kind) { Exact(kind, 16u, 5u, 300u); }},
        {"exact stores of 512 groups", [](Kind kind) { Exact(kind, 512u, 200u, 1000u); }},
        {"tail written behind a spin, 16 groups", [](Kind kind) { TailWrittenBehindSpin(kind, Calibrate(), 16u, 512u); }},
        {"tail written behind a spin, 512 groups", [](Kind kind) { TailWrittenBehindSpin(kind, Calibrate(), 512u, 1000u); }},
    }};
    for (const auto kind : {Kind::Shared, Kind::Private}) {
        for (const auto& [name, run] : cases) {
            try {
                run(kind);
            } catch (const std::exception& error) {
                Expect(false, std::string(name) + " in " + MirrorChecks::NameOf(kind) + ": exception: " + error.what());
            }
        }
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d linear store checks failed\n", failures);
        return 1;
    }
    std::printf("linear store tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
