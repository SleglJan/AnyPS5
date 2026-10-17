#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <exception>
#include <string>

namespace {

using namespace NewPath;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Records = 4;
constexpr std::uint32_t Untouched = 0xcafebabeu;
constexpr std::uint32_t FirstBase = 0x1000u;
constexpr std::uint32_t SecondBase = 0x2000u;
constexpr std::uint32_t LiveVector = 0x40000u;
constexpr std::uint32_t LiveScalar = 0x123u;
constexpr std::array<std::uint32_t, 2> Arguments{0x30000u, 0x50000u};

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

TestGuest::Array<std::uint32_t, Records * 4> Table;
TestGuest::Array<std::uint32_t, 8> Srt;
TestGuest::Array<std::uint32_t, Threads> Output;
TestGuest::Array<std::uint32_t, 2> ArgumentWords;

namespace Ops {

using namespace Isa;

constexpr std::uint32_t RecordIndex = 0x36020083u;
constexpr std::uint32_t MoveS21 = 0xb0150000u | LiveScalar;
constexpr std::uint32_t SaveExec = 0xbe9e047eu;
constexpr std::uint32_t ReadFirstRecord = 0x7e280501u;
constexpr std::uint32_t CompareRecord = 0x7d840214u;
constexpr std::uint32_t AndSaveExec = 0xbea0246au;
constexpr std::uint32_t RecordOffset = 0x8f148414u;
constexpr std::array<std::uint32_t, 2> LoadTable{0xf4080400u, 0xfa000000u};
constexpr std::uint32_t WaitScalar = 0xbf8cc07fu;
constexpr std::array<std::uint32_t, 2> LoadRecord{0xf4280208u, 0x28000000u};
constexpr std::uint32_t TargetLow = 0xbe8e0308u;
constexpr std::uint32_t TargetHigh = 0xbe8f0309u;
constexpr std::uint32_t SwapPc = 0xbe8e210eu;
constexpr std::uint32_t ClearDone = 0x8afe7e20u;
constexpr std::uint32_t LoopWhileLanes = 0xbf890000u;
constexpr std::uint32_t RestoreExec = 0xbefe041eu;
constexpr std::array<std::uint32_t, 2> LoadOutput{0xf4080600u, 0xfa000010u};
constexpr std::uint32_t LaneOffset = 0x34060082u;
constexpr std::array<std::uint32_t, 2> StoreResult{0xe0701000u, 0x80060203u};
constexpr std::uint32_t Return = 0xbe80200eu;
constexpr std::uint32_t ClobberS22 = 0xbe960380u;
constexpr std::array<std::uint32_t, 2> LoadArgument{0xf40005c5u, 0xfa000000u};

}

const Program& Caller() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << RecordIndex
              << Vop2(VLshlRevB32, 5, InlineInteger(3), 0)
              << Vop2(VAddNcU32, 5, Literal, 5) << LiveVector
              << MoveS21
              << SaveExec;
        const auto loop = built.size;
        built << ReadFirstRecord << CompareRecord << AndSaveExec << RecordOffset << LoadTable << WaitScalar << LoadRecord << WaitScalar
              << TargetLow << TargetHigh << SwapPc << ClearDone;
        const auto next = static_cast<std::int32_t>(built.size + 1u);
        built << (LoopWhileLanes | (static_cast<std::uint32_t>(static_cast<std::int32_t>(loop) - next) & 0xffffu))
              << RestoreExec
              << LoadOutput
              << Vop2(VAddNcU32, 2, Vgpr(2), 5)
              << Vop2(VAddNcU32, 2, 21, 2)
              << LaneOffset
              << WaitScalar
              << StoreResult
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& FirstCallee() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop2(VAddNcU32, 2, Literal, 0) << FirstBase
              << Vop1(VMovB32, 6, InlineInteger(0))
              << ClobberS22
              << Return;
        return built;
    }();
    return program;
}

const Program& SecondCallee() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << LoadArgument
              << Vop2(VAddNcU32, 2, Literal, 0) << SecondBase
              << Vop1(VMovB32, 7, InlineInteger(0))
              << WaitScalar
              << Vop2(VAddNcU32, 2, 23, 2)
              << Return;
        return built;
    }();
    return program;
}

void SetRecord(std::uint32_t record, std::uint64_t target, const void* argument) {
    const auto data = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(argument));
    Table[record * 4u] = Low(target);
    Table[record * 4u + 1u] = High(target);
    Table[record * 4u + 2u] = Low(data);
    Table[record * 4u + 3u] = High(data);
}

std::string Dispatch() {
    Output.fill(Untouched);
    const auto srt = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Srt.data()));
    const std::array<std::uint32_t, 2> userData{Low(srt), High(srt)};
    Commands commands;
    commands.Dispatch(Caller(), userData, Threads);
    std::string failure;
    StderrCapture capture;
    try {
        commands.Submit(0);
        WaitIdle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    try {
        AgcDriver::GuestMemory::FlushGpuWrites(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Output.data())), decltype(Output)::ByteSize);
    } catch (const std::exception& error) {
        failure += error.what();
    }
    const auto log = capture.Finish();
    return log + failure;
}

std::uint32_t Wanted(std::uint32_t lane) {
    const auto record = lane & 3u;
    const auto callee = (record & 1u) == 0u ? FirstBase + lane : SecondBase + lane + Arguments[record >> 1u];
    return callee + (lane << 3u) + LiveVector + LiveScalar;
}

void TwoTargets() {
    SetRecord(0, FirstCallee().Address(), nullptr);
    SetRecord(1, SecondCallee().Address(), &ArgumentWords[0]);
    SetRecord(2, FirstCallee().Address(), nullptr);
    SetRecord(3, SecondCallee().Address(), &ArgumentWords[1]);
    for (std::uint32_t round = 0; round < 2u; ++round) {
        const auto what = "two targets binned by lane, round " + std::to_string(round);
        const auto log = Dispatch();
        Expect(log.find("no linked callee") == std::string::npos && log.find("failed") == std::string::npos, what + ": the dispatch reported a failure: " + log);
        for (std::uint32_t lane = 0; lane < Threads; ++lane) {
            if (Output[lane] == Wanted(lane)) continue;
            Expect(false, what + ": lane " + std::to_string(lane) + " stored " + Hex(Output[lane]) + ", expected " + Hex(Wanted(lane)) + " (callee result plus the caller's live v5 and s21)");
            break;
        }
    }
}

void UnknownTarget() {
    SetRecord(0, FirstCallee().Address(), nullptr);
    SetRecord(1, 0, &ArgumentWords[0]);
    SetRecord(2, FirstCallee().Address(), nullptr);
    SetRecord(3, 0, &ArgumentWords[1]);
    const std::string what = "an unknown target beside a known one";
    const auto log = Dispatch();
    Expect(log.find("called address 0x0") != std::string::npos, what + ": no fault record names the unknown target: " + log);
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        const auto stored = Output[lane];
        const bool unknown = (lane & 1u) != 0u;
        if (unknown && stored == Untouched) continue;
        if (!unknown && (stored == Untouched || stored == Wanted(lane))) continue;
        Expect(false, what + ": lane " + std::to_string(lane) + " stored " + Hex(stored) + (unknown ? " after calling the unknown target" : ", expected its result or nothing"));
        break;
    }
}

}

int main() {
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    try {
        ArgumentWords[0] = Arguments[0];
        ArgumentWords[1] = Arguments[1];
        const auto table = BufferResource(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Table.data())), static_cast<std::uint32_t>(decltype(Table)::ByteSize));
        const auto output = BufferResource(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Output.data())), static_cast<std::uint32_t>(decltype(Output)::ByteSize));
        std::copy(table.begin(), table.end(), Srt.begin());
        std::copy(output.begin(), output.end(), Srt.begin() + 4);
        TwoTargets();
        UnknownTarget();
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        Expect(false, std::string("shutdown: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d table call continuation checks failed\n", failures);
        return 1;
    }
    std::printf("table call continuation tests passed\n");
    return 0;
}
