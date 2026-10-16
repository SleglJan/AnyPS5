#include "MirrorChecks.hpp"
#include "VulkanTestDevice.hpp"
#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32

namespace {

using namespace NewPath;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Groups = 256;
constexpr std::uint32_t Elements = Groups * Lanes;
constexpr std::uint32_t Tail = 512;
constexpr std::uint64_t Guard = 0x10000u - 0x40u;
constexpr std::uint32_t ElementBytes = 16;
constexpr std::uint64_t RangeBytes = std::uint64_t{Elements + Tail} * ElementBytes;
constexpr std::uint32_t StrideShift = 6;
constexpr std::uint32_t QueueA = 0x20;
constexpr std::uint32_t QueueB = 0x28;
constexpr std::uint32_t Equal = 3;
constexpr std::uint32_t CopyLanes = 64;
constexpr auto LabelLimit = std::chrono::seconds(10);

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

const Program& LinearProgram() {
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

const Program& StridedProgram() {
    static const Program program = [] {
        using namespace Ops;
        Program built;
        built << Vop3(VMadU32U24, 4, 8, InlineInteger(64), Vgpr(0))
              << Vop2(VLshlRevB32, 5, InlineInteger(StrideShift), 4)
              << Vop2(VAddNcU32, 6, Literal, 4) << 0x51000000u
              << Mubuf(BufferStoreDword, 6, 5, 0)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& CopyProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop1(VMovB32, 1, 8)
              << Vop2(VLshlRevB32, 1, InlineInteger(6), 1)
              << Vop2(VAddNcU32, 1, Vgpr(0), 1)
              << Vop2(VLshlRevB32, 1, InlineInteger(2), 1)
              << Mubuf(BufferLoadDword, 2, 1, 0)
              << WaitAll
              << Mubuf(BufferStoreDword, 2, 1, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

enum class Writer { Linear, Strided };

const char* NameOf(Writer writer) {
    return writer == Writer::Linear ? "a linear writer of 16-byte elements by global thread id" : "a writer of one dword per thread 64 bytes apart";
}

std::uint32_t Previous(std::uint64_t word) {
    return 0x3c000000u | static_cast<std::uint32_t>(word & 0x00ffffffu);
}

std::uint32_t CpuWord(std::uint64_t word) {
    return 0xc7000000u | static_cast<std::uint32_t>(word & 0x00ffffffu);
}

constexpr std::array<std::uint32_t, 4> Value{0x10101010u, 0x20202020u, 0x30303030u, 0x40404040u};

class Range {
public:
    Range() : pages(static_cast<std::uint32_t>((Guard + RangeBytes + Guard + PageBytes - 1u) / PageBytes)), copy(static_cast<std::uint32_t>((RangeBytes + PageBytes - 1u) / PageBytes)), values(1), labels(1) {
        for (std::uint32_t word = 0; word < 4u; ++word) values.Words()[word] = Value[word];
        Reset();
    }

    void Reset() {
        expected.resize(static_cast<std::size_t>(pages.Bytes() / 4u));
        for (std::size_t word = 0; word < expected.size(); ++word) {
            pages.Words()[word] = Previous(word);
            expected[word] = Previous(word);
        }
        for (std::uint64_t word = 0; word < copy.Bytes() / 4u; ++word) copy.Words()[word] = 0xdeadbeefu;
        for (std::uint32_t word = 0; word < 64u; ++word) labels.Words()[word] = 0u;
        AgcDriver::GuestMemory::BumpCollectEpoch();
    }

    std::uint64_t Base() const { return pages.Address(Guard); }
    std::uint64_t Label(std::uint32_t index) const { return labels.Address(index * 64u); }
    volatile std::uint32_t* LabelWord(std::uint32_t index) const { return labels.Words() + index * 16u; }

    void Write(Commands& commands, Writer writer) {
        const auto first = static_cast<std::size_t>(Guard / 4u);
        if (writer == Writer::Linear) {
            const auto buffer = BufferResource(Base(), Elements + Tail, ElementBytes);
            const std::array<std::uint32_t, 8> userData{buffer[0], buffer[1], buffer[2], buffer[3], Low(values.Address()), High(values.Address()), 0u, 0u};
            commands.Dispatch(LinearProgram(), userData, Lanes, Groups);
            for (std::uint32_t element = 0; element < Elements; ++element) {
                for (std::uint32_t word = 0; word < 4u; ++word) expected[first + element * 4u + word] = Value[word];
            }
            return;
        }
        constexpr std::uint32_t threads = static_cast<std::uint32_t>(RangeBytes >> StrideShift);
        const auto buffer = BufferResource(Base(), static_cast<std::uint32_t>(RangeBytes));
        const std::array<std::uint32_t, 8> userData{buffer[0], buffer[1], buffer[2], buffer[3], 0u, 0u, 0u, 0u};
        commands.Dispatch(StridedProgram(), userData, Lanes, threads / Lanes);
        for (std::uint32_t thread = 0; thread < threads; ++thread) expected[first + (std::size_t{thread} << StrideShift) / 4u] = 0x51000000u + thread;
    }

    void Copy(Commands& commands) {
        const auto from = BufferResource(Base(), static_cast<std::uint32_t>(RangeBytes));
        const auto to = BufferResource(copy.Address(), static_cast<std::uint32_t>(RangeBytes));
        const std::array<std::uint32_t, 8> userData{from[0], from[1], from[2], from[3], to[0], to[1], to[2], to[3]};
        commands.Dispatch(CopyProgram(), userData, CopyLanes, static_cast<std::uint32_t>(RangeBytes / 4u / CopyLanes));
    }

    void CpuWriteUnwritten(Writer writer) {
        for (std::size_t word = 0; word < expected.size(); word += 7u) {
            const auto offset = static_cast<std::int64_t>(word * 4u) - static_cast<std::int64_t>(Guard);
            bool written = false;
            if (offset >= 0 && offset < static_cast<std::int64_t>(RangeBytes)) {
                written = writer == Writer::Linear ? offset < static_cast<std::int64_t>(std::uint64_t{Elements} * ElementBytes) : (offset & ((std::int64_t{1} << StrideShift) - 1)) == 0;
            }
            if (written) continue;
            pages.Words()[word] = CpuWord(word);
            expected[word] = CpuWord(word);
        }
    }

    bool AwaitLabel(std::uint32_t index) const {
        for (const auto deadline = Clock::now() + LabelLimit; Clock::now() < deadline;) {
            if (LabelWord(index)[0] == 1u) return true;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return false;
    }

    void CheckRange(const std::string& what, bool hook) const {
        if (hook) AgcDriver::GuestMemory::FlushGpuWrites(pages.Address(), pages.Bytes());
        std::size_t wrong = 0;
        std::string first;
        for (std::size_t word = 0; word < expected.size(); ++word) {
            const auto actual = pages.Words()[word];
            if (actual == expected[word]) continue;
            if (wrong++ == 0u) first = "byte " + std::to_string(static_cast<std::int64_t>(word * 4u) - static_cast<std::int64_t>(Guard)) + " of the range holds " + Hex(actual) + ", expected " + Hex(expected[word]);
        }
        Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " words of the store range and its guards differ, first " + first);
    }

    void CheckCopy(const std::string& what) const {
        AgcDriver::GuestMemory::FlushGpuWrites(copy.Address(), copy.Bytes());
        std::size_t wrong = 0;
        std::string first;
        const auto offset = static_cast<std::size_t>(Guard / 4u);
        for (std::size_t word = 0; word < RangeBytes / 4u; ++word) {
            const auto actual = copy.Words()[word];
            if (actual == expected[offset + word]) continue;
            if (wrong++ == 0u) first = "byte " + std::to_string(word * 4u) + " read " + Hex(actual) + ", expected " + Hex(expected[offset + word]);
        }
        Expect(wrong == 0u, what + ": the reading dispatch read " + std::to_string(wrong) + " words unlike the store range, first " + first);
    }

private:
    PrivatePages pages;
    PrivatePages copy;
    PrivatePages values;
    PrivatePages labels;
    std::vector<std::uint32_t> expected;
};

std::string Settled() {
    std::string failure;
    try {
        WaitIdle();
        Settle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    return failure;
}

void ReadInSameSubmission(Range& range, Writer writer) {
    range.Reset();
    const std::string what = std::string(NameOf(writer)) + ", then a dispatch reading its whole range, in one submission";
    Commands commands;
    range.Write(commands, writer);
    range.Copy(commands);
    commands.WriteData(range.Label(0), {1u});
    commands.Submit(QueueA);
    Expect(range.AwaitLabel(0), what + ": the label never landed");
    const auto failure = Settled();
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    range.CheckCopy(what);
    range.CheckRange(what, true);
}

void ReadAtLabel(Range& range, Writer writer) {
    range.Reset();
    const std::string what = std::string(NameOf(writer)) + ", read by the CPU straight from guest memory once the label after it landed";
    Commands commands;
    range.Write(commands, writer);
    commands.WriteData(range.Label(0), {1u});
    commands.Submit(QueueA);
    Expect(range.AwaitLabel(0), what + ": the label never landed");
    range.CheckRange(what, false);
    const auto failure = Settled();
    Expect(failure.empty(), what + ": the submission failed: " + failure);
}

void CpuWritesBeforeExecution(Range& range, Writer writer) {
    range.Reset();
    const std::string what = std::string(NameOf(writer)) + " waiting on a CPU-written label, while the CPU writes bytes of its range and guards it does not store to before writing the label";
    Commands commands;
    commands.Wait(range.Label(1), Equal, 1u, 0xffffffffu);
    range.Write(commands, writer);
    commands.WriteData(range.Label(0), {1u});
    commands.Submit(QueueA);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    range.CpuWriteUnwritten(writer);
    range.LabelWord(1)[0] = 1u;
    Expect(range.AwaitLabel(0), what + ": the label after the writer never landed");
    range.CheckRange(what, false);
    const auto failure = Settled();
    Expect(failure.empty(), what + ": the submission failed: " + failure);
}

void ReadOnOtherStream(Range& range, Writer writer) {
    range.Reset();
    const std::string what = std::string(NameOf(writer)) + " on one queue, read by a dispatch on another queue that waits on the writer's label";
    Commands reader;
    reader.Wait(range.Label(0), Equal, 1u, 0xffffffffu);
    range.Copy(reader);
    reader.WriteData(range.Label(2), {1u});
    reader.Submit(QueueB);
    Commands writerCommands;
    range.Write(writerCommands, writer);
    writerCommands.WriteData(range.Label(0), {1u});
    writerCommands.Submit(QueueA);
    Expect(range.AwaitLabel(2), what + ": the reader's label never landed");
    const auto failure = Settled();
    Expect(failure.empty(), what + ": the submissions failed: " + failure);
    range.CheckCopy(what);
    range.CheckRange(what, true);
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
        Range range;
        for (const auto writer : {Writer::Linear, Writer::Strided}) {
            const auto before = MirrorChecks::Counts();
            ReadInSameSubmission(range, writer);
            ReadAtLabel(range, writer);
            CpuWritesBeforeExecution(range, writer);
            ReadOnOtherStream(range, writer);
            const auto after = MirrorChecks::Counts();
            std::printf("store range ordering: %llu bytes written back from residency mirrors in %llu CPU write-backs over the four cases of %s, reading dispatches included\n", static_cast<unsigned long long>(after.writtenBackBytes - before.writtenBackBytes), static_cast<unsigned long long>(after.cpuWriteBacks - before.cpuWriteBacks), NameOf(writer));
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
        std::fprintf(stderr, "%d store range ordering checks failed\n", failures);
        return 1;
    }
    std::printf("store range ordering tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
