#include "DeviceTableFixture.hpp"
#include <spirv/unified1/spirv.hpp>
#include <map>
#include "prx/libc/include/GuestAllocations.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

using namespace DeviceTable;
using AgcDriver::Graphics::Require;

constexpr std::uint32_t PointerSlot = 4;
constexpr std::uint32_t GuardSlot = 6;
constexpr std::uint32_t GuardSgpr = 8;
constexpr std::uint32_t PointerSgpr = 20;
constexpr std::uint32_t LoadedSgpr = 36;
constexpr std::uint32_t WiderSgpr = 40;
constexpr std::uint32_t DataWords = 16;
constexpr std::uint32_t WideWord = 1;
constexpr std::uint32_t WiderWord = 6;
constexpr std::uint32_t RowBytes = Lanes * LaneWords * 4u;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Barrier = 0xbf8a0000u;
constexpr std::uint32_t CmpxGreater = 0xd4u;
constexpr std::uint32_t CmpxLess = 0xd1u;

TestGuest::Array<std::uint32_t, DataWords> Data;

namespace Ops {

using namespace DeviceTable::Asm;

constexpr std::uint32_t SLoadDword = 0x00u;
constexpr std::uint32_t SLoadDwordX2 = 0x01u;
constexpr std::uint32_t SLoadDwordX8 = 0x03u;

constexpr std::uint32_t Vopc(std::uint32_t opcode, std::uint32_t src0, std::uint32_t vsrc1) {
    return 0x7c000000u | (opcode << 17u) | (vsrc1 << 9u) | src0;
}

}

std::uint32_t DataWord(std::uint32_t word) {
    return 0x5c000000u + word * 0x00010307u;
}

struct Built {
    Program program;
    std::vector<std::uint32_t> pcs;
};

void Prologue(Program& program, bool barrier, bool guard) {
    using namespace Ops;
    if (barrier) program << Barrier;
    program << Smem(SLoadDwordX4, 28, 0, OutputSlot * 4u) << Smem(SLoadDwordX2, PointerSgpr, 0, PointerSlot * 4u);
    if (guard) program << Smem(SLoadDword, GuardSgpr, 0, GuardSlot * 4u);
    program << WaitAll;
}

void Load(Built& built, std::uint32_t opcode, std::uint32_t sdata, std::uint32_t offset) {
    built.pcs.push_back(static_cast<std::uint32_t>(built.program.size * 4u));
    built.program << Ops::Smem(opcode, sdata, PointerSgpr, offset);
}

void StoreRows(Program& program, std::uint32_t rows) {
    using namespace Ops;
    program << WaitAll << Vop2(VLshlRevB32, 8, InlineInteger(4), 0);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t component = 0; component < LaneWords; ++component) program << Vop1(VMovB32, 4 + component, LoadedSgpr + row * LaneWords + component);
        program << Mubuf(BufferStoreDwordX4, 4, 8, 28, false, row * RowBytes);
    }
    program << EndProgram;
}

const Built& WideProgram() {
    static const Built built = [] {
        Built made;
        Prologue(made.program, false, false);
        Load(made, Ops::SLoadDwordX4, LoadedSgpr, WideWord * 4u);
        Load(made, Ops::SLoadDwordX8, WiderSgpr, WiderWord * 4u);
        StoreRows(made.program, 3);
        return made;
    }();
    return built;
}

const Built& ExecProgram(std::uint32_t compare) {
    static const auto build = [](std::uint32_t opcode) {
        Built made;
        Prologue(made.program, false, true);
        made.program << Ops::Vopc(opcode, GuardSgpr, 0);
        Load(made, Ops::SLoadDwordX4, LoadedSgpr, WideWord * 4u);
        Load(made, Ops::SLoadDwordX8, WiderSgpr, WiderWord * 4u);
        StoreRows(made.program, 3);
        return made;
    };
    static const Built greater = build(CmpxGreater);
    static const Built less = build(CmpxLess);
    return compare == CmpxGreater ? greater : less;
}

const Built& StraddleProgram(bool wider, bool barrier) {
    static const auto build = [](bool eight, bool withBarrier) {
        Built made;
        Prologue(made.program, withBarrier, false);
        Load(made, eight ? Ops::SLoadDwordX8 : Ops::SLoadDwordX4, LoadedSgpr, 0u);
        StoreRows(made.program, eight ? 2u : 1u);
        return made;
    };
    static const std::array<Built, 4> programs{build(false, false), build(false, true), build(true, false), build(true, true)};
    return programs[(wider ? 2u : 0u) + (barrier ? 1u : 0u)];
}

class StraddleBlock {
public:
    StraddleBlock() {
#ifdef _WIN32
        base = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2u * BlockBytes, MEM_RESERVE, PAGE_NOACCESS));
        Require(base != nullptr, "scalar load spans: cannot reserve the mapped and the inaccessible block");
        Require(VirtualAlloc(base, BlockBytes, MEM_COMMIT, PAGE_READWRITE) == base, "scalar load spans: cannot commit the mapped block");
#else
        reservation = mmap(nullptr, 3u * BlockBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(reservation != MAP_FAILED, "scalar load spans: cannot reserve the mapped and the inaccessible block");
        const auto begin = reinterpret_cast<std::uintptr_t>(reservation);
        base = reinterpret_cast<std::uint8_t*>((begin + BlockBytes - 1u) & ~(BlockBytes - 1u));
        Require(mprotect(base, BlockBytes, PROT_READ | PROT_WRITE) == 0, "scalar load spans: cannot map the mapped block");
#endif
        GuestAllocations::Mutation().Add(base, BlockBytes, true, true, true);
    }

    ~StraddleBlock() {
        GuestAllocations::Mutation().Remove(base);
#ifdef _WIN32
        VirtualFree(base, 0, MEM_RELEASE);
#else
        munmap(reservation, 3u * BlockBytes);
#endif
    }

    StraddleBlock(const StraddleBlock&) = delete;
    StraddleBlock& operator=(const StraddleBlock&) = delete;

    std::uint64_t Boundary() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base)) + BlockBytes; }

    void FillBefore(std::uint32_t words) {
        auto* first = reinterpret_cast<std::uint32_t*>(base + BlockBytes) - words;
        for (std::uint32_t word = 0; word < words; ++word) first[word] = DataWord(word);
    }

private:
    std::uint8_t* base = nullptr;
#ifndef _WIN32
    void* reservation = nullptr;
#endif
};

std::string Fault(std::uint64_t address, std::uint32_t pc) {
    char text[96];
    std::snprintf(text, sizeof(text), "address=0x%llx instruction=0x%x ", static_cast<unsigned long long>(address), pc);
    return text;
}

std::size_t Faults(const Outcome& outcome) {
    std::size_t count = 0;
    for (auto at = outcome.log.find("BDA access failed"); at != std::string::npos; at = outcome.log.find("BDA access failed", at + 1u)) ++count;
    return count;
}

std::size_t SegmentCallsFromMain(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        if ((words[cursor] & 0xffffu) == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
    }
    std::size_t calls = 0;
    std::string function;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) function = names[words[cursor + 2]];
        if (op == spv::OpFunctionCall && function == "main" && names[words[cursor + 3]] == "bda_segment") ++calls;
    }
    return calls;
}

Outcome Dispatch(AgcDriver::VulkanDevice& device, const Built& built, std::uint64_t pointer, std::uint32_t guard, const std::string& what) {
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Srt[PointerSlot] = static_cast<std::uint32_t>(pointer);
    Srt[PointerSlot + 1u] = static_cast<std::uint32_t>(pointer >> 32u);
    Srt[GuardSlot] = guard;
    const auto compiled = Compile(device, built.program, userData);
    ExpectNothingCaptured(compiled, what);
    Expect(SegmentCallsFromMain(compiled.result->spirv.Words()) != 0u, what + ": main reads its scalar loads without checking the invocation's cached read segment and refreshing it");
    AgcDriver::GuestMemory::BumpCollectEpoch();
    return DeviceTable::Run(device, compiled, 1);
}

void ExpectWords(const Outcome& outcome, std::uint32_t rows, const std::function<bool(std::uint32_t lane)>& active, const std::function<std::uint32_t(std::uint32_t row, std::uint32_t component)>& expected, const std::string& what) {
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            for (std::uint32_t component = 0; component < LaneWords; ++component) {
                const auto word = outcome.words[(row * Lanes + lane) * LaneWords + component];
                const auto wanted = active(lane) ? expected(row, component) : Untouched;
                if (word == wanted) continue;
                if (wrong++ == 0u) first = "row " + std::to_string(row) + " lane " + std::to_string(lane) + " component " + std::to_string(component) + " is " + Hex(word) + ", expected " + Hex(wanted);
            }
        }
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " words differ, first " + first);
}

std::uint32_t WideWords(std::uint32_t row, std::uint32_t component) {
    return row == 0u ? DataWord(WideWord + component) : DataWord(WiderWord + (row - 1u) * LaneWords + component);
}

void MappedReads(AgcDriver::VulkanDevice& device) {
    for (std::uint32_t word = 0; word < DataWords; ++word) Data[word] = DataWord(word);
    const auto pointer = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Data.data()));
    const auto all = [](std::uint32_t) { return true; };
    const auto wide = Dispatch(device, WideProgram(), pointer, 0u, "grouped scalar reads");
    ExpectClean(wide, "grouped scalar reads");
    ExpectWords(wide, 3, all, WideWords, "grouped scalar reads: an s_load_dwordx4 at dword 1 and an s_load_dwordx8 at dword 6 read at run time");

    const auto low = Dispatch(device, ExecProgram(CmpxGreater), pointer, 5u, "grouped scalar reads under EXEC");
    ExpectClean(low, "grouped scalar reads under EXEC");
    ExpectWords(low, 3, [](std::uint32_t lane) { return lane < 5u; }, WideWords, "grouped scalar reads: the loads with EXEC holding lanes 0 to 4");
    const auto high = Dispatch(device, ExecProgram(CmpxLess), pointer, 40u, "grouped scalar reads under EXEC");
    ExpectClean(high, "grouped scalar reads under EXEC");
    ExpectWords(high, 3, [](std::uint32_t lane) { return lane > 40u; }, WideWords, "grouped scalar reads: the loads with EXEC holding lanes 41 to 63 only");
}

void StraddlingReads(AgcDriver::VulkanDevice& device) {
    StraddleBlock block;
    const auto boundary = block.Boundary();
    for (const bool wider : {false, true}) {
        const std::uint32_t mapped = wider ? 5u : 2u;
        const std::uint32_t rows = wider ? 2u : 1u;
        block.FillBefore(mapped);
        const auto pointer = boundary - mapped * 4u;
        const std::string name = std::string(wider ? "an s_load_dwordx8" : "an s_load_dwordx4") + " with " + std::to_string(mapped) + " mapped dwords before an inaccessible page";
        for (const bool barrier : {true, false}) {
            const auto& built = StraddleProgram(wider, barrier);
            const auto what = "straddling scalar reads: " + name + (barrier ? " (a program with a barrier, whose invocations do not stop at a fault)" : "");
            const auto outcome = Dispatch(device, built, pointer, 0u, what);
            Expect(Faults(outcome) == 1u && outcome.log.find(Fault(boundary, built.pcs[0])) != std::string::npos && outcome.log.find("reason=1") != std::string::npos, what + ": did not report one unmapped fault at the first unmapped byte 0x" + Hex(static_cast<std::uint32_t>(boundary)) + " with the instruction's pc " + Hex(built.pcs[0]) + ":\n" + outcome.log);
            if (barrier) {
                ExpectWords(outcome, rows, [](std::uint32_t) { return true; }, [&](std::uint32_t row, std::uint32_t component) {
                    const auto dword = row * LaneWords + component;
                    return dword < mapped ? DataWord(dword) : 0u;
                }, what + ": the mapped dwords and zeros for the rest");
            } else {
                Expect(std::all_of(outcome.words.begin(), outcome.words.begin() + rows * Lanes * LaneWords, [](std::uint32_t word) { return word == Untouched || word == 0u; }), what + ": a faulting invocation stored data it did not read");
            }
        }
    }
}

void Run(AgcDriver::VulkanDevice& device) {
    MappedReads(device);
    StraddlingReads(device);
}

}

int main() {
    return Main("scalar load spans", Run);
}
