#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t Rounds = 32;
constexpr std::uint32_t RoundPages = 2;
constexpr std::uint32_t CpuBase = 0x6c000000u;
constexpr std::uint32_t CpuSubpage = 3;
constexpr std::uint32_t DirtyCpuSubpage = 5;
constexpr std::uint32_t UnsettledCpuSubpage = 7;

bool ImportSettled(std::uint64_t address) {
    bool settled = false;
    Locked([&] { settled = Residency().Manager().ImportSettled(PageOf(address)); });
    return settled;
}

std::uint32_t DirtyMask(std::uint64_t address) {
    std::uint32_t mask = 0;
    Locked([&] { mask = Residency().Manager().Dirty(PageOf(address)); });
    return mask;
}

void Round(MirrorReader& reader, std::uint32_t round) {
    const std::string what = "round " + std::to_string(round);
    const auto tag = 1u + round % 14u;
    PrivatePages pages(RoundPages);
    Fill(pages.Words(), RoundPages, tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    const auto initial = [&](std::uint64_t page, std::uint32_t subpage) { return Pattern(tag, page, subpage * WordsPerSubpage); };
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, initial, what + ": the read that admits the fresh import");
    Expect(Resident(pages.Address()) && Resident(pages.Address(PageBytes)), what + ": the read did not admit both pages");
    Expect(!AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address())), what + ": page 0 is guarded before a submission issued after its import completed");
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, initial, what + ": a read issued after the admission");
    pages.Words()[UnsettledCpuSubpage * WordsPerSubpage] = CpuBase + 0x20000u + round;
    const auto admitted = [&](std::uint64_t page, std::uint32_t subpage) { return page == 0 && subpage == UnsettledCpuSubpage ? CpuBase + 0x20000u + round : initial(page, subpage); };
    Settle();
    Expect(ImportSettled(pages.Address()) && ImportSettled(pages.Address(PageBytes)), what + ": the imports did not settle after a completed submission issued after them");
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, admitted, what + ": the read after the settling ordering point missed a CPU store made between the admission and the settle");
    Expect(AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address())) && AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address(PageBytes))), what + ": a settled CPU-writable private page has no write guard");
    Store(pages.Address(PageBytes));
    Expect(Resident(pages.Address(PageBytes)) && DirtyMask(pages.Address(PageBytes)) == 0xffffu, what + ": page 1 is not resident with 16 dirty subpages after the GPU store (mask " + Hex(DirtyMask(pages.Address(PageBytes))) + ")");
    const auto stored = [&](std::uint64_t page, std::uint32_t subpage) { return page == 1 ? WrittenBase + subpage : admitted(page, subpage); };
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, stored, what + ": the GPU read of the dirty mirror");
    pages.Words()[CpuSubpage * WordsPerSubpage] = CpuBase + round;
    pages.Words()[WordsPerPage + DirtyCpuSubpage * WordsPerSubpage] = CpuBase + 0x10000u + round;
    const auto written = [&](std::uint64_t page, std::uint32_t subpage) {
        if (page == 0 && subpage == CpuSubpage) return CpuBase + round;
        if (page == 1 && subpage == DirtyCpuSubpage) return CpuBase + 0x10000u + round;
        return stored(page, subpage);
    };
    ExpectRead(reader.Read(pages.Address(), RoundPages), RoundPages, written, what + ": the GPU read after CPU stores into a guarded clean subpage and a dirty one");
    std::uint32_t wrong = 0;
    for (std::uint32_t page = 0; page < RoundPages; ++page) {
        for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) wrong += pages.Words()[page * WordsPerPage + subpage * WordsPerSubpage] != written(page, subpage) ? 1u : 0u;
    }
    Expect(wrong == 0u, what + ": the CPU read " + std::to_string(wrong) + " wrong subpage words after the GPU store and the CPU stores");
    Settle();
}

void Run(MirrorReader& reader) {
    const auto before = Counts();
    for (std::uint32_t round = 0; round < Rounds && failures == 0; ++round) Round(reader, round);
    const auto after = Counts();
    Expect(after.importsSettled >= before.importsSettled + std::uint64_t{Rounds} * RoundPages, std::to_string(after.importsSettled - before.importsSettled) + " imports settled in " + std::to_string(Rounds) + " rounds of " + std::to_string(RoundPages) + " fresh pages");
    Expect(after.cpuWriteBacks > before.cpuWriteBacks, "no CPU read wrote a dirty private page back");
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency private guards", Run);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
