#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t MirrorMarker = 0x5b000001u;
constexpr std::uint32_t CpuStore = 0x5b000002u;
constexpr std::uint32_t MarkedSubpage = 2;
constexpr std::uint32_t StoredSubpage = 5;

void Run(MirrorReader& reader) {
    SharedPages pages(1);
    Fill(pages.Words(), 1, 5);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    Use(pages.Address());
    Expect(Resident(pages.Address()) && AgcDriver::GuestMemory::WriteGuarded(PageOf(pages.Address())), "subpage refresh: the page was not admitted with a write guard");
    FillMirror(pages.Address(0, MarkedSubpage), MirrorMarker);
    const auto expected = [](std::uint32_t storedSubpage) {
        return [storedSubpage](std::uint64_t page, std::uint32_t subpage) { return subpage == MarkedSubpage ? MirrorMarker : subpage == storedSubpage ? CpuStore : Pattern(5, page, subpage * WordsPerSubpage); };
    };
    ExpectRead(reader.Read(pages.Address(), 1), 1, expected(SubpagesPerPage), "subpage refresh: the read through the mirror before the CPU store");
    const auto before = Counts();
    Locked([&] {
        pages.Words(0)[StoredSubpage * WordsPerSubpage] = CpuStore;
        AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), PageBytes);
    });
    ExpectRead(reader.Read(pages.Address(), 1), 1, expected(StoredSubpage), "subpage refresh: the next dispatch after a CPU store into subpage 5 of a mirrored page (subpage 2 holds a change made only to the mirror, which a refresh of the stored 4 KiB alone keeps)");
    const auto after = Counts();
    Expect(after.faults == before.faults + 1u, "subpage refresh: the CPU store counted " + std::to_string(after.faults - before.faults) + " guard faults, expected one");
    Expect(after.copiedInBytes - before.copiedInBytes == SubpageBytes, "subpage refresh: the refresh after a CPU store into one 4 KiB subpage copied " + std::to_string(after.copiedInBytes - before.copiedInBytes) + " bytes, expected " + std::to_string(SubpageBytes));
    Settle();
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("new path residency subpage refresh", Run);
#else
    std::printf("skipped, commit imports and write guards are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
