#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t HotPages = MirrorReader::Pages;
constexpr std::uint32_t Tag = 12;
constexpr std::uint32_t FirstStore = 0x5e000001u;
constexpr std::uint32_t SecondStore = 0x5e000002u;

bool HotResident(std::uint64_t address) {
    bool hot = false;
    Locked([&] { hot = Residency().Manager().HotResident(PageOf(address)); });
    return hot;
}

void Run(MirrorReader& reader) {
    PrivatePages pages(HotPages);
    Fill(pages.Words(), HotPages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    bool stored = false;
    const auto expected = [&](std::uint64_t page, std::uint32_t subpage) {
        if (stored && page == 3 && subpage == 5) return FirstStore;
        if (stored && page == 11 && subpage == 0) return SecondStore;
        return Pattern(Tag, page, subpage * WordsPerSubpage);
    };
    ExpectRead(reader.Read(pages.Address(), HotPages), HotPages, expected, "hot refresh: the read that admits the pages");
    std::uint32_t hot = 0;
    for (std::uint32_t page = 0; page < HotPages; ++page) hot += HotResident(pages.Address(std::uint64_t{page} * PageBytes)) ? 1u : 0u;
    Expect(hot == HotPages, "hot refresh: " + std::to_string(hot) + " of " + std::to_string(HotPages) + " private pages admitted by a read are kept current from the write tracking");
    const auto quiet = Counts();
    for (std::uint32_t round = 0; round < 3; ++round) ExpectRead(reader.Read(pages.Address(), HotPages), HotPages, expected, "hot refresh: a read with no CPU store since the last one");
    const auto afterQuiet = Counts();
    Expect(afterQuiet.hotSweeps > quiet.hotSweeps, "hot refresh: three dispatches in new epochs collected no writes of the hot pages");
    Expect(afterQuiet.hotRefreshed == quiet.hotRefreshed && afterQuiet.copiedInBytes == quiet.copiedInBytes, "hot refresh: three dispatches with no CPU store refreshed " + std::to_string(afterQuiet.hotRefreshed - quiet.hotRefreshed) + " hot pages and copied " + std::to_string(afterQuiet.copiedInBytes - quiet.copiedInBytes) + " bytes");
    pages.Words()[3 * WordsPerPage + 5 * WordsPerSubpage] = FirstStore;
    pages.Words()[11 * WordsPerPage] = SecondStore;
    stored = true;
    const auto before = Counts();
    ExpectRead(reader.Read(pages.Address(), HotPages), HotPages, expected, "hot refresh: the read after CPU stores into two of the hot pages");
    const auto after = Counts();
    Expect(after.hotRefreshed - before.hotRefreshed == 2u, "hot refresh: CPU stores into 2 of " + std::to_string(HotPages) + " hot pages refreshed " + std::to_string(after.hotRefreshed - before.hotRefreshed) + " pages");
    Expect(after.copiedInBytes - before.copiedInBytes == 2u * SubpageBytes, "hot refresh: CPU stores into 2 subpages copied " + std::to_string(after.copiedInBytes - before.copiedInBytes) + " bytes, expected " + std::to_string(2u * SubpageBytes));
    const auto settled = Counts();
    ExpectRead(reader.Read(pages.Address(), HotPages), HotPages, expected, "hot refresh: the read after the refresh");
    Expect(Counts().copiedInBytes == settled.copiedInBytes, "hot refresh: a refreshed page was copied again with no CPU store");
    Settle();
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency hot refresh", Run);
#else
    std::printf("skipped, residency write tracking is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
