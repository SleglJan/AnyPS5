#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t HotPages = MirrorReader::Pages;
constexpr std::uint32_t Tag = 12;
constexpr std::uint32_t FirstStore = 0x5e000001u;
constexpr std::uint32_t SecondStore = 0x5e000002u;

bool Resident(std::uint64_t address) {
    bool resident = false;
    Locked([&] { resident = Residency().Manager().Slot(PageOf(address)).has_value(); });
    return resident;
}

void Run(MirrorReader& reader) {
    PrivatePages pages(HotPages);
    GuestArena::GuestArenaPinWritable_nid_postfix(pages.Data(), pages.Bytes());
    Fill(pages.Words(), HotPages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    bool stored = false;
    const auto expected = [&](std::uint64_t page, std::uint32_t subpage) {
        if (stored && page == 3 && subpage == 5) return FirstStore;
        if (stored && page == 11 && subpage == 0) return SecondStore;
        return Pattern(Tag, page, subpage * WordsPerSubpage);
    };
    ExpectRead(reader.Read(pages.Address(), HotPages), HotPages, expected, "hot refresh: the read that admits the pages");
    for (std::uint32_t round = 0; round < 3; ++round) ExpectRead(reader.Read(pages.Address(), HotPages), HotPages, expected, "hot refresh: a read while the imports settle");
    std::uint32_t served = 0;
    for (std::uint32_t page = 0; page < HotPages; ++page) served += Resident(pages.Address(std::uint64_t{page} * PageBytes)) ? 0u : 1u;
    Expect(served == HotPages, "hot refresh: " + std::to_string(served) + " of " + std::to_string(HotPages) + " pinned private pages, which libc refuses to guard, are served from their imports once their imports settled");
    const auto quiet = Counts();
    for (std::uint32_t round = 0; round < 3; ++round) ExpectRead(reader.Read(pages.Address(), HotPages), HotPages, expected, "hot refresh: a read with no CPU store since the last one");
    const auto afterQuiet = Counts();
    Expect(afterQuiet.hotCollects == quiet.hotCollects && afterQuiet.hotRefreshed == quiet.hotRefreshed && afterQuiet.copiedInBytes == quiet.copiedInBytes, "hot refresh: three dispatches with no CPU store collected " + std::to_string(afterQuiet.hotCollects - quiet.hotCollects) + " hot runs, refreshed " + std::to_string(afterQuiet.hotRefreshed - quiet.hotRefreshed) + " hot pages and copied " + std::to_string(afterQuiet.copiedInBytes - quiet.copiedInBytes) + " bytes");
    pages.Words()[3 * WordsPerPage + 5 * WordsPerSubpage] = FirstStore;
    pages.Words()[11 * WordsPerPage] = SecondStore;
    stored = true;
    const auto before = Counts();
    ExpectRead(reader.Read(pages.Address(), HotPages), HotPages, expected, "hot refresh: the read after CPU stores into two of the pages");
    const auto after = Counts();
    Expect(after.hotCollects == before.hotCollects && after.copiedInBytes == before.copiedInBytes && after.writtenBackBytes == before.writtenBackBytes, "hot refresh: CPU stores into 2 pages served from their imports reached the read with " + std::to_string(after.hotCollects - before.hotCollects) + " hot collects, " + std::to_string(after.copiedInBytes - before.copiedInBytes) + " bytes copied in and " + std::to_string(after.writtenBackBytes - before.writtenBackBytes) + " written back, expected none");
    Settle();
    GuestArena::GuestArenaUnpinWritable_nid_postfix(pages.Data(), pages.Bytes());
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
