#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t SharedPageCount = 3;

DWORD Protection(std::uint64_t address) {
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) == sizeof(memory), "residency: VirtualQuery failed");
    return memory.Protect;
}

void SharedViewsAdmittedEagerly(MirrorReader& reader) {
    const std::string what = "shared views under the guest-size budget";
    SharedPages pages(SharedPageCount);
    Fill(pages.Words(), SharedPageCount, 10);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    PrivatePages other(1);
    reader.Read(other.Address(), 1);
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < SharedPageCount; ++page) resident += Resident(pages.Address(page)) ? 1u : 0u;
    Expect(resident == SharedPageCount, what + ": " + std::to_string(resident) + " of the " + std::to_string(SharedPageCount) + " mapped shared pages have a mirror before any dispatch used them; shared views are admitted eagerly");
    ExpectRead(reader.Read(pages.Address(), SharedPageCount), SharedPageCount, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(10, page, subpage * WordsPerSubpage); }, what + ": the first read");
    Settle();
}

void UnsettledPrivatePagesUnguarded(MirrorReader& reader) {
    const std::string what = "a bounded store into private pages before their imports settled";
    PrivatePages pages(2);
    Fill(pages.Words(), 2, 11);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    Use(pages.Address());
    Store(pages.Address());
    std::uint32_t guarded = 0;
    std::string first;
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) {
        const auto protection = Protection(pages.Address(std::uint64_t{subpage} * SubpageBytes));
        if (protection == PAGE_READWRITE) continue;
        if (guarded++ == 0u) first = "subpage " + std::to_string(subpage) + " has protection " + Hex(protection);
    }
    Expect(guarded == 0u, what + ": " + std::to_string(guarded) + " subpages are protected after the store, first " + first + "; private pages are not guarded before their imports settled");
    AgcDriver::GuestMemory::FlushGpuWrites(pages.Address(), PageBytes);
    std::uint32_t wrong = 0;
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) {
        const auto wanted = word % WordsPerSubpage == 0u ? WrittenBase + word / WordsPerSubpage : Pattern(11, 0, word);
        wrong += pages.Words()[word] != wanted ? 1u : 0u;
    }
    Expect(wrong == 0u, what + ": the CPU read " + std::to_string(wrong) + " wrong words after the store");
    ExpectRead(reader.Read(pages.Address(), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return WrittenBase + subpage; }, what + ": the GPU read after the store");
    Settle();
}

void Run(MirrorReader& reader) {
    SharedViewsAdmittedEagerly(reader);
    UnsettledPrivatePagesUnguarded(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency shared and private pages", Run);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
