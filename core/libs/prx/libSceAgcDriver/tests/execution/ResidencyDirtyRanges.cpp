#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t Label = 0x5c000001u;
constexpr std::uint32_t CpuValue = 0x5c000002u;
constexpr std::uint32_t LabelSubpage = 3;
constexpr std::uint32_t LabelWord = 8;
constexpr std::uint32_t CpuSubpage = 5;
constexpr std::uint32_t CpuWord = 9;

using Model = std::function<std::uint32_t(std::uint32_t word)>;

void ExpectPage(volatile std::uint32_t* words, const Model& model, const std::string& what) {
    std::uint32_t wrong = 0;
    std::string first;
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) {
        const auto read = words[word];
        const auto wanted = model(word);
        if (read == wanted) continue;
        if (wrong++ == 0u) first = "subpage " + std::to_string(word / WordsPerSubpage) + " word " + std::to_string(word % WordsPerSubpage) + " is " + Hex(read) + ", expected " + Hex(wanted);
    }
    Expect(wrong == 0u, what + ": the CPU read " + std::to_string(wrong) + " wrong words, first " + first);
}

std::uint32_t Stored(std::uint32_t word, std::uint32_t tag) {
    if (word % WordsPerSubpage == 0u) return WrittenBase + word / WordsPerSubpage;
    return Pattern(tag, 0, word);
}

void ExpectKeptInMirror(std::uint64_t address, const ResidencyCounts& before, const std::string& what) {
    const auto after = Counts();
    Expect(Resident(address), what + ": the page a bounded store wrote left residency; it must stay in its mirror, dirty");
    Expect(after.demoted == before.demoted, what + ": the bounded store demoted " + std::to_string(after.demoted - before.demoted) + " pages; a bounded store keeps the page in its mirror");
}

void LabelInDirtySubpage(MirrorReader& reader) {
    const std::string what = "a label in a dirty subpage";
    SharedPages pages(2);
    Fill(pages.Words(), 2, 1);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    Use(pages.Address(0));
    Expect(Resident(pages.Address(0)), what + ": the page was not admitted");
    const auto before = Counts();
    Store(pages.Address(0));
    ExpectKeptInMirror(pages.Address(0), before, what);
    Commands commands;
    commands.WriteData(pages.Address(0, LabelSubpage, LabelWord), {Label});
    commands.Submit(0);
    WaitIdle();
    ExpectRead(reader.Read(pages.Address(0), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return WrittenBase + subpage; }, what + ": the GPU read of the stored words after the label");
    ExpectRead(reader.Read(pages.Address(0, 0, LabelWord), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return subpage == LabelSubpage ? Label : Pattern(1, 0, subpage * WordsPerSubpage + LabelWord); }, what + ": the GPU read of the label's word in every subpage");
    ExpectPage(pages.Words(0), [](std::uint32_t word) { return word == LabelSubpage * WordsPerSubpage + LabelWord ? Label : Stored(word, 1); }, what + ": the write-back of the dirty page kept the label written after the store");
    Settle();
}

void CpuWriteInDirtySubpage(MirrorReader& reader) {
    const std::string what = "a CPU write in a dirty subpage";
    SharedPages pages(2);
    Fill(pages.Words(), 2, 2);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    Use(pages.Address(0));
    Expect(Resident(pages.Address(0)), what + ": the page was not admitted");
    const auto before = Counts();
    Store(pages.Address(0));
    ExpectKeptInMirror(pages.Address(0), before, what);
    pages.Words(0)[CpuSubpage * WordsPerSubpage + CpuWord] = CpuValue;
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    ExpectPage(pages.Words(0), [](std::uint32_t word) { return word == CpuSubpage * WordsPerSubpage + CpuWord ? CpuValue : Stored(word, 2); }, what + ": the CPU read after its own write into the dirty subpage");
    ExpectRead(reader.Read(pages.Address(0), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return WrittenBase + subpage; }, what + ": the GPU read of the stored words after the CPU write");
    ExpectRead(reader.Read(pages.Address(0, 0, CpuWord), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return subpage == CpuSubpage ? CpuValue : Pattern(2, 0, subpage * WordsPerSubpage + CpuWord); }, what + ": the GPU read of the CPU's word in every subpage");
    Settle();
}

void SecondStoreAfterWriteBack(MirrorReader& reader) {
    const std::string what = "a second store after a write-back";
    SharedPages pages(2);
    Fill(pages.Words(), 2, 3);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    Use(pages.Address(0));
    Store(pages.Address(0));
    ExpectPage(pages.Words(0), [](std::uint32_t word) { return Stored(word, 3); }, what + ": the CPU read after the first store");
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) pages.Words(0)[subpage * WordsPerSubpage] = Pattern(4, 0, subpage);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    ExpectRead(reader.Read(pages.Address(0), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return Pattern(4, 0, subpage); }, what + ": the GPU read after the CPU overwrote the stored words");
    const auto before = Counts();
    Store(pages.Address(0));
    ExpectKeptInMirror(pages.Address(0), before, what);
    ExpectRead(reader.Read(pages.Address(0), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return WrittenBase + subpage; }, what + ": the GPU read after the second store");
    ExpectPage(pages.Words(0), [](std::uint32_t word) { return Stored(word, 3); }, what + ": the CPU read after the second store");
    Settle();
}

void Run(MirrorReader& reader) {
    LabelInDirtySubpage(reader);
    CpuWriteInDirtySubpage(reader);
    SecondStoreAfterWriteBack(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency dirty range", Run);
#else
    std::printf("skipped, dirty mirrors are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
