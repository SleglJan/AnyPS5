#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;
using AgcDriver::Graphics::Recorder;

constexpr std::uint32_t Tag = 13;
constexpr std::uint32_t Labels = 16;
constexpr std::uint32_t LabelSubpage = 2;
constexpr std::uint32_t FirstValue = 0x6a000001u;

void Run(MirrorReader& reader) {
    const std::string what = "label stores into a resident shared page";
    SharedPages pages(1);
    Fill(pages.Words(), 1, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    PrivatePages other(1);
    reader.Read(other.Address(), 1);
    Expect(Resident(pages.Address()), what + ": the shared page has a mirror before the labels");
    const auto label = pages.Address(0, LabelSubpage);
    const auto last = FirstValue + Labels - 1u;
    const auto expected = [&](std::uint64_t page, std::uint32_t subpage) {
        if (page == 0 && subpage == LabelSubpage) return last;
        return Pattern(Tag, page, subpage * WordsPerSubpage);
    };
    const auto before = Recorder::StoreCounts();
    Commands commands;
    reader.Record(commands, other.Address(), 1);
    for (std::uint32_t index = 0; index < Labels; ++index) commands.WriteData(label, {FirstValue + index});
    reader.Record(commands, pages.Address(), 1);
    commands.Submit(0);
    const auto lanes = reader.Collect();
    const auto after = Recorder::StoreCounts();
    const auto stores = after.stores - before.stores;
    const auto forced = after.runsForced - before.runsForced;
    Expect(stores >= Labels, what + ": " + std::to_string(stores) + " stores recorded for " + std::to_string(Labels) + " labels");
    Expect(forced == 1u, what + ": " + std::to_string(forced) + " store runs of " + std::to_string(Labels) + " labels were recorded in place, expected one, before the dispatch that reads their page; keeping the mirror current must not force a run per label");
    ExpectRead(lanes, 1, expected, what + ": a dispatch after the labels reads the last label through the mirror");
    Expect(pages.Words()[LabelSubpage * WordsPerSubpage] == last, what + ": guest memory holds " + Hex(pages.Words()[LabelSubpage * WordsPerSubpage]) + " after the labels, expected " + Hex(last));
    Settle();
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency label stores", Run);
#else
    std::printf("skipped, residency write tracking is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
