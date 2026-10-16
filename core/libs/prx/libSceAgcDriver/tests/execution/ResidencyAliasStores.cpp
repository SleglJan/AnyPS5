#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t SecondBase = 0x7e000000u;

const Program& SecondStoreProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(12), 0)
              << Vop2(VAddNcU32, 2, Literal, 0) << SecondBase
              << Mubuf(BufferStoreDword, 2, 1, 0, false, 4)
              << EndProgram;
        return built;
    }();
    return program;
}

void StoreSecond(std::uint64_t address) {
    Commands commands;
    commands.Dispatch(SecondStoreProgram(), BufferResource(address, static_cast<std::uint32_t>(PageBytes)), WriteLanes);
    commands.Submit(0);
    WaitIdle();
}

std::uint32_t Expected(std::uint32_t tag, std::uint32_t word, bool first, bool second) {
    const auto subpage = word / WordsPerSubpage;
    const auto inSubpage = word % WordsPerSubpage;
    if (first && inSubpage == 0u) return WrittenBase + subpage;
    if (second && inSubpage == 1u) return SecondBase + subpage;
    return Pattern(tag, 0, word);
}

void ExpectView(volatile std::uint32_t* words, std::uint32_t tag, bool first, bool second, const std::string& what) {
    std::uint32_t wrong = 0;
    std::string firstWrong;
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) {
        const auto read = words[word];
        const auto wanted = Expected(tag, word, first, second);
        if (read == wanted) continue;
        if (wrong++ == 0u) firstWrong = "subpage " + std::to_string(word / WordsPerSubpage) + " word " + std::to_string(word % WordsPerSubpage) + " is " + Hex(read) + ", expected " + Hex(wanted);
    }
    Expect(wrong == 0u, what + ": the CPU read " + std::to_string(wrong) + " wrong words, first " + firstWrong);
}

void ExpectGpu(MirrorReader& reader, const Aliases& aliases, std::uint32_t view, std::uint32_t tag, bool first, bool second, const std::string& what) {
    ExpectRead(reader.Read(aliases.Address(view), 1), 1, [&](std::uint64_t, std::uint32_t subpage) { return Expected(tag, subpage * WordsPerSubpage, first, second); }, what + ": the GPU read of word 0 through view " + std::to_string(view));
}

void StoreThroughOtherView(MirrorReader& reader) {
    const std::string what = "a bounded store through the view without a mirror";
    Aliases aliases(1);
    Fill(aliases.Words(0), 1, 5);
    aliases.Collect();
    Use(aliases.Address(0));
    Use(aliases.Address(1));
    const auto mirrored = Resident(aliases.Address(0)) ? 0u : 1u;
    const auto other = 1u - mirrored;
    Expect(Resident(aliases.Address(mirrored)), what + ": neither view's page was admitted");
    ExpectGpu(reader, aliases, mirrored, 5, false, false, what + " (before the store)");
    Store(aliases.Address(other));
    ExpectGpu(reader, aliases, mirrored, 5, true, false, what);
    ExpectGpu(reader, aliases, other, 5, true, false, what);
    ExpectView(aliases.Words(mirrored), 5, true, false, what + ", read by the CPU through the mirrored view");
    ExpectView(aliases.Words(other), 5, true, false, what + ", read by the CPU through the storing view");
    Settle();
}

void StoreThroughEachView(MirrorReader& reader) {
    const std::string what = "bounded stores through both views of one page";
    Aliases aliases(2);
    Fill(aliases.Words(0), 2, 6);
    aliases.Collect();
    Use(aliases.Address(0));
    Use(aliases.Address(1));
    Store(aliases.Address(0));
    StoreSecond(aliases.Address(1));
    for (std::uint32_t view = 0; view < 2u; ++view) ExpectGpu(reader, aliases, view, 6, true, true, what);
    ExpectRead(reader.Read(aliases.Address(1, 0, 0) + 4u, 1), 1, [](std::uint64_t, std::uint32_t subpage) { return SecondBase + subpage; }, what + ": the GPU read of word 1 through view 1");
    ExpectRead(reader.Read(aliases.Address(0, 0, 0) + 4u, 1), 1, [](std::uint64_t, std::uint32_t subpage) { return SecondBase + subpage; }, what + ": the GPU read of word 1 through view 0");
    ExpectView(aliases.Words(1), 6, true, true, what + ", read by the CPU through view 1");
    ExpectView(aliases.Words(0), 6, true, true, what + ", read by the CPU through view 0");
    Settle();
}

void Run(MirrorReader& reader) {
    StoreThroughOtherView(reader);
    StoreThroughEachView(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency alias store", Run);
#else
    std::printf("skipped, dirty mirrors are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
