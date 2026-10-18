#include "ResidencyRulesFixture.hpp"

#ifdef _WIN32

namespace {

using namespace ResidencyRules;
namespace BdaAbi = ShaderRecompiler::BdaAbi;

constexpr std::uint32_t AliasStore = 0x5b000003u;
constexpr std::uint32_t DriverValue = 0x5b000004u;
constexpr std::uint64_t PublishedDevice = 0x100000000ull;

void SharedMirrorsTakeStores(MirrorReader& reader) {
    SharedPages pages(2);
    Fill(pages.Words(), 2, 1);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    Use(pages.Address(0));
    Expect(Resident(pages.Address(0)), "rule 1: page 0 was not admitted");
    ExpectRead(reader.Read(pages.Address(), 1), 1, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(1, page, subpage * WordsPerSubpage); }, "rule 1: the read after admission");
    Settle();
    Expect(Resident(pages.Address(1)), "rule 1: page 1 of the shared view has no mirror after a dispatch under full residency");
    const std::array<BdaAbi::Range, 1> ranges{{{pages.Address(0), pages.Address(2), PublishedDevice, BdaAbi::Read | BdaAbi::Write, 0}}};
    std::vector<BdaAbi::Range> published;
    std::optional<std::uint64_t> mirror;
    Locked([&] {
        published = AgcDriver::Graphics::WithResidentPages(ranges);
        mirror = Residency().Manager().Translate(PageOf(pages.Address(0)));
        if (mirror) mirror = Residency().ResidentAddress(PageOf(pages.Address(0)), *mirror);
    });
    const auto find = [&](std::uint64_t begin) { return std::ranges::find_if(published, [&](const BdaAbi::Range& range) { return range.begin == begin; }); };
    const auto resident = find(pages.Address(0));
    std::optional<std::uint64_t> second;
    Locked([&] {
        second = Residency().Manager().Translate(PageOf(pages.Address(1)));
        if (second) second = Residency().ResidentAddress(PageOf(pages.Address(1)), *second);
    });
    const auto beside = find(pages.Address(1));
    Expect(mirror.has_value() && resident != published.end() && resident->end == pages.Address(1) && resident->deviceAddress == *mirror && resident->permissions == (BdaAbi::Read | BdaAbi::Write), "rule 1: a resident page of a shared view is not published at its mirror with the range's write permission");
    Expect(resident != published.end() && resident->deviceAddress == Residency().SparseAddress() + pages.Address(0) && resident->reserved == BdaAbi::RangeMirrored, "rule 1: a resident page of a shared view is not published at its guest offset in the sparse residency buffer with the mirrored flag");
    Expect(second.has_value() && beside != published.end() && beside->end == pages.Address(2) && beside->deviceAddress == *second && beside->permissions == (BdaAbi::Read | BdaAbi::Write), "rule 1: the resident page beside it is not published at its own mirror with the range's permissions");
    const auto before = Counts();
    Store(pages.Address(0));
    const auto after = Counts();
    Expect(Resident(pages.Address(0)) && after.demoted == before.demoted, "rule 1: a dispatch storing into a resident shared page demoted its mirror");
    Expect(after.writtenBackBytes == before.writtenBackBytes, "rule 1: a dispatch storing into a resident page wrote back " + std::to_string(after.writtenBackBytes - before.writtenBackBytes) + " bytes before any CPU read");
    AgcDriver::GuestMemory::FlushGpuWrites(pages.Address(0), PageBytes);
    Expect(Counts().dirty == 0u && Resident(pages.Address(0)), "rule 1: a CPU read left the page dirty or demoted it");
    for (std::uint32_t subpage = 0; subpage < SubpagesPerPage; ++subpage) {
        const auto stored = pages.Words(0)[subpage * WordsPerSubpage];
        if (stored == WrittenBase + subpage) continue;
        Expect(false, "rule 1: the store into a resident page did not reach guest memory: subpage " + std::to_string(subpage) + " holds " + Hex(stored));
        break;
    }
    ExpectRead(reader.Read(pages.Address(), 2), 2, [](std::uint64_t page, std::uint32_t subpage) { return page == 0 ? WrittenBase + subpage : Pattern(1, page, subpage * WordsPerSubpage); }, "rule 1: the read after the store");
    Settle();
}

void AliasStoreDemotes(MirrorReader& reader) {
    Aliases aliases(1);
    Fill(aliases.Words(0), 1, 2);
    aliases.Collect();
    Use(aliases.Address(0));
    Expect(Resident(aliases.Address(0)), "rule 2: the first view's page was not admitted");
    ExpectRead(reader.Read(aliases.Address(0), 1), 1, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(2, page, subpage * WordsPerSubpage); }, "rule 2: the read through the first view after admission");
    const auto before = Counts();
    Store(aliases.Address(1));
    const auto after = Counts();
    Expect((!Resident(aliases.Address(0)) && after.demoted > before.demoted) || after.refreshed > before.refreshed, "rule 2: a dispatch storing through the other view neither demoted nor refreshed the mirrored page");
    Expect(after.dirty == before.dirty, "rule 2: a store through the other view of an aliased page left a mirror dirty");
    ExpectRead(reader.Read(aliases.Address(0), 1), 1, [](std::uint64_t, std::uint32_t subpage) { return WrittenBase + subpage; }, "rule 2: the read through the first view after a store through the other view");
    Settle();
}

void UnboundedWriterKeepsMirrors(MirrorReader& reader) {
    SharedPages writable(3);
    PrivatePages addresses(1);
    Fill(writable.Words(), 3, 3);
    AgcDriver::GuestMemory::CollectWritesUncached(writable.Address(), writable.Bytes());
    const auto target = writable.Address(1, 0, 2);
    addresses.Words()[0] = Low(target);
    addresses.Words()[1] = High(target);
    AgcDriver::GuestMemory::CollectWritesUncached(addresses.Address(), addresses.Bytes());
    for (std::uint64_t page = 0; page < 3u; ++page) Use(writable.Address(page));
    Expect(Resident(writable.Address(0)) && Resident(writable.Address(1)) && Resident(writable.Address(2)), "rule 2: the three pages were not admitted");
    Commands commands;
    commands.Dispatch(UnboundedProgram(), BufferResource(addresses.Address(), 16u), 1);
    commands.Submit(0);
    WaitIdle();
    Expect(Resident(writable.Address(0)) && Resident(writable.Address(1)) && Resident(writable.Address(2)), "rule 2: a dispatch storing through an address it loads from memory demoted GPU-writable mirrors");
    AgcDriver::GuestMemory::FlushGpuWrites(writable.Address(1), PageBytes);
    Expect(writable.Words(1)[2] == UnboundedValue, "rule 2: the unbounded store did not reach guest memory: " + Hex(writable.Words(1)[2]));
    ExpectRead(reader.Read(writable.Address(), 3), 3, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(3, page, subpage * WordsPerSubpage); }, "rule 2: the read after the unbounded writer");
    Settle();
}

void DriverWriteThroughAlias(MirrorReader& reader) {
    Aliases aliases(1);
    Fill(aliases.Words(0), 1, 6);
    aliases.Collect();
    Use(aliases.Address(0));
    Expect(Resident(aliases.Address(0)), "rule 4: the first view's page was not admitted");
    ExpectRead(reader.Read(aliases.Address(0), 1), 1, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(6, page, subpage * WordsPerSubpage); }, "rule 4: the read after admission");
    Settle();
    Commands commands;
    commands.WriteData(aliases.Address(1, 0, 3), {DriverValue});
    reader.Record(commands, aliases.Address(0), 1);
    commands.Submit(0);
    ExpectRead(reader.Collect(), 1, [](std::uint64_t page, std::uint32_t subpage) { return subpage == 3u ? DriverValue : Pattern(6, page, subpage * WordsPerSubpage); }, "rule 4: a dispatch reading the mirrored view after a WRITE_DATA through the other view in the same submission");
    Settle();
}

void OneMirrorPerPhysicalPage(MirrorReader& reader) {
    Aliases aliases(1);
    Fill(aliases.Words(0), 1, 7);
    aliases.Collect();
    const auto before = Counts();
    Use(aliases.Address(0));
    Use(aliases.Address(1));
    const auto after = Counts();
    Expect(Resident(aliases.Address(0)) != Resident(aliases.Address(1)) && after.admitted == before.admitted + 1u && after.aliasSkipped == before.aliasSkipped + 1u, "rule 5: two views of one physical page took " + std::to_string(after.admitted - before.admitted) + " mirrors (aliases skipped " + std::to_string(after.aliasSkipped - before.aliasSkipped) + ")");
    const auto pattern = [](std::uint64_t page, std::uint32_t subpage) { return Pattern(7, page, subpage * WordsPerSubpage); };
    ExpectRead(reader.Read(aliases.Address(0), 1), 1, pattern, "rule 5: the read through the first view");
    ExpectRead(reader.Read(aliases.Address(1), 1), 1, pattern, "rule 5: the read through the second view");
    const auto mirrored = Resident(aliases.Address(0)) ? 0u : 1u;
    const auto other = 1u - mirrored;
    Locked([&] {
        aliases.Words(other)[6 * WordsPerSubpage] = AliasStore;
        aliases.Collect();
    });
    const auto stored = [&](std::uint64_t page, std::uint32_t subpage) { return subpage == 6u ? AliasStore : pattern(page, subpage); };
    ExpectRead(reader.Read(aliases.Address(mirrored), 1), 1, stored, "rule 5: the next dispatch reading the mirrored view after a CPU store through the other view of the same physical page");
    ExpectRead(reader.Read(aliases.Address(other), 1), 1, stored, "rule 5: the next dispatch reading the view that stored");
    Settle();
}

void Run(MirrorReader& reader) {
    SharedMirrorsTakeStores(reader);
    AliasStoreDemotes(reader);
    UnboundedWriterKeepsMirrors(reader);
    DriverWriteThroughAlias(reader);
    OneMirrorPerPhysicalPage(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("new path residency rule", Run);
#else
    std::printf("skipped, commit imports and write guards are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
