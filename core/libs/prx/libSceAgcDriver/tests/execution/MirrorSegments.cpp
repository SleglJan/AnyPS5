#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include <memory>
#include <optional>
#include <string>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;
namespace BdaAbi = ShaderRecompiler::BdaAbi;
using AgcDriver::Graphics::BdaPageTable;

constexpr std::uint32_t Tag = 0x4du;
constexpr std::uint32_t Pages = 3;

AgcDriver::Graphics::Context ContextOf(std::uint64_t residentAddress) {
    std::optional<AgcDriver::Graphics::Context> context;
    Locked([&] {
        const auto slot = Residency().Manager().Slot(PageOf(residentAddress));
        Require(slot.has_value(), "mirror segments: the page to take a context from is not resident");
        context = Residency().Mirrors().Chunk(slot->chunk)->context;
    });
    return *context;
}

std::shared_ptr<const BdaPageTable> Table(const AgcDriver::Graphics::Context& context) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    AgcDriver::Graphics::GuestBufferMemory memory(context);
    memory.AcquireRegistered();
    memory.Upload();
    auto table = memory.PageTable(context);
    memory.Release();
    return table;
}

std::string Describe(const std::optional<BdaAbi::Range>& range) {
    if (!range) return "no entry";
    return "entry " + Hex(range->begin) + ".." + Hex(range->end) + " at " + Hex(range->deviceAddress) + " reserved " + std::to_string(range->reserved);
}

bool MirroredAtGuestOffset(const std::optional<BdaAbi::Range>& range) {
    return range.has_value() && range->reserved == BdaAbi::RangeMirrored && range->deviceAddress == Residency().SparseAddress() + range->begin;
}

void ExpectPattern(MirrorReader& reader, std::uint64_t address, const std::string& what) {
    ExpectRead(reader.Read(address, Pages), Pages, [](std::uint64_t page, std::uint32_t subpage) { return Pattern(Tag, page, subpage * WordsPerSubpage); }, what);
}

void TouchingMirrorsFormOneSegment(MirrorReader& reader) {
    SharedPages pages(Pages);
    Fill(pages.Words(), Pages, Tag);
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(), pages.Bytes());
    for (std::uint32_t page = 0; page < Pages; ++page) Use(pages.Address(page));
    Settle();
    for (std::uint32_t page = 0; page < Pages; ++page) Expect(Resident(pages.Address(page)), "one segment: page " + std::to_string(page) + " has no mirror under full residency");
    const auto context = ContextOf(pages.Address(0));
    const auto table = Table(context);
    Expect(table != nullptr, "one segment: the address space has no page table");
    if (table == nullptr) return;
    const auto first = table->Find(pages.Address(0));
    Expect(MirroredAtGuestOffset(first) && first->begin <= pages.Address(0) && pages.Address(Pages) <= first->end, "one segment: three touching bound mirror pages are not one mirrored page-table entry at their guest offset in the sparse buffer: " + Describe(first));
    for (std::uint32_t page = 1; page < Pages; ++page) {
        const auto inside = table->Find(pages.Address(page) - 4u);
        const auto edge = table->Find(pages.Address(page));
        Expect(first.has_value() && inside.has_value() && edge.has_value() && inside->begin == first->begin && edge->begin == first->begin, "one segment: the addresses on both sides of page edge " + std::to_string(page) + " resolve to different entries: " + Describe(inside) + " and " + Describe(edge));
    }
    ExpectPattern(reader, pages.Address(), "one segment: the walk program's read across both page edges");

    Locked([&] { Residency().Manager().Demote(pages.Address(1), PageBytes); });
    Expect(!Resident(pages.Address(1)), "broken run: the middle page kept its mirror after the demotion");
    const auto broken = Table(context);
    Expect(broken != nullptr, "broken run: the address space has no page table");
    if (broken == nullptr) return;
    const auto before = broken->Find(pages.Address(0));
    const auto middle = broken->Find(pages.Address(1));
    const auto after = broken->Find(pages.Address(2));
    Expect(MirroredAtGuestOffset(before) && before->end == pages.Address(1), "broken run: the mirror before the import-served page does not end at it: " + Describe(before));
    Expect(middle.has_value() && middle->reserved == 0u, "broken run: the import-served page is not served by its import: " + Describe(middle));
    Expect(MirroredAtGuestOffset(after) && after->begin == pages.Address(2), "broken run: the mirror after the import-served page does not start at it: " + Describe(after));
    ExpectPattern(reader, pages.Address(), "broken run: the walk program's read across the import-served page");
    Settle();
}

void Run(MirrorReader& reader) {
    TouchingMirrorsFormOneSegment(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("mirror segment", Run);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
