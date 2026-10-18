#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"

#include <chrono>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;

constexpr std::uint32_t Dispatches = 16;
constexpr std::uint32_t Tag = 9;
constexpr std::uint32_t EagerPages = 2048;
constexpr std::uint64_t EagerPagesPerFrame = 512;

struct Work {
    ResidencyCounts residency;
    AgcDriver::Graphics::BdaPageTable::Counts tables;
    double microseconds = 0;
};

Work Between(const ResidencyCounts& before, const ResidencyCounts& after, const AgcDriver::Graphics::BdaPageTable::Counts& tablesBefore, double microseconds) {
    const auto tablesAfter = AgcDriver::Graphics::BdaPageTable::Totals();
    Work work;
    work.residency.admitted = after.admitted - before.admitted;
    work.residency.resident = after.resident;
    work.tables.tables = tablesAfter.tables - tablesBefore.tables;
    work.tables.encodedWindows = tablesAfter.encodedWindows - tablesBefore.encodedWindows;
    work.tables.sharedWindows = tablesAfter.sharedWindows - tablesBefore.sharedWindows;
    work.tables.encodedPages = tablesAfter.encodedPages - tablesBefore.encodedPages;
    work.tables.patchedWindows = tablesAfter.patchedWindows - tablesBefore.patchedWindows;
    work.microseconds = microseconds;
    return work;
}

std::string Describe(const Work& work) {
    return std::to_string(work.residency.resident) + " resident: per " + std::to_string(Dispatches) + " dispatches " + std::to_string(work.residency.admitted) + " admitted, " + std::to_string(work.tables.tables) + " page tables, " + std::to_string(work.tables.encodedWindows) + " encoded and " + std::to_string(work.tables.patchedWindows) + " patched windows, " + std::to_string(work.tables.encodedPages) + " encoded pages; " + std::to_string(work.microseconds) + " us per dispatch";
}

void Submit(MirrorReader& reader, std::uint64_t address) {
    Commands commands;
    reader.Record(commands, address, 1);
    commands.Submit(0);
    WaitIdle();
}

Work Dispatch(MirrorReader& reader, std::uint64_t address, const std::function<void(std::uint32_t)>& before = {}) {
    WaitIdle();
    const auto counts = Counts();
    const auto tables = AgcDriver::Graphics::BdaPageTable::Totals();
    double spent = 0;
    for (std::uint32_t dispatch = 0; dispatch < Dispatches; ++dispatch) {
        if (before) before(dispatch);
        const auto started = std::chrono::steady_clock::now();
        Submit(reader, address);
        spent += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
    }
    return Between(counts, Counts(), tables, spent / Dispatches);
}

void Admission(MirrorReader& reader) {
    PrivatePages target(1);
    Fill(target.Words(), 1, Tag);
    static_cast<void>(reader.Read(target.Address(), 1));
    SharedPages pending(EagerPages);
    const auto quiet = Dispatch(reader, target.Address());
    std::printf("[residency-work] %u whole shared pages waiting for eager admission, no frame boundary: %s\n", EagerPages, Describe(quiet).c_str());
    Expect(quiet.residency.admitted == 0u, "admission: dispatches inside one frame admitted " + std::to_string(quiet.residency.admitted) + " pages eagerly: " + Describe(quiet));
    const auto residentPending = [&] {
        std::uint64_t count = 0;
        Locked([&] {
            for (std::uint32_t page = 0; page < EagerPages; ++page) count += Residency().Manager().Slot(PageOf(pending.Address(page))).has_value() ? 1u : 0u;
        });
        return count;
    };
    std::uint32_t frames = 0;
    std::uint64_t largest = 0;
    std::uint64_t total = residentPending();
    while (total < EagerPages && frames < 2u * EagerPages / EagerPagesPerFrame) {
        Residency().Flip();
        ++frames;
        const auto frame = Dispatch(reader, target.Address());
        const auto now = residentPending();
        largest = std::max<std::uint64_t>(largest, now - total);
        total = now;
        std::printf("[residency-work] frame %u: %llu of the shared pages resident; %s\n", frames, static_cast<unsigned long long>(now), Describe(frame).c_str());
    }
    Expect(largest <= EagerPagesPerFrame, "admission: one frame admitted " + std::to_string(largest) + " pages eagerly, more than " + std::to_string(EagerPagesPerFrame));
    Expect(total == EagerPages, "admission: eager admission filled " + std::to_string(total) + " of " + std::to_string(EagerPages) + " whole shared pages within " + std::to_string(frames) + " frames");
    Settle();
}

Work OnePageChanged(MirrorReader& reader, PrivatePages& target, std::uint32_t sharedPages, std::uint32_t round) {
    PrivatePages fresh(Dispatches);
    Fill(fresh.Words(), Dispatches, Tag + round);
    AgcDriver::GuestMemory::CollectWritesUncached(fresh.Address(), fresh.Bytes());
    SharedPages shared(sharedPages);
    Fill(shared.Words(), sharedPages, Tag + round + 1u);
    AgcDriver::GuestMemory::CollectWritesUncached(shared.Address(), shared.Bytes());
    AdmitEagerly();
    Submit(reader, target.Address());
    const auto changing = Dispatch(reader, target.Address(), [&](std::uint32_t dispatch) { Use(fresh.Address(std::uint64_t{dispatch} * PageBytes)); });
    std::printf("[residency-work] one page changed per dispatch beside %u resident shared pages: %s\n", sharedPages, Describe(changing).c_str());
    const auto quiet = Dispatch(reader, target.Address());
    std::printf("[residency-work] nothing changed beside %u resident shared pages: %s\n", sharedPages, Describe(quiet).c_str());
    Expect(quiet.tables.tables == 0u && quiet.tables.encodedPages == 0u, "page table: dispatches with no residency change built " + std::to_string(quiet.tables.tables) + " page tables: " + Describe(quiet));
    ExpectRead(reader.Read(fresh.Address(), MirrorReader::Pages), MirrorReader::Pages, [&](std::uint64_t page, std::uint32_t subpage) { return Pattern(Tag + round, page, subpage * WordsPerSubpage); }, "page table: the pages admitted one per dispatch beside " + std::to_string(sharedPages) + " shared pages");
    ExpectRead(reader.Read(shared.Address(sharedPages - MirrorReader::Pages), MirrorReader::Pages), MirrorReader::Pages, [&](std::uint64_t page, std::uint32_t subpage) { return Pattern(Tag + round + 1u, sharedPages - MirrorReader::Pages + page, subpage * WordsPerSubpage); }, "page table: the last resident shared pages beside the changed ones");
    Settle();
    return changing;
}

void PageTable(MirrorReader& reader) {
    PrivatePages target(1);
    Fill(target.Words(), 1, Tag);
    static_cast<void>(reader.Read(target.Address(), 1));
    const auto few = OnePageChanged(reader, target, 64, 1);
    const auto many = OnePageChanged(reader, target, 4096, 3);
    Expect(many.tables.encodedPages <= few.tables.encodedPages + Dispatches, "page table: the page-table work of one changed page per dispatch grows with the resident pages: beside 64 shared pages " + Describe(few) + "; beside 4096 " + Describe(many));
}

void Run(MirrorReader& reader) {
    Admission(reader);
    PageTable(reader);
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency dispatch work", Run);
#else
    std::printf("skipped, residency write tracking is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
