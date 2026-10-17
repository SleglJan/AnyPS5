#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_MIRRORCHECKS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_MIRRORCHECKS_HPP

#include "NewPathFixture.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#ifdef _WIN32

namespace MirrorChecks {

using namespace NewPath;
using Range = std::pair<std::uint64_t, std::uint64_t>;

enum class Kind { Private, Shared };

inline const char* NameOf(Kind kind) {
    return kind == Kind::Shared ? "a shared view" : "private pages";
}

inline void PrepareResidency() {
    MirrorReader reader;
    PrivatePages warm(1);
    static_cast<void>(reader.Read(warm.Address(), 1));
    Settle();
    Locked([] { Residency().Manager().SetBudget(std::uint64_t{1} << 30u); });
}

class Pages {
public:
    Pages(Kind kind, std::uint32_t count) : kind(kind) {
        if (kind == Kind::Shared) shared = std::make_unique<SharedPages>(count);
        else privatePages = std::make_unique<PrivatePages>(count);
    }

    std::uint64_t Address(std::uint64_t offset = 0) const { return (shared != nullptr ? shared->Address() : privatePages->Address()) + offset; }
    volatile std::uint32_t* Words() const { return shared != nullptr ? shared->Words() : privatePages->Words(); }
    std::uint64_t Bytes() const { return shared != nullptr ? shared->Bytes() : privatePages->Bytes(); }
    Kind MemoryKind() const { return kind; }

private:
    Kind kind;
    std::unique_ptr<SharedPages> shared;
    std::unique_ptr<PrivatePages> privatePages;
};

inline AgcDriver::Graphics::ResidencyCounts Counts() {
    AgcDriver::Graphics::ResidencyCounts counts;
    Locked([&] { counts = Residency().Manager().Counts(); });
    return counts;
}

inline std::uint32_t StoredMask(std::uint64_t page, std::span<const Range> stored) {
    const auto pageBegin = page * PageBytes;
    const auto pageEnd = pageBegin + PageBytes;
    std::uint32_t mask = 0;
    for (const auto& [begin, end] : stored) {
        const auto low = std::max(begin, pageBegin);
        const auto high = std::min(end, pageEnd);
        if (low >= high) continue;
        for (auto subpage = (low - pageBegin) / SubpageBytes; subpage <= (high - 1u - pageBegin) / SubpageBytes; ++subpage) mask |= 1u << subpage;
    }
    return mask;
}

inline bool Overlaps(std::uint64_t page, std::span<const Range> ranges) {
    return std::ranges::any_of(ranges, [&](const Range& range) { return range.first < (page + 1u) * PageBytes && page * PageBytes < range.second; });
}

inline std::string StoresHeld(Kind kind, Range area, std::span<const Range> stored, std::span<const Range> allowed, const AgcDriver::Graphics::ResidencyCounts& before, std::span<const Range> cpuWritten = {}) {
    std::string failure;
    Locked([&] {
        auto& manager = Residency().Manager();
        const auto counts = manager.Counts();
        if (counts.writtenBackBytes != before.writtenBackBytes || counts.cpuWriteBacks != before.cpuWriteBacks) {
            failure = std::to_string(counts.writtenBackBytes - before.writtenBackBytes) + " bytes were written back from mirrors in " + std::to_string(counts.cpuWriteBacks - before.cpuWriteBacks) + " CPU write-backs before the CPU's read";
            return;
        }
        for (auto page = area.first / PageBytes; page * PageBytes < area.second; ++page) {
            const auto required = kind == Kind::Shared ? StoredMask(page, stored) : 0u;
            const auto permitted = kind == Kind::Shared ? StoredMask(page, allowed) | required : 0u;
            const auto resident = manager.Slot(page).has_value();
            const auto dirty = static_cast<std::uint32_t>(manager.Dirty(page));
            const auto dropped = kind == Kind::Shared && required == 0u && Overlaps(page, cpuWritten);
            if (dropped && resident) {
                failure = "page " + Hex(page * PageBytes) + " of the shared view keeps its mirror after the CPU wrote it, though no GPU work used it";
                return;
            }
            if (kind == Kind::Shared && !dropped && !resident) {
                failure = "page " + Hex(page * PageBytes) + " of the shared view has no mirror after the dispatch";
                return;
            }
            if ((dirty & required) != required || (dirty & ~permitted) != 0u) {
                failure = "page " + Hex(page * PageBytes) + " has dirty subpages " + Hex(dirty) + ", expected " + (required == permitted ? Hex(required) : Hex(required) + " up to " + Hex(permitted)) + (kind == Kind::Shared ? " (the subpages the dispatch stored to)" : " (private pages keep their stores in the import)");
                return;
            }
        }
    });
    return failure;
}

}

#endif

#endif
