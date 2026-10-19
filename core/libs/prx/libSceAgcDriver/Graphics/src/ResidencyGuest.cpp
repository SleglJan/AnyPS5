#include "prx/libSceAgcDriver/Graphics/include/ResidencyGuest.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CommittedImports.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <iterator>
#include <mutex>

namespace AgcDriver::Graphics {

std::optional<ResidencyGuestImport> ResidencyGuest::ImportOf(std::uint64_t page) {
    const auto imported = Graphics::ImportOf(device, page);
    if (!imported) return std::nullopt;
    const auto backing = GuestMemory::Backing(page);
    ResidencyGuestImport result;
    result.attributes.whole = imported->bytes == ResidencyPageBytes && backing.uniform;
    result.attributes.cpuWritable = imported->cpuWritable;
    result.attributes.gpuWritable = true;
    result.attributes.aliased = backing.aliased && !backing.resident;
    result.attributes.section = backing.resident ? 0 : backing.section;
    result.attributes.offset = backing.offset;
    result.attributes.serial = imported->serial;
    result.attributes.resident = backing.resident;
    result.buffer = imported->buffer;
    result.offset = imported->offset;
    result.address = imported->address;
    return result;
}

std::vector<ResidencyRange> ResidencyGuest::Aliases(std::uint64_t address, std::uint64_t bytes) {
    std::vector<ResidencyRange> ranges;
    for (const auto& [alias, size] : GuestMemory::Aliases(address, static_cast<std::size_t>(bytes))) ranges.push_back({alias, size});
    return ranges;
}

std::uint64_t ResidencyGuest::Generation() {
    return GuestMemory::TrackerGeneration();
}

void ResidencyGuest::ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) {
    Require(changed.size() == generations.size(), "residency: changed pages and generations differ in count");
    if (generations.empty()) return;
    const auto address = firstPage << ResidencyPageShift;
    const auto bytes = static_cast<std::size_t>(generations.size() << ResidencyPageShift);
    GuestMemory::CollectWrites(address, bytes);
    static_cast<void>(GuestMemory::ChangedBlocks(address, bytes, generations, changed));
    for (std::size_t index = 0; index < generations.size(); ++index) {
        if (generations[index] == ResidencyUnchecked) changed[index] = GuestMemory::BlockUnchanged;
    }
}

static_assert(static_cast<std::size_t>(GuestArena::GuardRefusals) == ResidencyRefusals);
static_assert(static_cast<std::uint8_t>(GuestArena::GuardRefusal::UnsettledImport) == static_cast<std::uint8_t>(ResidencyRefusal::UnsettledImport));
static_assert(static_cast<std::uint8_t>(GuestArena::GuardRefusal::PrivateHostWrite) == static_cast<std::uint8_t>(ResidencyRefusal::PrivateHostWrite));
static_assert(static_cast<std::uint8_t>(GuestArena::GuardRefusal::PrivatePinned) == static_cast<std::uint8_t>(ResidencyRefusal::PrivatePinned));
static_assert(static_cast<std::uint8_t>(GuestArena::GuardRefusal::ResidentView) == static_cast<std::uint8_t>(ResidencyRefusal::ResidentView));
static_assert(static_cast<std::uint8_t>(GuestArena::GuardRefusal::SharedPinned) == static_cast<std::uint8_t>(ResidencyRefusal::SharedPinned));
static_assert(static_cast<std::uint8_t>(GuestArena::GuardRefusal::AliasHostWrite) == static_cast<std::uint8_t>(ResidencyRefusal::AliasHostWrite));
static_assert(static_cast<std::uint8_t>(GuestArena::GuardRefusal::Unavailable) == static_cast<std::uint8_t>(ResidencyRefusal::Unavailable));

ResidencyRefusal ResidencyGuest::Guard(std::uint64_t page) {
    static_cast<void>(GuestMemory::SettleWrites(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes)));
    return static_cast<ResidencyRefusal>(GuestMemory::GuardWrites(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes)));
}

void ResidencyGuest::SettleWrites(std::uint64_t page) {
    static_cast<void>(GuestMemory::SettleWrites(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes)));
}

bool ResidencyGuest::ImportSettled(std::uint64_t page) {
    return GuestMemory::HostImportSettled(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes));
}

std::uint64_t ResidencyGuest::ImportStamp() {
    return GuestMemory::HostImportStamp();
}

std::optional<ResidencyReuseStamp> ResidencyGuest::ReuseStamp() {
    const auto collect = GuestMemory::ThreadCollectEpoch();
    if (collect == 0) return std::nullopt;
    return ResidencyReuseStamp{GuestMemory::TrackerGeneration(), collect, Graphics::CommittedImportGeneration(device), GuestMemory::HostImportStamp()};
}

void ResidencyGuest::SettleImport(std::uint64_t page, std::uint64_t stamp) {
    GuestMemory::SettleHostImports(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes), stamp);
}

ResidencyRefusal ResidencyGuest::GuardAccess(std::uint64_t page, ResidencyMask mask) {
    const auto base = page << ResidencyPageShift;
    std::vector<std::pair<std::uint64_t, std::size_t>> guarded;
    for (unsigned first = 0; first < ResidencySubpages;) {
        if ((mask & (1u << first)) == 0) {
            ++first;
            continue;
        }
        auto last = first;
        while (last + 1 < ResidencySubpages && (mask & (1u << (last + 1))) != 0) ++last;
        const auto address = base + std::uint64_t{first} * ResidencySubpageBytes;
        const auto bytes = static_cast<std::size_t>(std::uint64_t{last - first + 1} * ResidencySubpageBytes);
        if (const auto refusal = GuestMemory::GuardAccess(address, bytes); refusal != GuestArena::GuardRefusal::None) {
            for (const auto& [done, size] : guarded) GuestMemory::UnguardAccess(done, size);
            return static_cast<ResidencyRefusal>(refusal);
        }
        guarded.emplace_back(address, bytes);
        first = last + 1;
    }
    return ResidencyRefusal::None;
}

void ResidencyGuest::UnguardAccess(std::uint64_t page) {
    GuestMemory::UnguardAccess(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes));
}

std::vector<ResidencyRange> ResidencyGuest::WholeImports(std::uint64_t& generation) {
    std::vector<ResidencyRange> ranges;
    std::uint64_t current = generation;
    const auto imported = Graphics::CommittedImportRanges(device, current);
    if (current == generation) return ranges;
    generation = current;
    ranges.reserve(imported.size());
    for (const auto& [begin, end] : imported) {
        const auto first = (begin + ResidencyPageBytes - 1) & ~(ResidencyPageBytes - 1);
        const auto last = end & ~(ResidencyPageBytes - 1);
        if (last > first) ranges.push_back({first, last - first});
    }
    return ranges;
}

ResidencyWrites ResidencyGuest::PageWrites(std::uint64_t page, std::uint64_t generation) {
    const auto writes = GuestMemory::ResidencyPageWrites(page, generation);
    return {writes.cpuSubpages, writes.other, writes.generation};
}

void ResidencyGuest::CollectWrites(std::uint64_t address, std::uint64_t bytes) {
    static_cast<void>(GuestMemory::CollectWritesUncached(address, static_cast<std::size_t>(bytes)));
}

std::uint64_t ResidencyGuest::GuestBytes() {
    return CommittedImportCounters().viewBytes;
}

void ResidencyGuest::Unguard(std::span<const std::uint64_t> pages) {
    for (const auto page : pages) GuestMemory::UnguardWrites(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes));
}

ResidencyPolicy DeviceResidencyPolicy(const Context& context) {
    VkDeviceSize deviceLocal = 0;
    for (std::uint32_t heap = 0; heap < context.memory.memoryHeapCount; ++heap) {
        if ((context.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) deviceLocal = std::max(deviceLocal, context.memory.memoryHeaps[heap].size);
    }
    Require(deviceLocal != 0, "residency: the device has no device-local heap");
    ResidencyPolicy policy;
    policy.budgetBytes = ResidencyMeasuredBudget;
    return policy;
}

namespace {

bool anyResident(std::uint64_t address, std::size_t bytes) {
    auto* driver = ResidencyDriver::Active();
    return driver != nullptr && driver->Manager().AnyResident(address, bytes);
}

void cpuWritten(std::span<const std::uint64_t> pages) {
    if (auto* driver = ResidencyDriver::Active()) driver->Manager().QueueCpuWritten(pages);
}

void dropRange(std::uint64_t address, std::size_t bytes) {
    if (auto* driver = ResidencyDriver::Active()) driver->Manager().DropRange(address, bytes);
}

void writeBack(std::uint64_t address, std::size_t bytes) {
    auto* driver = ResidencyDriver::Active();
    if (driver == nullptr || (!driver->Manager().AnyResident(address, bytes) && !driver->Manager().StaleIn(address, bytes)) || driver->Manager().CpuReadSettled(address, bytes)) return;
    if (!GuestMemory::GpuMutex().HeldByThisThread() && driver->Manager().FlushForCpuReadSettled(address, bytes)) return;
    std::unique_lock lock(GuestMemory::GpuMutex());
    if (GuestMemory::GpuMutex().DepthOnThisThread() != 1) {
        driver->Manager().FlushForCpuRead(address, bytes);
        return;
    }
    static_cast<void>(driver->Manager().FlushForCpuReadUnlocked(address, bytes));
}

void accessFault(std::uint64_t page) {
    if (auto* driver = ResidencyDriver::Active()) driver->AccessFault(page);
}

void refusalCleared(std::uint64_t address, std::size_t bytes, std::uint32_t cleared) {
    if (auto* driver = ResidencyDriver::Active()) driver->Manager().QueueRefusalCleared(address, bytes, cleared);
}

const GuestMemory::ResidencyHooks residencyHooks{&anyResident, &cpuWritten, &dropRange, &writeBack, &accessFault, &refusalCleared};

}

void InstallResidencyHooks() {
    GuestMemory::SetResidencyHooks(&residencyHooks);
}

void RemoveResidencyHooks() {
    GuestMemory::SetResidencyHooks(nullptr);
}

void NoteImportWritten(std::uint64_t address, std::uint64_t bytes) {
    if (auto* driver = ResidencyDriver::Active(); driver != nullptr && bytes != 0) driver->Manager().NoteImportWritten(address, bytes);
}

void RefreshImportWritten(std::uint64_t address, std::uint64_t bytes) {
    if (auto* driver = ResidencyDriver::Active(); driver != nullptr && bytes != 0) driver->RefreshImportWritten(address, bytes);
}

void StoreThroughMirrors(std::uint64_t address, std::span<const std::byte> bytes) {
    if (auto* driver = ResidencyDriver::Active(); driver != nullptr && !bytes.empty()) driver->StoreThroughMirrors(address, bytes);
}

std::vector<ResidencyRange> MirrorStoreSpans(std::uint64_t address, std::uint64_t bytes) {
    std::vector<ResidencyRange> spans;
    if (ResidencyDriver::Active() == nullptr || bytes == 0) return spans;
    for (const auto& [alias, size] : GuestMemory::Aliases(address, static_cast<std::size_t>(bytes))) spans.push_back({alias, size});
    spans.push_back({address, bytes});
    return spans;
}

bool AnyMirrored(std::span<const ResidencyRange> spans) {
    auto* driver = ResidencyDriver::Active();
    if (driver == nullptr) return false;
    return std::any_of(spans.begin(), spans.end(), [&](const ResidencyRange& span) { return driver->Manager().AnyResident(span.address, span.bytes); });
}

void PrepareImportWrite(std::uint64_t address, std::uint64_t bytes) {
    if (auto* driver = ResidencyDriver::Active(); driver != nullptr && bytes != 0) driver->PrepareImportWrite(address, bytes);
}

std::vector<ShaderRecompiler::BdaAbi::Range> WithResidentPages(std::span<const ShaderRecompiler::BdaAbi::Range> ranges) {
    std::vector<ShaderRecompiler::BdaAbi::Range> result(ranges.begin(), ranges.end());
    auto* driver = ResidencyDriver::Active();
    if (driver == nullptr) return result;
    auto& manager = driver->Manager();
    struct Mirror {
        std::uint64_t begin;
        std::uint64_t end;
        std::uint64_t device;
    };
    std::vector<Mirror> mirrors;
    for (const auto& run : manager.ResidentRanges()) {
        for (auto page = run.address >> ResidencyPageShift; page < (run.address + run.bytes) >> ResidencyPageShift; ++page) {
            const auto mirror = manager.Translate(page);
            if (!mirror) continue;
            mirrors.push_back({page << ResidencyPageShift, (page + 1) << ResidencyPageShift, driver->ResidentAddress(page, *mirror)});
        }
    }
    if (mirrors.empty()) return result;
    std::sort(mirrors.begin(), mirrors.end(), [](const Mirror& left, const Mirror& right) { return left.begin < right.begin; });
    std::vector<ShaderRecompiler::BdaAbi::Range> merged;
    merged.reserve(ranges.size() + 2 * mirrors.size());
    auto next = mirrors.begin();
    for (const auto& range : ranges) {
        auto begin = range.begin;
        while (next != mirrors.end() && next->end <= begin) ++next;
        for (auto page = next; page != mirrors.end() && page->begin < range.end; ++page) {
            const auto low = std::max(page->begin, range.begin);
            const auto high = std::min(page->end, range.end);
            if (low > begin) merged.push_back({begin, low, range.deviceAddress + (begin - range.begin), range.permissions, range.reserved});
            const auto flags = page->device == driver->SparseAddress() + page->begin ? ShaderRecompiler::BdaAbi::RangeMirrored : 0u;
            merged.push_back({low, high, page->device + (low - page->begin), range.permissions, range.reserved | flags});
            begin = high;
        }
        if (begin < range.end) merged.push_back({begin, range.end, range.deviceAddress + (begin - range.begin), range.permissions, range.reserved});
    }
    return merged;
}

}
