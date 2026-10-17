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

bool ResidencyGuest::Guard(std::uint64_t page) {
    static_cast<void>(GuestMemory::SettleWrites(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes)));
    return GuestMemory::GuardWrites(page << ResidencyPageShift, static_cast<std::size_t>(ResidencyPageBytes));
}

bool ResidencyGuest::GuardAccess(std::uint64_t page, ResidencyMask mask) {
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
        if (!GuestMemory::GuardAccess(address, bytes)) {
            for (const auto& [done, size] : guarded) GuestMemory::UnguardAccess(done, size);
            return false;
        }
        guarded.emplace_back(address, bytes);
        first = last + 1;
    }
    return true;
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
    const std::lock_guard lock(GuestMemory::GpuMutex());
    driver->Manager().FlushForCpuRead(address, bytes);
}

void accessFault(std::uint64_t page) {
    if (auto* driver = ResidencyDriver::Active()) driver->AccessFault(page);
}

const GuestMemory::ResidencyHooks residencyHooks{&anyResident, &cpuWritten, &dropRange, &writeBack, &accessFault};

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
            mirrors.push_back({page << ResidencyPageShift, (page + 1) << ResidencyPageShift, *mirror});
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
            merged.push_back({low, high, page->device + (low - page->begin), range.permissions, range.reserved});
            begin = high;
        }
        if (begin < range.end) merged.push_back({begin, range.end, range.deviceAddress + (begin - range.begin), range.permissions, range.reserved});
    }
    return merged;
}

}
