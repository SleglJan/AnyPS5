#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <map>
#include <mutex>
#include <set>

namespace AgcDriver::Graphics {

namespace {

std::atomic<ResidencyDriver*> activeDriver{nullptr};

std::uint32_t DeviceLocalHeap(const Context& context) {
    Require(context.memoryProperties2 != nullptr && context.physical != VK_NULL_HANDLE, "residency: the device reports no memory budget (VK_EXT_memory_budget), which bounds residency to the guest size less the device-local memory in use");
    std::optional<std::uint32_t> heap;
    for (std::uint32_t index = 0; index < context.memory.memoryHeapCount; ++index) {
        if ((context.memory.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        if (!heap || context.memory.memoryHeaps[index].size > context.memory.memoryHeaps[*heap].size) heap = index;
    }
    Require(heap.has_value(), "residency: the device has no device-local heap");
    return *heap;
}

}

ResidencyDriver::ResidencyDriver(const Context& context, Recorder& recorder, IResidencyGuestMemory& guest, const ResidencyPolicy& policy)
    : context(context), recorder(recorder), guest(guest), deviceHeap(DeviceLocalHeap(context)), mirrors(context, policy.chunkSlots), manager(*this, policy) {
    presents = Recorder::Presents();
    activeDriver.store(this);
}

ResidencyDriver::~ResidencyDriver() {
    ResidencyDriver* expected = this;
    activeDriver.compare_exchange_strong(expected, nullptr);
}

ResidencyDriver* ResidencyDriver::Active() {
    return activeDriver.load();
}

void ResidencyDriver::OrderingPoint() {
    orderingPoints.fetch_add(1, std::memory_order_relaxed);
    if (!manager.Advance()) return;
    const std::lock_guard lock(GuestMemory::GpuMutex());
    manager.Apply();
    appliedAtOrderingPoints.fetch_add(1, std::memory_order_relaxed);
}

void ResidencyDriver::BeforeDispatch() {
    const std::lock_guard lock(GuestMemory::GpuMutex());
    manager.Apply();
    if (const auto now = Recorder::Presents(); now != presents) {
        presents = now;
        frameBoundary();
    }
    manager.RefreshHot();
}

std::uint64_t ResidencyDriver::Budget() {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    const auto mirrorBytes = mirrors.DeviceBytes();
    const auto used = reported.heapUsage[deviceHeap];
    const auto others = used > mirrorBytes ? used - mirrorBytes : 0;
    auto limit = guest.GuestBytes();
    if (reported.heapBudget[deviceHeap] != 0) limit = std::min<std::uint64_t>(limit, reported.heapBudget[deviceHeap]);
    budgetBytes = limit > others ? limit - others : 0;
    return budgetBytes;
}

std::uint64_t ResidencyDriver::GuestBytes() {
    return guest.GuestBytes();
}

void ResidencyDriver::frameBoundary() {
    ++frames;
    manager.ApplyBudget();
}

std::vector<ResidencyRange> ResidencyDriver::WholeImports(std::uint64_t& generation) {
    return guest.WholeImports(generation);
}

void ResidencyDriver::NoteRecordedWrites(std::uint64_t address, std::uint64_t bytes) {
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    manager.QueueGpuWritten(address, bytes);
}

std::optional<ResidencyDriver::MirrorTarget> ResidencyDriver::MirrorOf(std::uint64_t address) {
    const auto page = address >> ResidencyPageShift;
    const auto slot = manager.WriteSlot(page);
    if (!slot) return std::nullopt;
    MirrorTarget target;
    target.buffer = mirrors.Chunk(slot->chunk)->buffer;
    target.offset = (VkDeviceSize{slot->slot} << ResidencyPageShift) + (address & (ResidencyPageBytes - 1));
    target.end = (page + 1) << ResidencyPageShift;
    recorder.Keep(mirrors.Chunk(slot->chunk));
    return target;
}

void ResidencyDriver::NoteDriverWrite(std::uint64_t address, std::uint64_t bytes) {
    manager.Pin(address, bytes, ResidencyAccess::Write, recorder.CallerToken());
    manager.NoteGpuWritten(address, bytes, true);
}

void ResidencyDriver::PrepareImportWrite(std::uint64_t address, std::uint64_t bytes) {
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    manager.PrepareImportWrite(address, bytes);
}

void ResidencyDriver::RefreshImportWritten(std::uint64_t address, std::uint64_t bytes) {
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    manager.RefreshImportWritten(address, bytes);
}

void ResidencyDriver::AccessFault(std::uint64_t page) {
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    recorder.SyncThrough(page << ResidencyPageShift, ResidencyPageBytes, lock.owns_lock());
    manager.FlushForCpuRead(page << ResidencyPageShift, ResidencyPageBytes);
}

void ResidencyDriver::UseForRead(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    constexpr std::uint64_t largest = std::uint64_t{8u} << 20u;
    const std::lock_guard lock(GuestMemory::GpuMutex());
    for (const auto& [begin, end] : ranges) {
        if (end <= begin || end - begin > largest) continue;
        if (!AnyShadowedOverlaps(begin, static_cast<std::size_t>(end - begin))) {
            manager.Use(begin, end - begin, ResidencyAccess::Read);
            continue;
        }
        auto mappings = ShadowMappings(begin, static_cast<std::size_t>(end - begin));
        std::sort(mappings.begin(), mappings.end(), [](const ShadowMapping& left, const ShadowMapping& right) { return left.begin < right.begin; });
        auto cursor = begin;
        for (const auto& mapping : mappings) {
            const auto low = std::max(mapping.begin, begin);
            const auto high = std::min(mapping.end, end);
            if (high <= low) continue;
            if (low > cursor) manager.Use(cursor, low - cursor, ResidencyAccess::Read);
            cursor = std::max(cursor, high);
        }
        if (cursor < end) manager.Use(cursor, end - cursor, ResidencyAccess::Read);
    }
}

void ResidencyDriver::DropMirrors(std::span<const ResidencyRange> ranges) {
    for (const auto& range : ranges) manager.Demote(range.address, range.bytes);
}

void ResidencyDriver::BeforeWritingDispatch(std::span<const ResidencyRange> grants, bool writableImports) {
    const std::lock_guard lock(GuestMemory::GpuMutex());
    for (const auto& grant : grants) manager.NoteGpuWritten(grant.address, grant.bytes);
    if (writableImports) manager.DemoteWritable();
    for (const auto& grant : grants) manager.Pin(grant.address, grant.bytes, ResidencyAccess::Write, recorder.CallerToken());
}

ResidencyPublication ResidencyDriver::Publish() {
    ResidencyPublication publication;
    const auto changes = manager.TakeChanges();
    publication.serial = manager.Serial();
    publication.changed.reserve(changes.size());
    for (const auto page : changes) publication.changed.push_back({page, manager.Translate(page).value_or(0)});
    return publication;
}

std::vector<Recorder::SectionObject> ResidencyDriver::ChunkObjects(bool written) const {
    std::vector<Recorder::SectionObject> objects;
    for (const auto chunk : manager.Chunks()) objects.push_back({mirrors.Chunk(chunk).get(), written});
    return objects;
}

void ResidencyDriver::KeepChunks() {
    for (const auto chunk : manager.Chunks()) recorder.Keep(mirrors.Chunk(chunk));
}

ResidencyDriverCounts ResidencyDriver::Counts() const {
    return {orderingPoints.load(std::memory_order_relaxed), appliedAtOrderingPoints.load(std::memory_order_relaxed), vanishedCopies, frames, budgetBytes};
}

bool ResidencyDriver::Reached(const ResidencyToken& token) {
    return recorder.Reached(token);
}

void ResidencyDriver::Wait(const ResidencyToken& token) {
    recorder.Submit();
    recorder.FinishToken(token);
}

ResidencyToken ResidencyDriver::Newest() {
    return recorder.AllToken();
}

ResidencyToken ResidencyDriver::Current() {
    return recorder.CallerToken();
}

ResidencyImport ResidencyDriver::Import(std::uint64_t page) {
    const auto import = guest.ImportOf(page);
    return import ? import->attributes : ResidencyImport{};
}

std::vector<ResidencyRange> ResidencyDriver::Aliases(std::uint64_t address, std::uint64_t bytes) {
    return guest.Aliases(address, bytes);
}

std::uint64_t ResidencyDriver::Generation() {
    return guest.Generation();
}

void ResidencyDriver::ChangedPages(std::uint64_t firstPage, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) {
    guest.ChangedPages(firstPage, generations, changed);
}

std::uint64_t ResidencyDriver::AllocateChunk(std::uint32_t chunk, std::uint32_t) {
    return mirrors.Allocate(chunk);
}

void ResidencyDriver::ReleaseChunk(std::uint32_t chunk) {
    mirrors.Release(chunk);
}

ResidencyToken ResidencyDriver::CopyIn(std::span<const ResidencyCopy> copies) {
    return record(copies, true);
}

ResidencyToken ResidencyDriver::WriteBack(std::span<const ResidencyCopy> copies) {
    return record(copies, false);
}

bool ResidencyDriver::Guard(std::uint64_t page) {
    return guest.Guard(page);
}

void ResidencyDriver::Unguard(std::span<const std::uint64_t> pages) {
    guest.Unguard(pages);
}

bool ResidencyDriver::GuardAccess(std::uint64_t page, ResidencyMask mask) {
    return guest.GuardAccess(page, mask);
}

void ResidencyDriver::UnguardAccess(std::uint64_t page) {
    guest.UnguardAccess(page);
}

ResidencyToken ResidencyDriver::record(std::span<const ResidencyCopy> copies, bool intoMirror) {
    std::map<std::uint64_t, ResidencyImportLocation> locations;
    std::vector<ResidencyCopy> located;
    std::set<std::uint32_t> chunks;
    for (const auto& copy : copies) {
        auto found = locations.find(copy.page);
        if (found == locations.end()) {
            const auto import = guest.ImportOf(copy.page);
            found = locations.emplace(copy.page, import ? ResidencyImportLocation{import->buffer, import->offset} : ResidencyImportLocation{}).first;
        }
        if (found->second.buffer == VK_NULL_HANDLE) {
            ++vanishedCopies;
            continue;
        }
        located.push_back(copy);
        chunks.insert(copy.mirror.chunk);
    }
    if (located.empty()) return recorder.CallerToken();
    for (const auto& copy : located) {
        const auto address = (copy.page << ResidencyPageShift) + copy.offset;
        recorder.FlushStoresOverlapping(address, copy.bytes);
        recorder.FlushKeyStoresOverlapping(address, copy.bytes);
    }
    if (intoMirror) {
        std::vector<Recorder::SectionObject> objects;
        for (const auto chunk : chunks) objects.push_back({mirrors.Chunk(chunk).get(), true});
        recorder.DeclareSection(objects);
    }
    const auto commands = recorder.Commands();
    const ResidencyImportLocator locator = [&](std::uint64_t page) { return locations.at(page); };
    if (intoMirror) mirrors.RecordCopyIn(commands, located, locator);
    else mirrors.RecordWriteBack(commands, located, locator);
    recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    for (const auto chunk : chunks) recorder.Keep(mirrors.Chunk(chunk));
    return recorder.CallerToken();
}

}
