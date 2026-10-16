#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "BdaTouchAbi.hpp"
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
    harvestTouches();
    if (const auto now = Recorder::Presents(); now != presents || touchTable == nullptr) {
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

ResidencyWrites ResidencyDriver::PageWrites(std::uint64_t page, std::uint64_t generation) {
    return guest.PageWrites(page, generation);
}

VkDeviceAddress ResidencyDriver::TouchTableAddress() const {
    return touchAddress.load(std::memory_order_acquire);
}

void ResidencyDriver::frameBoundary() {
    namespace Touch = ShaderRecompiler::BdaTouchAbi;
    ++frames;
    manager.ApplyBudget();
    if (!context.bufferDeviceAddress) return;
    const auto commands = recorder.Commands();
    const auto fill = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    const auto copy = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    const VkMemoryBarrier between{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
    const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
    if (touchTable == nullptr) {
        touchTable = std::make_shared<Buffer>(context, Touch::TableBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        fill(commands, touchTable->Handle(), 0, Touch::TableBytes, 0u);
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        recorder.Keep(touchTable);
        touchAddress.store(touchTable->DeviceAddress(), std::memory_order_release);
        return;
    }
    auto* readback = &touchReadbacks[0];
    if (readback->pending) readback = &touchReadbacks[1];
    if (readback->pending) return;
    if (readback->buffer == nullptr) readback->buffer = std::make_shared<Buffer>(context, Touch::TableBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    const VkBufferCopy region{0, 0, Touch::TableBytes};
    copy(commands, touchTable->Handle(), readback->buffer->Handle(), 1, &region);
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &between, 0, nullptr, 0, nullptr);
    fill(commands, touchTable->Handle(), 0, Touch::TableBytes, 0u);
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    recorder.MarkCovered(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    recorder.Keep(touchTable);
    recorder.Keep(readback->buffer);
    readback->token = recorder.CallerToken();
    readback->pending = true;
}

void ResidencyDriver::harvestTouches() {
    namespace Touch = ShaderRecompiler::BdaTouchAbi;
    for (auto& readback : touchReadbacks) {
        if (!readback.pending || !recorder.Reached(readback.token)) continue;
        readback.pending = false;
        readback.buffer->Invalidate();
        const auto bytes = readback.buffer->Bytes();
        const auto* words = reinterpret_cast<const std::uint32_t*>(bytes.data());
        if (words[Touch::OverflowWord] != 0) ++touchOverflows;
        std::vector<std::uint64_t> touched;
        Touch::ForEachTouchedPage(words, [&](std::uint64_t page) { touched.push_back(page); });
        ++touchHarvests;
        touchedPages += touched.size();
        manager.AdmitTouched(touched);
    }
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

void ResidencyDriver::WriteBackAll() {
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    manager.WriteBackAll();
}

void ResidencyDriver::UseForRead(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    constexpr std::uint64_t largest = std::uint64_t{8u} << 20u;
    const std::lock_guard lock(GuestMemory::GpuMutex());
    for (const auto& [begin, end] : ranges) {
        if (end <= begin || end - begin > largest) continue;
        manager.Use(begin, end - begin, ResidencyAccess::Read);
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
    return {orderingPoints.load(std::memory_order_relaxed), appliedAtOrderingPoints.load(std::memory_order_relaxed), vanishedCopies, frames, touchHarvests, touchedPages, touchOverflows, budgetBytes};
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
