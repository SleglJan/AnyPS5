#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceMemoryAccounting.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
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
    : context(context), recorder(recorder), guest(guest), deviceHeap(DeviceLocalHeap(context)), mirrors(context, policy.chunkSlots), sparse(SparseResidency::Of(context)), manager(*this, policy) {
    activeDriver.store(this);
}

ResidencyDriver::~ResidencyDriver() {
    ResidencyDriver* expected = this;
    activeDriver.compare_exchange_strong(expected, nullptr);
    std::vector<VkSparseMemoryBind> binds;
    for (const auto& [page, mirror] : bound) binds.push_back({page << ResidencyPageShift, ResidencyPageBytes, VK_NULL_HANDLE, 0, 0});
    if (binds.empty()) return;
    std::unique_lock gpu(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) gpu.lock();
    if (context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(context.device) != VK_SUCCESS) return;
    sparse->Finish(sparse->Bind(binds));
    bindWaits.fetch_add(1, std::memory_order_relaxed);
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

void ResidencyDriver::Flip() {
    flips.fetch_add(1, std::memory_order_acq_rel);
    manager.NextFrame();
}

void ResidencyDriver::BeforeCompletionSignal() {
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    manager.RetileUnguarded();
}

void PublishForCompletionSignal() {
    static_cast<void>(StorageTexture::StorePendingForSignal());
    if (auto* residency = ResidencyDriver::Active()) residency->BeforeCompletionSignal();
}

bool CompletionPublicationPending() {
    if (StorageTexture::AnyPending()) return true;
    auto* residency = ResidencyDriver::Active();
    return residency != nullptr && residency->Manager().AnyStaleOwned();
}

void PublishBeforeCompletionSignal(SignalKind kind) {
    if (kind == SignalKind::BareWrite) return;
    if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->DeferSignalPublication()) return;
    PublishForCompletionSignal();
}

void ResidencyDriver::BeforeDispatch() {
    const std::lock_guard lock(GuestMemory::GpuMutex());
    manager.Apply();
    harvestTouches();
    if (const auto now = flips.load(std::memory_order_acquire); now != flipsSeen || touchTable == nullptr) {
        flipsSeen = now;
        frameBoundary();
    }
    manager.RefreshHot();
}

std::uint64_t ResidencyDriver::Budget() {
    const auto others = OtherDeviceBytes();
    const auto guestBytes = guest.GuestBytes();
    auto limit = guestBytes > others ? guestBytes - others : 0;
    if (reportedBudget != 0) limit = std::min<std::uint64_t>(limit, reportedBudget > heapOthers ? reportedBudget - heapOthers : 0);
    budgetBytes = limit;
    return budgetBytes;
}

std::uint64_t ResidencyDriver::OtherDeviceBytes() {
    const auto heaps = std::min<std::uint32_t>(context.memory.memoryHeapCount, VK_MAX_MEMORY_HEAPS);
    std::array<DeviceMemoryAccounting::HeapUsage, VK_MAX_MEMORY_HEAPS> counted{};
    std::uint64_t changes = 0;
    for (std::uint32_t heap = 0; heap < heaps; ++heap) {
        if ((context.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        counted[heap] = DeviceMemoryAccounting::Usage(context.device, heap);
        changes += counted[heap].changes;
    }
    if (usageStale || changes != reportedChanges) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        properties.pNext = &reported;
        context.memoryProperties2(context.physical, &properties);
        std::copy(std::begin(reported.heapUsage), std::end(reported.heapUsage), reportedUsage.begin());
        reportedBudget = reported.heapBudget[deviceHeap];
        reportedChanges = changes;
        usageStale = false;
    }
    std::uint64_t others = 0;
    for (std::uint32_t heap = 0; heap < heaps; ++heap) {
        if ((context.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        const auto& usage = counted[heap];
        const auto allocated = usage.allocated > usage.mirrors ? usage.allocated - usage.mirrors : 0;
        const auto used = reportedUsage[heap] > usage.mirrors ? reportedUsage[heap] - usage.mirrors : 0;
        const auto beside = std::max(allocated, used);
        if (heap == deviceHeap) heapOthers = beside;
        others += beside;
    }
    return others;
}

std::uint64_t ResidencyDriver::GuestBytes() {
    return guest.GuestBytes();
}

void ResidencyDriver::CollectWrites(std::uint64_t firstPage, std::uint64_t pages) {
    guest.CollectWrites(firstPage << ResidencyPageShift, pages << ResidencyPageShift);
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
    usageStale = true;
    manager.ApplyBudget();
    manager.AdmitEager();
    if (!context.bufferDeviceAddress) return;
    const std::array<Recorder::SectionObject, 1> objects{SectionAccess(true)};
    recorder.DeclareSection(objects);
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
    const std::lock_guard structureLock(manager.Mutex());
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

void ResidencyDriver::StoreThroughMirrors(std::uint64_t address, std::span<const std::byte> bytes) {
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    for (const auto& store : manager.StoreThroughMirrors(address, bytes.size(), recorder.CallerToken())) {
        const auto chunk = mirrors.Chunk(store.mirror.chunk);
        recorder.Keep(chunk);
        recorder.RecordStore(chunk->buffer, (VkDeviceSize{store.mirror.slot} << ResidencyPageShift) + store.offset, bytes.subspan(store.source, store.bytes), address + store.source);
    }
}

void ResidencyDriver::AccessFault(std::uint64_t page) {
    if (!GuestMemory::GpuMutex().HeldByThisThread() && !Recorder::SnapshotWriteOverlaps(page << ResidencyPageShift, ResidencyPageBytes) && manager.FlushForCpuReadSettled(page << ResidencyPageShift, ResidencyPageBytes)) return;
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    recorder.SyncThrough(page << ResidencyPageShift, ResidencyPageBytes, lock.owns_lock());
    if (!lock.owns_lock()) {
        manager.FlushForCpuRead(page << ResidencyPageShift, ResidencyPageBytes);
        return;
    }
    static_cast<void>(manager.FlushForCpuReadUnlocked(page << ResidencyPageShift, ResidencyPageBytes));
}

void ResidencyDriver::WriteBackAll() {
    std::unique_lock lock(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) lock.lock();
    manager.WriteBackAll();
}

void ResidencyDriver::UseForRead(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    constexpr std::uint64_t largest = std::uint64_t{8u} << 20u;
    const std::lock_guard lock(GuestMemory::GpuMutex());
    const std::lock_guard structureLock(manager.Mutex());
    for (const auto& [begin, end] : ranges) {
        if (end > begin) manager.UseRange(begin, end - begin, largest);
    }
}

void ResidencyDriver::CountMixedRanges(const BdaPageTable& table, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    constexpr std::uint64_t pageBytes = ResidencyPageBytes;
    for (const auto& [begin, end] : ranges) {
        bool mirrored = false;
        bool served = false;
        for (auto address = begin; address < end && !(mirrored && served);) {
            const auto range = table.Find(address);
            const auto pageEnd = (address & ~(pageBytes - 1)) + pageBytes;
            if (!range) {
                address = pageEnd;
                continue;
            }
            ((range->reserved & ShaderRecompiler::BdaAbi::RangeMirrored) != 0 ? mirrored : served) = true;
            address = std::max(pageEnd, range->end);
        }
        if (mirrored && served) mixedAtRecord.fetch_add(1, std::memory_order_relaxed);
    }
}

bool ResidencyDriver::SettledWhole(const BdaPageTable& table, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, bool unbounded) {
    const bool settled = !unbounded && std::ranges::all_of(ranges, [&](const auto& range) {
        if (range.second <= range.first) return true;
        const auto found = table.Find(range.first);
        return found.has_value() && found->begin <= range.first && range.second <= found->end;
    });
    (settled ? settledDispatches : unsettledDispatches).fetch_add(1, std::memory_order_relaxed);
    return settled;
}

void ResidencyDriver::DropMirrors(std::span<const ResidencyRange> ranges) {
    for (const auto& range : ranges) manager.Demote(range.address, range.bytes);
}

void ResidencyDriver::BeforeWritingDispatch(std::span<const ResidencyRange> grants, bool writableImports) {
    const std::lock_guard lock(GuestMemory::GpuMutex());
    const std::lock_guard structureLock(manager.Mutex());
    for (const auto& grant : grants) manager.NoteGpuWritten(grant.address, grant.bytes);
    if (writableImports) manager.NoteUnboundedWriter(recorder.CallerToken());
    for (const auto& grant : grants) manager.Pin(grant.address, grant.bytes, ResidencyAccess::Write, recorder.CallerToken());
}

ResidencyPublication ResidencyDriver::Publish() {
    ResidencyPublication publication;
    const std::lock_guard structureLock(manager.Mutex());
    const auto changes = manager.TakeChanges();
    publication.serial = manager.Serial();
    publication.changed.reserve(changes.size());
    for (const auto page : changes) publication.changed.push_back({page, manager.Translate(page).value_or(0)});
    return publication;
}

void ResidencyDriver::SyncBindings() {
    if (manager.Serial() == bindSerial.load(std::memory_order_acquire) && !bindsUnsettled.load(std::memory_order_acquire)) return;
    std::unique_lock gpu(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) gpu.lock();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> wanted;
    const auto since = bindSerial.load(std::memory_order_relaxed);
    std::uint64_t now = since;
    bool partial = true;
    if (manager.Serial() != since) {
        partial = manager.ChangesSince(since, now, wanted);
        if (!partial) wanted = manager.Mirrored(now);
    }
    const auto holders = Newest();
    const std::lock_guard lock(bindMutex);
    if (!partial) {
        std::set<std::uint64_t> held;
        for (const auto& [page, mirror] : wanted) held.insert(page);
        for (const auto& [page, mirror] : bound) {
            if (!held.contains(page)) wanted.emplace_back(page, 0u);
        }
    }
    bindSerial.store(now, std::memory_order_release);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> fresh;
    for (const auto& [page, mirror] : wanted) {
        if (((page + 1u) << ResidencyPageShift) > sparse->Bytes()) continue;
        const auto found = bound.find(page);
        const auto held = found != bound.end() ? found->second : 0u;
        if (held == mirror) {
            deferred.erase(page);
        } else if (held != 0) {
            auto& entry = deferred[page];
            entry.holders = MergeTokens(entry.holders, holders);
            entry.mirror = mirror;
            deferredUnbinds.fetch_add(1, std::memory_order_relaxed);
        } else {
            fresh.emplace_back(page, mirror);
        }
    }
    std::set<std::uint64_t> targets;
    for (const auto& [page, mirror] : fresh) targets.insert(mirror);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> settled;
    for (bool grown = true; grown;) {
        grown = false;
        for (auto entry = deferred.begin(); entry != deferred.end();) {
            if (!Reached(entry->second.holders) && !targets.contains(bound.at(entry->first))) {
                ++entry;
                continue;
            }
            settled.emplace_back(entry->first, entry->second.mirror);
            if (entry->second.mirror != 0) targets.insert(entry->second.mirror);
            entry = deferred.erase(entry);
            grown = true;
        }
    }
    std::vector<VkSparseMemoryBind> binds;
    bool releases = false;
    const auto bind = [&](std::uint64_t page, std::uint64_t mirror) {
        VkSparseMemoryBind entry{page << ResidencyPageShift, ResidencyPageBytes, VK_NULL_HANDLE, 0, 0};
        if (mirror != 0) {
            const auto [memory, offset] = mirrors.Locate(mirror);
            entry.memory = memory;
            entry.memoryOffset = offset;
            bound[page] = mirror;
        } else {
            bound.erase(page);
            releases = true;
        }
        binds.push_back(entry);
    };
    for (const auto& [page, mirror] : settled) bind(page, mirror);
    for (const auto& [page, mirror] : fresh) bind(page, mirror);
    const auto value = sparse->Bind(binds);
    if (releases) unbindValue = value;
    if (!retired.empty() && sparse->Completed() >= unbindValue) {
        std::erase_if(retired, [&](const std::shared_ptr<MirrorChunk>& chunk) {
            return std::none_of(bound.begin(), bound.end(), [&](const auto& held) { return held.second >= chunk->address && held.second - chunk->address < chunk->bytes; });
        });
    }
    bindsUnsettled.store(!deferred.empty() || !retired.empty(), std::memory_order_release);
}

std::uint64_t ResidencyDriver::ResidentAddress(std::uint64_t page, std::uint64_t mirror) const {
    const std::lock_guard lock(bindMutex);
    const auto found = bound.find(page);
    return found != bound.end() && found->second == mirror ? sparse->Address() + (page << ResidencyPageShift) : mirror;
}

bool ResidencyDriver::Bound(std::uint64_t page) const {
    const std::lock_guard lock(bindMutex);
    return bound.contains(page);
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
    return {orderingPoints.load(std::memory_order_relaxed), appliedAtOrderingPoints.load(std::memory_order_relaxed), vanishedCopies, frames, touchHarvests, touchedPages, touchOverflows, budgetBytes, bindWaits.load(std::memory_order_relaxed), deferredUnbinds.load(std::memory_order_relaxed), mixedAtRecord.load(std::memory_order_relaxed), settledDispatches.load(std::memory_order_relaxed), unsettledDispatches.load(std::memory_order_relaxed), segmentFaults.load(std::memory_order_relaxed)};
}

bool ResidencyDriver::Reached(const ResidencyToken& token) {
    return recorder.Reached(token);
}

void ResidencyDriver::Wait(const ResidencyToken& token) {
    recorder.Submit();
    recorder.FinishToken(token);
}

bool ResidencyDriver::WaitUnlocked(const ResidencyToken& token) {
    recorder.Submit();
    return recorder.WaitTokenUnlocked(token);
}

bool ResidencyDriver::Signaled(const ResidencyToken& token) {
    return recorder.TokenSignaled(token);
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
    {
        auto held = mirrors.Chunk(chunk);
        const std::lock_guard lock(bindMutex);
        if (std::any_of(bound.begin(), bound.end(), [&](const auto& page) { return page.second >= held->address && page.second - held->address < held->bytes; }) || sparse->Completed() < unbindValue) {
            retired.push_back(std::move(held));
            bindsUnsettled.store(true, std::memory_order_release);
        }
    }
    mirrors.Release(chunk);
}

ResidencyToken ResidencyDriver::CopyIn(std::span<const ResidencyCopy> copies) {
    return record(copies, true);
}

ResidencyToken ResidencyDriver::WriteBack(std::span<const ResidencyCopy> copies) {
    return record(copies, false);
}

ResidencyRefusal ResidencyDriver::Guard(std::uint64_t page) {
    return guest.Guard(page);
}

void ResidencyDriver::Unguard(std::span<const std::uint64_t> pages) {
    guest.Unguard(pages);
}

ResidencyRefusal ResidencyDriver::GuardAccess(std::uint64_t page, ResidencyMask mask) {
    return guest.GuardAccess(page, mask);
}

void ResidencyDriver::UnguardAccess(std::uint64_t page) {
    guest.UnguardAccess(page);
}

void ResidencyDriver::SettleWrites(std::uint64_t page) {
    guest.SettleWrites(page);
}

bool ResidencyDriver::ImportSettled(std::uint64_t page) {
    return guest.ImportSettled(page);
}

std::uint64_t ResidencyDriver::ImportStamp() {
    return guest.ImportStamp();
}

void ResidencyDriver::SettleImport(std::uint64_t page, std::uint64_t stamp) {
    guest.SettleImport(page, stamp);
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
        objects.push_back(SectionAccess(true));
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
