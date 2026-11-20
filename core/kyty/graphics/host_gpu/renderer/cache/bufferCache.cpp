#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "gpu_tiler_shaders/buffer_scatter_spv.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/spscQueue.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/scout.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/execHelpers.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsTitle.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <string>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write,
                                        bool wait = true) {
	const auto [mapped, offset] = staging.Map(size, alignment, wait);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferHostOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

constexpr uint64_t ReadAheadLimit = 256 * 1024;

struct ReadbackStats {
	std::atomic<uint64_t> cp_reads {0};
	std::atomic<uint64_t> cp_ns {0};
	std::atomic<uint64_t> cp_ordered {0};
	std::atomic<uint64_t> cp_ordered_ns {0};
	std::atomic<uint64_t> cp_not_waited {0};
	std::atomic<uint64_t> other_reads {0};
	std::atomic<uint64_t> other_ns {0};
	std::atomic<uint64_t> other_gpu_bytes {0};
	std::atomic<uint64_t> other_false_sharing {0};
	std::atomic<uint64_t> cp_staged {0};
	std::atomic<uint64_t> other_replayed {0};
	std::atomic<uint64_t> other_dispatch_ns {0};
	std::atomic<uint64_t> other_wait_ns {0};
	std::atomic<uint64_t> other_release_ns {0};
	std::atomic<uint64_t> released_async {0};
	std::atomic<uint64_t> replayed_after_copy {0};
	std::atomic<uint64_t> waited_after_copy {0};
	std::atomic<bool>     line_overflow {false};
	std::atomic<int64_t>  last_line {0};
};

ReadbackStats g_readback_stats;

struct WriteBlockStats {
	std::atomic<uint64_t> faults {0};
	std::atomic<uint64_t> blocks {0};
	std::atomic<uint64_t> pages {0};
	std::atomic<uint64_t> uploaded {0};
	std::atomic<uint64_t> command {0};
	std::atomic<uint64_t> segment {0};
	std::atomic<uint64_t> gpu {0};
	std::atomic<uint64_t> image {0};
	std::atomic<uint64_t> buffer {0};
	std::atomic<uint64_t> other {0};
	std::atomic<int64_t>  last_line {0};
};

WriteBlockStats g_write_block_stats;

struct ReadbackReplay {
	bool     allowed   = false;
	bool     published = false;
	uint64_t begin     = 0;
	uint64_t end       = 0;
};

thread_local ReadbackReplay g_readback_replay;

int64_t SteadyNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}


}

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	if constexpr (!insert) InvalidateCopyFeedback(buffer.CpuAddress(), buffer.Size());
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	{
		const auto epoch = m_registration_epoch.load(std::memory_order_relaxed) + 1;
		if (m_registration_spans.size() < 64) {
			m_registration_spans.push_back({buffer.CpuAddress(), buffer.Size()});
		} else {
			m_registration_spans_lost = true;
		}
		for (auto region = buffer.CpuAddress() / TRACKER_REGION_SIZE,
		          last   = (buffer.CpuAddress() + buffer.Size() - 1) / TRACKER_REGION_SIZE;
		     region <= last && region < TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE; ++region) {
			m_region_registrations[region].store(epoch, std::memory_order_release);
			BdaDirtyRegions::Mark(region);
		}
		m_registration_epoch.store(epoch, std::memory_order_release);
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_register_epoch++;
		m_buffer_set_epoch++;
		NoteBufferRegions(buffer, true);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_buffer_set_epoch++;
		NoteBufferRegions(buffer, false);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

bool BufferCache::TouchLiveBuffer(BufferId id, vk::Buffer handle, uint64_t vaddr, uint64_t size) {
	auto* buffer = GetBufferIfLive(id);
	if (buffer == nullptr || buffer->Handle() != handle || !buffer->IsInBounds(vaddr, size)) {
		return false;
	}
	TouchBuffer(*buffer);
	return true;
}

void BufferCache::EnsureBufferContents(uint64_t vaddr, uint64_t size) {
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	DiagArgsSyncNoteBuffer(11, m_slot_buffers[id].CpuAddress(), m_slot_buffers[id].Size(), 0);
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferHostOperation([this, id] {
			WaitReadbackQueueCopies(m_slot_buffers[id].Handle());
			m_slot_buffers.erase(id);
		});
	} else {
		m_slot_buffers.erase(id);
	}
}

void BufferCache::CollectDownloadCopies(const Buffer& buffer, uint64_t vaddr, uint64_t size,
                                        std::vector<vk::BufferCopy>& copies,
                                        uint64_t&                    total_size) {
	const auto buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    if (m_gpu_modified_ranges.Subtract(address, bytes)) {
			    NoteGpuModifiedChange(address, bytes, false);
		    }
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (!copies.empty()) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	CollectDownloadCopies(buffer, vaddr, size, copies, total_size);
	if (copies.empty()) {
		return false;
	}

	if constexpr (!async) {
		if (DownloadOnReadbackQueue(buffer, copies, total_size)) {
			return true;
		}
	}
	auto download = m_download_buffer.MapDownload(total_size, 64);
	for (auto& copy: copies) {
		copy.dstOffset += download.offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.buffer->Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.buffer->Handle();
	after.offset        = download.offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	uint64_t written_begin = UINT64_MAX;
	uint64_t written_end   = 0;
	for (const auto& copy: copies) {
		written_begin = std::min(written_begin, buffer_address + copy.srcOffset);
		written_end   = std::max(written_end, buffer_address + copy.srcOffset + copy.size);
	}
	auto publish = [total_size, buffer_address, copies = std::move(copies),
	                download = std::move(download)] {
		download.buffer->Invalidate(download.offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(
			    buffer_address + copy.srcOffset,
			    download.mapped + (copy.dstOffset - download.offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish), written_begin,
		                                   written_end - written_begin);
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache, RenderContext* resources)
    : m_graphics(graphics), m_scheduler(scheduler), m_resources(resources),
      m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	GetBuffer(null_id).Fill(0, 16, 0);
	m_scheduler.SetSubmitHook(
	    [](void* self, CommandBuffer& command) {
		    static_cast<BufferCache*>(self)->RecordMirrorCopies(command);
	    },
	    this);
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	m_scheduler.SetSubmitHook(nullptr, nullptr);
	if (m_scatter_pipeline) {
		m_graphics.device.destroyPipeline(m_scatter_pipeline);
	}
	if (m_scatter_layout) {
		m_graphics.device.destroyPipelineLayout(m_scatter_layout);
	}
	if (m_readback.pool) {
		m_graphics.device.destroyFence(m_readback.fence);
		m_graphics.device.destroyCommandPool(m_readback.pool);
	}
	for (auto& slot: m_readback_slots) {
		if (slot.queue.pool) {
			(void)m_graphics.device.waitForFences(1, &slot.queue.fence, VK_TRUE, UINT64_MAX);
			m_graphics.device.destroyFence(slot.queue.fence);
			m_graphics.device.destroyCommandPool(slot.queue.pool);
		}
	}
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size, bool guest_fault) {
	for (;;) {
		bool gpu_modified = false;
		m_memory_tracker.InvalidateRegion(vaddr, size, [&gpu_modified] { gpu_modified = true; });
		if (!gpu_modified) {
			return;
		}
		ReadMemory(vaddr, size, true, guest_fault);
	}
}

void BufferCache::InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	bool requested = false;
	m_memory_tracker.InvalidateRegion(vaddr, size, [&] {
		if (std::exchange(requested, true)) return;
		if (GuestGpu::IsGpuThread()) {
			DiscardGpuWrites(vaddr, size);
			return;
		}
		m_scheduler.Context().GetGpu().SendUrgentCommandSync([this, vaddr, size] { DiscardGpuWrites(vaddr, size); });
	});
}

void BufferCache::DiscardGpuWrites(uint64_t vaddr, uint64_t size) {
	const auto end   = vaddr + size;
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(end, TRACKER_PAGE_SIZE);
	if (!IsRegionRegistered(vaddr, size)) {
		return;
	}
	if (const auto inflight = InFlightReadbackTick(first, last); inflight != 0) {
		m_scheduler.Wait(inflight);
		m_scheduler.WaitPriorityOperations(inflight);
		std::erase_if(m_inflight_readbacks, [inflight](const InFlightReadback& readback) { return readback.tick <= inflight; });
	}
	for (const auto& [begin, stop]: {std::pair {first, vaddr}, std::pair {end, last}}) {
		if (begin >= stop || !m_gpu_modified_ranges.Intersects(begin, stop - begin)) continue;
		auto& buffer = m_slot_buffers[FindBuffer(Common::AlignDown(begin, TRACKER_PAGE_SIZE), TRACKER_PAGE_SIZE)];
		(void)DownloadBufferMemory<false>(buffer, begin, stop - begin);
	}
	if (m_gpu_modified_ranges.Subtract(first, last - first)) {
		NoteGpuModifiedChange(first, last - first, false);
	}
	ForgetColorMetadataChecks(first, last - first);
	UnmarkGpuModified(first, last - first);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
}

void BufferCache::InvalidateUnlessGpuModified(uint64_t vaddr, uint64_t size) noexcept {
	if (GuestRange {vaddr, size}.Valid()) {
		m_memory_tracker.InvalidateRegion(vaddr, size, [] {});
	}
}

static bool WriteFaultBlocks() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_WRITE_FAULT_BLOCKS");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}

BufferCache::WriteFaultOwner BufferCache::InvalidateFaultedPage(uint64_t vaddr) {
	static constexpr uint64_t FaultBlockSize = CommandBlocks::BlockSize;
	static_assert(TRACKER_REGION_SIZE % FaultBlockSize == 0);
	const auto page  = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	auto       owner = WriteFaultOwner::Other;
	if (GuestGpu::IsGpuThread() || GuestGpu::IsCommandProcessorThread() ||
	    !GuestRange {Common::AlignDown(page, FaultBlockSize), FaultBlockSize}.Valid()) {
		m_memory_tracker.NoteGuestWriteFault(page);
	} else {
		const bool command = m_memory_tracker.IsCommandBlock(page);
		const bool segment = !command && m_memory_tracker.IsSegmentBlock(page);
		const auto probe   = m_memory_tracker.OpenWriteBlock(page, FaultBlockSize, (command || segment) && WriteFaultBlocks());
		owner              = command       ? WriteFaultOwner::CommandBuffer
		                     : segment       ? WriteFaultOwner::CommandSegment
		                     : probe.gpu     ? WriteFaultOwner::GpuNeighbor
		                     : probe.tracked ? WriteFaultOwner::Buffer
		                                     : WriteFaultOwner::Other;
		if (probe.opened != 0) {
			auto& stats = g_write_block_stats;
			stats.blocks.fetch_add(1, std::memory_order_relaxed);
			stats.pages.fetch_add(probe.opened, std::memory_order_relaxed);
			return owner;
		}
	}
	const auto end  = Common::AlignDown(page, FaultBlockSize) + FaultBlockSize;
	if (page >= TRACKER_PAGE_SIZE && GuestRange {page - TRACKER_PAGE_SIZE, end - page}.Valid() &&
	    m_memory_tracker.IsRegionCpuModified(page - TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE) &&
	    !m_memory_tracker.IsRegionGpuModified(page, end - page)) {
		InvalidateMemory(page, end - page, true);
		return owner;
	}
	InvalidateMemory(vaddr, 1, true);
	return owner;
}

void BufferCache::CountWriteFault(WriteFaultOwner owner, bool image) {
	auto& stats = g_write_block_stats;
	stats.faults.fetch_add(1, std::memory_order_relaxed);
	auto& counter = owner == WriteFaultOwner::CommandBuffer    ? stats.command
	                : owner == WriteFaultOwner::CommandSegment ? stats.segment
	                : owner == WriteFaultOwner::GpuNeighbor    ? stats.gpu
	                : image                                 ? stats.image
	                : owner == WriteFaultOwner::Buffer      ? stats.buffer
	                                                        : stats.other;
	counter.fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write, bool guest_fault) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	if (!GuestGpu::IsGpuThread()) {
		ReadMemoryAsync(vaddr, size, is_write, guest_fault);
		return;
	}
	m_scheduler.Context().GetGpu().SendUrgentCommandSync([this, vaddr, size, is_write, guest_fault] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		if (TryGuestReadFromFeedback(vaddr, size, is_write)) {
			return;
		}
		auto&      buffer    = m_slot_buffers[FindBuffer(vaddr, size)];
		const auto read_size = guest_fault ? ReadAheadSize(buffer, vaddr, size) : size;

		const auto pages_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		const auto pages_end   = Common::AlignUp(vaddr + read_size, TRACKER_PAGE_SIZE);
		if (const auto inflight = InFlightReadbackTick(pages_begin, pages_end); inflight != 0) {
			m_scheduler.Wait(inflight);
			m_scheduler.WaitPriorityOperations(inflight);
			std::erase_if(m_inflight_readbacks, [inflight](const InFlightReadback& readback) {
				return readback.tick <= inflight;
			});
		}
		if (DownloadBufferMemory<false>(buffer, vaddr, read_size) ||
		    !m_gpu_modified_ranges.Intersects(pages_begin, pages_end - pages_begin)) {
			UnmarkGpuModified(vaddr, read_size);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	}, false);
}

void BufferCache::ReadMemoryAsync(uint64_t vaddr, uint64_t size, bool is_write, bool guest_fault) {
	const bool command_processor = GuestGpu::IsCommandProcessorThread();
	const auto started           = SteadyNs();
	ReadMemoryAsyncTimed(vaddr, size, is_write, guest_fault, command_processor);
	auto&      stats   = g_readback_stats;
	const auto elapsed = static_cast<uint64_t>(SteadyNs() - started);
	(command_processor ? stats.cp_reads : stats.other_reads).fetch_add(1, std::memory_order_relaxed);
	(command_processor ? stats.cp_ns : stats.other_ns).fetch_add(elapsed, std::memory_order_relaxed);
}

void BufferCache::AllowReplayAfterReadback() noexcept {
	g_readback_replay = {.allowed = true};
}

bool BufferCache::TakeReplayAfterReadback(uint64_t* begin, uint64_t* end) noexcept {
	const auto replay = std::exchange(g_readback_replay, {});
	*begin            = replay.begin;
	*end              = replay.end;
	return replay.published;
}

void BufferCache::NoteReplayAfterReadback(bool replayed) noexcept {
	(replayed ? g_readback_stats.replayed_after_copy : g_readback_stats.waited_after_copy)
	    .fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::WaitReadbackRelease() {
	m_scheduler.Context().GetGpu().SendReadbackCommandSync([] {});
}

void BufferCache::NoteReplayedRead() {
	g_readback_stats.other_replayed.fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::NoteStagedCommandProcessorRead() {
	g_readback_stats.cp_staged.fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::LogWriteBlockSummary() {
	auto&      stats = g_write_block_stats;
	const auto now   = SteadyNs();
	auto       last  = stats.last_line.load(std::memory_order_relaxed);
	if (last == 0) {
		stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000 || !stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto faults   = stats.faults.exchange(0, std::memory_order_relaxed);
	const auto blocks   = stats.blocks.exchange(0, std::memory_order_relaxed);
	const auto pages    = stats.pages.exchange(0, std::memory_order_relaxed);
	const auto uploaded = stats.uploaded.exchange(0, std::memory_order_relaxed);
	const auto command  = stats.command.exchange(0, std::memory_order_relaxed);
	const auto segment  = stats.segment.exchange(0, std::memory_order_relaxed);
	const auto gpu      = stats.gpu.exchange(0, std::memory_order_relaxed);
	const auto image    = stats.image.exchange(0, std::memory_order_relaxed);
	const auto buffer   = stats.buffer.exchange(0, std::memory_order_relaxed);
	const auto other    = stats.other.exchange(0, std::memory_order_relaxed);
	const auto opened   = RegionManager::TakeOpenedUploadBytes();
	if (faults == 0 && uploaded == 0) {
		return;
	}
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-write-blocks] summary of the last {:.0f} s: {} guest write faults, {} opened their 64 KiB block ({} clean pages "
	    "marked written with them), {} left to the page path; {:.1f} MiB of buffer uploads; faults by owner, first match: {} in "
	    "a command buffer the CP read, {} elsewhere in a command buffer segment, {} in a block with a page the GPU wrote, {} "
	    "watched by an image or texture, {} in a buffer of the cache, {} other; {:.1f} MiB of pages opened with a block handed "
	    "to uploads\n",
	    static_cast<double>(now - last) / 1e9, faults, blocks, pages, faults - std::min(faults, blocks),
	    static_cast<double>(uploaded) / (1024.0 * 1024.0), command, segment, gpu, image, buffer, other,
	    static_cast<double>(opened) / (1024.0 * 1024.0)));
}

void BufferCache::LogReadbackSummary() {
	LogWriteBlockSummary();
	auto&      stats = g_readback_stats;
	const auto now   = SteadyNs();
	auto       last  = stats.last_line.load(std::memory_order_relaxed);
	if (last == 0) {
		stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000 || !stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto cp_reads    = stats.cp_reads.exchange(0, std::memory_order_relaxed);
	const auto other_reads = stats.other_reads.exchange(0, std::memory_order_relaxed);
	const auto cp_ns       = stats.cp_ns.exchange(0, std::memory_order_relaxed);
	const auto other_ns    = stats.other_ns.exchange(0, std::memory_order_relaxed);
	const auto ordered     = stats.cp_ordered.exchange(0, std::memory_order_relaxed);
	const auto ordered_ns  = stats.cp_ordered_ns.exchange(0, std::memory_order_relaxed);
	const auto not_waited  = stats.cp_not_waited.exchange(0, std::memory_order_relaxed);
	const auto gpu_bytes   = stats.other_gpu_bytes.exchange(0, std::memory_order_relaxed);
	const auto false_share = stats.other_false_sharing.exchange(0, std::memory_order_relaxed);
	const auto staged      = stats.cp_staged.exchange(0, std::memory_order_relaxed);
	const auto replayed    = stats.other_replayed.exchange(0, std::memory_order_relaxed);
	const auto dispatch_ns = stats.other_dispatch_ns.exchange(0, std::memory_order_relaxed);
	const auto wait_ns     = stats.other_wait_ns.exchange(0, std::memory_order_relaxed);
	const auto release_ns  = stats.other_release_ns.exchange(0, std::memory_order_relaxed);
	const auto async_rel   = stats.released_async.exchange(0, std::memory_order_relaxed);
	const auto replayed_cp = stats.replayed_after_copy.exchange(0, std::memory_order_relaxed);
	const auto waited_cp   = stats.waited_after_copy.exchange(0, std::memory_order_relaxed);
	if (cp_reads + other_reads + staged + replayed == 0) {
		return;
	}
	const auto seconds = [](uint64_t ns) { return static_cast<double>(ns) / 1e9; };
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-readback] summary of the last {:.0f} s: command processor {} readbacks in {:.3f} s ({} after the "
	    "operations it is ordered after only, {:.3f} s waiting for them, {} operations not waited for), {} register "
	    "tables read without a fault; other threads {} readbacks in {:.3f} s ({} of faulted reads on bytes the GPU "
	    "wrote, {} on other bytes of a GPU written page, {} on other bytes replayed on the backing view without a readback; "
	    "{:.3f} s handing them to the execution thread, {:.3f} s waiting for the GPU copy, {:.3f} s releasing them through "
	    "the execution thread; {} released without waiting, the faulted instruction replayed on the copy {} times and "
	    "waiting for the release {} times){}\n",
	    seconds(static_cast<uint64_t>(now - last)), cp_reads, seconds(cp_ns), ordered, seconds(ordered_ns), not_waited,
	    staged, other_reads, seconds(other_ns), gpu_bytes, false_share, replayed, seconds(dispatch_ns), seconds(wait_ns),
	    seconds(release_ns), async_rel, replayed_cp, waited_cp,
	    stats.line_overflow.load(std::memory_order_relaxed) ? "; GPU line table overflowed" : ""));
}

void BufferCache::ReadMemoryAsyncTimed(uint64_t vaddr, uint64_t size, bool is_write, bool guest_fault,
                                       bool command_processor) {
	auto&             gpu         = m_scheduler.Context().GetGpu();
	uint64_t          tick        = 0;
	uint64_t          read_size   = size;
	uint64_t          pages_begin = 0;
	uint64_t          pages_end   = 0;
	ReadbackQueueRead queue_read;
	const bool        ordered        = command_processor && GuestGpu::OrderedReadbacks();
	const bool        readback_queue = !command_processor || ordered;
	if (ordered) {
		const auto started = SteadyNs();
		const auto skipped = gpu.WaitForOrderedOperations(
		    Common::AlignDown(vaddr, TRACKER_PAGE_SIZE), Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE) + ReadAheadLimit);
		auto& stats = g_readback_stats;
		stats.cp_ordered.fetch_add(1, std::memory_order_relaxed);
		stats.cp_ordered_ns.fetch_add(static_cast<uint64_t>(SteadyNs() - started), std::memory_order_relaxed);
		stats.cp_not_waited.fetch_add(skipped, std::memory_order_relaxed);
	}
	auto&      split          = g_readback_stats;
	const auto dispatch_start = SteadyNs();
	const auto note_split     = [&](std::atomic<uint64_t>& counter, int64_t since) {
		if (!command_processor) {
			counter.fetch_add(static_cast<uint64_t>(SteadyNs() - since), std::memory_order_relaxed);
		}
	};
	gpu.SendReadbackCommandSync([&] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		if (TryGuestReadFromFeedback(vaddr, size, is_write)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		read_size    = guest_fault ? ReadAheadSize(buffer, vaddr, size) : size;
		pages_begin  = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		pages_end    = Common::AlignUp(vaddr + read_size, TRACKER_PAGE_SIZE);
		tick         = InFlightReadbackTick(pages_begin, pages_end);
		NoteReadbackWait(pages_begin, pages_end, false,
		                 m_write_ticks.Latest(pages_begin, pages_end) >= m_scheduler.CurrentTick());
		const bool guest_read = guest_fault && !is_write;
		if (guest_read && !command_processor) {
			(m_gpu_modified_ranges.Intersects(vaddr, size) ? g_readback_stats.other_gpu_bytes
			                                                : g_readback_stats.other_false_sharing)
			    .fetch_add(1, std::memory_order_relaxed);
		}
		if (tick == 0 && readback_queue &&
		    StartReadbackQueueRead(buffer, vaddr, read_size, pages_begin, pages_end, queue_read,
		                           guest_read)) {
			tick = queue_read.recorded;
		} else if (DownloadBufferMemory<true>(buffer, vaddr, read_size)) {
			const auto recorded = m_scheduler.CurrentTick();
			m_inflight_readbacks.push_back({pages_begin, pages_end, recorded});
			tick = std::max(tick, recorded);
		}
		if (guest_read) {
			NoteGuestReadPages(pages_begin, pages_end);
		}
		if (tick == 0) {
			ReleaseReadback(vaddr, size, pages_begin, pages_end, 0, is_write);
			return;
		}
		if (queue_read.slot == nullptr && tick >= m_scheduler.CurrentTick()) {
			m_scheduler.Flush();
		}
	});
	note_split(split.other_dispatch_ns, dispatch_start);
	if (tick == 0) {
		return;
	}
	const bool replay_after = g_readback_replay.allowed && !is_write && !command_processor;
	const auto release      = [&](Common::UniqueFunction<void>&& operation) {
		const auto release_start = SteadyNs();
		if (replay_after && gpu.PostReadbackCommand(std::move(operation))) {
			g_readback_replay.published = true;
			g_readback_replay.begin     = vaddr;
			g_readback_replay.end       = vaddr + read_size;
			split.released_async.fetch_add(1, std::memory_order_relaxed);
		} else {
			gpu.SendReadbackCommandSync(std::move(operation));
		}
		note_split(split.other_release_ns, release_start);
	};
	const auto wait_start = SteadyNs();
	if (queue_read.slot != nullptr) {
		FinishReadbackQueueRead(queue_read);
		note_split(split.other_wait_ns, wait_start);
		release([this, vaddr, size, pages_begin, pages_end, is_write, recorded = queue_read.recorded,
		         followers = std::move(queue_read.followers)] {
			const auto own = std::find_if(m_inflight_readbacks.begin(), m_inflight_readbacks.end(),
			                              [&](const InFlightReadback& readback) {
				                              return readback.begin == pages_begin &&
				                                     readback.end == pages_end &&
				                                     readback.tick == recorded;
			                              });
			if (own != m_inflight_readbacks.end()) {
				m_inflight_readbacks.erase(own);
			}
			ReleaseReadback(vaddr, size, pages_begin, pages_end, 0, is_write);
			if (!followers.empty()) {
				std::vector<uint64_t> pages;
				pages.reserve(followers.size());
				for (const auto& follower: followers) {
					pages.push_back(follower.page);
				}
				std::sort(pages.begin(), pages.end());
				std::erase_if(m_inflight_readbacks, [&](const InFlightReadback& readback) {
					return readback.tick == recorded &&
					       readback.end - readback.begin == TRACKER_PAGE_SIZE &&
					       std::binary_search(pages.begin(), pages.end(), readback.begin);
				});
				for (const auto& follower: followers) {
					if (InFlightReadbackTick(follower.page, follower.page + TRACKER_PAGE_SIZE) == 0 &&
					    !m_gpu_modified_ranges.Intersects(follower.page, TRACKER_PAGE_SIZE)) {
						UnmarkGpuModified(follower.page, TRACKER_PAGE_SIZE);
					}
				}
			}
		});
	} else {
		m_scheduler.GetMasterSemaphore().Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		note_split(split.other_wait_ns, wait_start);
		release([this, vaddr, size, pages_begin, pages_end, tick, is_write] {
			ReleaseReadback(vaddr, size, pages_begin, pages_end, tick, is_write);
		});
	}
}

void BufferCache::NoteGuestReadPages(uint64_t pages_begin, uint64_t pages_end) {
	const auto read = ++m_guest_reads;
	for (auto page = pages_begin; page < pages_end; page += TRACKER_PAGE_SIZE) {
		auto found = std::ranges::find(m_guest_read_pages, page, &GuestReadPage::page);
		if (found == m_guest_read_pages.end()) {
			if (m_guest_read_pages.size() >= MaxGuestReadPages) {
				found = std::ranges::min_element(m_guest_read_pages, {}, &GuestReadPage::read);
			} else {
				found = m_guest_read_pages.emplace(m_guest_read_pages.end());
			}
			found->page = page;
		}
		found->read = read;
	}
}

bool BufferCache::StartReadbackQueueRead(const Buffer& buffer, uint64_t vaddr, uint64_t size,
                                         uint64_t pages_begin, uint64_t pages_end,
                                         ReadbackQueueRead& read, bool followers) {
	if (!m_graphics.readback_queue || m_device_address_pending) {
		return false;
	}
	uint64_t needed  = m_device_address_tick;
	bool     unknown = false;
	bool     written = false;
	m_gpu_modified_ranges.ForEachInRange(pages_begin, pages_end - pages_begin, [&](uint64_t start,
	                                                                              uint64_t end) {
		const auto latest = m_write_ticks.Latest(start, end);
		unknown |= latest == UINT64_MAX;
		needed  = std::max(needed, latest);
		written = true;
		for (const auto& [pending, bytes]: m_pending_writes) {
			unknown |= pending < end && start < pending + bytes;
		}
	});
	if (!written || unknown || needed > m_scheduler.SubmittedTick()) {
		return false;
	}
	ReadbackQueueSlot* slot = nullptr;
	for (auto& candidate: m_readback_slots) {
		if (!candidate.busy.load(std::memory_order_acquire)) {
			slot = &candidate;
			break;
		}
	}
	if (slot == nullptr) {
		return false;
	}
	read.copies.clear();
	read.followers.clear();
	read.total_size = 0;
	CollectDownloadCopies(buffer, vaddr, size, read.copies, read.total_size);
	if (read.copies.empty()) {
		return false;
	}
	if (followers) {
		std::erase_if(m_guest_read_pages, [&](const GuestReadPage& entry) {
			return entry.read + GuestReadPageLifetime < m_guest_reads;
		});
		for (const auto& entry: m_guest_read_pages) {
			const auto page = entry.page;
			const auto end  = page + TRACKER_PAGE_SIZE;
			if ((page < pages_end && pages_begin < end) ||
			    !m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) ||
			    InFlightReadbackTick(page, end) != 0 ||
			    m_texture_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
				continue;
			}
			const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
			if (owner == nullptr || !*owner) {
				continue;
			}
			const auto& other = m_slot_buffers[*owner];
			if (!other.IsInBounds(page, TRACKER_PAGE_SIZE)) {
				continue;
			}
			uint64_t latest       = m_device_address_tick;
			bool     page_unknown = false;
			m_gpu_modified_ranges.ForEachInRange(page, TRACKER_PAGE_SIZE,
			                                     [&](uint64_t start, uint64_t finish) {
				                                     const auto tick = m_write_ticks.Latest(start, finish);
				                                     page_unknown |= tick == UINT64_MAX;
				                                     latest = std::max(latest, tick);
				                                     for (const auto& [pending, bytes]: m_pending_writes) {
					                                     page_unknown |=
					                                         pending < finish && start < pending + bytes;
				                                     }
			                                     });
			if (page_unknown || latest > m_scheduler.SubmittedTick() ||
			    (latest > needed && !m_scheduler.IsFree(latest))) {
				continue;
			}
			auto& follower = read.followers.emplace_back();
			CollectDownloadCopies(other, page, TRACKER_PAGE_SIZE, follower.copies, read.total_size);
			if (follower.copies.empty()) {
				read.followers.pop_back();
				continue;
			}
			follower.page           = page;
			follower.buffer_address = other.CpuAddress();
			follower.source         = other.Handle();
			needed                  = std::max(needed, latest);
		}
	}
	slot->many   = !read.followers.empty();
	slot->source = buffer.Handle();
	slot->busy.store(true, std::memory_order_release);
	auto& queue  = slot->queue;
	auto& device = m_graphics.device;
	if (!queue.pool) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
		                             vk::CommandPoolCreateFlagBits::eTransient;
		pool_info.queueFamilyIndex = m_graphics.readback_family;
		vk::CommandBufferAllocateInfo allocate {};
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		vk::FenceCreateInfo fence_info {};
		fence_info.flags = vk::FenceCreateFlagBits::eSignaled;
		if (device.createCommandPool(&pool_info, nullptr, &queue.pool) != vk::Result::eSuccess ||
		    (allocate.commandPool = queue.pool,
		     device.allocateCommandBuffers(&allocate, &queue.command) != vk::Result::eSuccess) ||
		    device.createFence(&fence_info, nullptr, &queue.fence) != vk::Result::eSuccess) {
			EXIT("BufferCache: could not create the readback queue resources\n");
		}
	}
	if (!queue.staging || queue.staging->Size() < read.total_size) {
		queue.staging = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Download, 0,
		    vk::BufferUsageFlagBits::eTransferDst,
		    std::max<uint64_t>(Common::AlignUp(read.total_size, 1024 * 1024), 1024 * 1024));
	}
	const auto command = queue.command;
	(void)device.resetFences(1, &queue.fence);
	(void)command.reset({});
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	(void)command.begin(&begin_info);
	command.copyBuffer(buffer.Handle(), queue.staging->Handle(),
	                   static_cast<uint32_t>(read.copies.size()), read.copies.data());
	for (const auto& follower: read.followers) {
		command.copyBuffer(follower.source, queue.staging->Handle(),
		                   static_cast<uint32_t>(follower.copies.size()), follower.copies.data());
	}
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = queue.staging->Handle();
	after.offset              = 0;
	after.size                = read.total_size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &after, 0, nullptr);
	(void)command.end();
	const auto                      semaphore  = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &needed;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &semaphore;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	if (m_graphics.readback_queue.submit(1, &submit, queue.fence) != vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue submission failed\n");
	}
	read.slot           = slot;
	read.buffer_address = buffer.CpuAddress();
	read.needed         = needed;
	read.recorded       = m_scheduler.CurrentTick();
	read.published      = std::make_shared<std::atomic<bool>>(false);
	m_inflight_readbacks.push_back({pages_begin, pages_end, read.recorded});
	if (read.followers.empty()) {
		m_scheduler.DeferPriorityOperation(
		    [published = read.published] { published->wait(false, std::memory_order_acquire); },
		    pages_begin, pages_end - pages_begin);
		return true;
	}
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	ranges.reserve(read.followers.size() + 1);
	ranges.emplace_back(pages_begin, pages_end - pages_begin);
	for (const auto& follower: read.followers) {
		m_inflight_readbacks.push_back({follower.page, follower.page + TRACKER_PAGE_SIZE, read.recorded});
		ranges.emplace_back(follower.page, TRACKER_PAGE_SIZE);
	}
	m_scheduler.DeferPriorityOperation(
	    [published = read.published] { published->wait(false, std::memory_order_acquire); }, ranges);
	return true;
}

void BufferCache::WaitReadbackQueueCopies(vk::Buffer buffer) {
	for (auto& slot: m_readback_slots) {
		if (slot.busy.load(std::memory_order_acquire) && (slot.source == buffer || slot.many)) {
			(void)m_graphics.device.waitForFences(1, &slot.queue.fence, VK_TRUE, UINT64_MAX);
		}
	}
}

void BufferCache::FinishReadbackQueueRead(ReadbackQueueRead& read) {
	auto& queue = read.slot->queue;
	if (m_graphics.device.waitForFences(1, &queue.fence, VK_TRUE, UINT64_MAX) !=
	    vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue wait failed\n");
	}
	m_scheduler.WaitPriorityOperations(read.needed);
	queue.staging->Invalidate(0, read.total_size);
	const auto* mapped = queue.staging->Mapped().data();
	for (const auto& copy: read.copies) {
		Libs::LibKernel::Memory::WriteBacking(read.buffer_address + copy.srcOffset,
		                                      mapped + copy.dstOffset, copy.size);
	}
	for (const auto& follower: read.followers) {
		for (const auto& copy: follower.copies) {
			Libs::LibKernel::Memory::WriteBacking(follower.buffer_address + copy.srcOffset,
			                                      mapped + copy.dstOffset, copy.size);
		}
	}
	read.slot->busy.store(false, std::memory_order_release);
	read.published->store(true, std::memory_order_release);
	read.published->notify_all();
}

uint64_t BufferCache::InFlightReadbackTick(uint64_t begin, uint64_t end) const {
	uint64_t tick = 0;
	for (const auto& readback: m_inflight_readbacks) {
		if (readback.begin < end && begin < readback.end) {
			tick = std::max(tick, readback.tick);
		}
	}
	return tick;
}

static bool CopyFeedbackOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_COPY_FEEDBACK");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}

struct BufferCache::CopyFeedback {
	static constexpr uint64_t SlotSize  = 64 * 1024;
	static constexpr size_t   SlotCount = 512;
	struct Slot {
		uint64_t   address = 0, size = 0, tick = 0, mapping_epoch = 0;
		vk::Buffer owner   = nullptr;
	};
	Buffer                      download;
	std::array<Slot, SlotCount> slots {};
	std::vector<std::pair<uint64_t, size_t>> index;
	size_t                                   cursor = 0;
	[[nodiscard]] auto UpperBound(uint64_t address) {
		return std::upper_bound(index.begin(), index.end(), address,
		                        [](uint64_t value, const auto& entry) { return value < entry.first; });
	}
	[[nodiscard]] auto LowerBound(uint64_t address) {
		return std::lower_bound(index.begin(), index.end(), address,
		                        [](const auto& entry, uint64_t value) { return entry.first < value; });
	}
	bool Erase(uint64_t address) {
		const auto it = LowerBound(address);
		if (it == index.end() || it->first != address) return false;
		index.erase(it);
		return true;
	}
	bool Insert(uint64_t address, size_t slot) {
		const auto it = LowerBound(address);
		if (it != index.end() && it->first == address) return false;
		index.insert(it, {address, slot});
		return true;
	}
	std::array<uint16_t, 4096> granules {};
	void                       Count(const Slot& slot, int delta) {
		for (uint64_t g = slot.address >> 24u; g <= (slot.address + slot.size - 1) >> 24u; ++g)
			granules[g & 4095u] = static_cast<uint16_t>(granules[g & 4095u] + delta);
	}
	[[nodiscard]] bool MayOverlap(uint64_t vaddr, uint64_t size) const {
		const uint64_t first = vaddr >> 24u, last = (vaddr + size - 1) >> 24u;
		if (size == 0 || last - first >= granules.size()) return true;
		for (uint64_t g = first; g <= last; ++g)
			if (granules[g & 4095u] != 0) return true;
		return false;
	}
	CopyFeedback(GraphicContext& graphics, CommandScheduler& scheduler)
	    : download(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	               SlotCount * SlotSize) {
		const auto atom = graphics.physical_device_properties.limits.nonCoherentAtomSize;
		EXIT_IF(atom == 0 || SlotSize % atom != 0);
	}
};

void BufferCache::InvalidateCopyFeedback(uint64_t vaddr, uint64_t size) noexcept {
	if (!m_copy_feedback || m_copy_feedback->index.empty()) return;
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size)) return;
	auto it = feedback.LowerBound(vaddr);
	if (it != feedback.index.begin()) {
		const auto  prior = std::prev(it);
		const auto& slot  = feedback.slots[prior->second];
		if (slot.address + slot.size > vaddr) it = prior;
	}
	auto last = it;
	for (; last != feedback.index.end() && last->first < vaddr + size; ++last) {
		feedback.Count(feedback.slots[last->second], -1);
		feedback.slots[last->second].address = 0;
	}
	feedback.index.erase(it, last);
	m_copy_feedback_entries = feedback.index.size();
}

void BufferCache::ScheduleCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (!CopyFeedbackOn() || m_resources == nullptr) return;
	const auto mapping_epoch = m_resources->MappingEpoch();
	if (size == 0 || !GuestRange {vaddr, size}.Valid() || ((vaddr | size) & 3u) != 0 ||
	    size > CopyFeedback::SlotSize || !m_gpu_modified_ranges.Contains(vaddr, size) ||
	    m_texture_cache.IsRegionRegistered(vaddr, size) || !m_resources->IsMapped(vaddr, size)) {
		return;
	}
	const auto* owner_id = m_page_table.Find(vaddr >> PageTable::kPageBits);
	auto*       owner    = owner_id != nullptr && *owner_id ? GetBufferIfLive(*owner_id) : nullptr;
	if (owner == nullptr || !owner->IsInBounds(vaddr, size)) {
		return;
	}
	if (!m_copy_feedback) m_copy_feedback = std::make_unique<CopyFeedback>(m_graphics, m_scheduler);
	auto&  feedback = *m_copy_feedback;
	auto&  master   = m_scheduler.GetMasterSemaphore();
	size_t selected = CopyFeedback::SlotCount;
	for (unsigned refresh = 0; refresh < 2 && selected == CopyFeedback::SlotCount; ++refresh) {
		if (refresh != 0) master.Refresh();
		const auto completed = master.KnownGpuTick();
		for (unsigned evict = 0; evict < 2 && selected == CopyFeedback::SlotCount; ++evict) {
			for (size_t n = 0; n < CopyFeedback::SlotCount; ++n) {
				const auto  i    = (feedback.cursor + n) % CopyFeedback::SlotCount;
				const auto& slot = feedback.slots[i];
				if (slot.tick <= completed && (evict != 0 || slot.address == 0)) {
					selected = i;
					break;
				}
			}
		}
	}
	if (selected == CopyFeedback::SlotCount) {
		return;
	}
	if (m_resources->MappingEpoch() != mapping_epoch) return;
	auto& slot = feedback.slots[selected];
	if (slot.address != 0 && feedback.Erase(slot.address)) {
		feedback.Count(slot, -1);
	}
	InvalidateCopyFeedback(vaddr, size);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = owner->Handle();
	before.offset              = owner->Offset(vaddr);
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                       nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {owner->Offset(vaddr), selected * CopyFeedback::SlotSize, size};
	native.copyBuffer(owner->Handle(), feedback.download.Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = feedback.download.Handle();
	after.offset        = selected * CopyFeedback::SlotSize;
	after.size          = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                       1, &after, 0, nullptr);
	slot = {vaddr, size, m_scheduler.CurrentTick(), mapping_epoch, owner->Handle()};
	if (feedback.Insert(vaddr, selected)) feedback.Count(slot, 1);
	m_copy_feedback_entries = feedback.index.size();
	feedback.cursor         = (selected + 1) % CopyFeedback::SlotCount;
}

bool BufferCache::TryReadCopyFeedback(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	if (!m_copy_feedback || m_resources == nullptr) return false;
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (!buffer.IsInBounds(begin, end - begin) || end - begin > CopyFeedback::SlotSize ||
	    m_texture_cache.IsRegionRegistered(begin, end - begin)) {
		return false;
	}
	std::vector<std::pair<uint64_t, uint64_t>> copies;
	m_memory_tracker.ForEachDownloadRange<false>(begin, end - begin, [&](uint64_t address, uint64_t bytes) noexcept {
		m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes, "copy feedback");
		m_gpu_modified_ranges.ForEachInRange(address, bytes,
		                                     [&](uint64_t first, uint64_t last) { copies.emplace_back(first, last - first); });
	});
	if (copies.empty()) return false;
	struct Part {
		uint64_t address, size, offset;
	};
	std::vector<Part> parts;
	auto&             feedback      = *m_copy_feedback;
	const auto        mapping_epoch = m_resources->MappingEpoch();
	uint64_t          latest_tick   = 0;
	for (const auto& [address, bytes_in]: copies) {
		auto cursor = address;
		while (cursor < address + bytes_in) {
			auto it = feedback.UpperBound(cursor);
			if (it == feedback.index.begin()) return false;
			--it;
			const auto& slot = feedback.slots[it->second];
			if (cursor >= slot.address + slot.size) return false;
			if (slot.owner != buffer.Handle() || m_resources->UnmappedSince(slot.mapping_epoch, slot.address, slot.size) ||
			    !m_resources->IsMapped(slot.address, slot.size)) {
				return false;
			}
			const auto bytes = std::min(address + bytes_in, slot.address + slot.size) - cursor;
			parts.push_back({cursor, bytes, it->second * CopyFeedback::SlotSize + cursor - slot.address});
			latest_tick = std::max(latest_tick, slot.tick);
			cursor += bytes;
		}
	}
	if (!m_scheduler.IsFree(latest_tick)) return false;
	m_scheduler.WaitPriorityOperations(latest_tick);
	if (m_resources->MappingEpoch() != mapping_epoch) return false;
	for (const auto& part: parts) feedback.download.Invalidate(part.offset, part.size);
	for (const auto& part: parts) {
		LibKernel::Memory::WriteBacking(part.address, feedback.download.Mapped().data() + part.offset, part.size);
	}
	for (const auto& [address, bytes]: copies) {
		if (m_gpu_modified_ranges.Subtract(address, bytes)) {
			NoteGpuModifiedChange(address, bytes, false);
		}
		ForgetColorMetadataChecks(address, bytes);
	}
	UnmarkGpuModified(begin, end - begin);
	return true;
}

bool BufferCache::TryGuestReadFromFeedback(uint64_t vaddr, uint64_t size, bool is_write) {
	if (m_copy_feedback_entries == 0 || !m_copy_feedback) return false;
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size) || !IsRegionRegistered(vaddr, size)) return false;
	auto first = vaddr, last = vaddr + size;
	if (auto it = feedback.UpperBound(vaddr); it != feedback.index.begin()) {
		const auto& slot = feedback.slots[std::prev(it)->second];
		if (slot.address <= vaddr && vaddr < slot.address + slot.size) {
			first = std::min(first, slot.address);
			last  = std::max(last, slot.address + slot.size);
		}
	}
	if (InFlightReadbackTick(Common::AlignDown(first, TRACKER_PAGE_SIZE), Common::AlignUp(last, TRACKER_PAGE_SIZE)) != 0)
		return false;
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	if (last - first == size || !buffer.IsInBounds(first, last - first) ||
	    !TryReadCopyFeedback(buffer, first, last - first)) {
		if (!TryReadCopyFeedback(buffer, vaddr, size)) return false;
	}
	if (is_write) m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	return true;
}

void BufferCache::ReleaseReadback(uint64_t vaddr, uint64_t size, uint64_t pages_begin,
                                  uint64_t pages_end, uint64_t tick, bool is_write) {
	std::erase_if(m_inflight_readbacks,
	              [tick](const InFlightReadback& readback) { return readback.tick <= tick; });
	if (InFlightReadbackTick(pages_begin, pages_end) != 0 ||
	    m_gpu_modified_ranges.Intersects(pages_begin, pages_end - pages_begin)) {
		return;
	}
	UnmarkGpuModified(pages_begin, pages_end - pages_begin);
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

uint64_t BufferCache::ReadAheadSize(const Buffer& buffer, uint64_t vaddr, uint64_t size) const {
	static constexpr uint64_t MaxReadAhead = ReadAheadLimit;
	const auto                faulting_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto                faulting_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	const auto                limit =
	    std::min(buffer.CpuAddress() + buffer.Size(), faulting_end + MaxReadAhead);
	const auto latest_write = [&](uint64_t begin, uint64_t end) {
		uint64_t latest = 0;
		m_gpu_modified_ranges.ForEachInRange(begin, end - begin, [&](uint64_t first, uint64_t last) {
			latest = std::max(latest, m_write_ticks.Latest(first, last));
			for (const auto& [pending, bytes]: m_pending_writes) {
				if (pending < last && first < pending + bytes) {
					latest = UINT64_MAX;
				}
			}
		});
		return latest;
	};
	const auto needed = latest_write(faulting_begin, faulting_end);
	if (needed == 0) {
		return size;
	}
	const auto current = m_scheduler.CurrentTick();
	const auto bound   = needed >= current ? UINT64_MAX : current - 1;
	auto end    = faulting_end;
	auto cursor = faulting_end;
	while (cursor + TRACKER_PAGE_SIZE <= limit &&
	       !m_texture_cache.IsRegionGpuModified(cursor, TRACKER_PAGE_SIZE)) {
		if (!m_gpu_modified_ranges.Intersects(cursor, TRACKER_PAGE_SIZE)) {
			cursor += TRACKER_PAGE_SIZE;
			continue;
		}
		const auto tick = latest_write(cursor, cursor + TRACKER_PAGE_SIZE);
		if (tick == 0 || tick > bound) {
			break;
		}
		cursor += TRACKER_PAGE_SIZE;
		end = cursor;
	}
	return end > faulting_end ? end - vaddr : size;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

void BufferCache::CacheFaultedMemory(uint64_t vaddr, uint64_t size) {
	for (size_t i = 0; i < StaleRingSize; i++) {
		const auto address = m_stale_ring[i * 2].load(std::memory_order_relaxed);
		const auto length  = m_stale_ring[i * 2 + 1].load(std::memory_order_relaxed);
		if (length != 0 && address < vaddr + size && vaddr < address + length) {
			Common::HackNotice::Report(
			    "buffer released by age reached again by a device-address fault",
			    "BufferCache::CacheFaultedMemory", 0, 0,
			    "a shader read that memory before the buffer existed again (first read may be zero)");
			break;
		}
	}
	if (const auto id = FindBuffer(vaddr, size)) {
		m_slot_buffers[id].device_address_pinned = true;
	}
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.device_address_pinned |= overlap.device_address_pinned;
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	NoteWrite(overlap.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	{
		uint32_t joined = 0;
		uint64_t at     = overlap.begin;
		for (auto it = overlap.first; it != overlap.last; ++it) {
			const auto& old = m_slot_buffers[it->second];
			if (old.CpuAddress() > at) {
				DiagArgsSyncNoteBuffer(12, at, old.CpuAddress() - at, 0);
			}
			DiagArgsSyncNoteBuffer(10, old.CpuAddress(), old.Size(), 0);
			at = std::max(at, old.CpuAddress() + old.Size());
			joined++;
		}
		if (at < overlap.end) {
			DiagArgsSyncNoteBuffer(12, at, overlap.end - at, 0);
		}
		DiagArgsSyncNoteBuffer(9, overlap.begin, overlap.end - overlap.begin,
		                       joined | (overlap.has_stream_leap ? 0x80000000u : 0u));
	}
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::NoImageStartsAt(uint64_t vaddr) const noexcept {
	TextureCache::ImagePageTable::PageRange pages {};
	return TextureCache::ImagePageTable::TryGetPageRange(vaddr, 1, pages) &&
	       !m_texture_cache.m_registered_pages.Any(pages.first, pages.last_exclusive);
}

namespace {

bool ArgsSyncDiag() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DIAG_ARGS_SYNC");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool ArgsPreferGpu() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DIAG_ARGS_PREFER_GPU");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool SyncPreferGpu() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DIAG_SYNC_PREFER_GPU");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

uint64_t g_sync_skipped_runs  = 0;
uint64_t g_sync_skipped_bytes = 0;

template <typename Emit>
void SyncSkipMarked(std::span<const std::pair<uint64_t, uint64_t>> marked, uint64_t address, uint64_t bytes, Emit&& emit) {
	auto       at      = address;
	const auto end     = address + bytes;
	uint64_t   emitted = 0;
	for (const auto& [marked_begin, marked_size]: marked) {
		const auto marked_end = marked_begin + marked_size;
		if (marked_end <= at || marked_begin >= end) {
			continue;
		}
		if (marked_begin > at) {
			emit(at, marked_begin - at);
			emitted += marked_begin - at;
		}
		at = std::max(at, marked_end);
		if (at >= end) {
			break;
		}
	}
	if (at < end) {
		emit(at, end - at);
		emitted += end - at;
	}
	if (emitted != bytes) {
		g_sync_skipped_runs++;
		g_sync_skipped_bytes += bytes - emitted;
	}
}

struct ArgsSyncCounts {
	uint64_t reads    = 0;
	uint64_t dirty    = 0;
	uint64_t streamed = 0;
	uint64_t uploaded = 0;
	uint64_t over_gpu = 0;
	uint64_t cpu_only = 0;
	uint64_t kept_gpu = 0;
	uint64_t any_over_gpu = 0;
	uint64_t last_ns  = 0;
	uint32_t printed  = 0;
	uint32_t any_printed = 0;
};

ArgsSyncCounts g_args_sync;

struct SyncRingEntry {
	uint64_t    seq    = 0;
	uint64_t    vaddr  = 0;
	uint64_t    size   = 0;
	const void* caller = nullptr;
	uint8_t     kind   = 0;
	uint8_t     marks  = 0;
	uint8_t     op_kind    = 0xff;
	uint8_t     guest_sync = 0;
	uint8_t     path       = 0;
	int32_t     queue      = -1;
	uint64_t    op_seq     = 0;
	uint64_t    submit     = 0;
	uint64_t    shader     = 0;
	std::array<uint32_t, 4> words {};
	uint64_t    ind_base   = 0;
	uint32_t    ind_offset = 0;
	uint32_t    ind_count  = 0;
	uint32_t    ind_stride = 0;
	uint32_t    ind_run0   = 0;
	uint32_t    ind_runl   = 0;
	uint8_t     ind_run    = 0;
	uint32_t    pkt_header = 0;
	std::array<uint32_t, 3> pkt_body {};
	uint64_t    sub_order  = 0;
	int32_t     sub_queue  = -1;
};

constexpr size_t SyncRingSize = 4096;

struct SyncRing {
	std::mutex                              mutex;
	std::array<SyncRingEntry, SyncRingSize> uploads {};
	std::array<SyncRingEntry, SyncRingSize> writes {};
	std::array<SyncRingEntry, SyncRingSize> reads {};
	std::array<SyncRingEntry, SyncRingSize> cp {};
	std::array<SyncRingEntry, SyncRingSize> skips {};
	std::array<SyncRingEntry, SyncRingSize> life {};
	uint64_t                                seq          = 0;
	uint64_t                                upload_count = 0;
	uint64_t                                write_count  = 0;
	uint64_t                                read_count   = 0;
	uint64_t                                cp_count     = 0;
	uint64_t                                skip_count   = 0;
	uint64_t                                life_count   = 0;
	uint64_t                                skip_marked  = 0;
	uint64_t                                skip_window  = 0;
	uint64_t                                skip_window_marked = 0;
	MemoryTracker*                          tracker      = nullptr;
};

SyncRing g_sync_ring;

#if defined(__GNUC__) || defined(__clang__)
#define KYTY_DIAG_RETURN_ADDRESS() __builtin_return_address(0)
#else
#define KYTY_DIAG_RETURN_ADDRESS() nullptr
#endif

struct SyncOpContext {
	int32_t  queue      = -1;
	uint8_t  op_kind    = 0xff;
	uint8_t  guest_sync = 0;
	uint64_t op_seq     = 0;
	uint64_t submit     = 0;
};

thread_local SyncOpContext g_sync_op;

struct SyncBindingHint {
	uint8_t                 path = 0;
	std::array<uint32_t, 4> words {};
};

thread_local SyncBindingHint g_sync_hint;

struct SyncIndirect {
	uint64_t base   = 0;
	uint32_t offset = 0;
	uint32_t count  = 0;
	uint32_t stride = 0;
	uint32_t run0   = 0;
	uint32_t runl   = 0;
	uint8_t  run    = 0;
};

thread_local SyncIndirect g_sync_indirect;

thread_local uint64_t g_sync_shader = 0;

struct SyncDecision {
	uint8_t  source    = 0;
	uint64_t write_seq = 0;
	uint64_t emitted   = 0;
};

thread_local SyncDecision g_sync_decision;

thread_local const uint32_t* g_sync_packet = nullptr;

struct SyncWaitEntry {
	uint64_t seq       = 0;
	uint64_t address   = 0;
	uint64_t value     = 0;
	uint64_t reference = 0;
	uint64_t mask      = 0;
	uint64_t write_seq = 0;
	uint64_t emitted   = 0;
	uint64_t repeats   = 0;
	int32_t  queue     = -1;
	uint32_t size      = 0;
	uint32_t function  = 0;
	uint8_t  met       = 0;
	uint8_t  source    = 0;
	uint8_t  marks     = 0;
	const uint32_t* packet = nullptr;
	uint64_t hold_order  = 0;
	uint64_t hold_before = 0;
	uint64_t hold_done   = 0;
	uint8_t  hold_why    = 0xffu;
};

struct SyncWaits {
	std::array<SyncWaitEntry, 4096> ring {};
	std::array<uint64_t, 64>        last_by_queue {};
	uint64_t                        count = 0;
};

SyncWaits g_sync_waits;

struct SyncHoldCheck {
	std::atomic<uint64_t> order {0};
	std::atomic<uint64_t> before {0};
	std::atomic<uint64_t> done {0};
	std::atomic<uint32_t> why {0xffu};
};

std::array<SyncHoldCheck, 64>        g_sync_hold_last {};
std::array<std::atomic<uint64_t>, 4> g_sync_hold_why {};
thread_local const uint32_t*         g_sync_cp_packet = nullptr;
thread_local uint64_t                g_sync_sub_order = 0;
thread_local int32_t                 g_sync_sub_queue = -1;

constexpr const char* SyncHoldWhy[] = {
    "held, or would be held with KYTY_DIAG_ACE_HOLD off (an earlier q0 submission not fully parsed, q0's front not blocked)",
    "no hold: every q0 submission enqueued before it was fully parsed",
    "no hold: q0 empty although an earlier q0 submission is not marked done",
    "no hold: q0's front blocked (the no-deadlock exception)"};

struct SyncDecisionEntry {
	uint64_t seq        = 0;
	uint64_t address    = 0;
	uint64_t value      = 0;
	uint64_t reference  = 0;
	uint64_t mask       = 0;
	uint64_t write_seq  = 0;
	uint64_t emitted    = 0;
	int32_t  queue      = -1;
	uint32_t size       = 0;
	uint32_t function   = 0;
	uint32_t exec_count = 0;
	uint8_t  kind       = 0;
	uint8_t  taken      = 0;
	uint8_t  source     = 0;
	uint8_t  marks      = 0;
};

struct SyncDecisions {
	std::array<SyncDecisionEntry, 4096> ring {};
	uint64_t                            count = 0;
};

SyncDecisions g_sync_decisions;

struct CullBinding {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint8_t  written = 0;
};

struct CullSlot {
	uint64_t                   seq        = 0;
	uint64_t                   op_seq     = 0;
	uint64_t                   submit     = 0;
	uint64_t                   shader     = 0;
	uint64_t                   tick       = 0;
	uint64_t                   args_guest = 0;
	int32_t                    queue      = -1;
	uint32_t                   mode       = 0;
	uint32_t                   flags      = 0;
	uint8_t                    kind       = 0;
	uint8_t                    bindings   = 0;
	uint8_t                    post       = 0;
	uint8_t                    has_args   = 0;
	uint8_t                    has_groups = 0;
	uint32_t                   direct_groups[3] {};
	std::array<CullBinding, 8> b {};
	std::array<int8_t, 8>      post_head {};
};

constexpr uint32_t CullSlots     = 512;
constexpr uint64_t CullHeadBytes = 4096u;
constexpr uint64_t CullPreBase   = 1024u;
constexpr uint64_t CullPostBase  = CullPreBase + 8u * CullHeadBytes;
constexpr uint64_t CullSlotBytes = CullPostBase + 4u * CullHeadBytes;

struct CullState {
	std::unique_ptr<Buffer>          ring;
	CommandScheduler*                scheduler = nullptr;
	std::array<CullSlot, CullSlots> slots {};
	uint32_t                         next    = 0;
	int32_t                          current = -1;
};

CullState g_cull;

thread_local std::vector<CullBinding> g_cull_bindings;

struct CullCp {
	uint64_t op_seq = 0;
	uint64_t args   = 0;
	uint64_t base   = 0;
	uint32_t mode   = 0;
	int32_t  queue  = -1;
};

std::array<CullCp, 4096> g_cull_cp {};
uint64_t                 g_cull_cp_count = 0;

bool CullShader(uint64_t hash) {
	static const std::vector<uint64_t> set = [] {
		std::vector<uint64_t> values;
		const char*           text = std::getenv("KYTY_DIAG_CULL_SHADERS");
		const std::string     list = text != nullptr && text[0] != '\0'
		                                  ? std::string(text)
		                                  : std::string("7d3742bf962e7527,1b64e3043113a655,f757b6bfbb13cf67,"
		                                                "c820ae0968beaefd,0de564535eb5bf62,80a562a7c9fab830,"
		                                                "651dc20d4c870826,49c47c3080e7a0fc,05d7a53196247a0a,297c685a60d19e1b");
		size_t at = 0;
		while (at < list.size()) {
			const auto comma = list.find_first_of(",; :", at);
			const auto item  = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
			if (!item.empty()) {
				values.push_back(std::strtoull(item.c_str(), nullptr, 16));
			}
			if (comma == std::string::npos) {
				break;
			}
			at = comma + 1;
		}
		return values;
	}();
	return hash != 0 && std::find(set.begin(), set.end(), hash) != set.end();
}

uint64_t CullObjectBytes() {
	static const uint64_t bytes = [] {
		const char* text = std::getenv("KYTY_DIAG_CULL_SLOTS");
		if (text == nullptr || text[0] == '\0') {
			return uint64_t {0};
		}
		char*          end   = nullptr;
		const uint64_t slots = std::strtoull(text, &end, 10);
		if (end == nullptr || *end != ':') {
			return uint64_t {0};
		}
		return static_cast<uint64_t>(slots * std::strtoull(end + 1, nullptr, 10));
	}();
	return bytes;
}

struct RunShaderConfig {
	uint64_t hash    = 0;
	uint64_t threads = 0;
	uint64_t stride  = 0;
};

const RunShaderConfig& DiagRunShader() {
	static const RunShaderConfig config = [] {
		RunShaderConfig c {};
		const char*     text = std::getenv("KYTY_DIAG_RUN_SHADER");
		if (text == nullptr || text[0] == '\0') {
			return c;
		}
		char*          end  = nullptr;
		const uint64_t hash = std::strtoull(text, &end, 16);
		if (end == nullptr || *end != ':') {
			return c;
		}
		const uint64_t threads = std::strtoull(end + 1, &end, 10);
		if (end == nullptr || *end != ':') {
			return c;
		}
		const uint64_t stride = std::strtoull(end + 1, nullptr, 10);
		if (hash != 0 && threads != 0 && stride != 0) {
			c = {hash, threads, stride};
		}
		return c;
	}();
	return config;
}

constexpr const char* CullKinds[] = {"?", "dispatch direct", "dispatch indirect", "dispatch indirect (thread counts)"};

constexpr const char* SyncDecisionSources[] = {"?", "a parsed CP write of known value", "guest memory after waiting for a write's execution",
                                               "a guest-memory copy", "a guest-memory copy (no pipeline)"};

SyncRingEntry SyncEntry(uint8_t kind, uint64_t vaddr, uint64_t size, const void* caller, uint8_t marks, bool take_hint) {
	SyncRingEntry e {};
	e.vaddr      = vaddr;
	e.size       = size;
	e.caller     = caller;
	e.kind       = kind;
	e.marks      = marks;
	e.queue      = g_sync_op.queue;
	e.op_kind    = g_sync_op.op_kind;
	e.guest_sync = g_sync_op.guest_sync;
	e.op_seq     = g_sync_op.op_seq;
	e.submit     = g_sync_op.submit;
	e.shader     = Common::HackNotice::CurrentShaderSlot() != 0 ? Common::HackNotice::CurrentShaderSlot() : g_sync_shader;
	if (kind == 5) {
		e.ind_base   = g_sync_indirect.base;
		e.ind_offset = g_sync_indirect.offset;
		e.ind_count  = g_sync_indirect.count;
		e.ind_stride = g_sync_indirect.stride;
		e.ind_run0   = g_sync_indirect.run0;
		e.ind_runl   = g_sync_indirect.runl;
		e.ind_run    = g_sync_indirect.run;
	}
	if (take_hint) {
		e.path      = g_sync_hint.path;
		e.words     = g_sync_hint.words;
		g_sync_hint = {};
	}
	return e;
}

constexpr const char* SyncRingKinds[] = {"?", "upload (one buffer)", "upload (buffer spans)",
                                         "GPU write (a written binding)", "GPU write (write-bitmap readback)",
                                         "indirect args read", "CP write (parse time)", "?",
                                         "dispatch skipped on a CPU-read zero count", "buffer created",
                                         "old buffer joined into a new one (copied)", "buffer deleted",
                                         "GAP of a new buffer no old buffer covered", "GC eviction, clean per the tracker (no download)",
                                         "GC eviction, GPU-dirty (downloaded)"};

constexpr const char* SyncOpKinds[] = {"draw index", "draw auto", "draw indirect", "dispatch direct",
                                       "dispatch indirect", "callback"};

uint8_t SyncMarks(const RegionManager::ArgsPageProbe& probe) {
	return static_cast<uint8_t>((probe.cpu_dirty ? 1u : 0u) | (probe.gpu_dirty ? 2u : 0u) | (probe.gpu_written ? 4u : 0u) |
	                            (probe.gpu_noted ? 8u : 0u) | (probe.block_opened ? 16u : 0u));
}

void SyncRingRecordWrite(uint8_t kind, uint64_t vaddr, uint64_t size, const void* caller) {
	auto             e = SyncEntry(kind, vaddr, size, caller, 0, kind == 3);
	std::scoped_lock lock(g_sync_ring.mutex);
	auto&            r = g_sync_ring;
	e.seq             = ++r.seq;
	r.writes[r.write_count++ % SyncRingSize] = e;
}

void SyncCallerText(const void* caller, char* text, size_t size) {
	std::snprintf(text, size, "-");
	if (caller == nullptr) {
		return;
	}
	std::snprintf(text, size, "%p", caller);
#if defined(__linux__)
	Dl_info info {};
	if (dladdr(caller, &info) != 0 && info.dli_fbase != nullptr && info.dli_fname != nullptr) {
		const char* name = std::strrchr(info.dli_fname, '/');
		std::snprintf(text, size, "%s+0x%" PRIxPTR, name != nullptr ? name + 1 : info.dli_fname,
		              reinterpret_cast<uintptr_t>(caller) - reinterpret_cast<uintptr_t>(info.dli_fbase));
	}
#endif
}

void ArgsSyncReport() {
	auto&      c   = g_args_sync;
	const auto now = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
	if (c.last_ns == 0) {
		c.last_ns = now;
		return;
	}
	if (now - c.last_ns < 10'000'000'000ull) {
		return;
	}
	std::fprintf(stderr,
	             "[kyty-args-sync] last %.0f s: %" PRIu64 " args reads, %" PRIu64 " on CPU-dirty pages, %" PRIu64
	             " streamed from guest memory, %" PRIu64 " uploaded CPU-dirty pages; %" PRIu64
	             " of those over GPU-written pages, %" PRIu64 " with no GPU mark; %" PRIu64 " kept the GPU's bytes; %" PRIu64
	             " uploads by any reader over GPU-written pages\n",
	             static_cast<double>(now - c.last_ns) / 1e9, c.reads, c.dirty, c.streamed, c.uploaded, c.over_gpu,
	             c.cpu_only, c.kept_gpu, c.any_over_gpu);
	if (SyncPreferGpu()) {
		std::fprintf(stderr,
		             "[kyty-args-sync] last %.0f s: prefer GPU (all readers) skipped %" PRIu64 " upload runs, %" PRIu64
		             " bytes on GPU-marked pages\n",
		             static_cast<double>(now - c.last_ns) / 1e9, g_sync_skipped_runs, g_sync_skipped_bytes);
		g_sync_skipped_runs = g_sync_skipped_bytes = 0;
	}
	{
		std::scoped_lock lock(g_sync_ring.mutex);
		std::fprintf(stderr,
		             "[kyty-args-sync] last %.0f s: %" PRIu64 " indirect dispatches skipped on a CPU-read zero count, %" PRIu64
		             " of them with GPU marks on their args page; no-zero-skip %s\n",
		             static_cast<double>(now - c.last_ns) / 1e9, g_sync_ring.skip_window, g_sync_ring.skip_window_marked,
		             DiagNoZeroSkip() ? "on" : "off");
		g_sync_ring.skip_window = g_sync_ring.skip_window_marked = 0;
	}
	c.last_ns = now;
	c.reads = c.dirty = c.streamed = c.uploaded = c.over_gpu = c.cpu_only = c.kept_gpu = c.any_over_gpu = 0;
}

void ArgsSyncPrint(const char* what, uint32_t site, uint64_t vaddr, uint64_t size,
                   const RegionManager::ArgsPageProbe& probe) {
	auto& c = g_args_sync;
	if (c.printed >= 64) {
		return;
	}
	c.printed++;
	std::fprintf(stderr,
	             "[kyty-args-sync] %s: site %u args 0x%016" PRIx64 " size %" PRIu64 " pages %" PRIu64
	             ": cpu_dirty %d gpu_dirty %d gpu_written %d gpu_noted %d block_opened %d\n",
	             what, site, vaddr, size,
	             (vaddr + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE - vaddr / TRACKER_PAGE_SIZE,
	             probe.cpu_dirty ? 1 : 0, probe.gpu_dirty ? 1 : 0, probe.gpu_written ? 1 : 0,
	             probe.gpu_noted ? 1 : 0, probe.block_opened ? 1 : 0);
}

void ArgsSyncNoteUploads(MemoryTracker& tracker, const Buffer& buffer, std::span<const vk::BufferCopy> copies,
                         uint8_t kind, const void* caller) {
	auto& c = g_args_sync;
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		const auto probe   = tracker.ProbeArgsPages(address, copy.size);
		{
			auto             e = SyncEntry(kind, address, static_cast<uint64_t>(copy.size), caller, SyncMarks(probe), false);
			std::scoped_lock lock(g_sync_ring.mutex);
			auto&            r = g_sync_ring;
			r.tracker         = &tracker;
			e.seq             = ++r.seq;
			r.uploads[r.upload_count++ % SyncRingSize] = e;
		}
		if (!probe.gpu_written && !probe.gpu_noted) {
			continue;
		}
		c.any_over_gpu++;
		if (c.any_printed < 16) {
			c.any_printed++;
			std::fprintf(stderr,
			             "[kyty-args-sync] an upload (any reader) over GPU-written pages: address 0x%016" PRIx64
			             " bytes %" PRIu64 ": gpu_dirty %d gpu_written %d gpu_noted %d block_opened %d\n",
			             address, static_cast<uint64_t>(copy.size), probe.gpu_dirty ? 1 : 0, probe.gpu_written ? 1 : 0,
			             probe.gpu_noted ? 1 : 0, probe.block_opened ? 1 : 0);
		}
	}
}

}

void DiagArgsSyncSetOperation(uint32_t queue, uint32_t kind, uint64_t sequence, uint64_t submit, bool guest_sync) {
	if (!ArgsSyncDiag()) {
		return;
	}
	g_sync_op = {static_cast<int32_t>(queue), static_cast<uint8_t>(kind), static_cast<uint8_t>(guest_sync ? 1u : 0u), sequence,
	             submit};
	g_sync_indirect = {};
	g_sync_shader   = 0;
}

void DiagArgsSyncSetShader(uint64_t hash) {
	if (!ArgsSyncDiag()) {
		return;
	}
	g_sync_shader = hash;
	g_cull_bindings.clear();
}

void DiagArgsSyncNoteBinding(uint64_t address, uint64_t size, bool written) {
	if (!ArgsSyncDiag() || address == 0 || size == 0 || !CullShader(g_sync_shader)) {
		return;
	}
	for (auto& b: g_cull_bindings) {
		if (b.address == address && b.size == size) {
			b.written = static_cast<uint8_t>(b.written | (written ? 1u : 0u));
			return;
		}
	}
	if (g_cull_bindings.size() < 8u) {
		g_cull_bindings.push_back({address, size, static_cast<uint8_t>(written ? 1u : 0u)});
	}
}

void DiagArgsSyncNoteDispatchCp(uint32_t queue, uint64_t sequence, uint64_t args, uint64_t base, uint32_t mode) {
	if (!ArgsSyncDiag()) {
		return;
	}
	std::scoped_lock lock(g_sync_ring.mutex);
	g_cull_cp[g_cull_cp_count++ % g_cull_cp.size()] = {sequence, args, base, mode, static_cast<int32_t>(queue)};
}

bool DiagNoZeroSkip() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DIAG_NO_ZERO_SKIP");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void DiagArgsSyncNoteCpWrite(uint32_t queue, uint64_t sequence, uint64_t address, uint64_t size, const void* value) {
	if (!ArgsSyncDiag() || size == 0) {
		return;
	}
	SyncRingEntry e {};
	e.vaddr  = address;
	e.size   = size;
	e.kind   = 6;
	e.queue  = static_cast<int32_t>(queue);
	e.op_seq = sequence;
	if (const auto* p = g_sync_cp_packet; p != nullptr) {
		std::array<uint32_t, 4> pkt {};
		if (LibKernel::Memory::TryReadBacking(reinterpret_cast<uint64_t>(p), pkt.data(), sizeof(pkt))) {
			e.pkt_header = pkt[0];
			std::copy_n(pkt.begin() + 1, 3, e.pkt_body.begin());
		}
	}
	e.sub_order = g_sync_sub_order;
	e.sub_queue = g_sync_sub_queue;
	if (value != nullptr) {
		std::memcpy(e.words.data(), value, static_cast<size_t>(std::min<uint64_t>(size, sizeof(uint32_t) * 4u)));
	}
	std::scoped_lock lock(g_sync_ring.mutex);
	auto&            r = g_sync_ring;
	e.seq             = ++r.seq;
	r.cp[r.cp_count++ % SyncRingSize] = e;
}

void DiagArgsSyncNoteWaitPacket(const uint32_t* body) {
	if (!ArgsSyncDiag()) {
		return;
	}
	g_sync_packet = body;
}

void DiagArgsSyncNoteBuffer(uint8_t kind, uint64_t vaddr, uint64_t size, uint32_t extra) {
	if (!ArgsSyncDiag() || size == 0) {
		return;
	}
	auto*      tracker = g_sync_ring.tracker;
	const auto probe   = tracker != nullptr ? tracker->ProbeArgsPages(vaddr, size) : RegionManager::ArgsPageProbe {};
	auto       e       = SyncEntry(kind, vaddr, size, nullptr, SyncMarks(probe), false);
	e.words[0]         = extra;
	std::scoped_lock lock(g_sync_ring.mutex);
	auto&            r = g_sync_ring;
	e.seq             = ++r.seq;
	r.life[r.life_count++ % SyncRingSize] = e;
}

void DiagArgsSyncNoteDecisionSource(uint8_t source, uint64_t write_sequence, uint64_t emitted) {
	if (!ArgsSyncDiag()) {
		return;
	}
	g_sync_decision = {source, write_sequence, emitted};
}

void DiagArgsSyncNoteWait(uint32_t queue, uint64_t address, uint32_t size, uint64_t value, uint64_t reference, uint64_t mask,
                          uint32_t function, bool met) {
	if (!ArgsSyncDiag()) {
		return;
	}
	auto*      tracker  = g_sync_ring.tracker;
	const auto probe    = tracker != nullptr ? tracker->ProbeArgsPages(address, size) : RegionManager::ArgsPageProbe {};
	const auto decision = std::exchange(g_sync_decision, SyncDecision {});
	std::scoped_lock lock(g_sync_ring.mutex);
	auto&            w    = g_sync_waits;
	auto&            last = w.last_by_queue[queue % 64u];
	if (last != 0 && w.count - (last - 1) < w.ring.size()) {
		auto& previous = w.ring[(last - 1) % w.ring.size()];
		if (previous.met == 0 && !met && previous.address == address && previous.reference == reference &&
		    previous.mask == mask && previous.function == function) {
			previous.repeats++;
			previous.value   = value;
			previous.emitted = decision.emitted;
			return;
		}
	}
	SyncWaitEntry e {};
	e.seq       = ++g_sync_ring.seq;
	e.address   = address;
	e.value     = value;
	e.reference = reference;
	e.mask      = mask;
	e.write_seq = decision.write_seq;
	e.emitted   = decision.emitted;
	e.queue     = static_cast<int32_t>(queue);
	e.size      = size;
	e.function  = function;
	e.met       = met ? 1u : 0u;
	e.source    = decision.source;
	e.marks     = SyncMarks(probe);
	e.packet    = std::exchange(g_sync_packet, nullptr);
	if (e.queue >= 0) {
		const auto& h = g_sync_hold_last[static_cast<uint32_t>(e.queue) % 64u];
		e.hold_why    = static_cast<uint8_t>(h.why.load(std::memory_order_relaxed));
		e.hold_order  = h.order.load(std::memory_order_relaxed);
		e.hold_before = h.before.load(std::memory_order_relaxed);
		e.hold_done   = h.done.load(std::memory_order_relaxed);
	}
	w.ring[w.count % w.ring.size()] = e;
	last = ++w.count;
}

void DiagArgsSyncNoteDecision(uint32_t queue, uint8_t kind, uint64_t address, uint32_t size, uint64_t value, uint64_t reference,
                              uint64_t mask, uint32_t function, uint32_t exec_count, bool taken) {
	if (!ArgsSyncDiag()) {
		return;
	}
	auto*      tracker  = g_sync_ring.tracker;
	const auto probe    = tracker != nullptr ? tracker->ProbeArgsPages(address, size) : RegionManager::ArgsPageProbe {};
	const auto decision = std::exchange(g_sync_decision, SyncDecision {});
	SyncDecisionEntry e {};
	e.address    = address;
	e.value      = value;
	e.reference  = reference;
	e.mask       = mask;
	e.write_seq  = decision.write_seq;
	e.emitted    = decision.emitted;
	e.queue      = static_cast<int32_t>(queue);
	e.size       = size;
	e.function   = function;
	e.exec_count = exec_count;
	e.kind       = kind;
	e.taken      = taken ? 1u : 0u;
	e.source     = decision.source;
	e.marks      = SyncMarks(probe);
	std::scoped_lock lock(g_sync_ring.mutex);
	e.seq = ++g_sync_ring.seq;
	auto& d = g_sync_decisions;
	d.ring[d.count++ % d.ring.size()] = e;
}

std::array<std::atomic<uint64_t>, 256> g_predicated_skips {};
std::atomic<uint64_t>                  g_ace_holds {0};

void DiagArgsSyncNotePredicatedSkip(uint32_t queue, uint32_t opcode) {
	(void)queue;
	if (!ArgsSyncDiag()) {
		return;
	}
	g_predicated_skips[opcode & 0xffu].fetch_add(1, std::memory_order_relaxed);
}

const char* DiagAceHoldText() {
	const char* value = std::getenv("KYTY_DIAG_ACE_HOLD");
	return value != nullptr && value[0] == '1' ? "on" : "off";
}

void DiagArgsSyncNoteAceHold() {
	g_ace_holds.fetch_add(1, std::memory_order_relaxed);
}

void DiagArgsSyncNoteDispatchSkip(uint64_t args_address, const uint32_t* counts3, uint64_t shader_address) {
	if (!ArgsSyncDiag()) {
		return;
	}
	auto*      tracker = g_sync_ring.tracker;
	const auto probe   = tracker != nullptr ? tracker->ProbeArgsPages(args_address, sizeof(uint32_t) * 3u)
	                                       : RegionManager::ArgsPageProbe {};
	auto e     = SyncEntry(8, args_address, sizeof(uint32_t) * 3u, nullptr, SyncMarks(probe), false);
	e.shader   = shader_address;
	e.words[0] = counts3[0];
	e.words[1] = counts3[1];
	e.words[2] = counts3[2];
	const bool marked = probe.gpu_written || probe.gpu_noted;
	std::scoped_lock lock(g_sync_ring.mutex);
	auto&            r = g_sync_ring;
	e.seq             = ++r.seq;
	r.skips[r.skip_count++ % SyncRingSize] = e;
	r.skip_window++;
	if (marked) {
		r.skip_marked++;
		r.skip_window_marked++;
	}
}

void DiagArgsSyncSetIndirect(uint64_t base, uint32_t data_offset, uint32_t count, uint32_t stride, const uint32_t* run_offsets) {
	if (!ArgsSyncDiag()) {
		return;
	}
	g_sync_indirect = {.base = base, .offset = data_offset, .count = count, .stride = stride};
	if (run_offsets != nullptr && count != 0) {
		g_sync_indirect.run  = 1;
		g_sync_indirect.run0 = run_offsets[0];
		g_sync_indirect.runl = run_offsets[5u * (count - 1u)];
	}
}

void DiagArgsSyncHint(uint8_t path, const uint32_t* words4) {
	if (!ArgsSyncDiag()) {
		return;
	}
	g_sync_hint.path  = path;
	g_sync_hint.words = {};
	if (words4 != nullptr) {
		std::memcpy(g_sync_hint.words.data(), words4, sizeof(uint32_t) * 4u);
	}
}

namespace {

struct SyncSetBaseEntry {
	uint64_t                seq    = 0;
	const uint32_t*         packet = nullptr;
	std::array<uint32_t, 4> words {};
	uint8_t                 marks  = 0;
};

struct SyncDrawPacketEntry {
	uint64_t                 seq         = 0;
	uint64_t                 args_base   = 0;
	const uint32_t*          packet      = nullptr;
	SyncSetBaseEntry         set_base {};
	std::array<uint32_t, 10> words {};
	int32_t                  queue       = -1;
	uint32_t                 dw          = 0;
	uint32_t                 data_offset = 0;
	uint32_t                 count       = 0;
	uint32_t                 stride      = 0;
	uint8_t                  marks       = 0;
	uint8_t                  run         = 0;
};

struct SyncIbFetchEntry {
	uint64_t seq     = 0;
	uint64_t address = 0;
	uint64_t size    = 0;
	int32_t  queue   = -1;
	uint8_t  how     = 0;
	uint8_t  marks   = 0;
};

struct SyncPackets {
	std::array<SyncDrawPacketEntry, 512> draws {};
	std::array<SyncIbFetchEntry, 1024>   ibs {};
	std::array<uint64_t, 4>              ib_by_how {};
	uint64_t                             draw_count = 0;
	uint64_t                             ib_count   = 0;
};

std::array<SyncSetBaseEntry, 64> g_sync_set_base {};
SyncPackets                      g_sync_packets;
thread_local const uint32_t*     g_sync_draw_packet    = nullptr;
thread_local uint32_t            g_sync_draw_packet_dw = 0;

constexpr const char* SyncIbHow[] = {"an IB fetched with no PFP_SYNC_ME armed (never checked for GPU writes)",
                                     "an IB fetched after PFP_SYNC_ME (checked; not GPU-modified per the buffer cache)",
                                     "an IB downloaded after PFP_SYNC_ME (GPU-modified per the buffer cache)",
                                     "a top-level submission (never checked for GPU writes)"};

void SyncMarksText(uint8_t m, char* text, size_t size) {
	std::snprintf(text, size, "cpu_dirty %d gpu_dirty %d gpu_written %d gpu_noted %d block_opened %d", (m & 1u) != 0 ? 1 : 0,
	              (m & 2u) != 0 ? 1 : 0, (m & 4u) != 0 ? 1 : 0, (m & 8u) != 0 ? 1 : 0, (m & 16u) != 0 ? 1 : 0);
}

struct SyncPacketNow {
	std::array<uint32_t, 10> words {};
	uint8_t                  marks = 0;
	bool                     read  = false;
};

SyncPacketNow SyncReadPacketNow(const uint32_t* packet, uint32_t dw) {
	SyncPacketNow now {};
	if (packet == nullptr || dw == 0) {
		return now;
	}
	const auto address = reinterpret_cast<uint64_t>(packet);
	const auto bytes   = sizeof(uint32_t) * std::min<uint32_t>(dw, 10u);
	auto*      tracker = g_sync_ring.tracker;
	now.marks          = tracker != nullptr ? SyncMarks(tracker->ProbeArgsPages(address, bytes)) : 0;
	now.read           = LibKernel::Memory::TryReadBacking(address, now.words.data(), bytes);
	return now;
}

void SyncPrintWords(const char* what, const uint32_t* words, uint32_t dw) {
	std::fprintf(stderr, "[kyty-args-sync]       %s:", what);
	for (uint32_t i = 0; i < dw; i++) {
		std::fprintf(stderr, " %08x", words[i]);
	}
	std::fprintf(stderr, "\n");
}

void SyncPrintPageRings(uint64_t address, const char* what) {
	const auto page = address & ~(TRACKER_PAGE_SIZE - 1);
	auto&      r    = g_sync_ring;
	const auto ring = [&](const std::array<SyncRingEntry, SyncRingSize>& entries, uint64_t count, const char* name) {
		uint64_t hits  = 0;
		uint32_t shown = 0;
		for (uint64_t i = 0; i < std::min<uint64_t>(count, SyncRingSize); i++) {
			const auto& e = entries[(count - 1 - i) % SyncRingSize];
			if (e.vaddr >= page + TRACKER_PAGE_SIZE || page >= e.vaddr + e.size) {
				continue;
			}
			hits++;
			if (shown >= 6u) {
				continue;
			}
			shown++;
			std::fprintf(stderr,
			             "[kyty-args-sync]         %s: sequence %" PRIu64 " %s 0x%016" PRIx64 " bytes %" PRIu64
			             " queue %d op %s sequence %" PRIu64 " shader 0x%016" PRIx64 "\n",
			             name, e.seq, SyncRingKinds[e.kind < 15 ? e.kind : 0], e.vaddr, e.size, e.queue,
			             e.op_kind < 6 ? SyncOpKinds[e.op_kind] : "none", e.op_seq, e.shader);
		}
		std::fprintf(stderr, "[kyty-args-sync]       %s on the %s page 0x%016" PRIx64 ": %" PRIu64 " of the last %" PRIu64 "\n", name,
		             what, page, hits, std::min<uint64_t>(count, SyncRingSize));
	};
	ring(r.writes, r.write_count, "GPU writes");
	ring(r.uploads, r.upload_count, "uploads");
	ring(r.cp, r.cp_count, "CP writes");
	ring(r.life, r.life_count, "buffer lifecycle");
}

void SyncPrintIbOf(const uint32_t* packet, int32_t queue, uint64_t before_seq) {
	if (packet == nullptr) {
		return;
	}
	const auto  address = reinterpret_cast<uint64_t>(packet);
	const auto& p       = g_sync_packets;
	for (uint64_t i = 0; i < std::min<uint64_t>(p.ib_count, p.ibs.size()); i++) {
		const auto& e = p.ibs[(p.ib_count - 1 - i) % p.ibs.size()];
		if (e.queue != queue || e.seq > before_seq || address < e.address || address >= e.address + e.size) {
			continue;
		}
		char marks[128];
		SyncMarksText(e.marks, marks, sizeof(marks));
		std::fprintf(stderr,
		             "[kyty-args-sync]       its command buffer: %s, 0x%016" PRIx64 " bytes %" PRIu64 " (fetch sequence %" PRIu64
		             "), marks over it at the fetch: %s\n",
		             SyncIbHow[e.how & 3u], e.address, e.size, e.seq, marks);
		return;
	}
	std::fprintf(stderr, "[kyty-args-sync]       its command buffer: no fetch of this queue covering the packet in the last %" PRIu64 "\n",
	             std::min<uint64_t>(p.ib_count, p.ibs.size()));
}

}

void DiagArgsSyncSetCpPacket(const uint32_t* header) {
	g_sync_cp_packet = header;
}

void DiagArgsSyncSetSubmission(uint32_t queue, uint64_t order) {
	g_sync_sub_queue = static_cast<int32_t>(queue);
	g_sync_sub_order = order;
}

void DiagArgsSyncNoteHoldCheck(uint32_t queue, uint32_t why, uint64_t order, uint64_t before, uint64_t done) {
	auto& last = g_sync_hold_last[queue % 64u];
	last.order.store(order, std::memory_order_relaxed);
	last.before.store(before, std::memory_order_relaxed);
	last.done.store(done, std::memory_order_relaxed);
	last.why.store(why, std::memory_order_relaxed);
	g_sync_hold_why[why & 3u].fetch_add(1, std::memory_order_relaxed);
}

void DiagArgsSyncNoteSetBase(uint32_t queue, const uint32_t* body, uint32_t shader_type) {
	if (!ArgsSyncDiag() || shader_type != 0u || body == nullptr) {
		return;
	}
	auto*            tracker = g_sync_ring.tracker;
	SyncSetBaseEntry e {};
	e.packet = body - 1;
	std::memcpy(e.words.data(), e.packet, sizeof(uint32_t) * 4u);
	e.marks = tracker != nullptr ? SyncMarks(tracker->ProbeArgsPages(reinterpret_cast<uint64_t>(e.packet), sizeof(uint32_t) * 4u)) : 0;
	std::scoped_lock lock(g_sync_ring.mutex);
	e.seq                        = ++g_sync_ring.seq;
	g_sync_set_base[queue % 64u] = e;
}

void DiagArgsSyncNoteDrawPacket(const uint32_t* header, uint32_t dw) {
	if (!ArgsSyncDiag()) {
		return;
	}
	g_sync_draw_packet    = header;
	g_sync_draw_packet_dw = std::min<uint32_t>(dw, 10u);
}

void DiagArgsSyncNoteDrawParse(uint32_t queue, uint64_t args_base, uint32_t data_offset, uint32_t count, uint32_t stride,
                               bool run) {
	if (!ArgsSyncDiag()) {
		return;
	}
	const auto*         header  = std::exchange(g_sync_draw_packet, nullptr);
	const auto          dw      = std::exchange(g_sync_draw_packet_dw, 0u);
	auto*               tracker = g_sync_ring.tracker;
	SyncDrawPacketEntry e {};
	e.args_base   = args_base;
	e.packet      = header;
	e.queue       = static_cast<int32_t>(queue);
	e.dw          = header != nullptr ? dw : 0u;
	e.data_offset = data_offset;
	e.count       = count;
	e.stride      = stride;
	e.run         = run ? 1u : 0u;
	if (header != nullptr && e.dw != 0) {
		std::memcpy(e.words.data(), header, sizeof(uint32_t) * e.dw);
		e.marks = tracker != nullptr ? SyncMarks(tracker->ProbeArgsPages(reinterpret_cast<uint64_t>(header), sizeof(uint32_t) * e.dw))
		                             : 0;
	}
	std::scoped_lock lock(g_sync_ring.mutex);
	e.set_base = g_sync_set_base[queue % 64u];
	e.seq      = ++g_sync_ring.seq;
	auto& p    = g_sync_packets;
	p.draws[p.draw_count++ % p.draws.size()] = e;
}

void DiagArgsSyncNoteIbFetch(uint32_t queue, uint64_t address, uint64_t size, uint8_t how) {
	if (!ArgsSyncDiag() || size == 0) {
		return;
	}
	auto*            tracker = g_sync_ring.tracker;
	SyncIbFetchEntry e {};
	e.address = address;
	e.size    = size;
	e.queue   = static_cast<int32_t>(queue);
	e.how     = static_cast<uint8_t>(how & 3u);
	e.marks   = tracker != nullptr ? SyncMarks(tracker->ProbeArgsPages(address, size)) : 0;
	std::scoped_lock lock(g_sync_ring.mutex);
	e.seq  = ++g_sync_ring.seq;
	auto& p = g_sync_packets;
	p.ibs[p.ib_count++ % p.ibs.size()] = e;
	p.ib_by_how[e.how]++;
}

void DiagArgsSyncDumpPackets(uint64_t address, const char* why) {
	if (!ArgsSyncDiag()) {
		return;
	}
	std::vector<SyncDrawPacketEntry> draws;
	std::vector<SyncWaitEntry>       waits;
	bool                             covered    = false;
	uint64_t                         draw_count = 0;
	{
		std::scoped_lock lock(g_sync_ring.mutex);
		const auto&      p    = g_sync_packets;
		const auto       kept = std::min<uint64_t>(p.draw_count, p.draws.size());
		draw_count            = p.draw_count;
		for (uint64_t i = 0; i < kept && draws.size() < 4u; i++) {
			const auto& e  = p.draws[(p.draw_count - 1 - i) % p.draws.size()];
			const auto  lo = e.args_base + e.data_offset;
			const auto  hi = lo + uint64_t {std::max(e.count, 1u)} * uint64_t {std::max(e.stride, 4u)};
			if (address >= lo && address < hi) {
				draws.push_back(e);
			}
		}
		covered = !draws.empty();
		for (uint64_t i = 0; i < kept && draws.empty(); i++) {
			const auto& e = p.draws[(p.draw_count - 1 - i) % p.draws.size()];
			if (e.queue == 0) {
				draws.push_back(e);
			}
		}
		const auto& w = g_sync_waits;
		for (uint64_t i = 0; i < std::min<uint64_t>(w.count, w.ring.size()) && waits.size() < 8u; i++) {
			const auto& e = w.ring[(w.count - 1 - i) % w.ring.size()];
			if (e.queue == 0 || e.reference != 0 || e.packet == nullptr) {
				continue;
			}
			bool seen = false;
			for (const auto& x: waits) {
				seen = seen || x.packet == e.packet;
			}
			if (!seen) {
				waits.push_back(e);
			}
		}
	}
	std::vector<SyncPacketNow> draw_now;
	std::vector<SyncPacketNow> base_now;
	std::vector<SyncPacketNow> wait_now;
	for (const auto& e: draws) {
		draw_now.push_back(SyncReadPacketNow(e.packet, e.dw));
		base_now.push_back(SyncReadPacketNow(e.set_base.packet, e.set_base.packet != nullptr ? 4u : 0u));
	}
	for (const auto& e: waits) {
		wait_now.push_back(SyncReadPacketNow(e.packet - 1, 7u));
	}
	std::scoped_lock lock(g_sync_ring.mutex);
	const auto&      p = g_sync_packets;
	std::fprintf(stderr,
	             "[kyty-args-sync] %s: command packets for args 0x%016" PRIx64 ": %s (%" PRIu64
	             " indirect draws parsed so far); command buffer fetches so far: %" PRIu64 " IBs without PFP_SYNC_ME, %" PRIu64
	             " IBs checked after PFP_SYNC_ME, %" PRIu64 " IBs downloaded as GPU-modified, %" PRIu64 " top-level submissions\n",
	             why, address,
	             covered ? "parsed draws whose args range covers it, newest first"
	                     : (draws.empty() ? "no parsed draw in the ring" : "NO parsed draw in the ring covers it; the newest queue-0 draw instead"),
	             draw_count, p.ib_by_how[0], p.ib_by_how[1], p.ib_by_how[2], p.ib_by_how[3]);
	std::fprintf(stderr,
	             "[kyty-args-sync]   ACE hold checks so far (KYTY_DIAG_ACE_HOLD %s): %" PRIu64 " %s; %" PRIu64 " %s; %" PRIu64 " %s; %" PRIu64
	             " %s\n",
	             DiagAceHoldText(), g_sync_hold_why[0].load(std::memory_order_relaxed), SyncHoldWhy[0],
	             g_sync_hold_why[1].load(std::memory_order_relaxed), SyncHoldWhy[1], g_sync_hold_why[2].load(std::memory_order_relaxed),
	             SyncHoldWhy[2], g_sync_hold_why[3].load(std::memory_order_relaxed), SyncHoldWhy[3]);
	for (size_t i = 0; i < draws.size(); i++) {
		const auto& e = draws[i];
		char        then[128];
		char        now[128];
		SyncMarksText(e.marks, then, sizeof(then));
		SyncMarksText(draw_now[i].marks, now, sizeof(now));
		const bool same = draw_now[i].read && std::equal(e.words.begin(), e.words.begin() + e.dw, draw_now[i].words.begin());
		std::fprintf(stderr,
		             "[kyty-args-sync]     draw packet: queue %d parse sequence %" PRIu64 " at 0x%016" PRIx64 " (%u dw): args base 0x%016" PRIx64
		             " data offset 0x%x count %u stride %u run %u\n",
		             e.queue, e.seq, reinterpret_cast<uint64_t>(e.packet), e.dw, e.args_base, e.data_offset, e.count, e.stride,
		             static_cast<unsigned>(e.run));
		if (e.packet != nullptr) {
			SyncPrintWords("as parsed", e.words.data(), e.dw);
			SyncPrintWords(draw_now[i].read ? (same ? "guest backing now (backing read, no fault): EQUAL"
			                                        : "guest backing now (backing read, no fault): DIFFERENT")
			                                : "guest backing now: READ FAILED",
			               draw_now[i].words.data(), e.dw);
			std::fprintf(stderr, "[kyty-args-sync]       packet page marks at parse: %s; now: %s\n", then, now);
			SyncPrintIbOf(e.packet, e.queue, e.seq);
			SyncPrintPageRings(reinterpret_cast<uint64_t>(e.packet), "draw packet");
		}
		const auto& b = e.set_base;
		if (b.packet == nullptr) {
			std::fprintf(stderr, "[kyty-args-sync]     SET_BASE: none parsed on queue %d while the diag ran\n", e.queue);
			continue;
		}
		const uint64_t base_parsed = (uint64_t {b.words[2]} & ~0x7ull) | (uint64_t {b.words[3] & 0xffffu} << 32u);
		const bool base_same   = base_now[i].read && std::equal(b.words.begin(), b.words.end(), base_now[i].words.begin());
		SyncMarksText(b.marks, then, sizeof(then));
		SyncMarksText(base_now[i].marks, now, sizeof(now));
		std::fprintf(stderr,
		             "[kyty-args-sync]     SET_BASE packet: parse sequence %" PRIu64 " at 0x%016" PRIx64 ", base from it 0x%016" PRIx64
		             " (%s the draw's args base)\n",
		             b.seq, reinterpret_cast<uint64_t>(b.packet), base_parsed, base_parsed == e.args_base ? "equal to" : "DIFFERENT from");
		SyncPrintWords("as parsed", b.words.data(), 4u);
		SyncPrintWords(base_now[i].read ? (base_same ? "guest backing now (backing read, no fault): EQUAL"
		                                             : "guest backing now (backing read, no fault): DIFFERENT")
		                                : "guest backing now: READ FAILED",
		               base_now[i].words.data(), 4u);
		std::fprintf(stderr, "[kyty-args-sync]       packet page marks at parse: %s; now: %s\n", then, now);
		SyncPrintIbOf(b.packet, e.queue, b.seq);
		if ((reinterpret_cast<uint64_t>(b.packet) & ~(TRACKER_PAGE_SIZE - 1)) !=
		    (reinterpret_cast<uint64_t>(e.packet) & ~(TRACKER_PAGE_SIZE - 1))) {
			SyncPrintPageRings(reinterpret_cast<uint64_t>(b.packet), "SET_BASE packet");
		} else {
			std::fprintf(stderr, "[kyty-args-sync]       (same page as the draw packet)\n");
		}
	}
	std::fprintf(stderr, "[kyty-args-sync]   ACE WAIT_REG_MEM packets with reference 0, newest first (%zu distinct):\n", waits.size());
	for (size_t i = 0; i < waits.size(); i++) {
		const auto& e = waits[i];
		char        now[128];
		SyncMarksText(wait_now[i].marks, now, sizeof(now));
		std::fprintf(stderr,
		             "[kyty-args-sync]     wait: sequence %" PRIu64 " queue %d packet 0x%016" PRIx64 " address 0x%016" PRIx64
		             " function %u met %u; ref field now 0x%08x%s; packet page marks now: %s\n",
		             e.seq, e.queue, reinterpret_cast<uint64_t>(e.packet - 1), e.address, e.function, static_cast<unsigned>(e.met),
		             wait_now[i].words[4], wait_now[i].read ? "" : " (READ FAILED)", now);
		SyncPrintWords("packet now", wait_now[i].words.data(), 7u);
		SyncPrintIbOf(e.packet - 1, e.queue, e.seq);
		SyncPrintPageRings(reinterpret_cast<uint64_t>(e.packet - 1), "wait packet");
	}
}

void DiagArgsSyncDumpFor(uint64_t address, const char* why) {
	if (!ArgsSyncDiag()) {
		return;
	}
	const auto page    = address & ~(TRACKER_PAGE_SIZE - 1);
	auto*      tracker = g_sync_ring.tracker;
	const auto probe   = tracker != nullptr ? tracker->ProbeArgsPages(page, TRACKER_PAGE_SIZE) : RegionManager::ArgsPageProbe {};
	{
		uint32_t   guest[5] {};
		const bool read    = LibKernel::Memory::TryReadBacking(address, guest, sizeof(guest));
		const auto history = tracker != nullptr ? tracker->ProbePageHistory(page) : RegionManager::PageHistory {};
		std::fprintf(stderr,
		             "[kyty-args-sync] %s: guest memory at args 0x%016" PRIx64 " (backing read, no fault, no download)%s: %08x %08x %08x %08x %08x"
		             " (index count %u, instances %u, first index %u, vertex offset %d, first instance %u)\n",
		             why, address, read ? "" : " READ FAILED", guest[0], guest[1], guest[2], guest[3], guest[4], guest[0], guest[1],
		             guest[2], static_cast<int32_t>(guest[3]), guest[4]);
		std::fprintf(stderr,
		             "[kyty-args-sync]   page history: cpu_written consumers 0x%x; busy record %s (faulted %u, writes %u, seen %u, cleaned %u)\n",
		             history.cpu_written, history.busy != 0 ? "present" : "absent", static_cast<unsigned>(history.faulted),
		             static_cast<unsigned>(history.writes), history.seen, history.cleaned);
	}
	std::scoped_lock lock(g_sync_ring.mutex);
	auto&            r = g_sync_ring;
	std::fprintf(stderr,
	             "[kyty-args-sync] %s: args 0x%016" PRIx64 "; its 4 KiB page now: cpu_dirty %d gpu_dirty %d gpu_written %d "
	             "gpu_noted %d block_opened %d; sequence now %" PRIu64 ", %" PRIu64 " uploads and %" PRIu64
	             " GPU writes recorded so far\n",
	             why, address, probe.cpu_dirty ? 1 : 0, probe.gpu_dirty ? 1 : 0, probe.gpu_written ? 1 : 0,
	             probe.gpu_noted ? 1 : 0, probe.block_opened ? 1 : 0, r.seq, r.upload_count, r.write_count);
	const auto print_ring = [&](const std::array<SyncRingEntry, SyncRingSize>& ring, uint64_t count, const char* what) {
		const uint64_t kept  = std::min<uint64_t>(count, SyncRingSize);
		uint64_t       hits  = 0;
		uint32_t       shown = 0;
		for (uint64_t i = 0; i < kept; i++) {
			const auto& e = ring[(count - 1 - i) % SyncRingSize];
			if (e.vaddr >= page + TRACKER_PAGE_SIZE || page >= e.vaddr + e.size) {
				continue;
			}
			hits++;
			if (shown >= 24) {
				continue;
			}
			shown++;
			char where[192];
			SyncCallerText(e.caller, where, sizeof(where));
			std::fprintf(stderr,
			             "[kyty-args-sync]   %s: sequence %" PRIu64 " %s 0x%016" PRIx64 " bytes %" PRIu64
			             " marks after (cpu_dirty %d gpu_dirty %d gpu_written %d gpu_noted %d block_opened %d) caller %s;"
			             " queue %d op %s sequence %" PRIu64 " submit %" PRIu64 " guest_sync %u shader 0x%016" PRIx64
			             " path %u V# %08x %08x %08x %08x\n",
			             what, e.seq, SyncRingKinds[e.kind < 15 ? e.kind : 0], e.vaddr, e.size, (e.marks & 1u) != 0 ? 1 : 0,
			             (e.marks & 2u) != 0 ? 1 : 0, (e.marks & 4u) != 0 ? 1 : 0, (e.marks & 8u) != 0 ? 1 : 0,
			             (e.marks & 16u) != 0 ? 1 : 0, where, e.queue, e.op_kind < 6 ? SyncOpKinds[e.op_kind] : "none",
			             e.op_seq, e.submit, static_cast<unsigned>(e.guest_sync), e.shader, static_cast<unsigned>(e.path),
			             e.words[0], e.words[1], e.words[2], e.words[3]);
			if (e.kind == 5) {
				std::fprintf(stderr,
				             "[kyty-args-sync]     the draw: base 0x%016" PRIx64 " data offset 0x%x count %u stride %u run %u"
				             " (the command buffer's first data offset now 0x%x, last 0x%x)\n",
				             e.ind_base, e.ind_offset, e.ind_count, e.ind_stride, static_cast<unsigned>(e.ind_run), e.ind_run0,
				             e.ind_runl);
			}
		}
		std::fprintf(stderr, "[kyty-args-sync]   %s covering that page: %" PRIu64 " of the last %" PRIu64 " (newest first above)\n",
		             what, hits, kept);
	};
	print_ring(r.uploads, r.upload_count, "uploads");
	print_ring(r.writes, r.write_count, "GPU writes");
	print_ring(r.reads, r.read_count, "indirect args reads");
	print_ring(r.cp, r.cp_count, "CP writes");
	if (g_cull.ring != nullptr && g_cull.scheduler != nullptr) {
		g_cull.ring->Invalidate(0, g_cull.ring->Size());
		const auto mapped = g_cull.ring->Mapped();
		const auto dword  = [&](uint32_t index, uint64_t offset) {
			uint32_t value = 0;
			std::memcpy(&value, mapped.data() + uint64_t {index} * CullSlotBytes + offset, sizeof(value));
			return value;
		};
		std::vector<uint32_t> order;
		for (uint32_t i = 0; i < CullSlots; ++i) {
			if (g_cull.slots[i].seq != 0) {
				order.push_back(i);
			}
		}
		std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return g_cull.slots[a].seq > g_cull.slots[b].seq; });
		const uint64_t per_object = CullObjectBytes();
		const auto     covers     = [&](const CullSlot& s) {
			for (uint32_t k = 0; k < s.bindings; ++k) {
				if (s.b[k].written != 0 && address >= s.b[k].address && address < s.b[k].address + s.b[k].size) {
					return true;
				}
			}
			return false;
		};
		const auto print_slot = [&](uint32_t i, const char* why) {
			const auto&   s  = g_cull.slots[i];
			const CullCp* cp = nullptr;
			for (uint64_t j = 0; j < std::min<uint64_t>(g_cull_cp_count, g_cull_cp.size()); ++j) {
				const auto& x = g_cull_cp[(g_cull_cp_count - 1 - j) % g_cull_cp.size()];
				if (x.op_seq == s.op_seq) {
					cp = &x;
					break;
				}
			}
			std::fprintf(stderr,
			             "[kyty-args-sync]     cull (%s): sequence %" PRIu64 " queue %d op sequence %" PRIu64 " shader 0x%016" PRIx64
			             " %s, tick %s; args 0x%016" PRIx64 " (CP: dispatch base 0x%016" PRIx64 " + offset 0x%" PRIx64 ")\n",
			             why, s.seq, s.queue, s.op_seq, s.shader, CullKinds[s.kind < 4 ? s.kind : 0],
			             g_cull.scheduler->IsFree(s.tick) ? "done" : "NOT done (copies may be unwritten)", s.args_guest,
			             cp != nullptr ? cp->base : 0u, cp != nullptr && cp->base != 0 ? s.args_guest - cp->base : 0u);
			std::fprintf(stderr,
			             "[kyty-args-sync]       mode 0x%x: USE_THREAD_DIMENSIONS (0x20) %s; program has the dispatch-thread dword: %s;"
			             " input_info.dispatch_thread_dimensions: %s; input_info from TakeResolvedDispatch: %s\n",
			             s.mode, (s.mode & 0x20u) != 0 ? "set" : "clear", (s.flags & 2u) != 0 ? "yes" : "no",
			             (s.flags & 4u) != 0 ? "yes" : "no", (s.flags & 1u) != 0 ? "yes" : "no");
			if (s.has_args != 0) {
				std::fprintf(stderr, "[kyty-args-sync]       %s as the GPU read them: %u %u %u\n",
				             s.kind == 3 ? "thread counts" : "group counts", dword(i, 0), dword(i, 4), dword(i, 8));
			}
			if (s.has_groups != 0) {
				std::fprintf(stderr, "[kyty-args-sync]       groups launched (from the thread pass): %u %u %u\n", dword(i, 16),
				             dword(i, 20), dword(i, 24));
			}
			if (s.kind == 1) {
				std::fprintf(stderr, "[kyty-args-sync]       direct groups: %u %u %u\n", s.direct_groups[0], s.direct_groups[1],
				             s.direct_groups[2]);
			}
			if (const auto& run = DiagRunShader(); run.hash != 0 && s.shader == run.hash) {
				const uint64_t RunThreads = run.threads;
				const uint64_t RunStride  = run.stride;
				const uint64_t     groups =
				    s.kind == 1 ? uint64_t {s.direct_groups[0]} * std::max(s.direct_groups[1], 1u) * std::max(s.direct_groups[2], 1u)
				                : (s.has_args != 0 ? uint64_t {dword(i, 0)} * std::max(dword(i, 4), 1u) * std::max(dword(i, 8), 1u) : 0u);
				bool reported = false;
				for (uint32_t q = 0; q < s.bindings; ++q) {
					if (s.b[q].written != 0) {
						continue;
					}
					const uint64_t first = dword(i, CullPreBase + uint64_t {q} * CullHeadBytes);
					const uint64_t last  = groups != 0 ? first + RunThreads * groups - 1u : first;
					for (uint32_t t = 0; t < s.bindings; ++t) {
						const auto& w = s.b[t];
						if (w.written == 0 || address < w.address || address >= w.address + w.size || (address - w.address) % RunStride != 0) {
							continue;
						}
						const uint64_t element = (address - w.address) / RunStride;
						reported               = true;
						std::fprintf(stderr,
						             "[kyty-args-sync]       run shader: read binding %u dword 0 = %" PRIu64 ", %" PRIu64
						             " groups x %" PRIu64 " threads -> elements %" PRIu64 "..%" PRIu64 " of written binding %u; the args record is element %" PRIu64
						             ": %s\n",
						             q, first, groups, RunThreads, first, last, t, element,
						             groups != 0 && element >= first && element <= last ? "INSIDE the run" : "outside the run");
					}
				}
				if (!reported) {
					std::fprintf(stderr,
					             "[kyty-args-sync]       run shader: no read binding with a written binding at a %" PRIu64
					             "-byte step from the args record (the constant may come from user data rather than a buffer)\n",
					             RunStride);
				}
			}
			for (uint32_t k = 0; k < s.bindings; ++k) {
				const uint64_t head = CullPreBase + uint64_t {k} * CullHeadBytes;
				std::fprintf(stderr,
				             "[kyty-args-sync]       binding %u: 0x%016" PRIx64 " bytes %" PRIu64 " %s; first dwords before: %08x %08x %08x %08x",
				             k, s.b[k].address, s.b[k].size, s.b[k].written != 0 ? "written" : "read", dword(i, head),
				             dword(i, head + 4), dword(i, head + 8), dword(i, head + 12));
				if (s.b[k].written != 0 && s.post != 0) {
					const uint64_t tail = 64u + uint64_t {k} * 64u;
					std::fprintf(stderr, "; after: %08x %08x %08x %08x", dword(i, tail), dword(i, tail + 4), dword(i, tail + 8),
					             dword(i, tail + 12));
				}
				std::fprintf(stderr, "\n");
				const uint32_t counter = dword(i, head);
				const uint64_t window  = std::min<uint64_t>(s.b[k].size, CullHeadBytes) / 4u;
				if (s.b[k].written == 0 && counter > 0 && uint64_t {counter} + 3u < window) {
					std::fprintf(stderr, "[kyty-args-sync]         as a list: counter %u; entries %u..%u:", counter,
					             counter > 2u ? counter - 2u : 1u, counter + 2u);
					for (uint32_t e = counter > 2u ? counter - 2u : 1u; e <= counter + 2u; ++e) {
						std::fprintf(stderr, " %u", dword(i, head + uint64_t {e} * 4u));
					}
					std::fprintf(stderr, " (entries past the counter are stale tail)\n");
				}
				if (address >= s.b[k].address && address + 20u <= s.b[k].address + std::min<uint64_t>(s.b[k].size, CullHeadBytes)) {
					const uint64_t at = address - s.b[k].address;
					std::fprintf(stderr, "[kyty-args-sync]         the args record at 0x%016" PRIx64 " before: %08x %08x %08x %08x %08x",
					             address, dword(i, head + at), dword(i, head + at + 4), dword(i, head + at + 8),
					             dword(i, head + at + 12), dword(i, head + at + 16));
					if (s.post != 0 && s.post_head[k] >= 0) {
						const uint64_t ph = CullPostBase + uint64_t {static_cast<uint8_t>(s.post_head[k])} * CullHeadBytes + at;
						std::fprintf(stderr, "; after: %08x %08x %08x %08x %08x", dword(i, ph), dword(i, ph + 4), dword(i, ph + 8),
						             dword(i, ph + 12), dword(i, ph + 16));
					}
					std::fprintf(stderr, "\n");
				}
			}
			for (uint32_t k = 0; k < s.bindings; ++k) {
				const auto& t = s.b[k];
				if (t.written == 0 || per_object == 0 || address < t.address || address >= t.address + t.size ||
				    (address - t.address) % per_object != 0) {
					continue;
				}
				const uint64_t object = (address - t.address) / per_object;
				std::fprintf(stderr,
				             "[kyty-args-sync]       OVERRUN CHECK: args 0x%016" PRIx64 " = binding %u base + %" PRIu64 " x %" PRIu64
				             ": object %" PRIu64 "; groups %u; list entries equal to it:",
				             address, k, object, per_object, object,
				             s.has_groups != 0 ? dword(i, 16) : (s.has_args != 0 ? dword(i, 0) : s.direct_groups[0]));
				for (uint32_t q = 0; q < s.bindings; ++q) {
					if (s.b[q].written != 0) {
						continue;
					}
					const uint64_t head   = CullPreBase + uint64_t {q} * CullHeadBytes;
					const uint64_t window = std::min<uint64_t>(s.b[q].size, CullHeadBytes) / 4u;
					for (uint64_t e = 1; e < window; ++e) {
						if (dword(i, head + e * 4u) == object) {
							std::fprintf(stderr, " binding %u entry %" PRIu64 " (counter %u)", q, e, dword(i, head));
						}
					}
				}
				std::fprintf(stderr, "\n");
			}
		};
		std::fprintf(stderr,
		             "[kyty-args-sync]   culling, builder and args-writer dispatches: %zu of %u slots used; object stride %" PRIu64
		             " bytes%s; the newest 24, then every older slot whose written binding covers the args address:\n",
		             order.size(), CullSlots, per_object,
		             per_object == 0 ? " (overrun check off: KYTY_DIAG_CULL_SLOTS=<slots>:<record bytes> not set)" : "");
		uint32_t covering = 0;
		for (size_t n = 0; n < order.size(); ++n) {
			const bool hit = covers(g_cull.slots[order[n]]);
			covering += hit ? 1u : 0u;
			if (n < 24u) {
				print_slot(order[n], hit ? "newest, covers the args" : "newest");
			} else if (hit && covering <= 48u) {
				print_slot(order[n], "covers the args");
			}
		}
		std::fprintf(stderr, "[kyty-args-sync]   slots whose written binding covers the args address: %u\n", covering);
	}
	print_ring(r.life, r.life_count, "buffer lifecycle (created, joined, deleted, gaps, GC; words[0] of a created entry = joined count)");
	std::fprintf(stderr,
	             "[kyty-args-sync]   dispatch skips on a CPU-read zero count: %" PRIu64 " so far, %" PRIu64
	             " with GPU marks on their args page; the newest 24 (queue, op sequence, args, counts read, CS address):\n",
	             r.skip_count, r.skip_marked);
	uint64_t window_lo = UINT64_MAX;
	uint64_t window_hi = 0;
	uint64_t producer_hi = 0;
	{
		uint32_t taken = 0;
		for (uint64_t i = 0; i < std::min<uint64_t>(r.write_count, SyncRingSize) && taken < 16u; i++) {
			const auto& e = r.writes[(r.write_count - 1 - i) % SyncRingSize];
			if (e.vaddr >= page + TRACKER_PAGE_SIZE || page >= e.vaddr + e.size || e.op_seq == 0) {
				continue;
			}
			taken++;
			window_lo = std::min(window_lo, e.op_seq);
			producer_hi = std::max(producer_hi, e.op_seq);
		}
		for (uint64_t i = 0; i < std::min<uint64_t>(r.read_count, SyncRingSize); i++) {
			const auto& e = r.reads[(r.read_count - 1 - i) % SyncRingSize];
			if (e.vaddr >= page + TRACKER_PAGE_SIZE || page >= e.vaddr + e.size) {
				continue;
			}
			window_hi = e.op_seq;
			break;
		}
	}
	if (window_lo == UINT64_MAX) {
		window_lo = 0;
	} else {
		window_lo = window_lo > 200u ? window_lo - 200u : 0u;
	}
	if (window_hi == 0) {
		window_hi = UINT64_MAX;
	}
	const auto& w = g_sync_waits;
	std::fprintf(stderr,
	             "[kyty-args-sync]   WAIT_REG_MEM evaluations with the CP's emitted counter in [%" PRIu64 ", %" PRIu64
	             "] (200 before the oldest of the newest 16 GPU writes on the page, up to the newest args read), newest first; %" PRIu64
	             " waits recorded so far:\n",
	             window_lo, window_hi, w.count);
	for (int pass = 0; pass < 2; ++pass) {
	if (pass == 1 && producer_hi == 0) {
		break;
	}
	const uint64_t pass_lo = pass == 0 ? window_lo : producer_hi + 1u;
	const uint64_t pass_hi = pass == 0 ? (producer_hi != 0 ? producer_hi : window_hi) : window_hi;
	std::fprintf(stderr, "[kyty-args-sync]   %s [%" PRIu64 ", %" PRIu64 "]:\n",
	             pass == 0 ? "waits from 200 before the producers up to the last producer write"
	                       : "waits after the last producer write up to the newest args read",
	             pass_lo, pass_hi);
	uint32_t waits_shown = 0;
	for (uint64_t i = 0; i < std::min<uint64_t>(w.count, w.ring.size()) && waits_shown < (pass == 0 ? 48u : 16u); i++) {
		const auto& e = w.ring[(w.count - 1 - i) % w.ring.size()];
		if (e.emitted < pass_lo || e.emitted > pass_hi) {
			continue;
		}
		waits_shown++;
		std::fprintf(stderr,
		             "[kyty-args-sync]     wait: sequence %" PRIu64 " queue %d emitted %" PRIu64 " address 0x%016" PRIx64
		             " size %u value 0x%" PRIx64 " ref 0x%" PRIx64 " mask 0x%" PRIx64 " function %u met %u (unmet retries before: %" PRIu64
		             "), value from %s (write sequence %" PRIu64 "), page marks (gpu_dirty %d gpu_written %d gpu_noted %d)\n",
		             e.seq, e.queue, e.emitted, e.address, e.size, e.value, e.reference, e.mask, e.function,
		             static_cast<unsigned>(e.met), e.repeats, SyncDecisionSources[e.source < 5 ? e.source : 0], e.write_seq,
		             (e.marks & 2u) != 0 ? 1 : 0, (e.marks & 4u) != 0 ? 1 : 0, (e.marks & 8u) != 0 ? 1 : 0);
		if (e.packet != nullptr) {
			uint64_t   ref_now  = 0;
			const bool ref_read = LibKernel::Memory::TryReadBacking(reinterpret_cast<uint64_t>(e.packet + 3), &ref_now,
			                                                       e.size == 8u ? 8u : 4u);
			std::fprintf(stderr, "[kyty-args-sync]       ref in the command buffer now 0x%" PRIx64 " (%s the parsed ref)\n",
			             ref_now,
			             !ref_read ? "READ FAILED, not compared with" : ref_now == e.reference ? "equal to" : "DIFFERENT from");
		}
		if (e.hold_why != 0xffu) {
			std::fprintf(stderr,
			             "[kyty-args-sync]       hold check that admitted its submission: %s (submission order %" PRIu64
			             ", last q0 submission enqueued before it %" PRIu64 ", q0 done up to %" PRIu64 ")\n",
			             SyncHoldWhy[e.hold_why & 3u], e.hold_order, e.hold_before, e.hold_done);
		}
		if (e.source == 1) {
			const SyncRingEntry* cw = nullptr;
			for (uint64_t j = 0; j < std::min<uint64_t>(r.cp_count, SyncRingSize); j++) {
				const auto& x = r.cp[(r.cp_count - 1 - j) % SyncRingSize];
				if (x.op_seq == e.write_seq && x.vaddr <= e.address && e.address < x.vaddr + x.size) {
					cw = &x;
					break;
				}
			}
			if (cw == nullptr) {
				std::fprintf(stderr, "[kyty-args-sync]       the parsed CP write it used (write sequence %" PRIu64 "): not in the ring\n",
				             e.write_seq);
			} else {
				const uint32_t op   = (cw->pkt_header >> 8u) & 0xffu;
				const char*    name = op == 0x49u ? "RELEASE_MEM" : op == 0x37u ? "WRITE_DATA" : op == 0x47u ? "EVENT_WRITE_EOP"
				                    : op == 0x48u ? "EVENT_WRITE_EOS" : op == 0x50u ? "DMA_DATA" : op == 0x40u ? "COPY_DATA"
				                    : op == 0x1eu ? "ATOMIC_MEM" : "other";
				char fields[160] = "";
				if (op == 0x49u) {
					std::snprintf(fields, sizeof(fields), "; event_type 0x%x event_index %u data_sel %u int_sel %u dst_sel %u",
					              cw->pkt_body[0] & 0x3fu, (cw->pkt_body[0] >> 8u) & 0xfu, cw->pkt_body[1] >> 29u,
					              (cw->pkt_body[1] >> 24u) & 0x7u, (cw->pkt_body[1] >> 16u) & 0x3u);
				} else if (op == 0x37u) {
					std::snprintf(fields, sizeof(fields), "; dst_sel %u wr_confirm %u engine_sel %u", (cw->pkt_body[0] >> 8u) & 0xfu,
					              (cw->pkt_body[0] >> 20u) & 0x1u, cw->pkt_body[0] >> 30u);
				}
				const char* state = cw->sub_queue != 0      ? "not a queue-0 write"
				                    : e.hold_why == 0xffu    ? "q0 done order at admission unknown"
				                    : cw->sub_order > e.hold_done ? "its q0 submission was NOT fully parsed when this wait's submission was admitted"
				                                                  : "its q0 submission was fully parsed when this wait's submission was admitted";
				std::fprintf(stderr,
				             "[kyty-args-sync]       the parsed CP write it used: queue %d write sequence %" PRIu64 " at 0x%016" PRIx64
				             " from packet %s (opcode 0x%02x, header %08x, body %08x %08x %08x%s); submission order %" PRIu64
				             " on queue %d: %s\n",
				             cw->queue, cw->op_seq, cw->vaddr, name, op, cw->pkt_header, cw->pkt_body[0], cw->pkt_body[1],
				             cw->pkt_body[2], fields, cw->sub_order, cw->sub_queue, state);
			}
		}
		const auto last_before = [&](const std::array<SyncRingEntry, SyncRingSize>& ring, uint64_t count) -> const SyncRingEntry* {
			for (uint64_t j = 0; j < std::min<uint64_t>(count, SyncRingSize); j++) {
				const auto& x = ring[(count - 1 - j) % SyncRingSize];
				if (x.seq < e.seq && x.vaddr <= e.address && e.address < x.vaddr + x.size) {
					return &x;
				}
			}
			return nullptr;
		};
		if (const auto* x = last_before(r.cp, r.cp_count); x != nullptr) {
			std::fprintf(stderr,
			             "[kyty-args-sync]       last CP write to it before: queue %d op sequence %" PRIu64 " at 0x%016" PRIx64
			             " bytes %" PRIu64 " first dwords %08x %08x\n",
			             x->queue, x->op_seq, x->vaddr, x->size, x->words[0], x->words[1]);
		} else {
			std::fprintf(stderr, "[kyty-args-sync]       last CP write to it before: none in the ring\n");
		}
		if (const auto* x = last_before(r.writes, r.write_count); x != nullptr) {
			std::fprintf(stderr,
			             "[kyty-args-sync]       last GPU write over it before: %s queue %d op sequence %" PRIu64 " shader 0x%016" PRIx64
			             " range 0x%016" PRIx64 " bytes %" PRIu64 "\n",
			             SyncRingKinds[x->kind < 15 ? x->kind : 0], x->queue, x->op_seq, x->shader, x->vaddr, x->size);
		} else {
			std::fprintf(stderr, "[kyty-args-sync]       last GPU write over it before: none in the ring\n");
		}
	}
	}
	{
		const auto& dr = g_sync_decisions;
		uint64_t    stale_total = 0;
		for (uint64_t i = 0; i < std::min<uint64_t>(dr.count, dr.ring.size()); i++) {
			const auto& e = dr.ring[i];
			stale_total += e.source >= 3 && (e.marks & 14u) != 0 ? 1u : 0u;
		}
		std::fprintf(stderr,
		             "[kyty-args-sync]   COND_EXEC / branch decisions with the emitted counter up to %" PRIu64
		             " (the newest args read of the page), newest 32; %" PRIu64 " recorded so far, %" PRIu64
		             " in the ring taken from a guest-memory copy on a GPU-marked page:\n",
		             window_hi, dr.count, stale_total);
		uint32_t shown = 0;
		for (uint64_t i = 0; i < std::min<uint64_t>(dr.count, dr.ring.size()) && shown < 32u; i++) {
			const auto& e = dr.ring[(dr.count - 1 - i) % dr.ring.size()];
			if (e.emitted > window_hi) {
				continue;
			}
			shown++;
			const bool stale = e.source >= 3 && (e.marks & 14u) != 0;
			std::fprintf(stderr,
			             "[kyty-args-sync]     decision: sequence %" PRIu64 " queue %d emitted %" PRIu64 " %s at 0x%016" PRIx64
			             " size %u value 0x%" PRIx64 " (ref 0x%" PRIx64 " mask 0x%" PRIx64 " function %u) -> %s (%u dwords);"
			             " value from %s (write sequence %" PRIu64 "), page marks (gpu_dirty %d gpu_written %d gpu_noted %d)%s\n",
			             e.seq, e.queue, e.emitted, e.kind == 1 ? "COND_EXEC" : e.kind == 2 ? "branch" : e.kind == 3 ? "SET_PREDICATION op 1 (zpass)" : "SET_PREDICATION op 3 (memory)", e.address, e.size, e.value, e.reference,
			             e.mask, e.function,
			             e.kind == 1 ? (e.taken != 0 ? "executed the next" : "SKIPPED the next")
			             : e.kind == 2 ? (e.taken != 0 ? "then" : "else / none")
			                           : (e.taken != 0 ? "predicated packets run" : "predicated packets SKIPPED"),
			             e.exec_count, SyncDecisionSources[e.source < 5 ? e.source : 0], e.write_seq, (e.marks & 2u) != 0 ? 1 : 0,
			             (e.marks & 4u) != 0 ? 1 : 0, (e.marks & 8u) != 0 ? 1 : 0,
			             stale ? " STALE? (a guest-memory copy on a GPU-marked page)" : "");
		}
	}
	{
		uint64_t total = 0;
		for (const auto& n: g_predicated_skips) {
			total += n.load(std::memory_order_relaxed);
		}
		std::fprintf(stderr, "[kyty-args-sync]   predicated packets skipped so far: %" PRIu64 " (by opcode:", total);
		for (uint32_t op = 0; op < 256u; ++op) {
			const auto n = g_predicated_skips[op].load(std::memory_order_relaxed);
			if (n != 0) {
				std::fprintf(stderr, " 0x%02x x%" PRIu64, op, n);
			}
		}
		const char* hold = std::getenv("KYTY_DIAG_ACE_HOLD");
		std::fprintf(stderr, "); ACE submissions held behind earlier queue-0 work: %" PRIu64 " (KYTY_DIAG_ACE_HOLD %s)\n",
		             g_ace_holds.load(std::memory_order_relaxed), hold != nullptr && hold[0] == '1' ? "on" : "off");
	}
	for (uint64_t i = 0; i < std::min<uint64_t>({r.skip_count, SyncRingSize, 24u}); i++) {
		const auto& e = r.skips[(r.skip_count - 1 - i) % SyncRingSize];
		std::fprintf(stderr,
		             "[kyty-args-sync]     skip: sequence %" PRIu64 " queue %d op sequence %" PRIu64 " submit %" PRIu64
		             " args 0x%016" PRIx64 " counts %u %u %u marks (cpu_dirty %d gpu_dirty %d gpu_written %d gpu_noted %d) CS 0x%016" PRIx64 "\n",
		             e.seq, e.queue, e.op_seq, e.submit, e.vaddr, e.words[0], e.words[1], e.words[2], (e.marks & 1u) != 0 ? 1 : 0,
		             (e.marks & 2u) != 0 ? 1 : 0, (e.marks & 4u) != 0 ? 1 : 0, (e.marks & 8u) != 0 ? 1 : 0, e.shader);
	}
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	if (!is_written && !m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		if (is_texel_buffer && !NoImageStartsAt(vaddr)) {
			return SynchronizeBufferFromImage(buffer, vaddr, size);
		}
		return false;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	const MemoryTracker::UploadScope upload_scope;
	std::vector<std::pair<uint64_t, uint64_t>> marked;
	if (!is_written && SyncPreferGpu()) {
		m_memory_tracker.CollectGpuMarkedRuns(vaddr, size, marked);
	}
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    const auto emit = [&](uint64_t at, uint64_t run) noexcept {
			    copies.emplace_back(total_size, buffer.Offset(at), run);
			    total_size += run;
			    NoteUpload(at, run);
		    };
		    if (marked.empty()) {
			    emit(address, bytes);
		    } else {
			    SyncSkipMarked(marked, address, bytes, emit);
		    }
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); }, m_device_sync_pass);
	RecordBufferUpload(buffer, copies, source);
	if (!is_written && !copies.empty() && ArgsSyncDiag()) {
		ArgsSyncNoteUploads(m_memory_tracker, buffer, copies, 1, m_sync_caller);
	}
	if (is_texel_buffer && !is_written && !NoImageStartsAt(vaddr)) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

void BufferCache::RecordBufferUpload(Buffer& buffer, std::span<const vk::BufferCopy> copies,
                                     vk::Buffer source) {
	if (source && m_upload_batching && source == m_staging_buffer.Handle()) {
		m_upload_batch_buffers.push_back(
		    {buffer.Handle(), buffer.BufferDeviceAddress(), static_cast<uint32_t>(copies.size())});
		m_upload_batch_copies.insert(m_upload_batch_copies.end(), copies.begin(), copies.end());
	} else if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
}

void BufferCache::SynchronizeBuffers(std::span<Buffer* const>                       buffers,
                                     std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	if (m_span_copies.size() < buffers.size()) {
		m_span_copies.resize(buffers.size());
	}
	m_span_sizes.assign(buffers.size(), 0);
	for (size_t index = 0; index < buffers.size(); index++) {
		m_span_copies[index].clear();
	}
	const MemoryTracker::UploadScope upload_scope;
	std::vector<std::vector<std::pair<uint64_t, uint64_t>>> marked;
	if (SyncPreferGpu()) {
		marked.resize(ranges.size());
		for (size_t index = 0; index < ranges.size(); index++) {
			m_memory_tracker.CollectGpuMarkedRuns(ranges[index].first, ranges[index].second, marked[index]);
		}
	}
	m_memory_tracker.ForEachUploadRanges(
	    ranges,
	    [&](size_t index, uint64_t address, uint64_t bytes) noexcept {
		    const auto emit = [&](uint64_t at, uint64_t run) noexcept {
			    m_span_copies[index].emplace_back(m_span_sizes[index], buffers[index]->Offset(at), run);
			    m_span_sizes[index] += run;
			    NoteUpload(at, run);
		    };
		    if (marked.empty() || marked[index].empty()) {
			    emit(address, bytes);
		    } else {
			    SyncSkipMarked(marked[index], address, bytes, emit);
		    }
	    },
	    [&]() noexcept {
		    for (size_t index = 0; index < buffers.size(); index++) {
			    const auto source =
			        UploadCopies(*buffers[index], m_span_copies[index], m_span_sizes[index]);
			    RecordBufferUpload(*buffers[index], m_span_copies[index], source);
			    if (!m_span_copies[index].empty() && ArgsSyncDiag()) {
				    ArgsSyncNoteUploads(m_memory_tracker, *buffers[index], m_span_copies[index], 2, m_sync_caller);
			    }
		    }
	    },
	    m_device_sync_pass);
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);
	g_write_block_stats.uploaded.fetch_add(total_size, std::memory_order_relaxed);

	auto*      host_copies = m_scheduler.HostCopies();
	const auto ring        = m_staging_buffer.Mapped();
	const bool hand_over   = host_copies != nullptr && m_staging_buffer.IsCoherent() &&
	                         GuestGpu::IsGpuThread();
	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		const bool in_ring =
		    hand_over && mapped >= ring.data() && mapped < ring.data() + ring.size();
		if (auto& deferred = MemoryTracker::DeferredProtects(); !deferred.empty()) {
			bool queued = in_ring;
			for (const auto& copy: copies) {
				queued = queued && LibKernel::Memory::FindBackingPointer(buffer.CpuAddress() + copy.dstOffset, copy.size) != nullptr;
			}
			for (const auto& item: deferred) {
				if (queued) {
					host_copies->Call(
					    [](void* manager, uint64_t address, uint64_t size) {
						    static_cast<RegionManager*>(manager)->ApplyDeferredProtection(address, size);
					    },
					    item.manager, item.address, item.size);
				} else {
					item.manager->ApplyDeferredProtection(item.address, item.size);
				}
			}
			deferred.clear();
			if (queued) m_protect_jobs = host_copies->Flush();
		}
		for (auto& copy: copies) {
			const auto  address = buffer.CpuAddress() + copy.dstOffset;
			const void* backing =
			    in_ring ? LibKernel::Memory::FindBackingPointer(address, copy.size) : nullptr;
			if (backing != nullptr) {
				host_copies->Copy(mapped + copy.srcOffset, backing, copy.size);
			} else {
				if (m_protect_jobs != 0 && host_copies != nullptr) {
					host_copies->WaitFor(m_protect_jobs);
					m_protect_jobs = 0;
				}
				if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
					EXIT("BufferCache: failed to read sparse guest buffer memory\n");
				}
			}
			copy.srcOffset += base_offset;
		}
	}, !m_upload_batching);
	return source.first->Handle();
}

void BufferCache::StreamGuestData(uint8_t* destination, uint64_t vaddr, uint64_t size) {
	auto* copies = m_scheduler.HostCopies();
	if (copies != nullptr && m_stream_buffer.IsCoherent() && GuestGpu::IsGpuThread()) {
		if (const void* source = Libs::LibKernel::Memory::FindBackingPointer(vaddr, size)) {
			if (m_stream_in_place) {
				copies->CopyInline(destination, source, size);
			} else {
				copies->Copy(destination, source, size);
			}
			return;
		}
	}
	if (!LibKernel::Memory::TryReadBacking(vaddr, destination, size) &&
	    !LibKernel::Memory::CopySparseMemory(vaddr, destination, size)) {
		EXIT("BufferCache: failed to read sparse guest buffer memory at 0x%" PRIx64 " size 0x%" PRIx64
		     "\n",
		     vaddr, size);
	}
}

std::optional<uint64_t> BufferCache::StreamGuestRange(uint64_t vaddr, uint64_t size) {
	if (size == 0 || m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return std::nullopt;
	}
	const uint64_t base   = vaddr & ~uint64_t {255};
	const uint64_t bytes  = vaddr + size - base;
	const void*    source = Libs::LibKernel::Memory::FindBackingPointer(base, bytes);
	if (source == nullptr) {
		return std::nullopt;
	}
	auto [mapped, offset] = m_stream_buffer.Map(bytes, 256, false);
	if (mapped == nullptr) {
		return std::nullopt;
	}
	auto* copies = m_scheduler.HostCopies();
	if (copies != nullptr && m_stream_buffer.IsCoherent() && GuestGpu::IsGpuThread()) {
		if (m_stream_in_place) {
			copies->CopyInline(mapped, source, bytes);
		} else {
			copies->Copy(mapped, source, bytes);
		}
	} else {
		std::memcpy(mapped, source, bytes);
	}
	m_stream_buffer.Commit();
	return offset + (vaddr - base);
}

std::optional<uint64_t> BufferCache::FindStreamCopy(uint64_t vaddr, uint64_t size) {
	if (m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return std::nullopt;
	}
	const auto& stream = (*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) % StreamMemoSlots];
	if (stream.vaddr == vaddr && stream.size == size &&
	    stream.window == m_scheduler.Context().GetRenderExecutor().DrawWindow() &&
	    stream.lap == m_stream_buffer.Lap() && stream.write_generation == m_gpu_write_generation) {
		return stream.offset;
	}
	return std::nullopt;
}

void BufferCache::NoteStreamCopy(uint64_t vaddr, uint64_t size, uint64_t offset) {
	(*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) % StreamMemoSlots] = {
	    .vaddr            = vaddr,
	    .size             = size,
	    .window           = m_scheduler.Context().GetRenderExecutor().DrawWindow(),
	    .lap              = m_stream_buffer.Lap(),
	    .write_generation = m_gpu_write_generation,
	    .offset           = offset};
}

StreamBuffer& BufferCache::GetTableUploadBuffer() {
	if (!m_table_upload) {
		m_table_upload = std::make_unique<StreamBuffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 32 * MiB);
	}
	return *m_table_upload;
}

vk::DeviceAddress BufferCache::TableUpload(const void* data, uint64_t size) {
	if (!m_table_upload) {
		m_table_upload = std::make_unique<StreamBuffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 32 * MiB);
	}
	return m_table_upload->BufferDeviceAddress() + m_table_upload->Copy(data, size, 16);
}

uint64_t BufferCache::StreamUpload(const void* data, uint64_t size, uint64_t alignment) {
	auto* copies = m_scheduler.HostCopies();
	if (copies == nullptr || !m_stream_buffer.IsCoherent() || !GuestGpu::IsGpuThread()) {
		return m_stream_buffer.Copy(data, size, alignment);
	}
	const auto [mapped, offset] = m_stream_buffer.Map(size, alignment);
	EXIT_IF(mapped == nullptr);
	copies->CopyInline(mapped, data, size);
	m_stream_buffer.Commit();
	return offset;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));
	if (is_written) {
		m_scheduler.Context().CollectHotPageWrites(vaddr, size);
		if (ArgsSyncDiag()) {
			SyncRingRecordWrite(3, vaddr, size, KYTY_DIAG_RETURN_ADDRESS());
		}
	}

	const bool  small_read = !is_written && size <= CACHING_PAGESIZE;
	StreamMemo* stream     = nullptr;
	uint64_t    window     = 0;
	if (small_read && !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		window = m_scheduler.Context().GetRenderExecutor().DrawWindow();
		stream = &(*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) %
		                           StreamMemoSlots];
		if (stream->vaddr == vaddr && stream->size == size && stream->window == window &&
		    stream->lap == m_stream_buffer.Lap() &&
		    stream->write_generation == m_gpu_write_generation) {
			return {&m_stream_buffer, stream->offset};
		}
	}

	const bool  plain_read = !is_written;
	ObtainMemo* memo       = nullptr;
	uint64_t    cpu_epoch  = 0;
	const auto  page       = vaddr >> CACHING_PAGEBITS;
	const bool  in_page    = plain_read && size != 0 &&
	                         ((vaddr + size - 1) >> CACHING_PAGEBITS) == page;
	PageMemo*   page_memo  = nullptr;
	uint32_t    page_bits  = 0;
	if (plain_read) {
		cpu_epoch = CpuDirtyEpoch(vaddr, size);
		if (in_page) {
			page_memo = &(*m_page_memo)[(page * 0x9e3779b97f4a7c15ull >> 32u) % PageMemoSlots];
			const auto first = (vaddr & (CACHING_PAGESIZE - 1)) / TRACKER_PAGE_SIZE;
			const auto last  = ((vaddr + size - 1) & (CACHING_PAGESIZE - 1)) / TRACKER_PAGE_SIZE;
			page_bits        = ((2u << last) - 1u) & ~((1u << first) - 1u);
			if (page_memo->page == page && page_memo->buffer_epoch == m_buffer_set_epoch &&
			    page_memo->cpu_epoch == cpu_epoch && (page_memo->clean & page_bits) == page_bits &&
			    (!is_texel_buffer || NoImageStartsAt(vaddr))) {
				auto& buffer = m_slot_buffers[page_memo->id];
				TouchBuffer(buffer);
				return {&buffer, vaddr - page_memo->buffer_address};
			}
		} else {
			const auto hash = (vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull;
			memo            = &(*m_obtain_memo)[(hash >> 32u) % ObtainMemoSlots];
			if (memo->vaddr == vaddr && memo->size == size && memo->texel == is_texel_buffer &&
			    memo->buffer_epoch == m_buffer_set_epoch && memo->cpu_epoch == cpu_epoch &&
			    (!is_texel_buffer || NoImageStartsAt(vaddr))) {
				auto& buffer = m_slot_buffers[memo->id];
				TouchBuffer(buffer);
				return {&buffer, memo->offset};
			}
		}
	}

	if (stream != nullptr && m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			StreamGuestData(mapped, vaddr, size);
			m_stream_buffer.Commit();
			*stream = {.vaddr            = vaddr,
			           .size             = size,
			           .window           = window,
			           .lap              = m_stream_buffer.Lap(),
			           .write_generation = m_gpu_write_generation,
			           .offset           = offset};
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	WrittenMemo* written = nullptr;
	uint64_t     written_cpu_epoch = 0;
	uint64_t     written_gpu_epoch = 0;
	if (is_written) {
		const auto hash   = (vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull;
		written           = &(*m_written_memo)[(hash >> 32u) % WrittenMemoSlots];
		written_cpu_epoch = RegionManager::CpuDirtyEpoch();
		written_gpu_epoch = RegionManager::GpuCleanEpoch();
		if (written->vaddr == vaddr && written->size == size &&
		    written->buffer_epoch == m_buffer_set_epoch && written->cpu_epoch == written_cpu_epoch &&
		    written->gpu_epoch == written_gpu_epoch &&
		    (IsBufferInvalid(id) || id == written->id)) {
			auto& buffer = m_slot_buffers[written->id];
			TouchBuffer(buffer);
			m_gpu_write_generation++;
			if (m_gpu_modified_ranges.Add(vaddr, size)) {
				NoteGpuModifiedChange(vaddr, size, true);
			}
			ForgetColorMetadataChecks(vaddr, size);
			NoteWrite(vaddr, size);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}

	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
#if defined(__GNUC__) || defined(__clang__)
	m_sync_caller = __builtin_return_address(0);
#endif
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	m_sync_caller = nullptr;
	if (written != nullptr) {
		*written = {.vaddr        = vaddr,
		            .size         = size,
		            .buffer_epoch = m_buffer_set_epoch,
		            .cpu_epoch    = written_cpu_epoch,
		            .gpu_epoch    = written_gpu_epoch,
		            .id           = id};
	}
	if (page_memo != nullptr && (!is_texel_buffer || NoImageStartsAt(vaddr))) {
		const auto page_address = page << CACHING_PAGEBITS;
		if (page_memo->page == page && page_memo->buffer_epoch == m_buffer_set_epoch &&
		    page_memo->cpu_epoch == cpu_epoch && page_memo->id == id) {
			page_memo->clean |= page_bits;
		} else if (buffer.IsInBounds(page_address, CACHING_PAGESIZE)) {
			const bool all_clean =
			    !m_memory_tracker.IsRegionCpuModified(page_address, CACHING_PAGESIZE);
			*page_memo = {.page           = page,
			              .buffer_epoch   = m_buffer_set_epoch,
			              .cpu_epoch      = cpu_epoch,
			              .buffer_address = buffer.CpuAddress(),
			              .id             = id,
			              .clean          = all_clean ? AllPageBits : page_bits};
		}
	} else if (memo != nullptr && (!is_texel_buffer || NoImageStartsAt(vaddr))) {
		*memo = {.vaddr        = vaddr,
		         .size         = size,
		         .buffer_epoch = m_buffer_set_epoch,
		         .cpu_epoch    = cpu_epoch,
		         .offset       = buffer.Offset(vaddr),
		         .id           = id,
		         .texel        = is_texel_buffer};
	}
	if (is_written) {
		m_gpu_write_generation++;
		if (m_gpu_modified_ranges.Add(vaddr, size)) {
			NoteGpuModifiedChange(vaddr, size, true);
		}
		ForgetColorMetadataChecks(vaddr, size);
		NoteWrite(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainArgsBuffer(uint64_t vaddr, uint64_t size, uint32_t site) {
	const bool diag   = ArgsSyncDiag();
	const bool prefer = ArgsPreferGpu();
	if (!diag && !prefer) {
		return ObtainBuffer(vaddr, size, false);
	}
	const auto probe     = m_memory_tracker.ProbeArgsPages(vaddr, size);
	const bool gpu_marks = probe.gpu_written || probe.gpu_noted;
	if (diag) {
		auto e = SyncEntry(5, vaddr, size, KYTY_DIAG_RETURN_ADDRESS(), SyncMarks(probe), false);
		e.path = static_cast<uint8_t>(site);
		std::scoped_lock lock(g_sync_ring.mutex);
		auto&            r = g_sync_ring;
		r.tracker         = &m_memory_tracker;
		e.seq             = ++r.seq;
		r.reads[r.read_count++ % SyncRingSize] = e;
	}
	if (prefer && probe.cpu_dirty && gpu_marks) {
		const auto id     = FindBuffer(vaddr, size);
		auto&      buffer = m_slot_buffers[id];
		TouchBuffer(buffer);
		if (diag) {
			auto& c = g_args_sync;
			c.reads++;
			c.dirty++;
			c.kept_gpu++;
			ArgsSyncPrint("kept the GPU's bytes over CPU-dirty pages", site, vaddr, size, probe);
			ArgsSyncReport();
		}
		return {&buffer, buffer.Offset(vaddr)};
	}
	if (!diag) {
		return ObtainBuffer(vaddr, size, false);
	}
	const auto uploads   = m_upload_events;
	const auto result    = ObtainBuffer(vaddr, size, false);
	const bool streamed  = result.first == static_cast<Buffer*>(&m_stream_buffer);
	const bool uploaded  = m_upload_events != uploads;
	auto&      c         = g_args_sync;
	c.reads++;
	c.dirty += probe.cpu_dirty ? 1u : 0u;
	c.streamed += streamed ? 1u : 0u;
	c.uploaded += uploaded ? 1u : 0u;
	if ((streamed || uploaded) && gpu_marks) {
		c.over_gpu++;
		ArgsSyncPrint(streamed ? "streamed guest memory over GPU-written pages" : "uploaded CPU-dirty pages over GPU-written pages",
		              site, vaddr, size, probe);
	} else if (streamed || uploaded) {
		c.cpu_only++;
	}
	ArgsSyncReport();
	return result;
}

void BufferCache::DiagCullPre(CommandRecorder cmd, vk::Buffer args, uint64_t args_offset, uint64_t args_guest,
                              const uint32_t* direct_groups, uint8_t kind, uint32_t mode, uint32_t flags) {
	g_cull.current = -1;
	if (!ArgsSyncDiag() || !CullShader(g_sync_shader)) {
		return;
	}
	if (g_cull.ring == nullptr) {
		g_cull.ring      = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                           vk::BufferUsageFlagBits::eTransferDst, uint64_t {CullSlots} * CullSlotBytes);
		g_cull.scheduler = &m_scheduler;
	}
	const uint32_t index = g_cull.next++ % CullSlots;
	auto&          s     = g_cull.slots[index];
	s                    = {};
	s.post_head.fill(-1);
	s.op_seq     = g_sync_op.op_seq;
	s.submit     = g_sync_op.submit;
	s.queue      = g_sync_op.queue;
	s.shader     = g_sync_shader;
	s.tick       = m_scheduler.CurrentTick();
	s.args_guest = args_guest;
	s.kind       = kind;
	s.mode       = mode;
	s.flags      = flags;
	{
		std::scoped_lock lock(g_sync_ring.mutex);
		s.seq = ++g_sync_ring.seq;
	}
	if (direct_groups != nullptr) {
		std::copy_n(direct_groups, 3, s.direct_groups);
	}
	const uint64_t    base = uint64_t {index} * CullSlotBytes;
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
	                    0, nullptr);
	if (args) {
		const vk::BufferCopy copy {args_offset, base, 12};
		cmd.copyBuffer(args, g_cull.ring->Handle(), 1, &copy);
		s.has_args = 1;
	}
	for (const auto& b: g_cull_bindings) {
		if (s.bindings >= 8u) {
			break;
		}
		const auto* owner = m_page_table.Find(b.address >> PageTable::kPageBits);
		if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(b.address, b.size)) {
			continue;
		}
		auto&                buffer = m_slot_buffers[*owner];
		const vk::BufferCopy copy {buffer.Offset(b.address), base + CullPreBase + uint64_t {s.bindings} * CullHeadBytes,
		                           std::min<uint64_t>(b.size, CullHeadBytes)};
		cmd.copyBuffer(buffer.Handle(), g_cull.ring->Handle(), 1, &copy);
		s.b[s.bindings++] = b;
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &after, 0, nullptr, 0,
	                    nullptr);
	g_cull.current = static_cast<int32_t>(index);
}

void BufferCache::DiagCullPost(CommandRecorder cmd, vk::Buffer groups, uint64_t groups_offset) {
	if (g_cull.current < 0) {
		return;
	}
	const auto     index = static_cast<uint32_t>(g_cull.current);
	auto&          s     = g_cull.slots[index];
	const uint64_t base  = uint64_t {index} * CullSlotBytes;
	g_cull.current       = -1;
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
	                    0, nullptr);
	if (groups) {
		const vk::BufferCopy copy {groups_offset, base + 16u, 12};
		cmd.copyBuffer(groups, g_cull.ring->Handle(), 1, &copy);
		s.has_groups = 1;
	}
	int8_t heads = 0;
	for (uint32_t i = 0; i < s.bindings; ++i) {
		if (s.b[i].written == 0) {
			continue;
		}
		const auto* owner = m_page_table.Find(s.b[i].address >> PageTable::kPageBits);
		if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(s.b[i].address, s.b[i].size)) {
			continue;
		}
		auto&                buffer = m_slot_buffers[*owner];
		const vk::BufferCopy small {buffer.Offset(s.b[i].address), base + 64u + uint64_t {i} * 64u,
		                            std::min<uint64_t>(s.b[i].size, 64u)};
		cmd.copyBuffer(buffer.Handle(), g_cull.ring->Handle(), 1, &small);
		if (heads < 4) {
			const vk::BufferCopy head {buffer.Offset(s.b[i].address), base + CullPostBase + uint64_t {static_cast<uint8_t>(heads)} * CullHeadBytes,
			                           std::min<uint64_t>(s.b[i].size, CullHeadBytes)};
			cmd.copyBuffer(buffer.Handle(), g_cull.ring->Handle(), 1, &head);
			s.post_head[i] = heads++;
		}
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &after, 0, nullptr, 0,
	                    nullptr);
	s.post = 1;
}

bool BufferCache::PrepareDeviceAddressRead(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return true;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		Common::HackNotice::Report("device address read outside guest memory skipped with the ray tracing data that "
		                           "needed it",
		                           "BufferCache::PrepareDeviceAddressRead", 0, 0,
		                           fmt::format("0x{:x}, {} bytes", vaddr, size));
		return false;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
	return true;
}

static bool AsyncImageStaging() {
	static const bool enabled = [] {
		if (const char* value = std::getenv("KYTY_ASYNC_IMAGE_STAGING"); value != nullptr && value[0] != '\0')
			return value[0] != '0';
		return ::Loader::DemonsSoulsTitle::IsSupportedGame();
	}();
	return enabled;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto*      host_copies = m_scheduler.HostCopies();
	const auto ring        = m_staging_buffer.Mapped();
	const bool hand_over   = host_copies != nullptr && m_staging_buffer.IsCoherent() && GuestGpu::IsGpuThread() &&
	                         AsyncImageStaging();
	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (hand_over && mapped >= ring.data() && mapped < ring.data() + ring.size()) {
			if (const void* backing = Libs::LibKernel::Memory::FindBackingPointer(vaddr, size)) {
				host_copies->Copy(mapped, backing, size);
				(void)host_copies->Flush();
				return;
			}
			thread_local std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
			if (Libs::LibKernel::Memory::TryGetSparseBackingPieces(vaddr, size, pieces)) {
				uint64_t offset = 0;
				for (const auto& [piece, bytes]: pieces) {
					if (piece != nullptr) {
						host_copies->Copy(mapped + offset, piece, bytes);
					} else {
						host_copies->Call([](void*, uint64_t destination, uint64_t count) {
							std::memset(reinterpret_cast<void*>(destination), 0, count);
						}, nullptr, reinterpret_cast<uint64_t>(mapped + offset), bytes);
					}
					offset += bytes;
				}
				(void)host_copies->Flush();
				return;
			}
		}
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size) &&
		    !Libs::LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::DrainHostCopies() {
	if (auto* copies = m_scheduler.HostCopies(); copies != nullptr && GuestGpu::IsGpuThread()) {
		copies->Drain();
	}
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	m_gpu_write_generation++;
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		InvalidateMemory(vaddr, size);
		m_texture_cache.InvalidateMemory(vaddr, size);
		DrainHostCopies();
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		LibKernel::Memory::NoteEmulatorWrite(vaddr, size);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

bool BufferCache::ReplicateBuffer(uint64_t destination, uint64_t count, uint64_t source,
                                  uint64_t pattern) {
	const auto size = count * sizeof(uint32_t);
	const auto from = pattern * sizeof(uint32_t);
	if (destination == 0 || source == 0 || ((destination | source) & 3u) != 0 || count == 0 ||
	    pattern == 0 || (destination < source + from && source < destination + size) ||
	    IsRegionGpuModified(destination, size) || IsRegionGpuModified(source, from) ||
	    m_texture_cache.FindImageFromRange(source, from)) {
		return false;
	}
	m_gpu_write_generation++;
	(void)m_texture_cache.ClearMeta(destination);
	InvalidateMemory(destination, size);
	m_texture_cache.InvalidateMemory(destination, size);
	DrainHostCopies();
	auto*      target = reinterpret_cast<uint8_t*>(destination);
	const auto first  = std::min(size, from);
	std::memcpy(target, reinterpret_cast<const void*>(source), first);
	for (uint64_t written = first; written < size;) {
		const auto chunk = std::min(written, size - written);
		std::memcpy(target + written, target, chunk);
		written += chunk;
	}
	LibKernel::Memory::NoteEmulatorWrite(destination, size);
	return true;
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	m_gpu_write_generation++;
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		DrainHostCopies();
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		LibKernel::Memory::NoteEmulatorWrite(dst_vaddr, size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

void BufferCache::WriteTicks::Assign(uint64_t begin, uint64_t end, uint64_t tick) {
	if (begin >= end) {
		return;
	}
	auto& recent = m_recent[((begin ^ (end << 24u)) * 0x9e3779b97f4a7c15ull) >> 56u];
	if (recent.begin == begin && recent.end == end && recent.tick == tick) {
		return;
	}
	recent = {begin, end, tick};
	auto it = m_ranges.lower_bound(begin);
	if (it != m_ranges.end() && it->first == begin && it->second.end == end) {
		it->second.tick = tick;
		return;
	}
	if (it != m_ranges.begin()) {
		const auto previous = std::prev(it);
		if (previous->second.end > begin) {
			const auto right       = previous->second;
			previous->second.end = begin;
			it                     = m_ranges.emplace_hint(it, begin, right);
		}
	}
	while (it != m_ranges.end() && it->first < end) {
		if (it->second.end > end) {
			const auto tail = it->second;
			m_ranges.erase(it);
			m_ranges.emplace(end, tail);
			break;
		}
		it = m_ranges.erase(it);
	}
	m_ranges.emplace(begin, Range {end, tick});
}

uint64_t BufferCache::WriteTicks::Latest(uint64_t begin, uint64_t end) const {
	auto it = m_ranges.upper_bound(begin);
	if (it != m_ranges.begin()) {
		--it;
	}
	uint64_t latest = 0;
	uint64_t cursor = begin;
	for (; it != m_ranges.end() && it->first < end && cursor < end; ++it) {
		if (it->second.end <= cursor) {
			continue;
		}
		if (it->first > cursor) {
			return UINT64_MAX;
		}
		latest = std::max(latest, it->second.tick);
		cursor = it->second.end;
	}
	return cursor >= end ? latest : UINT64_MAX;
}

void BufferCache::NoteReadbackWait(uint64_t begin, uint64_t end, bool device_address,
                                   bool drained) {
	const auto count = ++m_readback_count;
	if (device_address && (drained || m_device_address_hot != 0)) {
		m_device_address_hot = count;
	}
	std::erase_if(m_readback_hot, [count](const HotReadback& hot) {
		return count - hot.last_hit >= HotReadbackLifetime;
	});
	begin = Common::AlignDown(begin, TRACKER_PAGE_SIZE);
	end   = Common::AlignUp(end, TRACKER_PAGE_SIZE);
	for (auto& hot: m_readback_hot) {
		if (begin < hot.end && hot.begin < end) {
			hot.begin    = std::min(hot.begin, begin);
			hot.end      = std::max(hot.end, end);
			hot.last_hit = count;
			return;
		}
	}
	if (drained && !device_address) {
		if (m_readback_hot.size() == MaxHotReadbacks) {
			m_readback_hot.erase(std::ranges::min_element(m_readback_hot, {}, &HotReadback::last_hit));
		}
		m_readback_hot.push_back({begin, end, count});
	}
}

bool BufferCache::TakeFlushRequest() {
	if (!m_flush_requested) {
		return false;
	}
	if (m_flush_deferred_tick != 0 && m_scheduler.CurrentTick() != m_flush_deferred_tick) {
		m_flush_requested     = false;
		m_flush_deferred_tick = 0;
		return false;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - m_last_requested_flush < MinRequestedFlushInterval) {
		m_flush_deferred_tick = m_scheduler.CurrentTick();
		return false;
	}
	m_last_requested_flush = now;
	m_flush_requested      = false;
	m_flush_deferred_tick  = 0;
	return true;
}

bool BufferCache::CommitWriteTicks() {
	if (m_pending_writes.empty() && !m_device_address_pending) {
		return TakeFlushRequest();
	}
	const auto tick = m_scheduler.CurrentTick();
	for (const auto& [vaddr, size]: m_pending_writes) {
		m_write_ticks.Assign(vaddr, vaddr + size, tick);
	}
	if (!m_mirror_lines.empty()) {
		for (const auto& [vaddr, size]: m_pending_writes) {
			MarkMirrorLinesWritten(vaddr, size, tick);
		}
	}
	m_pending_writes.clear();
	if (m_device_address_pending) {
		m_device_address_pending = false;
		m_device_address_tick    = tick;
	}
	return TakeFlushRequest();
}

void BufferCache::MarkMirrorLinesWritten(uint64_t vaddr, uint64_t size, uint64_t tick) {
	const auto end = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
	for (auto it = m_mirror_lines.lower_bound(Common::AlignDown(vaddr, MirrorLineSize));
	     it != m_mirror_lines.end() && it->first < end; ++it) {
		auto& line = it->second;
		line.tick  = 0;
		if (m_mirror_uses - line.last_use < MirrorFlushLifetime) {
			m_flush_requested = true;
		}
		if (line.dirty_tick != tick) {
			line.dirty_tick = tick;
			m_mirror_dirty.push_back(it->first);
		}
	}
}

void BufferCache::AddMirrorLine(uint64_t line) {
	if (m_mirror_free_slots.empty()) {
		if (m_mirror_lines.size() < MaxMirrorLines) {
			m_mirror_free_slots.push_back(static_cast<uint32_t>(m_mirror_lines.size()));
		} else {
			std::vector<uint64_t> uses;
			uses.reserve(m_mirror_lines.size());
			for (const auto& [address, entry]: m_mirror_lines) {
				uses.push_back(entry.last_use);
			}
			std::nth_element(uses.begin(), uses.begin() + uses.size() / 2, uses.end());
			const auto oldest = uses[uses.size() / 2];
			for (auto it = m_mirror_lines.begin(); it != m_mirror_lines.end();) {
				if (it->second.last_use <= oldest) {
					m_mirror_free_slots.push_back(it->second.slot);
					it = m_mirror_lines.erase(it);
				} else {
					++it;
				}
			}
		}
	}
	MirrorLine entry;
	entry.slot = m_mirror_free_slots.back();
	m_mirror_free_slots.pop_back();
	entry.last_use = ++m_mirror_uses;
	entry.dirty_tick = m_scheduler.CurrentTick();
	m_mirror_lines.emplace(line, entry);
	m_mirror_dirty.push_back(line);
}

bool BufferCache::TryReadMirroredGpuWrites(uint64_t vaddr, void* data, uint64_t size) {
	if (size == 0 || size > MirrorMaxRead || m_device_address_pending ||
	    !GuestGpu::IsGpuThread()) {
		return false;
	}
	if (InFlightReadbackTick(vaddr, vaddr + size) != 0) {
		return false;
	}
	for (const auto& [address, bytes]: m_pending_writes) {
		if (address < vaddr + size && vaddr < address + bytes) {
			return false;
		}
	}
	uint64_t needed  = m_device_address_tick;
	bool     unknown = false;
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
		const auto latest = m_write_ticks.Latest(start, end);
		unknown |= latest == UINT64_MAX;
		needed = std::max(needed, latest);
	});
	if (unknown) {
		return false;
	}
	const auto first = Common::AlignDown(vaddr, MirrorLineSize);
	const auto last  = Common::AlignUp(vaddr + size, MirrorLineSize);
	uint64_t   wait  = 0;
	bool       held  = true;
	std::array<uint32_t, MirrorMaxRead / MirrorLineSize + 1> slots {};
	for (auto line = first; line < last; line += MirrorLineSize) {
		const auto it = m_mirror_lines.find(line);
		if (it == m_mirror_lines.end()) {
			AddMirrorLine(line);
			held = false;
			continue;
		}
		auto& entry    = it->second;
		entry.last_use = ++m_mirror_uses;
		slots[(line - first) / MirrorLineSize] = entry.slot;
		if (entry.tick == 0 || entry.tick < needed) {
			held = false;
			if (entry.dirty_tick != m_scheduler.CurrentTick()) {
				entry.dirty_tick = m_scheduler.CurrentTick();
				m_mirror_dirty.push_back(line);
			}
		}
		wait = std::max(wait, entry.tick);
	}
	if (!held || wait >= m_scheduler.CurrentTick()) {
		return false;
	}
	m_scheduler.GetMasterSemaphore().Wait(wait);
	m_mirror_flush_use = m_mirror_uses;
	m_scheduler.WaitPriorityOperations(needed);
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, data, size)) {
		return false;
	}
	const auto* mapped = m_mirror->Mapped().data();
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
		while (start < end) {
			const auto line   = Common::AlignDown(start, MirrorLineSize);
			const auto chunk  = std::min(end, line + MirrorLineSize) - start;
			const auto slot   = slots[(line - first) / MirrorLineSize];
			const auto offset = uint64_t {slot} * MirrorLineSize + (start - line);
			m_mirror->Invalidate(offset, chunk);
			std::memcpy(static_cast<uint8_t*>(data) + (start - vaddr), mapped + offset, chunk);
			Libs::LibKernel::Memory::WriteBacking(start, mapped + offset, chunk);
			start += chunk;
		}
	});
	if (m_gpu_modified_ranges.Subtract(vaddr, size)) {
		NoteGpuModifiedChange(vaddr, size, false);
	}
	return true;
}

void BufferCache::RecordMirrorCopies(CommandBuffer& command) {
	if (m_mirror_dirty.empty() || !GuestGpu::IsGpuThread()) {
		return;
	}
	if (!m_mirror) {
		m_mirror = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                    vk::BufferUsageFlagBits::eTransferDst,
		                                    uint64_t {MaxMirrorLines} * MirrorLineSize);
	}
	std::sort(m_mirror_dirty.begin(), m_mirror_dirty.end());
	m_mirror_dirty.erase(std::unique(m_mirror_dirty.begin(), m_mirror_dirty.end()),
	                     m_mirror_dirty.end());
	const auto tick = m_scheduler.CurrentTick();
	command.EndRendering();
	const auto native = command.Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	std::vector<vk::BufferCopy> copies;
	const Buffer*               source = nullptr;
	const auto                  flush  = [&] {
		if (!copies.empty()) {
			native.copyBuffer(source->Handle(), m_mirror->Handle(),
			                  static_cast<uint32_t>(copies.size()), copies.data());
			copies.clear();
		}
	};
	for (const auto line: m_mirror_dirty) {
		const auto it = m_mirror_lines.find(line);
		if (it == m_mirror_lines.end()) {
			continue;
		}
		const auto* owner = m_page_table.Find(line >> PageTable::kPageBits);
		if (owner == nullptr || !*owner) {
			continue;
		}
		const auto& buffer = m_slot_buffers[*owner];
		if (buffer.is_deleted || !buffer.IsInBounds(line, MirrorLineSize)) {
			continue;
		}
		if (&buffer != source) {
			flush();
			source = &buffer;
		}
		copies.push_back({buffer.Offset(line), uint64_t {it->second.slot} * MirrorLineSize,
		                  MirrorLineSize});
		it->second.tick = tick;
	}
	flush();
	m_mirror_dirty.clear();
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &after, 0, nullptr, 0, nullptr);
}

bool BufferCache::DownloadOnReadbackQueue(Buffer& buffer, std::span<const vk::BufferCopy> copies,
                                          uint64_t total_size) {
	if (!m_graphics.readback_queue || m_device_address_pending) {
		return false;
	}
	const auto buffer_address = buffer.CpuAddress();
	uint64_t needed = m_device_address_tick;
	const auto readback_begin = buffer_address + copies.front().srcOffset;
	const auto readback_end   = buffer_address + copies.back().srcOffset + copies.back().size;
	for (const auto& copy: copies) {
		const auto begin = buffer_address + copy.srcOffset;
		const auto end   = begin + copy.size;
		for (const auto& [vaddr, size]: m_pending_writes) {
			if (vaddr < end && begin < vaddr + size) {
				NoteReadbackWait(readback_begin, readback_end, false, true);
				return false;
			}
		}
		needed = std::max(needed, m_write_ticks.Latest(begin, end));
	}
	const bool device_address = needed != 0 && needed == m_device_address_tick;
	if (needed >= m_scheduler.CurrentTick()) {
		NoteReadbackWait(readback_begin, readback_end, device_address, true);
		return false;
	}
	NoteReadbackWait(readback_begin, readback_end, device_address, false);
	while (m_scheduler.SubmittedTick() < needed) {
		Common::SpinPause();
	}
	auto& device = m_graphics.device;
	if (!m_readback.pool) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
		                             vk::CommandPoolCreateFlagBits::eTransient;
		pool_info.queueFamilyIndex = m_graphics.readback_family;
		vk::CommandBufferAllocateInfo allocate {};
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		const vk::FenceCreateInfo fence_info {};
		if (device.createCommandPool(&pool_info, nullptr, &m_readback.pool) != vk::Result::eSuccess ||
		    (allocate.commandPool = m_readback.pool,
		     device.allocateCommandBuffers(&allocate, &m_readback.command) != vk::Result::eSuccess) ||
		    device.createFence(&fence_info, nullptr, &m_readback.fence) != vk::Result::eSuccess) {
			EXIT("BufferCache: could not create the readback queue resources\n");
		}
	}
	if (!m_readback.staging || m_readback.staging->Size() < total_size) {
		m_readback.staging = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Download, 0,
		    vk::BufferUsageFlagBits::eTransferDst,
		    std::max<uint64_t>(Common::AlignUp(total_size, 1024 * 1024), 4 * MiB));
	}
	auto&                       staging = *m_readback.staging;
	const auto command = m_readback.command;
	(void)command.reset({});
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	(void)command.begin(&begin_info);
	command.copyBuffer(buffer.Handle(), staging.Handle(), static_cast<uint32_t>(copies.size()),
	                   copies.data());
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = staging.Handle();
	after.offset              = 0;
	after.size                = total_size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &after, 0, nullptr);
	(void)command.end();
	const auto                      semaphore = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &needed;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &semaphore;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	if (m_graphics.readback_queue.submit(1, &submit, m_readback.fence) != vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue submission failed\n");
	}
	for (;;) {
		const auto status = device.getFenceStatus(m_readback.fence);
		if (status == vk::Result::eSuccess) {
			break;
		}
		if (status != vk::Result::eNotReady) {
			EXIT("BufferCache: readback queue wait failed: %s\n", vk::to_string(status).c_str());
		}
		Common::SpinPause();
	}
	(void)device.resetFences(1, &m_readback.fence);
	m_scheduler.WaitPriorityOperations(needed);
	staging.Invalidate(0, total_size);
	const auto* mapped = staging.Mapped().data();
	for (const auto& copy: copies) {
		Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
		                                      mapped + copy.dstOffset, copy.size);
	}
	return true;
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
	m_texture_cache.NoteMetadataWrite(vaddr, size);
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::WriteAroundGpuWrites(uint64_t vaddr, const void* data, uint64_t size) {
	constexpr uint64_t MaxBytes = 4096;
	if (size == 0 || size > MaxBytes || (vaddr & 3u) != 0 || (size & 3u) != 0 ||
	    !GuestRange {vaddr, size}.Valid() || !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	Buffer*    buffer = nullptr;
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner && m_slot_buffers[*owner].IsInBounds(vaddr, size)) {
		buffer = &m_slot_buffers[*owner];
	} else if (IsRegionRegistered(vaddr, size)) {
		return false;
	}
	if (buffer == nullptr && HasGpuDirtyBytes(vaddr, size)) {
		return false;
	}
	if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	if (buffer == nullptr) {
		return true;
	}
	auto&      stream  = GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset  = StreamUpload(data, size, 4);
	auto&      command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer->Handle();
	before.offset              = buffer->Offset(vaddr);
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {offset, buffer->Offset(vaddr), size};
	native.copyBuffer(stream.Handle(), buffer->Handle(), 1, &copy);
	NoteWrite(vaddr, size);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	m_gpu_write_generation++;
	if (m_inflight_readbacks.empty() || InFlightReadbackTick(vaddr, vaddr + size) == 0) {
		if (m_gpu_modified_ranges.Subtract(vaddr, size)) {
			NoteGpuModifiedChange(vaddr, size, false);
		}
	}
	return true;
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t PageSize = 4096;
	const auto         page     = vaddr / PageSize;
	if (!m_inflight_readbacks.empty() && size != 0 &&
	    InFlightReadbackTick(vaddr, vaddr + size) != 0) {
		return true;
	}
	if (size == 0 || (vaddr + size - 1) / PageSize != page) {
		return m_gpu_modified_ranges.Intersects(vaddr, size);
	}
	auto& mask = m_dirty_masks[page % m_dirty_masks.size()];
	if (mask.page != page) {
		mask.page = page;
		mask.bits.fill(0);
		const auto base = page * PageSize;
		m_gpu_modified_ranges.ForEachInRange(base, PageSize, [&](uint64_t start, uint64_t end) {
			const auto low  = std::max(start, base) - base;
			const auto high = std::min(end, base + PageSize) - base - 1;
			for (auto word = low / 64; word <= high / 64; word++) {
				const auto from = word == low / 64 ? low % 64 : 0;
				const auto to   = word == high / 64 ? high % 64 : 63;
				mask.bits[word] |= (~uint64_t {0} >> (63 - to)) & (~uint64_t {0} << from);
			}
		});
	}
	const auto first = vaddr % PageSize;
	const auto last  = first + size - 1;
	for (auto word = first / 64; word <= last / 64; word++) {
		const auto low  = word == first / 64 ? first % 64 : 0;
		const auto high = word == last / 64 ? last % 64 : 63;
		const auto bits = (~uint64_t {0} >> (63 - high)) & (~uint64_t {0} << low);
		if ((mask.bits[word] & bits) != 0) {
			return true;
		}
	}
	return false;
}

namespace {

constexpr uint64_t GpuLinePageBits = 12;
constexpr uint64_t GpuLineBits     = 6;
constexpr uint64_t GpuLineMaxPages = uint64_t {1} << 16u;
constexpr uint64_t GpuLineMemoPages = 256;
constexpr uint64_t GpuLineChunk     = 512;

uint64_t GpuLineMask(uint64_t page, uint64_t vaddr, uint64_t size) {
	const auto base  = page << GpuLinePageBits;
	const auto low   = (std::max(vaddr, base) - base) >> GpuLineBits;
	const auto high  = (std::min(vaddr + size, base + (uint64_t {1} << GpuLinePageBits)) - base - 1) >> GpuLineBits;
	return (~uint64_t {0} >> (63 - high)) & (~uint64_t {0} << low);
}

size_t GpuLineHash(uint64_t page) {
	return static_cast<size_t>((page * 0x9e3779b97f4a7c15ull) >> 40u);
}

}

void BufferCache::NoteGpuLines(uint64_t vaddr, uint64_t size) noexcept {
	const auto first = vaddr >> GpuLinePageBits;
	const auto last  = (vaddr + size - 1) >> GpuLinePageBits;
	if (last - first >= GpuLineMaxPages) {
		const auto end = vaddr + size;
		for (auto* list: {&m_gpu_line_current_ranges, &m_gpu_line_ever_ranges}) {
			bool kept = false;
			for (auto& range: *list) {
				if (range.begin.load(std::memory_order_relaxed) <= vaddr && end <= range.end.load(std::memory_order_relaxed)) {
					kept = true;
					break;
				}
			}
			for (auto& range: *list) {
				if (kept) {
					break;
				}
				if (range.end.load(std::memory_order_relaxed) == 0) {
					range.begin.store(vaddr, std::memory_order_relaxed);
					range.end.store(end, std::memory_order_release);
					kept = true;
				}
			}
			if (!kept) {
				m_gpu_line_overflow.store(true, std::memory_order_release);
				g_readback_stats.line_overflow.store(true, std::memory_order_relaxed);
			}
		}
		return;
	}
	const bool memo = last - first >= GpuLineMemoPages;
	if (memo) {
		for (const auto& range: m_gpu_line_memo) {
			if (range.begin.load(std::memory_order_relaxed) <= vaddr && vaddr + size <= range.end.load(std::memory_order_relaxed)) {
				return;
			}
		}
	}
	for (auto page = first; page <= last; page++) {
		const auto mask   = GpuLineMask(page, vaddr, size);
		const auto key    = page + 1;
		bool       stored = false;
		for (size_t probe = 0; probe < GpuLineProbes && !stored; probe++) {
			auto&    slot    = m_gpu_line_slots[(GpuLineHash(page) + probe) % GpuLineSlots];
			uint64_t current = slot.page.load(std::memory_order_acquire);
			if (current == 0 && slot.page.compare_exchange_strong(current, key, std::memory_order_acq_rel)) {
				current = key;
			}
			if (current == key) {
				slot.ever.fetch_or(mask, std::memory_order_release);
				slot.current.fetch_or(mask, std::memory_order_release);
				stored = true;
			}
		}
		if (!stored) {
			m_gpu_line_overflow.store(true, std::memory_order_release);
			g_readback_stats.line_overflow.store(true, std::memory_order_relaxed);
			return;
		}
	}
	if (memo) {
		auto& range = m_gpu_line_memo[m_gpu_line_memo_next++ % GpuLineMemo];
		range.end.store(0, std::memory_order_relaxed);
		range.begin.store(vaddr, std::memory_order_relaxed);
		range.end.store(vaddr + size, std::memory_order_release);
	}
}

void BufferCache::ForgetCurrentGpuLines(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0) {
		return;
	}
	const auto end = vaddr + size;
	for (auto& range: m_gpu_line_memo) {
		if (vaddr < range.end.load(std::memory_order_relaxed) && range.begin.load(std::memory_order_relaxed) < end) {
			range.end.store(0, std::memory_order_release);
		}
	}
	for (auto& range: m_gpu_line_current_ranges) {
		const auto range_end   = range.end.load(std::memory_order_relaxed);
		const auto range_begin = range.begin.load(std::memory_order_relaxed);
		if (range_end != 0 && vaddr <= range_begin && range_end <= end &&
		    !m_gpu_modified_ranges.Intersects(range_begin, range_end - range_begin)) {
			range.end.store(0, std::memory_order_release);
		}
	}
	const auto first = vaddr >> GpuLinePageBits;
	const auto last  = (end - 1) >> GpuLinePageBits;
	if (last - first >= GpuLineMaxPages) {
		return;
	}
	std::array<GpuLineSlot*, GpuLineChunk> slots {};
	std::array<uint64_t, GpuLineChunk>     kept {};
	for (auto chunk = first; chunk <= last; chunk += GpuLineChunk) {
		const auto chunk_last = std::min(last, chunk + GpuLineChunk - 1);
		bool       any        = false;
		for (auto page = chunk; page <= chunk_last; page++) {
			auto&      found = slots[page - chunk];
			const auto key   = page + 1;
			found            = nullptr;
			for (size_t probe = 0; probe < GpuLineProbes; probe++) {
				auto&      slot    = m_gpu_line_slots[(GpuLineHash(page) + probe) % GpuLineSlots];
				const auto current = slot.page.load(std::memory_order_acquire);
				if (current == key) {
					if (slot.current.load(std::memory_order_relaxed) != 0) {
						found = &slot;
						any   = true;
					}
					break;
				}
				if (current == 0) {
					break;
				}
			}
		}
		if (!any) {
			continue;
		}
		std::fill_n(kept.begin(), chunk_last - chunk + 1, 0);
		const auto base = chunk << GpuLinePageBits;
		m_gpu_modified_ranges.ForEachInRange(base, (chunk_last - chunk + 1) << GpuLinePageBits, [&](uint64_t from, uint64_t to) {
			const auto from_page = std::max(from >> GpuLinePageBits, chunk);
			const auto to_page   = std::min((to - 1) >> GpuLinePageBits, chunk_last);
			for (auto page = from_page; page <= to_page; page++) {
				kept[page - chunk] |= GpuLineMask(page, from, to - from);
			}
		});
		for (auto page = chunk; page <= chunk_last; page++) {
			if (auto* slot = slots[page - chunk]; slot != nullptr) {
				slot->current.store(kept[page - chunk], std::memory_order_release);
			}
		}
	}
}

void BufferCache::UnmarkGpuModified(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
	ForgetCurrentGpuLines(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
	                      Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE) - Common::AlignDown(vaddr, TRACKER_PAGE_SIZE));
}

bool BufferCache::GpuLinesClean(uint64_t vaddr, uint64_t size, bool ever) const noexcept {
	if (size == 0 || m_gpu_line_overflow.load(std::memory_order_acquire)) {
		return false;
	}
	const auto end = vaddr + size;
	for (const auto& range: ever ? m_gpu_line_ever_ranges : m_gpu_line_current_ranges) {
		const auto range_end = range.end.load(std::memory_order_acquire);
		if (range_end != 0 && vaddr < range_end && range.begin.load(std::memory_order_relaxed) < end) {
			return false;
		}
	}
	const auto first = vaddr >> GpuLinePageBits;
	const auto last  = (end - 1) >> GpuLinePageBits;
	if (last - first >= GpuLineMaxPages) {
		return false;
	}
	for (auto page = first; page <= last; page++) {
		const auto key   = page + 1;
		bool       found = false;
		for (size_t probe = 0; probe < GpuLineProbes; probe++) {
			const auto& slot    = m_gpu_line_slots[(GpuLineHash(page) + probe) % GpuLineSlots];
			const auto  current = slot.page.load(std::memory_order_acquire);
			if (current == key) {
				const auto lines = (ever ? slot.ever : slot.current).load(std::memory_order_acquire);
				if ((lines & GpuLineMask(page, vaddr, size)) != 0) {
					return false;
				}
				found = true;
				break;
			}
			if (current == 0) {
				found = true;
				break;
			}
		}
		if (!found) {
			return false;
		}
	}
	return true;
}

bool BufferCache::NeverGpuWritten(uint64_t vaddr, uint64_t size) const noexcept {
	return GpuLinesClean(vaddr, size, true);
}

bool BufferCache::NoGpuBytesNow(uint64_t vaddr, uint64_t size) const noexcept {
	return GpuLinesClean(vaddr, size, false);
}

void BufferCache::NoteGpuModifiedChange(uint64_t vaddr, uint64_t size, bool dirty) {
	constexpr uint64_t PageSize = 4096;
	if (size == 0) {
		return;
	}
	if (dirty) {
		NoteGpuLines(vaddr, size);
	}
	const auto first = vaddr / PageSize;
	const auto last  = (vaddr + size - 1) / PageSize;
	const auto apply = [&](DirtyPageMask& mask) {
		const auto base = mask.page * PageSize;
		const auto low  = std::max(vaddr, base) - base;
		const auto high = std::min(vaddr + size, base + PageSize) - base - 1;
		for (auto word = low / 64; word <= high / 64; word++) {
			const auto from = word == low / 64 ? low % 64 : 0;
			const auto to   = word == high / 64 ? high % 64 : 63;
			const auto bits = (~uint64_t {0} >> (63 - to)) & (~uint64_t {0} << from);
			mask.bits[word] = dirty ? mask.bits[word] | bits : mask.bits[word] & ~bits;
		}
	};
	if (last - first >= m_dirty_masks.size()) {
		for (auto& mask: m_dirty_masks) {
			if (mask.page >= first && mask.page <= last) {
				apply(mask);
			}
		}
		return;
	}
	for (auto page = first; page <= last; page++) {
		auto& mask = m_dirty_masks[page % m_dirty_masks.size()];
		if (mask.page == page) {
			apply(mask);
		}
	}
}

bool BufferCache::HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Contains(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::CollectStaleBuffers(uint64_t tick) {
	static constexpr uint64_t StaleAge      = 2048;
	static constexpr uint64_t StalePeriod   = 8;
	static constexpr size_t   StaleVisitMax = 64;
	static constexpr size_t   StaleFreeMax  = 16;
	if (tick < StaleAge || tick % StalePeriod != 0) {
		return;
	}
	std::vector<BufferId> stale;
	std::vector<BufferId> kept;
	m_lru_cache.ForEachItemBelow(tick - StaleAge, [&](BufferId id) {
		const auto& buffer = m_slot_buffers[id];
		const bool  keep   = buffer.device_address_pinned ||
		                  m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size()) ||
		                  m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		(keep ? kept : stale).push_back(id);
		return stale.size() == StaleFreeMax || stale.size() + kept.size() == StaleVisitMax;
	});
	for (const auto id: kept) {
		m_lru_cache.Touch(m_slot_buffers[id].lru_id, tick);
	}
	for (const auto id: stale) {
		const auto& buffer = m_slot_buffers[id];
		const auto slot =
		    (m_stale_ring_next.fetch_add(1, std::memory_order_relaxed) % StaleRingSize) * 2;
		m_stale_ring[slot].store(buffer.CpuAddress(), std::memory_order_relaxed);
		m_stale_ring[slot + 1].store(buffer.Size(), std::memory_order_relaxed);
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_record_released) {
			m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
		}
		DeleteBuffer(id);
	}
}

uint64_t BufferCache::DeleteUnmappedBuffers(uint64_t vaddr, uint64_t size) {
	if (size == 0 || UINT64_MAX - vaddr < size) return 0;
	const auto            end = vaddr + size;
	std::vector<BufferId> ids;
	uint64_t              bytes = 0;
	for (auto it = m_buffers.lower_bound(vaddr); it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.is_deleted || buffer.CpuAddress() + buffer.Size() > end ||
		    m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()))
			continue;
		ids.push_back(it->second);
	}
	for (const auto id: ids) {
		auto& buffer = m_slot_buffers[id];
		bytes += buffer.Size();
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		DeleteBuffer(id);
	}
	return bytes;
}

uint64_t BufferCache::SplitUnmappedBuffers(uint64_t vaddr, uint64_t size, FreeBlocksQuery free_blocks,
                                           FreeBlockSplits* splits) {
	if (m_resources == nullptr || size == 0 || UINT64_MAX - vaddr < size) return 0;
	const auto            end = vaddr + size;
	std::vector<BufferId> candidates;
	uint64_t              cut = 0, lowest = vaddr, highest = end;
	auto                  it = m_buffers.lower_bound(vaddr);
	if (it != m_buffers.begin()) {
		const auto& buffer = m_slot_buffers[std::prev(it)->second];
		if (buffer.CpuAddress() + buffer.Size() > vaddr) candidates.push_back(std::prev(it)->second);
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.CpuAddress() + buffer.Size() > end) candidates.push_back(it->second);
	}
	for (const auto id: candidates) {
		lowest  = std::min(lowest, m_slot_buffers[id].CpuAddress());
		highest = std::max(highest, m_slot_buffers[id].CpuAddress() + m_slot_buffers[id].Size());
	}
	constexpr uint64_t Gap = CACHING_PAGESIZE * 4;
	constexpr uint64_t                         MinFreeBlockSplit = uint64_t {1} << 20u;
	std::vector<std::pair<uint64_t, uint64_t>> free_list;
	bool                                       listed = false, have_list = false;
	for (const auto id: candidates) {
		const auto& old = m_slot_buffers[id];
		if (old.is_deleted) continue;
		const auto begin = old.CpuAddress(), stop = begin + old.Size();
		RangeSet   mapped;
		m_resources->MappedRanges(begin, stop - begin, mapped);
		mapped.Subtract(vaddr, size);
		std::vector<std::pair<uint64_t, uint64_t>> pieces;
		uint64_t                                   kept = 0;
		const auto                                 cut_pieces = [&] {
			pieces.clear();
			mapped.ForEach([&](uint64_t from, uint64_t to) {
				from = std::max(begin, from & ~(CACHING_PAGESIZE - 1));
				to   = std::min(stop, (to + CACHING_PAGESIZE - 1) & ~(CACHING_PAGESIZE - 1));
				if (!pieces.empty() && from <= pieces.back().second + Gap) {
					pieces.back().second = std::max(pieces.back().second, to);
				} else if (from < to) {
					pieces.emplace_back(from, to);
				}
			});
			kept = 0;
			for (const auto& [from, to]: pieces) kept += to - from;
		};
		cut_pieces();
		bool over_free_blocks = false;
		if (kept * 2 > stop - begin) {
			if (free_blocks == nullptr || stop - begin < MinFreeBlockSplit) continue;
			if (!std::exchange(listed, true)) {
				have_list = free_blocks(lowest, highest, &free_list);
				if (splits != nullptr) splits->lists++;
			}
			if (!have_list) continue;
			std::vector<std::pair<uint64_t, uint64_t>> over;
			for (const auto& [address, bytes]: free_list) {
				const auto from = std::max(begin, address), to = std::min(stop, address + bytes);
				if (from < to) over.emplace_back(from, to - from);
			}
			if (over.empty()) continue;
			for (const auto& [address, bytes]: over) mapped.Subtract(address, bytes);
			cut_pieces();
			if (kept * 2 > stop - begin) continue;
			for (const auto& [address, bytes]: over) InvalidateMemory(address, bytes);
			if (m_slot_buffers[id].is_deleted) continue;
			over_free_blocks = true;
		}
		bool     owned_outside = false;
		uint64_t at            = begin;
		for (const auto& [from, to]: pieces) {
			owned_outside = owned_outside || (from > at && m_gpu_modified_ranges.Intersects(at, from - at));
			at            = to;
		}
		if (owned_outside || (stop > at && m_gpu_modified_ranges.Intersects(at, stop - at))) continue;
		std::vector<BufferId> ids;
		for (const auto& [from, to]: pieces) {
			ids.push_back(m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, from,
			                                    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, to - from));
		}
		const auto& source = m_slot_buffers[id];
		for (size_t i = 0; i < ids.size(); ++i) {
			auto&      piece      = m_slot_buffers[ids[i]];
			const auto [from, to] = pieces[i];
			piece.device_address_pinned |= source.device_address_pinned;
			SetVulkanObjectNameF(m_graphics.device, piece.Handle(), "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", from,
			                     to - from);
			piece.CopyFrom(m_scheduler.Current(), source, from - begin, 0, to - from);
			NoteWrite(from, to - from);
		}
		cut += stop - begin - kept;
		if (over_free_blocks && splits != nullptr) splits->cut += stop - begin - kept;
		DeleteBuffer(id);
		for (const auto piece: ids) Register(piece);
	}
	return cut;
}

uint64_t BufferCache::DropFreedRange(uint64_t vaddr, uint64_t size, bool read_back, FreeBlocksQuery free_blocks,
                                     FreeBlockSplits* splits) {
	if (size == 0 || UINT64_MAX - vaddr < size) return 0;
	ForgetCommandBuffers(vaddr, size);
	if (!IsRegionRegistered(vaddr, size)) return 0;
	if (read_back) {
		InvalidateMemory(vaddr, size);
	} else {
		DiscardGpuWrites(vaddr, size);
	}
	const auto deleted = DeleteUnmappedBuffers(vaddr, size);
	return deleted + SplitUnmappedBuffers(vaddr, size, free_blocks, splits);
}

void BufferCache::CollectMappedRegisteredRanges(const RangeSet& mapped, uint64_t begin, uint64_t end,
                                                std::vector<std::pair<uint64_t, uint64_t>>& ranges) const {
	ranges.clear();
	auto it = m_buffers.upper_bound(begin);
	if (it != m_buffers.begin()) --it;
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto first = std::max(it->first, begin), last = std::min(it->first + m_slot_buffers[it->second].Size(), end);
		if (first >= last) continue;
		mapped.ForEachInRange(first, last - first, [&](uint64_t from, uint64_t to) {
			if (!ranges.empty() && ranges.back().first + ranges.back().second == from)
				ranges.back().second += to - from;
			else ranges.emplace_back(from, to - from);
		});
	}
}

bool BufferCache::TakeRegistrationSpans(std::vector<GuestRange>& spans) {
	spans.insert(spans.end(), m_registration_spans.begin(), m_registration_spans.end());
	m_registration_spans.clear();
	return !std::exchange(m_registration_spans_lost, false);
}

void BufferCache::SynchronizeRegionRequest(SyncRegionRequest& request) {
	const auto epoch = CpuModificationEpoch(request.address, request.size);
	const auto registered = RegionRegistrationEpoch(request.address);
	if (epoch != 0 && request.cpu_epoch == epoch && request.registration_epoch == registered) {
		return;
	}
	SynchronizeBuffersInRange(request.address, request.size);
	request.cpu_epoch = 0;
	if (epoch != 0 && RegionRegistrationEpoch(request.address) == registered &&
	    CpuModificationEpoch(request.address, request.size) == epoch) {
		request.cpu_epoch          = epoch;
		request.registration_epoch = registered;
	}
}

void BufferCache::EnsureBdaReadable(std::span<const GuestRange> ranges) {
	constexpr uint64_t HeadBytes = uint64_t {64} << 10u;
	const auto         held      = [this](uint64_t page) {
        const auto* owner = m_page_table.Find(page);
        return owner != nullptr && *owner;
	};
	for (const auto& range: ranges) {
		const auto last = (range.address + std::min(range.size, HeadBytes) - 1) >> PageTable::kPageBits;
		for (auto page = range.address >> PageTable::kPageBits; page <= last;) {
			if (held(page)) {
				++page;
				continue;
			}
			auto end = page + 1;
			while (end <= last && !held(end)) ++end;
			const auto begin = page << PageTable::kPageBits, bytes = (end - page) << PageTable::kPageBits;
			if (m_resources != nullptr && m_resources->IsMapped(begin, bytes) && LibKernel::Memory::IsFullyMapped(begin, bytes)) {
				auto& buffer = m_slot_buffers[FindBuffer(begin, bytes)];
				TouchBuffer(buffer);
				(void)SynchronizeBuffer(buffer, begin, bytes, false, false);
			}
			page = end;
		}
	}
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	CollectStaleBuffers(tick);
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(
			    !DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()) &&
			    InFlightReadbackTick(buffer.CpuAddress(), buffer.CpuAddress() + buffer.Size()) == 0);
			DiagArgsSyncNoteBuffer(14, buffer.CpuAddress(), buffer.Size(), 0);
			dirty_buffers.push_back(id);
		} else {
			DiagArgsSyncNoteBuffer(13, buffer.CpuAddress(), buffer.Size(), 0);
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			if (m_record_released) {
				m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
			}
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		UnmarkGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_record_released) {
			m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
		}
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::ProcessUseBuffer() {
	m_fault_manager.ProcessUseBuffer([this](std::span<const uint64_t> pages) {
		for (const auto page: pages) {
			const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
			if (owner != nullptr && *owner) {
				TouchBuffer(m_slot_buffers[*owner]);
			}
		}
	});
}

void BufferCache::NoteGpuWrites(uint64_t vaddr, uint64_t size) {
	m_gpu_write_generation++;
	if (ArgsSyncDiag()) {
		SyncRingRecordWrite(4, vaddr, size, nullptr);
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid GPU write range\n");
	}
	m_scheduler.Context().CollectHotPageWrites(vaddr, size);
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		const auto  start  = std::max(buffer.CpuAddress(), vaddr);
		const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start >= finish) {
			continue;
		}
		TouchBuffer(buffer);
		NoteWrite(start, finish - start);
		m_memory_tracker.MarkRegionAsGpuModifiedUnlessCpuDirty(
		    start, finish - start, [&](uint64_t page, uint64_t bytes) {
			    const auto first = std::max(page, start);
			    const auto last  = std::min(page + bytes, finish);
			    if (m_gpu_modified_ranges.Add(first, last - first)) {
				    NoteGpuModifiedChange(first, last - first, true);
			    }
		    });
	}
}

void BufferCache::FlushUploadBatch() {
	m_upload_batching = false;
	if (m_upload_batch_buffers.empty()) {
		return;
	}
	EXIT_IF(m_scheduler.CurrentTick() != m_upload_batch_tick);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	if (m_upload_batch_buffers.size() > 1 && ScatterUploadBatch(native)) {
		m_upload_batch_buffers.clear();
		m_upload_batch_copies.clear();
		return;
	}
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion, 1,
	                       &before, 0, nullptr, 0, nullptr);
	const auto staging = m_staging_buffer.Handle();
	size_t     first   = 0;
	for (const auto& batch: m_upload_batch_buffers) {
		native.copyBuffer(staging, batch.buffer, batch.count, m_upload_batch_copies.data() + first);
		first += batch.count;
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eByRegion,
	                       1, &after, 0, nullptr, 0, nullptr);
	m_upload_batch_buffers.clear();
	m_upload_batch_copies.clear();
}

bool BufferCache::ScatterUploadBatch(const CommandRecorder& native) {
	struct Job {
		uint64_t source;
		uint64_t destination;
		uint32_t size;
		uint32_t pad0;
		uint64_t pad1;
	};
	static_assert(sizeof(Job) == 32);
	constexpr uint64_t ScatterJobBytes = 4 * 1024;
	constexpr uint32_t MaxJobs         = 65535;
	const auto         staging         = m_staging_buffer.BufferDeviceAddress();
	uint64_t           job_count = 0;
	size_t             first     = 0;
	for (const auto& batch: m_upload_batch_buffers) {
		if (batch.address == 0 || staging == 0) {
			return false;
		}
		for (uint32_t i = 0; i < batch.count; i++) {
			const auto& copy = m_upload_batch_copies[first + i];
			if (((copy.srcOffset | copy.dstOffset | copy.size) & 3u) != 0) {
				return false;
			}
			job_count += (copy.size + ScatterJobBytes - 1) / ScatterJobBytes;
		}
		first += batch.count;
	}
	if (job_count == 0 || job_count > MaxJobs) {
		return false;
	}
	const auto bytes            = job_count * sizeof(Job);
	const auto [mapped, offset] = m_staging_buffer.Map(bytes, 16, false);
	if (mapped == nullptr) {
		return false;
	}
	uint8_t* job = mapped;
	first        = 0;
	for (const auto& batch: m_upload_batch_buffers) {
		for (uint32_t i = 0; i < batch.count; i++) {
			const auto& copy = m_upload_batch_copies[first + i];
			for (uint64_t done = 0; done < copy.size; done += ScatterJobBytes) {
				const Job value {staging + copy.srcOffset + done, batch.address + copy.dstOffset + done,
				                 static_cast<uint32_t>(std::min(ScatterJobBytes, copy.size - done)), 0, 0};
				std::memcpy(job, &value, sizeof(value));
				job += sizeof(value);
			}
		}
		first += batch.count;
	}
	m_staging_buffer.Commit();
	if (!m_scatter_pipeline) {
		const auto                   device = m_graphics.device;
		const vk::PushConstantRange  push {vk::ShaderStageFlagBits::eCompute, 0, 16};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(device.createPipelineLayout(&layout_info, nullptr, &m_scatter_layout),
		                     "create buffer scatter pipeline layout");
		const auto                        module = CompileSPV(BUFFER_SCATTER_SPV, device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = m_scatter_layout;
		const auto result    = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_scatter_pipeline);
		device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create buffer scatter pipeline");
		SetVulkanObjectNameF(device, m_scatter_pipeline, "Kyty.BufferScatter");
	}
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eByRegion, 1,
	                       &before, 0, nullptr, 0, nullptr);
	struct Push {
		uint64_t jobs;
		uint32_t job_count;
		uint32_t pad;
	} push {staging + offset, static_cast<uint32_t>(job_count), 0};
	native.bindPipeline(vk::PipelineBindPoint::eCompute, m_scatter_pipeline);
	native.pushConstants(m_scatter_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	native.dispatch(push.job_count, 1, 1);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                       vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eByRegion,
	                       1, &after, 0, nullptr, 0, nullptr);
	return true;
}

struct BufferCache::UploadBatch {
	BufferCache& cache;
	explicit UploadBatch(BufferCache& owner): cache(owner) {
		cache.m_upload_batching   = true;
		cache.m_upload_batch_tick = cache.m_scheduler.CurrentTick();
	}
	~UploadBatch() { cache.FlushUploadBatch(); }
	UploadBatch(const UploadBatch&)            = delete;
	UploadBatch& operator=(const UploadBatch&) = delete;
};

void BufferCache::SynchronizeBuffersInSpans(std::span<const std::pair<uint64_t, uint64_t>> spans) {
	const UploadBatch batch {*this};
	m_sync_spans.assign(spans.begin(), spans.end());
	std::ranges::sort(m_sync_spans);
	size_t merged = 0;
	for (const auto& span: m_sync_spans) {
		if (merged != 0 && span.first <= m_sync_spans[merged - 1].second) {
			m_sync_spans[merged - 1].second = std::max(m_sync_spans[merged - 1].second, span.second);
		} else {
			m_sync_spans[merged++] = span;
		}
	}
	m_sync_spans.resize(merged);
	m_span_buffers.clear();
	m_span_ranges.clear();
	RegionBits dirty;
	uint64_t   dirty_region = UINT64_MAX;
	bool       region_dirty = false;
	for (const auto& [first, last]: m_sync_spans) {
		for (auto at = first; at < last;) {
			const auto index = at / TRACKER_REGION_SIZE;
			const auto base  = index * TRACKER_REGION_SIZE;
			const auto end   = std::min(last, base + TRACKER_REGION_SIZE);
			if (index != dirty_region) {
				dirty_region = index;
				region_dirty = m_memory_tracker.CopyRegionCpuDirty(index, dirty);
			}
			const auto end_page = (end - base + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
			for (auto page = (at - base) / TRACKER_PAGE_SIZE; region_dirty && page < end_page;) {
				const auto [run, after] = dirty.FirstRangeFrom(page);
				if (run >= end_page) {
					break;
				}
				const auto begin  = std::max(base + run * TRACKER_PAGE_SIZE, at);
				const auto finish = std::min(base + std::min<uint64_t>(after, end_page) * TRACKER_PAGE_SIZE, end);
				CollectSpanBuffers(begin, finish - begin, m_span_buffers, m_span_ranges);
				page = after;
			}
			at = end;
		}
	}
	if (!m_span_buffers.empty()) {
		SynchronizeBuffers(m_span_buffers, m_span_ranges);
	}
}

void BufferCache::GrowRegionTables(uint64_t region) {
	if (region < m_region_buffer_count.size()) {
		return;
	}
	const auto words = region / 64 + 1;
	m_region_buffer_count.resize(words * 64, 0);
	m_synced_region_epochs.resize(words * 64, 0);
	m_buffer_region_bits.resize(words, 0);
	m_forced_region_bits.resize(words, 0);
	m_region_forces.resize(words * 64, 0);
}

void BufferCache::NoteBufferRegions(const Buffer& buffer, bool insert) {
	const auto first = buffer.CpuAddress() / TRACKER_REGION_SIZE;
	const auto last  = (buffer.CpuAddress() + buffer.Size() - 1) / TRACKER_REGION_SIZE;
	GrowRegionTables(last);
	for (auto index = first; index <= last; index++) {
		const auto bit = uint64_t {1} << (index % 64);
		if (insert) {
			m_region_buffer_count[index]++;
			m_buffer_region_bits[index / 64] |= bit;
			m_forced_region_bits[index / 64] |= bit;
			m_region_forces[index]++;
		} else {
			EXIT_IF(m_region_buffer_count[index] == 0);
			if (--m_region_buffer_count[index] == 0) {
				m_buffer_region_bits[index / 64] &= ~bit;
			}
		}
	}
}

void BufferCache::ForceSynchronization(uint64_t vaddr, uint64_t size) {
	if (size == 0 || m_buffer_region_bits.empty()) {
		return;
	}
	const auto first = vaddr / TRACKER_REGION_SIZE;
	const auto last  = std::min<uint64_t>((vaddr + size - 1) / TRACKER_REGION_SIZE, m_region_buffer_count.size() - 1);
	for (auto index = first; index <= last; index++) {
		m_forced_region_bits[index / 64] |= uint64_t {1} << (index % 64);
		m_region_forces[index]++;
	}
}

void BufferCache::AddToSet(RangeSet& set, SetRegions& state, uint64_t address, uint64_t size) {
	const bool covered = state.version == set.Version();
	if (!set.Add(address, size)) {
		return;
	}
	const auto end = address + size;
	for (auto index = address / TRACKER_REGION_SIZE; index <= (end - 1) / TRACKER_REGION_SIZE; index++) {
		if (index < state.regions.size()) {
			state.regions[index] = {};
		}
	}
	if (!covered) {
		return;
	}
	for (auto at = address; at < end;) {
		const auto index    = at / TRACKER_REGION_SIZE;
		const auto base     = index * TRACKER_REGION_SIZE;
		const auto stop     = std::min(end, base + TRACKER_REGION_SIZE);
		const auto found    = std::ranges::lower_bound(state.covered, index);
		const auto position = static_cast<size_t>(found - state.covered.begin());
		if (found == state.covered.end()) {
			state.covered.push_back(index);
			state.covered_pages.emplace_back();
		} else if (*found != index) {
			return;
		}
		state.covered_pages[position].SetRange((at - base) / TRACKER_PAGE_SIZE,
		                                       (stop - base + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE);
		at = stop;
	}
	state.version = set.Version();
}

void BufferCache::SynchronizeBuffersInSet(const RangeSet& set, SetRegions& state) {
	const UploadBatch batch {*this};
	const auto sync = ++m_device_syncs;
	m_device_sync_pass = sync;
	struct EndPass {
		BufferCache& cache;
		~EndPass() {
			cache.m_device_sync_pass = 0;
			cache.m_kept_pages       = cache.m_memory_tracker.TakeKeptPages() || cache.m_kept_pages;
		}
	} end_pass {*this};
	auto& regions = state.regions;
	if (regions.size() < m_region_buffer_count.size()) {
		regions.resize(m_region_buffer_count.size());
	}
	if (state.version != set.Version()) {
		state.version = set.Version();
		state.covered.clear();
		state.covered_pages.clear();
		set.ForEach([&](uint64_t begin, uint64_t end) {
			for (auto at = begin; at < end;) {
				const auto index = at / TRACKER_REGION_SIZE;
				const auto base  = index * TRACKER_REGION_SIZE;
				const auto stop  = std::min(end, base + TRACKER_REGION_SIZE);
				if (state.covered.empty() || state.covered.back() != index) {
					state.covered.push_back(index);
					state.covered_pages.emplace_back();
				}
				state.covered_pages.back().SetRange((at - base) / TRACKER_PAGE_SIZE,
				                                    (stop - base + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE);
				at = stop;
			}
		});
	}
	m_set_visits.clear();
	size_t covered = 0;
	for (size_t word = 0; word < m_buffer_region_bits.size(); word++) {
		for (auto bits = m_buffer_region_bits[word]; bits != 0; bits &= bits - 1) {
			const auto index = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
			while (covered < state.covered.size() && state.covered[covered] < index) {
				covered++;
			}
			if (covered != state.covered.size() && state.covered[covered] == index) {
				m_set_visits.emplace_back(index, covered);
			}
		}
	}
	const auto visit = [&](const std::pair<uint64_t, size_t>& region, SetChunk& chunk) {
		const auto index = region.first;
		const auto epoch  = m_memory_tracker.RegionCpuDirtyEpoch(index);
		const auto forces = m_region_forces[index] + 1;
		auto&      synced = regions[index];
		const auto kept   = m_memory_tracker.RegionKeptSync(index);
		if (synced.second == forces && (epoch == 0 || synced.first == epoch) &&
		    (kept == 0 || kept < state.pass)) {
			return;
		}
		RegionBits dirty;
		if (m_memory_tracker.CopyRegionCpuDirty(index, dirty)) {
			dirty &= state.covered_pages[region.second];
			const auto base = index * TRACKER_REGION_SIZE;
			for (size_t page = 0; page < TRACKER_REGION_PAGES;) {
				const auto [first, after] = dirty.FirstRangeFrom(page);
				if (first >= TRACKER_REGION_PAGES) {
					break;
				}
				set.ForEachInRange(base + first * TRACKER_PAGE_SIZE, (after - first) * TRACKER_PAGE_SIZE,
				                   [&](uint64_t start, uint64_t end) {
					                   CollectSpanBuffers(start, end - start, chunk.buffers, chunk.ranges);
				                   });
				page = after;
			}
		}
		synced = {epoch, forces};
	};
	constexpr uint32_t Grain  = 8;
	const auto         count  = static_cast<uint32_t>(m_set_visits.size());
	const auto         chunks = (count + Grain - 1) / Grain;
	if (m_set_chunks.size() < chunks) {
		m_set_chunks.resize(chunks);
	}
	GetExecHelpers().ParallelFor(count, Grain, [&](uint32_t begin, uint32_t end) {
		auto& chunk = m_set_chunks[begin / Grain];
		chunk.buffers.clear();
		chunk.ranges.clear();
		for (auto visited = begin; visited < end; visited++) {
			visit(m_set_visits[visited], chunk);
		}
	});
	m_span_buffers.clear();
	m_span_ranges.clear();
	for (uint32_t chunk = 0; chunk < chunks; chunk++) {
		const auto& collected = m_set_chunks[chunk];
		m_span_buffers.insert(m_span_buffers.end(), collected.buffers.begin(), collected.buffers.end());
		m_span_ranges.insert(m_span_ranges.end(), collected.ranges.begin(), collected.ranges.end());
	}
	if (!m_span_buffers.empty()) {
		SynchronizeBuffers(m_span_buffers, m_span_ranges);
	}
	state.pass = sync;
}

void BufferCache::SynchronizeBuffersInRanges(const RangeSet& ranges, bool all) {
	const UploadBatch batch {*this};
	const auto sync    = ++m_device_syncs;
	m_device_sync_pass = sync;
	(void)m_memory_tracker.TakeKeptPages();
	struct EndPass {
		BufferCache& cache;
		~EndPass() {
			cache.m_device_sync_pass = 0;
			cache.m_kept_pages       = cache.m_memory_tracker.TakeKeptPages();
		}
	} end_pass {*this};
	m_sync_regions.clear();
	for (size_t word = 0; word < m_buffer_region_bits.size(); word++) {
		const auto forced = m_forced_region_bits[word];
		m_forced_region_bits[word] = 0;
		for (auto bits = m_buffer_region_bits[word]; bits != 0; bits &= bits - 1) {
			const auto index = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
			const auto epoch = m_memory_tracker.RegionCpuDirtyEpoch(index);
			const auto kept = m_memory_tracker.RegionKeptSync(index);
			if (!all && (forced & (uint64_t {1} << (index % 64))) == 0 &&
			    (epoch == 0 || m_synced_region_epochs[index] == epoch) &&
			    (kept == 0 || kept < m_device_full_pass)) {
				continue;
			}
			m_sync_regions.emplace_back(index, epoch);
		}
	}
	for (const auto& [index, epoch]: m_sync_regions) {
		ranges.ForEachInRange(index * TRACKER_REGION_SIZE, TRACKER_REGION_SIZE,
		                      [this](uint64_t start, uint64_t end) {
			                      SynchronizeBuffersInRange(start, end - start);
		                      });
		m_synced_region_epochs[index] = epoch;
	}
	m_device_full_pass = sync;
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t MaxGap = 32 * TRACKER_PAGE_SIZE;
	m_memory_tracker.ForEachCpuDirtySpan(vaddr, size, MaxGap, [&](uint64_t span, uint64_t bytes) {
		m_span_buffers.clear();
		m_span_ranges.clear();
		CollectSpanBuffers(span, bytes, m_span_buffers, m_span_ranges);
		if (m_span_buffers.size() == 1) {
			(void)SynchronizeBuffer(*m_span_buffers[0], m_span_ranges[0].first,
			                        m_span_ranges[0].second, false, false);
		} else if (!m_span_buffers.empty()) {
			SynchronizeBuffers(m_span_buffers, m_span_ranges);
		}
	});
}

void BufferCache::CollectSpanBuffers(uint64_t span, uint64_t bytes, std::vector<Buffer*>& buffers,
                                     std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	const auto end = span + bytes;
	auto       it  = m_buffers.upper_bound(span);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), span);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			buffers.push_back(&buffer);
			ranges.emplace_back(start, finish - start);
		}
	}
}

}
