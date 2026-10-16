#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_VULKANDEVICE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_VULKANDEVICE_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/ComputePipelineCapture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace ShaderRecompiler {
struct SourceHandle;
}

namespace AgcDriver {

class PipelineReplay;
class RecordUse;

namespace Graphics {
class StorageTexture;
}

// Stage A of a dispatch's resource build (see VulkanDevice::PrepareDispatch); opaque to callers.
struct PreparedDispatch;
struct RecordedDispatch;
struct ComputePipelineObjects;

class VulkanDevice {
public:
    explicit VulkanDevice(const PresentationWindow* window = nullptr);
    struct CompileOnly {};
    explicit VulkanDevice(CompileOnly);
    ComputePipelineReplayResult ReplayComputePipeline(const ComputePipelineCapture& capture, std::span<const std::byte> initialCache = {});
    ~VulkanDevice();
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    std::string DeviceName() const;
    std::string DriverInfo() const;
    bool CalibrateClock(double& gpuNs, std::chrono::steady_clock::time_point& cpu) const;
    ShaderRecompiler::SpirvTarget Target() const;
    std::optional<ShaderRecompiler::GeometryStageLimits> GeometryLimits() const;
    const std::uint32_t* WorkGroupCountLimit() const;
    std::uint32_t VendorId() const;
    ShaderRecompiler::SpirvTarget ComputeTarget(std::uint32_t waveSize) const;
    // Distinguishes this device from every earlier one in the process (a memo keyed by the device
    // cannot rely on the pointer, which a replacement may reuse).
    std::uint64_t Serial() const { return serial; }
    VkDevice Device() const;
    // Drains everything under the caller's GpuMutex: recorder Sync plus vkDeviceWaitIdle. For suspend,
    // resize, device replacement, CPU fill fallback and APS5_DRAIN_ALL; the packet-loop drains use the
    // three-step form below so the GPU wait happens without the mutex.
    void WaitIdle();
    void PrepareForReplacement();
    bool ImportGuestMemory(const GuestAllocations::Mapped& ranges, std::uint64_t generation, bool adoptDevice);
    // Sends recorded work to the GPU without waiting for it. With `reapFirst` it first retires batches
    // that already finished, so the in-flight list stays short (APS5_NO_OPPORTUNISTIC_REAP=1 skips
    // that). A reap runs completion actions, and a write-back can wait for a later batch under the
    // mutex through the flush hook, so a thread submitting on another queue's behalf (a WAIT_REG_MEM
    // poller, a compute worker's between-packet flush) passes false: it must only submit, never wait.
    void SubmitRecorded(bool reapFirst = true);
    void StoreIdleResults(bool reapFirst = true);
    // Unlocked drain: SubmitAndEpoch (under the mutex) returns the serial covering every recorded
    // batch (0: nothing in flight); WaitRecorded(serial) waits for it WITHOUT the mutex (the caller
    // holds a shared_ptr to this device); ReapRecorded(serial) (under the mutex) runs the completions
    // of the batches up to it. CanWaitUnlocked is false without timeline semaphores: use WaitIdle.
    bool CanWaitUnlocked() const;
    std::uint64_t SubmitAndEpoch();
    void WaitRecorded(std::uint64_t serial);
    void ReapRecorded(std::uint64_t serial);
    // Completes the batches that already finished (non-blocking); under the mutex.
    void ReapRecorded();
    // Records a store of `bytes` at a guest address behind the recorded work, so the value appears
    // once that work completed; the batch is submitted by the queue worker's flush rules (or at once
    // with APS5_LABEL_SUBMIT_NOW=1). `stamp` is the record-order stamp and `queue` the recording
    // queue for the recorder's pending-label table. Returns 0 when recorded on the GPU, 5 when kept
    // as a completion action behind pending write-backs, else why the CPU must write it:
    // 1 nothing recorded (the write is already ordered), 2 write-backs pending and 3 memory not
    // imported, 4 unsuitable size or alignment.
    // `reapFirst` retires finished batches before the checks; a caller recording a group of labels
    // under one lock passes it for the first label only.
    int WriteLabelOnGpu(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue, bool reapFirst = true);
    bool AfterRecordedWork(std::function<void()> action, bool reapFirst);
    // Pending-label table lookup and open-batch overlap test for WAIT_REG_MEM (see Recorder).
    std::optional<Graphics::Recorder::LabelHit> PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, const Graphics::Recorder::LabelAccepts& accepts, bool* behindCompletion = nullptr) const;
    bool LabelPendingAt(std::uint64_t address) const;
    Graphics::Recorder::Token LabelWriterBatches(std::uint64_t address, std::size_t bytes) const;
    bool TokenSignaled(const Graphics::Recorder::Token& token) const;
    bool OpenWriteOverlaps(std::uint64_t address, std::size_t bytes) const;
    bool RecordedWritesSettled(std::uint64_t address, std::size_t bytes) const;
    bool SeparateQueues() const;
    void OrderAfterLabel(std::uint64_t address, std::size_t bytes);
    void OrderAfterValue(std::uint64_t address, std::size_t bytes);
    void OrderAfterQueue(std::uint32_t queue);
    void OrderAfterBatch(std::uint32_t stream, std::uint64_t value);
    bool StoresPendingOver(std::uint64_t address, std::size_t bytes) const;
    // Fills [address, address + bytes) of host-imported guest memory with a repeating 16-byte pattern,
    // recorded behind the open batch; false when the range is not imported (the caller stores it).
    // Recorded GPU stores over the range are ordered before the fill by its barrier; finished
    // batches are retired first, and it waits only for stores a batch's completion makes on the
    // CPU (a copied buffer's write-back, a deferred label). Debug aid: APS5_FILL_SYNC=1 waits for
    // every recorded store over the range.
    bool FillBuffer(std::uint64_t address, std::size_t bytes, std::span<const std::uint32_t, 4> pattern);
    bool DumpSamplesOnGpu(std::uint64_t address);
    // Copies `bytes` of guest memory from `source` to `destination` (disjoint ranges) in place of the
    // engine's memcpy kernel (Driver.cpp copyBuffer). `path` 0: copied on the CPU at once, when
    // every test of the rule holds (each a pure query, nothing flushed or recorded before the
    // decision): at most `cpuMax` bytes; no storage image with results pending over either range;
    // the source settled (no recorded write, or every batch writing it signaled and no completion
    // store or label pending in it); no recorded write and no label (recorded or queued) in the
    // destination; and no unexecuted batch reads the destination in place (Recorder read tracking;
    // with APS5_COPY_READ_TRACKING=0 the recorder must be idle instead). The bytes are then settled:
    // later CPU readers never wait for a batch, and no GPU work sees them early. 1: recorded as a
    // transfer between the host imports behind the open batch, ordered like FillBuffer (recorded GPU
    // stores by its barrier, a completion's CPU store over either range by a wait, counted in
    // `synced`/`syncMs`), with the source noted as an in-place read, only for at most `gpuMax`
    // bytes (APS5_COPY_HLE_GPU_MAX; a larger copy answers 8 and the caller runs the kernel: its
    // dispatch costs the GPU the same as the transfer); 2: a range is not host-imported (the
    // caller runs the kernel). `reason` says which test refused the CPU path.
    // `queue` is the calling queue (the wait experiment below never waits on queue 0).
    // `noteWriter` is called for paths 1 and 3 right BEFORE the destination's pending write is
    // noted (the caller's evidence ring entry then precedes the note, as a dispatch's elements do)
    // with the destination's post-copy bytes when they are known now (path 1, at most `knownMax`
    // bytes, and the source CPU-current at the note: no recorded write, pending image, unpublished
    // shadow, pending label or copied writer over it; 0 disables) and the write-watch generation
    // MarkWritten(destination) stamped, else an empty span and 0.
    // `programAddress` keys the transfer's [gputime] range. Debug aids: APS5_COPY_SYNC=1 waits for
    // every recorded store over either range; APS5_COPY_WAIT_SOURCE=1 waits (unlocked) for a
    // source whose only unfinished writer is the oldest in-flight batch and decides again
    // (`waitedForSource`/`waitMs`); APS5_COPY_VERIFY=1 follows every CPU copy with a full sync and
    // byte compares (CopyVerifyCounts).
    struct CopyOutcome {
        // 0 a CPU copy, 1 a transfer between the host imports, 2 refused (not imported), 3 a
        // device copy between the surfaces' storage images (StorageTexture::CopyFrom; the
        // destination's write-back is deferred; APS5_NO_COPY_ALIAS=1 disables), 8 refused (over
        // `gpuMax`; Driver.cpp's CopyOverGpuMax).
        int path;
        bool synced;
        double syncMs;
        // DestinationPending: an unsignaled batch writes the destination; DestinationUnsettled:
        // every writer signaled but a completion store of one is still to land (as for the source).
        // Shadow: a unit shadow holds unpublished results over either range (the GPU path
        // publishes them; see UnitShadow.hpp).
        enum Refusal { None = 0, Size, Image, SourcePending, SourceUnsettled, DestinationPending, DestinationUnsettled, Label, Reader, Shadow, Refusals };
        int reason;
        // Path 0 taken although a batch had written the source: every such batch had signaled.
        bool sourceSettledBySignal;
        bool waitedForSource;
        double waitMs;
        // The reader that refused (reason Reader): batch serial, its queue, the reader kind
        // (Graphics::Recorder::ReadKind) and whether it is the open batch.
        std::uint64_t readerSerial;
        std::uint32_t readerQueue;
        int readerKind;
        bool readerOpen;
    };
    CopyOutcome CopyBuffer(std::uint64_t destination, std::uint64_t source, std::size_t bytes, std::size_t cpuMax, std::size_t gpuMax, std::uint64_t programAddress, std::uint32_t queue);
    // APS5_COPY_VERIFY=1 totals: CPU copies verified, whose source bytes changed over the sync (a
    // producer the rule missed), whose destination bytes were overwritten (a store the rule
    // missed), and whose destination an unsignaled batch read in place (the rule's own answer
    // re-checked without the fence shortcut).
    struct CopyVerification {
        std::uint64_t verified, sourceChanged, destinationChanged, readerMissed;
    };
    static CopyVerification CopyVerifyCounts();
    // Aliased copies (path 3): made, destination images created for them, and the transfers
    // taken instead because the source had no live image, the destination's image differed in
    // shape, or the image copy refused (keys, not imported, released).
    struct CopyAliasing {
        std::uint64_t aliased, created, noSource, shape, refused;
    };
    static CopyAliasing CopyAliasCounts();
    void ResolveMemory(std::uint64_t address, std::size_t bytes, bool writable);
    void* Window() const;
    void Resize(std::uint32_t width, std::uint32_t height);
    void SelectDynamicRange(const DisplayBuffer* buffer);
    bool Presentable() const;
    bool DualSrcBlend() const;
    bool PrimitiveListRestart() const;
    bool SamplerFilterMinmax() const;
    bool ConservativeRasterization() const;
    VkShaderStageFlags SubgroupStages() const;
    bool ProvokingVertexLast() const;
    bool GraphicsPipelineLibraries() const;
    // A presentation is a few steps so the presenter holds GuestMemory::GpuMutex only while it
    // touches the queue. Presentations are slots (FlipInFlight() + 1, each with its own command
    // buffer, fence, kept resident image and dump buffer): RetirePresents(keep) (no mutex) retires
    // the slots whose fence signaled, oldest first, and waits for the oldest ones until at most
    // `keep` are in flight (the only CPU wait of the flip path; it returns the time spent waiting);
    // AcquireImage takes the next swapchain image (no mutex needed: the swapchain and its fences
    // are the presenter's own, and in FIFO mode this is where a frame waits for the vblank);
    // PresentClear/PresentDisplayBuffer record and submit the frame into the next slot (under the
    // mutex; the blit follows the frame's batches in queue order, so the displayed image is always
    // complete) and return whether one was submitted; QueuePresent hands the image to the swapchain
    // (under the mutex). FinishPresent = RetirePresents(0): the render fence wait of the
    // synchronous paths (APS5_FLIP_INFLIGHT=0, APS5_SYNC_FLIP, PresentPixels). AcquireImage returns
    // false when the swapchain is out of date (the frame is dropped; the next Resize recreates it);
    // a present without a prior AcquireImage acquires itself, one whose slot is in flight waits for
    // it. PresentPixels does all steps itself. Retiring a slot releases its kept image and writes
    // its frame dump; WaitIdle, Resize and the destructor retire every slot before destroying what
    // an in-flight blit reads.
    // APS5_FLIP_INFLIGHT=<n>: presentations whose fence may be unsignaled when the next blit is
    // submitted (0: today's render fence wait before the flip completes, 1 default, 2; larger
    // values are refused; 0 when APS5_SYNC_FLIP is set).
    static std::size_t FlipInFlight();
    double RetirePresents(std::size_t keepInFlight);
    // Whether presenting `buffer` will use the single scaler source, staging or upload objects (a
    // GPU frame dump, the guest-memory path, a source-size change) while an in-flight blit still
    // reads them, i.e. wait for every slot: the presenter then retires them before taking the
    // mutex (the same wait under it is only the fallback when the answer changes in between).
    bool PresentWaitsForSlots(const DisplayBuffer* buffer) const;
    bool AcquireImage();
    bool PresentClear(std::uint32_t width, std::uint32_t height, bool opaque);
    void PresentPixels(std::uint32_t width, std::uint32_t height, std::span<const std::byte> pixels);
    bool PresentDisplayBuffer(const DisplayBuffer& buffer);
    double FinishPresent();
    void QueuePresent();
    // At the flip packet (under the mutex): the recorder's submissions so far and, under
    // APS5_PROFILE_DRAW, how many in flight have not signaled (the frame record's batches_at_flip).
    void FlipBatches(std::uint64_t& submissions, std::uint64_t& unsignaled) const;
    // Cumulative presentation counters for the [present] line (APS5_PROFILE_DRAW): presentations
    // submitted; over the batches the retired blits waited for: GPU busy time, idle gaps and the
    // batches whose completion record was missing; sums of the batches submitted between the flip
    // packet and the blit (ahead of it) and behind the blit (after_flip), of the last-submit and
    // blit-submit times after the flip packet; noted in-place reads checked
    // (APS5_FLIP_READ_CHECK=1) and found overwritten before the blit's fence.
    struct PresentStatistics {
        std::uint64_t presents, gpuUnread, batchesAheadOfBlit, batchesAfterFlip, readsChecked, readsOverwritten, acquireDrops;
        double gpuBusyMs, gpuGapMs, lastSubmitAfterFlipMs, blitSubmitAfterFlipMs;
    };
    static PresentStatistics PresentCounts();
    Graphics::DeviceImageTable::Counts DeviceImageCounts() const;
    std::shared_ptr<PreparedDispatch> PrepareDispatch(const ShaderRecompiler::RecompileResult& shader, std::uint64_t dispatchThreads = 0, std::uint64_t linearThreads = 0);
    // APS5_PROFILE_DRAW: the parts of a PrepareDispatch in milliseconds, in the order key, find,
    // precollect, presync, stage A (the driver's 'prepare:' rows).
    static std::span<const double, 5> PreparePhaseMs(const PreparedDispatch& prepared);
    void Dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint64_t programAddress = 0, std::shared_ptr<PreparedDispatch> prepared = nullptr);
    struct IndirectOutcome {
        int cpuReason;
        double argumentReadMs;
    };
    IndirectOutcome DispatchIndirect(const ShaderRecompiler::RecompileResult& shader, std::uint64_t arguments, std::uint64_t programAddress = 0, std::shared_ptr<PreparedDispatch> prepared = nullptr, std::optional<std::array<std::uint32_t, 3>> threadGroup = std::nullopt);
    void Draw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> shaders);
    std::optional<std::string> KnownDrawRejection(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> shaders) const;
    void ColorMetadataPass(const Graphics::ColorMetadataPass& pass);
    // The indirect draw features the device enabled, for the driver's decision whether an indirect
    // draw's records can be read by the GPU (Graphics::Draw) or must be resolved on the CPU.
    struct IndirectDrawSupport {
        bool firstInstance;
        bool multi;
        bool count;
    };
    IndirectDrawSupport DrawIndirectSupport() const;
    void EnqueueDraw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> shaders);
    bool PrebuildGraphicsLibraries(const Graphics::ShaderStages& stages, std::span<const Graphics::CompiledShader> shaders);
    bool PrepareDrawPipelines(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> shaders);
    std::uint64_t PresyncDraw(const Graphics::State& graphics);
    void PrewarmAddressSpace();
    struct PipelineCreations {
        std::uint64_t ahead = 0;
        std::uint64_t inFrames = 0;
        std::uint64_t underLock = 0;
        double aheadMs = 0;
        double inFramesMs = 0;
        double underLockMs = 0;
    };
    static PipelineCreations ComputePipelineCreations();
    bool PrepareComputePipeline(const ShaderRecompiler::RecompileResult& shader, std::uint64_t programAddress = 0);
    bool RecordComputePipeline(const ShaderRecompiler::RecompileResult& shader);
    std::size_t PrepareRecordedComputePipelines(const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::SourceHandle& handle);
    PipelineReplay* Replay() const;

private:
    VulkanDevice(const PresentationWindow* window, bool compileOnly);
    friend struct VulkanDeviceTestAccess;
    // The device's Graphics::Context: a copy of the one built at setup (its instance functions
    // resolved then, its function table filled then), or with APS5_NO_CONTEXT_CACHE=1 built anew.
    Graphics::Context graphicsContext() const;
    const Graphics::Context& readyContext() const;
    ShaderRecompiler::SpirvTarget buildTarget() const;
    Graphics::Context buildContext() const;
    // Body of Dispatch and DispatchIndirect: `arguments` 0 dispatches x, y, z groups.
    IndirectOutcome dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint64_t arguments, std::uint64_t programAddress, std::shared_ptr<PreparedDispatch> prepared, std::optional<std::array<std::uint32_t, 3>> threadGroup = std::nullopt);
    std::shared_ptr<ComputePipelineObjects> computePipeline(const ShaderRecompiler::RecompileResult& shader, std::uint64_t programAddress, bool* created = nullptr, std::shared_ptr<RecordUse> use = nullptr);
    // The stage-A pre-sync over `surfaces` (see PrepareDispatch): the serial waited for, 0 none.
    std::uint64_t presync(std::span<const std::pair<std::uint64_t, std::uint64_t>> surfaces);
    // A DISPATCH_INDIRECT's argument path (see DispatchIndirect): GPU-side from the host import, or
    // the counts read on the CPU (the record then continues as a direct dispatch).
    void decideIndirect(RecordedDispatch& record, IndirectOutcome& outcome, char* groupsText);
    // The tail of a dispatch's device call from the open batch's command buffer to the completion
    // registration: keeps, the template data refresh, barriers, bind, push, dispatch, marks.
    void recordDispatch(RecordedDispatch& record);
    bool present(std::uint32_t width, std::uint32_t height, bool opaque, std::span<const std::byte> pixels, const DisplayBuffer* display = nullptr, const std::shared_ptr<Graphics::StorageTexture>& resident = nullptr, VkFilter residentFilter = VK_FILTER_LINEAR, bool dumpFrame = false, bool residentConvert = false, const VkClearColorValue* uniform = nullptr, std::uint32_t residentLayer = 0, bool passthrough = false);
    void recreateSwapchain(std::uint32_t width, std::uint32_t height, bool force);
    void refreshSurfaceFormats();
    void setHdrMetadata();
    void ensureHdrScaler(VkFormat source);
    void useSurfaceFormat(VkSurfaceFormatKHR format);
    struct State;
    std::unique_ptr<State> state;
    std::uint64_t serial;
};

}

#endif
