#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRIVER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRIVER_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/Submission.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/DeviceAccess.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Packets/PacketHistory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchTiming.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawTiming.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/BufferCopy.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/IndirectDispatch.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Packets/PacketTiming.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stop_token>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ShaderRecompiler {
struct RunTimeProgram;
}

namespace AgcDriver::DriverDetail {

bool CacheableResult(const ShaderRecompiler::RecompileResult& compiled);

class Driver {
public:
    static Driver& Get();
    static std::uint64_t GpuMapImportThreads();
    ~Driver();
    void Shutdown();
    void WaitIdle();
    void CheckFailure();
    void ReportFailure(std::exception_ptr error);
    void Submit(const Packet* packet, std::uint32_t queue);
    void SuspendPoint();
    std::uint32_t ReserveEopFlip(const FlipInfo& info);
    void RegisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output);
    void UnregisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output);
    void AttachWindow(const PresentationWindow& window);
    void Present(const PresentationWindow& window, const DisplayBuffer* buffer, bool opaque, void (*gpuReady)(void*), void* context);
    void ReleaseWindow(void* window);
    void RegisterShader(const Shader* shader);
    void ResolveShaderAbi(const Shader* shader, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive);
    void ResolveGraphicsStagesAbi(std::span<const Shader* const> stages, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive);
    void ResolveGraphicsAbi(const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType);
    void SettleShaderPreparations(std::span<const Shader* const> stages);
    static std::optional<Graphics::IndirectDrawPath> ClassifyIndirectDraw(const ShaderRecompiler::RecompileResult& result, const Graphics::State& graphics, const DrawProgram& frontProgram, const std::shared_ptr<VulkanDevice>& localDevice, Pm4::DrawParameters& drawParameters, bool traceIndirect);
    static std::string dumpRequest(std::uint64_t address, const ShaderRecompiler::RecompileRequest& request);
    std::uint64_t DrawStageCompiles();
    static std::uint64_t TransferFills();
    static std::uint64_t TransferCopies();
    std::uint64_t DataIndependentDispatchHits();
    std::shared_ptr<const ShaderRegistry> RegisteredShaders();

private:
    friend struct PacketTimer;
    void stop();
    Driver();
    void rethrowFailure() const;
    void checkStopping() const;
    static bool& onWorkerThread();
    static void copyCommands(Submission& submission, const std::uint32_t* guest, std::size_t words);
    static bool copySegment(Submission& submission, const std::uint32_t* guest, std::size_t words, std::size_t& budget);
    static void readRegisterLists(Submission& submission);
    void waitForFlipRoom(const Submission& submission);
    void holdFlipBehindWorker(const Submission& submission);
    bool awaitsTitle(std::uint64_t awaited) const;
    void noteAwaitingTitle(std::uint32_t queue, std::uint64_t awaited);
    void releaseFlipHold();
    void reserveOutputs(Submission& submission);
    void executeRewindTail(const Submission& stalled);
    void enqueue(Submission submission);
    void noteHeldAtSubmit(Submission& submission, std::size_t cursor);
    static void forgetUnfinishedWrites(QueueWorker& worker, const Submission& submission);
    static bool waitFree(const Submission& submission);
    bool queue0Before(std::uint64_t received) const;
    bool orderReleased(std::uint32_t queue, std::uint64_t received) const;
    void noteWaitBlocked(std::uint32_t queue, std::uint64_t awaited, bool blocked);
    void reportPresents(double waitedMs, std::size_t inFlight);
    void dispatch(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::uint64_t indirectArguments = 0, bool threadDimensions = false);
    std::shared_ptr<const DrawDecode> drawDecode(const QueueState& queue, const Submission& submission, std::uint64_t deviceSerial);
    static bool drawPrecheck();
    std::optional<DrawVerdict> precheckDraw(const QueueState& queue, const Submission& submission, std::span<const std::uint32_t> packet, const Pm4::DrawParameters& drawParameters, std::string& rejected, bool& traceIndirect);
    static std::uint32_t drawUserWord(const DrawProgram& program, std::int32_t sgpr);
    DrawVerdict draw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::string& rejected);
    void addDriverPhases(DispatchClass which, const std::array<double, DriverPhaseCount>& ms, bool hit, bool validated);
    static PendingDispatchPhases& pendingDispatchPhases();
    static std::chrono::steady_clock::time_point& packetStartedAt();
    static PendingDrawPhases& pendingDrawPhases();
    void addDrawPhases(const std::array<double, DrawDriverPhaseCount>& ms, bool drawn, std::uint64_t captures);
    void noteLabelStore(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue);
    bool storedSince(std::span<const std::uint32_t> packet, std::uint64_t address, std::size_t bytes, std::uint64_t received, std::uint32_t* writer = nullptr);
    static void traceLabel(std::span<const std::uint32_t> packet, std::uint32_t queue);
    static std::array<WriteRecord, 16384>& writeHistory();
    static std::size_t& writeCursor();
    static void dumpPackets(std::span<const std::uint32_t> commands, const std::uint32_t* guest = nullptr);
    static void validate(const Submission& submission, const std::uint32_t* guest = nullptr);
    static void reportSkip(const char* kind, const std::string& what);
    static bool matchesFillKernel(std::span<const std::uint32_t> code, const std::vector<std::uint32_t>& userData, const ShaderRecompiler::ShaderComputeStageInfo& compute);
    static bool fillClearEnabled();
    static bool fillClearExactOnly();
    static void fillClearCount(const Graphics::StorageTexture::FillCoverage& coverage, std::size_t bytes, std::span<const std::uint32_t, 4> pattern, std::size_t discarded, bool cleared, const char* refusal);
    void storeBufferFill(std::uint32_t queueId, std::uint64_t base, std::size_t bytes, const std::array<std::uint32_t, 4>& pattern, const std::shared_ptr<VulkanDevice>& localDevice, bool memoryValueFill, bool noteMetadata);
    bool fillBuffer(QueueState& queue, std::uint32_t queueId, std::span<const std::uint32_t> packet, std::span<const std::uint32_t> code, const std::vector<std::uint32_t>& userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, const std::shared_ptr<VulkanDevice>& localDevice);
    static bool matchesCopyKernel(std::span<const std::uint32_t> code, const std::vector<std::uint32_t>& userData, const ShaderRecompiler::ShaderComputeStageInfo& compute);
    static void countCopy(int path, std::size_t bytes, std::chrono::steady_clock::time_point started, std::chrono::steady_clock::time_point locked, const VulkanDevice::CopyOutcome* outcome = nullptr);
    bool copyBuffer(QueueState& queue, std::uint32_t queueId, std::span<const std::uint32_t> packet, std::span<const std::uint32_t> code, const std::vector<std::uint32_t>& userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, const std::shared_ptr<VulkanDevice>& localDevice, std::uint64_t programAddress);
    bool copyRange(std::uint32_t queueId, std::uint64_t source, std::uint64_t destination, std::size_t bytes, const std::shared_ptr<VulkanDevice>& localDevice, std::uint64_t programAddress, std::chrono::steady_clock::time_point started, std::chrono::steady_clock::time_point locked);
    bool transferBuffer(std::uint32_t queueId, const ShaderRecompiler::RecompileResult& compiled, std::span<const std::uint32_t> userData, std::uint64_t threads, const std::shared_ptr<VulkanDevice>& localDevice, std::uint64_t programAddress);
    void traceCopyRefused(const VulkanDevice::CopyOutcome& outcome, std::uint64_t source, std::uint64_t destination, std::size_t bytes, const std::shared_ptr<VulkanDevice>& localDevice);
    static bool traceCopy();
    void traceCopyPending(const char* what, std::uint64_t address, std::size_t bytes, std::uint64_t source, std::uint64_t destination, std::size_t copyBytes, const std::shared_ptr<VulkanDevice>& localDevice);
    static void countIndirect(int path, double readMs);
    void dispatchIndirect(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission);
    using RunTimeProgram = ShaderRecompiler::RunTimeProgram;
    struct LinkedDispatch {
        std::shared_ptr<const RunTimeProgram> program;
        std::vector<std::uint32_t> linkedWords;
        std::vector<std::uint32_t> userData;
        std::vector<GuestMemory::UnchangedQuery> reads;
    };
    struct DispatchProgram {
        std::shared_ptr<const RunTimeProgram> program;
        std::shared_ptr<const LinkedDispatch> linked;
    };
    static bool readPrePassWord(void* waits, std::uint64_t at, std::uint32_t* value);
    static bool linkCurrent(const LinkedDispatch& linked, std::span<const std::uint32_t> userData);
    std::shared_ptr<const RunTimeProgram> runTimeDrawStage(const DrawProgram& program, const ShaderRecompiler::RecompileRequest& request, const std::shared_ptr<VulkanDevice>& device, std::uint64_t key);
    DispatchProgram runTimeDispatch(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, const std::shared_ptr<VulkanDevice>& device, std::uint64_t programKey, std::span<const std::uint32_t> userData);
    static std::shared_ptr<const ShaderRecompiler::RecompileResult> runTimeResult(const RunTimeProgram& program, std::span<const std::uint32_t> userData, std::uint64_t shaderBase, std::span<const std::uint32_t> linkedWords = {});
    static int waitTimeoutMs();
    void waitMemory(std::span<const std::uint32_t> packet, std::uint32_t queue, const PacketHistory& context, std::uint64_t received, bool heldAtSubmit);
    struct PendingWait {
        std::array<std::uint32_t, 8> words;
        std::size_t count;
        Graphics::Recorder::Token writer;
    };
    static std::vector<PendingWait>& pendingWaits();
    static void notePendingWait(std::span<const std::uint32_t> packet, std::uint32_t queue, const Graphics::Recorder::Token& writer);
    void landPendingWaits(std::uint32_t queue);
    static PollStats& pollStats();
    static WaitOutcomes& waitOutcomes();
    static EpochBumps& epochBumps();
    static void bumpEpoch(std::uint64_t EpochBumps::*counter);
    static bool packetEpoch();
    static bool labelTryEachPacket();
    static std::chrono::microseconds labelFlushDeadline();
    static bool labelFlushDue();
    static bool completionsPending();
    static bool reapEachPacket();
    static std::uint64_t batchCap();
    void recordDeferredLabels(VulkanDevice* localDevice, std::uint32_t queue);
    bool recordLabelsForPacket(VulkanDevice* localDevice, std::uint32_t queue);
    void recordQueuedLabelsBeforeRead(std::uint32_t queue);
    bool recordQueuedLabelsAfterPrePass(std::uint32_t queue);
    static std::vector<std::uint64_t>& prePassReads();
    void noteQueuedLabels(std::uint32_t queue);
    void recordQueuedLabelsByTry(std::uint32_t queue, std::atomic<std::uint64_t>& counter);
    static void recordQueuedLabelsFromHook();
    void flushBetweenPackets(std::uint32_t queue, std::uint32_t header, bool labelPacket);
    bool preparePacketMemory(const Submission& submission, QueueState& queue, std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode, bool& wroteOnGpu, bool& endOfPipeInterrupt, bool& interruptDeferred, bool& drawPacket, bool& sampleDump);
    void dumpSampleCounters(std::uint64_t address);
    template <typename TWork>
    static void timed(double WorkerProfile::*bucket, TWork&& work);
    FrameTiming* frameTiming();
    FrameTiming* includeTimingSubmission(const Submission& submission, bool firstSegment);
    void execute(const Submission& submission);
    void markCompleted(std::uint64_t serial);
    static const std::atomic<std::uint64_t>*& workerQueued();
    void reapCompletionLabels();
    void run(std::uint32_t id) noexcept;

    std::mutex mutex;
    std::mutex shutdownMutex;
    std::condition_variable changed;

    std::map<std::uint32_t, QueueWorker> workers;
    std::uint64_t frameSerial = 0;
    std::shared_ptr<FrameTiming> pendingFrameTiming;

    std::atomic<std::uint64_t> flipsCounted{0};
    std::atomic<std::uint64_t> flipSerial{0};
    std::atomic<std::uint64_t> flipBatchesUnsignaled{0};

    alignas(64) std::atomic<int> packetsInFlight{0};
    std::atomic<std::uint64_t> packetsDone{0};

    std::atomic<std::uint64_t> eventSerial{0};
    alignas(64) std::shared_ptr<ShaderRegistry> shaders = std::make_shared<ShaderRegistry>();

    std::mutex dispatchCacheMutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<const RunTimeProgram>> runTimeDispatches;
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<const LinkedDispatch>>> linkedDispatches;
    std::uint64_t dataIndependentHits = 0;
    std::uint64_t runTimeCompiles = 0;




    std::mutex drawCacheMutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<const DrawDecode>> drawDecodes;
    struct DrawStage {
        std::shared_ptr<const RunTimeProgram> program;
        std::shared_ptr<const ShaderSnapshot> snapshot;
        std::shared_ptr<const ShaderSnapshot> continuation;
    };
    std::unordered_map<std::uint64_t, DrawStage> drawStages;
    std::uint64_t drawStageCompiles = 0;

    std::mutex driverPhasesMutex;
    std::array<DriverPhaseTotals, DispatchClassCount> driverPhaseTotals{};
    std::chrono::steady_clock::time_point driverPhasesReport = std::chrono::steady_clock::now();

    std::mutex drawPhasesMutex;
    DrawPhaseTotals drawPhaseTotals;

    std::map<std::uint32_t, QueueState> queues;
    std::map<std::uint32_t, std::shared_ptr<IVideoOutput>> outputs;
    struct EopFlip {
        std::uint32_t handle;
        std::shared_ptr<IFlipRequest> request;
    };
    std::map<std::uint32_t, EopFlip> eopFlips;
    std::uint32_t nextEopToken = EopFlipTokenBase;
    DevicePointer device;

    std::mutex labelStoresMutex;
    std::unordered_map<std::uint64_t, std::array<LabelStore, LabelStoreHistory>> labelStores;

    std::vector<std::shared_ptr<VulkanDevice>> replacedDevices;
    std::stop_token shutdownToken = LibcShutdownToken_nid_postfix();
    DeviceUseGate deviceReplacement;
    std::uint64_t accepted = 0;
    std::uint64_t completed = 0;
    std::set<std::uint64_t> completedOutOfOrder;
    std::exception_ptr failure;

    std::atomic<bool> failed{false};
    std::atomic<bool> stopping{false};
    bool stopped = false;
    bool resetGraphics = false;

    std::uint32_t idleWaiters = 0;
    std::uint64_t queue0Executing = 0;
    std::atomic<std::uint32_t> orderHolders{0};
    std::atomic<std::uint32_t> runningWorkers{0};
    std::atomic<std::uint64_t> idleEpoch{0};
    std::atomic<std::uint64_t> queue0Awaited{0};
    std::atomic<std::uint32_t> flipsAhead{0};
    std::atomic<std::uint32_t> flipHolders{0};
    std::atomic<bool> queue0AwaitsTitle{false};





};

}

#endif
