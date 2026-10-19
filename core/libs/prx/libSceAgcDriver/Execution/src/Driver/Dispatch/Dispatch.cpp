#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/Dispatch/TableCallDispatch.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace AgcDriver::DriverDetail {

namespace {

struct PendingWrites {
    std::shared_ptr<const Graphics::Recorder::WriteRanges> snapshot;
    std::uint64_t generation = 0;
    bool loaded = false;

    bool Overlap(std::uint64_t address, std::size_t bytes) {
        const auto current = Graphics::Recorder::PublishGeneration();
        if (!loaded || current != generation) {
            snapshot = Graphics::Recorder::PendingWriteSnapshot();
            generation = current;
            loaded = true;
        }
        return Graphics::Recorder::SnapshotOverlaps(snapshot.get(), address, bytes);
    }
};

std::array<std::uint32_t, 3> DispatchGroups(std::span<const std::uint32_t> packet, std::uint64_t indirectArguments, const Registers& shader) {
    std::array<std::uint32_t, 3> groups{packet[1], packet[2], packet[3]};
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const auto threads = std::max(readRegister(shader, 0x207 + axis) & 0xffffu, 1u);
            groups[axis] = (groups[axis] + threads - 1) / threads;
        }
    }
    return groups;
}

}

bool CacheableResult(const ShaderRecompiler::RecompileResult& compiled) {
    return compiled.poisonedSrtReads == 0u;
}

std::vector<std::uint64_t>& Driver::prePassReads() {
    struct PrePassReadsTag {};
    return HostThreadLocal<std::vector<std::uint64_t>, PrePassReadsTag>();
}

std::uint64_t Driver::DataIndependentDispatchHits() {
    std::lock_guard cacheLock(runTimeProgramsMutex);
    return dataIndependentHits;
}

std::uint64_t Driver::RunTimeDispatchCompiles() {
    std::lock_guard cacheLock(runTimeProgramsMutex);
    return runTimeCompiles;
}

std::uint64_t Driver::DeferredDispatches() const {
    return deferredDispatches.load(std::memory_order_acquire);
}

Driver::PrePassDeferral& Driver::prePassDeferral() {
    struct PrePassDeferralTag {};
    return HostThreadLocal<PrePassDeferral, PrePassDeferralTag>();
}

bool Driver::readPrePassWord(void* waits, std::uint64_t at, std::uint32_t* value) {
    struct PendingWritesTag {};
    auto& pending = HostThreadLocal<PendingWrites, PendingWritesTag>();
    prePassReads().push_back(at);
    const bool waitForWrites = waits != nullptr && *static_cast<const bool*>(waits);
    if (pending.Overlap(at, sizeof(std::uint32_t))) {
        if (!waitForWrites) return false;
        auto& deferral = prePassDeferral();
        if (!deferral.allowed) return GuestMemory::ReadAfterPendingWrites(at, value);
        if (deferral.deferred) return false;
        auto* recorder = Graphics::Recorder::Active();
        auto resume = std::make_shared<std::atomic<bool>>(false);
        const auto wake = recorder == nullptr ? Graphics::Recorder::WriterWake::None : recorder->AfterPendingWrites(at, sizeof(std::uint32_t), [resume] {
            resume->store(true, std::memory_order_release);
            Get().changed.notify_all();
        });
        if (wake != Graphics::Recorder::WriterWake::None) {
            deferral.deferred = true;
            deferral.retry = wake == Graphics::Recorder::WriterWake::Retry;
            deferral.resume = std::move(resume);
            return false;
        }
    }
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(at), sizeof(std::uint32_t))) return false;
    std::memcpy(value, reinterpret_cast<const void*>(at), sizeof(std::uint32_t));
    return true;
}

Driver::DispatchProgram Driver::runTimeDispatch(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, const std::shared_ptr<VulkanDevice>& device, std::uint64_t programKey) {
    bool preparedAtUse = false;
    ShaderRecompiler::LinkedCode callees;
    const auto invocation = InvocationFor(snapshot, codeOffset, request, &preparedAtUse, &callees);
    const auto& prepared = invocation.Request();
    if (preparedAtUse) RecordPreparedProgram(device, snapshot, codeOffset, prepared);
    static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
    if (dumpShaders) static_cast<void>(dumpRequest(request.shader.codeAddress, request));
    const bool linked = !callees.calls.empty();
    auto compiled = invocation.MaterializeAtRunTime();
    require(compiled != nullptr, "a compute program that reads SRT words, uniform fills, image tables or bound descriptors at compile time is not implemented");
    require(CacheableResult(*compiled), "a compute program whose compile poisoned an SRT read is not implemented");
    require(std::ranges::none_of(compiled->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; }), "a compute program that binds a flattened SRT is not implemented");
    auto entry = std::make_shared<const RunTimeProgram>(RunTimeProgram{invocation.Plan(), std::move(compiled)});
    std::lock_guard cacheLock(runTimeProgramsMutex);
    ++dataIndependentHits;
    ++runTimeCompiles;
    if (!linked && entry->compiled->spirv.size() > ShaderRecompiler::GiantProgramWords) return {std::move(entry), false};
    if (!linked) return {runTimeDispatches.try_emplace(programKey, std::move(entry)).first->second, false};
    linkedPrograms[programKey].push_back(std::make_shared<const LinkedProgram>(LinkedProgram{entry, LinkedCalleesOf(request, callees.calls, std::move(callees.callees))}));
    return {std::move(entry), true};
}

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::runTimeResult(const RunTimeProgram& entry, std::span<const std::uint32_t> userData, std::uint64_t shaderBase) {
    ShaderRecompiler::SrtRuntime runtime;
    runtime.userData = userData;
    runtime.shaderBase = shaderBase;
    runtime.readMemory = &readPrePassWord;
    bool waitForWrites = true;
    auto settled = runtime;
    settled.userContext = &waitForWrites;
    return ShaderRecompiler::RunTimeResult(entry, runtime, settled);
}

void Driver::dispatch(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::uint64_t indirectArguments, bool threadDimensions) {
    PerformanceTimer timing("Driver.Dispatch");
    const auto address = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20c)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20d) & 0xffu) << 40u);
    if (address == SkippedShader() && address != 0) return;
    auto registeredShader = RegisteredProgram(*submission.shaders, address, {0});
    if (!registeredShader) registeredShader = ReadRawShader(address);
    const auto& snapshot = *registeredShader;
    require(snapshot.type == 0, "compute program refers to a non-compute shader");
    const auto userCount = (readRegister(queue.shader, 0x213) >> 1u) & 0x1fu;
    std::vector<std::uint32_t> userData;
    for (std::uint32_t i = 0; i < userCount; ++i) {
        userData.push_back(readUserData(queue.shader, 0x240 + i));
    }
    auto compute = Graphics::DecodeComputeStageInfo(queue.shader, snapshot.header);
    std::vector<ShaderRecompiler::MemoryRegion> memory{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}};
    if (!snapshot.header.empty()) memory.push_back({snapshot.headerAddress, snapshot.header});

    std::shared_ptr<VulkanDevice> localDevice = device.Load();
    if (localDevice == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) openDevice();
        localDevice = device;
    }
    const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / sizeof(std::uint32_t));
    std::array<std::uint32_t, 5> resolved{};
    if (indirectArguments != 0 && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) {
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        resolved = Pm4::ReadDispatchArguments(indirectArguments, packet[4]);
        if (!prePassDeferral().resumed) countIndirect(IndirectFillKernel, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        packet = resolved;
        indirectArguments = 0;
    }
    std::array<std::uint32_t, 3> partialGroups{};
    if (indirectArguments == 0 && (packet[4] & 0x2u) != 0) {
        partialGroups = Pm4::PartialGroupThreads({packet[1], packet[2], packet[3]}, {readRegister(queue.shader, 0x207), readRegister(queue.shader, 0x208), readRegister(queue.shader, 0x209)});
    }
    const bool partialLastGroup = partialGroups != std::array<std::uint32_t, 3>{};
    {
        auto& deferral = prePassDeferral();
        std::optional<Graphics::MemoryValueFill> fill;
        if (indirectArguments == 0 && (packet[4] & 0x20u) == 0 && !deferral.noted) fill = Graphics::MatchMemoryValueFill(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads, {packet[1], packet[2], packet[3]});
        if (fill && deferral.allowed) {
            bool waitForWrites = true;
            std::uint32_t value = 0;
            static_cast<void>(readPrePassWord(&waitForWrites, fill->valueAddress, &value));
            if (deferral.deferred) return;
        }
        if (const auto htile = Graphics::HtileDepthClearAddress(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads); htile != 0 && !deferral.noted) {
            const auto threads = (packet[4] & 0x20u) != 0 ? static_cast<std::uint64_t>(packet[1]) : static_cast<std::uint64_t>(packet[1]) * compute.numThreads[0];
            const auto nativeBytes = Graphics::HtileDepthClearBytes_nid_no_patch(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments);
            Graphics::NoteHtileDepthClear(htile, static_cast<std::size_t>(threads * 16u), nativeBytes);
        }
        if (fill) Graphics::NoteMemoryValueFill(*fill);
        if (deferral.allowed) deferral.noted = true;
        if (!partialLastGroup && fillBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice)) {
            pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
            return;
        }
        if (!partialLastGroup && indirectArguments == 0 && copyBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice, address)) {
            pendingDispatchPhases().outcome = DispatchOutcome::CopyHle;
            return;
        }
    }
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {
        const std::array<std::uint32_t, 3> threads{packet[1], packet[2], packet[3]};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if (threads[axis] % compute.numThreads[axis] != 0) compute.partialThreads = threads;
        }
    }
    if (partialLastGroup) compute.partialThreads = partialGroups;
    if (indirectArguments != 0 && threadDimensions) {
        compute.threadsFromMemory = true;
        compute.partialThreads = compute.numThreads;
    }
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, address, std::span(snapshot.code).subspan(codeOffset), snapshot.headerAddress, snapshot.header},
        {(packet[4] & 0x8000u) != 0 ? 32u : 64u, 0, userData, compute, std::nullopt, std::nullopt, memory, DispatchFloatMode(snapshot, queue.shader)},
        localDevice->ComputeTarget((packet[4] & 0x8000u) != 0 ? 32u : 64u),
        {0, 0, 0, 128}
    };
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static double recompileMs = 0, deviceMs = 0;
    static std::uint64_t dispatches = 0;
    auto lap = std::chrono::steady_clock::now();
    std::array<double, DriverPhaseCount> phaseMs{};
    auto phaseLap = lap;
    DispatchPhaseTiming phaseTiming{profile, lap, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[PhasePrologue] = std::chrono::duration<double, std::milli>(lap - packetStartedAt()).count();

    std::uint64_t programKey = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        programKey ^= value;
        programKey *= 0x100000001b3ull;
    };
    mix(address);
    mix(packet[4] & 0x8000u);
    for (const auto threads : compute.partialThreads) mix(threads);
    mix(compute.threadsFromMemory);
    for (const auto offset : {0x207u, 0x208u, 0x209u, 0x212u, 0x213u}) {
        const auto found = queue.shader.find(offset);
        mix(found == queue.shader.end() ? (1ull << 32u) : found->second);
    }
    mix(reinterpret_cast<std::uintptr_t>(registeredShader.get()));
    mix(localDevice->Serial());
    phaseTiming.Phase(PhaseKey);
    timing.Mark("decode_key_device_setup");
    std::shared_ptr<const RunTimeProgram> program;
    bool linked = false;
    std::vector<std::shared_ptr<const LinkedProgram>> linkedCandidates;
    bool hitCounted = false;
    {
        std::lock_guard cacheLock(runTimeProgramsMutex);
        if (const auto found = runTimeDispatches.find(programKey); found != runTimeDispatches.end()) {
            program = found->second;
            ++dataIndependentHits;
            hitCounted = true;
        } else if (const auto links = linkedPrograms.find(programKey); links != linkedPrograms.end()) {
            linkedCandidates = links->second;
        }
    }
    for (auto candidate = linkedCandidates.rbegin(); program == nullptr && candidate != linkedCandidates.rend(); ++candidate) {
        if (!(*candidate)->callees.Current(userData)) continue;
        linked = true;
        program = (*candidate)->program;
    }
    const auto deferred = [&] {
        if (!hitCounted) return;
        std::lock_guard cacheLock(runTimeProgramsMutex);
        --dataIndependentHits;
    };
    const auto rethrow = [&](const std::exception& error) {
        std::string reason = error.what();
        if (const auto newline = reason.find('\n'); newline != std::string::npos) reason.resize(newline);
        char where[64];
        std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
        throw std::runtime_error(where + reason);
    };
    if (program == nullptr) {
        std::shared_ptr<TableCallProgram> tableCalls;
        try {
            tableCalls = TableCallProgramFor(programKey, request);
            if (tableCalls == nullptr) tableCalls = RegisteredBranchSplitFor(programKey, snapshot, codeOffset, request);
            if (tableCalls == nullptr) {
                const auto made = runTimeDispatch(snapshot, codeOffset, request, localDevice, programKey);
                hitCounted = true;
                program = made.program;
                linked = made.linked;
                const auto words = program->compiled->spirv.size();
                if (linked && words > ShaderRecompiler::GiantProgramWords) {
                    std::shared_ptr<const LinkedProgram> entry;
                    {
                        std::lock_guard cacheLock(runTimeProgramsMutex);
                        auto& candidates = linkedPrograms[programKey];
                        const auto found = std::ranges::find_if(candidates, [&](const std::shared_ptr<const LinkedProgram>& candidate) { return candidate->program == program; });
                        if (found != candidates.end()) entry = *found;
                    }
                    require(entry != nullptr, "a linked compute program over the giant bound is not registered");
                    tableCalls = BranchSplitProgramFor(programKey, request, words, localDevice, snapshot, codeOffset, entry->callees);
                    if (tableCalls != nullptr) {
                        std::lock_guard cacheLock(runTimeProgramsMutex);
                        std::erase(linkedPrograms[programKey], entry);
                    }
                } else if (!linked && words > ShaderRecompiler::GiantProgramWords) {
                    tableCalls = BranchSplitProgramFor(programKey, request, words, localDevice, snapshot, codeOffset);
                    if (tableCalls == nullptr) {
                        std::lock_guard cacheLock(runTimeProgramsMutex);
                        program = runTimeDispatches.try_emplace(programKey, program).first->second;
                    }
                }
            }
        } catch (const std::exception& error) {
            rethrow(error);
        }
        if (tableCalls != nullptr) {
            auto groups = DispatchGroups(packet, indirectArguments, queue.shader);
            if (indirectArguments != 0) {
                require(!threadDimensions, "an indirect dispatch with thread dimensions of a program with table calls is not implemented");
                recordQueuedLabelsBeforeRead(submission.queue);
                const auto resolved = Pm4::ReadDispatchArguments(indirectArguments, packet[4]);
                groups = {resolved[1], resolved[2], resolved[3]};
            }
            TableCallDispatch call;
            call.device = localDevice;
            call.snapshot = &snapshot;
            call.codeOffset = codeOffset;
            call.request = &request;
            call.userData = userData;
            call.groups = groups;
            call.programAddress = address;
            call.result = [&](const RunTimeProgram& part, std::span<const std::uint32_t> words) { return runTimeResult(part, words, request.shader.codeAddress); };
            call.settle = [&] {
                if (!recordQueuedLabelsAfterPrePass(submission.queue)) return false;
                prePassReads().clear();
                return true;
            };
            call.locked = [&](const std::function<void()>& body) {
                GuestMemory::TagGpuLockSite(indirectArguments != 0 ? GuestMemory::GpuLockSite::Indirect : GuestMemory::GpuLockSite::Dispatch);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                recordLabelsForPacket(localDevice.get(), submission.queue);
                body();
            };
            prePassReads().clear();
            prePassDeferral().allowed = false;
            try {
                RecordTableCallDispatch(*tableCalls, call);
            } catch (const std::exception& error) {
                rethrow(error);
            }
            return;
        }
        recompileMs += phaseTiming.Elapsed();
        phaseTiming.Phase(PhaseRecompile);
    }
    prePassReads().clear();
    auto compiledResult = runTimeResult(*program, userData, request.shader.codeAddress);
    if (prePassDeferral().deferred) return deferred();
    if (recordQueuedLabelsAfterPrePass(submission.queue)) {
        prePassReads().clear();
        compiledResult = runTimeResult(*program, userData, request.shader.codeAddress);
        if (prePassDeferral().deferred) return deferred();
    }
    prePassDeferral().allowed = false;
    phaseTiming.Phase(PhaseLookup);
    const auto& compiled = *compiledResult;
    const auto groups = DispatchGroups(packet, indirectArguments, queue.shader);
    const std::uint64_t dispatchThreads = indirectArguments != 0 ? 0 : std::uint64_t{groups[0]} * groups[1] * groups[2] * compute.numThreads[0] * compute.numThreads[1] * compute.numThreads[2];
    const std::uint64_t linearThreads = indirectArguments != 0 || (packet[4] & 0x20u) != 0 || partialLastGroup || groups[1] != 1 || groups[2] != 1 ? 0 : std::uint64_t{groups[0]} * compute.numThreads[0];
    if (linearThreads != 0 && transferBuffer(submission.queue, compiled, userData, linearThreads, localDevice, address)) {
        pendingDispatchPhases().outcome = compiled.transferKernel->kind == ShaderRecompiler::TransferKernel::Kind::Copy ? DispatchOutcome::CopyHle : DispatchOutcome::FillHle;
        return;
    }
    static const bool traceIo = std::getenv("APS5_TRACE_DISPATCH_IO") != nullptr;
    if (traceIo) {
        std::string words;
        if (std::getenv("APS5_TRACE_DISPATCH_IO")[0] == '2') {
            for (const auto word : userData) {
                char text[12];
                std::snprintf(text, sizeof(text), " %08x", word);
                words += text;
            }
        }
        std::fprintf(stderr, "[dispatch-io] shader 0x%llx%s\n", static_cast<unsigned long long>(address), words.c_str());
    }
    phaseTiming.Phase(PhaseSnapshots);
    try {
        if (localDevice->PrepareComputePipeline(compiled, address)) localDevice->RecordComputePipeline(compiled);
    } catch (const std::exception& error) {
        rethrow(error);
    }
    std::optional<Graphics::DepthPlaneClear> depthPlaneClear;
    if (indirectArguments == 0 && !partialLastGroup) {
        if (const auto clear = Graphics::MatchStorageImageClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, groups)) {
            const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
            if (clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                std::lock_guard clearLock(GuestMemory::GpuMutex());
                recordLabelsForPacket(localDevice.get(), submission.queue);
                depthPlaneClear.emplace(*clear);
            }
        }
    }
    const auto probe = compiled.deviceImagesComplete ? nullptr : ShaderRecompiler::DeviceImageProbeResult(compiled);
    std::shared_ptr<PreparedDispatch> preparedProbe;
    std::shared_ptr<PreparedDispatch> prepared;
    try {
        if (probe != nullptr) {
            static_cast<void>(localDevice->PrepareComputePipeline(*probe, address));
            preparedProbe = localDevice->PrepareDispatch(*probe, dispatchThreads, linearThreads);
        }
        auto segmentRequest = request;
        segmentRequest.target.bdaSegments = true;
        segmentRequest.context.userData = {};
        segmentRequest.context.memory = {};
        auto segmentProgram = [registeredShader, codeOffset, segmentRequest, words = std::vector<std::uint32_t>(userData.begin(), userData.end()), regions = memory]() -> std::shared_ptr<const ShaderRecompiler::RecompileResult> {
            auto source = segmentRequest;
            source.context.userData = words;
            source.context.memory = regions;
            bool preparedAtUse = false;
            return InvocationFor(*registeredShader, codeOffset, source, &preparedAtUse, nullptr).MaterializeAtRunTime();
        };
        prepared = localDevice->PrepareDispatch(compiled, dispatchThreads, linearThreads, std::move(segmentProgram));
    } catch (const std::exception& error) {
        rethrow(error);
    }
    if (profile) {
        const auto now = std::chrono::steady_clock::now();
        const auto prepareMs = std::chrono::duration<double, std::milli>(now - phaseLap).count();
        phaseLap = now;
        double parts = 0;
        if (prepared != nullptr) {
            const auto phases = VulkanDevice::PreparePhaseMs(*prepared);
            phaseMs[PhasePrepareKey] += phases[0];
            phaseMs[PhasePrepareFind] += phases[1];
            phaseMs[PhasePreparePrecollect] += phases[2];
            phaseMs[PhasePreparePresync] += phases[3];
            phaseMs[PhasePrepareStageA] += phases[4];
            for (const auto part : phases) parts += part;
        }
        phaseMs[PhasePrepareOther] += std::max(0.0, prepareMs - parts);
    }
    {
        GuestMemory::TagGpuLockSite(indirectArguments != 0 ? GuestMemory::GpuLockSite::Indirect : GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        phaseTiming.Phase(PhaseLockWait);
        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(PhaseLabels);
        try {
            if (probe != nullptr && indirectArguments != 0) {
                static_cast<void>(localDevice->DispatchIndirect(*probe, indirectArguments, address, std::move(preparedProbe), compute.threadsFromMemory ? std::optional(compute.numThreads) : std::nullopt));
            } else if (probe != nullptr) {
                localDevice->Dispatch(*probe, groups[0], groups[1], groups[2], address, std::move(preparedProbe));
            }
            if (indirectArguments != 0) {
                const auto outcome = localDevice->DispatchIndirect(compiled, indirectArguments, address, std::move(prepared), compute.threadsFromMemory ? std::optional(compute.numThreads) : std::nullopt);
                countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else {
                localDevice->Dispatch(compiled, groups[0], groups[1], groups[2], address, std::move(prepared));
            }
        } catch (const std::exception& error) {
            rethrow(error);
        }
        phaseTiming.Phase(PhaseDevice);
        deviceMs += phaseTiming.Elapsed();
    }
    phaseTiming.Phase(PhaseTail);
    if (profile) {
        auto& pending = pendingDispatchPhases();
        pending.outcome = DispatchOutcome::Real;
        pending.phases = true;
        pending.hit = true;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
    }
    if (profile && ++dispatches % 100 == 0) {
        std::uint64_t hits = 0, compiles = 0;
        {
            std::lock_guard cacheLock(runTimeProgramsMutex);
            hits = dataIndependentHits;
            compiles = runTimeCompiles;
        }
        AgcDriver::ProfilePrint_nid_no_patch("[gpu] %llu dispatches (%llu through run-time programs, %llu programs compiled): compile %.1f s, device %.1f s\n", static_cast<unsigned long long>(dispatches), static_cast<unsigned long long>(hits), static_cast<unsigned long long>(compiles), recompileMs / 1000.0, deviceMs / 1000.0);
    }
}

}
