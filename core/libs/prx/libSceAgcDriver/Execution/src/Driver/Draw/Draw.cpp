#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawProgramTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRegisterKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "CacheKey.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <spirv/unified1/spirv.hpp>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t DrawDecodeEntries = 4096;

}

std::shared_ptr<const DrawDecode> Driver::drawDecode(const QueueState& queue, const Submission& submission, std::uint64_t deviceSerial) {
    auto key = RegisterShapeKey(queue);
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(deviceSerial);
    const bool workgroup = VertexWorkgroupRequired(queue, *submission.shaders);
    mix(workgroup ? 1u : 0u);
    for (const auto base : {0x008u, 0x088u, 0x0c8u, 0x108u, 0x148u}) {
        const auto low = queue.shader.find(base);
        const auto high = queue.shader.find(base + 1);
        if (low == queue.shader.end() || high == queue.shader.end()) {
            mix(0);
            continue;
        }
        const auto address = (static_cast<std::uint64_t>(low->second) << 8u) | (static_cast<std::uint64_t>(high->second & 0xffu) << 40u);
        auto it = submission.shaders->upper_bound(address);
        if (it == submission.shaders->begin()) {
            mix(1);
            continue;
        }
        --it;
        for (const auto& snapshot : it->second) mix(reinterpret_cast<std::uintptr_t>(snapshot.get()));
        mix(address - it->first);
    }
    std::shared_ptr<const DrawDecode> state;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        if (const auto found = drawDecodes.find(key); found != drawDecodes.end()) state = found->second;
    }
    if (state == nullptr) {
        auto fresh = std::make_shared<DrawDecode>();
        fresh->state = Graphics::DecodeState(queue, workgroup);
        fresh->pixel = Graphics::DecodePixelStageInfo(queue.context, Graphics::ExportMappings(fresh->state), Graphics::PixelProgramSkipped(queue));
        fresh->pixel.targetExportPacking = Graphics::ExportPackings(fresh->state);
        fresh->pixel.dualSourceBlend = fresh->state.dualSourceBlend;
        fresh->pixel.reversedBlend = fresh->state.reversedBlend;
        fresh->pixel.alphaToCoverage = fresh->state.alphaToCoverage;
        fresh->pixel.quadPixelMask = Graphics::QuadPixelMask(fresh->state);
        fresh->pixel.quadSampleMask = Graphics::QuadSampleMasks(fresh->state);
        if (fresh->pixel.stencilRefExportEnable && Graphics::StencilReferenceUnused(fresh->state)) {
            const auto localDevice = device.Load();
            const auto capabilities = localDevice == nullptr ? std::vector<std::uint32_t>{} : localDevice->Target().supportedCapabilities;
            if (std::ranges::find(capabilities, static_cast<std::uint32_t>(spv::CapabilityStencilExportEXT)) == capabilities.end()) fresh->pixel.stencilRefExportEnable = false;
        }
        if (fresh->pixel.ancillary) fresh->pixel.rasterPrimitive = Graphics::RasterPrimitiveOf(fresh->state);
        if (fresh->pixel.ancillary) fresh->pixel.layerExported = fresh->state.layerExports != 0;
        fresh->pixel.ditheredAlphaToCoverage = fresh->state.ditheredAlphaToCoverage;
        std::lock_guard cacheLock(drawCacheMutex);
        if (drawDecodes.size() >= DrawDecodeEntries) drawDecodes.clear();
        state = drawDecodes.try_emplace(key, std::move(fresh)).first->second;
    }
    auto decode = std::make_shared<DrawDecode>();
    decode->state = state->state;
    decode->pixel = state->pixel;
    DecodeGraphicsPrograms(*decode, queue, *submission.shaders, false, true);
    return decode;
}

std::shared_ptr<const Driver::RunTimeProgram> Driver::runTimeDrawStage(const DrawProgram& program, const ShaderRecompiler::RecompileRequest& request, const std::shared_ptr<VulkanDevice>& device, std::uint64_t key) {
    {
        std::lock_guard cacheLock(drawCacheMutex);
        if (const auto found = drawStages.find(key); found != drawStages.end() && SameCode(found->second.snapshot.get(), program.snapshot.get()) && SameCode(found->second.continuation.get(), program.tailContinuation.get())) return found->second.program;
    }
    bool preparedAtUse = false;
    const auto invocation = InvocationFor(*program.snapshot, program.codeOffset, request, &preparedAtUse);
    const auto& prepared = invocation.Request();
    if (preparedAtUse) RecordPreparedProgram(device, *program.snapshot, program.codeOffset, prepared);
    require((prepared.shader.code.size() == request.shader.code.size() || program.tailContinuation != nullptr) && prepared.shader.linkedLiterals.empty() && prepared.context.linkedWords.empty(), "a graphics program that calls code through its user data is not implemented");
    auto compiled = invocation.MaterializeAtRunTime();
    require(compiled != nullptr, "a graphics program that reads SRT words, uniform fills, image tables or bound descriptors at compile time is not implemented");
    require(CacheableResult(*compiled), "a graphics program whose compile poisoned an SRT read or an image table entry is not implemented");
    require(std::ranges::none_of(compiled->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; }), "a graphics program that binds a flattened SRT is not implemented");
    require(std::ranges::all_of(compiled->imageTableReadRanges, [](const auto& ranges) { return ranges.empty(); }), "a graphics program that reads image tables on the CPU is not implemented");
    auto entry = std::make_shared<const RunTimeProgram>(RunTimeProgram{invocation.Plan(), std::move(compiled)});
    std::lock_guard cacheLock(drawCacheMutex);
    ++drawStageCompiles;
    auto& stage = drawStages[key];
    if (stage.program != nullptr && SameCode(stage.snapshot.get(), program.snapshot.get()) && SameCode(stage.continuation.get(), program.tailContinuation.get())) return stage.program;
    stage = {std::move(entry), program.snapshot, program.tailContinuation};
    return stage.program;
}

std::uint64_t Driver::DrawStageCompiles() {
    std::lock_guard cacheLock(drawCacheMutex);
    return drawStageCompiles;
}

DrawVerdict Driver::draw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::string& rejected) {
    PerformanceTimer timing("Driver.Draw");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    auto phaseLap = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    DrawPhaseTiming phaseTiming{profile, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[DrawRowPrologue] = std::chrono::duration<double, std::milli>(phaseLap - packetStartedAt()).count();

    const auto drawn = [&] {
        phaseTiming.Phase(DrawRowVectors);
        if (!profile) return DrawVerdict::Drawn;
        auto& pending = pendingDrawPhases();
        pending.phases = true;
        pending.captures = 0;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
        return DrawVerdict::Drawn;
    };
    auto drawParameters = Pm4::ResolveDraw(packet, queue);
    bool traceIndirect = false;
    if (const auto verdict = precheckDraw(queue, submission, packet, drawParameters, rejected, traceIndirect)) return *verdict;
    phaseTiming.Phase(DrawRowPrecheck);
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;

    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
    std::shared_ptr<VulkanDevice> localDevice = device.Load();
    if (localDevice == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::lock_guard createLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    timing.Mark("device_setup");
    phaseTiming.Phase(DrawRowVectors);

    const auto decode = drawDecode(queue, submission, localDevice->Serial());
    const auto& graphics = decode->state;
    auto pixel = decode->pixel;
    std::vector<DrawProgram> programs = decode->programs;
    if (const auto skipped = SkippedShader(); skipped != 0 && std::any_of(programs.begin(), programs.end(), [skipped](const DrawProgram& program) { return program.binary.codeAddress == skipped; })) return drawn();
    const auto setMeshIndexBuffer = [&](const Pm4::DrawParameters& parameters) {
        if (!graphics.stages.mesh) return;
        auto& words = programs.front().userData;
        require(programs.front().firstUserSgpr == 0 && words.size() >= ShaderRecompiler::MeshIndexBufferUserWord + 4, "mesh program lacks the hidden user words");
        const auto descriptor = Graphics::MeshIndexBufferDescriptor(parameters);
        std::copy(descriptor.begin(), descriptor.end(), words.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    };
    if (!drawParameters.indirect) setMeshIndexBuffer(drawParameters);
    else if (graphics.stages.mesh) setMeshIndexBuffer(Pm4::DrawParameters{drawParameters.indexAddress, std::max(drawParameters.indexCount, 1u), drawParameters.indexSize, 1, 0, drawParameters.indexed});
    const std::vector<Role>& roles = decode->roles;
    static const bool tracePrograms = std::getenv("APS5_TRACE_DRAW_PROGRAMS") != nullptr;
    if (tracePrograms) {
        std::vector<DrawProgramTraceEntry> identities;
        identities.reserve(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) {
            identities.push_back({programs[i].binary.stage, roles[i], programs[i].snapshot->codeAddress, programs[i].binary.codeAddress});
        }
        const auto line = FormatDrawProgramTrace(submission.serial, submission.queue, identities);
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    phaseTiming.Phase(DrawRowDecode);

    const auto locate = [&](std::uint32_t location) -> std::optional<std::pair<std::size_t, std::size_t>> {
        if (location == 0x280u) return std::nullopt;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (roles[i] == Role::Fragment || roles[i] == Role::GeometryBack || location < programs[i].userDataBase) continue;
            const auto word = location - programs[i].userDataBase + (8u - programs[i].firstUserSgpr);
            if (word < programs[i].userData.size()) return std::make_pair(i, static_cast<std::size_t>(word));
        }
        return std::nullopt;
    };
    if (drawParameters.indirect) {
        auto& indirect = *drawParameters.indirect;
        const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
            const auto word = locate(location);
            if (!word || word->first != 0) return -1;
            return static_cast<std::int32_t>(programs.front().firstUserSgpr + word->second);
        };
        indirect.baseVertexSgpr = sgprOf(indirect.baseVertexLocation);
        indirect.startInstanceSgpr = sgprOf(indirect.startInstanceLocation);
        indirect.drawIndexSgpr = sgprOf(indirect.drawIndexLocation);
    }
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        for (const auto& region : program.memory) {
            if (!region.bytes.empty()) memory.push_back(region);
        }
        linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    timing.Mark("prepare");
    phaseTiming.Phase(DrawRowProgramPrepare);

    std::vector<ShaderRecompiler::RecompileResult> results;
    results.reserve(programs.size() + 2u);
    std::vector<std::shared_ptr<const RunTimeProgram>> stagePrograms(programs.size());
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> stageResults(programs.size());
    std::vector<Graphics::CompiledShader> stages;
    stages.reserve(programs.size() + 2u);
    std::vector<const ShaderRecompiler::RecompileResult*> programResults(programs.size(), nullptr);
    std::uint32_t pushCursorBytes = 0;

    const auto stageProgram = [&](std::size_t i, std::uint32_t pushOffset, const Pm4::DrawParameters& parameters) {
        const auto& program = programs[i];
        const auto waveSize = program.binary.stage == Stage::Fragment ? graphics.stages.fragmentWaveSize : graphics.stages.vertexWaveSize;
        std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
        if (program.binary.stage != Stage::Fragment) {
            vertex.emplace();
            vertex->paClVsOutCntl = PositionExportLayout(program.snapshot.get(), program.codeOffset, program.binary.code, readRegister(queue.context, 0x207), program.tailContinuation.get()) | graphics.layerExports;
        }
        const ShaderRecompiler::RecompileRequest request{
            program.binary,
            {waveSize, program.firstUserSgpr, program.userData, std::nullopt, program.binary.stage == Stage::Fragment ? std::optional(pixel) : std::nullopt, vertex, memory, RegisteredFloatMode(*program.snapshot)},
            localDevice->Target(),
            {0, 0, pushOffset, (graphics.stages.mesh ? ShaderRecompiler::MeshDrawPushOffsetBytes : graphics.stages.tessellation && pushOffset < Graphics::PipelinePushSlotBytes ? ShaderRecompiler::TessellationLevelPushOffsetBytes : Graphics::PipelinePushSlotBytes) - pushOffset % Graphics::PipelinePushSlotBytes},
            ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, graphics.stages.mesh, graphics.stages.tessellation, {parameters.indexAddress, parameters.indexCount, parameters.indexSize, parameters.instanceCount}}
        };
        auto key = ShaderRecompiler::RecompileCacheKey::ContextHash(request);
        const auto mix = [&](std::uint64_t value) {
            key ^= value;
            key *= 0x100000001b3ull;
        };
        mix(program.binary.codeAddress);
        mix(CodeIdentity(*program.snapshot));
        mix(program.codeOffset);
        mix(program.tailContinuation != nullptr ? CodeIdentity(*program.tailContinuation) : 0u);
        mix(request.layout.descriptorSet);
        mix(request.layout.firstBinding);
        mix(request.layout.pushConstantOffsetBytes);
        mix(request.layout.pushConstantSizeBytes);
        mix(localDevice->Serial());
        try {
            return runTimeDrawStage(program, request, localDevice, key);
        } catch (const std::exception& error) {
            std::string reason = error.what();
            if (const auto newline = reason.find('\n'); newline != std::string::npos) reason.resize(newline);
            char where[64];
            std::snprintf(where, sizeof(where), "shader 0x%llx stage %u: ", static_cast<unsigned long long>(program.binary.codeAddress), static_cast<unsigned>(program.binary.stage));
            throw std::runtime_error(where + reason);
        }
    };

    std::optional<Graphics::IndirectDrawPath> indirectCpu;
    std::vector<std::uint32_t> pushOffsets(programs.size(), 0);
    prePassReads().clear();
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        const auto& program = programs[i];
        pushCursorBytes = Graphics::StagePushOffset(pushCursorBytes, program.binary.stage, localDevice->GraphicsPipelineLibraries());
        pushOffsets[i] = pushCursorBytes;
        if (program.binary.stage == Stage::Fragment) {
            for (auto previous = i; previous-- > 0;) {
                if (programResults[previous] == nullptr) continue;
                pixel = Graphics::DefaultUnexportedInputs(decode->pixel, programResults[previous]->parameterExports);
                break;
            }
        }
        stagePrograms[i] = stageProgram(i, pushCursorBytes, drawParameters);
        phaseTiming.Phase(DrawRowRecompile);
        stageResults[i] = runTimeResult(*stagePrograms[i], program.userData, program.binary.codeAddress);
        phaseTiming.Phase(DrawRowKeyLookupValidate);
        programResults[i] = stageResults[i].get();
        const auto& result = *programResults[i];
        if (i == 0 && drawParameters.indirect) indirectCpu = ClassifyIndirectDraw(result, graphics, programs.front(), localDevice, drawParameters, traceIndirect);
        const auto slotEnd = (pushCursorBytes / Graphics::PipelinePushSlotBytes + 1u) * Graphics::PipelinePushSlotBytes;
        require(result.pushConstants.size() <= slotEnd - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
        stages.push_back({program.binary.stage, &result, result.pushConstants.empty() ? 0u : pushCursorBytes, program.binary.codeAddress});
        pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
    }
    timing.Mark("shader_compile_and_link");
    if (recordQueuedLabelsAfterPrePass(submission.queue)) {
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (stagePrograms[i] == nullptr) continue;
            const auto previous = programResults[i];
            stageResults[i] = runTimeResult(*stagePrograms[i], programs[i].userData, programs[i].binary.codeAddress);
            programResults[i] = stageResults[i].get();
            for (auto& stage : stages) {
                if (stage.program == previous) stage.program = programResults[i];
            }
        }
    }

    bool rectListBuilt = false;
    std::size_t rectIndex = 0;
    const auto buildRectList = [&] {
        phaseTiming.Phase(DrawRowVectors);
        require(programs.size() == 2 && programResults[0] != nullptr && programResults[1] != nullptr, "rect-list requires vertex and fragment programs");
        auto rectangle = DrawRectangle(*programs[0].snapshot, programs[1].snapshot, programResults[0]->variantId, programResults[1]->variantId, localDevice->Target());
        if (rectListBuilt) {
            results[rectIndex] = std::move(rectangle.control);
            results[rectIndex + 1] = std::move(rectangle.evaluation);
            phaseTiming.Phase(DrawRowRectList);
            return;
        }
        require(stages.size() == 2, "rect-list requires vertex and fragment programs");
        rectIndex = results.size();
        results.push_back(std::move(rectangle.control));
        results.push_back(std::move(rectangle.evaluation));
        stages.insert(stages.begin() + 1, {{Stage::TessellationControl, &results[rectIndex], 0}, {Stage::TessellationEvaluation, &results[rectIndex + 1], 0}});
        rectListBuilt = true;
        phaseTiming.Phase(DrawRowRectList);
    };
    if (graphics.rectList) buildRectList();
    if (!programs.empty() && programResults.back() != nullptr && programResults.back()->barycentricEmulation.active) {
        require(!graphics.rectList && graphics.stages.path == Graphics::ShaderPath::Vertex && programs.size() == 2 && programResults[0] != nullptr && stages.size() == 2, "a pixel shader that reads barycentrics without VK_KHR_fragment_shader_barycentric needs a vertex and a fragment program only");
        results.push_back(ShaderRecompiler::BuildBarycentricGeometryShader(*programResults[0], *programResults[1], localDevice->Target(), localDevice->GeometryLimits()));
        stages.insert(stages.begin() + 1, Graphics::CompiledShader{Stage::Geometry, &results.back(), 0});
    }
    timing.Mark("post_compile_prepare");
    const auto lockForDraw = [&] {
        if (gpuLock.owns_lock()) return;
        static_cast<void>(localDevice->PrepareDrawPipelines(graphics, stages));
        localDevice->PrewarmAddressSpace();
        phaseTiming.Phase(DrawRowVectors);
        const auto* presyncDevice = localDevice.get();
        const auto presynced = localDevice->PresyncDraw(graphics);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);
        if (auto current = device.Load(); current != nullptr && current != localDevice) {
            static std::atomic<std::uint64_t> replaced{0};
            std::fprintf(stderr, "[draw] device replaced during unlocked preparation (%llu)\n", static_cast<unsigned long long>(++replaced));
            localDevice = std::move(current);
        }
        if (presynced != 0 && localDevice.get() == presyncDevice) localDevice->ReapRecorded(presynced);
        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    };
    if (drawParameters.indirect && indirectCpu) {
        const auto indirect = *drawParameters.indirect;
        std::vector<std::size_t> resultIndex(programs.size(), 0);
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (programResults[i] == nullptr) continue;
            resultIndex[i] = results.size();
            results.push_back(ShaderRecompiler::RecompileResult(*programResults[i]));
            for (auto& stage : stages) {
                if (stage.program == programResults[i]) stage.program = &results[resultIndex[i]];
            }
            programResults[i] = &results[resultIndex[i]];
        }
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
        std::vector<Pm4::DrawArguments> records;
        for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(indirect, record));
        Graphics::CountIndirectDraw(*indirectCpu, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        const auto baseVertexWord = locate(indirect.baseVertexLocation);
        const auto startInstanceWord = locate(indirect.startInstanceLocation);
        const auto drawIndexWord = indirect.drawIndexEnabled ? locate(indirect.drawIndexLocation) : std::nullopt;
        for (std::uint32_t record = 0; record < records.size(); ++record) {
            const auto& arguments = records[record];
            if (traceIndirect) std::fprintf(stderr, "[draw]   record %u: count %u instances %u first %u vertexOffset %u startInstance %u\n", record, arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance);
            if (arguments.count == 0 || arguments.instances == 0) continue;
            std::set<std::size_t> patched;
            const auto patch = [&](const std::optional<std::pair<std::size_t, std::size_t>>& word, std::uint32_t value) {
                if (!word) return;
                programs[word->first].userData[word->second] = value;
                patched.insert(word->first);
            };
            patch(baseVertexWord, indirect.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex);
            patch(startInstanceWord, arguments.firstInstance);
            patch(drawIndexWord, record);
            if (graphics.stages.tessellation) {
                for (std::size_t i = 0; i < programs.size(); ++i) {
                    if (programs[i].binary.stage == Stage::TessellationControl || programs[i].binary.stage == Stage::TessellationEvaluation) patched.insert(i);
                }
            }
            Pm4::DrawParameters direct{0, arguments.count, 0, arguments.instances, drawParameters.flags, drawParameters.indexed, 0, 0};
            if (drawParameters.indexed) {
                if (arguments.firstVertexOrIndex >= drawParameters.indexCount) continue;
                direct.indexAddress = drawParameters.indexAddress + static_cast<std::uint64_t>(arguments.firstVertexOrIndex) * drawParameters.indexSize;
                direct.indexCount = std::min(arguments.count, drawParameters.indexCount - arguments.firstVertexOrIndex);
                direct.indexSize = drawParameters.indexSize;
            } else {
                direct.firstVertex = indirect.indxOffset;
            }
            if (graphics.stages.mesh) {
                setMeshIndexBuffer(direct);
                patched.insert(0);
            }
            for (const auto programIndex : patched) {
                auto& result = results[resultIndex[programIndex]];
                const auto pushBytes = result.pushConstants.size();
                if (programs[programIndex].binary.stage == Stage::TessellationControl || programs[programIndex].binary.stage == Stage::TessellationEvaluation) stagePrograms[programIndex] = stageProgram(programIndex, pushOffsets[programIndex], direct);
                result = *runTimeResult(*stagePrograms[programIndex], programs[programIndex].userData, programs[programIndex].binary.codeAddress);
                require(result.pushConstants.size() == pushBytes, "patched program changed its push constant layout");
            }
            if (graphics.rectList && patched.contains(0)) buildRectList();
            if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
                rejected = std::move(*known);
                return DrawVerdict::Rejected;
            }
            lockForDraw();
            phaseTiming.Phase(DrawRowVectors);
            localDevice->Draw(graphics, direct, stages);
            phaseTiming.Phase(DrawRowGraphics);
        }
        timing.Mark("draw_and_resource_release");
        return drawn();
    }

    if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
        rejected = std::move(*known);
        return DrawVerdict::Rejected;
    }
    lockForDraw();
    phaseTiming.Phase(DrawRowVectors);
    localDevice->Draw(graphics, drawParameters, stages);
    phaseTiming.Phase(DrawRowGraphics);
    timing.Mark("draw_and_resource_release");
    return drawn();
}

}
