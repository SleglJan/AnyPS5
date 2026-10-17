#include "prx/libSceAgcDriver/Execution/include/PipelineReplay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "CompiledVariant.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "ShaderCacheDirectory.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace AgcDriver {

namespace {

struct ReplayTotals {
    std::atomic<std::uint64_t> loaded{0};
    std::atomic<std::uint64_t> replayed{0};
    std::atomic<std::uint64_t> recorded{0};
    std::atomic<std::uint64_t> notRecordable{0};
    std::atomic<std::uint64_t> failed{0};
    std::atomic<std::uint64_t> inFrames{0};
    std::atomic<std::uint64_t> programsLoaded{0};
    std::atomic<std::uint64_t> programsAtUse{0};
    std::atomic<std::uint64_t> programsRecorded{0};
    std::atomic<std::uint64_t> programsReplayed{0};
    std::atomic<std::uint64_t> programsStale{0};
    std::atomic<std::uint64_t> programsFailed{0};
    std::atomic<std::uint64_t> programsRecompiled{0};
    std::atomic<std::uint64_t> programsDropped{0};
    std::atomic<std::uint64_t> dropped{0};
};

ReplayTotals& ReplayCounters() {
    static ReplayTotals totals;
    return totals;
}

bool Recordable(const Graphics::State& state, std::span<const Graphics::CompiledShader> shaders) {
    if (state.rectList) return false;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr || shader.stage == ShaderRecompiler::ShaderStage::Geometry || shader.program->identity == std::array<std::uint64_t, 2>{}) return false;
    }
    return true;
}

}

PipelineReplay::PipelineReplay(PipelineManifest& manifest, std::function<Graphics::Context()> context, std::function<Target()> target) : manifest(manifest), context(std::move(context)), target(std::move(target)) {
    for (auto& [record, use] : manifest.GraphicsRecords()) {
        auto& entry = pending.emplace_back();
        entry.use = use;
        entry.handles.resize(record.stages.size());
        entry.missing = record.stages.size();
        for (std::size_t stage = 0; stage < record.stages.size(); ++stage) waiting[record.stages[stage].program].emplace_back(pending.size() - 1u, stage);
        entry.record = std::move(record);
    }
    ReplayCounters().loaded.store(pending.size(), std::memory_order_relaxed);
    ReplayCounters().programsLoaded.store(manifest.LoadedRecords().prepared.size(), std::memory_order_relaxed);
}

std::shared_ptr<RecordUse> PipelineReplay::Created(const Graphics::Context& context, const Graphics::State& state, const Graphics::VertexInputLayout& vertexInput, const Graphics::PipelineSetLayouts& resources, std::span<const Graphics::CompiledShader> shaders, VkImageLayout attachmentLayout) {
    ReplayCounters().inFrames.fetch_add(1, std::memory_order_relaxed);
    if (!Recordable(state, shaders)) {
        ReplayCounters().notRecordable.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    const auto described = Graphics::DescribeSetLayouts(context, shaders);
    if (described.layouts.split != resources.split || !std::ranges::equal(described.mainKey, resources.mainKey) || !std::ranges::equal(described.fragmentKey, resources.fragmentKey)) {
        ReplayCounters().notRecordable.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    GraphicsPipelineRecord record;
    for (const auto& shader : shaders) {
        const auto& program = *shader.program;
        auto& stage = record.stages.emplace_back();
        stage.stage = static_cast<std::uint32_t>(shader.stage);
        stage.pushOffset = shader.pushConstantOffset;
        stage.program = program.identity;
        stage.constants = program.moduleConstants;
        stage.pushBytes = static_cast<std::uint32_t>(program.pushConstants.size());
        for (const auto& binding : program.bindings) stage.bindings.push_back({static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count});
    }
    record.description = Graphics::EncodePipelineDescription(state, vertexInput, attachmentLayout);
    const auto decoded = Graphics::DecodePipelineDescription(record.description);
    const Graphics::PipelineDescription original{state, vertexInput, attachmentLayout};
    if (!decoded.has_value() || !Graphics::SamePipelineKey(context, resources, shaders, original, *decoded)) throw std::logic_error("AGC driver: a graphics pipeline description does not rebuild its pipeline key");
    auto added = manifest.Record(std::move(record));
    if (added.added) ReplayCounters().recorded.fetch_add(1, std::memory_order_relaxed);
    return std::move(added.use);
}

std::vector<std::function<void()>> PipelineReplay::Prepared(const std::shared_ptr<const ShaderRecompiler::SourceHandle>& handle) {
    if (handle == nullptr || handle->artifact == nullptr) return {};
    return Prepared(handle, ShaderRecompiler::GetPreparedArtifact(*handle).identity);
}

std::optional<ProgramInterface> PipelineReplay::DerivedInterface(const ShaderRecompiler::RecompileRequest& request, const std::shared_ptr<const ShaderRecompiler::SourceHandle>& handle) {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> registration;
    try {
        if (handle == nullptr) return std::nullopt;
        registration = ShaderRecompiler::RegistrationResult(request, *handle);
        if (registration == nullptr) {
            if (const auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(request, handle)) registration = invocation->MaterializeAtRunTime();
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
    if (registration == nullptr) return std::nullopt;
    ProgramInterface derived;
    derived.pushBytes = static_cast<std::uint32_t>(registration->pushConstants.size());
    for (const auto& binding : registration->bindings) {
        derived.bindings.push_back({static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count});
        if (binding.kind == ShaderRecompiler::DescriptorKind::DeviceAddress || binding.kind == ShaderRecompiler::DescriptorKind::SamplerSlot) continue;
        if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable) continue;
        derived.computeBindings.emplace_back(binding.binding, binding.count);
    }
    return derived;
}

bool PipelineReplay::Recompiled(const PreparedProgramRecord& recorded, const ShaderRecompiler::RecompileRequest& request, const std::shared_ptr<const ShaderRecompiler::SourceHandle>& handle, const std::array<std::uint64_t, 2>& identity) {
    const auto derived = DerivedInterface(request, handle);
    manifest.Forget(recorded);
    const auto rekeyed = manifest.Rekey(recorded.identity, identity, derived);
    manifest.Record(PreparedProgramRecord{recorded.program, recorded.codeOffset, identity, ShaderRecompiler::RequestSerializer{}.Serialize(request)});
    {
        std::lock_guard lock(mutex);
        if (const auto found = waiting.find(recorded.identity); found != waiting.end()) {
            for (const auto& [index, stage] : found->second) {
                auto& entry = pending[index];
                if (!derived) {
                    entry.dropped = true;
                    continue;
                }
                entry.record.stages[stage].program = identity;
                entry.record.stages[stage].pushBytes = derived->pushBytes;
                entry.record.stages[stage].bindings = derived->bindings;
            }
            if (!derived) waiting.erase(found);
        }
    }
    if (!derived && rekeyed.droppedCompute + rekeyed.droppedGraphics != 0) {
        ReplayCounters().dropped.fetch_add(rekeyed.droppedCompute + rekeyed.droppedGraphics, std::memory_order_relaxed);
        std::fprintf(stderr, "[pipeline-manifest] the binding interface of a program prepared again from its request cannot be derived without its draw: %zu compute and %zu graphics pipeline records that named it are dropped\n", rekeyed.droppedCompute, rekeyed.droppedGraphics);
    }
    return derived.has_value();
}

void PipelineReplay::Drop(const PreparedProgramRecord& recorded) {
    manifest.Forget(recorded);
}

std::vector<std::function<void()>> PipelineReplay::Prepared(const std::shared_ptr<const ShaderRecompiler::SourceHandle>& handle, const std::array<std::uint64_t, 2>& identity) {
    std::vector<std::function<void()>> ready;
    if (handle == nullptr || handle->artifact == nullptr) return ready;
    std::lock_guard lock(mutex);
    const auto found = waiting.find(identity);
    if (found == waiting.end()) return ready;
    for (const auto& [index, stage] : found->second) {
        auto& entry = pending[index];
        if (entry.dropped || entry.handles[stage] != nullptr) continue;
        entry.handles[stage] = handle;
        if (--entry.missing != 0u) continue;
        ready.push_back([this, record = entry.record, use = entry.use, handles = std::move(entry.handles)] { create(record, use, handles); });
        entry.handles.clear();
    }
    waiting.erase(found);
    return ready;
}

void PipelineReplay::create(const GraphicsPipelineRecord& record, const std::shared_ptr<RecordUse>& use, const std::vector<std::shared_ptr<const ShaderRecompiler::SourceHandle>>& handles) {
    try {
        const auto device = context();
        const auto spirvTarget = target();
        const auto description = Graphics::DecodePipelineDescription(record.description);
        if (!description.has_value()) throw std::runtime_error("the recorded pipeline description does not decode");
        std::vector<std::shared_ptr<ShaderRecompiler::RecompileResult>> results;
        std::vector<Graphics::CompiledShader> shaders;
        for (std::size_t index = 0; index < record.stages.size(); ++index) {
            const auto& stage = record.stages[index];
            auto result = ShaderRecompiler::SpecializedStageResult(*handles[index], stage.constants, spirvTarget);
            if (result->moduleConstants != stage.constants) {
                manifest.Forget(record);
                throw std::runtime_error("the recorded specialization constants are not the ones the program's module takes: the record is dropped");
            }
            result->pushConstants.assign(stage.pushBytes, std::byte{0});
            for (const auto& shape : stage.bindings) {
                auto& binding = result->bindings.emplace_back();
                binding.kind = static_cast<ShaderRecompiler::DescriptorKind>(shape.kind);
                binding.role = static_cast<ShaderRecompiler::DescriptorRole>(shape.role);
                binding.descriptorSet = shape.descriptorSet;
                binding.binding = shape.binding;
                binding.count = shape.count;
            }
            shaders.push_back({static_cast<ShaderRecompiler::ShaderStage>(stage.stage), result.get(), stage.pushOffset});
            results.push_back(std::move(result));
        }
        const auto described = Graphics::DescribeSetLayouts(device, shaders);
        static_cast<void>(Graphics::PrepareCachedPipeline(device, description->state, description->vertexInput, described.View(), shaders, description->attachmentLayout, use));
        ReplayCounters().replayed.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& error) {
        ReplayCounters().failed.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "[pipeline-manifest] a graphics pipeline recorded by an earlier run could not be created ahead: %s\n", error.what());
    }
}

std::array<std::uint64_t, 2> PipelineReplay::ProgramKey(std::span<const std::uint32_t> registeredCode) {
    const auto bytes = std::as_bytes(registeredCode);
    return {ShaderRecompiler::HashBytes(bytes, 0x5eed0003ull), ShaderRecompiler::HashBytes(bytes, 0x5eed0004ull)};
}

void PipelineReplay::RecordPreparedProgram(std::span<const std::uint32_t> registeredCode, std::uint64_t codeOffset, std::span<const std::pair<std::uint64_t, std::uint64_t>> ownRegions, const ShaderRecompiler::RecompileRequest& request) {
    ReplayCounters().programsAtUse.fetch_add(1, std::memory_order_relaxed);
    auto own = request;
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    for (const auto& region : request.context.memory) {
        const auto end = region.guestAddress + region.bytes.size();
        if (std::ranges::any_of(ownRegions, [&](const auto& range) { return region.guestAddress < range.second && range.first < end; })) memory.push_back(region);
    }
    own.context.memory = memory;
    PreparedProgramRecord record{ProgramKey(registeredCode), codeOffset, ShaderRecompiler::ArtifactIdentity(request), ShaderRecompiler::RequestSerializer{}.Serialize(own)};
    if (manifest.Record(std::move(record))) ReplayCounters().programsRecorded.fetch_add(1, std::memory_order_relaxed);
}

std::vector<PreparedProgramRecord> PipelineReplay::PreparedPrograms(std::span<const std::uint32_t> registeredCode) const {
    return manifest.PreparedPrograms(ProgramKey(registeredCode));
}

void PipelineReplay::CountProgramReplay(ProgramReplay outcome) {
    auto& counters = ReplayCounters();
    switch (outcome) {
        case ProgramReplay::Replayed: counters.programsReplayed.fetch_add(1, std::memory_order_relaxed); break;
        case ProgramReplay::Stale: counters.programsStale.fetch_add(1, std::memory_order_relaxed); break;
        case ProgramReplay::Failed: counters.programsFailed.fetch_add(1, std::memory_order_relaxed); break;
        case ProgramReplay::Recompiled: counters.programsRecompiled.fetch_add(1, std::memory_order_relaxed); break;
        case ProgramReplay::Dropped: counters.programsDropped.fetch_add(1, std::memory_order_relaxed); break;
    }
}

PipelineReplay::Counters PipelineReplay::Totals() {
    auto& totals = ReplayCounters();
    return {totals.loaded.load(std::memory_order_relaxed), totals.replayed.load(std::memory_order_relaxed), totals.recorded.load(std::memory_order_relaxed), totals.notRecordable.load(std::memory_order_relaxed), totals.failed.load(std::memory_order_relaxed), totals.inFrames.load(std::memory_order_relaxed), totals.programsLoaded.load(std::memory_order_relaxed), totals.programsAtUse.load(std::memory_order_relaxed), totals.programsRecorded.load(std::memory_order_relaxed), totals.programsReplayed.load(std::memory_order_relaxed), totals.programsStale.load(std::memory_order_relaxed), totals.programsFailed.load(std::memory_order_relaxed), RecordUse::Marks().load(std::memory_order_relaxed), totals.programsRecompiled.load(std::memory_order_relaxed), totals.programsDropped.load(std::memory_order_relaxed), totals.dropped.load(std::memory_order_relaxed)};
}

}
