#include "Recompiler.hpp"
#include "StencilReplay.hpp"
#include <algorithm>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "CacheKey.hpp"
#include "CompiledVariant.hpp"
#include "VertexInputSpecialization.hpp"
#include "FragmentOutputSpecialization.hpp"
#include "SpirvBackend/SpirvSpecialization.hpp"
#include "SpirvBackend/SpirvImageEmitter.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "ShaderDiskCache.hpp"
#include "ShaderCacheDirectory.hpp"
#include <limits>
#include <list>
#include <functional>
#include <map>
#include <set>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <unordered_map>
#include "ControlFlow/include/ControlFlow/CallFunctions.hpp"
#include "ControlFlow/include/ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/include/ControlFlow/Structurizer.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "IntermediateRepresentation/include/IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/include/Optimization/BindingAllocator.hpp"
#include "Optimization/include/Optimization/ConstantFolder.hpp"
#include "Optimization/include/Optimization/DeadCodeEliminator.hpp"
#include "Optimization/include/Optimization/DenormalFlushEliminator.hpp"
#include "Optimization/include/Optimization/DescriptorBindingBuilder.hpp"
#include "Optimization/include/Optimization/HostInterpolationChecker.hpp"
#include "Optimization/include/Optimization/MaskedSelectEliminator.hpp"
#include "Optimization/include/Optimization/NanCheckEliminator.hpp"
#include "Optimization/include/Optimization/OperandIdiomMerger.hpp"
#include "Optimization/include/Optimization/ReadLaneEliminator.hpp"
#include "Optimization/include/Optimization/RedundantValueEliminator.hpp"
#include "Optimization/include/Optimization/RequestMemoryView.hpp"
#include "Optimization/include/Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/include/Optimization/ResourceTracker.hpp"
#include "Optimization/include/Optimization/LinearStore.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "Optimization/include/Optimization/ShaderInfoCollector.hpp"
#include "Optimization/include/Optimization/SrtWalker.hpp"
#include "Optimization/include/Optimization/SsaBuilder.hpp"
#include "Optimization/include/Optimization/TessellationLowering.hpp"
#include "Optimization/include/Optimization/VectorWindowStoreEliminator.hpp"
#include "SpirvBackend/include/SpirvBackend/SpirvAnalysis.hpp"
#include "SpirvBackend/include/SpirvBackend/SpirvEmitter.hpp"
#if ANYPS5_ENABLE_SPIRV_TOOLS
#include "SpirvBackend/SpirvOptimizer.hpp"
#endif
#include "SpirvBackend/SpirvMemory/SpirvInputOutput.hpp"
#include "Translation/include/Translation/InstructionTranslator.hpp"
#include "Translation/include/Translation/ShaderInputInfoBuilder.hpp"
#include <exception>
#include <stdexcept>
#include <string>
#include <ControlFlow/RequestSerializer.hpp>

namespace ShaderRecompiler {

namespace {

ShaderStageKind toShaderStageKind(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Compute:
        return ShaderStageKind::Compute;
    case ShaderStage::Vertex:
        return ShaderStageKind::Vertex;
    case ShaderStage::TessellationControl:
        return ShaderStageKind::TessellationControl;
    case ShaderStage::TessellationEvaluation:
        return ShaderStageKind::TessellationEvaluation;
    case ShaderStage::Fragment:
        return ShaderStageKind::Pixel;
    case ShaderStage::Local:
        return ShaderStageKind::Local;
    case ShaderStage::Mesh:
        return ShaderStageKind::Mesh;
    case ShaderStage::Geometry:
        break;
    }
    throw std::runtime_error("ShaderRecompiler::Recompile: unsupported shader stage");
}

}

namespace {

// The host subgroup width wave64 programs are laid out for. Debug aid: APS5_SINGLE_LANE=<hex code
// addresses, comma separated, or "all"> keeps the listed programs at one guest lane per invocation.
std::uint32_t HostSubgroupSize(const RecompileRequest& request) {
    const bool vertexStage = request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Local;
    if (vertexStage && !request.target.vertexSubgroups) return 1u;
    static const std::string list = [] { const char* text = std::getenv("APS5_SINGLE_LANE"); return text ? std::string(text) : std::string(); }();
    if (!list.empty()) {
        if (list == "all") return 64u;
        char address[32];
        std::snprintf(address, sizeof(address), "%llx", static_cast<unsigned long long>(request.shader.codeAddress));
        if (list.find(address) != std::string::npos) return 64u;
    }
    return request.target.subgroupSize;
}

enum class Wave64Layout {
    TwoLanes,
    OneLane,
    WaveExchange,
};

Wave64Layout ChooseWave64Layout(const RecompileRequest& request, const IrProgram& program, const ShaderStageInputInfo& inputInfo) {
    if (request.shader.stage != ShaderStage::Compute || inputInfo.compute == nullptr || program.WaveSize() != 64u || inputInfo.compute->hostSubgroupSize != 32u) return Wave64Layout::TwoLanes;
    static const bool twoLanes = std::getenv("APS5_TWO_LANE") != nullptr;
    if (twoLanes) return Wave64Layout::TwoLanes;
    const auto& threads = inputInfo.compute->threadsNum;
    const auto workgroupThreads = std::max(threads[0], 1u) * std::max(threads[1], 1u) * std::max(threads[2], 1u);
    IrOpcode blocker = IrOpcode::Void;
    const auto rejection = Wave64OneLaneRejection(program, workgroupThreads, request.target.supportedCapabilities, &blocker);
    const auto ldsBytes = request.context.compute.has_value() ? static_cast<std::uint64_t>(request.context.compute->ldsSizeDwords) * 4u : 0u;
    const auto sharedBytes = request.target.maxWorkgroupSharedMemoryBytes;
    const bool exchangeFits = ldsBytes + 4u > sharedBytes || ldsBytes + 4u + WaveExchangeDwords * 4u <= sharedBytes;
    const auto layout = rejection == OneLaneRejection::None ? Wave64Layout::OneLane : exchangeFits && Wave64ExchangeAllowed(program, workgroupThreads, request.target.supportedCapabilities) ? Wave64Layout::WaveExchange : Wave64Layout::TwoLanes;
    static const bool log = std::getenv("APS5_WAVE64_LAYOUT_LOG") != nullptr;
    if (log) std::fprintf(stderr, "[wave64] 0x%llx %s%s%s%s\n", static_cast<unsigned long long>(request.shader.codeAddress), layout == Wave64Layout::WaveExchange ? "one lane per invocation with wave operations through workgroup memory, else " : "", OneLaneRejectionName(rejection), blocker == IrOpcode::Void ? "" : ", at ", blocker == IrOpcode::Void ? "" : std::string(IrOpcodeName(blocker)).c_str());
    return layout;
}

RdnaProgram DecodeStageProgram(ShaderStageKind stage, std::span<const std::uint32_t> code, std::span<const std::uint32_t> linkedLiterals) {
    if (stage != ShaderStageKind::Local) {
        auto program = RdnaInstructionDecoder{}.Decode(code);
        LinkRdnaLiterals(program, linkedLiterals);
        return program;
    }
    if (!linkedLiterals.empty()) throw std::invalid_argument("a local shader has linked literals");
    auto program = DecodeRdnaFrontProgram(code);
    static constexpr std::array<std::uint32_t, 1> EndProgram{0xbf810000u};
    program.instructions.back() = DecodeRdnaInstruction(program.instructions.back().programCounter, EndProgram, 0u);
    return program;
}

ShaderStageInputInfo RequestInputInfo(const RecompileRequest& request) {
    const auto* mesh = request.graphics && request.graphics->mesh ? &*request.graphics->mesh : nullptr;
    std::optional<TessellationConfiguration> tessellationCopy;
    if (request.graphics && request.graphics->tessellation) {
        tessellationCopy = *request.graphics->tessellation;
        tessellationCopy->instancePatches = TessellationInstancePatches(request);
        const auto& capabilities = request.target.supportedCapabilities;
        tessellationCopy->drawParameters = std::find(capabilities.begin(), capabilities.end(), static_cast<std::uint32_t>(spv::CapabilityDrawParameters)) != capabilities.end();
    }
    const auto* tessellation = tessellationCopy ? &*tessellationCopy : nullptr;
    return BuildShaderStageInputInfo(toShaderStageKind(request.shader.stage), request.context, HostSubgroupSize(request), mesh, tessellation);
}

std::uint32_t DeviceMemoryLdsBytes(const RecompileRequest& request) {
    if (!request.context.compute.has_value()) return 0u;
    const auto bytes = static_cast<std::uint64_t>(request.context.compute->ldsSizeDwords) * 4u;
    return bytes + 4u > request.target.maxWorkgroupSharedMemoryBytes ? static_cast<std::uint32_t>(bytes) : 0u;
}

std::atomic<std::uint64_t> frontEndBuilds{0};

}

struct PreparedControlFlow {
    PreparedControlFlow(const RecompileRequest& request, const SwappcInfo& swappcInfo)
        : stage(request.shader.stage), swappc(swappcInfo), code(request.shader.code.begin(), request.shader.code.end()),
          linkedLiterals(request.shader.linkedLiterals.begin(), request.shader.linkedLiterals.end()),
          decoded(DecodeStageProgram(toShaderStageKind(request.shader.stage), code, linkedLiterals)), cfg(GraphBuilder{}.Build(decoded, &swappc)) {
        Structurizer{}.Structurize(cfg);
        AssignCallFunctions(cfg);
    }

    PreparedControlFlow(const PreparedControlFlow&) = delete;
    PreparedControlFlow& operator=(const PreparedControlFlow&) = delete;

    bool Matches(const RecompileRequest& request, const SwappcInfo& swappcInfo) const {
        return stage == request.shader.stage && swappc.fetchCallAllowed == swappcInfo.fetchCallAllowed &&
               swappc.userDataBaseRegister == swappcInfo.userDataBaseRegister && swappc.userDataCount == swappcInfo.userDataCount &&
               std::ranges::equal(code, request.shader.code) && std::ranges::equal(linkedLiterals, request.shader.linkedLiterals);
    }

    ShaderStage stage;
    SwappcInfo swappc;
    std::vector<std::uint32_t> code;
    std::vector<std::uint32_t> linkedLiterals;
    RdnaProgram decoded;
    ControlFlowGraph cfg;
};

std::shared_ptr<const PreparedControlFlow> ShaderPreparationContext::AcquireFrontend(const RecompileRequest& request) {
    const auto inputInfo = RequestInputInfo(request);
    SwappcInfo swappc{inputInfo.vertex != nullptr, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size())};
    swappc.callFunctions = request.shader.stage == ShaderStage::Compute;
    static const bool reuse = std::getenv("APS5_NO_PERF_FRONTEND_PAIR") == nullptr;
    if (reuse && frontend != nullptr && frontend->Matches(request, swappc)) {
        return frontend;
    }
    auto prepared = std::make_shared<const PreparedControlFlow>(request, swappc);
    if (reuse) {
        frontend = prepared;
    }
    return prepared;
}

std::uint64_t FrontEndBuilds() {
    return frontEndBuilds.load(std::memory_order_relaxed);
}

std::uint32_t ShaderPreparationContext::ResourcePrograms() const {
    return resourcePrograms;
}

IrProgram PrepareResourceProgram(const RecompileRequest& request) {
    return PrepareResourceProgram(request, nullptr);
}

IrProgram PrepareResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation) {
    frontEndBuilds.fetch_add(1, std::memory_order_relaxed);
    ShaderPreparationContext local;
    auto& context = preparation != nullptr ? *preparation : local;
    ++context.resourcePrograms;
    context.interfaceSource.reset();
    context.interfaceProgram.reset();
    const auto frontend = context.AcquireFrontend(request);
    const auto stageKind = toShaderStageKind(request.shader.stage);
    const auto inputInfo = RequestInputInfo(request);
    const auto& decoded = frontend->decoded;
    const auto& cfg = frontend->cfg;

    TranslateOptions translateOptions {};
    translateOptions.stage = stageKind;
    translateOptions.shaderHash = request.shader.codeAddress;
    translateOptions.waveSize = request.context.waveSize;
    translateOptions.userDataBaseRegister = request.context.userDataBaseRegister;
    translateOptions.userDataCount = static_cast<std::uint32_t>(request.context.userData.size());
    translateOptions.scratchDwords = request.context.compute.has_value() ? request.context.compute->scratchDwords : 0u;
    translateOptions.sharedMemoryBytes = DeviceMemoryLdsBytes(request);
    translateOptions.fragmentShaderBarycentricEnabled = request.target.fragmentShaderBarycentricEnabled;
    const auto supports = [&](spv::Capability capability) { return std::ranges::find(request.target.supportedCapabilities, static_cast<std::uint32_t>(capability)) != request.target.supportedCapabilities.end(); };
    translateOptions.nativeFloat16 = supports(spv::CapabilityFloat16) && supports(spv::CapabilityRoundingModeRTE) && supports(spv::CapabilityDenormPreserve);
    translateOptions.floatMode = request.context.floatMode;
    translateOptions.inputInfo = inputInfo;

    constexpr InstructionTranslator translator;

    EmbeddedFetchPlan embeddedFetch;
    if ((stageKind == ShaderStageKind::Vertex || stageKind == ShaderStageKind::Local) && inputInfo.vertex != nullptr && inputInfo.vertex->fetchEmbedded) {
        embeddedFetch = EmbeddedVertexFetchAnalyzer{}.Analyze(decoded, inputInfo.vertex->fetchAttribReg, inputInfo.vertex->fetchBufferReg, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size()), request.context.waveSize);
        translateOptions.embeddedFetch = &embeddedFetch;
    }
    auto program = translator.Translate(decoded, cfg, translateOptions);
    // Debug aid: APS5_DUMP_IR=<hex code address> (or "all") prints the program after each front-end pass.
    const auto dumpIr = [&](const char* pass) {
        static const std::string list = [] { const char* text = std::getenv("APS5_DUMP_IR"); return text ? std::string(text) : std::string(); }();
        if (list.empty()) return;
        char address[32];
        std::snprintf(address, sizeof(address), "%llx", static_cast<unsigned long long>(request.shader.codeAddress));
        if (list != "all" && list.find(address) == std::string::npos) return;
        std::fprintf(stderr, "==== IR 0x%s after %s\n%s\n", address, pass, ProgramToString(program).c_str());
    };

    constexpr SsaBuilder ssaBuilder;
    constexpr ConstantFolder constantFolder;
    constexpr DeadCodeEliminator deadCodeEliminator;
    constexpr ReadLaneEliminator readLaneEliminator;
    constexpr MaskedSelectEliminator maskedSelectEliminator;
    const auto simplify = [&] {
        dumpIr("translate");
        ssaBuilder.Rewrite(program);
        dumpIr("ssa");

        constantFolder.Fold(program);
        ResolveControlFlowIdentities(program);
        deadCodeEliminator.RemoveIdentities(program);
        deadCodeEliminator.Eliminate(program);
        dumpIr("fold");

        const auto readLaneStats = readLaneEliminator.Eliminate(program, translateOptions.waveSize);
        if (readLaneStats.rewrittenReads != 0u) {
            constantFolder.Fold(program);
            ResolveControlFlowIdentities(program);
            deadCodeEliminator.RemoveIdentities(program);
            deadCodeEliminator.Eliminate(program);
        }

        if (maskedSelectEliminator.Eliminate(program).removedSelects != 0u) {
            const bool liveAncillary = std::ranges::any_of(program.Blocks(), [](const auto& block) {
                return std::ranges::any_of(block->Instructions(), [](const IrValue* value) {
                    return value->Opcode() == IrOpcode::GetBuiltin && value->HasUses() && value->Argument(0)->Resolve()->ImmediateU32() == static_cast<std::uint32_t>(StageInputKind::PackedAncillary);
                });
            });
            if (liveAncillary) {
                constantFolder.Fold(program);
                ResolveControlFlowIdentities(program);
                deadCodeEliminator.RemoveIdentities(program);
            }
            deadCodeEliminator.Eliminate(program);
        }
        if (constantFolder.ComposePackedAncillary(program)) {
            deadCodeEliminator.Eliminate(program);
        }
    };
    simplify();
    if (stageKind == ShaderStageKind::Pixel && !translateOptions.fragmentShaderBarycentricEnabled && !HostInterpolationChecker{}.Lower(program, *inputInfo.pixel)) {
        translateOptions.fragmentShaderBarycentricEnabled = true;
        program = translator.Translate(decoded, cfg, translateOptions);
        program.Metadata().barycentricEmulation = true;
        simplify();
    }

    constexpr DenormalFlushEliminator denormalFlushEliminator;
    if (denormalFlushEliminator.Eliminate(program).removedFlushes != 0u) {
        deadCodeEliminator.Eliminate(program);
    }
    constexpr NanCheckEliminator nanCheckEliminator;
    if (nanCheckEliminator.Eliminate(program).Changed()) {
        deadCodeEliminator.Eliminate(program);
    }
    constexpr OperandIdiomMerger operandIdiomMerger;
    if (operandIdiomMerger.Merge(program).Changed()) {
        deadCodeEliminator.Eliminate(program);
    }

    constexpr VectorWindowStoreEliminator vectorWindowStoreEliminator;
    if (vectorWindowStoreEliminator.Eliminate(program).removedStores != 0u) {
        deadCodeEliminator.Eliminate(program);
    }

    if (inputInfo.vertex != nullptr) {
        constexpr TessellationLowering tessellationLowering;
        tessellationLowering.Lower(program, inputInfo.vertex->tess);
    }

    constexpr SrtWalker srtWalker;
    srtWalker.BuildPlan(program, supports(spv::CapabilityInt64) && supports(spv::CapabilityPhysicalStorageBufferAddresses));
    deadCodeEliminator.Eliminate(program);
    dumpIr("srt");

    constexpr ResourceTracker resourceTracker;
    resourceTracker.Track(program, supports(spv::CapabilityInt64) && supports(spv::CapabilityPhysicalStorageBufferAddresses));
    deadCodeEliminator.Eliminate(program);
    dumpIr("resources");
    if (request.context.compute && !request.context.compute->threadsFromMemory && !request.context.compute->PartialGroups()) program.Info().linearStore = FindLinearStore(program, request.context.compute->numThreads);
    program.Resources().srgbDecodeFormats = request.target.srgbDecodeFormats;

    return program;
}

// A materialized result of one variant over one snapshot (Recompile(request, capture)): the
// shared immutable object every later capture that reproduces the snapshot receives, so Populate
// and the per-request copy run once per distinct snapshot.
struct ResultMemoEntry {
    std::uint64_t variantId;
    std::uint64_t hash;
    std::shared_ptr<const RecompileResult> result;
};

struct EmissionFailure {
    std::uint64_t codeAddress;
    BindingLayout layout;
    std::exception_ptr failure;
};

struct SourceEntry : std::enable_shared_from_this<SourceEntry> {
    std::mutex mutex;
    // The code the entry was built for: the key carries only a hash of it, so a candidate entry is
    // accepted only when its code matches word for word. Owned here because the request's span
    // points into a registration the driver may replace while the entry lives on.
    std::vector<std::uint32_t> code;
    std::vector<std::uint32_t> linkedLiterals;
    std::shared_ptr<const IrResourcePlan> plan;
    // A plan build that threw (an unsupported resource chain or control flow) is remembered and
    // rethrown: the front end ran every pass before failing, ~13 ms per dispatch of a shader the
    // title issues every frame (0x1048947300 at the intro video). APS5_NO_FAILURE_MEMO=1 rebuilds.
    std::exception_ptr planFailure;
    std::unique_ptr<IrProgram> program;
    std::vector<std::shared_ptr<const CompiledVariant>> variants;
    std::vector<EmissionFailure> emissionFailures;
    // The result memo, most recently used first, at most ResultMemoEntries (under mutex).
    std::list<ResultMemoEntry> memo;
    std::unordered_map<std::uint64_t, std::list<ResultMemoEntry>::iterator> memoIndex;
};

namespace {

class PrimedSharedMutex : public std::shared_mutex {
public:
    PrimedSharedMutex() {
        lock();
        unlock();
    }
};

struct ResourceProgram {
    explicit ResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation = nullptr, bool reclaim = true) : program(std::make_unique<IrProgram>(PrepareResourceProgram(request, preparation))) {
        const auto clearRegisterState = [](const auto& storage) {
            for (const auto& block : storage) {
                block->ssaScalarValues.fill(nullptr);
                block->ssaVectorValues.fill(nullptr);
                block->ssaThreadBitScalarValues.fill(nullptr);
                block->ssaScalarMaskTags.fill(nullptr);
            }
        };
        clearRegisterState(program->Blocks());
        clearRegisterState(program->Resources().blockStorage);
        if (reclaim) program->ReclaimUnusedValues();
        plan = std::make_shared<const IrResourcePlan>(ResourceMaterializer{}.ExtractPlan(*program));
    }

    std::unique_ptr<IrProgram> program;
    std::shared_ptr<const IrResourcePlan> plan;
};

std::shared_ptr<const IrResourcePlan> makeResourcePlan(const RecompileRequest& request) {
    return ResourceProgram(request, nullptr, false).plan;
}

struct SourceKeyHash {
    std::size_t operator()(const std::vector<std::uint64_t>& key) const {
        std::size_t hash = 0;
        for (const auto value : key) {
            hash ^= static_cast<std::size_t>(value) + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) + (hash >> 2u);
            if constexpr (sizeof(std::size_t) < sizeof(value)) hash ^= static_cast<std::size_t>(value >> 32u);
        }
        return hash;
    }
};

bool FailureMemo() {
    static const bool memo = std::getenv("APS5_NO_FAILURE_MEMO") == nullptr;
    return memo;
}

bool diskPlans() {
    static const bool enabled = [] {
        const char* dumpIr = std::getenv("APS5_DUMP_IR");
        return std::getenv("APS5_NO_PERF_DISK_PLANS") == nullptr && (dumpIr == nullptr || *dumpIr == '\0');
    }();
    return enabled && ShaderDiskCache::Enabled() && !DebugProbeActive();
}

std::shared_ptr<SourceEntry> getSource(const RecompileRequest& request, ShaderPreparationContext* preparation = nullptr, bool reclaim = true) {
    static PrimedSharedMutex mutex;
    // Entries whose code hashes alike share a bucket; the code comparison picks the right one.
    static std::unordered_map<std::vector<std::uint64_t>, std::vector<std::shared_ptr<SourceEntry>>, SourceKeyHash> sources;
    struct SourceKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, SourceKeyStorage>();
    RecompileCacheKey::Build(request, key);
    key.push_back(HostSubgroupSize(request));
    const auto find = [&]() -> std::shared_ptr<SourceEntry> {
        const auto found = sources.find(key);
        if (found == sources.end()) return nullptr;
        for (const auto& entry : found->second) {
            if (std::equal(entry->code.begin(), entry->code.end(), request.shader.code.begin(), request.shader.code.end())) return entry;
        }
        return nullptr;
    };
    std::shared_ptr<SourceEntry> source;
    {
        std::shared_lock lock(mutex);
        source = find();
    }
    if (source == nullptr) {
        std::unique_lock lock(mutex);
        source = find();
        if (source == nullptr) {
            source = std::make_shared<SourceEntry>();
            source->code.assign(request.shader.code.begin(), request.shader.code.end());
            source->linkedLiterals.assign(request.shader.linkedLiterals.begin(), request.shader.linkedLiterals.end());
            auto& bucket = sources[key];
            if (!bucket.empty()) {
                // A second entry under one key is a code hash collision (or the unhashed key with
                // identical code, which cannot happen); each is reported under the profile switch.
                static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                static std::uint64_t collisions = 0;
                ++collisions;
                if (profile) std::fprintf(stderr, "[recompile] source key collision %llu: %zu entries share a key (%zu code words)\n", static_cast<unsigned long long>(collisions), bucket.size() + 1, request.shader.code.size());
            }
            bucket.push_back(source);
        }
    }
    {
        std::lock_guard lock(source->mutex);
        if (source->plan == nullptr) {
            if (FailureMemo() && source->planFailure) std::rethrow_exception(source->planFailure);
            const bool disk = diskPlans();
            std::vector<std::byte> planKey;
            if (disk) {
                ShaderDiskCache::BuildPlanKey(request, HostSubgroupSize(request), planKey);
                IrResourcePlan loaded;
                if (ShaderDiskCache::LoadPlan(planKey, loaded)) {
                    loaded.shaderHash = request.shader.codeAddress;
                    source->plan = std::make_shared<const IrResourcePlan>(std::move(loaded));
                }
            }
            if (source->plan == nullptr) {
                try {
                    ResourceProgram resource(request, preparation, reclaim);
                    source->plan = std::move(resource.plan);
                    source->program = std::move(resource.program);
                } catch (...) {
                    if (FailureMemo()) source->planFailure = std::current_exception();
                    throw;
                }
                if (disk) ShaderDiskCache::StorePlan(std::move(planKey), source->plan);
            }
        }
    }
    return source;
}

std::array<std::uint32_t, 3> partialThreads(const RecompileRequest& request) {
    return request.context.compute ? request.context.compute->partialThreads : std::array<std::uint32_t, 3>{};
}

std::optional<std::pair<std::uint64_t, std::uint64_t>> BufferStoreExtent(const DescriptorValue& value, const BufferStoreSite& site);

}

std::optional<LinearStore> MaterializeLinearStore(const ShaderInfo& info, std::span<const DescriptorValue> stores) {
    if (!info.linearStore || stores.size() != 1u || stores.front().dwordCount < 4u) return std::nullopt;
    const auto& words = stores.front().dwords;
    const auto stride = (words[1] >> 16u) & 0x3fffu;
    const bool swizzled = stride != 0u && (words[1] >> 31u) != 0u;
    const bool laneIndexed = ((words[3] >> 23u) & 1u) != 0u;
    if (swizzled || laneIndexed || (words[3] >> 30u) != 0u) return std::nullopt;
    auto elementBytes = info.linearStore->elementBytes;
    if (info.linearStore->descriptorFormat) {
        const auto format = GetFormatInfo(static_cast<IrBufferFormat>((words[3] >> 12u) & 0x7fu));
        if (format.type == SpirvFormatComponentType::Unknown || format.componentCount == 0u || format.componentCount > info.linearStore->components) return std::nullopt;
        std::uint32_t bits = 0;
        for (std::uint32_t component = 0; component < format.componentCount; ++component) bits += format.componentBits[component];
        elementBytes = format.packedBitfield ? 4u : bits / 8u;
    }
    if (elementBytes == 0u) return std::nullopt;
    const auto mode = (words[3] >> 28u) & 3u;
    if (mode == 3u && words[2] != 0u && words[2] < elementBytes) return std::nullopt;
    auto records = mode < 2u || words[2] == 0u ? words[2] : std::numeric_limits<std::uint32_t>::max();
    if (info.linearStore->groupIdBits < 32u) records = static_cast<std::uint32_t>(std::min<std::uint64_t>(records, (std::uint64_t{1} << info.linearStore->groupIdBits) * info.linearStore->groupThreads));
    return LinearStore{static_cast<std::uint64_t>(words[0]) | static_cast<std::uint64_t>(words[1] & 0xffffu) << 32u, stride, records, elementBytes};
}

void ApplyBufferRanges(const ShaderInfo& info, std::span<const DescriptorValue> stores, std::span<const DescriptorValue> loads, bool memory, ShaderInvocation& result) {
    const auto apply = [&](const std::vector<BufferStoreSite>& sites, std::span<const DescriptorValue> values, bool infoUnbounded, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, bool& unbounded, bool& fixedUnbounded, std::size_t& memoryRanges) {
        if (memory) {
            ranges.resize(ranges.size() - memoryRanges);
            memoryRanges = 0;
            unbounded = fixedUnbounded;
        } else {
            ranges.clear();
            memoryRanges = 0;
            unbounded = infoUnbounded || values.size() != sites.size();
        }
        const auto first = ranges.size();
        for (std::size_t index = 0; index < values.size() && index < sites.size(); ++index) {
            if (sites[index].fromMemory != memory) continue;
            const auto range = BufferStoreExtent(values[index], sites[index]);
            if (!range.has_value()) unbounded = true;
            else if (range->second > range->first) ranges.push_back(*range);
        }
        std::sort(ranges.begin() + static_cast<std::ptrdiff_t>(first), ranges.end());
        auto merged = first;
        for (auto index = first; index < ranges.size(); ++index) {
            if (merged != first && ranges[index].first <= ranges[merged - 1u].second) ranges[merged - 1u].second = std::max(ranges[merged - 1u].second, ranges[index].second);
            else ranges[merged++] = ranges[index];
        }
        ranges.resize(merged);
        if (memory) memoryRanges = merged - first;
        if (!memory) fixedUnbounded = unbounded;
    };
    apply(info.bufferStores, stores, info.unboundedStores, result.writeRanges, result.unboundedWrites, result.fixedUnboundedWrites, result.memoryWriteRanges);
    apply(info.bufferLoads, loads, info.unboundedLoads, result.readRanges, result.unboundedReads, result.fixedUnboundedReads, result.memoryReadRanges);
    if (!memory) result.linearStore = MaterializeLinearStore(info, stores);
}

bool RefreshMemoryBuffers(const IrResourcePlan& plan, const SrtRuntime& runtime, const ShaderInvocation& current, ShaderInvocation& refreshed) {
    std::vector<DescriptorValue> stores;
    std::vector<DescriptorValue> loads;
    ResourceMaterializer::EvaluateMemoryBuffers(plan, runtime, stores, loads);
    ShaderInvocation ranges;
    ranges.writeRanges.assign(current.writeRanges.begin(), current.writeRanges.end() - static_cast<std::ptrdiff_t>(current.memoryWriteRanges));
    ranges.readRanges.assign(current.readRanges.begin(), current.readRanges.end() - static_cast<std::ptrdiff_t>(current.memoryReadRanges));
    ranges.fixedUnboundedWrites = current.fixedUnboundedWrites;
    ranges.fixedUnboundedReads = current.fixedUnboundedReads;
    ApplyBufferRanges(plan.info, stores, loads, true, ranges);
    if (ranges.writeRanges == current.writeRanges && ranges.readRanges == current.readRanges && ranges.unboundedWrites == current.unboundedWrites && ranges.unboundedReads == current.unboundedReads) return false;
    refreshed.writeRanges = std::move(ranges.writeRanges);
    refreshed.readRanges = std::move(ranges.readRanges);
    refreshed.unboundedWrites = ranges.unboundedWrites;
    refreshed.unboundedReads = ranges.unboundedReads;
    refreshed.memoryWriteRanges = ranges.memoryWriteRanges;
    refreshed.memoryReadRanges = ranges.memoryReadRanges;
    return true;
}

namespace {

std::optional<std::pair<std::uint64_t, std::uint64_t>> BufferStoreExtent(const DescriptorValue& value, const BufferStoreSite& site) {
    if (site.address) {
        if (value.dwordCount != 2u && value.dwordCount != 3u) return std::nullopt;
        constexpr std::uint64_t addressMask = (std::uint64_t{1} << 48u) - 1u;
        const auto base = (static_cast<std::uint64_t>(value.dwords[0]) | (static_cast<std::uint64_t>(value.dwords[1]) << 32u)) & addressMask & ~std::uint64_t{3u};
        if (base == 0u) return std::pair<std::uint64_t, std::uint64_t>{0u, 0u};
        const auto registerOffset = value.dwordCount == 3u ? static_cast<std::uint64_t>(value.dwords[2] & ~3u) : std::uint64_t{0u};
        const auto first = base + registerOffset + static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(site.offset)));
        return std::pair<std::uint64_t, std::uint64_t>{first, first + ((site.bytes + 3u) & ~std::uint64_t{3u})};
    }
    const bool registerSoffset = site.dynamicSoffset && value.dwordCount == 5u;
    if (value.dwordCount != 4u && !registerSoffset) return std::nullopt;
    const auto soffset = registerSoffset ? static_cast<std::uint64_t>(value.dwords[4]) : static_cast<std::uint64_t>(site.soffset);
    const auto word1 = value.dwords[1];
    const auto records = value.dwords[2];
    const auto word3 = value.dwords[3];
    const auto base = static_cast<std::uint64_t>(value.dwords[0]) | (static_cast<std::uint64_t>(word1 & 0xffffu) << 32u);
    const std::pair<std::uint64_t, std::uint64_t> none{0u, 0u};
    if ((!site.scalar && (word3 >> 30u) != 0u) || base == 0u || records == 0u) return none;
    const auto stride = (word1 >> 16u) & 0x3fffu;
    if (site.scalar) return std::pair<std::uint64_t, std::uint64_t>{base & ~std::uint64_t{3u}, base + (stride == 0u ? records : static_cast<std::uint64_t>(records) * stride)};
    if (stride != 0u && (word1 >> 31u) != 0u) return std::nullopt;
    const auto mode = (word3 >> 28u) & 3u;
    if (mode == 3u) {
        if (stride != 0u && site.dynamicIndex) return std::nullopt;
        const auto first = static_cast<std::uint64_t>(site.index) * stride;
        const auto lanes = ((word3 >> 23u) & 1u) != 0u ? std::uint64_t{63u} * stride : std::uint64_t{0u};
        return std::pair<std::uint64_t, std::uint64_t>{base + first, base + first + lanes + records};
    }
    if ((site.dynamicSoffset && !registerSoffset) || (mode != 0u && (mode != 1u || site.dynamicOffset))) return std::nullopt;
    if (mode == 0u && stride == 0u) return none;
    constexpr std::uint64_t wrap = std::uint64_t{1} << 32u;
    const auto reach = mode == 0u ? static_cast<std::uint64_t>(records) * stride : static_cast<std::uint64_t>(records - 1u) * stride + site.offset;
    return std::pair<std::uint64_t, std::uint64_t>{base, base + std::min(reach + soffset + 16u, wrap)};
}

bool DeclaresSpecializationId(std::span<const std::uint32_t> words, std::uint32_t id) {
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        if (count == 0u) return false;
        if ((words[cursor] & 0xffffu) == spv::OpDecorate && count == 4u && cursor + 3u < words.size() && words[cursor + 2u] == spv::DecorationSpecId && words[cursor + 3u] == id) return true;
        cursor += count;
    }
    return false;
}

std::uint64_t nextVariantId() {
    static std::atomic<std::uint64_t> variants{0};
    return variants.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::array<std::uint64_t, 2> identityOf(std::span<const std::byte> diskKey) {
    return {HashBytes(diskKey, 0x5eed0001ull), HashBytes(diskKey, 0x5eed0002ull)};
}

std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources> vertexClassesOf(std::span<const PipelineSpecializationConstant> constants) {
    constexpr auto first = PipelineSpecialization::VertexBase;
    constexpr auto words = PipelineSpecialization::VertexWords;
    std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources> classes{};
    for (const auto& constant : constants) {
        if (constant.id < first || constant.id >= first + classes.size() * words || (constant.id - first) % words != 5u) continue;
        classes[(constant.id - first) / words] = constant.value;
    }
    return classes;
}

template<typename TStep>
auto recordEmissionFailure(std::exception_ptr* emissionFailure, TStep&& step) -> decltype(step()) {
    try {
        return step();
    } catch (const std::bad_alloc&) {
        throw;
    } catch (...) {
        if (emissionFailure != nullptr) *emissionFailure = std::current_exception();
        throw;
    }
}

void prepareInterface(const RecompileRequest& request, IrProgram& program) {
    const auto inputInfo = RequestInputInfo(request);
    constexpr DeadCodeEliminator deadCodeEliminator;
    constexpr ResourceMaterializer resourceMaterializer;
    const bool nativeSampleOffsets = request.target.nonConstantImageOffsets && std::find(request.target.supportedCapabilities.begin(), request.target.supportedCapabilities.end(), spv::CapabilityImageGatherExtended) != request.target.supportedCapabilities.end();
    resourceMaterializer.ApplyStaticInterface(program, nativeSampleOffsets);

    constexpr ConstantFolder constantFolder;
    if (request.shader.stage == ShaderStage::Vertex && constantFolder.FoldVertexLaneBounds(program, HostSubgroupSize(request), request.target.spirvVersion)) {
        constantFolder.Fold(program);
        ResolveControlFlowIdentities(program);
    }

    deadCodeEliminator.RemoveIdentities(program);
    constexpr RedundantValueEliminator redundantValueEliminator;
    static_cast<void>(redundantValueEliminator.Eliminate(program));
    deadCodeEliminator.Eliminate(program);

    constexpr ShaderInfoCollector shaderInfoCollector;
    shaderInfoCollector.Collect(program, inputInfo);
}

CompiledVariant emitVariant(const RecompileRequest& request, IrProgram& program) {
    const auto inputInfo = RequestInputInfo(request);
    constexpr BindingAllocator bindingAllocator;
    bool registration = request.target.registrationModules && (request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Fragment);
    program.Metadata().bindingLayoutComplete = false;
    const bool stencilReplay = request.shader.stage == ShaderStage::Fragment && std::ranges::any_of(program.Info().outputs, [](const StageOutput& output) { return output.kind == StageOutputKind::StencilRef; }) && std::ranges::find(request.target.supportedCapabilities, static_cast<std::uint32_t>(spv::CapabilityStencilExportEXT)) == request.target.supportedCapabilities.end();
    auto bindings = bindingAllocator.Allocate(program, request.layout, registration, stencilReplay);

    SpirvTargetOptions targetOptions {};
    targetOptions.vulkanVersion = request.target.vulkanVersion;
    targetOptions.spirvVersion = request.target.spirvVersion;
    targetOptions.subgroupSize = HostSubgroupSize(request) == 1u ? 1u : request.target.subgroupSize;
    targetOptions.bdaAbiVersion = request.target.bdaAbiVersion;
    targetOptions.supportedCapabilities = request.target.supportedCapabilities;
    targetOptions.supportedExtensions = request.target.supportedExtensions;
    targetOptions.nonConstantImageOffsets = request.target.nonConstantImageOffsets;
    targetOptions.narrowSubgroupClock = request.target.narrowSubgroupClock;
    targetOptions.float64DenormPreserve = request.target.float64DenormPreserve;
    targetOptions.registrationModules = registration;
    targetOptions.fragmentDescriptorSet = request.target.registrationModules;
    targetOptions.float16DenormPreserve = request.target.float16DenormPreserve;

    ShaderStageInputInfo emitInputInfo = inputInfo;
    ShaderComputeInputInfo oneLaneCompute{};
    if (const auto layout = ChooseWave64Layout(request, program, inputInfo); layout != Wave64Layout::TwoLanes) {
        oneLaneCompute = *inputInfo.compute;
        oneLaneCompute.hostSubgroupSize = 64u;
        oneLaneCompute.waveExchange = layout == Wave64Layout::WaveExchange;
        emitInputInfo.compute = &oneLaneCompute;
    }

    constexpr SpirvEmitter spirvEmitter;
    CompiledShaderArtifact result;
    result.variantId = nextVariantId();
    result.spirv = spirvEmitter.Emit(program, emitInputInfo, bindings, targetOptions);
    if (registration && result.spirv.size() > request.target.registrationWordLimit) {
        registration = false;
        program.Metadata().bindingLayoutComplete = false;
        bindings = bindingAllocator.Allocate(program, request.layout, false, stencilReplay);
        targetOptions.registrationModules = false;
        result.spirv = spirvEmitter.Emit(program, emitInputInfo, bindings, targetOptions);
    }
    result.registrationModule = registration;
    result.stencilReplaySafe = (request.shader.stage == ShaderStage::Vertex || stencilReplay) && StencilReplaySafe(program);
    if (stencilReplay) {
        result.stencilReferences = StencilReferenceValues(program);
        result.stencilReferenceDword = bindings.layout.stencilReferenceDword;
    }

    result.bdaAbiVersion = program.Info().usesDma || program.Info().usesFaultBuffer ? request.target.bdaAbiVersion : 0u;
    result.bdaWrites = program.Info().usesDma && program.Info().bdaWrites;
    result.imageStoresRecorded = ImageStoresRecorded(program, targetOptions.supportedCapabilities);
    result.usesShaderBase = DeclaresSpecializationId(result.spirv.Words(), PipelineSpecialization::ShaderBaseLow);
    result.memoryOffsetDword = bindings.layout.memoryOffsetDword;
    result.bufferAddressDword = bindings.layout.BufferAddressDword();
    result.samplerSlotDword = bindings.layout.SamplerSlotDword();
    result.dispatchThreadLimitDword = bindings.layout.DispatchThreadLimitDword();
    result.shaderDataDwords = bindings.layout.ShaderDataDwords();
    result.imageMetadataDword = bindings.layout.ImageMetadataDword();
    result.runtimeImageCount = bindings.layout.runtimeImageCount;
    result.hostSubgroupSize = HostSubgroupSize(request);
    result.vertexOffsetSgpr = program.Info().vertexOffsetSgpr;
    result.instanceOffsetSgpr = program.Info().instanceOffsetSgpr;
    result.vertexOffsetShared = program.Info().vertexOffsetShared;
    result.instanceOffsetShared = program.Info().instanceOffsetShared;
    result.vertexOffsetConflict = program.Info().vertexOffsetConflict;
    result.instanceOffsetConflict = program.Info().instanceOffsetConflict;
    for (const auto& output : program.Info().outputs) {
        if (output.kind == StageOutputKind::Parameter) result.parameterExports.push_back(output.location);
    }
    if (request.shader.stage == ShaderStage::Fragment) {
        result.fragmentParameters = DescribeFragmentParameters(program, inputInfo);
        const auto& inputs = program.Info().inputs;
        const auto reads = [&](StageInputKind kind) { return std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) { return input.kind == kind; }); };
        if (program.Metadata().barycentricEmulation) result.barycentricEmulation = {true, reads(StageInputKind::BaryCoordSmooth), reads(StageInputKind::BaryCoordNoPerspective)};
    }
    if (request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Local) {
        if (inputInfo.vertex == nullptr) throw std::runtime_error("vertex input metadata is missing");
        for (const auto& input : program.Info().inputs) {
            if (input.kind != StageInputKind::Parameter) continue;
            if (input.location >= static_cast<std::uint32_t>(inputInfo.vertex->resourcesNum)) throw std::runtime_error("vertex attribute location exceeds resource count");
            result.vertexInputs.push_back({input.location, input.componentCount, inputInfo.vertex->resourcesDst[input.location].fetchIndex});
            if (inputInfo.vertex->fetchEmbedded) {
                for (const auto& block : program.Blocks()) {
                    for (const auto* instruction : block->Instructions()) {
                        if (instruction->Opcode() == IrOpcode::GetAttribute && instruction->Argument(0)->Resolve()->ImmediateU32() == input.location) result.vertexInputs.back().outputMask |= 1u << instruction->Argument(1)->Resolve()->ImmediateU32();
                    }
                }
            }
        }
    }

    const bool embedded = !result.vertexInputs.empty() && inputInfo.vertex != nullptr && inputInfo.vertex->fetchEmbedded;
    if (embedded) PrepareVertexInputSpecialization(result);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    result.spirv = ValidateAndOptimizeSpirv(result.spirv, request.target.vulkanVersion, request.target.spirvVersion, request.target.nonConstantImageOffsets, !embedded);
    if (embedded) {
        std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources> classes{};
        for (std::uint32_t kind = 1u; kind < 3u; ++kind) {
            classes.fill(kind);
            const auto specialized = SpecializeVertexInputTypes(result, classes);
            static_cast<void>(ValidateAndOptimizeSpirv(specialized, request.target.vulkanVersion, request.target.spirvVersion, request.target.nonConstantImageOffsets, false));
        }
    }
#endif

    return {request.layout, {}, std::move(static_cast<CompiledBindingLayout&>(bindings)), std::move(result)};
}

CompiledVariant compileVariant(const RecompileRequest& request, IrProgram program, std::exception_ptr* emissionFailure = nullptr) {
    return recordEmissionFailure(emissionFailure, [&] {
        prepareInterface(request, program);
        auto variant = emitVariant(request, program);
        variant.info = std::move(program).TakeCompiledInfo();
        return variant;
    });
}

struct SpecializedModule {
    std::uint64_t specializationId = 0;
    std::vector<PipelineSpecializationConstant> constants;
    SharedSpirv spirv;
    std::vector<std::uint32_t> bindings;
    bool pushData = false;
    bool samplerHeap = false;
    bool imageHeap = false;
};

std::shared_ptr<const SpecializedModule> buildSpecializedModule(const CompiledShaderArtifact& artifact, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants, const SpirvTarget& target) {
    auto source = SpecializeVertexInputTypes(artifact, classes);
    std::map<std::uint32_t, std::uint32_t> supplied;
    for (const auto& constant : constants) {
        if (!supplied.emplace(constant.id, constant.value).second) throw std::runtime_error("duplicate prepared specialization ID");
    }
    std::map<std::uint32_t, std::uint32_t> values;
    std::set<std::uint32_t> specializedIds;
    std::set<std::uint32_t> pushTypes;
    std::optional<std::uint32_t> replaySlotBytes;
    const auto& words = source.Words();
    if (words.size() < 5u || words[0] != spv::MagicNumber) throw std::runtime_error("invalid prepared specialization module");
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
        if (count == 0u || count > words.size() - cursor) throw std::runtime_error("truncated prepared specialization instruction");
        if (!artifact.stencilReferences.empty() && op == spv::OpTypePointer && count == 4u && words[cursor + 2u] == spv::StorageClassPushConstant) pushTypes.insert(words[cursor + 3u]);
        if (op == spv::OpDecorate && count == 4u && words[cursor + 2u] == spv::DecorationSpecId) {
            const auto found = supplied.find(words[cursor + 3u]);
            if (found == supplied.end()) throw std::runtime_error("missing prepared specialization value");
            auto value = found->second;
            if (!artifact.stencilReferences.empty()) {
                if (found->first == PipelineSpecialization::StencilReferenceOffset) replaySlotBytes = value / PushData::DwordCount * NativePushSlotSize;
                if (found->first == PipelineSpecialization::StencilReferenceOffset || found->first == PipelineSpecialization::PushDataOffset) value %= PushData::DwordCount;
            }
            if (!values.emplace(words[cursor + 1u], value).second) throw std::runtime_error("duplicate prepared specialization value");
            specializedIds.insert(found->first);
        }
        cursor += count;
    }
    std::vector<std::uint32_t> materialized(words.begin(), words.begin() + 5);
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
        if (op == spv::OpSpecConstant) {
            if (count != 4u || !values.contains(words[cursor + 2u])) throw std::runtime_error("invalid prepared specialization constant");
            materialized.insert(materialized.end(), {(4u << 16u) | spv::OpConstant, words[cursor + 1u], words[cursor + 2u], values.at(words[cursor + 2u])});
        } else if (op == spv::OpMemberDecorate && count == 5u && words[cursor + 3u] == spv::DecorationOffset && pushTypes.contains(words[cursor + 1u])) {
            if (!replaySlotBytes || words[cursor + 2u] != 0u || *replaySlotBytes >= NativePushConstantSize) throw std::runtime_error("invalid stencil replay push constant slot");
            materialized.insert(materialized.end(), words.begin() + cursor, words.begin() + cursor + count - 1u);
            materialized.push_back(*replaySlotBytes);
        } else if (!(op == spv::OpDecorate && count == 4u && words[cursor + 2u] == spv::DecorationSpecId)) {
            materialized.insert(materialized.end(), words.begin() + cursor, words.begin() + cursor + count);
        }
        cursor += count;
    }
    materialized = SpecializeFragmentOutputs(SpecializeSpirv(materialized), constants, specializedIds);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    materialized = ValidateAndOptimizeSpirv(materialized, target.vulkanVersion, target.spirvVersion, target.nonConstantImageOffsets, true, true);
#endif
    std::map<std::uint32_t, std::uint32_t> descriptorVariables;
    std::set<std::uint32_t> usedDescriptors;
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationBinding) descriptorVariables.emplace(materialized[cursor + 1u], materialized[cursor + 3u]);
        if ((op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain || op == spv::OpPtrAccessChain || op == spv::OpInBoundsPtrAccessChain || op == spv::OpLoad || op == spv::OpCopyObject) && count >= 4u) usedDescriptors.insert(materialized[cursor + 3u]);
        cursor += count;
    }
    const auto unused = [&](std::uint32_t id) { return descriptorVariables.contains(id) && !usedDescriptors.contains(id); };
    std::vector<std::uint32_t> compact(materialized.begin(), materialized.begin() + 5);
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpEntryPoint) {
            const auto start = compact.size();
            std::size_t interfaceIndex = 3u;
            for (; interfaceIndex < count; ++interfaceIndex) {
                const auto word = materialized[cursor + interfaceIndex];
                if ((word & 0xffu) == 0u || (word & 0xff00u) == 0u || (word & 0xff0000u) == 0u || (word & 0xff000000u) == 0u) { ++interfaceIndex; break; }
            }
            compact.insert(compact.end(), materialized.begin() + cursor, materialized.begin() + cursor + interfaceIndex);
            for (auto index = interfaceIndex; index < count; ++index) if (!unused(materialized[cursor + index])) compact.push_back(materialized[cursor + index]);
            compact[start] = (static_cast<std::uint32_t>(compact.size() - start) << 16u) | spv::OpEntryPoint;
        } else if (!((op == spv::OpVariable && unused(materialized[cursor + 2u])) || ((op == spv::OpDecorate || op == spv::OpName) && unused(materialized[cursor + 1u])))) {
            compact.insert(compact.end(), materialized.begin() + cursor, materialized.begin() + cursor + count);
        }
        cursor += count;
    }
    materialized = std::move(compact);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    materialized = ValidateAndOptimizeSpirv(materialized, target.vulkanVersion, target.spirvVersion, target.nonConstantImageOffsets, false);
#endif
    auto module = std::make_shared<SpecializedModule>();
    std::map<std::uint32_t, std::uint32_t> bindingNumbers;
    std::set<std::uint32_t> heapVariables;
    std::set<std::uint32_t> workgroupVariables;
    bool compute = false;
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpEntryPoint) compute = materialized[cursor + 1u] == spv::ExecutionModelGLCompute;
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationBinding) bindingNumbers.emplace(materialized[cursor + 1u], materialized[cursor + 3u]);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationDescriptorSet && materialized[cursor + 3u] == RuntimeAbi::HeapDescriptorSet) heapVariables.insert(materialized[cursor + 1u]);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationDescriptorSet && compute && materialized[cursor + 3u] == WorkgroupMemoryDescriptorSet) workgroupVariables.insert(materialized[cursor + 1u]);
        if (op == spv::OpVariable && count >= 4u) {
            if (const auto found = bindingNumbers.find(materialized[cursor + 2u]); found != bindingNumbers.end() && !heapVariables.contains(materialized[cursor + 2u]) && !workgroupVariables.contains(materialized[cursor + 2u])) module->bindings.push_back(found->second);
            module->pushData |= materialized[cursor + 3u] == spv::StorageClassPushConstant;
            if (heapVariables.contains(materialized[cursor + 2u])) (bindingNumbers.at(materialized[cursor + 2u]) == RuntimeAbi::HeapSamplerBinding ? module->samplerHeap : module->imageHeap) = true;
        }
        cursor += count;
    }
    module->spirv = std::move(materialized);
    module->constants.assign(constants.begin(), constants.end());
    module->specializationId = constants.empty() ? 0u : nextVariantId();
    return module;
}

struct SpecializedModuleEntry {
    std::once_flag ready;
    std::exception_ptr failure;
    std::shared_ptr<const SpecializedModule> module;
};

struct PreparedModuleEntry {
    std::once_flag ready;
    std::exception_ptr failure;
    std::shared_ptr<const SpecializedModule> module;
    DescriptorBindingPlan bindings;
};

struct PreparedBindingPlan {
    std::once_flag ready;
    std::exception_ptr failure;
    DescriptorBindingPlan bindings;
    PrimedSharedMutex mutex;
    std::map<std::vector<std::uint32_t>, std::shared_ptr<PreparedModuleEntry>> modules;
};

struct RegistrationModuleEntry {
    std::once_flag ready;
    std::exception_ptr failure;
    std::shared_ptr<const SpecializedModule> module;
};

std::shared_ptr<const SpecializedModule> registrationModule(std::uint64_t variantId, std::span<const std::uint32_t> moduleKey, const std::function<std::shared_ptr<const SpecializedModule>()>& specialize) {
    std::vector<std::uint64_t> key{variantId};
    key.insert(key.end(), moduleKey.begin(), moduleKey.end());
    static PrimedSharedMutex mutex;
    static std::map<std::vector<std::uint64_t>, std::shared_ptr<RegistrationModuleEntry>> modules;
    std::shared_ptr<RegistrationModuleEntry> entry;
    {
        std::shared_lock lock(mutex);
        if (const auto found = modules.find(key); found != modules.end()) entry = found->second;
    }
    if (entry == nullptr) {
        std::unique_lock lock(mutex);
        const auto found = modules.find(key);
        entry = found != modules.end() ? found->second : modules.emplace(key, std::make_shared<RegistrationModuleEntry>()).first->second;
    }
    std::call_once(entry->ready, [&] {
        try {
            entry->module = specialize();
        } catch (...) {
            entry->failure = std::current_exception();
        }
    });
    if (entry->failure) std::rethrow_exception(entry->failure);
    return entry->module;
}

std::shared_ptr<PreparedBindingPlan> preparedBindingPlan(const CompiledVariant& variant, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exports, bool fragmentDescriptorSet) {
    struct BindingPlanKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, BindingPlanKeyStorage>();
    key.clear();
    key.push_back(variant.artifact.variantId);
    for (std::size_t index = 0; index < variant.info.info.images.size(); ++index) {
        const auto& descriptor = snapshot.images.at(index);
        key.push_back(descriptor.dwordCount);
        key.push_back(descriptor.dwords[0] != 0u || (descriptor.dwords[1] & 0xffu) != 0u);
        key.push_back(descriptor.dwords[1] & ~0xffu);
        for (std::size_t word = 2; word < 8; ++word) key.push_back(descriptor.dwords[word]);
    }
    for (std::size_t index = 0; index < variant.info.info.samplers.size(); ++index) {
        const auto& descriptor = snapshot.samplers.at(index);
        key.push_back(descriptor.dwordCount);
        for (std::size_t word = 0; word < 4; ++word) key.push_back(descriptor.dwords[word]);
    }
    key.push_back(exports.size());
    for (const auto mapping : exports) key.push_back(mapping);
    static PrimedSharedMutex mutex;
    static std::map<std::vector<std::uint64_t>, std::shared_ptr<PreparedBindingPlan>> plans;
    std::shared_ptr<PreparedBindingPlan> plan;
    {
        std::shared_lock lock(mutex);
        if (const auto found = plans.find(key); found != plans.end()) plan = found->second;
    }
    if (plan == nullptr) {
        std::unique_lock lock(mutex);
        const auto found = plans.find(key);
        plan = found != plans.end() ? found->second : plans.emplace(key, std::make_shared<PreparedBindingPlan>()).first->second;
    }
    std::call_once(plan->ready, [&] {
        try {
            plan->bindings = DescriptorBindingBuilder{}.Prepare(variant.bindings.layout, variant.info.info, variant.info.stage, snapshot, exports, variant.artifact.registrationModule, fragmentDescriptorSet);
        } catch (...) {
            plan->failure = std::current_exception();
        }
    });
    if (plan->failure) std::rethrow_exception(plan->failure);
    return plan;
}

std::shared_ptr<const SpecializedModule> specializeModule(const CompiledShaderArtifact& artifact, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants, const SpirvTarget& target) {
    struct ModuleKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, ModuleKeyStorage>();
    key.clear();
    key.push_back(artifact.variantId);
    for (const auto& constant : constants) key.push_back((static_cast<std::uint64_t>(constant.id) << 32u) | constant.value);
    static PrimedSharedMutex mutex;
    static std::map<std::vector<std::uint64_t>, std::shared_ptr<SpecializedModuleEntry>> modules;
    std::shared_ptr<SpecializedModuleEntry> entry;
    {
        std::shared_lock lock(mutex);
        if (const auto found = modules.find(key); found != modules.end()) entry = found->second;
    }
    if (entry == nullptr) {
        std::unique_lock lock(mutex);
        const auto found = modules.find(key);
        entry = found != modules.end() ? found->second : modules.emplace(key, std::make_shared<SpecializedModuleEntry>()).first->second;
    }
    std::call_once(entry->ready, [&] {
        try {
            entry->module = buildSpecializedModule(artifact, classes, constants, target);
        } catch (...) {
            entry->failure = std::current_exception();
        }
    });
    if (entry->failure) std::rethrow_exception(entry->failure);
    return entry->module;
}

RecompileResult materializeResult(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool registrationDefaults = false) {
    RecompileResult result;
    const auto& artifact = variant.artifact;
    static_cast<CompiledShaderArtifact&>(result) = {
        .memoryOffsetDword = artifact.memoryOffsetDword,
        .bufferAddressDword = artifact.bufferAddressDword,
        .samplerSlotDword = artifact.samplerSlotDword,
        .dispatchThreadLimitDword = artifact.dispatchThreadLimitDword,
        .shaderDataDwords = artifact.shaderDataDwords,
        .imageMetadataDword = artifact.imageMetadataDword,
        .runtimeImageCount = artifact.runtimeImageCount,
        .runtimeImageResources = artifact.runtimeImageResources,
        .registrationModule = artifact.registrationModule,
        .bdaAbiVersion = artifact.bdaAbiVersion,
        .bdaWrites = artifact.bdaWrites,
        .usesShaderBase = artifact.usesShaderBase,
        .imageStoresRecorded = artifact.imageStoresRecorded,
        .runtimeAbiVersion = artifact.runtimeAbiVersion,
        .vertexInputs = artifact.vertexInputs,
        .vertexOffsetSgpr = artifact.vertexOffsetSgpr,
        .instanceOffsetSgpr = artifact.instanceOffsetSgpr,
        .vertexOffsetShared = artifact.vertexOffsetShared,
        .instanceOffsetShared = artifact.instanceOffsetShared,
        .vertexOffsetConflict = artifact.vertexOffsetConflict,
        .instanceOffsetConflict = artifact.instanceOffsetConflict,
        .hostSubgroupSize = artifact.hostSubgroupSize,
        .parameterExports = artifact.parameterExports,
        .fragmentParameters = artifact.fragmentParameters,
        .barycentricEmulation = artifact.barycentricEmulation,
        .stencilReferences = artifact.stencilReferences,
        .stencilReplaySafe = artifact.stencilReplaySafe,
        .stencilReferenceDword = artifact.stencilReferenceDword,
        .variantId = artifact.variantId,
        .identity = artifact.identity
    };
    const auto plan = preparedBindingPlan(variant, snapshot, request.context.pixel ? std::span<const std::uint8_t>(request.context.pixel->targetExportMapping) : std::span<const std::uint8_t>{}, request.target.registrationModules);
    const DescriptorBindingPlan* bindingPlan = &plan->bindings;
    std::shared_ptr<PreparedModuleEntry> entry;
    struct LocalModuleKeyStorage {};
    auto& moduleKey = HostThreadLocal<std::vector<std::uint32_t>, LocalModuleKeyStorage>();
    moduleKey.clear();
    if (variant.bindings.layout.UsesPushData()) moduleKey.push_back(request.layout.pushConstantOffsetBytes / 4u);
    if (!artifact.stencilReferences.empty()) moduleKey.push_back(request.layout.pushConstantOffsetBytes / 4u + artifact.stencilReferenceDword);
    if (request.context.pixel) {
        for (const auto packing : request.context.pixel->targetExportPacking) moduleKey.push_back(static_cast<std::uint32_t>(packing));
        moduleKey.push_back(request.context.pixel->dualSourceBlend ? 1u : 0u);
        moduleKey.push_back(request.context.pixel->reversedBlend);
        if (variant.info.info.rasterPrimitive) moduleKey.push_back(static_cast<std::uint32_t>(request.context.pixel->rasterPrimitive));
        if (variant.info.info.layerExported) moduleKey.push_back(request.context.pixel->layerExported ? 1u : 0u);
    }
    result.vertexAttributes.reserve(result.vertexInputs.size());
    std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources / RuntimeAbi::VertexSelectorsPerDword> runtimeSelectors{};
    for (const auto& input : result.vertexInputs) {
        if (!request.context.vertex || input.location >= request.context.vertex->resourcesNum || input.location >= request.context.vertex->resources.size()) throw std::runtime_error("Shader cache: invalid vertex attribute metadata");
        const auto& vertex = *request.context.vertex;
        const auto& resource = vertex.resources[input.location];
        const auto fetchIndex = vertex.fetchEmbedded ? vertex.resourcesDst[input.location].fetchIndex : input.fetchIndex;
        result.vertexAttributes.push_back({input.location, input.components, resource, fetchIndex});
        if (!artifact.vertexInputPatches.empty()) {
            const auto format = static_cast<IrBufferFormat>((resource.fields[3] >> 12u) & 0x7fu);
            const auto numeric = registrationDefaults && format == IrBufferFormat::Invalid ? IrTextureNumericClass::Float : VertexInputNumericClass(format);
            if (numeric == IrTextureNumericClass::Unsupported) throw std::runtime_error("unsupported prepared vertex format");
            const auto kind = numeric == IrTextureNumericClass::Float ? 0u : numeric == IrTextureNumericClass::Sint ? 1u : 2u;
            std::uint32_t selectors = kind << 12u;
            auto& formatComponents = result.vertexAttributes.back().formatComponents;
            formatComponents = 1u;
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const auto selector = (input.outputMask & (1u << component)) != 0u ? (resource.fields[3] >> (component * 3u)) & 7u : 0u;
                if (selector == 2u || selector == 3u) throw std::runtime_error("reserved prepared vertex component selector");
                if (selector >= 4u) formatComponents = std::max(formatComponents, selector - 3u);
                selectors |= selector << (component * 3u);
            }
            moduleKey.push_back(artifact.registrationModule ? selectors & ~0xfffu : selectors);
            if (artifact.registrationModule) runtimeSelectors.at(input.location / RuntimeAbi::VertexSelectorsPerDword) |= (selectors & 0xfffu) << (input.location % RuntimeAbi::VertexSelectorsPerDword * 16u);
        }
    }
    if (artifact.usesShaderBase) {
        moduleKey.push_back(static_cast<std::uint32_t>(request.shader.codeAddress));
        moduleKey.push_back(static_cast<std::uint32_t>(request.shader.codeAddress >> 32u));
    }
    if (!plan->bindings.specialization.empty() || variant.bindings.layout.UsesPushData() || !artifact.vertexInputPatches.empty() || request.context.pixel || artifact.usesShaderBase) {
        {
            std::shared_lock lock(plan->mutex);
            if (const auto found = plan->modules.find(moduleKey); found != plan->modules.end()) entry = found->second;
        }
        if (entry == nullptr) {
            std::unique_lock lock(plan->mutex);
            const auto found = plan->modules.find(moduleKey);
            entry = found != plan->modules.end() ? found->second : plan->modules.emplace(moduleKey, std::make_shared<PreparedModuleEntry>()).first->second;
        }
        const auto prepare = [&] {
            auto constants = plan->bindings.specialization;
            std::size_t index = 0;
            if (variant.bindings.layout.UsesPushData()) constants.push_back({PipelineSpecialization::PushDataOffset, moduleKey[index++]});
            if (!artifact.stencilReferences.empty()) constants.push_back({PipelineSpecialization::StencilReferenceOffset, moduleKey[index++]});
            if (request.context.pixel) {
                for (std::uint32_t target = 0; target < request.context.pixel->targetExportPacking.size(); ++target) constants.push_back({PipelineSpecialization::ExportPackingBase + target, moduleKey[index++]});
                constants.push_back({PipelineSpecialization::DualSourceBlend, moduleKey[index++]});
                constants.push_back({PipelineSpecialization::ReversedBlend, moduleKey[index++]});
                if (variant.info.info.rasterPrimitive) constants.push_back({PipelineSpecialization::RasterPrimitive, moduleKey[index++]});
                if (variant.info.info.layerExported) constants.push_back({PipelineSpecialization::LayerExported, moduleKey[index++]});
            }
            if (!artifact.vertexInputPatches.empty()) {
                for (const auto& input : result.vertexInputs) {
                    const auto selectors = moduleKey[index++];
                    const auto first = PipelineSpecialization::VertexBase + input.location * PipelineSpecialization::VertexWords;
                    for (std::uint32_t component = 0; component < 4u; ++component) constants.push_back({first + component, (selectors >> (component * 3u)) & 7u});
                    const auto kind = selectors >> 12u;
                    constants.push_back({first + 4u, kind == 0u ? 0x3f800000u : 1u});
                    constants.push_back({first + 5u, kind});
                }
            }
            if (artifact.usesShaderBase) {
                constants.push_back({PipelineSpecialization::ShaderBaseLow, moduleKey[index++]});
                constants.push_back({PipelineSpecialization::ShaderBaseHigh, moduleKey[index++]});
            }
            const auto vertexClasses = vertexClassesOf(constants);
            const auto module = artifact.registrationModule && plan->bindings.specialization.empty() ? registrationModule(artifact.variantId, moduleKey, [&] { return specializeModule(variant.artifact, vertexClasses, constants, request.target); }) : specializeModule(variant.artifact, vertexClasses, constants, request.target);
            if (variant.bindings.layout.UsesPushData() && !module->pushData) throw std::runtime_error("specialization removed the prepared push constant interface");
            auto selected = DescriptorBindingBuilder{}.Select(plan->bindings, module->bindings, module->samplerHeap, module->imageHeap);
            entry->bindings = std::move(selected);
            entry->module = module;
        };
        std::call_once(entry->ready, [&] {
            try {
                prepare();
            } catch (...) {
                entry->failure = std::current_exception();
            }
        });
        if (entry->failure) std::rethrow_exception(entry->failure);
        const auto& module = entry->module;
        result.specializationId = module->specializationId;
        result.moduleConstants = module->constants;
        result.spirv = module->spirv;
        bindingPlan = &entry->bindings;
    } else {
        result.spirv = artifact.spirv;
    }
    const auto tables = ResourceMaterializer::ResolveImageTables(variant.info.info, snapshot, artifact.variantId);
    ResourceMaterializer::RegisterDeviceImageInterfaces(variant.info.info);
    const auto& images = variant.info.info.images;
    if (std::ranges::any_of(images, [](const ImageResource& image) { return image.deviceTable; })) {
        result.deviceImageInterfaces.assign(images.size(), 0u);
        for (std::size_t image = 0; image < images.size(); ++image) {
            if (images[image].deviceTable) result.deviceImageInterfaces[image] = ResourceMaterializer::DeviceImageInterfaceId(images[image], variant.info.info.runtimeImageModes.at(image));
        }
        result.deviceImages = snapshot.deviceImages;
        result.deviceImagesComplete = snapshot.deviceImagesComplete;
        DeviceImageTableAbi::NameInterfaces(result.deviceImages, result.deviceImageInterfaces);
        std::optional<RuntimeImageModeMatch> refusal;
        for (const auto& expectation : result.deviceImages) {
            if (expectation.candidate) continue;
            const bool runtimeSampler = (expectation.key[DeviceImageTableAbi::SamplerClassWord] & DeviceImageTableAbi::RuntimeSampler) != 0u;
            if (images.at(expectation.image).foldTexelOffsetsOnly && runtimeSampler && (expectation.key[DeviceImageTableAbi::SamplerWord] & DeviceImageTableAbi::SamplerUnnormalized) == 0u) {
                throw std::runtime_error("a normalized S# is used with an image sample that has a texel offset that is not a constant, which image sampling takes only with VK_KHR_maintenance8 and shaderImageGatherExtended");
            }
            DescriptorValue descriptor;
            descriptor.dwordCount = DeviceImageTableAbi::DescriptorWords;
            std::copy_n(expectation.key.begin(), DeviceImageTableAbi::DescriptorWords, descriptor.dwords.begin());
            const auto& modes = variant.info.info.runtimeImageModes.at(expectation.image);
            auto match = ResourceMaterializer::TryRuntimeImageMode(images.at(expectation.image), descriptor, modes);
            if (match.mode.has_value()) {
                const auto* samplerWords = runtimeSampler ? expectation.key.data() + DeviceImageTableAbi::SamplerWord : expectation.mapSampler < snapshot.samplers.size() && snapshot.samplers[expectation.mapSampler].dwordCount == 4u ? snapshot.samplers[expectation.mapSampler].dwords.data() : nullptr;
                if (samplerWords != nullptr) {
                    if (const auto* mismatch = ResourceMaterializer::VolumePlaneBiasSamplerMismatch(images.at(expectation.image), modes.at(*match.mode), std::span<const std::uint32_t, 4>(samplerWords, 4u))) throw std::runtime_error(mismatch);
                }
                if ((modes.at(*match.mode).emulatedCompare & EmulatedCompare::Enabled) != 0u) {
                    if (!runtimeSampler) throw std::runtime_error("comparison sampling of a color texture through a sampler the device image table does not read is not implemented");
                    static_cast<void>(ResourceMaterializer::ColorCompareState(images.at(expectation.image), descriptor, std::span<const std::uint32_t, 4>(expectation.key.data() + DeviceImageTableAbi::SamplerWord, DeviceImageTableAbi::SamplerWords)));
                }
                if (!runtimeSampler || (expectation.key[DeviceImageTableAbi::SamplerWord] & DeviceImageTableAbi::SamplerUnnormalized) == 0u) continue;
                if (const auto* mismatch = ResourceMaterializer::UnnormalizedSampleMismatch(modes.at(*match.mode))) throw std::runtime_error(std::string("an S# with unnormalized coordinates ") + mismatch + ", which is not implemented");
                for (const auto& use : variant.info.info.deviceImageUses) {
                    if (use.image != expectation.image || use.firstUsePc != expectation.pc || use.sampler == NoSampler) continue;
                    if (const auto unsupported = UnnormalizedUnsupportedUses(variant.info.info.samplers.at(use.sampler))) throw std::runtime_error(std::string("an S# with unnormalized coordinates ") + UnnormalizedUseReason(unsupported) + ", which is not implemented");
                }
                continue;
            }
            if (!refusal.has_value() || (match.reason == ImageTableAbi::PoisonReason::SrgbUnsupported && refusal->reason != ImageTableAbi::PoisonReason::SrgbUnsupported)) refusal = std::move(match);
        }
        if (refusal.has_value()) throw std::runtime_error(refusal->detail);
    }
    BindingAllocationResult bindings;
    DescriptorBindingBuilder{}.Populate(bindings, variant.bindings, *bindingPlan, variant.info.userDataBase, snapshot, partialThreads(request), &tables);
    result.workgroupMemoryDwords = WorkgroupMemoryStrideDwords(variant.info.info);
    result.bindings = std::move(bindings.bindings);
    result.pushConstants = std::move(bindings.pushConstants);
    if (!artifact.stencilReferences.empty()) {
        if (artifact.stencilReferenceDword * sizeof(std::uint32_t) != result.pushConstants.size() || result.pushConstants.size() + sizeof(std::uint32_t) > request.layout.pushConstantSizeBytes) throw std::runtime_error("stencil replay push constant lies outside the stage range");
        result.pushConstants.resize(result.pushConstants.size() + sizeof(std::uint32_t));
    }
    const auto& layout = variant.bindings.layout;
    result.userDataSlots.clear();
    for (const auto reg : layout.userDataRegisters) result.userDataSlots.push_back(reg - variant.info.userDataBase);
    if (artifact.registrationModule && request.shader.stage != ShaderStage::Fragment && layout.runtimeStateDwords != 0u) {
        const auto first = layout.RuntimeStateDword() + RuntimeAbi::StateVertexSelectors;
        if (layout.runtimeStateDwords > runtimeSelectors.size()) throw std::runtime_error("vertex selector state exceeds its capacity");
        if (layout.UsesPushData()) {
            if ((first + layout.runtimeStateDwords) * sizeof(std::uint32_t) > result.pushConstants.size()) throw std::runtime_error("vertex selector state lies outside the push data");
            std::memcpy(result.pushConstants.data() + first * sizeof(std::uint32_t), runtimeSelectors.data(), layout.runtimeStateDwords * sizeof(std::uint32_t));
        } else {
            const auto data = std::ranges::find_if(result.bindings, [](const DescriptorBinding& binding) { return binding.role == DescriptorRole::ShaderData; });
            if (data == result.bindings.end() || first + layout.runtimeStateDwords > data->guestDescriptor.size()) throw std::runtime_error("vertex selector state lies outside the shader data");
            std::copy_n(runtimeSelectors.begin(), layout.runtimeStateDwords, data->guestDescriptor.begin() + first);
        }
    }
    result.shaderDataUserWords = std::move(bindings.shaderDataUserWords);
    result.poisonedSrtReads = static_cast<std::uint32_t>(snapshot.srtPoison.size()) + snapshot.nullRootReads;
    ApplyBufferRanges(variant.info.info, snapshot.stores, snapshot.loads, false, result);
    ApplyBufferRanges(variant.info.info, snapshot.stores, snapshot.loads, true, result);
    if (!tables.map.empty()) {
        result.imageTablePoison.reserve(tables.poison.size());
        for (const auto& poison : tables.poison) result.imageTablePoison.push_back({poison.words, poison.dwordCount, poison.resource, static_cast<std::uint32_t>(poison.reason)});
        result.imageTableRanges = snapshot.tables.ranges;
        result.imageTableReadRanges.reserve(snapshot.tables.tables.size());
        for (const auto& table : snapshot.tables.tables) result.imageTableReadRanges.push_back(table.readRanges);
        result.imageTableShader = snapshot.tables.shader;
        result.imageTableFaults = tables.faults;
    }
    return result;
}

bool sameLayout(const BindingLayout& left, const BindingLayout& right) {
    return left.descriptorSet == right.descriptorSet && left.firstBinding == right.firstBinding && left.pushConstantOffsetBytes == right.pushConstantOffsetBytes && left.pushConstantSizeBytes == right.pushConstantSizeBytes;
}

}

struct PreparedInterface {
    static CompiledVariant Compile(SourceEntry& source, const RecompileRequest& request, ShaderPreparationContext& preparation, std::exception_ptr* emissionFailure) {
        if (preparation.interfaceSource.get() != &source || preparation.interfaceProgram->Resources().shaderHash != request.shader.codeAddress) {
            preparation.interfaceSource.reset();
            preparation.interfaceProgram.reset();
            auto program = source.program != nullptr ? std::move(source.program) : std::make_unique<IrProgram>(PrepareResourceProgram(request, &preparation));
            recordEmissionFailure(emissionFailure, [&] { prepareInterface(request, *program); });
            preparation.interfaceSource = source.shared_from_this();
            preparation.interfaceProgram = std::move(program);
        }
        auto& program = *preparation.interfaceProgram;
        return recordEmissionFailure(emissionFailure, [&] {
            auto variant = emitVariant(request, program);
            variant.info = program.CompiledInfo();
            return variant;
        });
    }
};

namespace {

std::shared_ptr<const CompiledVariant> findOrCompileVariant(SourceEntry& source, const RecompileRequest& request, bool& cacheHit, ShaderPreparationContext* preparation = nullptr) {
    for (const auto& candidate : source.variants) {
        if (sameLayout(candidate->layout, request.layout)) {
            cacheHit = true;
            return candidate;
        }
    }
    for (const auto& failure : source.emissionFailures) {
        if (failure.codeAddress == request.shader.codeAddress && sameLayout(failure.layout, request.layout)) std::rethrow_exception(failure.failure);
    }
    cacheHit = false;
    const bool disk = ShaderDiskCache::Enabled() && !DebugProbeActive();
    std::vector<std::byte> diskKey;
    ShaderDiskCache::BuildKey(request, HostSubgroupSize(request), diskKey);
    const auto identity = identityOf(diskKey);
    std::shared_ptr<const CompiledVariant> variant;
    if (disk) {
        CompiledVariant loaded;
        if (ShaderDiskCache::Load(diskKey, loaded)) {
            loaded.layout = request.layout;
            loaded.artifact.variantId = nextVariantId();
            loaded.artifact.identity = identity;
            variant = std::make_shared<const CompiledVariant>(std::move(loaded));
        }
    }
    if (variant == nullptr) {
        static const bool layoutPair = std::getenv("APS5_NO_PERF_LAYOUT_PAIR") == nullptr;
        std::exception_ptr emissionFailure;
        try {
            if (layoutPair && preparation != nullptr) {
                auto compiled = PreparedInterface::Compile(source, request, *preparation, &emissionFailure);
                compiled.artifact.identity = identity;
                variant = std::make_shared<const CompiledVariant>(std::move(compiled));
            } else {
                auto program = source.program != nullptr ? std::move(*source.program) : PrepareResourceProgram(request, preparation);
                source.program.reset();
                auto compiled = compileVariant(request, std::move(program), &emissionFailure);
                compiled.artifact.identity = identity;
                variant = std::make_shared<const CompiledVariant>(std::move(compiled));
            }
        } catch (...) {
            if (emissionFailure != nullptr && FailureMemo()) source.emissionFailures.push_back({request.shader.codeAddress, request.layout, emissionFailure});
            throw;
        }
        if (disk) {
            if (auto persisted = ShaderDiskCache::Persist(diskKey, *variant)) variant = std::move(persisted);
            else ShaderDiskCache::Store(std::move(diskKey), variant);
        }
    }
    source.program.reset();
    source.variants.push_back(variant);
    return variant;
}

RecompileResult materializeVariant(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot) {
    std::shared_ptr<const CompiledVariant> variant;
    bool cacheHit = false;
    {
        std::lock_guard lock(source.mutex);
        variant = findOrCompileVariant(source, request, cacheHit);
    }
    auto result = materializeResult(*variant, request, snapshot);
    result.cacheHit = cacheHit;
    return result;
}

void keepMemoryBuffers(const IrResourcePlan& plan, const SrtRuntime& runtime, ResourceSnapshot& snapshot) {
    std::vector<DescriptorValue> stores;
    std::vector<DescriptorValue> loads;
    ResourceMaterializer::EvaluateMemoryBuffers(plan, runtime, stores, loads);
    for (std::size_t index = 0; index < plan.info.bufferStores.size(); ++index) {
        if (plan.info.bufferStores[index].fromMemory) snapshot.stores.at(index) = stores[index];
    }
    for (std::size_t index = 0; index < plan.info.bufferLoads.size(); ++index) {
        if (plan.info.bufferLoads[index].fromMemory) snapshot.loads.at(index) = loads[index];
    }
}

RecompileResult RecompileImpl(const RecompileRequest& request) {
    static_cast<void>(RequestInputInfo(request));
    RequestMemoryView memory(request.context.memory);
    auto runtime = memory.MakeRuntime(request.context.userData, request.shader.codeAddress);
    runtime.linkedWords = request.context.linkedWords;
    ResourceSnapshot snapshot;
    constexpr ResourceMaterializer materializer;
    if (!request.useCache) {
        auto program = PrepareResourceProgram(request);
        const auto plan = materializer.ExtractPlan(program);
        materializer.Materialize(plan, runtime, snapshot);
        snapshot.deviceImages = ResourceMaterializer::ExpectedDeviceImages(plan, runtime, &snapshot.deviceImagesComplete);
        keepMemoryBuffers(plan, runtime, snapshot);
        const auto variant = compileVariant(request, std::move(program));
        return materializeResult(variant, request, snapshot);
    }
    const auto source = getSource(request);
    materializer.Materialize(*source->plan, runtime, snapshot);
    snapshot.deviceImages = ResourceMaterializer::ExpectedDeviceImages(*source->plan, runtime, &snapshot.deviceImagesComplete);
    keepMemoryBuffers(*source->plan, runtime, snapshot);
    return materializeVariant(*source, request, snapshot);
}

// APS5_NO_RESULT_MEMO=1: every Recompile(request, capture) materializes its own result as before.
bool ResultMemo() {
    static const bool resultMemo = std::getenv("APS5_NO_RESULT_MEMO") == nullptr;
    return resultMemo;
}

constexpr std::size_t ResultMemoEntries = 256;

struct ResultMemoCounters {
    std::atomic<std::uint64_t> hits{0}, misses{0}, evictions{0}, populateNanoseconds{0};
    std::atomic<std::int64_t> lastReport{0};
};

ResultMemoCounters& resultMemoCounters() {
    static ResultMemoCounters counters;
    return counters;
}

void reportResultMemo() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& counters = resultMemoCounters();
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000ll || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto hits = counters.hits.exchange(0, std::memory_order_relaxed);
    const auto misses = counters.misses.exchange(0, std::memory_order_relaxed);
    const auto evictions = counters.evictions.exchange(0, std::memory_order_relaxed);
    const auto populate = counters.populateNanoseconds.exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr, "[recompile] result memo (10 s): %llu hits, %llu misses (%.1f%% hits), Populate %.1f us per miss / %.1f ms in total, %llu evictions\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses), hits + misses != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses) : 0.0, misses != 0 ? static_cast<double>(populate) / 1000.0 / static_cast<double>(misses) : 0.0, static_cast<double>(populate) / 1e6, static_cast<unsigned long long>(evictions));
}

// Everything materializeResult reads besides the variant: the snapshot (the descriptor words, the
// flattened SRT, the user data, the uniform fill) and, for the vertex family, the V# table the
// attributes are resolved from.
std::uint64_t snapshotHash(const RecompileRequest& request, const ResourceSnapshot& snapshot) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ull;
    };
    const auto mixWords = [&](std::span<const std::uint32_t> words) {
        mix(words.size());
        for (const auto word : words) mix(word);
    };
    const auto mixDescriptors = [&](const std::vector<DescriptorValue>& values) {
        mix(values.size());
        for (const auto& value : values) {
            mix(value.dwordCount);
            mix(value.availableDwordMask);
            for (std::uint32_t i = 0; i < value.dwordCount && i < value.dwords.size(); ++i) mix(value.dwords[i]);
        }
    };
    mixDescriptors(snapshot.images);
    mixDescriptors(snapshot.samplers);
    mixDescriptors(snapshot.stores);
    mixDescriptors(snapshot.loads);
    mix(snapshot.deviceImages.size());
    mix(snapshot.deviceImagesComplete);
    for (const auto& image : snapshot.deviceImages) {
        mix(image.image);
        mix(image.pc);
        mix(image.mapSampler);
        mix(image.candidate);
        for (const auto word : image.key) mix(word);
    }
    mixWords(snapshot.flattenedSrt);
    mixWords(snapshot.userData);
    mix(static_cast<std::uint64_t>(snapshot.uniformFill.kind));
    mix(snapshot.uniformFill.resource);
    for (const auto stride : snapshot.uniformFill.groupStride) mix(stride);
    mix(snapshot.uniformFill.words);
    mix(snapshot.uniformFill.value);
    const auto& tables = snapshot.tables;
    mix(tables.shader);
    mix(tables.tables.size());
    for (const auto& table : tables.tables) {
        mix(table.base);
        mix(table.size);
        mix(table.records);
        mix(table.keys);
        mix(table.fault);
        mix(table.outside ? 1u : 0u);
        mixWords(table.codes);
        mix(table.readRanges.size());
        for (const auto& [base, size] : table.readRanges) {
            mix(base);
            mix(size);
        }
    }
    mixDescriptors(tables.words);
    mix(tables.ranges.size());
    for (const auto& [base, size] : tables.ranges) {
        mix(base);
        mix(size);
    }
    for (const auto threads : partialThreads(request)) mix(threads);
    mix(request.context.compute.has_value() && request.context.compute->threadsFromMemory);
    mix(request.layout.pushConstantOffsetBytes);
    if (request.context.pixel) {
        for (const auto mapping : request.context.pixel->targetExportMapping) mix(mapping);
        for (const auto packing : request.context.pixel->targetExportPacking) mix(static_cast<std::uint64_t>(packing));
        mix(request.context.pixel->dualSourceBlend);
        mix(request.context.pixel->reversedBlend);
        mix(static_cast<std::uint64_t>(request.context.pixel->rasterPrimitive));
        mix(request.context.pixel->layerExported);
    }
    if (request.context.vertex) {
        const auto& vertex = *request.context.vertex;
        const auto count = std::min<std::uint32_t>(vertex.resourcesNum, ShaderVertexStageInfo::MaxResources);
        mix(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            for (const auto field : vertex.resources[i].fields) mix(field);
            mix(vertex.resourcesDst[i].fetchIndex);
        }
    } else {
        mix(1ull << 32u);
    }
    return hash;
}

// The memo'd result of `source`'s variant for the snapshot (design13 R5): a hit returns the shared
// object, a miss materializes outside the source mutex and inserts (a concurrent miss's object is
// as good). `memoHit` reports the hit.
std::shared_ptr<const RecompileResult> materializePreparedMemoized(SourceEntry& source, const std::shared_ptr<const CompiledVariant>& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool cacheHit, bool* memoHit) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto hash = snapshotHash(request, snapshot);
    std::uint64_t index = 0;
    auto& counters = resultMemoCounters();
    {
        std::lock_guard lock(source.mutex);
        index = (variant->artifact.variantId * 0x9e3779b97f4a7c15ull) ^ hash;
        const auto found = source.memoIndex.find(index);
        if (found != source.memoIndex.end() && found->second->variantId == variant->artifact.variantId && found->second->hash == hash) {
            source.memo.splice(source.memo.begin(), source.memo, found->second);
            counters.hits.fetch_add(1, std::memory_order_relaxed);
            if (memoHit != nullptr) *memoHit = true;
            reportResultMemo();
            return found->second->result;
        }
    }
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto result = std::make_shared<RecompileResult>(materializeResult(*variant, request, snapshot));
    result->cacheHit = cacheHit;
    if (profile) counters.populateNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    counters.misses.fetch_add(1, std::memory_order_relaxed);
    std::shared_ptr<const RecompileResult> shared = std::move(result);
    {
        std::lock_guard lock(source.mutex);
        const auto found = source.memoIndex.find(index);
        if (found != source.memoIndex.end()) {
            if (found->second->variantId == variant->artifact.variantId && found->second->hash == hash) {
                source.memo.splice(source.memo.begin(), source.memo, found->second);
                shared = found->second->result;
            } else {
                source.memo.erase(found->second);
                source.memoIndex.erase(found);
            }
        }
        if (source.memoIndex.find(index) == source.memoIndex.end()) {
            source.memo.push_front({variant->artifact.variantId, hash, shared});
            source.memoIndex.emplace(index, source.memo.begin());
            while (source.memo.size() > ResultMemoEntries) {
                const auto& last = source.memo.back();
                source.memoIndex.erase((last.variantId * 0x9e3779b97f4a7c15ull) ^ last.hash);
                source.memo.pop_back();
                counters.evictions.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    reportResultMemo();
    return shared;
}

std::shared_ptr<const RecompileResult> materializeMemoized(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool* memoHit) {
    std::shared_ptr<const CompiledVariant> variant;
    bool cacheHit = false;
    {
        std::lock_guard lock(source.mutex);
        variant = findOrCompileVariant(source, request, cacheHit);
    }
    return materializePreparedMemoized(source, variant, request, snapshot, cacheHit, memoHit);
}

// The capture already resolved the source entry (stage input validation included) and materialized
// the request over exactly the words the driver captured, so neither is repeated here.
std::shared_ptr<const RecompileResult> RecompileImpl(const RecompileRequest& request, const ResourceCapture& capture, bool* memoHit) {
    if (!request.useCache || capture.source == nullptr) {
        auto program = PrepareResourceProgram(request);
        const auto variant = compileVariant(request, std::move(program));
        return std::make_shared<const RecompileResult>(materializeResult(variant, request, capture.snapshot));
    }
    if (!ResultMemo()) return std::make_shared<const RecompileResult>(materializeVariant(*capture.source, request, capture.snapshot));
    return materializeMemoized(*capture.source, request, capture.snapshot, memoHit);
}

template <typename Impl>
auto recompileReporting(const RecompileRequest& request, Impl&& impl) -> decltype(impl()) {
    try {
        return impl();
    } catch (const std::exception& e) {
        constexpr auto requestSerializer = RequestSerializer{};
        const auto inputInfo = "\nRecompileRequest:\n" + requestSerializer.Serialize(request);
        throw std::runtime_error(std::string("ShaderRecompiler::Recompile: ") + e.what() + inputInfo);
    } catch (...) {
        throw std::runtime_error("ShaderRecompiler::Recompile: unknown exception");
    }
}

}

std::shared_ptr<const IrResourcePlan> GetResourcePlan(const RecompileRequest& request) {
    static_cast<void>(RequestInputInfo(request));
    if (request.useCache) return getSource(request)->plan;
    return makeResourcePlan(request);
}

namespace {

// The capture's materialization; with pure flat slots in the plan the walk's read addresses are
// traced into the capture (ResourceCapture::readTrace).
void materializeCapture(ResourceCapture& capture, const SrtRuntime& runtime) {
    const auto& plan = *capture.plan;
    SrtRuntime now = runtime;
    if (runtime.readNow != nullptr && plan.srtReads.empty()) now.readMemory = runtime.readNow;
    now.readTrace = nullptr;
    SrtRuntime settled = now;
    if (runtime.readAfterWrites != nullptr && plan.srtReads.empty()) settled.readMemory = runtime.readAfterWrites;
    bool imagesComplete = true;
    auto images = ResourceMaterializer::ExpectedDeviceImages(plan, settled, &imagesComplete);
    std::vector<DescriptorValue> stores;
    std::vector<DescriptorValue> loads;
    ResourceMaterializer::EvaluateMemoryBuffers(plan, now, stores, loads);
    const auto keep = [&] {
        capture.snapshot.deviceImages = std::move(images);
        capture.snapshot.deviceImagesComplete = imagesComplete;
        for (std::size_t index = 0; index < plan.info.bufferStores.size(); ++index) {
            if (plan.info.bufferStores[index].fromMemory) capture.snapshot.stores.at(index) = stores[index];
        }
        for (std::size_t index = 0; index < plan.info.bufferLoads.size(); ++index) {
            if (plan.info.bufferLoads[index].fromMemory) capture.snapshot.loads.at(index) = loads[index];
        }
    };
    if (std::none_of(plan.pureFlatSlots.begin(), plan.pureFlatSlots.end(), [](std::uint8_t pure) { return pure != 0u; })) {
        ResourceMaterializer{}.Materialize(plan, runtime, capture.snapshot);
        keep();
        return;
    }
    SrtRuntime traced = runtime;
    traced.readTrace = &capture.readTrace;
    ResourceMaterializer{}.Materialize(plan, traced, capture.snapshot);
    keep();
    auto& other = capture.readTrace.otherReads;
    std::sort(other.begin(), other.end());
    other.erase(std::unique(other.begin(), other.end()), other.end());
}

}

std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Validates the stage inputs once per request, as GetResourcePlan and Recompile(request) do.
    static_cast<void>(RequestInputInfo(request));
    auto capture = std::make_shared<ResourceCapture>();
    if (request.useCache) {
        capture->source = getSource(request);
        capture->plan = capture->source->plan;
    } else {
        capture->plan = makeResourcePlan(request);
    }
    if (profile) capture->sourceNanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
    materializeCapture(*capture, runtime);
    return capture;
}

std::shared_ptr<const SourceHandle> ResolveSource(const RecompileRequest& request) {
    if (!request.useCache) return nullptr;
    static_cast<void>(RequestInputInfo(request));
    return std::make_shared<const SourceHandle>(SourceHandle{getSource(request)});
}

std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request) {
    return PrepareShader(request, nullptr);
}

std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request, ShaderPreparationContext* preparation) {
    return recompileReporting(request, [&]() -> std::shared_ptr<const SourceHandle> {
        static_cast<void>(RequestInputInfo(request));
        auto handle = std::make_shared<SourceHandle>();
        handle->source = getSource(request, preparation, false);
        RecompileCacheKey::BuildInterface(request, handle->staticKey);
        handle->staticKey.push_back(HostSubgroupSize(request));
        bool cacheHit = false;
        std::lock_guard lock(handle->source->mutex);
        handle->artifact = findOrCompileVariant(*handle->source, request, cacheHit, preparation);
        return handle;
    });
}

bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle) {
    struct PreparedKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, PreparedKeyStorage>();
    BuildPreparedShaderKey(request, key);
    return MatchesPreparedShader(request, handle, key);
}

void BuildPreparedShaderKey(const RecompileRequest& request, std::vector<std::uint64_t>& key) {
    RecompileCacheKey::BuildInterface(request, key);
    key.push_back(HostSubgroupSize(request));
}

bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle, std::span<const std::uint64_t> key) {
    if (handle.source == nullptr || handle.artifact == nullptr) return false;
    const auto& layout = request.layout;
    const auto& prepared = handle.artifact->bindings.layout;
    if (layout.descriptorSet != handle.artifact->layout.descriptorSet || layout.firstBinding != handle.artifact->layout.firstBinding) return false;
    if (layout.pushConstantOffsetBytes % 4u != 0u || layout.pushConstantSizeBytes % 4u != 0u || layout.pushConstantOffsetBytes > NativePushConstantSize || layout.pushConstantSizeBytes > NativePushConstantSize - layout.pushConstantOffsetBytes) return false;
    const auto words = prepared.ShaderDataDwords();
    const bool usesPush = prepared.runtimeImageCount == 0u && words != 0u && words <= layout.pushConstantSizeBytes / 4u;
    if (prepared.UsesPushData() != usesPush) return false;
    if (!handle.artifact->artifact.stencilReferences.empty() && (prepared.stencilReferenceDword >= layout.pushConstantSizeBytes / 4u || prepared.stencilReferenceDword != (usesPush ? words : 0u))) return false;
    if (!std::ranges::equal(key, handle.staticKey) || handle.source->code.size() != request.shader.code.size()) return false;
    return handle.source->code.data() == request.shader.code.data() || std::ranges::equal(handle.source->code, request.shader.code);
}

std::span<const std::uint32_t> GetPreparedCode(const SourceHandle& handle) {
    if (handle.source == nullptr || handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.source->code;
}

std::span<const std::uint32_t> GetPreparedLinkedLiterals(const SourceHandle& handle) {
    if (handle.source == nullptr || handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.source->linkedLiterals;
}

PreparedShaderInvocation::PreparedShaderInvocation(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle)
    : request(request), handle(handle), linkedWords(std::make_shared<const std::vector<std::uint32_t>>(request.context.linkedWords.begin(), request.context.linkedWords.end())) {
    this->request.context.linkedWords = *linkedWords;
}

std::optional<PreparedShaderInvocation> PreparedShaderInvocation::TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle) {
    struct PreparedKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, PreparedKeyStorage>();
    BuildPreparedShaderKey(request, key);
    return TryCreate(request, handle, key);
}

std::optional<PreparedShaderInvocation> PreparedShaderInvocation::TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle, std::span<const std::uint64_t> key) {
    if (handle == nullptr || !MatchesPreparedShader(request, *handle, key)) return std::nullopt;
    if (request.shader.stage != ShaderStage::Compute && request.shader.stage != ShaderStage::Fragment) static_cast<void>(RequestInputInfo(request));
    return PreparedShaderInvocation(request, handle);
}

std::shared_ptr<const ResourceCapture> PreparedShaderInvocation::Capture(const SrtRuntime& runtime) const {
    auto capture = std::make_shared<ResourceCapture>();
    capture->source = handle->source;
    capture->plan = handle->source->plan;
    materializeCapture(*capture, runtime);
    return capture;
}

std::shared_ptr<const RecompileResult> PreparedShaderInvocation::Materialize(const ResourceCapture& capture) const {
    if (capture.source != handle->source || capture.plan != handle->source->plan) throw std::runtime_error("ShaderRecompiler: resource capture belongs to another prepared shader");
    if (request.useCache && ResultMemo()) return materializePreparedMemoized(*handle->source, handle->artifact, request, capture.snapshot, true, nullptr);
    auto result = std::make_shared<RecompileResult>(materializeResult(*handle->artifact, request, capture.snapshot));
    result->cacheHit = true;
    return result;
}

std::shared_ptr<const IrResourcePlan> PreparedShaderInvocation::Plan() const {
    return handle->source->plan;
}

std::shared_ptr<const RecompileResult> PreparedShaderInvocation::MaterializeAtRunTime() const {
    const auto& plan = *handle->source->plan;
    if (!ResourceMaterializer::ReadsResourcesAtRunTime(plan)) return nullptr;
    SrtRuntime runtime;
    runtime.userData = request.context.userData;
    runtime.linkedWords = request.context.linkedWords;
    runtime.shaderBase = request.shader.codeAddress;
    runtime.readMemory = +[](void*, std::uint64_t, std::uint32_t*) { return false; };
    ResourceSnapshot snapshot;
    ResourceMaterializer::RunTimeSnapshot(plan, runtime, snapshot);
    auto result = std::make_shared<RecompileResult>(materializeResult(*handle->artifact, request, snapshot));
    result->cacheHit = true;
    return result;
}

std::optional<RunTimeProgram> CompileAtRunTime(const RecompileRequest& request) {
    const auto handle = PrepareShader(request);
    const auto invocation = PreparedShaderInvocation::TryCreate(request, handle);
    if (!invocation.has_value()) throw std::runtime_error("ShaderRecompiler: a prepared artifact does not match its request");
    auto compiled = invocation->MaterializeAtRunTime();
    if (compiled == nullptr) return std::nullopt;
    return RunTimeProgram{invocation->Plan(), std::move(compiled)};
}

std::shared_ptr<RecompileResult> RunTimeResult(const RunTimeProgram& program, const SrtRuntime& runtime, const SrtRuntime& settled) {
    const auto& plan = *program.plan;
    auto result = std::make_shared<RecompileResult>(*program.compiled);
    auto shaderData = std::ranges::find_if(result->bindings, [](const DescriptorBinding& binding) { return binding.role == DescriptorRole::ShaderData; });
    for (std::size_t slot = 0; slot < result->userDataSlots.size(); ++slot) {
        const auto index = result->userDataSlots[slot];
        if (index >= runtime.userData.size()) throw std::runtime_error("ShaderRecompiler: a program reads a user data register its dispatch or draw does not set");
        if (!result->pushConstants.empty()) {
            if ((slot + 1u) * sizeof(std::uint32_t) > result->pushConstants.size()) throw std::runtime_error("ShaderRecompiler: a program's user data lies outside its push data");
            std::memcpy(result->pushConstants.data() + slot * sizeof(std::uint32_t), &runtime.userData[index], sizeof(std::uint32_t));
        } else if (shaderData != result->bindings.end()) {
            if (slot >= shaderData->guestDescriptor.size()) throw std::runtime_error("ShaderRecompiler: a program's user data lies outside its shader data");
            shaderData->guestDescriptor[slot] = runtime.userData[index];
        }
    }
    std::vector<DescriptorValue> stores;
    std::vector<DescriptorValue> loads;
    ResourceMaterializer::EvaluateBufferSites(plan, runtime, stores, loads);
    ApplyBufferRanges(plan.info, stores, loads, false, *result);
    ApplyBufferRanges(plan.info, stores, loads, true, *result);
    bool complete = true;
    result->deviceImages = ResourceMaterializer::ExpectedDeviceImages(plan, settled, &complete);
    DeviceImageTableAbi::NameInterfaces(result->deviceImages, result->deviceImageInterfaces);
    result->deviceImagesComplete = complete;
    return result;
}

const CompiledShaderArtifact& GetPreparedArtifact(const SourceHandle& handle) {
    if (handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.artifact->artifact;
}

std::shared_ptr<const RecompileResult> RegistrationResult(const RecompileRequest& request, const SourceHandle& handle) {
    if (handle.artifact == nullptr || !MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    if (!request.target.registrationModules || !handle.artifact->artifact.registrationModule) return nullptr;
    const auto& info = handle.artifact->info.info;
    if (std::ranges::any_of(info.samplers, [](const SamplerResource& sampler) { return FoldsTexelOffsets(sampler); })) return nullptr;
    ResourceSnapshot snapshot;
    snapshot.images.assign(info.images.size(), {{}, 8u});
    snapshot.samplers.assign(info.samplers.size(), {{}, 4u});
    snapshot.userData.assign(RuntimeAbi::UserDataCapacity, 0u);
    snapshot.flattenedSrt.assign(1u, 0u);
    return std::make_shared<RecompileResult>(materializeResult(*handle.artifact, request, snapshot, true));
}

std::shared_ptr<RecompileResult> SpecializedComputeResult(const RecompileRequest& request, const SourceHandle& handle, std::span<const PipelineSpecializationConstant> constants) {
    if (handle.artifact == nullptr || !MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    if (request.shader.stage != ShaderStage::Compute) throw std::runtime_error("ShaderRecompiler: a recorded module specialization is replayed for compute programs only");
    auto result = SpecializedStageResult(handle, constants, request.target);
    if (handle.artifact->bindings.layout.UsesPushData()) result->pushConstants.resize(request.layout.pushConstantSizeBytes);
    return result;
}

std::shared_ptr<RecompileResult> SpecializedStageResult(const SourceHandle& handle, std::span<const PipelineSpecializationConstant> constants, const SpirvTarget& target) {
    if (handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    const auto& variant = *handle.artifact;
    auto result = std::make_shared<RecompileResult>();
    static_cast<CompiledShaderArtifact&>(*result) = variant.artifact;
    result->workgroupMemoryDwords = WorkgroupMemoryStrideDwords(variant.info.info);
    if (!constants.empty()) {
        const auto module = specializeModule(variant.artifact, vertexClassesOf(constants), constants, target);
        result->spirv = module->spirv;
        result->specializationId = module->specializationId;
        result->moduleConstants = module->constants;
    }
    return result;
}

std::array<std::uint64_t, 2> ArtifactIdentity(const RecompileRequest& request) {
    struct IdentityKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::byte>, IdentityKeyStorage>();
    ShaderDiskCache::BuildKey(request, HostSubgroupSize(request), key);
    return identityOf(key);
}

std::shared_ptr<const RecompileResult> MaterializeShader(const RecompileRequest& request, const ResourceCapture& capture, const SourceHandle& handle) {
    if (!MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    if (capture.source != handle.source || capture.plan != handle.source->plan) throw std::runtime_error("ShaderRecompiler: resource capture belongs to another prepared shader");
    if (request.useCache && ResultMemo()) return materializePreparedMemoized(*handle.source, handle.artifact, request, capture.snapshot, true, nullptr);
    auto result = std::make_shared<RecompileResult>(materializeResult(*handle.artifact, request, capture.snapshot));
    result->cacheHit = true;
    return result;
}

std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime, const SourceHandle& handle) {
    if (handle.source == nullptr) throw std::runtime_error("ShaderRecompiler: source handle is missing");
    if (handle.artifact != nullptr && !MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // The whole vertex family (Vertex, Local, TC, TE, Mesh) validates V# fields the memo key does not cover.
    if (request.shader.stage != ShaderStage::Compute && request.shader.stage != ShaderStage::Fragment) static_cast<void>(RequestInputInfo(request));
    auto capture = std::make_shared<ResourceCapture>();
    capture->source = handle.source;
    capture->plan = handle.source->plan;
    if (profile) capture->sourceNanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
    materializeCapture(*capture, runtime);
    return capture;
}

RecompileResult Recompile(const RecompileRequest& request) {
    return recompileReporting(request, [&] { return RecompileImpl(request); });
}

std::shared_ptr<const RecompileResult> Recompile(const RecompileRequest& request, const ResourceCapture& capture, bool* memoHit) {
    if (memoHit != nullptr) *memoHit = false;
    return recompileReporting(request, [&] { return RecompileImpl(request, capture, memoHit); });
}

}
