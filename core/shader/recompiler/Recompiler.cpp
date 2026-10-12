#include "Recompiler.hpp"
#include <algorithm>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "CacheKey.hpp"
#include "CompiledVariant.hpp"
#include "Optimization/include/Optimization/ResourceMaterializer.hpp"
#include "VertexInputSpecialization.hpp"
#include "FragmentOutputSpecialization.hpp"
#include "SpirvBackend/SpirvSpecialization.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "ShaderDiskCache.hpp"
#include "ResultKey.hpp"
#include <list>
#include <functional>
#include <map>
#include <set>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
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
#include "Optimization/include/Optimization/ReadLaneEliminator.hpp"
#include "Optimization/include/Optimization/RequestMemoryView.hpp"
#include "Optimization/include/Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/include/Optimization/ResourceTracker.hpp"
#include "Optimization/include/Optimization/ShaderInfoCollector.hpp"
#include "Optimization/include/Optimization/SrtWalker.hpp"
#include "Optimization/include/Optimization/SsaBuilder.hpp"
#include "Optimization/include/Optimization/TessellationLowering.hpp"
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

bool UsesTables(const ShaderInfo& info) {
    return std::ranges::any_of(info.images, [](const ImageResource& image) { return image.table != NoTable; }) || std::ranges::any_of(info.samplers, [](const SamplerResource& sampler) { return sampler.table != NoTable; });
}

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

bool MergedHullStage(const RecompileRequest& request) {
    const auto stage = request.shader.stage;
    return (stage == ShaderStage::Local || stage == ShaderStage::TessellationControl) && request.graphics && request.graphics->tessellation && request.graphics->tessellation->mergedHull;
}

RdnaProgram DecodeStageProgram(ShaderStageKind stage, std::span<const std::uint32_t> code, bool mergedHull) {
    static constexpr std::array<std::uint32_t, 1> EndProgram{0xbf810000u};
    if (stage == ShaderStageKind::Local && mergedHull) {
        RdnaProgram program;
        program.code = EndProgram;
        program.instructions.push_back(DecodeRdnaInstruction(0u, EndProgram, 0u));
        return program;
    }
    if (stage != ShaderStageKind::Local) return RdnaInstructionDecoder{}.Decode(code);
    auto program = DecodeRdnaFrontProgram(code);
    program.instructions.back() = DecodeRdnaInstruction(program.instructions.back().programCounter, EndProgram, 0u);
    return program;
}

ShaderStageInputInfo RequestInputInfo(const RecompileRequest& request) {
    const auto* mesh = request.graphics && request.graphics->mesh ? &*request.graphics->mesh : nullptr;
    const auto* tessellation = request.graphics && request.graphics->tessellation ? &*request.graphics->tessellation : nullptr;
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
        : stage(request.shader.stage), mergedHull(MergedHullStage(request)), swappc(swappcInfo), code(request.shader.code.begin(), request.shader.code.end()),
          decoded(DecodeStageProgram(toShaderStageKind(request.shader.stage), code, mergedHull)), cfg(GraphBuilder{}.Build(decoded, &swappc)) {
        Structurizer{}.Structurize(cfg);
    }

    PreparedControlFlow(const PreparedControlFlow&) = delete;
    PreparedControlFlow& operator=(const PreparedControlFlow&) = delete;

    bool Matches(const RecompileRequest& request, const SwappcInfo& swappcInfo) const {
        return stage == request.shader.stage && mergedHull == MergedHullStage(request) && swappc.fetchCallAllowed == swappcInfo.fetchCallAllowed &&
               swappc.userDataBaseRegister == swappcInfo.userDataBaseRegister && swappc.userDataCount == swappcInfo.userDataCount &&
               std::ranges::equal(code, request.shader.code);
    }

    ShaderStage stage;
    bool mergedHull;
    SwappcInfo swappc;
    std::vector<std::uint32_t> code;
    RdnaProgram decoded;
    ControlFlowGraph cfg;
};

std::shared_ptr<const PreparedControlFlow> ShaderPreparationContext::AcquireFrontend(const RecompileRequest& request) {
    const auto inputInfo = RequestInputInfo(request);
    const SwappcInfo swappc{inputInfo.vertex != nullptr, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size())};
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

IrProgram PrepareResourceProgram(const RecompileRequest& request) {
    return PrepareResourceProgram(request, nullptr);
}

IrProgram PrepareResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation) {
    frontEndBuilds.fetch_add(1, std::memory_order_relaxed);
    ShaderPreparationContext local;
    const auto frontend = (preparation != nullptr ? *preparation : local).AcquireFrontend(request);
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

    if (frontend->mergedHull && cfg.hasFetchCall) throw std::runtime_error("a merged hull program that calls a fetch subroutine is unsupported");
    EmbeddedFetchPlan embeddedFetch;
    if ((stageKind == ShaderStageKind::Vertex || stageKind == ShaderStageKind::Local) && !frontend->mergedHull && inputInfo.vertex != nullptr && inputInfo.vertex->fetchEmbedded) {
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
            constantFolder.Fold(program);
            ResolveControlFlowIdentities(program);
            deadCodeEliminator.RemoveIdentities(program);
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

    if (inputInfo.vertex != nullptr) {
        constexpr TessellationLowering tessellationLowering;
        tessellationLowering.Lower(program, inputInfo.vertex->tess, request.target.tessellation ? &*request.target.tessellation : nullptr);
    }

    constexpr SrtWalker srtWalker;
    srtWalker.BuildPlan(program);
    deadCodeEliminator.Eliminate(program);
    dumpIr("srt");

    constexpr ResourceTracker resourceTracker;
    resourceTracker.Track(program);
    deadCodeEliminator.Eliminate(program);
    dumpIr("resources");
    program.Resources().srgbDecodeFormats = request.target.srgbDecodeFormats;
    program.Resources().imageTableStorage = request.target.imageTableStorage;

    return program;
}

// A materialized result of one variant over one snapshot (Recompile(request, capture)): the
// shared immutable object every later capture that reproduces the snapshot receives, so Populate
// and the per-request copy run once per distinct snapshot.
struct ResultMemoEntry {
    std::uint64_t variantId;
    std::uint64_t hash;
    std::shared_ptr<const RecompileResult> result;
    // resultBytes(*result): what the entry keeps alive, for the memo's byte budget.
    std::size_t bytes = 0;
};

struct EmissionFailure {
    std::uint64_t codeAddress;
    BindingLayout layout;
    std::exception_ptr failure;
};

// The result memo key split by input, for APS5_RESULT_MEMO_DIAG (see resultKeyParts).
enum ResultKeyPart : std::size_t { KeyBuffers, KeyImages, KeySamplers, KeyFlatSrt, KeyUserData, KeyUniformFill, KeyTableColumns, KeyTableCodes, KeyTableReadRanges, KeyTableWords, KeyTableRanges, KeyPartialThreads, KeyPushOffset, KeyPixel, KeyVertex, KeyPartCount };
constexpr std::array<const char*, KeyPartCount> KeyPartNames{"buffers", "images", "samplers", "flat SRT", "user data", "uniform fill", "table columns", "table codes", "table read ranges", "table words", "table ranges", "partial threads", "push offset", "pixel exports", "vertex V#s"};

struct ResultKeyParts {
    std::array<std::uint64_t, KeyPartCount> hashes{};
    std::vector<std::uint32_t> userData;
    std::vector<std::uint32_t> flatSrt;
    // Every buffer descriptor as 8 dwords (unused ones zero).
    std::vector<std::uint32_t> bufferWords;
};

struct SourceEntry {
    std::mutex mutex;
    // The code the entry was built for: the key carries only a hash of it, so a candidate entry is
    // accepted only when its code matches word for word. Owned here because the request's span
    // points into a registration the driver may replace while the entry lives on.
    std::vector<std::uint32_t> code;
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
    // APS5_RESULT_MEMO_DIAG=1: each memo entry's key split by input (by memo index), and the
    // indices evicted since (a miss on one of them is a capacity miss, not a new snapshot).
    std::unordered_map<std::uint64_t, std::shared_ptr<const ResultKeyParts>> diagParts;
    std::unordered_set<std::uint64_t> diagEvicted;
    // The table memo (under mutex), most recently used first: a variant's resolved tables by the
    // inputs ResolveImageTables reads (the tables and the samplers), so a result memo miss whose
    // tables are a recent snapshot's skips the per-key resolution (APS5_NO_TABLE_RESULT_MEMO=1).
    struct TableMemoEntry {
        std::uint64_t variantId;
        std::uint64_t key;
        std::shared_ptr<const ResolvedImageTables> tables;
    };
    std::list<TableMemoEntry> tableMemo;
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

std::uint64_t nextVariantId() {
    static std::atomic<std::uint64_t> variants{0};
    return variants.fetch_add(1, std::memory_order_relaxed) + 1;
}

CompiledVariant compileVariant(const RecompileRequest& request, IrProgram program, std::exception_ptr* emissionFailure = nullptr) try {
    const auto inputInfo = RequestInputInfo(request);
    constexpr DeadCodeEliminator deadCodeEliminator;
    constexpr ResourceMaterializer resourceMaterializer;
    const bool nativeSampleOffsets = request.target.nonConstantImageOffsets && std::find(request.target.supportedCapabilities.begin(), request.target.supportedCapabilities.end(), spv::CapabilityImageGatherExtended) != request.target.supportedCapabilities.end();
    resourceMaterializer.ApplyStaticInterface(program, nativeSampleOffsets);

    deadCodeEliminator.RemoveIdentities(program);
    deadCodeEliminator.Eliminate(program);

    constexpr ShaderInfoCollector shaderInfoCollector;
    shaderInfoCollector.Collect(program, inputInfo);

    constexpr BindingAllocator bindingAllocator;
    const bool tables = UsesTables(program.Info());
    bool registration = request.target.registrationModules && !tables && (request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Fragment);
    auto bindings = bindingAllocator.Allocate(program, request.layout, registration);

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
    targetOptions.fragmentDescriptorSet = request.target.registrationModules && !tables;
    targetOptions.float16DenormPreserve = request.target.float16DenormPreserve;

    constexpr SpirvEmitter spirvEmitter;
    CompiledShaderArtifact result;
    result.variantId = nextVariantId();
    result.spirv = spirvEmitter.Emit(program, inputInfo, bindings, targetOptions);
    if (registration && result.spirv.size() > request.target.registrationWordLimit) {
        registration = false;
        program.Metadata().bindingLayoutComplete = false;
        bindings = bindingAllocator.Allocate(program, request.layout, false);
        targetOptions.registrationModules = false;
        result.spirv = spirvEmitter.Emit(program, inputInfo, bindings, targetOptions);
    }
    result.registrationModule = registration;

    result.bdaAbiVersion = program.Info().usesDma || program.Info().usesFaultBuffer ? request.target.bdaAbiVersion : 0u;
    result.memoryOffsetDword = bindings.layout.memoryOffsetDword;
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

    return {request.layout, std::move(program).TakeCompiledInfo(), std::move(static_cast<CompiledBindingLayout&>(bindings)), std::move(result)};
} catch (const std::bad_alloc&) {
    throw;
} catch (...) {
    if (emissionFailure != nullptr) *emissionFailure = std::current_exception();
    throw;
}

struct SpecializedModule {
    std::uint64_t specializationId = 0;
    SharedSpirv spirv;
    std::vector<std::uint32_t> bindings;
    bool pushData = false;
};

std::shared_ptr<const SpecializedModule> buildSpecializedModule(const CompiledShaderArtifact& artifact, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants, const SpirvTarget& target) {
    auto source = SpecializeVertexInputTypes(artifact, classes);
    std::map<std::uint32_t, std::uint32_t> supplied;
    for (const auto& constant : constants) {
        if (!supplied.emplace(constant.id, constant.value).second) throw std::runtime_error("duplicate prepared specialization ID");
    }
    std::map<std::uint32_t, std::uint32_t> values;
    std::set<std::uint32_t> specializedIds;
    const auto& words = source.Words();
    if (words.size() < 5u || words[0] != spv::MagicNumber) throw std::runtime_error("invalid prepared specialization module");
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
        if (count == 0u || count > words.size() - cursor) throw std::runtime_error("truncated prepared specialization instruction");
        if (op == spv::OpDecorate && count == 4u && words[cursor + 2u] == spv::DecorationSpecId) {
            const auto found = supplied.find(words[cursor + 3u]);
            if (found == supplied.end() || !values.emplace(words[cursor + 1u], found->second).second) throw std::runtime_error("missing or duplicate prepared specialization value");
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
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationBinding) bindingNumbers.emplace(materialized[cursor + 1u], materialized[cursor + 3u]);
        if (op == spv::OpVariable && count >= 4u) {
            if (const auto found = bindingNumbers.find(materialized[cursor + 2u]); found != bindingNumbers.end()) module->bindings.push_back(found->second);
            module->pushData |= materialized[cursor + 3u] == spv::StorageClassPushConstant;
        }
        cursor += count;
    }
    module->spirv = std::move(materialized);
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
    for (std::size_t index = 0; index < variant.info.info.buffers.size(); ++index) {
        const auto& descriptor = snapshot.buffers.at(index);
        key.push_back(descriptor.dwordCount);
        key.push_back(descriptor.dwords[1] & 0xffff0000u);
        key.push_back(descriptor.dwords[3]);
        const bool present = descriptor.dwords[2] != 0u && (descriptor.dwords[0] != 0u || (descriptor.dwords[1] & 0xffffu) != 0u);
        key.push_back(present ? 4u | (descriptor.dwords[0] & 3u) : 0u);
    }
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

// `resolved`: the snapshot's tables as ResolveImageTables gives them for the variant (the table
// memo's), resolved here when null.
RecompileResult materializeResult(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool registrationDefaults = false, const ResolvedImageTables* resolved = nullptr) {
    RecompileResult result;
    const auto& artifact = variant.artifact;
    static_cast<CompiledShaderArtifact&>(result) = {
        .memoryOffsetDword = artifact.memoryOffsetDword,
        .shaderDataDwords = artifact.shaderDataDwords,
        .imageMetadataDword = artifact.imageMetadataDword,
        .runtimeImageCount = artifact.runtimeImageCount,
        .runtimeImageResources = artifact.runtimeImageResources,
        .registrationModule = artifact.registrationModule,
        .bdaAbiVersion = artifact.bdaAbiVersion,
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
        .variantId = artifact.variantId
    };
    const auto plan = preparedBindingPlan(variant, snapshot, request.context.pixel ? std::span<const std::uint8_t>(request.context.pixel->targetExportMapping) : std::span<const std::uint8_t>{}, request.target.registrationModules && !UsesTables(variant.info.info));
    const DescriptorBindingPlan* bindingPlan = &plan->bindings;
    std::shared_ptr<PreparedModuleEntry> entry;
    struct LocalModuleKeyStorage {};
    auto& moduleKey = HostThreadLocal<std::vector<std::uint32_t>, LocalModuleKeyStorage>();
    moduleKey.clear();
    if (variant.bindings.layout.UsesPushData()) moduleKey.push_back(request.layout.pushConstantOffsetBytes / 4u);
    if (request.context.pixel) {
        for (const auto packing : request.context.pixel->targetExportPacking) moduleKey.push_back(static_cast<std::uint32_t>(packing));
        moduleKey.push_back(request.context.pixel->dualSourceBlend ? 1u : 0u);
        if (variant.info.info.rasterPrimitive) moduleKey.push_back(static_cast<std::uint32_t>(request.context.pixel->rasterPrimitive));
        if (variant.info.info.layerExported) moduleKey.push_back(request.context.pixel->layerExported ? 1u : 0u);
    }
    result.vertexAttributes.reserve(result.vertexInputs.size());
    std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources> vertexClasses{};
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
            vertexClasses.at(input.location) = kind;
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
    if (!plan->bindings.specialization.empty() || variant.bindings.layout.UsesPushData() || !artifact.vertexInputPatches.empty() || request.context.pixel) {
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
            if (request.context.pixel) {
                for (std::uint32_t target = 0; target < request.context.pixel->targetExportPacking.size(); ++target) constants.push_back({PipelineSpecialization::ExportPackingBase + target, moduleKey[index++]});
                constants.push_back({PipelineSpecialization::DualSourceBlend, moduleKey[index++]});
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
            const auto module = artifact.registrationModule && plan->bindings.specialization.empty() ? registrationModule(artifact.variantId, moduleKey, [&] { return specializeModule(variant.artifact, vertexClasses, constants, request.target); }) : specializeModule(variant.artifact, vertexClasses, constants, request.target);
            if (variant.bindings.layout.UsesPushData() && !module->pushData) throw std::runtime_error("specialization removed the prepared push constant interface");
            auto selected = DescriptorBindingBuilder{}.Select(plan->bindings, module->bindings);
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
        result.spirv = module->spirv;
        bindingPlan = &entry->bindings;
    } else {
        result.spirv = artifact.spirv;
    }
    ResolvedImageTables own;
    if (resolved == nullptr) own = ResourceMaterializer::ResolveImageTables(variant.info.info, snapshot, artifact.variantId);
    const auto& tables = resolved != nullptr ? *resolved : own;
    BindingAllocationResult bindings;
    DescriptorBindingBuilder{}.Populate(bindings, variant.bindings, *bindingPlan, variant.info.userDataBase, snapshot, partialThreads(request), &tables);
    result.workgroupMemoryDwords = WorkgroupMemoryStrideDwords(variant.info.info);
    result.bindings = std::move(bindings.bindings);
    result.pushConstants = std::move(bindings.pushConstants);
    const auto& layout = variant.bindings.layout;
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
    result.poisonedSrtReads = static_cast<std::uint32_t>(snapshot.srtPoison.size()) + snapshot.nullRootReads;
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
    std::shared_ptr<const CompiledVariant> variant;
    if (disk) {
        ShaderDiskCache::BuildKey(request, HostSubgroupSize(request), diskKey);
        CompiledVariant loaded;
        if (ShaderDiskCache::Load(diskKey, loaded)) {
            loaded.layout = request.layout;
            loaded.artifact.variantId = nextVariantId();
            variant = std::make_shared<const CompiledVariant>(std::move(loaded));
        }
    }
    if (variant == nullptr) {
        auto program = source.program != nullptr ? std::move(*source.program) : PrepareResourceProgram(request, preparation);
        source.program.reset();
        std::exception_ptr emissionFailure;
        try {
            variant = std::make_shared<const CompiledVariant>(compileVariant(request, std::move(program), &emissionFailure));
        } catch (...) {
            if (emissionFailure != nullptr && FailureMemo()) source.emissionFailures.push_back({request.shader.codeAddress, request.layout, emissionFailure});
            throw;
        }
        if (disk) ShaderDiskCache::Store(std::move(diskKey), variant);
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

RecompileResult RecompileImpl(const RecompileRequest& request) {
    static_cast<void>(RequestInputInfo(request));
    RequestMemoryView memory(request.context.memory);
    const auto runtime = memory.MakeRuntime(request.context.userData, request.shader.codeAddress);
    ResourceSnapshot snapshot;
    constexpr ResourceMaterializer materializer;
    if (!request.useCache) {
        auto program = PrepareResourceProgram(request);
        const auto plan = materializer.ExtractPlan(program);
        materializer.Materialize(plan, runtime, snapshot);
        const auto variant = compileVariant(request, std::move(program));
        return materializeResult(variant, request, snapshot);
    }
    const auto source = getSource(request);
    materializer.Materialize(*source->plan, runtime, snapshot);
    return materializeVariant(*source, request, snapshot);
}

// APS5_RESULT_MEMO_DIAG=1: every result memo miss is compared with the memo's nearest entry of the
// same variant (the fewest differing inputs), input by input, and the [recompile] line is followed
// by which inputs differed: the key inputs that change from draw to draw for one shader.
bool ResultMemoDiag() {
    static const bool diag = std::getenv("APS5_RESULT_MEMO_DIAG") != nullptr;
    return diag;
}

ResultKeyParts resultKeyParts(const RecompileRequest& request, const ResourceSnapshot& snapshot) {
    ResultKeyParts parts;
    std::uint64_t hash = 0;
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
            for (std::uint32_t i = 0; i < value.dwordCount && i < value.dwords.size(); ++i) mix(value.dwords[i]);
        }
    };
    const auto part = [&](ResultKeyPart index, auto&& body) {
        hash = 0xcbf29ce484222325ull;
        body();
        parts.hashes[index] = hash;
    };
    part(KeyBuffers, [&] { mixDescriptors(snapshot.buffers); });
    part(KeyImages, [&] { mixDescriptors(snapshot.images); });
    part(KeySamplers, [&] { mixDescriptors(snapshot.samplers); });
    part(KeyFlatSrt, [&] {
        mixWords(snapshot.flattenedSrt);
        mix(snapshot.nullRootReads);
    });
    part(KeyUserData, [&] { mixWords(snapshot.userData); });
    part(KeyUniformFill, [&] {
        mix(static_cast<std::uint64_t>(snapshot.uniformFill.kind));
        mix(snapshot.uniformFill.resource);
        for (const auto stride : snapshot.uniformFill.groupStride) mix(stride);
        mix(snapshot.uniformFill.words);
        mix(snapshot.uniformFill.value);
    });
    const auto& tables = snapshot.tables;
    part(KeyTableColumns, [&] {
        mix(tables.shader);
        mix(tables.tables.size());
        for (const auto& table : tables.tables) {
            mix(table.base);
            mix(table.size);
            mix(table.records);
            mix(table.keys);
            mix(table.fault);
            mix(table.outside ? 1u : 0u);
        }
    });
    part(KeyTableCodes, [&] {
        for (const auto& table : tables.tables) mixWords(table.codes);
    });
    part(KeyTableReadRanges, [&] {
        for (const auto& table : tables.tables) {
            mix(table.readRanges.size());
            for (const auto& [base, size] : table.readRanges) {
                mix(base);
                mix(size);
            }
        }
    });
    part(KeyTableWords, [&] { mixDescriptors(tables.words); });
    part(KeyTableRanges, [&] {
        mix(tables.ranges.size());
        for (const auto& [base, size] : tables.ranges) {
            mix(base);
            mix(size);
        }
    });
    part(KeyPartialThreads, [&] {
        for (const auto threads : partialThreads(request)) mix(threads);
    });
    part(KeyPushOffset, [&] { mix(request.layout.pushConstantOffsetBytes); });
    part(KeyPixel, [&] {
        if (!request.context.pixel) return;
        for (const auto mapping : request.context.pixel->targetExportMapping) mix(mapping);
        for (const auto packing : request.context.pixel->targetExportPacking) mix(static_cast<std::uint64_t>(packing));
        mix(request.context.pixel->dualSourceBlend);
    });
    part(KeyVertex, [&] {
        if (!request.context.vertex) return;
        const auto& vertex = *request.context.vertex;
        const auto count = std::min<std::uint32_t>(vertex.resourcesNum, ShaderVertexStageInfo::MaxResources);
        mix(count);
        mix(vertex.paClVsOutCntl);
        if (request.graphics && request.graphics->tessellation) {
            const auto& tessellation = *request.graphics->tessellation;
            for (const auto value : {tessellation.inputControlPoints, tessellation.outputControlPoints, tessellation.domain, tessellation.partitioning, tessellation.outputTopology}) mix(value);
            mix(tessellation.mergedHull);
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            for (const auto field : vertex.resources[i].fields) mix(field);
            mix(vertex.resourcesDst[i].fetchIndex);
        }
    });
    parts.userData = snapshot.userData;
    parts.flatSrt = snapshot.flattenedSrt;
    parts.bufferWords.reserve(snapshot.buffers.size() * 8u);
    for (const auto& value : snapshot.buffers) {
        for (std::uint32_t i = 0; i < 8u; ++i) parts.bufferWords.push_back(i < value.dwordCount ? value.dwords[i] : 0u);
    }
    return parts;
}

struct ResultMemoDiagCounters {
    std::mutex mutex;
    std::uint64_t misses = 0;
    std::uint64_t firstOfVariant = 0;
    std::uint64_t evictedBefore = 0;
    std::uint64_t identical = 0;
    std::array<std::uint64_t, KeyPartCount> differing{};
    std::array<std::uint64_t, KeyPartCount> alone{};
    // Misses whose only differing input is user data the variant's result never reads (no word of
    // it at a register of the layout's userDataRegisters).
    std::uint64_t unconsumedUserData = 0;
    std::map<std::uint32_t, std::uint64_t> userDataWords;
    std::map<std::uint32_t, std::uint64_t> flatSlots;
    std::map<std::uint32_t, std::uint64_t> bufferWords;
    std::map<std::uint64_t, std::uint64_t> shaders;
};

ResultMemoDiagCounters& resultMemoDiag() {
    static ResultMemoDiagCounters counters;
    return counters;
}

std::optional<std::uint32_t> firstDifference(const std::vector<std::uint32_t>& left, const std::vector<std::uint32_t>& right) {
    const auto count = std::min(left.size(), right.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (left[i] != right[i]) return static_cast<std::uint32_t>(i);
    }
    if (left.size() != right.size()) return static_cast<std::uint32_t>(count);
    return std::nullopt;
}

// Under source.mutex, before the miss's entry goes in: the nearest entry of the variant.
void diagnoseMiss(const SourceEntry& source, const CompiledVariant& variant, const RecompileRequest& request, std::uint64_t index, const ResultKeyParts& parts) {
    auto& counters = resultMemoDiag();
    const ResultKeyParts* nearest = nullptr;
    std::size_t nearestCount = KeyPartCount + 1u;
    for (const auto& entry : source.memo) {
        if (entry.variantId != variant.artifact.variantId) continue;
        const auto found = source.diagParts.find((entry.variantId * 0x9e3779b97f4a7c15ull) ^ entry.hash);
        if (found == source.diagParts.end()) continue;
        std::size_t count = 0;
        for (std::size_t part = 0; part < KeyPartCount; ++part) count += found->second->hashes[part] != parts.hashes[part] ? 1u : 0u;
        if (count < nearestCount) {
            nearestCount = count;
            nearest = found->second.get();
        }
    }
    std::lock_guard lock(counters.mutex);
    ++counters.misses;
    ++counters.shaders[request.shader.codeAddress];
    if (source.diagEvicted.contains(index)) ++counters.evictedBefore;
    if (nearest == nullptr) {
        ++counters.firstOfVariant;
        return;
    }
    if (nearestCount == 0u) {
        ++counters.identical;
        return;
    }
    for (std::size_t part = 0; part < KeyPartCount; ++part) {
        if (nearest->hashes[part] == parts.hashes[part]) continue;
        ++counters.differing[part];
        if (nearestCount == 1u) ++counters.alone[part];
    }
    if (nearest->hashes[KeyUserData] != parts.hashes[KeyUserData]) {
        if (const auto word = firstDifference(nearest->userData, parts.userData)) ++counters.userDataWords[*word];
        if (nearestCount == 1u && nearest->userData.size() == parts.userData.size()) {
            const auto& registers = variant.bindings.layout.userDataRegisters;
            bool consumed = false;
            for (std::size_t word = 0; word < parts.userData.size() && !consumed; ++word) {
                if (nearest->userData[word] == parts.userData[word]) continue;
                consumed = std::find(registers.begin(), registers.end(), variant.info.userDataBase + static_cast<std::uint32_t>(word)) != registers.end();
            }
            if (!consumed) ++counters.unconsumedUserData;
        }
    }
    if (nearest->hashes[KeyFlatSrt] != parts.hashes[KeyFlatSrt]) {
        if (const auto slot = firstDifference(nearest->flatSrt, parts.flatSrt)) ++counters.flatSlots[*slot];
    }
    if (nearest->hashes[KeyBuffers] != parts.hashes[KeyBuffers]) {
        if (const auto word = firstDifference(nearest->bufferWords, parts.bufferWords)) ++counters.bufferWords[*word];
    }
}

void reportResultMemoDiag() {
    auto& counters = resultMemoDiag();
    std::lock_guard lock(counters.mutex);
    std::string text;
    char buffer[512];
    const auto append = [&](const char* format, auto... values) {
        std::snprintf(buffer, sizeof(buffer), format, values...);
        text += buffer;
    };
    append("[recompile] result memo misses against the nearest entry of their variant (10 s): %llu misses, %llu first of their variant, %llu identical inputs (a hash-only difference), %llu of a key evicted before; differing inputs (alone):", static_cast<unsigned long long>(counters.misses), static_cast<unsigned long long>(counters.firstOfVariant), static_cast<unsigned long long>(counters.identical), static_cast<unsigned long long>(counters.evictedBefore));
    for (std::size_t part = 0; part < KeyPartCount; ++part) {
        if (counters.differing[part] != 0u) append(" %s %llu (%llu)", KeyPartNames[part], static_cast<unsigned long long>(counters.differing[part]), static_cast<unsigned long long>(counters.alone[part]));
    }
    append("; user data alone and never read by the result %llu", static_cast<unsigned long long>(counters.unconsumedUserData));
    const auto top = [&](const char* name, const std::map<std::uint32_t, std::uint64_t>& counts, bool buffers) {
        std::vector<std::pair<std::uint64_t, std::uint32_t>> sorted;
        for (const auto& [key, count] : counts) sorted.emplace_back(count, key);
        std::sort(sorted.rbegin(), sorted.rend());
        append("; first differing %s:", name);
        for (std::size_t i = 0; i < sorted.size() && i < 8u; ++i) {
            if (buffers) append(" %u.%u=%llu", sorted[i].second / 8u, sorted[i].second % 8u, static_cast<unsigned long long>(sorted[i].first));
            else append(" %u=%llu", sorted[i].second, static_cast<unsigned long long>(sorted[i].first));
        }
    };
    top("user-data words", counters.userDataWords, false);
    top("flat SRT slots", counters.flatSlots, false);
    top("buffer words (buffer.dword)", counters.bufferWords, true);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> shaders;
    for (const auto& [address, count] : counters.shaders) shaders.emplace_back(count, address);
    std::sort(shaders.rbegin(), shaders.rend());
    append("; misses by shader:");
    for (std::size_t i = 0; i < shaders.size() && i < 6u; ++i) append(" 0x%llx=%llu", static_cast<unsigned long long>(shaders[i].second), static_cast<unsigned long long>(shaders[i].first));
    std::fprintf(stderr, "%s\n", text.c_str());
    counters.misses = counters.firstOfVariant = counters.evictedBefore = counters.identical = counters.unconsumedUserData = 0;
    counters.differing.fill(0);
    counters.alone.fill(0);
    counters.userDataWords.clear();
    counters.flatSlots.clear();
    counters.bufferWords.clear();
    counters.shaders.clear();
}

// APS5_NO_RESULT_MEMO=1: every Recompile(request, capture) materializes its own result as before.
bool ResultMemo() {
    static const bool resultMemo = std::getenv("APS5_NO_RESULT_MEMO") == nullptr;
    return resultMemo;
}

// APS5_RESULT_MEMO_ENTRIES=<n>: the result memo's entries per shader (2048; 256 before the p7 diag
// counted 78% of the misses as keys evicted earlier).
std::size_t ResultMemoEntries() {
    static const std::size_t entries = [] {
        const char* text = std::getenv("APS5_RESULT_MEMO_ENTRIES");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull;
        return value != 0ull ? static_cast<std::size_t>(std::min<unsigned long long>(value, 65536ull)) : std::size_t{2048};
    }();
    return entries;
}

// APS5_RESULT_MEMO_MB=<n>: the bytes all shaders' result memos may keep together (1024 MiB). An
// image-table shader's result carries its whole table map and element list (~0.25-0.4 MiB for
// 007's 19k-key tables), so 2048 of them would be ~0.5-0.8 GiB per shader: past the budget, an
// insert evicts the inserting shader's least recently used entries (it keeps at least
// ResultMemoKeep), so the shaders that miss most give back what they take.
std::size_t ResultMemoBudget() {
    static const std::size_t bytes = [] {
        const char* text = std::getenv("APS5_RESULT_MEMO_MB");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull;
        return static_cast<std::size_t>(value != 0ull ? std::min<unsigned long long>(value, 1ull << 20u) : 1024ull) << 20u;
    }();
    return bytes;
}

constexpr std::size_t ResultMemoKeep = 64;

// What a memoized result keeps alive besides its shared SPIR-V (the module's, shared by every
// result of the variant): the bindings with their descriptor words and per-element vectors, the
// push constants, the image-table fields and the artifact's small copies.
std::size_t resultBytes(const RecompileResult& result) {
    std::size_t bytes = sizeof(RecompileResult);
    const auto bits = [](const std::vector<bool>& flags) { return (flags.size() + 7u) / 8u; };
    for (const auto& binding : result.bindings) {
        bytes += sizeof(DescriptorBinding) + binding.guestDescriptor.capacity() * sizeof(std::uint32_t) + binding.imageShapes.capacity() * sizeof(DescriptorImageShape) + binding.imageSamplers.capacity() * sizeof(std::uint32_t);
        bytes += bits(binding.samplerDepthCompare) + bits(binding.imageWritten) + bits(binding.imageDepthCompare) + bits(binding.imageAtomic) + bits(binding.imageAtomic64) + bits(binding.bufferAtomic) + bits(binding.bufferWritten) + bits(binding.samplerUnnormalized) + bits(binding.imageUnnormalized) + bits(binding.bufferRead);
    }
    bytes += result.specialization.capacity() * sizeof(PipelineSpecializationConstant) + result.pushConstants.capacity() + result.vertexAttributes.capacity() * sizeof(VertexAttribute);
    bytes += result.imageTablePoison.capacity() * sizeof(ImageTableEntryPoison) + result.imageTableRanges.capacity() * sizeof(std::pair<std::uint64_t, std::uint64_t>);
    for (const auto& ranges : result.imageTableReadRanges) bytes += sizeof(ranges) + ranges.capacity() * sizeof(std::pair<std::uint64_t, std::uint64_t>);
    bytes += result.vertexInputPatches.capacity() * sizeof(VertexInputPatch) + result.runtimeImageResources.capacity() * sizeof(std::uint32_t) + result.vertexInputs.capacity() * sizeof(VertexInput) + result.parameterExports.capacity() * sizeof(std::uint32_t) + result.fragmentParameters.capacity() * sizeof(FragmentParameter);
    return bytes;
}

// The bytes and entries all result memos keep, and the largest entry inserted in the window.
struct ResultMemoHeld {
    std::atomic<std::uint64_t> bytes{0};
    std::atomic<std::uint64_t> entries{0};
    std::atomic<std::uint64_t> largest{0};
    std::atomic<std::uint64_t> budgetEvictions{0};
};

ResultMemoHeld& resultMemoHeld() {
    static ResultMemoHeld held;
    return held;
}

// APS5_NO_CANONICAL_RESULT_KEY=1: the key hashes the whole user data and the tables with FNV-1a, as
// before ResultKey.
bool CanonicalResultKey() {
    static const bool canonical = std::getenv("APS5_NO_CANONICAL_RESULT_KEY") == nullptr;
    return canonical;
}

bool TableResultMemo() {
    static const bool memo = std::getenv("APS5_NO_TABLE_RESULT_MEMO") == nullptr;
    return memo;
}

// APS5_VERIFY_RESULT_MEMO=1: every result memo hit and every table memo hit is materialized
// (resolved) again from the snapshot and compared; a difference aborts.
bool VerifyResultMemo() {
    static const bool verify = std::getenv("APS5_VERIFY_RESULT_MEMO") != nullptr;
    return verify;
}

constexpr std::size_t TableMemoEntries = 32;

struct ResultMemoCounters {
    std::atomic<std::uint64_t> hits{0}, misses{0}, evictions{0}, populateNanoseconds{0}, tableHits{0}, tableMisses{0}, verified{0};
    std::atomic<std::int64_t> lastReport{0};
    // Never reset (ResultMemoTotalCounts).
    std::atomic<std::uint64_t> totalHits{0}, totalMisses{0}, totalTableHits{0}, totalTableMisses{0}, totalVerified{0};
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
    const auto tableHits = counters.tableHits.exchange(0, std::memory_order_relaxed);
    const auto tableMisses = counters.tableMisses.exchange(0, std::memory_order_relaxed);
    const auto verified = counters.verified.exchange(0, std::memory_order_relaxed);
    auto& held = resultMemoHeld();
    const auto largest = held.largest.exchange(0, std::memory_order_relaxed);
    const auto budgetEvictions = held.budgetEvictions.exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr, "[recompile] result memo (10 s): %llu hits, %llu misses (%.1f%% hits), Populate %.1f us per miss / %.1f ms in total, %llu evictions (%llu by the byte budget; %zu entries per shader, %zu MiB budget, %s key); held %.1f MiB in %llu entries, largest inserted %.1f KiB; tables of the misses: %llu from the table memo, %llu resolved; %llu verified\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses), hits + misses != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses) : 0.0, misses != 0 ? static_cast<double>(populate) / 1000.0 / static_cast<double>(misses) : 0.0, static_cast<double>(populate) / 1e6, static_cast<unsigned long long>(evictions), static_cast<unsigned long long>(budgetEvictions), ResultMemoEntries(), ResultMemoBudget() >> 20u, CanonicalResultKey() ? "canonical" : "full", static_cast<double>(held.bytes.load(std::memory_order_relaxed)) / 1048576.0, static_cast<unsigned long long>(held.entries.load(std::memory_order_relaxed)), static_cast<double>(largest) / 1024.0, static_cast<unsigned long long>(tableHits), static_cast<unsigned long long>(tableMisses), static_cast<unsigned long long>(verified));
    if (ResultMemoDiag()) reportResultMemoDiag();
}

// The key as it was before ResultKey (APS5_NO_CANONICAL_RESULT_KEY=1).
std::uint64_t fullSnapshotHash(const RecompileRequest& request, const ResourceSnapshot& snapshot) {
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
            for (std::uint32_t i = 0; i < value.dwordCount && i < value.dwords.size(); ++i) mix(value.dwords[i]);
        }
    };
    mixDescriptors(snapshot.buffers);
    mixDescriptors(snapshot.images);
    mixDescriptors(snapshot.samplers);
    mixWords(snapshot.flattenedSrt);
    mix(snapshot.nullRootReads);
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
    mix(request.layout.pushConstantOffsetBytes);
    if (request.context.pixel) {
        for (const auto mapping : request.context.pixel->targetExportMapping) mix(mapping);
        for (const auto packing : request.context.pixel->targetExportPacking) mix(static_cast<std::uint64_t>(packing));
        mix(request.context.pixel->dualSourceBlend);
        mix(static_cast<std::uint64_t>(request.context.pixel->rasterPrimitive));
        mix(request.context.pixel->layerExported);
    }
    if (request.context.vertex) {
        const auto& vertex = *request.context.vertex;
        const auto count = std::min<std::uint32_t>(vertex.resourcesNum, ShaderVertexStageInfo::MaxResources);
        mix(count);
        mix(vertex.paClVsOutCntl);
        if (request.graphics && request.graphics->tessellation) {
            const auto& tessellation = *request.graphics->tessellation;
            for (const auto value : {tessellation.inputControlPoints, tessellation.outputControlPoints, tessellation.domain, tessellation.partitioning, tessellation.outputTopology}) mix(value);
            mix(tessellation.mergedHull);
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            for (const auto field : vertex.resources[i].fields) mix(field);
            mix(vertex.resourcesDst[i].fetchIndex);
        }
    } else {
        mix(1ull << 32u);
    }
    return hash;
}

// Everything materializeResult reads besides the variant: the snapshot (the descriptor words, the
// flattened SRT, the user data, the uniform fill) and, for the vertex family, the V# table the
// attributes are resolved from. Canonical (the default): the user data only at the registers the
// variant's result reads (ResultKey::ConsumedUserDataHash) and the tables through
// ResultKey::TablesHash, which `tablesHash` receives (0 without tables) for the table memo.
std::uint64_t snapshotHash(const RecompileRequest& request, const ResourceSnapshot& snapshot, const CompiledVariant& variant, std::uint64_t& tablesHash) {
    // Not computed when neither the canonical key nor the table memo takes it (both switches set).
    const bool tablesKeyed = CanonicalResultKey() || TableResultMemo();
    tablesHash = snapshot.tables.tables.empty() || !tablesKeyed ? 0u : snapshot.tables.hash != 0u ? snapshot.tables.hash : ResultKey::TablesHash(snapshot.tables);
    if (!CanonicalResultKey()) return fullSnapshotHash(request, snapshot);
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ull;
    };
    const auto mixDescriptors = [&](const std::vector<DescriptorValue>& values) {
        mix(values.size());
        for (const auto& value : values) {
            mix(value.dwordCount);
            for (std::uint32_t i = 0; i < value.dwordCount && i < value.dwords.size(); ++i) mix(value.dwords[i]);
        }
    };
    mixDescriptors(snapshot.buffers);
    mixDescriptors(snapshot.images);
    mixDescriptors(snapshot.samplers);
    mix(ResultKey::HashWords(snapshot.flattenedSrt, snapshot.flattenedSrt.size()));
    mix(snapshot.nullRootReads);
    mix(ResultKey::ConsumedUserDataHash(snapshot.userData, variant.bindings.layout.userDataRegisters, variant.info.userDataBase));
    mix(static_cast<std::uint64_t>(snapshot.uniformFill.kind));
    mix(snapshot.uniformFill.resource);
    for (const auto stride : snapshot.uniformFill.groupStride) mix(stride);
    mix(snapshot.uniformFill.words);
    mix(snapshot.uniformFill.value);
    mix(tablesHash);
    for (const auto threads : partialThreads(request)) mix(threads);
    mix(request.layout.pushConstantOffsetBytes);
    if (request.context.pixel) {
        for (const auto mapping : request.context.pixel->targetExportMapping) mix(mapping);
        for (const auto packing : request.context.pixel->targetExportPacking) mix(static_cast<std::uint64_t>(packing));
        mix(request.context.pixel->dualSourceBlend);
    }
    if (request.context.vertex) {
        const auto& vertex = *request.context.vertex;
        const auto count = std::min<std::uint32_t>(vertex.resourcesNum, ShaderVertexStageInfo::MaxResources);
        mix(count);
        mix(vertex.paClVsOutCntl);
        if (request.graphics && request.graphics->tessellation) {
            const auto& tessellation = *request.graphics->tessellation;
            for (const auto value : {tessellation.inputControlPoints, tessellation.outputControlPoints, tessellation.domain, tessellation.partitioning, tessellation.outputTopology}) mix(value);
            mix(tessellation.mergedHull);
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            for (const auto field : vertex.resources[i].fields) mix(field);
            mix(vertex.resourcesDst[i].fetchIndex);
        }
    } else {
        mix(1ull << 32u);
    }
    return hash;
}

// The table memo's key: the tables and every sampler descriptor (ResolveImageTables reads the
// samplers paired with table images), for one variant.
std::uint64_t tableMemoKey(std::uint64_t tablesHash, const ResourceSnapshot& snapshot) {
    auto key = tablesHash;
    for (const auto& sampler : snapshot.samplers) key = ResultKey::Detail::Step(key, ResultKey::HashWords(std::span<const std::uint32_t>(sampler.dwords).first(std::min<std::size_t>(sampler.dwordCount, sampler.dwords.size())), sampler.dwordCount), ResultKey::Detail::LaneMultipliers[3]);
    return key;
}

// APS5_VERIFY_RESULT_MEMO: the first field in which two results differ (empty when none).
std::string resultDifference(const RecompileResult& left, const RecompileResult& right) {
    if (left.variantId != right.variantId || left.specializationId != right.specializationId || !(left.specialization == right.specialization)) return "variant or specialization";
    if (!(left.spirv == right.spirv)) return "SPIR-V";
    if (left.pushConstants != right.pushConstants) return "push constants";
    if (left.bindings.size() != right.bindings.size()) return "binding count";
    for (std::size_t index = 0; index < left.bindings.size(); ++index) {
        const auto& a = left.bindings[index];
        const auto& b = right.bindings[index];
        const auto at = [&](const char* field) { return "binding " + std::to_string(index) + " " + field; };
        if (a.kind != b.kind || a.role != b.role || a.descriptorSet != b.descriptorSet || a.binding != b.binding || a.count != b.count || a.readOnly != b.readOnly) return at("header");
        if (a.guestDescriptor != b.guestDescriptor) return at("guest descriptor");
        if (a.imageShape != b.imageShape || a.imageShapes != b.imageShapes) return at("image shapes");
        if (a.samplerDepthCompare != b.samplerDepthCompare || a.imageDepthCompare != b.imageDepthCompare) return at("depth compare");
        if (a.imageWritten != b.imageWritten || a.imageAtomic != b.imageAtomic || a.imageAtomic64 != b.imageAtomic64) return at("image access");
        if (a.bufferAtomic != b.bufferAtomic || a.bufferWritten != b.bufferWritten || a.bufferRead != b.bufferRead) return at("buffer access");
        if (a.samplerUnnormalized != b.samplerUnnormalized || a.imageUnnormalized != b.imageUnnormalized || a.imageSamplers != b.imageSamplers) return at("samplers");
    }
    if (left.vertexAttributes.size() != right.vertexAttributes.size()) return "vertex attribute count";
    for (std::size_t index = 0; index < left.vertexAttributes.size(); ++index) {
        const auto& a = left.vertexAttributes[index];
        const auto& b = right.vertexAttributes[index];
        if (a.location != b.location || a.components != b.components || a.resource.fields != b.resource.fields || a.fetchIndex != b.fetchIndex || a.formatComponents != b.formatComponents) return "vertex attribute " + std::to_string(index);
    }
    if (left.poisonedSrtReads != right.poisonedSrtReads) return "poisoned SRT reads";
    if (left.imageTablePoison.size() != right.imageTablePoison.size()) return "image table poison count";
    for (std::size_t index = 0; index < left.imageTablePoison.size(); ++index) {
        const auto& a = left.imageTablePoison[index];
        const auto& b = right.imageTablePoison[index];
        if (a.words != b.words || a.dwordCount != b.dwordCount || a.resource != b.resource || a.reason != b.reason) return "image table poison " + std::to_string(index);
    }
    if (left.imageTableRanges != right.imageTableRanges || left.imageTableReadRanges != right.imageTableReadRanges) return "image table ranges";
    if (left.imageTableShader != right.imageTableShader || left.imageTableFaults != right.imageTableFaults) return "image table shader or faults";
    if (left.workgroupMemoryDwords != right.workgroupMemoryDwords) return "workgroup memory";
    return {};
}

std::string tablesDifference(const ResolvedImageTables& left, const ResolvedImageTables& right) {
    if (left.map != right.map) return "map";
    if (left.faults != right.faults) return "faults";
    if (left.elements.size() != right.elements.size() || left.storageElements.size() != right.storageElements.size() || left.samplers.size() != right.samplers.size() || left.poison.size() != right.poison.size()) return "counts";
    for (std::size_t index = 0; index < left.elements.size(); ++index) {
        const auto& a = left.elements[index];
        const auto& b = right.elements[index];
        if (a.words != b.words || a.dimension != b.dimension || a.depthCompare != b.depthCompare || a.kind != b.kind || a.samplers != b.samplers) return "element " + std::to_string(index);
    }
    for (std::size_t index = 0; index < left.storageElements.size(); ++index) {
        const auto& a = left.storageElements[index];
        const auto& b = right.storageElements[index];
        if (a.words != b.words || a.dimension != b.dimension || a.depthCompare != b.depthCompare || a.kind != b.kind || a.samplers != b.samplers) return "storage element " + std::to_string(index);
    }
    for (std::size_t index = 0; index < left.samplers.size(); ++index) {
        const auto& a = left.samplers[index];
        const auto& b = right.samplers[index];
        if (a.words != b.words || a.compare != b.compare || a.point != b.point) return "sampler " + std::to_string(index);
    }
    for (std::size_t index = 0; index < left.poison.size(); ++index) {
        const auto& a = left.poison[index];
        const auto& b = right.poison[index];
        if (a.words != b.words || a.dwordCount != b.dwordCount || a.resource != b.resource || a.reason != b.reason) return "poison " + std::to_string(index);
    }
    return {};
}

[[noreturn]] void verifyFailed(const char* what, const RecompileRequest& request, const std::string& field) {
    std::fprintf(stderr, "[recompile] APS5_VERIFY_RESULT_MEMO: a %s of shader 0x%llx differs from a fresh materialization in %s\n", what, static_cast<unsigned long long>(request.shader.codeAddress), field.c_str());
    std::fflush(stderr);
    std::abort();
}

// The memo'd result of `source`'s variant for the snapshot (design13 R5): a hit returns the shared
// object, a miss materializes outside the source mutex and inserts (a concurrent miss's object is
// as good). `memoHit` reports the hit.
std::shared_ptr<const RecompileResult> materializePreparedMemoized(SourceEntry& source, const std::shared_ptr<const CompiledVariant>& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool cacheHit, bool* memoHit) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::uint64_t tablesHash = 0;
    const auto hash = snapshotHash(request, snapshot, *variant, tablesHash);
    std::uint64_t index = 0;
    auto& counters = resultMemoCounters();
    {
        std::shared_ptr<const RecompileResult> hit;
        {
            std::lock_guard lock(source.mutex);
            index = (variant->artifact.variantId * 0x9e3779b97f4a7c15ull) ^ hash;
            const auto found = source.memoIndex.find(index);
            if (found != source.memoIndex.end() && found->second->variantId == variant->artifact.variantId && found->second->hash == hash) {
                source.memo.splice(source.memo.begin(), source.memo, found->second);
                hit = found->second->result;
            }
        }
        if (hit != nullptr) {
            counters.hits.fetch_add(1, std::memory_order_relaxed);
            counters.totalHits.fetch_add(1, std::memory_order_relaxed);
            if (VerifyResultMemo()) {
                if (const auto field = resultDifference(*hit, materializeResult(*variant, request, snapshot)); !field.empty()) verifyFailed("result memo hit", request, field);
                counters.verified.fetch_add(1, std::memory_order_relaxed);
                counters.totalVerified.fetch_add(1, std::memory_order_relaxed);
            }
            if (memoHit != nullptr) *memoHit = true;
            reportResultMemo();
            return hit;
        }
    }
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // The tables of a recent snapshot of the variant with the same table inputs, resolved once.
    std::shared_ptr<const ResolvedImageTables> tables;
    if (TableResultMemo() && tablesHash != 0u) {
        const auto key = tableMemoKey(tablesHash, snapshot);
        {
            std::lock_guard lock(source.mutex);
            for (auto entry = source.tableMemo.begin(); entry != source.tableMemo.end(); ++entry) {
                if (entry->variantId != variant->artifact.variantId || entry->key != key) continue;
                source.tableMemo.splice(source.tableMemo.begin(), source.tableMemo, entry);
                tables = entry->tables;
                break;
            }
        }
        if (tables != nullptr) {
            counters.tableHits.fetch_add(1, std::memory_order_relaxed);
            counters.totalTableHits.fetch_add(1, std::memory_order_relaxed);
            if (VerifyResultMemo()) {
                if (const auto field = tablesDifference(*tables, ResourceMaterializer::ResolveImageTables(variant->info.info, snapshot, variant->artifact.variantId)); !field.empty()) verifyFailed("table memo hit", request, field);
                counters.verified.fetch_add(1, std::memory_order_relaxed);
                counters.totalVerified.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            tables = std::make_shared<const ResolvedImageTables>(ResourceMaterializer::ResolveImageTables(variant->info.info, snapshot, variant->artifact.variantId));
            counters.tableMisses.fetch_add(1, std::memory_order_relaxed);
            counters.totalTableMisses.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard lock(source.mutex);
            source.tableMemo.push_front({variant->artifact.variantId, key, tables});
            if (source.tableMemo.size() > TableMemoEntries) source.tableMemo.pop_back();
        }
    }
    auto result = std::make_shared<RecompileResult>(materializeResult(*variant, request, snapshot, false, tables.get()));
    result->cacheHit = cacheHit;
    if (profile) counters.populateNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    counters.misses.fetch_add(1, std::memory_order_relaxed);
    counters.totalMisses.fetch_add(1, std::memory_order_relaxed);
    std::shared_ptr<const RecompileResult> shared = std::move(result);
    const auto parts = ResultMemoDiag() ? std::make_shared<const ResultKeyParts>(resultKeyParts(request, snapshot)) : nullptr;
    const auto bytes = resultBytes(*shared);
    auto& held = resultMemoHeld();
    {
        std::lock_guard lock(source.mutex);
        const auto release = [&](const ResultMemoEntry& entry) {
            held.bytes.fetch_sub(entry.bytes, std::memory_order_relaxed);
            held.entries.fetch_sub(1, std::memory_order_relaxed);
        };
        const auto found = source.memoIndex.find(index);
        if (found != source.memoIndex.end()) {
            if (found->second->variantId == variant->artifact.variantId && found->second->hash == hash) {
                source.memo.splice(source.memo.begin(), source.memo, found->second);
                shared = found->second->result;
            } else {
                release(*found->second);
                source.memo.erase(found->second);
                source.memoIndex.erase(found);
                if (parts != nullptr) source.diagParts.erase(index);
            }
        }
        if (source.memoIndex.find(index) == source.memoIndex.end()) {
            if (parts != nullptr) {
                diagnoseMiss(source, *variant, request, index, *parts);
                source.diagParts[index] = parts;
                source.diagEvicted.erase(index);
            }
            source.memo.push_front({variant->artifact.variantId, hash, shared, bytes});
            source.memoIndex.emplace(index, source.memo.begin());
            held.bytes.fetch_add(bytes, std::memory_order_relaxed);
            held.entries.fetch_add(1, std::memory_order_relaxed);
            for (auto largest = held.largest.load(std::memory_order_relaxed); bytes > largest && !held.largest.compare_exchange_weak(largest, bytes, std::memory_order_relaxed);) {}
            for (;;) {
                const bool overCount = source.memo.size() > ResultMemoEntries();
                const bool overBudget = held.bytes.load(std::memory_order_relaxed) > ResultMemoBudget() && source.memo.size() > ResultMemoKeep;
                if (!overCount && !overBudget) break;
                const auto& last = source.memo.back();
                const auto evicted = (last.variantId * 0x9e3779b97f4a7c15ull) ^ last.hash;
                source.memoIndex.erase(evicted);
                if (parts != nullptr) {
                    source.diagParts.erase(evicted);
                    if (source.diagEvicted.size() >= 16384u) source.diagEvicted.clear();
                    source.diagEvicted.insert(evicted);
                }
                release(last);
                source.memo.pop_back();
                counters.evictions.fetch_add(1, std::memory_order_relaxed);
                if (!overCount) held.budgetEvictions.fetch_add(1, std::memory_order_relaxed);
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
    if (std::none_of(plan.pureFlatSlots.begin(), plan.pureFlatSlots.end(), [](std::uint8_t pure) { return pure != 0u; })) {
        ResourceMaterializer{}.Materialize(plan, runtime, capture.snapshot);
        return;
    }
    SrtRuntime traced = runtime;
    traced.readTrace = &capture.readTrace;
    ResourceMaterializer{}.Materialize(plan, traced, capture.snapshot);
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
    if (!std::ranges::equal(key, handle.staticKey) || handle.source->code.size() != request.shader.code.size()) return false;
    return handle.source->code.data() == request.shader.code.data() || std::ranges::equal(handle.source->code, request.shader.code);
}

std::span<const std::uint32_t> GetPreparedCode(const SourceHandle& handle) {
    if (handle.source == nullptr || handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.source->code;
}

PreparedShaderInvocation::PreparedShaderInvocation(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle) : request(request), handle(handle) {}

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

const CompiledShaderArtifact& GetPreparedArtifact(const SourceHandle& handle) {
    if (handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.artifact->artifact;
}

std::shared_ptr<const RecompileResult> RegistrationResult(const RecompileRequest& request, const SourceHandle& handle) {
    if (handle.artifact == nullptr || !MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    if (!request.target.registrationModules || !handle.artifact->artifact.registrationModule) return nullptr;
    const auto& info = handle.artifact->info.info;
    if (std::ranges::any_of(info.buffers, [](const BufferResource& buffer) { return buffer.descriptorFormatted; }) || std::ranges::any_of(info.samplers, [](const SamplerResource& sampler) { return FoldsTexelOffsets(sampler); })) return nullptr;
    ResourceSnapshot snapshot;
    snapshot.buffers.assign(info.buffers.size(), {{}, 4u});
    snapshot.images.assign(info.images.size(), {{}, 8u});
    snapshot.samplers.assign(info.samplers.size(), {{}, 4u});
    snapshot.userData.assign(RuntimeAbi::UserDataCapacity, 0u);
    snapshot.flattenedSrt.assign(1u, 0u);
    return std::make_shared<RecompileResult>(materializeResult(*handle.artifact, request, snapshot, true));
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

ResultMemoTotals ResultMemoTotalCounts() {
    const auto& counters = resultMemoCounters();
    const auto& held = resultMemoHeld();
    return {counters.totalHits.load(std::memory_order_relaxed), counters.totalMisses.load(std::memory_order_relaxed), counters.totalTableHits.load(std::memory_order_relaxed), counters.totalTableMisses.load(std::memory_order_relaxed), counters.totalVerified.load(std::memory_order_relaxed), held.bytes.load(std::memory_order_relaxed), held.entries.load(std::memory_order_relaxed)};
}

std::size_t ResultMemoEntryBytes(const RecompileResult& result) {
    return resultBytes(result);
}

}
