#include "prx/libSceAgcDriver/Execution/src/Driver/Dispatch/TableCallDispatch.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint64_t MaxHeadGroups = 65535;

struct ProgramCache {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<TableCallProgram>>> programs;
};

ProgramCache& Programs() {
    static ProgramCache cache;
    return cache;
}

struct SplitCounters {
    std::mutex mutex;
    BranchSplitCounts counts;
};

SplitCounters& Counters() {
    static SplitCounters counters;
    return counters;
}

void Remember(std::uint64_t programKey, const std::shared_ptr<TableCallProgram>& program) {
    auto& cache = Programs();
    std::lock_guard lock(cache.mutex);
    cache.programs[programKey].push_back(program);
}

}

TableCallProgram::TableCallProgram(ShaderRecompiler::TableCallSplit split, LinkedCallees callees)
    : split(std::move(split)), callees(std::move(callees)) {
}

std::vector<std::uint64_t> TableCallProgram::tablePointers(const ShaderRecompiler::RecompileRequest& request) const {
    std::vector<std::uint64_t> pointers;
    for (const auto& call : split.calls) {
        const auto slot = call.userDataRegister - request.context.userDataBaseRegister;
        pointers.push_back(static_cast<std::uint64_t>(request.context.userData[slot]) | static_cast<std::uint64_t>(request.context.userData[slot + 1u]) << 32u);
    }
    return pointers;
}

std::shared_ptr<const TableCallTargetSet> TableCallProgram::Targets(const ShaderRecompiler::RecompileRequest& request) {
    const auto pointers = tablePointers(request);
    std::lock_guard lock(mutex);
    if (targets != nullptr && targets->pointers == pointers) {
        const bool pending = std::ranges::any_of(targets->reads, [](const GuestMemory::UnchangedQuery& read) { return Graphics::Recorder::SnapshotWriteOverlaps(read.address, read.bytes); });
        if (!pending) {
            for (const auto& read : targets->reads) static_cast<void>(GuestMemory::CollectWrites(read.address, read.bytes));
            if (GuestMemory::UnchangedSinceAll(targets->reads)) return targets;
        }
    }
    auto set = std::make_shared<TableCallTargetSet>();
    set->pointers = pointers;
    set->targets = TableCallTargets(request, split, &set->reads);
    targets = set;
    return set;
}

ShaderRecompiler::RecompileRequest TableCallPartRequest(const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::TableCallSplit& split, TableCallPart& part, ShaderRecompiler::ContinuationAbi::Role role, std::uint32_t entrySite) {
    part.userData.assign(request.context.userData.begin(), request.context.userData.end());
    part.userData.resize(part.userData.size() + ShaderRecompiler::ContinuationAbi::UserDataWords, 0u);
    auto source = request;
    source.shader.code = part.code;
    source.context.userData = part.userData;
    source.shader.linkedLiterals = {};
    source.context.linkedWords = {};
    if (role == ShaderRecompiler::ContinuationAbi::Role::Continuation) {
        auto compute = *request.context.compute;
        compute.numThreads = {request.context.waveSize, 1u, 1u};
        compute.partialThreads = {};
        compute.threadsFromMemory = false;
        source.context.compute = compute;
    }
    source.context.continuation = ShaderRecompiler::ContinuationAbi::Shape{role, part.sites, split.scalarRegisters, split.vectorRegisters, split.ldsDwords, split.userDataScalars, entrySite};
    return source;
}

std::shared_ptr<const ShaderRecompiler::RunTimeProgram> TableCallPartProgram(const ShaderRecompiler::PreparedShaderInvocation& invocation) {
    auto compiled = invocation.MaterializeAtRunTime();
    require(compiled != nullptr, "a table-call program part that reads SRT words, uniform fills, image tables or bound descriptors at compile time is not implemented");
    require(CacheableResult(*compiled), "a table-call program part whose compile poisoned an SRT read or an image table entry is not implemented");
    require(std::ranges::none_of(compiled->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; }), "a table-call program part that binds a flattened SRT is not implemented");
    require(std::ranges::all_of(compiled->imageTableReadRanges, [](const auto& ranges) { return ranges.empty(); }), "a table-call program part that reads image tables on the CPU is not implemented");
    return std::make_shared<const ShaderRecompiler::RunTimeProgram>(ShaderRecompiler::RunTimeProgram{invocation.Plan(), std::move(compiled)});
}

std::shared_ptr<const TableCallPart> PrepareTableCallPart(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::TableCallSplit& split, ShaderRecompiler::SplitCode code, ShaderRecompiler::ContinuationAbi::Role role, std::uint32_t entrySite) {
    auto part = std::make_shared<TableCallPart>();
    part->code = std::move(code.code);
    part->sites = std::move(code.sites);
    const auto source = TableCallPartRequest(request, split, *part, role, entrySite);
    bool preparedAtUse = false;
    const auto invocation = TableCallPartInvocation(snapshot, codeOffset, source, &preparedAtUse);
    if (preparedAtUse) RecordPreparedProgram(device, snapshot, codeOffset, invocation.Request());
    part->program = TableCallPartProgram(invocation);
    return part;
}

std::shared_ptr<const TableCallPart> TableCallProgram::Head(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request) {
    std::lock_guard lock(mutex);
    if (head == nullptr) head = PrepareTableCallPart(device, snapshot, codeOffset, request, split, split.head, ShaderRecompiler::ContinuationAbi::Role::Head, 0u);
    return head;
}

void TableCallProgram::Prepared(std::shared_ptr<const TableCallPart> preparedHead, std::span<const std::shared_ptr<const TableCallPart>> outcomes) {
    const auto targets = ShaderRecompiler::BranchSplitTargets(split);
    if (outcomes.size() != targets.size()) throw std::logic_error("AGC driver: a split program's prepared parts and its outcomes differ in number");
    std::lock_guard lock(mutex);
    head = std::move(preparedHead);
    continuations.clear();
    for (std::size_t index = 0; index < targets.size(); ++index) continuations.push_back({{targets[index].site, targets[index].key[0], {}}, outcomes[index]});
}

std::shared_ptr<const TableCallPart> TableCallProgram::Continuation(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::TableCallTarget& target) {
    const auto outcome = split.branchSites.empty() ? 0u : target.key[0];
    std::lock_guard lock(mutex);
    for (const auto& [key, part] : continuations) {
        if (key.site == target.site && key.outcome == outcome && key.callee == target.callee) return part;
    }
    auto part = PrepareTableCallPart(device, snapshot, codeOffset, request, split, ShaderRecompiler::TableCallContinuation(split, target), ShaderRecompiler::ContinuationAbi::Role::Continuation, target.site);
    continuations.push_back({{target.site, outcome, target.callee}, part});
    return part;
}

std::shared_ptr<TableCallProgram> TableCallProgramFor(std::uint64_t programKey, const ShaderRecompiler::RecompileRequest& request) {
    auto& cache = Programs();
    {
        std::lock_guard lock(cache.mutex);
        if (const auto found = cache.programs.find(programKey); found != cache.programs.end()) {
            for (auto variant = found->second.rbegin(); variant != found->second.rend(); ++variant) {
                if ((*variant)->Current(request.context.userData)) return *variant;
            }
        }
    }
    auto split = SplitTableCallProgram(request);
    if (!split) return nullptr;
    auto callees = LinkedCalleesOf(request, split->userDataCalls, std::move(split->userDataCallees));
    auto program = std::make_shared<TableCallProgram>(std::move(*split), std::move(callees));
    Remember(programKey, program);
    return program;
}

std::shared_ptr<TableCallProgram> BranchSplitProgramFor(std::uint64_t programKey, const ShaderRecompiler::RecompileRequest& request, std::size_t programWords, const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, LinkedCallees callees) {
    std::vector<std::uint32_t> measuredSites;
    std::vector<std::shared_ptr<const TableCallPart>> measured;
    const ShaderRecompiler::BranchSplitPartWords partWords = [&](const ShaderRecompiler::TableCallSplit& candidate, const ShaderRecompiler::SplitCode& code, ShaderRecompiler::ContinuationAbi::Role role, std::uint32_t entrySite) {
        if (candidate.branchSites != measuredSites) {
            measuredSites = candidate.branchSites;
            measured.clear();
        }
        measured.push_back(PrepareTableCallPart(device, snapshot, codeOffset, request, candidate, code, role, entrySite));
        return measured.back()->program->compiled->spirv.size();
    };
    std::optional<ShaderRecompiler::TableCallSplit> split;
    try {
        split = ShaderRecompiler::SplitBranchProgram(request, programWords, callees.code, partWords);
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(std::string("AGC driver: ") + error.what());
    }
    if (!split) return nullptr;
    const bool prepared = split->branchSites == measuredSites && measured.size() == 1u + 2u * measuredSites.size();
    auto program = std::make_shared<TableCallProgram>(std::move(*split), std::move(callees));
    if (prepared) program->Prepared(measured.front(), std::span(measured).subspan(1u));
    Remember(programKey, program);
    auto& counters = Counters();
    std::lock_guard lock(counters.mutex);
    ++counters.counts.programs;
    counters.counts.largestProgramWords = std::max(counters.counts.largestProgramWords, programWords);
    return program;
}

BranchSplitCounts BranchSplitCounters() {
    auto& counters = Counters();
    std::lock_guard lock(counters.mutex);
    return counters.counts;
}

void RecordTableCallDispatch(TableCallProgram& program, const TableCallDispatch& dispatch) {
    auto& device = *dispatch.device;
    const auto& request = *dispatch.request;
    const auto& split = program.Split();
    const auto total = static_cast<std::uint64_t>(dispatch.groups[0]) * dispatch.groups[1] * dispatch.groups[2];
    if (total == 0u) return;
    const auto set = program.Targets(request);
    const auto head = program.Head(dispatch.device, *dispatch.snapshot, dispatch.codeOffset, request);
    const auto bins = static_cast<std::uint32_t>(set->targets.size());
    std::vector<std::uint64_t> roundBins;
    for (const auto sites : split.roundSites) {
        std::uint64_t mask = 0;
        for (std::uint32_t bin = 0; bin < bins; ++bin) {
            if (((sites >> set->targets[bin].site) & 1u) != 0u) mask |= std::uint64_t{1} << bin;
        }
        if (mask == 0u) break;
        roundBins.push_back(mask);
    }
    const auto rounds = static_cast<std::uint32_t>(roundBins.size());
    std::uint64_t liveBins = 0;
    for (const auto mask : roundBins) liveBins |= mask;
    const auto live = [&](std::uint32_t bin) { return ((liveBins >> bin) & 1u) != 0u; };
    std::vector<std::shared_ptr<const TableCallPart>> continuations(bins);
    for (std::uint32_t bin = 0; bin < bins; ++bin) {
        if (live(bin)) continuations[bin] = program.Continuation(dispatch.device, *dispatch.snapshot, dispatch.codeOffset, request, set->targets[bin]);
    }
    const auto waveSize = request.context.waveSize;
    const auto recordDwords = ShaderRecompiler::ContinuationAbi::RecordDwords(static_cast<std::uint32_t>(split.scalarRegisters.size()), static_cast<std::uint32_t>(split.vectorRegisters.size()), waveSize, split.ldsDwords);
    const std::uint64_t fixedBytes = (ShaderRecompiler::ContinuationAbi::CountersDword + static_cast<std::uint64_t>(rounds) * bins * ShaderRecompiler::ContinuationAbi::CounterDwords + 8u) * sizeof(std::uint32_t);
    const std::uint64_t groupBytes = static_cast<std::uint64_t>(split.wavesPerGroup) * (2u * bins + recordDwords) * sizeof(std::uint32_t);
    const std::uint64_t budget = device.ContinuationBlockBudget();
    require(fixedBytes + groupBytes <= budget, "the continuation records of one workgroup of a table-call program exceed the continuation block budget");
    const auto chunkGroups = std::min({total, (budget - fixedBytes) / groupBytes, MaxHeadGroups});
    const ShaderRecompiler::ContinuationAbi::Geometry geometry{bins, rounds, static_cast<std::uint32_t>(chunkGroups * split.wavesPerGroup), recordDwords};
    const auto block = device.ContinuationBlock(static_cast<std::size_t>(ShaderRecompiler::ContinuationAbi::BlockDwords(geometry) * sizeof(std::uint32_t)));
    const auto blockAddress = block->DeviceAddress();
    std::vector<std::uint32_t> words(static_cast<std::size_t>(ShaderRecompiler::ContinuationAbi::ListsDword(geometry)), 0u);
    words[ShaderRecompiler::ContinuationAbi::VersionWord] = ShaderRecompiler::ContinuationAbi::Version;
    words[ShaderRecompiler::ContinuationAbi::BinsWord] = bins;
    words[ShaderRecompiler::ContinuationAbi::RoundsWord] = rounds;
    words[ShaderRecompiler::ContinuationAbi::SlotsWord] = geometry.slots;
    words[ShaderRecompiler::ContinuationAbi::RecordDwordsWord] = recordDwords;
    words[ShaderRecompiler::ContinuationAbi::CountersWord] = ShaderRecompiler::ContinuationAbi::CountersDword;
    words[ShaderRecompiler::ContinuationAbi::ListsWord] = static_cast<std::uint32_t>(ShaderRecompiler::ContinuationAbi::ListsDword(geometry));
    words[ShaderRecompiler::ContinuationAbi::RecordsWord] = static_cast<std::uint32_t>(ShaderRecompiler::ContinuationAbi::RecordsDword(geometry));
    for (std::uint32_t bin = 0; bin < bins; ++bin) {
        const auto& target = set->targets[bin];
        const auto entry = ShaderRecompiler::ContinuationAbi::BinTableDword + bin * ShaderRecompiler::ContinuationAbi::BinEntryDwords;
        words[entry + ShaderRecompiler::ContinuationAbi::BinSiteWord] = target.site;
        words[entry + ShaderRecompiler::ContinuationAbi::BinKeyMaskWord] = split.head.sites.at(target.site).keyMask;
        std::copy(target.key.begin(), target.key.end(), words.begin() + entry + ShaderRecompiler::ContinuationAbi::BinKeyWord);
    }
    for (std::uint32_t round = 1; round <= rounds; ++round) {
        for (std::uint32_t bin = 0; bin < bins; ++bin) {
            const auto counter = static_cast<std::size_t>(ShaderRecompiler::ContinuationAbi::CounterDword(geometry, round, bin));
            words[counter + 1u] = 1u;
            words[counter + 2u] = 1u;
            words[counter + ShaderRecompiler::ContinuationAbi::CounterDispatchedWord] = static_cast<std::uint32_t>((roundBins[round - 1u] >> bin) & 1u);
        }
    }
    std::vector<std::uint32_t> extended(dispatch.userData.begin(), dispatch.userData.end());
    const auto base = extended.size();
    extended.resize(base + ShaderRecompiler::ContinuationAbi::UserDataWords, 0u);
    extended[base + ShaderRecompiler::ContinuationAbi::BlockLowWord] = static_cast<std::uint32_t>(blockAddress);
    extended[base + ShaderRecompiler::ContinuationAbi::BlockHighWord] = static_cast<std::uint32_t>(blockAddress >> 32u);
    extended[base + ShaderRecompiler::ContinuationAbi::GroupsXWord] = dispatch.groups[0];
    extended[base + ShaderRecompiler::ContinuationAbi::GroupsYWord] = dispatch.groups[1];
    const auto chunks = static_cast<std::size_t>((total + chunkGroups - 1u) / chunkGroups);
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> heads(chunks);
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> parts(bins);
    const auto results = [&] {
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            extended[base + ShaderRecompiler::ContinuationAbi::BinWord] = 0u;
            std::fill_n(extended.begin() + static_cast<std::ptrdiff_t>(base + ShaderRecompiler::ContinuationAbi::KeyWord), ShaderRecompiler::ContinuationAbi::KeyWords, 0u);
            extended[base + ShaderRecompiler::ContinuationAbi::FirstGroupWord] = static_cast<std::uint32_t>(chunk * chunkGroups);
            heads[chunk] = dispatch.result(*head->program, extended);
        }
        extended[base + ShaderRecompiler::ContinuationAbi::FirstGroupWord] = 0u;
        for (std::uint32_t bin = 0; bin < bins; ++bin) {
            if (!live(bin)) continue;
            extended[base + ShaderRecompiler::ContinuationAbi::BinWord] = bin;
            std::copy(set->targets[bin].key.begin(), set->targets[bin].key.end(), extended.begin() + static_cast<std::ptrdiff_t>(base + ShaderRecompiler::ContinuationAbi::KeyWord));
            parts[bin] = dispatch.result(*continuations[bin]->program, extended);
        }
    };
    results();
    if (dispatch.settle()) results();
    if (!split.branchSites.empty()) {
        std::size_t largest = heads.front()->spirv.size();
        for (const auto& compiled : parts) {
            if (compiled != nullptr) largest = std::max(largest, compiled->spirv.size());
        }
        auto& counters = Counters();
        std::lock_guard lock(counters.mutex);
        counters.counts.largestPartWords = std::max(counters.counts.largestPartWords, largest);
    }
    for (const auto& compiled : heads) {
        if (device.PrepareComputePipeline(*compiled, dispatch.programAddress)) device.RecordComputePipeline(*compiled);
    }
    for (const auto& compiled : parts) {
        if (compiled != nullptr && device.PrepareComputePipeline(*compiled, dispatch.programAddress)) device.RecordComputePipeline(*compiled);
    }
    auto headPrepared = device.PrepareDispatch(*heads.front());
    std::vector<std::shared_ptr<PreparedDispatch>> partPrepared(bins);
    for (std::uint32_t bin = 0; bin < bins; ++bin) {
        if (parts[bin] != nullptr) partPrepared[bin] = device.PrepareDispatch(*parts[bin]);
    }
    dispatch.locked([&] {
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            const auto first = chunk * chunkGroups;
            const auto count = std::min<std::uint64_t>(chunkGroups, total - first);
            device.InitializeContinuationBlock(block, words);
            device.Dispatch(*heads[chunk], static_cast<std::uint32_t>(count), 1u, 1u, dispatch.programAddress, chunk == 0u ? std::move(headPrepared) : nullptr);
            if (rounds == 0u) continue;
            std::vector<VulkanDevice::ContinuationPart> liveParts;
            for (std::uint32_t bin = 0; bin < bins; ++bin) {
                if (parts[bin] != nullptr) liveParts.push_back({parts[bin].get(), chunk == 0u ? std::move(partPrepared[bin]) : nullptr, bin});
            }
            device.DispatchContinuationRounds(block, liveParts, bins, roundBins, geometry.slots, dispatch.programAddress);
        }
    });
}

}
