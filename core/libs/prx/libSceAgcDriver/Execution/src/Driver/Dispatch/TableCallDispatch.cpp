#include "prx/libSceAgcDriver/Execution/src/Driver/Dispatch/TableCallDispatch.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {


constexpr std::size_t ProgramVariants = 8;
constexpr std::uint64_t MaxHeadGroups = 65535;

struct ProgramCache {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<TableCallProgram>>> programs;
};

ProgramCache& Programs() {
    static ProgramCache cache;
    return cache;
}

std::vector<std::uint32_t> UserDataCalleeWords(const ShaderRecompiler::RecompileRequest& request) {
    const auto program = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(request.shader.code);
    const ShaderRecompiler::SwappcInfo swappc{false, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size()), request.context.waveSize};
    std::vector<std::uint32_t> words;
    for (const auto& call : ShaderRecompiler::FindUserDataCalls(program, swappc)) {
        const auto slot = call.userDataRegister - request.context.userDataBaseRegister;
        for (const auto index : {slot, slot + 1u}) words.insert(words.end(), {index, request.context.userData[index]});
    }
    return words;
}

}

TableCallProgram::TableCallProgram(ShaderRecompiler::TableCallSplit split, std::vector<std::uint32_t> calleeWords, std::vector<GuestMemory::UnchangedQuery> calleeReads)
    : split(std::move(split)), calleeWords(std::move(calleeWords)), calleeReads(std::move(calleeReads)) {
}

bool TableCallProgram::Current(std::span<const std::uint32_t> userData) const {
    for (std::size_t index = 0; index + 1u < calleeWords.size(); index += 2u) {
        if (calleeWords[index] >= userData.size() || userData[calleeWords[index]] != calleeWords[index + 1u]) return false;
    }
    for (const auto& read : calleeReads) static_cast<void>(GuestMemory::CollectWrites(read.address, read.bytes));
    return GuestMemory::UnchangedSinceAll(calleeReads);
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

std::shared_ptr<const TableCallPart> TableCallProgram::prepare(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, ShaderRecompiler::SplitCode code, ShaderRecompiler::ContinuationAbi::Role role, std::uint32_t entrySite) {
    auto part = std::make_shared<TableCallPart>();
    part->code = std::move(code.code);
    part->sites = std::move(code.sites);
    part->userData.assign(request.context.userData.begin(), request.context.userData.end());
    part->userData.resize(part->userData.size() + ShaderRecompiler::ContinuationAbi::UserDataWords, 0u);
    auto source = request;
    source.shader.code = part->code;
    source.context.userData = part->userData;
    source.shader.linkedLiterals = {};
    source.context.linkedWords = {};
    if (role == ShaderRecompiler::ContinuationAbi::Role::Continuation) {
        auto compute = *request.context.compute;
        compute.numThreads = {request.context.waveSize, 1u, 1u};
        compute.partialThreads = {};
        compute.threadsFromMemory = false;
        source.context.compute = compute;
    }
    source.context.continuation = ShaderRecompiler::ContinuationAbi::Shape{role, part->sites, split.scalarRegisters, split.vectorRegisters, split.ldsDwords, split.userDataScalars, entrySite};
    bool preparedAtUse = false;
    const auto invocation = TableCallPartInvocation(snapshot, codeOffset, source, &preparedAtUse);
    if (preparedAtUse) RecordPreparedProgram(device, snapshot, codeOffset, invocation.Request());
    auto compiled = invocation.MaterializeAtRunTime();
    require(compiled != nullptr, "a table-call program part that reads SRT words, uniform fills, image tables or bound descriptors at compile time is not implemented");
    require(CacheableResult(*compiled), "a table-call program part whose compile poisoned an SRT read or an image table entry is not implemented");
    require(std::ranges::none_of(compiled->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; }), "a table-call program part that binds a flattened SRT is not implemented");
    require(std::ranges::all_of(compiled->imageTableReadRanges, [](const auto& ranges) { return ranges.empty(); }), "a table-call program part that reads image tables on the CPU is not implemented");
    part->program = std::make_shared<const ShaderRecompiler::RunTimeProgram>(ShaderRecompiler::RunTimeProgram{invocation.Plan(), std::move(compiled)});
    return part;
}

std::shared_ptr<const TableCallPart> TableCallProgram::Head(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request) {
    std::lock_guard lock(mutex);
    if (head == nullptr) head = prepare(device, snapshot, codeOffset, request, split.head, ShaderRecompiler::ContinuationAbi::Role::Head, 0u);
    return head;
}

std::shared_ptr<const TableCallPart> TableCallProgram::Continuation(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::TableCallTarget& target) {
    std::lock_guard lock(mutex);
    for (const auto& [key, part] : continuations) {
        if (key.first == target.site && key.second == target.callee) return part;
    }
    auto part = prepare(device, snapshot, codeOffset, request, ShaderRecompiler::TableCallContinuation(split, target), ShaderRecompiler::ContinuationAbi::Role::Continuation, target.site);
    continuations.push_back({{target.site, target.callee}, part});
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
    std::vector<GuestMemory::UnchangedQuery> reads;
    auto split = SplitTableCallProgram(request, &reads);
    if (!split) return nullptr;
    auto program = std::make_shared<TableCallProgram>(std::move(*split), UserDataCalleeWords(request), std::move(reads));
    std::lock_guard lock(cache.mutex);
    auto& variants = cache.programs[programKey];
    if (variants.size() == ProgramVariants) variants.erase(variants.begin());
    variants.push_back(program);
    return program;
}

void RecordTableCallDispatch(TableCallProgram& program, const TableCallDispatch& dispatch) {
    auto& device = *dispatch.device;
    const auto& request = *dispatch.request;
    const auto& split = program.Split();
    const auto total = static_cast<std::uint64_t>(dispatch.groups[0]) * dispatch.groups[1] * dispatch.groups[2];
    if (total == 0u) return;
    const auto set = program.Targets(request);
    const auto head = program.Head(dispatch.device, *dispatch.snapshot, dispatch.codeOffset, request);
    std::vector<std::shared_ptr<const TableCallPart>> continuations;
    for (const auto& target : set->targets) continuations.push_back(program.Continuation(dispatch.device, *dispatch.snapshot, dispatch.codeOffset, request, target));
    const auto waveSize = request.context.waveSize;
    const auto bins = static_cast<std::uint32_t>(set->targets.size());
    const auto recordDwords = ShaderRecompiler::ContinuationAbi::RecordDwords(static_cast<std::uint32_t>(split.scalarRegisters.size()), static_cast<std::uint32_t>(split.vectorRegisters.size()), waveSize, split.ldsDwords);
    const std::uint64_t fixedBytes = (ShaderRecompiler::ContinuationAbi::CountersDword + static_cast<std::uint64_t>(split.rounds) * bins * ShaderRecompiler::ContinuationAbi::CounterDwords + 8u) * sizeof(std::uint32_t);
    const std::uint64_t groupBytes = static_cast<std::uint64_t>(split.wavesPerGroup) * (2u * bins + recordDwords) * sizeof(std::uint32_t);
    const std::uint64_t budget = device.ContinuationBlockBudget();
    require(fixedBytes + groupBytes <= budget, "the continuation records of one workgroup of a table-call program exceed the continuation block budget");
    const auto chunkGroups = std::min({total, (budget - fixedBytes) / groupBytes, MaxHeadGroups});
    const ShaderRecompiler::ContinuationAbi::Geometry geometry{bins, split.rounds, static_cast<std::uint32_t>(chunkGroups * split.wavesPerGroup), recordDwords};
    const auto block = device.ContinuationBlock(static_cast<std::size_t>(ShaderRecompiler::ContinuationAbi::BlockDwords(geometry) * sizeof(std::uint32_t)));
    const auto blockAddress = block->DeviceAddress();
    std::vector<std::uint32_t> words(static_cast<std::size_t>(ShaderRecompiler::ContinuationAbi::ListsDword(geometry)), 0u);
    words[ShaderRecompiler::ContinuationAbi::VersionWord] = ShaderRecompiler::ContinuationAbi::Version;
    words[ShaderRecompiler::ContinuationAbi::BinsWord] = bins;
    words[ShaderRecompiler::ContinuationAbi::RoundsWord] = split.rounds;
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
    for (std::uint32_t round = 1; round <= split.rounds; ++round) {
        for (std::uint32_t bin = 0; bin < bins; ++bin) {
            const auto counter = static_cast<std::size_t>(ShaderRecompiler::ContinuationAbi::CounterDword(geometry, round, bin));
            words[counter + 1u] = 1u;
            words[counter + 2u] = 1u;
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
            extended[base + ShaderRecompiler::ContinuationAbi::BinWord] = bin;
            std::copy(set->targets[bin].key.begin(), set->targets[bin].key.end(), extended.begin() + static_cast<std::ptrdiff_t>(base + ShaderRecompiler::ContinuationAbi::KeyWord));
            parts[bin] = dispatch.result(*continuations[bin]->program, extended);
        }
    };
    results();
    if (dispatch.settle()) results();
    for (const auto& compiled : heads) {
        if (device.PrepareComputePipeline(*compiled, dispatch.programAddress)) device.RecordComputePipeline(*compiled);
    }
    for (const auto& compiled : parts) {
        if (device.PrepareComputePipeline(*compiled, dispatch.programAddress)) device.RecordComputePipeline(*compiled);
    }
    auto headPrepared = device.PrepareDispatch(*heads.front());
    std::vector<std::shared_ptr<PreparedDispatch>> partPrepared;
    for (const auto& compiled : parts) partPrepared.push_back(device.PrepareDispatch(*compiled));
    dispatch.locked([&] {
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            const auto first = chunk * chunkGroups;
            const auto count = std::min<std::uint64_t>(chunkGroups, total - first);
            device.InitializeContinuationBlock(block, words);
            device.Dispatch(*heads[chunk], static_cast<std::uint32_t>(count), 1u, 1u, dispatch.programAddress, chunk == 0u ? std::move(headPrepared) : nullptr);
            if (bins == 0u) continue;
            std::vector<VulkanDevice::ContinuationPart> rounds;
            for (std::uint32_t bin = 0; bin < bins; ++bin) rounds.push_back({parts[bin].get(), chunk == 0u ? std::move(partPrepared[bin]) : nullptr, bin});
            device.DispatchContinuationRounds(block, rounds, bins, split.rounds, geometry.slots, dispatch.programAddress);
        }
    });
}

}
