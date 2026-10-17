#include "ControlFlow/CalleeLinker.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

namespace {

std::string hexText(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::vector<std::uint32_t> readCallee(const CalleeMemory& memory, std::uint64_t address, std::uint32_t linkRegister) {
    constexpr std::uint64_t page = 4096u;
    constexpr std::size_t limitWords = 16384u;
    if (address % 4u != 0u) throw std::runtime_error("its callee address " + hexText(address) + " is not dword aligned");
    std::vector<std::uint32_t> words;
    while (words.size() < limitWords) {
        const auto next = address + words.size() * 4u;
        const auto count = static_cast<std::size_t>((page - next % page) / 4u);
        const auto old = words.size();
        words.resize(old + count);
        if (!memory.readCode(memory.context, next, std::span(words).subspan(old))) throw std::runtime_error("its callee at " + hexText(address) + " is not readable at " + hexText(next));
        if (const auto length = UserDataCalleeLength(words, linkRegister)) {
            words.resize(*length);
            return words;
        }
    }
    throw std::runtime_error("its callee at " + hexText(address) + " does not return within " + std::to_string(limitWords * 4u) + " bytes");
}

std::vector<std::uint32_t> readWords(const CalleeMemory& memory, std::uint64_t address, std::uint64_t count, const char* what) {
    const auto aligned = address & ~std::uint64_t{3u};
    const auto bytes = count * sizeof(std::uint32_t);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(count));
    if (aligned > std::numeric_limits<std::uint64_t>::max() - bytes || !memory.readData(memory.context, aligned, words)) {
        throw std::runtime_error(std::string("its ") + what + " at " + hexText(address) + " + " + hexText(bytes) + " is not readable");
    }
    return words;
}

struct ProgramCalls {
    RdnaProgram program;
    SwappcInfo swappc;
    std::vector<UserDataCall> userDataCalls;
    std::vector<TableCall> tableCalls;
};

std::optional<ProgramCalls> findCalls(const RecompileRequest& request) {
    if (request.shader.stage != ShaderStage::Compute) return std::nullopt;
    const auto code = request.shader.code;
    if (std::none_of(code.begin(), code.end(), [](std::uint32_t word) { return (word & 0xff80ff00u) == 0xbe802100u; })) return std::nullopt;
    ProgramCalls calls{RdnaInstructionDecoder{}.Decode(code), SwappcInfo{false, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size()), request.context.waveSize}, {}, {}};
    calls.userDataCalls = FindUserDataCalls(calls.program, calls.swappc);
    calls.tableCalls = FindTableCalls(calls.program, calls.swappc);
    if (calls.userDataCalls.empty() && calls.tableCalls.empty()) return std::nullopt;
    return calls;
}

LinkedCode linkUserDataCallees(const RecompileRequest& request, const ProgramCalls& calls, const CalleeMemory& memory) {
    std::vector<std::vector<std::uint32_t>> callees;
    for (const auto& call : calls.userDataCalls) {
        const auto slot = call.userDataRegister - calls.swappc.userDataBaseRegister;
        const auto address = static_cast<std::uint64_t>(request.context.userData[slot]) | static_cast<std::uint64_t>(request.context.userData[slot + 1u]) << 32u;
        try {
            callees.push_back(readCallee(memory, address, call.linkRegister));
        } catch (const std::exception& error) {
            throw std::runtime_error("compute shader " + hexText(request.shader.codeAddress) + " calls user data s[" + std::to_string(call.userDataRegister) + ":" +
                std::to_string(call.userDataRegister + 1u) + "] at program counter " + hexText(call.programCounter) + ", but " + error.what());
        }
    }
    auto linked = LinkCalls(calls.program, calls.userDataCalls, callees);
    linked.calls = calls.userDataCalls;
    linked.callees = std::move(callees);
    return linked;
}

std::string tableCallError(const RecompileRequest& request, const TableCall& call, const std::exception& error) {
    return "compute shader " + hexText(request.shader.codeAddress) + " calls a target loaded from a table at program counter " + hexText(call.programCounter) + ", but " + error.what();
}

void requireSplittable(const TableCallAnalysis& analysis, std::uint32_t wavesPerGroup, const char* what) {
    if (analysis.indexedRegisters) throw std::runtime_error(std::string(what) + " indexes registers dynamically (s_movrel or v_movrel), which a continuation's register layout does not cover");
    if (wavesPerGroup > 1u && analysis.sharedMemory) throw std::runtime_error(std::string(what) + " accesses LDS after the call in a workgroup of " + std::to_string(wavesPerGroup) + " waves, which one continuation per wave cannot share");
    if (wavesPerGroup > 1u && analysis.barrier) throw std::runtime_error(std::string(what) + " waits at s_barrier after the call in a workgroup of " + std::to_string(wavesPerGroup) + " waves, which one continuation per wave cannot join");
}

void fillLayout(TableCallSplit& split, const TableCallAnalysis& analysis, const RecompileRequest& request) {
    const auto& compute = *request.context.compute;
    split.rounds = analysis.rounds;
    split.roundSites = analysis.roundSites;
    split.ldsDwords = split.wavesPerGroup == 1u ? compute.ldsSizeDwords : 0u;
    std::set<std::uint32_t> scalars(analysis.scalarRegisters.begin(), analysis.scalarRegisters.end());
    const auto userData = static_cast<std::uint32_t>(request.context.userData.size());
    for (std::uint32_t index = 0; index < userData; ++index) scalars.insert(request.context.userDataBaseRegister + index);
    const auto systemRegisters = static_cast<std::uint32_t>(std::count(compute.groupIdEnable.begin(), compute.groupIdEnable.end(), true)) + (compute.tgSizeEnable ? 1u : 0u);
    for (std::uint32_t index = 0; index < systemRegisters; ++index) scalars.insert(request.context.userDataBaseRegister + userData + index);
    std::erase_if(scalars, [](std::uint32_t reg) { return reg > 105u; });
    split.scalarRegisters.assign(scalars.begin(), scalars.end());
    for (std::uint32_t index = 0; index < userData; ++index) {
        const auto reg = request.context.userDataBaseRegister + index;
        if (reg <= 105u && !std::ranges::binary_search(analysis.writtenScalars, reg)) split.userDataScalars.push_back(reg);
    }
    std::set<std::uint32_t> vectors(analysis.vectorRegisters.begin(), analysis.vectorRegisters.end());
    for (std::uint32_t index = 0; index < std::min(compute.threadIdComponentCount, 3u); ++index) vectors.insert(index);
    split.vectorRegisters.assign(vectors.begin(), vectors.end());
}

}

LinkedCode LinkCallees(const RecompileRequest& request, const CalleeMemory& memory) {
    const auto calls = findCalls(request);
    if (!calls) return {};
    if (!calls->tableCalls.empty()) {
        throw std::runtime_error("compute shader " + hexText(request.shader.codeAddress) + " calls a target loaded from a table at program counter " + hexText(calls->tableCalls.front().programCounter) +
            "; such a program is split into one program per target, not linked");
    }
    return linkUserDataCallees(request, *calls, memory);
}

std::optional<TableCallSplit> SplitTableCalls(const RecompileRequest& request, const CalleeMemory& memory) {
    const auto calls = findCalls(request);
    if (!calls || calls->tableCalls.empty()) return std::nullopt;
    if (!request.context.compute) throw std::runtime_error("compute shader " + hexText(request.shader.codeAddress) + " has table calls but no compute stage information");
    const auto& compute = *request.context.compute;
    TableCallSplit split;
    split.calls = calls->tableCalls;
    auto linked = linkUserDataCallees(request, *calls, memory);
    split.linked = std::move(linked.code);
    split.userDataCalls = std::move(linked.calls);
    split.userDataCallees = std::move(linked.callees);
    const auto waveSize = request.context.waveSize;
    const auto threads = std::max(compute.numThreads[0], 1u) * std::max(compute.numThreads[1], 1u) * std::max(compute.numThreads[2], 1u);
    split.wavesPerGroup = (threads + waveSize - 1u) / waveSize;
    const auto analysis = AnalyzeTableCalls(split.linked, split.calls, waveSize);
    try {
        requireSplittable(analysis, split.wavesPerGroup, "the caller");
    } catch (const std::exception& error) {
        throw std::runtime_error(tableCallError(request, split.calls.front(), error));
    }
    fillLayout(split, analysis, request);
    split.head = LinkTableCallHead(split.linked, split.calls);
    return split;
}

std::vector<TableCallTarget> ReadTableCallTargets(const RecompileRequest& request, const TableCallSplit& split, const CalleeMemory& memory) {
    if (!split.branchSites.empty()) return BranchSplitTargets(split);
    constexpr std::uint64_t limitBytes = 64u << 20u;
    std::vector<TableCallTarget> targets;
    std::map<std::uint64_t, std::vector<std::uint32_t>> codes;
    for (std::uint32_t site = 0; site < split.calls.size(); ++site) {
        const auto& call = split.calls[site];
        try {
            const auto slot = call.userDataRegister - request.context.userDataBaseRegister;
            const auto pointer = static_cast<std::uint64_t>(request.context.userData[slot]) | static_cast<std::uint64_t>(request.context.userData[slot + 1u]) << 32u;
            const auto descriptor = readWords(memory, pointer + call.descriptorOffset, 4u, "call table descriptor");
            const auto stride = (descriptor[1] >> 16u) & 0x3fffu;
            const auto bytes = stride == 0u ? static_cast<std::uint64_t>(descriptor[2]) : static_cast<std::uint64_t>(stride) * descriptor[2];
            const auto base = (static_cast<std::uint64_t>(descriptor[0]) | static_cast<std::uint64_t>(descriptor[1]) << 32u) & 0xffffffffffffull;
            if (bytes > limitBytes) throw std::runtime_error("its call table of " + std::to_string(bytes) + " bytes is larger than the " + std::to_string(limitBytes) + " bytes that are searched for targets");
            const auto table = bytes < sizeof(std::uint32_t) ? std::vector<std::uint32_t>{} : readWords(memory, base, bytes / sizeof(std::uint32_t), "call table");
            if (memory.observeTable != nullptr) memory.observeTable(memory.context, call, base & ~std::uint64_t{3u}, table);
            const auto& binSite = split.head.sites.at(site);
            std::set<std::array<std::uint32_t, ContinuationAbi::KeyWords>> found;
            for (std::uint64_t offset = call.recordOffset; offset + (binSite.keyDwords[1] + 1u) * 4u <= bytes; offset += call.entryStride) {
                std::array<std::uint32_t, ContinuationAbi::KeyWords> key{};
                for (std::uint32_t word = 0; word < ContinuationAbi::KeyWords; ++word) {
                    const auto dword = offset / 4u + binSite.keyDwords[word];
                    if ((binSite.keyMask >> word & 1u) != 0u && binSite.keyDwords[word] != ContinuationAbi::NoRecordRegister && dword < table.size()) key[word] = table[dword];
                }
                if (key[0] != 0u || key[1] != 0u) found.insert(key);
                if (call.entryStride == 0u) break;
            }
            for (const auto& key : found) {
                if (targets.size() == ContinuationAbi::MaxBins) throw std::runtime_error("its call tables hold more than the " + std::to_string(ContinuationAbi::MaxBins) + " distinct records one dispatch bins");
                const auto target = static_cast<std::uint64_t>(key[1]) << 32u | key[0];
                auto code = codes.find(target);
                if (code == codes.end()) {
                    code = codes.emplace(target, readCallee(memory, target, call.linkRegister)).first;
                    const auto callee = AnalyzeTableCallee(code->second);
                    requireSplittable(callee, split.wavesPerGroup, ("its callee at " + hexText(target)).c_str());
                    for (const auto reg : split.userDataScalars) {
                        if (std::ranges::binary_search(callee.writtenScalars, reg)) throw std::runtime_error("its callee at " + hexText(target) + " writes user-data register s" + std::to_string(reg) + ", which a continuation restores from its dispatch's user data");
                    }
                }
                targets.push_back({site, target, code->second, key});
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(tableCallError(request, call, error));
        }
    }
    return targets;
}

std::optional<TableCallSplit> SplitBranchProgram(const RecompileRequest& request, std::size_t programWords, std::span<const std::vector<std::uint32_t>> userDataCallees, const BranchSplitPartWords& partWords, std::size_t boundWords) {
    if (programWords <= GiantProgramWords || request.shader.stage != ShaderStage::Compute || !request.context.compute) return std::nullopt;
    const auto program = RdnaInstructionDecoder{}.Decode(request.shader.code);
    const SwappcInfo swappc{false, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size()), request.context.waveSize};
    auto calls = userDataCallees.empty() ? std::vector<UserDataCall>{} : FindUserDataCalls(program, swappc);
    if (calls.size() != userDataCallees.size()) throw std::logic_error("a split program's user-data calls and linked callees differ in number");
    const auto userDataCall = [&](const RdnaInstruction& instruction) {
        return std::ranges::any_of(calls, [&](const UserDataCall& call) { return call.programCounter == instruction.programCounter; });
    };
    if (std::ranges::any_of(program.instructions, [&](const RdnaInstruction& instruction) { return instruction.op == RdnaOpcode::SSetpcB64 || (instruction.op == RdnaOpcode::SSwappcB64 && !userDataCall(instruction)); })) return std::nullopt;
    const auto programEnd = (program.instructions.back().programCounter / 4u) + program.instructions.back().wordCount;
    TableCallSplit base;
    base.linked = LinkCalls(program, calls, userDataCallees).code;
    base.userDataCalls = std::move(calls);
    base.userDataCallees.assign(userDataCallees.begin(), userDataCallees.end());
    const auto& compute = *request.context.compute;
    const auto waveSize = request.context.waveSize;
    const auto threads = std::max(compute.numThreads[0], 1u) * std::max(compute.numThreads[1], 1u) * std::max(compute.numThreads[2], 1u);
    base.wavesPerGroup = (threads + waveSize - 1u) / waveSize;
    std::optional<TableCallSplit> best;
    std::size_t bestWords = 0;
    std::vector<double> corrections;
    std::vector<std::vector<std::uint32_t>> tried;
    for (std::size_t attempt = 0; attempt < BranchSplitAttempts; ++attempt) {
        auto sites = ChooseBranchSplitSites(base.linked, programWords, boundWords, programEnd, corrections);
        if (sites.empty() || std::ranges::find(tried, sites) != tried.end()) break;
        tried.push_back(sites);
        auto split = base;
        split.branchSites = std::move(sites);
        const auto analysis = AnalyzeBranchSplit(split.linked, split.branchSites);
        if (analysis.indexedRegisters || (split.wavesPerGroup > 1u && (analysis.sharedMemory || analysis.barrier))) break;
        fillLayout(split, analysis, request);
        split.head = LinkBranchSplitHead(split.linked, split.branchSites);
        if (!partWords) return split;
        std::vector<std::size_t> measured{partWords(split, split.head, ContinuationAbi::Role::Head, 0u)};
        for (const auto& target : BranchSplitTargets(split)) measured.push_back(partWords(split, TableCallContinuation(split, target), ContinuationAbi::Role::Continuation, target.site));
        const auto largest = *std::ranges::max_element(measured);
        if (!best || largest < bestWords) {
            best = split;
            bestWords = largest;
        }
        if (largest <= GiantProgramWords) break;
        corrections = CorrectBranchSplitEstimates(split.linked, programWords, split.branchSites, corrections, measured);
    }
    return best;
}

std::vector<TableCallTarget> BranchSplitTargets(const TableCallSplit& split) {
    std::vector<TableCallTarget> targets;
    for (std::uint32_t site = 0; site < split.branchSites.size(); ++site) {
        for (const std::uint32_t taken : {0u, 1u}) targets.push_back({site, 0u, {}, {taken, 0u, 0u, 0u}});
    }
    return targets;
}

SplitCode TableCallContinuation(const TableCallSplit& split, const TableCallTarget& target) {
    if (!split.branchSites.empty()) return LinkBranchSplitContinuation(split.linked, split.branchSites, target.site, target.key[0] != 0u);
    return LinkTableCallContinuation(split.linked, split.calls, target.site, target.callee);
}

ContinuationAbi::Shape TableCallShape(const TableCallSplit& split, const SplitCode& part, ContinuationAbi::Role role, std::uint32_t entrySite) {
    return {role, part.sites, split.scalarRegisters, split.vectorRegisters, split.ldsDwords, split.userDataScalars, entrySite};
}

}
