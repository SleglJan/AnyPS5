#include "Recompiler.hpp"
#include "ControlFlow/CalleeLinker.hpp"
#include "ControlFlow/GraphBuilder.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "execution/BranchSplitProgram.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;
using BranchSplitProgram::Condition;

constexpr std::uint32_t Operations = 600;
constexpr std::uint32_t EndpgmWord = 0xbf810000u;

void Require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::uint32_t BranchTarget(const std::vector<std::uint32_t>& code, std::size_t at) {
    return static_cast<std::uint32_t>(static_cast<std::int64_t>(at) + 1 + static_cast<std::int16_t>(code[at] & 0xffffu));
}

std::size_t SiteOf(const std::vector<std::uint32_t>& code, std::size_t skip = 0) {
    for (std::size_t at = 0; at < code.size(); ++at) {
        if ((code[at] & 0xfffc0000u) == 0xbf840000u && skip-- == 0u) return at;
    }
    throw std::runtime_error("the program has no such conditional branch");
}

std::size_t EntryOf(const std::vector<std::uint32_t>& code, std::size_t last) {
    std::size_t at = 0;
    for (std::size_t hops = 0; at <= last && (code[at] & 0xffff0000u) == 0xbf820000u; ++hops) {
        Require(hops < 8u, "a continuation entry does not reach its outcome");
        at = BranchTarget(code, at);
    }
    return at;
}

struct Atoms {
    std::size_t atomics = 0;
    std::size_t pointers = 0;
};

Atoms Count(std::span<const std::uint32_t> words) {
    Atoms atoms;
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = words[cursor] & 0xffffu;
        if (count == 0 || cursor + count > words.size()) throw std::runtime_error("truncated SPIR-V instruction");
        atoms.atomics += op == spv::OpAtomicIAdd ? 1u : 0u;
        atoms.pointers += op == spv::OpConvertUToPtr ? 1u : 0u;
        cursor += count;
    }
    return atoms;
}

class Source {
public:
    Source(std::vector<std::uint32_t> words, std::uint32_t threads = 64u, std::vector<std::uint32_t> data = {0x10000u, 0x11u}) : code(std::move(words)), userData(std::move(data)) {
        request.shader = {ShaderStage::Compute, 0x1100040000ull, code, 0, {}};
        request.context.waveSize = 64u;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        request.context.compute = ShaderComputeStageInfo{{threads, 1u, 1u}, 0u, {true, false, false}, false, 1u, {}};
        request.target.vulkanVersion = 0x00401000u;
        request.target.spirvVersion = 0x00010300u;
        request.target.subgroupSize = 32;
        request.target.bdaAbiVersion = 1;
        request.target.supportedCapabilities = capabilities;
        request.target.supportedExtensions = extensions;
        request.layout.pushConstantSizeBytes = 128;
        request.useCache = false;
        extended.assign(userData.begin(), userData.end());
        extended.resize(extended.size() + ContinuationAbi::UserDataWords, 0u);
    }

    std::size_t Whole() const {
        return Recompile(request).spirv.size();
    }

    std::vector<std::uint32_t> Part(const TableCallSplit& split, const SplitCode& part, ContinuationAbi::Role role, std::uint32_t entrySite) const {
        auto source = request;
        source.shader.code = part.code;
        source.context.userData = extended;
        if (role == ContinuationAbi::Role::Continuation) {
            auto compute = *request.context.compute;
            compute.numThreads = {request.context.waveSize, 1u, 1u};
            source.context.compute = compute;
        }
        source.context.continuation = TableCallShape(split, part, role, entrySite);
        const auto result = Recompile(source);
        Require(!result.spirv.empty(), "a split part did not recompile");
        return result.spirv.Copy();
    }

    std::vector<std::uint32_t> code;
    std::vector<std::uint32_t> userData;
    std::vector<std::uint32_t> extended;
    RecompileRequest request{};

private:
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    static constexpr std::array<std::uint32_t, 8> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityGroupNonUniform,
        spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformArithmetic};
};

struct Sizes {
    std::size_t whole = 0;
    std::size_t head = 0;
    std::array<std::size_t, 2> outcomes{};
};

Sizes CheckArms(Condition condition, std::uint32_t operations, const std::string& what) {
    const Source source(BranchSplitProgram::Code(condition, operations));
    Sizes sizes;
    const auto& code = source.code;
    const auto site = SiteOf(code, 1u);
    const auto armC = BranchTarget(code, site);
    const auto whole = source.Whole();
    sizes.whole = whole;
    Require(!SplitBranchProgram(source.request, GiantProgramWords).has_value(), what + ": a program at the giant bound was split");
    const auto split = SplitBranchProgram(source.request, GiantProgramWords + 100000u);
    Require(split.has_value(), what + ": a program over the giant bound with three equal arms was not split");
    Require(split->branchSites == std::vector<std::uint32_t>{static_cast<std::uint32_t>(site * 4u)}, what + ": the split point is not the branch between the last two arms");
    Require(split->rounds == 1u && split->roundSites == std::vector<std::uint64_t>{1u}, what + ": one split point does not take one round");
    Require(split->wavesPerGroup == 1u && split->ldsDwords == 0u, what + ": the waves per group or the LDS words are wrong");
    for (const auto reg : {0u, 1u, 2u, 3u, 24u, 27u}) Require(std::ranges::binary_search(split->scalarRegisters, reg), what + ": s" + std::to_string(reg) + " is not in the record");
    for (const auto reg : {0u, 1u, 2u, 3u}) Require(std::ranges::binary_search(split->vectorRegisters, reg), what + ": v" + std::to_string(reg) + " is not in the record");
    Require(split->userDataScalars == std::vector<std::uint32_t>{0u, 1u}, what + ": the SRT pointer is not restored from user data");
    const auto& head = split->head.code;
    Require(head[site] == code[site] && head[site + 1u] == EndpgmWord && head[armC] == EndpgmWord, what + ": the head does not end both outcomes of the split point");
    Require(split->head.sites.size() == 1u && split->head.sites[0].programCounter == site * 4u && split->head.sites[0].callProgramCounter == site * 4u && split->head.sites[0].keyMask == 1u, what + ": the head's bin site is wrong");
    const auto headWords = source.Part(*split, split->head, ContinuationAbi::Role::Head, 0u);
    const auto headAtoms = Count(headWords);
    Require(headAtoms.atomics >= 1u && headAtoms.pointers >= 1u, what + ": the head does not bin its waves");
    sizes.head = headWords.size();
    const auto targets = BranchSplitTargets(*split);
    Require(targets.size() == 2u && targets[0].key[0] == 0u && targets[1].key[0] == 1u, what + ": the targets are not the two outcomes");
    for (const auto& target : targets) {
        const auto part = TableCallContinuation(*split, target);
        Require(EntryOf(part.code, site) == (target.key[0] != 0u ? armC : site + 1u), what + ": a continuation does not enter at its outcome");
        Require(part.code[target.key[0] != 0u ? site + 1u : armC] == code[target.key[0] != 0u ? site + 1u : armC], what + ": a continuation changed the other outcome");
        const auto words = source.Part(*split, part, ContinuationAbi::Role::Continuation, target.site);
        Require(Count(words).pointers >= 1u, what + ": a continuation does not restore its record");
        sizes.outcomes[target.key[0]] = words.size();
    }
    return sizes;
}

void CheckArmGrowth(Condition condition, const std::string& what) {
    const auto small = CheckArms(condition, Operations, what);
    const auto large = CheckArms(condition, Operations * 2u, what);
    const auto grown = large.whole - small.whole;
    std::printf("%s: whole %zu, head %zu, outcomes %zu and %zu SPIR-V words; %.1f words per arm operation\n", what.c_str(), small.whole, small.head, small.outcomes[0], small.outcomes[1], static_cast<double>(grown) / (3.0 * Operations));
    Require(large.head > small.head && (large.head - small.head) * 20u < grown * 9u, what + ": the head does not hold the first arm alone");
    for (std::size_t outcome = 0; outcome < 2u; ++outcome) {
        Require(large.outcomes[outcome] > small.outcomes[outcome] && (large.outcomes[outcome] - small.outcomes[outcome]) * 20u < grown * 9u, what + ": a continuation does not hold its own arm alone");
    }
}

void CheckLongEntry() {
    const Source source(BranchSplitProgram::Code(Condition::Scc, 25000u));
    const auto& code = source.code;
    const auto site = SiteOf(code, 1u);
    const auto armC = BranchTarget(code, site);
    Require(armC > 0x8000u, "the long program's last arm is within one branch of its first instruction");
    const auto split = SplitBranchProgram(source.request, 540000u);
    Require(split.has_value() && split->branchSites == std::vector<std::uint32_t>{static_cast<std::uint32_t>(site * 4u)}, "the long program was not split between its last two arms");
    const auto part = LinkBranchSplitContinuation(split->linked, split->branchSites, 0u, true);
    Require(EntryOf(part.code, site) == armC, "a continuation whose outcome is out of one branch's range does not reach it");
    Require(Count(source.Part(*split, part, ContinuationAbi::Role::Continuation, 0u)).pointers >= 1u, "a continuation entered through its split point's word does not restore its record");
}

void CheckHopOverSite() {
    auto code = BranchSplitProgram::Prologue();
    BranchSplitProgram::AppendArm(code, 0u, 32760u);
    code.push_back(0xbf0d8002u);
    const auto first = code.size();
    Require(first == 0x8000u, "the first split point is not one branch away from the first instruction");
    code.push_back(0u);
    BranchSplitProgram::AppendArm(code, 1u, 10u);
    const auto skip = code.size();
    code.push_back(0u);
    const auto firstTaken = code.size();
    BranchSplitProgram::AppendArm(code, 2u, 10u);
    const auto join = code.size();
    code.insert(code.end(), {0x7e080280u, 0x87058202u, 0x7d8a0805u});
    const auto second = code.size();
    code.push_back(0u);
    BranchSplitProgram::AppendArm(code, 3u, 10u);
    BranchSplitProgram::AppendStore(code);
    const auto secondTaken = code.size();
    BranchSplitProgram::AppendArm(code, 4u, 10u);
    BranchSplitProgram::AppendStore(code);
    code[first] = BranchSplitProgram::Branch(5u, first, firstTaken);
    code[skip] = BranchSplitProgram::Branch(2u, skip, join);
    code[second] = BranchSplitProgram::Branch(7u, second, secondTaken);
    const Source source(code);
    TableCallSplit split;
    split.linked = code;
    split.linked.push_back(0xbf9f0000u);
    split.branchSites = {static_cast<std::uint32_t>(first * 4u), static_cast<std::uint32_t>(second * 4u)};
    const auto analysis = AnalyzeBranchSplit(split.linked, split.branchSites);
    split.scalarRegisters = analysis.scalarRegisters;
    split.vectorRegisters = analysis.vectorRegisters;
    split.userDataScalars = {0u, 1u};
    split.rounds = analysis.rounds;
    split.roundSites = analysis.roundSites;
    const auto part = LinkBranchSplitContinuation(split.linked, split.branchSites, 1u, true);
    Require((part.code[0] & 0xffff0000u) == 0xbf820000u && BranchTarget(part.code, 0u) == first && BranchTarget(part.code, first) == secondTaken, "the continuation's entry does not pass through the first split point's word");
    Require(Count(source.Part(split, part, ContinuationAbi::Role::Continuation, 1u)).pointers >= 1u, "a continuation whose entry passes through another split point's word does not recompile");
}

void CheckJoin() {
    auto code = BranchSplitProgram::Prologue();
    code.push_back(0xbf0d8002u);
    const auto first = code.size();
    code.push_back(0u);
    BranchSplitProgram::AppendArm(code, 0u, 100u);
    const auto firstJoin = code.size();
    BranchSplitProgram::AppendArm(code, 1u, 300u);
    code.push_back(0xbf0d8102u);
    const auto second = code.size();
    code.push_back(0u);
    BranchSplitProgram::AppendArm(code, 2u, 100u);
    const auto secondJoin = code.size();
    BranchSplitProgram::AppendArm(code, 3u, 300u);
    BranchSplitProgram::AppendStore(code);
    code[first] = BranchSplitProgram::Branch(5u, first, firstJoin);
    code[second] = BranchSplitProgram::Branch(5u, second, secondJoin);
    const Source source(code);
    const auto split = SplitBranchProgram(source.request, 700000u);
    Require(split.has_value() && split->branchSites == std::vector<std::uint32_t>{static_cast<std::uint32_t>(second * 4u)}, "an if-then branch whose skipped arm joins its taken outcome was not chosen as the cut point");
    const auto& head = split->head.code;
    Require(head[second] == code[second] && head[second + 1u] == EndpgmWord && head[secondJoin] == EndpgmWord && head[firstJoin] == code[firstJoin], "the head does not end both outcomes of the cut point alone");
    for (const auto taken : {false, true}) {
        const auto part = LinkBranchSplitContinuation(split->linked, split->branchSites, 0u, taken);
        Require(EntryOf(part.code, second) == (taken ? secondJoin : second + 1u), "a continuation of an if-then branch does not enter at its outcome");
        Require(part.code[secondJoin] == code[secondJoin], "a continuation of an if-then branch ended the join its arm reaches");
        Require(Count(source.Part(*split, part, ContinuationAbi::Role::Continuation, 0u)).pointers >= 1u, "a continuation of an if-then branch does not restore its record");
    }
}

void CheckSequence() {
    std::vector<std::uint32_t> code{0xf4080600u, 0xfa000000u, 0xbf8cc07fu, 0x8f038602u, 0x4a020003u, 0x34060282u, 0xbf0d8002u};
    const auto first = code.size();
    code.push_back(0u);
    BranchSplitProgram::AppendArm(code, 0u, 200u);
    const auto skipFirst = code.size();
    code.push_back(0u);
    const auto firstTaken = code.size();
    BranchSplitProgram::AppendArm(code, 1u, 200u);
    const auto join = code.size();
    code.insert(code.end(), {0x7e080280u, 0x87058202u, 0x7d8a0805u});
    const auto second = code.size();
    code.push_back(0u);
    BranchSplitProgram::AppendArm(code, 2u, 200u);
    const auto skipSecond = code.size();
    code.push_back(0u);
    const auto secondTaken = code.size();
    BranchSplitProgram::AppendArm(code, 3u, 200u);
    const auto store = code.size();
    code.insert(code.end(), {0xe0701000u, 0x80060203u, EndpgmWord});
    code[first] = BranchSplitProgram::Branch(5u, first, firstTaken);
    code[skipFirst] = BranchSplitProgram::Branch(2u, skipFirst, join);
    code[second] = BranchSplitProgram::Branch(7u, second, secondTaken);
    code[skipSecond] = BranchSplitProgram::Branch(2u, skipSecond, store);
    const Source source(code);
    const auto split = SplitBranchProgram(source.request, 1000000u);
    Require(split.has_value(), "two switches in sequence were not split");
    Require(split->branchSites == std::vector<std::uint32_t>{static_cast<std::uint32_t>(first * 4u), static_cast<std::uint32_t>(second * 4u)}, "two switches in sequence did not get a split point each");
    Require(split->rounds == 2u && split->roundSites == std::vector<std::uint64_t>{1u, 2u}, "two switches in sequence do not take one round each");
    Require(BranchSplitTargets(*split).size() == 4u, "two split points do not have four outcomes");
    const auto onlySecond = SplitBranchProgram(source.request, 700000u);
    Require(onlySecond.has_value() && onlySecond->branchSites == std::vector<std::uint32_t>{static_cast<std::uint32_t>(second * 4u)}, "the split point that lowers the largest part most was not taken first");
    for (std::uint32_t site = 0; site < 2u; ++site) {
        for (const auto taken : {false, true}) {
            const auto part = LinkBranchSplitContinuation(split->linked, split->branchSites, site, taken);
            const auto words = source.Part(*split, part, ContinuationAbi::Role::Continuation, site);
            Require(!words.empty(), "a continuation of two switches did not recompile");
        }
    }
}

void CheckLoops() {
    std::vector<std::uint32_t> code{0xf4080600u, 0xfa000000u, 0xbf8cc07fu, 0x8f038602u, 0x4a020003u, 0x34060282u};
    const auto loop = code.size();
    code.push_back(0xbf0d8002u);
    const auto inside = code.size();
    code.push_back(0u);
    BranchSplitProgram::AppendArm(code, 0u, 300u);
    const auto skip = code.size();
    BranchSplitProgram::AppendArm(code, 1u, 300u);
    const auto back = code.size();
    code.push_back(0u);
    code.insert(code.end(), {0xe0701000u, 0x80060203u, EndpgmWord});
    code[inside] = BranchSplitProgram::Branch(5u, inside, skip);
    code[back] = BranchSplitProgram::Branch(4u, back, loop);
    const Source source(code);
    Require(ChooseBranchSplitSites(code, 2000000u, GiantPartWords).empty(), "a branch inside a loop was chosen as a split point");
    Require(!SplitBranchProgram(source.request, 2000000u).has_value(), "a program whose only branches are inside a loop was split");
}

std::vector<std::size_t> TwiceSplitSizes(std::uint32_t operations) {
    const Source source(BranchSplitProgram::TreeCode(operations));
    const auto& code = source.code;
    const std::array<std::size_t, 2> sites{SiteOf(code, 1u), SiteOf(code, 2u)};
    std::vector<std::size_t> sizes{source.Whole()};
    const auto split = SplitBranchProgram(source.request, 1500000u);
    Require(split.has_value(), "a program with four arms over the giant bound was not split");
    Require(split->branchSites == std::vector<std::uint32_t>{static_cast<std::uint32_t>(sites[0] * 4u), static_cast<std::uint32_t>(sites[1] * 4u)}, "a program whose halves stay over the bound after one split point was not split again in each half (" + std::to_string(split->branchSites.size()) + " points)");
    Require(split->rounds == 1u && split->roundSites == std::vector<std::uint64_t>{3u}, "two split points the head reaches do not take one round");
    const auto& head = split->head.code;
    for (const auto site : sites) Require(head[site] == code[site] && head[site + 1u] == EndpgmWord && head[BranchTarget(code, site)] == EndpgmWord, "the head does not end both outcomes of each split point");
    sizes.push_back(source.Part(*split, split->head, ContinuationAbi::Role::Head, 0u).size());
    const auto targets = BranchSplitTargets(*split);
    Require(targets.size() == 4u, "two split points do not have four outcomes");
    for (const auto& target : targets) {
        const auto site = sites[target.site];
        const auto arm = target.key[0] != 0u ? BranchTarget(code, site) : site + 1u;
        const auto part = TableCallContinuation(*split, target);
        Require(EntryOf(part.code, site) == arm, "a continuation of a twice split program does not enter at its arm");
        const auto words = source.Part(*split, part, ContinuationAbi::Role::Continuation, target.site);
        Require(Count(words).pointers >= 1u, "a continuation of a twice split program does not restore its record");
        sizes.push_back(words.size());
    }
    return sizes;
}

void CheckTwiceSplit() {
    const auto small = TwiceSplitSizes(Operations);
    const auto large = TwiceSplitSizes(Operations * 2u);
    const auto grown = large[0] - small[0];
    std::printf("twice split: whole %zu, head %zu, arms %zu %zu %zu %zu SPIR-V words\n", small[0], small[1], small[2], small[3], small[4], small[5]);
    Require(large[1] - small[1] < grown / 20u, "the head of a twice split program holds an arm");
    for (std::size_t part = 2; part < small.size(); ++part) {
        Require(large[part] > small[part] && (large[part] - small[part]) * 10u < grown * 3u, "a continuation of a twice split program does not hold its own arm alone");
    }
}

void CheckMeasuredSplit() {
    const Source source(BranchSplitProgram::TreeCode(Operations));
    const auto& code = source.code;
    const std::array<std::size_t, 3> sites{SiteOf(code, 0u), SiteOf(code, 1u), SiteOf(code, 2u)};
    const auto twoArms = source.Whole() * 2u / 3u;
    std::size_t heads = 0;
    const BranchSplitPartWords partWords = [&](const TableCallSplit& split, const SplitCode& part, ContinuationAbi::Role role, std::uint32_t entrySite) {
        const auto words = source.Part(split, part, role, entrySite).size();
        if (role == ContinuationAbi::Role::Head) ++heads;
        return words > twoArms ? GiantProgramWords + 100000u : words;
    };
    Require(SplitBranchProgram(source.request, 700000u)->branchSites == std::vector<std::uint32_t>{static_cast<std::uint32_t>(sites[0] * 4u)}, "the estimate of a program with four arms does not split it once between its halves");
    const auto split = SplitBranchProgram(source.request, 700000u, {}, partWords);
    Require(split.has_value(), "a measured program over the giant bound was not split");
    const auto has = [&](std::size_t site) { return std::ranges::find(split->branchSites, static_cast<std::uint32_t>(site * 4u)) != split->branchSites.end(); };
    Require(has(sites[1]) && has(sites[2]), "parts measured over the giant bound were not split again (" + std::to_string(split->branchSites.size()) + " points)");
    Require(heads >= 2u && heads <= BranchSplitAttempts, "the split was not measured again after a part was over the bound (" + std::to_string(heads) + " measurements)");
    std::size_t always = 0;
    const BranchSplitPartWords over = [&](const TableCallSplit&, const SplitCode&, ContinuationAbi::Role role, std::uint32_t) {
        if (role == ContinuationAbi::Role::Head) ++always;
        return GiantProgramWords + 1u;
    };
    Require(SplitBranchProgram(source.request, 700000u, {}, over).has_value() && always <= BranchSplitAttempts, "a program whose parts stay over the bound was measured more often than the attempts allow");
}

void CheckUserDataCallee() {
    const auto caller = BranchSplitProgram::CallerCode(Operations);
    const auto callee = BranchSplitProgram::CalleeCode(300u);
    const Source source(caller.code, 64u, {0x10000u, 0x11u, 0x20000u, 0x11u});
    const auto& code = source.code;
    const auto site = SiteOf(code, 1u);
    const auto armC = BranchTarget(code, site);
    Require(!SplitBranchProgram(source.request, 1000000u).has_value(), "a program that calls user data was split without its callees");
    const std::vector<std::vector<std::uint32_t>> callees{callee, callee};
    const auto split = SplitBranchProgram(source.request, 1000000u, callees);
    Require(split.has_value(), "a program over the giant bound that calls user data was not split with its callees linked");
    Require(split->branchSites == std::vector<std::uint32_t>{static_cast<std::uint32_t>(site * 4u)}, "a program that calls user data was not split between its last two arms");
    Require(split->userDataCalls.size() == 2u && split->userDataCallees.size() == 2u && split->linked.size() == code.size() + 2u * callee.size() + 1u, "the split program does not carry its two linked callees");
    const auto calls = [&](const std::vector<std::uint32_t>& part, std::size_t call, std::size_t copy) {
        const auto word = caller.calls[call];
        const auto target = BranchTarget(part, word);
        return (part[word] & 0xff800000u) == 0xbb000000u && target == code.size() + copy * callee.size() && std::equal(callee.begin(), callee.end(), part.begin() + static_cast<std::ptrdiff_t>(target));
    };
    Require(calls(split->head.code, 0u, 0u), "the head does not call its linked callee");
    const auto headWords = source.Part(*split, split->head, ContinuationAbi::Role::Head, 0u);
    for (const auto& target : BranchSplitTargets(*split)) {
        const bool taken = target.key[0] != 0u;
        const auto part = TableCallContinuation(*split, target);
        Require(EntryOf(part.code, site) == (taken ? armC : site + 1u), "a continuation of a program that calls user data does not enter at its outcome");
        Require(!taken || calls(part.code, 1u, 1u), "the continuation of the arm that calls user data does not call its linked callee");
        Require(Count(source.Part(*split, part, ContinuationAbi::Role::Continuation, 0u)).pointers >= 1u, "a continuation of a program that calls user data does not restore its record");
    }
    Require(Count(headWords).atomics >= 1u, "the head of a program that calls user data does not bin its waves");
}

void CheckRefusals() {
    const auto code = BranchSplitProgram::Code(Condition::Scc, 60u);
    const auto site = SiteOf(code);
    auto barrier = code;
    barrier.insert(barrier.end() - 3, 0xbf8a0000u);
    Require(SplitBranchProgram(Source(barrier, 64u).request, 1000000u).has_value(), "a one-wave workgroup that waits at s_barrier after the split point was not split");
    Require(!SplitBranchProgram(Source(barrier, 128u).request, 1000000u).has_value(), "a two-wave workgroup that waits at s_barrier after the split point was split");
    auto swappc = code;
    swappc.insert(swappc.begin() + static_cast<std::ptrdiff_t>(site), 0xbe9e2100u);
    Require(!SplitBranchProgram(Source(swappc).request, 1000000u).has_value(), "a program with a computed call was split");
    auto indexed = code;
    indexed.insert(indexed.begin() + 3, 0xbe862e0au);
    const auto decoded = RdnaInstructionDecoder{}.Decode(indexed);
    Require(decoded.instructions.size() > 2u && decoded.instructions[2].op == RdnaOpcode::SMovrelsB32, "the indexed read is not s_movrels_b32");
    const auto split = SplitBranchProgram(Source(indexed).request, 1000000u);
    Require(split.has_value(), "a program that indexes scalar registers was not split");
    Require(split->scalarRegisters.size() == 106u, "a program that indexes scalar registers does not record every scalar register");
}

}

int main() {
    try {
        CheckArmGrowth(Condition::Scc, "scc");
        CheckArmGrowth(Condition::Vcc, "vcc");
        CheckLongEntry();
        CheckHopOverSite();
        CheckJoin();
        CheckSequence();
        CheckLoops();
        CheckTwiceSplit();
        CheckUserDataCallee();
        CheckMeasuredSplit();
        CheckRefusals();
    } catch (const std::exception& error) {
        std::cerr << "BranchSplitParts: " << error.what() << '\n';
        return 1;
    }
    std::puts("branch split part tests passed");
    return 0;
}
