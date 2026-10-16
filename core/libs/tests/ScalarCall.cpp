#include "Translation/TranslationContext.hpp"
#include "Translation/InstructionTranslator.hpp"
#include "RdnaDecoder/RdnaScalarOpDecoder.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "RdnaDecoder/RdnaMemoryOpDecoder.hpp"
#include "ControlFlow/CalleeLinker.hpp"
#include "ControlFlow/GraphBuilder.hpp"
#include <array>
#include <cstdio>
#include <algorithm>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ShaderRecompiler;

static void Require(bool value) {
    if (!value) throw std::runtime_error("scalar call regression");
}

template <typename TFunction>
static void ExpectThrow(const char* needle, TFunction fn) {
    try {
        fn();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) == std::string::npos) {
            throw std::runtime_error(std::string("expected ") + needle + ", got " + error.what());
        }
        return;
    }
    throw std::runtime_error(std::string("expected throw containing ") + needle);
}

static RdnaInstruction Call(std::uint32_t pc, std::uint32_t link, std::uint32_t immediate) {
    const std::array<std::uint32_t, 1> code{0xbb000000u | (link << 16u) | (immediate & 0xffffu)};
    return DecodeRdnaSopk(pc, code, 0u);
}

static RdnaInstruction Sop1(std::uint32_t pc, std::uint32_t destination, std::uint32_t op, std::uint32_t source) {
    const std::array<std::uint32_t, 1> code{0xbe800000u | (destination << 16u) | (op << 8u) | source};
    return DecodeRdnaSop1(pc, code, 0u);
}

static RdnaInstruction Sopp(std::uint32_t pc, std::uint32_t op, std::uint32_t immediate = 0u) {
    const std::array<std::uint32_t, 1> code{0xbf800000u | (op << 16u) | immediate};
    return DecodeRdnaSopp(pc, code, 0u);
}

static ControlFlowGraph Build(std::initializer_list<RdnaInstruction> instructions) {
    RdnaProgram program;
    program.instructions.assign(instructions);
    return GraphBuilder{}.Build(program);
}

static void CheckDecode() {
    const auto instruction = Call(0x40u, 8u, 7u);
    Require(instruction.op == RdnaOpcode::SCallB64);
    Require(instruction.family == RdnaInstructionFamily::SOPK);
    Require(instruction.destination.kind == RdnaOperandKind::ScalarRegister && instruction.destination.reg == 8u);
    Require(instruction.dataDwordCount == 2u && instruction.wordCount == 1u);
    Require(instruction.branchTarget == 0x60u);
    Require(Call(0x40u, 8u, 0xfff9u).branchTarget == 0x28u);
    Require(Call(0x20000u, 8u, 0x8000u).branchTarget == 4u);
    Require(Call(0u, 8u, 0x7fffu).branchTarget == 0x20000u);
}

static std::string TranslateDump(const RdnaInstruction& instruction) {
    IrProgram program;
    auto& block = program.CreateBlock();
    program.SetEntryBlock(block);
    program.BlockOrder() = {&block};
    program.Metadata().blockInfo.resize(1);
    TranslationContext context(program, block, 256);
    context.TranslateInstruction(instruction);
    return ProgramToString(program);
}

static void CheckLink() {
    const auto dump = TranslateDump(Call(0x50u, 8u, 2u));
    Require(dump.find("0x0000000000000054") != std::string::npos);
    Require(dump == TranslateDump(Sop1(0x50u, 8u, 0x1fu, 0u)));
}

static void CheckCallAfterEnd() {
    const std::array<std::uint32_t, 5> code{0xbb080001u, 0xbf810000u, 0xbf800000u, 0xbe802008u, 0xbf810000u};
    const auto decoded = RdnaInstructionDecoder{}.Decode(code);
    Require(decoded.instructions.size() == code.size());
    const auto cfg = GraphBuilder{}.Build(decoded);
    const auto& caller = cfg.FindBlockByProgramCounter(0u);
    const auto& callee = cfg.FindBlockByProgramCounter(8u);
    Require(caller.terminator.kind == TerminatorKind::Branch && caller.terminator.trueBlock == callee.id);
    Require(callee.terminator.kind == TerminatorKind::Branch && callee.terminator.trueBlock == cfg.FindBlockByProgramCounter(4u).id);
    ShaderComputeInputInfo computeInfo{};
    TranslateOptions options{};
    options.stage = ShaderStageKind::Compute;
    options.inputInfo.compute = &computeInfo;
    static_cast<void>(InstructionTranslator{}.Translate(decoded, cfg, options));
}

static std::vector<std::uint32_t> UserDataCaller() {
    return {0x7e020280u, 0xbe940383u, 0xbe8e0300u, 0xbe8f0301u, 0xbe8e210eu, 0xbe8e0302u, 0xbe8f0303u, 0xbe8e210eu,
        0x80948114u, 0xbf078014u, 0xbf85fff7u, 0xbf810000u, 0x30306c73u};
}

static std::vector<UserDataCall> UserDataCalls(const std::vector<std::uint32_t>& code, std::uint32_t userDataCount = 8u) {
    const SwappcInfo swappc{false, 0u, userDataCount};
    return FindUserDataCalls(RdnaInstructionDecoder{}.Decode(code), swappc);
}

static void CheckUserDataCalls() {
    const SwappcInfo swappc{false, 0u, 8u};
    const auto caller = UserDataCaller();
    const auto program = RdnaInstructionDecoder{}.Decode(caller);
    const auto calls = FindUserDataCalls(program, swappc);
    Require(calls.size() == 2u);
    Require(calls[0].programCounter == 0x10u && calls[0].linkRegister == 14u && calls[0].userDataRegister == 0u);
    Require(calls[1].programCounter == 0x1cu && calls[1].linkRegister == 14u && calls[1].userDataRegister == 2u);
    ExpectThrow("not statically resolvable", [&] { static_cast<void>(GraphBuilder{}.Build(program, &swappc)); });

    auto overwritten = caller;
    overwritten[1] = 0xbe800380u;
    const auto second = UserDataCalls(overwritten);
    Require(second.size() == 1u && second[0].programCounter == 0x1cu);
    auto entered = caller;
    entered[10] = 0xbf85fff8u;
    const auto afterLabel = UserDataCalls(entered);
    Require(afterLabel.size() == 1u && afterLabel[0].programCounter == 0x1cu);
    auto looped = caller;
    looped[8] = 0xbe820380u;
    const auto beforeWrite = UserDataCalls(looped);
    Require(beforeWrite.size() == 1u && beforeWrite[0].programCounter == 0x10u);
    Require(UserDataCalls({0xbe842104u, 0xbf810000u}, 6u).empty());
    const auto direct = UserDataCalls({0xbe8e2104u, 0xbf810000u}, 6u);
    Require(direct.size() == 1u && direct[0].linkRegister == 14u && direct[0].userDataRegister == 4u);

    const std::vector<std::uint32_t> addFive{0x4a020285u, 0xbe80200eu};
    const std::vector<std::uint32_t> addSixtyFour{0x4a0202c0u, 0xbe80200eu};
    Require(UserDataCalleeLength(addFive, 14u) == 2u);
    Require(!UserDataCalleeLength(std::vector<std::uint32_t>{0x4a020285u}, 14u).has_value());
    ExpectThrow("reads its program counter", [] { static_cast<void>(UserDataCalleeLength(std::vector<std::uint32_t>{0xbe901f00u, 0xbe80200eu}, 14u)); });
    ExpectThrow("makes a call", [] { static_cast<void>(UserDataCalleeLength(std::vector<std::uint32_t>{0xbe92210eu, 0xbe80200eu}, 14u)); });
    ExpectThrow("ends the program", [] { static_cast<void>(UserDataCalleeLength(std::vector<std::uint32_t>{0xbf810000u}, 14u)); });
    ExpectThrow("jumps through another register pair", [] { static_cast<void>(UserDataCalleeLength(std::vector<std::uint32_t>{0xbe802010u}, 14u)); });
    ExpectThrow("before code it branches to", [] { static_cast<void>(UserDataCalleeLength(std::vector<std::uint32_t>{0xbf840001u, 0xbe80200eu, 0x4a020285u, 0xbe80200eu}, 14u)); });

    const std::vector<std::vector<std::uint32_t>> callees{addFive, addSixtyFour};
    const auto linked = LinkUserDataCalls(program, calls, callees);
    Require(linked.size() == 17u && linked[4] == 0xbb0e0007u && linked[7] == 0xbb0e0006u && linked[16] == 0xbf9f0000u);
    Require(std::equal(addFive.begin(), addFive.end(), linked.begin() + 12) && std::equal(addSixtyFour.begin(), addSixtyFour.end(), linked.begin() + 14));
    auto readsPc = caller;
    readsPc[1] = 0xbe941f00u;
    const auto readsPcProgram = RdnaInstructionDecoder{}.Decode(readsPc);
    Require(LinkUserDataCalls(readsPcProgram, FindUserDataCalls(readsPcProgram, swappc), callees).size() == 17u);

    const auto linkedProgram = RdnaInstructionDecoder{}.Decode(linked);
    const auto cfg = GraphBuilder{}.Build(linkedProgram, &swappc);
    const auto block = [&](std::uint32_t programCounter) { return cfg.FindBlockByProgramCounter(programCounter).id; };
    Require(cfg.FindBlockByProgramCounter(0x08u).terminator.trueBlock == block(0x30u));
    Require(cfg.FindBlockByProgramCounter(0x30u).terminator.trueBlock == block(0x14u));
    Require(cfg.FindBlockByProgramCounter(0x14u).terminator.trueBlock == block(0x38u));
    Require(cfg.FindBlockByProgramCounter(0x38u).terminator.trueBlock == block(0x20u));
    ShaderComputeInputInfo computeInfo{};
    TranslateOptions options{};
    options.stage = ShaderStageKind::Compute;
    options.inputInfo.compute = &computeInfo;
    options.userDataCount = 8u;
    static_cast<void>(InstructionTranslator{}.Translate(linkedProgram, cfg, options));

    auto readsLink = linked;
    readsLink[8] = 0xbe94030eu;
    ExpectThrow("escapes the constant-offset call/return model", [&] { static_cast<void>(GraphBuilder{}.Build(RdnaInstructionDecoder{}.Decode(readsLink), &swappc)); });
    auto writesInside = linked;
    writesInside[12] = 0xbe8e0300u;
    ExpectThrow("escapes the constant-offset call/return model", [&] { static_cast<void>(GraphBuilder{}.Build(RdnaInstructionDecoder{}.Decode(writesInside), &swappc)); });
    auto sameCopy = linked;
    sameCopy[7] = 0xbb0e0004u;
    ExpectThrow("is shared with another call", [&] { static_cast<void>(GraphBuilder{}.Build(RdnaInstructionDecoder{}.Decode(sameCopy), &swappc)); });
}

static std::vector<std::uint32_t> TableCaller() {
    return {0x8f148414u, 0xf4080406u, 0xfa000270u, 0xbf8cc07fu, 0xf4280208u, 0x28000000u, 0xbf8cc07fu, 0xbe8e0308u, 0xbe8f0309u, 0xbe8e210eu, 0xbf810000u};
}

static ControlFlowGraph BuildCode(std::span<const std::uint32_t> code, const SwappcInfo& swappc) {
    return GraphBuilder{}.Build(RdnaInstructionDecoder{}.Decode(code), &swappc);
}

static IrProgram TranslateCode(std::span<const std::uint32_t> code, const SwappcInfo& swappc, std::span<const std::uint32_t> linkedLiterals = {}, const ContinuationAbi::Shape* continuation = nullptr) {
    auto program = RdnaInstructionDecoder{}.Decode(code);
    LinkRdnaLiterals(program, linkedLiterals);
    const auto cfg = GraphBuilder{}.Build(program, &swappc);
    ShaderComputeInputInfo computeInfo{};
    TranslateOptions options{};
    options.stage = ShaderStageKind::Compute;
    options.inputInfo.compute = &computeInfo;
    options.userDataCount = swappc.userDataCount;
    options.continuation = continuation;
    return InstructionTranslator{}.Translate(program, cfg, options);
}

static void CheckTableCalls() {
    const SwappcInfo swappc{false, 0u, 16u};
    const auto caller = TableCaller();
    const auto program = RdnaInstructionDecoder{}.Decode(caller);
    const auto calls = FindTableCalls(program, swappc);
    Require(calls.size() == 1u);
    Require(calls[0].programCounter == 0x24u && calls[0].linkRegister == 14u && calls[0].targetRegister == 14u && calls[0].userDataRegister == 12u);
    Require(calls[0].descriptorOffset == 0x270u && calls[0].entryOffset == 0u && calls[0].entryStride == 16u);
    const auto holds = [](const TableCall& call, std::initializer_list<std::pair<std::uint32_t, std::uint32_t>> expected) {
        return std::ranges::equal(call.recordRegisters, expected, [](const TableCallRegister& held, const std::pair<std::uint32_t, std::uint32_t>& pair) { return held.reg == pair.first && held.dword == pair.second; });
    };
    Require(calls[0].recordOffset == 0u && calls[0].recordDwords == 4u && holds(calls[0], {{8u, 0u}, {9u, 1u}, {10u, 2u}, {11u, 3u}}));
    auto moved = caller;
    moved.insert(moved.begin() + 9, 0xbe90040au);
    const auto movedCalls = FindTableCalls(RdnaInstructionDecoder{}.Decode(moved), swappc);
    Require(movedCalls.size() == 1u && holds(movedCalls[0], {{8u, 0u}, {9u, 1u}, {10u, 2u}, {11u, 3u}, {16u, 2u}, {17u, 3u}}));
    Require(FindUserDataCalls(program, swappc).empty());
    const auto table = [&](std::size_t word, std::uint32_t value, std::uint32_t userDataCount = 16u) {
        auto code = caller;
        code[word] = value;
        return FindTableCalls(RdnaInstructionDecoder{}.Decode(code), SwappcInfo{false, 0u, userDataCount});
    };
    Require(table(0u, 0x8f148114u).empty());
    Require(table(2u, 0x28000270u).empty());
    Require(table(8u, 0xbe8f030au).empty());
    Require(table(10u, 0xbf810000u, 12u).empty());
    Require(table(5u, 0x28000008u).size() == 1u && table(5u, 0x28000008u)[0].entryOffset == 8u);
    Require(table(0u, 0xbe940380u).empty());
    auto readsLane = caller;
    readsLane.insert(readsLane.begin(), {0xd4c2000bu, 0x00020300u});
    const auto readsLaneProgram = RdnaInstructionDecoder{}.Decode(readsLane);
    Require(FindTableCalls(readsLaneProgram, SwappcInfo{false, 0u, 16u, 32u}).size() == 1u);
    Require(FindTableCalls(readsLaneProgram, SwappcInfo{false, 0u, 16u, 64u}).empty());

    ExpectThrow("not statically resolvable", [&] { static_cast<void>(BuildCode(caller, swappc)); });
    const std::array<std::uint32_t, 1> listedCall{0x24u};
    SwappcInfo listed = swappc;
    listed.listedCalls = listedCall;
    Require(BuildCode(caller, listed).unlinkedCallProgramCounters == std::vector<std::uint32_t>{0x24u});
    const auto unlinked = TranslateCode(caller, listed);
    Require(unlinked.Info().usesFaultBuffer && ProgramToString(unlinked).find("UnlinkedCall") != std::string::npos);

    const auto linked = LinkCalls(program, {}, {}).code;
    Require(linked.size() == 12u && linked.back() == 0xbf9f0000u && std::equal(caller.begin(), caller.end(), linked.begin()));
    const auto analysis = AnalyzeTableCalls(linked, calls, 32u);
    Require(analysis.rounds == 1u && !analysis.sharedMemory && !analysis.barrier && !analysis.indexedRegisters);
    Require(std::ranges::includes(analysis.scalarRegisters, std::vector<std::uint32_t>{8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u, 20u}) && analysis.vectorRegisters.empty());
    const auto head = LinkTableCallHead(linked, calls);
    Require(head.code.size() == 14u && head.code[9] == 0xbf820001u && head.code[11] == 0xbe8e210eu && head.code[12] == 0xbf810000u && head.code[13] == 0xbf9f0000u);
    Require(head.sites.size() == 1u && head.sites[0].programCounter == 11u * 4u && head.sites[0].callProgramCounter == 0x24u && head.sites[0].keyMask == 0xfu);
    Require(head.sites[0].keyDwords == std::array<std::uint32_t, 4>{0u, 1u, 2u, 3u});
    Require(head.sites[0].recordRegisters == std::array<std::uint32_t, 8>{14u, 0x1000fu, 8u, 0x10009u, 0x2000au, 0x3000bu, ContinuationAbi::NoRecordRegister, ContinuationAbi::NoRecordRegister});
    const std::array<std::uint32_t, 1> headSite{11u * 4u};
    SwappcInfo headSwappc = swappc;
    headSwappc.listedCalls = headSite;
    Require(BuildCode(head.code, headSwappc).unlinkedCallProgramCounters == std::vector<std::uint32_t>{11u * 4u});
    const ContinuationAbi::Shape headShape{ContinuationAbi::Role::Head, head.sites, analysis.scalarRegisters, analysis.vectorRegisters, 0u};
    const auto headWords = ProgramToString(TranslateCode(head.code, headSwappc, {}, &headShape));
    Require(headWords.find("TableCallBin") != std::string::npos && headWords.find("ContinuationSpill") != std::string::npos && headWords.find("UnlinkedCall") == std::string::npos);

    const std::vector<std::uint32_t> first{0xbf068000u, 0x4a020285u, 0xbe80200eu};
    const auto continuation = LinkTableCallContinuation(linked, calls, 0u, first);
    Require(continuation.code.size() == 19u && continuation.code[0] == 0xbf82000au && continuation.code[9] == 0xbf820006u && continuation.code[11] == 0xbb0e0001u && continuation.code[12] == 0xbf82fffdu);
    Require(std::equal(first.begin(), first.end(), continuation.code.begin() + 13) && continuation.code[16] == 0xbe8e210eu && continuation.code[17] == 0xbf810000u && continuation.code[18] == 0xbf9f0000u);
    Require(continuation.sites.size() == 1u && continuation.sites[0].programCounter == 16u * 4u && continuation.sites[0].recordRegisters == head.sites[0].recordRegisters);
    const std::array<std::uint32_t, 1> continuationSite{16u * 4u};
    SwappcInfo continuationSwappc = swappc;
    continuationSwappc.listedCalls = continuationSite;
    const auto cfg = BuildCode(continuation.code, continuationSwappc);
    const auto block = [&](std::uint32_t word) { return cfg.FindBlockByProgramCounter(word * 4u).id; };
    Require(cfg.FindBlockByProgramCounter(0u).terminator.trueBlock == block(11u));
    Require(cfg.FindBlockByProgramCounter(11u * 4u).terminator.trueBlock == block(13u));
    Require(cfg.FindBlockByProgramCounter(13u * 4u).terminator.trueBlock == block(12u));
    Require(cfg.unlinkedCallProgramCounters == std::vector<std::uint32_t>{16u * 4u});
    const std::array<std::uint32_t, 2> userDataScalars{0u, 1u};
    const ContinuationAbi::Shape continuationShape{ContinuationAbi::Role::Continuation, continuation.sites, analysis.scalarRegisters, analysis.vectorRegisters, 0u, userDataScalars, 0u};
    const auto continuationWords = ProgramToString(TranslateCode(continuation.code, continuationSwappc, {}, &continuationShape));
    Require(continuationWords.find("ContinuationSlot") != std::string::npos && continuationWords.find("ContinuationLoad") != std::string::npos && continuationWords.find("GetUserData") != std::string::npos);

    auto looped = caller;
    looped[10] = 0xbf82fff5u;
    looped.push_back(0xbf810000u);
    const auto loopedProgram = RdnaInstructionDecoder{}.Decode(looped);
    const auto loopedCalls = FindTableCalls(loopedProgram, swappc);
    Require(loopedCalls.size() == 1u && AnalyzeTableCalls(LinkCalls(loopedProgram, {}, {}).code, loopedCalls, 32u).rounds == 32u);
    std::vector<std::uint32_t> large(0x8000u, 0xbf800000u);
    large.push_back(0xbe80200eu);
    ExpectThrow("out of the 16-bit branch range", [&] { static_cast<void>(LinkTableCallContinuation(linked, calls, 0u, large)); });
    auto entersFirst = caller;
    entersFirst[0] = 0xbf820000u;
    ExpectThrow("begins with a branch", [&] { static_cast<void>(LinkTableCallContinuation(LinkCalls(RdnaInstructionDecoder{}.Decode(entersFirst), {}, {}).code, calls, 0u, first)); });
    Require(AnalyzeTableCallee(std::vector<std::uint32_t>{0xd8d80000u, 0x01000000u, 0xbe80200eu}).sharedMemory);
    Require(AnalyzeTableCallee(std::vector<std::uint32_t>{0x7e008700u, 0xbe80200eu}).indexedRegisters);

    const std::vector<std::uint32_t> jumpTable{
        0xbe8c1f00u, 0x800cff0cu, 0x00000058u, 0x820d800du, 0x83928202u, 0xd4c40016u, 0x02000500u, 0x8f128212u, 0xf4000386u, 0x24000000u,
        0xbf8cc07fu, 0x7d820003u, 0x808c0e0cu, 0x828d800du, 0xbefd210cu, 0x7e020281u, 0xbf820003u, 0x7e020282u, 0xbf820001u, 0x7e020283u,
        0xe0700000u, 0x80000100u, 0xbf810000u, 0x00000020u, 0x00000018u, 0x00000010u};
    ExpectThrow("jumps through a table in its code", [&] { static_cast<void>(LinkCalls(RdnaInstructionDecoder{}.Decode(jumpTable), {}, {})); });
    const std::array<std::uint32_t, 1> compare{0xbf068008u};
    const auto comparesLink = Build({DecodeRdnaSopc(0u, compare, 0u), Call(4u, 8u, 1u), Sopp(8u, 1u), Sop1(12u, 0u, 0x20u, 8u)});
    Require(comparesLink.FindBlockByProgramCounter(0u).terminator.trueBlock == comparesLink.FindBlockByProgramCounter(12u).id);
    const auto readsFirst = Build({Sop1(0u, 0u, 3u, 8u), Call(4u, 8u, 1u), Sopp(8u, 1u), Sop1(12u, 0u, 0x20u, 8u)});
    Require(readsFirst.FindBlockByProgramCounter(0u).terminator.trueBlock == readsFirst.FindBlockByProgramCounter(12u).id);
    ExpectThrow("escapes the constant-offset call/return model", [] {
        Build({Sop1(0u, 0u, 3u, 8u), Call(4u, 8u, 2u), Sopp(8u, 2u, 0xfffdu), Sopp(12u, 1u), Sop1(16u, 0u, 0x20u, 8u)});
    });
}

struct CalleeRegions {
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions;
    std::vector<std::pair<std::uint64_t, std::size_t>> tables;
};

static bool ReadCalleeRegions(void* context, std::uint64_t address, std::span<std::uint32_t> words) {
    const auto& memory = *static_cast<const CalleeRegions*>(context);
    for (const auto& [base, region] : memory.regions) {
        if (address >= base && (address - base) / 4u + words.size() <= region.size()) {
            std::copy_n(region.begin() + static_cast<std::ptrdiff_t>((address - base) / 4u), words.size(), words.begin());
            return true;
        }
    }
    return false;
}

static void ObserveCalleeTable(void* context, const TableCall&, std::uint64_t base, std::span<const std::uint32_t> table) {
    static_cast<CalleeRegions*>(context)->tables.emplace_back(base, table.size());
}

static void CheckCalleeLinker() {
    constexpr std::uint64_t srt = 0x1100010000ull;
    constexpr std::uint64_t table = 0x1100020000ull;
    constexpr std::uint64_t code = 0x1100030000ull;
    const std::vector<std::uint32_t> first{0xbf068000u, 0x4a020285u, 0xbe80200eu};
    const std::vector<std::uint32_t> second{0x4a0202c0u, 0xbe80200eu};
    std::vector<std::uint32_t> page(1024u, 0u);
    std::copy(first.begin(), first.end(), page.begin());
    std::copy(second.begin(), second.end(), page.begin() + 0x200);
    std::vector<std::uint32_t> descriptors(0x270u / 4u + 4u, 0u);
    const std::array<std::uint32_t, 4> tableDescriptor{static_cast<std::uint32_t>(table), static_cast<std::uint32_t>(table >> 32u) | (16u << 16u), 4u, 0u};
    std::copy(tableDescriptor.begin(), tableDescriptor.end(), descriptors.begin() + 0x270 / 4);
    const auto low = [](std::uint64_t address) { return static_cast<std::uint32_t>(address); };
    const auto high = [](std::uint64_t address) { return static_cast<std::uint32_t>(address >> 32u); };
    const std::vector<std::uint32_t> records{low(code), high(code), 0x00c0ffeeu, 0x12u, low(code), high(code), 0x0badf00du, 0x13u, low(code + 0x800u), high(code + 0x800u), 0x00c0ffeeu, 0x12u, 0u, 0u, 0u, 0u};
    CalleeRegions memory{{{srt, descriptors}, {table, records}, {code, page}}, {}};
    std::array<std::uint32_t, 16> userData{};
    userData[12] = low(srt);
    userData[13] = high(srt);
    const auto caller = TableCaller();
    RecompileRequest request{};
    request.shader.stage = ShaderStage::Compute;
    request.shader.codeAddress = 0x1100040000ull;
    request.shader.code = caller;
    request.context.waveSize = 32u;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u, {}};
    const CalleeMemory callees{&memory, &ReadCalleeRegions, &ReadCalleeRegions, &ObserveCalleeTable};
    ExpectThrow("split into one program per target", [&] { static_cast<void>(LinkCallees(request, callees)); });
    const auto split = SplitTableCalls(request, callees);
    const auto program = RdnaInstructionDecoder{}.Decode(caller);
    const SwappcInfo swappc{false, 0u, 16u, 32u};
    const auto calls = FindTableCalls(program, swappc);
    Require(split.has_value() && split->calls.size() == 1u && split->rounds == 1u && split->wavesPerGroup == 1u && split->ldsDwords == 0u);
    Require(split->head.code == LinkTableCallHead(LinkCalls(program, {}, {}).code, calls).code);
    Require(std::ranges::includes(split->scalarRegisters, std::vector<std::uint32_t>{0u, 1u, 12u, 13u, 16u}) && split->vectorRegisters == std::vector<std::uint32_t>{0u});
    Require(std::ranges::includes(split->userDataScalars, std::vector<std::uint32_t>{0u, 1u, 12u, 13u}) && !std::ranges::binary_search(split->userDataScalars, 14u) && !std::ranges::binary_search(split->userDataScalars, 8u));
    Require(memory.tables.empty());
    const auto targets = ReadTableCallTargets(request, *split, callees);
    Require(targets.size() == 3u && targets[0].site == 0u && targets[0].target == code && targets[0].callee == first && targets[1].target == code && targets[2].target == code + 0x800u && targets[2].callee == second);
    Require(targets[0].key == std::array<std::uint32_t, 4>{low(code), high(code), 0x00c0ffeeu, 0x12u} && targets[1].key == std::array<std::uint32_t, 4>{low(code), high(code), 0x0badf00du, 0x13u});
    Require(memory.tables == std::vector<std::pair<std::uint64_t, std::size_t>>{{table, records.size()}});
    const auto continuation = TableCallContinuation(*split, targets[2]);
    Require(continuation.code == LinkTableCallContinuation(split->linked, split->calls, 0u, second).code);
    const auto shape = TableCallShape(*split, continuation, ContinuationAbi::Role::Continuation, targets[2].site);
    Require(shape.role == ContinuationAbi::Role::Continuation && shape.sites.size() == 1u && shape.scalarRegisters.size() == split->scalarRegisters.size() && shape.userDataScalars.size() == split->userDataScalars.size() && shape.entrySite == 0u);
    CalleeRegions withoutCode{{{srt, descriptors}, {table, records}}, {}};
    ExpectThrow("calls a target loaded from a table at program counter 0x24, but its callee at 0x1100030000 is not readable at 0x1100030000", [&] {
        static_cast<void>(ReadTableCallTargets(request, *split, CalleeMemory{&withoutCode, &ReadCalleeRegions, &ReadCalleeRegions, nullptr}));
    });
    CalleeRegions withoutTable{{{srt, descriptors}, {code, page}}, {}};
    ExpectThrow("its call table at 0x1100020000 + 0x40 is not readable", [&] {
        static_cast<void>(ReadTableCallTargets(request, *split, CalleeMemory{&withoutTable, &ReadCalleeRegions, &ReadCalleeRegions, nullptr}));
    });

    const auto userDataCaller = UserDataCaller();
    const std::array<std::uint32_t, 8> pointers{low(code), high(code), low(code + 0x800u), high(code + 0x800u), 0u, 0u, 0u, 0u};
    request.shader.code = userDataCaller;
    request.context.userData = pointers;
    const auto userDataProgram = RdnaInstructionDecoder{}.Decode(userDataCaller);
    const std::vector<std::vector<std::uint32_t>> userDataCallees{first, second};
    Require(LinkCallees(request, callees).code == LinkUserDataCalls(userDataProgram, FindUserDataCalls(userDataProgram, SwappcInfo{false, 0u, 8u, 32u}), userDataCallees));
    request.shader.stage = ShaderStage::Fragment;
    Require(LinkCallees(request, callees).code.empty());
}

static void CheckRejections() {
    ExpectThrow("invalid instruction boundary", [] {
        Build({Call(0u, 8u, 16u), Sopp(4u, 1u)});
    });
    ExpectThrow("invalid instruction boundary", [] {
        const std::array<std::uint32_t, 7> code{0xbb080002u, 0xbf810000u, 0xbe8003ffu, 0u, 0xbe802008u, 0xbf810000u, 0xbf810000u};
        static_cast<void>(GraphBuilder{}.Build(RdnaInstructionDecoder{}.Decode(code)));
    });
    ExpectThrow("code span bounds", [] {
        const std::array<std::uint32_t, 2> outside{0xbb080010u, 0xbf810000u};
        static_cast<void>(RdnaInstructionDecoder{}.Decode(outside));
    });
    ExpectThrow("code span bounds", [] {
        const std::array<std::uint32_t, 2> outside{0xbb08fffeu, 0xbf810000u};
        static_cast<void>(RdnaInstructionDecoder{}.Decode(outside));
    });
    ExpectThrow("no paired s_setpc_b64 return", [] {
        Build({Call(0u, 8u, 0u), Sopp(4u, 1u)});
    });
    for (const auto link : {9u, 105u, 106u, 125u}) {
        ExpectThrow("ordinary aligned scalar register pair", [=] {
            Build({Call(0u, link, 0u), Sopp(4u, 1u)});
        });
    }
    ExpectThrow("escapes the constant-offset call/return model", [] {
        Build({Call(0u, 8u, 1u), Sopp(4u, 1u), Sop1(8u, 8u, 3u, 128u), Sop1(12u, 0u, 0x20u, 8u)});
    });
    ExpectThrow("recursive scalar call", [] {
        Build({Call(0u, 8u, 0xffffu), Sop1(4u, 0u, 0x20u, 8u), Sopp(8u, 1u)});
    });
    ExpectThrow("is shared with another call", [] {
        Build({Call(0u, 8u, 2u), Call(4u, 8u, 1u), Sopp(8u, 1u), Sop1(12u, 0u, 0x20u, 8u)});
    });
    ExpectThrow("multiple s_setpc_b64 returns", [] {
        Build({Call(0u, 8u, 1u), Sopp(4u, 1u), Sop1(8u, 0u, 0x20u, 8u), Sop1(12u, 0u, 0x20u, 8u)});
    });
    ExpectThrow("control flow enters the call region", [] {
        Build({Call(0u, 8u, 2u), Sopp(4u, 2u, 1u), Sopp(8u, 1u), Sopp(12u, 0u), Sop1(16u, 0u, 0x20u, 8u)});
    });
    ExpectThrow("control flow leaves the call region", [] {
        Build({Call(0u, 8u, 1u), Sopp(4u, 1u), Sopp(8u, 2u, 2u), Sop1(12u, 0u, 0x20u, 8u), Sopp(16u, 1u), Sopp(20u, 1u)});
    });
}

static void CheckGuards(const std::string& only) {
    bool checked = false;
    const auto check = [&](const char* name, const char* needle, auto fn) {
        if (only.empty() || only == name) {
            checked = true;
            ExpectThrow(needle, fn);
        }
    };
    check("tuple", "escapes the constant-offset call/return model", [] {
        auto load = Sopp(8u, 0u);
        load.family = RdnaInstructionFamily::SMEM;
        load.op = RdnaOpcode::SLoadDwordx8;
        load.dataDwordCount = 8u;
        load.destination.kind = RdnaOperandKind::ScalarRegister;
        load.destination.reg = 4u;
        Build({Call(0u, 8u, 1u), Sopp(4u, 1u), load, Sop1(12u, 0u, 0x20u, 8u)});
    });
    check("wrong_return", "escapes the constant-offset call/return model", [] {
        Build({Call(0u, 8u, 1u), Sopp(4u, 1u), Sop1(8u, 0u, 0x20u, 9u)});
    });
    check("fallthrough", "control flow enters the call region", [] {
        Build({Call(0u, 8u, 2u), Sopp(4u, 0u), Sopp(8u, 0u), Sopp(12u, 0u), Sop1(16u, 0u, 0x20u, 8u)});
    });
    check("conditional_fallthrough", "control flow enters the call region", [] {
        Build({Call(0u, 8u, 2u), Sopp(4u, 2u), Sopp(8u, 4u, 2u), Sopp(12u, 0u), Sop1(16u, 0u, 0x20u, 8u), Sopp(20u, 1u)});
    });
    check("return_before_target", "return precedes the call target", [] {
        Build({Call(0u, 8u, 2u), Sopp(4u, 1u), Sop1(8u, 0u, 0x20u, 8u), Sopp(12u, 0u), Sopp(16u, 1u)});
    });
    check("descriptor", "escapes the constant-offset call/return model", [] {
        const std::array<std::uint32_t, 2> code{0xe0381000u, 0x80020401u};
        const auto load = DecodeRdnaMubuf(8u, code, 0u);
        Require(load.source1.reg == 8u);
        Build({Call(0u, 10u, 1u), Sopp(4u, 1u), load, Sop1(16u, 0u, 0x20u, 10u)});
    });
    check("relative_read", "escapes the constant-offset call/return model", [] {
        Build({Call(0u, 8u, 1u), Sopp(4u, 1u), Sop1(8u, 0u, 0x2eu, 0u), Sop1(12u, 0u, 0x20u, 8u)});
    });
    if (only.empty()) {
        for (const auto link : {0u, 104u}) {
            const auto cfg = Build({Call(0u, link, 1u), Sopp(4u, 1u), Sop1(8u, 0u, 0x20u, link)});
            Require(cfg.FindBlockByProgramCounter(0u).terminator.kind == TerminatorKind::Branch);
        }
    }
    Require(checked);
}

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            CheckDecode();
            CheckLink();
            CheckCallAfterEnd();
            CheckRejections();
            CheckUserDataCalls();
            CheckTableCalls();
            CheckCalleeLinker();
            CheckGuards("");
        } else {
            Require(argc == 2);
            CheckGuards(argv[1]);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "scalar call regression: %s\n", error.what());
        return 1;
    }
    return 0;
}
