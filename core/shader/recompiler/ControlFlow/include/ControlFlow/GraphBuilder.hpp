#ifndef CORE_SHADER_RECOMPILIER_CONTROLFLOW_INCLUDE_CONTROLFLOW_GRAPHBUILDER_HPP
#define CORE_SHADER_RECOMPILIER_CONTROLFLOW_INCLUDE_CONTROLFLOW_GRAPHBUILDER_HPP

#include "ContinuationAbi.hpp"
#include "ControlFlow/ControlFlowGraph.hpp"
#include "RdnaDecoder/RdnaProgram.hpp"
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ShaderRecompiler {

struct UserDataTail {
    std::uint32_t programCounter;
    std::uint32_t userDataRegister;
    std::optional<std::uint32_t> constantHigh;
};

[[nodiscard]] std::optional<UserDataTail> FindUserDataTail(const RdnaProgram& program, std::uint32_t userDataBase, std::uint32_t userDataCount);
[[nodiscard]] std::vector<std::uint32_t> LinkUserDataTail(const RdnaProgram& program, const UserDataTail& tail, const RdnaProgram& continuation);

struct SwappcInfo {
    bool fetchCallAllowed = false;
    std::uint32_t userDataBaseRegister = 0;
    std::uint32_t userDataCount = 0;
    std::uint32_t waveSize = 0;
    bool callFunctions = false;
    std::span<const std::uint32_t> listedCalls;
};

struct SwappcCall {
    std::uint32_t callIndex = 0;
    std::uint32_t linkRegister = 0;
    bool fetch = false;
    std::uint32_t targetIndex = 0;
    std::uint32_t targetProgramCounter = 0;
    std::uint32_t returnIndex = 0;
    std::uint32_t returnTargetProgramCounter = 0;
};

struct UserDataCall {
    std::uint32_t programCounter = 0;
    std::uint32_t linkRegister = 0;
    std::uint32_t userDataRegister = 0;
};

struct TableCallRegister {
    std::uint32_t reg = 0;
    std::uint32_t dword = 0;
};

struct TableCall {
    std::uint32_t programCounter = 0;
    std::uint32_t linkRegister = 0;
    std::uint32_t targetRegister = 0;
    std::uint32_t userDataRegister = 0;
    std::uint32_t descriptorOffset = 0;
    std::uint32_t entryOffset = 0;
    std::uint32_t entryStride = 0;
    std::uint32_t recordOffset = 0;
    std::uint32_t recordDwords = 0;
    std::vector<TableCallRegister> recordRegisters;
};

struct LinkedCode {
    std::vector<std::uint32_t> code;
    std::vector<UserDataCall> calls;
    std::vector<std::vector<std::uint32_t>> callees;
};

[[nodiscard]] std::optional<std::uint32_t> UnresolvableSwappcTarget(const RdnaProgram& program, const SwappcInfo* swappc);
[[nodiscard]] std::vector<UserDataCall> FindUserDataCalls(const RdnaProgram& program, const SwappcInfo& swappc);
[[nodiscard]] std::optional<std::uint32_t> UserDataCalleeLength(std::span<const std::uint32_t> callee, std::uint32_t linkRegister);
[[nodiscard]] std::vector<std::uint32_t> LinkUserDataCalls(const RdnaProgram& program, std::span<const UserDataCall> calls, std::span<const std::vector<std::uint32_t>> callees);
[[nodiscard]] std::vector<TableCall> FindTableCalls(const RdnaProgram& program, const SwappcInfo& swappc);
[[nodiscard]] LinkedCode LinkCalls(const RdnaProgram& program, std::span<const UserDataCall> userDataCalls, std::span<const std::vector<std::uint32_t>> userDataCallees);

struct SplitCode {
    std::vector<std::uint32_t> code;
    std::vector<ContinuationAbi::BinSite> sites;
};

struct TableCallAnalysis {
    std::vector<std::uint32_t> scalarRegisters;
    std::vector<std::uint32_t> vectorRegisters;
    std::vector<std::uint32_t> writtenScalars;
    bool sharedMemory = false;
    bool barrier = false;
    bool indexedRegisters = false;
    std::uint32_t rounds = 0;
    std::vector<std::uint64_t> roundSites;
};

[[nodiscard]] SplitCode LinkTableCallHead(std::span<const std::uint32_t> linked, std::span<const TableCall> calls);
[[nodiscard]] SplitCode LinkTableCallContinuation(std::span<const std::uint32_t> linked, std::span<const TableCall> calls, std::size_t site, std::span<const std::uint32_t> callee);
[[nodiscard]] TableCallAnalysis AnalyzeTableCalls(std::span<const std::uint32_t> linked, std::span<const TableCall> calls, std::uint32_t waveSize);
[[nodiscard]] TableCallAnalysis AnalyzeTableCallee(std::span<const std::uint32_t> callee);
[[nodiscard]] std::vector<std::uint32_t> ChooseBranchSplitSites(std::span<const std::uint32_t> code, std::size_t programWords, std::size_t boundWords, std::size_t calleeWord = SIZE_MAX, std::span<const double> corrections = {});
[[nodiscard]] std::vector<double> CorrectBranchSplitEstimates(std::span<const std::uint32_t> code, std::size_t programWords, std::span<const std::uint32_t> sites, std::span<const double> corrections, std::span<const std::size_t> partWords);
[[nodiscard]] SplitCode LinkBranchSplitHead(std::span<const std::uint32_t> linked, std::span<const std::uint32_t> sites);
[[nodiscard]] SplitCode LinkBranchSplitContinuation(std::span<const std::uint32_t> linked, std::span<const std::uint32_t> sites, std::size_t site, bool taken);
[[nodiscard]] TableCallAnalysis AnalyzeBranchSplit(std::span<const std::uint32_t> linked, std::span<const std::uint32_t> sites);

class GraphBuilder {
public:
    [[nodiscard]] ControlFlowGraph Build(const RdnaProgram& program, const SwappcInfo* swappc = nullptr) const;

private:
    [[nodiscard]] std::vector<BasicBlock> splitIntoBlocks(const RdnaProgram& program, const std::vector<SwappcCall>& calls) const;
    void linkBlocks(std::vector<BasicBlock>& blocks, const RdnaProgram& program, const std::vector<SwappcCall>& calls) const;
};

}

#endif
