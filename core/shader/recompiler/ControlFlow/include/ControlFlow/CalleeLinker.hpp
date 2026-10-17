#ifndef CORE_SHADER_RECOMPILIER_CONTROLFLOW_INCLUDE_CONTROLFLOW_CALLEELINKER_HPP
#define CORE_SHADER_RECOMPILIER_CONTROLFLOW_INCLUDE_CONTROLFLOW_CALLEELINKER_HPP

#include "ContinuationAbi.hpp"
#include "ControlFlow/GraphBuilder.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ShaderRecompiler {

struct CalleeMemory {
    void* context = nullptr;
    bool (*readCode)(void* context, std::uint64_t address, std::span<std::uint32_t> words) = nullptr;
    bool (*readData)(void* context, std::uint64_t address, std::span<std::uint32_t> words) = nullptr;
    void (*observeTable)(void* context, const TableCall& call, std::uint64_t base, std::span<const std::uint32_t> table) = nullptr;
};

[[nodiscard]] LinkedCode LinkCallees(const RecompileRequest& request, const CalleeMemory& memory);

struct TableCallSplit {
    std::vector<TableCall> calls;
    std::vector<std::uint32_t> linked;
    SplitCode head;
    std::vector<std::uint32_t> scalarRegisters;
    std::vector<std::uint32_t> vectorRegisters;
    std::uint32_t ldsDwords = 0;
    std::uint32_t rounds = 0;
    std::uint32_t wavesPerGroup = 1;
    std::vector<std::uint32_t> userDataScalars;
    std::vector<std::uint64_t> roundSites;
    std::vector<std::uint32_t> branchSites;
};

struct TableCallTarget {
    std::uint32_t site = 0;
    std::uint64_t target = 0;
    std::vector<std::uint32_t> callee;
    std::array<std::uint32_t, ContinuationAbi::KeyWords> key{};
};

[[nodiscard]] std::optional<TableCallSplit> SplitTableCalls(const RecompileRequest& request, const CalleeMemory& memory);
[[nodiscard]] std::vector<TableCallTarget> ReadTableCallTargets(const RecompileRequest& request, const TableCallSplit& split, const CalleeMemory& memory);
[[nodiscard]] SplitCode TableCallContinuation(const TableCallSplit& split, const TableCallTarget& target);
[[nodiscard]] ContinuationAbi::Shape TableCallShape(const TableCallSplit& split, const SplitCode& part, ContinuationAbi::Role role, std::uint32_t entrySite = 0);
inline constexpr std::size_t GiantProgramWords = 500000;
inline constexpr std::size_t GiantPartWords = 400000;
[[nodiscard]] std::optional<TableCallSplit> SplitBranchProgram(const RecompileRequest& request, std::size_t programWords, std::size_t boundWords = GiantPartWords);
[[nodiscard]] std::vector<TableCallTarget> BranchSplitTargets(const TableCallSplit& split);

}

#endif
