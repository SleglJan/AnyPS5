#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_DISPATCH_TABLECALLDISPATCH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_DISPATCH_TABLECALLDISPATCH_HPP

#include "ControlFlow/CalleeLinker.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace AgcDriver {
class VulkanDevice;
}

namespace AgcDriver::DriverDetail {

struct ShaderSnapshot;

[[nodiscard]] LinkedCallees LinkedCalleesOf(const ShaderRecompiler::RecompileRequest& request, std::span<const ShaderRecompiler::UserDataCall> calls, std::vector<std::vector<std::uint32_t>> callees);

struct TableCallPart {
    std::vector<std::uint32_t> code;
    std::vector<std::uint32_t> userData;
    std::vector<ShaderRecompiler::ContinuationAbi::BinSite> sites;
    std::shared_ptr<const ShaderRecompiler::RunTimeProgram> program;
};

struct TableCallTargetSet {
    std::vector<std::uint64_t> pointers;
    std::vector<GuestMemory::UnchangedQuery> reads;
    std::vector<ShaderRecompiler::TableCallTarget> targets;
};

class TableCallProgram {
public:
    TableCallProgram(ShaderRecompiler::TableCallSplit split, LinkedCallees callees);

    [[nodiscard]] const ShaderRecompiler::TableCallSplit& Split() const { return split; }
    [[nodiscard]] bool Current(std::span<const std::uint32_t> userData) const { return callees.Current(userData); }
    [[nodiscard]] std::shared_ptr<const TableCallTargetSet> Targets(const ShaderRecompiler::RecompileRequest& request);
    [[nodiscard]] std::shared_ptr<const TableCallPart> Head(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request);
    [[nodiscard]] std::shared_ptr<const TableCallPart> Continuation(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::TableCallTarget& target);

private:
    std::shared_ptr<const TableCallPart> prepare(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, ShaderRecompiler::SplitCode code, ShaderRecompiler::ContinuationAbi::Role role, std::uint32_t entrySite);
    std::vector<std::uint64_t> tablePointers(const ShaderRecompiler::RecompileRequest& request) const;

    ShaderRecompiler::TableCallSplit split;
    LinkedCallees callees;
    std::mutex mutex;
    std::shared_ptr<const TableCallPart> head;
    struct ContinuationKey {
        std::uint32_t site = 0;
        std::uint32_t outcome = 0;
        std::vector<std::uint32_t> callee;
    };
    std::vector<std::pair<ContinuationKey, std::shared_ptr<const TableCallPart>>> continuations;
    std::shared_ptr<const TableCallTargetSet> targets;
};

[[nodiscard]] ShaderRecompiler::RecompileRequest TableCallPartRequest(const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::TableCallSplit& split, TableCallPart& part, ShaderRecompiler::ContinuationAbi::Role role, std::uint32_t entrySite);
[[nodiscard]] std::shared_ptr<const ShaderRecompiler::RunTimeProgram> TableCallPartProgram(const ShaderRecompiler::PreparedShaderInvocation& invocation);
[[nodiscard]] std::shared_ptr<TableCallProgram> TableCallProgramFor(std::uint64_t programKey, const ShaderRecompiler::RecompileRequest& request);
[[nodiscard]] std::shared_ptr<TableCallProgram> BranchSplitProgramFor(std::uint64_t programKey, const ShaderRecompiler::RecompileRequest& request, std::size_t programWords);

struct BranchSplitCounts {
    std::uint64_t programs = 0;
    std::size_t largestProgramWords = 0;
    std::size_t largestPartWords = 0;
};
[[nodiscard]] BranchSplitCounts BranchSplitCounters();

struct TableCallDispatch {
    std::shared_ptr<VulkanDevice> device;
    const ShaderSnapshot* snapshot = nullptr;
    std::size_t codeOffset = 0;
    const ShaderRecompiler::RecompileRequest* request = nullptr;
    std::span<const std::uint32_t> userData;
    std::array<std::uint32_t, 3> groups{};
    std::uint64_t programAddress = 0;
    std::function<std::shared_ptr<const ShaderRecompiler::RecompileResult>(const ShaderRecompiler::RunTimeProgram&, std::span<const std::uint32_t>)> result;
    std::function<bool()> settle;
    std::function<void(const std::function<void()>&)> locked;
};

void RecordTableCallDispatch(TableCallProgram& program, const TableCallDispatch& dispatch);

}

#endif
