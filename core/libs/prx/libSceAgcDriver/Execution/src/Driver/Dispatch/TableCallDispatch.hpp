#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_DISPATCH_TABLECALLDISPATCH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_DISPATCH_TABLECALLDISPATCH_HPP

#include "ControlFlow/CalleeLinker.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
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
    TableCallProgram(ShaderRecompiler::TableCallSplit split, std::vector<std::uint32_t> calleeWords, std::vector<GuestMemory::UnchangedQuery> calleeReads);

    [[nodiscard]] const ShaderRecompiler::TableCallSplit& Split() const { return split; }
    [[nodiscard]] bool Current(std::span<const std::uint32_t> userData) const;
    [[nodiscard]] std::shared_ptr<const TableCallTargetSet> Targets(const ShaderRecompiler::RecompileRequest& request);
    [[nodiscard]] std::shared_ptr<const TableCallPart> Head(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request);
    [[nodiscard]] std::shared_ptr<const TableCallPart> Continuation(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::TableCallTarget& target);

private:
    std::shared_ptr<const TableCallPart> prepare(const std::shared_ptr<VulkanDevice>& device, const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, ShaderRecompiler::SplitCode code, ShaderRecompiler::ContinuationAbi::Role role, std::uint32_t entrySite);
    std::vector<std::uint64_t> tablePointers(const ShaderRecompiler::RecompileRequest& request) const;

    ShaderRecompiler::TableCallSplit split;
    std::vector<std::uint32_t> calleeWords;
    std::vector<GuestMemory::UnchangedQuery> calleeReads;
    std::mutex mutex;
    std::shared_ptr<const TableCallPart> head;
    std::vector<std::pair<std::pair<std::uint32_t, std::vector<std::uint32_t>>, std::shared_ptr<const TableCallPart>>> continuations;
    std::shared_ptr<const TableCallTargetSet> targets;
};

[[nodiscard]] std::shared_ptr<TableCallProgram> TableCallProgramFor(std::uint64_t programKey, const ShaderRecompiler::RecompileRequest& request);

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
