#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PIPELINEREPLAY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PIPELINEREPLAY_HPP

#include "prx/libSceAgcDriver/Execution/include/PipelineManifest.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineDescription.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace ShaderRecompiler {
struct SourceHandle;
}

namespace AgcDriver {

class PipelineReplay final : public Graphics::IGraphicsPipelineRecorder {
public:
    using Target = ShaderRecompiler::SpirvTarget;
    struct Counters {
        std::uint64_t loaded = 0;
        std::uint64_t replayed = 0;
        std::uint64_t recorded = 0;
        std::uint64_t notRecordable = 0;
        std::uint64_t failed = 0;
        std::uint64_t inFrames = 0;
        std::uint64_t programsLoaded = 0;
        std::uint64_t programsAtUse = 0;
        std::uint64_t programsRecorded = 0;
        std::uint64_t programsReplayed = 0;
        std::uint64_t programsStale = 0;
        std::uint64_t programsFailed = 0;
        std::uint64_t used = 0;
        std::uint64_t programsRecompiled = 0;
        std::uint64_t programsDropped = 0;
    };
    enum class ProgramReplay { Replayed, Stale, Failed, Recompiled, Dropped };

    PipelineReplay(PipelineManifest& manifest, std::function<Graphics::Context()> context, std::function<Target()> target);

    std::shared_ptr<RecordUse> Created(const Graphics::Context& context, const Graphics::State& state, const Graphics::VertexInputLayout& vertexInput, const Graphics::PipelineSetLayouts& resources, std::span<const Graphics::CompiledShader> shaders, VkImageLayout attachmentLayout) override;
    std::vector<std::function<void()>> Prepared(const std::shared_ptr<const ShaderRecompiler::SourceHandle>& handle);
    std::vector<std::function<void()>> Prepared(const std::shared_ptr<const ShaderRecompiler::SourceHandle>& handle, const std::array<std::uint64_t, 2>& recordedIdentity);
    void Recompiled(const PreparedProgramRecord& recorded, const ShaderRecompiler::RecompileRequest& request, const std::array<std::uint64_t, 2>& identity);
    void Drop(const PreparedProgramRecord& recorded);
    void RecordPreparedProgram(std::span<const std::uint32_t> registeredCode, std::uint64_t codeOffset, std::span<const std::pair<std::uint64_t, std::uint64_t>> ownRegions, const ShaderRecompiler::RecompileRequest& request);
    std::vector<PreparedProgramRecord> PreparedPrograms(std::span<const std::uint32_t> registeredCode) const;
    static void CountProgramReplay(ProgramReplay outcome);
    static std::array<std::uint64_t, 2> ProgramKey(std::span<const std::uint32_t> registeredCode);
    static Counters Totals();

private:
    struct Pending {
        GraphicsPipelineRecord record;
        std::shared_ptr<RecordUse> use;
        std::vector<std::shared_ptr<const ShaderRecompiler::SourceHandle>> handles;
        std::size_t missing = 0;
    };

    void create(const GraphicsPipelineRecord& record, const std::shared_ptr<RecordUse>& use, const std::vector<std::shared_ptr<const ShaderRecompiler::SourceHandle>>& handles);

    PipelineManifest& manifest;
    std::function<Graphics::Context()> context;
    std::function<Target()> target;
    mutable std::mutex mutex;
    std::vector<Pending> pending;
    std::map<std::array<std::uint64_t, 2>, std::vector<std::pair<std::size_t, std::size_t>>> waiting;
};

}

#endif
