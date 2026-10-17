#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PIPELINEMANIFEST_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PIPELINEMANIFEST_HPP

#include "PipelineSpecialization.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver {

class RecordUse;

struct ComputePipelineRecord {
    std::array<std::uint64_t, 2> program{};
    std::vector<ShaderRecompiler::PipelineSpecializationConstant> constants;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bindings;

    bool operator==(const ComputePipelineRecord&) const = default;
};

struct GraphicsBindingShape {
    std::uint32_t kind = 0;
    std::uint32_t role = 0;
    std::uint32_t descriptorSet = 0;
    std::uint32_t binding = 0;
    std::uint32_t count = 0;
    std::uint32_t written = 0;

    bool operator==(const GraphicsBindingShape&) const = default;
};

struct GraphicsStageRecord {
    std::uint32_t stage = 0;
    std::uint32_t pushOffset = 0;
    std::array<std::uint64_t, 2> program{};
    std::vector<ShaderRecompiler::PipelineSpecializationConstant> constants;
    std::uint32_t pushBytes = 0;
    std::vector<GraphicsBindingShape> bindings;

    bool operator==(const GraphicsStageRecord&) const = default;
};

struct GraphicsPipelineRecord {
    std::vector<GraphicsStageRecord> stages;
    std::vector<std::byte> description;

    bool operator==(const GraphicsPipelineRecord&) const = default;
};

struct PreparedProgramRecord {
    std::array<std::uint64_t, 2> program{};
    std::uint64_t codeOffset = 0;
    std::array<std::uint64_t, 2> identity{};
    std::string request;

    bool operator==(const PreparedProgramRecord&) const = default;
};

struct ProgramInterface {
    std::uint32_t pushBytes = 0;
    std::vector<GraphicsBindingShape> bindings;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> computeBindings;

    bool operator==(const ProgramInterface&) const = default;
};

struct Rekeyed {
    std::size_t compute = 0;
    std::size_t graphics = 0;
    std::size_t droppedCompute = 0;
    std::size_t droppedGraphics = 0;
};

struct RecordUsage {
    std::uint32_t misses = 0;
    std::uint64_t progress = 0;

    bool operator==(const RecordUsage&) const = default;
};

template <typename TRecord>
struct Recorded {
    TRecord record;
    RecordUsage usage;

    bool operator==(const Recorded&) const = default;
};

struct ManifestRecords {
    std::vector<Recorded<ComputePipelineRecord>> compute;
    std::vector<Recorded<GraphicsPipelineRecord>> graphics;
    std::vector<PreparedProgramRecord> prepared;

    bool operator==(const ManifestRecords&) const = default;
};

class PipelineManifest {
public:
    static constexpr int SaveIntervalSeconds = 10;
    static constexpr std::uint32_t ExpiryRuns = 3;

    struct Added {
        bool added = false;
        std::shared_ptr<RecordUse> use;
    };

    PipelineManifest(std::filesystem::path path, std::uint64_t sourceVersion);
    ~PipelineManifest();
    PipelineManifest(const PipelineManifest&) = delete;
    PipelineManifest& operator=(const PipelineManifest&) = delete;

    Added Record(ComputePipelineRecord record);
    Added Record(GraphicsPipelineRecord record);
    bool Record(PreparedProgramRecord record);
    std::vector<std::pair<ComputePipelineRecord, std::shared_ptr<RecordUse>>> Records(const std::array<std::uint64_t, 2>& program) const;
    std::vector<std::pair<GraphicsPipelineRecord, std::shared_ptr<RecordUse>>> GraphicsRecords() const;
    std::vector<PreparedProgramRecord> PreparedPrograms(const std::array<std::uint64_t, 2>& program) const;
    std::size_t Size() const;
    std::size_t Loaded() const { return loaded.compute.size() + loaded.graphics.size() + loaded.prepared.size(); }
    std::size_t Expired() const { return expired; }
    bool RecordedByAnotherBuild() const { return recordedSource != sourceVersion; }
    Rekeyed Rekey(const std::array<std::uint64_t, 2>& from, const std::array<std::uint64_t, 2>& to, const std::optional<ProgramInterface>& derived);
    void Forget(const PreparedProgramRecord& record);
    void Forget(const GraphicsPipelineRecord& record);
    const ManifestRecords& LoadedRecords() const { return loaded; }
    ManifestRecords Current() const;
    void Save();

    static std::vector<std::byte> Encode(std::uint64_t sourceVersion, const ManifestRecords& records);
    static std::optional<ManifestRecords> Decode(std::span<const std::byte> bytes, std::uint64_t& recordedSource);
    static ManifestRecords Unexpired(ManifestRecords records, std::size_t* dropped = nullptr);

private:
    template <typename TRecord>
    struct Stored {
        TRecord record;
        RecordUsage usage;
        std::shared_ptr<RecordUse> use;
    };

    void run();
    RecordUsage usageNow(const RecordUsage& loaded, const RecordUse& use, std::uint64_t extent) const;

    std::filesystem::path path;
    std::uint64_t sourceVersion;
    std::uint64_t recordedSource = 0;
    ManifestRecords loaded;
    std::size_t expired = 0;
    std::shared_ptr<std::atomic<std::uint64_t>> uses;
    mutable std::mutex mutex;
    std::map<std::array<std::uint64_t, 2>, std::vector<Stored<ComputePipelineRecord>>> compute;
    std::vector<Stored<GraphicsPipelineRecord>> graphics;
    std::vector<PreparedProgramRecord> prepared;
    std::size_t count = 0;
    std::uint64_t savedUses = 0;
    bool dirty = false;
    bool stopping = false;
    std::condition_variable wake;
    std::thread saver;
};

}

#endif
