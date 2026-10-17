#include "prx/libSceAgcDriver/Execution/include/PipelineManifest.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "ShaderCacheDirectory.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>

namespace AgcDriver {

namespace {

constexpr std::uint64_t Magic = 0x314d504335535041ull;
constexpr std::uint32_t FormatVersion = 5;

class Writer {
public:
    explicit Writer(std::vector<std::byte>& out) : out(out) {}

    template <typename T>
    void Value(T value) {
        const auto bytes = std::as_bytes(std::span(&value, 1));
        out.insert(out.end(), bytes.begin(), bytes.end());
    }

    template <typename T, typename TWrite>
    void List(const std::vector<T>& values, TWrite&& write) {
        Value(static_cast<std::uint32_t>(values.size()));
        for (const auto& value : values) write(value);
    }

    void Constants(const std::vector<ShaderRecompiler::PipelineSpecializationConstant>& constants) {
        List(constants, [&](const ShaderRecompiler::PipelineSpecializationConstant& constant) {
            Value(constant.id);
            Value(constant.value);
        });
    }

    void Usage(const RecordUsage& usage) {
        Value(usage.misses);
        Value(usage.progress);
    }

private:
    std::vector<std::byte>& out;
};

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes(bytes) {}

    template <typename T>
    bool Value(T& value) {
        if (bytes.size() - offset < sizeof(T)) return false;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return true;
    }

    template <typename T, typename TRead>
    bool List(std::vector<T>& values, std::size_t minimumBytes, TRead&& read) {
        std::uint32_t size = 0;
        if (!Value(size) || size > Remaining() / minimumBytes) return false;
        values.resize(size);
        for (auto& value : values) {
            if (!read(value)) return false;
        }
        return true;
    }

    bool Constants(std::vector<ShaderRecompiler::PipelineSpecializationConstant>& constants) {
        return List(constants, 8u, [&](ShaderRecompiler::PipelineSpecializationConstant& constant) { return Value(constant.id) && Value(constant.value); });
    }

    bool Usage(RecordUsage& usage) {
        return Value(usage.misses) && Value(usage.progress);
    }

    std::size_t Remaining() const { return bytes.size() - offset; }

private:
    std::span<const std::byte> bytes;
    std::size_t offset = 0;
};

}

PipelineManifest::PipelineManifest(std::filesystem::path path, std::uint64_t sourceVersion) : path(std::move(path)), sourceVersion(sourceVersion), uses(std::make_shared<std::atomic<std::uint64_t>>(0)) {
    recordedSource = sourceVersion;
    if (this->path.empty()) return;
    std::vector<std::byte> file;
    if (ShaderRecompiler::ReadWholeFile(this->path, file)) {
        if (auto decoded = Decode(file, recordedSource)) {
            loaded = Unexpired(std::move(*decoded), &expired);
            for (const auto& entry : loaded.compute) compute[entry.record.program].push_back({entry.record, entry.usage, std::make_shared<RecordUse>(uses)});
            for (const auto& entry : loaded.graphics) graphics.push_back({entry.record, entry.usage, std::make_shared<RecordUse>(uses)});
            prepared = loaded.prepared;
            count = Loaded();
            dirty = expired != 0 || recordedSource != sourceVersion;
            if (recordedSource != sourceVersion) std::fprintf(stderr, "[pipeline-manifest] %s was recorded by another recompiler build: its programs are prepared again from their requests\n", this->path.string().c_str());
        } else {
            std::fprintf(stderr, "[pipeline-manifest] ignoring %s: it is not a manifest of this build\n", this->path.string().c_str());
        }
    }
    saver = std::thread([this] { run(); });
}

PipelineManifest::~PipelineManifest() {
    if (saver.joinable()) {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        saver.join();
    }
    Save();
}

PipelineManifest::Added PipelineManifest::Record(ComputePipelineRecord record) {
    std::lock_guard lock(mutex);
    auto& program = compute[record.program];
    if (const auto found = std::ranges::find_if(program, [&](const auto& stored) { return stored.record == record; }); found != program.end()) {
        found->use->Mark();
        return {false, found->use};
    }
    auto use = std::make_shared<RecordUse>(uses);
    use->Mark();
    program.push_back({std::move(record), {}, use});
    ++count;
    dirty = true;
    return {true, use};
}

PipelineManifest::Added PipelineManifest::Record(GraphicsPipelineRecord record) {
    std::lock_guard lock(mutex);
    if (const auto found = std::ranges::find_if(graphics, [&](const auto& stored) { return stored.record == record; }); found != graphics.end()) {
        found->use->Mark();
        return {false, found->use};
    }
    auto use = std::make_shared<RecordUse>(uses);
    use->Mark();
    graphics.push_back({std::move(record), {}, use});
    ++count;
    dirty = true;
    return {true, use};
}

bool PipelineManifest::Record(PreparedProgramRecord record) {
    std::lock_guard lock(mutex);
    if (std::ranges::any_of(prepared, [&](const PreparedProgramRecord& existing) { return existing.identity == record.identity && existing.program == record.program && existing.codeOffset == record.codeOffset; })) return false;
    prepared.push_back(std::move(record));
    ++count;
    dirty = true;
    return true;
}

Rekeyed PipelineManifest::Rekey(const std::array<std::uint64_t, 2>& from, const std::array<std::uint64_t, 2>& to, const std::optional<ProgramInterface>& derived) {
    Rekeyed result;
    if (from == to) return result;
    std::lock_guard lock(mutex);
    if (const auto found = compute.find(from); found != compute.end()) {
        auto moved = std::move(found->second);
        compute.erase(found);
        dirty = true;
        if (!derived) {
            result.droppedCompute = moved.size();
            count -= moved.size();
        } else {
            auto& target = compute[to];
            for (auto& stored : moved) {
                stored.record.program = to;
                stored.record.bindings = derived->computeBindings;
                if (std::ranges::none_of(target, [&](const auto& existing) { return existing.record == stored.record; })) {
                    target.push_back(std::move(stored));
                    ++result.compute;
                } else {
                    --count;
                }
            }
        }
    }
    const auto names = [&](const Stored<GraphicsPipelineRecord>& stored) { return std::ranges::any_of(stored.record.stages, [&](const GraphicsStageRecord& stage) { return stage.program == from; }); };
    if (!derived) {
        result.droppedGraphics = static_cast<std::size_t>(std::erase_if(graphics, names));
        count -= result.droppedGraphics;
        dirty = dirty || result.droppedGraphics != 0;
        return result;
    }
    for (auto& stored : graphics) {
        if (!names(stored)) continue;
        for (auto& stage : stored.record.stages) {
            if (stage.program != from) continue;
            stage.program = to;
            stage.pushBytes = derived->pushBytes;
            stage.bindings = derived->bindings;
        }
        ++result.graphics;
        dirty = true;
    }
    return result;
}

void PipelineManifest::Forget(const GraphicsPipelineRecord& record) {
    std::lock_guard lock(mutex);
    const auto removed = std::erase_if(graphics, [&](const Stored<GraphicsPipelineRecord>& stored) { return stored.record == record; });
    if (removed == 0) return;
    count -= removed;
    dirty = true;
}

void PipelineManifest::Forget(const PreparedProgramRecord& record) {
    std::lock_guard lock(mutex);
    const auto removed = std::erase_if(prepared, [&](const PreparedProgramRecord& existing) { return existing == record; });
    if (removed == 0) return;
    count -= removed;
    dirty = true;
}

std::vector<PreparedProgramRecord> PipelineManifest::PreparedPrograms(const std::array<std::uint64_t, 2>& program) const {
    std::lock_guard lock(mutex);
    std::vector<PreparedProgramRecord> result;
    for (const auto& record : prepared) {
        if (record.program == program) result.push_back(record);
    }
    return result;
}

std::vector<std::pair<ComputePipelineRecord, std::shared_ptr<RecordUse>>> PipelineManifest::Records(const std::array<std::uint64_t, 2>& program) const {
    std::lock_guard lock(mutex);
    std::vector<std::pair<ComputePipelineRecord, std::shared_ptr<RecordUse>>> result;
    if (const auto found = compute.find(program); found != compute.end()) {
        for (const auto& stored : found->second) result.emplace_back(stored.record, stored.use);
    }
    return result;
}

std::vector<std::pair<GraphicsPipelineRecord, std::shared_ptr<RecordUse>>> PipelineManifest::GraphicsRecords() const {
    std::lock_guard lock(mutex);
    std::vector<std::pair<GraphicsPipelineRecord, std::shared_ptr<RecordUse>>> result;
    for (const auto& stored : graphics) result.emplace_back(stored.record, stored.use);
    return result;
}

std::size_t PipelineManifest::Size() const {
    std::lock_guard lock(mutex);
    return count;
}

RecordUsage PipelineManifest::usageNow(const RecordUsage& loaded, const RecordUse& use, std::uint64_t extent) const {
    const auto order = use.Order();
    if (order != 0u) return {0u, loaded.progress == 0u ? order : std::min(loaded.progress, order)};
    if (loaded.progress != 0u && extent >= loaded.progress) return {loaded.misses + 1u, loaded.progress};
    return loaded;
}

ManifestRecords PipelineManifest::Current() const {
    std::lock_guard lock(mutex);
    const auto extent = uses->load(std::memory_order_relaxed);
    ManifestRecords all;
    for (const auto& [program, list] : compute) {
        for (const auto& stored : list) all.compute.push_back({stored.record, usageNow(stored.usage, *stored.use, extent)});
    }
    for (const auto& stored : graphics) all.graphics.push_back({stored.record, usageNow(stored.usage, *stored.use, extent)});
    all.prepared = prepared;
    return all;
}

ManifestRecords PipelineManifest::Unexpired(ManifestRecords records, std::size_t* dropped) {
    const auto before = records.compute.size() + records.graphics.size() + records.prepared.size();
    std::erase_if(records.compute, [](const auto& entry) { return entry.usage.misses >= ExpiryRuns; });
    std::erase_if(records.graphics, [](const auto& entry) { return entry.usage.misses >= ExpiryRuns; });
    std::set<std::array<std::uint64_t, 2>> named;
    for (const auto& entry : records.compute) named.insert(entry.record.program);
    for (const auto& entry : records.graphics) {
        for (const auto& stage : entry.record.stages) named.insert(stage.program);
    }
    std::erase_if(records.prepared, [&](const auto& record) { return !named.contains(record.identity); });
    if (dropped != nullptr) *dropped = before - (records.compute.size() + records.graphics.size() + records.prepared.size());
    return records;
}

void PipelineManifest::Save() {
    if (path.empty()) return;
    {
        std::lock_guard lock(mutex);
        const auto extent = uses->load(std::memory_order_relaxed);
        if (!dirty && extent == savedUses) return;
        dirty = false;
        savedUses = extent;
    }
    const auto bytes = Encode(sourceVersion, Current());
    if (!ShaderRecompiler::WriteFileAtomically(path, bytes)) {
        std::fprintf(stderr, "[pipeline-manifest] cannot write %s\n", path.string().c_str());
        std::lock_guard lock(mutex);
        dirty = true;
    }
}

void PipelineManifest::run() {
    std::unique_lock lock(mutex);
    while (!stopping) {
        wake.wait_for(lock, std::chrono::seconds(SaveIntervalSeconds), [&] { return stopping; });
        if (stopping) break;
        lock.unlock();
        Save();
        lock.lock();
    }
}

std::vector<std::byte> PipelineManifest::Encode(std::uint64_t sourceVersion, const ManifestRecords& records) {
    std::vector<std::byte> out;
    Writer writer(out);
    writer.Value(Magic);
    writer.Value(FormatVersion);
    writer.Value(sourceVersion);
    writer.List(records.compute, [&](const Recorded<ComputePipelineRecord>& entry) {
        const auto& record = entry.record;
        writer.Usage(entry.usage);
        writer.Value(record.program[0]);
        writer.Value(record.program[1]);
        writer.Constants(record.constants);
        writer.List(record.bindings, [&](const std::pair<std::uint32_t, std::uint32_t>& binding) {
            writer.Value(binding.first);
            writer.Value(binding.second);
        });
    });
    writer.List(records.graphics, [&](const Recorded<GraphicsPipelineRecord>& entry) {
        const auto& record = entry.record;
        writer.Usage(entry.usage);
        writer.List(record.stages, [&](const GraphicsStageRecord& stage) {
            writer.Value(stage.stage);
            writer.Value(stage.pushOffset);
            writer.Value(stage.program[0]);
            writer.Value(stage.program[1]);
            writer.Constants(stage.constants);
            writer.Value(stage.pushBytes);
            writer.List(stage.bindings, [&](const GraphicsBindingShape& binding) {
                writer.Value(binding.kind);
                writer.Value(binding.role);
                writer.Value(binding.descriptorSet);
                writer.Value(binding.binding);
                writer.Value(binding.count);
            });
        });
        writer.List(record.description, [&](std::byte value) { writer.Value(value); });
    });
    writer.List(records.prepared, [&](const PreparedProgramRecord& record) {
        writer.Value(record.program[0]);
        writer.Value(record.program[1]);
        writer.Value(record.codeOffset);
        writer.Value(record.identity[0]);
        writer.Value(record.identity[1]);
        writer.Value(static_cast<std::uint32_t>(record.request.size()));
        for (const char character : record.request) writer.Value(character);
    });
    return out;
}

std::optional<ManifestRecords> PipelineManifest::Decode(std::span<const std::byte> bytes, std::uint64_t& recordedSource) {
    Reader reader(bytes);
    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    std::uint64_t source = 0;
    if (!reader.Value(magic) || !reader.Value(version) || !reader.Value(source)) return std::nullopt;
    if (magic != Magic || version != FormatVersion) return std::nullopt;
    recordedSource = source;
    ManifestRecords records;
    const bool compute = reader.List(records.compute, 36u, [&](Recorded<ComputePipelineRecord>& entry) {
        auto& record = entry.record;
        return reader.Usage(entry.usage) && reader.Value(record.program[0]) && reader.Value(record.program[1]) && reader.Constants(record.constants)
            && reader.List(record.bindings, 8u, [&](std::pair<std::uint32_t, std::uint32_t>& binding) { return reader.Value(binding.first) && reader.Value(binding.second); });
    });
    const bool graphics = compute && reader.List(records.graphics, 20u, [&](Recorded<GraphicsPipelineRecord>& entry) {
        auto& record = entry.record;
        return reader.Usage(entry.usage) && reader.List(record.stages, 36u, [&](GraphicsStageRecord& stage) {
            return reader.Value(stage.stage) && reader.Value(stage.pushOffset) && reader.Value(stage.program[0]) && reader.Value(stage.program[1]) && reader.Constants(stage.constants) && reader.Value(stage.pushBytes)
                && reader.List(stage.bindings, 20u, [&](GraphicsBindingShape& binding) {
                       return reader.Value(binding.kind) && reader.Value(binding.role) && reader.Value(binding.descriptorSet) && reader.Value(binding.binding) && reader.Value(binding.count);
                   });
        }) && reader.List(record.description, 1u, [&](std::byte& value) { return reader.Value(value); });
    });
    const bool prepared = graphics && reader.List(records.prepared, 44u, [&](PreparedProgramRecord& record) {
        std::uint32_t size = 0;
        if (!reader.Value(record.program[0]) || !reader.Value(record.program[1]) || !reader.Value(record.codeOffset) || !reader.Value(record.identity[0]) || !reader.Value(record.identity[1]) || !reader.Value(size) || size > reader.Remaining()) return false;
        record.request.resize(size);
        for (auto& character : record.request) {
            if (!reader.Value(character)) return false;
        }
        return true;
    });
    if (!prepared || reader.Remaining() != 0) return std::nullopt;
    return records;
}

}
