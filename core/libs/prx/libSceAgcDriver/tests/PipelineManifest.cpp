#include "prx/libSceAgcDriver/Execution/include/PipelineManifest.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "ShaderCacheDirectory.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<AgcDriver::ComputePipelineRecord> ComputeRecords() {
    AgcDriver::ComputePipelineRecord first;
    first.program = {0x1111222233334444ull, 0x5555666677778888ull};
    first.constants = {{16777219u, 0u}, {8188u, 0u}};
    first.bindings = {{3u, 1u}, {7u, 1u}};
    AgcDriver::ComputePipelineRecord second = first;
    second.constants[0].value = 2u;
    AgcDriver::ComputePipelineRecord third;
    third.program = {1u, 2u};
    return {first, second, third};
}

std::vector<AgcDriver::GraphicsPipelineRecord> GraphicsRecords() {
    AgcDriver::GraphicsPipelineRecord first;
    first.stages.push_back({1u, 0u, {0xaaaau, 0xbbbbu}, {{4096u + 5u, 2u}, {8188u, 4u}}, 64u, {{1u, 7u, 0u, 3u, 1u}}});
    first.stages.push_back({5u, 128u, {0xccccu, 0xddddu}, {{8224u, 1u}}, 0u, {{1u, 0u, 1u, 0u, 4u}, {8u, 2u, 1u, 9u, 1u}}});
    first.description = {std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0x37}};
    auto second = first;
    second.description.back() = std::byte{0x38};
    return {first, second};
}

std::vector<AgcDriver::PreparedProgramRecord> PreparedRecords() {
    AgcDriver::PreparedProgramRecord first{{7u, 8u}, 64u, {0x1111222233334444ull, 0x5555666677778888ull}, "stage 0 code AQID"};
    auto second = first;
    second.codeOffset = 128u;
    second.identity = {0xaaaau, 0xbbbbu};
    AgcDriver::PreparedProgramRecord other{{9u, 9u}, 0u, {5u, 6u}, ""};
    return {first, second, other};
}

std::vector<AgcDriver::BranchSplitRecord> SplitRecords() {
    AgcDriver::BranchSplitRecord named{{7u, 8u}, 0u, {3u, 4u}, {0xaaaau, 0xbbbbu}, {12u, 40u}};
    AgcDriver::BranchSplitRecord unnamed{{7u, 8u}, 64u, {5u, 6u}, {1u, 2u}, {9u}};
    return {named, unnamed};
}

template <typename TRecord>
std::vector<AgcDriver::Recorded<TRecord>> WithUsage(const std::vector<TRecord>& records, std::uint32_t misses) {
    std::vector<AgcDriver::Recorded<TRecord>> result;
    for (std::size_t index = 0; index < records.size(); ++index) result.push_back({records[index], {misses, index + 1u}});
    return result;
}

AgcDriver::ManifestRecords Records() {
    return {WithUsage(ComputeRecords(), 1u), WithUsage(GraphicsRecords(), 2u), PreparedRecords(), SplitRecords()};
}

void Codec() {
    const auto records = Records();
    const auto bytes = AgcDriver::PipelineManifest::Encode(77u, records);
    std::uint64_t recordedSource = 0;
    const auto decoded = AgcDriver::PipelineManifest::Decode(bytes, recordedSource);
    Require(decoded.has_value() && *decoded == records, "a manifest did not decode to the records it was encoded from");
    Require(recordedSource == 77u, "the manifest did not report the recompiler build that recorded it");
    std::uint64_t otherSource = 0;
    const auto other = AgcDriver::PipelineManifest::Decode(AgcDriver::PipelineManifest::Encode(78u, records), otherSource);
    Require(other.has_value() && *other == records && otherSource == 78u, "a manifest of another recompiler build was not read with its records and its build");
    for (std::size_t size = 0; size < bytes.size(); ++size) Require(!AgcDriver::PipelineManifest::Decode(std::span(bytes).first(size), recordedSource).has_value(), "a manifest cut to " + std::to_string(size) + " bytes was accepted");
    auto longer = bytes;
    longer.push_back(std::byte{0});
    Require(!AgcDriver::PipelineManifest::Decode(longer, recordedSource).has_value(), "a manifest with trailing bytes was accepted");
    auto corrupt = bytes;
    corrupt[52] = std::byte{0xff};
    corrupt[53] = std::byte{0xff};
    corrupt[54] = std::byte{0xff};
    corrupt[55] = std::byte{0x7f};
    Require(!AgcDriver::PipelineManifest::Decode(corrupt, recordedSource).has_value(), "a manifest whose constant count exceeds its bytes was accepted");
    auto older = bytes;
    older[8] = std::byte{3};
    Require(!AgcDriver::PipelineManifest::Decode(older, recordedSource).has_value(), "a manifest of an earlier format was accepted");
}

std::filesystem::path Directory() {
    const auto directory = std::filesystem::temp_directory_path() / ("aps5-manifest-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    return directory;
}

void File() {
    const auto directory = Directory();
    const auto path = directory / "pipelines.manifest";
    const auto compute = ComputeRecords();
    const auto graphics = GraphicsRecords();
    const auto prepared = PreparedRecords();
    const auto splits = SplitRecords();
    {
        AgcDriver::PipelineManifest manifest(path, 77u);
        Require(manifest.Loaded() == 0u && manifest.Size() == 0u, "a manifest without a file has records");
        for (const auto& record : compute) Require(manifest.Record(record).added, "a new compute record was refused");
        for (const auto& record : graphics) Require(manifest.Record(record).added, "a new graphics record was refused");
        for (const auto& record : prepared) Require(manifest.Record(record), "a new record of a program prepared at use was refused");
        for (const auto& record : splits) Require(manifest.Record(record), "a new record of a split program was refused");
        Require(!manifest.Record(compute[1]).added && !manifest.Record(graphics[1]).added && !manifest.Record(prepared[1]) && !manifest.Record(splits[1]), "a record seen before was recorded again");
        Require(manifest.Records(compute[0].program).size() == 2u && manifest.Records(compute[2].program).size() == 1u && manifest.Records({9u, 9u}).empty(), "the records of a program are not the records recorded for it");
    }
    {
        AgcDriver::PipelineManifest manifest(path, 77u);
        const auto& loaded = manifest.LoadedRecords();
        Require(loaded.compute.size() == 3u && loaded.graphics.size() == 2u && manifest.Records(compute[0].program).front().first == compute[0], "a manifest did not load the pipeline records the previous one saved");
        Require(loaded.prepared == std::vector<AgcDriver::PreparedProgramRecord>{prepared[0], prepared[1]}, "a manifest kept a program prepared at use that no pipeline record names, or lost one");
        Require(manifest.PreparedPrograms({7u, 8u}).size() == 2u && manifest.PreparedPrograms({9u, 9u}).empty(), "the programs prepared at use of a registered program are not the ones saved");
        Require(manifest.BranchSplits({7u, 8u}) == splits && manifest.BranchSplits({9u, 9u}).empty(), "the splits of a registered program are not the ones saved");
        for (const auto& entry : loaded.compute) Require(entry.usage.misses == 0u && entry.usage.progress != 0u, "a record made in a run was saved without that run's use");
    }
    const std::array<std::uint64_t, 2> rebuilt{0x9999u, 0x9898u};
    const std::array<std::uint64_t, 2> rebuiltStage{0x7777u, 0x7676u};
    {
        AgcDriver::PipelineManifest manifest(path, 78u);
        Require(manifest.RecordedByAnotherBuild() && manifest.Loaded() == 9u && manifest.Records(compute[0].program).size() == 2u && manifest.GraphicsRecords().size() == 2u, "a manifest of another recompiler build was not loaded with its records");
        const AgcDriver::ProgramInterface computeInterface{0u, {}, {{9u, 2u}}};
        const AgcDriver::ProgramInterface stageInterface{32u, {{1u, 7u, 0u, 4u, 1u}}, {}};
        const auto movedCompute = manifest.Rekey(compute[0].program, rebuilt, computeInterface);
        const auto movedStage = manifest.Rekey(graphics[0].stages[0].program, rebuiltStage, stageInterface);
        Require(movedCompute.compute == 2u && movedStage.graphics == 2u && movedCompute.droppedCompute + movedCompute.droppedGraphics + movedStage.droppedCompute + movedStage.droppedGraphics == 0u, "rekeying with a derived interface dropped records or moved the wrong number");
        manifest.Forget(prepared[0]);
        manifest.Forget(prepared[1]);
        auto first = prepared[0];
        first.identity = rebuilt;
        auto second = prepared[1];
        second.identity = rebuiltStage;
        Require(manifest.Record(first) && manifest.Record(second), "a program prepared at use under its new identity was refused");
        Require(manifest.Records(compute[0].program).empty() && manifest.Records(rebuilt).size() == 2u && manifest.Records(rebuilt).front().first.program == rebuilt, "rekeying moved no compute record to the program's new identity");
        Require(std::ranges::all_of(manifest.GraphicsRecords(), [&](const auto& entry) { return entry.first.stages[0].program == rebuiltStage && entry.first.stages[0].pushBytes == 32u && entry.first.stages[0].bindings == stageInterface.bindings && entry.first.stages[1].bindings == graphics[0].stages[1].bindings; }), "rekeying left a graphics stage on the program's old identity or old interface, or changed another stage");
        Require(std::ranges::all_of(manifest.Records(rebuilt), [&](const auto& entry) { return entry.first.bindings == computeInterface.computeBindings; }), "rekeying left a compute record on its old bindings");
        Require(manifest.PreparedPrograms({7u, 8u}) == std::vector<AgcDriver::PreparedProgramRecord>{first, second}, "a forgotten program prepared at use was kept, or its new record lost");
    }
    {
        AgcDriver::PipelineManifest manifest(path, 78u);
        Require(!manifest.RecordedByAnotherBuild() && manifest.Records(rebuilt).size() == 2u && manifest.Records(compute[0].program).empty() && manifest.PreparedPrograms({7u, 8u}).size() == 2u && manifest.GraphicsRecords().size() == 2u, "a rekeyed manifest did not save its new identities under the build that rekeyed it");
        const auto droppedCompute = manifest.Rekey(compute[2].program, {1u, 1u}, std::nullopt);
        const auto droppedGraphics = manifest.Rekey(graphics[0].stages[1].program, {2u, 2u}, std::nullopt);
        Require(droppedCompute.droppedCompute == 1u && droppedCompute.compute == 0u && manifest.Records(compute[2].program).empty() && manifest.Records({1u, 1u}).empty(), "a compute record of a program whose interface cannot be derived was kept");
        Require(droppedGraphics.droppedGraphics == 2u && manifest.GraphicsRecords().empty(), "a graphics record naming a program whose interface cannot be derived was kept");
    }
    std::error_code error;
    std::filesystem::remove_all(directory, error);
}

void Expiry() {
    const auto directory = Directory();
    const auto path = directory / "pipelines.manifest";
    const auto compute = ComputeRecords();
    const auto graphics = GraphicsRecords();
    const auto prepared = PreparedRecords();
    {
        AgcDriver::PipelineManifest manifest(path, 77u);
        static_cast<void>(manifest.Record(compute[0]));
        static_cast<void>(manifest.Record(graphics[0]));
        static_cast<void>(manifest.Record(compute[1]));
        static_cast<void>(manifest.Record(compute[2]));
        static_cast<void>(manifest.Record(prepared[1]));
    }
    const auto usage = [&](const AgcDriver::ManifestRecords& records, const AgcDriver::ComputePipelineRecord& record) {
        for (const auto& entry : records.compute) {
            if (entry.record == record) return entry.usage;
        }
        return AgcDriver::RecordUsage{~0u, 0u};
    };
    const auto run = [&](std::initializer_list<const AgcDriver::ComputePipelineRecord*> used, bool graphicsUsed) {
        AgcDriver::PipelineManifest manifest(path, 77u);
        for (const auto* record : used) static_cast<void>(manifest.Record(*record));
        if (graphicsUsed) static_cast<void>(manifest.Record(graphics[0]));
        return manifest.LoadedRecords();
    };
    auto loaded = run({}, false);
    Require(usage(loaded, compute[0]) == AgcDriver::RecordUsage{0u, 1u} && usage(loaded, compute[2]) == AgcDriver::RecordUsage{0u, 4u}, "the first run's records did not keep the order of their first use");
    loaded = run({&compute[0]}, false);
    Require(usage(loaded, compute[0]) == AgcDriver::RecordUsage{0u, 1u} && usage(loaded, compute[1]) == AgcDriver::RecordUsage{0u, 3u}, "a run that used nothing changed its records");
    loaded = run({&compute[0], &compute[2], &compute[1]}, true);
    Require(usage(loaded, compute[0]) == AgcDriver::RecordUsage{0u, 1u} && usage(loaded, compute[1]) == AgcDriver::RecordUsage{0u, 3u} && usage(loaded, compute[2]) == AgcDriver::RecordUsage{0u, 4u}, "a run that ended before the later records' first use counted it against them");
    for (std::uint32_t miss = 1; miss <= AgcDriver::PipelineManifest::ExpiryRuns; ++miss) {
        loaded = run({&compute[0], &compute[2]}, true);
        Require(usage(loaded, compute[1]) == AgcDriver::RecordUsage{miss - 1u, 3u}, "a record unused by a run that went past its first use did not count a miss");
        Require(usage(loaded, compute[0]).misses == 0u && usage(loaded, compute[2]).misses == 0u, "a record used in every run counted a miss");
    }
    AgcDriver::PipelineManifest manifest(path, 77u);
    Require(manifest.Expired() == 1u && manifest.Records(compute[1].program).size() == 1u && manifest.Records(compute[1].program).front().first == compute[0], "a record unused in " + std::to_string(AgcDriver::PipelineManifest::ExpiryRuns) + " runs that went past its first use was kept");
    Require(manifest.GraphicsRecords().size() == 1u && manifest.LoadedRecords().prepared.size() == 1u, "a used graphics record or the program it names was dropped");
    std::error_code error;
    std::filesystem::remove_all(directory, error);
}

void Unnamed() {
    AgcDriver::ManifestRecords records{WithUsage(ComputeRecords(), AgcDriver::PipelineManifest::ExpiryRuns), WithUsage(GraphicsRecords(), 0u), PreparedRecords(), SplitRecords()};
    std::size_t dropped = 0;
    const auto kept = AgcDriver::PipelineManifest::Unexpired(records, &dropped);
    Require(kept.compute.empty() && kept.graphics.size() == 2u && kept.prepared == std::vector<AgcDriver::PreparedProgramRecord>{PreparedRecords()[1]} && kept.splits == std::vector<AgcDriver::BranchSplitRecord>{SplitRecords()[0]} && dropped == 6u, "expiry did not drop the expired pipelines and the programs and splits only they named");
}

}

int main() {
    try {
        Codec();
        File();
        Expiry();
        Unnamed();
        std::puts("pipeline manifest tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
