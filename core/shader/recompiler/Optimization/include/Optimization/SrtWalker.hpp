#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SRTWALKER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SRTWALKER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

using SrtMemoryReader = bool (*)(void* userData, std::uint64_t address, std::uint32_t* value);
using SrtMemoryProbe = bool (*)(void* userData, std::uint64_t address, std::uint64_t bytes);
// Reads `count` consecutive dwords from `address` exactly as `count` readMemory calls in address
// order would (the same values, the same recorded reads and waits), only cheaper.
using SrtMemoryBulkReader = bool (*)(void* userData, std::uint64_t address, std::uint32_t* values, std::uint64_t count);

// The guest addresses a walk dereferenced (Detail::Evaluator::EvaluateRawRead): the leaf read of
// each pure flat slot (IrResourcePlan::pureFlatSlots) as (flat offset, address), set by the
// flattened loop through `leaf`/`leafSlot` around that slot's evaluation, and every other raw read
// address.
struct SrtReadTrace {
    const IrValue* leaf = nullptr;
    std::uint32_t leafSlot = 0;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> leaves;
    std::vector<std::uint64_t> otherReads;
};

// Optional memory services for image-table snapshots that skip reading a table whose memory is
// unchanged (ResourceMaterializer's column memo): the host's write tracking and the capture's page
// store. Every member is required when the struct is given.
struct SrtTableMemory {
    // A write generation for the range, collected now: what the range holds afterwards stays what
    // it holds while `unchanged` answers true for it. 0 when the host does not track the range.
    std::uint64_t (*collect)(void* userData, std::uint64_t address, std::uint64_t bytes) = nullptr;
    // Whether the range still holds what it held at `generation`, after storing into it whatever a
    // read would have flushed first, with no recorded work still writing it, accessible whole.
    bool (*unchanged)(void* userData, std::uint64_t address, std::uint64_t bytes, std::uint64_t generation) = nullptr;
    // Reads the 4 KiB page at `base` from memory now, as a whole-page fetch would; false when it is
    // not accessible whole.
    bool (*readPage)(void* userData, std::uint64_t base, std::uint32_t* words) = nullptr;
    // Records the dwords set in `mask` (1024 bits as 16 words) of the page at `base` as read, with
    // `words` (the page's 1024 dwords) as the capture's copy of the page, exactly as reading them
    // would; false, recording nothing of the page, when the capture cannot take the page that way.
    bool (*replayPage)(void* userData, std::uint64_t base, const std::uint32_t* words, const std::uint64_t* mask) = nullptr;
};

struct SrtRuntime {
    std::span<const std::uint32_t> userData;
    std::uint64_t shaderBase = 0;
    SrtMemoryReader readMemory = nullptr;
    void* userContext = nullptr;
    SrtMemoryReader readSpecializationMemory = nullptr;
    SrtReadTrace* readTrace = nullptr;
    SrtMemoryProbe accessible = nullptr;
    SrtMemoryProbe pendingWrite = nullptr;
    // Optional bulk form of readMemory (image-table snapshots); null reads word by word.
    SrtMemoryBulkReader readWords = nullptr;
    // Optional (image-table snapshots): null always reads the tables.
    const SrtTableMemory* tableMemory = nullptr;
};

enum class RuntimeValueType {
    Any,
    Integer
};

class SrtWalker {
public:
    void BuildPlan(IrProgram& program) const;
    [[nodiscard]] bool ValidateRuntimeValue(const IrResourcePlan& program, const IrValue* value, RuntimeValueType type = RuntimeValueType::Any) const;
    void EvaluateUniformValues(const IrResourcePlan& program, std::span<IrValue* const> values, const SrtRuntime& runtime, std::span<std::uint32_t> results) const;
    void EvaluateDescriptorSource(const IrResourcePlan& program, std::uint32_t source, const SrtRuntime& runtime, DescriptorValue& result) const;
    void EvaluateDescriptorSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results) const;
    void EvaluateRuntimeSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, std::span<const std::uint8_t> cleanFlatSlots, std::vector<std::uint8_t>& activeSources, std::vector<SrtReadPoison>* poison = nullptr, std::uint32_t* nullRootReads = nullptr) const;
    void Walk(const IrResourcePlan& program, const SrtRuntime& runtime, std::vector<std::uint32_t>& flat) const;

};

}

#endif
