#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>
#undef SPV_ENABLE_UTILITY_CODE
#include "SpirvBackend/SpirvProbeModule.hpp"
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ShaderRecompiler {

namespace {

bool DeviceStorage(std::uint32_t storageClass) {
    switch (static_cast<spv::StorageClass>(storageClass)) {
        case spv::StorageClassFunction:
        case spv::StorageClassPrivate:
        case spv::StorageClassWorkgroup:
        case spv::StorageClassInput:
            return false;
        default:
            return true;
    }
}

bool AtomicWithResult(spv::Op op) {
    switch (op) {
        case spv::OpAtomicExchange:
        case spv::OpAtomicCompareExchange:
        case spv::OpAtomicCompareExchangeWeak:
        case spv::OpAtomicIIncrement:
        case spv::OpAtomicIDecrement:
        case spv::OpAtomicIAdd:
        case spv::OpAtomicISub:
        case spv::OpAtomicSMin:
        case spv::OpAtomicUMin:
        case spv::OpAtomicSMax:
        case spv::OpAtomicUMax:
        case spv::OpAtomicAnd:
        case spv::OpAtomicOr:
        case spv::OpAtomicXor:
        case spv::OpAtomicFlagTestAndSet:
        case spv::OpAtomicFAddEXT:
        case spv::OpAtomicFMinEXT:
        case spv::OpAtomicFMaxEXT:
            return true;
        default:
            return false;
    }
}

}

std::optional<std::vector<std::uint32_t>> DeviceImageProbeModule(std::span<const std::uint32_t> words) {
    constexpr std::size_t header = 5;
    if (words.size() <= header || words[0] != spv::MagicNumber) return std::nullopt;
    std::unordered_map<std::uint32_t, std::string> names;
    std::unordered_map<std::uint32_t, std::uint32_t> pointerClasses;
    std::unordered_map<std::uint32_t, std::uint32_t> types;
    std::size_t firstFunction = 0;
    std::uint32_t entry = 0;
    for (std::size_t cursor = header; cursor < words.size();) {
        const auto count = words[cursor] >> spv::WordCountShift;
        const auto op = static_cast<spv::Op>(words[cursor] & spv::OpCodeMask);
        if (count == 0 || cursor + count > words.size()) return std::nullopt;
        if (op == spv::OpName && count > 2) {
            const auto* text = reinterpret_cast<const char*>(words.data() + cursor + 2);
            names[words[cursor + 1]] = std::string(text, strnlen(text, (count - 2u) * 4u));
        }
        if (op == spv::OpTypePointer && count == 4) pointerClasses[words[cursor + 1]] = words[cursor + 2];
        if (op == spv::OpTypeForwardPointer && count == 3) pointerClasses[words[cursor + 1]] = words[cursor + 2];
        bool hasResult = false;
        bool hasType = false;
        spv::HasResultAndType(op, &hasResult, &hasType);
        if (hasResult && hasType && count >= 3) types[words[cursor + 2]] = words[cursor + 1];
        if (op == spv::OpFunction && firstFunction == 0) firstFunction = cursor;
        if (op == spv::OpEntryPoint && count >= 3 && entry == 0) entry = words[cursor + 2];
        cursor += count;
    }
    if (firstFunction == 0) return std::nullopt;
    const auto named = [&](std::string_view name) {
        for (const auto& [id, text] : names) {
            if (text == name) return id;
        }
        return 0u;
    };
    const auto lookup = named("device_image_lookup");
    if (lookup == 0u) return std::nullopt;
    const std::set<std::uint32_t> kept{lookup, named("device_image_record_miss")};
    std::set<std::uint32_t> records;
    for (const auto* name : {"record_bda_fault", "note_bda_write", "note_written_chunk", "note_bda_touch", "record_image_store"}) {
        if (const auto id = named(name); id != 0u) records.insert(id);
    }
    const auto device = [&](std::uint32_t pointer) {
        const auto type = types.find(pointer);
        if (type == types.end()) return true;
        const auto storage = pointerClasses.find(type->second);
        return storage == pointerClasses.end() || DeviceStorage(storage->second);
    };
    std::unordered_map<std::uint32_t, std::set<std::uint32_t>> callees;
    std::set<std::uint32_t> polling;
    {
        std::uint32_t current = 0;
        for (std::size_t cursor = firstFunction; cursor < words.size();) {
            const auto count = words[cursor] >> spv::WordCountShift;
            const auto op = static_cast<spv::Op>(words[cursor] & spv::OpCodeMask);
            if (op == spv::OpFunction) current = words[cursor + 2];
            if (op == spv::OpFunctionCall && !records.contains(words[cursor + 3])) callees[current].insert(words[cursor + 3]);
            if (op == spv::OpAtomicLoad && device(words[cursor + 3])) polling.insert(current);
            cursor += count;
        }
    }
    std::set<std::uint32_t> reached;
    std::vector<std::uint32_t> pending{entry};
    while (!pending.empty()) {
        const auto current = pending.back();
        pending.pop_back();
        if (current == 0u || kept.contains(current) || !reached.insert(current).second) continue;
        if (polling.contains(current)) return std::nullopt;
        for (const auto callee : callees[current]) pending.push_back(callee);
    }
    auto bound = words[3];
    std::map<std::uint32_t, std::uint32_t> nulls;
    const auto nullOf = [&](std::uint32_t type) {
        const auto [found, inserted] = nulls.try_emplace(type, 0u);
        if (inserted) found->second = bound++;
        return found->second;
    };
    std::vector<std::uint32_t> body;
    body.reserve(words.size() - firstFunction);
    std::uint32_t function = 0;
    for (std::size_t cursor = firstFunction; cursor < words.size();) {
        const auto count = words[cursor] >> spv::WordCountShift;
        const auto op = static_cast<spv::Op>(words[cursor] & spv::OpCodeMask);
        const auto* operands = words.data() + cursor;
        cursor += count;
        if (op == spv::OpFunction) function = operands[2];
        const bool strip = !kept.contains(function);
        if (strip) {
            if ((op == spv::OpStore || op == spv::OpCopyMemory || op == spv::OpCopyMemorySized) && device(operands[1])) continue;
            if (op == spv::OpImageWrite) continue;
            if ((op == spv::OpAtomicStore || op == spv::OpAtomicFlagClear) && device(operands[1])) continue;
            if (op == spv::OpFunctionCall && records.contains(operands[3])) continue;
            if (AtomicWithResult(op) && device(operands[3])) {
                body.insert(body.end(), {4u << spv::WordCountShift | spv::OpCopyObject, operands[1], operands[2], nullOf(operands[1])});
                continue;
            }
        }
        body.insert(body.end(), operands, operands + count);
        if (op == spv::OpFunctionEnd) function = 0;
    }
    std::vector<std::uint32_t> module(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(firstFunction));
    for (const auto& [type, id] : nulls) module.insert(module.end(), {3u << spv::WordCountShift | spv::OpConstantNull, type, id});
    module.insert(module.end(), body.begin(), body.end());
    module[3] = bound;
    return module;
}

}
