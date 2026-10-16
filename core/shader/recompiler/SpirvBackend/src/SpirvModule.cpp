#define SPV_ENABLE_UTILITY_CODE
#include "SpirvBackend/SpirvModule.hpp"
#undef SPV_ENABLE_UTILITY_CODE

#include <algorithm>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <spirv/unified1/spirv.hpp>
#include <stdexcept>

namespace ShaderRecompiler {
namespace {

constexpr std::size_t InitialSpirvSectionReserve = 4096;
constexpr std::size_t InitialSpirvFunctionSectionReserve = 32768;

void AppendInstructionWords(std::vector<std::uint32_t>& section, std::span<const std::uint32_t> words) {
    if (words.empty()) {
        return;
    }
    const auto opcode = words[0];
    const auto wordCount = static_cast<std::uint32_t>(words.size());
    section.push_back((wordCount << spv::WordCountShift) | opcode);
    section.insert(section.end(), words.begin() + 1, words.end());
}

bool MovesValue(std::uint32_t opcode) {
    switch (opcode) {
    case spv::OpPhi:
    case spv::OpCopyObject:
    case spv::OpCompositeExtract:
    case spv::OpCompositeInsert:
    case spv::OpCompositeConstruct:
    case spv::OpVectorShuffle:
    case spv::OpBitcast:
    case spv::OpUConvert:
    case spv::OpSConvert:
        return true;
    default:
        return false;
    }
}

void MergeSorted(std::vector<std::uint32_t>& into, const std::vector<std::uint32_t>& from, bool& changed) {
    if (from.empty()) return;
    std::vector<std::uint32_t> merged;
    merged.reserve(into.size() + from.size());
    std::set_union(into.begin(), into.end(), from.begin(), from.end(), std::back_inserter(merged));
    if (merged.size() != into.size()) {
        into = std::move(merged);
        changed = true;
    }
}

std::vector<std::size_t> CallsToSpill(std::span<const std::uint32_t> function, const std::set<std::uint32_t>& outOfLine, const std::set<std::uint32_t>& registerMemory) {
    constexpr auto none = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::size_t> offsets;
    std::vector<std::uint32_t> blockOf;
    std::vector<std::size_t> blockLast;
    std::unordered_map<std::uint32_t, std::uint32_t> blockOfLabel;
    struct Site {
        std::size_t instruction;
        std::uint32_t callee;
    };
    std::vector<Site> sites;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> origins;
    for (std::size_t offset = 0; offset < function.size(); offset += function[offset] >> spv::WordCountShift) {
        if ((function[offset] >> spv::WordCountShift) == 0u) throw std::runtime_error("SpirvModule::SpillCallResults found an empty instruction");
        const auto opcode = function[offset] & spv::OpCodeMask;
        if (opcode == spv::OpFunctionEnd) {
            blockOf.push_back(none);
            offsets.push_back(offset);
            continue;
        }
        if (opcode == spv::OpLabel) {
            blockOfLabel.emplace(function[offset + 1u], static_cast<std::uint32_t>(blockLast.size()));
            blockLast.push_back(offsets.size());
        }
        blockOf.push_back(blockLast.empty() ? none : static_cast<std::uint32_t>(blockLast.size() - 1u));
        if (!blockLast.empty()) blockLast.back() = offsets.size();
        if (opcode == spv::OpFunctionCall && outOfLine.contains(function[offset + 3u])) {
            origins[function[offset + 2u]].push_back(static_cast<std::uint32_t>(sites.size()));
            sites.push_back({offsets.size(), function[offset + 3u]});
        }
        offsets.push_back(offset);
    }
    if (sites.empty()) return {};
    const auto words = [&](std::size_t instruction) { return function.subspan(offsets[instruction], function[offsets[instruction]] >> spv::WordCountShift); };
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t instruction = 0; instruction < offsets.size(); ++instruction) {
            const auto op = words(instruction);
            const auto opcode = op[0] & spv::OpCodeMask;
            if (!MovesValue(opcode) || op.size() < 4u) continue;
            std::size_t last = op.size();
            if (opcode == spv::OpCompositeExtract) last = 4u;
            if (opcode == spv::OpCompositeInsert || opcode == spv::OpVectorShuffle) last = 5u;
            for (std::size_t index = 3; index < last; index += opcode == spv::OpPhi ? 2u : 1u) {
                const auto found = origins.find(op[index]);
                if (found == origins.end() || op[index] == op[2]) continue;
                const auto from = found->second;
                MergeSorted(origins[op[2]], from, changed);
            }
        }
    }
    const auto blocks = blockLast.size();
    std::vector<std::vector<std::uint32_t>> predecessors(blocks);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        const auto op = words(blockLast[block]);
        const auto opcode = op[0] & spv::OpCodeMask;
        std::size_t first = op.size();
        if (opcode == spv::OpBranch) first = 1u;
        if (opcode == spv::OpBranchConditional) first = 2u;
        if (opcode == spv::OpSwitch) first = 2u;
        const std::size_t last = opcode == spv::OpBranchConditional ? 4u : op.size();
        std::set<std::uint32_t> targets;
        for (auto index = first; index < last; ++index) {
            if (const auto found = blockOfLabel.find(op[index]); found != blockOfLabel.end()) targets.insert(found->second);
        }
        for (const auto target : targets) predecessors[target].push_back(block);
    }
    struct Value {
        std::size_t definition = 0;
        std::vector<std::size_t> uses;
        std::vector<std::uint32_t> edges;
    };
    std::unordered_map<std::uint32_t, Value> values;
    std::unordered_map<std::uint32_t, std::uint32_t> types;
    for (std::size_t instruction = 0; instruction < offsets.size(); ++instruction) {
        const auto op = words(instruction);
        bool hasResult = false;
        bool hasType = false;
        spv::HasResultAndType(static_cast<spv::Op>(op[0] & spv::OpCodeMask), &hasResult, &hasType);
        if (hasResult && hasType && op.size() > 2u) types.emplace(op[2], op[1]);
    }
    std::vector<bool> spilled(sites.size(), false);
    for (std::size_t instruction = 0; instruction < offsets.size(); ++instruction) {
        const auto op = words(instruction);
        const auto opcode = static_cast<spv::Op>(op[0] & spv::OpCodeMask);
        bool hasResult = false;
        bool hasType = false;
        spv::HasResultAndType(opcode, &hasResult, &hasType);
        if (opcode == spv::OpStore && op.size() > 2u) {
            const auto stored = origins.find(op[2]);
            const auto type = types.find(op[1]);
            if (stored != origins.end() && (registerMemory.contains(op[1]) || (type != types.end() && registerMemory.contains(type->second)))) {
                for (const auto origin : stored->second) spilled[origin] = true;
            }
        }
        const std::size_t resultIndex = hasType ? 2u : 1u;
        if (hasResult && op.size() > resultIndex && origins.contains(op[resultIndex])) values[op[resultIndex]].definition = instruction;
        const std::size_t first = (hasResult ? 1u : 0u) + (hasType ? 1u : 0u) + 1u;
        for (auto index = first; index < op.size(); ++index) {
            if (!origins.contains(op[index])) continue;
            if (opcode == spv::OpPhi && index + 1u < op.size() && (index - first) % 2u == 0u) {
                if (const auto parent = blockOfLabel.find(op[index + 1u]); parent != blockOfLabel.end()) values[op[index]].edges.push_back(parent->second);
                continue;
            }
            values[op[index]].uses.push_back(instruction);
        }
    }
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> sitesOf;
    for (std::uint32_t site = 0; site < sites.size(); ++site) sitesOf[sites[site].callee].push_back(site);
    std::vector<std::uint32_t> liveIn(blocks, none);
    std::vector<std::uint32_t> liveOut(blocks, none);
    std::vector<std::uint32_t> pending;
    std::uint32_t generation = 0;
    for (const auto& [id, value] : values) {
        const auto& from = origins.at(id);
        ++generation;
        const auto home = blockOf[value.definition];
        const auto enter = [&](std::uint32_t block) {
            if (block == home || liveIn[block] == generation) return;
            liveIn[block] = generation;
            pending.push_back(block);
        };
        for (const auto use : value.uses) enter(blockOf[use]);
        for (const auto edge : value.edges) {
            liveOut[edge] = generation;
            enter(edge);
        }
        while (!pending.empty()) {
            const auto block = pending.back();
            pending.pop_back();
            for (const auto predecessor : predecessors[block]) {
                liveOut[predecessor] = generation;
                enter(predecessor);
            }
        }
        std::set<std::uint32_t> callees;
        for (const auto origin : from) callees.insert(sites[origin].callee);
        for (const auto callee : callees) {
            for (const auto site : sitesOf.at(callee)) {
                const auto at = sites[site].instruction;
                const auto block = blockOf[at];
                const bool before = (block == home && value.definition < at) || liveIn[block] == generation;
                if (!before) continue;
                const bool after = liveOut[block] == generation || std::any_of(value.uses.begin(), value.uses.end(), [&](std::size_t use) { return blockOf[use] == block && use > at; });
                if (!after) continue;
                for (const auto origin : from) {
                    if (sites[origin].callee == callee) spilled[origin] = true;
                }
            }
        }
    }
    std::vector<std::size_t> spills;
    for (std::uint32_t site = 0; site < sites.size(); ++site) {
        if (spilled[site]) spills.push_back(offsets[sites[site].instruction]);
    }
    std::sort(spills.begin(), spills.end());
    return spills;
}

}

SpirvModule::SpirvModule(std::uint32_t version) : version(version) {
    debug.reserve(InitialSpirvSectionReserve);
    annotations.reserve(InitialSpirvSectionReserve);
    declarations.reserve(InitialSpirvSectionReserve);
    functionInstructions.reserve(InitialSpirvFunctionSectionReserve);
}

std::uint32_t SpirvModule::AllocateId() {
    return nextId++;
}

void SpirvModule::EmitCapability(std::uint32_t capability) {
    if (requiredCapabilities.insert(capability).second) {
        appendInstruction(capabilities, spv::OpCapability, capability);
    }
}

void SpirvModule::EmitExtension(const std::string& extensionName) {
    if (requiredExtensions.insert(extensionName).second) {
        std::vector<std::uint32_t> operands;
        appendString(operands, extensionName);
        appendInstruction(extensions, spv::OpExtension, operands);
    }
}

void SpirvModule::EmitEntryPoint(std::uint32_t executionModel, std::uint32_t entryPointId, const std::string& entryPointName, const std::vector<std::uint32_t>& interfaceIds) {
    std::vector<std::uint32_t> operands;
    appendOperands(operands, executionModel, entryPointId);
    appendString(operands, entryPointName);
    operands.insert(operands.end(), interfaceIds.begin(), interfaceIds.end());
    if (version >= 0x00010400u) {
        const auto scanForVariables = [&](const std::vector<std::uint32_t>& section) {
            for (std::size_t offset = 0; offset < section.size();) {
                const auto count = section[offset] >> spv::WordCountShift;
                if ((section[offset] & spv::OpCodeMask) == spv::OpVariable) {
                    const auto id = section[offset + 2u];
                    if (std::find(interfaceIds.begin(), interfaceIds.end(), id) == interfaceIds.end()) {
                        operands.push_back(id);
                    }
                }
                offset += count;
            }
        };
        scanForVariables(declarations);
        scanForVariables(globalVariables);
    }
    appendInstruction(entryPoints, spv::OpEntryPoint, operands);
}

void SpirvModule::EmitTypeDeclaration(std::vector<std::uint32_t> words) {
    AppendInstructionWords(typeDeclarations, words);
}

void SpirvModule::EmitGlobalVariable(std::vector<std::uint32_t> words) {
    AppendInstructionWords(globalVariables, words);
}

void SpirvModule::EmitFunctionInstruction(std::vector<std::uint32_t> words) {
    AppendInstructionWords(functionInstructions, words);
}

std::vector<std::uint32_t> SpirvModule::Finalize() const {
    if (unpatchedPhiIncomings != 0) {
        throw std::runtime_error("SpirvModule::Finalize called with unpatched OpPhi incoming pairs");
    }
    if (inHelperFunction) throw std::runtime_error("SpirvModule::Finalize called inside a helper function");
    std::vector<std::uint32_t> module;
    module.reserve(5u + capabilities.size() + extensions.size() + extInstImports.size() + memoryModel.size() + entryPoints.size() + executionModes.size() + debug.size() + annotations.size() + typeDeclarations.size() + declarations.size() + globalVariables.size() + functionInstructions.size() + helperFunctionInstructions.size());
    module.push_back(spv::MagicNumber);
    module.push_back(version);
    module.push_back(0u);
    module.push_back(nextId);
    module.push_back(0u);
    module.insert(module.end(), capabilities.begin(), capabilities.end());
    module.insert(module.end(), extensions.begin(), extensions.end());
    module.insert(module.end(), extInstImports.begin(), extInstImports.end());
    module.insert(module.end(), memoryModel.begin(), memoryModel.end());
    module.insert(module.end(), entryPoints.begin(), entryPoints.end());
    module.insert(module.end(), executionModes.begin(), executionModes.end());
    module.insert(module.end(), debug.begin(), debug.end());
    module.insert(module.end(), annotations.begin(), annotations.end());
    module.insert(module.end(), typeDeclarations.begin(), typeDeclarations.end());
    module.insert(module.end(), declarations.begin(), declarations.end());
    module.insert(module.end(), globalVariables.begin(), globalVariables.end());
    module.insert(module.end(), functionInstructions.begin(), functionInstructions.end());
    module.insert(module.end(), helperFunctionInstructions.begin(), helperFunctionInstructions.end());
    return module;
}

void SpirvModule::BeginHelperFunction() {
    if (inHelperFunction) throw std::runtime_error("SpirvModule helper functions cannot nest");
    std::swap(functionInstructions, helperFunctionInstructions);
    inHelperFunction = true;
}

void SpirvModule::EndHelperFunction() {
    if (!inHelperFunction) throw std::runtime_error("SpirvModule::EndHelperFunction without a helper function");
    std::swap(functionInstructions, helperFunctionInstructions);
    inHelperFunction = false;
}

void SpirvModule::SpillCallResults() {
    if (inHelperFunction) throw std::runtime_error("SpirvModule::SpillCallResults called inside a helper function");
    if (unpatchedPhiIncomings != 0) throw std::runtime_error("SpirvModule::SpillCallResults called with unpatched OpPhi incoming pairs");
    const auto voidType = Type(spv::OpTypeVoid);
    const std::array<std::vector<std::uint32_t>*, 2> sections{&functionInstructions, &helperFunctionInstructions};
    const auto functions = [](const std::vector<std::uint32_t>& section, auto&& visit) {
        std::size_t begin = 0;
        for (std::size_t offset = 0; offset < section.size(); offset += section[offset] >> spv::WordCountShift) {
            if ((section[offset] >> spv::WordCountShift) == 0u) throw std::runtime_error("SpirvModule::SpillCallResults found an empty instruction");
            if ((section[offset] & spv::OpCodeMask) != spv::OpFunctionEnd) continue;
            const auto end = offset + (section[offset] >> spv::WordCountShift);
            visit(std::span<const std::uint32_t>(section.data() + begin, end - begin));
            begin = end;
        }
        return begin;
    };
    std::map<std::uint32_t, std::uint32_t> resultTypes;
    for (const auto* section : sections) {
        functions(*section, [&](std::span<const std::uint32_t> function) {
            if ((function[0] & spv::OpCodeMask) == spv::OpFunction && function[1] != voidType && (function[3] & spv::FunctionControlDontInlineMask) != 0u) resultTypes.emplace(function[2], function[1]);
        });
    }
    if (resultTypes.empty()) return;
    std::set<std::uint32_t> outOfLine;
    for (const auto& [function, type] : resultTypes) outOfLine.insert(function);
    std::set<std::uint32_t> registerMemory;
    for (const auto* section : {&typeDeclarations, &declarations, &globalVariables}) {
        for (std::size_t offset = 0; offset < section->size(); offset += (*section)[offset] >> spv::WordCountShift) {
            const auto opcode = (*section)[offset] & spv::OpCodeMask;
            if (opcode == spv::OpTypePointer && ((*section)[offset + 2u] == spv::StorageClassFunction || (*section)[offset + 2u] == spv::StorageClassPrivate)) registerMemory.insert((*section)[offset + 1u]);
            if (opcode == spv::OpVariable && (*section)[offset + 3u] == spv::StorageClassPrivate) registerMemory.insert((*section)[offset + 2u]);
        }
    }
    std::map<std::uint32_t, std::vector<std::size_t>> spills;
    std::set<std::uint32_t> anchored;
    for (const auto* section : sections) {
        functions(*section, [&](std::span<const std::uint32_t> function) {
            auto calls = CallsToSpill(function, outOfLine, registerMemory);
            if (calls.empty()) return;
            for (const auto at : calls) anchored.insert(function[at + 3u]);
            spills.emplace(function[2], std::move(calls));
        });
    }
    if (anchored.empty()) return;
    const auto pointerTo = [&](std::uint32_t type) { return Type(spv::OpTypePointer, static_cast<std::uint32_t>(spv::StorageClassFunction), type); };
    for (auto* section : sections) {
        std::vector<std::uint32_t> rewritten;
        rewritten.reserve(section->size());
        const auto tail = functions(*section, [&](std::span<const std::uint32_t> function) {
            const auto id = function[2];
            const auto found = spills.find(id);
            const std::span<const std::size_t> spilled = found == spills.end() ? std::span<const std::size_t>() : std::span<const std::size_t>(found->second);
            std::map<std::uint32_t, std::uint32_t> variables;
            for (std::size_t at = 0; at < function.size(); at += function[at] >> spv::WordCountShift) {
                if ((function[at] & spv::OpCodeMask) == spv::OpFunctionCall && anchored.contains(function[at + 3u]) && !variables.contains(function[at + 1u])) variables.emplace(function[at + 1u], AllocateId());
            }
            const bool anchoredFunction = anchored.contains(id);
            if (variables.empty() && !anchoredFunction) {
                rewritten.insert(rewritten.end(), function.begin(), function.end());
                return;
            }
            bool entry = true;
            std::vector<std::uint32_t> signature{function[1]};
            for (std::size_t at = 0; at < function.size(); at += function[at] >> spv::WordCountShift) {
                const auto words = function.subspan(at, function[at] >> spv::WordCountShift);
                const auto opcode = words[0] & spv::OpCodeMask;
                if (opcode == spv::OpFunction && anchoredFunction) {
                    for (std::size_t parameter = words.size(); parameter < function.size() && (function[parameter] & spv::OpCodeMask) == spv::OpFunctionParameter; parameter += function[parameter] >> spv::WordCountShift) signature.push_back(function[parameter + 1u]);
                    signature.push_back(pointerTo(function[1]));
                    appendInstruction(rewritten, spv::OpFunction, words[1], words[2], words[3], Type(spv::OpTypeFunction, std::span<const std::uint32_t>(signature)));
                    continue;
                }
                if (opcode == spv::OpLabel && entry) {
                    entry = false;
                    if (anchoredFunction) appendInstruction(rewritten, spv::OpFunctionParameter, pointerTo(function[1]), AllocateId());
                    rewritten.insert(rewritten.end(), words.begin(), words.end());
                    for (const auto& [type, variable] : variables) appendInstruction(rewritten, spv::OpVariable, pointerTo(type), variable, spv::StorageClassFunction);
                    continue;
                }
                if (opcode == spv::OpFunctionCall && anchored.contains(words[3])) {
                    const auto variable = variables.at(words[1]);
                    if (!std::binary_search(spilled.begin(), spilled.end(), at)) {
                        appendInstruction(rewritten, spv::OpFunctionCall, words.subspan(1), variable);
                        continue;
                    }
                    const auto returned = AllocateId();
                    appendInstruction(rewritten, spv::OpFunctionCall, words[1], returned, words.subspan(3), variable);
                    appendInstruction(rewritten, spv::OpStore, variable, returned, static_cast<std::uint32_t>(spv::MemoryAccessVolatileMask));
                    appendInstruction(rewritten, spv::OpLoad, words[1], words[2], variable, static_cast<std::uint32_t>(spv::MemoryAccessVolatileMask));
                    continue;
                }
                rewritten.insert(rewritten.end(), words.begin(), words.end());
            }
        });
        rewritten.insert(rewritten.end(), section->begin() + static_cast<std::ptrdiff_t>(tail), section->end());
        *section = std::move(rewritten);
    }
}

void SpirvModule::RequireVersion(std::uint32_t version) {
    this->version = std::max(this->version, version);
}

std::uint32_t SpirvModule::Import(const std::string& name) {
    if (const auto it = importIds.find(name); it != importIds.end()) {
        return it->second;
    }
    const auto id = AllocateId();
    importIds.emplace(name, id);
    std::vector<std::uint32_t> operands = {id};
    appendString(operands, name);
    appendInstruction(extInstImports, spv::OpExtInstImport, operands);
    return id;
}

std::uint32_t SpirvModule::declareType(std::uint32_t opcode, std::vector<std::uint32_t> key) {
    if (const auto it = declarationIds.find(key); it != declarationIds.end()) {
        return it->second;
    }
    const auto id = AllocateId();
    appendInstruction(declarations, opcode, id, std::span<const std::uint32_t>(key).subspan(2));
    declarationIds.emplace(std::move(key), id);
    return id;
}

std::uint32_t SpirvModule::declareDecoratedType(std::uint32_t opcode, std::vector<std::uint32_t> key, std::initializer_list<SpirvTypeAnnotation> annotationList) {
    if (annotationList.size() == 0) {
        return declareType(opcode, std::move(key));
    }
    const auto operandCount = key[1];
    key.push_back(static_cast<std::uint32_t>(annotationList.size()));
    for (const auto& annotation : annotationList) {
        appendOperand(key, annotation.opcode);
        key.push_back(static_cast<std::uint32_t>(annotation.operands.size()));
        key.insert(key.end(), annotation.operands.begin(), annotation.operands.end());
    }
    if (const auto it = declarationIds.find(key); it != declarationIds.end()) {
        return it->second;
    }
    const auto id = AllocateId();
    appendInstruction(declarations, opcode, id, std::span<const std::uint32_t>(key).subspan(2, operandCount));
    declarationIds.emplace(std::move(key), id);
    for (const auto& annotation : annotationList) {
        appendInstruction(annotations, annotation.opcode, id, annotation.operands);
    }
    return id;
}

std::uint32_t SpirvModule::SpecializationConstant(std::uint32_t type, std::uint32_t constantId, std::uint32_t defaultValue) {
    if (const auto found = specializationIds.find(constantId); found != specializationIds.end()) return found->second;
    const auto id = AllocateId();
    appendInstruction(declarations, spv::OpSpecConstant, type, id, defaultValue);
    AddAnnotation(spv::OpDecorate, id, spv::DecorationSpecId, constantId);
    specializationIds.emplace(constantId, id);
    return id;
}

std::uint32_t SpirvModule::declareConstant(std::uint32_t opcode, std::vector<std::uint32_t> key) {
    if (const auto it = declarationIds.find(key); it != declarationIds.end()) {
        return it->second;
    }
    const auto id = AllocateId();
    appendInstruction(declarations, opcode, key[1], id, std::span<const std::uint32_t>(key).subspan(2));
    declarationIds.emplace(std::move(key), id);
    return id;
}

std::uint32_t SpirvModule::DefineGlobalVariable(std::uint32_t pointerType, std::uint32_t storageClass) {
    const auto id = AllocateId();
    DefineGlobalVariable(id, pointerType, storageClass);
    return id;
}

void SpirvModule::DefineGlobalVariable(std::uint32_t id, std::uint32_t pointerType, std::uint32_t storageClass) {
    appendInstruction(declarations, spv::OpVariable, pointerType, id, storageClass);
}

std::uint32_t SpirvModule::DefineInitializedGlobalVariable(std::uint32_t pointerType, std::uint32_t storageClass, std::uint32_t initializer) {
    const auto id = AllocateId();
    appendInstruction(declarations, spv::OpVariable, pointerType, id, storageClass, initializer);
    return id;
}

void SpirvModule::AddMemoryModel(std::uint32_t addressingModel, std::uint32_t memoryModel) {
    appendInstruction(this->memoryModel, spv::OpMemoryModel, addressingModel, memoryModel);
}

void SpirvModule::AddName(std::uint32_t target, const std::string& name) {
    std::vector<std::uint32_t> operands = {target};
    appendString(operands, name);
    appendInstruction(debug, spv::OpName, operands);
}

void SpirvModule::AddFunction(std::span<const std::uint32_t> words) {
    AppendInstructionWords(functionInstructions, words);
}

std::size_t SpirvModule::FunctionWordCount() const {
    return functionInstructions.size();
}

std::vector<std::uint32_t> SpirvModule::TakeFunctionWords(std::size_t offset) {
    if (offset > functionInstructions.size()) throw std::runtime_error("SpirvModule::TakeFunctionWords offset is past the function instructions");
    std::vector<std::uint32_t> words(functionInstructions.begin() + static_cast<std::ptrdiff_t>(offset), functionInstructions.end());
    functionInstructions.resize(offset);
    return words;
}

void SpirvModule::AppendFunctionWords(std::span<const std::uint32_t> words) {
    functionInstructions.insert(functionInstructions.end(), words.begin(), words.end());
}

std::size_t SpirvModule::AnnotationWordCount() const {
    return annotations.size();
}

std::vector<std::uint32_t> SpirvModule::TakeAnnotationWords(std::size_t offset) {
    if (offset > annotations.size()) throw std::runtime_error("SpirvModule::TakeAnnotationWords offset is past the annotations");
    std::vector<std::uint32_t> words(annotations.begin() + static_cast<std::ptrdiff_t>(offset), annotations.end());
    annotations.resize(offset);
    return words;
}

void SpirvModule::AppendAnnotationWords(std::span<const std::uint32_t> words) {
    annotations.insert(annotations.end(), words.begin(), words.end());
}

void SpirvModule::appendString(std::vector<std::uint32_t>& words, const std::string& text) {
    const auto length = text.size();
    const auto wordCount = (length + 1u + 3u) / 4u;
    for (std::size_t i = 0; i < wordCount; i++) {
        std::uint32_t word = 0;
        for (std::size_t byte = 0; byte < 4; byte++) {
            const auto index = i * 4u + byte;
            if (index < length) {
                word |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[index])) << (byte * 8u);
            }
        }
        words.push_back(word);
    }
}

SpirvDeferredPhi SpirvModule::AddDeferredPhi(std::uint32_t type, std::uint32_t result, std::size_t incomingCount) {
    std::vector<std::uint32_t> words = {spv::OpPhi, type, result};
    words.resize(words.size() + incomingCount * 2u);
    const SpirvDeferredPhi phi {functionInstructions.size()};
    AddFunction(std::span<const std::uint32_t>(words));
    unpatchedPhiIncomings += incomingCount;
    return phi;
}

void SpirvModule::PatchDeferredPhi(SpirvDeferredPhi phi, std::size_t incoming, std::uint32_t value, std::uint32_t parent) {
    const auto incomingCount = ((functionInstructions.at(phi.wordOffset) >> spv::WordCountShift) - 3u) / 2u;
    if (incoming >= incomingCount || value == 0u || parent == 0u) {
        throw std::runtime_error("SpirvModule::PatchDeferredPhi received an invalid incoming index or a zero id");
    }
    const auto valueWord = phi.wordOffset + 3u + incoming * 2u;
    const auto parentWord = valueWord + 1u;
    if (functionInstructions.at(valueWord) != 0u || functionInstructions.at(parentWord) != 0u) {
        throw std::runtime_error("SpirvModule::PatchDeferredPhi target incoming pair is already patched");
    }
    functionInstructions[valueWord] = value;
    functionInstructions[parentWord] = parent;
    unpatchedPhiIncomings--;
}

}
