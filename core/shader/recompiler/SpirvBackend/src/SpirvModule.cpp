#include "SpirvBackend/SpirvModule.hpp"

#include <algorithm>
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

void SpirvModule::ReturnThroughResultPointers() {
    if (inHelperFunction) throw std::runtime_error("SpirvModule::ReturnThroughResultPointers called inside a helper function");
    if (unpatchedPhiIncomings != 0) throw std::runtime_error("SpirvModule::ReturnThroughResultPointers called with unpatched OpPhi incoming pairs");
    struct Rewrite {
        std::uint32_t result;
        std::uint32_t pointer;
        std::uint32_t type;
        std::uint32_t parameter;
    };
    const auto voidType = Type(spv::OpTypeVoid);
    const std::array<std::vector<std::uint32_t>*, 2> sections{&functionInstructions, &helperFunctionInstructions};
    const auto instructions = [](const std::vector<std::uint32_t>& section, auto&& visit) {
        for (std::size_t offset = 0; offset < section.size(); offset += section[offset] >> spv::WordCountShift) {
            visit(std::span<const std::uint32_t>(section.data() + offset, section[offset] >> spv::WordCountShift));
        }
    };
    std::map<std::uint32_t, Rewrite> rewrites;
    for (const auto* section : sections) {
        std::uint32_t function = 0;
        std::vector<std::uint32_t> types;
        instructions(*section, [&](std::span<const std::uint32_t> words) {
            const auto opcode = words[0] & spv::OpCodeMask;
            if (opcode == spv::OpFunction) {
                function = words[1] != voidType && (words[3] & spv::FunctionControlDontInlineMask) != 0u ? words[2] : 0u;
                types = {voidType};
                if (function != 0u) rewrites.emplace(function, Rewrite{words[1], Type(spv::OpTypePointer, static_cast<std::uint32_t>(spv::StorageClassFunction), words[1]), 0u, AllocateId()});
            } else if (opcode == spv::OpFunctionParameter && function != 0u) {
                types.push_back(words[1]);
            } else if (opcode == spv::OpLabel && function != 0u) {
                auto& rewrite = rewrites.at(function);
                types.push_back(rewrite.pointer);
                rewrite.type = Type(spv::OpTypeFunction, std::span<const std::uint32_t>(types));
                function = 0u;
            }
        });
    }
    if (rewrites.empty()) return;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> variables;
    std::map<std::uint32_t, std::vector<std::uint32_t>> callees;
    for (const auto* section : sections) {
        std::uint32_t caller = 0;
        instructions(*section, [&](std::span<const std::uint32_t> words) {
            const auto opcode = words[0] & spv::OpCodeMask;
            if (opcode == spv::OpFunction) caller = words[2];
            if (opcode != spv::OpFunctionCall || !rewrites.contains(words[3])) return;
            if (variables.emplace(std::pair{caller, words[3]}, 0u).second) callees[caller].push_back(words[3]);
        });
    }
    for (auto& [key, variable] : variables) variable = AllocateId();
    for (auto* section : sections) {
        std::vector<std::uint32_t> rewritten;
        rewritten.reserve(section->size());
        std::uint32_t caller = 0;
        const Rewrite* current = nullptr;
        bool entry = false;
        instructions(*section, [&](std::span<const std::uint32_t> words) {
            const auto opcode = words[0] & spv::OpCodeMask;
            if (opcode == spv::OpFunction) {
                caller = words[2];
                entry = true;
                const auto found = rewrites.find(caller);
                current = found == rewrites.end() ? nullptr : &found->second;
                if (current != nullptr) {
                    appendInstruction(rewritten, spv::OpFunction, voidType, caller, words[3], current->type);
                    return;
                }
            } else if (opcode == spv::OpLabel && entry) {
                entry = false;
                if (current != nullptr) appendInstruction(rewritten, spv::OpFunctionParameter, current->pointer, current->parameter);
                rewritten.insert(rewritten.end(), words.begin(), words.end());
                if (const auto found = callees.find(caller); found != callees.end()) {
                    for (const auto callee : found->second) appendInstruction(rewritten, spv::OpVariable, rewrites.at(callee).pointer, variables.at({caller, callee}), spv::StorageClassFunction);
                }
                return;
            } else if (opcode == spv::OpFunctionEnd) {
                current = nullptr;
            } else if (opcode == spv::OpReturnValue && current != nullptr) {
                appendInstruction(rewritten, spv::OpStore, current->parameter, words[1]);
                appendInstruction(rewritten, spv::OpReturn);
                return;
            } else if (opcode == spv::OpFunctionCall) {
                const auto found = rewrites.find(words[3]);
                if (found != rewrites.end()) {
                    const auto variable = variables.at({caller, words[3]});
                    appendInstruction(rewritten, spv::OpFunctionCall, voidType, AllocateId(), words.subspan(3), variable);
                    appendInstruction(rewritten, spv::OpLoad, found->second.result, words[2], variable);
                    return;
                }
            }
            rewritten.insert(rewritten.end(), words.begin(), words.end());
        });
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
