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

void AddSharedCallees(std::span<const std::uint32_t> function, const std::set<std::uint32_t>& outOfLine, std::set<std::uint32_t>& shared) {
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> blockOf;
    std::vector<std::size_t> terminators;
    std::unordered_map<std::uint32_t, std::size_t> blockOfLabel;
    std::map<std::uint32_t, std::vector<std::size_t>> sites;
    for (std::size_t offset = 0; offset < function.size(); offset += function[offset] >> spv::WordCountShift) {
        if ((function[offset] >> spv::WordCountShift) == 0u) throw std::runtime_error("SpirvModule::ReturnSharedResultsThroughPointers found an empty instruction");
        const auto opcode = function[offset] & spv::OpCodeMask;
        if (opcode == spv::OpLabel) {
            blockOfLabel.emplace(function[offset + 1u], terminators.size());
            terminators.push_back(offset);
        }
        if (opcode != spv::OpLabel && opcode != spv::OpFunctionEnd && !terminators.empty()) terminators.back() = offset;
        if (opcode == spv::OpFunctionCall && outOfLine.contains(function[offset + 3u]) && !terminators.empty()) sites[function[offset + 3u]].push_back(terminators.size() - 1u);
    }
    if (sites.empty()) return;
    const auto blocks = terminators.size();
    std::vector<std::vector<std::size_t>> successors(blocks);
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto offset = terminators[block];
        const auto count = function[offset] >> spv::WordCountShift;
        const auto opcode = function[offset] & spv::OpCodeMask;
        std::size_t first = count;
        std::size_t last = count;
        if (opcode == spv::OpBranch) first = 1u;
        if (opcode == spv::OpBranchConditional) {
            first = 2u;
            last = 4u;
        }
        if (opcode == spv::OpSwitch) first = 2u;
        for (auto index = first; index < last; ++index) {
            if (const auto found = blockOfLabel.find(function[offset + index]); found != blockOfLabel.end()) successors[block].push_back(found->second);
        }
    }
    constexpr auto unvisited = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> order(blocks, unvisited);
    std::vector<std::size_t> low(blocks, 0u);
    std::vector<bool> onStack(blocks, false);
    std::vector<bool> cyclic(blocks, false);
    std::vector<std::size_t> stack;
    std::vector<std::pair<std::size_t, std::size_t>> frames;
    std::size_t counter = 0;
    for (std::size_t root = 0; root < blocks; ++root) {
        if (order[root] != unvisited) continue;
        frames.push_back({root, 0u});
        while (!frames.empty()) {
            auto& [block, next] = frames.back();
            if (next == 0u && order[block] == unvisited) {
                order[block] = low[block] = counter++;
                stack.push_back(block);
                onStack[block] = true;
            }
            if (next < successors[block].size()) {
                const auto successor = successors[block][next++];
                if (successor == block) cyclic[block] = true;
                if (order[successor] == unvisited) frames.push_back({successor, 0u});
                else if (onStack[successor]) low[block] = std::min(low[block], order[successor]);
                continue;
            }
            const auto done = block;
            frames.pop_back();
            if (!frames.empty()) low[frames.back().first] = std::min(low[frames.back().first], low[done]);
            if (low[done] != order[done]) continue;
            std::vector<std::size_t> component;
            for (;;) {
                const auto member = stack.back();
                stack.pop_back();
                onStack[member] = false;
                component.push_back(member);
                if (member == done) break;
            }
            if (component.size() > 1u) {
                for (const auto member : component) cyclic[member] = true;
            }
        }
    }
    for (const auto& [callee, calls] : sites) {
        if (calls.size() > 1u || cyclic[calls.front()]) shared.insert(callee);
    }
}

}

SpirvLocalMemoryUsers CountLocalMemoryUsers(std::span<const std::uint32_t> module) {
    struct Variable {
        std::uint32_t storage = 0;
        bool dynamic = false;
        bool passed = false;
        bool volatileAccess = false;
        std::set<std::uint32_t> functions;
    };
    std::set<std::uint32_t> constants;
    std::unordered_map<std::uint32_t, Variable> variables;
    std::unordered_map<std::uint32_t, std::uint32_t> roots;
    const auto rootOf = [&](std::uint32_t id) -> Variable* {
        if (const auto found = variables.find(id); found != variables.end()) return &found->second;
        if (const auto found = roots.find(id); found != roots.end()) return &variables.at(found->second);
        return nullptr;
    };
    std::uint32_t function = 0;
    for (std::size_t offset = 5; offset < module.size(); offset += module[offset] >> spv::WordCountShift) {
        const auto count = module[offset] >> spv::WordCountShift;
        if (count == 0u || offset + count > module.size()) throw std::runtime_error("CountLocalMemoryUsers found a truncated instruction");
        const auto op = module.subspan(offset, count);
        const auto opcode = op[0] & spv::OpCodeMask;
        switch (opcode) {
        case spv::OpConstant:
        case spv::OpConstantNull:
        case spv::OpConstantTrue:
        case spv::OpConstantFalse:
        case spv::OpConstantComposite:
        case spv::OpSpecConstant:
        case spv::OpSpecConstantTrue:
        case spv::OpSpecConstantFalse:
        case spv::OpSpecConstantComposite:
        case spv::OpSpecConstantOp:
            if (count > 2u) constants.insert(op[2]);
            continue;
        case spv::OpFunction:
            function = op[2];
            continue;
        case spv::OpFunctionEnd:
            function = 0;
            continue;
        case spv::OpVariable:
            if (count > 3u && (op[3] == spv::StorageClassFunction || op[3] == spv::StorageClassPrivate)) variables[op[2]].storage = op[3];
            continue;
        default:
            break;
        }
        if (function == 0u) continue;
        if (opcode == spv::OpAccessChain || opcode == spv::OpInBoundsAccessChain || opcode == spv::OpPtrAccessChain || opcode == spv::OpInBoundsPtrAccessChain || opcode == spv::OpCopyObject) {
            if (count > 3u) {
                if (auto* root = rootOf(op[3]); root != nullptr) {
                    roots[op[2]] = variables.contains(op[3]) ? op[3] : roots.at(op[3]);
                    for (std::size_t index = 4; index < count; ++index) {
                        if (!constants.contains(op[index])) root->dynamic = true;
                    }
                    root->functions.insert(function);
                }
            }
            continue;
        }
        if (opcode == spv::OpFunctionCall) {
            for (std::size_t index = 4; index < count; ++index) {
                if (auto* root = rootOf(op[index]); root != nullptr) {
                    root->passed = true;
                    root->functions.insert(function);
                }
            }
            continue;
        }
        if (opcode == spv::OpLoad && count > 3u) {
            if (auto* root = rootOf(op[3]); root != nullptr) {
                root->functions.insert(function);
                if (count > 4u && (op[4] & spv::MemoryAccessVolatileMask) != 0u) root->volatileAccess = true;
            }
            continue;
        }
        if (opcode == spv::OpStore && count > 2u) {
            if (auto* root = rootOf(op[1]); root != nullptr) {
                root->functions.insert(function);
                if (count > 3u && (op[3] & spv::MemoryAccessVolatileMask) != 0u) root->volatileAccess = true;
            }
            continue;
        }
        for (std::size_t index = 1; index < count; ++index) {
            if (auto* root = rootOf(op[index]); root != nullptr) root->functions.insert(function);
        }
    }
    SpirvLocalMemoryUsers users;
    for (const auto& [id, variable] : variables) {
        const bool shared = variable.storage == spv::StorageClassPrivate && variable.functions.size() > 1u;
        users.dynamicallyIndexed += variable.dynamic ? 1u : 0u;
        users.passedToCalls += variable.passed ? 1u : 0u;
        users.volatileAccess += variable.volatileAccess ? 1u : 0u;
        users.sharedPrivate += shared ? 1u : 0u;
        users.total += variable.dynamic || variable.passed || variable.volatileAccess || shared ? 1u : 0u;
    }
    return users;
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

void SpirvModule::LocalizePrivateVariables(std::span<const std::uint32_t> variables) {
    if (inHelperFunction) throw std::runtime_error("SpirvModule::LocalizePrivateVariables called inside a helper function");
    struct Local {
        std::uint32_t pointer = 0;
        std::uint32_t initializer = 0;
    };
    std::map<std::uint32_t, Local> locals;
    for (const auto variable : variables) {
        if (variable != 0u) locals.emplace(variable, Local{});
    }
    if (locals.empty()) return;
    const auto each = [](const std::vector<std::uint32_t>& section, auto&& visit) {
        for (std::size_t offset = 0; offset < section.size(); offset += section[offset] >> spv::WordCountShift) {
            if ((section[offset] >> spv::WordCountShift) == 0u) throw std::runtime_error("SpirvModule::LocalizePrivateVariables found an empty instruction");
            visit(offset, std::span<const std::uint32_t>(section.data() + offset, section[offset] >> spv::WordCountShift));
        }
    };
    std::map<std::uint32_t, std::uint32_t> pointees;
    for (const auto* section : {&declarations, &typeDeclarations}) {
        each(*section, [&](std::size_t, std::span<const std::uint32_t> words) {
            if ((words[0] & spv::OpCodeMask) == spv::OpTypePointer && words.size() == 4u && words[2] == spv::StorageClassPrivate) pointees.emplace(words[1], words[3]);
        });
    }
    std::vector<std::uint32_t> kept;
    kept.reserve(globalVariables.size());
    each(globalVariables, [&](std::size_t, std::span<const std::uint32_t> words) {
        const auto found = (words[0] & spv::OpCodeMask) == spv::OpVariable && words.size() >= 4u ? locals.find(words[2]) : locals.end();
        if (found == locals.end()) {
            kept.insert(kept.end(), words.begin(), words.end());
            return;
        }
        if (words[3] != spv::StorageClassPrivate || !pointees.contains(words[1])) throw std::runtime_error("SpirvModule::LocalizePrivateVariables was given a variable that is not a Private scalar");
        found->second.pointer = Type(spv::OpTypePointer, static_cast<std::uint32_t>(spv::StorageClassFunction), pointees.at(words[1]));
        found->second.initializer = words.size() > 4u ? words[4] : 0u;
    });
    for (const auto& [variable, local] : locals) {
        if (local.pointer == 0u) throw std::runtime_error("SpirvModule::LocalizePrivateVariables was given a variable it does not declare");
    }
    globalVariables = std::move(kept);
    for (auto* section : {&functionInstructions, &helperFunctionInstructions}) {
        std::map<std::uint32_t, std::map<std::uint32_t, std::uint32_t>> copies;
        std::uint32_t function = 0;
        each(*section, [&](std::size_t, std::span<const std::uint32_t> words) {
            const auto opcode = words[0] & spv::OpCodeMask;
            if (opcode == spv::OpFunction) function = words[2];
            const auto pointer = opcode == spv::OpLoad && words.size() >= 4u ? words[3] : opcode == spv::OpStore && words.size() >= 3u ? words[1] : 0u;
            if (pointer != 0u && locals.contains(pointer)) copies[function].emplace(pointer, 0u);
        });
        for (auto& [owner, renames] : copies) {
            for (auto& [variable, copy] : renames) copy = AllocateId();
        }
        std::vector<std::uint32_t> rewritten;
        rewritten.reserve(section->size());
        const std::map<std::uint32_t, std::uint32_t>* renames = nullptr;
        bool entry = false;
        each(*section, [&](std::size_t, std::span<const std::uint32_t> words) {
            const auto opcode = words[0] & spv::OpCodeMask;
            const auto start = rewritten.size();
            rewritten.insert(rewritten.end(), words.begin(), words.end());
            if (opcode == spv::OpFunction) {
                const auto found = copies.find(words[2]);
                renames = found == copies.end() ? nullptr : &found->second;
                entry = true;
                return;
            }
            if (renames == nullptr) return;
            if (opcode == spv::OpLabel && entry) {
                entry = false;
                for (const auto& [variable, copy] : *renames) {
                    const auto& local = locals.at(variable);
                    if (local.initializer != 0u) appendInstruction(rewritten, spv::OpVariable, local.pointer, copy, spv::StorageClassFunction, local.initializer);
                    else appendInstruction(rewritten, spv::OpVariable, local.pointer, copy, spv::StorageClassFunction);
                }
                return;
            }
            const auto operand = opcode == spv::OpLoad && words.size() >= 4u ? 3u : opcode == spv::OpStore && words.size() >= 3u ? 1u : 0u;
            if (operand == 0u) return;
            if (const auto found = renames->find(words[operand]); found != renames->end()) rewritten[start + operand] = found->second;
        });
        *section = std::move(rewritten);
    }
    for (auto* section : {&debug, &annotations}) {
        std::vector<std::uint32_t> kept;
        each(*section, [&](std::size_t, std::span<const std::uint32_t> words) {
            const auto opcode = words[0] & spv::OpCodeMask;
            const bool targets = (opcode == spv::OpName || opcode == spv::OpDecorate) && words.size() >= 2u && locals.contains(words[1]);
            if (!targets) kept.insert(kept.end(), words.begin(), words.end());
        });
        *section = std::move(kept);
    }
}

void SpirvModule::ReturnSharedResultsThroughPointers() {
    if (inHelperFunction) throw std::runtime_error("SpirvModule::ReturnSharedResultsThroughPointers called inside a helper function");
    if (unpatchedPhiIncomings != 0) throw std::runtime_error("SpirvModule::ReturnSharedResultsThroughPointers called with unpatched OpPhi incoming pairs");
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
            if ((section[offset] >> spv::WordCountShift) == 0u) throw std::runtime_error("SpirvModule::ReturnSharedResultsThroughPointers found an empty instruction");
            visit(std::span<const std::uint32_t>(section.data() + offset, section[offset] >> spv::WordCountShift));
        }
    };
    std::set<std::uint32_t> outOfLine;
    for (const auto* section : sections) {
        instructions(*section, [&](std::span<const std::uint32_t> words) {
            if ((words[0] & spv::OpCodeMask) == spv::OpFunction && words[1] != voidType && (words[3] & spv::FunctionControlDontInlineMask) != 0u) outOfLine.insert(words[2]);
        });
    }
    if (outOfLine.empty()) return;
    std::set<std::uint32_t> shared;
    for (const auto* section : sections) {
        std::size_t begin = 0;
        for (std::size_t offset = 0; offset < section->size(); offset += (*section)[offset] >> spv::WordCountShift) {
            if (((*section)[offset] & spv::OpCodeMask) != spv::OpFunctionEnd) continue;
            const auto end = offset + ((*section)[offset] >> spv::WordCountShift);
            AddSharedCallees(std::span<const std::uint32_t>(section->data() + begin, end - begin), outOfLine, shared);
            begin = end;
        }
    }
    if (shared.empty()) return;
    std::map<std::uint32_t, Rewrite> rewrites;
    for (const auto* section : sections) {
        std::uint32_t function = 0;
        std::vector<std::uint32_t> types;
        instructions(*section, [&](std::span<const std::uint32_t> words) {
            const auto opcode = words[0] & spv::OpCodeMask;
            if (opcode == spv::OpFunction) {
                function = shared.contains(words[2]) ? words[2] : 0u;
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
