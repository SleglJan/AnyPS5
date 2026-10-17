#include "Recompiler.hpp"
#include "ControlFlow/GraphBuilder.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "Translation/InstructionTranslator.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint32_t MoveM0Two = 0xbefc0382u;
constexpr std::uint32_t MoveM0GroupX = 0xbefc0304u;
constexpr std::uint32_t MoveV4One = 0x7e0802f2u;
constexpr std::uint32_t MoveV5Two = 0x7e0a02f4u;
constexpr std::uint32_t MoveV6Four = 0x7e0c02f6u;
constexpr std::uint32_t MovrelsV1V4 = 0x7e028704u;
constexpr std::uint32_t MovreldV4V0 = 0x7e088500u;
constexpr std::uint32_t MoveV1V5 = 0x7e020305u;
constexpr std::uint32_t EndProgram = 0xbf810000u;
constexpr std::uint32_t MoveV2Zero = 0x7e040280u;
constexpr std::uint32_t MoveV3Zero = 0x7e060280u;
constexpr std::uint32_t WritelaneV4 = 0xd7610004u;
constexpr std::uint32_t FromS8Lane0 = 0x00010008u;
constexpr std::uint32_t ReadlaneS8 = 0xd7600008u;
constexpr std::uint32_t FromV4Lane0 = 0x00010104u;
constexpr std::uint32_t ImageSampleLz2d = 0xf09c0f08u;
constexpr std::uint32_t SampleV8V2S0S8 = 0x00400802u;
constexpr std::uint32_t AddV8V1V8 = 0x4a101101u;
constexpr std::uint32_t BufferStoreDword = 0xe0700000u;
constexpr std::uint32_t StoreV8WithS12 = 0x80030800u;
constexpr std::uint32_t MovrelsV3V4 = 0x7e068704u;
constexpr std::uint32_t CompareGroupXZero = 0xbf068004u;
constexpr std::uint32_t BranchSccOverOne = 0xbf850001u;
constexpr std::uint32_t MoveS5Three = 0xbe850383u;
constexpr std::uint32_t AddV2V1V2 = 0x4a040501u;
constexpr std::uint32_t AddV4OneV4 = 0x4a080881u;
constexpr std::uint32_t SubtractS5One = 0x80858105u;
constexpr std::uint32_t CompareS5NotZero = 0xbf078005u;
constexpr std::uint32_t BranchSccBackSix = 0xbf85fffau;

constexpr std::uint32_t MoveVector(std::uint32_t reg, std::uint32_t value) {
    return 0x7e000200u | (reg << 17u) | (128u + value);
}

struct WindowAccess {
    bool store = false;
    std::optional<std::uint32_t> slot;
    std::optional<std::uint32_t> value;

    bool operator==(const WindowAccess& other) const = default;
};

struct Shape {
    std::size_t functionArrays = 0;
    std::uint32_t arrayLength = 0;
    std::size_t accessChains = 0;
    std::size_t slots = 0;
    std::size_t switchedAccesses = 0;
    std::size_t selects = 0;
    std::vector<WindowAccess> windowAccesses;
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint32_t> RecompileProgram(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData) {
    const std::array<std::uint32_t, 8> capabilities{spv::CapabilityShader, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess};
    const std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x20000u, code, 0, {}};
    request.context.waveSize = 32;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 4u, {true, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00403000u;
    request.target.spirvVersion = 0x00010600u;
    request.target.subgroupSize = 32;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.maxWorkgroupSize = {1024u, 1024u, 64u};
    request.target.maxWorkgroupInvocations = 1024;
    request.target.maxWorkgroupSharedMemoryBytes = 49152;
    request.layout = {0, 0, 0, 128};
    request.useCache = false;
    auto result = Recompile(request);
    Require(!result.spirv.empty(), "the relative-addressing program did not recompile");
    return std::vector<std::uint32_t>(result.spirv.Words().begin(), result.spirv.Words().end());
}

std::vector<std::uint32_t> RecompileProgram(std::span<const std::uint32_t> code) {
    const std::array<std::uint32_t, 4> userData{0x10000000u, 0u, 0x40u, 0x31016facu};
    return RecompileProgram(code, userData);
}

Shape Scan(std::span<const std::uint32_t> words) {
    Shape shape;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> constants;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> arrays;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> functionPointers;
    std::vector<std::uint32_t> windowVariables;
    std::unordered_map<std::uint32_t, std::optional<std::uint32_t>> windowElements;
    std::unordered_map<std::uint32_t, std::uint32_t> slots;
    bool afterWindow = false;
    std::uint32_t pendingMerge = 0;
    std::vector<std::uint32_t> armLabels;
    std::uint32_t armsMerge = 0;
    bool inArm = false;
    bool armRecorded = false;
    const auto constantValue = [&](std::uint32_t id) -> std::optional<std::uint32_t> {
        for (const auto& [constant, value] : constants) {
            if (constant == id) return value;
        }
        return std::nullopt;
    };
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> spv::WordCountShift;
        const auto op = static_cast<spv::Op>(words[cursor] & spv::OpCodeMask);
        Require(count != 0 && cursor + count <= words.size(), "truncated SPIR-V instruction");
        if (op == spv::OpConstant && count == 4) constants.emplace_back(words[cursor + 2], words[cursor + 3]);
        if (op == spv::OpTypeArray) arrays.emplace_back(words[cursor + 1], words[cursor + 3]);
        if (op == spv::OpTypePointer && words[cursor + 2] == spv::StorageClassFunction) functionPointers.emplace_back(words[cursor + 1], words[cursor + 3]);
        if (op == spv::OpVariable && words[cursor + 3] == spv::StorageClassFunction && afterWindow && count == 5) {
            slots.emplace(words[cursor + 2], static_cast<std::uint32_t>(slots.size()));
            cursor += count;
            continue;
        }
        if (op != spv::OpVariable) afterWindow = false;
        if (op == spv::OpVariable && words[cursor + 3] == spv::StorageClassFunction) {
            afterWindow = false;
            for (const auto& [pointer, pointee] : functionPointers) {
                if (pointer != words[cursor + 1]) continue;
                for (const auto& [array, length] : arrays) {
                    if (array != pointee) continue;
                    ++shape.functionArrays;
                    windowVariables.push_back(words[cursor + 2]);
                    afterWindow = true;
                    for (const auto& [constant, value] : constants) {
                        if (constant == length) shape.arrayLength = value;
                    }
                }
            }
        }
        if (op == spv::OpAccessChain) {
            for (const auto variable : windowVariables) {
                if (words[cursor + 3] != variable) continue;
                ++shape.accessChains;
                windowElements[words[cursor + 2]] = count == 5 ? constantValue(words[cursor + 4]) : std::nullopt;
            }
        }
        if (op == spv::OpStore && windowElements.contains(words[cursor + 1])) shape.windowAccesses.push_back({true, windowElements[words[cursor + 1]], constantValue(words[cursor + 2])});
        if (op == spv::OpLoad && windowElements.contains(words[cursor + 3])) shape.windowAccesses.push_back({false, windowElements[words[cursor + 3]], std::nullopt});
        if (op == spv::OpSelectionMerge) pendingMerge = words[cursor + 1];
        if (op == spv::OpSwitch && armsMerge == 0u) {
            armLabels.clear();
            for (std::size_t operand = 4; operand < count; operand += 2u) armLabels.push_back(words[cursor + operand]);
            armsMerge = pendingMerge;
            armRecorded = false;
        }
        if (op == spv::OpLabel) {
            if (words[cursor + 1] == armsMerge) {
                armsMerge = 0u;
                inArm = false;
            } else if (armsMerge != 0u && std::ranges::find(armLabels, words[cursor + 1]) != armLabels.end()) {
                inArm = true;
            }
        }
        const auto slotAccess = [&](std::uint32_t variable, bool store, std::optional<std::uint32_t> value) {
            const auto found = slots.find(variable);
            if (found == slots.end()) return;
            if (!inArm) {
                shape.windowAccesses.push_back({store, found->second, value});
                return;
            }
            if (armRecorded) return;
            armRecorded = true;
            ++shape.switchedAccesses;
            shape.windowAccesses.push_back({store, std::nullopt, value});
        };
        if (op == spv::OpStore) slotAccess(words[cursor + 1], true, constantValue(words[cursor + 2]));
        if (op == spv::OpLoad) slotAccess(words[cursor + 3], false, std::nullopt);
        if (op == spv::OpSelect) ++shape.selects;
        cursor += count;
    }
    shape.slots = slots.size();
    return shape;
}

bool WindowAccessesAre(std::span<const std::uint32_t> code, std::initializer_list<WindowAccess> expected) {
    const auto accesses = Scan(RecompileProgram(code)).windowAccesses;
    if (std::ranges::equal(accesses, expected)) return true;
    std::string text;
    for (const auto& access : accesses) {
        text += access.store ? " store[" : " load[";
        text += access.slot ? std::to_string(*access.slot) : std::string("index");
        text += access.store ? (access.value ? "]=" + std::to_string(*access.value) : std::string("]=value")) : std::string("]");
    }
    std::fprintf(stderr, "  window accesses:%s\n", text.c_str());
    return false;
}

IrProgram TranslateStage(ShaderStageKind stage, std::uint32_t waveSize, std::uint32_t base = 0u, std::uint32_t fetchStart = 0u, std::uint32_t fetchCount = 0u) {
    ShaderComputeInputInfo compute{};
    compute.threadIdsNum = 3;
    compute.threadsNum[0] = waveSize;
    ShaderVertexInputInfo vertex{};
    vertex.mesh.threadsNum[0] = waveSize;
    vertex.mesh.inputPrimitive = 1u;
    vertex.mesh.primitivesPerGroup = 1u;
    vertex.mesh.verticesPerGroup = 1u;
    vertex.mesh.esgsItemSize = 1u;
    ShaderPixelInputInfo pixel{};
    std::uint32_t next = 0u;
    for (const auto input : {PixelInput::PerspectiveCenter, PixelInput::PerspectiveCentroid, PixelInput::LinearCenter, PixelInput::LinearCentroid,
                            PixelInput::PositionX, PixelInput::PositionY, PixelInput::PositionZ, PixelInput::PositionW,
                            PixelInput::FrontFace, PixelInput::Ancillary, PixelInput::LineStipple, PixelInput::PositionFixedPoint}) {
        pixel.psInputVgpr[static_cast<std::uint32_t>(input)] = next;
        next += PixelInputVgprCount(input);
    }
    std::vector<std::uint32_t> code{MoveM0Two};
    if (fetchCount != 0u) {
        code.push_back(0xbe802102u);
        code.push_back(0x7e008700u | base);
        code.push_back(0x7e1002f2u);
        vertex.resourcesNum = 1;
        vertex.resourcesDst[0] = {static_cast<int>(fetchStart), static_cast<int>(fetchCount), 0, 0u};
    }
    code.push_back(0x7e008700u | base);
    code.push_back(EndProgram);
    const auto decoded = RdnaInstructionDecoder{}.Decode(code);
    if (fetchCount != 0u) {
        const auto& call = decoded.instructions.at(1);
        Require(call.op == RdnaOpcode::SSwappcB64 && call.family == RdnaInstructionFamily::SOP1 && call.opcodeId == 0x21u &&
                call.destination.kind == RdnaOperandKind::ScalarRegister && call.destination.reg == 0u &&
                call.source0.kind == RdnaOperandKind::ScalarRegister && call.source0.reg == 2u,
                "external fetch fixture did not decode s_swappc_b64 s[0:1], s[2:3]");
    }
    const SwappcInfo swappc{fetchCount != 0u, 0u, 8u};
    const auto graph = GraphBuilder{}.Build(decoded, &swappc);
    Require(graph.hasFetchCall == (fetchCount != 0u), "stage fixture did not identify its external fetch call");
    TranslateOptions options{};
    options.stage = stage;
    options.waveSize = waveSize;
    options.userDataCount = 8u;
    options.inputInfo = {&vertex, &pixel, &compute};
    return InstructionTranslator{}.Translate(decoded, graph, options);
}

void RequireWindow(const IrProgram& program, std::uint32_t base, std::uint32_t limit, std::span<const std::uint32_t> initialized) {
    Require(program.Metadata().vectorWindowBase == base && program.Metadata().vectorWindowSize == limit - base, "relative window does not include exactly the initialized stage registers");
    std::array<bool, NumVectorRegs> stored{};
    for (const auto* instruction : program.BlockOrder().front()->Instructions()) {
        if (instruction->Opcode() == IrOpcode::StoreVectorWindow) {
            Require(instruction->Argument(0)->HasImmediate(), "stage input uses a dynamic window index");
            const auto index = base + instruction->Argument(0)->ImmediateU32();
            Require(index < limit, "stage input lies past the window");
            stored[index] = true;
        } else if (instruction->Opcode() == IrOpcode::SetVectorRegister) {
            const auto index = instruction->Argument(0)->Register().index;
            Require(index < base || !program.Metadata().vectorWindowIndexedWrites, "stage input escaped the relative window into SSA");
            if (index < base) stored[index] = true;
        }
    }
    for (const auto index : initialized) Require(stored[index], "stage input v" + std::to_string(index) + " was not initialized");
    bool bounded = false;
    for (const auto& block : program.Blocks()) {
        for (const auto* instruction : block->Instructions()) {
            if (instruction->Opcode() == IrOpcode::ULessThan32 && instruction->Argument(1)->HasImmediate() && instruction->Argument(1)->ImmediateU32() == limit - base) bounded = true;
        }
    }
    Require(bounded, "relative access did not use the final stage window bound");
}

template<typename TAction>
bool RunStage(const char* name, TAction action) {
    try {
        action();
        std::printf("vector window stage inputs: %s passed\n", name);
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "vector window stage inputs: %s: %s\n", name, error.what());
        return false;
    }
}

bool StageInputs() {
    bool passed = true;
    const auto stage = [&](const char* name, ShaderStageKind kind, std::uint32_t limit, std::initializer_list<std::uint32_t> initialized) {
        passed &= RunStage(name, [&] {
            for (const auto wave : {32u, 64u}) RequireWindow(TranslateStage(kind, wave), 0u, limit, {initialized.begin(), initialized.size()});
        });
    };
    stage("compute", ShaderStageKind::Compute, 3u, {0u, 1u, 2u});
    stage("vertex", ShaderStageKind::Vertex, 9u, {5u, 8u});
    stage("local", ShaderStageKind::Local, 6u, {2u, 3u, 5u});
    stage("tessellation control", ShaderStageKind::TessellationControl, 2u, {0u, 1u});
    stage("tessellation evaluation", ShaderStageKind::TessellationEvaluation, 9u, {0u, 5u, 6u, 7u, 8u});
    stage("mesh", ShaderStageKind::Mesh, 9u, {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u});
    stage("pixel", ShaderStageKind::Pixel, 16u, {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u});
    passed &= RunStage("SSA prefix", [] {
        const std::array<std::uint32_t, 3> local{2u, 3u, 5u};
        const std::array<std::uint32_t, 2> vertex{5u, 8u};
        for (const auto wave : {32u, 64u}) {
            RequireWindow(TranslateStage(ShaderStageKind::Local, wave, 4u), 4u, 6u, local);
            RequireWindow(TranslateStage(ShaderStageKind::Vertex, wave, 16u), 16u, 17u, vertex);
        }
    });
    passed &= RunStage("external fetch", [] {
        const std::array<std::uint32_t, 2> vertex{5u, 8u};
        for (const auto wave : {32u, 64u}) {
            for (const auto start : {20u, 252u}) {
                for (const auto count : {1u, 2u, 3u, 4u}) {
                    const auto program = TranslateStage(ShaderStageKind::Vertex, wave, 0u, start, count);
                    RequireWindow(program, 0u, start + count, vertex);
                    std::uint32_t fetched = 0u;
                    std::uint32_t boundedReads = 0u;
                    for (const auto& block : program.Blocks()) {
                        for (const auto* instruction : block->Instructions()) {
                            if (instruction->Opcode() == IrOpcode::StoreVectorWindow) {
                                const auto* value = instruction->Argument(1);
                                if (value->Opcode() == IrOpcode::SelectU32) value = value->Argument(1);
                                if (value->Opcode() == IrOpcode::GetAttribute) {
                                    Require(instruction->Argument(0)->ImmediateU32() == start + fetched && value->Argument(1)->ImmediateU32() == fetched,
                                            "external fetch did not initialize its entire window span");
                                    ++fetched;
                                }
                            }
                            if (instruction->Opcode() == IrOpcode::ULessThan32 && instruction->Argument(1)->HasImmediate() && instruction->Argument(1)->ImmediateU32() == start + count) ++boundedReads;
                        }
                    }
                    Require(fetched == count && boundedReads == 2u, "relative accesses to the external fetch disagree about the window");
                }
            }
        }
    });
    if (passed) std::puts("Vector window stage input and fetch tests passed");
    return passed;
}

bool run(const char* name, bool (*test)()) {
    try {
        if (test()) return true;
        std::fprintf(stderr, "%s\n", name);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: %s\n", name, error.what());
    }
    return false;
}

}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--stage-inputs-only") return StageInputs() ? 0 : 1;
    if (argc != 1) return 2;
    bool passed = true;

    passed &= run("observable relative reads and writes index a function-scope window", [] {
        const std::array<std::uint32_t, 12> code{MoveM0GroupX, MoveV4One, MoveV5Two, MoveV6Four, MovrelsV1V4,
            0xe0700000u, 0x80000100u, MovreldV4V0, MoveV1V5, 0xe0700004u, 0x80000100u, EndProgram};
        const auto shape = Scan(RecompileProgram(code));
        const bool indexed = shape.accessChains >= 2 || (shape.slots == 3 && shape.switchedAccesses >= 2);
        if (shape.functionArrays != 1 || shape.arrayLength != 3 || !indexed) std::fprintf(stderr, "  window shape: arrays %zu length %u access chains %zu slots %zu switched accesses %zu selects %zu\n", shape.functionArrays, shape.arrayLength, shape.accessChains, shape.slots, shape.switchedAccesses, shape.selects);
        return shape.functionArrays == 1 && shape.arrayLength == 3 && indexed;
    });

    passed &= run("a program without relative addressing keeps its registers in SSA values", [] {
        const std::array<std::uint32_t, 7> code{MoveV4One, MoveV5Two, MoveV6Four, MoveV1V5, 0xe0700000u, 0x80000100u, EndProgram};
        const auto shape = Scan(RecompileProgram(code));
        return shape.functionArrays == 0 && shape.accessChains == 0 && shape.slots == 0;
    });

    passed &= run("a sampler dword spilled into a relatively read register keeps its definition through v_readlane", [] {
        const std::array<std::uint32_t, 14> code{MoveM0GroupX, MoveV2Zero, MoveV3Zero, WritelaneV4, FromS8Lane0, MovrelsV1V4, ReadlaneS8, FromV4Lane0, ImageSampleLz2d, SampleV8V2S0S8, AddV8V1V8, BufferStoreDword, StoreV8WithS12, EndProgram};
        const std::array<std::uint32_t, 16> userData{};
        const auto shape = Scan(RecompileProgram(code, userData));
        return shape.functionArrays == 1;
    });

    passed &= run("a register rewritten before a relative read is stored once, and not after the last read", [] {
        const std::array<std::uint32_t, 11> code{MoveM0GroupX, MoveVector(4, 1), MoveVector(5, 2), MoveVector(4, 3), MoveVector(5, 4), MovrelsV1V4, 0xe0700000u, 0x80000100u, MoveVector(4, 5), MoveVector(5, 6), EndProgram};
        return WindowAccessesAre(code, {{true, 0u, 3u}, {true, 1u, 4u}, {false, std::nullopt, std::nullopt}});
    });

    passed &= run("a register written in one arm of a branch is stored in the arm for the relative read after the merge", [] {
        const std::array<std::uint32_t, 11> code{MoveM0GroupX, MoveVector(4, 1), MoveVector(5, 7), MoveVector(4, 2), CompareGroupXZero, BranchSccOverOne, MoveVector(4, 3), MovrelsV1V4, 0xe0700000u, 0x80000100u, EndProgram};
        return WindowAccessesAre(code, {{true, 1u, 7u}, {true, 0u, 2u}, {true, 0u, 3u}, {false, std::nullopt, std::nullopt}});
    });

    passed &= run("a register rewritten in a loop is stored for the relative read of the next iteration", [] {
        const std::array<std::uint32_t, 18> code{MoveM0GroupX, MoveVector(4, 1), MoveVector(5, 2), MoveS5Three, MovrelsV1V4, AddV2V1V2, AddV4OneV4, SubtractS5One, CompareS5NotZero, BranchSccBackSix,
            MoveVector(4, 8), MovrelsV3V4, 0xe0700000u, 0x80000200u, 0xe0700004u, 0x80000300u, MoveVector(5, 9), EndProgram};
        return WindowAccessesAre(code, {{true, 0u, 1u}, {true, 1u, 2u}, {false, std::nullopt, std::nullopt}, {true, 0u, std::nullopt}, {true, 0u, 8u}, {false, std::nullopt, std::nullopt}});
    });

    if (passed) std::puts("Vector window shape and SSA control tests passed");
    return passed ? 0 : 1;
}
