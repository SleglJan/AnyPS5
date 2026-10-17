#include "DeviceTableFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "SpirvBackend/SpirvModule.hpp"
#include "VulkanTestDevice.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using namespace AgcDriver::Graphics;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t InputWords = 16;
constexpr std::uint32_t Magic = 0x5a17c0deu;
constexpr std::uint32_t PairLow = 0x11112222u;
constexpr std::uint32_t PairHigh = 0x33334444u;
constexpr std::uint32_t Count = 0x55556666u;
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t LoopRounds = 4;
constexpr VkShaderStageFlags Stages = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT;

enum class Storage { Private, PrivateInitialized, Function, FunctionInitialized };
enum class Reader { Main, Callee, MainLoop, CalleeLoop };
constexpr std::uint32_t HeaderWords = 8;
constexpr std::uint32_t HeaderTag = 4;
constexpr std::uint32_t Grants = (InputWords - HeaderWords) / 4u;

enum class Stage { Compute, Mesh };
enum class Guard { ArrayLength, BufferValue, Header };


void Check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(result));
}

std::string Name(Storage storage, Reader reader, Guard guard) {
    std::string name = storage == Storage::Private ? "Private" : storage == Storage::PrivateInitialized ? "Private with an initializer" : storage == Storage::Function ? "Function" : "Function with an initializer";
    name += reader == Reader::Main ? ", read in main" : reader == Reader::Callee ? ", read in a callee" : reader == Reader::MainLoop ? ", read in a loop in main" : ", read in a callee called in a loop";
    name += guard == Guard::ArrayLength ? ", guarded by the buffer's length" : guard == Guard::BufferValue ? ", guarded by a buffer value" : ", a table header checked under the buffer's length";
    return name;
}

std::vector<std::uint32_t> Module(Stage stage, Storage storage, Reader reader, Guard guard, bool wrapped) {
    ShaderRecompiler::SpirvModule module;
    const bool mesh = stage == Stage::Mesh;
    if (mesh) {
        module.RequireVersion(0x00010400u);
        module.EmitCapability(spv::CapabilityMeshShadingEXT);
        module.EmitExtension("SPV_EXT_mesh_shader");
    }
    module.EmitCapability(spv::CapabilityShader);
    module.EmitCapability(spv::CapabilityInt64);
    module.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto boolType = module.Type(spv::OpTypeBool);
    const auto u32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto u64 = module.Type(spv::OpTypeInt, 64u, 0u);
    const auto uvec2 = module.Type(spv::OpTypeVector, u32, 2u);
    const auto uvec3 = module.Type(spv::OpTypeVector, u32, 3u);
    const auto uvec4 = module.Type(spv::OpTypeVector, u32, 4u);
    const auto words = module.Type(spv::OpTypeRuntimeArray, u32);
    module.AddAnnotation(spv::OpDecorate, words, spv::DecorationArrayStride, 4u);
    const auto block = module.Type(spv::OpTypeStruct, words);
    module.AddAnnotation(spv::OpDecorate, block, spv::DecorationBlock);
    module.AddAnnotation(spv::OpMemberDecorate, block, 0u, spv::DecorationOffset, 0u);
    const auto blockPointer = module.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, block);
    const auto wordPointer = module.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, u32);
    const auto inputPointer = module.Type(spv::OpTypePointer, spv::StorageClassInput, uvec3);
    const bool privateClass = storage == Storage::Private || storage == Storage::PrivateInitialized;
    const bool initialized = storage == Storage::PrivateInitialized || storage == Storage::FunctionInitialized;
    const auto variableClass = privateClass ? spv::StorageClassPrivate : spv::StorageClassFunction;
    const auto u32Pointer = module.Type(spv::OpTypePointer, variableClass, u32);
    const auto u64Pointer = module.Type(spv::OpTypePointer, variableClass, u64);
    const auto input = module.DefineGlobalVariable(blockPointer, spv::StorageClassStorageBuffer);
    const auto output = module.DefineGlobalVariable(blockPointer, spv::StorageClassStorageBuffer);
    const auto invocation = module.DefineGlobalVariable(inputPointer, spv::StorageClassInput);
    module.AddAnnotation(spv::OpDecorate, input, spv::DecorationDescriptorSet, 0u);
    module.AddAnnotation(spv::OpDecorate, input, spv::DecorationBinding, 0u);
    module.AddAnnotation(spv::OpDecorate, output, spv::DecorationDescriptorSet, 0u);
    module.AddAnnotation(spv::OpDecorate, output, spv::DecorationBinding, 1u);
    module.AddAnnotation(spv::OpDecorate, invocation, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
    const auto constant = [&](std::uint32_t value) { return module.Constant(spv::OpConstant, u32, value); };
    const auto zero = constant(0u);
    const auto one = constant(1u);
    const auto zero64 = module.Constant(spv::OpConstant, u64, 0u, 0u);
    std::array<std::uint32_t, 3> globals{};
    if (storage == Storage::Private) {
        globals = {module.DefineGlobalVariable(u32Pointer, spv::StorageClassPrivate), module.DefineGlobalVariable(u64Pointer, spv::StorageClassPrivate), module.DefineGlobalVariable(u32Pointer, spv::StorageClassPrivate)};
    } else if (storage == Storage::PrivateInitialized) {
        globals = {module.DefineInitializedGlobalVariable(u32Pointer, spv::StorageClassPrivate, zero), module.DefineInitializedGlobalVariable(u64Pointer, spv::StorageClassPrivate, zero64), module.DefineInitializedGlobalVariable(u32Pointer, spv::StorageClassPrivate, zero)};
    }
    const auto id = [&] { return module.AllocateId(); };
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = id();
        module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto word = [&](std::uint32_t buffer, std::uint32_t index) {
        const auto pointer = id();
        module.AddFunction(spv::OpAccessChain, wordPointer, pointer, buffer, zero, index);
        return pointer;
    };
    const auto binary = [&](std::uint32_t opcode, std::uint32_t type, std::uint32_t left, std::uint32_t right) {
        const auto value = id();
        module.AddFunction(opcode, type, value, left, right);
        return value;
    };
    const auto readValues = [&](const std::array<std::uint32_t, 3>& variables) {
        const auto valid = load(u32, variables[0]);
        const auto pair = load(u64, variables[1]);
        const auto count = load(u32, variables[2]);
        const auto halves = id();
        module.AddFunction(spv::OpBitcast, uvec2, halves, pair);
        const auto low = id();
        module.AddFunction(spv::OpCompositeExtract, u32, low, halves, 0u);
        const auto high = id();
        module.AddFunction(spv::OpCompositeExtract, u32, high, halves, 1u);
        const auto packed = id();
        module.AddFunction(spv::OpCompositeConstruct, uvec4, packed, valid, low, high, count);
        return packed;
    };
    const bool callee = reader == Reader::Callee || reader == Reader::CalleeLoop;
    std::uint32_t reading = 0;
    if (callee) {
        reading = id();
        std::vector<std::uint32_t> parameterTypes;
        if (!privateClass) parameterTypes = {u32Pointer, u64Pointer, u32Pointer};
        std::uint32_t functionType = 0;
        if (parameterTypes.empty()) functionType = module.Type(spv::OpTypeFunction, uvec4);
        else functionType = module.Type(spv::OpTypeFunction, uvec4, parameterTypes[0], parameterTypes[1], parameterTypes[2]);
        module.AddFunction(spv::OpFunction, uvec4, reading, spv::FunctionControlDontInlineMask, functionType);
        std::array<std::uint32_t, 3> parameters = globals;
        if (!privateClass) {
            for (std::uint32_t index = 0; index < 3u; ++index) {
                parameters[index] = id();
                module.AddFunction(spv::OpFunctionParameter, parameterTypes[index], parameters[index]);
            }
        }
        module.AddFunction(spv::OpLabel, id());
        module.AddFunction(spv::OpReturnValue, readValues(parameters));
        module.AddFunction(spv::OpFunctionEnd);
    }
    const auto main = id();
    const auto body = wrapped ? id() : main;
    module.AddFunction(spv::OpFunction, voidType, body, wrapped ? spv::FunctionControlDontInlineMask : spv::FunctionControlMaskNone, module.Type(spv::OpTypeFunction, voidType));
    const auto entry = id();
    module.AddFunction(spv::OpLabel, entry);
    std::array<std::uint32_t, 3> variables = globals;
    if (storage == Storage::Function) {
        variables = {id(), id(), id()};
        module.AddFunction(spv::OpVariable, u32Pointer, variables[0], spv::StorageClassFunction);
        module.AddFunction(spv::OpVariable, u64Pointer, variables[1], spv::StorageClassFunction);
        module.AddFunction(spv::OpVariable, u32Pointer, variables[2], spv::StorageClassFunction);
    } else if (storage == Storage::FunctionInitialized) {
        variables = {id(), id(), id()};
        module.AddFunction(spv::OpVariable, u32Pointer, variables[0], spv::StorageClassFunction, zero);
        module.AddFunction(spv::OpVariable, u64Pointer, variables[1], spv::StorageClassFunction, zero64);
        module.AddFunction(spv::OpVariable, u32Pointer, variables[2], spv::StorageClassFunction, zero);
    }
    if (mesh && !wrapped) module.AddFunction(spv::OpSetMeshOutputsEXT, zero, zero);
    if (!initialized) {
        module.AddFunction(spv::OpStore, variables[0], zero);
        module.AddFunction(spv::OpStore, variables[1], zero64);
        module.AddFunction(spv::OpStore, variables[2], zero);
    }
    const auto lane = id();
    module.AddFunction(spv::OpCompositeExtract, u32, lane, load(uvec3, invocation), 0u);
    std::uint32_t condition = 0;
    std::uint32_t length = 0;
    if (guard == Guard::Header) {
        length = id();
        module.AddFunction(spv::OpArrayLength, u32, length, input, 0u);
        condition = binary(spv::OpUGreaterThanEqual, boolType, length, constant(HeaderWords));
    } else if (guard == Guard::ArrayLength) {
        const auto length = id();
        module.AddFunction(spv::OpArrayLength, u32, length, input, 0u);
        condition = binary(spv::OpUGreaterThanEqual, boolType, length, constant(InputWords));
    } else {
        condition = binary(spv::OpIEqual, boolType, load(u32, word(input, zero)), constant(Magic));
    }
    const auto then = id();
    const auto merge = id();
    module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    module.AddFunction(spv::OpBranchConditional, condition, then, merge);
    module.AddFunction(spv::OpLabel, then);
    std::array<std::uint32_t, 3> results{};
    if (guard == Guard::Header) {
        std::array<std::uint32_t, HeaderWords> fields{};
        for (std::uint32_t index = 0; index < HeaderWords; ++index) fields[index] = load(u32, word(input, constant(index)));
        const auto rest = binary(spv::OpISub, u32, length, constant(HeaderWords));
        const auto wrongTag = binary(spv::OpINotEqual, boolType, fields[0], constant(HeaderTag));
        const auto wrongZero = binary(spv::OpINotEqual, boolType, fields[3], zero);
        const auto ranges = binary(spv::OpUGreaterThan, boolType, fields[1], binary(spv::OpShiftRightLogical, u32, rest, constant(3u)));
        const auto grants = binary(spv::OpUGreaterThan, boolType, fields[2], binary(spv::OpShiftRightLogical, u32, rest, constant(2u)));
        const auto size = binary(spv::OpIAdd, u32, binary(spv::OpIMul, u32, fields[1], constant(HeaderWords)), binary(spv::OpIMul, u32, fields[2], constant(4u)));
        const auto wrongSize = binary(spv::OpINotEqual, boolType, size, rest);
        const auto someRanges = binary(spv::OpINotEqual, boolType, fields[1], zero);
        const auto invalid = binary(spv::OpLogicalOr, boolType, binary(spv::OpLogicalOr, boolType, binary(spv::OpLogicalOr, boolType, wrongTag, wrongZero), binary(spv::OpLogicalOr, boolType, binary(spv::OpLogicalOr, boolType, grants, ranges), wrongSize)), someRanges);
        const auto valid = id();
        module.AddFunction(spv::OpSelect, u32, valid, invalid, zero, one);
        const auto wideHigh = id();
        module.AddFunction(spv::OpUConvert, u64, wideHigh, fields[5]);
        const auto shifted = binary(spv::OpShiftLeftLogical, u64, wideHigh, module.Constant(spv::OpConstant, u64, 32u, 0u));
        const auto wideLow = id();
        module.AddFunction(spv::OpUConvert, u64, wideLow, fields[4]);
        results = {valid, binary(spv::OpBitwiseOr, u64, wideLow, shifted), fields[2]};
    } else {
        const auto check = load(u32, word(input, one));
        const auto low = load(u32, word(input, constant(2u)));
        const auto high = load(u32, word(input, constant(3u)));
        const auto count = load(u32, word(input, constant(4u)));
        const auto matches = binary(spv::OpIEqual, boolType, check, constant(Magic));
        const auto valid = id();
        module.AddFunction(spv::OpSelect, u32, valid, matches, one, zero);
        const auto halves = id();
        module.AddFunction(spv::OpCompositeConstruct, uvec2, halves, low, high);
        const auto pair = id();
        module.AddFunction(spv::OpBitcast, u64, pair, halves);
        results = {valid, pair, count};
    }
    module.AddFunction(spv::OpBranch, merge);
    module.AddFunction(spv::OpLabel, merge);
    const std::array<std::uint32_t, 3> types{u32, u64, u32};
    const std::array<std::uint32_t, 3> zeros{zero, zero64, zero};
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto merged = id();
        module.AddFunction(spv::OpPhi, types[index], merged, results[index], then, zeros[index], entry);
        module.AddFunction(spv::OpStore, variables[index], merged);
    }
    const auto base = binary(spv::OpShiftLeftLogical, u32, lane, constant(2u));
    const auto read = [&] {
        if (!callee) return readValues(variables);
        const auto result = id();
        if (!privateClass) module.AddFunction(spv::OpFunctionCall, uvec4, result, reading, variables[0], variables[1], variables[2]);
        else module.AddFunction(spv::OpFunctionCall, uvec4, result, reading);
        return result;
    };
    const auto write = [&](std::uint32_t packed) {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto value = id();
            module.AddFunction(spv::OpCompositeExtract, u32, value, packed, component);
            module.AddFunction(spv::OpStore, word(output, binary(spv::OpIAdd, u32, base, constant(component))), value);
        }
    };
    if (reader == Reader::Main || reader == Reader::Callee) {
        write(read());
    } else {
        const auto header = id();
        const auto body = id();
        const auto next = id();
        const auto done = id();
        const auto counter = id();
        const auto advanced = id();
        module.AddFunction(spv::OpBranch, header);
        module.AddFunction(spv::OpLabel, header);
        module.AddFunction(spv::OpPhi, u32, counter, zero, merge, advanced, next);
        module.AddFunction(spv::OpLoopMerge, done, next, spv::LoopControlMaskNone);
        module.AddFunction(spv::OpBranchConditional, binary(spv::OpULessThan, boolType, counter, constant(LoopRounds)), body, done);
        module.AddFunction(spv::OpLabel, body);
        write(read());
        module.AddFunction(spv::OpBranch, next);
        module.AddFunction(spv::OpLabel, next);
        module.AddFunction(spv::OpIAdd, u32, advanced, counter, one);
        module.AddFunction(spv::OpBranch, header);
        module.AddFunction(spv::OpLabel, done);
    }
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    if (wrapped) {
        module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, module.Type(spv::OpTypeFunction, voidType));
        module.AddFunction(spv::OpLabel, id());
        if (mesh) module.AddFunction(spv::OpSetMeshOutputsEXT, zero, zero);
        module.AddFunction(spv::OpFunctionCall, voidType, id(), body);
        module.AddFunction(spv::OpReturn);
        module.AddFunction(spv::OpFunctionEnd);
    }
    if (mesh) {
        std::vector<std::uint32_t> interfaces{invocation, input, output};
        if (privateClass) interfaces.insert(interfaces.end(), globals.begin(), globals.end());
        module.EmitEntryPoint(spv::ExecutionModelMeshEXT, main, "main", interfaces);
        module.AddExecutionMode(main, spv::ExecutionModeOutputVertices, 3u);
        module.AddExecutionMode(main, spv::ExecutionModeOutputPrimitivesEXT, 1u);
        module.AddExecutionMode(main, spv::ExecutionModeOutputTrianglesEXT);
    } else {
        module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", {invocation});
    }
    module.AddExecutionMode(main, spv::ExecutionModeLocalSize, Lanes, 1u, 1u);
    return module.Finalize();
}

class Runner {
public:
    Runner(AgcDriver::VulkanDevice& device, const Context& context) : device(device), context(context) {
        const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, Stages, nullptr}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, Stages, nullptr}}};
        VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        setInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        setInfo.pBindings = bindings.data();
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &setInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &setLayout;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &size;
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &setLayout;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets");
        input = std::make_shared<Buffer>(context, InputWords * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        output = std::make_shared<Buffer>(context, Lanes * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const std::array<VkDescriptorBufferInfo, 2> infos{{{input->Handle(), 0, InputWords * 4u}, {output->Handle(), 0, Lanes * 16u}}};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (std::uint32_t binding = 0; binding < 2u; ++binding) {
            writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[binding], nullptr};
        }
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 2, writes.data(), 0, nullptr);
    }

    ~Runner() {
        context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    std::vector<std::uint32_t> Run(Stage stage, const std::vector<std::uint32_t>& code, const std::array<std::uint32_t, InputWords>& inputs) {
        std::memcpy(input->Bytes().data(), inputs.data(), sizeof(inputs));
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = code.size() * 4u;
        moduleInfo.pCode = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        const bool mesh = stage == Stage::Mesh;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult created = VK_SUCCESS;
        if (mesh) {
            const VkPipelineShaderStageCreateInfo meshStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, module, "main", nullptr};
            VkPipelineRenderingCreateInfoKHR rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
            VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rasterization.rasterizerDiscardEnable = VK_TRUE;
            rasterization.lineWidth = 1.0f;
            VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
            pipelineInfo.stageCount = 1;
            pipelineInfo.pStages = &meshStage;
            pipelineInfo.pRasterizationState = &rasterization;
            pipelineInfo.layout = layout;
            created = context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        } else {
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            pipelineInfo.layout = layout;
            created = context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        }
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        Check(created, "vkCreatePipelines");
        auto* words = reinterpret_cast<std::uint32_t*>(output->Bytes().data());
        for (std::uint32_t index = 0; index < Lanes * 4u; ++index) words[index] = Untouched;
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            auto* recorder = Recorder::Active();
            if (recorder == nullptr) throw std::runtime_error("the device has no recorder");
            const std::vector<Recorder::SectionObject> objects;
            recorder->DeclareSection(objects);
            const auto commands = recorder->Commands();
            const auto bindPoint = mesh ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE;
            context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, bindPoint, pipeline);
            context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, bindPoint, layout, 0, 1, &set, 0, nullptr);
            if (mesh) {
                VkRenderingInfoKHR rendering{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
                rendering.renderArea = {{0, 0}, {1, 1}};
                rendering.layerCount = 1;
                context.Function<PFN_vkCmdBeginRenderingKHR>("vkCmdBeginRenderingKHR")(commands, &rendering);
                context.Function<PFN_vkCmdDrawMeshTasksEXT>("vkCmdDrawMeshTasksEXT")(commands, 1, 1, 1);
                context.Function<PFN_vkCmdEndRenderingKHR>("vkCmdEndRenderingKHR")(commands);
            } else {
                context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, 1, 1, 1);
            }
            const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, mesh ? VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
            recorder->Keep(input);
            recorder->Keep(output);
            device.SubmitRecorded(false);
        }
        device.WaitIdle();
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        return {words, words + Lanes * 4u};
    }

private:
    AgcDriver::VulkanDevice& device;
    Context context;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::shared_ptr<Buffer> input;
    std::shared_ptr<Buffer> output;
};

void RunModules(AgcDriver::VulkanDevice& device) {
    Runner runner(device, AgcDriver::VulkanDeviceTestAccess::ContextOf(device));
    const std::array<std::uint32_t, InputWords> plain{Magic, Magic, PairLow, PairHigh, Count};
    const std::array<std::uint32_t, InputWords> header{HeaderTag, 0u, Grants, 0u, PairLow, PairHigh};
    for (const auto stage : {Stage::Compute, Stage::Mesh}) {
        for (const auto storage : {Storage::Private, Storage::PrivateInitialized, Storage::Function, Storage::FunctionInitialized}) {
            for (const auto reader : {Reader::Main, Reader::Callee, Reader::MainLoop, Reader::CalleeLoop}) {
                for (const auto guard : {Guard::ArrayLength, Guard::BufferValue, Guard::Header}) {
                    for (const bool wrapped : {false, true}) {
                        const auto what = std::string(stage == Stage::Mesh ? "mesh: " : "compute: ") + Name(storage, reader, guard) + (wrapped ? ", in a callee of main" : "");
                        const auto words = runner.Run(stage, Module(stage, storage, reader, guard, wrapped), guard == Guard::Header ? header : plain);
                        const std::array<std::uint32_t, 4> expected{1u, PairLow, PairHigh, guard == Guard::Header ? Grants : Count};
                        std::uint32_t wrong = 0;
                        std::string first;
                        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
                            for (std::uint32_t component = 0; component < 4u; ++component) {
                                const auto value = words[lane * 4u + component];
                                if (value == expected[component]) continue;
                                if (wrong++ == 0u) first = "lane " + std::to_string(lane) + " component " + std::to_string(component) + " is " + DeviceTable::Hex(value) + ", expected " + DeviceTable::Hex(expected[component]);
                            }
                        }
                        DeviceTable::Expect(wrong == 0u, what + ": values produced in the guarded branch and stored after its merge were read wrong in " + std::to_string(wrong) + " words, first " + first);
                    }
                }
            }
        }
    }
}

struct PrivateStores {
    std::uint32_t total = 0;
    std::uint32_t guarded = 0;
};

PrivateStores CountPrivateStores(std::span<const std::uint32_t> words) {
    std::vector<std::uint32_t> privates;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        if ((words[cursor] & 0xffffu) == spv::OpVariable && words[cursor + 3] == spv::StorageClassPrivate) privates.push_back(words[cursor + 2]);
    }
    PrivateStores stores;
    std::vector<std::uint32_t> merges;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpFunction) merges.clear();
        if (op == spv::OpLabel) {
            while (!merges.empty() && merges.back() == words[cursor + 1]) merges.pop_back();
        }
        if (op == spv::OpSelectionMerge) merges.push_back(words[cursor + 1]);
        if (op != spv::OpStore || std::find(privates.begin(), privates.end(), words[cursor + 1]) == privates.end()) continue;
        ++stores.total;
        if (!merges.empty()) ++stores.guarded;
    }
    return stores;
}

constexpr std::uint32_t LaneMark = 0x6d61726bu;

DeviceTable::Program LaneProgram() {
    using namespace DeviceTable::Asm;
    DeviceTable::Program built;
    LoadSrt(built, {{28, DeviceTable::OutputSlot}});
    built << Vop1(VMovB32, 4, Vgpr(0)) << Vop1(VMovB32, 5, 2) << Vop1(VMovB32, 6, Literal) << LaneMark << Vop1(VMovB32, 7, Literal) << ~LaneMark;
    StoreLane(built, 2, 28);
    return built;
}

void RunRecompiled(AgcDriver::VulkanDevice& device) {
    DeviceTable::Srt.fill(0u);
    DeviceTable::Place(DeviceTable::OutputSlot, DeviceTable::Buffer(DeviceTable::Output.data(), decltype(DeviceTable::Output)::ByteSize));
    const auto compiled = DeviceTable::Compile(device, LaneProgram(), DeviceTable::SrtUserData());
    const auto stores = CountPrivateStores(compiled.result->spirv.Words());
    const std::string what = "a recompiled program that stores through the device address table";
    DeviceTable::Expect(stores.total != 0u, what + ": the module stores no private variable, so it does not keep the validated table header");
    DeviceTable::Expect(stores.guarded == 0u, what + ": " + std::to_string(stores.guarded) + " of " + std::to_string(stores.total) + " private variable stores are inside a selection construct, expected each value produced under a guard to be stored after the guard's merge");
    const auto outcome = DeviceTable::Run(device, compiled, DeviceTable::Height);
    std::uint32_t wrong = 0;
    for (std::uint32_t group = 0; group < DeviceTable::Height; ++group) {
        for (std::uint32_t lane = 0; lane < DeviceTable::Lanes; ++lane) {
            const std::array<std::uint32_t, 4> expected{lane, group, LaneMark, ~LaneMark};
            for (std::uint32_t component = 0; component < 4u; ++component) {
                if (outcome.words[(group * DeviceTable::Lanes + lane) * 4u + component] != expected[component]) ++wrong;
            }
        }
    }
    DeviceTable::Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " stored words are wrong; log: " + outcome.log);
}

void Run(AgcDriver::VulkanDevice& device) {
    RunRecompiled(device);
    RunModules(device);
}

}

int main() {
    return DeviceTable::Main("private store after branch", Run);
}
