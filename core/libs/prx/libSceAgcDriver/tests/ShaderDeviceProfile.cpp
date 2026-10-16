#include "prx/libSceAgcDriver/Execution/include/ShaderDeviceProfile.hpp"
#include "BdaAbi.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "PipelineSpecialization.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

struct ProfileInput {
    std::vector<std::uint32_t> capabilities{spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess};
    std::vector<std::string> extensionStrings{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    std::vector<std::string_view> extensions{extensionStrings[0], extensionStrings[1]};
    VkPhysicalDeviceRobustness2FeaturesEXT robustness{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, nullptr, VK_FALSE, VK_FALSE, VK_TRUE};
    VkPhysicalDeviceDescriptorIndexingFeatures indexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, &robustness};
    VkPhysicalDevice8BitStorageFeatures bytes{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &indexing, VK_TRUE};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, &bytes, VK_TRUE};
    VkPhysicalDeviceFeatures core{};
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    VkPhysicalDeviceLimits limits{};

    ProfileInput() {
        core.shaderInt64 = VK_TRUE;
        core.vertexPipelineStoresAndAtomics = VK_TRUE;
        core.fragmentStoresAndAtomics = VK_TRUE;
        device.pEnabledFeatures = &core;
        device.pNext = &bda;
        limits.maxPushConstantsSize = 128u;
        limits.maxBoundDescriptorSets = 3u;
        limits.maxStorageBufferRange = sizeof(ShaderRecompiler::RuntimeAbi::ShaderData);
    }

    ShaderRecompiler::SpirvTarget Target() const {
        ShaderRecompiler::SpirvTarget target{};
        target.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
        target.supportedCapabilities = capabilities;
        target.supportedExtensions = extensions;
        return target;
    }
};

template<typename TAction>
void Reject(TAction action, const char* expected) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos, "unexpected profile validation error");
        return;
    }
    throw std::runtime_error(std::string("invalid shader device profile was accepted: ") + expected);
}

void CheckProfile() {
    ProfileInput input;
    const AgcDriver::ShaderDeviceProfile profile(input.Target(), input.device, input.limits);
    input.capabilities.clear();
    input.extensionStrings[0] = "changed";
    input.core.shaderInt64 = VK_FALSE;
    input.robustness.nullDescriptor = VK_FALSE;
    input.limits.maxPushConstantsSize = 0u;
    const auto target = profile.Target();
    Require(target.supportedCapabilities.size() == 4u && std::ranges::is_sorted(target.supportedCapabilities), "profile capabilities were not frozen");
    Require(std::ranges::find(target.supportedExtensions, "SPV_KHR_physical_storage_buffer") != target.supportedExtensions.end(), "profile extension storage is borrowed");
    Require(profile.Limits().maxPushConstantsSize == 128u, "profile limits were not frozen");
    Require(profile.NullDescriptors(), "profile null descriptor support was not frozen");
    static_assert(!std::is_copy_constructible_v<AgcDriver::ShaderDeviceProfile> && !std::is_move_constructible_v<AgcDriver::ShaderDeviceProfile>);
    const auto invalid = [](auto mutate, const char* expected) {
        ProfileInput candidate;
        mutate(candidate);
        Reject([&] { AgcDriver::ShaderDeviceProfile rejected(candidate.Target(), candidate.device, candidate.limits); }, expected);
    };
    invalid([](auto& value) { value.device.pEnabledFeatures = nullptr; }, "enabled core features");
    invalid([](auto& value) { value.bda.bufferDeviceAddress = VK_FALSE; }, "bufferDeviceAddress");
    invalid([](auto& value) { value.device.pNext = nullptr; }, "bufferDeviceAddress");
    invalid([](auto& value) { value.core.shaderInt64 = VK_FALSE; }, "shaderInt64");
    invalid([](auto& value) { value.bytes.storageBuffer8BitAccess = VK_FALSE; }, "storageBuffer8BitAccess");
    ProfileInput withoutNull;
    withoutNull.robustness.nullDescriptor = VK_FALSE;
    Require(!AgcDriver::ShaderDeviceProfile(withoutNull.Target(), withoutNull.device, withoutNull.limits).NullDescriptors(), "a device without nullDescriptor was profiled with it");
    ProfileInput withoutRobustness;
    withoutRobustness.indexing.pNext = nullptr;
    Require(!AgcDriver::ShaderDeviceProfile(withoutRobustness.Target(), withoutRobustness.device, withoutRobustness.limits).NullDescriptors(), "a device without VK_EXT_robustness2 was profiled with nullDescriptor");
    invalid([](auto& value) { value.core.fragmentStoresAndAtomics = VK_FALSE; }, "graphics stores and atomics");
    invalid([](auto& value) { value.limits.maxPushConstantsSize = 127u; }, "exceeds device limits");
    invalid([](auto& value) { value.limits.maxBoundDescriptorSets = 0u; }, "exceeds device limits");
    invalid([](auto& value) { --value.limits.maxStorageBufferRange; }, "ShaderData exceeds device limits");
    invalid([](auto& value) { value.capabilities.clear(); }, "missing runtime capabilities");
    invalid([](auto& value) { value.extensions[0] = "missing"; }, "missing runtime extensions");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilitySampledImageArrayDynamicIndexing); }, "shaderSampledImageArrayDynamicIndexing");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityStorageImageArrayNonUniformIndexing); }, "shaderStorageImageArrayNonUniformIndexing");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityStorageImageReadWithoutFormat); }, "shaderStorageImageReadWithoutFormat");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityStorageImageMultisample); }, "shaderStorageImageMultisample");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityFragmentBarycentricKHR); }, "fragmentShaderBarycentric");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityMeshShadingEXT); }, "meshShader");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityDrawParameters); }, "shaderDrawParameters");
    invalid([](auto& value) { value.extensions.push_back("SPV_KHR_terminate_invocation"); }, "shaderTerminateInvocation");
    ProfileInput enabled;
    enabled.capabilities.push_back(spv::CapabilitySampledImageArrayDynamicIndexing);
    enabled.capabilities.push_back(spv::CapabilityStorageImageArrayNonUniformIndexing);
    enabled.core.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    enabled.indexing.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    const AgcDriver::ShaderDeviceProfile accepted(enabled.Target(), enabled.device, enabled.limits);
    Require(accepted.Target().supportedCapabilities.size() == 6u, "enabled descriptor indexing was rejected");
    ProfileInput drawParameters;
    drawParameters.capabilities.push_back(spv::CapabilityDrawParameters);
    VkPhysicalDeviceShaderDrawParametersFeatures drawFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, &drawParameters.bda, VK_TRUE};
    drawParameters.device.pNext = &drawFeatures;
    const AgcDriver::ShaderDeviceProfile drawParametersAccepted(drawParameters.Target(), drawParameters.device, drawParameters.limits);
    Require(std::ranges::find(drawParametersAccepted.Target().supportedCapabilities, spv::CapabilityDrawParameters) != drawParametersAccepted.Target().supportedCapabilities.end(), "enabled shaderDrawParameters was rejected");
    ProfileInput terminating;
    VkPhysicalDeviceShaderTerminateInvocationFeaturesKHR terminate{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR, &terminating.bda, VK_TRUE};
    terminating.device.pNext = &terminate;
    terminating.extensions.push_back("SPV_KHR_terminate_invocation");
    const AgcDriver::ShaderDeviceProfile terminated(terminating.Target(), terminating.device, terminating.limits);
    Require(std::ranges::find(terminated.Target().supportedExtensions, "SPV_KHR_terminate_invocation") != terminated.Target().supportedExtensions.end(), "enabled shaderTerminateInvocation was rejected");
    ProfileInput disabled;
    VkPhysicalDeviceShaderTerminateInvocationFeaturesKHR disabledTerminate{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR, &disabled.bda, VK_FALSE};
    disabled.device.pNext = &disabledTerminate;
    disabled.extensions.push_back("SPV_KHR_terminate_invocation");
    Reject([&] { AgcDriver::ShaderDeviceProfile rejected(disabled.Target(), disabled.device, disabled.limits); }, "shaderTerminateInvocation");
}

void CheckAbi() {
    using ShaderRecompiler::RuntimeAbi::Binding;
    using ShaderRecompiler::RuntimeAbi::BindingNumber;
    using ShaderRecompiler::RuntimeAbi::Stage;
    std::set<std::uint32_t> bindings;
    for (std::uint32_t stage = 0u; stage < ShaderRecompiler::RuntimeAbi::StageCount; ++stage) {
        for (std::uint32_t binding = 0u; binding < static_cast<std::uint32_t>(Binding::Count); ++binding) {
            Require(bindings.insert(BindingNumber(static_cast<Stage>(stage), static_cast<Binding>(binding))).second, "runtime ABI bindings overlap");
        }
    }
    Require(BindingNumber(Stage::Main, Binding::ShaderData) == 62u && BindingNumber(Stage::Fragment, Binding::ShaderData) == 127u, "runtime ABI binding numbers changed");
    Require(BindingNumber(Stage::Main, Binding::ImageTable) == ShaderRecompiler::RuntimeAbi::ImageTableBinding && BindingNumber(Stage::TessellationEvaluation, Binding::ImageTable) == ShaderRecompiler::RuntimeAbi::ImageTableBinding && !bindings.contains(ShaderRecompiler::RuntimeAbi::ImageTableBinding), "the shared image table binding overlaps a stage binding");
    Reject([] { BindingNumber(static_cast<Stage>(4u), Binding::Buffers); }, "invalid stage or binding");
    Reject([] { BindingNumber(Stage::Main, Binding::Count); }, "invalid stage or binding");
    Reject([] { ShaderRecompiler::RuntimeAbi::RequireVersion(0u); }, "incompatible version");
    ShaderRecompiler::RuntimeAbi::RequireVersion(ShaderRecompiler::RuntimeAbi::Version);
}

void CheckSpecializationIds() {
    namespace Spec = ShaderRecompiler::PipelineSpecialization;
    constexpr std::uint32_t samplers = 100u;
    std::set<std::uint32_t> ids;
    const auto add = [&](std::uint32_t id, const char* what) { Require(ids.insert(id).second, std::string("specialization constant ID collision in ") + what); };
    for (std::uint32_t sampler = 0u; sampler < samplers; ++sampler) {
        add(Spec::FoldOffset(sampler), "the texel offset folds of the samplers");
        add(Spec::UnnormalizedBaseLevel(sampler), "the unnormalized base level flags of the samplers");
        Require(Spec::FoldOffset(sampler) >= Spec::SamplerBase && Spec::UnnormalizedBaseLevel(sampler) < Spec::BufferBase, "a sampler specialization ID left its range");
    }
    for (std::uint32_t image = 0u; image < ShaderRecompiler::RuntimeAbi::ImageCapacity; ++image) {
        for (std::uint32_t word = 0u; word < Spec::ImageWords; ++word) add(Spec::ImageBase + image * Spec::ImageWords + word, "the image constants");
        for (std::uint32_t word = 0u; word < Spec::CompareWords; ++word) add(Spec::CompareBase + image * Spec::CompareWords + word, "the comparison constants");
    }
    for (std::uint32_t buffer = 0u; buffer < ShaderRecompiler::RuntimeAbi::BufferCapacity; ++buffer) {
        for (std::uint32_t word = 0u; word < Spec::BufferWords; ++word) add(Spec::BufferBase + buffer * Spec::BufferWords + word, "the buffer constants");
    }
    for (std::uint32_t binding = 0u; binding < static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count); ++binding) {
        for (std::uint32_t element = 0u; element < Spec::DescriptorIndexStride; ++element) add(Spec::DescriptorIndex(binding, element), "the descriptor indices");
    }
    Require(Spec::FoldOffset(0u) > Spec::CompareBase + ShaderRecompiler::RuntimeAbi::ImageCapacity * Spec::CompareWords, "sampler specialization IDs start inside the comparison constants");
}

void CheckHeaps() {
    using namespace ShaderRecompiler;
    const auto allocate = [](const ImageResource& image, std::uint32_t count, std::uint32_t samplers = 0u) {
        IrProgram program;
        program.Metadata().shaderInfoComplete = true;
        program.Resources().info.images.assign(count, image);
        program.Resources().info.samplers.resize(samplers);
        return BindingAllocator{}.Allocate(program, {0u, 0u, 0u, 128u});
    };
    ImageResource image;
    image.resourceClass = ImageResourceClass::Sampled;
    image.numericClass = IrTextureNumericClass::Float;
    image.dimension = RdnaImageDimension::Dim2D;
    const auto single = allocate(image, 1u);
    const auto sixteenSamplers = allocate(image, 1u, 16u);
    Require(BindingAllocator{}.FindBinding(sixteenSamplers.layout, DescriptorBindingKind::Samplers).resources.size() == 32u, "sixteen logical samplers must fit with both descriptor variants");
    constexpr std::uint32_t manyImages = 64u;
    const auto full = allocate(image, manyImages, RuntimeAbi::SamplerHeapCapacity / 2u);
    const auto expectSlots = [](const IrBindingLayout& layout, std::vector<std::uint32_t> bases, std::uint32_t count, std::string_view what) {
        Require(layout.imageSlotBases == bases && layout.imageSlotCount == count, std::string(what) + ": the image slots are not consecutive in image order");
        Require(layout.runtimeImageCount == 0u && layout.ImageSlotDword() == layout.SamplerSlotDword() + layout.samplerSlotCount && layout.ShaderDataDwords() == layout.samplerSlotCount + count, std::string(what) + ": the image slots do not follow the sampler slots, or direct images allocated runtime metadata");
    };
    const auto sequence = [](std::uint32_t count, std::uint32_t step) {
        std::vector<std::uint32_t> result(count);
        for (std::uint32_t index = 0u; index < count; ++index) result[index] = index * step;
        return result;
    };
    expectSlots(single.layout, {0u}, 1u, "one sampled image takes one slot shared by all its typed heaps");
    expectSlots(full.layout, sequence(manyImages, 1u), manyImages, "many sampled images");
    Require(single.layout.memoryOffsetDword == full.layout.memoryOffsetDword && single.layout.SamplerSlotDword() == full.layout.SamplerSlotDword(), "runtime layout before the image slots depends on the image count");
    Require(full.layout.samplerSlotCount == RuntimeAbi::SamplerHeapCapacity && full.layout.SamplerSlotDword() == 0u, "direct samplers did not take one heap slot per descriptor variant");
    Require(BindingAllocator{}.FindBinding(allocate(image, 40u).layout, DescriptorBindingForImage(image)).resources.size() == 40u, "a typed group does not hold 40 images of one class");
    const auto limit = allocate(image, RuntimeAbi::ImageCapacity);
    Require(BindingAllocator{}.FindBinding(limit.layout, DescriptorBindingForImage(image)).resources.size() == RuntimeAbi::ImageCapacity, "a typed group does not hold as many images as the runtime metadata does");
    expectSlots(limit.layout, sequence(RuntimeAbi::ImageCapacity, 1u), RuntimeAbi::ImageCapacity, "as many sampled images as the runtime metadata holds");
    Reject([&] { allocate(image, RuntimeAbi::ImageCapacity + 1u); }, "metadata capacity");
    constexpr std::uint32_t manySamplers = 40u;
    static_assert(manySamplers > RuntimeAbi::SamplerHeapCapacity);
    const auto beyondMetadata = allocate(image, 1u, manySamplers);
    Require(BindingAllocator{}.FindBinding(beyondMetadata.layout, DescriptorBindingKind::Samplers).resources.size() == 2u * manySamplers && beyondMetadata.layout.samplerSlotCount == 2u * manySamplers, "a shader with more samplers than the runtime metadata holds did not get two slots for each");
    expectSlots(beyondMetadata.layout, {0u}, 1u, "one sampled image after more samplers than the runtime metadata holds");
    image.resourceClass = ImageResourceClass::Storage;
    const auto plainStorage = allocate(image, 2u);
    expectSlots(plainStorage.layout, {0u, 1u}, 2u, "storage images without dynamic mips");
    image.mipMode = ImageMipMode::DynamicStorage;
    image.mipCount = RuntimeAbi::StorageMipSlots;
    const auto storage = allocate(image, 1u);
    Require(BindingAllocator{}.FindBinding(storage.layout, DescriptorBindingForImage(image)).resources.size() == image.mipCount, "storage heap did not reserve each mip");
    const auto shared = allocate(image, 2u);
    Require(BindingAllocator{}.FindBinding(shared.layout, DescriptorBindingForImage(image)).resources.size() == 2u * image.mipCount, "a storage heap did not hold two dynamic-mip images");
    constexpr std::uint32_t manyStorage = RuntimeAbi::ImageCapacity;
    const auto filled = allocate(image, manyStorage);
    Require(BindingAllocator{}.FindBinding(filled.layout, DescriptorBindingForImage(image)).resources.size() == manyStorage * image.mipCount, "a storage group did not hold a slot per mip of every image");
    expectSlots(storage.layout, {0u}, RuntimeAbi::StorageMipSlots, "one dynamic-mip storage image takes a slot per mip");
    expectSlots(shared.layout, {0u, RuntimeAbi::StorageMipSlots}, 2u * RuntimeAbi::StorageMipSlots, "two dynamic-mip storage images");
    expectSlots(filled.layout, sequence(manyStorage, RuntimeAbi::StorageMipSlots), manyStorage * RuntimeAbi::StorageMipSlots, "as many dynamic-mip storage images as the runtime metadata holds");
    Reject([&] { allocate(image, manyStorage + 1u); }, "metadata capacity");
    {
        ImageResource sampled;
        sampled.resourceClass = ImageResourceClass::Sampled;
        sampled.numericClass = IrTextureNumericClass::Float;
        sampled.dimension = RdnaImageDimension::Dim2D;
        ImageResource plain = sampled;
        plain.resourceClass = ImageResourceClass::Storage;
        IrProgram mixed;
        mixed.Metadata().shaderInfoComplete = true;
        ImageResource tabled = sampled;
        tabled.table = 0u;
        mixed.Resources().info.images = {sampled, image, tabled, sampled, plain};
        const auto result = BindingAllocator{}.Allocate(mixed, {0u, 0u, 0u, 128u});
        expectSlots(result.layout, {0u, 1u, 0u, 1u + RuntimeAbi::StorageMipSlots, 2u + RuntimeAbi::StorageMipSlots}, 3u + RuntimeAbi::StorageMipSlots, "sampled, dynamic-mip storage, table and plain storage images");
    }
    constexpr std::uint32_t hundredSamplers = 100u;
    const auto hundred = allocate(image, 0u, hundredSamplers);
    Require(BindingAllocator{}.FindBinding(hundred.layout, DescriptorBindingKind::Samplers).resources.size() == 2u * hundredSamplers && hundred.layout.samplerSlotCount == 2u * hundredSamplers && hundred.layout.ImageSlotDword() == 2u * hundredSamplers && hundred.layout.ShaderDataDwords() == 2u * hundredSamplers, "a hundred samplers did not take a pair of slots each");
    const std::array dimensions{RdnaImageDimension::Dim1D, RdnaImageDimension::Dim1DArray, RdnaImageDimension::Dim2D, RdnaImageDimension::Dim2DArray, RdnaImageDimension::Dim3D, RdnaImageDimension::Dim2DMsaa, RdnaImageDimension::Dim2DMsaaArray};
    std::set<std::uint32_t> classes;
    for (std::uint32_t group = 0u; group < 8u; ++group) {
        for (const auto dimension : dimensions) {
            image = {};
            image.resourceClass = group < 4u ? ImageResourceClass::Sampled : ImageResourceClass::Storage;
            image.numericClass = group == 1u || group >= 5u ? IrTextureNumericClass::Uint : group == 2u ? IrTextureNumericClass::Sint : IrTextureNumericClass::Float;
            image.depthCompare = group == 3u;
            image.atomic = group >= 6u;
            image.atomic64 = group == 7u;
            image.dimension = dimension;
            const auto binding = DescriptorBindingForImage(image);
            Require(classes.insert(static_cast<std::uint32_t>(binding)).second, "typed image classes overlap");
        }
    }
    Require(classes.size() == RuntimeAbi::ImageBindingCount && *classes.begin() == 1u && *classes.rbegin() == RuntimeAbi::ImageBindingCount, "typed image class mapping is incomplete");
}

}

int main() {
    try {
        CheckAbi();
        CheckSpecializationIds();
        CheckHeaps();
        CheckProfile();
        std::cout << "shader runtime ABI and device profile tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
