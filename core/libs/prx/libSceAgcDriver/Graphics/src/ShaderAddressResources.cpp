#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "StorageImageMemory.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace AgcDriver::Graphics {

void ShaderResources::prepareAddressBindings(std::span<const CompiledShader> shaders) {
    static const bool loopGuarded = std::getenv("APS5_LOOP_GUARD") != nullptr;
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        std::uint32_t tables = 0;
        std::uint32_t faults = 0;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::BdaPagetable && binding.role != ShaderRecompiler::DescriptorRole::FaultBuffer) continue;
            Require(binding.kind == ShaderRecompiler::DescriptorKind::StorageBuffer && binding.count == 1 && binding.guestDescriptor.empty() && !binding.readOnly, "invalid BDA descriptor contract");
            if (binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable) ++tables;
            else ++faults;
        }
        const bool faultOnly = tables == 0 && faults == 1 && shader.program->bdaAbiVersion == ShaderRecompiler::BdaAbi::Version;
        Require((tables == faults || faultOnly) && tables <= 1 && faults <= 1, "invalid BDA table and fault descriptors");
        Require(shader.program->bdaAbiVersion == (faults == 0 ? 0u : ShaderRecompiler::BdaAbi::Version), "incompatible BDA ABI version");
        // Rect-list validation needs a fault buffer, but never accesses guest addresses.
        usesBda = usesBda || tables != 0;
        pageTableStores = pageTableStores || (tables != 0 && shader.program->bdaWrites);
        if (tables != 0 && shader.program->bdaWrites) {
            storesUnbounded = storesUnbounded || shader.program->unboundedWrites;
            storeRanges.insert(storeRanges.end(), shader.program->writeRanges.begin(), shader.program->writeRanges.end());
        }
        if (tables != 0) {
            accessesUnbounded = accessesUnbounded || shader.program->unboundedReads || (shader.program->bdaWrites && shader.program->unboundedWrites);
            accessRanges.insert(accessRanges.end(), shader.program->readRanges.begin(), shader.program->readRanges.end());
            readRanges.insert(readRanges.end(), shader.program->readRanges.begin(), shader.program->readRanges.end());
            if (shader.program->bdaWrites) accessRanges.insert(accessRanges.end(), shader.program->writeRanges.begin(), shader.program->writeRanges.end());
        }
        usesFaultBuffer = usesFaultBuffer || faults != 0;
        const bool tableFaultsOnly = std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap; });
        const bool device = UsesDeviceImageTable(*shader.program);
        deviceImages = deviceImages || device;
        faultChecks = faultChecks || tables != 0 || (faults != 0 && (!tableFaultsOnly || device || shader.program->imageTableFaults != 0 || shader.program->poisonedSrtReads != 0 || loopGuarded));
    }
    if (const auto linear = linearStoreRange(); linear.has_value() && pageTableStores) {
        storesUnbounded = false;
        storeRanges.assign(1, *linear);
        accessRanges.push_back(*linear);
        accessesUnbounded = std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program->unboundedReads; });
    }
    if (usesBda) selectMemoryImages(shaders);
    if (usesBda) {
        Require(context.bufferDeviceAddress, "buffer device address is not enabled");
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        guestMemory.AcquireRegistered();
        guestMemory.SetAccessRanges(accessRanges, accessesUnbounded);
        if (profile) GuestBufferMemory::CountAddressBuild();
    }
}

void ShaderResources::selectMemoryImages(std::span<const CompiledShader> shaders) {
    namespace Memory = ShaderRecompiler::StorageImageMemory;
    struct Element {
        const ShaderRecompiler::DescriptorBinding* binding;
        std::uint32_t element;
        std::uint64_t begin;
        std::uint64_t end;
        Memory::Access access;
        bool memory;
    };
    std::vector<Element> elements;
    for (const auto& shader : shaders) {
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.count == 0) continue;
            const auto elementWords = binding.guestDescriptor.size() / binding.count;
            if (elementWords < 8u) continue;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                std::array<std::uint32_t, 8> words{};
                std::copy_n(binding.guestDescriptor.begin() + static_cast<std::ptrdiff_t>(element * elementWords), words.size(), words.begin());
                if (IsNullTextureResource(words)) continue;
                std::uint64_t begin = 0;
                std::uint64_t end = 0;
                try {
                    const auto resource = DecodeTextureResource(words);
                    begin = resource.baseAddress;
                    end = begin + DescribeSurface(resource).guestBytes;
                } catch (const std::runtime_error&) {
                    continue;
                }
                const auto code = element < binding.imageMemoryAccess.size() ? binding.imageMemoryAccess[element] : std::uint8_t{0};
                const auto access = static_cast<Memory::Access>(code == 0u ? 0u : code - 1u);
                const bool memory = code != 0u && element < binding.imageSlotDwords.size() && end > begin && Memory::SurfaceThroughMemory(words, access) && !DepthSurfaceAt(begin);
                elements.push_back({&binding, element, begin, end, access, memory});
            }
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& element : elements) {
            if (!element.memory) continue;
            const bool viewed = std::any_of(elements.begin(), elements.end(), [&](const Element& other) { return !other.memory && other.begin < element.end && element.begin < other.end; });
            if (!viewed) continue;
            element.memory = false;
            changed = true;
        }
    }
    for (const auto& element : elements) {
        if (!element.memory) continue;
        memoryImages.emplace(element.binding, element.element);
        const std::pair<std::uint64_t, std::uint64_t> range{element.begin, element.end};
        accessRanges.push_back(range);
        if (element.access != Memory::Access::Store) readRanges.push_back(range);
        if (element.access != Memory::Access::Load) storeRanges.push_back(range);
    }
}

std::pair<VkDeviceAddress, std::uint64_t> ShaderResources::bufferAddress(Allocation& allocation) {
    Require(allocation.deviceAddressed, "the allocation is not a guest buffer bound by device address");
    if (allocation.guest) return guestMemory.Address(allocation.address, allocation.size, allocation.adjustment);
    return {0, 0};
}

void ShaderResources::patchBufferAddress(const Allocation& allocation, VkDeviceAddress address, std::uint64_t bytes) {
    Require(bytes / 4u <= std::numeric_limits<std::uint32_t>::max(), "guest buffer view exceeds the shader's 32-bit dword count");
    const std::array<std::uint32_t, ShaderRecompiler::RuntimeAbi::BufferAddressDwords> words{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), static_cast<std::uint32_t>(bytes / 4u)};
    patchShaderData(allocation.addressPushByte, allocation.dataAllocation, allocation.addressDataByte, words);
}

void ShaderResources::patchShaderData(std::int32_t pushByte, std::int64_t dataAllocation, std::uint32_t dataByte, std::span<const std::uint32_t> words) {
    const auto raw = std::as_bytes(words);
    if (pushByte >= 0) {
        for (std::size_t byte = 0; byte < raw.size(); ++byte) pushPatches.emplace_back(static_cast<std::uint32_t>(pushByte) + static_cast<std::uint32_t>(byte), static_cast<std::uint32_t>(raw[byte]));
        return;
    }
    Require(dataAllocation >= 0, "a shader data patch in a shader without shader data");
    auto& data = allocations[static_cast<std::size_t>(dataAllocation)];
    Require(data.buffer != nullptr && dataByte + raw.size() <= data.size, "a shader data patch lies outside the shader's data buffer");
    for (std::size_t byte = 0; byte < raw.size(); ++byte) {
        data.buffer->Bytes()[dataByte + byte] = raw[byte];
        dataPatches.push_back({static_cast<std::size_t>(dataAllocation), dataByte + static_cast<std::uint32_t>(byte), static_cast<std::uint32_t>(raw[byte])});
    }
}

VkDescriptorBufferInfo ShaderResources::descriptor(Allocation& allocation) {
    Require(!allocation.deviceAddressed, "a guest buffer read through its device address has no descriptor");
    if (allocation.guest) return guestMemory.Descriptor(allocation.address, allocation.size, allocation.adjustment);
    if (allocation.role == ShaderRecompiler::DescriptorRole::BdaPagetable || allocation.role == ShaderRecompiler::DescriptorRole::FaultBuffer) {
        Require(bda != nullptr, "BDA descriptors have no memory owner");
        return allocation.role == ShaderRecompiler::DescriptorRole::BdaPagetable ? bda->Table() : bda->Fault();
    }
    Require(allocation.buffer != nullptr, "shader data has no buffer owner");
    return {allocation.buffer->Handle(), 0, allocation.size};
}

}
