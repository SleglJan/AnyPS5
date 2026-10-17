#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "StorageImageMemory.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace AgcDriver::Graphics {

void ShaderResources::prepareAddressBindings(std::span<const CompiledShader> shaders) {
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
        const bool device = UsesDeviceImageTable(*shader.program);
        deviceImages = deviceImages || device;
        faultChecks = faultChecks || tables != 0 || faults != 0;
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
    namespace Keys = ShaderRecompiler::DeviceImageTableAbi;
    for (const auto& shader : shaders) {
        for (const auto& expectation : shader.program->deviceImages) {
            const auto& key = expectation.key;
            const auto id = static_cast<std::uint64_t>(key[Keys::InterfaceWord]) | (static_cast<std::uint64_t>(key[Keys::InterfaceWord + 1u]) << 32u);
            const auto interface = ShaderRecompiler::ResourceMaterializer::FindDeviceImageInterface(id);
            if (interface == nullptr || !ShaderRecompiler::MemoryPathRuntimeImage(interface->image)) continue;
            std::array<std::uint32_t, 8> words{};
            std::copy_n(key.begin(), words.size(), words.begin());
            const auto access = Memory::AccessOf(interface->image);
            if (IsNullTextureResource(words) || !Memory::SurfaceThroughMemory(words, access)) continue;
            std::uint64_t begin = 0;
            std::uint64_t end = 0;
            try {
                const auto resource = DecodeTextureResource(words);
                begin = resource.baseAddress;
                end = begin + DescribeSurface(resource).guestBytes;
            } catch (const std::runtime_error&) {
                continue;
            }
            if (DepthSurfaceAt(begin)) continue;
            const std::pair<std::uint64_t, std::uint64_t> range{begin, end};
            accessRanges.push_back(range);
            if (access != Memory::Access::Store) readRanges.push_back(range);
            if (access != Memory::Access::Load) storeRanges.push_back(range);
        }
    }
}

void ShaderResources::patchShaderData(std::int32_t pushByte, std::int64_t dataAllocation, std::uint32_t dataByte, std::span<const std::uint32_t> words) {
    const auto raw = std::as_bytes(words);
    if (pushByte >= 0) {
        for (std::size_t byte = 0; byte < raw.size(); ++byte) pushPatches.emplace_back(static_cast<std::uint32_t>(pushByte) + static_cast<std::uint32_t>(byte), static_cast<std::uint32_t>(raw[byte]));
        return;
    }
    Require(dataAllocation >= 0, "a shader data patch in a shader without shader data");
    const auto index = static_cast<std::size_t>(dataAllocation);
    Require(index < dispatchData.size() && dispatchData[index] != nullptr, "a per-dispatch shader data patch targets data its program shares");
    auto& data = *dispatchData[index];
    Require(dataByte + raw.size() <= data.Bytes().size(), "a shader data patch lies outside the shader's data buffer");
    std::memcpy(data.Bytes().data() + dataByte, raw.data(), raw.size());
}

}
