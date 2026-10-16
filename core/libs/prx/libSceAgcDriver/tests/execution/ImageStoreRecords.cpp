#include "GuestTestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <span>
#include <string>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Size = 64;
constexpr std::uint32_t Format32 = 20;
constexpr std::uint32_t Tile4KBS = 5;
constexpr std::uint32_t Type2D = 9;

alignas(256) constexpr std::array<std::uint32_t, 10> CopyCode{0xd7460004, 0x04010610, 0xd7460005, 0x04050611, 0xf0000f08, 0x00000004, 0xbf8c3f70, 0xf0200f08, 0x00020004, 0xbf810000};
alignas(256) constexpr std::array<std::uint32_t, 7> BufferCode{0x34020082, 0xe0301000, 0x80000201, 0xbf8c3f70, 0xe0701000, 0x80010201, 0xbf810000};

std::array<std::uint32_t, 8> TextureDescriptor(const void* texels) {
    const auto address = reinterpret_cast<std::uintptr_t>(texels) >> 8u;
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffu) | Format32 << 20u | ((Size - 1u) & 3u) << 30u, ((Size - 1u) >> 2u) | (Size - 1u) << 14u, 0xfacu | Tile4KBS << 20u | Type2D << 28u, 0u, 0u, 0u, 0u};
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

ShaderRecompiler::RecompileResult Recompile(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, std::uint32_t waveSize) {
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

std::size_t PrivateArrays(std::span<const std::uint32_t> words) {
    std::set<std::uint32_t> arrays;
    std::set<std::uint32_t> pointers;
    std::size_t variables = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpTypeArray) arrays.insert(words[cursor + 1]);
        if (op == spv::OpTypePointer && words[cursor + 2] == spv::StorageClassPrivate && arrays.contains(words[cursor + 3])) pointers.insert(words[cursor + 1]);
        if (op == spv::OpVariable && pointers.contains(words[cursor + 1])) ++variables;
    }
    return variables;
}

bool BindsFaultBuffer(const ShaderRecompiler::RecompileResult& result) {
    return std::ranges::any_of(result.bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer; });
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto layout = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kStandard4KB, Format32, Size, Size, 1u);
        const auto bytes = static_cast<std::size_t>(AgcDriver::Graphics::ComputeSurfaceSize(layout, 1u));
        auto* source = reinterpret_cast<std::uint8_t*>(TestGuest::Allocate(bytes, 65536));
        auto* destination = reinterpret_cast<std::uint8_t*>(TestGuest::Allocate(bytes, 65536));
        for (std::size_t index = 0; index < bytes; ++index) {
            source[index] = static_cast<std::uint8_t>(index * 13u + (index >> 9u) * 7u + 1u);
            destination[index] = 0u;
        }
        TestGuest::Vector<std::uint32_t> userData(16, 0u);
        const auto from = TextureDescriptor(source);
        const auto to = TextureDescriptor(destination);
        std::copy(from.begin(), from.end(), userData.begin());
        std::copy(to.begin(), to.end(), userData.begin() + 8);
        const ShaderRecompiler::ShaderComputeStageInfo copy{{8, 8, 1}, 0u, {true, true, false}, false, 2};
        const auto result = Recompile(*device, CopyCode, userData, copy, 64);
        Require(result.imageStoresRecorded, "image store records: an image copy kernel was not marked as recording its image stores");
        Require(BindsFaultBuffer(result), "image store records: an image copy kernel recording its stores binds no fault buffer");
        Require(PrivateArrays(result.spirv.Words()) == 0u, "image store records: the image copy kernel keeps " + std::to_string(PrivateArrays(result.spirv.Words())) + " Private arrays, expected the record's format and tail tables selected from constants");
        device->Dispatch(result, Size / 8u, Size / 8u, 1, reinterpret_cast<std::uintptr_t>(CopyCode.data()));
        device->WaitIdle();
        AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(destination), bytes, nullptr, "test");
        device->WaitIdle();
        for (std::size_t index = 0; index < bytes; ++index) {
            Require(destination[index] == source[index], "image store records: byte " + std::to_string(index) + " of the copied image is " + std::to_string(destination[index]) + ", expected " + std::to_string(source[index]));
        }
        TestGuest::Array<std::uint32_t, 32> input;
        TestGuest::Array<std::uint32_t, 32> output;
        std::array<std::uint32_t, 8> buffers{};
        const auto in = BufferDescriptor(input.data(), static_cast<std::uint32_t>(decltype(input)::ByteSize));
        const auto out = BufferDescriptor(output.data(), static_cast<std::uint32_t>(decltype(output)::ByteSize));
        std::copy(in.begin(), in.end(), buffers.begin());
        std::copy(out.begin(), out.end(), buffers.begin() + 4);
        const ShaderRecompiler::ShaderComputeStageInfo lanes{{32, 1, 1}, 0u, {false, false, false}, false, 1};
        Require(!Recompile(*device, BufferCode, buffers, lanes, 32).imageStoresRecorded, "image store records: a kernel without image stores was marked as recording them");
        std::puts("image store record tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
