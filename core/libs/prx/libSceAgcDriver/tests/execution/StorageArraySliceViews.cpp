#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::StorageTexture;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Side = 128;
constexpr std::uint32_t Slices = 4;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileStandard64KB = 0x09;
constexpr std::uint32_t Type2DArray = 13;
constexpr std::size_t SliceBytes = 65536;
constexpr std::size_t SurfaceBytes = SliceBytes * Slices;
constexpr std::uint32_t Initial = 0x5a5a5a5au;
constexpr std::array<std::uint32_t, Slices> Values{0x10000000u, 0x11000000u, 0x12000000u, 0x13000000u};

alignas(256) constexpr std::array<std::uint32_t, 6> StoreCode{
    0x7e020280, 0x7e040280, 0x7e06020c, 0xf0201128, 0x00010300, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 9> LoadCode{
    0x7e020280, 0x7e04020d, 0xf0001128, 0x00010300, 0x34080082, 0xbf8c3f70, 0xe0701000, 0x80000304, 0xbf810000,
};

TestGuest::Array<std::uint32_t, Threads> Output;

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

class GuestBlock {
public:
    GuestBlock() {
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(SurfaceBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, SurfaceBytes, TestGuest::PageReadWrite, SurfaceBytes);
        Require(block != nullptr, "array slice views: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, SurfaceBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
        GuestArena::GuestArenaReset_nid_postfix(block, SurfaceBytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, SurfaceBytes);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }

private:
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 8> SliceDescriptor(const std::uint8_t* texels, std::uint32_t first, std::uint32_t last) {
    const auto address = AddressOf(texels);
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (TileStandard64KB << 20u) | (Type2DArray << 28u),
        last | (first << 16u),
        0u,
        0u,
        0u,
    };
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = AddressOf(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void Dispatch(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const TestGuest::Vector<std::uint32_t>& userData) {
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void StoreSlice(AgcDriver::VulkanDevice& device, const std::uint8_t* texels, std::uint32_t slice) {
    TestGuest::Vector<std::uint32_t> userData(16, 0u);
    const auto texture = SliceDescriptor(texels, slice, slice);
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    userData[12] = Values[slice];
    Dispatch(device, StoreCode, userData);
}

void LoadThroughArray(AgcDriver::VulkanDevice& device, const std::uint8_t* texels, std::uint32_t slice) {
    Output.fill(0xdeadbeefu);
    TestGuest::Vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    const auto texture = SliceDescriptor(texels, 0, Slices - 1u);
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    userData[13] = slice;
    Dispatch(device, LoadCode, userData);
}

void RequireSlice(AgcDriver::VulkanDevice& device, const std::uint8_t* texels, std::uint32_t slice, std::uint32_t expected, const std::string& what) {
    LoadThroughArray(device, texels, slice);
    for (std::uint32_t x = 0; x < Threads; ++x) {
        Require(Output[x] == expected, what + ": slice " + std::to_string(slice) + " texel (" + std::to_string(x) + ", 0) reads " + Hex(Output[x]) + ", expected " + Hex(expected));
    }
}

void Run(AgcDriver::VulkanDevice& device, std::uint8_t* texels) {
    const auto surface = AddressOf(texels);
    std::fill_n(reinterpret_cast<std::uint32_t*>(texels), SurfaceBytes / 4u, Initial);
    AgcDriver::GuestMemory::CollectWrites(surface, SurfaceBytes);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    StoreSlice(device, texels, 0);
    RequireSlice(device, texels, 0, Values[0], "the array view after a store through slice 0 alone");
    for (std::uint32_t slice = 1; slice < Slices; ++slice) RequireSlice(device, texels, slice, Initial, "the array view after a store through slice 0 alone");
    StoreSlice(device, texels, 2);
    const auto image = StorageTexture::FindPending(surface, SurfaceBytes);
    Require(image != nullptr && image->Descriptor().depthOrLastArray == Slices - 1u, "a store through one slice of the array left no pending image of the whole array");
    StoreSlice(device, texels, 1);
    Require(StorageTexture::FindPending(surface, SurfaceBytes) == image, "a store through slice 1 used another image than the store through slice 2");
    StoreSlice(device, texels, 3);
    Require(StorageTexture::FindPending(surface, SurfaceBytes) == image, "a store through slice 3 used another image than the store through slice 2");
    for (std::uint32_t slice = 0; slice < Slices; ++slice) RequireSlice(device, texels, slice, Values[slice], "the array view after a store through each slice");
    Require(StorageTexture::FindPending(surface, SurfaceBytes) == image, "the load through the array wrote the slices' results back instead of reading them in place");
    StorageTexture::FlushPending(surface, SurfaceBytes, nullptr, "test");
    device.WaitIdle();
    for (std::uint32_t slice = 0; slice < Slices; ++slice) {
        const auto* words = reinterpret_cast<const std::uint32_t*>(texels + slice * SliceBytes);
        Require(std::count(words, words + SliceBytes / 4u, Values[slice]) == static_cast<std::ptrdiff_t>(Threads), "guest slice " + std::to_string(slice) + " does not hold the 32 texels stored through its view after the flush");
        Require(std::count(words, words + SliceBytes / 4u, Initial) == static_cast<std::ptrdiff_t>(SliceBytes / 4u - Threads), "guest slice " + std::to_string(slice) + " changed outside the texels stored through its view");
    }
    AgcDriver::Graphics::ClearCachedTextures(device.Device());
}

}

int main() {
    try {
        GuestBlock block;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device, block.Data());
        std::puts("storage array slice view tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
