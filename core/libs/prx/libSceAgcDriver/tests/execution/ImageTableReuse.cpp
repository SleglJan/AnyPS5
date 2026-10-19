#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t TextureBytes = Width * Height * 4;
constexpr std::uint32_t Stride = 48;
constexpr std::uint32_t Records = 48;
constexpr std::uint32_t Moved = 64;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Format32Float = 22;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t TexturesOffset = 65536;
constexpr std::size_t BlockBytes = TexturesOffset + 2 * 65536;
constexpr std::size_t UnimportableBytes = BlockBytes - 256;

TestGuest::Array<std::uint32_t, Records> Keys;
TestGuest::Array<std::uint32_t, Lanes * 4> Output;
TestGuest::Array<std::uint32_t, 32> Srt;

alignas(256) constexpr std::array<std::uint32_t, 28> TableCode{
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080700u, 0xfa000020u, 0x8f108202u, 0xf4200444u,
    0x20000000u, 0x9312b011u, 0xf42c0802u, 0x24000000u, 0x360200bfu, 0x7e020d01u, 0x060202f0u, 0x100202ffu,
    0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf09c8f08u, 0x01090401u, 0x7e120202u, 0x34101286u, 0x4a101100u,
    0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 7> StoreCode{
    0x7e020d00u, 0x082802f7u, 0x7e3c0300u, 0x7e3e0280u, 0xf0201108u, 0x0001141eu, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 4> PointClamp{0x92u, (4u * 256u) << 12u, 0u, 0u};

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::array<std::uint32_t, 4> Image(const void* texels) {
    const auto address = AddressOf(texels);
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32Float << 20u) | (((Width - 1u) & 3u) << 30u), ((Width - 1u) >> 2u) | ((Height - 1u) << 14u) | (1u << 31u), 0xfacu | (Type2D << 28u)};
}

std::array<std::uint32_t, 4> Buffer(const void* base, std::uint32_t bytes) {
    const auto address = AddressOf(base);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

class GuestBlock {
public:
    explicit GuestBlock(std::size_t registered = BlockBytes) {
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, TestGuest::PageReadWrite, BlockBytes);
        Require(block != nullptr, "image table reuse: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, registered, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
        GuestArena::GuestArenaReset_nid_postfix(block, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, BlockBytes);
    }

    std::uint32_t* Table() const { return reinterpret_cast<std::uint32_t*>(block); }
    float* Texels(std::uint32_t texture) const { return reinterpret_cast<float*>(block + TexturesOffset + static_cast<std::size_t>(texture) * TextureBytes); }

private:
    std::uint8_t* block = nullptr;
};

void SetRecord(const GuestBlock& block, std::uint32_t record, std::uint32_t texture) {
    auto* words = block.Table() + record * (Stride / 4u);
    std::fill(words, words + Stride / 4u, 0u);
    std::copy(PointClamp.begin(), PointClamp.end(), words);
    const auto image = Image(block.Texels(texture));
    std::copy(image.begin(), image.end(), words + 4);
}

void Fill(const GuestBlock& block, std::uint32_t texture, float bias) {
    for (std::uint32_t texel = 0; texel < Width * Height; ++texel) block.Texels(texture)[texel] = bias + static_cast<float>(texture) * 100.0f + static_cast<float>(texel);
}

ShaderRecompiler::RecompileResult Compile(AgcDriver::VulkanDevice& device, const GuestBlock& block) {
    const auto srtAddress = AddressOf(Srt.data());
    const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(srtAddress), static_cast<std::uint32_t>(srtAddress >> 32u)};
    const std::span<const std::uint32_t> code(TableCode);
    const std::span<const std::uint32_t> table(block.Table(), Records * Stride / 4u);
    const std::array<ShaderRecompiler::MemoryRegion, 4> memory{{
        {AddressOf(code.data()), std::as_bytes(code)},
        {srtAddress, std::as_bytes(std::span(Srt))},
        {AddressOf(table.data()), std::as_bytes(table)},
        {AddressOf(Keys.data()), std::as_bytes(std::span(Keys))},
    }};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {true, false, false}, false, 1};
    const ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, AddressOf(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    auto result = ShaderRecompiler::Recompile(request);
    Require(AgcDriver::Graphics::UsesDeviceImageTable(result), "image table reuse: the shader does not read its table entries through the device image table");
    return result;
}

void SetOutput(const void* output) {
    const auto view = Buffer(output, static_cast<std::uint32_t>(decltype(Output)::ByteSize));
    std::copy(view.begin(), view.end(), Srt.begin() + 8);
}

void Store(AgcDriver::VulkanDevice& device, const GuestBlock& block, std::uint32_t texture) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto image = Image(block.Texels(texture));
    std::copy(image.begin(), image.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(StoreCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{AddressOf(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, AddressOf(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, AddressOf(code.data()));
    device.WaitIdle();
}

struct Sampled {
    std::vector<std::uint32_t> red;
    std::uint64_t misses = 0;
};

Sampled Dispatch(AgcDriver::VulkanDevice& device, const ShaderRecompiler::RecompileResult& result, std::uint32_t record) {
    Keys[0] = record;
    std::fill(Output.begin(), Output.end(), 0xdeadbeefu);
    const auto before = device.DeviceImageCounts().misses;
    device.Dispatch(result, 1, 1, 1, AddressOf(TableCode.data()));
    device.WaitIdle();
    Sampled sampled;
    sampled.red.resize(Lanes);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) sampled.red[lane] = Output[lane * 4u];
    sampled.misses = device.DeviceImageCounts().misses - before;
    return sampled;
}

std::vector<std::uint32_t> Row(const float* texels) {
    std::vector<std::uint32_t> row(Lanes);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) row[lane] = std::bit_cast<std::uint32_t>(texels[lane % Width]);
    return row;
}

std::string Mismatch(const Sampled& sampled, const std::vector<std::uint32_t>& expected) {
    char text[160];
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        if (sampled.red[lane] == expected[lane]) continue;
        std::snprintf(text, sizeof(text), "lane %u read %g, expected %g; %llu misses", lane, std::bit_cast<float>(sampled.red[lane]), std::bit_cast<float>(expected[lane]), static_cast<unsigned long long>(sampled.misses));
        return text;
    }
    std::snprintf(text, sizeof(text), "every lane read the expected texels; %llu misses", static_cast<unsigned long long>(sampled.misses));
    return text;
}

enum class Use { First, Again };

void Sample(AgcDriver::VulkanDevice& device, const ShaderRecompiler::RecompileResult& result, std::uint32_t record, const std::vector<std::uint32_t>& expected, Use use, const std::string& what) {
    const auto sampled = Dispatch(device, result, record);
    if (use == Use::First && sampled.misses != 0u && sampled.red == std::vector<std::uint32_t>(Lanes, 0u)) {
        const auto again = Dispatch(device, result, record);
        Require(again.red == expected && again.misses == 0u, "image table reuse: " + what + " (the dispatch after the entry's first-use miss): " + Mismatch(again, expected));
        return;
    }
    Require(sampled.red == expected, "image table reuse: " + what + ": " + Mismatch(sampled, expected));
    Require(sampled.misses == 0u, "image table reuse: " + what + ": the dispatch read the texels yet counted " + std::to_string(sampled.misses) + " misses");
}

void RunImageStore(AgcDriver::VulkanDevice& device, const GuestBlock& block) {
    for (std::uint32_t texture = 0; texture < Records; ++texture) {
        Fill(block, texture, 0.125f);
        SetRecord(block, texture, texture);
    }
    const auto srt = Buffer(block.Table(), Records * Stride);
    std::copy(srt.begin(), srt.end(), Srt.begin());
    const auto result = Compile(device, block);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    Sample(device, result, 25, Row(block.Texels(25)), Use::First, "the table before an image store sampled the wrong entry");
    Sample(device, result, 27, Row(block.Texels(27)), Use::First, "the table resolved again before an image store sampled the wrong entry");
    Sample(device, result, 23, Row(block.Texels(23)), Use::First, "the table reused before an image store sampled the wrong entry");
    Store(device, block, 23);
    std::vector<float> stored(Width);
    for (std::uint32_t x = 0; x < Width; ++x) stored[x] = -4.0f - static_cast<float>(x);
    Sample(device, result, 23, Row(stored.data()), Use::Again, "an entry an image store left results pending over kept its old texels");
    AgcDriver::Graphics::StorageTexture::FlushPending(AddressOf(block.Texels(23)), TextureBytes, nullptr, "test");
    device.WaitIdle();
    Require(Row(block.Texels(23)) == Row(stored.data()), "image table reuse: the image store wrote the wrong values");
}

void RunTests(AgcDriver::VulkanDevice& device, const GuestBlock& block, const GuestBlock& stores) {
    for (std::uint32_t texture = 0; texture < Records; ++texture) {
        Fill(block, texture, 0.25f);
        SetRecord(block, texture, texture);
    }
    Fill(block, Moved, 0.5f);
    for (std::uint32_t key = 0; key < Records; ++key) Keys[key] = key;
    const auto srt = Buffer(block.Table(), Records * Stride);
    const auto keys = Buffer(Keys.data(), static_cast<std::uint32_t>(decltype(Keys)::ByteSize));
    std::copy(srt.begin(), srt.end(), Srt.begin());
    std::copy(keys.begin(), keys.end(), Srt.begin() + 4);
    SetOutput(Output.data());
    const auto result = Compile(device, block);
    const bool watched = AgcDriver::GuestMemory::WriteWatched();

    std::thread thread([&] {
        Sample(device, result, 3, Row(block.Texels(3)), Use::First, "a thread without a collect epoch sampled the wrong entry");
        Sample(device, result, 3, Row(block.Texels(3)), Use::Again, "a thread without a collect epoch sampled the wrong entry again");
    });
    thread.join();

    AgcDriver::GuestMemory::BumpCollectEpoch();
    Sample(device, result, 5, Row(block.Texels(5)), Use::First, "the first dispatch of the epoch sampled the wrong entry");
    Sample(device, result, 9, Row(block.Texels(9)), Use::First, "a dispatch reusing the table sampled the wrong entry");
    Sample(device, result, 5, Row(block.Texels(5)), Use::Again, "the third dispatch of the epoch sampled the wrong entry");
    if (!watched) {
        std::puts("guest memory has no write watch: writes over table entries are not tested");
        return;
    }

    AgcDriver::Graphics::ClearCachedTextures(device.Device());
    Sample(device, result, 7, Row(block.Texels(7)), Use::First, "the table after the texture cache was cleared sampled the wrong entry");
    Sample(device, result, 9, Row(block.Texels(9)), Use::Again, "an entry resolved before the texture cache was cleared sampled the wrong texels");

    std::vector<float> stored(Width * Height);
    for (std::uint32_t texel = 0; texel < Width * Height; ++texel) stored[texel] = -1.0f - static_cast<float>(texel);
    Sample(device, result, 11, Row(block.Texels(11)), Use::First, "an entry before the driver stored over it sampled the wrong texels");
    AgcDriver::GuestMemory::Write(AddressOf(block.Texels(11)), std::as_bytes(std::span(stored)));
    Sample(device, result, 11, Row(stored.data()), Use::Again, "an entry the driver stored over kept its old texels");
    Sample(device, result, 11, Row(stored.data()), Use::Again, "an entry the driver stored over sampled the wrong texels again");

    Sample(device, result, 17, Row(block.Texels(17)), Use::First, "an entry before the CPU wrote it sampled the wrong texels");
    Fill(block, 17, 0.75f);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    Sample(device, result, 17, Row(block.Texels(17)), Use::Again, "an entry the CPU wrote kept its old texels");
    Sample(device, result, 3, Row(block.Texels(3)), Use::Again, "an entry next to one the CPU wrote sampled the wrong texels");
    Sample(device, result, 17, Row(block.Texels(17)), Use::Again, "an entry the CPU wrote sampled the wrong texels again");

    Sample(device, result, 19, Row(block.Texels(19)), Use::First, "a record before it was pointed at another texture sampled the wrong texels");
    SetRecord(block, 19, Moved);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    const auto moved = Compile(device, block);
    Sample(device, moved, 19, Row(block.Texels(Moved)), Use::First, "a record pointed at another texture sampled the old one");
    Sample(device, result, 19, Row(block.Texels(Moved)), Use::Again, "a program compiled before a record was pointed at another texture sampled the old texture");
    Sample(device, result, 21, Row(block.Texels(21)), Use::First, "the table after a record moved sampled the wrong entry");

    SetOutput(block.Texels(13));
    const auto writer = Compile(device, block);
    static_cast<void>(Dispatch(device, writer, 7));
    SetOutput(Output.data());
    const auto row = Row(block.Texels(7));
    const std::vector<float> written(block.Texels(13), block.Texels(13) + Width * Height);
    Require(std::bit_cast<std::uint32_t>(written[0]) == row[0], "image table reuse: the dispatch storing into an entry's texels stored the wrong values");
    Sample(device, result, 13, Row(written.data()), Use::First, "an entry a dispatch wrote in the epoch kept its old texels");
    Sample(device, result, 15, Row(block.Texels(15)), Use::First, "the table after a dispatch wrote an entry sampled the wrong entry");

    RunImageStore(device, stores);
}

void RunImportedTable(AgcDriver::VulkanDevice& device, const GuestBlock& block) {
    for (std::uint32_t texture = 0; texture < Records; ++texture) {
        Fill(block, texture, 0.75f);
        SetRecord(block, texture, texture);
    }
    for (std::uint32_t key = 0; key < Records; ++key) Keys[key] = key;
    const auto srt = Buffer(block.Table(), Records * Stride);
    std::copy(srt.begin(), srt.end(), Srt.begin());
    SetOutput(Output.data());
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    Require(AgcDriver::Graphics::HostImportFor(context, AddressOf(block.Table()), BlockBytes) != nullptr, "image table reuse: the block could not be imported");
    const auto result = Compile(device, block);
    Sample(device, result, 6, Row(block.Texels(6)), Use::First, "the dispatch that settles the address space over imported memory sampled the wrong entry");
    AgcDriver::GuestMemory::BumpCollectEpoch();
    Sample(device, result, 6, Row(block.Texels(6)), Use::Again, "the first dispatch over imported memory sampled the wrong entry");
    Sample(device, result, 8, Row(block.Texels(8)), Use::First, "a dispatch reusing a table of storage views sampled the wrong entry");
    Sample(device, result, 6, Row(block.Texels(6)), Use::Again, "the third dispatch over imported memory sampled the wrong entry");
}

}

int main() {
    try {
        GuestBlock block;
        GuestBlock stores(UnimportableBytes);
        GuestBlock imported;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        RunTests(*device, block, stores);
        RunImportedTable(*device, imported);
        std::puts("image table reuse tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
