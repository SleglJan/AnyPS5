#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Dwords = 1u << 18u;
constexpr std::uint32_t LaneStride = Dwords / Threads;

TestGuest::Array<std::uint32_t, 8> Code{0x3402008eu, 0x4a040081u, 0xe0701000u, 0x80000201u, 0xbf810000u, 0u, 0u, 0u};
TestGuest::Array<std::uint32_t, Dwords> Target;

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::uint32_t Pattern(std::uint32_t index) {
    return index * 2654435761u + 0x5a5a5a5au;
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

struct MirrorState {
    std::uint64_t writtenBack = 0;
    std::uint64_t cpuWriteBacks = 0;
    std::uint32_t dirtyPages = 0;
};

MirrorState Mirrors(std::uint64_t address, std::uint64_t bytes) {
    MirrorState state;
    auto* residency = AgcDriver::Graphics::ResidencyDriver::Active();
    if (residency == nullptr) return state;
    const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    auto& manager = residency->Manager();
    const auto counts = manager.Counts();
    state.writtenBack = counts.writtenBackBytes;
    state.cpuWriteBacks = counts.cpuWriteBacks;
    for (auto page = address >> AgcDriver::Graphics::ResidencyPageShift; page <= (address + bytes - 1u) >> AgcDriver::Graphics::ResidencyPageShift; ++page) {
        if (manager.Dirty(page) != 0) ++state.dirtyPages;
    }
    return state;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        for (std::uint32_t index = 0; index < Dwords; ++index) Target[index] = Pattern(index);
        std::vector<std::uint32_t> userData(4, 0u);
        const auto target = BufferDescriptor(Target.data(), static_cast<std::uint32_t>(decltype(Target)::ByteSize));
        std::copy(target.begin(), target.end(), userData.begin());
        const std::span<const std::uint32_t> code(Code.data(), 8u);
        const auto address = reinterpret_cast<std::uintptr_t>(code.data());
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{address, std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
        ShaderRecompiler::RecompileRequest request{
            {ShaderStage::Compute, address, code, 0, {}},
            {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
            device->Target(),
            {0, 0, 0, 128}
        };
        request.useCache = false;
        const auto result = ShaderRecompiler::Recompile(request);
        const auto targetAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Target.data()));
        const auto before = Mirrors(targetAddress, decltype(Target)::ByteSize);
        device->Dispatch(result, 1, 1, 1, address);
        device->WaitIdle();
        const auto after = Mirrors(targetAddress, decltype(Target)::ByteSize);
        Require(after.writtenBack == before.writtenBack && after.cpuWriteBacks == before.cpuWriteBacks && after.dirtyPages == 0u, "staged stores: a dispatch storing one dword per lane 16 KiB apart through a 1 MiB V# of private pages, which writes only part of its range, left " + std::to_string(after.dirtyPages) + " of its pages dirty in mirrors and " + std::to_string(after.writtenBack - before.writtenBack) + " bytes written back over the range, where its stores reach the import alone");
        for (std::uint32_t index = 0; index < Dwords; ++index) {
            const bool stored = index % LaneStride == 0u;
            const auto expected = stored ? index / LaneStride + 1u : Pattern(index);
            Require(Target[index] == expected, "staged stores: dword " + std::to_string(index) + " is " + Hex(Target[index]) + ", expected " + Hex(expected) + (stored ? " from the store" : " kept from before the dispatch"));
        }
        std::puts("staged stores tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
