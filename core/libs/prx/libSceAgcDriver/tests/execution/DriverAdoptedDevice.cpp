#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "DepthFastClearHarness.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Recorder* RecorderOf(const VulkanDevice& device) { return device.graphicsContext().recorder; }
};

}

namespace {

using DepthFastClearHarness::Covered;
using DepthFastClearHarness::Draw;
using DepthFastClearHarness::Height;
using DepthFastClearHarness::Require;
using DepthFastClearHarness::Surface;
using DepthFastClearHarness::Width;

constexpr std::array<std::uint32_t, 15> ClearKernel{0xd7460004u, 0x0401060cu, 0xf4280204u, 0xfa000000u, 0xd7460005u, 0x0405060du, 0x7e0c020eu, 0xbf8cc07fu, 0x7e000208u, 0x7e020209u, 0x7e04020au, 0x7e06020bu, 0xf0200f28u, 0x00000004u, 0xbf810000u};

std::uint8_t* NewBlock() {
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    auto* block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint8_t*>(std::aligned_alloc(bytes, bytes));
#endif
    Require(block != nullptr, "adopted device: cannot allocate a guest block");
    std::memset(block, 0, bytes);
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    return block;
}

std::uint64_t Address(const std::uint8_t* block) { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

std::vector<std::uint32_t> Pm4Packet(std::uint32_t opcode, std::initializer_list<std::uint32_t> body) {
    std::vector<std::uint32_t> words{0xc0000000u | (static_cast<std::uint32_t>(body.size() - 1) << 16u) | (opcode << 8u)};
    words.insert(words.end(), body);
    return words;
}

void DispatchClear(std::uint64_t kernel, std::uint64_t plane, std::uint64_t color) {
    const std::array<std::uint32_t, 8> texture{static_cast<std::uint32_t>(plane >> 8u), static_cast<std::uint32_t>((plane >> 40u) & 0xffu) | (22u << 20u) | (((Width - 1u) & 3u) << 30u), ((Width - 1u) >> 2u) | ((Height - 1u) << 14u), 0xfacu | (13u << 28u), 0u, 0u, 0u, 0u};
    const std::array<std::uint32_t, 4> buffer{static_cast<std::uint32_t>(color), static_cast<std::uint32_t>(color >> 32u) & 0xffffu, 16u, 0x31016facu};
    std::vector<std::vector<std::uint32_t>> packets{
        Pm4Packet(0x76, {0x20c, static_cast<std::uint32_t>(kernel >> 8u)}),
        Pm4Packet(0x76, {0x20d, static_cast<std::uint32_t>(kernel >> 40u) & 0xffu}),
        Pm4Packet(0x76, {0x207, 8}),
        Pm4Packet(0x76, {0x208, 8}),
        Pm4Packet(0x76, {0x209, 1}),
        Pm4Packet(0x76, {0x213, (12u << 1u) | 0x80u | 0x100u | 0x200u | (1u << 11u)}),
    };
    for (std::uint32_t word = 0; word < 8u; ++word) packets.push_back(Pm4Packet(0x76, {0x240u + word, texture[word]}));
    for (std::uint32_t word = 0; word < 4u; ++word) packets.push_back(Pm4Packet(0x76, {0x248u + word, buffer[word]}));
    packets.push_back(Pm4Packet(0x15, {Width / 8u, Height / 8u, 1, 0x41}));
    std::vector<std::uint32_t> words;
    for (const auto& packet : packets) words.insert(words.end(), packet.begin(), packet.end());
    Packet dcb{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&dcb) == 0, "adopted device: DCB submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

}

int main() {
    try {
        std::shared_ptr<AgcDriver::VulkanDevice> device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        AgcDriver::DriverDetail::Driver::Get().AdoptDeviceForTest(device);
        const auto shaders = DepthFastClearHarness::Compile(*device);
        auto* code = NewBlock();
        std::memcpy(code, ClearKernel.data(), sizeof(ClearKernel));
        auto* plane = NewBlock();
        auto* color = NewBlock();
        const Surface surface{Address(plane), 0, 0, false, VK_FORMAT_D32_SFLOAT};
        Require(Draw(*device, shaders, surface, {.depthWrite = true, .depthCompare = VK_COMPARE_OP_ALWAYS}) == Covered, "adopted device: the producer did not cover the plane");
        const std::array<float, 4> one{1.0f, 1.0f, 1.0f, 1.0f};
        std::memcpy(color, one.data(), sizeof(one));
        AgcDriver::GuestMemory::MarkWritten(Address(color), sizeof(one));
        DispatchClear(Address(code), Address(plane), Address(color));
        Require(AgcDriver::Graphics::Recorder::Active() == AgcDriver::VulkanDeviceTestAccess::RecorderOf(*device), "a driver-submitted dispatch opened a device of its own: the test device's later draws record into another device's recorder");
        const auto less = Draw(*device, shaders, surface, {.depthCompare = VK_COMPARE_OP_LESS});
        Require(less == Covered, "after the driver's clear to 1.0 the depth test 0.5 < depth passed for " + std::to_string(less) + " of " + std::to_string(Covered) + " pixels");
        Require(Draw(*device, shaders, surface, {.depthCompare = VK_COMPARE_OP_GREATER}) == 0u, "after the driver's clear to 1.0 a pixel passed 0.5 > depth");
        LibcRunShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
    std::puts("driver adopted device tests passed");
    return 0;
}
