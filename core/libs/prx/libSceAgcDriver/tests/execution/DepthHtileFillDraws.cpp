#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "DepthFastClearHarness.hpp"
#include "GuestTestMemory.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

namespace {

using DepthFastClearHarness::Compile;
using DepthFastClearHarness::Covered;
using DepthFastClearHarness::Draw;
using DepthFastClearHarness::HtileBytes;
using DepthFastClearHarness::Record;
using DepthFastClearHarness::Require;
using DepthFastClearHarness::Surface;

constexpr std::size_t Tiles = HtileBytes / 4u;
constexpr std::uint32_t Expanded = 0xfffc000fu;
constexpr std::uint32_t Frames = 3;
constexpr std::array<std::uint32_t, 2> DrawsPerFrame{1, 8};

template <typename T>
T* Block() {
    return reinterpret_cast<T*>(TestGuest::Allocate(65536, 65536));
}

float* const Depth = Block<float>();
std::uint32_t* const Htile = Block<std::uint32_t>();

std::uint64_t Address(const void* pointer) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

bool Holds(std::uint32_t word) {
    return std::all_of(Htile, Htile + Tiles, [&](std::uint32_t value) { return value == word; });
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto shaders = Compile(*device);
        std::fill_n(Htile, Tiles, Expanded);
        const Surface depth{Address(Depth), 0, Address(Htile), false, VK_FORMAT_D32_SFLOAT};
        Require(Draw(*device, shaders, depth, {.depthWrite = true}) == Covered, "a new depth surface was not cleared to DB_DEPTH_CLEAR (1.0)");
        const std::array<std::uint32_t, 4> zero{};
        for (const auto draws : DrawsPerFrame) {
            for (std::uint32_t frame = 0; frame < Frames; ++frame) {
                const auto what = std::to_string(draws) + " draws, frame " + std::to_string(frame) + ": ";
                Require(Draw(*device, shaders, depth, {}) == 0u, what + "a draw at depth 0.5 passed LESS against the 0.5 written before the fill");
                const auto reads = AgcDriver::Graphics::HtileStoreBackReads();
                Require(device->FillBuffer(Address(Htile), HtileBytes, zero), what + "the device did not store the HTILE fill on the GPU");
                AgcDriver::Graphics::NoteDepthMetadataFill(Address(Htile), HtileBytes, 0u);
                for (std::uint32_t draw = 0; draw < draws; ++draw) Record(*device, shaders, depth, {});
                device->WaitIdle();
                const auto read = AgcDriver::Graphics::HtileStoreBackReads() - reads;
                Require(read == 0u, what + "the draws after a GPU fill of their HTILE read it on the CPU " + std::to_string(read) + " times, waiting for the fill");
                Require(Holds(Expanded), what + "the HTILE of the cleared depth was not stored back as expanded");
                Require(Draw(*device, shaders, depth, {}) == Covered, what + "the HTILE fill did not clear the depth to DB_DEPTH_CLEAR (1.0)");
                Require(Draw(*device, shaders, depth, {.depthWrite = true}) == Covered, what + "a depth write of 0.5 did not pass LESS against the cleared depth");
                Require(Draw(*device, shaders, depth, {}) == 0u, what + "the depth written after the fill's clear was cleared again");
            }
        }
        std::puts("depth HTILE fill draws tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
