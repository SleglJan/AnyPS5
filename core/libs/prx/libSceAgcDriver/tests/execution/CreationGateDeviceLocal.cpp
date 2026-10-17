#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "VulkanTestDevice.hpp"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::int64_t PrivateBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) return static_cast<std::int64_t>(counters.PrivateUsage);
#endif
    return 0;
}

}

int main() {
#ifndef _WIN32
    std::puts("skipped: the RAM side is measured on Windows");
    return VulkanTestSkipped;
#else
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        constexpr std::int64_t Bytes = std::int64_t{512} << 20u;
        const auto privateBefore = PrivateBytes();
        const auto ramBefore = static_cast<std::int64_t>(AgcDriver::CreationGate::RamSideBytes());
        const auto block = device->ContinuationBlock(static_cast<std::size_t>(Bytes));
        const auto privateAfter = PrivateBytes();
        const auto ramAfter = static_cast<std::int64_t>(AgcDriver::CreationGate::RamSideBytes());
        const auto grewPrivate = privateAfter - privateBefore;
        const auto grewRam = ramAfter - ramBefore;
        Require(block != nullptr, "the device-local buffer was not created");
        Require(grewRam <= grewPrivate - Bytes / 2, "a 512 MiB device-local buffer grew the creation gate's RAM side by " + std::to_string(grewRam >> 20) + " MiB while the private bytes grew by " + std::to_string(grewPrivate >> 20) + " MiB: its backing was not subtracted");
        std::printf("creation gate device-local backing tests passed (private %+lld MiB, RAM side %+lld MiB)\n", static_cast<long long>(grewPrivate >> 20), static_cast<long long>(grewRam >> 20));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
