#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {

using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Blocks = 20000;
constexpr std::uint32_t Creations = 4;
constexpr std::int64_t RetainedBound = std::int64_t{128} << 20u;

TestGuest::Array<std::uint32_t, 2048> Input;

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

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

TestGuest::Vector<std::uint32_t> Program(std::uint32_t blocks, std::uint32_t seed) {
    TestGuest::Vector<std::uint32_t> code{0x7e020300u, 0x34060082u};
    for (std::uint32_t block = 0; block < blocks; ++block) {
        code.insert(code.end(), {0x4a0202ffu, block * 7u + seed * 3u + 1u, 0x3a020101u, 0x16020101u});
    }
    code.insert(code.end(), {0xe0701000u, 0x80000103u, 0xbf810000u});
    return code;
}

struct Created {
    std::size_t words = 0;
    std::int64_t held = 0;
    double milliseconds = 0;
};

struct Module {
    TestGuest::Vector<std::uint32_t> code;
    ShaderRecompiler::RecompileResult result;
};

Module Compile(AgcDriver::VulkanDevice& device, std::uint32_t blocks, std::uint32_t seed) {
    auto code = Program(blocks, seed);
    TestGuest::Vector<std::uint32_t> userData(4, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(decltype(Input)::ByteSize));
    std::copy(input.begin(), input.end(), userData.begin());
    const std::span<const std::uint32_t> words(code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{reinterpret_cast<std::uintptr_t>(words.data()), std::as_bytes(words)}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(words.data()), words, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    auto result = ShaderRecompiler::Recompile(request);
    return {std::move(code), std::move(result)};
}

Created Create(AgcDriver::VulkanDevice& device, const Module& module) {
    const auto before = PrivateBytes();
    const auto started = std::chrono::steady_clock::now();
    device.Dispatch(module.result, 1, 1, 1, reinterpret_cast<std::uintptr_t>(module.code.data()));
    device.WaitIdle();
    return {module.result.spirv.size(), PrivateBytes() - before, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count()};
}


}

int main() {
#ifndef _WIN32
    std::puts("skipped: private bytes are measured on Windows");
    return VulkanTestSkipped;
#else
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        for (std::uint32_t index = 0; index < Input.size(); ++index) Input[index] = index * 2654435761u;
        const auto seed = static_cast<std::uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 1u;
        static_cast<void>(Create(*device, Compile(*device, 4u, seed)));
        std::vector<Module> modules;
        for (std::uint32_t index = 0; index < Creations; ++index) modules.push_back(Compile(*device, Blocks, seed + index + 1u));
        const auto compiledBefore = AgcDriver::VulkanDevice::DisposableCompiles();
        std::vector<Created> results(Creations);
        std::vector<std::exception_ptr> failures(Creations);
        std::mutex mutex;
        std::condition_variable changed;
        std::uint32_t turn = 0;
        bool finished = false;
        std::vector<std::thread> creators;
        for (std::uint32_t index = 0; index < Creations; ++index) {
            creators.emplace_back([&, index] {
                {
                    std::unique_lock lock(mutex);
                    changed.wait(lock, [&] { return turn == index; });
                }
                try {
                    results[index] = Create(*device, modules[index]);
                } catch (...) {
                    failures[index] = std::current_exception();
                }
                {
                    std::lock_guard lock(mutex);
                    ++turn;
                }
                changed.notify_all();
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return finished; });
            });
        }
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return turn == Creations; });
            finished = true;
        }
        changed.notify_all();
        for (auto& creator : creators) creator.join();
        for (const auto& failure : failures) {
            if (failure != nullptr) std::rethrow_exception(failure);
        }
        std::int64_t held = 0;
        for (std::uint32_t index = 0; index < Creations; ++index) {
            const auto& created = results[index];
            held += created.held;
            std::printf("creation %u: %zu SPIR-V words in %.0f ms, private bytes %+.1f MiB across the creation\n", index, created.words, created.milliseconds, created.held / 1048576.0);
            Require(created.words >= AgcDriver::VulkanDevice::DisposableCompileWords, "the test module has " + std::to_string(created.words) + " SPIR-V words, fewer than a disposable compile takes");
        }
        const auto compiled = AgcDriver::VulkanDevice::DisposableCompiles();
        std::printf("disposable compiles %llu, hits %llu, misses %llu, devices %llu in %.0f ms; private bytes %+.1f MiB across the %u creations\n", static_cast<unsigned long long>(compiled.compiles - compiledBefore.compiles), static_cast<unsigned long long>(compiled.hits - compiledBefore.hits), static_cast<unsigned long long>(compiled.misses - compiledBefore.misses), static_cast<unsigned long long>(compiled.devices - compiledBefore.devices), compiled.deviceMs - compiledBefore.deviceMs, held / 1048576.0, Creations);
        Require(compiled.compiles - compiledBefore.compiles == Creations, "large pipelines were not compiled on a disposable device");
        Require(compiled.hits - compiledBefore.hits == Creations && compiled.misses == compiledBefore.misses, "a pipeline compiled on a disposable device was not created from its cache data");
        Require(compiled.devices - compiledBefore.devices == Creations, "a disposable compile device outlived its compile");
        Require(held <= RetainedBound, "private bytes grew by " + std::to_string(held >> 20) + " MiB across " + std::to_string(Creations) + " large pipeline creations, each on its own thread kept alive, more than the " + std::to_string(RetainedBound >> 20) + " MiB their pipelines may hold");
        std::puts("pipeline compile retention tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
