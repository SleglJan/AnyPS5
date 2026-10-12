#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

alignas(256) std::array<std::uint32_t, 4> StoreOneCode{0x7e000281u, 0xe0700000u, 0x80000000u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 4> StoreTwoCode{0x7e000282u, 0xe0700000u, 0x80000000u, 0xbf810000u};

alignas(256) std::array<std::uint32_t, 4> First{};
alignas(256) std::array<std::uint32_t, 4> Second{};
alignas(256) volatile std::uint32_t Done = 0;

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

std::array<std::uint32_t, 4> Descriptor(const void* pointer, std::uint32_t bytes) {
    return {Low(pointer), High(pointer) & 0xffffu, bytes, 0x31016facu};
}

void SetShaderRegister(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

void Store(std::vector<std::uint32_t>& words, const void* code, const void* target) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(code));
    SetShaderRegister(words, 0x20c, static_cast<std::uint32_t>(address >> 8u));
    SetShaderRegister(words, 0x20d, static_cast<std::uint32_t>(address >> 40u) & 0xffu);
    for (const std::uint32_t offset : {0x207u, 0x208u, 0x209u}) SetShaderRegister(words, offset, 1);
    SetShaderRegister(words, 0x212, 0);
    SetShaderRegister(words, 0x213, 4u << 1u);
    std::uint32_t offset = 0x240;
    for (const auto word : Descriptor(target, 16)) SetShaderRegister(words, offset++, word);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_AHEAD_PIPELINES", "2");
#else
    setenv("APS5_AHEAD_PIPELINES", "2", 1);
#endif
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        device.reset();
        const auto queuedBefore = AgcDriver::VulkanDevice::AheadPipelinesQueued();
        std::vector<std::uint32_t> words;
        Store(words, StoreOneCode.data(), First.data());
        Store(words, StoreTwoCode.data(), Second.data());
        words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(&Done), High(&Done), 1u});
        Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
        Require(sceAgcDriverSubmitAcb(0x20, &packet) == 0, "compute submit failed");
        AgcDriverWaitIdle_nid_postfix();
        Require(Done == 1, "the submission did not finish");
        Require(First[0] == 1, "the first dispatch did not store 1 (" + std::to_string(First[0]) + ")");
        Require(Second[0] == 2, "the dispatch whose pipeline was created ahead did not store 2 (" + std::to_string(Second[0]) + ")");
        Require(AgcDriver::VulkanDevice::AheadPipelinesQueued() > queuedBefore, "the second program's pipeline was not queued ahead while the first one's was created");
        std::cout << "pipelines created ahead test passed\n";
        AgcDriverShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << '\n';
        }
        return 1;
    }
}
