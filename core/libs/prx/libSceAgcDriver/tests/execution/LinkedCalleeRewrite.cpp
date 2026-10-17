#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>

namespace {

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Untouched = 0xcafebabeu;
constexpr std::uint32_t UserWords = 9;
constexpr std::uint32_t EndProgram = 0xbf810000u;

alignas(256) constexpr std::array<std::uint32_t, 6> Caller{0x34020082u, 0x7e040300u, 0xbe8e2102u, 0xe0701000u, 0x80010201u, EndProgram};
TestGuest::Array<std::uint32_t, 64> Callee;
TestGuest::Array<std::uint32_t, 64> CalleeCopy;
TestGuest::Array<std::uint32_t, Threads> Output;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

std::uint32_t AddInline(std::uint32_t value) {
    return 0x4a040480u + value;
}

void WriteCallee(TestGuest::Array<std::uint32_t, 64>& code, std::uint32_t add) {
    code.fill(EndProgram);
    code[0] = AddInline(add);
    code[1] = 0xbe80200eu;
}

TestGuest::Vector<std::uint32_t> Commands(const void* callee, std::uint32_t extra) {
    TestGuest::Vector<std::uint32_t> words;
    const auto registers = [&](std::uint32_t first, std::span<const std::uint32_t> values) {
        words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
        words.push_back(first);
        words.insert(words.end(), values.begin(), values.end());
    };
    const std::array<std::uint32_t, 3> shape{Threads, 1, 1};
    registers(0x207, shape);
    const auto address = reinterpret_cast<std::uintptr_t>(Caller.data());
    const std::array<std::uint32_t, 2> program{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u)};
    registers(0x20c, program);
    const std::array<std::uint32_t, 1> resource{UserWords << 1u};
    registers(0x213, resource);
    const auto target = reinterpret_cast<std::uintptr_t>(callee);
    const auto output = reinterpret_cast<std::uintptr_t>(Output.data());
    const std::array<std::uint32_t, UserWords> data{0u, 0u, static_cast<std::uint32_t>(target), static_cast<std::uint32_t>(target >> 32u),
        static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), static_cast<std::uint32_t>(decltype(Output)::ByteSize), 0x31016facu, extra};
    registers(0x240, data);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x41u});
    return words;
}

std::uint64_t Dispatch(const std::string& what, const void* callee, std::uint32_t extra, std::uint32_t add) {
    Output.fill(Untouched);
    const auto compiles = AgcDriver::DriverDetail::Driver::Get().RunTimeDispatchCompiles();
    const auto words = Commands(callee, extra);
    Packet packet{const_cast<std::uint32_t*>(words.data()), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Output.data()), decltype(Output)::ByteSize);
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        const auto expected = lane + add;
        if (Output[lane] == expected) continue;
        Expect(false, what + ": lane " + std::to_string(lane) + " is " + std::to_string(Output[lane]) + ", expected " + std::to_string(expected));
        break;
    }
    return AgcDriver::DriverDetail::Driver::Get().RunTimeDispatchCompiles() - compiles;
}

void Run() {
    WriteCallee(Callee, 5);
    WriteCallee(CalleeCopy, 5);
    const auto first = Dispatch("the first dispatch", Callee.data(), 1u, 5u);
    Expect(first == 1u, "the first dispatch compiled " + std::to_string(first) + " programs, expected 1");
    const auto otherData = Dispatch("a dispatch with other user data outside the call's address", Callee.data(), 2u, 5u);
    Expect(otherData == 0u, "a dispatch whose user data changed only outside the callee address compiled " + std::to_string(otherData) + " programs, expected 0: the linked program depends on the callee's code, not on the user data");
    const auto otherAddress = Dispatch("a dispatch that calls the same code at another address", CalleeCopy.data(), 3u, 5u);
    Expect(otherAddress == 0u, "a dispatch that calls identical callee code at another address compiled " + std::to_string(otherAddress) + " programs, expected 0");
    WriteCallee(Callee, 7);
    const auto rewritten = Dispatch("a dispatch after the guest rewrote the callee", Callee.data(), 3u, 7u);
    Expect(rewritten == 1u, "a dispatch after the guest rewrote its callee compiled " + std::to_string(rewritten) + " programs, expected 1 for the new code");
    const auto back = Dispatch("a dispatch that calls the first code again", CalleeCopy.data(), 4u, 5u);
    Expect(back == 0u, "a dispatch that calls the first callee code again compiled " + std::to_string(back) + " programs, expected 0");
    const auto again = Dispatch("a second dispatch of the rewritten callee", Callee.data(), 5u, 7u);
    Expect(again == 0u, "a second dispatch of the rewritten callee compiled " + std::to_string(again) + " programs, expected 0");
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Run();
        AgcDriverShutdown_nid_postfix();
        if (failures != 0) {
            std::cerr << failures << " linked callee checks failed\n";
            return 1;
        }
        std::puts("linked callee tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception&) {
        }
        return 1;
    }
}
