#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/Dispatch/TableCallDispatch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "BranchSplitProgram.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using BranchSplitProgram::Condition;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Groups = 4;
constexpr std::uint32_t Operations = 25000;
constexpr std::uint32_t TreeOperations = 30000;
constexpr std::uint32_t CalleeOperations = 2000;
constexpr std::uint32_t Untouched = 0xcafebabeu;

TestGuest::Array<std::uint32_t, 4> Srt;
TestGuest::Array<std::uint32_t, Threads * Groups> Output;

using ExpectedValue = std::function<std::uint32_t(std::uint32_t group, std::uint32_t thread)>;

std::vector<std::uint32_t> SrtUserData() {
    const auto srt = reinterpret_cast<std::uintptr_t>(Srt.data());
    return {static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u)};
}

TestGuest::Vector<std::uint32_t> Commands(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData) {
    TestGuest::Vector<std::uint32_t> words;
    const auto registers = [&](std::uint32_t first, std::span<const std::uint32_t> values) {
        words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
        words.push_back(first);
        words.insert(words.end(), values.begin(), values.end());
    };
    const std::array<std::uint32_t, 3> shape{Threads, 1, 1};
    registers(0x207, shape);
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    const std::array<std::uint32_t, 2> program{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u)};
    registers(0x20c, program);
    const std::array<std::uint32_t, 1> resource{(static_cast<std::uint32_t>(userData.size()) << 1u) | (1u << 7u)};
    registers(0x213, resource);
    registers(0x240, userData);
    words.insert(words.end(), {0xc0031500u, Groups, 1, 1, 0x41u});
    return words;
}

void Dispatch(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ExpectedValue& expectedValue, const std::string& what) {
    Output.fill(Untouched);
    const auto words = Commands(code, userData);
    Packet packet{const_cast<std::uint32_t*>(words.data()), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Output.data()), decltype(Output)::ByteSize);
    for (std::uint32_t group = 0; group < Groups; ++group) {
        for (std::uint32_t lane = 0; lane < Threads; ++lane) {
            const auto thread = group * Threads + lane;
            const auto expected = expectedValue(group, thread);
            Require(Output[thread] == expected, what + ": group " + std::to_string(group) + " lane " + std::to_string(lane) + " is " + std::to_string(Output[thread]) + ", expected " + std::to_string(expected));
        }
    }
}

std::span<std::uint32_t> GuestCode(const std::vector<std::uint32_t>& generated) {
    const std::span code(reinterpret_cast<std::uint32_t*>(TestGuest::Allocate(generated.size() * sizeof(std::uint32_t), 256)), generated.size());
    std::copy(generated.begin(), generated.end(), code.begin());
    return code;
}

void CheckSplit(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ExpectedValue& expectedValue, const std::string& what) {
    const auto before = AgcDriver::DriverDetail::BranchSplitCounters();
    Dispatch(code, userData, expectedValue, what);
    const auto split = AgcDriver::DriverDetail::BranchSplitCounters();
    Require(split.programs == before.programs + 1u, what + ": the program was dispatched whole, not split at its branch");
    Require(split.largestProgramWords > ShaderRecompiler::GiantProgramWords, what + ": the whole program is " + std::to_string(split.largestProgramWords) + " SPIR-V words, not over the giant bound");
    Require(split.largestPartWords != 0u && split.largestPartWords <= ShaderRecompiler::GiantProgramWords, what + ": the largest part is " + std::to_string(split.largestPartWords) + " SPIR-V words, over the giant bound");
    std::printf("%s: whole %zu SPIR-V words, largest part %zu\n", what.c_str(), split.largestProgramWords, split.largestPartWords);
    Dispatch(code, userData, expectedValue, what + ", dispatched again");
    Require(AgcDriver::DriverDetail::BranchSplitCounters().programs == split.programs, what + ": the program was split again on its second dispatch");
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        const auto output = std::array<std::uint32_t, 4>{static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(Output.data())), static_cast<std::uint32_t>((reinterpret_cast<std::uintptr_t>(Output.data()) >> 32u) & 0xffffu), static_cast<std::uint32_t>(decltype(Output)::ByteSize), 0x31016facu};
        std::copy(output.begin(), output.end(), Srt.begin());
        const auto srt = SrtUserData();
        for (const auto condition : {Condition::Scc, Condition::Vcc}) {
            const auto arms = [](std::uint32_t group, std::uint32_t thread) { return BranchSplitProgram::Expected(BranchSplitProgram::ArmOf(group), Operations, thread); };
            CheckSplit(GuestCode(BranchSplitProgram::Code(condition, Operations)), srt, arms, condition == Condition::Scc ? "a program over the giant bound that branches on scc" : "a program over the giant bound that branches on vcc");
        }
        const auto tree = [](std::uint32_t group, std::uint32_t thread) { return BranchSplitProgram::Expected(BranchSplitProgram::TreeArmOf(group), TreeOperations, thread); };
        CheckSplit(GuestCode(BranchSplitProgram::TreeCode(TreeOperations)), srt, tree, "a program whose halves stay over the giant bound after one split point");
        const auto callee = GuestCode(BranchSplitProgram::CalleeCode(CalleeOperations));
        const auto calleeAddress = reinterpret_cast<std::uintptr_t>(callee.data());
        const std::vector<std::uint32_t> callerData{srt[0], srt[1], static_cast<std::uint32_t>(calleeAddress), static_cast<std::uint32_t>(calleeAddress >> 32u)};
        const auto called = [](std::uint32_t group, std::uint32_t thread) { return BranchSplitProgram::CallerExpected(group, Operations, CalleeOperations, false, thread); };
        CheckSplit(GuestCode(BranchSplitProgram::CallerCode(Operations, false).code), callerData, called, "a program over the giant bound that calls user data");
        AgcDriverShutdown_nid_postfix();
        std::puts("compute branch split tests passed");
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
