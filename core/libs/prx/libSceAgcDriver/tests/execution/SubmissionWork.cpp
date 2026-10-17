#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "GuestTestMemory.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr std::uint32_t Nops = 40;
constexpr std::uint32_t Lists = 12;
constexpr std::uint32_t Pairs = 2;
constexpr std::uint32_t Submissions = 64;
constexpr std::uint32_t CommandWalksPerSubmission = 4;

TestGuest::Array<std::uint32_t, Lists * Pairs * 2> RegisterPairs;

void Check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("submission work: " + what);
}

struct Counts {
    std::uint64_t packets;
    std::uint64_t walks;
    std::uint64_t lists;
    std::uint64_t bytes;
};

Counts Read() {
    auto& costs = AgcDriver::DriverDetail::submissionCosts(0);
    return {costs.validatedPackets.load(), costs.commandWalks.load(), costs.registerLists.load(), costs.registerBytes.load()};
}

TestGuest::Vector<std::uint32_t> Commands() {
    TestGuest::Vector<std::uint32_t> words;
    for (std::uint32_t list = 0; list < Lists; ++list) {
        const auto address = reinterpret_cast<std::uintptr_t>(RegisterPairs.data() + list * Pairs * 2u);
        words.insert(words.end(), {0xc0036300u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), 0x80000000u, Pairs});
        for (std::uint32_t nop = 0; nop < Nops / Lists; ++nop) words.insert(words.end(), {0xc0001000u, 0u});
    }
    for (std::uint32_t nop = Nops / Lists * Lists; nop < Nops; ++nop) words.insert(words.end(), {0xc0001000u, 0u});
    return words;
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        for (std::uint32_t list = 0; list < Lists; ++list) {
            for (std::uint32_t pair = 0; pair < Pairs; ++pair) {
                RegisterPairs[(list * Pairs + pair) * 2u] = 0x00cu + pair;
                RegisterPairs[(list * Pairs + pair) * 2u + 1u] = list * 16u + pair;
            }
        }
        auto words = Commands();
        for (std::uint32_t submission = 0; submission < Submissions; ++submission) {
            const auto before = Read();
            Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
            AgcDriver::Submit(&packet, 0);
            const auto after = Read();
            const auto at = " (submission " + std::to_string(submission) + ")";
            Check(after.packets - before.packets == Nops + Lists, "validation checked " + std::to_string(after.packets - before.packets) + " packets of a submission of " + std::to_string(Nops + Lists) + at);
            Check(after.walks - before.walks == CommandWalksPerSubmission, "the submission walked its commands " + std::to_string(after.walks - before.walks) + " times, expected " + std::to_string(CommandWalksPerSubmission) + at);
            Check(after.lists - before.lists == Lists, "the submission read " + std::to_string(after.lists - before.lists) + " register lists, expected " + std::to_string(Lists) + at);
            Check(after.bytes - before.bytes == Lists * Pairs * 2u * sizeof(std::uint32_t), "the submission read " + std::to_string(after.bytes - before.bytes) + " bytes of register lists" + at);
        }
        AgcDriverWaitIdle_nid_postfix();
        LibcRunShutdown_nid_postfix();
        std::puts("submission work tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
