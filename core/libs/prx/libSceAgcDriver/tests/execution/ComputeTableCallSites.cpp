#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Untouched = 0xcafebabeu;

alignas(256) constexpr std::array<std::uint32_t, 31> Caller{
    0x7e040300u, 0x7e280500u, 0x8f148414u, 0xf4080400u, 0xfa000000u, 0xbf8cc07fu, 0xf4280208u, 0x28000000u,
    0xbf8cc07fu, 0xbe8e0308u, 0xbe8f0309u, 0xbe8e210eu, 0x7e280500u, 0x80148114u, 0x8f148414u, 0xf4080400u,
    0xfa000000u, 0xbf8cc07fu, 0xf4280208u, 0x28000000u, 0xbf8cc07fu, 0xbe8e0308u, 0xbe8f0309u, 0xbe8e210eu,
    0xf4080600u, 0xfa000010u, 0x34060082u, 0xbf8cc07fu, 0xe0701000u, 0x80060203u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 2> AddFive{0x4a040485u, 0xbe80200eu};
alignas(256) constexpr std::array<std::uint32_t, 2> Double{0x34040481u, 0xbe80200eu};
TestGuest::Array<std::uint32_t, 16> Table;
TestGuest::Array<std::uint32_t, 8> Srt;
TestGuest::Array<std::uint32_t, Threads> Output;

std::array<std::uint32_t, 4> Buffer(const void* base, std::uint32_t bytes) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base));
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void SetRecord(std::uint32_t record, const void* target) {
    const auto code = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(target));
    Table[record * 4u] = static_cast<std::uint32_t>(code);
    Table[record * 4u + 1u] = static_cast<std::uint32_t>(code >> 32u);
    Table[record * 4u + 2u] = 0u;
    Table[record * 4u + 3u] = 0u;
}

TestGuest::Vector<std::uint32_t> Commands() {
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
    const std::array<std::uint32_t, 1> resource{2u << 1u};
    registers(0x213, resource);
    const auto srt = reinterpret_cast<std::uintptr_t>(Srt.data());
    const std::array<std::uint32_t, 2> data{static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u)};
    registers(0x240, data);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x41u});
    return words;
}

void Check(const void* first, const void* second, const std::string& what) {
    SetRecord(0, first);
    SetRecord(1, second);
    SetRecord(2, first);
    SetRecord(3, second);
    Output.fill(Untouched);
    const auto words = Commands();
    Packet packet{const_cast<std::uint32_t*>(words.data()), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Output.data()), decltype(Output)::ByteSize);
    const auto apply = [](const void* target, std::uint32_t value) { return target == AddFive.data() ? value + 5u : value * 2u; };
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        const auto expected = apply(second, apply(first, lane));
        Require(Output[lane] == expected, what + ": lane " + std::to_string(lane) + " is " + std::to_string(Output[lane]) + ", expected " + std::to_string(expected));
    }
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        const auto table = Buffer(Table.data(), static_cast<std::uint32_t>(decltype(Table)::ByteSize));
        const auto output = Buffer(Output.data(), static_cast<std::uint32_t>(decltype(Output)::ByteSize));
        std::copy(table.begin(), table.end(), Srt.begin());
        std::copy(output.begin(), output.end(), Srt.begin() + 4);
        Check(AddFive.data(), Double.data(), "two table calls one after the other, each in its own round");
        Check(Double.data(), AddFive.data(), "the two records' callees swapped");
        AgcDriverShutdown_nid_postfix();
        std::puts("compute table call site tests passed");
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
