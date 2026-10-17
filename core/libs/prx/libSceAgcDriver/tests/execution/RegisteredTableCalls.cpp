#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "SceShaders.hpp"
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

struct alignas(8) Header {
    Shader shader{};
    std::array<ShaderRegister, 7> registers{};
    ShaderSpecialRegs specials{};
};

Header Registered;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

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

std::string Dispatch(const std::string& what) {
    Output.fill(Untouched);
    const auto words = Commands();
    Packet packet{const_cast<std::uint32_t*>(words.data()), static_cast<std::uint32_t>(words.size()), 0, {}};
    std::string log;
    {
        NewPath::StderrCapture capture;
        AgcDriver::Submit(&packet, 0);
        AgcDriverWaitIdle_nid_postfix();
        log = capture.Finish();
    }
    std::cerr << log;
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Output.data()), decltype(Output)::ByteSize);
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        const auto expected = (lane + 5u) * 2u;
        Require(Output[lane] == expected, what + ": lane " + std::to_string(lane) + " is " + std::to_string(Output[lane]) + ", expected " + std::to_string(expected));
    }
    return log;
}

void Register() {
    const auto address = reinterpret_cast<std::uintptr_t>(Caller.data());
    Registered.shader.file_header = 0x34333231u;
    Registered.shader.version = 0x18;
    Registered.shader.header_size = sizeof(Header);
    Registered.shader.shader_size = sizeof(Caller);
    Registered.shader.code = Caller.data();
    Registered.shader.sh_registers = Registered.registers.data();
    Registered.shader.num_sh_registers = Registered.registers.size();
    Registered.shader.specials = &Registered.specials;
    Registered.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, Threads}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 2u << 1u}}};
    std::string log;
    {
        NewPath::StderrCapture capture;
        AgcDriverRegisterShader_nid_postfix(&Registered.shader);
        AgcDriver::DriverDetail::AwaitRegistrationPreparation();
        AgcDriver::DriverDetail::CloseRegistrationPreparation();
        log = capture.Finish();
    }
    std::cerr << log;
    Expect(log.find("could not be prepared at registration") == std::string::npos, "the registration of a program whose s_swappc_b64 calls a target loaded from a table left it to be prepared at use");
    Expect(log.find("could not be created at registration") == std::string::npos, "the registration could not create the head of the table-call program");
    const auto registry = AgcDriver::DriverDetail::Driver::Get().RegisteredShaders();
    Require(registry != nullptr, "the registration published no shader registry");
    const auto snapshot = AgcDriver::DriverDetail::RegisteredHeader(*registry, &Registered.shader);
    Require(snapshot != nullptr, "the registration published no snapshot");
    AgcDriver::DriverDetail::ShaderPreparationTransaction transaction;
    const auto& state = transaction.Read(*snapshot);
    Expect(!state.deferred, "the registration deferred the preparation of the table-call program to its use");
    Expect(!state.entries.empty(), "the registration prepared no head of the table-call program");
}

void Run() {
    const auto table = Buffer(Table.data(), static_cast<std::uint32_t>(decltype(Table)::ByteSize));
    const auto output = Buffer(Output.data(), static_cast<std::uint32_t>(decltype(Output)::ByteSize));
    std::copy(table.begin(), table.end(), Srt.begin());
    std::copy(output.begin(), output.end(), Srt.begin() + 4);
    SetRecord(0, AddFive.data());
    SetRecord(1, Double.data());
    SetRecord(2, AddFive.data());
    SetRecord(3, Double.data());
    const auto before = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Register();
    const auto registered = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Expect(registered.ahead == before.ahead + 1u && registered.inFrames == before.inFrames, "the registration created " + std::to_string(registered.ahead - before.ahead) + " compute pipelines ahead and " + std::to_string(registered.inFrames - before.inFrames) + " inside frames, expected the head's one ahead");
    const auto continuations = AgcDriver::DriverDetail::TableCallContinuationsPreparedAtUse();
    const auto log = Dispatch("the first dispatch");
    Expect(log.find("has no artifact prepared at registration") == std::string::npos, "the first dispatch prepared a part of the table-call program at use that its registration had prepared");
    const auto named = AgcDriver::DriverDetail::TableCallContinuationsPreparedAtUse() - continuations;
    Expect(named != 0u, "the first dispatch counted no continuation for the callees its records first name");
    const auto dispatched = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Expect(dispatched.ahead == registered.ahead && dispatched.inFrames == registered.inFrames + named, "the first dispatch created " + std::to_string(dispatched.inFrames - registered.inFrames) + " compute pipelines inside frames, expected only its " + std::to_string(named) + " continuations': the head was created at registration");
    static_cast<void>(Dispatch("the second dispatch"));
    const auto again = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Expect(again.inFrames == dispatched.inFrames && again.ahead == dispatched.ahead, "the second dispatch created compute pipelines");
    Expect(AgcDriver::DriverDetail::TableCallContinuationsPreparedAtUse() == continuations + named, "the second dispatch prepared continuations at use");
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
            std::cerr << failures << " registered table call checks failed\n";
            return 1;
        }
        std::puts("registered table call tests passed");
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
