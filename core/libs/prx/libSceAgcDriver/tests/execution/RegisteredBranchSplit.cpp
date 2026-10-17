#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/Dispatch/TableCallDispatch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "BranchSplitProgram.hpp"
#include "SceShaders.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
constexpr std::uint32_t Untouched = 0xcafebabeu;
constexpr std::uint32_t Resource = (2u << 1u) | (1u << 7u);

TestGuest::Array<std::uint32_t, 4> Srt;
TestGuest::Array<std::uint32_t, Threads * Groups> Output;

struct alignas(8) Header {
    Shader shader{};
    std::array<ShaderRegister, 7> registers{};
    ShaderSpecialRegs specials{};
};

Header Registered;

std::span<std::uint32_t> GuestCode(const std::vector<std::uint32_t>& generated) {
    const std::span code(reinterpret_cast<std::uint32_t*>(TestGuest::Allocate(generated.size() * sizeof(std::uint32_t), 256)), generated.size());
    std::copy(generated.begin(), generated.end(), code.begin());
    return code;
}

TestGuest::Vector<std::uint32_t> Commands(std::span<const std::uint32_t> code) {
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
    const std::array<std::uint32_t, 1> resource{Resource};
    registers(0x213, resource);
    const auto srt = reinterpret_cast<std::uintptr_t>(Srt.data());
    const std::array<std::uint32_t, 2> data{static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u)};
    registers(0x240, data);
    words.insert(words.end(), {0xc0031500u, Groups, 1, 1, 0x41u});
    return words;
}

void Register(std::span<const std::uint32_t> code, const std::string& what) {
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    Registered.shader.file_header = 0x34333231u;
    Registered.shader.version = 0x18;
    Registered.shader.header_size = sizeof(Header);
    Registered.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
    Registered.shader.code = code.data();
    Registered.shader.sh_registers = Registered.registers.data();
    Registered.shader.num_sh_registers = Registered.registers.size();
    Registered.shader.specials = &Registered.specials;
    Registered.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, Threads}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, Resource}}};
    AgcDriverRegisterShader_nid_postfix(&Registered.shader);
    AgcDriver::DriverDetail::AwaitRegistrationPreparation();
    AgcDriver::DriverDetail::CloseRegistrationPreparation();
    const auto registry = AgcDriver::DriverDetail::Driver::Get().RegisteredShaders();
    Require(registry != nullptr, what + ": the registration published no shader registry");
    Require(AgcDriver::DriverDetail::RegisteredHeader(*registry, &Registered.shader) != nullptr, what + ": the registration published no snapshot");
}

void Dispatch(std::span<const std::uint32_t> code, const std::string& what) {
    Output.fill(Untouched);
    const auto words = Commands(code);
    Packet packet{const_cast<std::uint32_t*>(words.data()), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Output.data()), decltype(Output)::ByteSize);
    for (std::uint32_t group = 0; group < Groups; ++group) {
        for (std::uint32_t lane = 0; lane < Threads; ++lane) {
            const auto thread = group * Threads + lane;
            const auto expected = BranchSplitProgram::Expected(BranchSplitProgram::ArmOf(group), Operations, thread);
            Require(Output[thread] == expected, what + ": group " + std::to_string(group) + " lane " + std::to_string(lane) + " is " + std::to_string(Output[thread]) + ", expected " + std::to_string(expected));
        }
    }
}

void Run(bool warm) {
    const auto what = std::string(warm ? "a warm start" : "a cold start");
    const auto output = std::array<std::uint32_t, 4>{static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(Output.data())), static_cast<std::uint32_t>((reinterpret_cast<std::uintptr_t>(Output.data()) >> 32u) & 0xffffu), static_cast<std::uint32_t>(decltype(Output)::ByteSize), 0x31016facu};
    std::copy(output.begin(), output.end(), Srt.begin());
    const auto code = GuestCode(BranchSplitProgram::Code(Condition::Scc, Operations));
    const auto before = AgcDriver::DriverDetail::BranchSplitCounters();
    const auto start = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Register(code, what);
    const auto registered = AgcDriver::DriverDetail::BranchSplitCounters();
    const auto ahead = AgcDriver::VulkanDevice::ComputePipelineCreations();
    const auto variants = registered.ahead - before.ahead;
    Require(variants != 0u && registered.programs == before.programs + variants, what + ": the registration of a program over the giant bound did not split it");
    Require(registered.recorded == before.recorded + (warm ? variants : 0u), what + (warm ? ": the registration measured the split again instead of the split the manifest recorded" : ": the registration split from a record it had no manifest for"));
    Require(ahead.inFrames == start.inFrames && ahead.ahead >= start.ahead + 3u, what + ": the registration created " + std::to_string(ahead.ahead - start.ahead) + " compute pipelines ahead and " + std::to_string(ahead.inFrames - start.inFrames) + " inside frames, expected the head and both continuations ahead");
    std::printf("%s: the registration split %llu variants of a program of %zu SPIR-V words and created %llu pipelines ahead\n", what.c_str(), static_cast<unsigned long long>(variants), registered.largestProgramWords, static_cast<unsigned long long>(ahead.ahead - start.ahead));
    Dispatch(code, what + ", the first dispatch");
    const auto dispatched = AgcDriver::DriverDetail::BranchSplitCounters();
    const auto first = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Require(dispatched.programs == registered.programs && dispatched.registeredHits == registered.registeredHits + 1u, what + ": the first dispatch did not use the split its registration prepared");
    Require(dispatched.largestPartWords != 0u && dispatched.largestPartWords <= ShaderRecompiler::GiantProgramWords, what + ": the largest part is " + std::to_string(dispatched.largestPartWords) + " SPIR-V words");
    Require(first.inFrames == ahead.inFrames && first.ahead == ahead.ahead, what + ": the first dispatch created " + std::to_string(first.inFrames - ahead.inFrames) + " compute pipelines inside frames that its registration had created");
    Dispatch(code, what + ", the second dispatch");
    const auto again = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Require(again.inFrames == first.inFrames && AgcDriver::DriverDetail::BranchSplitCounters().programs == dispatched.programs, what + ": the second dispatch split or created again");
    AgcDriverShutdown_nid_postfix();
}

void SetVariable(const char* name, const std::string& value) {
#ifdef _WIN32
    Require(_putenv_s(name, value.c_str()) == 0, std::string("cannot set ") + name);
#else
    Require((value.empty() ? unsetenv(name) : setenv(name, value.c_str(), 1)) == 0, std::string("cannot set ") + name);
#endif
}

int RunCold(const char* executable) {
#ifdef _WIN32
    const std::string command = "\"\"" + std::string(executable) + "\" --cold\"";
#else
    const std::string command = "'" + std::string(executable) + "' --cold";
#endif
    return std::system(command.c_str());
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--cold") {
            Run(false);
            return 0;
        }
        Require(argc == 1, "usage: [--cold]");
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        const auto directory = std::filesystem::temp_directory_path() / ("aps5-registered-split-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        SetVariable("ANYPS5_NO_SHADER_CACHE", "");
        SetVariable("ANYPS5_SHADER_CACHE_DIR", directory.string());
        Require(RunCold(argv[0]) == 0, "the cold start failed");
        Run(true);
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        std::puts("registered branch split tests passed");
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
