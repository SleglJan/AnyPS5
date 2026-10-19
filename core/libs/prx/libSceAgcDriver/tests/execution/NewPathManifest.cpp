#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "SceShaders.hpp"
#include "VulkanTestDevice.hpp"
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>

#ifdef _WIN32

namespace {

using namespace NewPath;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Programs = 4;
constexpr std::uint32_t Untouched = 0xcafebabeu;

struct Header {
    Shader shader{};
    std::array<ShaderRegister, 7> registers{};
    ShaderSpecialRegs specials{};
};

Program Codes[Programs];
Header Headers[Programs];

void BuildPrograms() {
    using namespace Isa;
    for (std::uint32_t index = 0; index < Programs; ++index) {
        Codes[index] << Vop2(VLshlRevB32, 1, InlineInteger(2), 0)
                     << Mubuf(BufferLoadDword, 2, 1, 0)
                     << WaitAll
                     << Vop2(VAddNcU32, 2, InlineInteger(3 + index), 2)
                     << Mubuf(BufferStoreDword, 2, 1, 4)
                     << EndProgram;
    }
}

void Register(std::uint32_t index) {
    auto& header = Headers[index];
    const auto address = Codes[index].Address();
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(Codes[index].Code().size_bytes());
    header.shader.code = Codes[index].words.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000u;
    header.registers = {{{0x20c, Low(address >> 8u)}, {0x20d, Low(address >> 40u)}, {0x207, Threads}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 8u << 1u}}};
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void RegisterAll() {
    for (std::uint32_t index = 0; index < Programs; ++index) Register(index);
    AgcDriver::DriverDetail::CloseRegistrationPreparation();
}

class Buffers {
public:
    explicit Buffers(std::uint32_t offset) : memory(1), offset(offset) {
        for (std::uint32_t lane = 0; lane < Threads; ++lane) {
            Input()[lane] = lane * 2654435761u + offset;
            Output()[lane] = Untouched;
        }
    }

    void Dispatch(std::uint32_t index, const std::string& what) {
        for (std::uint32_t lane = 0; lane < Threads; ++lane) Output()[lane] = Untouched;
        std::array<std::uint32_t, 8> user{};
        const auto source = BufferResource(memory.Address(offset), Threads * 4u);
        const auto target = BufferResource(memory.Address(offset + 0x8000u), Threads * 4u + offset);
        std::copy(source.begin(), source.end(), user.begin());
        std::copy(target.begin(), target.end(), user.begin() + 4);
        std::vector<std::uint32_t> words;
        const auto registers = [&](std::uint32_t first, std::initializer_list<std::uint32_t> values) {
            words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
            words.push_back(first);
            words.insert(words.end(), values);
        };
        const auto address = Codes[index].Address();
        registers(0x207, {Threads, 1u, 1u});
        registers(0x20c, {Low(address >> 8u), Low(address >> 40u)});
        registers(0x213, {8u << 1u});
        words.push_back(0xc0007600u | (8u << 16u));
        words.push_back(0x240u);
        words.insert(words.end(), user.begin(), user.end());
        words.insert(words.end(), {0xc0031500u, 1u, 1u, 1u, 0x8041u});
        Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
        AgcDriver::Submit(&packet, 0);
        WaitIdle();
        AgcDriver::CreationPool::WaitBackground();
        AgcDriver::GuestMemory::FlushGpuWrites(memory.Address(offset + 0x8000u), Threads * 4u);
        for (std::uint32_t lane = 0; lane < Threads; ++lane) Require(Output()[lane] == Input()[lane] + 3u + index, what + ": lane " + std::to_string(lane) + " is " + Hex(Output()[lane]) + ", expected " + Hex(Input()[lane] + 3u + index));
    }

private:
    volatile std::uint32_t* Input() const { return reinterpret_cast<volatile std::uint32_t*>(memory.Data() + offset); }
    volatile std::uint32_t* Output() const { return reinterpret_cast<volatile std::uint32_t*>(memory.Data() + offset + 0x8000u); }

    PrivatePages memory;
    std::uint32_t offset;
};

void SetVariable(const char* name, const std::string& value) {
    Require(_putenv_s(name, value.c_str()) == 0, std::string("cannot set ") + name);
}

int RunChild(const char* executable, const char* mode) {
    const std::string command = "\"\"" + std::string(executable) + "\" " + mode + "\"";
    return std::system(command.c_str());
}

void Record() {
    RegisterAll();
    Buffers buffers(0x100u);
    for (std::uint32_t index = 0; index < Programs; ++index) buffers.Dispatch(index, "the recording run, program " + std::to_string(index));
    const auto creations = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Require(creations.inFrames == Programs, "the recording run created " + std::to_string(creations.inFrames) + " compute pipelines inside frames, expected " + std::to_string(Programs));
    Require(creations.segmentsInFrames == Programs, "the recording run created " + std::to_string(creations.segmentsInFrames) + " segment program pipelines inside frames, expected " + std::to_string(Programs));
    AgcDriverShutdown_nid_postfix();
}

void UseSome() {
    RegisterAll();
    Buffers buffers(0x200u);
    buffers.Dispatch(0, "a later run, program 0");
    buffers.Dispatch(2, "a later run, program 2");
    AgcDriverShutdown_nid_postfix();
}

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

void Check() {
    const auto before = AgcDriver::VulkanDevice::ComputePipelineCreations();
    RegisterAll();
    const auto registered = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Expect(registered.segmentsAhead == before.segmentsAhead + 3u && registered.segmentsInFrames == before.segmentsInFrames, "registration after three runs created " + std::to_string(registered.segmentsAhead - before.segmentsAhead) + " segment program pipelines ahead, expected 3 (program 1's expired with its walk program's record, program 3's kept)");
    Expect(registered.ahead == before.ahead + 3u && registered.inFrames == before.inFrames, "registration after three runs that used programs 0 and 2 and reached past program 1's first use created " + std::to_string(registered.ahead - before.ahead) + " pipelines ahead, expected 3 (program 1's record expired, program 3's kept: the runs ended before its first use)");
    Buffers first(0x300u);
    Buffers second(0x1000u);
    first.Dispatch(3, "program 3, kept although the runs never reached it");
    second.Dispatch(0, "program 0 through other buffers than the recording run's");
    first.Dispatch(0, "program 0 through a second set of buffers");
    const auto kept = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Expect(kept.segmentsInFrames == registered.segmentsInFrames, "dispatches of programs whose segment program pipelines were created ahead created " + std::to_string(kept.segmentsInFrames - registered.segmentsInFrames) + " inside frames");
    Expect(kept.inFrames == registered.inFrames, "dispatches of programs whose pipelines were created ahead created " + std::to_string(kept.inFrames - registered.inFrames) + " inside frames: a pipeline depended on the buffers it was dispatched with");
    first.Dispatch(1, "program 1, whose record expired");
    const auto expired = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Expect(expired.segmentsInFrames == kept.segmentsInFrames + 1u, "the dispatch of program 1, whose record expired, created " + std::to_string(expired.segmentsInFrames - kept.segmentsInFrames) + " segment program pipelines inside its frame, expected one");
    Expect(expired.inFrames == kept.inFrames + 1u, "the dispatch of program 1, whose record expired, created " + std::to_string(expired.inFrames - kept.inFrames) + " pipelines inside its frame, expected one");
    AgcDriverShutdown_nid_postfix();
}

}

#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    try {
        BuildPrograms();
        if (argc == 2 && std::string(argv[1]) == "--record") {
            Record();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--use") {
            UseSome();
            return 0;
        }
        Require(argc == 1, "usage: [--record | --use]");
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        const auto directory = std::filesystem::temp_directory_path() / ("aps5-new-path-manifest-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        SetVariable("ANYPS5_NO_SHADER_CACHE", "");
        SetVariable("ANYPS5_SHADER_CACHE_DIR", directory.string());
        Require(RunChild(argv[0], "--record") == 0, "the recording run failed");
        for (std::uint32_t run = 0; run < 3; ++run) Require(RunChild(argv[0], "--use") == 0, "later run " + std::to_string(run + 2) + " failed");
        Check();
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception&) {
        }
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d new path manifest checks failed\n", failures);
        return 1;
    }
    std::printf("new path manifest tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
