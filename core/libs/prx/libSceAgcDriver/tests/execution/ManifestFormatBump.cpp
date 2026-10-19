#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineManifest.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineReplay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PipelineCreation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "SceShaders.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include "ShaderCacheDirectory.hpp"
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint32_t Threads = 64;

alignas(256) constexpr std::array<std::uint32_t, 64> Caller{
    0x7e020280u, 0xbe940383u, 0xbe8e0300u, 0xbe8f0301u, 0xbe8e210eu, 0xbe8e0302u, 0xbe8f0303u, 0xbe8e210eu,
    0x80948114u, 0xbf078014u, 0xbf85fff7u, 0x34040082u, 0xe0701000u, 0x80010102u, 0xbf810000u, 0xbf9f0000u,
    0x30306c73u, 0x00000040u, 0x00000008u};
alignas(256) constexpr std::array<std::uint32_t, 2> AddFive{0x4a020285u, 0xbe80200eu};
alignas(256) constexpr std::array<std::uint32_t, 2> AddSixtyFour{0x4a0202c0u, 0xbe80200eu};
TestGuest::Array<std::uint32_t, Threads> Output;

alignas(256) constexpr std::array<std::uint32_t, 8> VertexCode{0x7e000208u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u, 0xf80008cfu, 0x03020100u, 0xbf810000u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 8> PixelCode{0x7e000200u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};
TestGuest::Array<std::byte, 1024> ColorMemory;

void Append(TestGuest::Vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

void RegisterCaller() {
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 7> registers{};
        ShaderSpecialRegs specials{};
    };
    alignas(8) static Header header{};
    const auto address = reinterpret_cast<std::uintptr_t>(Caller.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = sizeof(Caller);
    header.shader.code = Caller.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000u;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, Threads}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 8u << 1u}}};
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

struct GraphicsHeader {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) GraphicsHeader VertexHeader{};
alignas(8) GraphicsHeader PixelHeader{};

void RegisterGraphics(GraphicsHeader& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(GraphicsHeader);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void Register() {
    RegisterCaller();
    RegisterGraphics(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
    RegisterGraphics(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
}

void Dispatch() {
    Output.fill(0xcafebabeu);
    const auto low = reinterpret_cast<std::uintptr_t>(AddFive.data());
    const auto high = reinterpret_cast<std::uintptr_t>(AddSixtyFour.data());
    const auto output = reinterpret_cast<std::uintptr_t>(Output.data());
    const std::array<std::uint32_t, 8> data{static_cast<std::uint32_t>(low), static_cast<std::uint32_t>(low >> 32u), static_cast<std::uint32_t>(high), static_cast<std::uint32_t>(high >> 32u),
        static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), static_cast<std::uint32_t>(decltype(Output)::ByteSize), 0x31016facu};
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
    const std::array<std::uint32_t, 1> resource{static_cast<std::uint32_t>(data.size()) << 1u};
    registers(0x213, resource);
    registers(0x240, data);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Output.data()), decltype(Output)::ByteSize);
    for (std::uint32_t lane = 0; lane < Threads; ++lane) Require(Output[lane] == 3u * (5u + 64u), "the user-data callees: lane " + std::to_string(lane) + " is " + std::to_string(Output[lane]));
}

void Draw() {
    const auto color = reinterpret_cast<std::uintptr_t>(ColorMemory.data());
    const auto vertex = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(PixelCode.data());
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(color >> 8u)}, {0x390, static_cast<std::uint32_t>(color >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
    };
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 1u << 1u}, {0x8c, 0},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 1u << 1u}, {0x00c, std::bit_cast<std::uint32_t>(1.0f)},
    };
    TestGuest::Vector<std::uint32_t> commands;
    Append(commands, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    Append(commands, 0x2d, {3, 2});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

void SetVariable(const char* name, const std::string& value) {
#ifdef _WIN32
    Require(_putenv_s(name, value.c_str()) == 0, std::string("cannot set ") + name);
#else
    Require((value.empty() ? unsetenv(name) : setenv(name, value.c_str(), 1)) == 0, std::string("cannot set ") + name);
#endif
}

int RunRecording(const char* executable) {
#ifdef _WIN32
    const std::string command = "\"\"" + std::string(executable) + "\" --record\"";
#else
    const std::string command = "'" + std::string(executable) + "' --record";
#endif
    return std::system(command.c_str());
}

void Record() {
    Register();
    AgcDriver::DriverDetail::CloseRegistrationPreparation();
    const auto before = AgcDriver::PipelineReplay::Totals();
    Dispatch();
    Draw();
    AgcDriver::CreationPool::WaitBackground();
    const auto after = AgcDriver::PipelineReplay::Totals();
    const auto compute = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Require(after.programsAtUse == before.programsAtUse + 3u && after.programsRecorded == before.programsRecorded + 3u, "the recording run prepared " + std::to_string(after.programsAtUse - before.programsAtUse) + " programs at use and recorded " + std::to_string(after.programsRecorded - before.programsRecorded) + ", expected the linked compute program and both graphics stages");
    Require(compute.inFrames == 1u && after.inFrames == before.inFrames + 1u && after.recorded == before.recorded + 1u, "the recording run did not create and record one compute and one graphics pipeline inside frames");
    Require(after.segmentProgramsAtUse == before.segmentProgramsAtUse + 1u && after.segmentProgramsRecorded == before.segmentProgramsRecorded + 1u && compute.segmentsInFrames == 1u, "the recording run prepared " + std::to_string(after.segmentProgramsAtUse - before.segmentProgramsAtUse) + " segment programs, recorded " + std::to_string(after.segmentProgramsRecorded - before.segmentProgramsRecorded) + " and created " + std::to_string(compute.segmentsInFrames) + " segment program pipelines, expected the compute program's one");
    AgcDriverShutdown_nid_postfix();
}

std::array<std::uint64_t, 2> Older(const std::array<std::uint64_t, 2>& identity) {
    return {identity[0] ^ 0x0123456789abcdefull, identity[1] ^ 0xfedcba9876543210ull};
}

std::string RefusedRequest(const std::string& request) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    Require(request.size() >= 8u, "a recorded request is too short to carry a version");
    auto text = request;
    const auto value = [&](char symbol) { return static_cast<std::uint32_t>(std::string_view(alphabet).find(symbol)); };
    std::uint32_t chunk = (value(text[4]) << 18u) | (value(text[5]) << 12u) | (value(text[6]) << 6u) | value(text[7]);
    chunk = (chunk & 0xff00ffu) | (0x63u << 8u);
    text[4] = alphabet[(chunk >> 18u) & 63u];
    text[5] = alphabet[(chunk >> 12u) & 63u];
    text[6] = alphabet[(chunk >> 6u) & 63u];
    text[7] = alphabet[chunk & 63u];
    return text;
}

void AgeCache(const std::filesystem::path& directory) {
    std::filesystem::path manifestPath;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_directory()) std::filesystem::remove_all(entry.path());
        else if (entry.path().extension() == ".manifest") manifestPath = entry.path();
    }
    Require(!manifestPath.empty(), "the recording run left no pipeline manifest");
    std::vector<std::byte> bytes;
    Require(ShaderRecompiler::ReadWholeFile(manifestPath, bytes), "the pipeline manifest cannot be read");
    std::uint64_t source = 0;
    auto records = AgcDriver::PipelineManifest::Decode(bytes, source);
    Require(records.has_value() && records->prepared.size() == 4u && records->compute.size() == 2u && records->graphics.size() == 1u, "the recording run did not leave three programs, one segment program, their two compute pipelines and one graphics pipeline in its manifest");
    for (auto& entry : records->compute) {
        entry.record.program = Older(entry.record.program);
        entry.record.bindings.emplace_back(99u, 1u);
    }
    for (auto& entry : records->graphics) {
        for (auto& stage : entry.record.stages) {
            stage.program = Older(stage.program);
            stage.pushBytes += 4u;
            stage.bindings.push_back({1u, 0u, 0u, 99u, 1u});
        }
    }
    for (auto& record : records->prepared) record.identity = Older(record.identity);
    auto refused = records->prepared.front();
    refused.codeOffset += 4u;
    refused.request = RefusedRequest(refused.request);
    records->prepared.push_back(refused);
    const auto aged = AgcDriver::PipelineManifest::Encode(source ^ 1u, *records);
    Require(ShaderRecompiler::WriteFileAtomically(manifestPath, aged), "the aged pipeline manifest cannot be written");
}

void Replay() {
    const auto start = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Register();
    AgcDriver::DriverDetail::CloseRegistrationPreparation();
    const auto registered = AgcDriver::PipelineReplay::Totals();
    const auto ahead = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Require(registered.segmentProgramsRecompiled == 1u && registered.segmentProgramsReplayed == 0u, "the registrations prepared again " + std::to_string(registered.segmentProgramsRecompiled) + " of the one segment program an earlier build prepared at use (" + std::to_string(registered.segmentProgramsReplayed) + " replayed)");
    Require(registered.programsLoaded == 5u && registered.programsRecompiled == 3u && registered.programsReplayed == 0u && registered.programsStale == 0u && registered.programsFailed == 0u, "the registrations prepared again " + std::to_string(registered.programsRecompiled) + " of the three programs an earlier build prepared at use (" + std::to_string(registered.programsReplayed) + " replayed, " + std::to_string(registered.programsStale) + " stale, " + std::to_string(registered.programsFailed) + " failed)");
    Require(registered.programsDropped == 1u, "the program whose request version is refused was not dropped");
    Require(ahead.ahead == start.ahead + 1u && registered.replayed == 1u, "the programs prepared at registration did not create their compute and graphics pipelines ahead");
    Require(ahead.segmentsAhead == start.segmentsAhead + 1u, "registration created " + std::to_string(ahead.segmentsAhead - start.segmentsAhead) + " segment program pipelines ahead, expected the compute program's one");
    Dispatch();
    Draw();
    AgcDriver::CreationPool::WaitBackground();
    const auto used = AgcDriver::PipelineReplay::Totals();
    const auto compute = AgcDriver::VulkanDevice::ComputePipelineCreations();
    Require(used.programsAtUse == registered.programsAtUse, "the dispatch and the draw prepared " + std::to_string(used.programsAtUse - registered.programsAtUse) + " programs at use that registration had prepared");
    Require(compute.inFrames == ahead.inFrames && used.inFrames == registered.inFrames, "the dispatch and the draw created pipelines that registration had created");
    Require(used.segmentProgramsAtUse == registered.segmentProgramsAtUse && compute.segmentsInFrames == ahead.segmentsInFrames, "the dispatch prepared " + std::to_string(used.segmentProgramsAtUse - registered.segmentProgramsAtUse) + " segment programs and created " + std::to_string(compute.segmentsInFrames - ahead.segmentsInFrames) + " segment program pipelines that registration had prepared and created");
    AgcDriverShutdown_nid_postfix();
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--record") {
            Record();
            return 0;
        }
        Require(argc == 1, "usage: [--record]");
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
            if (!device->GraphicsPipelineLibraries()) return VulkanTestSkipped;
        }
        const auto directory = std::filesystem::temp_directory_path() / ("aps5-manifest-bump-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        SetVariable("ANYPS5_NO_SHADER_CACHE", "");
        SetVariable("ANYPS5_SHADER_CACHE_DIR", directory.string());
        Require(RunRecording(argv[0]) == 0, "the recording run failed");
        AgeCache(directory);
        Replay();
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        std::puts("manifest format bump tests passed");
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
