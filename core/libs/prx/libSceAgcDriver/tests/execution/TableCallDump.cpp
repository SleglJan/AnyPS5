#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "ControlFlow/CalleeLinker.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Optimization/RequestMemoryView.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("table call dump: " + what);
}

constexpr std::array<std::uint32_t, 11> Caller{0x8f148414u, 0xf4080406u, 0xfa000270u, 0xbf8cc07fu, 0xf4280208u, 0x28000000u, 0xbf8cc07fu, 0xbe8e0308u, 0xbe8f0309u, 0xbe8e210eu, 0xbf810000u};
constexpr std::array<std::uint32_t, 3> First{0xbf068000u, 0x4a020285u, 0xbe80200eu};
constexpr std::array<std::uint32_t, 2> Second{0x4a0202c0u, 0xbe80200eu};

TestGuest::Array<std::uint32_t, 256> CallerCode;
TestGuest::Array<std::uint32_t, 1024> CalleeCode;
TestGuest::Array<std::uint32_t, 0x270u / 4u + 4u> Descriptors;
TestGuest::Array<std::uint32_t, 16> Records;

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::uint32_t Low(std::uint64_t value) { return static_cast<std::uint32_t>(value); }
std::uint32_t High(std::uint64_t value) { return static_cast<std::uint32_t>(value >> 32u); }

bool ReadRegions(void* context, std::uint64_t address, std::span<std::uint32_t> words) {
    const auto& runtime = *static_cast<const ShaderRecompiler::SrtRuntime*>(context);
    for (std::size_t word = 0; word < words.size(); ++word) {
        if (!runtime.readMemory(runtime.userContext, address + word * sizeof(std::uint32_t), &words[word])) return false;
    }
    return true;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        std::copy(Caller.begin(), Caller.end(), CallerCode.begin());
        std::copy(First.begin(), First.end(), CalleeCode.begin());
        std::copy(Second.begin(), Second.end(), CalleeCode.begin() + 0x200);
        const auto code = AddressOf(CalleeCode.data());
        const auto table = AddressOf(Records.data());
        const std::array<std::uint32_t, 4> tableDescriptor{Low(table), High(table) | (16u << 16u), 4u, 0u};
        std::copy(tableDescriptor.begin(), tableDescriptor.end(), Descriptors.begin() + 0x270 / 4);
        Records = {Low(code), High(code), 0x00c0ffeeu, 0x12u, Low(code), High(code), 0x0badf00du, 0x13u, Low(code + 0x800u), High(code + 0x800u), 0x00c0ffeeu, 0x12u, 0u, 0u, 0u, 0u};
        std::array<std::uint32_t, 16> userData{};
        userData[12] = Low(AddressOf(Descriptors.data()));
        userData[13] = High(AddressOf(Descriptors.data()));
        const std::span<const std::uint32_t> callerWords(CallerCode.data(), Caller.size());
        const std::array<ShaderRecompiler::MemoryRegion, 3> captured{{
            {AddressOf(CallerCode.data()), std::as_bytes(callerWords)},
            {AddressOf(Descriptors.data()), std::as_bytes(std::span(Descriptors.data(), Descriptors.size()))},
            {table, std::as_bytes(std::span(Records.data(), Records.size()))},
        }};
        ShaderRecompiler::RecompileRequest request{};
        request.shader.stage = ShaderRecompiler::ShaderStage::Compute;
        request.shader.codeAddress = AddressOf(CallerCode.data());
        request.shader.code = callerWords;
        request.context.waveSize = 32u;
        request.context.userData = userData;
        request.context.compute = ShaderRecompiler::ShaderComputeStageInfo{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u, {}};
        request.context.memory = captured;
        request.target = device->ComputeTarget(32u);
        request.layout = {0u, 0u, 0u, 128u};

        const auto directory = std::filesystem::temp_directory_path() / ("aps5-table-dump-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        const auto previous = std::filesystem::current_path();
        std::filesystem::current_path(directory);
        const auto name = AgcDriver::DriverDetail::Driver::dumpRequest(request.shader.codeAddress, request);
        std::filesystem::current_path(previous);
        std::ifstream file(directory / name, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        std::error_code removed;
        std::filesystem::remove_all(directory, removed);
        Require(!text.empty(), "the dump wrote no request");

        auto dumped = ShaderRecompiler::RequestSerializer{}.Deserialize(text);
        const auto covered = [&](std::uint64_t address, std::size_t bytes) {
            return std::ranges::any_of(dumped.request.context.memory, [&](const ShaderRecompiler::MemoryRegion& region) { return region.guestAddress <= address && address + bytes <= region.guestAddress + region.bytes.size(); });
        };
        Require(covered(code, First.size() * 4u) && covered(code + 0x800u, Second.size() * 4u), "the dumped request does not carry the code of its table-call targets");
        Require(ShaderRecompiler::RequestSerializer{}.Serialize(dumped.request) == text, "the dumped request does not serialize back to itself");

        ShaderRecompiler::RequestMemoryView view(dumped.request.context.memory);
        auto runtime = view.MakeRuntime(dumped.request.context.userData, dumped.request.shader.codeAddress);
        const ShaderRecompiler::CalleeMemory memory{&runtime, &ReadRegions, &ReadRegions, nullptr};
        const auto split = ShaderRecompiler::SplitTableCalls(dumped.request, memory);
        Require(split.has_value(), "the dumped request does not split at its table call");
        const auto targets = ShaderRecompiler::ReadTableCallTargets(dumped.request, *split, memory);
        Require(targets.size() == 3u && targets[0].target == code && targets[2].target == code + 0x800u, "the dumped request does not read its three table-call targets");
        const auto compile = [&](const ShaderRecompiler::SplitCode& part, ShaderRecompiler::ContinuationAbi::Role role, std::uint32_t entrySite) {
            std::vector<std::uint32_t> words(dumped.request.context.userData.begin(), dumped.request.context.userData.end());
            words.resize(words.size() + ShaderRecompiler::ContinuationAbi::UserDataWords, 0u);
            auto source = dumped.request;
            source.shader.code = part.code;
            source.context.userData = words;
            if (role == ShaderRecompiler::ContinuationAbi::Role::Continuation) {
                auto compute = *dumped.request.context.compute;
                compute.numThreads = {dumped.request.context.waveSize, 1u, 1u};
                source.context.compute = compute;
            }
            source.context.continuation = ShaderRecompiler::TableCallShape(*split, part, role, entrySite);
            source.useCache = false;
            const auto result = ShaderRecompiler::Recompile(source);
            Require(result.spirv.size() > 5u, "a split part compiled to no module");
        };
        compile(split->head, ShaderRecompiler::ContinuationAbi::Role::Head, 0u);
        for (const auto& target : targets) compile(ShaderRecompiler::TableCallContinuation(*split, target), ShaderRecompiler::ContinuationAbi::Role::Continuation, target.site);
        std::puts("table call dump tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
