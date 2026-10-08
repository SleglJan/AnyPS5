#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ShaderRecompiler::ShaderStage;

std::vector<std::uint32_t> ReadWords(const char* path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<std::uint32_t> words;
    std::string token;
    while (file >> token) words.push_back(static_cast<std::uint32_t>(std::stoul(token, nullptr, 0)));
    return words;
}

std::uint32_t Setting(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    return value ? static_cast<std::uint32_t>(std::stoul(value, nullptr, 0)) : fallback;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), count, 0x11016facu};
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const char* codePath = std::getenv("REPLAY_CODE");
        const char* rowsPath = std::getenv("REPLAY_ROWS");
        if (!codePath || !rowsPath) throw std::runtime_error("REPLAY_CODE and REPLAY_ROWS are required");
        const std::uint32_t inputs = Setting("REPLAY_INPUTS", 8);
        const std::uint32_t results = Setting("REPLAY_RESULTS", 8);
        const std::uint32_t threads = Setting("REPLAY_THREADS", 32);
        const std::uint32_t lds = Setting("REPLAY_LDS", 0);
        const std::uint32_t userDataWords = Setting("REPLAY_USER_DATA", 8);
        const std::uint32_t wave = Setting("REPLAY_WAVE", 32);
        std::optional<ShaderRecompiler::ShaderFloatMode> floatMode;
        if (const char* mode = std::getenv("REPLAY_MODE"); mode && *mode) {
            int fm = 0, dx10 = 0, ieee = 0, ovfl = 0;
            if (std::sscanf(mode, "%i,%i,%i,%i", &fm, &dx10, &ieee, &ovfl) != 4) throw std::runtime_error("REPLAY_MODE is float_mode,dx10_clamp,ieee_mode,fp16_ovfl");
            floatMode = ShaderRecompiler::ShaderFloatMode{static_cast<std::uint32_t>(fm), dx10 != 0, ieee != 0, ovfl != 0};
        }
        const auto codeWords = ReadWords(codePath);
        const auto rowWords = ReadWords(rowsPath);
        if (rowWords.size() % inputs != 0) throw std::runtime_error("the rows file is not a multiple of REPLAY_INPUTS dwords");
        const std::uint32_t rows = static_cast<std::uint32_t>(rowWords.size() / inputs);
        if (rows > threads) throw std::runtime_error("more rows than REPLAY_THREADS");

        std::vector<std::uint32_t> code(codeWords.size() + 64, 0xbf810000u);
        std::copy(codeWords.begin(), codeWords.end(), code.begin());
        alignas(256) static std::uint32_t input[1024 * 64];
        alignas(256) static std::uint32_t output[1024 * 64];
        if (threads * inputs > 1024 * 64 || threads * results > 1024 * 64) throw std::runtime_error("buffers too large");
        std::fill(input, input + threads * inputs, 0u);
        std::copy(rowWords.begin(), rowWords.end(), input);
        std::fill(output, output + threads * results, 0xdeadbeefu);

        std::vector<std::uint32_t> userData(userDataWords, 0u);
        const auto in = BufferDescriptor(input, threads * inputs);
        const auto out = BufferDescriptor(output, threads * results);
        std::copy(in.begin(), in.end(), userData.begin());
        std::copy(out.begin(), out.end(), userData.begin() + 4);

        const std::span<const std::uint32_t> span(code);
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(span.data()), std::as_bytes(span)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{threads, 1, 1}, lds, {false, false, false}, false, 1};
        ShaderRecompiler::RecompileRequest request{
            {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(span.data()), span, 0, {}},
            {wave, 0, userData, compute, std::nullopt, std::nullopt, memory},
            device->Target(),
            {0, 0, 0, 128}
        };
        request.useCache = false;
        request.context.floatMode = floatMode;
        const auto result = ShaderRecompiler::Recompile(request);
        device->Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(span.data()));
        device->WaitIdle();

        for (std::uint32_t tid = 0; tid < rows; ++tid) {
            for (std::uint32_t j = 0; j < results; ++j) std::printf("%s%08x", j ? " " : "", output[tid * results + j]);
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
