#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "GuestTestMemory.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("DMA copy after a GPU write: " + what);
}

std::array<std::uint32_t, 7> DmaData(bool immediate, std::uint64_t source, std::uint32_t data, std::uint64_t destination, std::uint32_t bytes) {
    return {0xc0055000u, immediate ? (2u << 29u) : 0u, immediate ? data : static_cast<std::uint32_t>(source), immediate ? 0u : static_cast<std::uint32_t>(source >> 32u), static_cast<std::uint32_t>(destination), static_cast<std::uint32_t>(destination >> 32u), bytes};
}

struct alignas(8) ShaderHeader {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(256) constexpr std::array<std::uint32_t, 1> EndProgram{0xbf810000u};

void OpenDevice() {
    static ShaderHeader header;
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(ShaderHeader);
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(EndProgram));
    header.shader.code = EndProgram.data();
    header.shader.type = 1;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void Submit(TestGuest::Vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
}

}

int main() {
    try {
        OpenDevice();
        static TestGuest::Array<std::uint32_t, 32768> filled;
        static TestGuest::Array<std::uint32_t, 1024> copied;
        static TestGuest::Array<std::uint32_t, 1024> plain;
        static TestGuest::Array<std::uint32_t, 1024> plainCopied;
        filled.fill(0);
        copied.fill(0);
        plainCopied.fill(0);
        for (std::size_t i = 0; i < plain.size(); ++i) plain[i] = 0xc0de0000u + static_cast<std::uint32_t>(i);
        using namespace AgcDriver::DriverDetail;
        const auto fillsBefore = dmaFills.load();
        const auto gpuCopiesBefore = dmaCopiesOnGpu.load();
        const auto drainedBefore = storesDrained.load();
        TestGuest::Vector<std::uint32_t> words;
        const auto fill = DmaData(true, 0, 0x5eed1234u, reinterpret_cast<std::uintptr_t>(filled.data()), static_cast<std::uint32_t>(decltype(filled)::ByteSize));
        const auto copy = DmaData(false, reinterpret_cast<std::uintptr_t>(filled.data()) + 4096, 0, reinterpret_cast<std::uintptr_t>(copied.data()), 512);
        const auto plainCopy = DmaData(false, reinterpret_cast<std::uintptr_t>(plain.data()), 0, reinterpret_cast<std::uintptr_t>(plainCopied.data()), 1024);
        words.insert(words.end(), fill.begin(), fill.end());
        words.insert(words.end(), copy.begin(), copy.end());
        words.insert(words.end(), plainCopy.begin(), plainCopy.end());
        Submit(words);
        AgcDriverWaitIdle_nid_postfix();
        std::vector<std::uint32_t> source(128), result(copied.size()), plainResult(plainCopied.size());
        AgcDriver::GuestMemory::Read(reinterpret_cast<std::uintptr_t>(copied.data()), std::as_writable_bytes(std::span(result)));
        AgcDriver::GuestMemory::Read(reinterpret_cast<std::uintptr_t>(plainCopied.data()), std::as_writable_bytes(std::span(plainResult)));
        AgcDriver::GuestMemory::Read(reinterpret_cast<std::uintptr_t>(filled.data()) + 4096, std::as_writable_bytes(std::span(source)));
        check(dmaFills.load() == fillsBefore + 1, "the large immediate fill was not recorded on the GPU");
        for (std::size_t i = 0; i < result.size(); ++i) check(result[i] == (i < 128 ? 0x5eed1234u : 0u), "the copy did not see the GPU fill before it, or wrote past its bytes (word " + std::to_string(i) + " is " + std::to_string(result[i]) + ", the source word " + std::to_string(source[i % 128]) + ")");
        for (std::size_t i = 0; i < plainResult.size(); ++i) check(plainResult[i] == (i < 256 ? plain[i] : 0u), "the copy of CPU-written bytes is wrong");
        const auto gpuCopies = dmaCopiesOnGpu.load() - gpuCopiesBefore;
        check(gpuCopies == 1, "a small copy whose source a recorded GPU write produces must be recorded as a device copy, not read on the CPU after waiting for the GPU; the copy of CPU-written bytes stays a store (" + std::to_string(gpuCopies) + " device copies)");
        check(storesDrained.load() == drainedBefore, "a DMA_DATA packet drained");
        LibcRunShutdown_nid_postfix();
        std::puts("DMA copy after a GPU write tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
