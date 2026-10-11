// Local benchmark (not a ctest): an image-table shader over a large table in write-watched arena
// memory, captured and materialized the way the draw path does it, so the table snapshot's skip
// path and the result memo can be timed and profiled without a title.
// usage: agc_table_bench [captures=2000] [entries=18960] [vary=0|1 (user data varies per capture)]
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "BdaAbi.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
}

namespace {

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

std::array<std::uint32_t, 8> TableT(std::uint64_t address, std::uint32_t format, std::uint32_t type, std::uint32_t height) {
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (3u << 30u), height << 14u, 0xfacu | (type << 28u), 0u, 0u, 0u, 0u};
}

std::array<std::uint32_t, 4> TableV(std::uint64_t address, std::uint32_t stride, std::uint32_t records) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), records, stride == 0u ? 0x31016facu : 0x01016facu};
}

const std::vector<std::uint32_t> TableWholeCode{0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080300u, 0xfa000020u, 0xf4080700u, 0xfa000030u, 0x7e200500u, 0x8f108510u, 0xf42c0502u, 0x20000000u, 0xf09c0f08u, 0x00450000u, 0xe0700000u, 0x80070000u, 0xbf810000u};

ShaderRecompiler::RecompileRequest TableRequest(const std::vector<std::uint32_t>& code, std::span<const std::uint32_t> userData) {
    using namespace ShaderRecompiler;
    static constexpr std::array<std::uint32_t, 4> capabilities{29u, spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess};
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x7770000u, code, 0, {}};
    request.context.waveSize = 64;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32u;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.fragmentShaderBarycentricEnabled = false;
    request.layout.pushConstantSizeBytes = 128;
    return request;
}

double Since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

}

int main(int argc, char** argv) {
    try {
        const auto captures = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 2000ul;
        const auto entries = argc > 2 ? static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 0)) : 18960u;
        // SRT copies the per-capture user data cycles through: equal contents ("same"), or each
        // with its own output V# ("buffers": a per-draw constant buffer).
        const auto copies = argc > 3 ? std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 0))) : 1u;
        const bool buffers = argc > 4 && std::string_view(argv[4]) == "buffers";
        require(copies <= 4096u, "at most 4096 SRT copies");
        require(AgcDriver::GuestMemory::WriteWatched(), "write tracking unavailable");
        const std::size_t bytes = 4u << 20u;
        std::int64_t physical = -1;
        void* data = nullptr;
        require(sceKernelAllocateDirectMemory(0, 0x10000000000ll, bytes, 65536, 3, &physical) == 0, "allocate direct memory failed");
        require(sceKernelMapDirectMemory(&data, bytes, 3, 0, physical, 65536) == 0, "map direct memory failed");
        const auto base = reinterpret_cast<std::uint64_t>(data);
        auto* words = static_cast<std::uint32_t*>(data);
        const auto output = base + 0xf0000u;
        const auto heap = base + 0x100000u;
        const auto textures = base + 0x300000u;
        for (std::uint32_t entry = 0; entry < entries; ++entry) {
            const auto t = TableT(textures + (entry % 97u) * 0x100u, entry % 5u == 0u ? 20u : 56u, 9u, 3u + entry % 7u);
            std::memcpy(words + (heap - base) / 4u + entry * 8u, t.data(), sizeof(t));
        }
        std::vector<std::array<std::uint32_t, 2>> users;
        for (std::uint32_t copy = 0; copy < copies; ++copy) {
            const auto srt = base + copy * 0x40u;
            auto* at = words + (srt - base) / 4u;
            const auto heapV = TableV(heap, 32u, entries);
            const auto outputV = TableV(output + (buffers ? copy * 0x10u : 0u), 0u, 16u);
            std::memcpy(at, heapV.data(), 16u);
            std::memset(at + 4, 0, 16u);
            std::memcpy(at + 8, outputV.data(), 16u);
            std::memcpy(at + 12, outputV.data(), 16u);
            users.push_back({static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u)});
        }
        double captureUs = 0, regionsUs = 0, materializeUs = 0;
        std::uint64_t memoHits = 0;
        for (unsigned long i = 0; i < captures; ++i) {
            AgcDriver::GuestMemory::BumpCollectEpoch();
            auto request = TableRequest(TableWholeCode, users[i % copies]);
            AgcDriver::ShaderMemory memory({});
            auto started = std::chrono::steady_clock::now();
            const auto capture = memory.Capture(request);
            captureUs += Since(started);
            started = std::chrono::steady_clock::now();
            const auto recent = memory.TakeRecentRegions();
            const auto regions = memory.Regions();
            regionsUs += Since(started);
            request.context.memory = regions;
            started = std::chrono::steady_clock::now();
            bool hit = false;
            const auto result = ShaderRecompiler::Recompile(request, *capture, &hit);
            materializeUs += Since(started);
            memoHits += hit ? 1u : 0u;
            require(result != nullptr && !recent.empty(), "nothing compiled");
            if (i == 0) std::printf("first: %zu regions, %zu table keys\n", regions.size(), capture->snapshot.tables.tables.empty() ? 0u : capture->snapshot.tables.tables[0].codes.size());
        }
        const auto n = static_cast<double>(captures);
        std::printf("%lu captures of %u entries, %u SRT copies%s: capture %.1f us, regions %.1f us, materialize %.1f us (%llu memo hits)\n", captures, entries, copies, buffers ? " (own buffers)" : "", captureUs / n, regionsUs / n, materializeUs / n, static_cast<unsigned long long>(memoHits));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "table bench: %s\n", error.what());
        return 1;
    }
}
