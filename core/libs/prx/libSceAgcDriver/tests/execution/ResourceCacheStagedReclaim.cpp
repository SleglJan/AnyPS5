#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <vector>


namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint32_t Dispatches = 12;
constexpr std::size_t SrtBytes = 64;
constexpr std::size_t OutputBytes = 256;
constexpr std::size_t SrtOffset = 4096;
constexpr std::size_t OutputOffset = SrtOffset + static_cast<std::size_t>(Dispatches) * SrtBytes;
constexpr std::size_t BlockBytes = (OutputOffset + static_cast<std::size_t>(Dispatches) * OutputBytes + 65535) & ~static_cast<std::size_t>(65535);

constexpr std::array<std::uint32_t, 7> StoreCode{0xf4080100u, 0xfa000000u, 0xbf8c0000u, 0x7e020281u, 0xe0700000u, 0x80010100u, 0xbf810000u};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "staged reclaim: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t At(std::size_t offset) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block + offset)); }
    std::uint8_t* Bytes(std::size_t offset) const { return block + offset; }

private:
    std::uint8_t* block = nullptr;
};

std::vector<std::uint32_t> Pm4(std::uint32_t opcode, std::initializer_list<std::uint32_t> body) {
    std::vector<std::uint32_t> words{0xc0000000u | (static_cast<std::uint32_t>(body.size() - 1) << 16u) | (opcode << 8u)};
    words.insert(words.end(), body);
    return words;
}

void Prepare(const GuestBlock& block) {
    std::memcpy(block.Bytes(0), StoreCode.data(), sizeof(StoreCode));
    for (std::uint32_t dispatch = 0; dispatch < Dispatches; ++dispatch) {
        const auto output = block.At(OutputOffset + static_cast<std::size_t>(dispatch) * OutputBytes);
        const std::array<std::uint32_t, 4> descriptor{static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), static_cast<std::uint32_t>(OutputBytes), 0x31016facu};
        std::memcpy(block.Bytes(SrtOffset + static_cast<std::size_t>(dispatch) * SrtBytes), descriptor.data(), sizeof(descriptor));
    }
    std::memset(block.Bytes(OutputOffset), 0, static_cast<std::size_t>(Dispatches) * OutputBytes);
}

void RunPass(const GuestBlock& block, std::uint32_t count = Dispatches) {
    const auto kernel = block.At(0);
    std::vector<std::uint32_t> words;
    for (std::uint32_t dispatch = 0; dispatch < count; ++dispatch) {
        const auto srt = block.At(SrtOffset + static_cast<std::size_t>(dispatch) * SrtBytes);
        for (const auto& packet : {Pm4(0x76, {0x20c, static_cast<std::uint32_t>(kernel >> 8u)}), Pm4(0x76, {0x20d, static_cast<std::uint32_t>(kernel >> 40u) & 0xffu}), Pm4(0x76, {0x207, 64}), Pm4(0x76, {0x208, 1}), Pm4(0x76, {0x209, 1}), Pm4(0x76, {0x213, 2u << 1u}), Pm4(0x76, {0x240, static_cast<std::uint32_t>(srt)}), Pm4(0x76, {0x241, static_cast<std::uint32_t>(srt >> 32u)}), Pm4(0x15, {1, 1, 1, 0x41})}) {
            words.insert(words.end(), packet.begin(), packet.end());
        }
    }
    Packet dcb{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&dcb) == 0, "staged reclaim: the dispatch submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

void StageEveryWrittenBuffer() {
#ifdef _WIN32
    _putenv_s("APS5_NO_DISPATCH_RECIPE", "1");
    _putenv_s("APS5_WRITTEN_SHADOW_MIN_KIB", "0");
#else
    setenv("APS5_NO_DISPATCH_RECIPE", "1", 1);
    setenv("APS5_WRITTEN_SHADOW_MIN_KIB", "0", 1);
#endif
}

void ShortfallDropsTheOldestStagedEntries(const GuestBlock& block) {
    auto& cache = AgcDriver::Graphics::SharedResourceCache();
    RunPass(block);
    for (std::uint32_t dispatch = 0; dispatch < Dispatches; ++dispatch) {
        std::uint32_t stored = 0;
        AgcDriver::GuestMemory::Read(block.At(OutputOffset + static_cast<std::size_t>(dispatch) * OutputBytes), std::as_writable_bytes(std::span(&stored, 1)), 4);
        Require(stored == 1u, "staged reclaim: dispatch " + std::to_string(dispatch) + " stored " + std::to_string(stored) + ", expected 1");
    }
    Require(cache.Size() == Dispatches, "staged reclaim: " + std::to_string(Dispatches) + " dispatches with distinct written buffers left " + std::to_string(cache.Size()) + " cache entries");
    const auto staged = cache.StagedBytes();
    Require(staged == static_cast<std::uint64_t>(Dispatches) * OutputBytes, "staged reclaim: the entries hold " + std::to_string(staged) + " staged bytes, expected " + std::to_string(Dispatches * OutputBytes) + " (every written buffer staged device-local)");

    std::vector<std::shared_ptr<AgcDriver::Graphics::ShaderResources>> evicted;
    std::uint64_t freed = 0;
    Require(cache.TryReclaimStaged(2 * OutputBytes, evicted, freed), "staged reclaim: the cache mutex was busy");
    Require(evicted.size() == 2 && freed == 2 * OutputBytes, "staged reclaim: a shortfall of two buffers dropped " + std::to_string(evicted.size()) + " entries freeing " + std::to_string(freed) + " bytes");
    Require(cache.Size() == Dispatches - 2 && cache.StagedBytes() == staged - 2 * OutputBytes, "staged reclaim: the cache did not lose exactly the two dropped entries");

    freed = 0;
    Require(cache.TryReclaimStaged(2 * OutputBytes + OutputBytes / 2, evicted, freed), "staged reclaim: the cache mutex was busy");
    Require(evicted.size() == 5 && freed == 3 * OutputBytes, "staged reclaim: a shortfall of two and a half buffers dropped " + std::to_string(evicted.size() - 2) + " entries freeing " + std::to_string(freed) + " bytes, expected 3 and " + std::to_string(3 * OutputBytes) + " (stop at the shortfall)");
    Require(cache.Size() == Dispatches - 5, "staged reclaim: the cache did not lose exactly the three dropped entries");
    evicted.clear();

    RunPass(block, 5);
    Require(cache.Size() == Dispatches, "staged reclaim: re-dispatching the five oldest builds left " + std::to_string(cache.Size()) + " entries, expected " + std::to_string(Dispatches) + " (the drops were not the oldest entries)");

    freed = 0;
    Require(cache.TryReclaimStaged(std::uint64_t{1} << 40u, evicted, freed), "staged reclaim: the cache mutex was busy");
    Require(evicted.size() == Dispatches && freed == static_cast<std::uint64_t>(Dispatches) * OutputBytes && cache.Size() == 0 && cache.StagedBytes() == 0, "staged reclaim: a shortfall larger than every shadow left " + std::to_string(cache.Size()) + " entries");
    evicted.clear();

    freed = 0;
    Require(cache.TryReclaimStaged(OutputBytes, evicted, freed) && evicted.empty() && freed == 0, "staged reclaim: an empty cache dropped something");
}

}

int main() {
    StageEveryWrittenBuffer();
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        auto& cache = AgcDriver::Graphics::SharedResourceCache();
        cache.Clear();
        GuestBlock block;
        Prepare(block);
        ShortfallDropsTheOldestStagedEntries(block);
        AgcDriverShutdown_nid_postfix();
        std::puts("resource cache staged reclaim tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
