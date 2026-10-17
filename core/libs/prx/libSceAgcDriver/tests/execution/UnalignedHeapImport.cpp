#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CommittedImports.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestHeap.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

namespace Graphics = AgcDriver::Graphics;
using Graphics::Require;

constexpr std::array<std::size_t, 5> BlockBytes{0x100, 0x10100, 0x100, 0x10100, 0x100};

std::uint64_t AddressOf(const void* pointer) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::string Hex(std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

void ExpectImport(const Graphics::Context& context, const void* block, std::size_t bytes, const std::string& what) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const auto begin = AddressOf(block);
    const auto* import = Graphics::HostImportFor(context, begin, bytes);
    Require(import != nullptr, what + " at " + Hex(begin) + "+" + Hex(bytes) + " has no host import");
    Require(import->base <= begin && begin + bytes <= import->base + import->bytes, what + " at " + Hex(begin) + " is not inside its import " + Hex(import->base) + "+" + Hex(import->bytes));
    Require(import->base % context.hostImportAlignment == 0 && import->bytes % context.hostImportAlignment == 0, what + "'s import " + Hex(import->base) + "+" + Hex(import->bytes) + " is not at the host import alignment");
}

void Run(const Graphics::Context& context) {
    std::array<void*, BlockBytes.size()> blocks{};
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        blocks[index] = GuestHeap::GuestHeapAlignForGpu_nid_postfix(256, BlockBytes[index]);
        Require(blocks[index] != nullptr, "unaligned heap import: the heap refused a block");
        std::memset(blocks[index], static_cast<int>(index + 1), BlockBytes[index]);
    }
    for (const std::size_t index : {0u, 2u, 4u}) GuestHeap::GuestHeapFree_nid_postfix(blocks[index]);
    const auto alignment = context.hostImportAlignment;
    for (const std::size_t index : {1u, 3u}) {
        const auto begin = AddressOf(blocks[index]);
        std::printf("lone GPU heap block %s+%s\n", Hex(begin).c_str(), Hex(BlockBytes[index]).c_str());
        Require(begin % alignment != 0 || (begin + BlockBytes[index]) % alignment != 0, "unaligned heap import: the heap placed a lone block at the host import alignment, so the test does not reach its case");
    }
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        Graphics::GuestBufferMemory memory(context);
        memory.AcquireRegistered();
        memory.Upload();
        static_cast<void>(memory.PageTable(context));
        memory.Release();
    }
    ExpectImport(context, blocks[1], BlockBytes[1], "a lone writable heap block");
    ExpectImport(context, blocks[3], BlockBytes[3], "a second lone writable heap block");
    ExpectImport(context, blocks[1], BlockBytes[1], "the first block looked up again");
    for (const std::size_t index : {1u, 3u}) GuestHeap::GuestHeapFree_nid_postfix(blocks[index]);
}

void BuildAddressSpace(const Graphics::Context& context) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Graphics::GuestBufferMemory memory(context);
    memory.AcquireRegistered();
    memory.Upload();
    static_cast<void>(memory.PageTable(context));
    memory.Release();
}

#ifdef _WIN32
void Uncovered(const Graphics::Context& context) {
    constexpr std::size_t RegionBytes = 0x40000;
    struct Piece {
        std::size_t offset;
        std::size_t bytes;
    };
    constexpr std::array<Piece, 3> Pieces{{{0x100, 0x20000}, {0x30100, 0x700}, {0x30900, 0x1000}}};
    auto* base = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(RegionBytes, 65536));
    Require(base != nullptr, "unaligned heap import: cannot reserve guest memory");
    GuestArena::GuestArenaCommit_nid_postfix(base, RegionBytes, PAGE_READONLY, RegionBytes);
    DWORD previous = 0;
    Require(VirtualProtect(base, RegionBytes, PAGE_READWRITE, &previous) != 0, "unaligned heap import: cannot make the region writable");
    Require(!Graphics::CommittedImportsOverlap(context.device, AddressOf(base), AddressOf(base) + RegionBytes), "unaligned heap import: committed imports cover memory committed read-only, so the test does not reach its case");
    for (const auto& piece : Pieces) {
        std::memset(base + piece.offset, 0x5a, piece.bytes);
        GuestAllocations::Mutation().Add(base + piece.offset, piece.bytes, true, true, true);
    }
    BuildAddressSpace(context);
    ExpectImport(context, base + Pieces[0].offset, Pieces[0].bytes, "a lone writable block no committed import covers");
    ExpectImport(context, base + Pieces[1].offset, Pieces[1].bytes, "a lone writable block sharing a page with another");
    ExpectImport(context, base + Pieces[2].offset, Pieces[2].bytes, "the block it shares the page with");
    BuildAddressSpace(context);
    for (const auto& piece : Pieces) GuestAllocations::Mutation().Remove(base + piece.offset);
    GuestArena::GuestArenaReset_nid_postfix(base, RegionBytes);
    GuestArena::GuestArenaRelease_nid_postfix(base, RegionBytes);
}
#endif

void SharedPage(const Graphics::Context& context) {
    constexpr std::size_t Small = 0xf0;
    std::array<void*, 6> blocks{};
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        blocks[index] = GuestHeap::GuestHeapAlignForGpu_nid_postfix(256, Small);
        Require(blocks[index] != nullptr, "unaligned heap import: the heap refused a small block");
        std::memset(blocks[index], static_cast<int>(index + 1), Small);
    }
    for (const std::size_t index : {0u, 2u, 4u}) GuestHeap::GuestHeapFree_nid_postfix(blocks[index]);
    const auto alignment = context.hostImportAlignment;
    const auto page = [&](const void* block) { return AddressOf(block) / alignment; };
    std::printf("lone small GPU heap blocks %s, %s, %s\n", Hex(AddressOf(blocks[1])).c_str(), Hex(AddressOf(blocks[3])).c_str(), Hex(AddressOf(blocks[5])).c_str());
    Require(page(blocks[1]) == page(blocks[3]) || page(blocks[3]) == page(blocks[5]), "unaligned heap import: no two lone small blocks share an import page, so the test does not reach its case");
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        Graphics::GuestBufferMemory memory(context);
        memory.AcquireRegistered();
        memory.Upload();
        static_cast<void>(memory.PageTable(context));
        memory.Release();
    }
    for (const std::size_t index : {1u, 3u, 5u}) ExpectImport(context, blocks[index], Small, "a lone small heap block sharing a page");
    for (const std::size_t index : {1u, 3u, 5u}) GuestHeap::GuestHeapFree_nid_postfix(blocks[index]);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        if (context.hostImportAlignment == 0 || !context.bufferDeviceAddress) {
            std::puts("unaligned heap import: the device imports no host memory");
            return VulkanTestSkipped;
        }
        Run(context);
        SharedPage(context);
#ifdef _WIN32
        Uncovered(context);
#endif
        std::puts("unaligned heap import tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
