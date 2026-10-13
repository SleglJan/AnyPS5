#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using namespace ShaderRecompiler;
using AgcDriver::Graphics::Require;

constexpr std::size_t BlockBytes = 1u << 20u;
constexpr std::uint32_t Width = 256;
constexpr std::uint32_t Height = 128;
constexpr std::size_t ImageBytes = Width * Height * 4u;
constexpr std::uint32_t Initial = 0x3f800000u;
constexpr std::uint32_t Stored = 0x11111111u;
constexpr std::uint32_t Overwritten = 0x22222222u;
constexpr std::size_t OverwrittenDword = 256u / 4u;
constexpr std::size_t FarDword = 100u * Width;

alignas(256) constexpr std::array<std::uint32_t, 11> StoreImageCode{0x7e000280u, 0x7e020280u, 0x7e0402ffu, 0x11111111u, 0xf0200108u, 0x00010200u, 0x7e0202ffu, 0x00000064u, 0xf0200108u, 0x00010200u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 7> StoreGlobalCode{0x7e000208u, 0x7e020209u, 0x7e0402ffu, 0x22222222u, 0xdc708100u, 0x007d0200u, 0xbf810000u};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "storage flush ordering: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;
    std::uint32_t* Words() const { return reinterpret_cast<std::uint32_t*>(block); }
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

private:
    std::uint8_t* block = nullptr;
};

PFN_vkDeviceWaitIdle RawDeviceWaitIdle() {
#ifdef _WIN32
    const auto module = GetModuleHandleA("vulkan-1.dll");
    return module != nullptr ? reinterpret_cast<PFN_vkDeviceWaitIdle>(GetProcAddress(module, "vkDeviceWaitIdle")) : nullptr;
#else
    void* handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
    return handle != nullptr ? reinterpret_cast<PFN_vkDeviceWaitIdle>(dlsym(handle, "vkDeviceWaitIdle")) : nullptr;
#endif
}

auto Compile(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const SpirvTarget& target) {
    const std::array<MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderComputeStageInfo compute{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64u, 0, userData, compute, std::nullopt, std::nullopt, regions},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return Recompile(request);
}

void Run(AgcDriver::VulkanDevice& device) {
    GuestBlock guest;
    auto* words = guest.Words();
    std::fill(words, words + ImageBytes / 4u, Initial);
    const auto base = guest.Address();
    std::vector<std::uint32_t> imageUser(12, 0u);
    imageUser[4] = static_cast<std::uint32_t>(base >> 8u);
    imageUser[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (20u << 20u) | (((Width - 1u) & 3u) << 30u);
    imageUser[6] = ((Width - 1u) >> 2u) | ((Height - 1u) << 14u);
    imageUser[7] = 0x90000facu;
    std::vector<std::uint32_t> globalUser(10, 0u);
    globalUser[8] = static_cast<std::uint32_t>(base);
    globalUser[9] = static_cast<std::uint32_t>(base >> 32u);
    const auto target = device.Target();
    const auto storeImage = Compile(StoreImageCode, imageUser, target);
    const auto storeGlobal = Compile(StoreGlobalCode, globalUser, target);
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    device.Dispatch(storeGlobal, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreGlobalCode.data()));
    const auto rawWait = RawDeviceWaitIdle();
    if (rawWait != nullptr) rawWait(device.Device());
    else std::puts("storage flush ordering: no raw device wait, the completions may land before the flush");
    const auto waitedBefore = AgcDriver::Graphics::StorageTexture::FlushesBehindWriters();
    AgcDriver::Graphics::StorageTexture::FlushPending(base, ImageBytes, nullptr, "test");
    device.WaitIdle();
    Require(AgcDriver::Graphics::StorageTexture::FlushesBehindWriters() == waitedBefore + 1u, "storage flush ordering: the flush did not wait for the in-flight address-based writer");
    Require(words[FarDword] == Stored, "storage flush ordering: the pending image's texel away from the newer write was not stored");
    if (!AgcDriver::GuestMemory::WriteWatched()) {
        std::puts("storage flush ordering: no write watching, the kept-block check is skipped");
        return;
    }
    Require(words[OverwrittenDword] == Overwritten, "storage flush ordering: the pending image's store put its older texels over a newer address-based store");
}

void RunAhead(AgcDriver::VulkanDevice& device) {
    GuestBlock guest;
    auto* words = guest.Words();
    std::fill(words, words + ImageBytes / 4u, Initial);
    const auto base = guest.Address();
    std::vector<std::uint32_t> imageUser(12, 0u);
    imageUser[4] = static_cast<std::uint32_t>(base >> 8u);
    imageUser[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (20u << 20u) | (((Width - 1u) & 3u) << 30u);
    imageUser[6] = ((Width - 1u) >> 2u) | ((Height - 1u) << 14u);
    imageUser[7] = 0x90000facu;
    std::vector<std::uint32_t> globalUser(10, 0u);
    globalUser[8] = static_cast<std::uint32_t>(base);
    globalUser[9] = static_cast<std::uint32_t>(base >> 32u);
    const auto target = device.Target();
    const auto storeImage = Compile(StoreImageCode, imageUser, target);
    const auto storeGlobal = Compile(StoreGlobalCode, globalUser, target);
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    device.WaitIdle();
    device.Dispatch(storeGlobal, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreGlobalCode.data()));
    const auto aheadBefore = AgcDriver::Graphics::StorageTexture::FlushesStoredAhead();
    AgcDriver::Graphics::StorageTexture::FlushPending(base, ImageBytes, nullptr, "test");
    Require(AgcDriver::Graphics::StorageTexture::FlushesStoredAhead() == aheadBefore + 1u, "storage flush ordering: the store behind a writer in the open batch was not submitted ahead of it");
    device.WaitIdle();
    Require(words[FarDword] == Stored, "storage flush ordering: the image's texel away from the open batch's write was not stored");
    Require(words[OverwrittenDword] == Overwritten, "storage flush ordering: the store submitted ahead of the open batch did not land before its writer");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        if (std::getenv("APS5_FLUSH_STORE_AHEAD") != nullptr) RunAhead(*device);
        else std::puts("storage flush ordering: APS5_FLUSH_STORE_AHEAD is not set, the store-ahead case is skipped");
        std::puts("storage flush ordering tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "storage flush ordering: %s\n", error.what());
        return 1;
    }
}
