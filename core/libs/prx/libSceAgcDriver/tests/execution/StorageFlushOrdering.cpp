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

constexpr std::size_t BlockBytes = 4u << 20u;
constexpr std::uint32_t Width = 256;
constexpr std::uint32_t Height = 128;
constexpr std::size_t ImageBytes = Width * Height * 4u;
constexpr std::uint32_t Initial = 0x3f800000u;
constexpr std::uint32_t Stored = 0x11111111u;
constexpr std::uint32_t Overwritten = 0x22222222u;
constexpr std::size_t OverwrittenDword = 256u / 4u;
constexpr std::size_t FarDword = 100u * Width;
constexpr std::uint32_t LargeWidth = 1024;
constexpr std::uint32_t LargeHeight = 512;
constexpr std::size_t LargeImageBytes = LargeWidth * LargeHeight * 4u;
constexpr std::size_t LargeFarDword = 100u * LargeWidth;

alignas(256) constexpr std::array<std::uint32_t, 11> StoreImageCode{0x7e000280u, 0x7e020280u, 0x7e0402ffu, 0x11111111u, 0xf0200108u, 0x00010200u, 0x7e0202ffu, 0x00000064u, 0xf0200108u, 0x00010200u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 11> StoreImageAgainCode{0x7e000280u, 0x7e020280u, 0x7e0402ffu, 0x33333333u, 0xf0200108u, 0x00010200u, 0x7e0202ffu, 0x00000064u, 0xf0200108u, 0x00010200u, 0xbf810000u};
constexpr std::uint32_t StoredAgain = 0x33333333u;
alignas(256) constexpr std::array<std::uint32_t, 5> StoreBufferCode{0x7e0402ffu, 0x22222222u, 0xe0700100u, 0x80010200u, 0xbf810000u};
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

}

void RunUploadOrder(AgcDriver::VulkanDevice& device) {
    GuestBlock guest;
    auto* words = guest.Words();
    std::fill(words, words + ImageBytes / 4u, Initial);
    const auto base = guest.Address();
    std::vector<std::uint32_t> imageUser(12, 0u);
    imageUser[4] = static_cast<std::uint32_t>(base >> 8u);
    imageUser[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (20u << 20u) | (((Width - 1u) & 3u) << 30u);
    imageUser[6] = ((Width - 1u) >> 2u) | ((Height - 1u) << 14u);
    imageUser[7] = 0x90000facu;
    std::vector<std::uint32_t> bufferUser(8, 0u);
    bufferUser[4] = static_cast<std::uint32_t>(base);
    bufferUser[5] = static_cast<std::uint32_t>(base >> 32u);
    bufferUser[6] = static_cast<std::uint32_t>(ImageBytes);
    bufferUser[7] = 0x31016facu;
    const auto target = device.Target();
    const auto storeBuffer = Compile(StoreBufferCode, bufferUser, target);
    const auto storeImage = Compile(StoreImageCode, imageUser, target);
    const auto waitedBefore = AgcDriver::Graphics::StorageTexture::UploadWaits();
    device.Dispatch(storeBuffer, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreBufferCode.data()));
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    Require(AgcDriver::Graphics::StorageTexture::UploadWaits() == waitedBefore + 1u, "storage upload order: the upload did not wait for the recorded buffer writer of its memory");
    AgcDriver::Graphics::StorageTexture::FlushPending(base, ImageBytes, nullptr, "test");
    device.WaitIdle();
    Require(words[OverwrittenDword] == Overwritten, "storage upload order: the image's write-back put its upload-era texel over the buffer write recorded before its upload");
    Require(words[FarDword] == Stored, "storage upload order: the image's far texel was not stored");
}

void RunRefreshBehindWriter(AgcDriver::VulkanDevice& device) {
    GuestBlock guest;
    auto* words = guest.Words();
    std::fill(words, words + ImageBytes / 4u, Initial);
    const auto base = guest.Address();
    std::vector<std::uint32_t> imageUser(12, 0u);
    imageUser[4] = static_cast<std::uint32_t>(base >> 8u);
    imageUser[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (20u << 20u) | (((Width - 1u) & 3u) << 30u);
    imageUser[6] = ((Width - 1u) >> 2u) | ((Height - 1u) << 14u);
    imageUser[7] = 0x90000facu;
    std::vector<std::uint32_t> bufferUser(8, 0u);
    bufferUser[4] = static_cast<std::uint32_t>(base);
    bufferUser[5] = static_cast<std::uint32_t>(base >> 32u);
    bufferUser[6] = static_cast<std::uint32_t>(ImageBytes);
    bufferUser[7] = 0x31016facu;
    const auto target = device.Target();
    const auto storeImage = Compile(StoreImageCode, imageUser, target);
    const auto storeBuffer = Compile(StoreBufferCode, bufferUser, target);
    const auto storeImageAgain = Compile(StoreImageAgainCode, imageUser, target);
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    AgcDriver::Graphics::StorageTexture::FlushPending(base, ImageBytes, nullptr, "test");
    device.WaitIdle();
    Require(words[FarDword] == Stored, "storage refresh order: the first image write was not stored");
    device.Dispatch(storeBuffer, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreBufferCode.data()));
    device.Dispatch(storeImageAgain, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageAgainCode.data()));
    AgcDriver::Graphics::StorageTexture::FlushPending(base, ImageBytes, nullptr, "test");
    device.WaitIdle();
    Require(words[FarDword] == StoredAgain, "storage refresh order: the second image write was not stored");
    Require(words[OverwrittenDword] == Overwritten, "storage refresh order: the image's refresh advanced past a buffer write recorded before it and its write-back put an upload-era texel over the write");
}

void RunRefresh(AgcDriver::VulkanDevice& device) {
    GuestBlock guest;
    auto* words = guest.Words();
    std::fill(words, words + ImageBytes / 4u, Initial);
    const auto base = guest.Address();
    std::vector<std::uint32_t> imageUser(12, 0u);
    imageUser[4] = static_cast<std::uint32_t>(base >> 8u);
    imageUser[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (20u << 20u) | (((Width - 1u) & 3u) << 30u);
    imageUser[6] = ((Width - 1u) >> 2u) | ((Height - 1u) << 14u);
    imageUser[7] = 0x90000facu;
    std::vector<std::uint32_t> bufferUser(8, 0u);
    bufferUser[4] = static_cast<std::uint32_t>(base);
    bufferUser[5] = static_cast<std::uint32_t>(base >> 32u);
    bufferUser[6] = static_cast<std::uint32_t>(ImageBytes);
    bufferUser[7] = 0x31016facu;
    const auto target = device.Target();
    const auto storeImage = Compile(StoreImageCode, imageUser, target);
    const auto storeBuffer = Compile(StoreBufferCode, bufferUser, target);
    const auto storeImageAgain = Compile(StoreImageAgainCode, imageUser, target);
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    device.Dispatch(storeBuffer, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreBufferCode.data()));
    device.Dispatch(storeImageAgain, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageAgainCode.data()));
    device.WaitIdle();
    AgcDriver::Graphics::StorageTexture::FlushPending(base, ImageBytes, nullptr, "test");
    device.WaitIdle();
    std::printf("storage flush ordering (refresh): dword 0 = 0x%08x, dword %zu = 0x%08x, far = 0x%08x\n", words[0], OverwrittenDword, words[OverwrittenDword], words[FarDword]);
    Require(words[OverwrittenDword] == Overwritten, "storage flush ordering: the buffer write between two image writes was lost");
    Require(words[FarDword] == StoredAgain, "storage flush ordering: the image's far texel from the second write was not stored after a buffer write in its range");
    Require(words[0] == StoredAgain, "storage flush ordering: the image's texel written after the buffer write was dropped with its block");
}

void RunMarkedStore(AgcDriver::VulkanDevice& device) {
    GuestBlock guest;
    auto* words = guest.Words();
    std::fill(words, words + ImageBytes / 4u, Initial);
    const auto base = guest.Address();
    std::vector<std::uint32_t> imageUser(12, 0u);
    imageUser[4] = static_cast<std::uint32_t>(base >> 8u);
    imageUser[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (20u << 20u) | (((Width - 1u) & 3u) << 30u);
    imageUser[6] = ((Width - 1u) >> 2u) | ((Height - 1u) << 14u);
    imageUser[7] = 0x90000facu;
    const auto target = device.Target();
    const auto storeImage = Compile(StoreImageCode, imageUser, target);
    const auto storeImageAgain = Compile(StoreImageAgainCode, imageUser, target);
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    AgcDriver::Graphics::StorageTexture::FlushPending(base, ImageBytes, nullptr, "test");
    device.Dispatch(storeImageAgain, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageAgainCode.data()));
    const auto behindBefore = AgcDriver::Graphics::StorageTexture::FlushesBehindWriters();
    const auto skippedBefore = AgcDriver::Graphics::StorageTexture::FlushWaitsSkippedMarked();
    AgcDriver::Graphics::StorageTexture::FlushPending(base, ImageBytes, nullptr, "test");
    const auto behind = AgcDriver::Graphics::StorageTexture::FlushesBehindWriters() - behindBefore;
    const auto skipped = AgcDriver::Graphics::StorageTexture::FlushWaitsSkippedMarked() - skippedBefore;
    device.WaitIdle();
    std::printf("storage flush ordering (recorded store): dword 0 = 0x%08x, far = 0x%08x, behind %llu, skipped %llu\n", words[0], words[FarDword], static_cast<unsigned long long>(behind), static_cast<unsigned long long>(skipped));
    Require(words[0] == StoredAgain && words[FarDword] == StoredAgain, "storage flush ordering: the newer results did not replace a recorded store of the same memory");
    Require(skipped <= behind, "storage flush ordering: a flush that was not behind a writer counted a skipped wait");
    if (std::getenv("APS5_FLUSH_SKIP_MARKED") == nullptr) Require(skipped == 0, "storage flush ordering: a wait was skipped without APS5_FLUSH_SKIP_MARKED");
}

void RunRefreshLarge(AgcDriver::VulkanDevice& device) {
    GuestBlock guest;
    auto* words = guest.Words();
    std::fill(words, words + LargeImageBytes / 4u, Initial);
    const auto base = guest.Address();
    std::vector<std::uint32_t> imageUser(12, 0u);
    imageUser[4] = static_cast<std::uint32_t>(base >> 8u);
    imageUser[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (20u << 20u) | (((LargeWidth - 1u) & 3u) << 30u);
    imageUser[6] = ((LargeWidth - 1u) >> 2u) | ((LargeHeight - 1u) << 14u);
    imageUser[7] = 0x90000facu;
    std::vector<std::uint32_t> bufferUser(8, 0u);
    bufferUser[4] = static_cast<std::uint32_t>(base);
    bufferUser[5] = static_cast<std::uint32_t>(base >> 32u);
    bufferUser[6] = static_cast<std::uint32_t>(LargeImageBytes);
    bufferUser[7] = 0x31016facu;
    const auto target = device.Target();
    const auto storeImage = Compile(StoreImageCode, imageUser, target);
    const auto storeBuffer = Compile(StoreBufferCode, bufferUser, target);
    const auto storeImageAgain = Compile(StoreImageAgainCode, imageUser, target);
    device.Dispatch(storeImage, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
    device.Dispatch(storeBuffer, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreBufferCode.data()));
    device.Dispatch(storeImageAgain, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageAgainCode.data()));
    device.WaitIdle();
    AgcDriver::Graphics::StorageTexture::FlushPending(base, LargeImageBytes, nullptr, "test");
    device.WaitIdle();
    std::printf("storage flush ordering (large): dword 0 = 0x%08x, dword %zu = 0x%08x, far = 0x%08x\n", words[0], OverwrittenDword, words[OverwrittenDword], words[LargeFarDword]);
    Require(words[OverwrittenDword] == Overwritten, "storage flush ordering: the buffer write between two writes of a 2 MiB image was lost");
    Require(words[LargeFarDword] == StoredAgain, "storage flush ordering: the 2 MiB image's far texel from the second write was not stored");
    Require(words[0] == StoredAgain, "storage flush ordering: the 2 MiB image's texel written after the buffer write was dropped with its block");
}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        RunUploadOrder(*device);
        RunRefreshBehindWriter(*device);
        RunRefresh(*device);
        RunRefreshLarge(*device);
        RunMarkedStore(*device);
        std::puts("storage flush ordering tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "storage flush ordering: %s\n", error.what());
        return 1;
    }
}
