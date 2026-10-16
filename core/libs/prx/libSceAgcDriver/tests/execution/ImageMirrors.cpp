#include "NewPathFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

#ifdef _WIN32

namespace {

namespace Graphics = AgcDriver::Graphics;
using namespace NewPath;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t MirroredWords = 4096;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t LaneStride = MirroredWords / Lanes;

constexpr std::array<std::uint32_t, MirroredWords> MakeWords() {
    std::array<std::uint32_t, MirroredWords> words{};
    for (std::uint32_t index = 0; index < MirroredWords; ++index) words[index] = 0x7e000000u | (index * 2654435761u >> 8u);
    return words;
}

alignas(4096) const std::array<std::uint32_t, MirroredWords> Mirrored = MakeWords();

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::uint64_t Address() {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Mirrored.data()));
}

constexpr std::uint64_t Bytes() {
    return std::uint64_t{MirroredWords} * 4u;
}

void Build(const Graphics::Context& context) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Graphics::GuestBufferMemory memory(context);
    memory.AcquireRegistered();
    memory.Upload();
    static_cast<void>(memory.PageTable(context));
    memory.Release();
}

std::uint64_t Serial(const Graphics::Context& context) {
    return Graphics::ImageMirrorSerial(context, Address(), static_cast<std::size_t>(Bytes()));
}

const Program& ReadProgram() {
    static const Program program = [] {
        using namespace Isa;
        Program built;
        built << Vop2(VLshlRevB32, 1, InlineInteger(8), 0)
              << Global(GlobalLoadDword, 2, 1, 0, 0)
              << WaitAll
              << Vop2(VLshlRevB32, 3, InlineInteger(2), 0)
              << Global(GlobalStoreDword, 0, 3, 2, 2)
              << EndProgram;
        return built;
    }();
    return program;
}

std::vector<std::uint32_t> Read(AgcDriver::VulkanDevice& device, const std::string& what) {
    PrivatePages output(1);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) output.Words()[lane] = 0xdeadbeefu;
    const std::array<std::uint32_t, 4> userData{Low(Address()), High(Address()), Low(output.Address()), High(output.Address())};
    const auto& program = ReadProgram();
    const auto code = program.Code();
    const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{static_cast<std::uintptr_t>(program.Address()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, program.Address(), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, regions},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    std::string failure;
    {
        StderrCapture capture;
        try {
            const auto result = ShaderRecompiler::Recompile(request);
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            device.Dispatch(result, 1, 1, 1, static_cast<std::uintptr_t>(program.Address()));
            device.WaitIdle();
        } catch (const std::exception& error) {
            failure = error.what();
        }
        const auto log = capture.Finish();
        if (log.find("BDA access failed") != std::string::npos || log.find("failed") != std::string::npos) failure += "\n" + log;
    }
    Expect(failure.empty(), what + ": the dispatch reading the mirrored range failed: " + failure);
    AgcDriver::GuestMemory::FlushGpuWrites(output.Address(), output.Bytes());
    std::vector<std::uint32_t> words(Lanes);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) words[lane] = output.Words()[lane];
    return words;
}

void ExpectRead(AgcDriver::VulkanDevice& device, const std::string& what) {
    const auto words = Read(device, what);
    std::size_t wrong = 0;
    std::string first;
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const auto expected = reinterpret_cast<const volatile std::uint32_t*>(Mirrored.data())[lane * LaneStride];
        if (words[lane] == expected) continue;
        if (wrong++ == 0u) first = "lane " + std::to_string(lane) + " read " + Hex(words[lane]) + ", expected " + Hex(expected);
    }
    Expect(wrong == 0u, what + ": " + std::to_string(wrong) + " of " + std::to_string(Lanes) + " lanes read other bytes than the range holds, first " + first);
}

void Protect(bool gpu, const std::function<void()>& apply) {
    GuestAllocations::Mutation().Protect(Mirrored.data(), static_cast<std::size_t>(Bytes()), true, false, gpu, apply);
}

void Rewrite(std::uint32_t seed) {
    DWORD previous = 0;
    auto* words = const_cast<volatile std::uint32_t*>(reinterpret_cast<const volatile std::uint32_t*>(Mirrored.data()));
    Require(VirtualProtect(const_cast<std::uint32_t*>(Mirrored.data()), static_cast<SIZE_T>(Bytes()), PAGE_READWRITE, &previous) != 0, "image mirrors: cannot make the read-only words writable");
    for (std::uint32_t index = 0; index < MirroredWords; ++index) words[index] = seed ^ (index * 40503u);
    DWORD restored = 0;
    Require(VirtualProtect(const_cast<std::uint32_t*>(Mirrored.data()), static_cast<SIZE_T>(Bytes()), previous, &restored) != 0, "image mirrors: cannot restore the words' protection");
}

void ExpectDeviceCopy(const std::string& what) {
    const auto stats = Graphics::MirrorCounters();
    Expect(stats.bytes != 0u && stats.deviceBytes == stats.bytes, "image mirrors: " + what + ": " + std::to_string(stats.bytes - stats.deviceBytes) + " of " + std::to_string(stats.bytes) + " mirrored bytes are a host-side copy, not device memory");
}

void Mirrors(AgcDriver::VulkanDevice& device, const Graphics::Context& context) {
    MEMORY_BASIC_INFORMATION info{};
    Require(VirtualQuery(Mirrored.data(), &info, sizeof(info)) == sizeof(info) && info.Type == MEM_IMAGE && (info.Protect & 0xffu) == PAGE_READONLY, "image mirrors: the test's constant words are not read-only image pages");
    GuestAllocations::Mutation().RegisterMainImage();
    Protect(true, [] {});
    {
        const auto range = GuestAllocations::GuestAllocationsLookup_nid_postfix(static_cast<std::uintptr_t>(Address()), false);
        Require(range != nullptr && range->address == Address() && range->bytes == Bytes() && range->gpu && !range->writable && !range->releasable, "image mirrors: the words are not one read-only GPU range of the main image");
    }

    Build(context);
    const auto first = Serial(context);
    Expect(first != 0u && Graphics::HostImportSerial(context, Address(), static_cast<std::size_t>(Bytes()), false) == 0u, "image mirrors: the read-only range of the main image is not served by a mirror (mirror serial " + Hex(first) + ")");
    ExpectRead(device, "a dispatch reading a read-only range of the main image");
    ExpectDeviceCopy("the first mirror");
    Build(context);
    Expect(Serial(context) == first, "image mirrors: a build with no mapping change built the mirror again (serial " + Hex(first) + ", now " + Hex(Serial(context)) + ")");

    Protect(true, [] { Rewrite(0x5d000000u); });
    Build(context);
    const auto rewritten = Serial(context);
    Expect(rewritten != 0u && rewritten != first, "image mirrors: after the guest rewrote the range under a protection change, the mirror kept its serial " + Hex(first));
    ExpectRead(device, "a dispatch reading the range after the guest rewrote it under a protection change");
    ExpectDeviceCopy("the mirror rebuilt after the rewrite");

    Protect(false, [] {});
    Build(context);
    Expect(Serial(context) == 0u, "image mirrors: the mirror of a range no longer mapped for the GPU is still served (serial " + Hex(Serial(context)) + ")");
    Protect(true, [] { Rewrite(0x2b000000u); });
    Build(context);
    const auto mappedAgain = Serial(context);
    Expect(mappedAgain != 0u && mappedAgain != first && mappedAgain != rewritten, "image mirrors: the range mapped for the GPU again did not get a new mirror (serial " + Hex(mappedAgain) + ")");
    ExpectRead(device, "a dispatch reading the range mapped for the GPU again");
    ExpectDeviceCopy("the mirror of the range mapped again");

    GuestAllocations::Mutation().UnregisterImage(GetModuleHandleW(nullptr));
    Build(context);
    Expect(Serial(context) == 0u, "image mirrors: the mirror outlived the unregistered image (serial " + Hex(Serial(context)) + ")");
}

}

#endif

int main() {
#ifdef _WIN32
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        if (!context.bufferDeviceAddress) {
            std::puts("image mirrors: buffer device addresses unavailable");
            return VulkanTestSkipped;
        }
        Mirrors(*device, context);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d image mirror checks failed\n", failures);
        return 1;
    }
    std::puts("image mirror tests passed");
    return 0;
#else
    return VulkanTestSkipped;
#endif
}
