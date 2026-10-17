#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Residency.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

#ifdef _WIN32
namespace {

using AgcDriver::Graphics::Require;
namespace Graphics = AgcDriver::Graphics;

constexpr std::size_t RegionBytes = std::size_t{1} << 20u;
constexpr std::size_t OwnedBytes = std::size_t{1} << 20u;
constexpr std::size_t FillBytes = 4096;
constexpr std::uint32_t ComputeQueue = 0x20;
constexpr std::uint32_t FreshQueue = 0x28;
constexpr std::uint64_t PassKey = 0x5157;
constexpr std::array<std::uint32_t, 4> Before{0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u};
constexpr std::array<std::uint32_t, 4> After{0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};

class GuestRegion {
public:
    GuestRegion() {
        region = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, RegionBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        Require(region != nullptr, "stream dependency publication: cannot allocate the guest region");
        GuestAllocations::Mutation().Add(region, RegionBytes, true, true, true);
    }

    ~GuestRegion() {
        GuestAllocations::Mutation().Remove(region);
        VirtualFree(region, 0, MEM_RELEASE);
    }

    GuestRegion(const GuestRegion&) = delete;
    GuestRegion& operator=(const GuestRegion&) = delete;

    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(region)); }

    std::uint32_t Word(std::size_t index) const {
        std::uint32_t word = 0;
        std::memcpy(&word, region + index * 4, 4);
        return word;
    }

private:
    std::uint8_t* region = nullptr;
};

class EmptyPass {
public:
    explicit EmptyPass(const Graphics::Context& context) : context(context) {
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        passInfo.subpassCount = 1;
        passInfo.pSubpasses = &subpass;
        Graphics::Check(context.Function<PFN_vkCreateRenderPass>("vkCreateRenderPass")(context.device, &passInfo, nullptr, &renderPass), "vkCreateRenderPass empty pass");
        VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.width = 8;
        framebufferInfo.height = 8;
        framebufferInfo.layers = 1;
        const auto created = context.Function<PFN_vkCreateFramebuffer>("vkCreateFramebuffer")(context.device, &framebufferInfo, nullptr, &framebuffer);
        if (created != VK_SUCCESS) {
            context.Function<PFN_vkDestroyRenderPass>("vkDestroyRenderPass")(context.device, renderPass, nullptr);
            Graphics::Check(created, "vkCreateFramebuffer empty pass");
        }
    }

    ~EmptyPass() {
        context.Function<PFN_vkDestroyFramebuffer>("vkDestroyFramebuffer")(context.device, framebuffer, nullptr);
        context.Function<PFN_vkDestroyRenderPass>("vkDestroyRenderPass")(context.device, renderPass, nullptr);
    }

    EmptyPass(const EmptyPass&) = delete;
    EmptyPass& operator=(const EmptyPass&) = delete;

    void LeaveOpen(Graphics::Recorder& recorder) const {
        const auto start = recorder.StartDrawPass(PassKey, true, false, Graphics::PassAccess{});
        VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        begin.renderPass = renderPass;
        begin.framebuffer = framebuffer;
        begin.renderArea = {{0, 0}, {8, 8}};
        context.Function<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass")(start.commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
        recorder.LeaveRenderPassOpen(PassKey, Graphics::Recorder::NoTiming, false, Graphics::PassAccess{});
    }

private:
    const Graphics::Context& context;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
};

class SubmittingOwner final : public Graphics::IResidencyOwner {
public:
    void Retile(std::span<const Graphics::ResidencyRange> ranges) override {
        static_cast<void>(ranges);
        ++retiles;
        Graphics::Recorder::Active()->Submit();
    }

    std::size_t retiles = 0;
};

void CallerBatchSubmittedByPublication() {
    GuestRegion region;
    auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "the Vulkan device opened by the parent did not open in the scenario");
    bool imported = true;
    std::thread warm([&] {
        AgcDriver::GuestMemory::TagGpuLockThread(ComputeQueue);
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        imported = device->FillBuffer(region.Address(), FillBytes, Before);
        device->SubmitRecorded(false);
        device->WaitIdle();
    });
    warm.join();
    Require(imported, "the guest region was not imported");
    auto* recorder = Graphics::Recorder::Active();
    auto* residency = Graphics::ResidencyDriver::Active();
    Require(recorder != nullptr && residency != nullptr, "the device has no recorder or residency driver");
    auto* owned = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, OwnedBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    Require(owned != nullptr, "cannot allocate the owned range");
    const auto ownedAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(owned));
    SubmittingOwner owner;
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    {
        const EmptyPass pass(context);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            residency->Manager().SetBudget(0);
            residency->Manager().Own(ownedAddress, OwnedBytes, &owner);
            residency->Manager().MarkStale(ownedAddress, OwnedBytes, &owner);
            Require(residency->Manager().AnyStale(), "the owned range is not stale");
            Require(owner.retiles == 0, "the owned range was retiled when it was marked stale");
            pass.LeaveOpen(*recorder);
            Graphics::PublishBeforeCompletionSignal(Graphics::SignalKind::Completion);
            Require(owner.retiles == 0, "the completion signal inside an open render pass was published at once, not when its batch is submitted");
        }
        std::thread compute([&] {
            AgcDriver::GuestMemory::TagGpuLockThread(ComputeQueue);
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            Require(device->FillBuffer(region.Address(), FillBytes, After), "the compute queue's fill was not recorded");
            device->OrderAfterQueue(0);
            std::uint32_t stream = 0;
            std::uint64_t value = 0;
            Require(recorder->CallerBatchValue(stream, value), "the compute queue has no open batch to carry its wait for the graphics queue");
            Require(device->FillBuffer(region.Address() + FillBytes, FillBytes, After), "the compute queue's fill after the wait was not recorded");
            device->SubmitRecorded(false);
            device->WaitIdle();
        });
        compute.join();
        Require(owner.retiles == 1, "the publication owed by the graphics batch ran " + std::to_string(owner.retiles) + " times, expected once at its submission");
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        residency->Manager().Forget(&owner);
        device->SubmitRecorded(false);
        device->WaitIdle();
    }
    VirtualFree(owned, 0, MEM_RELEASE);
    for (std::size_t index = 0; index < 2 * FillBytes / 4; ++index) {
        if (region.Word(index) == After[index % 4]) continue;
        Require(false, "word " + std::to_string(index) + " holds " + std::to_string(region.Word(index)) + ", not the compute queue's fill");
    }
}

void FreshQueueOrderedAfterGraphics() {
    GuestRegion region;
    auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "the Vulkan device opened by the parent did not open in the scenario");
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Require(device->FillBuffer(region.Address(), FillBytes, Before), "the guest region was not imported");
    }
    auto* recorder = Graphics::Recorder::Active();
    Require(recorder != nullptr, "the device has no recorder");
    std::thread compute([&] {
        AgcDriver::GuestMemory::TagGpuLockThread(FreshQueue);
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device->OrderAfterQueue(0);
        std::uint32_t stream = 0;
        std::uint64_t value = 0;
        Require(recorder->CallerBatchValue(stream, value) && stream != 0, "a queue that had recorded nothing has no open batch of its own to carry its wait for the graphics queue");
        Require(device->FillBuffer(region.Address(), FillBytes, After), "the fresh queue's fill was not recorded");
        device->SubmitRecorded(false);
        device->WaitIdle();
    });
    compute.join();
    for (std::size_t index = 0; index < FillBytes / 4; ++index) {
        if (region.Word(index) == After[index % 4]) continue;
        Require(false, "word " + std::to_string(index) + " holds " + std::to_string(region.Word(index)) + ", not the fresh queue's fill ordered after the graphics fill");
    }
}

int RunChild(const char* scenario) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    try {
        if (std::strcmp(scenario, "fresh-queue") == 0) FreshQueueOrderedAfterGraphics();
        else if (std::strcmp(scenario, "caller-submitted") == 0) CallerBatchSubmittedByPublication();
        else throw std::runtime_error(std::string("unknown scenario ") + scenario);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << scenario << ": " << error.what() << '\n';
        return 1;
    }
}

DWORD RunScenario(const char* scenario) {
    std::wstring executable(MAX_PATH, L'\0');
    executable.resize(GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size())));
    std::wstring commandLine = L"\"" + executable + L"\" --scenario ";
    for (const char* c = scenario; *c != '\0'; ++c) commandLine.push_back(static_cast<wchar_t>(*c));
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    Require(CreateProcessW(executable.c_str(), commandLine.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process) != 0, "cannot start the scenario child");
    const auto waited = WaitForSingleObject(process.hProcess, 90000);
    if (waited != WAIT_OBJECT_0) TerminateProcess(process.hProcess, 0xdead);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return code;
}

std::size_t RecorderStreams() {
    GuestRegion region;
    auto device = OpenVulkanTestDevice();
    if (!device) return 0;
    bool imported = true;
    std::thread compute([&] {
        AgcDriver::GuestMemory::TagGpuLockThread(ComputeQueue);
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        imported = device->FillBuffer(region.Address(), FillBytes, Before);
        device->SubmitRecorded(false);
        device->WaitIdle();
    });
    compute.join();
    if (!imported) return 0;
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    return Graphics::Recorder::Active()->Streams();
}

}
#endif

int main(int argc, char** argv) {
#ifndef _WIN32
    std::puts("skipped: the scenario runs in a child process started on Windows");
    return VulkanTestSkipped;
#else
    if (argc == 3 && std::strcmp(argv[1], "--scenario") == 0) return RunChild(argv[2]);
    try {
        const auto streams = RecorderStreams();
        if (streams < 2) {
            std::printf("skipped, the device records on %zu stream(s) or does not import guest memory\n", streams);
            return VulkanTestSkipped;
        }
        for (const char* scenario : {"fresh-queue", "caller-submitted"}) {
            const auto code = RunScenario(scenario);
            char text[16];
            std::snprintf(text, sizeof(text), "%lx", static_cast<unsigned long>(code));
            Require(code == 0, std::string("scenario ") + scenario + " ended with exit code 0x" + text + ": a dependency on another stream lost the caller's open batch");
        }
        std::printf("stream dependency publication tests passed (%zu recorder streams)\n", streams);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
