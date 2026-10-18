#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

#ifdef _WIN32

namespace {

using namespace NewPath;
using namespace NewPath::Isa;
using AgcDriver::Graphics::Recorder;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t GraphicsQueue = 0;
constexpr std::uint32_t OtherQueue = 0x20;
constexpr std::uint32_t HeldQueue = 0x28;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t WaitEqual = 3;
constexpr std::uint32_t Open = 0x6a7e0001u;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Side = 64;
constexpr VkComponentMapping Identity{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};

int failures = 0;
const int otherObject = 0;
const int firstObject = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

Recorder& ActiveRecorder() {
    Recorder* recorder = nullptr;
    Locked([&] { recorder = Recorder::Active(); });
    Require(recorder != nullptr, "barrier sections: the device has no recorder");
    return *recorder;
}

void SectionOnOtherQueue(Recorder& recorder, std::uint32_t& stream) {
    std::jthread other([&] {
        AgcDriver::GuestMemory::TagGpuLockThread(OtherQueue);
        Locked([&] {
            stream = recorder.CallerStream();
            const std::array<Recorder::SectionObject, 1> objects{{{&otherObject, true}}};
            recorder.DeclareSection(objects);
            static_cast<void>(recorder.Commands());
        });
    });
}

void FamilyBesideOtherQueue(AgcDriver::VulkanDevice& device, const std::string& family, const std::function<void()>& section) {
    auto& recorder = ActiveRecorder();
    Locked([&] {
        device.SubmitRecorded(true);
        device.WaitIdle();
    });
    std::uint64_t open = 0;
    Locked([&] {
        section();
        open = recorder.OpenBatchValue(GraphicsQueue);
    });
    Require(open != 0, family + ": the section on queue 0 opened no batch");
    const auto orderSubmits = recorder.OrderSubmits();
    std::uint32_t stream = 0;
    SectionOnOtherQueue(recorder, stream);
    const bool forced = recorder.OrderSubmits() != orderSubmits;
    const auto after = recorder.OpenBatchValue(GraphicsQueue);
    std::printf("%s on queue 0, then a section on queue 0x%x (stream %u) touching another object: queue 0's open batch %s\n", family.c_str(), OtherQueue, stream, forced || after != open ? "was submitted" : "stayed open");
    Require(stream != GraphicsQueue, family + ": queue 0x20 records on queue 0's stream");
    Expect(!forced, family + ": the other queue's section submitted queue 0's open batch to order after it");
    Expect(after == open, family + ": queue 0's open batch did not stay open beside the other queue's section");
    Locked([&] {
        device.SubmitRecorded(true);
        device.WaitIdle();
    });
}

void Families() {
    auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "barrier sections: no device");
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    auto& recorder = ActiveRecorder();
    if (recorder.Streams() < 2) {
        std::printf("the device records every queue on one stream; family cases skipped\n");
        return;
    }
    PrivatePages texels(1);
    PrivatePages colors(1);
    PrivatePages depths(1);
    PrivatePages labels(1);
    for (std::uint32_t texel = 0; texel < Side * 4u; ++texel) texels.Words()[texel] = texel;
    AgcDriver::GuestMemory::CollectWritesUncached(texels.Address(), texels.Bytes());

    if (context.deviceImages != nullptr) {
        FamilyBesideOtherQueue(*device, "the device image table's update", [&] { context.deviceImages->BeforeDispatch(context, recorder); });
    } else {
        std::printf("the device has no device image table; that case skipped\n");
    }
    std::shared_ptr<AgcDriver::Graphics::Texture> texture;
    FamilyBesideOtherQueue(*device, "a sampled texture's upload", [&] {
        const auto words = ImageResource(texels.Address(), Format32UInt, Side, 4);
        texture = AgcDriver::Graphics::CachedSampledTexture(context, words);
        Require(texture != nullptr, "barrier sections: the sampled texture was not made");
    });
    FamilyBesideOtherQueue(*device, "a depth surface's creation", [&] {
        const AgcDriver::Graphics::DepthTarget target{depths.Address(), 0, {Side, Side}, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
        Require(AgcDriver::Graphics::DepthSurfaceView(context, target) != VK_NULL_HANDLE, "barrier sections: the depth surface has no view");
    });
    if ((context.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0) {
        FamilyBesideOtherQueue(*device, "a multisampled target's creation", [&] {
            AgcDriver::Graphics::ColorTarget target{};
            target.address = colors.Address();
            target.surfaceAddress = target.address;
            target.extent = {Side, Side};
            target.surfaceExtent = target.extent;
            target.format = VK_FORMAT_R8G8B8A8_UNORM;
            target.tileMode = AgcDriver::Graphics::ColorTileMode::RenderTarget;
            target.elementBytes = 4;
            target.bytes = colors.Bytes() / 4u;
            target.samples = 4;
            Require(AgcDriver::Graphics::MultisampleTargetView(context, target) != VK_NULL_HANDLE, "barrier sections: the multisampled target has no view");
        });
    } else {
        std::printf("the device has no 4x color targets; that case skipped\n");
    }
    FamilyBesideOtherQueue(*device, "a completion label", [&] {
        const std::array<Recorder::SectionObject, 1> objects{{{&firstObject, true}}};
        recorder.DeclareSection(objects);
        static_cast<void>(recorder.Commands());
        const std::array<std::uint32_t, 1> value{1u};
        recorder.AfterCompletions(labels.Address(), std::as_bytes(std::span(value)), 1, GraphicsQueue, true);
    });
    texture.reset();
    Locked([&] {
        device->SubmitRecorded(true);
        device->WaitIdle();
        AgcDriver::Graphics::ClearMultisampleTargets(context.device);
        AgcDriver::Graphics::ClearDepthSurfaces(context.device);
        AgcDriver::Graphics::ClearCachedTextures(context.device);
    });
}

Program Storer;

bool LandsWithin(volatile std::uint32_t* word, std::uint32_t value, Clock::duration limit) {
    const auto start = Clock::now();
    while (*word != value && Clock::now() - start < limit) std::this_thread::yield();
    return *word == value;
}

void SetWord(PrivatePages& pages, std::uint64_t offset, std::uint32_t value) {
    pages.Words()[offset / 4u] = value;
    AgcDriver::GuestMemory::CollectWritesUncached(pages.Address(offset), 4u);
}

void OtherQueueSubmission(PrivatePages& stored, PrivatePages& labels, std::uint32_t value) {
    Commands commands;
    commands.Dispatch(Storer, BufferResource(stored.Address(), Lanes * 4u), Lanes);
    commands.WriteData(labels.Address(), {value});
    commands.Submit(OtherQueue);
}

std::vector<std::uint64_t> OpenBatches(const Recorder& recorder) {
    std::vector<std::uint64_t> values(recorder.Streams());
    for (std::uint32_t stream = 0; stream < values.size(); ++stream) values[stream] = recorder.OpenBatchValue(stream);
    return values;
}

void SubmissionEnd() {
    const std::string what = "the end of a submission on queue 0x20 while queue 0x28's batch is open";
    Storer << Vop2(VLshlRevB32, 1, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 0, 1, 0) << EndProgram;
    PrivatePages stored(1);
    PrivatePages other(1);
    PrivatePages labels(1);
    PrivatePages gates(1);
    std::uint32_t label = 0x56000000u;
    {
        for (const auto queue : {GraphicsQueue, HeldQueue}) {
            Commands warm;
            warm.Dispatch(Storer, BufferResource(stored.Address(), Lanes * 4u), Lanes);
            warm.Submit(queue);
            WaitIdle();
        }
        OtherQueueSubmission(other, labels, ++label);
        Require(LandsWithin(labels.Words(), label, std::chrono::seconds(10)), what + ": the first label did not land");
        WaitIdle();
        Settle();
    }
    auto& recorder = ActiveRecorder();
    Require(recorder.Streams() > 1, what + ": the device records every queue on one stream");
    for (std::uint32_t attempt = 0; attempt < 3; ++attempt) {
        SetWord(gates, 0, 0u);
        const auto done = attempt + 1u;
        Commands held;
        held.Dispatch(Storer, BufferResource(stored.Address(), Lanes * 4u), Lanes);
        held.Wait(gates.Address(), WaitEqual, Open, 0xffffffffu);
        held.WriteData(gates.Address(8), {done});
        held.Submit(HeldQueue);
        std::vector<std::uint64_t> before;
        std::uint32_t heldStream = 0;
        auto anyOpen = [&] {
            before = OpenBatches(recorder);
            for (std::uint32_t stream = 1; stream < before.size(); ++stream) {
                if (before[stream] != 0) {
                    heldStream = stream;
                    return true;
                }
            }
            return false;
        };
        const auto start = Clock::now();
        while (!anyOpen() && Clock::now() - start < std::chrono::seconds(10)) std::this_thread::yield();
        Require(heldStream != 0, what + ": queue 0x28 opened no batch for its dispatch");
        OtherQueueSubmission(other, labels, ++label);
        const bool landed = LandsWithin(labels.Words(), label, std::chrono::seconds(10));
        OtherQueueSubmission(other, labels, ++label);
        const bool nextLanded = LandsWithin(labels.Words(), label, std::chrono::seconds(10));
        const auto after = recorder.OpenBatchValue(heldStream);
        const bool waiting = gates.Words()[2] != done;
        std::printf("%s (attempt %u): labels %s, queue 0x28's batch %s after two submissions on queue 0x20 ended, queue 0x28 %s\n", what.c_str(), attempt, landed && nextLanded ? "landed" : "did not land", after == before[heldStream] ? "stayed open" : "was submitted", waiting ? "waited" : "had finished");
        Expect(landed && nextLanded, what + ": the labels on queue 0x20 did not land");
        Expect(waiting, what + ": queue 0x28 did not wait for its gate");
        Expect(after == before[heldStream], what + ": the submission end on queue 0x20 submitted queue 0x28's open batch");
        SetWord(gates, 0, Open);
        Expect(LandsWithin(gates.Words() + 2, done, std::chrono::seconds(10)), what + ": queue 0x28 did not finish after its gate opened");
        WaitIdle();
        Settle();
    }
}

}

#endif

int main() {
#ifdef _WIN32
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    try {
        Families();
        SubmissionEnd();
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d barrier section checks failed\n", failures);
        return 1;
    }
    std::printf("barrier section tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
