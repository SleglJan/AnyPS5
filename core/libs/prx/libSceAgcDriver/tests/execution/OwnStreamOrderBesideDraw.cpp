#include "DrawSubmitFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32

namespace {

using DeviceTable::Expect;
using NewPath::Commands;
using NewPath::High;
using NewPath::Low;
using NewPath::PrivatePages;
using NewPath::Program;
using NewPath::WordsPerPage;

constexpr std::uint32_t GraphicsQueue = 0;
constexpr std::uint32_t WaitFreeQueue = 0x20;
constexpr std::uint32_t TargetWidth = 128;
constexpr std::uint32_t TargetHeight = 128;
constexpr std::uint32_t RenderTargetAttrib3 = (0x1bu << 14u) | (1u << 24u) | (1u << 27u);
constexpr std::uint32_t VertexUserWords = 4;
constexpr std::uint32_t Red = 0xff0000ffu;
constexpr std::uint32_t RectList = 17;
constexpr std::uint32_t Draws = 256;

using Corners = std::array<std::array<float, 4>, 3>;
constexpr Corners FullRectangle{{{-1.0f, -1.0f, 0.5f, 1.0f}, {1.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 1.0f, 0.5f, 1.0f}}};

DrawSubmit::Header VertexHeader;
DrawSubmit::Header PixelHeader;

const Program& BoundedVertex() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Indexed(BufferLoadDwordX4, 10, 5, 8)
              << WaitAll
              << Export(PositionTarget, 10, false)
              << EndProgram;
        return built;
    }();
    return program;
}

const Program& RedPixel() {
    static const Program program = [] {
        using namespace DrawSubmit::Isa;
        Program built;
        built << Vop1(VMovB32, 0, InlineOne)
              << Vop1(VMovB32, 1, InlineInteger(0))
              << Vop1(VMovB32, 2, InlineInteger(0))
              << Vop1(VMovB32, 3, InlineOne)
              << Export(ColorTarget, 0, true)
              << EndProgram;
        return built;
    }();
    return program;
}

struct VertexBuffer {
    PrivatePages pages{1};

    VertexBuffer() { std::memcpy(pages.Data(), FullRectangle.data(), sizeof(FullRectangle)); }

    std::array<std::uint32_t, VertexUserWords> Descriptor() const {
        const auto address = pages.Address();
        return {Low(address), (High(address) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(FullRectangle.size()), 0x01016facu};
    }
};

std::size_t Count(const std::string& text, const std::string& what) {
    std::size_t count = 0;
    for (auto at = text.find(what); at != std::string::npos; at = text.find(what, at + 1u)) ++count;
    return count;
}

AgcDriver::Graphics::Recorder& ActiveRecorder() {
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    NewPath::Require(recorder != nullptr, "own-stream order beside a draw: the device has no recorder");
    return *recorder;
}

void RecordDraw(Commands& commands, const PrivatePages& target, const VertexBuffer& corners) {
    DrawSubmit::Baseline(commands, {target.Data(), TargetWidth, TargetHeight}, DrawSubmit::CenterAndPosition);
    DrawSubmit::Context(commands, 0x3b8, RenderTargetAttrib3);
    DrawSubmit::Context(commands, 0x31b, 0u);
    DrawSubmit::Programs(commands, BoundedVertex(), VertexUserWords, RedPixel(), 0u);
    DrawSubmit::Instances(commands, 1u);
    DrawSubmit::PrimitiveType(commands, RectList);
    DrawSubmit::VertexUserData(commands, corners.Descriptor());
    DrawSubmit::DrawAuto(commands, static_cast<std::uint32_t>(FullRectangle.size()));
}

std::string Submitted(Commands& commands, std::string& log) {
    std::string failure;
    NewPath::StderrCapture capture;
    try {
        commands.Submit(GraphicsQueue);
        NewPath::WaitIdle();
        NewPath::Settle();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    log = capture.Finish();
    return failure;
}

std::size_t NotRed(const PrivatePages& target) {
    const auto pixels = DrawSubmit::ReadBack(reinterpret_cast<const std::uint32_t*>(target.Data()), TargetWidth * TargetHeight);
    return static_cast<std::size_t>(std::ranges::count_if(pixels, [](std::uint32_t pixel) { return pixel != Red; }));
}

void ClearTarget(PrivatePages& target) {
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) target.Words()[word] = 0u;
    AgcDriver::GuestMemory::CollectWritesUncached(target.Address(), NewPath::PageBytes);
}

void DrawAfterWaitFreeOrder() {
    DrawSubmit::Register(VertexHeader, BoundedVertex(), DrawSubmit::VertexShaderType);
    DrawSubmit::Register(PixelHeader, RedPixel(), DrawSubmit::PixelShaderType);
    PrivatePages target(1);
    const VertexBuffer corners;
    ClearTarget(target);
    {
        Commands warm;
        RecordDraw(warm, target, corners);
        std::string log;
        const auto failure = Submitted(warm, log);
        Expect(failure.empty(), "the first rect-list draw failed: " + failure + "\n" + log);
        Expect(NotRed(target) == 0u, "the first rect-list draw left pixels not red");
    }
    ClearTarget(target);
    auto& recorder = ActiveRecorder();
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> ownStream{0};
    std::atomic<std::uint64_t> locked{0};
    std::thread waitFree([&] {
        AgcDriver::GuestMemory::TagGpuLockThread(WaitFreeQueue);
        while (!stop.load(std::memory_order_acquire)) {
            if (recorder.OrderAfterQueueOnOwnStream(GraphicsQueue)) {
                ownStream.fetch_add(1, std::memory_order_relaxed);
            } else {
                NewPath::Locked([&] { recorder.OrderAfterQueue(GraphicsQueue); });
                locked.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    });
    Commands commands;
    for (std::uint32_t draw = 0; draw < Draws; ++draw) RecordDraw(commands, target, corners);
    std::string log;
    const auto failure = Submitted(commands, log);
    stop.store(true, std::memory_order_release);
    waitFree.join();
    NewPath::Settle();
    const auto text = failure + "\n" + log;
    const std::string what = std::to_string(Draws) + " queue-0 rect-list draws recorded while another queue orders itself after queue 0's open batch on its own stream (" + std::to_string(ownStream.load()) + " orders on its own stream, " + std::to_string(locked.load()) + " under the device lock)";
    Expect(ownStream.load() != 0u, what + ": no order after queue 0 was recorded on the other queue's own stream");
    Expect(failure.empty(), what + ": the submission failed: " + failure);
    Expect(Count(text, "a build was recorded into a batch other than its fault record's") == 0u, what + ": a build was recorded into a batch other than its fault record's:\n" + text);
    Expect(Count(text, "deferred write-back failed") == 0u, what + ": a write-back failed:\n" + text);
    Expect(NotRed(target) == 0u, what + ": the draws left " + std::to_string(NotRed(target)) + " of " + std::to_string(TargetWidth * TargetHeight) + " pixels not red");
}

}

#endif

int main() {
#ifdef _WIN32
    return DrawSubmit::Main("own-stream order beside a draw", DrawAfterWaitFreeOrder);
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
