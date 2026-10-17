#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <chrono>
#include <exception>
#include <optional>
#include <thread>

#ifdef _WIN32

namespace {

using namespace NewPath;

constexpr std::uint32_t WriterQueue = 0x20;
constexpr std::uint32_t WaiterQueue = 0x28;
constexpr std::uint32_t CopyPages = 512;
constexpr std::uint32_t Copies = 4;
constexpr std::uint32_t Equal = 3;
constexpr std::uint32_t Later = 0x6c000001u;

constexpr std::uint32_t SpinLanes = 64;
constexpr std::uint32_t SpinRounds = 1u << 20u;
constexpr std::uint32_t SpinOutput = 64;
constexpr std::uint32_t VLshrRevB32 = 0x16u;
constexpr std::uint32_t VXorB32 = 0x1du;
constexpr std::uint32_t MoveLiteralToS9 = 0xbe8903ffu;
constexpr std::uint32_t DecrementS9 = 0x80898109u;
constexpr std::uint32_t CompareS9NotZero = 0xbf078009u;
constexpr std::uint32_t BranchBackWhileNotZero = 0xbf85fff7u;

constexpr std::uint32_t SpinAttempts = 4;

Program Opener;
std::array<Program, SpinAttempts> Spinners;

void BuildPrograms() {
    using namespace Isa;
    Opener << Vop2(VLshlRevB32, 1, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 0, 1, 0) << EndProgram;
    for (std::uint32_t attempt = 0; attempt < SpinAttempts; ++attempt) {
        Spinners[attempt] << Vop2(VAddNcU32, 1, InlineInteger(1), 0) << MoveLiteralToS9 << (SpinRounds << attempt)
                          << Vop2(VLshlRevB32, 2, InlineInteger(13), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
                          << Vop2(VLshrRevB32, 2, InlineInteger(17), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
                          << Vop2(VLshlRevB32, 2, InlineInteger(5), 1) << Vop2(VXorB32, 1, Vgpr(1), 2)
                          << DecrementS9 << CompareS9NotZero << BranchBackWhileNotZero
                          << Vop2(VLshlRevB32, 3, InlineInteger(2), 0) << Mubuf(BufferStoreDword, 1, 3, 0) << EndProgram;
    }
}

std::uint32_t Spun(std::uint32_t lane, std::uint32_t attempt) {
    std::uint32_t value = lane + 1u;
    for (std::uint32_t round = 0; round < (SpinRounds << attempt); ++round) {
        value ^= value << 13u;
        value ^= value >> 17u;
        value ^= value << 5u;
    }
    return value;
}

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

bool Recorded(std::uint64_t address) {
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr && recorder->LabelPendingAt(address);
}

struct Observation {
    bool recorded = false;
    bool beforeLanding = false;
};

Observation WatchRecord(std::uint64_t target, volatile std::uint32_t* label, std::uint32_t initial) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        if (Recorded(target)) return {true, *label == initial};
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return {};
}

class Memory {
public:
    Memory() : sources(Copies * CopyPages), destinations(Copies * CopyPages), checks(TailBytes / PageBytes), slots(1) {
        std::memset(sources.Data(), 0x3c, sources.Bytes());
        for (std::uint32_t copy = 0; copy < Copies; ++copy) {
            auto* tail = sources.Data() + (std::uint64_t{copy + 1u} * CopyPages) * PageBytes - 16u;
            for (std::uint32_t byte = 0; byte < 16u; ++byte) tail[byte] = static_cast<std::uint8_t>(0x80u + copy * 16u + byte);
        }
        std::memset(slots.Data(), 0, slots.Bytes());
    }

    void Reset() {
        std::memset(destinations.Data(), 0xc3, destinations.Bytes());
        std::memset(checks.Data(), 0xa5, checks.Bytes());
        std::memset(slots.Data(), 0, slots.Bytes());
    }

    void SlowWrites(Commands& commands) const {
        for (std::uint32_t copy = 0; copy < Copies; ++copy) {
            const auto offset = std::uint64_t{copy} * CopyPages * PageBytes;
            commands.DmaCopy(destinations.Address(offset), sources.Address(offset), static_cast<std::uint32_t>(CopyPages * PageBytes));
        }
    }

    void Spin(Commands& commands, std::uint32_t attempt) const {
        const auto output = BufferResource(Slot(SpinOutput), SpinLanes * 4u);
        commands.Dispatch(Spinners[attempt], output, SpinLanes);
    }

    void CopyTail(Commands& commands) const {
        commands.DmaCopy(checks.Address(), destinations.Address(Total - TailBytes), static_cast<std::uint32_t>(TailBytes));
    }

    std::optional<std::uint64_t> TailMismatch() const {
        AgcDriver::GuestMemory::FlushGpuWrites(checks.Address(), TailBytes);
        const auto* expected = sources.Data() + (Total - TailBytes);
        for (std::uint64_t byte = 0; byte < TailBytes; ++byte) {
            if (checks.Data()[byte] != expected[byte]) return byte;
        }
        return std::nullopt;
    }

    std::uint8_t Checked(std::uint64_t byte) const { return checks.Data()[byte]; }
    std::uint8_t Source(std::uint64_t byte) const { return sources.Data()[Total - TailBytes + byte]; }

    std::uint64_t Slot(std::uint32_t index) const { return slots.Address(std::uint64_t{index} * 64u); }
    volatile std::uint32_t* SlotWords(std::uint32_t index) const { return slots.Words() + index * 16u; }

    static constexpr std::uint64_t Total = std::uint64_t{Copies} * CopyPages * PageBytes;
    static constexpr std::uint64_t TailBytes = 2 * PageBytes;

private:
    PrivatePages sources;
    PrivatePages destinations;
    PrivatePages checks;
    PrivatePages slots;
};

void GpuSideWait(Memory& memory) {
    for (std::uint32_t attempt = 0; attempt < SpinAttempts; ++attempt) {
        memory.Reset();
        const auto label = memory.Slot(0);
        const auto target = memory.Slot(1);
        Commands waiter;
        waiter.Wait(label, Equal, 1u, 0xffffffffu);
        waiter.WriteData(target, {Later});
        memory.CopyTail(waiter);
        waiter.Submit(WaiterQueue);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        Expect(!Recorded(target), "GPU-side wait: the work after a wait was recorded before any writer of its label existed");
        Commands spin;
        memory.Spin(spin, attempt);
        spin.Submit(WriterQueue);
        Commands writer;
        memory.SlowWrites(writer);
        writer.WriteData(label, {1u});
        writer.Submit(WriterQueue);
        const auto seen = WatchRecord(target, memory.SlotWords(0), 0u);
        Expect(seen.recorded, "GPU-side wait: the waiter never recorded the work after its wait");
        WaitIdle();
        if (seen.recorded && !seen.beforeLanding) continue;
        AgcDriver::GuestMemory::FlushGpuWrites(memory.Slot(0), 2 * 64u);
        AgcDriver::GuestMemory::FlushGpuWrites(memory.Slot(SpinOutput), SpinLanes * 4u);
        Expect(memory.SlotWords(1)[0] == Later, "GPU-side wait: the store after the wait did not land");
        Expect(memory.SlotWords(SpinOutput)[SpinLanes - 1u] == Spun(SpinLanes - 1u, attempt), "GPU-side wait: the writer's spinning dispatch stored " + Hex(memory.SlotWords(SpinOutput)[SpinLanes - 1u]) + ", expected " + Hex(Spun(SpinLanes - 1u, attempt)));
        if (const auto byte = memory.TailMismatch()) {
            Expect(false, "GPU-side wait: the GPU copy after the wait read byte " + std::to_string(*byte) + " of the writer's last copy as " + Hex(memory.Checked(*byte)) + ", before the writer's copy in the label's batch landed (" + Hex(memory.Source(*byte)) + ")");
        }
        return;
    }
    Expect(false, "GPU-side wait: in all " + std::to_string(SpinAttempts) + " attempts, the writer's spin of up to " + std::to_string(SpinRounds << (SpinAttempts - 1u)) + " rounds finished and its label landed before the waiter recorded the work after its wait (a CPU wait, or a waiter too slow to tell)");
}

void CpuWrittenLabel(Memory& memory) {
    memory.Reset();
    const auto label = memory.Slot(4);
    const auto target = memory.Slot(5);
    Commands waiter;
    waiter.Wait(label, Equal, 7u, 0xffffffffu);
    waiter.WriteData(target, {Later});
    waiter.Submit(WaiterQueue);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Expect(!Recorded(target), "CPU-written label: the work after a wait on a value no recorded work writes was recorded before the CPU stored it");
    Expect(memory.SlotWords(5)[0] == 0u, "CPU-written label: the work after the wait ran before the CPU stored the value");
    memory.SlotWords(4)[0] = 7u;
    WaitIdle();
    AgcDriver::GuestMemory::FlushGpuWrites(target, 4u);
    Expect(memory.SlotWords(5)[0] == Later, "CPU-written label: the work after the wait did not run once the CPU stored the value");
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
        Memory memory;
        {
            BuildPrograms();
            Commands open;
            const auto target = BufferResource(memory.Slot(8), 256u);
            open.Dispatch(Opener, target, 64u);
            open.Submit(0);
            WaitIdle();
        }
        const auto* recorder = AgcDriver::Graphics::Recorder::Active();
        for (auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60); recorder == nullptr && std::chrono::steady_clock::now() < deadline;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            recorder = AgcDriver::Graphics::Recorder::Active();
        }
        Require(recorder != nullptr, "stream waits: the driver opened no device within 60 s of its first submission");
        if (recorder->Streams() < 2) {
            std::printf("skipped, the device gives the recorder one stream\n");
            AgcDriverShutdown_nid_postfix();
            return VulkanTestSkipped;
        }
        GpuSideWait(memory);
        CpuWrittenLabel(memory);
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d new path stream wait checks failed\n", failures);
        return 1;
    }
    std::printf("new path stream wait tests passed\n");
    return 0;
#else
    std::printf("skipped, the guest arena is Windows-only here\n");
    return VulkanTestSkipped;
#endif
}
