#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::size_t CopyBytes = std::size_t{32} << 20u;
constexpr std::size_t Copies = 3;
constexpr std::size_t SlotBytes = 64;
constexpr std::size_t Slots = 64;
constexpr std::size_t RegionBytes = 2 * Copies * CopyBytes + Slots * SlotBytes;
constexpr std::uint32_t WriterQueue = 0x20;
constexpr std::uint32_t WaitFreeQueue = 0x28;
constexpr std::uint32_t Earlier = 0xbbbbbbbbu;
constexpr std::uint32_t Later = 0x55555555u;

alignas(256) constexpr std::array<std::uint32_t, 69> FillKernel{
    0xbfa00003u, 0xd7460002u, 0x04010c0au, 0x7da80408u, 0xbf88003fu, 0x7e000c09u, 0xbf070980u, 0x858a807eu,
    0x7e005700u, 0x100000ffu, 0x4f800000u, 0x7e060f00u, 0xd5766a00u, 0x02020609u, 0x7d8a0280u, 0x4c020080u,
    0x02000101u, 0xd56a0001u, 0x00020700u, 0x4c000303u, 0x4a020303u, 0x02000101u, 0xd56a0000u, 0x00020500u,
    0xd5690001u, 0x00020009u, 0x4c060302u, 0x7d860609u, 0x7d8c02f9u, 0x06068c02u, 0x87ea6a0cu, 0x50000080u,
    0xd5286a00u, 0x003200c1u, 0xd5010000u, 0x002a00c1u, 0xd5690000u, 0x00020009u, 0x4c000102u, 0x7d0a0080u,
    0xbe88246au, 0xbf880015u, 0x7d0a0081u, 0xbe8a246au, 0xbf88000cu, 0x7d0a0082u, 0xbeea246au, 0xbf880003u,
    0x7e000207u, 0xe0102000u, 0x80000002u, 0x8afe7e6au, 0xbf880003u, 0x7e000206u, 0xe0102000u, 0x80000002u,
    0xbefe046au, 0x8afe7e0au, 0xbf880003u, 0x7e000205u, 0xe0102000u, 0x80000002u, 0xbefe040au, 0x8afe7e08u,
    0xbf880003u, 0x7e000204u, 0xe0102000u, 0x80000002u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 256> WarmUpOutput{};

class GuestRegion {
public:
    GuestRegion() {
#ifdef _WIN32
        region = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, RegionBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        region = static_cast<std::uint8_t*>(std::aligned_alloc(65536, RegionBytes));
#endif
        Require(region != nullptr, "queue label order: cannot allocate the guest region");
        std::memset(region, 0x5a, RegionBytes);
        std::memset(region + 2 * Copies * CopyBytes, 0, Slots * SlotBytes);
        GuestAllocations::Mutation().Add(region, RegionBytes, true, true, true);
    }

    ~GuestRegion() {
        GuestAllocations::Mutation().Remove(region);
#ifdef _WIN32
        VirtualFree(region, 0, MEM_RELEASE);
#else
        std::free(region);
#endif
    }

    GuestRegion(const GuestRegion&) = delete;
    GuestRegion& operator=(const GuestRegion&) = delete;

    std::uint64_t Source(std::size_t copy) const { return Address(copy * CopyBytes); }
    std::uint64_t Destination(std::size_t copy) const { return Address((Copies + copy) * CopyBytes); }
    std::uint64_t Slot(std::size_t slot) const { return Address(2 * Copies * CopyBytes + slot * SlotBytes); }

    volatile std::uint32_t* Words(std::uint64_t address) const { return reinterpret_cast<volatile std::uint32_t*>(static_cast<std::uintptr_t>(address)); }

private:
    std::uint64_t Address(std::size_t offset) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(region + offset)); }

    std::uint8_t* region = nullptr;
};

std::uint32_t Low(std::uint64_t value) {
    return static_cast<std::uint32_t>(value);
}

std::uint32_t High(std::uint64_t value) {
    return static_cast<std::uint32_t>(value >> 32u);
}

void WriteData(std::vector<std::uint32_t>& words, std::uint64_t address, std::initializer_list<std::uint32_t> values) {
    words.push_back(0xc0003700u | ((2u + static_cast<std::uint32_t>(values.size())) << 16u));
    words.insert(words.end(), {0x00100200u, Low(address), High(address)});
    words.insert(words.end(), values);
}

void DmaCopy(std::vector<std::uint32_t>& words, std::uint64_t destination, std::uint64_t source, std::size_t bytes) {
    words.insert(words.end(), {0xc0055000u, 0u, Low(source), High(source), Low(destination), High(destination), static_cast<std::uint32_t>(bytes)});
}

struct WaitForm {
    const char* name;
    bool wide;
    std::uint32_t function;
    std::uint64_t initial;
    std::uint64_t written;
    std::uint64_t reference;
    std::uint64_t mask;
};

void Wait(std::vector<std::uint32_t>& words, std::uint64_t address, const WaitForm& form) {
    if (form.wide) {
        words.insert(words.end(), {0xc0079300u, 0x10u | form.function, Low(address), High(address), Low(form.reference), High(form.reference), Low(form.mask), High(form.mask), 0x19u});
    } else {
        words.insert(words.end(), {0xc0053c00u, 0x10u | form.function, Low(address), High(address), Low(form.reference), Low(form.mask), 0x19u});
    }
}

void Submit(std::uint32_t queue, const std::vector<std::uint32_t>& words) {
    Packet packet{const_cast<std::uint32_t*>(words.data()), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, queue);
}

void OpenDriverDevice() {
    std::vector<std::uint32_t> words;
    const auto registers = [&](std::uint32_t first, std::initializer_list<std::uint32_t> values) {
        words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
        words.push_back(first);
        words.insert(words.end(), values);
    };
    const auto program = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(FillKernel.data()));
    const auto output = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(WarmUpOutput.data()));
    registers(0x207, {64, 1, 1});
    registers(0x20c, {static_cast<std::uint32_t>(program >> 8u), static_cast<std::uint32_t>(program >> 40u)});
    registers(0x213, {(10u << 1u) | (1u << 7u)});
    registers(0x240, {Low(output), (High(output) & 0xffffu) | (4u << 16u), 256, 0x00014004u, 1, 2, 3, 4, 256, 4});
    words.insert(words.end(), {0xc0031500u, 4, 1, 1, 0x41u});
    Submit(0, words);
    AgcDriverWaitIdle_nid_postfix();
    Require(AgcDriver::Graphics::Recorder::Active() != nullptr, "queue label order: the driver opened no Vulkan device");
}

std::vector<std::uint32_t> SlowWrites(const GuestRegion& region, std::uint64_t target, std::size_t rounds = 1) {
    std::vector<std::uint32_t> words;
    for (std::size_t round = 0; round < rounds; ++round) {
        for (std::size_t copy = 0; copy < Copies; ++copy) DmaCopy(words, region.Destination(copy), region.Source(copy), CopyBytes);
    }
    WriteData(words, target, {Earlier, Earlier, Earlier, Earlier});
    return words;
}

std::size_t RecorderStreams() {
    const auto* recorder = AgcDriver::Graphics::Recorder::Active();
    return recorder != nullptr ? recorder->Streams() : 0;
}

std::vector<std::string>& Failures() {
    static std::vector<std::string> failures;
    return failures;
}

void RequireLater(const GuestRegion& region, std::uint64_t target, const std::string& what) {
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(target, 16);
    for (std::size_t index = 0; index < 4; ++index) {
        const std::uint32_t word = region.Words(target)[index];
        if (word == Later) continue;
        std::array<char, 320> message{};
        std::snprintf(message.data(), message.size(), "queue label order: %s: word %zu holds 0x%08x, not the later write's 0x%08x (%zu recorder streams)", what.c_str(), index, word, Later, RecorderStreams());
        Failures().emplace_back(message.data());
        return;
    }
}

void SetLabel(const GuestRegion& region, std::uint64_t label, std::uint64_t value) {
    region.Words(label)[0] = Low(value);
    region.Words(label)[1] = High(value);
}

void LabelWaitCase(const GuestRegion& region, std::size_t slot, const WaitForm& form, bool waiterFirst) {
    const auto label = region.Slot(slot);
    const auto target = label + 16;
    SetLabel(region, label, form.initial);
    auto writer = SlowWrites(region, target);
    if (form.wide) WriteData(writer, label, {Low(form.written), High(form.written)});
    else WriteData(writer, label, {Low(form.written)});
    std::vector<std::uint32_t> waiter;
    Wait(waiter, label, form);
    WriteData(waiter, target, {Later, Later, Later, Later});
    if (waiterFirst) {
        Submit(0, waiter);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Submit(WriterQueue, writer);
    } else {
        Submit(WriterQueue, writer);
        Submit(0, waiter);
    }
    RequireLater(region, target, std::string(form.name) + (waiterFirst ? ", wait submitted before its writer" : ", wait submitted after its writer"));
}

void ArrivalOrderCase(const GuestRegion& region, std::size_t slot) {
    const auto target = region.Slot(slot) + 16;
    Submit(0, SlowWrites(region, target));
    std::vector<std::uint32_t> later;
    WriteData(later, target, {Later, Later, Later, Later});
    Submit(WaitFreeQueue, later);
    RequireLater(region, target, "a wait-free submission of another queue after queue 0's earlier submission");
}

void LandingCase(const GuestRegion& region, std::size_t slot) {
    constexpr std::size_t Attempts = 4;
    constexpr std::size_t FirstRounds = 12;
    const auto label = region.Slot(slot);
    const auto condition = label + 16;
    const auto flag = label + 32;
    for (std::size_t attempt = 0; attempt < Attempts; ++attempt) {
        const std::uint64_t value = attempt + 1u;
        SetLabel(region, label, 0);
        region.Words(condition)[0] = 0;
        region.Words(flag)[0] = 0;
        auto writer = SlowWrites(region, label + 48, FirstRounds << attempt);
        WriteData(writer, label, {Low(value)});
        std::vector<std::uint32_t> waiter;
        Wait(waiter, label, {"landing", false, 3, 0, value, value, 0xffffffffu});
        waiter.insert(waiter.end(), {0xc0032200u, Low(condition), High(condition), 0u, 5u});
        WriteData(waiter, flag, {1});
        Submit(WriterQueue, waiter);
        Submit(0, writer);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        AgcDriver::GuestMemory::TagGpuLockThread(0x7f);
        while (AgcDriver::Graphics::Recorder::LookupLabelValue(label, 4) != std::optional<std::uint64_t>{value}) {
            Require(std::chrono::steady_clock::now() < deadline, "queue label order: the writer's label was never recorded");
            std::this_thread::yield();
        }
        AgcDriver::GuestMemory::TagGpuLockThread(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const bool landedEarly = region.Words(label)[0] == Low(value);
        region.Words(condition)[0] = 1;
        AgcDriverWaitIdle_nid_postfix();
        AgcDriver::GuestMemory::FlushGpuWrites(flag, 4);
        if (landedEarly) continue;
        if (region.Words(flag)[0] != 1) Failures().emplace_back("queue label order: a COND_EXEC behind a wait on another queue's recorded label read its condition before the label landed, and missed the CPU's store made after the submission (" + std::to_string(RecorderStreams()) + " recorder streams)");
        return;
    }
    Failures().emplace_back("queue label order: the landing case's writer finished before the CPU's store in all " + std::to_string(Attempts) + " attempts, up to " + std::to_string(FirstRounds << (Attempts - 1u)) + " rounds of copies, so it tests nothing");
}

void CpuValueCase(const GuestRegion& region, std::size_t slot) {
    const auto label = region.Slot(slot);
    const auto flag = label + 16;
    SetLabel(region, label, 0);
    region.Words(flag)[0] = 0;
    std::vector<std::uint32_t> waiter;
    Wait(waiter, label, {"cpu", false, 3, 0, 1, 1, 0xffffffffu});
    WriteData(waiter, flag, {1});
    Submit(WriterQueue, waiter);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Require(region.Words(flag)[0] == 0, "queue label order: a wait on a value no recorded work writes passed before the CPU stored it");
    region.Words(label)[0] = 1;
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(flag, 4);
    Require(region.Words(flag)[0] == 1, "queue label order: the work after a wait on a CPU-stored value did not run once the CPU stored it");
}

}

int main() {
    {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
    }
    GuestRegion region;
    try {
        OpenDriverDevice();
        const std::array<WaitForm, 8> forms{{
            {"less", false, 1, 9, 3, 5, 0xffffffffu},
            {"less or equal", false, 2, 9, 5, 5, 0xffffffffu},
            {"equal under a mask", false, 3, 0, 0x1201u, 0x01u, 0xffu},
            {"not equal", false, 4, 7, 8, 7, 0xffffffffu},
            {"greater or equal", false, 5, 0, 6, 5, 0xffffffffu},
            {"greater under a mask", false, 6, 0, 0x60u, 0x50u, 0xf0u},
            {"64-bit equal", true, 3, 0, 0x100000002ull, 0x100000002ull, ~std::uint64_t{0}},
            {"64-bit greater or equal in the upper word", true, 5, 0, 0x200000000ull, 0x100000000ull, 0xffffffff00000000ull},
        }};
        std::size_t slot = 0;
        for (const auto& form : forms) LabelWaitCase(region, slot++, form, false);
        LabelWaitCase(region, slot++, forms[2], true);
        LabelWaitCase(region, slot++, forms[6], true);
        ArrivalOrderCase(region, slot++);
        LandingCase(region, slot++);
        CpuValueCase(region, slot++);
        for (const auto& failure : Failures()) std::cerr << failure << '\n';
        Require(Failures().empty(), "queue label order: " + std::to_string(Failures().size()) + " cases failed");
        const auto streams = RecorderStreams();
        AgcDriverShutdown_nid_postfix();
        std::printf("queue label order tests passed (%zu recorder streams)\n", streams);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
