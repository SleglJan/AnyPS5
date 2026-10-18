#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CommittedImports.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
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
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

#ifdef _WIN32
namespace {

using AgcDriver::Graphics::Require;
using namespace AgcDriver::Graphics;

constexpr std::size_t Page = 65536;
constexpr std::size_t Granule = 16384;

std::uint64_t pageOf(const void* address) {
    return reinterpret_cast<std::uint64_t>(address) >> AgcDriver::GuestMemory::ResidencyPageShift;
}

std::uint8_t* commit(std::size_t bytes, DWORD protection = PAGE_READWRITE) {
    auto* memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, Page));
    GuestArena::GuestArenaCommit_nid_postfix(memory, bytes, protection, Granule);
    return memory;
}

bool fill(AgcDriver::VulkanDevice& device, const void* address, std::size_t bytes, std::uint32_t value) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    const std::array<std::uint32_t, 4> pattern{value, value, value, value};
    const bool filled = device.FillBuffer(reinterpret_cast<std::uint64_t>(address), bytes, pattern);
    device.WaitIdle();
    return filled;
}

bool holds(const std::uint8_t* memory, std::size_t bytes, std::uint32_t value) {
    for (std::size_t offset = 0; offset < bytes; offset += 4) {
        std::uint32_t word = 0;
        std::memcpy(&word, memory + offset, 4);
        if (word != value) return false;
    }
    return true;
}

void CheckCommitImported(AgcDriver::VulkanDevice& device) {
    const auto vk = device.Device();
    auto* memory = commit(4 * Page);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto first = ImportOf(vk, pageOf(memory));
    const auto last = ImportOf(vk, pageOf(memory + 3 * Page));
    Require(first && last && first->bytes == Page && last->bytes == Page && first->serial != 0, "a guest commit was not imported before the commit returned");
    Require(first->buffer == last->buffer && last->offset == 3 * Page && last->address == first->address + 3 * Page, "one commit was not one import");
    const auto committed = AgcDriver::GuestMemory::CollectWritesUncached(base, 4 * Page);
    Require(AgcDriver::GuestMemory::UnchangedSinceCollected(base, 4 * Page, committed), "the import's own marks were collected as CPU stores");
    memory[5000] = 1;
    Require(!AgcDriver::GuestMemory::UnchangedSinceCollected(base, 4 * Page, committed), "a CPU store into imported memory was not collected");
    Require(fill(device, memory, 4 * Page, 0x05050505u) && holds(memory, 4 * Page, 0x05050505u), "a GPU fill through the commit import did not reach guest memory");
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    const auto* import = HostImportFor(context, base + 64, 256);
    Require(import != nullptr && import->serial == first->serial && HostImportSerial(context, base + 64, 256, true) == first->serial, "the bind path did not serve the commit import");
    std::printf("commit import: ok\n");
}

void CheckPartialRelease(AgcDriver::VulkanDevice& device) {
    const auto vk = device.Device();
    auto* memory = commit(4 * Page);
    const auto before = ImportOf(vk, pageOf(memory));
    Require(before.has_value(), "the commit was not imported");
    GuestArena::GuestArenaReset_nid_postfix(memory + Page, Page);
    Require(!ImportOf(vk, pageOf(memory + Page)), "a released page kept its import");
    const auto kept = ImportOf(vk, pageOf(memory));
    const auto tail = ImportOf(vk, pageOf(memory + 3 * Page));
    Require(kept && tail && kept->bytes == Page && tail->bytes == Page && kept->serial != before->serial, "the rest of a partly released commit lost its import");
    GuestArena::GuestArenaCommit_nid_postfix(memory + Page, Page, PAGE_READWRITE, Granule);
    const auto again = ImportOf(vk, pageOf(memory + Page));
    Require(again && again->buffer != kept->buffer, "a page committed again was not imported again");
    Require(fill(device, memory + Page, Page, 0x0a0a0a0au) && holds(memory + Page, Page, 0x0a0a0a0au), "a GPU fill of a page committed again went to the released pages");
    Require(fill(device, memory, Page, 0x0b0b0b0bu) && holds(memory, Page, 0x0b0b0b0bu), "the kept part of the import does not reach guest memory");
    std::printf("partial release: ok\n");
}

void CheckSharedMapping(AgcDriver::VulkanDevice& device) {
    const auto vk = device.Device();
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(2 * Page), nullptr);
    Require(section != nullptr, "cannot create a section");
    auto* first = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(2 * Page, Page));
    auto* second = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(2 * Page, Page));
    GuestArena::GuestArenaMap_nid_postfix(first, 2 * Page, section, 0, PAGE_READWRITE);
    GuestArena::GuestArenaMap_nid_postfix(second, 2 * Page, section, 0, PAGE_READWRITE);
    const auto import = ImportOf(vk, pageOf(first));
    Require(import && import->bytes == Page && ImportOf(vk, pageOf(second)), "a shared mapping was not imported at its mapping");
    Require(fill(device, first + Page, Page, 0x0c0c0c0cu) && holds(second + Page, Page, 0x0c0c0c0cu), "a GPU fill through the shared import did not reach the section");
    std::printf("shared mapping import: ok\n");
}

void CheckUnimportable(AgcDriver::VulkanDevice& device) {
    auto* readOnly = commit(Page, PAGE_READONLY);
    Require(!ImportOf(device.Device(), pageOf(readOnly)), "a read-only commit was imported");
    std::printf("read-only commit: ok\n");
}

void CheckBudget(AgcDriver::VulkanDevice& device) {
    SetCommittedImportBudget(CommittedImportPinnedBytes() + Page);
    auto* memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(2 * Page, Page));
    std::string error;
    try {
        GuestArena::GuestArenaCommit_nid_postfix(memory, 2 * Page, PAGE_READWRITE, Granule);
    } catch (const std::runtime_error& refused) {
        error = refused.what();
    }
    SetCommittedImportBudget(0);
    Require(error.find("refused") != std::string::npos && error.find("> budget") != std::string::npos && error.find("physical") != std::string::npos, "an import past the budget did not throw with its numbers: " + error);
    Require(!ImportOf(device.Device(), pageOf(memory)), "a refused import was made");
    std::printf("budget refusal: ok (%s)\n", error.c_str());
}

void CheckAttach(AgcDriver::VulkanDevice& device) {
    const auto vk = device.Device();
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    DetachCommittedImports(vk);
    auto* memory = commit(Page);
    Require(!ImportOf(vk, pageOf(memory)), "a commit was imported while no device was attached");
    AttachCommittedImports(context, PrepareImportWatch(context) == ImportWatch::Watch);
    Require(ImportOf(vk, pageOf(memory)).has_value(), "attaching a device did not import memory committed before it");
    Require(fill(device, memory, Page, 0x0d0d0d0du) && holds(memory, Page, 0x0d0d0d0du), "the attach import does not reach guest memory");
    std::printf("attach: ok\n");
}

std::vector<std::uint64_t> guardReports;

void noteGuardReport(std::span<const std::uint64_t> pages) {
    guardReports.insert(guardReports.end(), pages.begin(), pages.end());
}

void CheckGuardOverImport(AgcDriver::VulkanDevice& device) {
    static const AgcDriver::GuestMemory::ResidencyHooks hooks{[](std::uint64_t, std::size_t) { return false; }, &noteGuardReport, nullptr, nullptr};
    AgcDriver::GuestMemory::SetResidencyHooks(&hooks);
    const auto guardedRounds = [&](std::uint8_t* memory, const char* kind) {
        const auto base = reinterpret_cast<std::uint64_t>(memory);
        for (std::uint32_t round = 0; round < 64; ++round) {
            guardReports.clear();
            Require(ImportOf(device.Device(), pageOf(memory)) && AgcDriver::GuestMemory::GuardWrites(base, Page) == GuestArena::GuardRefusal::None, std::string("an imported ") + kind + " page could not be guarded");
            Require(fill(device, memory, Page, 0x0e0e0e0eu + round) && holds(memory, Page, 0x0e0e0e0eu + round), std::string("a GPU fill through the import of a guarded ") + kind + " page did not land");
            Require(guardReports.empty() && AgcDriver::GuestMemory::WriteGuarded(pageOf(memory)), std::string("a GPU write tripped the CPU write guard of a ") + kind + " page");
            memory[8 + round] = 1;
            Require(guardReports == std::vector<std::uint64_t>{pageOf(memory)} && memory[8 + round] == 1, std::string("a CPU store to a guarded imported ") + kind + " page was not reported");
        }
    };
    auto* privateMemory = commit(Page);
    const auto privateBase = reinterpret_cast<std::uint64_t>(privateMemory);
    Require(ImportOf(device.Device(), pageOf(privateMemory)).has_value(), "a private commit was not imported");
    Require(AgcDriver::GuestMemory::GuardWrites(privateBase, Page) == GuestArena::GuardRefusal::UnsettledImport && AgcDriver::GuestMemory::GuardAccess(privateBase, Page) == GuestArena::GuardRefusal::UnsettledImport, "an imported private page was guarded before its import settled: the GPU would lose the device on its read-only pages");
    Require(!AgcDriver::GuestMemory::WriteGuarded(pageOf(privateMemory)) && !AgcDriver::GuestMemory::AccessGuarded(pageOf(privateMemory)), "a refused guard armed the private page");
    const auto stamp = AgcDriver::GuestMemory::HostImportStamp();
    Require(fill(device, privateMemory, Page, 0x0f0f0f0fu) && holds(privateMemory, Page, 0x0f0f0f0fu), "a GPU fill through a fresh private import did not land");
    AgcDriver::GuestMemory::SettleHostImports(privateBase, Page, stamp);
    guardedRounds(privateMemory, "private");
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(Page), nullptr);
    Require(section != nullptr, "cannot create a section");
    auto* memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(Page, Page));
    GuestArena::GuestArenaMap_nid_postfix(memory, Page, section, 0, PAGE_READWRITE);
    guardedRounds(memory, "shared");
    AgcDriver::GuestMemory::SetResidencyHooks(nullptr);
    std::printf("guard over an import: ok\n");
}

void CheckCpuWritable(AgcDriver::VulkanDevice& device) {
    const auto vk = device.Device();
    auto* memory = commit(3 * Page);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(memory, Page, false, false, true);
        mutation.Add(memory + Page, Page, true, true, true);
        mutation.Add(memory + 2 * Page, Page / 2, true, false, true);
    }
    const auto gpuOnly = ImportOf(vk, pageOf(memory));
    const auto writable = ImportOf(vk, pageOf(memory + Page));
    const auto partly = ImportOf(vk, pageOf(memory + 2 * Page));
    Require(gpuOnly && !gpuOnly->cpuWritable, "a page mapped without CPU write was reported CPU-writable");
    Require(writable && writable->cpuWritable, "a CPU-writable page was reported unwritable");
    Require(partly && partly->cpuWritable, "a page with unregistered committed bytes was reported unwritable");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(memory);
        mutation.Remove(memory + Page);
        mutation.Remove(memory + 2 * Page);
    }
    std::printf("cpu-writable attribute: ok\n");
}

}
#endif

int main() {
#ifdef _WIN32
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (!ImportOf(device->Device(), pageOf(commit(Page)))) {
            std::printf("skipped, the device has no host imports\n");
            return VulkanTestSkipped;
        }
        CheckCommitImported(*device);
        CheckPartialRelease(*device);
        CheckSharedMapping(*device);
        CheckUnimportable(*device);
        CheckBudget(*device);
        CheckAttach(*device);
        CheckGuardOverImport(*device);
        CheckCpuWritable(*device);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "committed import test failed: %s\n", error.what());
        return 1;
    }
    std::printf("committed import tests passed\n");
    return 0;
#else
    std::printf("skipped, commit imports are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
