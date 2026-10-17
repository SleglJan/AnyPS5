#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_RESIDENCYBUDGETHARNESS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_RESIDENCYBUDGETHARNESS_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace ResidencyBudgetTest {

using namespace AgcDriver::Graphics;

constexpr std::uint32_t ChunkSlots = 16;
constexpr std::uint64_t ChunkBytes = std::uint64_t{ChunkSlots} << ResidencyPageShift;
constexpr std::uint32_t GuestPages = 16 * ChunkSlots;
constexpr std::uint64_t GuestChunks = 8;

inline int failures = 0;

inline void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

inline std::string MiB(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f MiB", static_cast<double>(bytes) / static_cast<double>(1u << 20u));
    return text;
}

inline std::uint32_t DeviceHeap(const Context& context) {
    for (std::uint32_t heap = 0; heap < context.memory.memoryHeapCount; ++heap) {
        if ((context.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) return heap;
    }
    return 0;
}

inline std::uint64_t HeapUsage(const Context& context) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    std::uint64_t used = 0;
    for (std::uint32_t heap = 0; heap < context.memory.memoryHeapCount; ++heap) {
        if ((context.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) used += reported.heapUsage[heap];
    }
    return used;
}

class GuestRegion {
public:
    GuestRegion() : bytes(std::size_t{GuestPages} << ResidencyPageShift) {
        region = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes + ResidencyPageBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        Require(region != nullptr, "residency budget: cannot allocate the guest region");
        base = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(region) + ResidencyPageBytes - 1) & ~(ResidencyPageBytes - 1));
        auto* words = reinterpret_cast<std::uint32_t*>(base);
        for (std::size_t word = 0; word < bytes / 4; ++word) words[word] = static_cast<std::uint32_t>(word);
        GuestAllocations::Mutation().Add(base, bytes, true, true, true);
    }

    ~GuestRegion() {
        GuestAllocations::Mutation().Remove(base);
        VirtualFree(region, 0, MEM_RELEASE);
    }

    GuestRegion(const GuestRegion&) = delete;
    GuestRegion& operator=(const GuestRegion&) = delete;

    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(base); }
    std::uint64_t First() const { return Address() >> ResidencyPageShift; }
    std::uint64_t Bytes() const { return bytes; }

private:
    std::uint8_t* region = nullptr;
    std::uint8_t* base = nullptr;
    std::size_t bytes = 0;
};

class FixedGuest final : public IResidencyGuestMemory {
public:
    FixedGuest(const Context& context, const GuestRegion& region) : context(context), region(region) {}

    std::optional<ResidencyGuestImport> ImportOf(std::uint64_t page) override {
        if (page < region.First() || page >= region.First() + GuestPages) return std::nullopt;
        const auto address = page << ResidencyPageShift;
        const auto* import = HostImportFor(context, address, ResidencyPageBytes);
        if (import == nullptr) return std::nullopt;
        const bool whole = import->base <= address && address + ResidencyPageBytes <= import->base + import->bytes;
        ResidencyImport attributes{whole, true, true};
        attributes.section = page + 1;
        return ResidencyGuestImport{attributes, import->buffer, address - import->base, import->address + (address - import->base)};
    }

    std::vector<ResidencyRange> Aliases(std::uint64_t, std::uint64_t) override { return {}; }
    std::uint64_t Generation() override { return 1; }

    void ChangedPages(std::uint64_t, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed) override {
        for (std::size_t k = 0; k < generations.size(); ++k) changed[k] = generations[k] == 0 ? 1 : 0;
    }

    bool Guard(std::uint64_t) override { return true; }
    void Unguard(std::span<const std::uint64_t>) override {}
    bool GuardAccess(std::uint64_t, ResidencyMask) override { return true; }
    void UnguardAccess(std::uint64_t) override {}

    std::vector<ResidencyRange> WholeImports(std::uint64_t& generation) override {
        if (generation == 1) return {};
        generation = 1;
        return {ResidencyRange{region.Address(), region.Bytes()}};
    }

    std::uint64_t GuestBytes() override { return guestBytes; }

    std::uint64_t guestBytes = 0;

private:
    const Context& context;
    const GuestRegion& region;
};

struct Harness {
    Harness(AgcDriver::VulkanDevice& device, const Context& context, Recorder& recorder, const GuestRegion& region)
        : device(device), guest(context, region), driver(context, recorder, guest, ResidencyPolicy{ChunkSlots, ResidencyMeasuredBudget, 2, 2}) {}

    ~Harness() { Drain(); }

    void Drain() {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device.SubmitRecorded(false);
        device.WaitIdle();
    }

    void Prepare() {
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            driver.Manager().Prepare();
        }
        Drain();
    }

    void FrameBoundary() {
        driver.Flip();
        driver.BeforeDispatch();
        Drain();
    }

    void Fill() {
        std::uint64_t resident = ~std::uint64_t{0};
        for (int round = 0; round < 64; ++round) {
            Prepare();
            const auto now = driver.Manager().Counts().resident;
            if (now == resident) break;
            resident = now;
        }
        Prepare();
    }

    void Start() {
        guest.guestBytes = driver.OtherDeviceBytes() + GuestChunks * ChunkBytes;
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            driver.Manager().SetBudget(ResidencyMeasuredBudget);
        }
        FrameBoundary();
        Fill();
    }

    std::uint64_t Mirrors() const { return driver.Mirrors().DeviceBytes(); }

    std::uint64_t Others() {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        return driver.OtherDeviceBytes();
    }

    void ExpectWithin(std::uint64_t others, const std::string& what) {
        const auto guestBytes = guest.guestBytes;
        const auto budget = guestBytes > others ? guestBytes - others : 0;
        const auto mirrors = Mirrors();
        Expect(mirrors <= budget, what + ": mirrors hold " + MiB(mirrors) + ", above the " + MiB(budget) + " the guest size leaves beside " + MiB(others) + " of other device-local usage");
    }

    void ExpectFilled(std::uint64_t others, const std::string& what) {
        ExpectWithin(others, what);
        const auto guestBytes = guest.guestBytes;
        const auto budget = guestBytes > others ? guestBytes - others : 0;
        const auto mirrors = Mirrors();
        Expect(mirrors + 2 * ChunkBytes >= budget, what + ": eager admission stopped at " + MiB(mirrors) + " of mirrors, more than two chunks below the " + MiB(budget) + " budget");
    }

    AgcDriver::VulkanDevice& device;
    FixedGuest guest;
    ResidencyDriver driver;
};

template<typename TBody>
int Run(const char* name, TBody&& body) {
    try {
        GuestRegion region;
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Recorder* recorder = nullptr;
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            recorder = Recorder::Active();
        }
        Require(recorder != nullptr, "residency budget: the device has no recorder");
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            if (HostImportFor(context, region.Address(), region.Bytes()) == nullptr) {
                std::printf("skipped, the device does not import guest memory\n");
                return VulkanTestSkipped;
            }
        }
        if (context.memoryProperties2 == nullptr) {
            std::printf("skipped, the device reports no memory budget\n");
            return VulkanTestSkipped;
        }
        body(*device, context, *recorder, region);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d %s checks failed\n", failures, name);
        return 1;
    }
    std::printf("%s tests passed\n", name);
    return 0;
}

}

#endif
