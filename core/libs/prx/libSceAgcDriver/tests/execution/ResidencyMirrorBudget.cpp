#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyDriver.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

#ifdef _WIN32

namespace {

using namespace AgcDriver::Graphics;

constexpr std::uint32_t ChunkSlots = 16;
constexpr std::uint64_t ChunkBytes = std::uint64_t{ChunkSlots} << ResidencyPageShift;
constexpr std::uint32_t GuestPages = 16 * ChunkSlots;
constexpr std::uint64_t GuestChunks = 8;
constexpr std::uint64_t FirstOther = 2 * ChunkBytes;
constexpr std::uint64_t LaterOther = 3 * ChunkBytes;
constexpr std::uint64_t Overflow = 9 * ChunkBytes;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::string MiB(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f MiB", static_cast<double>(bytes) / static_cast<double>(1u << 20u));
    return text;
}

class GuestRegion {
public:
    GuestRegion() : bytes(std::size_t{GuestPages} << ResidencyPageShift) {
        region = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes + ResidencyPageBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        Require(region != nullptr, "mirror budget: cannot allocate the guest region");
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

class DeviceLocal {
public:
    DeviceLocal(const Context& context, std::uint64_t bytes) : context(context) {
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = bytes;
        allocation.memoryTypeIndex = context.MemoryType(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory other device-local memory");
    }

    ~DeviceLocal() { Free(); }

    DeviceLocal(const DeviceLocal&) = delete;
    DeviceLocal& operator=(const DeviceLocal&) = delete;

    void Free() {
        if (memory == VK_NULL_HANDLE) return;
        context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }

private:
    const Context& context;
    VkDeviceMemory memory = VK_NULL_HANDLE;
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

    std::uint64_t Mirrors() const { return driver.Mirrors().DeviceBytes(); }

    void ExpectWithin(std::uint64_t others, const std::string& what) {
        const auto guestBytes = guest.guestBytes;
        const auto budget = guestBytes > others ? guestBytes - others : 0;
        const auto mirrors = Mirrors();
        const auto measured = driver.OtherDeviceBytes();
        Expect(measured == others, what + ": the driver counts " + MiB(measured) + " of other device-local memory, expected " + MiB(others));
        Expect(mirrors + measured <= std::max(guestBytes, measured), what + ": mirrors " + MiB(mirrors) + " and other device-local memory " + MiB(measured) + " exceed the guest's " + MiB(guestBytes));
        Expect(mirrors <= budget, what + ": mirrors hold " + MiB(mirrors) + ", above the " + MiB(budget) + " the guest size leaves beside " + MiB(others) + " of other device-local memory");
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

void BudgetIsTheRemainder(Harness& harness, const Context& context) {
    const auto base = harness.driver.OtherDeviceBytes();
    harness.guest.guestBytes = base + GuestChunks * ChunkBytes;
    {
        const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        harness.driver.Manager().SetBudget(ResidencyMeasuredBudget);
    }
    DeviceLocal first(context, FirstOther);
    harness.Fill();
    harness.ExpectFilled(base + FirstOther, "with " + MiB(FirstOther) + " of other device-local memory");

    std::optional<DeviceLocal> later;
    later.emplace(context, LaterOther);
    const auto evictedBefore = harness.driver.Manager().Counts().evicted;
    harness.Prepare();
    Expect(harness.driver.Manager().Counts().evicted > evictedBefore, "a later allocation of " + MiB(LaterOther) + " evicted no mirror at the next admission point");
    harness.Prepare();
    harness.ExpectWithin(base + FirstOther + LaterOther, "after a later allocation of " + MiB(LaterOther) + ", before any frame boundary");
    harness.Fill();
    harness.ExpectFilled(base + FirstOther + LaterOther, "with both allocations");

    later.reset();
    harness.Fill();
    harness.ExpectFilled(base + FirstOther, "after freeing the later allocation");

    {
        DeviceLocal overflow(context, Overflow);
        harness.Prepare();
        harness.Prepare();
        harness.ExpectWithin(base + FirstOther + Overflow, "with other device-local memory above the guest size");
        harness.Fill();
        Expect(harness.Mirrors() == 0 && harness.driver.Manager().Counts().chunks == 0, "other device-local memory above the guest size left " + MiB(harness.Mirrors()) + " of mirrors");
    }
    harness.Fill();
    harness.ExpectFilled(base + FirstOther, "after freeing the memory above the guest size");
}

}

int main() {
    try {
        GuestRegion region;
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Recorder* recorder = nullptr;
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            recorder = Recorder::Active();
        }
        Require(recorder != nullptr, "mirror budget: the device has no recorder");
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        {
            const std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            if (HostImportFor(context, region.Address(), region.Bytes()) == nullptr) {
                std::printf("skipped, the device does not import guest memory\n");
                return VulkanTestSkipped;
            }
        }
        {
            Harness harness(*device, context, *recorder, region);
            BudgetIsTheRemainder(harness, context);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d mirror budget checks failed\n", failures);
        return 1;
    }
    std::printf("mirror budget tests passed\n");
    return 0;
}

#else

int main() {
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
}

#endif
