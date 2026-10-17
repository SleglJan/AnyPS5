#include "ResidencyRulesFixture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <memory>

#ifdef _WIN32

namespace {

using namespace ResidencyRules;
using AgcDriver::Graphics::ResidencyMeasuredBudget;

constexpr std::uint32_t GuestChunks = 2;
constexpr std::uint64_t Tolerance = std::uint64_t{16} << 20u;
constexpr std::uint64_t OtherBytes = std::uint64_t{1} << 30u;

struct Measured {
    std::uint64_t budget = 0;
    std::uint64_t expected = 0;
    std::uint64_t guest = 0;
    std::uint64_t reported = 0;
    std::uint64_t room = 0;
    bool full = false;
};

Measured Measure(const AgcDriver::Graphics::Context& context) {
    Measured measured;
    Locked([&] {
        auto& residency = Residency();
        residency.Manager().SetBudget(ResidencyMeasuredBudget);
        measured.budget = residency.Manager().BudgetBytes();
        measured.full = residency.Manager().FullResidency();
        measured.guest = residency.GuestBytes();
        measured.reported = residency.Counts().budgetBytes;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        properties.pNext = &reported;
        context.memoryProperties2(context.physical, &properties);
        std::optional<std::uint32_t> heap;
        for (std::uint32_t index = 0; index < properties.memoryProperties.memoryHeapCount; ++index) {
            if ((properties.memoryProperties.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
            if (!heap || properties.memoryProperties.memoryHeaps[index].size > properties.memoryProperties.memoryHeaps[*heap].size) heap = index;
        }
        Require(heap.has_value(), "budget: the device has no device-local heap");
        const auto mirrors = residency.Mirrors().DeviceBytes();
        const auto used = reported.heapUsage[*heap];
        const auto others = used > mirrors ? used - mirrors : 0;
        measured.room = ~std::uint64_t{0};
        if (reported.heapBudget[*heap] != 0) measured.room = reported.heapBudget[*heap] > others ? reported.heapBudget[*heap] - others : 0;
        measured.expected = std::min(measured.guest, measured.room);
    });
    return measured;
}

void ExpectBudget(const Measured& measured, const std::string& what) {
    const auto difference = measured.budget > measured.expected ? measured.budget - measured.expected : measured.expected - measured.budget;
    Expect(difference <= Tolerance, what + ": the residency budget is " + std::to_string(measured.budget) + " bytes, expected the guest's " + std::to_string(measured.guest) + " bytes, capped only by the heap budget left beside the other device-local memory, " + std::to_string(measured.expected) + " bytes");
    Expect(measured.budget <= measured.guest, what + ": the budget of " + std::to_string(measured.budget) + " bytes exceeds the guest's " + std::to_string(measured.guest) + " bytes");
    Expect(measured.reported == measured.budget, what + ": the driver reports a budget of " + std::to_string(measured.reported) + " bytes, the manager uses " + std::to_string(measured.budget));
    Expect(measured.full == (measured.budget >= measured.guest), what + ": full residency is " + (measured.full ? "on" : "off") + " with a budget of " + std::to_string(measured.budget) + " bytes for " + std::to_string(measured.guest) + " guest bytes");
}

void BudgetIsTheGuestSize(MirrorReader& reader) {
    static_cast<void>(reader);
    std::uint64_t chunkBytes = 0;
    Locked([&] { chunkBytes = std::uint64_t{Residency().Manager().ChunkSlots()} * PageBytes; });
    const auto guestPages = static_cast<std::uint32_t>(GuestChunks * chunkBytes / PageBytes);
    SharedPages base(guestPages);
    Fill(base.Words(), 1, 12);
    AgcDriver::GuestMemory::CollectWritesUncached(base.Address(), PageBytes);
    Locked([&] { Residency().Manager().SetBudget(std::uint64_t{GuestChunks} * chunkBytes); });
    Use(base.Address());
    std::optional<AgcDriver::Graphics::Context> context;
    Locked([&] {
        auto& residency = Residency();
        const auto chunks = residency.Manager().Chunks();
        if (!chunks.empty()) {
            if (const auto chunk = residency.Mirrors().Chunk(chunks.front())) context = chunk->context;
        }
    });
    Expect(context.has_value(), "a budget of " + std::to_string(GuestChunks) + " chunks admitted no page of a " + std::to_string(GuestChunks) + "-chunk guest allocation, so no mirror chunk exists");
    if (!context) return;
    const auto first = Measure(*context);
    ExpectBudget(first, "with a " + std::to_string(GuestChunks) + "-chunk shared allocation");
    {
        SharedPages more(guestPages);
        const auto grown = Measure(*context);
        Expect(grown.guest == first.guest + more.Bytes(), "mapping " + std::to_string(more.Bytes()) + " more shared bytes took the guest bytes from " + std::to_string(first.guest) + " to " + std::to_string(grown.guest));
        ExpectBudget(grown, "after mapping " + std::to_string(GuestChunks) + " more chunks of shared memory");
    }
    const auto shrunk = Measure(*context);
    Expect(shrunk.guest == first.guest, "unmapping the second allocation left the guest bytes at " + std::to_string(shrunk.guest) + ", expected " + std::to_string(first.guest));
    ExpectBudget(shrunk, "after unmapping the second allocation");
    {
        AgcDriver::Graphics::Buffer other(*context, OtherBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const auto beside = Measure(*context);
        ExpectBudget(beside, "with " + std::to_string(OtherBytes) + " bytes of other device-local memory");
        if (beside.room >= beside.guest) Expect(beside.budget == beside.guest, "a device-local allocation of " + std::to_string(OtherBytes) + " bytes outside the mirrors took the mirror budget to " + std::to_string(beside.budget) + " bytes, below the guest's " + std::to_string(beside.guest) + " bytes");
    }
    std::uint32_t slots = 0;
    bool full = true;
    Locked([&] {
        auto& manager = Residency().Manager();
        slots = manager.ChunkSlots();
        manager.SetBudget(chunkBytes);
        full = manager.FullResidency();
    });
    Expect(!full, "a budget of one mirror chunk covers the guest, so eager admission is not tested below the guest size");
    for (int round = 0; round < 8; ++round) {
        Settle();
        Locked([&] { Residency().Manager().Prepare(); });
    }
    AgcDriver::Graphics::ResidencyCounts counts{};
    Locked([&] { counts = Residency().Manager().Counts(); });
    Expect(counts.chunks == 1 && counts.resident == slots, "eager admission under a one-chunk budget below the guest size left " + std::to_string(counts.resident) + " of " + std::to_string(slots) + " slots resident in " + std::to_string(counts.chunks) + " chunks, expected the budget filled");
    Locked([&] { Residency().Manager().SetBudget(ResidencyMeasuredBudget); });
    Settle();
}

}

#endif

int main() {
#ifdef _WIN32
    return ResidencyRules::Main("residency guest-size budget", BudgetIsTheGuestSize);
#else
    std::printf("skipped, residency is Windows-only\n");
    return VulkanTestSkipped;
#endif
}
