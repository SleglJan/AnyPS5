#include "NewPathFixture.hpp"
#include "VulkanTestDevice.hpp"
#include <exception>

#ifdef _WIN32

namespace {

using namespace NewPath;
using AgcDriver::Graphics::ResidencyAccess;
using AgcDriver::Graphics::ResidencyCounts;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

ResidencyCounts Counts() {
    ResidencyCounts counts;
    Locked([&] { counts = Residency().Manager().Counts(); });
    return counts;
}

bool Resident(std::uint64_t address) {
    bool resident = false;
    Locked([&] { resident = Residency().Manager().Slot(PageOf(address)).has_value(); });
    return resident;
}

bool Guarded(std::uint64_t address) {
    return AgcDriver::GuestMemory::WriteGuarded(PageOf(address));
}

bool Hot(std::uint64_t address) {
    bool hot = false;
    Locked([&] { hot = Residency().Manager().Hot(PageOf(address)) || Residency().Manager().HotResident(PageOf(address)); });
    return hot;
}

void Use(std::uint64_t address) {
    Locked([&] { Residency().Manager().Use(address, PageBytes, ResidencyAccess::Read); });
}

void OrderingPoint() {
    Locked([] {
        auto& manager = Residency().Manager();
        static_cast<void>(manager.Advance());
        manager.Apply();
    });
}

void Collect(std::uint64_t address) {
    AgcDriver::GuestMemory::CollectWritesUncached(address, PageBytes);
}

void Fill(volatile std::uint32_t* words, std::uint32_t tag) {
    for (std::uint32_t word = 0; word < WordsPerPage; ++word) words[word] = (tag << 24u) | word;
}

std::string Faults(const ResidencyCounts& before, const ResidencyCounts& after) {
    return std::to_string(after.faults - before.faults);
}

void StoreBeforeAdmission() {
    SharedPages pages(1);
    Fill(pages.Words(), 1);
    Use(pages.Address());
    Expect(Resident(pages.Address()) && Guarded(pages.Address()), "store before admission: the page was not admitted with a write guard");
    const auto before = Counts();
    Collect(pages.Address());
    OrderingPoint();
    const auto after = Counts();
    Expect(after.faults == before.faults, "store before admission: CPU stores made before the page was admitted counted " + Faults(before, after) + " guard faults once the write watch was collected, expected none");
    Expect(Resident(pages.Address()) && Guarded(pages.Address()) && !Hot(pages.Address()), "store before admission: the page left residency, lost its guard or turned hot");
}

void OneStoreOneFault() {
    SharedPages pages(1);
    Fill(pages.Words(), 2);
    Collect(pages.Address());
    Use(pages.Address());
    Expect(Resident(pages.Address()) && Guarded(pages.Address()), "one store: the page was not admitted with a write guard");
    const auto before = Counts();
    pages.Words()[5] = 0x2a000005u;
    OrderingPoint();
    Collect(pages.Address());
    OrderingPoint();
    const auto after = Counts();
    Expect(after.faults == before.faults + 1u, "one store: one CPU store to a guarded resident page counted " + Faults(before, after) + " guard faults when its write-watch report came after an ordering point, expected one");
    Expect(!Hot(pages.Address()) && Resident(pages.Address()) && Guarded(pages.Address()), "one store: one CPU store made the page hot, or it left residency or its guard");
}

void HotAcrossRemap() {
    SharedPages pages(1);
    Fill(pages.Words(), 3);
    Collect(pages.Address());
    Use(pages.Address());
    for (std::uint32_t store = 0; store < 2u; ++store) {
        Locked([] { Residency().Manager().NextFrame(); });
        pages.Words()[7] = 0x3b000000u + store;
        Collect(pages.Address());
        OrderingPoint();
    }
    Expect(Hot(pages.Address()), "remap: two CPU stores in two frames did not make the page hot");
    pages.Unmap();
    pages.Map();
    OrderingPoint();
    Expect(!Hot(pages.Address()), "remap: an unmap and a new mapping at the same address kept the old memory's hot state");
    Fill(pages.Words(), 4);
    Collect(pages.Address());
    Use(pages.Address());
    Expect(Resident(pages.Address()), "remap: a new mapping at an address that was hot before the unmap was not admitted");
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
        MirrorReader reader;
        {
            PrivatePages warm(1);
            reader.Read(warm.Address(), 1);
        }
        Locked([] { Residency().Manager().SetBudget(std::uint64_t{1} << 30); });
        StoreBeforeAdmission();
        OneStoreOneFault();
        HotAcrossRemap();
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception: %s\n", error.what());
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d new path residency fault checks failed\n", failures);
        return 1;
    }
    std::printf("new path residency fault tests passed\n");
    return 0;
#else
    std::printf("skipped, residency guards are Windows-only\n");
    return VulkanTestSkipped;
#endif
}
