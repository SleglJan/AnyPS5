#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

#ifdef _WIN32

namespace {

using AgcDriver::Graphics::BdaPageTable;
using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;
namespace Abi = ShaderRecompiler::BdaAbi;
namespace Graphics = AgcDriver::Graphics;

constexpr std::uint64_t Unit = 65536;
constexpr std::uint32_t Units = 10;
constexpr std::uint64_t FakeDevice = 0x6000000000ull;
constexpr std::uint64_t FakeStride = 0x1000000ull;
constexpr std::uint32_t Threads = 32;
constexpr std::uint8_t ImportByte = 0x11;

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

struct Translation {
    bool mapped = false;
    std::uint64_t device = 0;
    std::uint32_t permissions = 0;
    bool operator==(const Translation&) const = default;
};

Translation Through(const std::optional<Abi::Range>& range, std::uint64_t address) {
    if (!range) return {};
    return {true, range->deviceAddress + (address - range->begin), range->permissions};
}

Translation Spliced(const BdaPageTable& base, std::span<const Abi::Range> overrides, std::uint64_t address) {
    const auto found = std::upper_bound(overrides.begin(), overrides.end(), address, [](std::uint64_t value, const Abi::Range& range) { return value < range.begin; });
    if (found != overrides.begin() && address < std::prev(found)->end) return Through(*std::prev(found), address);
    return Through(base.Find(address), address);
}

std::string Describe(const Translation& translation) {
    return translation.mapped ? Hex(translation.device) + " permissions " + std::to_string(translation.permissions) : "unmapped";
}

void ExpectTranslations(const BdaPageTable& table, const BdaPageTable& base, std::span<const Abi::Range> overrides, std::span<const std::uint64_t> probes, const std::string& name) {
    std::size_t wrong = 0;
    std::string first;
    for (const auto address : probes) {
        const auto expected = Spliced(base, overrides, address);
        const auto actual = Through(table.Find(address), address);
        if (actual == expected) continue;
        if (wrong++ == 0u) first = Hex(address) + " translated to " + Describe(actual) + ", expected " + Describe(expected);
    }
    Expect(wrong == 0u, name + ": " + std::to_string(wrong) + " of " + std::to_string(probes.size()) + " probed addresses translated wrongly, first " + first);
}

void AddBoundaries(std::vector<std::uint64_t>& probes, std::span<const Abi::Range> ranges) {
    for (const auto& range : ranges) {
        for (const auto address : {range.begin - 1u, range.begin, range.end - 1u, range.end}) probes.push_back(address);
    }
}

Abi::Range Override(std::uint64_t begin, std::uint64_t end, std::uint32_t index, std::uint32_t permissions) {
    return {begin, end, FakeDevice + index * FakeStride, permissions, 0u};
}

std::vector<Translation> Translations(const BdaPageTable& table, std::span<const std::uint64_t> probes) {
    std::vector<Translation> translations;
    translations.reserve(probes.size());
    for (const auto address : probes) translations.push_back(Through(table.Find(address), address));
    return translations;
}

std::vector<VkDeviceAddress> Leaves(const BdaPageTable& table) {
    std::vector<VkDeviceAddress> leaves(Abi::WindowCount);
    for (std::uint32_t window = 0; window < Abi::WindowCount; ++window) leaves[window] = table.LeafAddress(window);
    return leaves;
}

class Reservation {
public:
    Reservation() {
        base = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(Units * Unit, Unit));
        Require(base != nullptr, "override page tables: cannot reserve guest units");
        GuestArena::GuestArenaCommit_nid_postfix(base, Units * Unit, PAGE_READWRITE, 16384);
        std::memset(base, ImportByte, Units * Unit);
    }

    ~Reservation() {
        for (const auto unit : registered) GuestAllocations::Mutation().Remove(base + unit * Unit);
        GuestArena::GuestArenaReset_nid_postfix(base, Units * Unit);
        GuestArena::GuestArenaRelease_nid_postfix(base, Units * Unit);
    }

    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;

    void Register(std::uint32_t unit, std::uint32_t units, bool writable) {
        GuestAllocations::Mutation().Add(base + unit * Unit, units * Unit, true, writable, true);
        registered.push_back(unit);
    }

    std::uint64_t Address(std::uint32_t unit, std::uint64_t offset = 0) const { return reinterpret_cast<std::uint64_t>(base) + unit * Unit + offset; }

private:
    std::uint8_t* base = nullptr;
    std::vector<std::uint32_t> registered;
};

std::optional<Abi::Range> Resident(const BdaPageTable& base, std::uint64_t address, std::uint64_t end, const std::string& name) {
    const auto found = base.Find(address);
    Expect(found.has_value() && found->begin == address && found->end == end, "override page tables: the base table does not hold " + name + " as one range from " + Hex(address) + " to " + Hex(end) + (found ? " (it holds " + Hex(found->begin) + " to " + Hex(found->end) + ")" : " (unmapped)"));
    return found;
}

void SplicesAndSharing(const Graphics::Context& context) {
    Reservation units;
    units.Register(0, 2, true);
    units.Register(3, 1, true);
    units.Register(4, 1, false);
    units.Register(6, 2, true);
    units.Register(9, 1, true);
    std::unique_lock gpu(AgcDriver::GuestMemory::GpuMutex());
    Graphics::GuestBufferMemory memory(context);
    memory.AcquireRegistered();
    memory.Upload();
    Expect(memory.HoldsLease(), "override page tables: the build holds no address space");
    const auto base = memory.PageTable(context);
    Require(base != nullptr, "override page tables: the address space built no base table");
    const auto r1 = Resident(*base, units.Address(0), units.Address(2), "the first two registered units");
    const auto r2 = Resident(*base, units.Address(3), units.Address(4), "the writable unit 3");
    const auto r3 = Resident(*base, units.Address(4), units.Address(5), "the read-only unit 4 beside it");
    const auto r4 = Resident(*base, units.Address(6), units.Address(8), "units 6 and 7");
    const auto r5 = Resident(*base, units.Address(9), units.Address(10), "unit 9");
    Require(r1 && r2 && r3 && r4 && r5, "override page tables: the registered units are not the expected resident ranges");
    Expect(r3->permissions == Abi::Read && r2->permissions == (Abi::Read | Abi::Write), "override page tables: the read-only unit's range is not read-only beside a writable one");
    for (const auto address : {units.Address(2), units.Address(5), units.Address(8, Unit - 1u)}) Expect(!base->Find(address).has_value(), "override page tables: an unregistered unit at " + Hex(address) + " is in the base table");

    const std::vector<Abi::Range> cuts{
        Override(r1->begin, r1->begin + 0x1234u, 0, Abi::Read),
        Override(r1->begin + 0x8010u, r1->begin + 0x18007u, 1, Abi::Read | Abi::Write),
        Override(r1->end - 0x321u, r1->end, 2, Abi::Read),
        Override(r2->begin + 0x4000u, r3->begin + 0x2000u, 3, Abi::Read | Abi::Write),
        Override(r4->begin + 0x100u, r4->begin + 0x900u, 4, Abi::Read),
        Override(r4->begin + 0x900u, r4->begin + 0x1100u, 5, Abi::Read | Abi::Write),
        Override(r5->begin, r5->end, 6, Abi::Read),
    };
    std::vector<std::uint64_t> every;
    for (auto address = units.Address(0) - 0x100u; address < units.Address(Units) + 0x100u; ++address) every.push_back(address);
    const auto baseEvery = Translations(*base, every);
    const auto spliced = memory.PageTableWith(context, cuts);
    Require(spliced != nullptr, "override page tables: no table for the override cuts");
    ExpectTranslations(*spliced, *base, cuts, every, "override page tables: cuts at a range's start, middle and end, over two adjacent ranges, two adjacent overrides and a whole range");
    Expect(Translations(*base, every) == baseEvery, "override page tables: the base table translates differently after the override table was built");

    const auto window = static_cast<std::uint32_t>(units.Address(0) >> Abi::WindowShift);
    Require(window == static_cast<std::uint32_t>((units.Address(Units) - 1u) >> Abi::WindowShift), "override page tables: the registered units straddle two page table windows");
    const auto other = (window + 1u) % Abi::WindowCount;
    const auto third = (window + 2u) % Abi::WindowCount;
    const auto start = [](std::uint32_t at) { return static_cast<std::uint64_t>(at) << Abi::WindowShift; };
    const std::vector<Abi::Range> overridesA{Override(r4->begin + 0x1000u, r4->begin + 0x3000u, 10, Abi::Read), Override(start(other) + 4u * Unit, start(other) + 6u * Unit, 11, Abi::Read | Abi::Write)};
    std::vector<Abi::Range> overridesB{Override(r4->begin + 0x5000u, r4->begin + 0x7000u, 10, Abi::Read), overridesA[1]};
    if (other < window) std::swap(overridesB[0], overridesB[1]);
    std::vector<Abi::Range> sortedA = overridesA;
    if (other < window) std::swap(sortedA[0], sortedA[1]);
    const std::vector<Abi::Range> overridesC{Override(start(third) + Unit, start(third) + 2u * Unit, 12, Abi::Read)};

    std::vector<std::uint64_t> pages;
    for (const auto at : {window, other, third}) {
        for (std::uint64_t page = 0; page < Abi::WindowPages; ++page) pages.push_back(start(at) + (page << Abi::PageShift));
    }
    for (auto address = units.Address(0); address < units.Address(Units); address += 16u) pages.push_back(address);
    AddBoundaries(pages, sortedA);
    AddBoundaries(pages, overridesB);
    AddBoundaries(pages, overridesC);
    AddBoundaries(pages, std::array{*r1, *r2, *r3, *r4, *r5});
    const auto baseLeaves = Leaves(*base);
    const auto basePages = Translations(*base, pages);
    const auto a = memory.PageTableWith(context, sortedA);
    const auto b = memory.PageTableWith(context, overridesB);
    const auto c = memory.PageTableWith(context, overridesC);
    Require(a != nullptr && b != nullptr && c != nullptr, "override page tables: an override table was not built");
    ExpectTranslations(*a, *base, sortedA, pages, "override page tables: table A");
    ExpectTranslations(*b, *base, overridesB, pages, "override page tables: table B, with A's override moved within its window");
    ExpectTranslations(*c, *base, overridesC, pages, "override page tables: table C, with one override in a third window");

    const auto leavesA = Leaves(*a);
    const auto leavesB = Leaves(*b);
    const auto leavesC = Leaves(*c);
    std::size_t aWrong = 0;
    std::size_t bWrong = 0;
    std::size_t cWrong = 0;
    for (std::uint32_t at = 0; at < Abi::WindowCount; ++at) {
        if (at == window || at == other) {
            aWrong += leavesA[at] != 0u && leavesA[at] != baseLeaves[at] ? 0u : 1u;
        } else {
            aWrong += leavesA[at] == baseLeaves[at] ? 0u : 1u;
        }
        if (at == window) {
            bWrong += leavesB[at] != 0u && leavesB[at] != leavesA[at] && leavesB[at] != baseLeaves[at] ? 0u : 1u;
        } else {
            bWrong += leavesB[at] == leavesA[at] ? 0u : 1u;
        }
        if (at == third) {
            cWrong += leavesC[at] != 0u && leavesC[at] != baseLeaves[at] ? 0u : 1u;
        } else {
            cWrong += leavesC[at] == baseLeaves[at] ? 0u : 1u;
        }
    }
    Expect(aWrong == 0u, "override page tables: table A has " + std::to_string(aWrong) + " windows that are not the base's leaves outside its two override windows, or not new leaves inside them");
    Expect(bWrong == 0u, "override page tables: table B has " + std::to_string(bWrong) + " windows that are not A's leaves outside the window whose override moved, or not a new leaf inside it");
    Expect(cWrong == 0u, "override page tables: table C has " + std::to_string(cWrong) + " windows that are not the base's leaves outside its override window, or not a new leaf inside it");
    Expect(memory.PageTable(context) == base && Leaves(*base) == baseLeaves, "override page tables: the base table or one of its leaves changed after the override tables were built");
    Expect(Translations(*base, pages) == basePages, "override page tables: the base table translates differently after tables A, B and C were built");
    memory.Release();
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        if (context.hostImportAlignment == 0 || !context.bufferDeviceAddress) {
            std::puts("override page tables: host imports or buffer device addresses unavailable");
            return VulkanTestSkipped;
        }
        SplicesAndSharing(context);
    } catch (const std::exception& error) {
        Expect(false, std::string("exception: ") + error.what());
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d override page table checks failed\n", failures);
        return 1;
    }
    std::puts("override page table tests passed");
    return 0;
}

#else

int main() {
    return VulkanTestSkipped;
}

#endif
