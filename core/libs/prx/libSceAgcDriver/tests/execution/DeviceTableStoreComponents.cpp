#include "DeviceTableFixture.hpp"
#include <cstdio>
#include <unordered_set>

namespace {

using namespace DeviceTable;

constexpr std::size_t ComponentSelections = 16;

TestGuest::Array<float, Width * Height> Texels;

Program StoresProgram(std::uint32_t stores) {
    using namespace Asm;
    Program built;
    LoadSrt(built, {{28, OutputSlot}, {36, ImageSlot}});
    TexelIndex(built, 2);
    for (std::uint32_t component = 0; component < 4u; ++component) built << Vop1(VMovB32, 4 + component, InlineInteger(component + 1u));
    for (std::uint32_t store = 0; store < stores; ++store) {
        built << Mimg(ImageStore, 0xfu, true, 4, 1, 36) << Vop2(VAddNcU32, 1, InlineInteger(1), 1);
    }
    StoreLane(built, 2, 28);
    return built;
}

std::size_t SwizzleSelections(std::span<const std::uint32_t> words) {
    constexpr std::uint32_t OpConstant = 43;
    constexpr std::uint32_t OpIEqual = 170;
    std::unordered_set<std::uint32_t> destinations;
    for (std::size_t at = 5; at < words.size();) {
        const auto count = words[at] >> 16u;
        if (count == 0u) break;
        if ((words[at] & 0xffffu) == OpConstant && count == 4u && words[at + 3] >= 4u && words[at + 3] <= 7u) destinations.insert(words[at + 2]);
        at += count;
    }
    std::size_t selections = 0;
    for (std::size_t at = 5; at < words.size();) {
        const auto count = words[at] >> 16u;
        if (count == 0u) break;
        if ((words[at] & 0xffffu) == OpIEqual && count == 5u && destinations.contains(words[at + 4])) ++selections;
        at += count;
    }
    return selections;
}

std::size_t Selections(AgcDriver::VulkanDevice& device, const Program& program, std::span<const std::uint32_t> userData, const std::string& what) {
    const auto compiled = Compile(device, program, userData);
    ExpectRouted(compiled, what);
    return SwizzleSelections(compiled.result->spirv.Words());
}

void Run(AgcDriver::VulkanDevice& device) {
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(ImageSlot, Image(Texels.data(), Format32Float));
    const auto perStore = Selections(device, StoresProgram(3), userData, "three stores") - Selections(device, StoresProgram(2), userData, "two stores");
    std::printf("store components: one device-table store selects %zu components\n", perStore);
    Expect(perStore == ComponentSelections, "store components: one store to a device-table storage image selects " + std::to_string(perStore) + " components by swizzle, expected " + std::to_string(ComponentSelections) + " (one component set for all its storage mode arms)");
}

}

int main() {
    return Main("device table store components", Run);
}
