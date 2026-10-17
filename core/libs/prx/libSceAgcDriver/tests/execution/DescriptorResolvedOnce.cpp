#include "DeviceTableFixture.hpp"
#include <spirv/unified1/spirv.hpp>
#include <map>
#include <set>

namespace {

using namespace DeviceTable;

constexpr std::uint32_t InputSlot = 20;
constexpr std::uint32_t InputBase = 0x6e000000u;
constexpr std::uint32_t Accesses = 4;
constexpr std::uint32_t AccessStride = 0x400u;
constexpr std::uint32_t InputWords = Accesses * AccessStride / 4u;

TestGuest::Array<std::uint32_t, InputWords> Input;

constexpr std::uint32_t BufferLoadFormatX = 0x00u;

const Program& LoadProgram(std::uint32_t opcode, std::uint32_t accesses) {
    static std::map<std::pair<std::uint32_t, std::uint32_t>, Program> programs;
    auto& built = programs[{opcode, accesses}];
    if (built.size != 0u) return built;
    using namespace NewPath::Isa;
    using namespace Asm;
    constexpr std::uint32_t VMadU32U24 = 0x143u;
    built << Smem(SLoadDwordX4, 28, 0, OutputSlot * 4u)
          << Smem(SLoadDwordX4, 24, 0, InputSlot * 4u)
          << std::array<std::uint32_t, 2>{0xd4000000u | (VMadU32U24 << 16u) | 3u, 2u | (InlineInteger(64) << 9u) | (Vgpr(0) << 18u)}
          << Vop2(VLshlRevB32, 3, InlineInteger(2), 3);
    for (std::uint32_t component = 0; component < 4u; ++component) built << Vop1(VMovB32, 4 + component, InlineInteger(0));
    built << WaitAll;
    for (std::uint32_t access = 0; access < accesses; ++access) built << Mubuf(opcode, 4 + access, 3, 24, false, access * AccessStride);
    StoreLane(built, 2, 28);
    return built;
}

std::size_t TranslationsAtLoads(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> calls;
    std::uint32_t function = 0;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        if (op == spv::OpName) names[words[cursor + 1]] = reinterpret_cast<const char*>(&words[cursor + 2]);
        if (op == spv::OpFunction) function = words[cursor + 2];
        if (op == spv::OpFunctionEnd) function = 0;
        if (op == spv::OpFunctionCall) calls.emplace_back(function, words[cursor + 3]);
    }
    std::size_t sites = 0;
    for (const auto& [caller, callee] : calls) {
        if (names[caller] != "main") continue;
        const auto& name = names[callee];
        if (name == "bda_translate" || name == "bda_segment") ++sites;
    }
    return sites;
}

void Check(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> userData, std::uint32_t opcode, const std::string& kind) {
    const auto single = Compile(device, LoadProgram(opcode, 1u), userData);
    const auto many = Compile(device, LoadProgram(opcode, Accesses), userData);
    const auto singleSites = TranslationsAtLoads(single.result->spirv.Words());
    const auto manySites = TranslationsAtLoads(many.result->spirv.Words());
    Expect(manySites == singleSites, kind + ": main calls the page-table translation itself at " + std::to_string(manySites) + " sites for " + std::to_string(Accesses) + " accesses through one V#, against " + std::to_string(singleSites) + " for one access; the V# must be translated once where it is loaded, the accesses' helpers translating only on a miss of its segment");
    for (const auto* compiled : {&single, &many}) {
        const auto accesses = compiled == &single ? 1u : Accesses;
        const auto what = kind + ", " + std::to_string(accesses) + " access(es)";
        const auto outcome = Run(device, *compiled, Height);
        ExpectClean(outcome, what);
        const Expected expected{Height, 0, 4, [accesses](std::uint32_t row, std::uint32_t x, std::uint32_t component, std::uint32_t word) {
            if (component >= accesses) return word == 0u;
            return word == InputBase + row * Lanes + x + component * (AccessStride / 4u);
        }};
        std::string mismatch;
        Expect(Classify(outcome, expected, mismatch) == Read::Expected, what + ": " + mismatch);
    }
}

void Run(AgcDriver::VulkanDevice& device) {
    for (std::uint32_t index = 0; index < InputWords; ++index) Input[index] = InputBase + index;
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(InputSlot, Buffer(Input.data(), static_cast<std::uint32_t>(decltype(Input)::ByteSize)));
    Check(device, userData, NewPath::Isa::BufferLoadDword, "dword loads");
    Check(device, userData, BufferLoadFormatX, "formatted loads");
}

}

int main() {
    return Main("descriptor resolved once", Run);
}
