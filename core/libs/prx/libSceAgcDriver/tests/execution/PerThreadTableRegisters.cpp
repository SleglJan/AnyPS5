#include "DeviceTableFixture.hpp"
#include <spirv/unified1/spirv.hpp>
#include <map>
#include <set>

namespace {

using namespace DeviceTable;

constexpr std::uint32_t InputSlot = 20;
constexpr std::uint32_t IndexSlot = 24;
constexpr std::uint32_t TableRegisters = 8;
constexpr std::uint32_t TableBase = 10;
constexpr std::uint32_t TableStride = 0x100u;
constexpr std::uint32_t Replaced = 0xabcu;
constexpr std::uint32_t InputBase = 0x6d000000u;

TestGuest::Array<std::uint32_t, Height * Lanes> Input;

namespace Ops {

using namespace NewPath::Isa;

constexpr std::uint32_t SLoadDword = 0x00u;
constexpr std::uint32_t VMovrelsB32 = 0x43u;
constexpr std::uint32_t VMovreldB32 = 0x42u;
constexpr std::uint32_t MoveM0FromS4 = 0xbefc0304u;
constexpr std::uint32_t VMadU32U24 = 0x143u;

constexpr std::array<std::uint32_t, 2> Vop3(std::uint32_t opcode, std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1, std::uint32_t src2) {
    return {0xd4000000u | (opcode << 16u) | vdst, src0 | (src1 << 9u) | (src2 << 18u)};
}

}

const Program& TableProgram() {
    static const Program program = [] {
        using namespace Ops;
        using namespace Asm;
        Program built;
        built << Smem(SLoadDwordX4, 28, 0, OutputSlot * 4u)
              << Smem(SLoadDwordX4, 24, 0, InputSlot * 4u)
              << Smem(SLoadDword, 4, 0, IndexSlot * 4u);
        for (std::uint32_t entry = 0; entry < TableRegisters; ++entry) built << Vop2(VAddNcU32, TableBase + entry, Literal, 0) << entry * TableStride;
        built << Vop3(VMadU32U24, 3, 2, InlineInteger(64), Vgpr(0))
              << Vop2(VLshlRevB32, 3, InlineInteger(2), 3)
              << WaitAll
              << MoveM0FromS4
              << Vop1(VMovrelsB32, 4, Vgpr(TableBase))
              << Vop1(VMovreldB32, TableBase, Literal) << Replaced
              << Vop1(VMovB32, 5, Vgpr(TableBase + 3u))
              << Vop1(VMovrelsB32, 6, Vgpr(TableBase))
              << Mubuf(BufferLoadDword, 7, 3, 24);
        StoreLane(built, 2, 28);
        return built;
    }();
    return program;
}

struct Violations {
    std::vector<std::string> dynamicIndexing;
    std::vector<std::string> resultPointers;
};

Violations Inspect(std::span<const std::uint32_t> words) {
    std::map<std::uint32_t, std::string> names;
    std::set<std::uint32_t> constants;
    std::map<std::uint32_t, std::uint32_t> pointerStorage;
    std::map<std::uint32_t, std::uint32_t> pointerPointee;
    std::set<std::uint32_t> arrays;
    std::set<std::uint32_t> localArrays;
    std::map<std::uint32_t, std::uint32_t> localPointers;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        const auto operand = [&](std::size_t index) { return words[cursor + 1u + index]; };
        if (op == spv::OpName) names[operand(0)] = reinterpret_cast<const char*>(&words[cursor + 2]);
        if (op == spv::OpConstant || op == spv::OpSpecConstant || op == spv::OpConstantNull) constants.insert(operand(1));
        if (op == spv::OpTypeArray || op == spv::OpTypeRuntimeArray) arrays.insert(operand(0));
        if (op == spv::OpTypePointer) {
            pointerStorage[operand(0)] = operand(1);
            pointerPointee[operand(0)] = operand(2);
        }
        if (op == spv::OpVariable) {
            const auto storage = operand(2);
            if (storage != spv::StorageClassFunction && storage != spv::StorageClassPrivate) continue;
            localPointers[operand(1)] = storage;
            const auto pointee = pointerPointee[operand(0)];
            if (arrays.contains(pointee)) localArrays.insert(operand(1));
        }
    }
    const auto name = [&](std::uint32_t id) { return names.contains(id) ? names[id] : "%" + std::to_string(id); };
    Violations found;
    std::string function;
    for (std::size_t cursor = 5; cursor < words.size() && (words[cursor] >> 16u) != 0u; cursor += words[cursor] >> 16u) {
        const auto op = words[cursor] & 0xffffu;
        const auto count = words[cursor] >> 16u;
        const auto operand = [&](std::size_t index) { return words[cursor + 1u + index]; };
        if (op == spv::OpFunction) function = name(operand(1));
        if (op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain || op == spv::OpPtrAccessChain || op == spv::OpInBoundsPtrAccessChain) {
            const auto base = operand(2);
            if (localPointers.contains(base)) localPointers[operand(1)] = localPointers[base];
            if (!localArrays.contains(base)) continue;
            for (std::size_t index = 3; index + 1u < count; ++index) {
                if (constants.contains(operand(index))) continue;
                found.dynamicIndexing.push_back(function + " indexes " + name(base) + " with a non-constant index");
                break;
            }
        }
        if (op == spv::OpFunctionCall) {
            for (std::size_t argument = 3; argument + 1u < count; ++argument) {
                const auto value = operand(argument);
                if (!localPointers.contains(value) || localPointers[value] != spv::StorageClassFunction) continue;
                found.resultPointers.push_back(function + " passes the Function-storage pointer " + name(value) + " to " + name(operand(2)));
            }
        }
    }
    return found;
}

std::string Join(const std::vector<std::string>& lines) {
    std::string text;
    for (std::size_t index = 0; index < lines.size() && index < 8u; ++index) text += "\n  " + lines[index];
    if (lines.size() > 8u) text += "\n  ... " + std::to_string(lines.size() - 8u) + " more";
    return text;
}

void Run(AgcDriver::VulkanDevice& device) {
    for (std::uint32_t index = 0; index < Height * Lanes; ++index) Input[index] = InputBase + index;
    Srt.fill(0u);
    const auto userData = SrtUserData();
    Place(InputSlot, Buffer(Input.data(), static_cast<std::uint32_t>(decltype(Input)::ByteSize)));
    for (const std::uint32_t index : {3u, 6u}) {
        const auto what = "a per-thread table indexed by " + std::to_string(index);
        Srt[IndexSlot] = index;
        const auto compiled = Compile(device, TableProgram(), userData);
        const auto violations = Inspect(compiled.result->spirv.Words());
        Expect(violations.dynamicIndexing.empty(), what + ": the module indexes Function or Private arrays dynamically, which the GPU places in local memory:" + Join(violations.dynamicIndexing));
        Expect(violations.resultPointers.empty(), what + ": the module returns helper results through Function-storage pointers instead of by value:" + Join(violations.resultPointers));
        const auto outcome = Run(device, compiled, Height);
        ExpectClean(outcome, what);
        const Expected expected{Height, 0, 4, [index](std::uint32_t row, std::uint32_t x, std::uint32_t component, std::uint32_t word) {
            switch (component) {
            case 0: return word == x + index * TableStride;
            case 1: return word == (index == 3u ? Replaced : x + 3u * TableStride);
            case 2: return word == Replaced;
            default: return word == InputBase + row * Lanes + x;
            }
        }};
        std::string mismatch;
        Expect(Classify(outcome, expected, mismatch) == Read::Expected, what + ": " + mismatch);
    }
}

}

int main() {
    return Main("per-thread table registers", Run);
}
