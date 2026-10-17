#include "Optimization/FloatIdioms.hpp"
#include <bit>

namespace ShaderRecompiler {

namespace {

bool isImmediate(const IrValue* value, std::uint32_t bits) {
    return value->HasImmediate() && value->ImmediateU32() == bits;
}

}

IrValue* MaskedOperand(const IrValue& value, IrOpcode opcode, std::uint32_t bits) {
    if (value.Opcode() != opcode || value.ArgumentCount() != 2u) return nullptr;
    IrValue* lhs = value.Argument(0)->Resolve();
    IrValue* rhs = value.Argument(1)->Resolve();
    if (isImmediate(rhs, bits)) return lhs;
    if (isImmediate(lhs, bits)) return rhs;
    return nullptr;
}

IrValue* FlushedF32Source(const IrValue& value) {
    if (value.Opcode() != IrOpcode::SelectU32 || value.ArgumentCount() != 3u) return nullptr;
    IrValue* source = value.Argument(2)->Resolve();
    const IrValue* exponent = MaskedOperand(*value.Argument(0)->Resolve(), IrOpcode::IEqual32, 0u);
    if (exponent == nullptr || MaskedOperand(*exponent, IrOpcode::BitwiseAnd32, 0x7f800000u) != source) return nullptr;
    return MaskedOperand(*value.Argument(1)->Resolve(), IrOpcode::BitwiseAnd32, 0x80000000u) == source ? source : nullptr;
}

IrValue* NanTestedF32Source(const IrValue& value) {
    if (value.Opcode() != IrOpcode::UGreaterThan32 || value.ArgumentCount() != 2u || !isImmediate(value.Argument(1)->Resolve(), 0x7f800000u)) return nullptr;
    return MaskedOperand(*value.Argument(0)->Resolve(), IrOpcode::BitwiseAnd32, 0x7fffffffu);
}

IrValue* QuietedF32Source(const IrValue& value) {
    return MaskedOperand(value, IrOpcode::BitwiseOr32, 0x00400000u);
}

IrValue* MagnitudeSource(const IrValue& value) {
    IrValue* current = value.Resolve();
    while (true) {
        IrValue* inner = MaskedOperand(*current, IrOpcode::BitwiseAnd32, 0x7fffffffu);
        if (inner == nullptr) inner = MaskedOperand(*current, IrOpcode::BitwiseXor32, 0x80000000u);
        if (inner == nullptr) return current;
        current = inner;
    }
}

std::optional<std::uint32_t> ExactF32Product(std::uint32_t lhs, std::uint32_t rhs) {
    const auto normalOrZero = [](std::uint32_t bits) {
        const std::uint32_t exponent = bits & 0x7f800000u;
        return exponent == 0u ? (bits & 0x007fffffu) == 0u : exponent != 0x7f800000u;
    };
    if (!normalOrZero(lhs) || !normalOrZero(rhs)) return std::nullopt;
    const std::uint32_t product = std::bit_cast<std::uint32_t>(std::bit_cast<float>(lhs) * std::bit_cast<float>(rhs));
    if (!normalOrZero(product)) return std::nullopt;
    const bool zeroOperand = (lhs & 0x7fffffffu) == 0u || (rhs & 0x7fffffffu) == 0u;
    if ((product & 0x7fffffffu) == 0u && !zeroOperand) return std::nullopt;
    return product;
}

}
