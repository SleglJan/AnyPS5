"""Apply the quarter-turn fold and the linear small-argument path on top of #1979's EmitTrigCycleF32. Usage: apply_fold.py <SpirvAluEmitterMath.cpp>"""
import pathlib
import sys

SIN_HEAD = "std::uint32_t EmitFPSin(SpirvEmitterState& state, std::uint32_t arg0) {"
COS_HEAD = "std::uint32_t EmitFPCos(SpirvEmitterState& state, std::uint32_t arg0) {"
NEW = '''std::uint32_t TrigLinearNearZero(SpirvEmitterState& state, std::uint32_t value, std::uint32_t distance, std::uint32_t linearConstant) {
    const auto linear = Binary(state, spv::OpFMul, TypeF32(state), distance, linearConstant);
    const auto magnitude = Binary(state, spv::OpBitwiseAnd, TypeU32(state), Unary(state, spv::OpBitcast, TypeU32(state), distance), ConstantU32(state, 0x7fffffffu));
    const auto near = Binary(state, spv::OpULessThan, TypeBool(state), magnitude, ConstantU32(state, 0x39800000u));
    return Select(state, TypeF32(state), near, linear, value);
}

std::uint32_t EmitFPSin(SpirvEmitterState& state, std::uint32_t arg0) {
    const auto cycle = EmitTrigCycleF32(state, arg0, true);
    const auto absolute = EmitExt(state, TypeF32(state), GLSLstd450FAbs, {cycle});
    const auto signBits = Binary(state, spv::OpBitwiseAnd, TypeU32(state), Unary(state, spv::OpBitcast, TypeU32(state), cycle), ConstantU32(state, 0x80000000u));
    const auto value = EmitExt(state, TypeF32(state), GLSLstd450Sin, {Binary(state, spv::OpFMul, TypeF32(state), absolute, ConstantF32(state, 0x40c90fdbu))});
    const auto nearHalf = Binary(state, spv::OpFOrdGreaterThan, TypeBool(state), absolute, ConstantF32(state, 0x3e800000u));
    const auto distance = Select(state, TypeF32(state), nearHalf, Binary(state, spv::OpFSub, TypeF32(state), ConstantF32(state, 0x3f000000u), absolute), absolute);
    const auto linearConstant = Select(state, TypeF32(state), nearHalf, ConstantF32(state, 0x40c90fdbu), ConstantF32(state, 0x40c90fd5u));
    const auto result = TrigLinearNearZero(state, value, distance, linearConstant);
    return Unary(state, spv::OpBitcast, TypeF32(state), Binary(state, spv::OpBitwiseOr, TypeU32(state), Unary(state, spv::OpBitcast, TypeU32(state), result), signBits));
}

std::uint32_t EmitFPCos(SpirvEmitterState& state, std::uint32_t arg0) {
    const auto cycle = EmitTrigCycleF32(state, arg0, false);
    const auto absolute = EmitExt(state, TypeF32(state), GLSLstd450FAbs, {cycle});
    const auto value = EmitExt(state, TypeF32(state), GLSLstd450Cos, {Binary(state, spv::OpFMul, TypeF32(state), absolute, ConstantF32(state, 0x40c90fdbu))});
    const auto distance = Binary(state, spv::OpFSub, TypeF32(state), ConstantF32(state, 0x3e800000u), absolute);
    return TrigLinearNearZero(state, value, distance, ConstantF32(state, 0x40c90fdbu));
}
'''


def main():
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    start = text.index(SIN_HEAD)
    cos = text.index(COS_HEAD, start)
    end = text.index("\n}\n", cos) + 3
    old = text[start:end]
    if "EmitTrigCycleF32(state, arg0, true)" not in old or "GLSLstd450Cos" not in old:
        raise SystemExit("unexpected EmitFPSin/EmitFPCos")
    path.write_text(text[:start] + NEW + text[end:])
    print("fold applied on top of EmitTrigCycleF32 in", path)


if __name__ == "__main__":
    main()
