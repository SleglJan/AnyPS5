"""Apply the quarter-turn fold and the linear small-argument path on top of #1979's EmitTrigCycleF32. Usage: apply_fold.py <SpirvAluEmitterMath.cpp>"""
import pathlib
import sys

SIN_HEAD = "std::uint32_t EmitFPSin(SpirvEmitterState& state, std::uint32_t arg0) {"
COS_HEAD = "std::uint32_t EmitFPCos(SpirvEmitterState& state, std::uint32_t arg0) {"
NEW = '''std::uint32_t TrigSineOfTurn(SpirvEmitterState& state, std::uint32_t turn, std::uint32_t linearConstant) {
    const auto radians = Binary(state, spv::OpFMul, TypeF32(state), turn, ConstantF32(state, 0x40c90fdbu));
    const auto value = EmitExt(state, TypeF32(state), GLSLstd450Sin, {radians});
    const auto linear = Binary(state, spv::OpFMul, TypeF32(state), turn, linearConstant);
    const auto magnitude = Binary(state, spv::OpBitwiseAnd, TypeU32(state), Unary(state, spv::OpBitcast, TypeU32(state), turn), ConstantU32(state, 0x7fffffffu));
    const auto small = Binary(state, spv::OpULessThan, TypeBool(state), magnitude, ConstantU32(state, 0x39800000u));
    return Select(state, TypeF32(state), small, linear, value);
}

std::uint32_t EmitFPSin(SpirvEmitterState& state, std::uint32_t arg0) {
    const auto cycle = EmitTrigCycleF32(state, arg0, true);
    const auto absolute = EmitExt(state, TypeF32(state), GLSLstd450FAbs, {cycle});
    const auto signBits = Binary(state, spv::OpBitwiseAnd, TypeU32(state), Unary(state, spv::OpBitcast, TypeU32(state), cycle), ConstantU32(state, 0x80000000u));
    const auto beyondQuarter = Binary(state, spv::OpFOrdGreaterThan, TypeBool(state), absolute, ConstantF32(state, 0x3e800000u));
    const auto turn = Select(state, TypeF32(state), beyondQuarter, Binary(state, spv::OpFSub, TypeF32(state), ConstantF32(state, 0x3f000000u), absolute), absolute);
    const auto linearConstant = Select(state, TypeF32(state), beyondQuarter, ConstantF32(state, 0x40c90fdbu), ConstantF32(state, 0x40c90fd5u));
    const auto value = TrigSineOfTurn(state, turn, linearConstant);
    return Unary(state, spv::OpBitcast, TypeF32(state), Binary(state, spv::OpBitwiseOr, TypeU32(state), Unary(state, spv::OpBitcast, TypeU32(state), value), signBits));
}

std::uint32_t EmitFPCos(SpirvEmitterState& state, std::uint32_t arg0) {
    const auto cycle = EmitTrigCycleF32(state, arg0, false);
    const auto absolute = EmitExt(state, TypeF32(state), GLSLstd450FAbs, {cycle});
    const auto direct = EmitExt(state, TypeF32(state), GLSLstd450Cos, {Binary(state, spv::OpFMul, TypeF32(state), absolute, ConstantF32(state, 0x40c90fdbu))});
    const auto folded = TrigSineOfTurn(state, Binary(state, spv::OpFSub, TypeF32(state), ConstantF32(state, 0x3e800000u), absolute), ConstantF32(state, 0x40c90fdbu));
    const auto nearQuarter = Binary(state, spv::OpFOrdGreaterThanEqual, TypeBool(state), absolute, ConstantF32(state, 0x3e000000u));
    return Select(state, TypeF32(state), nearQuarter, folded, direct);
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
