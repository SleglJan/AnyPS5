"""Replace EmitFPLog2 with the gated near-one series. Usage: apply_log.py <SpirvAluEmitterMath.cpp>"""
import pathlib
import sys

OLD = '''std::uint32_t EmitFPLog2(SpirvEmitterState& state, std::uint32_t arg0) {
    return EmitExt(state, TypeF32(state), GLSLstd450Log2, {EmitFlushF32DenormToSignedZero(state, arg0)});
}
'''
NEW = '''std::uint32_t Log2SeriesNearOne(SpirvEmitterState& state, std::uint32_t difference) {
    auto polynomial = ConstantF32(state, 0xbe000000u);
    for (const std::uint32_t coefficient : {0x3e124925u, 0xbe2aaaabu, 0x3e4ccccdu, 0xbe800000u, 0x3eaaaaabu, 0xbf000000u, 0x3f800000u}) {
        polynomial = Exact(state, EmitExt(state, TypeF32(state), GLSLstd450Fma, {polynomial, difference, ConstantF32(state, coefficient)}));
    }
    const auto product = Exact(state, Binary(state, spv::OpFMul, TypeF32(state), difference, polynomial));
    return Exact(state, Binary(state, spv::OpFMul, TypeF32(state), product, ConstantF32(state, 0x3fb8aa3bu)));
}

std::uint32_t EmitFPLog2(SpirvEmitterState& state, std::uint32_t arg0) {
    const auto source = EmitFlushF32DenormToSignedZero(state, arg0);
    const auto value = EmitExt(state, TypeF32(state), GLSLstd450Log2, {source});
    const auto difference = Exact(state, Binary(state, spv::OpFSub, TypeF32(state), source, ConstantF32(state, 0x3f800000u)));
    const auto series = Log2SeriesNearOne(state, difference);
    const auto valueBits = Unary(state, spv::OpBitcast, TypeU32(state), value);
    const auto seriesBits = Unary(state, spv::OpBitcast, TypeU32(state), series);
    const auto distance = EmitExt(state, TypeU32(state), GLSLstd450UMin, {Binary(state, spv::OpISub, TypeU32(state), valueBits, seriesBits), Binary(state, spv::OpISub, TypeU32(state), seriesBits, valueBits)});
    const auto near = Binary(state, spv::OpFOrdLessThanEqual, TypeBool(state), EmitExt(state, TypeF32(state), GLSLstd450FAbs, {difference}), ConstantF32(state, 0x3e000000u));
    const auto far = Binary(state, spv::OpUGreaterThan, TypeBool(state), distance, ConstantU32(state, 4u));
    return Select(state, TypeF32(state), Binary(state, spv::OpLogicalAnd, TypeBool(state), near, far), series, value);
}
'''


def main():
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    if text.count(OLD) != 1:
        raise SystemExit("EmitFPLog2 not found once")
    path.write_text(text.replace(OLD, NEW, 1))
    print("EmitFPLog2 replaced in", path)


if __name__ == "__main__":
    main()
