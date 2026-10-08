"""The execution tests' own pinned values, read from their source tables, with each test's assertion rule.

pins(file, kernel) -> Pins or None (None: the test computes its expectation in code; no table to compare with).
Pins.at(row, col) -> (pin, asserted, accepts) where pin is the table value (int) or None, asserted says whether Check()
asserts that cell at all, and accepts(actual) is Check()'s own comparison (exact, NaN-tolerant or a tolerance).
Rows are the test's lanes / vectors in table order (the adapters' table and first4 rows use the same order).
"""
import math
import os
import pathlib
import struct
import sys

ROOT = pathlib.Path(os.environ.get("ANYPS5_AUDIT_ROOT") or pathlib.Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / "scripts" / "oracle"))
import srctables as st  # noqa: E402

TESTS = ROOT / "AnyPS5" / "core" / "libs" / "prx" / "libSceAgcDriver" / "tests" / "execution"


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xffffffff))[0]


def nan16(b):
    return (b & 0x7fff) > 0x7c00


def nan32(b):
    return (b & 0x7fffffff) > 0x7f800000


def denormal32(b):
    return (b & 0x7f800000) == 0 and (b & 0x007fffff) != 0


def exact(pin):
    return lambda actual: actual == pin


class Pins:
    def __init__(self, source, rule, at):
        self.source, self.rule, self._at = source, rule, at

    def at(self, row, col):
        return self._at(row, col)


def _src(file):
    return (TESTS / file).read_text()


def _table2d(file, name, rule, asserted=lambda r, c: True, accept=None):
    t = st.table(_src(file), name)

    def at(r, c):
        if r >= len(t) or c >= len(t[r]):
            return None, False, None
        pin = t[r][c]
        return pin, asserted(r, c), (accept(r, c, pin) if accept else exact(pin))
    return Pins(f"{file}:{name}[{len(t)}][{len(t[0])}]", rule, at)


def _broadcast(file, name, rule):
    t = st.table(_src(file), name)

    def at(r, c):
        if c >= len(t):
            return None, False, None
        return t[c], True, exact(t[c])
    return Pins(f"{file}:{name}[{len(t)}] (same for every lane)", rule, at)


def _float16_result_modifiers():
    def half_value(b):
        mag = math.ldexp(b & 0x3ff, -24) if (b & 0x7c00) == 0 else math.ldexp((b & 0x3ff) | 0x400, ((b >> 10) & 0x1f) - 25)
        return -mag if b & 0x8000 else mag

    def same_half(a, e):
        return a == e or (nan16(a) and nan16(e))

    def near_half(a, e, clamped):
        if clamped and a & 0x8000:
            return False
        if not clamped and (e & 0x7fff) == 0:
            return a == e
        if nan16(a) or nan16(e) or (a & 0x7fff) == 0x7c00 or (e & 0x7fff) == 0x7c00:
            return same_half(a, e)
        return abs(half_value(a) - half_value(e)) <= 2 ** -10

    def accept(r, c, pin):
        if c in (3, 4):
            return lambda x: (x >> 16) == (pin >> 16) and near_half(x & 0xffff, pin & 0xffff, c == 4)
        return lambda x: same_half(x & 0xffff, pin & 0xffff) and same_half(x >> 16, pin >> 16)
    return _table2d("Float16ResultModifiers.cpp", "Expected", "every cell; halves NaN-tolerant; cols 3,4 within 2^-10 (col 4 non-negative)", accept=accept)


def _vop1_float_unary():
    src = _src("Vop1FloatUnary.cpp")
    checked = st.table(src, "Checked")
    tol = st.table(src, "Tolerance")
    rows = st.table(src, "Rows")

    def half(b):
        e = (b >> 10) & 0x1f
        if e == 0x1f:
            mag = math.nan if b & 0x3ff else math.inf
        else:
            mag = math.ldexp((b & 0x3ff) | (0x400 if e else 0), (e if e else 1) - 25)
        return -mag if b & 0x8000 else mag

    def finite(v):
        return math.isfinite(v) and v != 0.0

    def accept(r, c, pin):
        if tol[c] == 0:
            return exact(pin)

        def near(x):
            h = tol[c] >= 5
            a = half(x & 0xffff) if h else f32(x)
            e = half(pin & 0xffff) if h else f32(pin)
            if not finite(e) or (h and (x >> 16) != (pin >> 16)):
                return x == pin
            xin = half((rows[r][2] >> (16 if c % 2 else 0)) & 0xffff) if h else f32(rows[r][c % 2])
            err = abs(a - e)
            rel = err / abs(e)
            t = tol[c]
            if t == 1:
                return rel <= 2 ** -20
            if t == 2:
                return err <= 2 ** -21 or rel <= 2 ** -20
            if t == 3:
                return rel <= (4.0 + 2.0 * abs(xin)) * 2 ** -23
            if t == 4:
                return err <= 2 ** -11
            if t == 5:
                return rel <= 2 ** -9
            return err <= 2 ** -10
        return near
    return _table2d("Vop1FloatUnary.cpp", "Expected", "Checked[lane] bit mask; Tolerance[col] 0 exact, else relative/absolute tolerance",
                    asserted=lambda r, c: bool((checked[r] >> c) & 1), accept=accept)


def _named(file, rule):
    names = st.names(_src(file), "Names")
    return _table2d(file, "Expected", rule, asserted=lambda r, c: c < len(names) and names[c] is not None)


def _vectors(file, offset, rule, asserted, accept):
    t = st.table(_src(file), "Vectors")

    def at(r, c):
        if r >= len(t) or offset + c >= len(t[r]):
            return None, False, None
        pin = t[r][offset + c]
        return pin, asserted(t[r], c), accept(c, pin)
    return Pins(f"{file}:Vectors[{len(t)}].expected", rule, at)


def _float16_misc():
    def matches(c, pin):
        def f(x):
            if x == pin:
                return True
            if c <= 1:
                return (x >> 16) == (pin >> 16) and nan16(x) and nan16(pin)
            return c >= 9 and nan32(x) and nan32(pin)
        return f
    return _vectors("Float16Misc.cpp", 3, "cols >= 9 skipped when a, b or expected[9] is an f32 denormal; cols 0,1 f16-NaN-tolerant, cols 9,10 f32-NaN-tolerant",
                    lambda v, c: not (c >= 9 and (denormal32(v[0]) or denormal32(v[1]) or denormal32(v[3 + 9]))), matches)


def _float16_rounding():
    def matches(c, pin):
        return lambda x: x == pin or ((x >> 16) == (pin >> 16) and nan16(x) and nan16(pin))
    return _vectors("Float16Rounding.cpp", 3, "every cell; f16 NaN-tolerant with equal high half", lambda v, c: True, matches)


def _division_helper():
    t = st.table(_src("Division.cpp"), "HelperVectors")

    def at(r, c):
        if r >= len(t) or c > 3:
            return None, False, None
        v = t[r]
        pin = v[4 + c]
        asserted = (v[0] == v[1] or v[0] == v[2]) if c <= 1 else True
        return pin, asserted, exact(pin)
    return Pins("Division.cpp:HelperVectors[256].{scale,scaleVcc,fmas,fixup}",
                "cols 0,1 only when s0 == s1 or s0 == s2; col 2 only when the host fuses FMA (asserted here); col 3 always; exact", at)


def _division_quotient():
    t = st.table(_src("Division.cpp"), "DivisionVectors")

    def at(r, c):
        if r >= len(t) or c != 0:
            return None, False, None
        return t[r][2], True, exact(t[r][2])
    return Pins("Division.cpp:DivisionVectors[256].quotient", "col 0 exact (the test runs it only on hosts that fuse FMA)", at)


def _fma_rounding():
    t = st.table(_src("FmaRounding.cpp"), "Rows")
    field = [5, 5, 6, 7, 8, 8, 9, 10, 8, 8]

    def at(r, c):
        if r >= len(t) or c >= len(field):
            return None, False, None
        pin = t[r][field[c]]
        return pin, True, exact(pin)
    return Pins("FmaRounding.cpp:Rows[64].{fused,fusedAk,fusedMk,separate,...}",
                "cols 0..3 only when the host fuses FMA (asserted here); cols 4..9 always; exact", at)


def pins(file, kernel):
    key = (file, kernel)
    if key == ("Float16ResultModifiers.cpp", "Code"):
        return _float16_result_modifiers()
    if key == ("RelativeIndexing.cpp", "Code"):
        return _table2d(file, "Expected", "every cell exact")
    if key in (("ScalarRelativeIndexing.cpp", "Code"), ("ScalarSopkMisc.cpp", "Code")):
        return _broadcast(file, "Expected", "every cell exact")
    if key in (("ScalarSop1.cpp", "Code"), ("ScalarSop2.cpp", "Code")):
        return _named(file, "columns whose Names[] entry is not nullptr; exact (ScalarSop1 masks exec-reading results with the host's lanes, all 32 here)")
    if key == ("Vop1FloatUnary.cpp", "Code"):
        return _vop1_float_unary()
    if key in (("Vop1Vop2Integer.cpp", "Code"), ("Vop3IntegerAlu.cpp", "Code")):
        return _table2d(file, "Expected", "every cell exact")
    if key in (("LdsAddtid.cpp", "Wave32Code"), ("LdsIntegerAtomics.cpp", "Wave32Code"), ("LdsReadWrite.cpp", "Wave32Code")):
        return _table2d(file, "Expected32", "every cell exact")
    if key == ("ScalarProgramFlow.cpp", "Wave32Code"):
        return _table2d(file, "Expected32", "every cell exact; cols 0..11 (CrossLane) only when the host subgroup covers the wave")
    if key == ("ScalarProgramFlow.cpp", "Wave64Code"):
        return _table2d(file, "Expected64", "every cell exact; cols 0..11 (CrossLane) only when the host subgroup covers the wave")
    if key == ("Float16Misc.cpp", "MiscCode"):
        return _float16_misc()
    if key == ("Float16Rounding.cpp", "RoundingCode"):
        return _float16_rounding()
    if key == ("Division.cpp", "HelperCode"):
        return _division_helper()
    if key == ("Division.cpp", "DivisionCode"):
        return _division_quotient()
    if key == ("FmaRounding.cpp", "FmaCode"):
        return _fma_rounding()
    return None
