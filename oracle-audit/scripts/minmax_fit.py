"""Fit the IEEE_MODE min/max rules to the matrices measured by minmax_matrix.py.

Usage: python3 -s scripts/oracle/minmax_fit.py notes/oracle-minmax/matrix.json

Rule: with IEEE_MODE=0 a NaN operand loses to a non-NaN one and two NaNs give the first operand; with IEEE_MODE=1 the
first signalling NaN operand (first, then second) is returned quieted, otherwise as IEEE_MODE=0; for equal zeros min gives
-0 and max +0; denormal operands are flushed to signed zero first when the type's denormal field says so. min3/max3 are two
nested pairs; med3 is min3 when any operand is NaN, else min(max(a, b), max(min(a, b), c)).
"""
import json
import struct
import sys

f64 = lambda v: struct.unpack("<d", struct.pack("<Q", v))[0]
f16 = lambda h: struct.unpack("<e", struct.pack("<H", h))[0]


def nan64(v): return (v & 0x7fffffffffffffff) > 0x7ff0000000000000
def snan64(v): return nan64(v) and not (v & 0x0008000000000000)
def nan16(h): return (h & 0x7fff) > 0x7c00
def snan16(h): return nan16(h) and not (h & 0x0200)
def flush64(v, f): return (v & 0x8000000000000000) if f and (v & 0x7ff0000000000000) == 0 and (v & 0x000fffffffffffff) else v
def flush16(h, f): return (h & 0x8000) if f and (h & 0x7c00) == 0 and (h & 0x03ff) else h


def pair(a, b, op, ieee, nan, snan, qbit, val, sign):
    if ieee:
        if snan(a): return a | qbit
        if snan(b): return b | qbit
    if nan(a): return a if nan(b) else b
    if nan(b): return a
    fa, fb = val(a), val(b)
    if fa == fb == 0.0:
        na = (a >> sign) & 1
        return (a if na else b) if op == "min" else (b if na else a)
    return (a if fa < fb else b) if op == "min" else (a if fa > fb else b)


def main():
    m = json.load(open(sys.argv[1]))
    total = 0
    s = m["f64_minmax"]
    for mode, rows in s["modes"].items():
        ieee = mode.startswith("ieee1"); fl = mode.endswith("flushed"); mism = [0, 0]
        for r, out in zip(s["rows"], rows):
            a = flush64(r[0] | (r[1] << 32), fl); b = flush64(r[2] | (r[3] << 32), fl)
            got = [out[0] | (out[1] << 32), out[2] | (out[3] << 32)]
            for i, op in enumerate(("min", "max")):
                if got[i] != pair(a, b, op, ieee, nan64, snan64, 0x0008000000000000, f64, 63): mism[i] += 1
        total += sum(mism)
        print(f"f64 {mode:<14} v_min_f64 {mism[0]}/{len(rows)}  v_max_f64 {mism[1]}/{len(rows)}")
    s = m["f16_minmax"]
    def p(a, b, op, ieee): return pair(a, b, op, ieee, nan16, snan16, 0x0200, f16, 15)
    def m3(a, b, c, op, ieee):
        if op == "min3": return p(p(a, b, "min", ieee), c, "min", ieee)
        if op == "max3": return p(p(a, b, "max", ieee), c, "max", ieee)
        if any(nan16(x) for x in (a, b, c)): return p(p(a, b, "min", ieee), c, "min", ieee)
        return p(p(a, b, "max", ieee), p(p(a, b, "min", ieee), c, "max", ieee), "min", ieee)
    for mode, rows in s["modes"].items():
        ieee = mode.startswith("ieee1"); fl = mode.endswith("flushed"); mism = [0] * 5
        for r, out in zip(s["rows"], rows):
            a, b, c = (flush16(x, fl) for x in r[:3])
            exp = [p(a, b, "min", ieee), p(a, b, "max", ieee), m3(a, b, c, "min3", ieee), m3(a, b, c, "max3", ieee), m3(a, b, c, "med3", ieee)]
            for i in range(5):
                if (out[i] & 0xffff) != exp[i]: mism[i] += 1
        total += sum(mism)
        print(f"f16 {mode:<14} " + " ".join(f"{s['ops'][i]} {mism[i]}/{len(rows)}" for i in range(5)))
    print("total mismatches:", total)


if __name__ == "__main__":
    main()
