"""Measure the hardware's sine and cosine just off the half and quarter turns: is sin(0.5 - q) the linear term in q, with which constant, up to which q?

Usage: python3 -s scripts/oracle/fold_constants.py out-dir
Rows: for q = 2^-k * (1 + m/16), k = 8..22, m = 0..15: q, 0.5 - q, 0.5 + q, 1 - q, -0.5 + q, 0.25 - q, 0.25 + q, 0.75 - q (sin and cos of each).
"""
import importlib.util
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep", HERE / "transcendental_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

FAMILIES = ["q", "0.5-q", "0.5+q", "1-q", "-0.5+q", "0.25-q", "0.25+q", "0.75-q"]


def f32(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def fl(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def mul(a_bits, b_bits):
    return f32(fl(a_bits) * fl(b_bits)) if abs(fl(a_bits) * fl(b_bits)) >= 2 ** -126 else f32(0.0)


def main():
    out = Path(sys.argv[1]).resolve()
    qs = [f32(2.0 ** -k * (1 + m / 16)) for k in range(8, 23) for m in range(16)]
    rows = []
    for q in qs:
        v = fl(q)
        rows += [q, f32(0.5 - v), f32(0.5 + v), f32(1.0 - v), f32(-0.5 + v), f32(0.25 - v), f32(0.25 + v), f32(0.75 - v)]
    while len(rows) % 64:
        rows.append(0)
    got = sweep.run("fold-constants", "v_sin_f32 v10, v4\nv_cos_f32 v11, v4\n", rows, 2, out, sweep.flags(0))
    if got is None:
        raise SystemExit("oracle failed")
    hw = {}
    for i, q in enumerate(qs):
        for j, fam in enumerate(FAMILIES):
            hw[(q, fam)] = got[i * 8 + j]
    fd5, fdb = 0x40c90fd5, 0x40c90fdb
    print("per exponent of q: how many of the 16 hardware results equal the candidate (sin families use v_sin, cos families v_cos); sign-folded to the q-side")
    print(f"{'k':>2} {'fam':>7} | {'=sin(q)':>7} {'=q*fd5':>7} {'=q*fdb':>7} {'maxULP vs sin(q)':>17}")
    for fam in FAMILIES[1:]:
        for k in range(8, 23):
            eq_sinq = eq5 = eqb = 0; worst = 0
            for m in range(16):
                q = f32(2.0 ** -k * (1 + m / 16)); v = fl(q)
                col = 1 if fam in ("0.25-q", "0.25+q", "0.75-q") else 0
                r = hw[(q, fam)][col] & 0x7fffffff
                s = hw[(q, "q")][0] & 0x7fffffff
                eq_sinq += r == s; eq5 += r == mul(q, fd5); eqb += r == mul(q, fdb)
                worst = max(worst, abs(r - s))
            print(f"{k:>2} {fam:>7} | {eq_sinq:>7} {eq5:>7} {eqb:>7} {worst:>17}")
    print("direct: sin(q) == q*fd5 per k:", [sum(hw[(f32(2.0 ** -k * (1 + m / 16)), 'q')][0] == mul(f32(2.0 ** -k * (1 + m / 16)), fd5) for m in range(16)) for k in range(8, 23)])
    print("direct: sin(q) == q*fdb per k:", [sum(hw[(f32(2.0 ** -k * (1 + m / 16)), 'q')][0] == mul(f32(2.0 ** -k * (1 + m / 16)), fdb) for m in range(16)) for k in range(8, 23)])


if __name__ == "__main__":
    main()
