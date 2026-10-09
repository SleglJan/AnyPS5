"""Measure v_log_f32 around 1.0 on the oracle and replay the same rows through the recompiler on both hosts.

Usage: python3 -s scripts/oracle/log_near_one.py out-dir
Rows: x = 1 + s * 2^-k * (1 + m/64) for s = +1, -1; k = 1..24; m = 0..63 (rounded to f32), plus x = 1 and the f32 neighbours of 1.
Compares the hardware with the correctly rounded log2 (double precision) and with fl((x - 1) * C) for C near 1/ln2, per exponent of |x - 1|.
"""
import importlib.util
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep", HERE / "transcendental_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
spec2 = importlib.util.spec_from_file_location("hosts", HERE / "transcendental_hosts.py")
hosts = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(hosts)


def f32(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def fl(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def rn_log2(x):
    return f32(math.log2(x))


def ulp(a, b):
    return abs(hosts.ordered(a, 32) - hosts.ordered(b, 32))


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = [0x3f800000, 0x3f800001, 0x3f7fffff, 0x3f800002, 0x3f7ffffe]
    for s in (1.0, -1.0):
        for k in range(1, 25):
            for m in range(64):
                rows.append(f32(1.0 + s * 2.0 ** -k * (1 + m / 64)))
    rows = sorted(set(rows))
    while len(rows) % 64:
        rows.append(0x3f800000)
    body = "v_log_f32 v10, v4\n"
    hw = sweep.run("log-near-one", body, rows, 1, out, sweep.flags(0))
    if hw is None:
        raise SystemExit("oracle failed")
    hw = [r[0] for r in hw]
    print(f"{len(rows)} rows on the oracle")
    one_ln2 = 0x3fb8aa3b
    buckets = {}
    for x, h in zip(rows, hw):
        v = fl(x)
        if v == 1.0: continue
        d = v - 1.0
        k = -math.frexp(abs(d))[1] + 1
        b = buckets.setdefault((k, d > 0), [0, 0, 0, 0, 0])
        b[0] += 1
        b[1] += h == rn_log2(v)
        b[2] += ulp(h, rn_log2(v))
        b[3] = max(b[3], ulp(h, rn_log2(v)))
        b[4] += h == f32(d * fl(one_ln2))
    print("hardware v_log_f32 near 1 per exponent of |x-1| (k: |x-1| in [2^-k, 2^-k+1)): n, == RN(log2 x), mean/max ULP vs RN, == fl((x-1)/ln2)")
    for (k, pos) in sorted(buckets):
        n, eq, tot, mx, lin = buckets[(k, pos)]
        print(f"  k={k:2d} {'1+d' if pos else '1-d'}: n={n:3d} exact={eq:3d} meanULP={tot / n:6.2f} maxULP={mx:4d} linear={lin:3d}")
    # hosts
    for gpu in ("Ryzen", "NVIDIA"):
        got = hosts.host_sweep(gpu, body, rows, hosts.MODES32["flushed"], out / "hosts" / (sys.argv[2] if len(sys.argv) > 2 else "main"))
        res = [r[0] for r in got]
        (out / "hosts" / (sys.argv[2] if len(sys.argv) > 2 else "main") / f"{gpu}-results.txt").write_text("".join(f"{v:08x}\n" for v in res))
        hb = {}
        for x, h, g in zip(rows, hw, res):
            v = fl(x)
            if v == 1.0: continue
            d = v - 1.0; k = -math.frexp(abs(d))[1] + 1
            b = hb.setdefault(k, [0, 0, 0, None])
            b[0] += 1; b[1] += g == h
            u = ulp(g, h) if hosts.classify(g, 32) == hosts.classify(h, 32) else 10 ** 9
            if u > b[2]: b[2] = u; b[3] = (x, g, h)
        print(f"{gpu} (main lowering) vs hardware per exponent of |x-1|: n, exact, max ULP, worst")
        for k in sorted(hb):
            n, eq, mx, w = hb[k]
            print(f"  k={k:2d}: n={n:3d} exact={eq:3d} maxULP={mx:10d} worst={'in=%08x host=%08x hw=%08x' % w if w else ''}")


if __name__ == "__main__":
    main()
