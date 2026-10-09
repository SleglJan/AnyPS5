"""Per-input comparison of two host sweeps against the oracle. Usage: compare_variants.py <tagA> <tagB> (hosts-<tag>/ dirs under notes/oracle-transcendentals)."""
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

R = Path("/home/slegl/PycharmProjects/anyps5")
D = R / "notes/oracle-transcendentals"
spec = importlib.util.spec_from_file_location("h", R / "scripts/oracle/transcendental_hosts.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def load(tag, setname, gpu):
    return [[int(x, 16) for x in l.split()] for l in (D / f"hosts-{tag}" / setname / f"{gpu}-results.txt").read_text().splitlines() if l.strip()]


def distance(host, hw, width):
    m = 0xffff if width == 16 else 0xffffffff
    host &= m; hw &= m
    if host == hw: return 0, "same"
    ch, cw = h.classify(host, width), h.classify(hw, width)
    if ch == cw == "nan": return 0, "same"
    if ch != cw and ("nan" in (ch, cw) or "inf" in (ch, cw)): return None, "class"
    return abs(h.ordered(host, width) - h.ordered(hw, width)), "ulp"


def reduced(x):
    v = f32(x)
    if not math.isfinite(v): return None
    return abs(v - round(v)) if abs(v) < 2 ** 23 else 0.0


def region(r):
    if r is None: return "special"
    if r == 0: return "cycle 0"
    if r < 2 ** -14: return "|r| < 2^-14"
    if r < 0.125: return "2^-14 <= |r| < 0.125"
    if r <= 0.25: return "0.125 <= |r| <= 0.25"
    if r < 0.5 - 2 ** -14: return "0.25 < |r| < 0.5 - 2^-14"
    return "|r| within 2^-14 of 0.5"


def main():
    a, b = sys.argv[1], sys.argv[2]
    sets = [("f32-flushed", 32, json.load(open(D / "f32-inputs.json")), h.load_oracle(D / "f32-flushed.txt"), [(5, "v_sin_f32"), (6, "v_cos_f32")]),
            ("f16", 16, list(range(65536)), h.load_oracle(D / "f16-title.txt"), [(5, "v_sin_f16"), (6, "v_cos_f16")])]
    for setname, width, inputs, oracle, cols in sets:
        for gpu in ("Ryzen", "NVIDIA"):
            ra, rb = load(a, setname, gpu), load(b, setname, gpu)
            for col, op in cols:
                better = worse = same = 0; worse_max = 0; worse_regions = {}; better_regions = {}; a_class_r = []; b_class = 0
                worst_worse = None
                for i, x in enumerate(inputs):
                    if i >= len(oracle): break
                    hw = oracle[i][col]
                    da, ka = distance(ra[i][col], hw, width); db, kb = distance(rb[i][col], hw, width)
                    r = reduced(x) if width == 32 else None
                    if ka == "class": a_class_r.append(r)
                    if kb == "class": b_class += 1
                    sa = float("inf") if da is None else da; sb = float("inf") if db is None else db
                    if sa == sb: same += 1
                    elif sb < sa: better += 1; better_regions[region(r)] = better_regions.get(region(r), 0) + 1
                    else:
                        worse += 1; worse_regions[region(r)] = worse_regions.get(region(r), 0) + 1
                        if sb > worse_max: worse_max = sb; worst_worse = (x, ra[i][col], rb[i][col], hw, r)
                print(f"{gpu} {op}: {b} vs {a}: better {better}, worse {worse}, same {same}; worst of the worse: {worse_max} ULP {worst_worse and 'in=%08x a=%08x b=%08x hw=%08x |r|=%s' % worst_worse}")
                if worse_regions: print(f"    worse by region: {worse_regions}")
                if better_regions: print(f"    better by region: {better_regions}")
                if a_class_r and width == 32:
                    finite = [r for r in a_class_r if r is not None]
                    print(f"    {a} class mismatches: {len(a_class_r)}; max |r| among them {max(finite) if finite else None}; {b} class mismatches: {b_class}")


if __name__ == "__main__":
    main()
