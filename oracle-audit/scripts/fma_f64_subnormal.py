"""Measure v_fma_f64, v_mul_f64 and v_add_f64 with subnormal operands and results on the oracle, and fit an exact model.

Usage: python3 -s scripts/oracle/fma_f64_subnormal.py out-dir
The third operand c comes through the oracle's --extra blob (8 bytes per lane after the padded rows).
Modes: IEEE 0/1 x f16/f64 denormals kept/flushed. Model: the exact rational result rounded once to f64 (round to nearest
even, gradual underflow); with the denormal field at 0, subnormal inputs are flushed to signed zero before and subnormal
results after.
"""
import json
import random
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "AnyPS5" / "tools" / "hw-oracle" / "hw_oracle.py"
MODES = {"ieee0_kept": (0, 3), "ieee1_kept": (1, 3), "ieee0_flushed": (0, 0), "ieee1_flushed": (1, 0)}
MIN_SUB = 0x0000000000000001
MIN_NORMAL = 0x0010000000000000
ONE = 0x3ff0000000000000


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def val(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def round_f64(x):
    if x == 0:
        return 0
    sign = x < 0
    x = abs(x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if x < Fraction(2) ** e:
        e -= 1
    e = max(e, -1022)
    q = x / Fraction(2) ** (e - 52)
    n, r = divmod(q.numerator, q.denominator)
    rem = Fraction(r, q.denominator)
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2):
        n += 1
    if n >= 2 ** 53:
        n //= 2
        e += 1
    if e > 1023:
        return bits(float("inf")) | (0x8000000000000000 if sign else 0)
    result = bits(float(Fraction(n) * Fraction(2) ** (e - 52)))
    return result | (0x8000000000000000 if sign else 0)


def is_sub(b):
    return (b & 0x7ff0000000000000) == 0 and (b & 0x000fffffffffffff) != 0


def flush(b):
    return (b & 0x8000000000000000) if is_sub(b) else b


def finite(b):
    return (b & 0x7ff0000000000000) != 0x7ff0000000000000


def model(op, a, b, c, flushed):
    if flushed:
        a, b, c = flush(a), flush(b), flush(c)
    if not all(finite(x) for x in (a, b, c)):
        return None
    fa, fb, fc = Fraction(val(a)), Fraction(val(b)), Fraction(val(c))
    exact = {"fma": fa * fb + fc, "mul": fa * fb, "add": fa + fc}[op]
    if exact == 0:
        signs = {"fma": (a ^ b) >> 63 if fa * fb != 0 else None, "mul": (a ^ b) >> 63, "add": None}[op]
        if op == "mul":
            return signs << 63
        if op == "add" or op == "fma":
            za = (a if op == "add" else ((a ^ b) & 0x8000000000000000)) >> 63
            zc = c >> 63
            return (1 << 63) if (za and zc) else 0
    r = round_f64(exact)
    if flushed and is_sub(r):
        r &= 0x8000000000000000
    return r


def double_rounded(a, b, c):
    p = round_f64(Fraction(val(a)) * Fraction(val(b)))
    if not finite(p):
        return None
    return round_f64(Fraction(val(p)) + Fraction(val(c)))


def discriminating(rnd, wanted):
    found = []
    tries = 0
    while len(found) < wanted and tries < 200000:
        tries += 1
        scale = rnd.choice([-1074, -1070, -1060, -1030, -1023, -1022, -1000, -60, 0])
        a = bits((1.0 + rnd.getrandbits(52) / 2.0 ** 52) * 2.0 ** rnd.randint(-10, 10))
        b = bits((1.0 + rnd.getrandbits(52) / 2.0 ** 52) * 2.0 ** (scale - rnd.randint(-10, 10)))
        if not finite(b) or val(b) == 0:
            continue
        product = round_f64(Fraction(val(a)) * Fraction(val(b)))
        if not finite(product) or product == 0:
            continue
        c = product ^ (1 << 63)
        c += rnd.choice([-3, -2, -1, 0, 1, 2, 3]) * rnd.choice([1, 1, 1, 2 ** rnd.randint(0, 40)])
        if not finite(c & 0x7fffffffffffffff | 0):
            continue
        single = model("fma", a, b, c, False)
        double = double_rounded(a, b, c)
        if single is not None and double is not None and single != double:
            found.append((a, b, c))
    return found


def rows():
    rnd = random.Random(981)
    out = []
    subs = [MIN_SUB, 0x0000000000000003, 0x0008000000000000, 0x000fffffffffffff, 0x0000000080000000]
    normals = [ONE, bits(1.5), bits(0.5), MIN_NORMAL, bits(2.0 ** -1021), bits(3.0), bits(2.0 ** 52), bits(1.0 + 2.0 ** -52)]
    for a in subs:
        for b in normals:
            out += [(a, b, 0), (a, b, MIN_SUB), (a, b, a), (a | (1 << 63), b, MIN_NORMAL), (a, b, bits(-1.0))]
    for a in normals:
        for b in [MIN_NORMAL, bits(2.0 ** -1022 * 1.5), bits(2.0 ** -1000), bits(2.0 ** -60)]:
            out += [(a, b, MIN_SUB), (a, b, bits(-(2.0 ** -1074) * 3)), (a, b, bits(-(2.0 ** -1022)))]
    for _ in range(128):
        a = rnd.getrandbits(52) | rnd.choice([0, MIN_NORMAL, 0x0020000000000000, 0x3ff0000000000000])
        b = rnd.getrandbits(52) | rnd.choice([0, MIN_NORMAL, 0x3fe0000000000000, 0x0030000000000000])
        c = rnd.getrandbits(52) | rnd.choice([0, MIN_NORMAL, 1 << 63, 0x8010000000000000])
        out.append((a, b, c))
    out += discriminating(rnd, 96)
    uniq = list(dict.fromkeys(out))
    while len(uniq) % 32:
        uniq.append((0, 0, 0))
    return uniq


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rs = rows()
    padded = len(rs)
    rowfile = out / "rows.txt"
    rowfile.write_text("".join(f"0x{a & 0xffffffff:08x} 0x{a >> 32:08x} 0x{b & 0xffffffff:08x} 0x{b >> 32:08x}\n" for a, b, c in rs))
    extra = out / "extra.bin"
    extra.write_bytes(b"".join(struct.pack("<Q", c) for a, b, c in rs))
    body = out / "body.s"
    body.write_text(f"""s_add_u32 s10, s4, {padded * 16}
s_addc_u32 s11, s5, 0
v_lshlrev_b32 v3, 3, v0
global_load_dwordx2 v[8:9], v3, s[10:11]
s_waitcnt vmcnt(0)
v_fma_f64 v[10:11], v[4:5], v[6:7], v[8:9]
v_mul_f64 v[12:13], v[4:5], v[6:7]
v_add_f64 v[14:15], v[4:5], v[8:9]
""")
    results = {"rows": [[a, b, c] for a, b, c in rs], "modes": {}}
    for mode, (ieee, denorm16) in MODES.items():
        cmd = [sys.executable, "-s", str(ORACLE), "--ieee", str(ieee), "--denorm32", "0", "--denorm16", str(denorm16), "--dx10-clamp", "1",
               "--round32", "0", "--round16", "0", "--fp16-overflow", "0", "--outs", "6", "--extra", str(extra), str(body), str(rowfile)]
        p = subprocess.run(cmd, cwd=ROOT / "AnyPS5", capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            print(mode, "FAILED:", p.stderr.strip().splitlines()[-3:])
            continue
        got = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if l.strip()]
        results["modes"][mode] = got
        flushed = denorm16 == 0
        mism = {"fma": 0, "mul": 0, "add": 0}
        checked = {"fma": 0, "mul": 0, "add": 0}
        examples = []
        discr = 0
        discr_double = 0
        for (a, b, c), o in zip(rs, got):
            hw = {"fma": o[0] | (o[1] << 32), "mul": o[2] | (o[3] << 32), "add": o[4] | (o[5] << 32)}
            if not flushed:
                sd = model("fma", a, b, c, False); dd = double_rounded(a, b, c)
                if sd is not None and dd is not None and sd != dd:
                    discr += 1
                    if hw["fma"] == dd:
                        discr_double += 1
            for op in mism:
                exp = model(op, a, b, c, flushed)
                if exp is None:
                    continue
                checked[op] += 1
                if hw[op] != exp:
                    mism[op] += 1
                    if len(examples) < 4:
                        examples.append(f"{op}({a:016x},{b:016x},{c:016x}) hw {hw[op]:016x} model {exp:016x}")
        print(f"{mode:<14} " + " ".join(f"{op} {mism[op]}/{checked[op]}" for op in mism) + (f"  discriminating rows {discr}, of which hardware matches the double-rounded model {discr_double}" if not flushed else "") + ("  e.g. " + " | ".join(examples) if examples else ""))
    json.dump(results, open(out / "matrix.json", "w"))
    print("rows", padded, "wrote", out / "matrix.json")


if __name__ == "__main__":
    main()
