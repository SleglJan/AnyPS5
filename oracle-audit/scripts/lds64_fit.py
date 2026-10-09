"""Fit the LdsAtomics64 test's CPU model to the oracle matrices measured by lds64_matrix.py.

Usage: python3 -s scripts/oracle/lds64_fit.py notes/oracle-lds64/matrix.json
Model (core/libs/prx/libSceAgcDriver/tests/execution/LdsAtomics64.cpp): min/max order the bits with OrderKey, a NaN
operand loses; cmpst stores data1 when memory equals data0 (bitwise, NaN never equal, both zeros equal); the integer
forms as in the ISA; every _rtn form returns the old value.
"""
import json
import re
import sys

SIGN = 1 << 63
MASK = (1 << 64) - 1


def nan(b): return (b & ~SIGN) > 0x7ff0000000000000
def denormal(b): return (b & 0x7ff0000000000000) == 0 and (b & 0x000fffffffffffff) != 0
def order_key(b): return (~b) & MASK if b & SIGN else b | SIGN


def minmax(old, data, maximum):
    if nan(old): return data
    if nan(data): return old
    better = order_key(data) > order_key(old) if maximum else order_key(data) < order_key(old)
    return data if better else old


def float_equal(a, b): return (a == b and not nan(a)) or ((a | b) & ~SIGN) == 0


def signed(v): return v - (1 << 64) if v & SIGN else v


INT_MODEL = {
    "add_u": lambda a, b, c: (a + b) & MASK, "sub_u": lambda a, b, c: (a - b) & MASK, "rsub_u": lambda a, b, c: (b - a) & MASK,
    "inc_u": lambda a, b, c: 0 if a >= b else a + 1, "dec_u": lambda a, b, c: b if (a == 0 or a > b) else a - 1,
    "min_i": lambda a, b, c: b if signed(b) < signed(a) else a, "max_i": lambda a, b, c: b if signed(b) > signed(a) else a,
    "min_u": lambda a, b, c: min(a, b), "max_u": lambda a, b, c: max(a, b),
    "and_b": lambda a, b, c: a & b, "or_b": lambda a, b, c: a | b, "xor_b": lambda a, b, c: a ^ b,
    "mskor_b": lambda a, b, c: (a & ~b & MASK) | c, "cmpst_b": lambda a, b, c: c if a == b else a, "wrxchg_b": lambda a, b, c: b,
}


def q(lo, hi): return lo | (hi << 32)


def classify(a, b):
    tags = []
    if nan(a) or nan(b): tags.append("nan")
    if ((a | b) & ~SIGN) == 0: tags.append("zeros")
    if denormal(a) or denormal(b): tags.append("denormal")
    return "+".join(tags) or "ordinary"


def main():
    m = json.load(open(sys.argv[1]))
    total = 0
    s = m["f64_minmax"]
    for mode, got in s["modes"].items():
        mism = {}
        examples = {}
        for (a, b), o in zip(s["rows"], got):
            cells = {"min_rtn mem": (q(o[0], o[1]), minmax(a, b, False)), "min_rtn rtn": (q(o[2], o[3]), a),
                     "max_rtn mem": (q(o[4], o[5]), minmax(a, b, True)), "max_rtn rtn": (q(o[6], o[7]), a),
                     "min mem": (q(o[8], o[9]), minmax(a, b, False)), "max mem": (q(o[10], o[11]), minmax(a, b, True))}
            for col, (hw, exp) in cells.items():
                if hw != exp:
                    key = (col, classify(a, b))
                    mism[key] = mism.get(key, 0) + 1
                    examples.setdefault(key, []).append(f"a={a:016x} b={b:016x} hw={hw:016x} model={exp:016x}")
        total += sum(mism.values())
        print(f"f64_minmax {mode:<14} mismatches {sum(mism.values())}/{len(got) * 6}")
        for key, n in sorted(mism.items()):
            print(f"   {key[0]:<12} {key[1]:<14} {n:4d}  e.g. {examples[key][0]}" + (f" | {examples[key][1]}" if len(examples[key]) > 1 else ""))
    s = m["f64_cmpst"]
    for mode, got in s["modes"].items():
        mism = {}
        examples = {}
        for (a, b, c), o in zip(s["rows"], got):
            cells = {"cmpst_rtn mem": (q(o[0], o[1]), c if float_equal(a, b) else a), "cmpst_rtn rtn": (q(o[2], o[3]), a), "cmpst mem": (q(o[4], o[5]), c if float_equal(a, b) else a)}
            for col, (hw, exp) in cells.items():
                if hw != exp:
                    key = (col, classify(a, b))
                    mism[key] = mism.get(key, 0) + 1
                    examples.setdefault(key, []).append(f"a={a:016x} b={b:016x} hw={hw:016x} model={exp:016x}")
        total += sum(mism.values())
        print(f"f64_cmpst  {mode:<14} mismatches {sum(mism.values())}/{len(got) * 3}")
        for key, n in sorted(mism.items()):
            print(f"   {key[0]:<14} {key[1]:<14} {n:4d}  e.g. {examples[key][0]}" + (f" | {examples[key][1]}" if len(examples[key]) > 1 else ""))
    s = m["int"]
    for op, got in s["ops"].items():
        kind = op.replace("ds_", "").replace("_rtn", "").replace("64", "")
        model = INT_MODEL[kind]
        mism = []
        for (a, b, c), o in zip(s["rows"], got):
            mem, rtn = q(o[0], o[1]), q(o[2], o[3])
            exp = model(a, b, c)
            if mem != exp or ("_rtn_" in op and rtn != a):
                mism.append(f"a={a:016x} b={b:016x} c={c:016x} mem={mem:016x} model={exp:016x} rtn={rtn:016x}")
        total += len(mism)
        print(f"int {op:<20} mismatches {len(mism)}/{len(got)}" + (f"  e.g. {mism[0]}" if mism else ""))
    if "f32" in m:
        s = m["f32"]
        def nan32(b): return (b & 0x7fffffff) > 0x7f800000
        def den32(b): return (b & 0x7f800000) == 0 and (b & 0x007fffff) != 0
        def key32(b): return (~b) & 0xffffffff if b & 0x80000000 else b | 0x80000000
        def mm32(old, data, maximum):
            if nan32(old): return data
            if nan32(data): return old
            better = key32(data) > key32(old) if maximum else key32(data) < key32(old)
            return data if better else old
        def eq32(a, b): return (a == b and not nan32(a)) or ((a | b) & 0x7fffffff) == 0
        def cls32(a, b):
            tags = []
            if nan32(a) or nan32(b): tags.append("nan")
            if ((a | b) & 0x7fffffff) == 0: tags.append("zeros")
            if den32(a) or den32(b): tags.append("denormal")
            return "+".join(tags) or "ordinary"
        for mode, got in s["modes"].items():
            mism = {}; examples = {}
            for (a, b, c), o in zip(s["rows"], got):
                cells = {"min_rtn mem": (o[0], mm32(a, b, False)), "min_rtn rtn": (o[1], a), "max_rtn mem": (o[2], mm32(a, b, True)), "max_rtn rtn": (o[3], a),
                         "cmpst_rtn mem": (o[4], c if eq32(a, b) else a), "cmpst_rtn rtn": (o[5], a), "min mem": (o[6], mm32(a, b, False)), "max mem": (o[7], mm32(a, b, True)),
                         "cmpst mem": (o[8], c if eq32(a, b) else a)}
                for col, (hw, exp) in cells.items():
                    if hw != exp:
                        key = (col, cls32(a, b)); mism[key] = mism.get(key, 0) + 1
                        examples.setdefault(key, []).append(f"a={a:08x} b={b:08x} hw={hw:08x} model={exp:08x}")
            total += sum(mism.values())
            print(f"f32        {mode:<14} mismatches {sum(mism.values())}/{len(got) * 9}")
            for key, n in sorted(mism.items()):
                print(f"   {key[0]:<14} {key[1]:<14} {n:4d}  e.g. {examples[key][0]}" + (f" | {examples[key][1]}" if len(examples[key]) > 1 else ""))
    print("total mismatches:", total)
    # NaN-class rules for f64 min, ieee0_kept and ieee0_flushed: result class per (old class, data class)
    def cls(b):
        if (b & ~SIGN) > 0x7ff8000000000000 or (b & ~SIGN) == 0x7ff8000000000000: return "qNaN"
        if nan(b): return "sNaN"
        if denormal(b): return "den"
        if (b & ~SIGN) == 0: return "zero"
        return "num"
    s = m["f64_minmax"]
    for mode in ("ieee0_kept", "ieee0_flushed"):
        print(f"\nf64 min_rtn memory result by operand class, {mode} (old class, data class -> result)")
        table = {}
        for (a, b), o in zip(s["rows"], s["modes"][mode]):
            hw = q(o[0], o[1])
            res = "old" if hw == a else "data" if hw == b else "old quieted" if hw == (a | 0x0008000000000000) else "data quieted" if hw == (b | 0x0008000000000000) else "old flushed" if hw == (a & SIGN) and denormal(a) else "data flushed" if hw == (b & SIGN) and denormal(b) else f"{hw:016x}"
            table.setdefault((cls(a), cls(b)), set()).add(res)
        for key in sorted(table): print(f"   old {key[0]:<5} data {key[1]:<5} -> {', '.join(sorted(table[key]))}")


if __name__ == "__main__":
    main()
