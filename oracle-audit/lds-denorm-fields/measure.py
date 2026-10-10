"""Measure ds_min/max/cmpst(_rtn)_f32/_f64 on the oracle for every value 0-3 of the type's denormal field and compare with two rules.

Usage: python3 -s measure.py out-dir [oracle-worktree]
Bodies, operand sets (14 x 14 per type) and run helper come from scripts/oracle/lds64_matrix.py. The f32 forms run with
--denorm32 = field and --denorm16 3; the f64 forms with --denorm16 = field and --denorm32 0 (as the PR's measurement did).
Rules compared, both otherwise the PR's (signalling NaN data stored quieted, else signalling NaN old quieted, else quiet NaN
loses, else ordered compare with -0 below +0; cmpst compares the flushed bits; returned = old):
  pr:       inputs flushed when field == 0
  reviewer: inputs flushed when (field & 1) == 0
Only the 196 real operand pairs are counted (the padding lanes are left out).
"""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

SCRIPTS = Path("/home/slegl/PycharmProjects/anyps5/scripts/oracle")
spec = importlib.util.spec_from_file_location("lds64_matrix", SCRIPTS / "lds64_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class Fmt:
    def __init__(self, bits, exp_mask, quiet):
        self.sign = 1 << (bits - 1)
        self.exp = exp_mask
        self.man = self.sign - 1 - exp_mask
        self.quiet = quiet
        self.mask = (1 << bits) - 1

    def nan(self, b): return (b & ~self.sign & self.mask) > self.exp
    def snan(self, b): return self.nan(b) and not (b & self.quiet)
    def den(self, b): return (b & self.exp) == 0 and (b & self.man) != 0
    def flush(self, b): return b & self.sign if self.den(b) else b
    def key(self, b): return (~b) & self.mask if b & self.sign else b | self.sign

    def minmax(self, old, data, maximum, flush):
        a, b = (self.flush(old), self.flush(data)) if flush else (old, data)
        if self.snan(b): return b | self.quiet
        if self.snan(a): return a | self.quiet
        if self.nan(a): return b
        if self.nan(b): return a
        better = self.key(b) > self.key(a) if maximum else self.key(b) < self.key(a)
        return b if better else a

    def cmpst(self, old, compare, data, flush):
        a, b = (self.flush(old), self.flush(compare)) if flush else (old, compare)
        equal = (a == b and not self.nan(a)) or ((a | b) & ~self.sign & self.mask) == 0
        return data if equal else old


F32 = Fmt(32, 0x7f800000, 0x00400000)
F64 = Fmt(64, 0x7ff0000000000000, 0x0008000000000000)
RULES = {"pr": lambda field: field == 0, "reviewer": lambda field: (field & 1) == 0}


def q(lo, hi): return lo | (hi << 32)


def expected_f32(a, b, c, flush):
    return [F32.minmax(a, b, False, flush), a, F32.minmax(a, b, True, flush), a, F32.cmpst(a, b, c, flush), a,
            F32.minmax(a, b, False, flush), F32.minmax(a, b, True, flush), F32.cmpst(a, b, c, flush)]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2:
        matrix.ORACLE = Path(sys.argv[2]) / "tools" / "hw-oracle" / "hw_oracle.py"
    real = len(matrix.F64) * len(matrix.F64)
    data = {"oracle": str(matrix.ORACLE), "f32": {}, "f64_minmax": {}, "f64_cmpst": {}}
    rows32 = matrix.pad((a, b, matrix.MARKER & 0xffffffff) for a, b in itertools.product(matrix.F32, matrix.F32))
    body32 = matrix.prologue(len(rows32), True)
    for op, mem, rtn, three in (("ds_min_rtn_f32", "v10", "v11", False), ("ds_max_rtn_f32", "v12", "v13", False), ("ds_cmpst_rtn_f32", "v14", "v15", True),
                                ("ds_min_f32", "v16", "", False), ("ds_max_f32", "v17", "", False), ("ds_cmpst_f32", "v18", "", True)):
        operands = "v3, v6" + (", v8" if three else "")
        body32 += f"ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\n" + (f"{op} {rtn}, {operands}" if rtn else f"{op} {operands}") + "\ns_waitcnt lgkmcnt(0)\n"
        body32 += f"ds_read_b32 {mem}, v3\ns_waitcnt lgkmcnt(0)\n"
    rows64 = matrix.pad((a, b, 0) for a, b in itertools.product(matrix.F64, matrix.F64))
    body64 = matrix.prologue(len(rows64), False)
    body64 += matrix.op_body("ds_min_rtn_f64", "v[12:13]", "v[10:11]") + matrix.op_body("ds_max_rtn_f64", "v[16:17]", "v[14:15]")
    body64 += matrix.op_body("ds_min_f64", "", "v[18:19]") + matrix.op_body("ds_max_f64", "", "v[20:21]")
    rowsc = matrix.pad((a, b, matrix.MARKER) for a, b in itertools.product(matrix.F64, matrix.F64))
    bodyc = matrix.prologue(len(rowsc), True)
    bodyc += matrix.op_body("ds_cmpst_rtn_f64", "v[12:13]", "v[10:11]") + matrix.op_body("ds_cmpst_f64", "", "v[14:15]")
    data["f32"]["rows"] = [list(r) for r in rows32]
    data["f64_minmax"]["rows"] = [list(r) for r in rows64]
    data["f64_cmpst"]["rows"] = [list(r) for r in rowsc]
    summary = []
    for field in range(4):
        for ieee in (0, 1):
            mode = f"ieee{ieee}_field{field}"
            got32 = matrix.run(matrix.flags(ieee, 3, field), body32, rows32, 9, True, out, f"f32-{mode}", True)
            got64 = matrix.run(matrix.flags(ieee, field, 0), body64, rows64, 12, None, out, f"f64_minmax-{mode}", True)
            gotc = matrix.run(matrix.flags(ieee, field, 0), bodyc, rowsc, 6, True, out, f"f64_cmpst-{mode}", True)
            if not (got32 and got64 and gotc):
                summary.append({"mode": mode, "error": "oracle run failed"})
                continue
            data["f32"][mode], data["f64_minmax"][mode], data["f64_cmpst"][mode] = got32, got64, gotc
            line = {"mode": mode, "field": field, "ieee": ieee, "results_f32": real * 9, "results_f64": real * 9}
            for rule, flushes in RULES.items():
                flush = flushes(field)
                m32, m64, examples = 0, 0, []
                for (a, b, c), o in list(zip(rows32, got32))[:real]:
                    for col, (hw, exp) in enumerate(zip(o, expected_f32(a, b, c, flush))):
                        if hw != exp:
                            m32 += 1
                            examples.append(f"f32 col {col} a={a:08x} b={b:08x} hw={hw:08x} rule={exp:08x}")
                for (a, b, c), o in list(zip(rows64, got64))[:real]:
                    exp = [F64.minmax(a, b, False, flush), a, F64.minmax(a, b, True, flush), a, F64.minmax(a, b, False, flush), F64.minmax(a, b, True, flush)]
                    for col, e in enumerate(exp):
                        hw = q(o[2 * col], o[2 * col + 1])
                        if hw != e:
                            m64 += 1
                            examples.append(f"f64 minmax col {col} a={a:016x} b={b:016x} hw={hw:016x} rule={e:016x}")
                for (a, b, c), o in list(zip(rowsc, gotc))[:real]:
                    exp = [F64.cmpst(a, b, c, flush), a, F64.cmpst(a, b, c, flush)]
                    for col, e in enumerate(exp):
                        hw = q(o[2 * col], o[2 * col + 1])
                        if hw != e:
                            m64 += 1
                            examples.append(f"f64 cmpst col {col} a={a:016x} b={b:016x} hw={hw:016x} rule={e:016x}")
                line[f"{rule}_f32"] = m32
                line[f"{rule}_f64"] = m64
                line[f"{rule}_examples"] = examples[:6]
            summary.append(line)
            print(f"{mode}: pr f32 {line['pr_f32']}/{real * 9} f64 {line['pr_f64']}/{real * 9} | reviewer f32 {line['reviewer_f32']}/{real * 9} f64 {line['reviewer_f64']}/{real * 9}", flush=True)
    data["summary"] = summary
    json.dump(data, open(out / "matrix.json", "w"))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print("wrote", out / "matrix.json")


if __name__ == "__main__":
    main()
