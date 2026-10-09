"""Measure the 64-bit LDS atomics on the oracle: f64 min/max and cmpst matrices, the integer forms on edge pairs.

Usage: python3 -s scripts/oracle/lds64_matrix.py out-dir [set ...]
Sets: f64_minmax (14 x 14 operands, 4 modes), f64_cmpst (14 x 14, marker as data1, 4 modes), int (29 forms, edge pairs).
Each lane owns the LDS slot lane * 8: the slot is written with a (v4:5), the op runs with b (v6:7) and, for cmpst and
mskor, c (8 bytes per lane after the padded rows, loaded into v8:9); the slot is read back into v10:11 and the
returned value lands in v12:13. Non-returning forms store their slot into v14:15.
"""
import itertools
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "AnyPS5" / "tools" / "hw-oracle" / "hw_oracle.py"

F64 = [0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000, 0x7ff0000000000000, 0xfff0000000000000,
       0x0000000000000001, 0x8000000000000001, 0x7ff8000000000000, 0x7ff8000000000123, 0xfff8000000000000,
       0x7ff0000000000001, 0x7ff0000000000123, 0xfff0000000000001]
MARKER = 0x4142434445464748
F32 = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x00000001, 0x80000001, 0x7fc00000, 0x7fc00123, 0xffc00000,
       0x7f800001, 0x7f800123, 0xff800001]
MODES32 = {"ieee0_kept": (0, 3), "ieee1_kept": (1, 3), "ieee0_flushed": (0, 0), "ieee1_flushed": (1, 0)}
MODES = {"ieee0_kept": (0, 3), "ieee1_kept": (1, 3), "ieee0_flushed": (0, 0), "ieee1_flushed": (1, 0)}
INT_PAIRS = [
    (0, 0), (0, 1), (1, 0), (0xffffffffffffffff, 1), (1, 0xffffffffffffffff), (0xffffffff, 1), (0xffffffff, 0xffffffff),
    (0x100000000, 1), (0x100000000, 0x100000000), (0x7fffffffffffffff, 1), (0x8000000000000000, 1), (0x8000000000000000, 0x8000000000000000),
    (0x7fffffffffffffff, 0x8000000000000000), (0x8000000000000000, 0x7fffffffffffffff), (0xfffffffffffffffe, 0xffffffffffffffff),
    (0xffffffffffffffff, 0xfffffffffffffffe), (5, 5), (5, 4), (4, 5), (0, 0xffffffffffffffff), (0xffffffffffffffff, 0),
    (0x123456789abcdef0, 0x0fedcba987654321), (0x0fedcba987654321, 0x123456789abcdef0), (0xdeadbeefcafebabe, 0x00000000ffffffff),
    (0x00000000ffffffff, 0xdeadbeefcafebabe), (0xffffffff00000000, 0x00000000ffffffff), (6, 0x100000000), (0x100000000, 6),
    (0x8000000000000001, 0xffffffffffffffff), (3, 2), (2, 3), (0xaaaaaaaaaaaaaaaa, 0x5555555555555555),
]
INT_OPS = ["ds_add_u64", "ds_sub_u64", "ds_rsub_u64", "ds_inc_u64", "ds_dec_u64", "ds_min_i64", "ds_max_i64", "ds_min_u64", "ds_max_u64",
           "ds_and_b64", "ds_or_b64", "ds_xor_b64", "ds_mskor_b64", "ds_cmpst_b64",
           "ds_add_rtn_u64", "ds_sub_rtn_u64", "ds_rsub_rtn_u64", "ds_inc_rtn_u64", "ds_dec_rtn_u64", "ds_min_rtn_i64", "ds_max_rtn_i64",
           "ds_min_rtn_u64", "ds_max_rtn_u64", "ds_and_rtn_b64", "ds_or_rtn_b64", "ds_xor_rtn_b64", "ds_mskor_rtn_b64", "ds_wrxchg_rtn_b64",
           "ds_cmpst_rtn_b64"]
THREE_OPERAND = ("mskor", "cmpst")
WAVE = 64


def pad(rows):
    rows = list(rows)
    while len(rows) % WAVE:
        rows.append((0, 0, 0))
    return rows


def prologue(padded, with_c):
    text = "v_lshlrev_b32 v3, 3, v0\n"
    if with_c:
        text += f"s_add_u32 s10, s4, {padded * 16}\ns_addc_u32 s11, s5, 0\nglobal_load_dwordx2 v[8:9], v3, s[10:11]\ns_waitcnt vmcnt(0)\n"
    return text


def op_body(op, rtn_regs, mem_regs):
    three = any(k in op for k in THREE_OPERAND)
    operands = "v3, v[6:7]" + (", v[8:9]" if three else "")
    body = "ds_write_b64 v3, v[4:5]\ns_waitcnt lgkmcnt(0)\n"
    body += (f"{op} {rtn_regs}, {operands}\n" if "_rtn_" in op else f"{op} {operands}\n") + "s_waitcnt lgkmcnt(0)\n"
    body += f"ds_read_b64 {mem_regs}, v3\ns_waitcnt lgkmcnt(0)\n"
    return body


def run(mode_flags, body, rows, outs, extra, out, name, wave64):
    body_file = out / f"{name}-body.s"
    rows_file = out / f"{name}-rows.txt"
    body_file.write_text(body)
    rows_file.write_text("".join(f"0x{a & 0xffffffff:08x} 0x{a >> 32:08x} 0x{b & 0xffffffff:08x} 0x{b >> 32:08x}\n" for a, b, c in rows))
    cmd = [sys.executable, "-s", str(ORACLE)] + mode_flags + ["--outs", str(outs), str(body_file), str(rows_file)]
    if extra is not None:
        extra_file = out / f"{name}-extra.bin"
        extra_file.write_bytes(b"".join(struct.pack("<Q", c) for a, b, c in rows))
        cmd += ["--extra", str(extra_file)]
    if wave64:
        cmd.append("--wave64")
    p = subprocess.run(cmd, cwd=ROOT / "AnyPS5", capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        print(name, "FAILED:", "\n".join(p.stderr.strip().splitlines()[-4:]), flush=True)
        return None
    (out / f"{name}.txt").write_text(p.stdout)
    return [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if l.strip()]


def flags(ieee, denorm16, denorm32=0):
    return ["--ieee", str(ieee), "--denorm32", str(denorm32), "--denorm16", str(denorm16), "--dx10-clamp", "1", "--round32", "0", "--round16", "0", "--fp16-overflow", "0"]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[2:]) or {"f64_minmax", "f64_cmpst", "int", "f32"}
    wave64 = WAVE == 64
    matrix = {}
    if "f64_minmax" in wanted:
        rows = pad((a, b, 0) for a, b in itertools.product(F64, F64))
        body = prologue(len(rows), False)
        body += op_body("ds_min_rtn_f64", "v[12:13]", "v[10:11]") + op_body("ds_max_rtn_f64", "v[16:17]", "v[14:15]")
        body += op_body("ds_min_f64", "", "v[18:19]") + op_body("ds_max_f64", "", "v[20:21]")
        matrix["f64_minmax"] = {"rows": [[a, b] for a, b, c in rows], "columns": ["min_rtn mem", "min_rtn rtn", "max_rtn mem", "max_rtn rtn", "min mem", "max mem"], "modes": {}}
        for mode, (ieee, d16) in MODES.items():
            got = run(flags(ieee, d16), body, rows, 12, None, out, f"f64_minmax-{mode}", wave64)
            if got:
                matrix["f64_minmax"]["modes"][mode] = got
                print(f"f64_minmax {mode}: {len(got)} rows", flush=True)
    if "f64_cmpst" in wanted:
        rows = pad((a, b, MARKER) for a, b in itertools.product(F64, F64))
        body = prologue(len(rows), True)
        body += op_body("ds_cmpst_rtn_f64", "v[12:13]", "v[10:11]") + op_body("ds_cmpst_f64", "", "v[14:15]")
        matrix["f64_cmpst"] = {"rows": [[a, b, c] for a, b, c in rows], "columns": ["cmpst_rtn mem", "cmpst_rtn rtn", "cmpst mem"], "modes": {}}
        for mode, (ieee, d16) in MODES.items():
            got = run(flags(ieee, d16), body, rows, 6, True, out, f"f64_cmpst-{mode}", wave64)
            if got:
                matrix["f64_cmpst"]["modes"][mode] = got
                print(f"f64_cmpst {mode}: {len(got)} rows", flush=True)
    if "int" in wanted:
        rows = pad((a, b, MARKER) for a, b in INT_PAIRS)
        matrix["int"] = {"rows": [[a, b, c] for a, b, c in rows], "ops": {}}
        for op in INT_OPS:
            three = any(k in op for k in THREE_OPERAND)
            body = prologue(len(rows), three) + op_body(op, "v[12:13]", "v[10:11]")
            got = run(flags(0, 3), body, rows, 4, True if three else None, out, f"int-{op}", wave64)
            if got:
                matrix["int"]["ops"][op] = got
        print(f"int: {len(matrix['int']['ops'])} of {len(INT_OPS)} forms ran", flush=True)
    if "f32" in wanted:
        rows = pad((a, b, MARKER & 0xffffffff) for a, b in itertools.product(F32, F32))
        body = prologue(len(rows), True)
        body += ("ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_min_rtn_f32 v11, v3, v6\ns_waitcnt lgkmcnt(0)\nds_read_b32 v10, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_max_rtn_f32 v13, v3, v6\ns_waitcnt lgkmcnt(0)\nds_read_b32 v12, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_cmpst_rtn_f32 v15, v3, v6, v8\ns_waitcnt lgkmcnt(0)\nds_read_b32 v14, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_min_f32 v3, v6\ns_waitcnt lgkmcnt(0)\nds_read_b32 v16, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_max_f32 v3, v6\ns_waitcnt lgkmcnt(0)\nds_read_b32 v17, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_cmpst_f32 v3, v6, v8\ns_waitcnt lgkmcnt(0)\nds_read_b32 v18, v3\ns_waitcnt lgkmcnt(0)\n")
        matrix["f32"] = {"rows": [[a, b, c] for a, b, c in rows], "columns": ["min_rtn mem", "min_rtn rtn", "max_rtn mem", "max_rtn rtn", "cmpst_rtn mem", "cmpst_rtn rtn", "min mem", "max mem", "cmpst mem"], "modes": {}}
        for mode, (ieee, d32) in MODES32.items():
            got = run(flags(ieee, 3, d32), body, rows, 9, True, out, f"f32-{mode}", wave64)
            if got:
                matrix["f32"]["modes"][mode] = got
                print(f"f32 {mode}: {len(got)} rows", flush=True)
    json.dump(matrix, open(out / "matrix.json", "w"))
    print("wrote", out / "matrix.json")


if __name__ == "__main__":
    main()
