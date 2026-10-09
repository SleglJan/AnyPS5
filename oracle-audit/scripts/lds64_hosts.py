"""Run the 64-bit LDS atomic bodies through the recompiler on both hosts and compare with the oracle.

Usage: python3 -s scripts/oracle/lds64_hosts.py notes/oracle-lds64 [set ...]
Sets: f64_minmax f64_cmpst f32 int contention. Each body is wrapped in a host kernel (buffer loads from the input V#
in s[0:3], stores to the output V# in s[4:7], as the execution tests do), assembled with llvm-mc for gfx1030, and run by
build/tests/agc_driver_kernel_replay_tests (REPLAY_LDS 1024 dwords = the oracle's 4 KiB, REPLAY_MODE per float mode). Cells are compared bit for bit
with the oracle outputs stored by lds64_matrix.py and lds64_contention.py.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARNESS = ROOT / "build" / "tests" / "agc_driver_kernel_replay_tests"
HEX_ROW = re.compile(r"^[0-9a-f]{8}( [0-9a-f]{8})*$")


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = load("lds64_matrix")
contention = load("lds64_contention")

MODES = {"ieee0_kept": "0xf0,1,0,0", "ieee1_kept": "0xf0,1,1,0", "ieee0_flushed": "0x30,1,0,0", "ieee1_flushed": "0x30,1,1,0"}
MODES32 = {"ieee0_kept": "0xf0,1,0,0", "ieee1_kept": "0xf0,1,1,0", "ieee0_flushed": "0xc0,1,0,0", "ieee1_flushed": "0xc0,1,1,0"}


def host_kernel(body, inputs, with_c):
    shift = {4: 4, 8: 5}[inputs]
    text = f"v_lshlrev_b32 v1, {shift}, v0\nv_lshlrev_b32 v2, 6, v0\nbuffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen\n"
    if with_c:
        text += "buffer_load_dwordx2 v[8:9], v1, s[0:3], 0 offen offset:16\n"
    text += "s_waitcnt vmcnt(0)\n" + "".join(f"v_mov_b32 v{r}, 0\n" for r in range(10, 26))
    text += body
    text += "buffer_store_dwordx4 v[10:13], v2, s[4:7], 0 offen\nbuffer_store_dwordx4 v[14:17], v2, s[4:7], 0 offen offset:16\n"
    text += "buffer_store_dwordx4 v[18:21], v2, s[4:7], 0 offen offset:32\nbuffer_store_dwordx4 v[22:25], v2, s[4:7], 0 offen offset:48\ns_endpgm\n"
    return text


def assemble(text, wave64):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "k.s"
        obj = Path(tmp) / "k.o"
        raw = Path(tmp) / "k.bin"
        src.write_text(text)
        mattr = ["-mattr=+wavefrontsize64"] if wave64 else ["-mattr=+wavefrontsize32"]
        subprocess.run(["llvm-mc", "-triple=amdgcn--amdpal", "-mcpu=gfx1030", *mattr, "-filetype=obj", str(src), "-o", str(obj)], check=True, capture_output=True, text=True)
        subprocess.run(["llvm-objcopy", "-O", "binary", "--only-section=.text", str(obj), str(raw)], check=True)
        data = raw.read_bytes()
    return [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]


def host_run(gpu, words, rows, inputs, threads, mode, wave64, work, lds=1024):
    work.mkdir(parents=True, exist_ok=True)
    (work / "code.txt").write_text("\n".join(f"0x{w:08x}" for w in words) + "\n")
    (work / "rows.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in r) + "\n" for r in rows))
    env = dict(os.environ, REPLAY_CODE=str(work / "code.txt"), REPLAY_ROWS=str(work / "rows.txt"), REPLAY_INPUTS=str(inputs), REPLAY_RESULTS="16",
               REPLAY_THREADS=str(threads), REPLAY_WAVE="64" if wave64 else "32", REPLAY_LDS=str(lds), REPLAY_MODE=mode, ANYPS5_GPU=gpu)
    p = subprocess.run([str(HARNESS)], env=env, capture_output=True, text=True, timeout=300)
    out = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if HEX_ROW.match(l.strip())]
    if p.returncode != 0 or len(out) != len(rows):
        return None, (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or ["no output"])[-1][:200]
    return out, ""


def split_rows(rows3, with_c):
    if with_c:
        return [[a & 0xffffffff, a >> 32, b & 0xffffffff, b >> 32, c & 0xffffffff, c >> 32, 0, 0] for a, b, c in rows3]
    return [[a & 0xffffffff, a >> 32, b & 0xffffffff, b >> 32] for a, b, *_ in rows3]


def compare(name, host, oracle, columns):
    equal = sum(1 for h, o in zip(host, oracle) for c in range(columns) if h[c] == o[c])
    total = len(oracle) * columns
    examples = [(i, c, f"{h[c]:08x}", f"{o[c]:08x}") for i, (h, o) in enumerate(zip(host, oracle)) for c in range(columns) if h[c] != o[c]][:3]
    return total - equal, total, examples


def main():
    out = Path(sys.argv[1]).resolve()
    wanted = set(sys.argv[2:]) or {"f64_minmax", "f64_cmpst", "f32", "int", "contention"}
    m = json.load(open(out / "matrix.json"))
    report = json.load(open(out / "hosts-report.json")) if (out / "hosts-report.json").exists() else {}
    work = out / "hosts"
    for gpu in ("Ryzen", "NVIDIA"):
        if "f64_minmax" in wanted:
            rows3 = [(a, b, 0) for a, b in m["f64_minmax"]["rows"]]
            body = "v_lshlrev_b32 v3, 3, v0\n" + matrix.op_body("ds_min_rtn_f64", "v[12:13]", "v[10:11]") + matrix.op_body("ds_max_rtn_f64", "v[16:17]", "v[14:15]") \
                + matrix.op_body("ds_min_f64", "", "v[18:19]") + matrix.op_body("ds_max_f64", "", "v[20:21]")
            words = assemble(host_kernel(body, 4, False), True)
            for mode, flags in MODES.items():
                got, err = host_run(gpu, words, split_rows(rows3, False), 4, len(rows3), flags, True, work / f"f64_minmax-{mode}-{gpu}")
                key = f"{gpu}/f64_minmax/{mode}"
                report[key] = {"error": err} if got is None else dict(zip(("differs", "cells", "examples"), compare(key, got, m["f64_minmax"]["modes"][mode], 12)))
                print(key, report[key], flush=True)
        if "f64_cmpst" in wanted:
            rows3 = [tuple(r) for r in m["f64_cmpst"]["rows"]]
            body = "v_lshlrev_b32 v3, 3, v0\n" + matrix.op_body("ds_cmpst_rtn_f64", "v[12:13]", "v[10:11]") + matrix.op_body("ds_cmpst_f64", "", "v[14:15]")
            words = assemble(host_kernel(body, 8, True), True)
            for mode, flags in MODES.items():
                got, err = host_run(gpu, words, split_rows(rows3, True), 8, len(rows3), flags, True, work / f"f64_cmpst-{mode}-{gpu}")
                key = f"{gpu}/f64_cmpst/{mode}"
                report[key] = {"error": err} if got is None else dict(zip(("differs", "cells", "examples"), compare(key, got, m["f64_cmpst"]["modes"][mode], 6)))
                print(key, report[key], flush=True)
        if "f32" in wanted:
            rows3 = [tuple(r) for r in m["f32"]["rows"]]
            body = "v_lshlrev_b32 v3, 3, v0\n" + ("ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_min_rtn_f32 v11, v3, v6\ns_waitcnt lgkmcnt(0)\nds_read_b32 v10, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_max_rtn_f32 v13, v3, v6\ns_waitcnt lgkmcnt(0)\nds_read_b32 v12, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_cmpst_rtn_f32 v15, v3, v6, v8\ns_waitcnt lgkmcnt(0)\nds_read_b32 v14, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_min_f32 v3, v6\ns_waitcnt lgkmcnt(0)\nds_read_b32 v16, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_max_f32 v3, v6\ns_waitcnt lgkmcnt(0)\nds_read_b32 v17, v3\ns_waitcnt lgkmcnt(0)\n"
                 "ds_write_b32 v3, v4\ns_waitcnt lgkmcnt(0)\nds_cmpst_f32 v3, v6, v8\ns_waitcnt lgkmcnt(0)\nds_read_b32 v18, v3\ns_waitcnt lgkmcnt(0)\n")
            words = assemble(host_kernel(body, 8, True), True)
            for mode, flags in MODES32.items():
                got, err = host_run(gpu, words, split_rows(rows3, True), 8, len(rows3), flags, True, work / f"f32-{mode}-{gpu}")
                key = f"{gpu}/f32/{mode}"
                report[key] = {"error": err} if got is None else dict(zip(("differs", "cells", "examples"), compare(key, got, m["f32"]["modes"][mode], 9)))
                print(key, report[key], flush=True)
        if "int" in wanted:
            rows3 = [tuple(r) for r in m["int"]["rows"]]
            differs = 0
            cells = 0
            failed = []
            for op, oracle in m["int"]["ops"].items():
                three = any(k in op for k in matrix.THREE_OPERAND)
                body = "v_lshlrev_b32 v3, 3, v0\n" + matrix.op_body(op, "v[12:13]", "v[10:11]")
                words = assemble(host_kernel(body, 8, True), True)
                got, err = host_run(gpu, words, split_rows(rows3, True), 8, len(rows3), "0xf0,1,0,0", True, work / f"int-{op}-{gpu}")
                if got is None:
                    failed.append(f"{op}: {err}")
                    continue
                d, t, _ = compare(op, got, oracle, 4)
                differs += d
                cells += t
            report[f"{gpu}/int"] = {"differs": differs, "cells": cells, "failed": failed}
            print(f"{gpu}/int", report[f"{gpu}/int"], flush=True)
        if "contention" in wanted:
            c = json.load(open(out / "contention.json"))
            for wave64 in (False, True):
                tag = "wave64" if wave64 else "wave32"
                for name, body in contention.bodies(wave64).items():
                    if f"{name}-{tag}" not in c:
                        continue
                    rows = contention.rows_for(name)
                    words = assemble(host_kernel(body, 4, False), wave64)
                    got, err = host_run(gpu, words, [[a & 0xffffffff, a >> 32, b & 0xffffffff, b >> 32] for a, _, b, _ in rows], 4, len(rows), "0xf0,1,0,0", wave64, work / f"{name}-{tag}-{gpu}")
                    key = f"{gpu}/{name}-{tag}"
                    outs = {"contention": 12, "mixed": 2, "exec": 4, "offsets": 4, "range": 6}[name]
                    if got is None:
                        report[key] = {"error": err}
                    elif name == "contention":
                        q = lambda lo, hi: lo | (hi << 32)
                        final = {q(o[6], o[7]) for o in got}
                        adds = sorted(q(o[0], o[1]) for o in got)
                        report[key] = {"add_final": [hex(v) for v in final], "serial": adds == [i * 0xffffffff for i in range(len(got))],
                                       "max_final": [hex(q(o[8], o[9])) for o in got[:1]], "inc_final": [hex(q(o[10], o[11])) for o in got[:1]]}
                    elif name == "mixed":
                        report[key] = {"final": [hex(o[0] | (o[1] << 32)) for o in got[:1]], "lanes_agree": len({(o[0], o[1]) for o in got}) == 1,
                                       "expected": hex((128 * 0xffffffff + 128 * (1 << 32)) & ((1 << 64) - 1))}
                    else:
                        report[key] = dict(zip(("differs", "cells", "examples"), compare(key, got, c[f"{name}-{tag}"]["out"], outs)))
                    print(key, report[key], flush=True)
    json.dump(report, open(out / "hosts-report.json", "w"), indent=1)
    print("wrote", out / "hosts-report.json")


if __name__ == "__main__":
    main()
