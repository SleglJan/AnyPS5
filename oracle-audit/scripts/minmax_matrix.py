"""Measure min/max families over the 14-value operand set of #1826 in IEEE x denormal modes on the oracle.

Usage: python3 -s scripts/oracle/minmax_matrix.py out-dir
Writes out-dir/<set>-<mode>.txt (oracle output) and out-dir/<set>-rows.txt, plus matrix.json with every cell.
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "AnyPS5" / "tools" / "hw-oracle" / "hw_oracle.py"

F64 = [0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000, 0x7ff0000000000000, 0xfff0000000000000,
       0x0000000000000001, 0x8000000000000001, 0x7ff8000000000000, 0x7ff8000000000123, 0xfff8000000000000,
       0x7ff0000000000001, 0x7ff0000000000123, 0xfff0000000000001]
F16 = [0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x0001, 0x8001, 0x7e00, 0x7e23, 0xfe00, 0x7c01, 0x7c23, 0xfc01]

SETS = {
    "f64_minmax": {
        "body": "v_min_f64 v[10:11], v[4:5], v[6:7]\nv_max_f64 v[12:13], v[4:5], v[6:7]\n",
        "rows": [[a & 0xffffffff, a >> 32, b & 0xffffffff, b >> 32] for a, b in itertools.product(F64, F64)],
        "outs": 4, "ops": ["v_min_f64", "v_max_f64"],
    },
    "f16_minmax": {
        "body": ("v_min_f16_e64 v10, v4, v5\nv_max_f16_e64 v11, v4, v5\nv_min3_f16 v12, v4, v5, v6\n"
                 "v_max3_f16 v13, v4, v5, v6\nv_med3_f16 v14, v4, v5, v6\n"),
        "rows": [[a, b, c, 0] for a, b, c in itertools.product(F16, F16, F16)],
        "outs": 5, "ops": ["v_min_f16", "v_max_f16", "v_min3_f16", "v_max3_f16", "v_med3_f16"],
    },
}
MODES = {"ieee0_kept": (0, 3), "ieee1_kept": (1, 3), "ieee0_flushed": (0, 0), "ieee1_flushed": (1, 0)}


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    matrix = {}
    for name, spec in SETS.items():
        body = out / f"{name}-body.s"
        rows = out / f"{name}-rows.txt"
        body.write_text(spec["body"])
        rows.write_text("".join(" ".join(f"0x{v:08x}" for v in r) + "\n" for r in spec["rows"]))
        matrix[name] = {"ops": spec["ops"], "rows": spec["rows"], "modes": {}}
        for mode, (ieee, denorm16) in MODES.items():
            cmd = [sys.executable, "-s", str(ORACLE), "--ieee", str(ieee), "--denorm32", "0", "--denorm16", str(denorm16), "--dx10-clamp", "1",
                   "--round32", "0", "--round16", "0", "--fp16-overflow", "0", "--outs", str(spec["outs"]), str(body), str(rows)]
            p = subprocess.run(cmd, cwd=ROOT / "AnyPS5", capture_output=True, text=True, timeout=1800)
            if p.returncode != 0:
                print(name, mode, "FAILED:", p.stderr.strip().splitlines()[-1:], flush=True)
                continue
            lines = [l for l in p.stdout.splitlines() if l.strip()]
            (out / f"{name}-{mode}.txt").write_text(p.stdout)
            matrix[name]["modes"][mode] = [[int(x, 16) for x in l.split()] for l in lines]
            print(f"{name} {mode}: {len(lines)} rows", flush=True)
    json.dump(matrix, open(out / "matrix.json", "w"))
    print("wrote", out / "matrix.json")


if __name__ == "__main__":
    main()
