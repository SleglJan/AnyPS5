"""Sweep the transcendental and reciprocal VALU ops on the oracle: every f16 input, stratified f32 inputs, specials.

Usage: python3 -s scripts/oracle/transcendental_sweep.py out-dir [f16|f32]
f16: v_rcp/rsq/sqrt/log/exp/sin/cos_f16 over all 65,536 inputs (v4 low half), results v10..v16 (low halves).
f32: v_rcp/rsq/sqrt/log/exp/sin/cos_f32 and v_rcp_iflag_f32 over 65,536 inputs: every exponent (256) with 128 mantissas
     of both signs, plus specials and the sin/cos domain edges (|x| around 256 and far beyond); results v10..v17.
Modes: the titles' mode (IEEE 0, f32 denormals flushed, f16 kept, DX10_CLAMP 1, RNE); f32 also with denormals kept.
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "AnyPS5" / "tools" / "hw-oracle" / "hw_oracle.py"
F16_OPS = ["v_rcp_f16", "v_rsq_f16", "v_sqrt_f16", "v_log_f16", "v_exp_f16", "v_sin_f16", "v_cos_f16"]
F32_OPS = ["v_rcp_f32", "v_rsq_f32", "v_sqrt_f32", "v_log_f32", "v_exp_f32", "v_sin_f32", "v_cos_f32", "v_rcp_iflag_f32"]


def flags(denorm32):
    return ["--ieee", "0", "--denorm32", str(denorm32), "--denorm16", "3", "--dx10-clamp", "1", "--round32", "0", "--round16", "0", "--fp16-overflow", "0"]


def run(name, body, rows, outs, out, mode_flags):
    bf = out / f"{name}-body.s"
    rf = out / f"{name}-rows.txt"
    bf.write_text(body)
    rf.write_text("".join(f"0x{r:08x} 0x00000000 0x00000000 0x00000000\n" for r in rows))
    p = subprocess.run([sys.executable, "-s", str(ORACLE)] + mode_flags + ["--outs", str(outs), str(bf), str(rf)], cwd=ROOT / "AnyPS5", capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        print(name, "FAILED:", [l for l in p.stderr.splitlines() if "error" in l.lower()][:2] or p.stderr.strip().splitlines()[-1:])
        return None
    (out / f"{name}.txt").write_text(p.stdout)
    return [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if l.strip()]


def f32_inputs():
    values = []
    for exponent in range(256):
        for k in range(64):
            mantissa = (k * 0x7fffff) // 63
            for sign in (0, 1):
                values.append((sign << 31) | (exponent << 23) | mantissa)
    specials = [0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001, 0x7fc00123, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff]
    edges = []
    for cycles in (255.0, 255.5, 255.75, 256.0, 256.25, 256.5, 257.0, 300.0, 1000.0, 65536.0, 1e6, 1e9, 3.4e38):
        for sign in (1.0, -1.0):
            edges.append(struct.unpack("<I", struct.pack("<f", sign * cycles))[0])
    values = values + specials + edges
    while len(values) % 64:
        values.append(0)
    return values


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    which = set(sys.argv[2:]) or {"f16", "f32"}
    manifest = {}
    if "f16" in which:
        rows = list(range(65536))
        body = "".join(f"{op} v{10 + i}, v4\n" for i, op in enumerate(F16_OPS))
        got = run("f16-title", body, rows, len(F16_OPS), out, flags(0))
        if got:
            manifest["f16-title"] = {"ops": F16_OPS, "rows": len(rows), "file": "f16-title.txt"}
            print(f"f16: {len(got)} inputs x {len(F16_OPS)} ops", flush=True)
    if "f32" in which:
        rows = f32_inputs()
        (out / "f32-inputs.json").write_text(json.dumps(rows))
        body = "".join(f"{op} v{10 + i}, v4\n" for i, op in enumerate(F32_OPS))
        for mode, d32 in (("flushed", 0), ("kept", 3)):
            got = run(f"f32-{mode}", body, rows, len(F32_OPS), out, flags(d32))
            if got:
                manifest[f"f32-{mode}"] = {"ops": F32_OPS, "rows": len(rows), "file": f"f32-{mode}.txt"}
                print(f"f32 {mode}: {len(got)} inputs x {len(F32_OPS)} ops", flush=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print("wrote", out / "manifest.json")


if __name__ == "__main__":
    main()
