"""Scalar-call, null-destination swappc, shader clocks and v_cmp_lg e32/e64 on the oracle.

Usage: python3 -s scripts/oracle/scalar_misc.py out-dir
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "AnyPS5" / "tools" / "hw-oracle" / "hw_oracle.py"
FLAGS = ["--ieee", "0", "--denorm32", "0", "--denorm16", "3", "--dx10-clamp", "1", "--round32", "0", "--round16", "0", "--fp16-overflow", "0"]

CALL = """s_mov_b32 s20, 0
s_getpc_b64 s[12:13]
s_call_b64 s[14:15], 3
s_add_u32 s20, s20, 2
s_branch 2f
s_nop 0
s_add_u32 s20, s20, 1
s_mov_b32 s19, 0x2222
s_setpc_b64 s[14:15]
2:
s_sub_u32 s16, s14, s12
s_subb_u32 s17, s15, s13
v_mov_b32 v10, s16
v_mov_b32 v11, s17
v_mov_b32 v12, s19
v_mov_b32 v13, s20
"""
SWAPPC = """s_mov_b32 s14, 0xabcd
s_mov_b32 s15, 0xef01
s_mov_b32 s24, 0
s_getpc_b64 s[12:13]
s_add_u32 s22, s12, 16
s_addc_u32 s23, s13, 0
s_swappc_b64 null, s[22:23]
s_mov_b32 s24, 17
s_mov_b32 s25, 34
s_getpc_b64 s[26:27]
s_add_u32 s28, s26, 16
s_addc_u32 s29, s27, 0
s_swappc_b64 s[30:31], s[28:29]
s_mov_b32 s25, 51
s_mov_b32 s34, 60
s_sub_u32 s32, s30, s26
s_subb_u32 s33, s31, s27
v_mov_b32 v10, s14
v_mov_b32 v11, s15
v_mov_b32 v12, s24
v_mov_b32 v13, s25
v_mov_b32 v14, s32
v_mov_b32 v15, s33
v_mov_b32 v16, s34
"""
CLOCK = """s_memtime s[12:13]
s_memrealtime s[14:15]
s_waitcnt lgkmcnt(0)
s_getreg_b32 s22, hwreg(HW_REG_SHADER_CYCLES)
s_mov_b32 s20, @N@
1:
v_add_f32 v8, v4, v8
v_add_f32 v9, v5, v9
s_sub_u32 s20, s20, 1
s_cmp_lg_u32 s20, 0
s_cbranch_scc1 1b
s_memtime s[16:17]
s_memrealtime s[18:19]
s_waitcnt lgkmcnt(0)
s_getreg_b32 s23, hwreg(HW_REG_SHADER_CYCLES)
v_mov_b32 v10, s12
v_mov_b32 v11, s13
v_mov_b32 v12, s14
v_mov_b32 v13, s15
v_mov_b32 v14, s16
v_mov_b32 v15, s17
v_mov_b32 v16, s18
v_mov_b32 v17, s19
v_mov_b32 v18, s22
v_mov_b32 v19, s23
v_mov_b32 v20, v8
"""
LG = """v_cmp_lg_f32 vcc_lo, v4, v5
v_cndmask_b32 v10, 0, 1, vcc_lo
v_cmp_lg_f32_e64 s12, v4, v5
v_cndmask_b32_e64 v11, 0, 1, s12
v_cmp_lg_f16 vcc_lo, v6, v7
v_cndmask_b32 v12, 0, 1, vcc_lo
v_cmp_lg_f16_e64 s12, v6, v7
v_cndmask_b32_e64 v13, 0, 1, s12
v_cmp_nlg_f32 vcc_lo, v4, v5
v_cndmask_b32 v14, 0, 1, vcc_lo
v_cmp_neq_f32 vcc_lo, v4, v5
v_cndmask_b32 v15, 0, 1, vcc_lo
"""
F32 = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x00000001, 0x7fc00000, 0xfffffffe, 0x7f800001, 0xff800001, 0x007fffff, 0x3f800001]
F16 = [0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x0001, 0x7e00, 0xfffe, 0x7c01, 0xfc01, 0x7fff, 0x3c01]


def run(name, body, rows, outs, out, wave64=False):
    bf = out / f"{name}-body.s"
    rf = out / f"{name}-rows.txt"
    bf.write_text(body)
    rf.write_text("".join(" ".join(f"0x{v:08x}" for v in r) + "\n" for r in rows))
    cmd = [sys.executable, "-s", str(ORACLE)] + FLAGS + ["--outs", str(outs), str(bf), str(rf)] + (["--wave64"] if wave64 else [])
    p = subprocess.run(cmd, cwd=ROOT / "AnyPS5", capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        print(name, "FAILED:", [l for l in p.stderr.splitlines() if "error" in l.lower()][:3] or p.stderr.strip().splitlines()[-2:])
        return None
    (out / f"{name}.txt").write_text(p.stdout)
    return [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if l.strip()]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows32 = [[0, 0, 0, 0]] * 32
    results = {}
    for wave64 in (False, True):
        tag = "wave64" if wave64 else "wave32"
        got = run(f"call-{tag}", CALL, rows32 if not wave64 else [[0, 0, 0, 0]] * 64, 4, out, wave64)
        if got:
            results[f"call-{tag}"] = sorted({tuple(o[:4]) for o in got})
            print(f"s_call_b64 {tag}: distinct (link-pc lo, hi, callee marker, order counter) = {[tuple(hex(v) for v in r) for r in results[f'call-{tag}']]}  (expected ('0x4','0x0','0x2222','0x3'))")
        got = run(f"swappc-{tag}", SWAPPC, rows32 if not wave64 else [[0, 0, 0, 0]] * 64, 7, out, wave64)
        if got:
            results[f"swappc-{tag}"] = sorted({tuple(o[:7]) for o in got})
            print(f"s_swappc_b64 {tag}: distinct (s14, s15 untouched by null swappc, s24 skipped by its jump, s25 skipped by the second jump, link-pc lo, hi, s34 reached) = {[tuple(hex(v) for v in r) for r in results[f'swappc-{tag}']]}  (expected ('0xabcd','0xef01','0x0','0x22','0xc','0x0','0x3c'))")
    for n in (1000, 20000, 400000):
        got = run(f"clock-{n}", CLOCK.replace("@N@", str(n)), [[0x3f800000, 0x3f800000, 0, 0]] * 32, 11, out)
        if got:
            o = got[0]
            mt0, rt0, mt1, rt1 = o[0] | (o[1] << 32), o[2] | (o[3] << 32), o[4] | (o[5] << 32), o[6] | (o[7] << 32)
            cyc0, cyc1 = o[8], o[9]
            results[f"clock-{n}"] = {"memtime": [mt0, mt1], "memrealtime": [rt0, rt1], "shader_cycles": [cyc0, cyc1], "uniform": len({tuple(x[:10]) for x in got}) == 1}
            dmt, drt, dcyc = mt1 - mt0, rt1 - rt0, (cyc1 - cyc0) & 0xfffff
            print(f"clock loop {n}: memtime delta {dmt} ({dmt.bit_length()} bits), memrealtime delta {drt} (100 MHz -> {drt / 100:.1f} us), ratio {dmt / drt if drt else float('nan'):.2f}; SHADER_CYCLES before {cyc0:#x} after {cyc1:#x} (20-bit delta {dcyc}); uniform over the wave: {results[f'clock-{n}']['uniform']}")
    rows = []
    for i, a in enumerate(F32):
        for j, b in enumerate(F32):
            rows.append([a, b, F16[i], F16[j]])
    while len(rows) % 32:
        rows.append([0, 0, 0, 0])
    got = run("lg", LG, rows, 6, out)
    if got:
        def nan32(x): return (x & 0x7fffffff) > 0x7f800000
        def nan16(x): return (x & 0x7fff) > 0x7c00
        import struct
        f32 = lambda v: struct.unpack("<f", struct.pack("<I", v))[0]
        f16 = lambda v: struct.unpack("<e", struct.pack("<H", v))[0]
        mism = {"lg_f32 e32 vs e64": 0, "lg_f16 e32 vs e64": 0, "lg_f32 vs model(ordered !=)": 0, "lg_f16 vs model": 0, "nlg_f32 vs !lg": 0, "neq_f32 vs model(unordered or !=)": 0}
        for r, o in zip(rows, got):
            a, b, c, d = r
            lg32 = 0 if (nan32(a) or nan32(b)) else int(f32(a) != f32(b))
            lg16 = 0 if (nan16(c) or nan16(d)) else int(f16(c) != f16(d))
            neq32 = 1 if (nan32(a) or nan32(b)) else int(f32(a) != f32(b))
            mism["lg_f32 e32 vs e64"] += o[0] != o[1]; mism["lg_f16 e32 vs e64"] += o[2] != o[3]
            mism["lg_f32 vs model(ordered !=)"] += o[0] != lg32; mism["lg_f16 vs model"] += o[2] != lg16
            mism["nlg_f32 vs !lg"] += o[4] != (1 - lg32); mism["neq_f32 vs model(unordered or !=)"] += o[5] != neq32
        results["lg"] = {"rows": len(rows), "mismatches": mism}
        print(f"v_cmp_lg on {len(rows)} rows ({len(F32)}x{len(F32)} f32 and f16 pairs incl. NaNs): {mism}")
    json.dump(results, open(out / "results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
