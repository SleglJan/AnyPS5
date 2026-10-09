"""64-bit LDS atomics under contention, mixed 32/64-bit widths, EXEC masking, immediate offsets and out-of-range addresses.

Usage: python3 -s scripts/oracle/lds64_contention.py out-dir
Runs each body in wave32 (8 waves of a 256-lane workgroup) and wave64 (4 waves).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "AnyPS5" / "tools" / "hw-oracle" / "hw_oracle.py"
LANES = 256
FLAGS = ["--ieee", "0", "--denorm32", "0", "--denorm16", "3", "--dx10-clamp", "1", "--round32", "0", "--round16", "0", "--fp16-overflow", "0"]


def exec_ops(wave64):
    if wave64:
        return dict(cmp="v_cmp_eq_u32 vcc, 0, v8", saveexec="s_and_saveexec_b64 s[12:13], vcc", other="s_andn2_b64 exec, s[12:13], vcc", restore="s_mov_b64 exec, s[12:13]",
                    half="s_mov_b32 exec_lo, 0x55555555\ns_mov_b32 exec_hi, 0x55555555", full="s_mov_b64 exec, -1")
    return dict(cmp="v_cmp_eq_u32 vcc_lo, 0, v8", saveexec="s_and_saveexec_b32 s12, vcc_lo", other="s_andn2_b32 exec_lo, s12, vcc_lo", restore="s_mov_b32 exec_lo, s12",
                half="s_mov_b32 exec_lo, 0x55555555", full="s_mov_b32 exec_lo, -1")


INIT = "v_mov_b32 v8, 0\nv_mov_b32 v9, 0\nv_mov_b32 v3, 0\nds_write_b64 v3, v[8:9]\nds_write_b64 v3, v[8:9] offset:8\nds_write_b64 v3, v[8:9] offset:16\nds_write_b64 v3, v[8:9] offset:24\ns_waitcnt lgkmcnt(0)\ns_barrier\n"
BARRIER = "s_waitcnt lgkmcnt(0)\ns_barrier\n"


def bodies(wave64):
    e = exec_ops(wave64)
    return {
        "contention": INIT + "ds_add_rtn_u64 v[10:11], v3, v[4:5]\nds_max_rtn_u64 v[12:13], v3, v[6:7] offset:8\nds_inc_rtn_u64 v[14:15], v3, v[6:7] offset:16\n" + BARRIER
                      + "ds_read_b64 v[16:17], v3\nds_read_b64 v[18:19], v3 offset:8\nds_read_b64 v[20:21], v3 offset:16\ns_waitcnt lgkmcnt(0)\n",
        "mixed": INIT + "v_and_b32 v8, 1, v0\n" + e["cmp"] + "\n" + e["saveexec"] + "\nds_add_u64 v3, v[4:5]\ns_waitcnt lgkmcnt(0)\n" + e["other"]
                 + "\nds_add_u32 v3, v6 offset:4\ns_waitcnt lgkmcnt(0)\n" + e["restore"] + "\n" + BARRIER + "ds_read_b64 v[10:11], v3\ns_waitcnt lgkmcnt(0)\n",
        "exec": "v_lshlrev_b32 v3, 3, v0\nds_write_b64 v3, v[4:5]\ns_waitcnt lgkmcnt(0)\n" + e["half"] + "\nds_add_rtn_u64 v[12:13], v3, v[6:7]\ns_waitcnt lgkmcnt(0)\n" + e["full"]
                + "\nds_read_b64 v[10:11], v3\ns_waitcnt lgkmcnt(0)\n",
        "offsets": "v_lshlrev_b32 v3, 4, v0\nds_write_b64 v3, v[4:5] offset:8\ns_waitcnt lgkmcnt(0)\nds_add_rtn_u64 v[12:13], v3, v[6:7] offset:8\ns_waitcnt lgkmcnt(0)\n"
                   "ds_read_b64 v[10:11], v3 offset:8\nds_read_b64 v[14:15], v3\ns_waitcnt lgkmcnt(0)\n",
        "range": "v_lshlrev_b32 v3, 3, v0\nds_write_b64 v3, v[4:5]\ns_waitcnt lgkmcnt(0)\nv_mov_b32 v8, 0x1000\nv_add_nc_u32 v8, v8, v3\nds_add_rtn_u64 v[12:13], v8, v[6:7]\n"
                 "s_waitcnt lgkmcnt(0)\nds_read_b64 v[10:11], v3\nv_mov_b32 v8, 0xfff8\nds_add_rtn_u64 v[14:15], v8, v[6:7]\ns_waitcnt lgkmcnt(0)\n",
    }


def rows_for(name):
    if name == "contention":
        return [(0xffffffff, 0, (lane * 0x9e3779b97f4a7c15) & ((1 << 64) - 1), 0) for lane in range(LANES)]
    if name == "mixed":
        return [(0xffffffff, 0, 1, 0) for lane in range(LANES)]
    if name == "range":
        return [(0x1111111122222222, 0, 0x0000000100000001, 0) for lane in range(LANES)]
    return [((lane + 1) * 0x0101010101010101 & ((1 << 64) - 1), 0, 0x00000000ffffffff, 0) for lane in range(LANES)]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for wave64 in (False, True):
        tag = "wave64" if wave64 else "wave32"
        for name, body in bodies(wave64).items():
            rows = rows_for(name)
            bf = out / f"{name}-{tag}-body.s"
            rf = out / f"{name}-{tag}-rows.txt"
            bf.write_text(body)
            rf.write_text("".join(f"0x{a & 0xffffffff:08x} 0x{a >> 32:08x} 0x{b & 0xffffffff:08x} 0x{b >> 32:08x}\n" for a, _, b, _ in rows))
            outs = {"contention": 12, "mixed": 2, "exec": 4, "offsets": 6, "range": 6}[name]
            cmd = [sys.executable, "-s", str(ORACLE)] + FLAGS + ["--outs", str(outs), str(bf), str(rf)] + (["--wave64"] if wave64 else [])
            p = subprocess.run(cmd, cwd=ROOT / "AnyPS5", capture_output=True, text=True, timeout=600)
            if p.returncode != 0:
                print(name, tag, "FAILED:", "\n".join(p.stderr.strip().splitlines()[-3:]), flush=True)
                continue
            got = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if l.strip()]
            (out / f"{name}-{tag}.txt").write_text(p.stdout)
            results[f"{name}-{tag}"] = {"rows": [[a, b] for a, _, b, _ in rows], "out": got}
            print(f"{name} {tag}: {len(got)} lanes", flush=True)
    json.dump(results, open(out / "contention.json", "w"))
    print("wrote", out / "contention.json")


if __name__ == "__main__":
    main()
