"""Repeat the mixed 32/64-bit LDS atomic kernel on the oracle and on both hosts, recording every run's final value.

Usage: python3 -s scripts/oracle/lds64_mixed_repeat.py notes/oracle-lds64 [host-runs] [oracle-runs]
Expected final: 128 * 0xffffffff + 128 * 2^32 = 0xffffffff80; a loss is counted in 32-bit increments (2^32 each).
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hosts = load("lds64_hosts")
contention = load("lds64_contention")
EXPECTED = (128 * 0xffffffff + 128 * (1 << 32)) & ((1 << 64) - 1)


def main():
    out = Path(sys.argv[1]).resolve()
    host_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    oracle_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    result = {"expected": f"0x{EXPECTED:x}", "oracle": {}, "hosts": {}}
    for wave64 in (False, True):
        tag = "wave64" if wave64 else "wave32"
        body = contention.bodies(wave64)["mixed"]
        rows = contention.rows_for("mixed")
        bf = out / f"mixed-{tag}-body.s"
        rf = out / f"mixed-{tag}-rows.txt"
        finals = []
        for _ in range(oracle_runs):
            cmd = [sys.executable, "-s", str(contention.ORACLE)] + contention.FLAGS + ["--outs", "2", str(bf), str(rf)] + (["--wave64"] if wave64 else [])
            p = subprocess.run(cmd, cwd=contention.ROOT / "AnyPS5", capture_output=True, text=True, timeout=600, check=True)
            got = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if l.strip()]
            values = {o[0] | (o[1] << 32) for o in got}
            finals.append(f"0x{values.pop():x}" if len(values) == 1 else "lanes disagree")
        result["oracle"][tag] = finals
        print(f"oracle {tag}: {finals}", flush=True)
        words = hosts.assemble(hosts.host_kernel(body, 4, False), wave64)
        host_rows = [[a & 0xffffffff, a >> 32, b & 0xffffffff, b >> 32] for a, _, b, _ in rows]
        for gpu in ("Ryzen", "NVIDIA"):
            finals = []
            for i in range(host_runs):
                got, err = hosts.host_run(gpu, words, host_rows, 4, len(rows), "0xf0,1,0,0", wave64, out / "hosts" / f"mixed-repeat-{tag}-{gpu}")
                if got is None:
                    finals.append(f"error: {err}")
                    continue
                values = {o[0] | (o[1] << 32) for o in got}
                finals.append(f"0x{values.pop():x}" if len(values) == 1 else "lanes disagree")
            lost = [((EXPECTED - int(f, 16)) >> 32) if f.startswith("0x") else None for f in finals]
            result["hosts"][f"{gpu}/{tag}"] = {"finals": finals, "lost_increments": lost}
            print(f"{gpu} {tag}: lost increments per run {lost}", flush=True)
    json.dump(result, open(out / "mixed-repeats.json", "w"), indent=1)
    print("wrote", out / "mixed-repeats.json")


if __name__ == "__main__":
    main()
