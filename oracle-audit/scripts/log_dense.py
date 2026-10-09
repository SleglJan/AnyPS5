"""Dense v_log_f32 sample of [0.875, 1.125] on the oracle (every 2^-15 of x, 16,384 rows) with a main-lowering replay on both hosts.

Usage: python3 -s scripts/oracle/log_dense.py out-dir
"""
import importlib.util
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep", HERE / "transcendental_sweep.py"); sweep = importlib.util.module_from_spec(spec); spec.loader.exec_module(sweep)
spec2 = importlib.util.spec_from_file_location("hosts", HERE / "transcendental_hosts.py"); hosts = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(hosts)
f32 = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = [f32(0.875 + i * 2.0 ** -15) for i in range(16384)]
    body = "v_log_f32 v10, v4\n"
    hw = sweep.run("log-dense", body, rows, 1, out, sweep.flags(0))
    if hw is None:
        raise SystemExit("oracle failed")
    print(len(rows), "rows on the oracle")
    for gpu in ("Ryzen", "NVIDIA"):
        got = hosts.host_sweep(gpu, body, rows, hosts.MODES32["flushed"], out / "hosts" / ((sys.argv[2] if len(sys.argv) > 2 else "main") + "-dense"))
        (out / "hosts" / ((sys.argv[2] if len(sys.argv) > 2 else "main") + "-dense") / f"{gpu}-results.txt").write_text("".join(f"{r[0]:08x}\n" for r in got))
        exact = sum(g[0] == h[0] for g, h in zip(got, hw)); worst = max((abs(hosts.ordered(g[0], 32) - hosts.ordered(h[0], 32)), i) for i, (g, h) in enumerate(zip(got, hw)))
        print(f"{gpu} (main) vs hardware: exact {exact}/{len(rows)}, max {worst[0]} ULP at x={rows[worst[1]]:08x}")


if __name__ == "__main__":
    main()
