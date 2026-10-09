"""Replay the dense (0, 0.25] sample through a recompiler build on both GPUs and compare with the oracle. Usage: dense_compare.py <tag> (LD_LIBRARY_PATH selects the build)."""
import importlib.util, struct, sys
from pathlib import Path
R = Path("/home/slegl/PycharmProjects/anyps5"); D = R / "notes/oracle-transcendentals"
spec = importlib.util.spec_from_file_location("h", R / "scripts/oracle/transcendental_hosts.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
fl = lambda b: struct.unpack("<f", struct.pack("<I", b))[0]
inputs = [int(l.split()[0], 16) for l in (D / "dense-sincos-rows.txt").read_text().splitlines() if l.strip()]
oracle = h.load_oracle(D / "dense-sincos.txt")
tag = sys.argv[1]
for gpu in ("Ryzen", "NVIDIA"):
    got = h.host_sweep(gpu, "v_sin_f32 v10, v4\nv_cos_f32 v11, v4\n", inputs, h.MODES32["flushed"], D / "hosts" / f"dense-{tag}")
    (D / "hosts" / f"dense-{tag}" / f"{gpu}-results.txt").write_text("".join(" ".join(f"{v:08x}" for v in row) + "\n" for row in got))
    for col, op in ((0, "sin"), (1, "cos")):
        exact = 0; worst = 0; worst_in = None; bands = {}
        for i, x in enumerate(inputs):
            a, b = got[i][col], oracle[i][col]
            if a == b: exact += 1; continue
            if h.classify(a, 32) != h.classify(b, 32): d = 10 ** 9
            else: d = abs(h.ordered(a, 32) - h.ordered(b, 32))
            v = fl(x); band = "<2^-12" if v < 2 ** -12 else "<0.125" if v < 0.125 else "<0.25-2^-12" if v < 0.25 - 2 ** -12 else "near 0.25"
            bands[band] = max(bands.get(band, 0), d)
            if d > worst: worst, worst_in = d, (x, a, b)
        print(f"{tag} {gpu} {op}: exact {exact}/{len(inputs)}, max {worst} at {worst_in and 'in=%08x host=%08x hw=%08x' % worst_in}, max by band {bands}")
