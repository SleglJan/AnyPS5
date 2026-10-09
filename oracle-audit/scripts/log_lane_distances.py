"""Replay the LogNearOne test rows on both hosts and print per-lane ULP distances to the oracle (LD_LIBRARY_PATH selects the build). Usage: log_lane_distances.py <tag>"""
import importlib.util, sys
from pathlib import Path
R = Path("/home/slegl/PycharmProjects/anyps5")
spec = importlib.util.spec_from_file_location("t", R / "scripts/oracle/log_make_test.py"); t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
spec2 = importlib.util.spec_from_file_location("h", R / "scripts/oracle/transcendental_hosts.py"); h = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(h)
out = R / "notes/oracle-log"
expected = [[int(x, 16) for x in l.split()] for l in (out / "log-test.txt").read_text().splitlines() if l.strip()]
rows = [[a, b, 0, 0] for a, b in zip(t.ROWS32, t.ROWS16)]
kernel = "v_lshlrev_b32 v1, 4, v0\nv_lshlrev_b32 v2, 6, v0\nbuffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen\ns_waitcnt vmcnt(0)\n" + "".join(f"v_mov_b32 v{r}, 0\n" for r in range(10, 26)) + t.BODY + "buffer_store_dwordx4 v[10:13], v2, s[4:7], 0 offen\ns_endpgm\n"
words = t.hosts.assemble(kernel, False)
tag = sys.argv[1]
for gpu in ("Ryzen", "NVIDIA"):
    got, err = t.hosts.host_run(gpu, words, rows, 4, 64, "0xc0,1,0,0", False, out / "hosts" / f"log-test-{tag}-{gpu}")
    (out / "hosts" / f"log-test-{tag}-{gpu}" / "results.txt").write_text("".join(" ".join(f"{v:08x}" for v in row) + "\n" for row in got))
    line = []; dist = []
    for i, (e, g) in enumerate(zip(expected, got)):
        ds = []
        for c, w in enumerate((32, 16)):
            m = 0xffff if w == 16 else 0xffffffff
            ev, gv = e[c] & m, g[c] & m
            if ev == gv: ds.append(0); continue
            ce, cg = h.classify(ev, w), h.classify(gv, w)
            ds.append(10 ** 9 if ce != cg else abs(h.ordered(gv, w) - h.ordered(ev, w)))
        dist.append(max(ds))
        if any(ds): line.append(f"{i}:{'/'.join(map(str, ds))}")
    print(f"{tag} {gpu}: lanes off (log32/log16): {' '.join(line) or 'none'}")
    print(f"{tag} {gpu} per-lane max: {','.join(map(str, dist))}")
