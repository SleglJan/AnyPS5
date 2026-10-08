import collections
import json
import sys

MODES = ["title", "old", "ieee0_keep32", "ieee1_flush32", "title_flush16", "title_noclamp", "title_fp16ovfl"]


def nan32(v):
    return (v & 0x7fffffff) > 0x7f800000


def nan16(h):
    return (h & 0x7fff) > 0x7c00


def equal(a, b, f16):
    if a == b:
        return True
    if f16:
        lo = (a & 0xffff, b & 0xffff)
        hi = (a >> 16, b >> 16)
        return all(x == y or (nan16(x) and nan16(y)) for x, y in (lo, hi))
    return nan32(a) and nan32(b)


def main():
    report = json.load(open(sys.argv[1]))
    summary = []
    for k in report:
        if "cells" not in k:
            continue
        f16 = "f16" in k.get("areas", [])
        groups = collections.Counter()
        payload = 0
        hosts_differ = 0
        for c in k["flagged_cells"]:
            if not c["hosts"] or "title" not in c["oracle"]:
                continue
            host = int(c["hosts"].get("NVIDIA") or c["hosts"].get("Ryzen"), 16)
            radv = c["hosts"].get("Ryzen")
            if radv is not None and not equal(int(radv, 16), host, f16):
                hosts_differ += 1
                host = int(radv, 16)
            o = {m: int(v, 16) for m, v in c["oracle"].items()}
            if equal(o["title"], host, f16):
                continue
            if o["title"] == host or (f16 and (nan16(o["title"] & 0xffff) or nan16(o["title"] >> 16))) or nan32(o["title"]):
                if equal(o["title"], host, f16):
                    payload += 1
                    continue
            match = tuple(m for m in MODES if m in o and equal(o[m], host, f16))
            groups[match or ("none",)] += 1
        if groups or hosts_differ:
            summary.append((k["file"], k["kernel"], dict(groups), hosts_differ))
    total = 0
    for f, kn, g, hd in summary:
        n = sum(g.values())
        total += n
        print(f"{f:<30} {n:3d} pinned != hw(title)" + (f", hosts differ {hd}" if hd else "") + ": " + "; ".join(f"{c} match {'+'.join(m)}" for m, c in sorted(g.items(), key=lambda kv: -kv[1])))
    print("total pinned != hardware in the titles' mode:", total)


if __name__ == "__main__":
    main()
