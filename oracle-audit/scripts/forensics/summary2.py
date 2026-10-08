"""Run 2 summary: coverage, host-vs-hardware cells and anomalies, from audit.json / values.json / skipped.json.

python3 -s summary2.py [audit-dir] > summary.txt
"""
import collections
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("ANYPS5_AUDIT_ROOT") or HERE.parents[2])
sys.path.insert(0, str(HERE))
from colinstr import producer  # noqa: E402

M = ["title", "old", "ieee0_keep32", "ieee1_flush32", "title_flush16", "title_noclamp", "title_fp16ovfl"]
AB = dict(title="T", old="O", ieee0_keep32="K", ieee1_flush32="F", title_flush16="h", title_noclamp="c", title_fp16ovfl="v")


def main():
    d = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notes" / "oracle-audit-run2"
    audit = json.load(open(d / "audit.json"))
    skipped = json.load(open(d / "skipped.json"))
    kernels = collections.OrderedDict()
    for a in audit:
        kernels.setdefault((a["file"], a["kernel"]), []).append(a)
    sk = collections.OrderedDict()
    for s in skipped:
        sk.setdefault((s["file"], s["kernel"]), []).append(s)
    print("## Coverage")
    print(f"adapter files run: {len(audit)}; skipped: {len(skipped)}; kernels with at least one run: {len(kernels)}; "
          f"kernels only skipped: {len([k for k in sk if k not in kernels])}")
    print(f"total cells: {sum(a.get('cells', 0) for a in audit)}")
    for (f, k), items in kernels.items():
        variants = [a["variant"] or "-" for a in items]
        cells = sum(a.get("cells", 0) for a in items)
        rows = sorted({a.get("rows") for a in items})
        host = items[0].get("host_run", "")
        extra = f"; skipped variants: {', '.join(s['variant'] or '-' for s in sk.get((f, k), []))}" if (f, k) in sk else ""
        print(f"  {f}:{k}: {len(items)} file(s) [{', '.join(variants)}], rows {rows}, cells {cells}; host: {host}{extra}")
    print("\n## Skipped adapter files")
    for s in skipped:
        print(f"  {s['adapter']}: {s['reason']}")
    print("\n## Errors")
    errs = 0
    for a in audit:
        for k in ("error", "oracle_errors", "host_errors"):
            if a.get(k):
                errs += 1
                print(f"  {a['adapter']}: {k}: {a[k]}")
    if not errs:
        print("  none")
    print("\n## Masks taken from adapter notes, and ShaderClock's relation check")
    for a in audit:
        if a.get("note_masks"):
            print(f"  {a['adapter']}: {a['note_masks']['masks']} ({a['note_masks']['reason']})")
        if a.get("clock_relations"):
            print(f"  {a['adapter']}: clock relations per mode: {a['clock_relations']}")
    print("\n## Columns whose titles'-mode value (& mask) is 0 in every row")
    for a in audit:
        if a.get("all_zero_columns_title"):
            print(f"  {a['adapter']}: cols {a['all_zero_columns_title']} ({a['rows']} rows): "
                  + "; ".join(f"c{c}: {producer(a['adapter'], c)}" for c in a["all_zero_columns_title"]))
    print("\n## Host vs hardware (titles' mode), bit-exact, where the harness ran")
    tot = collections.Counter()
    for a in audit:
        if not a.get("host_run", "").startswith("whole kernel"):
            continue
        tot["cells"] += a["cells"]
        cells = [c for c in a["flagged_cells"] if "recompiler-differs-from-hardware-title-mode" in c["flags"] or "hosts-differ" in c["flags"]]
        if not cells:
            continue
        print(f"  {a['adapter']}: {len(cells)} cells")
        for c in cells:
            tot["differ"] += "recompiler-differs-from-hardware-title-mode" in c["flags"]
            tot["hosts-differ"] += "hosts-differ" in c["flags"]
            tot["quiet"] += "recompiler-differs-quiet-bit-only" in c["flags"]
            for g, v in c["hosts"].items():
                tot[f"{g} != hw"] += v != c["oracle"]["title"]
            same = "+".join(AB[m] for m in M if c["oracle"][m] in c["hosts"].values()) or "none"
            print(f"     r{c['row']:3d} c{c['col']:2d} hw={c['oracle']['title']} " + " ".join(f"{g}={v}" for g, v in c["hosts"].items())
                  + f" | modes equal to a host value: {same} | {','.join(c['flags'])} | in={' '.join(c['inputs'])} | {producer(a['adapter'], c['col'])}")
    print(f"  totals: {dict(tot)}")


if __name__ == "__main__":
    main()
