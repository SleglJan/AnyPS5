"""Run 2: the tests' own pinned values against the titles'-mode hardware value, bit-exact, per adapter cell.

python3 -s allpinned2.py [audit-dir] > pinned-vs-hardware.txt   (audit-dir defaults to notes/oracle-audit-run2)
       --json out.json   also writes the per-cell records

Per cell of every adapter (values.json, mapped through the adapter's columns and column_masks): pin = the test's table
value & mask, title = the titles'-mode oracle value & mask. Cells with pin != title are listed, asserted ones first,
with the modes whose value equals the pin, both hosts' values (where the harness ran), whether Check()'s own comparison
accepts the hardware value, and the instruction that writes the column.
"""
import collections
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("ANYPS5_AUDIT_ROOT") or HERE.parents[2])
sys.path.insert(0, str(HERE))
import expectations as ex  # noqa: E402
from colinstr import producer  # noqa: E402

M = ["title", "old", "ieee0_keep32", "ieee1_flush32", "title_flush16", "title_noclamp", "title_fp16ovfl"]


def analyse(audit_dir):
    values = json.load(open(audit_dir / "values.json"))
    per_kernel = collections.OrderedDict()
    for name in sorted(values):
        v = values[name]
        key = (v["file"], v["kernel"])
        k = per_kernel.setdefault(key, dict(adapters=[], cells=0, compared=0, asserted=0, diff=[], pins=None, error=None))
        k["adapters"].append(name)
        k["cells"] += len(v["inputs"]) * len(v["columns"])
        if k["pins"] is None and k["error"] is None:
            try:
                k["pins"] = ex.pins(*key) or False
            except Exception as error:
                k["error"] = str(error)
        pins = k["pins"]
        if not pins or "title" not in v["oracle"]:
            continue
        for r, inputs in enumerate(v["inputs"]):
            for j, c in enumerate(v["columns"]):
                mask = int(v["masks"][j], 16)
                pin, asserted, accepts = pins.at(r, int(c))
                if pin is None:
                    continue
                k["compared"] += 1
                k["asserted"] += bool(asserted)
                o = {m: int(v["oracle"][m][r][j], 16) for m in M if m in v["oracle"]}
                t = o["title"] & mask
                if (pin & mask) == t:
                    continue
                hosts = {g: v["hosts"][g][r][j] for g in v["hosts"] if v["hosts"][g][r][j] is not None}
                rec = dict(adapter=name, row=r, col=int(c), mask=v["masks"][j], inputs=inputs, pin=f"{pin & mask:08x}", title=f"{t:08x}",
                           asserted=bool(asserted), reproduced_by=[m for m in M if (o[m] & mask) == (pin & mask)],
                           oracle={m: f"{o[m] & mask:08x}" for m in o}, hosts={g: f"{int(h, 16) & mask:08x}" for g, h in hosts.items()},
                           check_accepts_hardware=(accepts(o["title"]) if mask == 0xffffffff and accepts else None))
                k["diff"].append(rec)
    return per_kernel


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    audit_dir = pathlib.Path(args[0]) if args else ROOT / "notes" / "oracle-audit-run2"
    jout = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    per_kernel = analyse(audit_dir)
    print("Run 2: pinned value (the test's own table, & column mask) vs titles'-mode hardware value, bit-exact.")
    print("'reproduced by' = modes whose oracle value equals the pin; 'Check() accepts hw' = the test's own comparison applied to the hardware value.")
    print("Modes: T=title O=old K=ieee0_keep32 F=ieee1_flush32 h=title_flush16 c=title_noclamp v=title_fp16ovfl\n")
    ab = dict(title="T", old="O", ieee0_keep32="K", ieee1_flush32="F", title_flush16="h", title_noclamp="c", title_fp16ovfl="v")
    none, summary = [], []
    for (file, kernel), k in per_kernel.items():
        if k["error"]:
            none.append(f"{file}:{kernel} (expectation parse error: {k['error']})")
            continue
        if not k["pins"]:
            none.append(f"{file}:{kernel}")
            continue
        asserted = [d for d in k["diff"] if d["asserted"]]
        unasserted = [d for d in k["diff"] if not d["asserted"]]
        summary.append((file, kernel, k["compared"], k["asserted"], len(asserted), len(unasserted)))
        print(f"## {file} {kernel}: {len(k['adapters'])} adapter file(s), {k['cells']} cells, {k['compared']} with a pin, {k['asserted']} asserted;"
              f" pin != hardware: {len(asserted)} asserted, {len(unasserted)} unasserted")
        print(f"   pins: {k['pins'].source}; Check(): {k['pins'].rule}")
        for label, group in (("ASSERTED", asserted), ("unasserted", unasserted)):
            if not group:
                continue
            by = collections.Counter(("+".join(ab[m] for m in d["reproduced_by"]) or "none") for d in group)
            print(f"   {label}: {len(group)} cells; reproduced by {dict(by)}")
            for d in group:
                instr = producer(d["adapter"], d["col"])
                h = " ".join(f"{g}={x}" for g, x in d["hosts"].items()) or "hosts: not run"
                acc = {True: "yes", False: "NO", None: "n/a"}[d["check_accepts_hardware"]]
                mask = "" if d["mask"] == "0xffffffff" else f" mask={d['mask']}"
                print(f"     r{d['row']:3d} c{d['col']:2d}{mask} pin={d['pin']} hw={d['title']} by={'+'.join(ab[m] for m in d['reproduced_by']) or 'none'}"
                      f" | {h} | Check() accepts hw: {acc} | in={' '.join(d['inputs'])} | {d['adapter'][:-5]} | {instr}")
        print()
    print("## Summary (kernel: cells with a pin, asserted, asserted pin!=hw, unasserted pin!=hw)")
    for s in summary:
        print(f"   {s[0]}:{s[1]}: {s[2]}, {s[3]}, {s[4]}, {s[5]}")
    print(f"   total: {sum(s[2] for s in summary)} cells with a pin, {sum(s[3] for s in summary)} asserted, "
          f"{sum(s[4] for s in summary)} asserted pin!=hw, {sum(s[5] for s in summary)} unasserted pin!=hw")
    print("\n## Kernels with no Expected table (the test computes its expectation in code; not compared)")
    for n in none:
        print("   " + n)
    if jout:
        json.dump({f"{f}:{k}": dict(adapters=v["adapters"], cells=v["cells"], compared=v["compared"], asserted=v["asserted"], diff=v["diff"],
                                    pins=(v["pins"].source if v["pins"] else None)) for (f, k), v in per_kernel.items()}, open(jout, "w"), indent=1)


if __name__ == "__main__":
    main()
