"""Run 2 mode sensitivity: per kernel and result column, which of the seven modes give a value (& column mask) that
differs bit-exactly from the titles' mode, and per-field totals over all cells.

python3 -s sens2.py [audit-dir] > mode-sensitivity.txt

Fields (each the titles' mode with one field changed): IEEE_MODE = ieee1_flush32, f32 denormals = ieee0_keep32,
f16/f64 denormals = title_flush16, DX10_CLAMP = title_noclamp, FP16_OVFL = title_fp16ovfl. 'old' (IEEE 1 + f32 kept)
is reported separately; a cell that differs only under 'old' and under neither single field is counted as
'IEEE+f32 combination only'. A cell counts once per field. 'quiet bit only' = every differing value differs from
the titles' value only in the NaN quiet bit of an f32, an f16 half or an f64 high word (both values NaN).
"""
import collections
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("ANYPS5_AUDIT_ROOT") or HERE.parents[2])
sys.path.insert(0, str(ROOT / "scripts" / "oracle"))
from audit_adapters import is_quiet_bit_only  # noqa: E402

FIELDS = [("IEEE_MODE", "ieee1_flush32"), ("f32 denormals", "ieee0_keep32"), ("f16/f64 denormals", "title_flush16"),
          ("DX10_CLAMP", "title_noclamp"), ("FP16_OVFL", "title_fp16ovfl")]
ALL = ["old"] + [m for _, m in FIELDS]
AB = dict(old="O", ieee0_keep32="K", ieee1_flush32="F", title_flush16="h", title_noclamp="c", title_fp16ovfl="v")


def main():
    audit_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notes" / "oracle-audit-run2"
    values = json.load(open(audit_dir / "values.json"))
    per_col = collections.OrderedDict()
    total_cells = dep_cells = dep_quiet_only = 0
    field_tot = collections.Counter()
    field_quiet = collections.Counter()
    combo_only = 0
    missing = []
    for name in sorted(values):
        v = values[name]
        if "title" not in v["oracle"] or any(m not in v["oracle"] for m in ALL):
            missing.append(name)
            continue
        for j, c in enumerate(v["columns"]):
            mask = int(v["masks"][j], 16)
            key = (v["file"], v["kernel"], int(c))
            rec = per_col.setdefault(key, dict(cells=0, modes=collections.Counter(), quiet=collections.Counter(), adapters=set()))
            rec["adapters"].add(name[:-5])
            for r in range(len(v["inputs"])):
                total_cells += 1
                rec["cells"] += 1
                t = int(v["oracle"]["title"][r][j], 16) & mask
                d = {m: int(v["oracle"][m][r][j], 16) & mask for m in ALL}
                diff = {m: x for m, x in d.items() if x != t}
                if not diff:
                    continue
                dep_cells += 1
                q = all(is_quiet_bit_only(x, t) for x in diff.values())
                dep_quiet_only += q
                for m, x in diff.items():
                    rec["modes"][m] += 1
                    rec["quiet"][m] += is_quiet_bit_only(x, t)
                for f, m in FIELDS:
                    if m in diff:
                        field_tot[f] += 1
                        field_quiet[f] += is_quiet_bit_only(diff[m], t)
                if "old" in diff and "ieee1_flush32" not in diff and "ieee0_keep32" not in diff:
                    combo_only += 1
                    field_tot["IEEE+f32 combination only"] += 1
                    field_quiet["IEEE+f32 combination only"] += is_quiet_bit_only(diff["old"], t)
    print("Run 2 mode sensitivity, bit-exact against the titles' mode (value & column mask).")
    print("Mode letters: O=old (IEEE 1, f32 kept) F=IEEE 1 K=f32 denormals kept h=f16/f64 flushed c=DX10_CLAMP 0 v=FP16_OVFL 1")
    print(f"\nTotal cells: {total_cells}; mode-dependent: {dep_cells} ({dep_quiet_only} of them only in the NaN quiet bit)")
    print("\nPer field (a cell counts once per field; 'quiet bit only' cells in parentheses):")
    for f, _ in FIELDS:
        print(f"  {f:28s} {field_tot[f]:6d} ({field_quiet[f]})")
    print(f"  {'old only (IEEE+f32 together)':28s} {combo_only:6d} ({field_quiet['IEEE+f32 combination only']})")
    print(f"  {'old (IEEE 1 + f32 kept), any':28s} {sum(r['modes']['old'] for r in per_col.values()):6d} ({sum(r['quiet']['old'] for r in per_col.values())})")
    print("\nPer kernel and column (only columns with a mode-dependent cell): cells; per mode: differing cells (quiet-bit-only)")
    last = None
    quiet_cols = 0
    for (file, kernel, col), rec in per_col.items():
        if not rec["modes"]:
            continue
        if (file, kernel) != last:
            print(f"\n## {file} {kernel}")
            last = (file, kernel)
        modes = "  ".join(f"{AB[m]}={rec['modes'][m]}" + (f"({rec['quiet'][m]})" if rec["quiet"][m] else "") for m in ALL if rec["modes"][m])
        print(f"   c{col:3d} cells={rec['cells']:4d}  {modes}")
    insensitive = sorted({f"{f}:{k}" for (f, k, c), r in per_col.items()} - {f"{f}:{k}" for (f, k, c), r in per_col.items() if r["modes"]})
    print(f"\n## Kernels with no mode-dependent cell ({len(insensitive)})")
    for k in insensitive:
        print("   " + k)
    if missing:
        print("\n## Adapters without all seven modes (oracle errors): " + ", ".join(missing))


if __name__ == "__main__":
    main()
