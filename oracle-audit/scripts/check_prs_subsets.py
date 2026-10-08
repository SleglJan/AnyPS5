"""Count, per kernel, the cells whose oracle value depends on a mode field and how the hosts do on exactly those cells.

Usage: python3 -s scripts/oracle/check_prs_subsets.py notes/oracle-check-1866-1867 [dx10|ieee]
dx10: cells where the oracle differs between title and title_noclamp; hosts compared in title_noclamp.
ieee: cells where the oracle differs between title and ieee1_flush32; hosts compared in title (IEEE 0) and ieee1_flush32.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_prs import classify  # noqa: E402

PAIRS = {"dx10": ("title", "title_noclamp", ["title_noclamp"]), "ieee": ("title", "ieee1_flush32", ["title", "ieee1_flush32"])}


def main():
    out = Path(sys.argv[1]).resolve()
    which = sys.argv[2:] or ["dx10", "ieee"]
    for name in which:
        base, other, host_modes = PAIRS[name]
        print(f"== {name}: cells with oracle[{base}] != oracle[{other}]")
        totals = {}
        for work in sorted(out.iterdir()):
            f = work / "values.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            o_base, o_other = d["oracle"][base], d["oracle"][other]
            cells = [k for k in o_base if k in o_other and o_base[k] != o_other[k]]
            if not cells:
                continue
            line = f"{work.name:<32} {len(cells):4d}"
            for gpu in ("Ryzen", "NVIDIA"):
                for mode in host_modes:
                    got = d["hosts"].get(f"{gpu}/{mode}")
                    if got is None:
                        line += f"  {gpu}/{mode}: throws"
                        totals.setdefault((gpu, mode), {}).setdefault("throws", 0)
                        totals[(gpu, mode)]["throws"] += len(cells)
                        continue
                    counts = {"equal": 0, "nan-payload": 0, "differs": 0}
                    for k in cells:
                        r, c = map(int, k.split(","))
                        oracle_row = [d["oracle"][mode].get(f"{r},{cc}", 0) for cc in range(d["results"])]
                        counts[classify(work.name.split(".")[0], got[r][c], d["oracle"][mode][k], c, got[r], oracle_row)] += 1
                    line += f"  {gpu}/{mode}: ={counts['equal']} nan={counts['nan-payload']} diff={counts['differs']}"
                    t = totals.setdefault((gpu, mode), {})
                    for kk, vv in counts.items():
                        t[kk] = t.get(kk, 0) + vv
            print(line)
        for key, t in totals.items():
            print("   total", key, t)


if __name__ == "__main__":
    main()
