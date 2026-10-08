"""Cells whose oracle value does not depend on a field must not move on the hosts when the field changes.

Usage: python3 -s scripts/oracle/check_prs_invariance.py notes/oracle-check-1866-1867
"""
import json
import sys
from pathlib import Path


def main():
    out = Path(sys.argv[1]).resolve()
    for base, other in (("title", "title_noclamp"), ("title", "ieee1_flush32")):
        moved, checked = {}, 0
        for work in sorted(out.iterdir()):
            f = work / "values.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            sensitive = {k for k in d["oracle"][base] if k in d["oracle"][other] and d["oracle"][base][k] != d["oracle"][other][k]}
            for gpu in ("Ryzen", "NVIDIA"):
                h1, h2 = d["hosts"].get(f"{gpu}/{base}"), d["hosts"].get(f"{gpu}/{other}")
                if h1 is None or h2 is None:
                    continue
                for k in d["oracle"][base]:
                    if k in sensitive:
                        continue
                    r, c = map(int, k.split(","))
                    checked += 1
                    if h1[r][c] != h2[r][c]:
                        moved.setdefault(f"{work.name}/{gpu}", []).append(k)
        print(f"{base} vs {other}: {checked} host cells the oracle does not change; moved on the hosts: {sum(map(len, moved.values()))} {moved}")
    moved, checked = 0, 0
    for work in sorted(out.iterdir()):
        f = work / "values.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        for gpu in ("Ryzen", "NVIDIA"):
            for ra, rb in zip(d["hosts"][f"{gpu}/none"], d["hosts"][f"{gpu}/title"]):
                for c in range(d["used"]):
                    checked += 1
                    moved += ra[c] != rb[c]
    print(f"no mode vs titles' mode: {checked} host cells, {moved} differ")


if __name__ == "__main__":
    main()
