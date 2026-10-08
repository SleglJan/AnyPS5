"""Compare the oracle values of two run-2 audit directories (repeatability check).

python3 -s repeat2.py dirA dirB
"""
import collections
import json
import sys

A = json.load(open(sys.argv[1] + "/values.json"))
B = json.load(open(sys.argv[2] + "/values.json"))
diff, total = collections.Counter(), 0
for k in A:
    a, b = A[k], B[k]
    for m in a["oracle"]:
        for ra, rb in zip(a["oracle"][m], b["oracle"][m]):
            for j, (x, y) in enumerate(zip(ra, rb)):
                total += 1
                if x != y:
                    diff[(k, a["columns"][j], a["masks"][j])] += 1
print(f"compared {total} oracle values (all adapters, rows, columns, 7 modes); differing: {sum(diff.values())}")
for (k, c, m), n in sorted(diff.items()):
    print(f"  {k} col {c} (comparison mask {m}): {n} values differ")
