"""Compare the oracle results of denormal field 2 with field 0 and of field 1 with field 3, per group and IEEE mode.

Reads data/matrix.json written by measure.py; the first 196 rows are the 14 x 14 operand pairs, the rest padding.
"""
import json
import sys
from pathlib import Path

REAL = 196
matrix = json.load(open(Path(sys.argv[1] if len(sys.argv) > 1 else "data/matrix.json")))
for group in ("f32", "f64_minmax", "f64_cmpst"):
    for ieee in (0, 1):
        for a, b in ((2, 0), (1, 3)):
            left = matrix[group][f"ieee{ieee}_field{a}"][:REAL]
            right = matrix[group][f"ieee{ieee}_field{b}"][:REAL]
            differ = sum(1 for x, y in zip(left, right) if x != y)
            print(f"{group} ieee{ieee} field{a} vs field{b}: {differ} of {len(left)} rows differ")
