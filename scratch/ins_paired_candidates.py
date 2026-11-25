"""Paired exact McNemar between insertion validation candidates (500 validation seeds, strict). No rollouts. Run: python scratch/ins_paired_candidates.py"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from referee.stats import mcnemar_exact

V = Path(__file__).resolve().parents[1] / "experiments" / "insertion" / "stage3" / "validation"


def rows(n):
    p = V / f"{n}.jsonl"
    if not p.exists():
        p = Path(str(p) + ".gz")
    op = gzip.open if p.suffix == ".gz" else open
    return {r["seed"]: r for r in (json.loads(l) for l in op(p, "rt") if l.strip())}


o = rows("original")
for a, b in (("r1", "c06"), ("r1", "r2"), ("r1", "base_constant_offset")):
    A, B = rows(a), rows(b)
    x = sum(A[s]["strict_success"] and not B[s]["strict_success"] for s in o)
    y = sum(B[s]["strict_success"] and not A[s]["strict_success"] for s in o)
    print(f"{a} only {x}, {b} only {y}, exact McNemar p {mcnemar_exact(x, y):.3g}")
