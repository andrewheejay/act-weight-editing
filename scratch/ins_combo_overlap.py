"""Offline counts on finished design rows (no rollouts), session 25. python scratch/ins_combo_overlap.py
1. Group A strict successes and rescues (strict where the original was not) per arm, shared with c06 and the union (the numbers of the correction in hypotheses.md).
2. The original and the listed candidates on all 1,000 design seeds: pins, late share, strict among pins, parts.
3. The design box and the fifths of peg x and peg y that analysis.py ZONES refers to.
4. c06 against the best setting of each baseline family on the design seeds, paired, strict.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from awe.candidates_ins import design_rows
from awe.combo_ins import ref_rows
from awe.dose_ins import timing
from awe.paths import exp
from awe.sweep_ins_report import part
from awe.sweep_report import find, rows_of
from referee.stats import mcnemar_exact

d = exp("insertion")
base = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
A = range(20500, 21000)
g = {"single": {r["seed"]: r for r in ref_rows("insertion")}}
for r in rows_of(find(d / "stage3", "combo_A.jsonl")):
    g.setdefault(r["arm"], {})[r["seed"]] = r
succ = lambda a: {s for s in A if g[a][s]["strict_success"]}
resc = lambda a: {s for s in succ(a) if not base[s]["strict_success"]}
print("1. group A, original strict", sum(base[s]["strict_success"] for s in A))
for a in ("c06", "c08", "c02", "c01", "c15", "single"):
    print(f"  {a:7} strict {len(succ(a))} (shared {len(succ(a) & succ('c06'))}, union {len(succ(a) | succ('c06'))})  rescues {len(resc(a))} (shared {len(resc(a) & resc('c06'))}, union {len(resc(a) | resc('c06'))})")
print("2. 1,000 design seeds")
for a, rows in [("original", base)] + [(c, design_rows("insertion", c)) for c in ("c06", "r1", "r2", "c08", "r6", "r5", "r3", "c02", "r4", "c01")]:
    rs = list(rows.values())
    pins = [r for r in rs if r["t_pin"] >= 0]
    print(f"  {a:8} n {len(rs)} met {sum(r['t_meet'] >= 0 for r in rs)} pins {len(pins)} late {timing(rs)['late_pin_share']} strict/pins {sum(r['strict_success'] for r in pins)}/{len(pins)} "
          f"parts {({k: sum(part(r) == k for r in rs) for k in ('S', 'A', 'M')})} grasp {sum(part(r) not in 'SAM' for r in rs)}")
print("3. box and fifths")
for k in ("peg_x0", "peg_y0", "sock_x0", "sock_y0"):
    v = np.array([r[k] for r in base.values()])
    print(f"  {k} {v.min():.3f} to {v.max():.3f} quintile edges {np.round(np.quantile(v, [.2, .4, .6, .8]), 3)}")
print("4. c06 against baselines on the design seeds, strict")
bs = json.loads((d / "stage3" / "baselines_summary.json").read_text())["families"]
R = {}
for f in ("baselines_screen.jsonl", "baselines_full.jsonl"):
    for r in rows_of(find(d / "stage3", f)):
        R.setdefault(r["arm"], {})[r["seed"]] = r
R["chunk50"] = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_chunk50.jsonl"))}
c = design_rows("insertion", "c06")
for fam, v in bs.items():
    q = R.get(v["best"], {})
    ss = sorted(set(q) & set(c))
    f = sum(c[s]["strict_success"] and not q[s]["strict_success"] for s in ss)
    b = sum(q[s]["strict_success"] and not c[s]["strict_success"] for s in ss)
    print(f"  {fam} {v['best']} n {len(ss)} c06 only {f} baseline only {b} diff {100 * (f - b) / max(1, len(ss)):+.1f} p {mcnemar_exact(f, b):.4f}")
