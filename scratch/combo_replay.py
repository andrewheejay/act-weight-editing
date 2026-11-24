"""Replay 38 rows of stage3/combo.jsonl (2 per arm) with the current code and compare the full records.

Reason: awe/arms.py, rollout.py and export_checkpoint.py were edited while combo_a ran (PROGRESS.md, session 10 corrections).
Rows are drawn per arm with a fixed rng, one from the first and one from the second half of the arm's rows in file order.
Outcomes are not used to choose them. PYTHONPATH=. python scratch/combo_replay.py
"""
import json
import random

from awe import pool
from awe.combo import ALL, spec
from awe.paths import ROOT, exp

OUT = ROOT / "scratch" / "combo_replay.jsonl"


def main():
    rows = [json.loads(l) for l in (exp("transfer_cube") / "stage3" / "combo.jsonl").read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    print(len(rows), "rows", {a: len(v) for a, v in by.items()})
    assert set(by) == set(ALL) and all(len({r["seed"] for r in v}) == 500 for v in by.values())
    rng = random.Random(11)
    pick = []
    for a in sorted(by):
        v = by[a]
        pick += [rng.choice(v[:250]), rng.choice(v[250:])]
    pool.run("transfer_cube", [(spec(r["arm"]), r["seed"], None) for r in pick], OUT, budget="agent", purpose="combo_replay_check")
    new = {(r["arm"], r["seed"]): r for r in map(json.loads, OUT.read_text().splitlines())}
    same_hash = sum(new[(r["arm"], r["seed"])]["traj_hash"] == r["traj_hash"] for r in pick)
    same_rec = sum(new[(r["arm"], r["seed"])] == r for r in pick)
    print(f"{len(pick)} replayed, same traj_hash {same_hash}, same full record {same_rec}")


if __name__ == "__main__":
    main()
