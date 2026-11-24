"""The auto-greedy edit, SPEC.md Controls and Stage 4: a greedy search over the single-head sweep, with no reasoning. Budget `autogreedy`.

python -m awe.autogreedy run       stage3/autogreedy.jsonl, then freeze/autogreedy.json
python -m awe.autogreedy report    only rebuild freeze/autogreedy.json from the files already run
python -m awe.autogreedy check     no rollouts: the pool, the number of planned rollouts, the single-head starts on the search seeds

Rule, rewritten and fixed before the first rollout of this script (docstring of this file).
- Pool. The single-head edits that survived the split-half check (stage2/split_half.json, `kept`, the same sign of gain in both halves), with a
  positive gain, ranked by the mean of the two half gains on the benchmark score (`gain_both_halves`). The first six are the pool (x0 or x2 only,
  the two scales of the sweep).
- Search seeds. 450 design seeds, 20500 to 20949, paired against the original (stage1/design_default.jsonl). Results of the six single edits on
  these seeds come from the sweep (stage2/sweep_half2.jsonl, no new rollouts).
- Step 1. Every pair of pool edits (15) on the search seeds, 6,750 rollouts. "Starts from the six": every pair contains two of the six, so the
  search covers all six starts. The best is the pair with the largest net gain (fixed minus broken, benchmark score). Ties: fewer broken, then
  the smaller sum of pool ranks. It is accepted only if its net is larger than the best single edit's net on the same seeds (else the single edit
  is the result).
- Steps 2 and 3. From the accepted set, every single addition of a pool edit not yet used (4, then 3), 1,800 and 1,350 rollouts. The best by the same
  rule is accepted if its net gain is larger than the current set's, else the search stops. Up to 4 heads. At most 9,900 of the 10,000 rollouts.
- The result is the set with the largest net gain found along this path (a set that fails to beat its parent is never the result).
- Everything is computed by this script from stored files. No step reads the agent's candidates, their design numbers or any validation file.

What the author had seen when the rule was written. This file was rewritten on 2026-11-24 after the author (the agent) had seen the combination table
of its own candidates (stage3/combo_summary.json) on design seeds, including that c08 (enc3.h4.x3 + cross.h1.x0) scored -3.2 points on seeds 20500 to
20999. It had also seen the validation results of the agent's candidates. The rewrite replaced a version that took only the top single edit (cross.h1.x0)
as start, which could have ended early. The new rule is the plainer one and it uses the SPEC.md budget in full. The agent's own edit (the rule in
awe/candidates.py) was chosen on 1,000 design seeds and 500 validation seeds, this script tunes on 450 design seeds and nothing else, so the rival
is tuned on less data than the edit. Its result cannot influence the edit, which is frozen in choice.json before the script runs.
"""
import argparse
import json
from itertools import combinations

from . import pool
from .arms import make_spec
from .combo import LONG
from .paths import exp
from .sweep_report import find, rows_of

SEEDS = range(20500, 20950)
POOL_SIZE = 6
MAX_HEADS = 4


def parse(eid):
    """'cross.h1.x0' -> op dict"""
    b, h, x = eid.split(".")
    s = float(x[1:])
    return {"op": "scale_head", "block": LONG[b], "head": int(h[1:]), "scale": int(s) if s == int(s) else s}


def base_rows(task):
    return {r["seed"]: r for r in rows_of(find(exp(task) / "stage1", "design_default.jsonl")) if r["seed"] in SEEDS}


def net(base, rows):
    f = sum(rows[s]["benchmark_success"] and not base[s]["benchmark_success"] for s in SEEDS)
    b = sum(base[s]["benchmark_success"] and not rows[s]["benchmark_success"] for s in SEEDS)
    return f, b


def pool_list(task):
    sh = json.loads((exp(task) / "stage2" / "split_half.json").read_text())
    kept = [e for e in sh["kept"] if sh["results"][e]["gain_both_halves"] > 0]
    kept.sort(key=lambda e: (-sh["results"][e]["gain_both_halves"], e))
    return kept[:POOL_SIZE]


def single_rows(task, eid):
    rows = {r["seed"]: r for r in rows_of(find(exp(task) / "stage2", "sweep_half2.jsonl")) if r["arm"] == eid and r["seed"] in SEEDS}
    assert len(rows) == len(SEEDS), f"{eid}: {len(rows)} second-half rows"
    return rows


def distinct_heads(edits):
    ks = [(parse(e)["block"], parse(e)["head"]) for e in edits]
    return len(set(ks)) == len(ks)


def combo_id(edits):
    return "ag_" + "_".join(edits)


def canon(edits, pl):
    return tuple(sorted(edits, key=pl.index))


def runs_file(task):
    return exp(task) / "stage3" / "autogreedy.jsonl"


def all_runs(task):
    """arm id -> {seed: row}, from the plain or gzipped file"""
    f = find(exp(task) / "stage3", "autogreedy.jsonl")
    g = {}
    if f.exists():
        for r in rows_of(f):
            g.setdefault(r["arm"], {})[r["seed"]] = r
    return g


def run_sets(task, sets):
    have = all_runs(task)
    todo = [e for e in sets if len(have.get(combo_id(e), {})) < len(SEEDS)]
    if not todo:
        return
    assert runs_file(task).exists() or not find(exp(task) / "stage3", "autogreedy.jsonl").exists(), "unpack stage3/autogreedy.jsonl.gz first"
    specs = [make_spec(combo_id(e), [parse(x) for x in e]) for e in todo]
    pool.run(task, [(sp, s, None) for sp in specs for s in SEEDS], runs_file(task), budget="autogreedy", purpose="autogreedy")


def rows_for(task, edits):
    rows = all_runs(task).get(combo_id(edits), {})
    assert len(rows) == len(SEEDS), combo_id(edits)
    return rows


def search(task):
    base = base_rows(task)
    pl = pool_list(task)
    rank = {e: i for i, e in enumerate(pl)}
    log = []

    def score(edits):
        rows = single_rows(task, edits[0]) if len(edits) == 1 else rows_for(task, edits)
        f, b = net(base, rows)
        return {"edits": list(edits), "fixed": f, "broken": b, "net": f - b, "n_seeds": len(SEEDS)}

    key = lambda o: (-o["net"], o["broken"], sum(rank[e] for e in o["edits"]))
    singles = sorted((score((e,)) for e in pl), key=key)
    log.append({"step": 1, "what": "single edits from the sweep, no new rollouts", "options": singles, "best": singles[0]["edits"]})
    cur = singles[0]

    def advance(options, what):
        """Run the sets, score them, keep the best if it beats the current net."""
        nonlocal cur
        run_sets(task, options)
        scored = sorted((score(o) for o in options), key=key)
        best = scored[0]
        acc = best["net"] > cur["net"]
        log.append({"step": len(log) + 1, "what": what, "options": scored, "best": best["edits"], "current_net_before": cur["net"], "accepted": acc})
        if acc:
            cur = best
        return acc

    pairs = [canon(p, pl) for p in combinations(pl, 2) if distinct_heads(p)]
    if advance(pairs, "all pairs"):
        while len(cur["edits"]) < MAX_HEADS:
            ex = [canon(cur["edits"] + [e], pl) for e in pl if e not in cur["edits"] and distinct_heads(list(cur["edits"]) + [e])]
            if not ex or not advance(ex, f"{len(cur['edits']) + 1} heads"):
                break
    return cur, log, pl


def planned(task):
    n = len(SEEDS)
    return (15 + 4 + 3) * n


def write(task, cur, log, pl):
    d = exp(task) / "freeze"
    d.mkdir(parents=True, exist_ok=True)
    ops = [parse(e) for e in cur["edits"]]
    out = {
        "built_by": "awe/autogreedy.py (python -m awe.autogreedy run)", "rule": __doc__,
        "edits": cur["edits"], "ops": ops, "n_heads": len(ops), "pool": pl,
        "search_seeds": f"{SEEDS[0]} to {SEEDS[-1]} ({len(SEEDS)} design seeds), paired against the original",
        "final": cur, "search_log": log,
    }
    (d / "autogreedy.json").write_text(json.dumps(out, indent=1) + "\n")
    print("autogreedy:", cur["edits"], "net", cur["net"], "of", len(SEEDS), "ops", ops)


def check(task):
    base = base_rows(task)
    pl = pool_list(task)
    print("pool", pl)
    for e in pl:
        print(" ", e, "single on search seeds (fixed, broken)", net(base, single_rows(task, e)))
    pairs = [canon(p, pl) for p in combinations(pl, 2) if distinct_heads(p)]
    print(len(pairs), "pairs, planned rollouts at most", planned(task), "cap 10,000")
    assert planned(task) <= 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "check"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    if a.cmd == "check":
        return check(a.task)
    # `report` runs the same search, all rollouts are already on file so nothing new is run (pool.run skips done seeds)
    cur, log, pl = search(a.task)
    write(a.task, cur, log, pl)


if __name__ == "__main__":
    main()
