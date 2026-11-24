"""The auto-greedy edit, SPEC.md Controls and Stage 4: a greedy search over the single-head sweep, with no reasoning. Budget `autogreedy`.

python -m awe.autogreedy run       stage3/autogreedy.jsonl, then freeze/autogreedy.json
python -m awe.autogreedy report    only rebuild freeze/autogreedy.json from the files already run

Rule, fixed before the first rollout of this script (docstring of this file).
- Pool. The single-head edits that survived the split-half check (stage2/split_half.json, `kept`, the same sign of gain in both halves),
  with a positive gain, ranked by the mean of the two half gains on the benchmark score (`gain_both_halves`). The first six are the pool.
- Start. The top edit of the pool alone, with its recorded results on seeds 20500 to 20999 (stage2/sweep_half2.jsonl, no new rollouts).
- Step. For each pool edit not yet used (a head may appear once), run the current set plus that edit on seeds 20500 to 20999 (500 rollouts,
  original as the paired baseline from stage1/design_default). Take the candidate with the largest net gain (fixed minus broken, benchmark score).
  Ties: fewer broken, then pool order. Accept it if its net gain is larger than the current set's net gain on the same seeds, else stop.
- Stop at 4 heads or when no addition improves. At most 5+4+3 = 12 combinations, 6,000 rollouts, all on design seeds. The validation seeds
  are never used.
- Everything is computed by this script from stored files. It was written while the validation job ran and before any validation result was
  read, after the design-seed choice of the edit candidates. It reads neither the candidates nor any validation file, and it is meant to be
  run only after the final edit is frozen (stage 4), so its result cannot influence the edit.
"""
import argparse
import json

from . import pool
from .arms import make_spec
from .combo import LONG
from .paths import exp
from .sweep_report import find, rows_of

SEEDS = range(20500, 21000)
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
    d = exp(task) / "stage2"
    sh = json.loads((d / "split_half.json").read_text())
    kept = [e for e in sh["kept"] if sh["results"][e]["gain_both_halves"] > 0]
    kept.sort(key=lambda e: (-sh["results"][e]["gain_both_halves"], e))
    return kept[:POOL_SIZE]


def single_rows(task, eid):
    rows = {}
    for p in (find(exp(task) / "stage2", "sweep_half2.jsonl"),):
        for r in rows_of(p):
            if r["arm"] == eid and r["seed"] in SEEDS:
                rows[r["seed"]] = r
    assert len(rows) == len(SEEDS), f"{eid}: {len(rows)} second-half rows"
    return rows


def combo_id(edits):
    return "ag_" + "_".join(edits)


def run_combo(task, edits):
    path = exp(task) / "stage3" / "autogreedy.jsonl"
    spec = make_spec(combo_id(edits), [parse(e) for e in edits])
    pool.run(task, [(spec, s, None) for s in SEEDS], path, budget="autogreedy", purpose="autogreedy")
    rows = {r["seed"]: r for r in rows_of(path) if r["arm"] == combo_id(edits)}
    assert len(rows) == len(SEEDS), combo_id(edits)
    return rows


def search(task, dry=False):
    """dry only builds the pool and the start (no rollouts), used in tests"""
    base = base_rows(task)
    pl = pool_list(task)
    log = []
    cur = [pl[0]]
    f, b = net(base, single_rows(task, pl[0]))
    log.append({"step": 1, "edits": cur[:], "fixed": f, "broken": b, "net": f - b, "from": "stage2/sweep_half2.jsonl.gz"})
    cur_net = f - b
    n_runs = 0
    while len(cur) < MAX_HEADS:
        options = []
        for e in pl:
            if e in cur or any(parse(e)["block"] == parse(c)["block"] and parse(e)["head"] == parse(c)["head"] for c in cur):
                continue
            rows = run_combo(task, cur + [e]) if not dry else None
            if rows is None:
                continue
            n_runs += len(SEEDS)
            f, b = net(base, rows)
            options.append({"edit": e, "fixed": f, "broken": b, "net": f - b, "gain": (f - b) / len(SEEDS)})
        if not options:
            break
        best = min(options, key=lambda o: (-o["net"], o["broken"], pl.index(o["edit"])))
        log.append({"step": len(cur) + 1, "current": cur[:], "options": options, "best": best["edit"], "accepted": best["net"] > cur_net})
        if best["net"] <= cur_net:
            break
        cur = cur + [best["edit"]]
        cur_net = best["net"]
    return cur, log, cur_net


def write(task, cur, log, cur_net):
    d = exp(task) / "freeze"
    d.mkdir(parents=True, exist_ok=True)
    ops = [parse(e) for e in cur]
    out = {
        "built_by": "awe/autogreedy.py (python -m awe.autogreedy run)", "rule": __doc__,
        "edits": cur, "ops": ops, "n_heads": len(cur), "pool": pool_list(task),
        "design_seeds_used": "20500 to 20999 (500), paired against the original",
        "final_net_seeds_of_500": cur_net, "search_log": log,
    }
    (d / "autogreedy.json").write_text(json.dumps(out, indent=1) + "\n")
    print("autogreedy:", cur, "ops", ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    # `report` runs the same search, all rollouts are already on file so nothing new is run (pool.run skips done seeds)
    cur, log, cur_net = search(a.task)
    write(a.task, cur, log, cur_net)


if __name__ == "__main__":
    main()
