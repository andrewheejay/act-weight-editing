"""Stage 3 step 5: validation checks on seeds 30000 to 30499. Resumable, every rollout goes to the ledger with its candidate id.

python -m awe.validate original            the original on all 500 validation seeds, stage3/validation/original.jsonl
python -m awe.validate candidate ID        one candidate from stage3/candidates.json, stage3/validation/ID.jsonl
python -m awe.validate baseline FAMILY     the frozen best setting of one baseline family (budget baselines), validation/base_FAMILY.jsonl
python -m awe.validate report              stage3/validation/summary.json, all checked candidates paired against the original

Rules fixed before the first validation rollout.
- At most 10 candidates ever. A candidate needs a line in hypotheses.md and an entry in stage3/candidates.json, both written before its run.
- Each candidate is run once on all 500 seeds and never re-run or changed. The file name is the id, so a finished file blocks a re-run.
- The pass rule is SPEC.md Stage 3: share of failures removed >= 25%, exact McNemar p < 0.05, strict gain >= -2 points.
- Baselines: each family gets one check, with the setting chosen on the design seeds (stage3/baselines_summary.json, best per family).
"""
import argparse
import json

from . import pool
from .arms import make_spec
from .ledger import ROOT
from .paths import exp
from .sweep_report import rows_of, stats, refs
from referee.stats import mcnemar_exact

SEEDS = range(30000, 30500)
MAX_CANDIDATES = 10


def vdir(task):
    d = exp(task) / "stage3" / "validation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def candidates(task):
    return json.loads((exp(task) / "stage3" / "candidates.json").read_text())


def candidate_spec(task, cid):
    c = candidates(task)[cid]
    return make_spec(cid, c["ops"])


def baseline_spec(task, family):
    s = json.loads((exp(task) / "stage3" / "baselines_summary.json").read_text())["families"][family]
    best = s["best"]
    if family == "shorter_chunk":
        return make_spec("base_shorter_chunk", n_action_steps=s["results"][best]["n_action_steps"])
    return make_spec(f"base_{family}", post=s["recipes"][best])


def used_candidates(task):
    """Distinct agent-budget candidates already in the ledger on validation seeds (any file)."""
    out = set()
    for f in sorted((ROOT / "ledger").glob("*.jsonl*")):
        import gzip

        op = gzip.open if f.name.endswith(".gz") else open
        with op(f, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("task") == task and r.get("seed") in SEEDS and r.get("budget") == "agent":
                    c = r.get("candidate") or r.get("arm")
                    if c != "original":
                        out.add(c)
    return out


def complete(path):
    return path.exists() and len({json.loads(l)["seed"] for l in path.read_text().splitlines() if l.strip()}) == len(SEEDS)


def run_one(task, spec, path, budget, purpose):
    if complete(path):
        print(f"{path.name} already complete, not re-run")
        return
    pool.run(task, [(spec, s, None) for s in SEEDS], path, budget=budget, purpose=purpose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["original", "candidate", "baseline", "report"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    t = a.task
    d = vdir(t)
    if a.cmd == "original":
        run_one(t, make_spec("original"), d / "original.jsonl", "agent", "validation_original")
    elif a.cmd == "candidate":
        assert a.id, "candidate needs an id"
        hyp = (exp(t) / "hypotheses.md").read_text()
        assert a.id in candidates(t), f"{a.id} is not in stage3/candidates.json"
        assert f"- {a.id} " in hyp or f"- {a.id}:" in hyp or f"{a.id} (" in hyp, f"{a.id} has no line in hypotheses.md"
        used = used_candidates(t)
        assert a.id in used or len(used) < MAX_CANDIDATES, f"{len(used)} candidates already checked"
        run_one(t, candidate_spec(t, a.id), d / f"{a.id}.jsonl", "agent", "validation_candidate")
    elif a.cmd == "baseline":
        assert a.id, "baseline needs a family"
        run_one(t, baseline_spec(t, a.id), d / f"base_{a.id}.jsonl", "baselines", "validation_baseline")
    else:
        report(t)


def report(task):
    d = vdir(task)
    reach, lat = refs(task)
    base = {r["seed"]: r for r in rows_of(d / "original.jsonl")}
    n_fail = sum(not r["benchmark_success"] for r in base.values())
    n = len(base)
    out = {"n": n, "original_failures": n_fail, "original_success": 1 - n_fail / n, "rule": __doc__, "results": {}}
    for f in sorted(d.glob("*.jsonl")):
        if f.name == "original.jsonl":
            continue
        rows = rows_of(f)
        if len({r["seed"] for r in rows}) != n:
            print("incomplete", f.name)
            continue
        s = stats(base, rows, reach, lat)
        share = (s["fixed"] - s["broken"]) / n_fail
        p = mcnemar_exact(s["fixed"], s["broken"])
        passed = share >= 0.25 and p < 0.05 and s["strict_gain"] >= -0.02
        out["results"][f.stem] = {**s, "share_of_failures_removed": round(share, 4), "p": p, "passes": bool(passed)}
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"original success {out['original_success']:.3f}, {n_fail} failures of {n}")
    for k, v in sorted(out["results"].items(), key=lambda x: -x[1]["gain"]):
        print(f"{k}: gain {100*v['gain']:+.1f} strict {100*v['strict_gain']:+.1f} fixed {v['fixed']} broken {v['broken']} share {v['share_of_failures_removed']:.2f} p {v['p']:.2e} passes {v['passes']}")


if __name__ == "__main__":
    main()
