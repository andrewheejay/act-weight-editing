"""Stage 3 step 5 for insertion: validation checks on seeds 30000 to 30499. Resumable, every rollout goes to the ledger with its candidate id.
Same rules and files as awe/validate.py (which serves transfer cube). What differs is the pass rule, because the primary score is strict:

python -m awe.validate_ins original            the original on all 500 validation seeds, stage3/validation/original.jsonl
python -m awe.validate_ins candidate ID        one candidate from stage3/candidates.json, stage3/validation/ID.jsonl
python -m awe.validate_ins baseline FAMILY     the frozen best setting of one baseline family (budget baselines), validation/base_FAMILY.jsonl
python -m awe.validate_ins report              stage3/validation/summary.json, all checked candidates paired against the original

Rules fixed before the first validation rollout.
- At most 10 candidates ever. A candidate needs a line in hypotheses.md and an entry in stage3/candidates.json, both written before its run.
- Each candidate is run once on all 500 seeds and never re-run or changed. The file name is the id, so a finished file blocks a re-run.
- The pass rule (SPEC.md Stage 3 with the primary and guard of decisions.json): share of strict failures removed >= 25%, exact McNemar p on
  strict < 0.05, benchmark success at most 2 points below the original's.
- Baselines: each family gets one check, with the setting chosen on the design seeds (stage3/baselines_summary.json, best per family).
"""
import argparse
import json

from .arms import make_spec
from .candidates_ins import net
from .paths import exp
from .sweep_ins_report import refs, stats
from .sweep_report import rows_of
from .validate import SEEDS, MAX_CANDIDATES, baseline_spec, candidate_spec, candidates, run_one, used_candidates, vdir
from referee.stats import mcnemar_exact


def report(task):
    d = vdir(task)
    ref = refs(task)
    base = {r["seed"]: r for r in rows_of(d / "original.jsonl")}
    n_fail = sum(not r["strict_success"] for r in base.values())
    n = len(base)
    out = {"n": n, "original_strict_failures": n_fail, "original_strict": 1 - n_fail / n, "rule": __doc__, "results": {}}
    for f in sorted(d.glob("*.jsonl")):
        if f.name == "original.jsonl":
            continue
        rows = rows_of(f)
        if len({r["seed"] for r in rows}) != n:
            print("incomplete", f.name)
            continue
        s = stats(base, rows, ref)
        share = (s["fixed"] - s["broken"]) / n_fail
        p = mcnemar_exact(s["fixed"], s["broken"])
        passed = share >= 0.25 and p < 0.05 and s["guard_gain"] >= -0.02
        out["results"][f.stem] = {**s, "strict_gain": s["gain"], "share_of_failures_removed": round(share, 4), "p": p, "passes": bool(passed)}
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"original strict {out['original_strict']:.3f}, {n_fail} strict failures of {n}")
    for k, v in sorted(out["results"].items(), key=lambda x: -x[1]["gain"]):
        print(f"{k}: strict gain {100*v['gain']:+.1f} guard {100*v['guard_gain']:+.1f} fixed {v['fixed']} broken {v['broken']} share {v['share_of_failures_removed']:.2f} p {v['p']:.2e} passes {v['passes']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["original", "candidate", "baseline", "report"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--task", default="insertion")
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


if __name__ == "__main__":
    main()
