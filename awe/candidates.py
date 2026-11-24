"""Stage 3 steps 5 and 6: the validation list and the choice rule, both fixed before any validation rollout.

python -m awe.candidates write       stage3/candidates.json, the ids below with their ops (a copy of awe/combo.py, refused if the file exists with other content)
python -m awe.candidates choose      stage3/choice.json, the rule below applied to the finished design and validation rollouts

List. Every id has a hypotheses.md line. All were run on the 1,000 design seeds (stage3/combo.jsonl.gz) before this list was written,
and they are the best of those 25 arms by design benchmark gain with at least two heads, plus the best single head:
r4, r5, r1, r2, r6, c06, c03 (two or three heads) and s1 (one head, reference only).
The design numbers are selected and optimistic. They only pick who gets a validation check, not who wins.

Choice rule.
1. A candidate passes validation when it removes at least 25% of the original's validation failures, the exact McNemar p is below 0.05,
   and its strict success is no more than 2 points below the original's on the validation seeds (SPEC.md Stage 3).
2. Only candidates with at least 40 distinct matched random edits can be chosen (7 other heads per op, so 7 for s1 and 49 or more for the
   others), because H2 cannot be tested otherwise. s1 is reference only: it can not be chosen.
3. Among the passing candidates allowed by 2, the one with the largest pooled net gain over the 1,000 design seeds and the 500 validation
   seeds (fixed minus broken on the benchmark score, each seed counted once). Ties by fewer broken seeds pooled, then fewer heads, then id.
   Design data is optimistic for every candidate. It is pooled because it has twice the seeds and all candidates were selected on it.
4. If no eligible candidate passes, the stop rule applies: the eligible one with the largest pooled net gain is frozen anyway and the
   report says it missed.
"""
import argparse
import json

from .combo import ALL
from .paths import exp
from .sweep_report import find, rows_of
from referee.stats import mcnemar_exact

IDS = ["r4", "r5", "r1", "r2", "r6", "c06", "c03", "s1"]
MIN_MATCHED = 40


def n_matched(cid):
    """distinct matched random edits: one other head (of 7) per op, same block, same scale. Blocks within a candidate are distinct."""
    assert len({b for b, _, _ in ALL[cid]}) == len(ALL[cid]), cid
    return 7 ** len(ALL[cid])


def write(task):
    from .combo import spec

    out = {c: {"ops": spec(c)["ops"], "heads": [list(x) for x in ALL[c]], "n_matched": n_matched(c)} for c in IDS}
    p = exp(task) / "stage3" / "candidates.json"
    if p.exists():
        assert json.loads(p.read_text()) == out, "candidates.json differs from awe/candidates.py"
        print("unchanged", p)
        return
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(p)


def paired_net(base, rows, seeds):
    f = sum(1 for s in seeds if rows[s]["benchmark_success"] and not base[s]["benchmark_success"])
    b = sum(1 for s in seeds if base[s]["benchmark_success"] and not rows[s]["benchmark_success"])
    return f, b


def choose(task):
    d = exp(task)
    cands = json.loads((d / "stage3" / "candidates.json").read_text())
    db = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
    vb = {r["seed"]: r for r in rows_of(d / "stage3" / "validation" / "original.jsonl")}
    design = {}
    for r in rows_of(find(d / "stage3", "combo.jsonl")):
        design.setdefault(r["arm"], {})[r["seed"]] = r
    nv_fail = sum(not r["benchmark_success"] for r in vb.values())
    tab = {}
    for c in IDS:
        vr = {r["seed"]: r for r in rows_of(d / "stage3" / "validation" / f"{c}.jsonl")}
        assert len(vr) == len(vb) == 500 and len(design[c]) == 1000, c
        fv, bv = paired_net(vb, vr, sorted(vb))
        fd, bd = paired_net(db, design[c], sorted(db))
        strict = sum(int(vr[s]["strict_success"]) - int(vb[s]["strict_success"]) for s in vb) / len(vb)
        p = mcnemar_exact(fv, bv)
        tab[c] = {"val_fixed": fv, "val_broken": bv, "val_gain": (fv - bv) / len(vb), "val_share_removed": (fv - bv) / nv_fail, "val_p": p,
                  "val_strict_gain": strict, "passes": bool((fv - bv) / nv_fail >= 0.25 and p < 0.05 and strict >= -0.02),
                  "design_fixed": fd, "design_broken": bd, "pooled_net": fv - bv + fd - bd, "pooled_broken": bv + bd,
                  "n_heads": len(ALL[c]), "n_matched": n_matched(c), "eligible": n_matched(c) >= MIN_MATCHED}
    passing = [c for c in IDS if tab[c]["passes"] and tab[c]["eligible"]]
    pool = passing or [c for c in IDS if tab[c]["eligible"]] or IDS
    key = lambda c: (-tab[c]["pooled_net"], tab[c]["pooled_broken"], tab[c]["n_heads"], c)
    best = min(pool, key=key)
    out = {"rule": __doc__, "original_validation_failures": nv_fail, "table": tab, "passing_and_eligible": passing,
           "chosen": best, "chosen_passes_validation": tab[best]["passes"],
           "stop_rule_used": not passing}
    (d / "stage3" / "choice.json").write_text(json.dumps(out, indent=1) + "\n")
    for c in sorted(IDS, key=key):
        t = tab[c]
        print(f"{c:4} val gain {100*t['val_gain']:+.1f} (f{t['val_fixed']} b{t['val_broken']}) share {t['val_share_removed']:.2f} p {t['val_p']:.2e} strict {100*t['val_strict_gain']:+.1f} "
              f"pooled {t['pooled_net']} broken {t['pooled_broken']} passes {t['passes']}")
    print("chosen", best, "passing", passing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "choose"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    {"write": write, "choose": choose}[a.cmd](a.task)
