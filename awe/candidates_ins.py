"""Stage 3 steps 5 and 6 for insertion: the validation list and the choice rule. The rule and the list rule below are fixed before any
validation rollout (written in session 25, before the group B and refinement results were read). The ids come out of the design files.

python -m awe.candidates_ins list        prints the ids that the list rule gives (stage3/combo_*.jsonl.gz must be complete)
python -m awe.candidates_ins write       stage3/candidates.json, the ids with their ops (refused if the file exists with other content)
python -m awe.candidates_ins choose      stage3/choice.json, the choice rule applied to the finished design and validation rollouts

List rule. Every combination or refinement (awe/combo_ins.py) with rows on all 1,000 design seeds (c06, c08, c02, c01 and r1 to r6) and at least two heads
is eligible to be checked. The 8 with the largest net strict gain on the 1,000 design seeds (fixed minus broken, primary score) go to validation. Ties: fewer
broken, fewer heads, id. Every id has a hypotheses.md line written before its rollouts. The design numbers are selected and optimistic. They only pick who gets a
validation check.

Choice rule (insertion: strict is the primary score, benchmark the guard).
1. A candidate passes validation when it removes at least 25% of the original's strict failures on the validation seeds, the exact McNemar p on strict is below
   0.05, and its benchmark success is no more than 2 points below the original's (SPEC.md Stage 3, with the primary and the guard of decisions.json).
2. Only candidates with at least 40 distinct matched random edits can be chosen (7 other heads per op), because H2 cannot be tested otherwise.
3. Among the passing candidates allowed by 2, the one with the largest pooled net strict gain over the 1,000 design seeds and the 500 validation seeds (fixed
   minus broken on strict, each seed once). Ties by fewer broken seeds pooled, then fewer heads, then id. Design data is optimistic for every candidate. It is
   pooled because it has twice the seeds and all candidates were selected on it.
4. If no eligible candidate passes, the stop rule applies: the eligible one with the largest pooled net strict gain is frozen anyway and the report says it missed.
   The 25% bar is far out of reach on this task (about 93% of episodes are strict failures), so this is the expected path.
"""
import argparse
import json

from .combo_ins import ALL, GROUP, spec
from .paths import exp
from .sweep_report import find, rows_of
from referee.stats import mcnemar_exact

N_LIST = 8
MIN_MATCHED = 40


def n_matched(cid):
    """distinct matched random edits: one other head (of 7) per op, same block, same scale. Blocks within a candidate are distinct."""
    assert len({b for b, _, _ in ALL[cid]}) == len(ALL[cid]), cid
    return 7 ** len(ALL[cid])


def design_rows(task, cid):
    rows = {}
    for grp in GROUP:
        p = find(exp(task) / "stage3", f"combo_{grp}.jsonl")
        if p.exists():
            for r in rows_of(p):
                if r["arm"] == cid:
                    rows[r["seed"]] = r
    return rows


def eligible_ids(task):
    """ids with rows on every one of the 1,000 design seeds, at least two heads and 40 matched random edits"""
    out = []
    for c in ALL:
        if len(ALL[c]) >= 2 and n_matched(c) >= MIN_MATCHED and len(design_rows(task, c)) == 1000:
            out.append(c)
    return out


def net(base, rows, seeds, key="strict_success"):
    f = sum(1 for s in seeds if rows[s][key] and not base[s][key])
    b = sum(1 for s in seeds if base[s][key] and not rows[s][key])
    return f, b


def the_list(task):
    base = {r["seed"]: r for r in rows_of(find(exp(task) / "stage1", "design_default.jsonl"))}
    seeds = sorted(base)
    assert len(seeds) == 1000
    tab = {}
    for c in eligible_ids(task):
        f, b = net(base, design_rows(task, c), seeds)
        tab[c] = (f - b, b, len(ALL[c]), c)
    ids = sorted(tab, key=lambda c: (-tab[c][0], tab[c][1], tab[c][2], tab[c][3]))
    return ids, {c: {"design_net": tab[c][0], "design_broken": tab[c][1]} for c in ids}


def the_ids(task):
    ids, _ = the_list(task)
    return ids[:N_LIST]


def write(task):
    ids = the_ids(task)
    out = {c: {"ops": spec(c)["ops"], "heads": [list(x) for x in ALL[c]], "n_matched": n_matched(c)} for c in ids}
    p = exp(task) / "stage3" / "candidates.json"
    if p.exists():
        assert json.loads(p.read_text()) == out, "candidates.json differs from awe/candidates_ins.py"
        print("unchanged", p)
        return
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(p, ids)


def choose(task):
    d = exp(task)
    ids = list(json.loads((d / "stage3" / "candidates.json").read_text()))
    db = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
    vb = {r["seed"]: r for r in rows_of(d / "stage3" / "validation" / "original.jsonl")}
    nv_fail = sum(not r["strict_success"] for r in vb.values())
    tab = {}
    for c in ids:
        dr = design_rows(task, c)
        vr = {r["seed"]: r for r in rows_of(d / "stage3" / "validation" / f"{c}.jsonl")}
        assert len(vr) == len(vb) == 500 and len(dr) == 1000, c
        fv, bv = net(vb, vr, sorted(vb))
        fd, bd = net(db, dr, sorted(db))
        gv, gbv = net(vb, vr, sorted(vb), "benchmark_success")
        p = mcnemar_exact(fv, bv)
        tab[c] = {"val_fixed": fv, "val_broken": bv, "val_strict_gain": (fv - bv) / len(vb), "val_share_removed": (fv - bv) / nv_fail, "val_p": p,
                  "val_benchmark_gain": (gv - gbv) / len(vb), "passes": bool((fv - bv) / nv_fail >= 0.25 and p < 0.05 and (gv - gbv) / len(vb) >= -0.02),
                  "design_fixed": fd, "design_broken": bd, "pooled_net": fv - bv + fd - bd, "pooled_broken": bv + bd,
                  "n_heads": len(ALL[c]), "n_matched": n_matched(c), "eligible": n_matched(c) >= MIN_MATCHED}
    passing = [c for c in ids if tab[c]["passes"] and tab[c]["eligible"]]
    pool_ = passing or [c for c in ids if tab[c]["eligible"]]
    key = lambda c: (-tab[c]["pooled_net"], tab[c]["pooled_broken"], tab[c]["n_heads"], c)
    best = min(pool_, key=key)
    out = {"rule": __doc__, "original_validation_strict_failures": nv_fail, "table": tab, "passing_and_eligible": passing,
           "chosen": best, "chosen_passes_validation": tab[best]["passes"], "stop_rule_used": not passing}
    (d / "stage3" / "choice.json").write_text(json.dumps(out, indent=1) + "\n")
    for c in sorted(ids, key=key):
        t = tab[c]
        print(f"{c:4} val strict gain {100*t['val_strict_gain']:+.1f} (f{t['val_fixed']} b{t['val_broken']}) share {t['val_share_removed']:.2f} p {t['val_p']:.2e} "
              f"bench {100*t['val_benchmark_gain']:+.1f} pooled {t['pooled_net']} broken {t['pooled_broken']} passes {t['passes']}")
    print("chosen", best, "passing", passing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "write", "choose"])
    ap.add_argument("--task", default="insertion")
    a = ap.parse_args()
    if a.cmd == "list":
        ids, tab = the_list(a.task)
        for c in ids:
            print(c, tab[c], "IN" if c in the_ids(a.task) else "out")
    else:
        {"write": write, "choose": choose}[a.cmd](a.task)
