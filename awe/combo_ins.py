"""Stage 3 step 3 for insertion: hand-written combinations of up to 5 heads. Resumable, every rollout goes to the ledger.

python -m awe.combo_ins run A [ID ...]     ids (default all) on design seeds 20500 to 20999, stage3/combo_A.jsonl
python -m awe.combo_ins run B ID ...       the named ids on design seeds 20000 to 20499, stage3/combo_B.jsonl
python -m awe.combo_ins run AB ID ...      refinement ids (r*) on all 1,000 design seeds, stage3/combo_AB.jsonl
python -m awe.combo_ins report             stage3/combo_summary.json, paired against the original (stage1/design_default.jsonl)

Every combination has a line in hypotheses.md, written before it runs. Nothing here chooses heads or scales by search.
Seeds. The heads came from 20000 to 20499 (sweep) and 20500 to 20999 (recheck), and the dose scales from 20000 to 20299, so group A
(20500 to 20999) is the half that did not choose the scales. Numbers from B are optimistic for anything that was chosen there.
The primary score is strict success. benchmark is the guard (it may drop by no more than 2 points, SPEC.md Stage 3).
"""
import argparse
import json

from . import pool
from .arms import make_spec
from .dose_ins import timing
from .paths import exp
from .sweep import SHORT, arm_id
from .sweep_ins_report import refs, stats
from .sweep_report import find, rows_of

GROUP = {"A": range(20500, 21000), "B": range(20000, 20500), "AB": range(20000, 21000)}
LONG = {v: k for k, v in SHORT.items()}

# (short block name, head, scale), one head per block. Reasons are in hypotheses.md.
COMBOS = {
    "c01": [("cross", 0, 0), ("enc3", 4, 2)],
    "c02": [("cross", 0, 0), ("enc0", 6, 2)],
    "c03": [("cross", 0, 0), ("enc2", 1, 0)],
    "c04": [("cross", 0, 0), ("enc2", 6, 0)],
    "c05": [("cross", 0, 0), ("enc1", 1, 2)],
    "c06": [("cross", 0, 0), ("enc3", 3, 0)],
    "c07": [("cross", 0, 0.5), ("enc3", 4, 2)],
    "c08": [("cross", 0, 0), ("enc3", 4, 2), ("enc0", 6, 2)],
    "c09": [("cross", 0, 0), ("enc3", 4, 2), ("enc2", 1, 0)],
    "c10": [("cross", 0, 0), ("enc0", 6, 2), ("enc2", 1, 0)],
    "c11": [("cross", 0, 0), ("enc3", 4, 2), ("enc0", 6, 2), ("enc2", 1, 0)],
    "c12": [("cross", 0, 0.5), ("enc3", 4, 2), ("enc0", 6, 2)],
    "c13": [("cross", 2, 0), ("enc3", 4, 2)],
    "c14": [("cross", 2, 0), ("enc0", 6, 2), ("enc2", 1, 0)],
    "c15": [("enc3", 4, 2), ("enc0", 6, 2), ("enc2", 1, 0)],
    "c16": [("cross", 0, 0), ("enc3", 4, 2), ("enc0", 6, 2), ("enc2", 1, 0), ("enc1", 1, 2)],
}
REFINE = {}  # filled after the group A and B results, each id with a line in hypotheses.md written before it runs
ALL = {**COMBOS, **REFINE}

# reference single: cross.h0.x0 has rows on both halves in the sweep files (stage2/sweep.jsonl, sweep_half2.jsonl)
SINGLE_REFS = {"s_cross.h0.x0": ("cross", 0, 0)}


def spec(cid):
    assert len({b for b, _, _ in ALL[cid]}) == len(ALL[cid]), f"{cid}: one head per block"
    ops = [{"op": "scale_head", "block": LONG[b], "head": h, "scale": s} for b, h, s in ALL[cid]]
    return make_spec(cid, ops)


def run(task, group, ids):
    sp = [spec(i) for i in (ids or list(ALL))]
    pool.run(task, [(x, s, None) for x in sp for s in GROUP[group]], exp(task) / "stage3" / f"combo_{group}.jsonl",
             budget="agent", purpose=f"combo_{group}")


def ref_rows(task):
    """the single-head reference on the design seeds (cross.h0.x0 rows of the sweep)"""
    d = exp(task) / "stage2"
    rows = {}
    for name in ("sweep.jsonl", "sweep_half2.jsonl"):
        for r in rows_of(find(d, name)):
            if r["arm"] == arm_id(LONG["cross"], 0, 0):
                rows[r["seed"]] = r
    return list(rows.values())


def report(task):
    d = exp(task)
    ref = refs(task)
    base = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
    g = {"s_cross.h0.x0": ref_rows(task)}
    for grp in GROUP:
        p = find(d / "stage3", f"combo_{grp}.jsonl")
        if p.exists():
            for r in rows_of(p):
                g.setdefault(r["arm"], []).append(r)
    out = {}
    for cid, rows in g.items():
        heads = [list(x) for x in ALL[cid]] if cid in ALL else [list(SINGLE_REFS.get(cid, ()))]
        rec = {"heads": heads}
        for name, rng in (("A", GROUP["A"]), ("B", GROUP["B"]), ("all", range(20000, 21000))):
            sub = [r for r in rows if r["seed"] in rng]
            if sub:
                rec[name] = {**stats(base, sub, ref), **timing(sub)}
        out[cid] = rec
    (d / "stage3" / "combo_summary.json").write_text(json.dumps({"rule": __doc__, "combos": out}, indent=1) + "\n")
    for cid in sorted(out):
        for name in ("A", "B", "all"):
            if name in out[cid]:
                s = out[cid][name]
                print(f"{cid:14} {name:>3}: n {s['n']} strict {100*s['strict']:.1f} gain {100*s['gain']:+.1f} guard {100*s['guard_gain']:+.1f} "
                      f"fixed {s['fixed']} broken {s['broken']} p {s['p_mcnemar']:.4f} pin {s['n_pin']} late {s['late_pin_share']} tip {s['tip_median_cm']} "
                      f"A/M {s['parts']['A']}/{s['parts']['M']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("group", nargs="?", choices=list(GROUP))
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--task", default="insertion")
    a = ap.parse_args()
    if a.cmd == "run":
        assert a.group, "run needs a group"
        assert all(i in ALL for i in a.ids), a.ids
        run(a.task, a.group, a.ids)
    else:
        report(a.task)
