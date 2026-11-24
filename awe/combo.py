"""Stage 3 step 3: hand-written combinations of up to 6 heads. Resumable, every rollout goes to the ledger.

python -m awe.combo run A [ID ...]      ids (default all) on design seeds 20500 to 20999, stage3/combo.jsonl
python -m awe.combo run B ID ...        the named ids on design seeds 20000 to 20499, same file
python -m awe.combo run AB ID ...       the named ids on all 1,000 design seeds (refinement round r1 to r6)
python -m awe.combo report              stage3/combo_summary.json, paired against the original (stage1/design_default.jsonl)

Every combination has a line in hypotheses.md, written before it runs. Nothing here chooses heads or scales by search.
Seeds. The heads came from 20000 to 20499 (and partly from 20500 to 20999), and the dose scales from 20000 to 20299, so group A
(20500 to 20999) is the least selected half. Numbers from B are optimistic for anything that was chosen there.
"""
import argparse
import json

from . import pool
from .arms import make_spec
from .edits import BLOCKS
from .paths import exp
from .sweep import SHORT, arm_id
from .sweep_report import find, refs, rows_of, stats

GROUP = {"A": range(20500, 21000), "B": range(20000, 20500), "AB": range(20000, 21000)}
LONG = {v: k for k, v in SHORT.items()}

# (short block name, head, scale)
SINGLES = {
    "s1": [("enc3", 4, 3)],
    "s2": [("cross", 1, 0.5)],
}
COMBOS = {
    "c01": [("enc3", 4, 3), ("cross", 1, 0.5)],
    "c02": [("enc3", 4, 3), ("cross", 3, 2)],
    "c03": [("enc3", 4, 3), ("enc0", 1, 0)],
    "c04": [("enc3", 4, 3), ("enc0", 6, 0)],
    "c05": [("cross", 1, 0.5), ("cross", 3, 2)],
    "c06": [("cross", 1, 0.5), ("enc0", 1, 0)],
    "c07": [("cross", 3, 2), ("enc0", 1, 0)],
    "c08": [("enc3", 4, 3), ("cross", 1, 0)],
    "c09": [("enc3", 4, 3), ("cross", 1, 0.5), ("cross", 3, 2)],
    "c10": [("enc3", 4, 3), ("cross", 1, 0.5), ("enc0", 1, 0)],
    "c11": [("enc3", 4, 3), ("cross", 3, 2), ("enc0", 1, 0)],
    "c12": [("enc3", 4, 3), ("cross", 1, 0.5), ("enc0", 6, 0)],
    "c13": [("enc3", 4, 2), ("cross", 1, 0.5), ("cross", 3, 2), ("enc0", 1, 0)],
    "c14": [("enc3", 4, 3), ("cross", 1, 0.5), ("enc0", 1, 0), ("enc0", 6, 0)],
    "c15": [("enc3", 4, 2), ("cross", 1, 0.5), ("cross", 3, 1.5), ("enc0", 1, 0), ("enc1", 7, 0)],
    "c16": [("enc3", 4, 3), ("cross", 1, 0.5), ("cross", 3, 2), ("enc0", 1, 0), ("enc0", 6, 0)],
    "c17": [("enc3", 4, 3), ("cross", 6, 0.5), ("cross", 1, 0.5)],
}
# the refinement round, written after the group A and B results of the combinations (hypotheses.md), run with `run AB`
REFINE = {
    "r1": [("enc3", 4, 3), ("enc1", 7, 0)],
    "r2": [("enc3", 4, 2), ("enc0", 1, 0)],
    "r3": [("cross", 1, 0.5), ("enc0", 1, 0), ("enc1", 7, 0)],
    "r4": [("enc3", 4, 2), ("cross", 1, 0.5)],
    "r5": [("enc3", 4, 3), ("enc2", 3, 0.5)],
    "r6": [("cross", 1, 0.5), ("enc0", 1, 0), ("enc3", 4, 1.5)],
}
ALL = {**SINGLES, **COMBOS, **REFINE}


def spec(cid):
    ops = [{"op": "scale_head", "block": LONG[b], "head": h, "scale": s} for b, h, s in ALL[cid]]
    return make_spec(cid, ops)


def run(task, group, ids):
    sp = [spec(i) for i in (ids or list(ALL))]
    pool.run(task, [(x, s, None) for x in sp for s in GROUP[group]], exp(task) / "stage3" / "combo.jsonl",
             budget="agent", purpose=f"combo_{group}")


def report(task):
    d = exp(task)
    reach, lat = refs(task)
    base = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
    g = {}
    for r in rows_of(find(d / "stage3", "combo.jsonl")):
        g.setdefault(r["arm"], []).append(r)
    out = {}
    for cid, rows in g.items():
        heads = [list(x) for x in ALL[cid]]
        rec = {"heads": heads}
        for name, rng in (("A", GROUP["A"]), ("B", GROUP["B"]), ("all", range(20000, 21000))):
            sub = [r for r in rows if r["seed"] in rng]
            if sub:
                rec[name] = stats(base, sub, reach, lat)
        out[cid] = rec
    (d / "stage3" / "combo_summary.json").write_text(json.dumps({"rule": __doc__, "combos": out}, indent=1))
    for cid in sorted(out):
        for name in ("A", "B", "all"):
            if name in out[cid]:
                s = out[cid][name]
                print(f"{cid} {name:>3}: n {s['n']} gain {100*s['gain']:+.1f} strict {100*s['strict_gain']:+.1f} "
                      f"fixed {s['fixed']} broken {s['broken']} p {s['p_mcnemar']:.4f} gf {s['grasp_fail']} "
                      f"reach {s['reach_err_median_cm']} lat {s['cmd_lat8_dev_median_cm_clean']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("group", nargs="?", choices=list(GROUP))
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    if a.cmd == "run":
        assert a.group, "run needs a group"
        assert all(i in ALL for i in a.ids), a.ids
        run(a.task, a.group, a.ids)
    else:
        report(a.task)
