"""Stage 3 dose response for insertion, SPEC.md Stage 3 step 2. Resumable, every rollout goes to the ledger.

python -m awe.dose_ins run       x0.5, x1.5, x3, x4 of the 8 candidate heads on seeds 20000 to 20299
python -m awe.dose_ins report    stage3/dose_response.json (x0, x1 and x2 come from the sweep rows on the same seeds)

Rules fixed before the run (2026-11-25, session 23, the sweep and split-half results were known)
- Heads: positive strict gain in both halves of the split-half check, the 8 largest mean gains (stage2/split_half.json).
- Tracking number: the median of pre_tip_lat over the episodes that meet (cm), with the met count beside it (stage1/decisions.json).
- Dial: the tracking number moves the same way at every step of x0.5, x1, x1.5, x2, x3, x4. The same test is reported with x0 in front.
  Beside it, the strict gain at every scale (the same test), the met count and the share of pin contacts that start after step 375.
- The seeds 20000 to 20299 also chose the heads (they are the first 300 of the sweep's 500), so the x0 and x2 values are optimistic.
- No scale is chosen by a search. The numbers inform hand-written candidates, each listed in hypotheses.md first.
"""
import argparse
import json

import numpy as np

from . import pool
from .arms import make_spec
from .paths import exp
from .sweep import SHORT, arm_id
from .sweep_ins_report import refs, stats
from .sweep_report import find, grouped

SEEDS = range(20000, 20300)
NEW, ALL = (0.5, 1.5, 3, 4), (0, 0.5, 1, 1.5, 2, 3, 4)
LONG = {v: k for k, v in SHORT.items()}
N_HEADS_CHOSEN = 8
LATE = 375  # a pin contact starting at step 375 or later cannot reach 25 strict steps within the 400-step episode


def heads(task):
    r = json.loads((exp(task) / "stage2" / "split_half.json").read_text())["results"]
    pos = [a for a, v in r.items() if v["first_half"]["gain"] > 0 and v["second_half"]["gain"] > 0]
    pos.sort(key=lambda a: (-r[a]["gain_both_halves"], a))
    out = []
    for a in pos:
        b, h, _ = a.split(".")[:3]
        if (b, int(h[1:])) not in out:
            out.append((b, int(h[1:])))
    return out[:N_HEADS_CHOSEN]


def specs(task):
    sp = []
    for b, h in heads(task):
        for s in NEW:
            a = arm_id(LONG[b], h, s)
            sp.append(make_spec(a, [{"op": "scale_head", "block": LONG[b], "head": h, "scale": s}]))
    return sp


def run(task):
    pool.run(task, [(x, s, None) for x in specs(task) for s in SEEDS], exp(task) / "stage3" / "dose.jsonl", budget="agent", purpose="dose")


def monotone(v):
    if any(x is None for x in v):
        return None
    d = [b - a for a, b in zip(v, v[1:])]
    return all(x > 0 for x in d) or all(x < 0 for x in d)


def timing(rows):
    meet = [r["t_meet"] for r in rows if r["t_meet"] >= 0]
    pin = [r["t_pin"] for r in rows if r["t_pin"] >= 0]
    return {"median_t_meet": float(np.median(meet)) if meet else None, "median_t_pin": float(np.median(pin)) if pin else None,
            "late_pin_share": round(float(np.mean([p >= LATE for p in pin])), 4) if pin else None}


def report(task):
    d = exp(task)
    ref = refs(task)
    g2 = grouped(find(d / "stage2", "sweep.jsonl"))
    g3 = grouped(find(d / "stage3", "dose.jsonl"))
    keep = lambda rows: [r for r in rows if r["seed"] in SEEDS]
    base = {r["seed"]: r for r in keep(g2["original"])}
    out = {}
    for b, h in heads(task):
        tab = {}
        for s in ALL:
            if s == 1:
                rows = list(base.values())
            else:
                a = arm_id(LONG[b], h, s)
                rows = keep(g2[a]) if s in (0, 2) else g3[a]
            assert len({r["seed"] for r in rows}) == len(SEEDS), (b, h, s)
            tab[str(s)] = {**stats(base, rows, ref), **timing(rows)}
        num = {k: [tab[str(s)][k] for s in ALL] for k in ("tip_median_cm", "gain", "n_met", "guard_gain", "median_t_pin", "late_pin_share")}
        dial = {k: {"x0.5_to_x4": monotone(v[1:]), "x0_to_x4": monotone(v)} for k, v in num.items() if k in ("tip_median_cm", "gain", "n_met")}
        out[f"{b}.h{h}"] = {"by_scale": tab, "dial": dial, "tracking_by_scale": num["tip_median_cm"]}
        print(f"{b}.h{h}", {k: dict(zip(ALL, v)) for k, v in num.items()}, dial)
    (d / "stage3").mkdir(exist_ok=True)
    (d / "stage3" / "dose_response.json").write_text(json.dumps({
        "seeds": [SEEDS[0], SEEDS[-1]], "scales": list(ALL), "tracking_number": "tip_median_cm (median pre_tip_lat over the episodes that meet)",
        "rule": __doc__, "heads": out}, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--task", default="insertion")
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a.task)
