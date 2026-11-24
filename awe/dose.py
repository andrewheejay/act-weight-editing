"""Stage 3 dose response, SPEC.md Stage 3 step 2. Resumable, every rollout goes to the ledger.

python -m awe.dose run       x0.5, x1.5, x3, x4 of the 8 candidate heads on seeds 20000 to 20299
python -m awe.dose report    stage3/dose_response.json (x0, x1 and x2 come from the sweep rows on the same seeds)

Rules fixed before the run
- Heads: positive gain in both halves, the 8 largest mean gains in split_half.json.
- Dial: the tracking number (median reach error, all episodes that have it) moves the same way at every step of
  x0.5, x1, x1.5, x2, x3, x4. The same test is reported with x0 in front, for the grasp failure count and for cmd_lat8.
- No scale is chosen by a search. The numbers inform hand-written candidates, each listed in hypotheses.md first.
"""
import argparse
import json

from . import edits, pool
from .arms import make_spec
from .paths import exp
from .sweep import SHORT, arm_id
from .sweep_report import find, grouped, refs, stats

SEEDS = range(20000, 20300)
NEW, ALL = (0.5, 1.5, 3, 4), (0, 0.5, 1, 1.5, 2, 3, 4)
LONG = {v: k for k, v in SHORT.items()}
N_HEADS_CHOSEN = 8


def heads(task):
    r = json.loads((exp(task) / "stage2" / "split_half.json").read_text())["results"]
    pos = [a for a, v in r.items() if v["first_half"]["gain"] > 0 and v["second_half"]["gain"] > 0]
    pos.sort(key=lambda a: -r[a]["gain_both_halves"])
    out = []
    for a in pos:
        b, h, _ = a.split(".")[:3]
        if (b, int(h[1:])) not in out:
            out.append((b, int(h[1:])))
    return out[:N_HEADS_CHOSEN]


def run(task):
    sp = []
    for b, h in heads(task):
        for s in NEW:
            a = arm_id(LONG[b], h, s)
            sp.append(make_spec(a, [{"op": "scale_head", "block": LONG[b], "head": h, "scale": s}]))
    pool.run(task, [(x, s, None) for x in sp for s in SEEDS], exp(task) / "stage3" / "dose.jsonl", budget="agent", purpose="dose")


def monotone(v):
    d = [b - a for a, b in zip(v, v[1:])]
    return all(x > 0 for x in d) or all(x < 0 for x in d)


def report(task):
    d = exp(task)
    reach, lat = refs(task)
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
            tab[str(s)] = stats(base, rows, reach, lat)
        num = {k: [tab[str(s)][k] for s in ALL] for k in ("reach_err_median_cm", "cmd_lat8_dev_median_cm_clean", "grasp_fail", "gain")}
        out[f"{b}.h{h}"] = {
            "by_scale": tab,
            "dial": {k: {"x0.5_to_x4": monotone(v[1:]), "x0_to_x4": monotone(v)} for k, v in num.items() if k != "gain"},
        }
        print(f"{b}.h{h}", {k: dict(zip(ALL, v)) for k, v in num.items()}, out[f"{b}.h{h}"]["dial"])
    (d / "stage3" / "dose_response.json").write_text(json.dumps({
        "seeds": [SEEDS[0], SEEDS[-1]], "scales": list(ALL), "tracking_number": "reach_err_median_cm", "rule": __doc__, "heads": out}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a.task)
