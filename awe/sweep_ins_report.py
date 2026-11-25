"""Analysis of the insertion stage 2 sweep. Selection rules are fixed in the awe/atlas_ins.py docstring.

python -m awe.sweep_ins_report first      stage2/sweep_summary.json and recheck_arms.json (top 10 by strict gain, top 5 by overlap)
python -m awe.sweep_ins_report second     stage2/split_half.json after `python -m awe.sweep recheck --task insertion`

The primary score is strict success, benchmark is the guard (stage1/decisions.json). Tracking number: pre_tip_lat (median, cm) over the
episodes that meet, always with the met count beside it. Parts: the grasp failures split by the object that was not lifted (z > 0.06 m).
Reach deviations: distance of the recorded gripper-minus-object vector at the first close from its mean over the design strict successes.
"""
import argparse
import json

import numpy as np

from referee.stats import mcnemar_exact

from .paths import exp
from .sweep_report import find, grouped, rows_of

LIFT_Z = 0.06


def part(r):
    """S, A (class 4a), M (class 3), or the grasp failures by which object was lifted. As in stage1/review1_ins.py."""
    if r["strict_success"]:
        return "S"
    if r["failure_class"] in ("4a", "3"):
        return "A" if r["failure_class"] == "4a" else "M"
    p, s = r["peg_z_max"] > LIFT_Z, r["sock_z_max"] > LIFT_Z
    return {(True, True): "both_lifted", (True, False): "socket_not_lifted", (False, True): "peg_not_lifted", (False, False): "neither_lifted"}[(p, s)]


def refs(task):
    out = {}
    R = rows_of(find(exp(task) / "stage1", "design_default.jsonl"))
    for side in ("r", "l"):
        v = [r["reach_rel_" + side] for r in R if r["strict_success"] and r.get("reach_rel_" + side)]
        out[side] = np.mean(v, 0)
    return out


def cm(v, nd=3):
    return round(100 * float(np.median(v)), nd) if len(v) else None


def stats(base, rows, ref):
    """Paired against base (dict seed -> row)."""
    r = {x["seed"]: x for x in rows}
    seeds = sorted(set(r) & set(base))
    x, y = [r[s] for s in seeds], [base[s] for s in seeds]
    get = lambda rs, k: np.array([bool(q[k]) for q in rs])
    e, b, ge, gb = get(x, "strict_success"), get(y, "strict_success"), get(x, "benchmark_success"), get(y, "benchmark_success")
    f, br = int((e & ~b).sum()), int((b & ~e).sum())
    parts = [part(q) for q in x]
    met = [q for q in x if q.get("pre_tip_lat") is not None]
    M = [q["pre_tip_lat"] for q in met if part(q) == "M"]
    A = [q["strict_run"] for q in x if part(q) == "A"]
    reach = {s: [np.linalg.norm(np.array(q["reach_rel_" + s]) - ref[s]) for q in x if q.get("reach_rel_" + s)] for s in ("r", "l")}
    return {
        "n": len(seeds), "strict": round(float(e.mean()), 4), "gain": round(float(e.mean() - b.mean()), 4), "fixed": f, "broken": br, "p_mcnemar": mcnemar_exact(f, br),
        "benchmark": round(float(ge.mean()), 4), "guard_gain": round(float(ge.mean() - gb.mean()), 4), "benchmark_fixed": int((ge & ~gb).sum()), "benchmark_broken": int((gb & ~ge).sum()),
        "n_met": sum(q["t_meet"] >= 0 for q in x), "n_pin": sum(q["t_pin"] >= 0 for q in x), "parts": {k: parts.count(k) for k in ("S", "A", "M", "peg_not_lifted", "socket_not_lifted", "neither_lifted", "both_lifted")},
        "tip_median_cm": cm([q["pre_tip_lat"] for q in met]), "n_tip": len(met), "tip_median_cm_M": cm(M), "tip_within_0.8cm_share_of_met": round(float(np.mean([q["pre_tip_lat"] <= 0.008 for q in met])), 4) if met else None,
        "strict_run_median_A": float(np.median(A)) if A else None, "reach_dev_median_cm": {s: cm(v) for s, v in reach.items()},
        "identical_traj": sum(r[s]["traj_hash"] == base[s]["traj_hash"] for s in seeds),
    }


def first(task):
    d = exp(task) / "stage2"
    g = grouped(find(d, "sweep.jsonl"))
    ref = refs(task)
    base = {r["seed"]: r for r in g["original"]}
    out = {"original": stats(base, g["original"], ref)}
    for arm in sorted(a for a in g if a != "original"):
        assert len({r["seed"] for r in g[arm]}) == 500, arm
        out[arm] = stats(base, g[arm], ref)
    edits = [a for a in out if a != "original"]
    assert len(edits) == 96, len(edits)
    by_gain = sorted(edits, key=lambda a: (-out[a]["gain"], -out[a]["guard_gain"], a))[:10]
    by_overlap = json.loads((d / "atlas" / "heads.json").read_text())["top5_by_overlap_edits"]
    arms = by_gain + [a for a in by_overlap if a not in by_gain]
    (d / "sweep_summary.json").write_text(json.dumps({"reach_reference": {s: v.tolist() for s, v in ref.items()}, "arms": out}, indent=1))
    (d / "recheck_arms.json").write_text(json.dumps({"arms": arms, "top10_by_gain": by_gain, "top5_by_overlap": by_overlap}, indent=1))
    print("original", out["original"])
    print("recheck:", arms)


def second(task):
    d = exp(task) / "stage2"
    ref = refs(task)
    sel = json.loads((d / "recheck_arms.json").read_text())
    g1, g2 = grouped(find(d, "sweep.jsonl")), grouped(find(d, "sweep_half2.jsonl"))
    b1, b2 = ({r["seed"]: r for r in g["original"]} for g in (g1, g2))
    res, kept = {}, []
    for a in sel["arms"]:
        assert len({r["seed"] for r in g2[a]}) == 500, a
        h1, h2 = stats(b1, g1[a], ref), stats(b2, g2[a], ref)
        same = h1["gain"] * h2["gain"] > 0
        res[a] = {"first_half": h1, "second_half": h2, "same_direction": same, "gain_both_halves": round((h1["gain"] + h2["gain"]) / 2, 4)}
        if same:
            kept.append(a)
    (d / "split_half.json").write_text(json.dumps({"rechecked": sel["arms"], "kept": kept, "top10_by_gain": sel["top10_by_gain"], "top5_by_overlap": sel["top5_by_overlap"],
                                                   "rule": "kept if the strict (primary) gain has the same sign (and is not zero) in both halves", "original_strict": {"first": float(np.mean([r["strict_success"] for r in b1.values()])), "second": float(np.mean([r["strict_success"] for r in b2.values()]))}, "results": res}, indent=1))
    print(len(kept), "of", len(sel["arms"]), "kept:", kept)
    for a in sel["arms"]:
        print(a, res[a]["first_half"]["gain"], res[a]["second_half"]["gain"], res[a]["first_half"]["fixed"], res[a]["first_half"]["broken"], res[a]["second_half"]["fixed"], res[a]["second_half"]["broken"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["first", "second"])
    ap.add_argument("--task", default="insertion")
    a = ap.parse_args()
    (first if a.cmd == "first" else second)(a.task)
