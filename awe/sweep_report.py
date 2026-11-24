"""Analysis of the stage 2 sweep. Selection rules are fixed in the awe/atlas.py docstring.

python -m awe.sweep_report first      stage2/sweep_summary.json and recheck_arms.json (top 10 by gain, top 5 by overlap)
python -m awe.sweep_report second     stage2/split_half.json after `python -m awe.sweep recheck`

Tracking numbers per arm: reach error (SPEC tracking number) and cmd_lat8 (contact-free companion), both against the mean of the
original's successful design episodes (stage 1). Medians over all episodes that have the number; the lat8 one over clean episodes.
"""
import argparse
import gzip
import json

import numpy as np

from referee.stats import mcnemar_exact

from .paths import exp


def rows_of(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def find(d, name):
    return d / (name + ".gz") if (d / (name + ".gz")).exists() else d / name


def refs(task):
    """Reference vectors from the original's successful design episodes (stage 1)."""
    from .stage1_analysis import load_features

    F = load_features(task, "default")
    ok = [r for r in F if r["benchmark_success"] and r.get("reach_rel")]
    reach = np.mean([r["reach_rel"] for r in ok], 0)
    c = [r["cmd_lat_h"]["0.08"] for r in ok if r.get("cmd_clean") and r["cmd_lat_h"]["0.08"] is not None]
    return reach, float(np.mean(c))


def stats(base, rows, reach_ref, lat_ref):
    """Paired against base (dict seed -> row)."""
    r = {x["seed"]: x for x in rows}
    seeds = sorted(set(r) & set(base))
    b = np.array([base[s]["benchmark_success"] for s in seeds])
    e = np.array([r[s]["benchmark_success"] for s in seeds])
    sb = np.array([base[s]["strict_success"] for s in seeds])
    se = np.array([r[s]["strict_success"] for s in seeds])
    fixed, broken = int((e & ~b).sum()), int((b & ~e).sum())
    x = [r[s] for s in seeds]
    re = [np.linalg.norm(np.array(q["reach_rel"]) - reach_ref) for q in x if q.get("reach_rel")]
    lat = [abs(q["cmd_lat8"] - lat_ref) for q in x if q.get("cmd_clean") and q["cmd_lat8"] is not None]
    return {
        "n": len(seeds), "success": round(float(e.mean()), 4), "gain": round(float(e.mean() - b.mean()), 4), "strict_gain": round(float(se.mean() - sb.mean()), 4),
        "fixed": fixed, "broken": broken, "p_mcnemar": mcnemar_exact(fixed, broken), "grasp_fail": sum(q["grasp_fail"] for q in x) if all("grasp_fail" in q for q in x) else None,
        "reach_err_median_cm": round(100 * float(np.median(re)), 3) if re else None,
        "cmd_lat8_dev_median_cm_clean": round(100 * float(np.median(lat)), 3) if lat else None, "n_clean": len(lat),
        "identical_traj": sum(r[s]["traj_hash"] == base[s]["traj_hash"] for s in seeds),
    }


def grouped(path):
    g = {}
    for r in rows_of(path):
        g.setdefault(r["arm"], []).append(r)
    return g


def first(task):
    d = exp(task) / "stage2"
    g = grouped(find(d, "sweep.jsonl"))
    reach, lat = refs(task)
    base = {r["seed"]: r for r in g["original"]}
    out = {"original": stats(base, g["original"], reach, lat)}
    for arm in sorted(a for a in g if a != "original"):
        assert len({r["seed"] for r in g[arm]}) == 500, arm
        out[arm] = stats(base, g[arm], reach, lat)
    edits = [a for a in out if a != "original"]
    by_gain = sorted(edits, key=lambda a: (-out[a]["gain"], -out[a]["strict_gain"], a))[:10]
    by_overlap = json.loads((d / "atlas" / "heads.json").read_text())["top5_by_overlap_edits"]
    arms = by_gain + [a for a in by_overlap if a not in by_gain]
    (d / "sweep_summary.json").write_text(json.dumps({"reference": {"reach_rel_mean": reach.tolist(), "cmd_lat8_mean": lat}, "arms": out}, indent=1))
    (d / "recheck_arms.json").write_text(json.dumps({"arms": arms, "top10_by_gain": by_gain, "top5_by_overlap": by_overlap}, indent=1))
    print("original", out["original"]["success"], "recheck:", arms)
    for a in sorted(edits, key=lambda a: -out[a]["gain"]):
        print(a, out[a])


def second(task):
    d = exp(task) / "stage2"
    reach, lat = refs(task)
    sel = json.loads((d / "recheck_arms.json").read_text())
    g1, g2 = grouped(find(d, "sweep.jsonl")), grouped(find(d, "sweep_half2.jsonl"))
    b1, b2 = ({r["seed"]: r for r in g["original"]} for g in (g1, g2))
    res, kept = {}, []
    for a in sel["arms"]:
        h1, h2 = stats(b1, g1[a], reach, lat), stats(b2, g2[a], reach, lat)
        both = h1["gain"] * h2["gain"] > 0
        res[a] = {"first_half": h1, "second_half": h2, "same_direction": both, "gain_both_halves": round((h1["gain"] + h2["gain"]) / 2, 4)}
        if both:
            kept.append(a)
    (d / "split_half.json").write_text(json.dumps({"rechecked": sel["arms"], "kept": kept, "top10_by_gain": sel["top10_by_gain"], "top5_by_overlap": sel["top5_by_overlap"],
                                                   "rule": "kept if the benchmark gain has the same sign (and is not zero) in both halves", "original_success": {"first": float(np.mean([r["benchmark_success"] for r in b1.values()])), "second": float(np.mean([r["benchmark_success"] for r in b2.values()]))}, "results": res}, indent=1))
    print(len(kept), "of", len(sel["arms"]), "kept:", kept)
    for a in sel["arms"]:
        print(a, res[a]["first_half"]["gain"], res[a]["second_half"]["gain"], res[a]["first_half"]["fixed"], res[a]["first_half"]["broken"], res[a]["second_half"]["fixed"], res[a]["second_half"]["broken"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["first", "second"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    (first if a.cmd == "first" else second)(a.task)
