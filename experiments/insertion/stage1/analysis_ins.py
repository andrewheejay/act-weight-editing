"""Stage 1 analysis for insertion. Standalone (not in awe/, so it does not trigger a Gate 0 replay).

python experiments/insertion/stage1/analysis_ins.py features default|chunk50|grid_peg|grid_sock
python experiments/insertion/stage1/analysis_ins.py stats default|...      stage1/ins_stats_<name>.json
    stage1/features_<name>.jsonl.gz from the records and the per-step traces (cache, not in git, rebuilt by re-running)

Definitions (DEVIATIONS.md, 2026-11-25, and referee/RULES.md for the score):
  kind        S  strict success
              A  class 4a, pin contact at some step but never strict
              M  class 3, the peg touched the socket off the table but never the pin
              G0 class 0, G1 class 1, G2 class 2: no peg-socket contact off the table
  pre_*       the peg in the socket frame one control step before the first peg-socket contact off the table (t_meet - 1).
              Nothing has touched the socket yet, so it is free of contact forces from the insertion itself (the grasps are there).
              axial = |offset along the socket axis|, lat = max |offset across it|, ang = angle between the two axes (degrees),
              d = signed offset in the socket frame (3 numbers).
  reach_rel_r  right gripper centre minus peg centre at the first right close; reach_rel_l  the same for the left and the socket.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from awe import measure  # noqa: E402
from awe.paths import cache_dir, exp  # noqa: E402
from awe.pool import trace_name  # noqa: E402

T = exp("insertion")
S1 = T / "stage1"
FILES = {"default": "design_default.jsonl", "chunk50": "design_chunk50.jsonl", "chunk25": "design_chunk25.jsonl", "ens": "design_ens.jsonl",
         "grid_peg": "grid_peg_default.jsonl", "grid_sock": "grid_sock_default.jsonl"}
TRACE = {"default": "default", "chunk50": "chunk50", "chunk25": "chunk25", "ens": "ens", "grid_peg": "grid_peg", "grid_sock": "grid_sock"}


def records(name):
    f = S1 / FILES[name]
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def load_trace(name, rec):
    return np.load(cache_dir("insertion") / "traces" / TRACE[name] / trace_name(rec.get("seed"), rec.get("pose")))


def kind(r):
    if r["strict_success"]:
        return "S"
    return {"4a": "A", "3": "M", "2": "G2", "1": "G1", "0": "G0"}[r["failure_class"]]


def feature_row(rec, z):
    d = dict(rec)
    d["kind"] = kind(rec)
    for k in ("peg_x0", "peg_y0", "sock_x0", "sock_y0"):
        d[k] = float(rec.get(k, np.nan))
    t = rec["t_meet"]
    if t is not None and t >= 1:
        ax, lat, ang, off = measure.peg_in_socket(z["peg"].astype(float), z["pegR"].astype(float), z["sock"].astype(float), z["sockR"].astype(float))
        k = t - 1
        d["pre_ax"], d["pre_lat"], d["pre_ang"] = float(ax[k]), float(lat[k]), float(ang[k])
        d["pre_d"] = [float(x) for x in off[k]]
        # same 20 steps earlier: is the approach closing in
        j = max(0, k - 20)
        d["pre20_lat"], d["pre20_ang"], d["pre20_ax"] = float(lat[j]), float(ang[j]), float(ax[j])
    return d


def cmd_features(name):
    R = records(name)
    rows = []
    for r in R:
        rows.append(feature_row(r, load_trace(name, r)))
    rows.sort(key=lambda r: (r.get("seed") is None, r.get("seed") or 0, str(r.get("pose"))))
    out = S1 / f"features_{name}.jsonl.gz"
    with gzip.GzipFile(out, "wb", mtime=0) as fh:
        for r in rows:
            fh.write((json.dumps(r) + "\n").encode())
    print(name, len(rows), "->", out)


def load(name):
    with gzip.open(S1 / f"features_{name}.jsonl.gz", "rt") as fh:
        return [json.loads(l) for l in fh]


KS = ["S", "A", "M", "G2", "G1", "G0"]


def col(F, k, kinds=None):
    return np.array([np.nan if r.get(k) is None else r[k] for r in F if kinds is None or r["kind"] in kinds], float)


def auc_by(F, key, pos, neg, fn=lambda r, k: r.get(k)):
    R = [r for r in F if r["kind"] in pos + neg and fn(r, key) is not None]
    x = np.array([fn(r, key) for r in R], float)
    y = np.array([r["kind"] in pos for r in R])
    return round(auc(x, y), 4)


def cmd_stats(name):
    F = load(name)
    n = len(F)
    out = {"name": name, "n": n, "benchmark": sum(r["benchmark_success"] for r in F) / n, "strict": sum(r["strict_success"] for r in F) / n}
    out["gap_points"] = round(100 * (out["benchmark"] - out["strict"]), 2)
    out["kinds"] = {k: sum(r["kind"] == k for r in F) for k in KS}
    out["classes"] = {c: sum(r["failure_class"] == c for r in F) for c in ("0", "1", "2", "3", "4a", "4b")}
    out["by_kind_median"] = {k: {c: (None if not np.isfinite(v := np.nanmedian(col(F, k, [c]))) else round(float(v), 4)) for c in KS} for k in
                             ("pre_lat", "pre_ang", "pre_ax", "meet_lat", "meet_ang", "t_grasp", "t_meet", "held_steps", "chunk_jump")}
    sgn = lambda i: (lambda r, k: None if "pre_d" not in r else r["pre_d"][i])
    out["auc"] = {
        "pre_lat_M_vs_S": auc_by(F, "pre_lat", ["M"], ["S"]), "pre_lat_M_vs_SA": auc_by(F, "pre_lat", ["M"], ["S", "A"]),
        "pre_ang_M_vs_S": auc_by(F, "pre_ang", ["M"], ["S"]),
        "pre_lat_A_vs_S": auc_by(F, "pre_lat", ["A"], ["S"]),
        "abs_dy_M_vs_SA": auc_by(F, None, ["M"], ["S", "A"], lambda r, k: None if "pre_d" not in r else abs(r["pre_d"][1])),
        "abs_dz_M_vs_SA": auc_by(F, None, ["M"], ["S", "A"], lambda r, k: None if "pre_d" not in r else abs(r["pre_d"][2])),
        "signed_dy_M_vs_SA": auc_by(F, None, ["M"], ["S", "A"], sgn(1)), "signed_dz_M_vs_SA": auc_by(F, None, ["M"], ["S", "A"], sgn(2))}
    ref = {}
    for side in ("r", "l"):
        key = "reach_rel_" + side
        S = np.array([r[key] for r in F if r["kind"] == "S" and r.get(key) is not None])
        mu = S.mean(0)
        ref[side] = {"mean_over_strict_m": [round(float(x), 5) for x in mu], "n": len(S)}
        dev = {c: np.array([np.linalg.norm(np.array(r[key]) - mu) for r in F if r["kind"] == c and r.get(key) is not None]) for c in KS}
        out[f"reach_{side}_dev_median_cm"] = {c: round(100 * float(np.median(v)), 3) for c, v in dev.items() if len(v)}
        x = np.concatenate([dev[c] for c in KS])
        y = np.concatenate([np.full(len(dev[c]), c in ("G0", "G1", "G2")) for c in KS])
        out[f"reach_{side}_auc_G_vs_rest"] = round(auc(x, y), 4)
    out["reach_reference"] = ref
    (S1 / f"ins_stats_{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    from awe.stage1_analysis import auc  # noqa: E402

    if sys.argv[1] == "features":
        cmd_features(sys.argv[2])
    elif sys.argv[1] == "stats":
        cmd_stats(sys.argv[2])
