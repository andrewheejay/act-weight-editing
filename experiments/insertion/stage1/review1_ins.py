"""Numbers for the answer to the Gate 1 review of insertion, round 1. Standalone (not in awe/). No rollouts.

python experiments/insertion/stage1/review1_ins.py depth    -> stage1/depth_<run>.jsonl.gz  (per episode, from the trace cache, committed)
python experiments/insertion/stage1/review1_ins.py stats    -> stage1/ins_review1.json       (from committed files only)

depth. Every step from t_meet on. The tip is the peg's leading end (analysis_ins.tip_offset). Socket frame: mouth at |axial| 0.06, pin face at 0.04,
the opening is +-0.018 across, the peg is +-0.010 across, so an aligned peg inside the opening has |tip lat| <= 0.008.
  deep_*     the step with the smallest |tip axial|, no lateral condition (the number the first diagnosis used)
  g08, g12, g18   smallest |tip axial| over the steps with max(|y|,|z|) of the tip below 0.8, 1.2, 1.8 cm
  face       smallest |tip axial| over the steps where all four corners of the leading face are inside the opening (+-1.8 cm, 2 mm slack)
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from analysis_ins import load, load_trace, tip_offset  # noqa: E402
from awe.stage1_analysis import auc  # noqa: E402

S1 = Path(__file__).resolve().parent
RUNS = ["default", "chunk50", "chunk25", "ens", "grid_peg", "grid_sock"]
CENTRE = {"peg_x0": 0.15, "peg_y0": 0.5, "sock_x0": -0.15, "sock_y0": 0.5}
HALF = {"peg_x0": 0.05, "peg_y0": 0.10, "sock_x0": 0.05, "sock_y0": 0.10}
LIFT_Z = 0.06


def depth_row(r, z):
    t = r["t_meet"]
    td = tip_offset(z)
    peg, pR, so, sR = [z[k].astype(float) for k in ("peg", "pegR", "sock", "sockR")]
    ax = pR[:, :, 0]
    e1, e2 = peg + 0.06 * ax, peg - 0.06 * ax
    tip = np.where((np.linalg.norm(e1 - so, axis=1) < np.linalg.norm(e2 - so, axis=1))[:, None], e1, e2)
    corners = np.stack([tip + 0.01 * (a * pR[:, :, 1] + b * pR[:, :, 2]) for a in (-1, 1) for b in (-1, 1)], 1)
    c = np.einsum("tij,tki->tkj", sR, corners - so[:, None])
    face_in = (np.abs(c[:, :, 1:]).max((1, 2)) <= 0.020)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ti,ti->t", ax, sR[:, :, 0])), 0, 1)))
    held = (z["left"] > 0) & (z["right"] > 0) & (z["pegtable"] == 0) & (z["socktable"] == 0)
    a = np.abs(td[:, 0])
    lat = np.abs(td[:, 1:]).max(1)
    T = np.arange(len(a)) >= t
    k = int(np.argmin(np.where(T, a, 9.0)))
    d = {"seed": r.get("seed"), "pose": r.get("pose"), "kind": r["kind"], "t_meet": t, "deep_t": k, "deep_ax": float(a[k]), "deep_lat": float(lat[k]),
         "deep_ang": float(ang[k]), "deep_held": bool(held[k]), "deep_face_in": bool(face_in[k])}
    for nm, sel in (("g08", lat < 0.008), ("g12", lat < 0.012), ("g18", lat < 0.018), ("face", face_in)):
        m = T & sel
        d[nm] = float(a[m].min()) if m.any() else None
        d[nm + "_held"] = float(a[m & held].min()) if (m & held).any() else None
    d["steps_after_meet"] = int(T.sum())
    return d


def longest_run_info(z):
    """The strict condition run (pin, right finger on peg, left finger on socket, no table) and what breaks it, and the pin-only run."""
    from awe.measure import longest_run

    ok = (z["pin"] > 0) & (z["right"] > 0) & (z["left"] > 0) & (z["table"] == 0)
    pin = z["pin"] > 0
    L = longest_run(ok)
    brk = "end_of_episode"
    if L:
        cur, end = 0, None
        for t, v in enumerate(ok):
            cur = cur + 1 if v else 0
            if cur == L and end is None:
                end = t
        if end + 1 < len(ok):
            t = end + 1
            brk = "pin" if not pin[t] else "right" if not z["right"][t] > 0 else "left" if not z["left"][t] > 0 else "table" if z["table"][t] > 0 else "other"
    return {"strict_run": int(L), "pin_run": int(longest_run(pin)), "pin_steps": int(pin.sum()), "break": brk}


def cmd_runs():
    rows = [{"seed": r["seed"], "kind": r["kind"], "t_meet": r["t_meet"], **longest_run_info(load_trace("default", r))} for r in load("default") if r["kind"] in "SAM"]
    with gzip.GzipFile(S1 / "runs_default.jsonl.gz", "wb", mtime=0) as fh:
        for r in rows:
            fh.write((json.dumps(r) + "\n").encode())
    print(len(rows))


def load_runs():
    with gzip.open(S1 / "runs_default.jsonl.gz", "rt") as fh:
        return [json.loads(l) for l in fh]


def issue_4a(F):
    R = load_runs()
    out = {}
    for k in "SA":
        v = np.array([r["strict_run"] for r in R if r["kind"] == k])
        p = np.array([r["pin_run"] for r in R if r["kind"] == k])
        out[k] = {"n": len(v), "strict_run_p10_50_90": [int(x) for x in np.percentile(v, [10, 50, 90])], "strict_run_ge_15": int((v >= 15).sum()), "strict_run_ge_20": int((v >= 20).sum()),
                  "pin_only_run_p10_50_90": [int(x) for x in np.percentile(p, [10, 50, 90])], "pin_only_run_ge_25": int((p >= 25).sum()),
                  "break_reason": {b: sum(r["break"] == b for r in R if r["kind"] == k) for b in ("end_of_episode", "pin", "right", "left", "table", "other")}}
    return out


def M_breakdown():
    """What M does: by deepest step (lateral offset of the tip at the deepest tip position, and whether it passed the mouth plane at 6 cm)."""
    out = {}
    for name in ("default", "chunk50", "grid_peg", "grid_sock"):
        D = [d for d in load_depth(name) if d["kind"] == "M"]
        past = [d for d in D if d["deep_ax"] < 0.055]
        stay = [d for d in D if d["deep_ax"] >= 0.055]
        o = {"n": len(D), "stayed_within_5mm_of_mouth_plane": len(stay), "passed_mouth_plane_5mm": len(past),
             "passed_with_tip_in_opening_lat_below_1.8cm": sum(d["deep_lat"] < 0.018 for d in past),
             "passed_beside_lat_at_least_1.8cm": sum(d["deep_lat"] >= 0.018 for d in past),
             "stayed_tip_lat_at_deepest_below_0.8 / 0.8to1.8 / above_1.8": [sum(d["deep_lat"] < 0.008 for d in stay), sum(0.008 <= d["deep_lat"] < 0.018 for d in stay), sum(d["deep_lat"] >= 0.018 for d in stay)],
             "median_deepest_tip_cm": round(100 * float(np.median([d["deep_ax"] for d in D])), 3), "p10_deepest_tip_cm": round(100 * float(np.percentile([d["deep_ax"] for d in D], 10)), 3)}
        out[name] = o
    return out


def radial_auc(F):
    G = [r for r in F if r["kind"] in "SAM" and r.get("pre_tip_d") is not None]
    k = np.array([r["kind"] for r in G])
    v = np.array([np.hypot(r["pre_tip_d"][1], r["pre_tip_d"][2]) for r in G])
    f = lambda pos, neg: round(float(auc(v[np.isin(k, pos + neg)], np.isin(k[np.isin(k, pos + neg)], pos))), 4)
    return {"M_vs_S": f(["M"], ["S"]), "M_vs_SA": f(["M"], ["S", "A"]), "A_vs_S": f(["A"], ["S"])}


def near_miss_shares(F):
    """Share of M with the tip offset before contact under multiples of the 0.8 cm limit."""
    M = np.array([r["pre_tip_lat"] for r in F if r["kind"] == "M"])
    return {f"M_below_{m}x_limit_({m*0.8:.1f}cm)": round(float((M < 0.008 * m).mean()), 4) for m in (1, 1.5, 2, 3)}


def cmd_depth():
    for name in RUNS:
        rows = [depth_row(r, load_trace(name, r)) for r in load(name) if r.get("pre_tip_lat") is not None]
        rows.sort(key=lambda r: (r["seed"] is None, r["seed"] or 0, str(r["pose"])))
        with gzip.GzipFile(S1 / f"depth_{name}.jsonl.gz", "wb", mtime=0) as fh:
            for r in rows:
                fh.write((json.dumps(r) + "\n").encode())
        print(name, len(rows))


def load_depth(name):
    with gzip.open(S1 / f"depth_{name}.jsonl.gz", "rt") as fh:
        return [json.loads(l) for l in fh]


def rank(a):
    a = np.asarray(a, float)
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt)
    return (cs - (cnt - 1) / 2.0)[inv]          # average ranks, ties handled


def spear(a, b):
    return round(float(np.corrcoef(rank(a), rank(b))[0, 1]), 4)


def qs(v):
    return [round(100 * float(x), 3) for x in np.percentile(v, [10, 50, 90])] if len(v) else None


# ---------------------------------------------------------------- issue 1
def issue1(name):
    D = load_depth(name)
    out = {"n": {c: sum(d["kind"] == c for d in D) for c in "SAM"}}
    for c in "SAM":
        R = [d for d in D if d["kind"] == c]
        if not R:
            continue
        n = len(R)
        o = {"deepest_ungated_cm_p10_50_90": qs([d["deep_ax"] for d in R])}
        for lab, thr in (("5mm", 0.055), ("15mm", 0.045)):
            o[f"ungated_past_{lab}"] = sum(d["deep_ax"] < thr for d in R)
            for g in ("g08", "g12", "g18", "face"):
                o[f"{g}_past_{lab}"] = sum(d[g] is not None and d[g] < thr for d in R)
            o[f"at_deepest_step_lat_below_1.8cm_past_{lab}"] = sum(d["deep_ax"] < thr and d["deep_lat"] < 0.018 for d in R)
            o[f"at_deepest_step_lat_above_1.8cm_past_{lab}"] = sum(d["deep_ax"] < thr and d["deep_lat"] >= 0.018 for d in R)
        o["impossible_below_3.8cm_ungated"] = sum(d["deep_ax"] < 0.038 for d in R)
        o["impossible_below_3.8cm_and_outside_1.8"] = sum(d["deep_ax"] < 0.038 and d["deep_lat"] >= 0.018 for d in R)
        o["face_ever_in_opening_past_the_mouth"] = sum(d["face"] is not None and d["face"] < 0.060 for d in R)
        o["n"] = n
        out[c] = o
    return out


# ---------------------------------------------------------------- issue 2
def dev(F, side):
    key = "reach_rel_" + side
    mu = np.array([r[key] for r in F if r["kind"] == "S" and r.get(key) is not None]).mean(0)
    return {id(r): float(np.linalg.norm(np.array(r[key]) - mu)) for r in F if r.get(key) is not None}


def scaled_dist(r, obj):
    return float(np.hypot((r[obj + "_x0"] - CENTRE[obj + "_x0"]) / HALF[obj + "_x0"], (r[obj + "_y0"] - CENTRE[obj + "_y0"]) / HALF[obj + "_y0"]))


def lifted(r):
    return r["peg_z_max"] > LIFT_Z, r["sock_z_max"] > LIFT_Z


def part_of(r):
    if r["kind"] in "SAM":
        return r["kind"]
    p, s = lifted(r)
    return {(True, True): "both_lifted", (True, False): "socket_not_lifted", (False, True): "peg_not_lifted", (False, False): "neither_lifted"}[(p, s)]


def nn_cluster(F, flag):
    P = np.array([[(r[c] - CENTRE[c]) / HALF[c] for c in CENTRE] for r in F])
    D = ((P[:, None] - P[None]) ** 2).sum(-1)
    np.fill_diagonal(D, 9e9)
    nn = D.argmin(1)
    y = np.array(flag)
    base = float(y.mean())
    return {"base_rate": round(base, 4), "nn_same_given_in": round(float(y[nn][y].mean()), 4), "ratio_to_base": round(float(y[nn][y].mean()) / base, 2), "n": int(y.sum())}


def issue2(F):
    out = {}
    for c in ("0", "1", "2"):
        R = [r for r in F if r["failure_class"] == c]
        out[f"class_{c}"] = {p: sum(part_of(r) == p for r in R) for p in ("peg_not_lifted", "socket_not_lifted", "neither_lifted", "both_lifted")}
    out["lift_cut_sensitivity_407"] = {str(cut): {"peg_not_lifted_only": sum(r["peg_z_max"] <= cut < r["sock_z_max"] for r in F if r["kind"] in ("G0", "G1", "G2")),
                                                  "socket_not_lifted_only": sum(r["sock_z_max"] <= cut < r["peg_z_max"] for r in F if r["kind"] in ("G0", "G1", "G2")),
                                                  "neither": sum(r["sock_z_max"] <= cut and r["peg_z_max"] <= cut for r in F if r["kind"] in ("G0", "G1", "G2"))} for cut in (0.055, 0.06, 0.07)}
    er, el = dev(F, "r"), dev(F, "l")
    ref = [r for r in F if r["kind"] in "SAM"]      # episodes that met
    parts = {"right_arm_fails (peg_not_lifted)": ("peg_not_lifted", er, "peg"), "left_arm_fails (socket_not_lifted)": ("socket_not_lifted", el, "sock")}
    for lab, (p, e, obj) in parts.items():
        R = [r for r in F if part_of(r) == p]
        vs = [e[id(r)] for r in R if id(r) in e]
        rs = [e[id(r)] for r in ref if id(r) in e]
        x = np.r_[vs, rs]
        y = np.r_[np.ones(len(vs), bool), np.zeros(len(rs), bool)]
        Sset = [e[id(r)] for r in F if r["kind"] == "S" and id(r) in e]
        ds = [scaled_dist(r, obj) for r in R]
        o = {"n": len(R), "n_with_reach": len(vs), "median_reach_dev_cm": round(100 * float(np.median(vs)), 2), "median_reach_dev_cm_met": round(100 * float(np.median(rs)), 2),
             "median_reach_dev_cm_S": round(100 * float(np.median(Sset)), 2), "auc_vs_met": round(float(auc(x, y)), 4),
             "auc_vs_S": round(float(auc(np.r_[vs, Sset], np.r_[np.ones(len(vs), bool), np.zeros(len(Sset), bool)])), 4),
             "median_distance_from_centre_scaled": round(float(np.median(ds)), 3), "median_distance_all": round(float(np.median([scaled_dist(r, obj) for r in F])), 3),
             "cluster": nn_cluster(F, [part_of(r) == p for r in F])}
        own = [part_of(r) == p for r in F]
        for c in ("_x0", "_y0"):
            v = np.array([r[obj + c] for r in F])
            ed = np.quantile(v, [0, .2, .4, .6, .8, 1])
            b = np.clip(np.searchsorted(ed, v, side="right") - 1, 0, 4)
            o[f"rate_by_fifth_of_{obj}{c}"] = [round(float(np.array(own)[b == i].mean()), 3) for i in range(5)]
        o["reach_dev_spearman_with_distance_in_part"] = spear([e[id(r)] for r in R if id(r) in e], [scaled_dist(r, obj) for r in R if id(r) in e]) if len(vs) > 4 else None
        out[lab] = o
    return out


def issue2_M(F):
    """The same breakdown for M."""
    D = {d["seed"]: d for d in load_depth("default")}
    M = [r for r in F if r["kind"] == "M"]
    p90S = float(np.percentile([r["pre_tip_lat"] for r in F if r["kind"] == "S"], 90))
    o = {"n": len(M), "meet_at_or_after_350": sum(r["t_meet"] >= 350 for r in M), "deepest_ungated_below_5.5cm": sum(D[r["seed"]]["deep_ax"] < 0.055 for r in M),
         "pass_outside_1.8cm_at_deepest (deep_ax<5.5, lat>=1.8)": sum(D[r["seed"]]["deep_ax"] < 0.055 and D[r["seed"]]["deep_lat"] >= 0.018 for r in M),
         "tip_lat_below_0.8cm_pre_contact": sum(r["pre_tip_lat"] < 0.008 for r in M), "tip_lat_above_S_p90": sum(r["pre_tip_lat"] > p90S for r in M)}
    late = [r for r in M if r["t_meet"] >= 350]
    early = [r for r in M if r["t_meet"] < 350]
    o["late_vs_early_median_tip_lat_cm"] = [round(100 * float(np.median([r["pre_tip_lat"] for r in late])), 2), round(100 * float(np.median([r["pre_tip_lat"] for r in early])), 2)]
    o["cluster"] = nn_cluster(F, [r["kind"] == "M" for r in F])
    own = np.array([r["kind"] == "M" for r in F])
    for obj in ("peg", "sock"):
        for c in ("_x0", "_y0"):
            v = np.array([r[obj + c] for r in F])
            ed = np.quantile(v, [0, .2, .4, .6, .8, 1])
            b = np.clip(np.searchsorted(ed, v, side="right") - 1, 0, 4)
            o[f"M_rate_by_fifth_of_{obj}{c}"] = [round(float(own[b == i].mean()), 3) for i in range(5)]
    return o


def knn_pred(P, Y, Q_idx, k):
    D = ((P[Q_idx][:, None] - P[None]) ** 2).sum(-1)
    D[np.arange(len(Q_idx)), Q_idx] = 9e9
    nn = np.argsort(D, 1)[:, :k]
    return Y[nn].mean(1)


def funnel(F, G):
    """What fixing each stage could give. Rates conditional on reaching the stage, and a local (nearest met neighbour) estimate for the rescued seeds."""
    n = len(F)
    met = [r for r in F if r["kind"] in "SAM"]
    pin = [r for r in F if r["kind"] in "SA"]
    S = [r for r in F if r["kind"] == "S"]
    o = {"n": n, "met": len(met), "pin": len(pin), "strict": len(S), "P_met": round(len(met) / n, 4), "P_pin_given_met": round(len(pin) / len(met), 4), "P_strict_given_pin": round(len(S) / len(pin), 4)}
    # strict rate if a stage succeeded always, with the later conditional rates as observed
    pm, pp, ps = len(met) / n, len(pin) / len(met), len(S) / len(pin)
    o["strict_if_every_episode_met"] = round(pp * ps, 4)
    o["strict_if_every_met_reached_the_pin"] = round(pm * ps, 4)
    o["strict_if_every_pin_contact_became_strict"] = round(pm * pp, 4)
    # local: for every grasp-class seed, the strict rate among its k nearest seeds that met (4-D, box-scaled)
    P = np.array([[(r[c] - CENTRE[c]) / HALF[c] for c in CENTRE] for r in F])
    g = [i for i, r in enumerate(F) if r["kind"] in ("G0", "G1", "G2")]
    metidx = np.array([i for i, r in enumerate(F) if r["kind"] in "SAM"])
    sv = np.array([r["kind"] == "S" for r in F], float)
    pv = np.array([r["kind"] in "SA" for r in F], float)
    for k in (10, 20):
        D = ((P[g][:, None] - P[metidx][None]) ** 2).sum(-1)
        nn = metidx[np.argsort(D, 1)[:, :k]]
        o[f"local_k{k}_for_407_grasp_seeds"] = {"mean_strict_rate_of_nearest_met": round(float(sv[nn].mean()), 4), "mean_pin_rate_of_nearest_met": round(float(pv[nn].mean()), 4)}
    o["note"] = "conditional rates assume a rescued episode behaves like the episodes that reached the stage; this is not tested"
    return o


# ---------------------------------------------------------------- issue 3
def grid_local(name, obj):
    """Signed tip offset on a 41x41 grid of met episodes: spatial variation between neighbours against the total variation."""
    F = [r for r in load(name) if r.get("pre_tip_d") is not None]
    xs = np.array([r[obj + "_x0"] for r in F])
    ys = np.array([r[obj + "_y0"] for r in F])
    gx = np.round((xs - CENTRE[obj + "_x0"] + 0.05) / 0.0025).astype(int)
    gy = np.round((ys - 0.4) / 0.005).astype(int)
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(gx, gy))}
    out = {"n_met": len(F)}
    for axn, i in (("y", 1), ("z", 2)):
        v = np.array([r["pre_tip_d"][i] for r in F])
        tot = v.var()
        res = {}
        for lag in (1, 2, 4, 8):
            for lab, (dx, dy) in (("x", (lag, 0)), ("y", (0, lag))):
                pr = [(v[j], v[idx[(gx[j] + dx, gy[j] + dy)]]) for j in range(len(F)) if (gx[j] + dx, gy[j] + dy) in idx]
                a = np.array(pr)
                if len(a) < 20:
                    continue
                res[f"lag{lag}_{lab}"] = {"n_pairs": len(a), "corr": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3),
                                          "half_mean_sq_diff_over_variance": round(float(0.5 * ((a[:, 0] - a[:, 1]) ** 2).mean() / tot), 3)}
        out[f"signed_{axn}"] = {"sd_cm": round(100 * float(np.sqrt(tot)), 3), "pairs": res}
        # LOO k-nearest-neighbour R2 over the met grid points, in scaled grid units
        Q = np.array([[(xs[j] - CENTRE[obj + "_x0"]) / HALF[obj + "_x0"], (ys[j] - CENTRE[obj + "_y0"]) / HALF[obj + "_y0"]] for j in range(len(F))])
        for k in (4, 8, 16):
            p = knn_pred(Q, v, np.arange(len(F)), k)
            out[f"signed_{axn}"][f"loo_knn_R2_k{k}"] = round(float(1 - ((v - p) ** 2).sum() / ((v - v.mean()) ** 2).sum()), 3)
    return out


def design_local(F):
    """Same question on the design seeds, 4-D. LOO nearest-neighbour R2 (box-scaled) of the signed tip offset, over the met episodes."""
    G = [r for r in F if r.get("pre_tip_d") is not None]
    P = np.array([[(r[c] - CENTRE[c]) / HALF[c] for c in CENTRE] for r in G])
    out = {"n_met": len(G), "median_nn_distance_scaled": None}
    D = ((P[:, None] - P[None]) ** 2).sum(-1)
    np.fill_diagonal(D, 9e9)
    out["median_nn_distance_scaled"] = round(float(np.median(np.sqrt(D.min(1)))), 3)
    for axn, i in (("y", 1), ("z", 2)):
        v = np.array([r["pre_tip_d"][i] for r in G])
        for k in (1, 2, 4, 8, 16, 32):
            p = knn_pred(P, v, np.arange(len(G)), k)
            out[f"signed_{axn}_loo_knn_R2_k{k}"] = round(float(1 - ((v - p) ** 2).sum() / ((v - v.mean()) ** 2).sum()), 3)
    vv = np.array([r["pre_tip_lat"] for r in G])
    for k in (4, 8, 16, 32):
        p = knn_pred(P, vv, np.arange(len(G)), k)
        out[f"abs_tip_lat_loo_knn_R2_k{k}"] = round(float(1 - ((vv - p) ** 2).sum() / ((vv - vv.mean()) ** 2).sum()), 3)
    return out


def four_coord_positions(F):
    """Spearman of the tip offset with each start coordinate and peg y reported (met episodes)."""
    G = [r for r in F if r.get("pre_tip_lat") is not None]
    v = np.array([r["pre_tip_lat"] for r in G])
    out = {"n": len(G)}
    for c in CENTRE:
        x = np.array([r[c] for r in G]) - CENTRE[c]
        if np.ptp(x) < 1e-6:      # held fixed on this grid
            continue
        ed = np.quantile(x, [0, .2, .4, .6, .8, 1])
        b = np.clip(np.searchsorted(ed, x, side="right") - 1, 0, 4)
        out[c] = {"spearman": spear(v, x), "median_tip_lat_cm_by_fifth": [round(100 * float(np.median(v[b == i])), 2) for i in range(5)]}
    return out


def cmd_stats():
    F = load("default")
    out = {"issue1_depth": {n: issue1(n) for n in RUNS}}
    out["issue2_grasp_parts"] = issue2(F)
    out["issue2_M_parts"] = issue2_M(F)
    out["issue1_M_breakdown"] = M_breakdown()
    out["class4a_hold"] = issue_4a(F)
    out["radial_tip_offset_auc_design"] = radial_auc(F)
    out["near_miss_shares_design"] = near_miss_shares(F)
    out["issue2_funnel"] = funnel(F, None)
    out["issue3_design_local"] = design_local(F)
    out["issue3_grid_peg_local"] = grid_local("grid_peg", "peg")
    out["issue3_grid_sock_local"] = grid_local("grid_sock", "sock")
    out["issue3_positions"] = {n: four_coord_positions(load(n)) for n in ("default", "grid_peg", "grid_sock")}
    (S1 / "ins_review1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"depth": cmd_depth, "runs": cmd_runs, "stats": cmd_stats}[sys.argv[1]]()
