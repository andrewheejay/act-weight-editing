"""Stage 1 numbers for the insertion diagnosis. Standalone (not in awe/). Reads the committed features and, for the time course only, the trace cache.

python experiments/insertion/stage1/report_ins.py stats      -> stage1/ins_pattern.json
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
from awe.stage1_report import wilson  # noqa: E402

S1 = Path(__file__).resolve().parent
RUNS = ["default", "chunk50", "chunk25", "ens", "grid_peg", "grid_sock"]
TOL = 0.008   # m, half-gap 0.018 minus peg half-width 0.010 (awe/measure.py)
CENTRE = {"peg_x0": 0.15, "peg_y0": 0.5, "sock_x0": -0.15, "sock_y0": 0.5}


def met(F):
    return [r for r in F if r["kind"] in "SAM" and r.get("pre_tip_lat") is not None]


def A(v, kinds, pos, neg):
    m = np.isin(kinds, pos + neg)
    return None if len(set(kinds[m])) < 2 else round(float(auc(v[m], np.isin(kinds[m], pos))), 4)


def pct(v):
    return [round(100 * float(x), 3) for x in np.percentile(v, [10, 50, 90])] if len(v) else None


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def tracking(F):
    G = met(F)
    k = np.array([r["kind"] for r in G])
    out = {"n": {c: int((k == c).sum()) for c in "SAM"}}
    for key in ("pre_tip_lat", "pre_lat", "pre_ang", "pre_tip_lat_20", "post_min_tip_ax"):
        v = np.array([r[key] for r in G], float)
        out[key] = {"M_vs_S": A(v, k, ["M"], ["S"]), "M_vs_SA": A(v, k, ["M"], ["S", "A"]), "A_vs_S": A(v, k, ["A"], ["S"]), "AM_vs_S": A(v, k, ["A", "M"], ["S"])}
    v = np.array([r["pre_tip_lat"] for r in G])
    out["tip_lat_cm_p10_50_90"] = {c: pct(v[k == c]) for c in "SAM"}
    out["share_within_tolerance_0.8cm"] = {c: round(float((v[k == c] < TOL).mean()), 4) for c in "SAM" if (k == c).any()}
    p90 = float(np.percentile(v[k == "S"], 90))
    out["S_p90_cm"] = round(100 * p90, 3)
    out["share_above_S_p90"] = {c: round(float((v[k == c] > p90).mean()), 4) for c in "AM"}
    d = np.array([r["pre_tip_d"] for r in G])
    out["signed_tip_cm_mean_sd"] = {c: {ax: [round(100 * float(d[k == c, i].mean()), 3), round(100 * float(d[k == c, i].std()), 3)] for ax, i in (("y", 1), ("z", 2))} for c in "SAM"}
    return out


def boot_auc(F, n=2000):
    G = met(F)
    k = np.array([r["kind"] for r in G]); v = np.array([r["pre_tip_lat"] for r in G])
    s, m = v[k == "S"], v[k == "M"]
    rng = np.random.default_rng(0)
    bs = [auc(np.r_[rng.choice(s, len(s)), rng.choice(m, len(m))], np.r_[np.zeros(len(s), bool), np.ones(len(m), bool)]) for _ in range(n)]
    return [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]


def timing(F):
    G = met(F)
    k = np.array([r["kind"] for r in G]); t = np.array([r["t_meet"] for r in G]); v = np.array([r["pre_tip_lat"] for r in G])
    out = {"t_meet_p10_50_90": {c: [int(x) for x in np.percentile(t[k == c], [10, 50, 90])] for c in "SAM"}}
    out["first_contact_recorded_before_t_meet"] = int(sum(0 <= r["first_pegsock"] < r["t_meet"] for r in G))
    for nm, sel in (("meet_before_step_300", t < 300), ("meet_from_step_300", t >= 300)):
        out[nm] = {"n": {c: int((sel & (k == c)).sum()) for c in "SAM"}, "median_tip_lat_cm": {c: pct(v[sel & (k == c)])[1] for c in "SAM" if (sel & (k == c)).any()},
                   "auc_M_vs_S": A(v[sel], k[sel], ["M"], ["S"])}
    out["meet_in_last_50_steps"] = {c: round(float((t[k == c] >= 350).mean()), 4) for c in "SAM"}
    return out


def depth(F):
    G = met(F)
    k = np.array([r["kind"] for r in G]); a = np.array([r["post_min_tip_ax"] for r in G])
    # the mouth is at 0.06 and the pin is reached at 0.04 (tip coordinates)
    return {"deepest_tip_cm_p10_50_90": {c: pct(a[k == c]) for c in "SAM"}, "share_entered_5mm": {c: round(float((a[k == c] < 0.055).mean()), 4) for c in "SAM"},
            "share_entered_15mm": {c: round(float((a[k == c] < 0.045).mean()), 4) for c in "SAM"}}


def time_course(F):
    ks = (60, 40, 20, 10, 5, 1)
    rows = {c: {q: [] for q in ks} for c in "SAM"}
    for r in met(F):
        if r["t_meet"] < 61:
            continue
        d = tip_offset(load_trace("default", r))
        for q in ks:
            rows[r["kind"]][q].append(float(np.abs(d[r["t_meet"] - q][1:]).max()))
    return {c: {f"t_meet-{q}": pct(v)[1] for q, v in rows[c].items()} for c in "SAM"}


def positions(F):
    G = met(F)
    k = np.array([r["kind"] for r in G]); v = np.array([r["pre_tip_lat"] for r in G]); M = k == "M"
    out = {"n_met": len(G)}
    for key, c in CENTRE.items():
        x = np.array([r[key] for r in G]) - c
        if np.ptp(x) < 1e-6:
            continue
        ed = np.quantile(x, [0, .2, .4, .6, .8, 1]); b = np.clip(np.searchsorted(ed, x, side="right") - 1, 0, 4)
        out[key] = {"spearman_tip_lat": spear(v, x), "spearman_abs_from_centre": spear(v, np.abs(x)),
                    "M_share_by_fifth": [round(float(M[b == i].mean()), 3) for i in range(5)], "median_tip_lat_cm_by_fifth": [round(100 * float(np.median(v[b == i])), 2) for i in range(5)]}
    for o in ("peg", "sock"):
        x = np.array([r[o + "_x0"] for r in G]) - CENTRE[o + "_x0"]; y = np.array([r[o + "_y0"] for r in G]) - CENTRE[o + "_y0"]
        if np.ptp(x) > 1e-6:
            # distance from the centre of the object's own start box, each axis scaled by its half-width (5 and 10 cm)
            out[o + "_distance_from_centre"] = {"spearman_tip_lat": spear(v, np.hypot(x / 0.05, y / 0.10))}
    # a linear fit of the signed tip offset on the start positions (R2), the test of a position-dependent bias
    X = np.array([[1] + [r[c] - CENTRE[c] for c in CENTRE] for r in G]); keep = X.std(0) > 1e-9; keep[0] = True; X = X[:, keep]
    d = np.array([r["pre_tip_d"] for r in G])
    for nm, i in (("y", 1), ("z", 2)):
        beta, *_ = np.linalg.lstsq(X, d[:, i], rcond=None); res = d[:, i] - X @ beta
        out[f"linear_R2_signed_{nm}"] = round(float(1 - res.var() / d[:, i].var()), 4)
    return out


def neighbours(F):
    P = np.array([[(r[c] - CENTRE[c]) / (0.05 if "x0" in c else 0.10) for c in CENTRE] for r in F])
    D = ((P[:, None] - P[None]) ** 2).sum(-1); np.fill_diagonal(D, 9e9); nn = D.argmin(1)
    out = {}
    for nm, ks in (("M", ["M"]), ("grasp_classes_0_1_2", ["G0", "G1", "G2"]), ("class_0", ["G0"]), ("class_1", ["G1"])):
        y = np.array([r["kind"] in ks for r in F])
        out[nm] = {"base_rate": round(float(y.mean()), 4), "given_nn_same": round(float(y[nn][y].mean()), 4), "given_nn_not": round(float(y[nn][~y].mean()), 4)}
    return out


def cells(name, obj):
    F = load(name)
    x = np.array([r[obj + "_x0"] for r in F]) - CENTRE[obj + "_x0"]; y = np.array([r[obj + "_y0"] for r in F]) - CENTRE[obj + "_y0"]
    k = np.array([r["kind"] for r in F])
    ix = np.clip(((x + 0.05) / 0.10 * 5 + 1e-9).astype(int), 0, 4); iy = np.clip(((y + 0.10) / 0.20 * 5 + 1e-9).astype(int), 0, 4)
    f = lambda ok: [[round(float(ok[(ix == i) & (iy == j)].mean()), 3) for i in range(5)] for j in range(5)]
    return {"rows": "y from low to high (5 bins of 4 cm), columns x from low to high (5 bins of 2 cm)", "grasp_classes_0_1_2": f(np.isin(k, ["G0", "G1", "G2"])), "met_but_no_pin": f(k == "M"), "strict": f(k == "S")}


def reach_by_distance(F):
    out = {}
    for side, obj in (("r", "peg"), ("l", "sock")):
        R = [r for r in F if r.get("reach_rel_" + side) is not None]
        mu = np.array([r["reach_rel_" + side] for r in R if r["kind"] == "S"]).mean(0)
        e = np.array([np.linalg.norm(np.array(r["reach_rel_" + side]) - mu) for r in R])
        dist = np.array([np.hypot((r[obj + "_x0"] - CENTRE[obj + "_x0"]) / 0.05, (r[obj + "_y0"] - CENTRE[obj + "_y0"]) / 0.10) for r in R])
        ed = np.quantile(dist, [0, .2, .4, .6, .8, 1]); b = np.clip(np.searchsorted(ed, dist, side="right") - 1, 0, 4)
        k = np.array([r["kind"] for r in R])
        out[side] = {"n": len(R), "median_cm_by_distance_fifth_all": [round(100 * float(np.median(e[b == i])), 2) for i in range(5)],
                     "spearman_with_distance_all": spear(e, dist), "spearman_with_distance_strict_only": spear(e[k == "S"], dist[k == "S"]),
                     "median_cm_by_kind": {c: round(100 * float(np.median(e[k == c])), 2) for c in ("S", "A", "M", "G2", "G1", "G0") if (k == c).any()}}
    return out


def cmd_stats():
    F = load("default")
    n = len(F)
    out = {"n": n, "rates": {}, "classes": {}}
    for k in ("benchmark_success", "strict_success"):
        c = sum(r[k] for r in F)
        out["rates"][k] = {"count": c, "rate": c / n, "wilson95": [round(x, 4) for x in wilson(c, n)]}
    out["classes"] = {c: sum(r["failure_class"] == c for r in F) for c in ("0", "1", "2", "3", "4a", "4b")}
    out["strict_variants"] = {"strict_depth": sum(r["strict_depth"] for r in F), "strict_both_fingers": sum(r["strict_both_fingers"] for r in F)}
    out["tracking"] = {m: tracking(load(m)) for m in RUNS}
    out["tracking"]["default"]["auc_tip_lat_M_vs_S_bootstrap95"] = boot_auc(F)
    out["timing_default"] = timing(F)
    out["depth_default"] = depth(F)
    out["time_course_default_tip_lat_cm_median"] = time_course(F)
    out["positions"] = {m: positions(load(m)) for m in ("default", "grid_peg", "grid_sock")}
    out["neighbours_default"] = neighbours(F)
    out["reach_error_default"] = reach_by_distance(F)
    out["cells"] = {"grid_peg_by_peg_start": cells("grid_peg", "peg"), "grid_sock_by_socket_start": cells("grid_sock", "sock"), "default_by_peg_start": cells("default", "peg"), "default_by_socket_start": cells("default", "sock")}
    (S1 / "ins_pattern.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("rates", "classes", "strict_variants", "timing_default", "depth_default", "time_course_default_tip_lat_cm_median", "neighbours_default")}, indent=0))


if __name__ == "__main__":
    {"stats": cmd_stats}[sys.argv[1]]()
