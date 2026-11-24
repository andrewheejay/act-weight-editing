"""Numbers, figures and decisions for Stage 1. Everything here is recomputed from files under experiments/<task>/stage1."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import exp
from .stage1_analysis import CMD_HEIGHTS, auc, load_features

BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8985", "#fcfcfb"
GRAY = "#c9c8c3"
KINDS = (("grasp, never lifted", "grasp_fail", BLUE, "s"), ("lifted, no handover", "lift_fail", ORANGE, "D"))
LOW_Y = 0.44
CMD_HEIGHTS_STR = [f"{h:.2f}" for h in CMD_HEIGHTS]


def records(task, name):
    f = exp(task) / "stage1" / ("grid_default.jsonl" if name == "grid" else f"design_{name}.jsonl")
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def demos(task):
    d = json.load(open(exp(task) / "stage1" / "demo_cubes.json"))["demos"]
    return np.array([[r["x"], r["y"]] for r in d])


def wilson(k, n, z=1.96):
    if not n:
        return [float("nan")] * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def arrays(F):
    xy = np.array([[r["cube_x0"], r["cube_y0"]] for r in F])
    ok = np.array([r["benchmark_success"] for r in F])
    gf = np.array([r["grasp_fail"] for r in F])
    lf = np.array([r["lift_fail"] for r in F])
    return xy, ok, gf, lf


def reach_error(F, ok):
    """Reach error, SPEC.md Measurements: gripper centre minus cube centre at the first close, minus the mean of that vector
    over the successful episodes. Returns (norm per episode, mean vector)."""
    rel = np.array([r.get("reach_rel", [np.nan] * 3) for r in F], float)
    mu = np.nanmean(rel[ok], axis=0)
    return np.linalg.norm(rel - mu, axis=1), rel, mu


def contact_free(FF, ref):
    """Contact-free geometry on one set of features. ref holds the reference vectors/scalars (means over clean successes of the
    design seeds). Thresholds (p90, p95 of the success deviations) come from the clean successes of the same set."""
    R = [r for r in FF if r.get("cmd_clean") and (r["benchmark_success"] or r["grasp_fail"])]
    ok = np.array([r["benchmark_success"] for r in R])
    gf = np.array([r["grasp_fail"] for r in R])

    def block(e, o, g):
        p90, p95 = np.percentile(e[o], [90, 95])
        return {"n_gf": int(g.sum()), "n_ok": int(o.sum()), "auc_gf_vs_ok": auc(e, g), "median_ok_cm": float(100 * np.median(e[o])),
                "median_gf_cm": float(100 * np.median(e[g])), "p90_ok_cm": float(100 * p90), "p95_ok_cm": float(100 * p95),
                "gf_above_p90": int((e[g] > p90).sum()), "gf_above_p95": int((e[g] > p95).sum()), "gf_above_p90_if_like_success": float(0.1 * g.sum())}

    aim = np.array([r["aim_rel"] for r in R])
    cmd = np.array([r["cmd_rel"] for r in R])
    df = np.array([r["defl"] for r in R])
    out = {"n_success_clean": int(ok.sum()), "n_grasp_fail_clean": int(gf.sum())}
    out["aim_measured_at_close"] = block(np.linalg.norm(aim - ref["aim"], axis=1), ok, gf)
    out["cmd_at_close"] = block(np.linalg.norm(cmd - ref["cmd"], axis=1), ok, gf)
    out["deflection_at_close"] = {"median_norm_ok_cm": float(100 * np.median(np.linalg.norm(df[ok], axis=1))), "median_norm_gf_cm": float(100 * np.median(np.linalg.norm(df[gf], axis=1))),
                                  "median_z_ok_cm": float(100 * np.median(df[ok, 2])), "median_z_gf_cm": float(100 * np.median(df[gf, 2]))}
    pre = np.array([r["defl_pre"] if r["defl_pre"] is not None else [np.nan] * 3 for r in R], float)
    hp = ~np.isnan(pre[:, 0])
    out["deflection_one_step_before_first_contact"] = {"n_ok": int((hp & ok).sum()), "n_gf": int((hp & gf).sum()),
        "median_norm_ok_cm": float(100 * np.median(np.linalg.norm(pre[hp & ok], axis=1))), "median_norm_gf_cm": float(100 * np.median(np.linalg.norm(pre[hp & gf], axis=1))),
        "median_z_ok_cm": float(100 * np.median(pre[hp & ok, 2])), "median_z_gf_cm": float(100 * np.median(pre[hp & gf, 2])),
        "p90_z_gf_cm": float(100 * np.percentile(pre[hp & gf, 2], 90))}
    gap = np.array([(r["t_rclose"] - r["t_touch"]) if r["t_touch"] is not None and 0 <= r["t_touch"] <= r["t_rclose"] else np.nan for r in R], float)
    out["clean_contact_steps_before_close"] = {"gf_median": float(np.nanmedian(gap[gf])), "gf_more_than_20": int((gap[gf] > 20).sum()), "ok_median": float(np.nanmedian(gap[ok]))}
    tb = np.array([r["t_touch"] is not None and 0 <= r["t_touch"] < r["t_rclose"] for r in R])
    ea, ec = np.linalg.norm(aim - ref["aim"], axis=1), np.linalg.norm(cmd - ref["cmd"], axis=1)
    p90a = np.percentile(ea[ok], 90)
    out["grasp_fail_by_contact_before_close"] = {lab: {"n": int((gf & m).sum()), "aim_median_cm": float(100 * np.median(ea[gf & m])), "cmd_median_cm": float(100 * np.median(ec[gf & m])),
                                                         "defl_norm_median_cm": float(100 * np.median(np.linalg.norm(df[gf & m], axis=1))), "defl_z_median_cm": float(100 * np.median(df[gf & m, 2])),
                                                         "aim_share_above_success_p90": float((ea[gf & m] > p90a).mean())} for lab, m in (("contact_before_close", tb), ("no_contact_before_close", ~tb))}
    out["cmd_lat"] = {}
    for key in CMD_HEIGHTS_STR + ["close"]:
        v = np.array([r["cmd_lat_close"] if key == "close" else (np.nan if r["cmd_lat_h"][key] is None else r["cmd_lat_h"][key]) for r in R], float)
        have = ~np.isnan(v)
        d = v[have] - ref["lat"][key]
        o, g = ok[have], gf[have]
        out["cmd_lat"][key] = block(np.abs(d), o, g)
        out["cmd_lat"][key].update({"n_crossing": int(have.sum()), "signed_mean_dev_gf_cm": float(100 * d[g].mean()),
                                    "signed_dev_positive_share_gf": float((d[g] > 0).mean()), "signed_auc": auc(d, g)})
    return out


def contact_free_stats(F, FG=None):
    """Reference vectors/scalars = mean over clean successful design episodes."""
    C = [r for r in F if r.get("cmd_clean") and r["benchmark_success"]]
    ref = {"aim": np.nanmean([r["aim_rel"] for r in C], axis=0), "cmd": np.nanmean([r["cmd_rel"] for r in C], axis=0), "lat": {}}
    for h in CMD_HEIGHTS_STR:
        ref["lat"][h] = float(np.nanmean([np.nan if r["cmd_lat_h"][h] is None else r["cmd_lat_h"][h] for r in C]))
    ref["lat"]["close"] = float(np.mean([r["cmd_lat_close"] for r in C]))
    out = {"definition": "clean subset (chunk at the first close decided before any right-finger contact), reference vectors from clean successful design episodes",
           "reference_lat_cm": {k: 100 * v for k, v in ref["lat"].items()}, "design": contact_free(F, ref)}
    if FG is not None:
        out["grid_design_reference"] = contact_free(FG, ref)
    # per-class share of grasp failures above the success p90 of reach error (spec definition), for the taxonomy text
    return out


def pair_table(task, names):
    """Paired against the default run on benchmark success."""
    base = {r["seed"]: r["benchmark_success"] for r in records(task, "default")}
    out = {}
    for n in names:
        r = {x["seed"]: x["benchmark_success"] for x in records(task, n)}
        both = sorted(set(r) & set(base))
        if not both:
            continue
        b = sum(r[s] and not base[s] for s in both)
        c = sum(base[s] and not r[s] for s in both)
        out[n] = {"n": len(both), "fixed": int(b), "broken": int(c), "net_points": round(100 * (b - c) / len(both), 2)}
    return out


def stats(task):
    st = {}
    for n in ("default", "chunk50", "chunk25", "ens"):
        R = records(task, n)
        if R:
            k = sum(r["benchmark_success"] for r in R)
            s = sum(r["strict_success"] for r in R)
            st[f"rate_{n}"] = {"n": len(R), "benchmark": k / len(R), "strict": s / len(R), "benchmark_ci95": wilson(k, len(R)), "classes": {c: sum(r["failure_class"] == c for r in R) for c in ("0", "1", "2", "3", "4a", "4b")}}
    st["paired_vs_default_benchmark"] = pair_table(task, ["chunk50", "chunk25", "ens"])
    F = load_features(task, "default")
    xy, ok, gf, lf = arrays(F)
    fail = ~ok
    err, rel, mu = reach_error(F, ok)
    n_fail = int(fail.sum())
    st["default_taxonomy"] = {
        "n_fail": n_fail, "grasp_fail": int(gf.sum()), "lift_fail": int(lf.sum()),
        "grasp_fail_share": float(gf.sum() / n_fail),
        "class_by_kind": {k: {c: int(sum(r["failure_class"] == c and r[k] for r in F)) for c in ("0", "1", "2", "3", "4a", "4b")} for k in ("grasp_fail", "lift_fail")},
        "lift_fail_dropped": int(sum(1 for r in F if r["lift_fail"] and r["t_drop"] >= 0)),
        "lift_fail_held_to_end": int(sum(1 for r in F if r["lift_fail"] and r["t_drop"] < 0)),
        "strict_fail_benchmark_ok": int(sum((not r["strict_success"]) and r["benchmark_success"] for r in F)),
    }
    grasp = gf | ok
    touch = np.array([r["t_touch"] for r in F])
    close = np.array([r["t_rclose"] for r in F])
    early = (touch >= 0) & (close - touch > 20)
    knock = np.array([r["knock_m"] for r in F])
    st["reach_error"] = {
        "reference_mean_rel_ok": [float(v) for v in mu], "median_ok": float(np.median(err[ok])), "median_grasp_fail": float(np.median(err[gf])), "median_lift_fail": float(np.median(err[lf])),
        "auc_grasp_fail_vs_ok": auc(err[grasp], gf[grasp]), "auc_lift_fail_vs_ok": auc(err[lf | ok], lf[lf | ok]),
        "p90_ok": float(np.percentile(err[ok], 90)), "share_gf_above_p90_ok": float((err[gf] > np.percentile(err[ok], 90)).mean()),
        "knock_median_ok_gf": [float(np.median(knock[ok])), float(np.median(knock[gf]))],
        "auc_knock": auc(knock[grasp], gf[grasp]),
        "early_touch_gt20_before_close": {"ok": float(early[ok].mean()), "grasp_fail": float(early[gf].mean()), "lift_fail": float(early[lf].mean())},
        "touch_gap_median_ok_gf": [float(np.median((close - touch)[ok & (touch >= 0)])), float(np.median((close - touch)[gf & (touch >= 0)]))],
    }
    # position dependence of the reach vector among successes: gain error
    dev = rel - mu
    cx, cy = xy[:, 0] - 0.1, xy[:, 1] - 0.5
    reg = {}
    for c, j in (("x", 0), ("y", 1), ("z", 2)):
        A = np.c_[np.ones(ok.sum()), cx[ok], cy[ok]]
        b, *_ = np.linalg.lstsq(A, rel[ok, j], rcond=None)
        reg["rel_" + c] = {"intercept": float(b[0]), "slope_cube_x": float(b[1]), "slope_cube_y": float(b[2]), "resid_sd": float((rel[ok, j] - A @ b).std())}
    st["reach_rel_position_regression_on_successes"] = reg
    zone = {}
    low = xy[:, 1] < LOW_Y
    for nm, m in (("y_below_0.44", low), ("y_above_0.44", ~low)):
        zone[nm] = {"n": int(m.sum()), "fail": float(fail[m].mean()), "grasp_fail": float(gf[m].mean()), "lift_fail": float(lf[m].mean()), "median_reach_err_ok": float(np.median(err[m & ok])), "median_reach_err_gf": float(np.median(err[m & gf])) if (m & gf).any() else None, "auc_gf_vs_ok_in_zone": auc(err[m & grasp], gf[m & grasp])}
    xb = np.linspace(0, 0.2, 6)
    yb = np.linspace(0.4, 0.6, 6)
    zone["grid5x5_failrate"] = [[float(fail[(xy[:, 0] >= xb[i]) & (xy[:, 0] < xb[i + 1] + 1e-9 * (i == 4)) & (xy[:, 1] >= yb[j]) & (xy[:, 1] < yb[j + 1] + 1e-9 * (j == 4))].mean()) for i in range(5)] for j in range(5)]
    zone["grid5x5_graspfail"] = [[float(gf[(xy[:, 0] >= xb[i]) & (xy[:, 0] < xb[i + 1] + 1e-9 * (i == 4)) & (xy[:, 1] >= yb[j]) & (xy[:, 1] < yb[j + 1] + 1e-9 * (j == 4))].mean()) for i in range(5)] for j in range(5)]
    st["zones"] = zone
    # first right-finger contact, in the gripper frame, one step before contact (cube not moved yet)
    tl, tf, th = (np.array([r.get(k, np.nan) for r in F], float) for k in ("touch_lat", "touch_fwd", "touch_height"))
    st["first_contact_in_gripper_frame"] = {
        nm: {"n": int((sel & ~np.isnan(tl)).sum()), "lat_median_m": float(np.nanmedian(tl[sel])), "abs_lat_median_m": float(np.nanmedian(np.abs(tl[sel]))), "lat_positive_share": float(np.nanmean((tl[sel] > 0))), "fwd_median_m": float(np.nanmedian(tf[sel])), "height_above_cube_median_m": float(np.nanmedian(th[sel]))}
        for nm, sel in (("success", ok), ("grasp_fail", gf), ("lift_fail", lf))}
    shift = np.array([r.get("pre_touch_cube_shift_m", np.nan) for r in F], float)
    st["pre_touch_cube_shift_mm"] = {nm: {"median": float(1000 * np.nanmedian(shift[sel])), "p95": float(1000 * np.nanpercentile(shift[sel], 95)), "max": float(1000 * np.nanmax(shift[sel]))} for nm, sel in (("success", ok), ("grasp_fail", gf), ("lift_fail", lf))}
    st["first_contact_in_gripper_frame"]["auc_abs_lat_grasp_fail_vs_success"] = auc(np.abs(tl)[gf | ok], gf[gf | ok])
    st["first_contact_in_gripper_frame"]["auc_height_grasp_fail_vs_success"] = auc(th[gf | ok], gf[gf | ok])
    # neighbour agreement: does the nearest design seed (cube position) fail too
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    st["neighbours"] = {"nearest_seed_fails_given_fail": float(fail[nn][fail].mean()), "base_rate_fail": float(fail.mean()), "nearest_seed_fails_given_success": float(fail[nn][~fail].mean())}
    dm = demos(task)
    dd = np.linalg.norm(xy[:, None] - dm[None], axis=-1)
    near = dd.min(1)
    st["demos"] = {"n": int(len(dm)), "nearest_demo_dist_median": float(np.median(near)), "auc_fail_by_demo_distance": auc(near, fail), "failrate_by_demo_distance_quartile": [float(fail[(near >= q0) & (near < q1)].mean()) for q0, q1 in zip(np.percentile(near, [0, 25, 50, 75]), list(np.percentile(near, [25, 50, 75])) + [np.inf])]}
    meet = [r for r in F if r["lift_fail"]]
    mr = np.array([r["meet_rel"] for r in F if r["benchmark_success"] and "meet_rel" in r], float)
    mm = np.median(mr, axis=0)
    mf = np.array([r["meet_rel"] for r in meet if "meet_rel" in r], float)
    st["lift_fail_meet"] = {"n": len(meet), "left_close_started": int(sum((r["t_lclose"] or -1) >= 0 for r in meet)), "meet_rel_reference_median_success": [float(v) for v in mm],
                            "meet_error_median_success_m": float(np.median(np.linalg.norm(mr - mm, axis=1))) if len(mr) else None,
                            "meet_error_median_lift_fail_m": float(np.median(np.linalg.norm(mf - mm, axis=1))) if len(mf) else None, "n_lift_fail_with_meet": len(mf)}
    # robustness of the reach error as a tracking number (the reference vector was built from the same successes it is tested on)
    def gf_vs_ok(FF, use_mu, sel_rows=None):
        XY, OK, GF, LF = arrays(FF)
        rr = np.array([r.get("reach_rel", [np.nan] * 3) for r in FF], float)
        e = np.linalg.norm(rr - use_mu, axis=1)
        m = (GF | OK) if sel_rows is None else (GF | OK) & sel_rows
        return auc(e[m], GF[m])
    odd = np.array([r["seed"] % 2 == 1 for r in F])
    rob = {"reference_from": "even-seed successes, tested on odd seeds, and the reverse"}
    for nm, half in (("ref_even_test_odd", odd), ("ref_odd_test_even", ~odd)):
        mu_h = np.nanmean(rel[ok & ~half], axis=0)
        rob[nm] = gf_vs_ok(F, mu_h, half)
    if (exp(task) / "stage1" / "features_grid.jsonl.gz").exists():
        FG = load_features(task, "grid")
        rob["grid_auc_reach_error_design_reference"] = gf_vs_ok(FG, mu)
        _, OKG, GFG, _ = arrays(FG)
        tlg, thg = (np.array([r.get(k, np.nan) for r in FG], float) for k in ("touch_lat", "touch_height"))
        rob["grid_auc_abs_lat_first_contact"] = auc(np.abs(tlg)[GFG | OKG], GFG[GFG | OKG])
        rob["grid_auc_height_first_contact"] = auc(thg[GFG | OKG], GFG[GFG | OKG])
        rob["grid_counts"] = {"n": len(FG), "success": int(OKG.sum()), "grasp_fail": int(GFG.sum())}
    rob["seed_position_is_independent_for_grid"] = "the grid is a separate set of poses, the policy is deterministic given the state"
    st["tracking_robustness"] = rob
    # SPEC.md "the one diagnostic": is the error tied to distance from the centre (gain) or to zones. Failure against position, on the grid and the design seeds
    def pattern(FF):
        XY, OK, GF, LF = arrays(FF)
        fl = ~OK
        rc = np.hypot(XY[:, 0] - 0.1, XY[:, 1] - 0.5)
        q = np.percentile(rc, [25, 50, 75])
        edges = [-1, *q, 9]
        def fr(sel):
            return float(fl[sel].mean())
        xb = XY[:, 0]; yb = XY[:, 1]
        cx = np.abs(xb - 0.1)
        return {"fail_rate_by_centre_distance_quartile": [fr((rc >= a) & (rc < b)) for a, b in zip(edges[:-1], edges[1:])],
                "auc_fail_by_centre_distance": auc(rc, fl),
                "fail_rate_y_below_0.44": fr(yb < LOW_Y), "fail_rate_y_above_0.44": fr(yb >= LOW_Y),
                "auc_fail_by_minus_y": auc(-yb, fl), "auc_fail_by_abs_x_minus_0.1": auc(cx, fl),
                "fail_rate_grid": float(fl.mean()), "n": len(FF)}
    st["pattern"] = {"design_seeds": pattern(F)}
    if (exp(task) / "stage1" / "features_grid.jsonl.gz").exists():
        st["pattern"]["grid"] = pattern(FG)
    rcok = np.hypot(xy[:, 0] - 0.1, xy[:, 1] - 0.5)
    st["pattern"]["success_reach_error_vs_centre_distance_spearman"] = float(np.corrcoef(np.argsort(np.argsort(rcok[ok])), np.argsort(np.argsort(err[ok])))[0, 1])
    st["pattern"]["success_reach_error_vs_y_spearman"] = float(np.corrcoef(np.argsort(np.argsort(xy[ok, 1])), np.argsort(np.argsort(err[ok])))[0, 1])
    # centre pull: mean gripper deviation (reach_rel minus the success mean) per 4 cm cell, projected on the direction to the centre of the square
    def centre_pull(sel):
        bx_, by_ = bins_xy(5)
        out = {}
        for i in range(5):
            for j in range(5):
                m = sel & (xy[:, 0] >= bx_[i]) & (xy[:, 0] < bx_[i + 1] + (i == 4)) & (xy[:, 1] >= by_[j]) & (xy[:, 1] < by_[j + 1] + (j == 4))
                c = np.array([(bx_[i] + bx_[i + 1]) / 2, (by_[j] + by_[j + 1]) / 2])
                to = np.array([0.1, 0.5]) - c
                if m.sum() < 3 or np.linalg.norm(to) < 1e-9:
                    continue
                kind = "corner" if i in (0, 4) and j in (0, 4) else ("edge" if i in (0, 4) or j in (0, 4) else "interior")
                out.setdefault(kind, []).append(float(np.nanmean(dev[m, :2], 0) @ (to / np.linalg.norm(to))))
        return {k: {"cells": len(v), "toward_centre": int(sum(x > 0 for x in v)), "mean_projection_m": float(np.mean(v)), "max_projection_m": float(np.max(v))} for k, v in out.items()}
    st["pattern"]["centre_pull_success_only"] = centre_pull(ok)
    # circularity check: gripper at the first close against the cube's initial (untouched) position. The bump can still enter through
    # the policy, if a chunk decided after the bump follows the moved cube, hence the clean subset below
    def aim_auc(FF, use_mu=None):
        _, OK, GF, _ = arrays(FF)
        a = np.array([r.get("aim_rel", [np.nan] * 3) for r in FF], float)
        m_ = np.nanmean(a[OK], axis=0) if use_mu is None else use_mu
        e = np.linalg.norm(a - m_, axis=1)
        sel = GF | OK
        return auc(e[sel], GF[sel]), float(np.nanmedian(e[OK])), float(np.nanmedian(e[GF])), m_, e, a
    aa, mok, mgf, amu, aerr, aim = aim_auc(F)
    st["aim_error_initial_cube"] = {"auc_grasp_fail_vs_ok": aa, "median_ok_m": mok, "median_grasp_fail_m": mgf,
                                    "xy_only_auc": auc(np.linalg.norm((aim - amu)[:, :2], axis=1)[gf | ok], gf[gf | ok]),
                                    "z_only_auc": auc(np.abs((aim - amu)[:, 2])[gf | ok], gf[gf | ok])}
    if (exp(task) / "stage1" / "features_grid.jsonl.gz").exists():
        ga = aim_auc(FG, amu)
        st["aim_error_initial_cube"]["grid_auc_design_reference"] = ga[0]
        st["aim_error_initial_cube"]["grid_medians_ok_gf_m"] = [ga[1], ga[2]]
    # clean subset: the chunk that is active at the first close was decided (step 100*floor(t/100)) before any right-finger contact
    def clean(FF):
        tc_ = np.array([r["t_rclose"] if r["t_rclose"] is not None else -1 for r in FF]); tt_ = np.array([r["t_touch"] if r["t_touch"] is not None else -1 for r in FF])
        return (tc_ >= 0) & ((tt_ < 0) | (tt_ >= 100 * (tc_ // 100)))
    def clean_auc(FF, mu_aim, mu_reach):
        _, OK, GF, LF_ = arrays(FF)
        c_ = clean(FF)
        a = np.array([r.get("aim_rel", [np.nan] * 3) for r in FF], float); q_ = np.array([r.get("reach_rel", [np.nan] * 3) for r in FF], float)
        ea, eq = np.linalg.norm(a - mu_aim, axis=1), np.linalg.norm(q_ - mu_reach, axis=1)
        sel = c_ & (GF | OK)
        return {"n_success": int((c_ & OK).sum()), "n_grasp_fail": int((c_ & GF).sum()), "of_success": int(OK.sum()), "of_grasp_fail": int(GF.sum()),
                "aim_auc": auc(ea[sel], GF[sel]), "reach_auc": auc(eq[sel], GF[sel]), "aim_median_ok_m": float(np.nanmedian(ea[c_ & OK])), "aim_median_gf_m": float(np.nanmedian(ea[c_ & GF]))}
    st["clean_subset_chunk_decided_before_contact"] = {"design": clean_auc(F, amu, mu)}
    if (exp(task) / "stage1" / "features_grid.jsonl.gz").exists():
        st["clean_subset_chunk_decided_before_contact"]["grid_design_reference"] = clean_auc(FG, amu, mu)
    st["contact_free_geometry"] = contact_free_stats(F, FG if (exp(task) / "stage1" / "features_grid.jsonl.gz").exists() else None)
    # per-class share of grasp failures above the success p90 of reach error
    p90r = np.percentile(err[ok], 90)
    st["reach_error"]["share_gf_above_p90_by_class"] = {c: {"n": int(sum(1 for i, r in enumerate(F) if r["failure_class"] == c and r["grasp_fail"])), "above_p90": int(sum(1 for i, r in enumerate(F) if r["failure_class"] == c and r["grasp_fail"] and err[i] > p90r))} for c in ("0", "1", "2")}
    # zone refinement in 4 cm cells and a 2 by 2 look at the hot block (design seeds)
    fb = ((xy[:, 0] >= 0.08) & (xy[:, 0] < 0.16) & (xy[:, 1] >= 0.40) & (xy[:, 1] < 0.48))
    st["zones"]["block_x0.08_0.16_y0.40_0.48"] = {"n": int(fb.sum()), "fail": int((fail & fb).sum()), "grasp_fail": int((gf & fb).sum()), "share_of_all_failures": float((fail & fb).sum() / fail.sum())}
    # SPEC diagnostic proper: reach error (and aim error) against the start position, over ALL episodes with a close, not successes only
    rc_all = np.hypot(xy[:, 0] - 0.1, xy[:, 1] - 0.5)
    eb = [0, 0.04, 0.06, 0.08, 0.10, 0.15]
    st["error_vs_centre_distance"] = {"bins_m": eb, "n": [], "fail_rate": [], "reach_median_all_m": [], "reach_median_success_m": [], "aim_median_all_m": [], "aim_median_success_m": []}
    for lo, hi in zip(eb[:-1], eb[1:]):
        m = (rc_all >= lo) & (rc_all < hi)
        o = st["error_vs_centre_distance"]
        o["n"].append(int(m.sum())); o["fail_rate"].append(float(fail[m].mean()))
        o["reach_median_all_m"].append(float(np.nanmedian(err[m]))); o["reach_median_success_m"].append(float(np.nanmedian(err[m & ok])))
        o["aim_median_all_m"].append(float(np.nanmedian(aerr[m]))); o["aim_median_success_m"].append(float(np.nanmedian(aerr[m & ok])))
    st["aim_rel_regression_all_episodes"] = {}
    good = ~np.isnan(aim[:, 0])
    for c, j in (("x", 0), ("y", 1), ("z", 2)):
        A = np.c_[np.ones(good.sum()), cx[good], cy[good]]
        b, *_ = np.linalg.lstsq(A, aim[good, j], rcond=None)
        st["aim_rel_regression_all_episodes"]["rel_" + c] = {"slope_cube_x": float(b[1]), "slope_cube_y": float(b[2]), "resid_sd": float((aim[good, j] - A @ b).std())}
    # label cut and lateness of the lift among lift failures
    zm = np.array([r["zmax20"] for r in F]); tl6 = np.array([r["t_lift60"] for r in F])
    st["label_cut"] = {"grasp_fail_max_zmax20": float(zm[gf].max()), "lift_fail_min_zmax20": float(zm[lf].min()), "success_min_zmax20": float(zm[ok].min()),
                       "grasp_fail_count_at_0.05_0.06_0.08": [int((fail & (zm <= c)).sum()) for c in (0.05, 0.06, 0.08)],
                       "lift_fail_below_0.10": int((lf & (zm < 0.10)).sum()), "class2_not_lifted_zmax20_median": float(np.median([r["zmax20"] for r in F if r["failure_class"] == "2" and r["grasp_fail"]]))}
    st["lift_fail_lateness"] = {"t_lift60_median_success": float(np.median(tl6[ok])), "t_lift60_p95_success": float(np.percentile(tl6[ok], 95)), "t_lift60_median_lift_fail": float(np.median(tl6[lf])),
                                "lift_fail_later_than_success_p95": int((lf & (tl6 > np.percentile(tl6[ok], 95))).sum()), "lift_fail_after_step_250": int((lf & (tl6 > 250)).sum()), "n_lift_fail": int(lf.sum())}
    # max-reward class 0 means the stock touch flag never fired. The flag looks at one finger geom (10_right_gripper_finger).
    R0 = [r for r in F if r["failure_class"] == "0"]
    st["class0_contact_by_any_right_finger"] = {"n": len(R0), "with_t_touch": int(sum((r["t_touch"] or -1) >= 0 for r in R0))}
    if len(mf):
        dz = (mf - mm)[:, 2]
        st["lift_fail_meet"].update({"z_error_above_10cm": int((dz > 0.10).sum()), "z_error_within_10cm": int((np.abs(dz) <= 0.10).sum()),
                                     "xy_error_median_m_when_z_within_10cm": float(np.median(np.linalg.norm((mf - mm)[np.abs(dz) <= 0.10][:, :2], axis=1)))})
    held = np.array([r["lift_fail"] and r["t_drop"] < 0 for r in F])
    st["lift_fail_lateness"]["held_to_end"] = int(held.sum())
    st["lift_fail_lateness"]["held_to_end_lifted_after_step_250"] = int((held & (tl6 > 250)).sum())
    st["strict_variant_both_left_fingers"] = {"strict_any_left_finger": float(np.mean([r["strict_success"] for r in records(task, "default")])), "strict_both_left_fingers": float(np.mean([r["strict_both_fingers"] for r in records(task, "default")]))}
    rg = records(task, "grid")
    if rg:
        st["grid_rate"] = {"n": len(rg), "benchmark": float(np.mean([r["benchmark_success"] for r in rg])), "strict": float(np.mean([r["strict_success"] for r in rg])), "low_y_strip_success": float(np.mean([r["benchmark_success"] for r in rg if r["cube_y0"] < LOW_Y])) if any(r["cube_y0"] < LOW_Y for r in rg) else None}
    return st, (xy, ok, gf, lf, err, rel, mu, near, touch, close)


def cmd_decisions(task):
    """Apply the Gate 1 rules of referee/RULES.md to the design runs. Nothing is chosen by hand except the two free-text entries."""
    R = {n: records(task, n) for n in ("default", "chunk50", "chunk25")}
    rate = lambda rows, key: sum(r[key] for r in rows) / len(rows)
    bench, strict = rate(R["default"], "benchmark_success"), rate(R["default"], "strict_success")
    primary = "strict" if (bench - strict) * 100 >= 10 else "benchmark"
    key = primary + "_success"
    base = rate(R["default"], key)
    gains = {c: (rate(R[f"chunk{c}"], key) - base) * 100 for c in (50, 25)}
    best = max(gains, key=lambda c: gains[c])
    n_steps = best if gains[best] >= 3 else 100
    d = {"primary_metric": primary, "operating_point": {"n_action_steps": n_steps}, "proceed": bool(1 - base >= 0.05), "target_failure_class": "grasp_fail (benchmark failure and the cube centre never above 0.06 m after step 20. Contains max-reward class 0 (the stock touch flag, which watches one finger geom, never fired), class 1 and the class 2 episodes with no real lift)",
         "tracking_number": "reach_error (3D, gripper centre minus cube centre at the first close of the right gripper, minus the mean of that vector over the 816 successful design episodes at the default setting)",
         "evidence": {"benchmark_success": bench, "strict_success": strict, "chunk_gains_points": gains}}
    (exp(task) / "stage1" / "decisions.json").write_text(json.dumps(d, indent=1))
    print(json.dumps(d, indent=1))


def cmd_stats(task):
    st, _ = stats(task)
    (exp(task) / "stage1" / "stage1_stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


def _axis_style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors="#52514e")


def bins_xy(n=5):
    return np.linspace(0, 0.2, n + 1), np.linspace(0.4, 0.6, n + 1)


def meet_figure(task, F, ok, lf):
    """Meet error for the handover, SPEC.md: left gripper centre minus cube centre at its first close after the lift."""
    mr = np.array([r["meet_rel"] for r in F if r["benchmark_success"] and "meet_rel" in r], float)
    mm = np.median(mr, axis=0)
    fl = [r for r in F if r["lift_fail"] and "meet_rel" in r]
    fx = np.array([r["meet_rel"] for r in fl], float)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURF)
    for ax, j, lab in ((axs[0], 1, "y"), (axs[1], 2, "z")):
        ax.scatter((mr - mm)[:, 0] * 100, (mr - mm)[:, j] * 100, s=7, c=GRAY, label=f"success ({len(mr)})", zorder=2)
        if len(fx):
            ax.scatter((fx - mm)[:, 0] * 100, (fx - mm)[:, j] * 100, s=26, c=ORANGE, marker="D", edgecolors=SURF, linewidths=0.8, label=f"lifted, no handover, left gripper closed ({len(fx)})", zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":"), ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("x, relative to the success median (cm)"), ax.set_ylabel(lab + " (cm)")
        _axis_style(ax)
    axs[0].set_title("Meet error of the left gripper at its first close", fontsize=10, loc="left", color=INK)
    axs[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.savefig(exp(task) / "stage1" / "meet_error.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def cmd_figures(task):
    _, (xy, ok, gf, lf, err, rel, mu, near, touch, close) = stats(task)
    s1 = exp(task) / "stage1"
    dm = demos(task)
    fail_any = ~ok
    # ---------------------------------------------------------------- failure map
    G = records(task, "grid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), facecolor=SURF)
    # SPEC.md step 3 asks for the failure map by class: max-reward class of every design episode
    ax = axs[2]
    F_ = load_features(task, "default")
    cls = np.array([r["failure_class"] for r in F_])
    for c, lab, col, mk, sz in (("4b", "4b handover complete (success)", GRAY, "o", 9), ("4a", "4a touch in the air, no complete handover (success)", AQUA, "o", 22),
                                ("2", "2 lift flag, left never touched", ORANGE, "D", 26), ("1", "1 touch, no lift", BLUE, "s", 26), ("0", "0 touch flag never fired", "#7a3fa0", "^", 34)):
        m_ = cls == c
        if m_.any():
            ax.scatter(*xy[m_].T, s=sz, c=col, marker=mk, edgecolors=SURF, linewidths=0.6, label=f"{lab} ({m_.sum()})", zorder=3 if c != "4b" else 2)
    ax.scatter(*dm.T, s=34, c=INK, marker="x", linewidths=1.3, label="training demo cube", zorder=4)
    ax.set_title("Design seeds by SPEC class (max reward reached)", fontsize=10, loc="left", color=INK)
    ax.set_xlabel("cube start x (m)"), ax.set_ylabel("cube start y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=8)
    _axis_style(ax)
    ax = axs[0]
    ax.scatter(*xy[ok].T, s=9, c=GRAY, marker="o", label=f"success ({ok.sum()})", zorder=2)
    for lab, key, c, m in KINDS:
        sel = gf if key == "grasp_fail" else lf
        ax.scatter(*xy[sel].T, s=26, c=c, marker=m, edgecolors=SURF, linewidths=0.8, label=f"{lab} ({sel.sum()})", zorder=3)
    ax.scatter(*dm.T, s=34, c=INK, marker="x", linewidths=1.3, label=f"training demo cube ({len(dm)})", zorder=4)
    ax.set_title("Design seeds, original policy, 100-step chunks", fontsize=10, loc="left", color=INK)
    ax.set_xlabel("cube start x (m)"), ax.set_ylabel("cube start y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8)
    _axis_style(ax)
    ax = axs[1]
    if G:
        gx = np.array([r["pose"][0] for r in G]); gy = np.array([r["pose"][1] for r in G])
        v = np.array([r["benchmark_success"] for r in G], float)
        ux, uy = np.unique(np.round(gx, 6)), np.unique(np.round(gy, 6))
        Z = np.full((len(uy), len(ux)), np.nan)
        for a, b, w in zip(gx, gy, v):
            Z[np.searchsorted(uy, round(b, 6)), np.searchsorted(ux, round(a, 6))] = w
        dx = (ux[1] - ux[0]) / 2
        ext = [ux[0] - dx, ux[-1] + dx, uy[0] - dx, uy[-1] + dx]
        im = ax.imshow(Z, origin="lower", extent=ext, cmap=plt.matplotlib.colors.ListedColormap([ORANGE, GRAY]), vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        ax.scatter(*dm.T, s=30, c=INK, marker="x", linewidths=1.2, label="training demo cube")
        ax.set_title(f"41 by 41 grid, same policy ({100 * v.mean():.1f}% success)", fontsize=10, loc="left", color=INK)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=GRAY, label="success"), Patch(color=ORANGE, label="benchmark failure")] + [plt.Line2D([], [], color=INK, marker="x", ls="none", label="training demo cube")], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    ax.set_xlabel("cube start x (m)")
    _axis_style(ax)
    fig.savefig(s1 / "failure_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    # ---------------------------------------------------------------- reach error
    dev = rel - mu
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax = axs[0]
    bx, by = bins_xy(5)
    for i in range(5):
        for j in range(5):
            m = (xy[:, 0] >= bx[i]) & (xy[:, 0] < bx[i + 1] + (i == 4)) & (xy[:, 1] >= by[j]) & (xy[:, 1] < by[j + 1] + (j == 4))
            if m.sum() < 5:
                continue
            c = plt.cm.Oranges(0.15 + 0.8 * min(1, fail_any[m].mean() / 0.5))
            ax.add_patch(plt.Rectangle((bx[i], by[j]), bx[1], by[1] - by[0], color=c, zorder=1))
            d = dev[m].mean(0)
            ax.annotate("", xy=((bx[i] + bx[i + 1]) / 2 + 2.0 * d[0], (by[j] + by[j + 1]) / 2 + 2.0 * d[1]), xytext=((bx[i] + bx[i + 1]) / 2, (by[j] + by[j + 1]) / 2),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.6), zorder=4)
    ax.scatter(*dm.T, s=14, c=INK, marker="x", linewidths=1, zorder=3)
    ax.set_xlim(0, 0.2), ax.set_ylim(0.4, 0.6)
    ax.set_title("Reach error of the gripper at the first close,\nmean by 4 cm cell (arrow, 2x length), shade = failure rate", fontsize=9, loc="left", color=INK)
    ax.set_xlabel("cube start x (m)"), ax.set_ylabel("cube start y (m)")
    _axis_style(ax)
    ax = axs[1]
    r0 = np.hypot(xy[:, 0] - 0.1, xy[:, 1] - 0.5)
    ax.scatter(r0[ok] * 100, err[ok] * 100, s=6, c=GRAY, label="success", zorder=2)
    for lab, key, c, m in KINDS:
        sel = gf if key == "grasp_fail" else lf
        ax.scatter(r0[sel] * 100, err[sel] * 100, s=18, c=c, marker=m, edgecolors=SURF, linewidths=0.6, label=lab, zorder=3)
    ax.set_xlabel("cube distance from the square's centre (cm)"), ax.set_ylabel("reach error, 3D (cm)")
    ax.set_ylim(0, 10)
    ax.set_title("Reach error against distance from centre", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _axis_style(ax)
    ax = axs[2]
    binse = np.linspace(0, 8, 33)
    for lab, sel, c in (("success", ok, MUTED), ("grasp, never lifted", gf, BLUE), ("lifted, no handover", lf, ORANGE)):
        ax.hist(err[sel] * 100, bins=binse, histtype="step", density=True, color=c, lw=2, label=f"{lab} ({sel.sum()})")
    ax.set_xlabel("reach error, 3D (cm)"), ax.set_ylabel("density")
    ax.set_title(f"Separation, AUC grasp failure vs success {auc(err[gf | ok], gf[gf | ok]):.2f}", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _axis_style(ax)
    fig.savefig(s1 / "reach_error.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    meet_figure(task, load_features(task, "default"), ok, lf)
