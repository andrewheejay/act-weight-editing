"""Numbers, figures and decisions for Stage 1. Everything here is recomputed from files under experiments/<task>/stage1."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import exp
from .stage1_analysis import auc, load_features

BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8985", "#fcfcfb"
GRAY = "#c9c8c3"
KINDS = (("grasp, never lifted", "grasp_fail", BLUE, "s"), ("lifted, no handover", "lift_fail", ORANGE, "D"))
LOW_Y = 0.44


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
    # max-reward class 0 means the stock touch flag never fired. The flag looks at one finger geom (10_right_gripper_finger).
    R0 = [r for r in F if r["failure_class"] == "0"]
    st["class0_contact_by_any_right_finger"] = {"n": len(R0), "with_t_touch": int(sum((r["t_touch"] or -1) >= 0 for r in R0))}
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
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=SURF)
    ax.scatter((mr - mm)[:, 0] * 100, (mr - mm)[:, 1] * 100, s=7, c=GRAY, label=f"success ({len(mr)})", zorder=2)
    if len(fx):
        ax.scatter((fx - mm)[:, 0] * 100, (fx - mm)[:, 1] * 100, s=26, c=ORANGE, marker="D", edgecolors=SURF, linewidths=0.8, label=f"lifted, no handover ({len(fx)})", zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":"), ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("x, relative to the success median (cm)"), ax.set_ylabel("y (cm)")
    ax.set_title("Meet error of the left gripper at its first close", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _axis_style(ax)
    fig.savefig(exp(task) / "stage1" / "meet_error.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def cmd_figures(task):
    _, (xy, ok, gf, lf, err, rel, mu, near, touch, close) = stats(task)
    s1 = exp(task) / "stage1"
    dm = demos(task)
    fail_any = ~ok
    # ---------------------------------------------------------------- failure map
    G = records(task, "grid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), facecolor=SURF)
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
