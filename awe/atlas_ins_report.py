"""Turn insertion stage2/atlas/raw.npz into heads.json, effect_maps.npz and the figures. Definitions: awe/atlas_ins.py docstring."""
import json

import numpy as np

from .atlas import JOINTS, KH, KW
from .atlas_ins import FULL_STEPS, GRID_N, GRIDS, STEPS
from .atlas_report import VAR, r4, rho
from .paths import exp

SMOOTH_M = 0.01
DX, DY = 0.0025, 0.005  # grid spacing in x and y (m)
EXT = {"peg": (10, 20, 40, 60), "sock": (-20, -10, 40, 60)}
TITLE = {"peg": "peg", "sock": "socket"}
COL = {"peg": slice(0, 2), "sock": slice(7, 9)}


def smooth(m):
    from scipy.ndimage import gaussian_filter

    s = (SMOOTH_M / DY, SMOOTH_M / DX)  # axes: y, x
    return gaussian_filter(m, s, mode="constant") / gaussian_filter(np.ones_like(m), s, mode="constant")


def failure_maps(task):
    """0/1 maps per grid, indexed [iy, ix]: strict failure (the score), class 3, benchmark failure, grasp failure (classes 0 to 2)."""
    out = {}
    for g in GRIDS:
        rows = [json.loads(l) for l in (exp(task) / "stage1" / f"grid_{g}_default.jsonl").read_text().splitlines()]
        x0, x1, y0, y1 = (v / 100 for v in EXT[g])
        ix = {round(v, 4): i for i, v in enumerate(np.linspace(x0, x1, GRID_N))}
        iy = {round(v, 4): i for i, v in enumerate(np.linspace(y0, y1, GRID_N))}
        m = {k: np.full((GRID_N, GRID_N), np.nan) for k in ("strict", "class3", "benchmark", "grasp")}
        for r in rows:
            x, y = (round(v, 4) for v in r["pose"][COL[g]])
            a = (iy[y], ix[x])
            m["strict"][a] = not r["strict_success"]
            m["class3"][a] = r["failure_class"] == "3"
            m["benchmark"][a] = not r["benchmark_success"]
            m["grasp"][a] = r["failure_class"] in ("0", "1", "2")
        assert not any(np.isnan(v).any() for v in m.values())
        out[g] = m
    return out


def shift_p(eff, fmap):
    """Share of the 1,680 toroidal shifts of the failure map whose rho is at least the observed one (one-sided, spatial null)."""
    from scipy.stats import rankdata

    e = rankdata(eff.ravel()) - (GRID_N * GRID_N + 1) / 2
    r = rankdata(fmap.ravel()).reshape(GRID_N, GRID_N) - (GRID_N * GRID_N + 1) / 2
    obs = float((e * r.ravel()).sum())
    n = sum(float((e * np.roll(r, (a, b), (0, 1)).ravel()).sum()) >= obs for a in range(GRID_N) for b in range(GRID_N) if (a, b) != (0, 0))
    return (n + 1) / (GRID_N * GRID_N)


def build(task):
    from .edits import BLOCKS, N_HEADS
    from .sweep import SHORT

    z = {k: v for k, v in np.load(exp(task) / "stage2" / "atlas" / "raw.npz").items()}
    ix = {n: i for i, n in enumerate(z["names"])}
    step, grid, index, eff = z["step"], z["grid"], z["index"], z["eff"].astype(np.float64)
    fm = failure_maps(task)
    sm = {g: {k: smooth(v) for k, v in fm[g].items()} for g in GRIDS}
    sel = {(g, s): np.nonzero((step == s) & (grid == g))[0] for g in GRIDS for s in STEPS}
    for g in GRIDS:
        for s in FULL_STEPS:
            assert list(index[sel[(g, s)]]) == list(range(GRID_N * GRID_N))
    kinds = {b: ("encoder self-attention" if b.startswith("model.encoder") else "decoder cross-attention" if b.endswith("multihead_attn") else "decoder self-attention") for b in BLOCKS}
    base_var = z["var_sum"][-1]  # per grid
    KG = {(g, s): gi * len(STEPS) + si for gi, g in enumerate(GRIDS) for si, s in enumerate(STEPS)}
    maps, heads = {}, []

    def mean2(d):
        return None if any(v is None for v in d.values()) else round(float(np.mean(list(d.values()))), 4)

    for b in BLOCKS:
        for h in range(N_HEADS):
            hid = f"{SHORT[b]}.h{h}"
            e = {"id": hid, "block": b, "head": h, "kind": kinds[b], "effect_rms": {}, "object_signal_drop": {}, "overlap": {}, "joints": {}, "time_profile": {}}
            for v in VAR:
                i = ix[f"{hid}.{v}"]
                e["effect_rms"][v] = {g: {str(s): round(float(eff[i, sel[(g, s)]].mean()), 5) for s in STEPS} for g in GRIDS}
                e["object_signal_drop"][v] = {g: round(1 - float(z["var_sum"][i, gi]) / float(base_var[gi]), 4) for gi, g in enumerate(GRIDS)}
                ov = {}
                for s in FULL_STEPS:
                    for g in GRIDS:
                        maps[f"{hid}.{v}.{g}.step{s}"] = eff[i, sel[(g, s)]].reshape(GRID_N, GRID_N).astype(np.float16)
                    for k in ("strict", "class3", "benchmark", "grasp"):
                        ov[f"step{s}_{k}"] = {g: r4(rho(eff[i, sel[(g, s)]].reshape(GRID_N, GRID_N), sm[g][k])) for g in GRIDS}
                e["overlap"][v] = {"step0": mean2(ov["step0_strict"]), **{k: {**d, "mean": mean2(d)} for k, d in ov.items()}}
                e["joints"][v] = {str(s): {JOINTS[j]: round(float(m[j]), 4) for j in np.argsort(-m)[:3]} for s in STEPS for m in [(z["joint"][i, KG[("peg", s)]] + z["joint"][i, KG[("sock", s)]]) / 2]}
                e["time_profile"][v] = {str(s): [round(float(x), 4) for x in (z["timep"][i, KG[("peg", s)]] + z["timep"][i, KG[("sock", s)]]) / 2] for s in STEPS}

            def mm(g, s):
                return (maps[f"{hid}.x0.{g}.step{s}"].astype(float) + maps[f"{hid}.x2.{g}.step{s}"]) / 2

            e["overlap"]["x0_x2_mean"] = {}
            for s in FULL_STEPS:
                d = {g: rho(mm(g, s), sm[g]["strict"]) for g in GRIDS}
                e["overlap"]["x0_x2_mean"][f"step{s}"] = mean2(d)
                e["overlap"]["x0_x2_mean"][f"step{s}_by_grid"] = {g: r4(v) for g, v in d.items()}
            e["overlap"]["x0_x2_mean"]["step0_shift_p_by_grid"] = {g: round(shift_p(mm(g, 0), sm[g]["strict"]), 4) for g in GRIDS}
            if SHORT[b] == "cross":
                att = {}
                for g, c in (("peg", 0), ("sock", 1)):
                    A = [z["attn"][KG[(g, s)], h] for s in STEPS]
                    att[g] = {"mass_latent": [round(float(a[0]), 4) for a in A], "mass_state": [round(float(a[1]), 4) for a in A], "peak_cell_row_col_per_step": [[int(v) for v in np.unravel_index(int(a[2:].argmax()), (KH, KW))] for a in A]}
                    for s in FULL_STEPS:
                        o = sel[(g, s)]
                        att[g][f"mass_on_{g}_cells_step{s}"] = round(float(z["attn_cell"][o, h, c].mean()), 4)
                        att[g][f"uniform_mass_on_{g}_cells_step{s}"] = round(float(z["on_cells"][o, c].mean() / 302), 4)
                    o = sel[(g, 0)]
                    cen, ac = z["cen"][o, c] / 32 - 0.5, z["attn_cen"][o, h]
                    k = ~np.isnan(cen).any(1)
                    att[g]["centroid_tracks_object_corr_xy_step0"] = [round(float(np.corrcoef(cen[k, d], ac[k, d])[0, 1]), 4) for d in (0, 1)]
                e["attention"] = att
            heads.append(e)

    score = lambda e, v: -9.0 if e["overlap"][v]["step0"] is None else e["overlap"][v]["step0"]
    top = sorted(heads, key=lambda e: -score(e, "x0_x2_mean"))
    picks = [f"{e['id']}.{max(('x0', 'x2'), key=lambda v: score(e, v))}" for e in top[:5]]
    lat = {n: {g: {str(s): round(float(eff[ix[n], sel[(g, s)]].mean()), 6) for s in STEPS} for g in GRIDS} for n in z["names"] if n.startswith("lat")}
    meta = {"heads": heads, "top5_by_overlap_edits": picks, "latent_axes_effect_rms": lat, "step0_variance_original": {g: float(base_var[gi]) for gi, g in enumerate(GRIDS)},
            "failure_rate_grid": {g: {k: round(float(v.mean()), 4) for k, v in fm[g].items()} for g in GRIDS},
            "recorded_action_check_max_abs_by_step": {str(s): round(float(z["recheck"][z["recheck"][:, 0] == s, 1].max()), 5) for s in STEPS},
            "definitions": "awe/atlas_ins.py docstring"}
    return meta, maps, sm, z, ix


def figures(task, meta, maps, sm, z, ix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = exp(task) / "stage2" / "atlas"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a8a"})
    heads = meta["heads"]
    ids = [e["id"] for e in heads]
    xs = {g: np.linspace(EXT[g][0], EXT[g][1], GRID_N) for g in GRIDS}
    ys = np.linspace(40, 60, GRID_N)
    for g in GRIDS:
        fs = sm[g]["strict"]
        lev = float(np.quantile(fs, 0.85))
        for s in (0, 200):
            fig, ax = plt.subplots(6, 8, figsize=(10, 13), sharex=True, sharey=True)
            for a, e in zip(ax.flat, heads):
                m = maps[f"{e['id']}.x0.{g}.step{s}"].astype(float)
                a.imshow(m, origin="lower", extent=EXT[g], cmap="Blues", aspect="auto")
                a.contour(xs[g], ys, fs, levels=[lev], colors="#b0413e", linewidths=0.9)
                a.set_title(f"{e['id']}  max {m.max():.2g}", fontsize=7, color="#333333")
            fig.suptitle(f"Removing each head (x0): effect on the chunk decided at step {s}, over the {TITLE[g]} start (cm), own scale per panel\nRed line: top 15% of the smoothed strict-failure rate ({lev:.2f})", fontsize=10)
            fig.tight_layout(rect=(0.01, 0.01, 1, 0.96))
            fig.savefig(d / f"effect_maps_x0_{g}_step{s}.png", dpi=90)
            plt.close(fig)

    r = np.array([np.nan if e["overlap"]["x0_x2_mean"]["step0"] is None else e["overlap"]["x0_x2_mean"]["step0"] for e in heads])
    o = np.argsort(np.nan_to_num(r, nan=-9))
    fig, ax = plt.subplots(1, 3, figsize=(15, 8), gridspec_kw={"width_ratios": [0.6, 0.6, 1.2]})
    for a, g in zip(ax, GRIDS):
        im = a.imshow(sm[g]["strict"], origin="lower", extent=EXT[g], cmap="Blues", aspect="auto")
        a.set_title(f"Smoothed strict-failure rate\n{TITLE[g]} grid, default policy")
        a.set_xlabel(f"{TITLE[g]} x (cm)")
        a.set_ylabel(f"{TITLE[g]} y (cm)")
        fig.colorbar(im, ax=a, fraction=0.046)
    by = np.array([[np.nan if e["overlap"]["x0_x2_mean"]["step0_by_grid"][g] is None else e["overlap"]["x0_x2_mean"]["step0_by_grid"][g] for g in GRIDS] for e in heads], float)
    ax[2].hlines(range(48), np.nanmin(by, 1)[o], np.nanmax(by, 1)[o], color="#c9d3df", lw=2)
    for c, g, col in ((0, "peg", "#2f5d8c"), (1, "sock", "#8c6d2f")):
        ax[2].plot(by[o, c], range(48), "o", ms=5, color=col, label=f"{TITLE[g]} grid")
    ax[2].plot(r[o], range(48), "|", ms=12, color="black", label="mean (the score)")
    ax[2].legend(frameon=False, loc="lower right")
    ax[2].set_yticks(range(48), [ids[i] for i in o], fontsize=6)
    ax[2].axvline(0, color="#8a8a8a", lw=0.8)
    ax[2].set_xlabel("Spearman correlation of the effect map (mean of x0 and x2, step 0) with the failure map")
    ax[2].set_title("Overlap score per head")
    fig.tight_layout()
    fig.savefig(d / "overlap.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(3, 2, figsize=(12, 18), gridspec_kw={"width_ratios": [14, 10]})
    for row, s in enumerate((0, 100, 200)):
        ks = [gi * len(STEPS) + STEPS.index(s) for gi in (0, 1)]
        J = np.array([z["joint"][ix[f"{e['id']}.x0"], ks].mean(0) for e in heads])
        T = np.array([z["timep"][ix[f"{e['id']}.x0"], ks].mean(0) for e in heads])
        for a, M, lab in ((ax[row, 0], J, JOINTS), (ax[row, 1], T, [f"{10 * i}-{10 * i + 9}" for i in range(10)])):
            a.imshow(M / np.maximum(M.max(1, keepdims=True), 1e-9), aspect="auto", cmap="Blues", vmin=0, vmax=1)
            a.set_yticks(range(48), ids, fontsize=5.5)
            a.set_xticks(range(len(lab)), lab, rotation=60, ha="right", fontsize=7)
        ax[row, 0].set_title(f"Step {s}: which joints the x0 edit moves (each row over its own maximum)")
        ax[row, 1].set_title(f"Step {s}: which part of the chunk (steps)")
    fig.tight_layout()
    fig.savefig(d / "joint_time_profiles_x0.png", dpi=100)
    plt.close(fig)

    cols = [("peg", "Peg signal lost at x0\n(1 - variance ratio, peg grid)", lambda e: e["object_signal_drop"]["x0"]["peg"]), ("sock", "Socket signal lost at x0\n(1 - variance ratio, socket grid)", lambda e: e["object_signal_drop"]["x0"]["sock"]),
            ("size", "Mean effect size at x0, step 200\n(normalized rms, both grids)", lambda e: np.mean([e["effect_rms"]["x0"][g]["200"] for g in GRIDS]))]
    V = [np.array([fn(e) for e in heads]) for _, _, fn in cols]
    o = np.argsort(V[0] + V[1])
    fig, ax = plt.subplots(1, 3, figsize=(13, 8), sharey=True)
    for a, (_, t, _), v in zip(ax, cols, V):
        a.hlines(range(48), 0, v[o], color="#c9d3df", lw=2)
        a.plot(v[o], range(48), "o", ms=6, color="#2f5d8c")
        a.axvline(0, color="#8a8a8a", lw=0.8)
        a.set_title(t)
    ax[0].set_yticks(range(48), [ids[i] for i in o], fontsize=6)
    fig.tight_layout()
    fig.savefig(d / "object_signal.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(8, 8, figsize=(17, 14))
    for h in range(8):
        for gi, g in enumerate(GRIDS):
            for si, s in enumerate(STEPS):
                a = ax[h, gi * 4 + si]
                w = z["attn"][gi * len(STEPS) + si, h, 2:]
                a.imshow(w.reshape(KH, KW), cmap="Blues")
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(f"cross.h{h} {TITLE[g]} grid, step {s}, image mass {w.sum():.2f}", fontsize=6)
    fig.suptitle("Where the cross-attention heads look (mean over poses and chunk steps, 15 x 20 cells of the top camera, own scale)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(d / "cross_attention.png", dpi=90)
    plt.close(fig)


def report(task):
    meta, maps, sm, z, ix = build(task)
    d = exp(task) / "stage2" / "atlas"
    (d / "heads.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(d / "effect_maps.npz", **{f"{k}_{g}": sm[g][k].astype(np.float32) for g in GRIDS for k in sm[g]}, **maps)
    figures(task, meta, maps, sm, z, ix)
    print(len(meta["heads"]), "heads ->", d)
    print("top 5 by overlap:", meta["top5_by_overlap_edits"])
