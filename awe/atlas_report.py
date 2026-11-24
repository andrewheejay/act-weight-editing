"""Turn stage2/atlas/raw.npz into heads.json, effect_maps.npz and the figures. Definitions: awe/atlas.py docstring."""
import json

import numpy as np

from .atlas import JOINTS, KH, KW, SMOOTH_SIGMA_M, STEPS
from .paths import exp

GRID = 41
VAR = ("x0", "x2", "sharp2", "soft0.5")


def failure_maps(task):
    rows = [json.loads(l) for l in (exp(task) / "stage1" / "grid_default.jsonl").read_text().splitlines()]
    from .stage1_analysis import load_features

    gf = {tuple(round(v, 4) for v in r["pose"][:2]): r["grasp_fail"] for r in load_features(task, "grid")}
    xs, ys = np.linspace(0.0, 0.2, GRID), np.linspace(0.4, 0.6, GRID)
    ix = {round(v, 4): i for i, v in enumerate(xs)}
    iy = {round(v, 4): i for i, v in enumerate(ys)}
    fail, grasp = np.full((GRID, GRID), np.nan), np.full((GRID, GRID), np.nan)
    for r in rows:
        x, y = (round(v, 4) for v in r["pose"][:2])
        fail[iy[y], ix[x]] = not r["benchmark_success"]
        grasp[iy[y], ix[x]] = gf[(x, y)]
    assert not np.isnan(fail).any() and not np.isnan(grasp).any()
    return fail, grasp


def smooth(m):
    from scipy.ndimage import gaussian_filter

    s = SMOOTH_SIGMA_M / 0.005
    return gaussian_filter(m, s, mode="constant") / gaussian_filter(np.ones_like(m), s, mode="constant")


def rho(a, b):
    from scipy.stats import rankdata

    a, b = rankdata(a.ravel()), rankdata(b.ravel())
    return float(np.corrcoef(a, b)[0, 1])


def shift_p(eff, fmap):
    """Share of the 1,680 toroidal shifts of the failure map whose rho is at least the observed one (one-sided, spatial null)."""
    from scipy.stats import rankdata

    e = rankdata(eff.ravel()) - (GRID * GRID + 1) / 2
    r = rankdata(fmap.ravel()).reshape(GRID, GRID) - (GRID * GRID + 1) / 2
    obs = float((e * r.ravel()).sum())
    n = sum(float((e * np.roll(r, (a, b), (0, 1)).ravel()).sum()) >= obs for a in range(GRID) for b in range(GRID) if (a, b) != (0, 0))
    return (n + 1) / (GRID * GRID)


def load(task):
    z = np.load(exp(task) / "stage2" / "atlas" / "raw.npz")
    return z, {n: i for i, n in enumerate(z["names"])}


def build(task):
    from .edits import BLOCKS, N_HEADS
    from .sweep import SHORT

    z, ix = load(task)
    step = z["step"]
    eff = z["eff"].astype(np.float64)
    fail, grasp = failure_maps(task)
    fs, gs = smooth(fail), smooth(grasp)
    order0 = np.argsort(z["index"][step == 0])
    order100 = np.argsort(z["index"][step == 100])
    maps = {}
    heads = []
    kinds = {b: ("encoder self-attention" if b.startswith("model.encoder") else "decoder cross-attention" if b.endswith("multihead_attn") else "decoder self-attention") for b in BLOCKS}
    base_var = float(z["var_sum"][-1])
    for b in BLOCKS:
        for h in range(N_HEADS):
            hid = f"{SHORT[b]}.h{h}"
            e = {"id": hid, "block": b, "head": h, "kind": kinds[b], "effect_rms": {}, "cube_signal_drop": {}, "overlap": {}, "joints": {}, "time_profile": {}}
            for v in VAR:
                i = ix[f"{hid}.{v}"]
                e["effect_rms"][v] = {str(s): round(float(eff[i, step == s].mean()), 5) for s in STEPS}
                e["cube_signal_drop"][v] = round(1 - float(z["var_sum"][i]) / base_var, 4)
                m0 = eff[i, step == 0][order0].reshape(GRID, GRID)
                m100 = eff[i, step == 100][order100].reshape(GRID, GRID)
                maps[f"{hid}.{v}.step0"], maps[f"{hid}.{v}.step100"] = m0.astype(np.float16), m100.astype(np.float16)
                e["overlap"][v] = {"step0": round(rho(m0, fs), 4), "step100": round(rho(m100, fs), 4), "step0_grasp": round(rho(m0, gs), 4), "step100_grasp": round(rho(m100, gs), 4)}
                e["joints"][v] = {str(s): {JOINTS[j]: round(float(z["joint"][i, k, j]), 4) for j in np.argsort(-z["joint"][i, k])[:3]} for k, s in enumerate(STEPS)}
                e["time_profile"][v] = {str(s): [round(float(x), 4) for x in z["timep"][i, k]] for k, s in enumerate(STEPS)}
            mm = lambda s: (maps[f"{hid}.x0.{s}"].astype(float) + maps[f"{hid}.x2.{s}"]) / 2
            e["overlap"]["x0_x2_mean"] = {"step0": round(rho(mm("step0"), fs), 4), "step0_shift_p": round(shift_p(mm("step0"), fs), 4), "step100": round(rho(mm("step100"), fs), 4)}
            if SHORT[b] == "cross":
                A = z["attn"][:, h]  # (4, 302)
                ok = step == 0
                cen, ac = z["cen"][ok] / 32 - 0.5, z["attn_cen"][ok, h]
                g = ~np.isnan(cen).any(1)
                e["attention"] = {
                    "mass_latent": [round(float(v), 4) for v in A[:, 0]], "mass_state": [round(float(v), 4) for v in A[:, 1]],
                    "peak_cell_row_col_per_step": [[int(v) for v in np.unravel_index(int(A[k, 2:].argmax()), (KH, KW))] for k in range(4)],
                    "mass_on_red_cells_step0": round(float(z["attn_cell"][ok, h].mean()), 4),
                    "uniform_mass_on_red_cells_step0": round(float(z["red_cells"][ok].mean() / 302), 4),
                    "mass_on_red_cells_step100": round(float(z["attn_cell"][step == 100, h].mean()), 4),
                    "uniform_mass_on_red_cells_step100": round(float(z["red_cells"][step == 100].mean() / 302), 4),
                    "centroid_tracks_cube_corr_xy_step0": [round(float(np.corrcoef(cen[g, d], ac[g, d])[0, 1]), 4) for d in (0, 1)],
                }
            heads.append(e)
    top = sorted(heads, key=lambda e: -e["overlap"]["x0_x2_mean"]["step0"])
    picks = [f"{e['id']}.{max(('x0', 'x2'), key=lambda v: e['overlap'][v]['step0'])}" for e in top[:5]]
    lat = {n: {str(s): round(float(eff[ix[n], step == s].mean()), 6) for s in STEPS} for n in z["names"] if n.startswith("lat")}
    meta = {"heads": heads, "top5_by_overlap_edits": picks, "latent_axes_effect_rms": lat, "step0_variance_original": base_var,
            "failure_rate_grid": round(float(fail.mean()), 4), "grasp_fail_rate_grid": round(float(grasp.mean()), 4),
            "recorded_action_check_max_abs_by_step": {str(int(s)): round(float(z["recheck"][z["recheck"][:, 0] == s, 1].max()), 5) for s in STEPS},
            "definitions": "awe/atlas.py docstring"}
    return meta, maps, fs, gs


def figures(task, meta, maps, fs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = exp(task) / "stage2" / "atlas"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a8a"})
    heads = meta["heads"]
    ext = (0, 20, 40, 60)

    fig, ax = plt.subplots(6, 8, figsize=(14, 11), sharex=True, sharey=True)
    for a, e in zip(ax.flat, heads):
        m = maps[f"{e['id']}.x0.step0"].astype(float)
        a.imshow(m, origin="lower", extent=ext, cmap="Blues")
        a.contour(np.linspace(0, 20, GRID), np.linspace(40, 60, GRID), fs, levels=[0.35], colors="#b0413e", linewidths=0.9)
        a.set_title(f"{e['id']}  max {m.max():.2f}", fontsize=8, color="#333333")
    fig.suptitle("Effect size of removing each head (x0) on the first action chunk, over the cube start (x, y in cm). Own scale per panel. Red line: smoothed failure rate 0.35", fontsize=10)
    fig.supxlabel("cube x (cm)", fontsize=9)
    fig.supylabel("cube y (cm)", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    fig.savefig(d / "effect_maps_x0_step0.png", dpi=110)
    plt.close(fig)

    ids = [e["id"] for e in heads]
    r = np.array([e["overlap"]["x0_x2_mean"]["step0"] for e in heads])
    o = np.argsort(r)
    fig, ax = plt.subplots(1, 2, figsize=(11, 8), gridspec_kw={"width_ratios": [1, 1.1]})
    im = ax[0].imshow(fs, origin="lower", extent=ext, cmap="Blues")
    ax[0].set_title("Smoothed benchmark failure rate, default policy")
    ax[0].set_xlabel("cube x (cm)")
    ax[0].set_ylabel("cube y (cm)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].hlines(range(48), 0, r[o], color="#c9d3df", lw=2)
    ax[1].plot(r[o], range(48), "o", ms=6, color="#2f5d8c")
    ax[1].set_yticks(range(48), [ids[i] for i in o], fontsize=6)
    ax[1].axvline(0, color="#8a8a8a", lw=0.8)
    ax[1].set_xlabel("Spearman correlation of the effect map with the failure map (mean of x0 and x2, step 0)")
    ax[1].set_title("Overlap score per head")
    fig.tight_layout()
    fig.savefig(d / "overlap.png", dpi=120)
    plt.close(fig)

    z, ix = load(task)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={"width_ratios": [14, 10]})
    for row, (k, s) in enumerate([(0, 0), (1, 100)]):
        J = np.array([z["joint"][ix[f"{e['id']}.x0"], k] for e in heads])
        T = np.array([z["timep"][ix[f"{e['id']}.x0"], k] for e in heads])
        for a, M, lab in ((ax[row, 0], J, JOINTS), (ax[row, 1], T, [f"{10 * i}-{10 * i + 9}" for i in range(10)])):
            im = a.imshow(M / np.maximum(M.max(1, keepdims=True), 1e-9), aspect="auto", cmap="Blues", vmin=0, vmax=1)
            a.set_yticks(range(48), ids, fontsize=5.5)
            a.set_xticks(range(len(lab)), lab, rotation=60, ha="right", fontsize=7)
        ax[row, 0].set_title(f"Step {s}: which joints the x0 edit moves (each row over its own maximum)")
        ax[row, 1].set_title(f"Step {s}: which part of the chunk (steps)")
    fig.tight_layout()
    fig.savefig(d / "joint_time_profiles_x0.png", dpi=110)
    plt.close(fig)

    sig = np.array([e["cube_signal_drop"]["x0"] for e in heads])
    size = np.array([e["effect_rms"]["x0"]["0"] for e in heads])
    o = np.argsort(sig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    for a, v, t in ((ax[0], sig, "Cube-position signal lost at x0\n(1 - variance ratio of the first chunk, can be negative)"), (ax[1], size, "Mean effect size at x0, step 0 (normalized rms)")):
        a.hlines(range(48), 0, v[o], color="#c9d3df", lw=2)
        a.plot(v[o], range(48), "o", ms=6, color="#2f5d8c")
        a.axvline(0, color="#8a8a8a", lw=0.8)
        a.set_title(t)
    ax[0].set_yticks(range(48), [ids[i] for i in o], fontsize=6)
    fig.tight_layout()
    fig.savefig(d / "cube_signal.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(8, 4, figsize=(10, 14))
    for h in range(8):
        for k, s in enumerate(STEPS):
            a = ax[h, k]
            a.imshow(z["attn"][k, h, 2:].reshape(KH, KW), cmap="Blues")
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"cross.h{h} step {s}  image mass {z['attn'][k, h, 2:].sum():.2f}", fontsize=7)
    fig.suptitle("Where the cross-attention heads look (mean over poses and chunk steps, 15 x 20 cells of the top camera, own scale)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(d / "cross_attention.png", dpi=110)
    plt.close(fig)


def report(task):
    meta, maps, fs, gs = build(task)
    d = exp(task) / "stage2" / "atlas"
    (d / "heads.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(d / "effect_maps.npz", failure_smoothed=fs.astype(np.float32), grasp_fail_smoothed=gs.astype(np.float32), **maps)
    figures(task, meta, maps, fs)
    print(len(meta["heads"]), "heads ->", d)
    print("top 5 by overlap:", meta["top5_by_overlap_edits"])
