"""Stage 1 figures for insertion. python experiments/insertion/stage1/figures_ins.py  -> failure_map.png, reach_error.png (standalone, reads the committed features)"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_ins import load  # noqa: E402
from awe.stage1_analysis import auc  # noqa: E402

S1 = Path(__file__).resolve().parent
# slots 1 to 3 of the reference categorical order, plus gray for the grasp classes. Marker shape is a second code.
STY = {"S": ("#2a78d6", "o", "strict success"), "A": ("#1baf7a", "^", "class 4a, pin touched, not strict"), "M": ("#eb6834", "D", "class 3, met, no pin"), "G": ("#8a8985", "x", "classes 0 to 2, no peg-socket contact")}
GRP = {"S": "S", "A": "A", "M": "M", "G2": "G", "G1": "G", "G0": "G"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED, "text.color": INK})
F = load("default")
DEMO = json.load(open(S1 / "demo_objects.json"))["demos"]
BOX = {"peg": (0.1, 0.2, 0.4, 0.6), "sock": (-0.2, -0.1, 0.4, 0.6)}


def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)


def kinds(R):
    return np.array([GRP[r["kind"]] for r in R])


# ---- failure map ----
fig, axs = plt.subplots(2, 3, figsize=(14.5, 9.2))
for j, obj in enumerate(("peg", "sock")):
    ax = axs[j, 0]
    x = np.array([r[obj + "_x0"] for r in F]); y = np.array([r[obj + "_y0"] for r in F]); k = kinds(F)
    for c in ("G", "M", "A", "S"):
        col, m, lab = STY[c]
        ax.scatter(x[k == c], y[k == c], s=14 if m != "x" else 16, c=col, marker=m, linewidths=0.9 if m == "x" else 0, alpha=0.75, label=f"{lab} ({int((k == c).sum())})")
    ax.scatter([d[obj + "_x"] for d in DEMO], [d[obj + "_y"] for d in DEMO], s=26, facecolors="none", edgecolors=INK, linewidths=1.0, label="50 training demos, first frame")
    style(ax, f"Design seeds by {'peg' if obj == 'peg' else 'socket'} start (n {len(F)})", "x (m)", "y (m)")
    ax.set_xlim(BOX[obj][0] - 0.005, BOX[obj][1] + 0.005); ax.set_ylim(BOX[obj][2] - 0.01, BOX[obj][3] + 0.01)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
for j, obj in enumerate(("peg", "sock")):
    G = load("grid_" + obj)
    ax = axs[j, 1]
    x = np.array([r[obj + "_x0"] for r in G]); y = np.array([r[obj + "_y0"] for r in G]); k = kinds(G)
    ux, uy = np.unique(np.round(x, 5)), np.unique(np.round(y, 5))
    code = {"S": 0, "A": 1, "M": 2, "G": 3}
    im = np.full((len(uy), len(ux)), np.nan)
    for xi, yi, ki in zip(np.round(x, 5), np.round(y, 5), k):
        im[np.searchsorted(uy, yi), np.searchsorted(ux, xi)] = code[ki]
    ax.imshow(im, origin="lower", extent=(ux[0], ux[-1], uy[0], uy[-1]), aspect="auto", interpolation="nearest", cmap=matplotlib.colors.ListedColormap([STY[c][0] for c in "SAMG"]), vmin=-0.5, vmax=3.5)
    ax.scatter([d[obj + "_x"] for d in DEMO], [d[obj + "_y"] for d in DEMO], s=26, facecolors="none", edgecolors="white", linewidths=1.2)
    style(ax, f"41 x 41 grid: {'peg' if obj == 'peg' else 'socket'} moved, other at box centre", "x (m)", "y (m)")
    ax.text(0.0, -0.30, f"n {len(G)}: strict {int((k=='S').sum())}, 4a {int((k=='A').sum())}, class 3 {int((k=='M').sum())}, classes 0-2 {int((k=='G').sum())}. White rings: demos", transform=ax.transAxes, fontsize=7.5, color=MUTED)
for j, (obj, name) in enumerate((("peg", "grid_peg"), ("sock", "grid_sock"))):
    ax = axs[j, 2]
    A = np.array(json.load(open(S1 / "ins_pattern.json"))["cells"]["grid_peg_by_peg_start" if obj == "peg" else "grid_sock_by_socket_start"]["met_but_no_pin"])
    nb = 5
    im = ax.imshow(A, origin="lower", extent=(BOX[obj][0], BOX[obj][1], BOX[obj][2], BOX[obj][3]), aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    for i in range(nb):
        for j2 in range(nb):
            ax.text(BOX[obj][0] + 0.1 * (i + 0.5) / nb, BOX[obj][2] + 0.2 * (j2 + 0.5) / nb, f"{100 * A[j2, i]:.0f}", ha="center", va="center", fontsize=8, color="white" if A[j2, i] > 0.55 else INK)
    style(ax, f"Class 3 share, 5 x 5 cells (%), {'peg' if obj == 'peg' else 'socket'} moved", "x (m)", "y (m)")
fig.suptitle("Insertion, original policy: where the policy fails. Design seeds 20000 to 20999 and the two grids, same policy (not independent samples)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.subplots_adjust(hspace=0.62, wspace=0.22)
fig.savefig(S1 / "failure_map.png", dpi=150); plt.close(fig)

# ---- reach error and misalignment ----
fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.4))
ref = {}
for side, obj in (("r", "peg"), ("l", "sock")):
    S = np.array([r["reach_rel_" + side] for r in F if r["kind"] == "S" and r.get("reach_rel_" + side) is not None]); ref[side] = S.mean(0)
for j, (side, obj, lab) in enumerate((("r", "peg", "right gripper to peg"), ("l", "sock", "left gripper to socket"))):
    ax = axs[0, j]
    R = [r for r in F if r.get("reach_rel_" + side) is not None]
    e = np.array([np.linalg.norm(np.array(r["reach_rel_" + side]) - ref[side]) for r in R]) * 100
    dist = np.array([np.hypot((r[obj + "_x0"] - {"peg": 0.15, "sock": -0.15}[obj]) / 0.05, (r[obj + "_y0"] - 0.5) / 0.10) for r in R])
    k = kinds(R)
    for c in ("G", "M", "A", "S"):
        col, m, labc = STY[c]
        ax.scatter(dist[k == c], e[k == c], s=12, c=col, marker=m, linewidths=0.8 if m == "x" else 0, alpha=0.6, label=f"{labc} ({int((k == c).sum())})")
    ed = np.quantile(dist, [0, .2, .4, .6, .8, 1]); b = np.clip(np.searchsorted(ed, dist, side="right") - 1, 0, 4)
    ax.plot([ed[i:i + 2].mean() for i in range(5)], [np.median(e[b == i]) for i in range(5)], color=INK, lw=2, label="median by fifth, all")
    ax.set_ylim(0, 12); style(ax, f"Reach error: {lab}\n(first close, minus mean over strict successes)", "distance of the object's start from its box centre (box half-widths)", "cm")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
ax = axs[0, 2]
G = [r for r in F if r.get("pre_tip_lat") is not None and r["kind"] in "SAM"]
kk = np.array([r["kind"] for r in G]); v = np.array([r["pre_tip_lat"] for r in G]) * 100
for c in "SAM":
    s = np.sort(v[kk == c]); ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=STY[c][0], lw=2, label=f"{STY[c][2]} ({len(s)})")
ax.axvline(0.8, color=MUTED, lw=1, ls="--"); ax.text(2.6, 0.28, "dashed: 0.8 cm clearance\n(half-gap 1.8 minus peg half-width 1.0)", fontsize=7, color=MUTED)
ax.set_xlim(0, 4.5); style(ax, "Tip misalignment one step before contact (ECDF)", "cm across the socket axis (larger of two axes)", "share of episodes")
ax.legend(fontsize=7, frameon=False, loc="lower right")
for j, (key, obj, xl) in enumerate((("peg_x0", "peg", "peg start x (m)"), ("sock_y0", "sock", "socket start y (m)"))):
    ax = axs[1, j]
    x = np.array([r[key] for r in G])
    for c in ("M", "A", "S"):
        ax.scatter(x[kk == c], v[kk == c], s=12, c=STY[c][0], marker=STY[c][1], alpha=0.6, edgecolors="none", label=STY[c][2])
    ed = np.quantile(x, [0, .2, .4, .6, .8, 1]); b = np.clip(np.searchsorted(ed, x, side="right") - 1, 0, 4)
    ax.plot([ed[i:i + 2].mean() for i in range(5)], [np.median(v[b == i]) for i in range(5)], color=INK, lw=2, label="median by fifth")
    ax.axhline(0.8, color=MUTED, lw=1, ls="--"); ax.set_ylim(0, 6); style(ax, f"Tip misalignment against {xl.split(' (')[0]}", xl, "cm")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
ax = axs[1, 2]
tc = json.load(open(S1 / "ins_pattern.json"))["time_course_default_tip_lat_cm_median"]
ks = [60, 40, 20, 10, 5, 1]
for c in "SAM":
    ax.plot([-q for q in ks], [tc[c][f"t_meet-{q}"] for q in ks], color=STY[c][0], marker=STY[c][1], lw=2, ms=5, label=STY[c][2])
ax.axhline(0.8, color=MUTED, lw=1, ls="--"); ax.set_ylim(0, 6); style(ax, "Median tip misalignment, steps before contact", "control steps before first peg-socket contact (50 Hz)", "cm")
ax.legend(fontsize=7, frameon=False)
fig.suptitle("Insertion, original policy: reach error, and the peg-tip misalignment at contact (the tracking number). Colours are outcomes, not arms", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5, h_pad=2.5); fig.savefig(S1 / "reach_error.png", dpi=150)
