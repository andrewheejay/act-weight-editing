"""Failure map redrawn from the hub's per-seed file. Re-derives the numbers quoted in SPEC.md, What we already know."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compare_renderers import published
from .paths import exp


def cube_xy(seed):
    return np.random.RandomState(seed).uniform([0.0, 0.4, 0.05], [0.2, 0.6, 0.05])[:2]  # gym_aloha.utils.sample_box_pose


def main():
    pub = published()
    seeds = sorted(pub)
    xy = np.array([cube_xy(s) for s in seeds])
    ok = np.array([pub[s][0] for s in seeds])
    mr = np.array([pub[s][1] for s in seeds])
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(1)
    low = xy[:, 1] < 0.44
    stats = {
        "n": len(seeds),
        "success": float(ok.mean()),
        "failures": int((~ok).sum()),
        "max_reward_counts": {str(int(k)): int((mr == k).sum()) for k in np.unique(mr)},
        "nearest_seed_also_fails_given_fail": float((~ok[nn])[~ok].mean()),
        "low_y_strip_below_0.44": {"n": int(low.sum()), "fail": int((~ok & low).sum()), "n_rest": int((~low).sum()), "fail_rest": int((~ok & ~low).sum())},
    }
    fig, ax = plt.subplots(figsize=(6.4, 6.0), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.scatter(*xy[ok].T, s=14, c="#c9c8c3", marker="o", label=f"success ({ok.sum()})", zorder=2)
    for k, c, m, lab in ((0, "#1baf7a", "^", "never touched"), (1, "#2a78d6", "s", "touched, not lifted"), (2, "#eb6834", "D", "lifted, no handover")):
        sel = ~ok & (mr == k)
        ax.scatter(*xy[sel].T, s=44, c=c, marker=m, edgecolors="#fcfcfb", linewidths=1.2, label=f"{lab} ({sel.sum()})", zorder=3)
    ax.axhline(0.44, color="#8a8985", lw=1, ls=":")
    ax.set_xlabel("cube start x (m)"), ax.set_ylabel("cube start y (m)")
    ax.set_title(f"Published evaluation, 500 seeds, {100 * ok.mean():.1f}% success", fontsize=11, color="#0b0b0b", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
    s0 = exp("transfer_cube") / "stage0"
    fig.savefig(s0 / "published_failure_map.png", dpi=150, bbox_inches="tight")
    (s0 / "published_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
