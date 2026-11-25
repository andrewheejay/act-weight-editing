"""Descriptive tables the frozen freeze/analysis.py does not make. Reads test/arms/*.jsonl only, runs no rollout, decides no verdict. Written after the test was read.

python experiments/transfer_cube/report/extra_tables.py        -> report/extra.json, report/extra_tables.md, report/random_by_head.png, report/fixed_broken_by_class.png

Paired differences use referee.stats (exact McNemar, 10,000 bootstrap resamples, seed 0), as in freeze/analysis.py.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from referee.stats import bootstrap_ci, paired  # noqa: E402

T = ROOT / "experiments" / "transfer_cube"
OUT = T / "report"
BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8985", "#fcfcfb"
man = {a["id"]: a for a in json.loads((T / "freeze" / "test_manifest.json").read_text())["arms"]}
rows = {a: {r["seed"]: r for r in map(json.loads, open(T / "test" / "arms" / f"{a}.jsonl"))} for a in man}


def col(a, k="benchmark_success"):
    return {s: bool(r[k]) for s, r in rows[a].items()}


def versus(a, b, k="benchmark_success"):
    """arm b against arm a, same seeds: gain of b over a"""
    p, c = paired(col(a, k), col(b, k)), bootstrap_ci(col(a, k), col(b, k), 10_000, 0)
    return {"diff_points": p["gain_points"], "ci95": c["gain_points"], "only_edit": p["fixed"], "only_other": p["broke"], "p": p["p_mcnemar"]}


out = {"note": "descriptive, computed after the test was read"}
others = ["base_deviation_gain", "base_simplest_fix", "base_constant_offset", "base_shorter_chunk", "original_ens", "autogreedy", "random_00", "dose_k0.5", "dose_k1.5"]
out["edit_vs"] = {o: {"benchmark": versus(o, "agent_edit"), "strict": versus(o, "agent_edit", "strict_success")} for o in others}
out["ens_edit_vs_ens_original"] = versus("original_ens", "agent_edit_ens")

# class of each seed under the original and under the edit, fixed and broken by class
o, e = rows["original"], rows["agent_edit"]
seeds = sorted(o)
out["transitions"] = {f"{a}->{b}": n for (a, b), n in sorted(Counter((o[s]["failure_class"], e[s]["failure_class"]) for s in seeds).items())}
out["fixed_by_original_class"] = dict(sorted(Counter(o[s]["failure_class"] for s in seeds if not o[s]["benchmark_success"] and e[s]["benchmark_success"]).items()))
out["original_failures_by_class"] = dict(sorted(Counter(o[s]["failure_class"] for s in seeds if not o[s]["benchmark_success"]).items()))
out["broken_by_edit_class"] = dict(sorted(Counter(e[s]["failure_class"] for s in seeds if o[s]["benchmark_success"] and not e[s]["benchmark_success"]).items()))
out["strict"] = {"fixed": sum(e[s]["strict_success"] and not o[s]["strict_success"] for s in seeds), "broke": sum(o[s]["strict_success"] and not e[s]["strict_success"] for s in seeds)}

# the random family as a table, encoder head x2 by cross head x0.5
g = {a: paired(col("original"), col(a))["gain_points"] for a in man if a.startswith("random_")}
key = {a: (man[a]["ops"][0]["head"], man[a]["ops"][1]["head"]) for a in g}
grid = {key[a]: v for a, v in g.items()}
out["random_grid"] = {f"enc3.h{i}.x2+cross.h{j}.x0.5": v for (i, j), v in sorted(grid.items())}
out["random_mean_by_cross_head"] = {f"cross.h{j}": {"n": len(x), "mean": round(float(np.mean(x)), 2), "min": min(x), "max": max(x)} for j in range(8) if (x := [v for (a, b), v in grid.items() if b == j])}
out["random_mean_by_enc3_head"] = {f"enc3.h{i}": {"n": len(x), "mean": round(float(np.mean(x)), 2), "min": min(x), "max": max(x)} for i in range(8) if (x := [v for (a, b), v in grid.items() if a == i])}
a = np.array(list(g.values()))
out["random_summary"] = {"n": len(a), "mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2), "n_positive": int((a > 0).sum()), "n_p_below_0.05_positive": sum(paired(col("original"), col(x))["p_mcnemar"] < 0.05 and v > 0 for x, v in g.items()), "max": float(a.max()), "min": float(a.min()),
                         "not_drawn_of_49": [f"enc3.h{i}+cross.h{j}" for i in range(8) for j in range(8) if i != 4 and j != 1 and (i, j) not in grid]}
gs = {x: paired(col("original", "strict_success"), col(x, "strict_success"))["gain_points"] for x in g}
eg = paired(col("original", "strict_success"), col("agent_edit", "strict_success"))["gain_points"]
out["random_strict"] = {"edit_strict_gain": eg, "n_gaining_as_much_strict": sum(v >= eg - 1e-12 for v in gs.values()), "best_strict": max(gs.values()), "best_arm": max(gs, key=gs.get)}
(OUT / "extra.json").write_text(json.dumps(out, indent=1) + "\n")

L = ["# Extra descriptive tables (test data)", "", "The edit against other arms on the same seeds. Positive means the edit did better.", "",
     "| other arm | difference (points) | 95% interval | only the edit succeeds | only the other succeeds | p | strict difference (points) |", "|---|---|---|---|---|---|---|"]
for k, v in out["edit_vs"].items():
    b = v["benchmark"]
    L.append(f"| {k} | {b['diff_points']:+.1f} | {b['ci95'][0]:+.1f} to {b['ci95'][1]:+.1f} | {b['only_edit']} | {b['only_other']} | {b['p']:.2g} | {v['strict']['diff_points']:+.1f} |")
L += ["", "Random edits, gain in points by heads:", "", "| enc3 head x2 | " + " | ".join(f"cross.h{j} x0.5" for j in range(8) if j != 1) + " |", "|---|" + "---|" * 7]
for i in range(8):
    if i != 4:
        L.append(f"| enc3.h{i} | " + " | ".join(f"{grid[(i, j)]:+.1f}" if (i, j) in grid else "not drawn" for j in range(8) if j != 1) + " |")
(OUT / "extra_tables.md").write_text("\n".join(L) + "\n")


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors="#52514e")


# fixed and broken by class. Class 4a and 4b are benchmark successes, so they hold no failures
cls = ["0", "1", "2"]
fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURF)
for x, d, c, lab in ((-0.27, out["original_failures_by_class"], "#c9c8c3", "original failures of this class"), (0.0, out["fixed_by_original_class"], AQUA, "of those, fixed by the edit"), (0.27, out["broken_by_edit_class"], ORANGE, "original successes broken into this class")):
    ax.bar_label(ax.bar(np.arange(3) + x, [d.get(k, 0) for k in cls], 0.25, color=c, label=lab), fontsize=8, color=INK)
ax.set_xticks(range(3), ["0, never touched", "1, touched, no lift", "2, lifted, no left touch in the air"])
ax.set_ylabel("test seeds", fontsize=9)
ax.set_title("Fixed and broken by failure class, 1,000 test seeds", fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left")
style(ax)
fig.savefig(OUT / "fixed_broken_by_class.png", dpi=140, bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# random edits by cross head
fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURF)
for a_, v in g.items():
    i, j = key[a_]
    ax.scatter(j, v, s=34, color=MUTED, edgecolors=SURF, linewidths=1, zorder=2)
ax.scatter([1], [8.4], s=90, color=BLUE, edgecolors=SURF, linewidths=1.5, zorder=3, label="agent's edit (cross.h1)")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(8), [f"cross.h{j}" for j in range(8)])
ax.set_xlabel("cross-attention head halved (with one other encoder layer 3 head doubled)", fontsize=9)
ax.set_ylabel("gain over the original (points)", fontsize=9)
ax.set_title("40 matched random edits, grouped by the cross-attention head", fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)
fig.savefig(OUT / "random_by_head.png", dpi=140, bbox_inches="tight", facecolor=SURF)
print(json.dumps(out, indent=1))
