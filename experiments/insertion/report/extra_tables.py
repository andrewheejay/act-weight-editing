"""Descriptive tables the frozen freeze/analysis.py does not make. Reads test/arms/*.jsonl and freeze/ only, runs no rollout, decides no verdict.
Written after the test was read (2026-11-26, session 28).

python experiments/insertion/report/extra_tables.py   -> report/extra.json, report/extra_tables.md, report/fixed_broken_by_kind.png,
                                                          report/fixed_broken_map.png, report/random_by_head.png

Primary score is strict success, guard is benchmark success (as in freeze/analysis.py). Paired differences use referee.stats (exact McNemar,
10,000 bootstrap resamples, seed 0). "Kinds" are awe.sweep_ins_report.part: S strict, A class 4a, M class 3, grasp failures by the object not lifted.
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

from awe.dose_ins import timing  # noqa: E402
from awe.sweep_ins_report import part  # noqa: E402
from referee.stats import bootstrap_ci, paired  # noqa: E402

T = ROOT / "experiments" / "insertion"
OUT = T / "report"
BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8985", "#fcfcfb", "#c9c8c3"
man = {a["id"]: a for a in json.loads((T / "freeze" / "test_manifest.json").read_text())["arms"]}
rows = {a: {r["seed"]: r for r in map(json.loads, open(T / "test" / "arms" / f"{a}.jsonl"))} for a in man}
for a in rows:
    assert len(rows[a]) == 1000, a


def col(a, k):
    return {s: bool(r[k]) for s, r in rows[a].items()}


def versus(a, b, k="strict_success"):
    """arm b against arm a, same seeds: gain of b over a"""
    p, c = paired(col(a, k), col(b, k)), bootstrap_ci(col(a, k), col(b, k), 10_000, 0)
    return {"diff_points": p["gain_points"], "ci95": c["gain_points"], "only_b": p["fixed"], "only_a": p["broke"], "p": p["p_mcnemar"]}


out = {"note": "descriptive, computed after the test was read. No verdict."}
others = ["original", "autogreedy", "base_constant_offset", "base_simplest_fix", "base_deviation_gain", "base_shorter_chunk", "dose_k0.5", "original_ens", "agent_edit_ens"]
out["edit_vs"] = {o: {"strict": versus(o, "agent_edit"), "benchmark": versus(o, "agent_edit", "benchmark_success")} for o in others}
out["ens_edit_vs_ens_original"] = {"strict": versus("original_ens", "agent_edit_ens"), "benchmark": versus("original_ens", "agent_edit_ens", "benchmark_success")}

o, e = rows["original"], rows["agent_edit"]
seeds = sorted(o)
out["kind_transitions"] = {f"{a}->{b}": n for (a, b), n in sorted(Counter((part(o[s]), part(e[s])) for s in seeds).items())}
out["original_failures_by_kind"] = dict(sorted(Counter(part(o[s]) for s in seeds if not o[s]["strict_success"]).items()))
out["fixed_by_original_kind"] = dict(sorted(Counter(part(o[s]) for s in seeds if not o[s]["strict_success"] and e[s]["strict_success"]).items()))
out["broken_into_kind"] = dict(sorted(Counter(part(e[s]) for s in seeds if o[s]["strict_success"] and not e[s]["strict_success"]).items()))
out["kinds"] = {a: dict(sorted(Counter(part(rows[a][s]) for s in seeds).items())) for a in ("original", "agent_edit", "autogreedy")}
out["failure_class_counts"] = {a: dict(sorted(Counter(rows[a][s]["failure_class"] for s in seeds).items())) for a in ("original", "agent_edit")}

# timing, the measures named by the written hypothesis (the approach is slowed by the heads when they are on)
tm = {}
for a in ("original", "agent_edit", "autogreedy", "dose_k0.5", "dose_km0.5", "dose_km1", "dose_km2"):
    R = list(rows[a].values())
    t = timing(R)
    pins = [r for r in R if r["t_pin"] >= 0]
    tm[a] = {**t, "n_pin": len(pins), "n_pin_strict": sum(r["strict_success"] for r in pins)}
out["timing"] = tm

# the random family by the cross-attention head that was turned off
g = {a: paired(col("original", "strict_success"), col(a, "strict_success"))["gain_points"] for a in man if a.startswith("random_")}
key = {a: tuple(op["head"] for op in man[a]["ops"]) for a in g}  # (cross head x0, enc3 head x0, enc0 head x2)
out["random_by_cross_head"] = {f"cross.h{j}": {"n": len(x), "mean": round(float(np.mean(x)), 2), "min": min(x), "max": max(x)} for j in range(8) if (x := [v for a, v in g.items() if key[a][0] == j])}
out["random_by_enc3_head"] = {f"enc3.h{j}": {"n": len(x), "mean": round(float(np.mean(x)), 2), "min": min(x), "max": max(x)} for j in range(8) if (x := [v for a, v in g.items() if key[a][1] == j])}
out["random_by_enc0_head"] = {f"enc0.h{j}": {"n": len(x), "mean": round(float(np.mean(x)), 2), "min": min(x), "max": max(x)} for j in range(8) if (x := [v for a, v in g.items() if key[a][2] == j])}
arr = np.array(list(g.values()))
gb = {a: paired(col("original", "benchmark_success"), col(a, "benchmark_success"))["gain_points"] for a in g}
eb = paired(col("original", "benchmark_success"), col("agent_edit", "benchmark_success"))["gain_points"]
out["random_summary"] = {"n": len(arr), "mean": round(float(arr.mean()), 2), "median": round(float(np.median(arr)), 2), "n_positive": int((arr > 0).sum()), "max": float(arr.max()), "min": float(arr.min()),
                         "n_positive_p_below_0.05": sum(v > 0 and paired(col("original", "strict_success"), col(a, "strict_success"))["p_mcnemar"] < 0.05 for a, v in g.items()),
                         "benchmark": {"edit_gain": eb, "n_gaining_as_much": sum(v >= eb - 1e-12 for v in gb.values()), "best": max(gb.values()), "best_arm": max(gb, key=gb.get)}}
out["random_arms"] = {a: {"cross_x0": key[a][0], "enc3_x0": key[a][1], "enc0_x2": key[a][2], "strict_gain": g[a], "benchmark_gain": gb[a]} for a in sorted(g)}

# design -> validation -> test for the edit (strict gain in points)
ev = json.loads((T / "freeze" / "edit.json").read_text())
out["shrinkage"] = {"design": {"n": 1000, "fixed": ev["design"]["fixed"], "broken": ev["design"]["broken"], "gain_points": round((ev["design"]["fixed"] - ev["design"]["broken"]) / 10, 1)},
                    "validation": {"n": 500, "fixed": ev["validation_strict"]["fixed"], "broken": ev["validation_strict"]["broken"], "gain_points": ev["validation_strict"]["strict_gain_points"],
                                   "share_removed": ev["validation_strict"]["share_of_failures_removed"]},
                    "test": {"n": 1000, **{k: out["edit_vs"]["original"]["strict"][k] for k in ("diff_points", "ci95")}, "fixed": out["edit_vs"]["original"]["strict"]["only_b"], "broken": out["edit_vs"]["original"]["strict"]["only_a"]}}
(OUT / "extra.json").write_text(json.dumps(out, indent=1) + "\n")

L = ["# Extra descriptive tables (test data)", "", "The edit against other arms on the same 1,000 test seeds. Positive means the edit did better. Descriptive, after the test.", "",
     "| other arm | strict difference (points) | 95% interval | only the edit | only the other | p | benchmark difference (points) | 95% interval |", "|---|---|---|---|---|---|---|---|"]
for k, v in out["edit_vs"].items():
    s, b = v["strict"], v["benchmark"]
    L.append(f"| {k} | {s['diff_points']:+.1f} | {s['ci95'][0]:+.1f} to {s['ci95'][1]:+.1f} | {s['only_b']} | {s['only_a']} | {s['p']:.2g} | {b['diff_points']:+.1f} | {b['ci95'][0]:+.1f} to {b['ci95'][1]:+.1f} |")
x = out["ens_edit_vs_ens_original"]
L += ["", f"With ensembling, edit against original: strict {x['strict']['diff_points']:+.1f} ({x['strict']['ci95'][0]:+.1f} to {x['strict']['ci95'][1]:+.1f}, {x['strict']['only_b']} against {x['strict']['only_a']}, p {x['strict']['p']:.2g}), "
      f"benchmark {x['benchmark']['diff_points']:+.1f}.", "", "Timing (steps of 400; pin = first step the peg tip is in the socket, late = at or after step 375):", "",
      "| arm | median first meet | median first pin | episodes with a pin | of those strict | late pin share |", "|---|---|---|---|---|---|"]
for a, t in tm.items():
    L.append(f"| {a} | {t['median_t_meet']:.0f} | {t['median_t_pin']:.0f} | {t['n_pin']} | {t['n_pin_strict']} | {t['late_pin_share']:.3f} |")
L += ["", "Random edits by the cross-attention head turned off (strict gain, points):", "", "| head | n | mean | min | max |", "|---|---|---|---|---|"]
L += [f"| {h} | {v['n']} | {v['mean']:+.2f} | {v['min']:+.1f} | {v['max']:+.1f} |" for h, v in out["random_by_cross_head"].items()]
L += ["", "Random edits by the encoder layer 3 head turned off:", "", "| head | n | mean | min | max |", "|---|---|---|---|---|"]
L += [f"| {h} | {v['n']} | {v['mean']:+.2f} | {v['min']:+.1f} | {v['max']:+.1f} |" for h, v in out["random_by_enc3_head"].items()]
(OUT / "extra_tables.md").write_text("\n".join(L) + "\n")


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors="#52514e")


# fixed and broken by kind
K = ["A", "M", "peg_not_lifted", "socket_not_lifted", "neither_lifted", "both_lifted"]
lab = ["A, class 4a\n(touched in air)", "M, class 3\n(met, no strict)", "peg not\nlifted", "socket not\nlifted", "neither\nlifted", "both lifted,\nnever met"]
fig, ax = plt.subplots(figsize=(9.5, 4), facecolor=SURF)
for dx, d, c, lb in ((-0.27, out["original_failures_by_kind"], GRAY, "original failures of this kind"), (0.0, out["fixed_by_original_kind"], AQUA, "of those, fixed by the edit"),
                     (0.27, out["broken_into_kind"], ORANGE, "original successes broken into this kind")):
    ax.bar_label(ax.bar(np.arange(len(K)) + dx, [d.get(k, 0) for k in K], 0.25, color=c, label=lb), fontsize=8, color=INK)
ax.set_xticks(range(len(K)), lab, fontsize=8)
ax.set_ylabel("test seeds", fontsize=9)
ax.set_title("Strict failures fixed and broken, by kind, 1,000 test seeds", fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper right")
style(ax)
fig.savefig(OUT / "fixed_broken_by_kind.png", dpi=140, bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# fixed and broken seeds on the start positions
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURF)
fx = [s for s in seeds if e[s]["strict_success"] and not o[s]["strict_success"]]
br = [s for s in seeds if o[s]["strict_success"] and not e[s]["strict_success"]]
same = [s for s in seeds if s not in set(fx) | set(br)]
for ax, (xk, yk, name) in zip(axs, (("peg_x0", "peg_y0", "peg"), ("sock_x0", "sock_y0", "socket"))):
    ax.scatter([o[s][xk] for s in same], [o[s][yk] for s in same], s=5, color=GRAY, label=f"unchanged ({len(same)})")
    ax.scatter([o[s][xk] for s in fx], [o[s][yk] for s in fx], s=16, color=AQUA, edgecolors=SURF, linewidths=0.5, label=f"fixed ({len(fx)})")
    ax.scatter([o[s][xk] for s in br], [o[s][yk] for s in br], s=22, color=ORANGE, marker="x", label=f"broken ({len(br)})")
    if name == "peg":
        ax.axvline(0.143, color=MUTED, lw=0.7, ls="--")
        ax.axhline(0.483, color=MUTED, lw=0.7, ls="--")
    ax.set_xlabel(f"{name} start x (m)", fontsize=9)
    ax.set_ylabel(f"{name} start y (m)", fontsize=9)
    ax.set_title(f"Seeds the edit changed, start of the {name}", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    style(ax)
fig.savefig(OUT / "fixed_broken_map.png", dpi=140, bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# random edits by cross head
fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURF)
for a_, v in g.items():
    ax.scatter(key[a_][0], v, s=34, color=MUTED, edgecolors=SURF, linewidths=1, zorder=2)
ax.scatter([0], [out["edit_vs"]["original"]["strict"]["diff_points"]], s=90, color=BLUE, edgecolors=SURF, linewidths=1.5, zorder=3, label="agent's edit (cross.h0)")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(8), [f"cross.h{j}" for j in range(8)])
ax.set_xlabel("cross-attention head turned off (with one enc3 head off and one enc0 head doubled)", fontsize=9)
ax.set_ylabel("strict gain over the original (points)", fontsize=9)
ax.set_title("40 matched random edits, grouped by the cross-attention head", fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)
fig.savefig(OUT / "random_by_head.png", dpi=140, bbox_inches="tight", facecolor=SURF)
print(json.dumps({k: v for k, v in out.items() if k != "random_arms"}, indent=1))
