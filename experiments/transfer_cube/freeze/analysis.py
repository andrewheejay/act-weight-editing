"""Descriptive tables and figures for the freeze package and the report. It describes results and never decides the verdict (the referee does).

python experiments/transfer_cube/freeze/analysis.py --mode validation     the validation files, output freeze/analysis_validation/
python experiments/transfer_cube/freeze/analysis.py --mode test           test/arms/<arm id>.jsonl of the manifest, output report/analysis/
Options: --arms-dir DIR (read <arm id>.jsonl from there instead), --out DIR. Arms with no file are skipped and listed under `missing`.

Writes tables.json, tables.md and the figures. Every comparison is paired by seed against the arm `original`, on the seeds both arms have.
Intervals are 95% percentile bootstrap intervals (referee.stats.bootstrap_ci, 10,000 resamples, seed 0). They are not the verdict's intervals,
which the referee seeds from the beacon.
"""
import argparse
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

from awe.paths import exp  # noqa: E402
from awe.sweep_report import find, refs, rows_of, stats as arm_stats  # noqa: E402
from referee.stats import bootstrap_ci, paired  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8985", "#fcfcfb", "#c9c8c3"
# The first two zones are named in diagnosis.md (stage 1, design seeds), before any edit. The third was defined on 2026-11-24 after looking at the
# validation map of the chosen edit, because the broken seeds gather there. It is post hoc on design and validation seeds and only the test seeds check it.
ZONES = {"hot block x 0.08-0.16, y 0.40-0.48": lambda x, y: 0.08 <= x < 0.16 and 0.40 <= y < 0.48, "low strip y < 0.44": lambda x, y: y < 0.44,
         "low corner x < 0.06, y < 0.47 (post hoc)": lambda x, y: x < 0.06 and y < 0.47}


def load(task, mode, arms_dir):
    """-> (list of (arm id, role, family, extra), {arm id: {seed: row}}, missing ids)"""
    e = exp(task)
    meta, files = [], {}
    if mode == "validation" and arms_dir is None:
        v = e / "stage3" / "validation"
        cid = json.loads((e / "stage3" / "choice.json").read_text())["chosen"]
        meta = [("original", "original", None, {}), ("agent_edit", "agent_edit", None, {"candidate": cid})]
        files = {"original": find(v, "original.jsonl"), "agent_edit": find(v, f"{cid}.jsonl")}
        for fam in ("constant_offset", "deviation_gain", "shorter_chunk", "simplest_fix"):
            meta.append((f"base_{fam}", "baseline", fam, {}))
            files[f"base_{fam}"] = find(v, f"base_{fam}.jsonl")
        for c in json.loads((e / "stage3" / "candidates.json").read_text()):
            if c != cid:
                meta.append((f"cand_{c}", "other_candidate", None, {}))
                files[f"cand_{c}"] = find(v, f"{c}.jsonl")
    else:
        m = json.loads((e / "freeze" / "test_manifest.json").read_text())
        d = Path(arms_dir) if arms_dir else e / "test" / "arms"
        for a in m["arms"]:
            meta.append((a["id"], a["role"], a.get("family"), {k: a[k] for k in ("k", "ops", "setting", "edits") if k in a}))
            files[a["id"]] = find(d, f"{a['id']}.jsonl")
    rows, missing = {}, []
    for aid, *_ in meta:
        if files[aid].exists():
            rows[aid] = {r["seed"]: r for r in rows_of(files[aid])}
        else:
            missing.append(aid)
    return [x for x in meta if x[0] in rows], rows, missing


def col(rows, key):
    return {s: bool(r[key]) for s, r in rows.items()}


def compare(o, a):
    """one arm against the original, both scores"""
    b, st = paired(col(o, "benchmark_success"), col(a, "benchmark_success")), paired(col(o, "strict_success"), col(a, "strict_success"))
    ci = bootstrap_ci(col(o, "benchmark_success"), col(a, "benchmark_success"), 10_000, 0)
    return {"n": b["n"], "orig_success": b["orig_success"], "success": b["new_success"], "gain_points": b["gain_points"], "gain_ci95": ci["gain_points"],
            "fixed": b["fixed"], "broke": b["broke"], "p_mcnemar": b["p_mcnemar"], "share_removed": b["share_removed"], "share_ci95": ci["share_removed"],
            "strict_orig": st["orig_success"], "strict": st["new_success"], "strict_gain_points": st["gain_points"]}


def kind(r):
    """success, grasp_fail (never lifted, the target failure of diagnosis.md) or other_fail"""
    return "success" if r["benchmark_success"] else ("grasp_fail" if r["grasp_fail"] else "other_fail")


def tables(task, meta, rows, missing, mode):
    o = rows["original"]
    reach, lat = refs(task)
    per = {}
    for aid, role, fam, extra in meta:
        if aid != "original":
            per[aid] = {"role": role, "family": fam, **extra, **compare(o, rows[aid])}
    beh = {}
    for aid, *_ in meta:
        s = arm_stats(o, list(rows[aid].values()), reach, lat)
        beh[aid] = {k: s[k] for k in ("n", "grasp_fail", "reach_err_median_cm", "cmd_lat8_dev_median_cm_clean", "n_clean")}
    out = {"mode": mode, "n_original": len(o), "missing": missing, "arms": per, "behaviour": beh}
    ed = rows.get("agent_edit")
    if ed:
        seeds = sorted(set(o) & set(ed))
        out["class_counts"] = {"original": dict(sorted(Counter(o[s]["failure_class"] for s in seeds).items())), "agent_edit": dict(sorted(Counter(ed[s]["failure_class"] for s in seeds).items()))}
        grew = {c: [out["class_counts"]["original"].get(c, 0), n] for c, n in out["class_counts"]["agent_edit"].items() if c in "0123" and n > out["class_counts"]["original"].get(c, 0)}
        out["failure_classes_that_grew"] = grew
        out["class_transitions"] = {f"{a}->{b}": n for (a, b), n in sorted(Counter((o[s]["failure_class"], ed[s]["failure_class"]) for s in seeds).items())}
        out["fixed_by_kind"] = {k: {"original_failures": sum(kind(o[s]) == k for s in seeds), "fixed": sum(kind(o[s]) == k and ed[s]["benchmark_success"] for s in seeds)} for k in ("grasp_fail", "other_fail")}
        out["broken_into_kind"] = dict(Counter(kind(ed[s]) for s in seeds if o[s]["benchmark_success"] and not ed[s]["benchmark_success"]))
        z = {}
        for name, f in ZONES.items():
            for lab, want in (("inside", True), ("outside", False)):
                ss = [s for s in seeds if f(o[s]["cube_x0"], o[s]["cube_y0"]) == want]
                z[f"{name}, {lab}"] = {"n": len(ss), "original": float(np.mean([o[s]["benchmark_success"] for s in ss])), "agent_edit": float(np.mean([ed[s]["benchmark_success"] for s in ss])),
                                       "fixed": sum(ed[s]["benchmark_success"] and not o[s]["benchmark_success"] for s in ss), "broke": sum(o[s]["benchmark_success"] and not ed[s]["benchmark_success"] for s in ss)}
        out["zones"] = z
    rnd = [a for a, r, *_ in meta if r == "random"]
    if rnd and ed:
        g = sorted((per[a]["gain_points"] for a in rnd), reverse=True)
        out["random"] = {"n": len(g), "gains_points": g, "edit_gain_points": per["agent_edit"]["gain_points"], "n_gaining_as_much": sum(x >= per["agent_edit"]["gain_points"] - 1e-12 for x in g)}
    # the dial: k = 0 is the original (every scale 1) and k = 1 is the edit
    dose = [(0.0, "original")] + ([(1.0, "agent_edit")] if ed else []) + [(float(x[3]["k"]), x[0]) for x in meta if x[1] == "dose"]
    if len(dose) > 2:
        out["dose"] = [{"k": k, "arm": a, "gain_points": 0.0 if a == "original" else per[a]["gain_points"], **beh[a]} for k, a in sorted(dose)]
        v = [d["reach_err_median_cm"] for d in out["dose"]]
        out["dose_moves_in_step"] = bool(all(x > y for x, y in zip(v, v[1:])) or all(x < y for x, y in zip(v, v[1:])))
    return out


def markdown(t):
    L = [f"# Descriptive tables ({t['mode']} data)", "", "Paired by seed against the original. Intervals are 95% bootstrap intervals. The referee decides the verdict.", ""]
    if t["missing"]:
        L += ["Arms with no data: " + ", ".join(t["missing"]), ""]
    L += ["| arm | role | n | success | gain (points) | 95% interval | fixed | broke | p | share of failures removed | strict gain (points) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in sorted(t["arms"].items(), key=lambda kv: -kv[1]["gain_points"]):
        sh = "n/a" if r["share_removed"] is None else f"{100 * r['share_removed']:.1f}%"
        L.append(f"| {a} | {r['role']} | {r['n']} | {100 * r['success']:.1f}% | {r['gain_points']:+.1f} | {r['gain_ci95'][0]:+.1f} to {r['gain_ci95'][1]:+.1f} | {r['fixed']} | {r['broke']} | {r['p_mcnemar']:.2g} | {sh} | {r['strict_gain_points']:+.1f} |")
    L += ["", "## Behaviour numbers", "", "| arm | grasp failures | median reach error (cm) | median lateral command deviation (cm) |", "|---|---|---|---|"]
    for a, b in t["behaviour"].items():
        if not a.startswith("random_"):
            L.append(f"| {a} | {b['grasp_fail']} | {b['reach_err_median_cm']} | {b['cmd_lat8_dev_median_cm_clean']} |")
    if "class_counts" in t:
        cs = sorted(set(t["class_counts"]["original"]) | set(t["class_counts"]["agent_edit"]))
        L += ["", "## Failure classes, original against the edit", "", "| class | original | edit |", "|---|---|---|"] + [f"| {c} | {t['class_counts']['original'].get(c, 0)} | {t['class_counts']['agent_edit'].get(c, 0)} |" for c in cs]
        L += ["", "Failure classes that grew: " + (", ".join(f"{c} ({a} to {b})" for c, (a, b) in t["failure_classes_that_grew"].items()) or "none")]
        L += ["", "Original failures fixed: " + ", ".join(f"{k} {v['fixed']} of {v['original_failures']}" for k, v in t["fixed_by_kind"].items()) + f". Broken successes became: {t['broken_into_kind']}"]
        L += ["", "## Zones", "", "| zone | n | original | edit | fixed | broke |", "|---|---|---|---|---|---|"] + [f"| {k} | {z['n']} | {100 * z['original']:.1f}% | {100 * z['agent_edit']:.1f}% | {z['fixed']} | {z['broke']} |" for k, z in t["zones"].items()]
    if "random" in t:
        r = t["random"]
        L += ["", f"Matched random edits: {r['n_gaining_as_much']} of {r['n']} gained as much as the edit ({r['edit_gain_points']:+.1f} points). Best random {r['gains_points'][0]:+.1f}."]
    if "dose" in t:
        L += ["", "## Dose (k = 0 is the original, k = 1 the edit)", "", "| k | arm | gain (points) | grasp failures | median reach error (cm) |", "|---|---|---|---|---|"] + [f"| {d['k']:g} | {d['arm']} | {d['gain_points']:+.1f} | {d['grasp_fail']} | {d['reach_err_median_cm']} |" for d in t["dose"]]
        L += ["", f"The tracking number moves the same way at every step: {t['dose_moves_in_step']}"]
    return "\n".join(L) + "\n"


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors="#52514e")


def figures(t, rows, out):
    made = []

    def save(fig, name):
        fig.savefig(out / name, dpi=140, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
        made.append(name)

    arms = sorted(((a, r) for a, r in t["arms"].items() if r["role"] != "random"), key=lambda x: x[1]["gain_points"])
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(arms) + 1.4), facecolor=SURF)
    for i, (a, r) in enumerate(arms):
        c = BLUE if a == "agent_edit" else MUTED
        ax.plot(r["gain_ci95"], [i, i], color=c, lw=2, zorder=2)
        ax.scatter([r["gain_points"]], [i], s=56, color=c, edgecolors=SURF, linewidths=1.5, zorder=3)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(arms)), [f"{a} (n={r['n']})" for a, r in arms])
    ax.set_xlabel("gain over the original, points of benchmark success (95% bootstrap interval)", fontsize=9)
    ax.set_title(f"Gain of each arm, {t['mode']} seeds. Blue is the agent's edit", fontsize=10, loc="left", color=INK)
    style(ax)
    save(fig, "gain_by_arm.png")

    o, e = rows["original"], rows.get("agent_edit")
    if e:
        ss = sorted(set(o) & set(e))
        fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.4), facecolor=SURF, sharex=True, sharey=True)
        for ax, lab, src in ((axs[0], "Original", o), (axs[1], "Agent's edit", e)):
            for nm, want, c, m, z in (("success", True, GRAY, "o", 10), ("failure", False, ORANGE, "D", 24)):
                p = np.array([(o[s]["cube_x0"], o[s]["cube_y0"]) for s in ss if src[s]["benchmark_success"] == want])
                if len(p):
                    ax.scatter(*p.T, s=z, c=c, marker=m, edgecolors=SURF, linewidths=0.5, label=f"{nm} ({len(p)})", zorder=2 + (not want))
            ax.set_title(f"{lab}, {t['mode']} seeds", fontsize=10, loc="left", color=INK)
        for nm, f, c, m, z in (("no change", lambda a, b: a == b, GRAY, "o", 8), ("fixed", lambda a, b: b and not a, AQUA, "o", 30), ("broken", lambda a, b: a and not b, ORANGE, "X", 36)):
            p = np.array([(o[s]["cube_x0"], o[s]["cube_y0"]) for s in ss if f(o[s]["benchmark_success"], e[s]["benchmark_success"])])
            if len(p):
                axs[2].scatter(*p.T, s=z, c=c, marker=m, edgecolors=SURF, linewidths=0.6, label=f"{nm} ({len(p)})", zorder=2 if nm == "no change" else 3)
        axs[2].set_title("What changed", fontsize=10, loc="left", color=INK)
        for ax in axs:
            ax.set_xlabel("cube start x (m)"), style(ax)
            ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
        axs[0].set_ylabel("cube start y (m)")
        save(fig, "before_after_map.png")
    if "random" in t:
        r = t["random"]
        fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURF)
        ax.hist(r["gains_points"], bins=20, color=GRAY, edgecolor=SURF, linewidth=1)
        ax.axvline(r["edit_gain_points"], color=BLUE, lw=2)
        ax.annotate(f"agent's edit {r['edit_gain_points']:+.1f}", (r["edit_gain_points"], ax.get_ylim()[1]), xytext=(-4, -4), textcoords="offset points", ha="right", va="top", fontsize=9, color=INK)
        ax.set_xlabel("gain over the original (points)"), ax.set_ylabel("matched random edits")
        ax.set_title(f"{r['n']} matched random edits, {r['n_gaining_as_much']} gained as much as the edit", fontsize=10, loc="left", color=INK)
        style(ax)
        save(fig, "random_edits.png")
    if "dose" in t:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF)
        k = [d["k"] for d in t["dose"]]
        for ax, key, lab in ((axs[0], "gain_points", "gain over the original (points)"), (axs[1], "reach_err_median_cm", "median reach error (cm), the tracking number")):
            ax.plot(k, [d[key] for d in t["dose"]], color=MUTED, lw=2, marker="o", ms=7, mec=SURF, zorder=2)
            hit = [(d["k"], d[key]) for d in t["dose"] if d["arm"] == "agent_edit"]
            if hit:
                ax.scatter(*zip(*hit), s=90, color=BLUE, edgecolors=SURF, linewidths=1.5, zorder=3, label="agent's edit (k = 1)")
                ax.legend(frameon=False, fontsize=8)
            ax.set_xlabel("strength k (0 is the original)"), ax.set_ylabel(lab)
            style(ax)
        axs[0].set_title("Dose response of the edit's heads", fontsize=10, loc="left", color=INK)
        save(fig, "dose_response.png")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="transfer_cube")
    ap.add_argument("--mode", choices=["validation", "test"], required=True)
    ap.add_argument("--arms-dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    meta, rows, missing = load(a.task, a.mode, a.arms_dir)
    out = Path(a.out) if a.out else (exp(a.task) / "freeze" / "analysis_validation" if a.mode == "validation" else exp(a.task) / "report" / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    t = tables(a.task, meta, rows, missing, a.mode)
    t["figures"] = figures(t, rows, out)
    (out / "tables.json").write_text(json.dumps(t, indent=1) + "\n")
    (out / "tables.md").write_text(markdown(t))
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main()
