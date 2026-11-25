"""Descriptive tables and figures for the insertion freeze package and the report. It describes results and never decides the verdict (the referee does).

python experiments/insertion/freeze/analysis.py --mode validation     the validation files, output freeze/analysis_validation/
python experiments/insertion/freeze/analysis.py --mode test           test/arms/<arm id>.jsonl of the manifest, output report/analysis/
Options: --arms-dir DIR (read <arm id>.jsonl from there instead), --out DIR. Arms with no file are skipped and listed under `missing`.

The primary score is strict success and the guard is benchmark success (stage1/decisions.json). Every comparison is paired by seed against the
arm `original`. Intervals are 95% percentile bootstrap intervals (referee.stats.bootstrap_ci, 10,000 resamples, seed 0), not the verdict's.
Behaviour numbers: the parts of awe/sweep_ins_report.py (S strict success, A class 4a, M class 3, and the grasp failures by the object not
lifted), the tip offset before contact over the episodes that meet (tracking number, median cm, with the met count), pin contacts and the share of
pins that start after step 375.
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

from awe.dose_ins import timing  # noqa: E402
from awe.paths import exp  # noqa: E402
from awe.sweep_ins_report import part, refs, stats  # noqa: E402
from awe.sweep_report import find, rows_of  # noqa: E402
from referee.stats import bootstrap_ci, paired  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8985", "#fcfcfb", "#c9c8c3"
# Zones. The directions come from diagnosis.md (stage 1, before any edit): the tip offset is larger in the two lowest fifths of peg x and in the three
# highest fifths of peg y. The cutoffs are the edges of those fifths on the design seeds (scratch/ins_combo_overlap.py, part 3). diagnosis.md names no zone
# and no cutoff. They were set in session 25, after the design results of the combinations, and a first version of this file used peg y 0.52, which does not match the fifths. It
# was changed for that reason, after its zone table had been printed once for c06 on group A.
ZONES = {"peg x below 0.143 (larger tip offset)": lambda r: r["peg_x0"] < 0.143, "peg y above 0.483 (larger tip offset)": lambda r: r["peg_y0"] > 0.483}
KINDS = ("S", "A", "M", "peg_not_lifted", "socket_not_lifted", "neither_lifted", "both_lifted")


def load(task, mode, arms_dir):
    e = exp(task)
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
        meta, files = [], {}
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
    st, b = paired(col(o, "strict_success"), col(a, "strict_success")), paired(col(o, "benchmark_success"), col(a, "benchmark_success"))
    ci = bootstrap_ci(col(o, "strict_success"), col(a, "strict_success"), 10_000, 0)
    return {"n": st["n"], "orig_success": st["orig_success"], "success": st["new_success"], "gain_points": st["gain_points"], "gain_ci95": ci["gain_points"],
            "fixed": st["fixed"], "broke": st["broke"], "p_mcnemar": st["p_mcnemar"], "share_removed": st["share_removed"], "share_ci95": ci["share_removed"],
            "guard_orig": b["orig_success"], "guard": b["new_success"], "guard_gain_points": b["gain_points"]}


def tables(task, meta, rows, missing, mode):
    o = rows["original"]
    ref = refs(task)
    per, beh = {}, {}
    for aid, role, fam, extra in meta:
        if aid != "original":
            per[aid] = {"role": role, "family": fam, **extra, **compare(o, rows[aid])}
        s = stats(o, list(rows[aid].values()), ref)
        beh[aid] = {k: s[k] for k in ("n", "n_met", "n_pin", "parts", "tip_median_cm", "n_tip", "strict_run_median_A")}
        beh[aid]["late_pin_share"] = timing(list(rows[aid].values()))["late_pin_share"]
    out = {"mode": mode, "primary": "strict_success", "n_original": len(o), "missing": missing, "arms": per, "behaviour": beh}
    ed = rows.get("agent_edit")
    if ed:
        seeds = sorted(set(o) & set(ed))
        kinds = lambda rs: {k: sum(part(rs[s]) == k for s in seeds) for k in KINDS}
        out["parts_counts"] = {"original": kinds(o), "agent_edit": kinds(ed)}
        out["classes_that_grew"] = {k: [out["parts_counts"]["original"][k], out["parts_counts"]["agent_edit"][k]] for k in KINDS if out["parts_counts"]["agent_edit"][k] > out["parts_counts"]["original"][k]}
        out["part_transitions"] = {f"{a}->{b}": n for (a, b), n in sorted(Counter((part(o[s]), part(ed[s])) for s in seeds).items())}
        out["fixed_by_kind"] = {k: {"original_failures": sum(part(o[s]) == k for s in seeds if not o[s]["strict_success"]), "fixed": sum(part(o[s]) == k and ed[s]["strict_success"] for s in seeds if not o[s]["strict_success"])} for k in KINDS if k != "S"}
        out["broken_into_kind"] = dict(Counter(part(ed[s]) for s in seeds if o[s]["strict_success"] and not ed[s]["strict_success"]))
        z = {}
        for name, f in ZONES.items():
            for lab, want in (("inside", True), ("outside", False)):
                ss = [s for s in seeds if f(o[s]) == want]
                z[f"{name}, {lab}"] = {"n": len(ss), "original": float(np.mean([o[s]["strict_success"] for s in ss])), "agent_edit": float(np.mean([ed[s]["strict_success"] for s in ss])),
                                       "fixed": sum(ed[s]["strict_success"] and not o[s]["strict_success"] for s in ss), "broke": sum(o[s]["strict_success"] and not ed[s]["strict_success"] for s in ss)}
        out["zones"] = z
    rnd = [a for a, r, *_ in meta if r == "random"]
    if rnd and ed:
        g = sorted((per[a]["gain_points"] for a in rnd), reverse=True)
        out["random"] = {"n": len(g), "gains_points": g, "edit_gain_points": per["agent_edit"]["gain_points"], "n_gaining_as_much": sum(x >= per["agent_edit"]["gain_points"] - 1e-12 for x in g)}
    dose = [(0.0, "original")] + ([(1.0, "agent_edit")] if ed else []) + [(float(x[3]["k"]), x[0]) for x in meta if x[1] == "dose"]
    if len(dose) > 2:
        out["dose"] = [{"k": k, "arm": a, "gain_points": 0.0 if a == "original" else per[a]["gain_points"], **beh[a]} for k, a in sorted(dose)]
        v = [d["tip_median_cm"] for d in out["dose"]]
        out["dose_moves_in_step"] = bool(None not in v and (all(x > y for x, y in zip(v, v[1:])) or all(x < y for x, y in zip(v, v[1:]))))
    return out


def markdown(t):
    L = [f"# Descriptive tables ({t['mode']} data)", "", "Primary score: strict success. Paired by seed against the original. Intervals are 95% bootstrap intervals. The referee decides the verdict.", ""]
    if t["missing"]:
        L += ["Arms with no data: " + ", ".join(t["missing"]), ""]
    L += ["| arm | role | n | strict success | gain (points) | 95% interval | fixed | broke | p | share of failures removed | benchmark gain (points) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in sorted(t["arms"].items(), key=lambda kv: -kv[1]["gain_points"]):
        sh = "n/a" if r["share_removed"] is None else f"{100 * r['share_removed']:.1f}%"
        L.append(f"| {a} | {r['role']} | {r['n']} | {100 * r['success']:.1f}% | {r['gain_points']:+.1f} | {r['gain_ci95'][0]:+.1f} to {r['gain_ci95'][1]:+.1f} | {r['fixed']} | {r['broke']} | {r['p_mcnemar']:.2g} | {sh} | {r['guard_gain_points']:+.1f} |")
    L += ["", "## Behaviour numbers (parts: S strict, A class 4a, M class 3, grasp failures by what was not lifted; tip: median offset before contact over episodes that meet)", "", "| arm | " + " | ".join(KINDS) + " | met | pin | late pin share | tip (cm) |", "|---|" + "---|" * (len(KINDS) + 4)]
    for a, b in t["behaviour"].items():
        if not a.startswith("random_"):
            L.append(f"| {a} | " + " | ".join(str(b["parts"][k]) for k in KINDS) + f" | {b['n_met']} | {b['n_pin']} | {b['late_pin_share']} | {b['tip_median_cm']} |")
    if "parts_counts" in t:
        L += ["", "Parts that grew in the edit: " + (", ".join(f"{k} ({a} to {b})" for k, (a, b) in t["classes_that_grew"].items()) or "none")]
        L += ["", "Original failures fixed by kind: " + ", ".join(f"{k} {v['fixed']} of {v['original_failures']}" for k, v in t["fixed_by_kind"].items()) + f". Broken successes became: {t['broken_into_kind']}"]
        L += ["", "## Zones", "", "| zone | n | original | edit | fixed | broke |", "|---|---|---|---|---|---|"] + [f"| {k} | {z['n']} | {100 * z['original']:.1f}% | {100 * z['agent_edit']:.1f}% | {z['fixed']} | {z['broke']} |" for k, z in t["zones"].items()]
    if "random" in t:
        r = t["random"]
        L += ["", f"Matched random edits: {r['n_gaining_as_much']} of {r['n']} gained as much as the edit ({r['edit_gain_points']:+.1f} points). Best random {r['gains_points'][0]:+.1f}."]
    if "dose" in t:
        L += ["", "## Dose (k = 0 is the original, k = 1 the edit)", "", "| k | arm | gain (points) | met | median tip offset (cm) |", "|---|---|---|---|---|"] + [f"| {d['k']:g} | {d['arm']} | {d['gain_points']:+.1f} | {d['n_met']} | {d['tip_median_cm']} |" for d in t["dose"]]
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
    ax.set_xlabel("gain over the original, points of strict success (95% bootstrap interval)", fontsize=9)
    ax.set_title(f"Gain of each arm, {t['mode']} seeds. Blue is the agent's edit", fontsize=10, loc="left", color=INK)
    style(ax)
    save(fig, "gain_by_arm.png")

    o, e = rows["original"], rows.get("agent_edit")
    if e:
        ss = sorted(set(o) & set(e))
        fig, axs = plt.subplots(2, 2, figsize=(12, 10.5), facecolor=SURF)
        for row, (xk, yk, lab) in enumerate((("peg_x0", "peg_y0", "peg"), ("sock_x0", "sock_y0", "socket"))):
            for ax, name, src in ((axs[row, 0], "Original", o), (axs[row, 1], "Agent's edit", e)):
                for nm, want, c, m, sz in (("strict success", True, AQUA, "o", 18), ("failure", False, GRAY, "o", 8)):
                    p = np.array([(o[s][xk], o[s][yk]) for s in ss if src[s]["strict_success"] == want])
                    if len(p):
                        ax.scatter(*p.T, s=sz, c=c, marker=m, edgecolors=SURF, linewidths=0.5, label=f"{nm} ({len(p)})", zorder=2 + want)
                ax.set_title(f"{name}, start of the {lab}, {t['mode']} seeds", fontsize=10, loc="left", color=INK)
                ax.set_xlabel(f"{lab} start x (m)"), ax.set_ylabel(f"{lab} start y (m)"), style(ax)
                ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        fig.tight_layout()
        save(fig, "before_after_map.png")
    if "random" in t:
        r = t["random"]
        fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURF)
        ax.hist(r["gains_points"], bins=20, color=GRAY, edgecolor=SURF, linewidth=1)
        ax.axvline(r["edit_gain_points"], color=BLUE, lw=2)
        ax.set_xlabel("gain over the original (points of strict success)"), ax.set_ylabel("matched random edits")
        ax.set_title(f"{r['n']} matched random edits, {r['n_gaining_as_much']} gained as much as the edit ({r['edit_gain_points']:+.1f})", fontsize=10, loc="left", color=INK)
        style(ax)
        save(fig, "random_edits.png")
    if "dose" in t:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF)
        k = [d["k"] for d in t["dose"]]
        for ax, key, lab in ((axs[0], "gain_points", "gain over the original (points)"), (axs[1], "tip_median_cm", "median tip offset before contact (cm), the tracking number")):
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
    ap.add_argument("--task", default="insertion")
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
