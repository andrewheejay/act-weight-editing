"""Stage 4 for insertion: assemble the freeze package from files already in the repo. Nothing is tuned here. Built on awe/freeze.py (transfer cube).

python -m awe.freeze_ins random      freeze/random_family.json (the rule and draw of awe/freeze.py, RNG_SEED there)
python -m awe.freeze_ins build       random_family.json, baselines.json, test_manifest.json, then edit.json (exports the edited checkpoint). Safe to rerun.
python -m awe.freeze_ins check       prints the arm list and the invariants
python -m awe.freeze_ins relation    adds `relation_to_agent_candidates` to freeze/autogreedy.json (run by `build` too, idempotent, no rollouts)

Needs stage3/choice.json, freeze/autogreedy.json (from `python -m awe.autogreedy_ins run`) and freeze/analysis_validation/tables.json
(from `python experiments/insertion/freeze/analysis.py --mode validation`).

What differs from awe/freeze.py (all decided before the validation results except where said, see DEVIATIONS.md and PROGRESS.md session 25):
- Primary score is strict success, guard is benchmark success (decisions.json).
- Dose arms: k in (0.5, -0.5, -1, -2), scale = max(0, 1 + k (s - 1)), ids with m for the minus sign (dose_km0.5). Rule written in session 25 for a c06-type
  edit before any validation result. awe/freeze.py's k in (0.5, 1.5, 2, 3) would leave x0 heads at x0 (k >= 1 is the edit itself). The rule is applied as written
  to whatever edit choice.json names, with no change after seeing its values. An op with a scale s > 1 moves the other way (its scale falls as k falls), so
  the four arms keep the pairwise distinctness check of awe/freeze.py's kind (the check is below and fails loudly if two arms are the same).
- Random family, baseline arms and ensembling arms: the rules of awe/freeze.py unchanged.
- baselines.json: the insertion summary has no strict/benchmark pair in the same names, so the numbers are copied from
  stage3/baselines_summary.json (design) and stage3/validation/summary.json (validation, each family once).
"""
import argparse
import json
import re
import sys
from collections import Counter

from . import freeze as F
from .freeze import ENS_COEFF, FAMILIES, base_arm_id, baseline_specs, chosen, dump, exp, fd, j
from .paths import ROOT

TASK = "insertion"
DOSE_K = (0.5, -0.5, -1, -2)
NAMES = {"model.decoder.layers.0.multihead_attn": "cross", "model.decoder.layers.0.self_attn": "dself"}


def short(op):
    b = op["block"]
    n = NAMES.get(b) or f"enc{b.split('.')[3]}"
    return f"{n}.h{op['head']}"


def dose_arms(ops):
    out = []
    for k in DOSE_K:
        scaled = []
        for o in ops:
            s = max(0.0, round(1 + k * (o["scale"] - 1), 6))
            scaled.append({**o, "scale": int(s) if float(s).is_integer() else s})
        lab = ("m" if k < 0 else "") + f"{abs(k):g}"
        out.append({"id": f"dose_k{lab}", "role": "dose", "k": k, "ops": scaled,
                    "note": "the edit's heads at strength k, scale = max(0, 1 + k (s - 1)), rule of session 25 (PROGRESS.md), k = 1 is the edit"})
    keys = [json.dumps([[o["block"], o["head"], float(o["scale"])] for o in a["ops"]]) for a in out]
    keys.append(json.dumps([[o["block"], o["head"], float(o["scale"])] for o in ops]))
    assert len(set(keys)) == len(keys), "dose arms must differ from each other and from the edit"
    return out


def build_baselines(task):
    val = j(exp(task) / "stage3" / "validation" / "summary.json")["results"]
    rows = []
    for fam, (best, spec, design, nset) in baseline_specs(task).items():
        v = val[base_arm_id(fam)]
        rows.append({
            "family": fam, "arm": base_arm_id(fam), "setting": best, "recipe": spec, "n_settings_tried": nset,
            "tuned_on": "design seeds 20000 to 20999 (screening on 200, the best settings on all 1,000), see stage3/baselines_summary.json",
            "design_strict": {k: design[k] for k in ("n", "strict", "gain", "fixed", "broken", "p_mcnemar", "benchmark", "guard_gain")},
            "validation_one_check": {k: v[k] for k in ("n", "strict", "gain", "fixed", "broken", "p", "share_of_failures_removed", "guard_gain")},
        })
    dump(fd(task) / "baselines.json", {"built_by": "python -m awe.freeze_ins build", "selection_rule": "stage3/baselines_summary.json, `rule`, written before the first insertion baseline rollout (session 23)",
                                       "note": "design_strict and validation numbers are of the primary score (strict success); guard_gain is benchmark success. Each family got one validation check, with the setting chosen on the design seeds.",
                                       "baselines": rows})
    print("baselines.json", [r["setting"] for r in rows])


def build_manifest(task):
    cid, ops, _ = chosen(task)
    dec = j(exp(task) / "stage1" / "decisions.json")
    ck = j(exp(task) / "stage0" / "checkpoint.json")
    ag = j(fd(task) / "autogreedy.json")
    rf = j(fd(task) / "random_family.json")
    arms = [{"id": "original", "role": "original"},
            {"id": "agent_edit", "role": "agent_edit", "ops": ops, "candidate": cid},
            {"id": "autogreedy", "role": "autogreedy", "ops": ag["ops"], "edits": ag["edits"]}]
    arms += [{"id": d["id"], "role": "random", "ops": d["ops"]} for d in rf["draws"]]
    for fam, (best, spec, _, _) in baseline_specs(task).items():
        arms.append({"id": base_arm_id(fam), "role": "baseline", "family": fam, "setting": best, **spec})
    arms += dose_arms(ops)
    for role, aid, a_ops in (("ensembling_original", "original_ens", []), ("ensembling_edit", "agent_edit_ens", ops)):
        arms.append({"id": aid, "role": role, "ops": a_ops, "temporal_ensemble_coeff": ENS_COEFF, "n_action_steps": dec["operating_point"]["n_action_steps"]})
    dump(fd(task) / "test_manifest.json", {
        "task": task, "primary_metric": dec["primary_metric"], "operating_point": dec["operating_point"],
        "original_checkpoint_sha256": ck["model_sha256"], "built_by": "python -m awe.freeze_ins build", "arms": arms})
    print("test_manifest.json", len(arms), "arms")


def _ops_key(ops):
    return tuple(sorted((o["block"], o["head"], float(o["scale"])) for o in ops))


def _paired(base, rows, seeds, key="strict_success"):
    f = sum(bool(rows[s][key]) and not base[s][key] for s in seeds)
    b = sum(bool(base[s][key]) and not rows[s][key] for s in seeds)
    return {"n": len(seeds), "fixed": f, "broken": b, "gain_points": round(100 * (f - b) / len(seeds), 1)}


def autogreedy_relation(task):
    """How the auto-greedy edit relates to what the agent had run. Every number comes from files already on disk."""
    from . import autogreedy as A
    from .combo_ins import ALL, spec
    from .sweep_report import find, rows_of

    d = exp(task)
    ag = j(fd(task) / "autogreedy.json")
    base = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
    combo = {}
    for grp in ("A", "B", "AB"):
        f = find(d / "stage3", f"combo_{grp}.jsonl")
        if f.exists():
            for r in rows_of(f):
                combo.setdefault(r["arm"], {})[r["seed"]] = r
    runs = {}
    for r in rows_of(find(d / "stage3", "autogreedy.jsonl")):
        runs.setdefault(r["arm"], {})[r["seed"]] = r
    val0 = {r["seed"]: r for r in rows_of(d / "stage3" / "validation" / "original.jsonl")}
    cand = j(d / "stage3" / "candidates.json")
    same = [c for c, v in cand.items() if _ops_key(v["ops"]) == _ops_key(ag["ops"])]
    pool_ops = {_ops_key([A.parse(e)])[0] for e in ag["pool"]}
    from_pool, same_all = {}, []
    for cid in ALL:
        k = _ops_key(spec(cid)["ops"])
        if set(k) <= pool_ops and cid in combo:
            from_pool[cid] = {"ops": [list(x) for x in k], "design": _paired(base, combo[cid], sorted(combo[cid]))}
        if k == _ops_key(ag["ops"]) and cid in combo:
            same_all.append(cid)
    out = {"identical_to_validated_candidate": same, "identical_to_any_agent_arm": sorted(set(same_all) | set(same)),
           "agent_arms_built_only_from_pool_edits": from_pool}
    for cid in same_all:
        g = runs.get(A.combo_id(A.canon(ag["edits"], ag["pool"])), {})
        e = {"design": _paired(base, combo[cid], sorted(combo[cid])), "search_seeds_450": _paired(base, combo[cid], [s for s in A.SEEDS if s in combo[cid]])}
        vf = d / "stage3" / "validation" / f"{cid}.jsonl"
        if vf.exists():
            v = {r["seed"]: r for r in rows_of(vf)}
            e["validation_500"] = _paired(val0, v, sorted(val0))
            e["validation_500_benchmark"] = _paired(val0, v, sorted(val0), "benchmark_success")
        e["same_traj_hash_as_autogreedy_rows"] = [sum(combo[cid][s]["traj_hash"] == g[s]["traj_hash"] for s in A.SEEDS if s in combo[cid] and s in g), len(A.SEEDS)]
        out[cid] = e
    n = len(ag["ops"])
    out["statement"] = (
        "The search script awe/autogreedy_ins.py was committed before any validation rollout and run after stage3/choice.json, from the stage 2 single-head sweep. "
        f"It returned {n} heads: {', '.join(ag['edits'])}. "
        + (f"Equal, op for op, to agent arm(s) {', '.join(out['identical_to_any_agent_arm'])}. " if out["identical_to_any_agent_arm"] else "No agent arm equals it op for op. ")
        + "Its pool contains all three heads and scales of the frozen edit r1, and the author had seen the hand-written combinations (which contain r1 and its pieces) and, when the run "
        "was started, the validation results of r1 and the other candidates. The search reads only stage 2 files and the original's rows of stage 1. So this arm is not "
        "an independent no-LLM result, and if it equals the agent's arm it is the same arm. The report must not describe it as independent.")
    return out


def relation(task):
    p = fd(task) / "autogreedy.json"
    ag = j(p)
    ag["relation_to_agent_candidates"] = autogreedy_relation(task)
    dump(p, ag)
    print("autogreedy.json relation:", ag["relation_to_agent_candidates"]["identical_to_any_agent_arm"])


def atlas_evidence(task, ops):
    """Stage 2 numbers for the edit's heads, from stage2/atlas/heads.json (fields named as there)."""
    heads = j(exp(task) / "stage2" / "atlas" / "heads.json")
    heads = heads["heads"] if isinstance(heads, dict) else heads
    by = {x["id"]: x for x in heads}
    sig = lambda x: (x["object_signal_drop"]["x0"]["peg"] + x["object_signal_drop"]["x0"]["sock"]) / 2
    ov = lambda x: x["overlap"]["x0"]["step100_strict"]["mean"]
    rank_sig = {x["id"]: i for i, x in enumerate(sorted(heads, key=lambda x: -sig(x)), 1)}
    rank_ov = {x["id"]: i for i, x in enumerate(sorted(heads, key=lambda x: ov(x)), 1)}
    rows = {}
    for o in ops:
        n = short(o)
        x = by[n]
        rows[n] = {"scale_in_edit": o["scale"],
                   "object_signal_drop_x0_peg_grid": x["object_signal_drop"]["x0"]["peg"], "object_signal_drop_x0_socket_grid": x["object_signal_drop"]["x0"]["sock"],
                   "rank_of_48_by_mean_object_signal_drop_x0": rank_sig[n], "overlap_x0_step0_strict_mean": x["overlap"]["x0"]["step0_strict"]["mean"],
                   "overlap_x0_step100_strict_mean": ov(x), "rank_of_48_by_overlap_step100_strict_ascending": rank_ov[n],
                   "overlap_x0_step200_strict_mean": x["overlap"]["x0"]["step200_strict"]["mean"]}
    return rows


def evidence(task):
    """What the files say about the written hypothesis, with the numbers read from them."""
    import numpy as np

    from .dose_ins import timing
    from .sweep_report import rows_of
    from referee.stats import mcnemar_exact

    t = j(fd(task) / "analysis_validation" / "tables.json")
    cid, ops, choice = chosen(task)
    vd = exp(task) / "stage3" / "validation"
    R = lambda n: {r["seed"]: r for r in rows_of(vd / f"{n}.jsonl")}
    o, e = R("original"), R(cid)
    seeds = sorted(o)
    b = t["behaviour"]
    pin = lambda rows: sum(rows[s]["t_pin"] >= 0 for s in seeds)
    conv = lambda rows: sum(bool(rows[s]["strict_success"]) for s in seeds if rows[s]["t_pin"] >= 0)
    others = {}
    for fam in FAMILIES:
        f = vd / f"base_{fam}.jsonl"
        if f.exists():
            x = R(f"base_{fam}")
            a_only = sum(bool(e[s]["strict_success"]) and not x[s]["strict_success"] for s in seeds)
            x_only = sum(bool(x[s]["strict_success"]) and not e[s]["strict_success"] for s in seeds)
            others[fam] = {"edit_only": a_only, "baseline_only": x_only, "p_mcnemar": mcnemar_exact(a_only, x_only)}
    dose = j(exp(task) / "stage3" / "dose_response.json")["heads"]
    val = j(vd / "summary.json")["results"]
    best = max(others, key=lambda f: val[f"base_{f}"]["gain"])
    bv = others[best]
    vs_base = (f"Against the best simple baseline on these validation seeds ({best}, {100 * val[f'base_{best}']['gain']:+.1f} points) the edit is {100 * (sum(bool(e[s]['strict_success']) for s in seeds) - sum(bool(R(f'base_{best}')[s]['strict_success']) for s in seeds)) / len(seeds):+.1f} points ahead, "
               f"paired: {bv['edit_only']} seeds only the edit solves, {bv['baseline_only']} only the baseline, exact p {bv['p_mcnemar']:.2g}, so on validation the edit is not distinguishable from it at 0.05. " if bv['p_mcnemar'] >= 0.05 else "")
    tab = choice["table"][cid]
    gr = t["classes_that_grew"]
    missed = (f"The edit {'passes' if choice['chosen_passes_validation'] else 'does not pass'} the validation rule: on the 500 validation seeds it fixed {tab['val_fixed']} and broke {tab['val_broken']} "
              f"strict seeds ({100 * tab['val_strict_gain']:+.1f} points, exact McNemar p {tab['val_p']:.1e}), which is {100 * tab['val_share_removed']:.1f}% of the original's "
              f"{choice['original_validation_strict_failures']} strict failures (the bar is 25%). "
              f"Benchmark success changed by {100 * tab['val_benchmark_gain']:+.1f} points (the guard allows a drop of at most 2). ")
    summary = (missed + f"Where the files agree with the written hypothesis: the median step of the first pin contact went from {timing(list(o.values()))['median_t_pin']:.0f} to "
               f"{timing(list(e.values()))['median_t_pin']:.0f} and the share of pins that start after step 375 from {b['original']['late_pin_share']:.3f} to {b['agent_edit']['late_pin_share']:.3f}, "
               f"and {conv(e)} of {pin(e)} pin contacts became strict successes against {conv(o)} of {pin(o)} for the original. "
               "Where the files do not support it: the tracking number (the median offset of the peg tip before contact) got worse, not better "
               f"({b['original']['tip_median_cm']:.2f} cm for the original, {b['agent_edit']['tip_median_cm']:.2f} cm for the edit, over the {b['original']['n_tip']} and {b['agent_edit']['n_tip']} episodes that meet), so this "
               "edit is not shown to fix the target failure the way diagnosis.md describes it. The timing story was first formed on the design seeds, and the validation numbers just given also took part in choosing "
               "the edit (the choice pooled design and validation gains), so they are not an independent check of that story. "
               "The medians above are over the episodes that reach the pin, and the edit also changes which episodes do, so part of the change is composition. "
               + vs_base)
    return {
        "summary": summary,
        "timing_validation": {"original": timing(list(o.values())), "agent_edit": timing(list(e.values()))},
        "pin_contacts_strict": {"original": [conv(o), pin(o)], "agent_edit": [conv(e), pin(e)]},
        "tip_offset_median_cm_validation": {"original": b["original"]["tip_median_cm"], "agent_edit": b["agent_edit"]["tip_median_cm"],
                                            "note": "pre_tip_lat over the episodes that meet, tracking number of decisions.json. The edit also changes which episodes meet."},
        "paired_against_the_baselines_validation": others,
        "failure_parts_validation": t["parts_counts"], "parts_that_grew_validation": gr, "original_failures_fixed_by_kind": t["fixed_by_kind"],
        "broken_successes_became": t["broken_into_kind"], "zones_validation": t["zones"],
        "dial_test_single_heads": {short(op): dose[short(op)]["dial"] for op in ops if short(op) in dose},
        "dial_test_note": "design seeds 20000 to 20299, one head at a time, rule in awe/dose_ins.py. The combination itself had no dose response before the test.",
        "atlas_of_the_edited_heads": atlas_evidence(task, ops),
    }


def hypothesis(task, cid):
    txt = (exp(task) / "hypotheses.md").read_text()
    sec = txt[txt.index("validation round, written before any validation rollout"):]
    line = next(l[2:] for l in sec.splitlines() if l.startswith(f"- {cid} "))
    ref = next(l[2:] for l in sec.splitlines() if l.startswith("- c06 "))
    if cid != "c06":
        line = f"{line} [The line for c06, to which this one refers, verbatim: {ref}]"
    return line


def build_edit(task):
    from .export_checkpoint import export
    from .sweep_report import rows_of

    cid, ops, choice = chosen(task)
    d = fd(task)
    out = ROOT / "outputs" / f"edit_ckpt_{task}"
    export(task, "agent_edit", d / "test_manifest.json", out)
    sha = F.sha256(out / "model.safetensors")
    tab = choice["table"][cid]
    val = j(exp(task) / "stage3" / "validation" / "summary.json")["results"][cid]
    ev = evidence(task)
    T = choice["table"]
    best_id = max(T, key=lambda c: T[c]["val_strict_gain"])
    best_net = T[best_id]["val_fixed"] - T[best_id]["val_broken"]
    bar_net = int(-(-0.25 * choice["original_validation_strict_failures"] // 1))
    tied = next((c for c in T if c != cid and T[c]["pooled_net"] == T[cid]["pooled_net"] and T[c]["pooled_broken"] > T[cid]["pooled_broken"]), None)
    tied_broken = T[tied]["pooled_broken"] if tied else None
    tab = T[cid]
    dump(d / "edit.json", {
        "task": task, "candidate": cid, "family": "head_scaling", "ops": ops, "free_numbers": 0, "heads": len(ops),
        "hypothesis": hypothesis(task, cid),
        "hypothesis_source": "experiments/insertion/hypotheses.md, section 'validation round, written before any validation rollout' of 2026-11-25 (session 25), copied word for word. Written after the design results (the candidate's design numbers are in it) and before any validation rollout.",
        "evidence_on_the_hypothesis": ev,
        "chosen_by": "stage3/choice.json, the rule of awe/candidates_ins.py, fixed before any validation rollout (the rule's numbers are in choice.json, table)",
        "stop_rule_used": choice["stop_rule_used"], "passes_validation_rule": choice["chosen_passes_validation"],
        "validation_strict": {"n": val["n"], "strict_gain_points": round(100 * val["gain"], 1), "fixed": val["fixed"], "broken": val["broken"],
                              "share_of_failures_removed": val["share_of_failures_removed"], "p": val["p"], "passes": val["passes"], "benchmark_gain_points": round(100 * val["guard_gain"], 1)},
        "design": {"fixed": tab["design_fixed"], "broken": tab["design_broken"], "pooled_net_design_and_validation": tab["pooled_net"], "note": "1,000 design seeds, used for choosing, optimistic"},
        "stop_rule_statement": (f"No candidate passed the validation rule (25% of strict failures removed, p < 0.05, guard drop at most 2 points). Stop rule, SPEC.md Stage 3: the candidate with the "
                                f"largest pooled net gain is frozen anyway and reported as a miss. {len(choice['table'])} of the 10 validation checks were used, {10 - len(choice['table'])} were not. "
                                f"Reason for not using them: the bar needs a net of {bar_net} fixes of {choice['original_validation_strict_failures']} strict failures, the best of the {len(choice['table'])} reached {best_net} net "
                                f"({best_id}), and any further candidate would have been designed after reading these validation results."),
        "autogreedy_arm": "see relation_to_agent_candidates in freeze/autogreedy.json. Not an independent no-LLM result.",
        "edited_checkpoint": f"outputs/edit_ckpt_{task} (gitignored, rebuilt by awe.export_checkpoint)",
        "edited_checkpoint_sha256": sha, "original_checkpoint_sha256": j(exp(task) / "stage0" / "checkpoint.json")["model_sha256"],
        "limits": (f"Chosen among {len(choice['table'])} validation-checked candidates by the pooled net strict gain over 1,000 design and 500 validation seeds"
                   + (f", level with {tied} at a pooled net gain of {tab['pooled_net']} and chosen over it by the fixed tie rule (fewer broken seeds, {tab['pooled_broken']} against {tied_broken})" if tied else "") + ". "
                   f"{best_id} had the largest validation strict gain ({100 * choice['table'][best_id]['val_strict_gain']:+.1f} points against {100 * tab['val_strict_gain']:+.1f} for the edit) and the candidates are not "
                   "distinguishable from each other on the validation seeds. The design numbers are selected. The edit has no dose response of its own before the test."),
    })
    print("edit.json", cid, "sha", sha)


def build(task):
    chosen(task)
    relation(task)
    F.build_random(task)
    build_baselines(task)
    build_manifest(task)
    build_edit(task)


def check(task):
    m = j(fd(task) / "test_manifest.json")
    ids = [a["id"] for a in m["arms"]]
    print(len(ids), "arms, unique", len(set(ids)) == len(ids), "bad ids", [i for i in ids if not re.fullmatch(r"[A-Za-z0-9._-]+", i)])
    print(dict(Counter(a["role"] for a in m["arms"])))
    for a in m["arms"]:
        for o in a.get("ops", []):
            assert o["op"] == "scale_head" and 0 <= o["head"] < 8, a
    rs = [json.dumps(a["ops"], sort_keys=True) for a in m["arms"] if a["role"] == "random"]
    print("random distinct", len(set(rs)) == len(rs), len(rs))
    print("dose", [(a["id"], [(short(o), o["scale"]) for o in a["ops"]]) for a in m["arms"] if a["role"] == "dose"])
    e = j(fd(task) / "edit.json")
    print("edit", e["candidate"], e["edited_checkpoint_sha256"][:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "random", "baselines", "relation", "dose"])
    ap.add_argument("--task", default=TASK)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.task)
    elif a.cmd == "random":
        F.build_random(a.task)
    elif a.cmd == "baselines":
        build_baselines(a.task)
    elif a.cmd == "relation":
        relation(a.task)
    elif a.cmd == "dose":
        print([(x["id"], [(short(o), o["scale"]) for o in x["ops"]]) for x in dose_arms(chosen(a.task)[1])])
    else:
        check(a.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
