"""Stage 4: assemble the freeze package for one task. Everything is built from files already in the repo, nothing is tuned here.

python -m awe.freeze build     random_family.json, baselines.json, test_manifest.json, the exported checkpoint, edit.json, in that order.
                               Needs stage3/choice.json and freeze/autogreedy.json (from `python -m awe.autogreedy run`). Safe to rerun.
python -m awe.freeze check     prints the arm list and the invariants (arm counts, unique ids, ops legal, random edits distinct, shas)

The agent's edit is the candidate named in stage3/choice.json, with the ops in stage3/candidates.json. The rules for the other arms:
- random: the edit's ops with each head replaced by one of the 7 other heads of its block, same block and scale, 40 drawn without
  repeats by random.Random(RNG_SEED).sample over the full list of distinct combinations (RNG_SEED is fixed here, before any random edit ran).
- baseline: the best setting per family (stage3/baselines_summary.json, the pre-registered selection on the 1,000 design seeds).
- dose: the edit's own heads at four other strengths. A dial k multiplies each op's distance from 1 (scale = max(0, 1 + k (s - 1))), k in DOSE_K.
  k = 1 is the edit itself and is not a dose arm. The four arms are checked to differ from each other and from the edit.
- ensembling: original and edit with temporal ensembling 0.01 at the checkpoint's own chunk, on all 1,000 test seeds (no n_seeds: the box
  renders in software but runs inference on a GPU, and 1,000 ensembled episodes took 2.2 hours in stage 1).
- autogreedy: the ops in freeze/autogreedy.json, written by the script before the final manifest.
- The operating point is 100 steps (Gate 1), so there are no *_default arms.
"""
import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path

from .paths import ROOT, exp

RNG_SEED = 20261124
N_RANDOM = 40
DOSE_K = (0.5, 1.5, 2, 3)
FAMILIES = ("constant_offset", "deviation_gain", "shorter_chunk", "simplest_fix")
ENS_COEFF = 0.01
RANDOM_RULE = ("Matched random edits: the agent edit's ops with each head replaced by one of the 7 other heads of the same block (same layer, same "
               "attention type, same scale, same number of heads). All distinct combinations are listed in a fixed order (product of the other heads, ascending, "
               "ops in the order of edit.json) and 40 are drawn without repeats by random.Random(rng_seed).sample. If fewer than 40 exist, all are used.")


def fd(task):
    d = exp(task) / "freeze"
    d.mkdir(parents=True, exist_ok=True)
    return d


def j(p):
    return json.loads(Path(p).read_text())


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1) + "\n")


def chosen(task):
    c = j(exp(task) / "stage3" / "choice.json")
    cand = j(exp(task) / "stage3" / "candidates.json")
    return c["chosen"], cand[c["chosen"]]["ops"], c


def all_matched(ops):
    """Every edit with the same blocks and scales, one other head (of 7) per op."""
    blocks = [(o["block"], o["head"], o["scale"]) for o in ops]
    assert len({b for b, _, _ in blocks}) == len(blocks), "matched rule assumes one op per block"
    choices = [[h for h in range(8) if h != head] for _, head, _ in blocks]
    out = []
    for hs in itertools.product(*choices):
        out.append([{"op": "scale_head", "block": b, "head": h, "scale": s} for (b, _, s), h in zip(blocks, hs)])
    return out


def build_random(task):
    cid, ops, _ = chosen(task)
    pool_ = all_matched(ops)
    rng = random.Random(RNG_SEED)
    n = min(N_RANDOM, len(pool_))
    draws = rng.sample(pool_, n)
    keys = {json.dumps(d, sort_keys=True) for d in draws}
    assert len(keys) == n
    dump(fd(task) / "random_family.json", {
        "rule": RANDOM_RULE,
        "rng_seed": RNG_SEED, "n_available": len(pool_), "n_drawn": n, "for_candidate": cid,
        "matched_on": "number of heads, block (layer and attention type) and scale. Different heads.",
        "draws": [{"id": f"random_{i:02d}", "ops": d} for i, d in enumerate(draws)]})
    print("random_family.json", n, "of", len(pool_))


def base_arm_id(family):
    return f"base_{family}"


def baseline_specs(task):
    s = j(exp(task) / "stage3" / "baselines_summary.json")
    out = {}
    for fam in FAMILIES:
        f = s["families"][fam]
        best = f["best"]
        if fam == "shorter_chunk":
            spec = {"n_action_steps": f["results"][best]["n_action_steps"]}
            design = f["results"][best]["all"]
        else:
            spec = {"post": f["recipes"][best]}
            design = f["results"][best]["all"]
        out[fam] = (best, spec, design, f.get("n_settings"))
    return out


def build_baselines(task):
    val = j(exp(task) / "stage3" / "validation" / "summary.json")["results"]
    rows = []
    for fam, (best, spec, design, nset) in baseline_specs(task).items():
        v = val[base_arm_id(fam)]
        rows.append({
            "family": fam, "arm": base_arm_id(fam), "setting": best, "recipe": spec, "n_settings_tried": nset,
            "tuned_on": "design seeds 20000 to 20999 (screening on 200, the best settings on all 1,000), see stage3/baselines_summary.json",
            "design": {k: design[k] for k in ("n", "gain", "strict_gain", "fixed", "broken", "p_mcnemar")},
            "validation_one_check": {k: v[k] for k in ("n", "gain", "strict_gain", "fixed", "broken", "p", "share_of_failures_removed")},
        })
    dump(fd(task) / "baselines.json", {"built_by": "python -m awe.freeze build", "selection_rule": "stage3/baselines_summary.json, `rule`, before any baseline rollout",
                                       "deviations": "two extra deviation gains and both gripper readings (DEVIATIONS.md, 2026-11-24)", "baselines": rows})
    print("baselines.json", [r["setting"] for r in rows])


def dose_arms(task, ops):
    out = []
    for k in DOSE_K:
        scaled = [{**o, "scale": max(0.0, round(1 + k * (o["scale"] - 1), 6))} for o in ops]
        scaled = [{**o, "scale": int(o["scale"]) if float(o["scale"]).is_integer() else o["scale"]} for o in scaled]
        out.append({"id": f"dose_k{k:g}", "role": "dose", "k": k, "ops": scaled, "note": "the edit's heads at strength k, scale = max(0, 1 + k (s - 1))"})
    keys = [json.dumps([[o["block"], o["head"], float(o["scale"])] for o in a["ops"]]) for a in out]
    keys.append(json.dumps([[o["block"], o["head"], float(o["scale"])] for o in ops]))
    assert len(set(keys)) == len(keys), "dose arms must differ from each other and from the edit"
    return out


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
    arms += dose_arms(task, ops)
    for role, aid, a_ops in (("ensembling_original", "original_ens", []), ("ensembling_edit", "agent_edit_ens", ops)):
        arms.append({"id": aid, "role": role, "ops": a_ops, "temporal_ensemble_coeff": ENS_COEFF, "n_action_steps": dec["operating_point"]["n_action_steps"]})
    dump(fd(task) / "test_manifest.json", {
        "task": task, "primary_metric": dec["primary_metric"], "operating_point": dec["operating_point"],
        "original_checkpoint_sha256": ck["model_sha256"], "built_by": "python -m awe.freeze build", "arms": arms})
    print("test_manifest.json", len(arms), "arms")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hypothesis_line(task, cid):
    """The candidate's line in the validation section of hypotheses.md, verbatim (written before any validation rollout)."""
    txt = (exp(task) / "hypotheses.md").read_text()
    sec = txt[txt.index("validation round, written before any validation rollout"):]
    return next(l[2:] for l in sec.splitlines() if l.startswith(f"- {cid} "))


def atlas_evidence(task):
    """What the stage 2 head atlas says about the two edited heads: the cube-position signal they carry and the failure-map overlap."""
    heads = j(exp(task) / "stage2" / "atlas" / "heads.json")
    heads = heads["heads"] if isinstance(heads, dict) else heads
    by = {x["id"]: x for x in heads}
    order = sorted(heads, key=lambda x: -x["cube_signal_drop"]["x0"])
    rank = {x["id"]: i for i, x in enumerate(order, 1)}
    low = {x["id"]: i for i, x in enumerate(sorted(heads, key=lambda x: x["overlap"]["x0"]["step100"]), 1)}  # 1 is the most negative overlap
    cid, ops, _ = chosen(task)
    rows = {}
    for o in ops:
        enc = o["block"].startswith("model.encoder")
        name = f"enc{o['block'].split('.')[3]}.h{o['head']}" if enc else ("cross" if "multihead" in o["block"] else "dself") + f".h{o['head']}"
        x = by[name]
        rows[name] = {"scale_in_edit": o["scale"], "cube_signal_drop_x0": x["cube_signal_drop"]["x0"], "rank_of_48_by_cube_signal_drop_x0": rank[name],
                      "overlap_with_failure_map_x0": {"step0": x["overlap"]["x0"]["step0"], "step100": x["overlap"]["x0"]["step100"]},
                      "rank_of_48_by_overlap_step100_ascending": low[name]}
    e, c = rows["enc3.h4"], rows["cross.h1"]
    sentence = (f"enc3.h4, the head the hypothesis says carries the cube signal too weakly, has a cube-signal drop of {e['cube_signal_drop_x0']:+.3f} when off "
                f"(rank {e['rank_of_48_by_cube_signal_drop_x0']} of 48, so it carries almost none of it), and its effect map overlaps the failure map at step 100 at "
                f"{e['overlap_with_failure_map_x0']['step100']:+.2f} (rank {e['rank_of_48_by_overlap_step100_ascending']} of 48, 1 is the most negative). "
                f"cross.h1, which the edit halves, has a cube-signal drop of {c['cube_signal_drop_x0']:+.3f}, rank {c['rank_of_48_by_cube_signal_drop_x0']} of 48 (1 is the strongest reader). The validation gain stands on the score, not on the story.")
    return {"heads": rows, "sentence": sentence}


def evidence(task):
    """What the files say about the written hypothesis, with the numbers read from them (analysis.py on validation data, the dose response)."""
    t = j(fd(task) / "analysis_validation" / "tables.json")
    h = t["behaviour"]
    dose = j(exp(task) / "stage3" / "dose_response.json")["heads"]
    atlas = atlas_evidence(task)
    return {
        "summary": ("On the validation seeds the edit passes the validation rule, but the files do not support the written mechanism (a better first reach): "
                    "the median reach error went up, the gain is as large on the other failures as on the target failures (fixed_by_kind), and the edit "
                    "breaks successes in the low corner of the square (zones). The head atlas does not support it either: " + atlas["sentence"]),
        "atlas_of_the_edited_heads": atlas["heads"],
        "reach_error_median_cm_validation": {"original": h["original"]["reach_err_median_cm"], "agent_edit": h["agent_edit"]["reach_err_median_cm"]},
        "original_failures_fixed_by_kind": t["fixed_by_kind"],
        "failure_class_counts_validation": t["class_counts"],
        "failure_classes_that_grew_validation": t["failure_classes_that_grew"],
        "zones_validation": t["zones"],
        "dial_test_single_heads": {k: dose[k]["dial"] for k in ("enc3.h4", "cross.h1")},
        "dial_test_note": "design seeds 20000 to 20299, one head at a time, rule in awe/dose.py. The pair itself has no dose response before the test.",
    }


def build_edit(task, out_dir=None):
    from .export_checkpoint import export

    cid, ops, choice = chosen(task)
    d = fd(task)
    out = Path(out_dir or ROOT / "outputs" / f"edit_ckpt_{task}")
    export(task, "agent_edit", d / "test_manifest.json", out)
    sha = sha256(out / "model.safetensors")
    t = choice["table"][cid]
    val = j(exp(task) / "stage3" / "validation" / "summary.json")["results"][cid]
    dump(d / "edit.json", {
        "task": task, "candidate": cid, "family": "head_scaling", "ops": ops, "free_numbers": 0,
        "heads": len(ops), "hypothesis": hypothesis_line(task, cid),
        "hypothesis_source": "hypotheses.md, validation section of 2026-11-24, copied word for word. Written before any validation rollout and after the design results.",
        "evidence_on_the_hypothesis": evidence(task),
        "chosen_by": "stage3/choice.json, the rule of awe/candidates.py, fixed before any validation rollout",
        "stop_rule_used": choice["stop_rule_used"], "passes_validation_rule": choice["chosen_passes_validation"],
        "validation": {k: val[k] for k in ("n", "gain", "strict_gain", "fixed", "broken", "share_of_failures_removed", "p", "passes")},
        "design": {"fixed": t["design_fixed"], "broken": t["design_broken"], "pooled_net": t["pooled_net"], "note": "1,000 design seeds, used for choosing"},
        "edited_checkpoint": f"outputs/edit_ckpt_{task} (gitignored, rebuilt by awe.export_checkpoint)",
        "edited_checkpoint_sha256": sha, "original_checkpoint_sha256": j(exp(task) / "stage0" / "checkpoint.json")["model_sha256"],
        "limits": "Chosen among 8 validation-checked candidates (pooled design and validation net gain). Validation gains of the 8 were 7.6 to 10.4 points, "
                  "not distinguishable from each other. r6 (three heads) had the largest validation gain (+10.4, the edit +9.2) and r4 the largest pooled net "
                  "(152 against 150). s1 (one head) was excluded from the choice because only 7 matched random edits exist for it.",
    })
    print("edit.json", cid, "sha", sha)


def build(task):
    chosen(task)
    build_random(task)
    build_baselines(task)
    build_manifest(task)
    build_edit(task)


def check(task):
    m = j(fd(task) / "test_manifest.json")
    ids = [a["id"] for a in m["arms"]]
    print(len(ids), "arms, unique", len(set(ids)) == len(ids), "bad ids", [i for i in ids if not re.fullmatch(r"[A-Za-z0-9._-]+", i)])
    from collections import Counter

    print(dict(Counter(a["role"] for a in m["arms"])))
    for a in m["arms"]:
        for o in a.get("ops", []):
            assert o["op"] == "scale_head" and 0 <= o["head"] < 8, a
    rs = [json.dumps(a["ops"], sort_keys=True) for a in m["arms"] if a["role"] == "random"]
    print("random distinct", len(set(rs)) == len(rs), len(rs))
    print("dose", [(a["id"], a["ops"]) for a in m["arms"] if a["role"] == "dose"])
    e = j(fd(task) / "edit.json")
    print("edit", e["candidate"], e["edited_checkpoint_sha256"][:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "random", "baselines"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.task)
    elif a.cmd == "random":
        build_random(a.task)
    elif a.cmd == "baselines":
        build_baselines(a.task)
    else:
        check(a.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
