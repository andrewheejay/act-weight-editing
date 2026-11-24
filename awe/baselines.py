"""Stage 3 step 4: the simple output-level baselines, SPEC.md Controls. Resumable, budget `baselines`.

python -m awe.baselines mean       freeze/mean_action.json, the mean executed action of the original at each step (1,000 design episodes)
python -m awe.baselines screen     every setting on the 200 screening seeds, stage3/baselines_screen.jsonl
python -m awe.baselines full       the best settings on the other 800 design seeds, stage3/baselines_full.jsonl
python -m awe.baselines report     stage3/baselines_summary.json (paired against the original, stage1/design_default.jsonl)

Rules fixed before the first baseline rollout.
- Screening seeds 20000 to 20199. Order by benchmark gain, then strict gain, then id. Top 5 of constant_offset, deviation_gain and
  simplest_fix each go on to all 1,000 design seeds. The final setting per family is the best benchmark gain on the 1,000 (ties as above).
- constant_offset, 80 settings: 10 joints (right arm 7 to 13, left waist, shoulder, elbow 0 to 2) x 8 shifts.
- deviation_gain, 36 settings: g in 0.8, 0.9, 1.1, 1.2, 1.3, 1.5 on the right arm, the left arm or both, each with and without the
  gripper joint (SPEC.md does not say whether an arm includes it, so both are tuned).
- simplest_fix, 40 settings: a shift of one right-arm target (waist 7, shoulder 8, elbow 9, wrist angle 11, gripper 13) only until the
  right gripper starts to close. Two free numbers (joint, size). Idea: the diagnosis says the gripper arrives a centimetre off and lands
  on the cube during the approach, so correct the approach and leave lift and handover alone. Chosen before any baseline rollout.
- shorter_chunk: 50 and 25, already run on all design seeds in stage 1. No new rollouts.
"""
import argparse
import glob
import json

import numpy as np

from . import pool
from .arms import make_spec
from .paths import cache_dir, exp
from .post import mean_path
from .sweep_report import find, refs, rows_of, stats

SCREEN, REST = range(20000, 20200), range(20200, 21000)
TOP = 5
ARM = (0.01, 0.02, 0.04, 0.08)
GRIP = (0.05, 0.1, 0.2, 0.4)
OFFSET_JOINTS = (7, 8, 9, 10, 11, 12, 13, 0, 1, 2)
FIX_JOINTS = (7, 8, 9, 11, 13)
GAINS = (0.8, 0.9, 1.1, 1.2, 1.3, 1.5)
LIMBS = {"right": range(7, 14), "left": range(0, 7), "both": range(0, 14)}


def _levels(j):
    lv = GRIP if j == 13 else ARM
    return [s * v for v in lv for s in (1, -1)]


def settings():
    """family -> {arm id: recipe}"""
    out = {"constant_offset": {}, "deviation_gain": {}, "simplest_fix": {}}
    for fam, kind, joints, tag in (("constant_offset", "offset", OFFSET_JOINTS, "off"), ("simplest_fix", "approach_offset", FIX_JOINTS, "app")):
        for j in joints:
            for d in _levels(j):
                out[fam][f"{tag}_j{j}_{'p' if d > 0 else 'm'}{abs(d):g}"] = {"kind": kind, "joint": j, "delta": d}
    for limb, js in LIMBS.items():
        for grip in (True, False):
            joints = [j for j in js if grip or j not in (6, 13)]
            for g in GAINS:
                out["deviation_gain"][f"gain_{limb}_{'grip' if grip else 'nogrip'}_{g:g}"] = {"kind": "gain", "g": g, "joints": joints}
    return out


def mean(task):
    files = sorted(glob.glob(str(cache_dir(task) / "traces" / "default" / "2*.npz")))
    assert len(files) == 1000, len(files)
    m = np.mean([np.load(f)["action"].astype(np.float64) for f in files], 0)
    p = mean_path(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "built_by": "python -m awe.baselines mean", "source": "executed actions of the original on design seeds 20000 to 20999, 100-step chunks (stage 1 traces)",
        "mean_action": np.round(m, 6).tolist()}) + "\n")
    print(p, m.shape)


def run_jobs(task, arms, seeds, name, purpose):
    sp = [make_spec(i, post=r) for i, r in arms.items()]
    pool.run(task, [(s, x, None) for s in sp for x in seeds], exp(task) / "stage3" / name, budget="baselines", purpose=purpose)


def screen(task):
    for fam, arms in settings().items():
        run_jobs(task, arms, SCREEN, "baselines_screen.jsonl", "baseline_screen")


def tables(task):
    d = exp(task)
    reach, lat = refs(task)
    base = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
    g = {}
    for name in ("baselines_screen.jsonl", "baselines_full.jsonl"):
        p = find(d / "stage3", name)
        if p.exists():
            for r in rows_of(p):
                g.setdefault(r["arm"], {})[r["seed"]] = r
    st = lambda rows: stats(base, rows, reach, lat)
    return {a: {"screen": st([r for s, r in rows.items() if s in SCREEN]),
                **({"all": st(list(rows.values()))} if len(rows) == 1000 else {})} for a, rows in g.items()}


def ranked(tab, ids, key):
    return sorted(ids, key=lambda a: (-tab[a][key]["gain"], -tab[a][key]["strict_gain"], a))


def full(task):
    tab = tables(task)
    for fam, arms in settings().items():
        assert all(tab.get(a, {}).get("screen", {}).get("n") == len(SCREEN) for a in arms), fam
        top = ranked(tab, arms, "screen")[:TOP]
        print(fam, top)
        run_jobs(task, {a: arms[a] for a in top}, REST, "baselines_full.jsonl", "baseline_full")


def report(task):
    d = exp(task)
    tab = tables(task)
    reach, lat = refs(task)
    base = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
    fams = {}
    for fam, arms in settings().items():
        ids = [a for a in arms if a in tab]
        done = [a for a in ids if "all" in tab[a]]
        fams[fam] = {"n_settings": len(arms), "screened": len(ids), "full": ranked(tab, done, "all"), "best": (ranked(tab, done, "all") or [None])[0],
                     "recipes": arms, "results": {a: tab[a] for a in ids}}
    chunk = {}
    for n in (50, 25):
        rows = rows_of(find(d / "stage1", f"design_chunk{n}.jsonl"))
        chunk[f"chunk{n}"] = {"n_action_steps": n, "all": stats(base, rows, reach, lat)}
    fams["shorter_chunk"] = {"n_settings": 2, "results": chunk, "best": max(chunk, key=lambda c: (chunk[c]["all"]["gain"], chunk[c]["all"]["strict_gain"]))}
    (d / "stage3" / "baselines_summary.json").write_text(json.dumps({"rule": __doc__, "families": fams}, indent=1) + "\n")
    for fam, f in fams.items():
        print(fam, "best", f["best"])
        for a in (f.get("full") or []):
            print("  ", a, f["results"][a]["all"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mean", "screen", "full", "report"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    {"mean": mean, "screen": screen, "full": full, "report": report}[a.cmd](a.task)
