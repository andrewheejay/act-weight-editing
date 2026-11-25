"""Stage 3 step 4 for insertion: the simple output-level baselines, SPEC.md Controls. Resumable, budget `baselines`.

python -m awe.baselines_ins mean       freeze/mean_action.json, the mean executed action of the original at each step (1,000 design episodes)
python -m awe.baselines_ins screen     every setting on the 200 screening seeds, stage3/baselines_screen.jsonl
python -m awe.baselines_ins full       the best settings on the other 800 design seeds, stage3/baselines_full.jsonl
python -m awe.baselines_ins report     stage3/baselines_summary.json (paired against the original, stage1/design_default.jsonl)

Rules fixed before the first baseline rollout for insertion (2026-11-25, session 23). The primary score is strict success, benchmark the guard.
- Screening seeds 20000 to 20199. Order by strict gain, then guard gain, then id. Top 5 of constant_offset, deviation_gain and
  simplest_fix each go on to all 1,000 design seeds. The final setting per family is the best strict gain on the 1,000 (ties as above).
- constant_offset, 80 settings: 10 joints (left waist 0, shoulder 1, elbow 2, wrist angle 4, right waist 7, shoulder 8, elbow 9, wrist
  angle 11, the two grippers 6 and 13) x 8 shifts.
- deviation_gain, 36 settings: g in 0.8, 0.9, 1.1, 1.2, 1.3, 1.5 on the right arm, the left arm or both, each with and without the gripper joints.
- simplest_fix, 48 settings: a shift of one arm joint (0 to 5 left, 7 to 12 right, 12 joints) from step 200 on (late_offset in awe/post.py), by
  +/-0.02 or +/-0.04 rad. Two free numbers, joint and size (the start step is fixed at 200, the decision that covers the approach). Idea: the
  target failure is the peg tip arriving about 2 cm off the socket axis at the first contact, so move one joint late, in the approach and
  after it, and leave the grasp and the lift alone. The sizes were set to 0.02 and 0.04 (first written 0.03 and 0.08) after a 4-seed wiring smoke, where +0.08 on
  the right shoulder removed every peg-socket meeting. No other setting was changed after the smoke.
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
from .sweep_ins_report import refs, stats
from .sweep_report import find, rows_of

SCREEN, REST = range(20000, 20200), range(20200, 21000)
TOP = 5
ARM = (0.01, 0.02, 0.04, 0.08)
GRIP = (0.05, 0.1, 0.2, 0.4)
OFFSET_JOINTS = (0, 1, 2, 4, 7, 8, 9, 11, 6, 13)
GAINS = (0.8, 0.9, 1.1, 1.2, 1.3, 1.5)
LATE_JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
LATE_DELTAS = (0.02, 0.04)
LATE_T0 = 200
LIMBS = {"right": range(7, 14), "left": range(0, 7), "both": range(0, 14)}
FAMILIES = ("constant_offset", "deviation_gain", "simplest_fix")


def _levels(j):
    lv = GRIP if j in (6, 13) else ARM
    return [s * v for v in lv for s in (1, -1)]


def settings():
    """family -> {arm id: recipe}"""
    out = {"constant_offset": {}, "deviation_gain": {}, "simplest_fix": {}}
    for j in OFFSET_JOINTS:
        for d in _levels(j):
            out["constant_offset"][f"off_j{j}_{'p' if d > 0 else 'm'}{abs(d):g}"] = {"kind": "offset", "joint": j, "delta": d}
    for limb, js in LIMBS.items():
        for grip in (True, False):
            joints = [j for j in js if grip or j not in (6, 13)]
            for g in GAINS:
                out["deviation_gain"][f"gain_{limb}_{'grip' if grip else 'nogrip'}_{g:g}"] = {"kind": "gain", "g": g, "joints": joints}
    for j in LATE_JOINTS:
        for d in LATE_DELTAS:
            for s in (1, -1):
                out["simplest_fix"][f"late_j{j}_{'p' if s > 0 else 'm'}{d:g}"] = {"kind": "late_offset", "joint": j, "delta": s * d, "t0": LATE_T0}
    return out


def mean(task):
    files = sorted(glob.glob(str(cache_dir(task) / "traces" / "default" / "2*.npz")))
    assert len(files) == 1000, len(files)
    m = np.mean([np.load(f)["action"].astype(np.float64) for f in files], 0)
    assert m.shape == (400, 14), m.shape
    p = mean_path(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "built_by": "python -m awe.baselines_ins mean", "source": "executed actions of the original on design seeds 20000 to 20999, 100-step chunks (stage 1 traces)",
        "mean_action": np.round(m, 6).tolist()}) + "\n")
    print(p, m.shape)


def run_jobs(task, arms, seeds, name, purpose):
    sp = [make_spec(i, post=r) for i, r in arms.items()]
    pool.run(task, [(s, x, None) for s in sp for x in seeds], exp(task) / "stage3" / name, budget="baselines", purpose=purpose)


def screen(task):
    for fam, arms in settings().items():
        run_jobs(task, arms, SCREEN, "baselines_screen.jsonl", "baseline_screen")


def base_rows(task):
    return {r["seed"]: r for r in rows_of(find(exp(task) / "stage1", "design_default.jsonl"))}


def tables(task):
    d = exp(task)
    ref, base = refs(task), base_rows(task)
    g = {}
    for name in ("baselines_screen.jsonl", "baselines_full.jsonl"):
        p = find(d / "stage3", name)
        if p.exists():
            for r in rows_of(p):
                g.setdefault(r["arm"], {})[r["seed"]] = r
    st = lambda rows: stats(base, rows, ref)
    return {a: {"screen": st([r for s, r in rows.items() if s in SCREEN]),
                **({"all": st(list(rows.values()))} if len(rows) == 1000 else {})} for a, rows in g.items()}


def ranked(tab, ids, key):
    return sorted(ids, key=lambda a: (-tab[a][key]["gain"], -tab[a][key]["guard_gain"], a))


def stage1_features(task, name):
    """Stage 1 rows lack the tip numbers, the feature files have them (same definition, within 5e-6 m)."""
    return {r["seed"]: r for r in rows_of(exp(task) / "stage1" / f"features_{name}.jsonl.gz")}


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
    ref, base = refs(task), base_rows(task)
    fams = {}
    for fam, arms in settings().items():
        ids = [a for a in arms if a in tab]
        done = [a for a in ids if "all" in tab[a]]
        fams[fam] = {"n_settings": len(arms), "screened": len(ids), "full": ranked(tab, done, "all"), "best": (ranked(tab, done, "all") or [None])[0],
                     "recipes": arms, "results": {a: tab[a] for a in ids}}
    chunk = {}
    for n in (50, 25):
        f = stage1_features(task, f"chunk{n}")
        rows = [{**r, **{k: f[r["seed"]][k] for k in ("pre_tip_lat", "pre_tip_d") if k in f[r["seed"]]}} for r in rows_of(find(d / "stage1", f"design_chunk{n}.jsonl"))]
        chunk[f"chunk{n}"] = {"n_action_steps": n, "all": stats(base, rows, ref)}
    fams["shorter_chunk"] = {"n_settings": 2, "results": chunk, "best": max(chunk, key=lambda c: (chunk[c]["all"]["gain"], chunk[c]["all"]["guard_gain"]))}
    (d / "stage3" / "baselines_summary.json").write_text(json.dumps({"rule": __doc__, "families": fams}, indent=1) + "\n")
    for fam, f in fams.items():
        print(fam, "best", f["best"])
        for a in (f.get("full") or []):
            t = f["results"][a]["all"]
            print("  ", a, t["gain"], t["guard_gain"], t["fixed"], t["broken"], t["strict"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mean", "screen", "full", "report"])
    ap.add_argument("--task", default="insertion")
    a = ap.parse_args()
    {"mean": mean, "screen": screen, "full": full, "report": report}[a.cmd](a.task)
