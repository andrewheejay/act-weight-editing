"""Stage 1 analysis: features from the traces, the figures and the Gate 1 decisions.

python -m awe.stage1_analysis features NAME     stage1/features_NAME.jsonl.gz from design_NAME / grid_NAME records and traces
python -m awe.stage1_analysis stats             the numbers for diagnosis.md, stage1/stage1_stats.json
python -m awe.stage1_analysis figures           failure_map.png, reach_error.png, meet_error.png
python -m awe.stage1_analysis decisions         stage1/decisions.json (the Gate 1 rules, referee/RULES.md)

Labels, per episode (the benchmark score decides failure):
  zmax20      highest cube centre height after step 20 (the first steps hold the 3 cm reset drop, so they are skipped)
  grasp_fail  benchmark failure and zmax20 <= 0.06 m. The cube never left the table area. Includes classes 0, 1 and the
              class 2 episodes where a finger touched the cube off the table only as a bump
  lift_fail   benchmark failure and zmax20 > 0.06 m. The cube was really lifted, the left gripper did not finish
Geometry in the gripper frame: columns of the right gripper_link rotation, origin at the mean of the two finger links.
  lat         cube offset along the axis from the left to the right finger body (finger spacing about 0.11 m)
  fwd         cube offset along the gripper x axis (approach direction)
"""
import argparse
import gzip
import json
import math

import numpy as np

from . import render
from .paths import cache_dir, exp

LIFT_Z = 0.06
SETTLE = 20


def load_records(task, name):
    f = exp(task) / "stage1" / f"design_{name}.jsonl"
    if name == "grid":
        f = exp(task) / "stage1" / "grid_default.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def load_trace(task, name, rec):
    from .pool import trace_name
    d = cache_dir(task) / "traces" / ("ens" if name == "ens" else name)
    return np.load(d / trace_name(rec.get("seed"), rec.get("pose")))


def gripper_frames(task):
    from .env import FastAloha
    env = FastAloha(task)
    ph = env.physics
    gb = env.bid["vx300s_right/gripper_link"]
    lf, rf = env.bid["vx300s_right/left_finger_link"], env.bid["vx300s_right/right_finger_link"]

    def at(q):
        with ph.reset_context():
            ph.data.qpos[:] = q
        R = np.array(ph.data.xmat[gb]).reshape(3, 3)
        a, b = np.array(ph.data.xpos[lf]), np.array(ph.data.xpos[rf])
        return R, a, b

    return at


def feature_row(rec, z, frame_at):
    cube, tr, act = z["cube"], z["tcpr"], z["action"]
    q = z["qpos"]
    d = {"seed": rec.get("seed"), "pose": rec.get("pose"), "failure_class": rec["failure_class"],
         "benchmark_success": rec["benchmark_success"], "strict_success": rec["strict_success"],
         "cube_x0": rec["cube_x0"], "cube_y0": rec["cube_y0"], "t_rclose": rec["t_rclose"], "t_touch": rec["t_touch"],
         "knock_m": rec["knock_m"], "dual_hold": rec["dual_hold_steps"], "t_lclose": rec["t_lclose"], "vis_px": rec["vis_px"]}
    zs = cube[SETTLE:, 2]
    d["zmax20"] = float(zs.max())
    d["final_z"] = float(cube[-1, 2])
    lifted = (cube[:, 2] > LIFT_Z)
    lifted[:SETTLE] = False
    d["t_lift60"] = int(np.argmax(lifted)) if lifted.any() else -1
    tl = d["t_lift60"]
    drop = np.nonzero(cube[tl:, 2] < 0.035)[0] if tl >= 0 else []
    d["t_drop"] = int(tl + drop[0]) if len(drop) else -1
    d["min_left_dist"] = float(np.linalg.norm(z["tcpl"][SETTLE:] - cube[SETTLE:], axis=1).min())
    fail = not rec["benchmark_success"]
    d["grasp_fail"] = bool(fail and d["zmax20"] <= LIFT_Z)
    d["lift_fail"] = bool(fail and d["zmax20"] > LIFT_Z)
    t = rec["t_rclose"]
    if t is not None and t >= 0:
        rel = tr[t] - cube[t]
        d["reach_rel"] = [float(x) for x in rel]
        # the same gripper position against where the cube was BEFORE anything touched it (settled after the reset drop)
        d["aim_rel"] = [float(x) for x in tr[t] - cube[min(SETTLE, t)]]
        R, a, b = frame_at(q[t + 1])
        off = cube[t] - 0.5 * (a + b)
        ax = (b - a) / np.linalg.norm(b - a)
        d["fwd"] = float((R.T @ off)[0])
        d["lat"] = float(off @ ax)
        d["spacing"] = float(np.linalg.norm(b - a))
    tt = rec["t_touch"]
    if tt is not None and tt >= 1:
        k = tt - 1  # the last step before the first right-finger contact, the cube is still where it was
        R, a, b = frame_at(q[k + 1])
        off = cube[k] - 0.5 * (a + b)
        ax = (b - a) / np.linalg.norm(b - a)
        d["touch_lat"] = float(off @ ax)
        d["touch_fwd"] = float((R.T @ off)[0])
        d["touch_height"] = float(tr[k][2] - cube[k][2])
        b0 = cube[min(SETTLE, k)]  # the cube has settled from the reset drop by then
        d["pre_touch_cube_shift_m"] = float(np.linalg.norm(cube[k][:2] - b0[:2]))
    if "meet_rel" in rec:
        d["meet_rel"] = rec["meet_rel"]
    return d


def cmd_features(task, name):
    frame_at = gripper_frames(task)
    rows = [feature_row(r, load_trace(task, name, r), frame_at) for r in load_records(task, name)]
    out = exp(task) / "stage1" / f"features_{name}.jsonl.gz"
    with gzip.GzipFile(out, "wb", mtime=0) as fh:
        for r in sorted(rows, key=lambda r: (r["seed"] is None, r["seed"] or 0, str(r["pose"]))):
            fh.write((json.dumps(r) + "\n").encode())
    print(name, len(rows), "->", out)


def load_features(task, name):
    with gzip.open(exp(task) / "stage1" / f"features_{name}.jsonl.gz", "rt") as fh:
        return [json.loads(l) for l in fh]


def auc(x, y):
    """P(x of a positive > x of a negative), ties at half weight. Mann-Whitney U over n1 n0."""
    x, y = np.asarray(x, float), np.asarray(y, bool)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    order = x.argsort(kind="stable")
    ranks = np.empty(len(x))
    i = 0
    sx = x[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "stats", "figures", "decisions"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    render.configure(render.chosen(a.task))
    if a.cmd == "features":
        cmd_features(a.task, a.name or "default")
    else:
        from . import stage1_report

        getattr(stage1_report, "cmd_" + a.cmd)(a.task)
