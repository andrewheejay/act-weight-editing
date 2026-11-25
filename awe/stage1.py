"""Stage 1 rollouts on the design seeds, and the 41 by 41 cube grid. Resumable, every rollout goes to the ledger.

python -m awe.stage1 runs [--task T] [--only default,chunk50,...]     the job, in the order of ORDER[T]
python -m awe.stage1 strips      frame strips of 30 failures and 10 successes, re-run with the camera on (hashes must match)
"""
import argparse
import json

import numpy as np

from . import pool
from .arms import make_spec
from .paths import cache_dir, exp

DESIGN = range(20000, 21000)
ENS_COEFF = 0.01
GRID_N = 41


def grid_poses():
    """Cube positions on an even 41 by 41 grid over the start square, with the seed sampler's height and heading."""
    xs, ys = np.linspace(0.0, 0.2, GRID_N), np.linspace(0.4, 0.6, GRID_N)
    return [[float(x), float(y), 0.05, 1.0, 0.0, 0.0, 0.0] for y in ys for x in xs]


# Insertion has two bodies (DEVIATIONS.md, 2026-11-25). One body is moved over an even 41 by 41 grid of its own start box, the other
# stays at the centre of its box. Pose order is peg then socket, 7 numbers each, as gym_aloha's sampler concatenates them.
INS_BOX = {"peg": ((0.1, 0.2), (0.4, 0.6)), "sock": ((-0.2, -0.1), (0.4, 0.6))}
INS_CENTRE = {"peg": (0.15, 0.5), "sock": (-0.15, 0.5)}


def insertion_grid_poses(moving):
    (x0, x1), (y0, y1) = INS_BOX[moving]
    out = []
    for y in np.linspace(y0, y1, GRID_N):
        for x in np.linspace(x0, x1, GRID_N):
            at = dict(INS_CENTRE)
            at[moving] = (float(x), float(y))
            out.append([at["peg"][0], at["peg"][1], 0.05, 1.0, 0.0, 0.0, 0.0, at["sock"][0], at["sock"][1], 0.05, 1.0, 0.0, 0.0, 0.0])
    return out


def settings(task="transfer_cube"):
    """name -> (file, arm spec, seeds or None, poses or None)"""
    d = lambda **kw: make_spec("original", **kw)
    ens = d(temporal_ensemble_coeff=ENS_COEFF)
    if task == "insertion":
        return {
            "default": ("design_default.jsonl", d(), DESIGN, None),
            "chunk50": ("design_chunk50.jsonl", d(n_action_steps=50), DESIGN, None),
            "chunk25": ("design_chunk25.jsonl", d(n_action_steps=25), DESIGN, None),
            "ens200": ("design_ens.jsonl", ens, range(20000, 20200), None),
            "grid_peg": ("grid_peg_default.jsonl", d(), None, insertion_grid_poses("peg")),
            "grid_sock": ("grid_sock_default.jsonl", d(), None, insertion_grid_poses("sock")),
        }
    return {
        "default": ("design_default.jsonl", d(), DESIGN, None),
        "chunk50": ("design_chunk50.jsonl", d(n_action_steps=50), DESIGN, None),
        "chunk25": ("design_chunk25.jsonl", d(n_action_steps=25), DESIGN, None),
        "grid": ("grid_default.jsonl", d(), None, grid_poses()),
        "ens200": ("design_ens.jsonl", ens, range(20000, 20200), None),
        "ens": ("design_ens.jsonl", ens, DESIGN, None),
    }


ORDER = {"transfer_cube": "default,chunk50,chunk25,grid,ens200,ens", "insertion": "default,chunk50,chunk25,ens200,grid_peg,grid_sock"}


def run(task, names):
    st = settings(task)
    for name in names:
        fname, spec, seeds, poses = st[name]
        jobs = [(spec, s, None) for s in seeds] if seeds is not None else [(spec, None, p) for p in poses]
        pool.run(task, jobs, exp(task) / "stage1" / fname, budget="agent", purpose=f"stage1 {name}",
                 trace_dir=cache_dir(task) / "traces" / ("ens" if name.startswith("ens") else name))


def choose_strip_seeds(task):
    """30 failures (by kind, in proportion to their counts) and 10 strict successes, drawn with a fixed random seed."""
    from .stage1_analysis import load_features

    F = load_features(task, "default")
    rng = np.random.default_rng(0)
    kinds = {"class0": [r for r in F if r["grasp_fail"] and r["failure_class"] == "0"],
             "class1": [r for r in F if r["grasp_fail"] and r["failure_class"] == "1"],
             "class2_not_lifted": [r for r in F if r["grasp_fail"] and r["failure_class"] == "2"],
             "class2_lifted": [r for r in F if r["lift_fail"]]}
    total = sum(len(v) for v in kinds.values())
    want, picked = {}, []
    for k, v in kinds.items():
        want[k] = max(1, round(30 * len(v) / total))
    while sum(want.values()) != 30:  # fix rounding on the largest kind
        k = max(want, key=lambda k: len(kinds[k]))
        want[k] += 30 - sum(want.values())
    for k, v in kinds.items():
        idx = sorted(rng.choice(len(v), want[k], replace=False))
        picked += [(k, v[i]["seed"]) for i in idx]
    good = [r for r in F if r["strict_success"]]
    picked += [("success", good[i]["seed"]) for i in sorted(rng.choice(len(good), 10, replace=False))]
    return picked


def strips(task):
    from PIL import Image

    s1 = exp(task) / "stage1"
    d = s1 / "strips"
    d.mkdir(exist_ok=True)
    picked = choose_strip_seeds(task)
    kind_of = {s: k for k, s in picked}
    spec = make_spec("original")
    ref = {json.loads(l)["seed"]: json.loads(l)["traj_hash"] for l in (s1 / "design_default.jsonl").read_text().splitlines()}
    info = {}

    def save(rec, frames):
        f = [Image.fromarray(x).resize((200, 150)) for x in frames]
        w = Image.new("RGB", (200 * len(f), 150))
        for i, x in enumerate(f):
            w.paste(x, (200 * i, 0))
        name = f"{kind_of[rec['seed']]}_{rec['seed']}_class{rec['failure_class']}.jpg"
        w.save(d / name, quality=80)
        info[rec["seed"]] = {"kind": kind_of[rec["seed"]], "file": name, "class": rec["failure_class"], "hash_matches_design_default": rec["traj_hash"] == ref[rec["seed"]]}

    out = s1 / "strips_runs.jsonl"
    if not out.exists() or len(out.read_text().splitlines()) < len(picked):
        pool.run(task, [(spec, s, None) for _, s in picked], out, budget="agent", purpose="stage1 strips", frames_cb=save, n_frames=9)
        (s1 / "strips_index.json").write_text(json.dumps({str(k): v for k, v in sorted(info.items())}, indent=1))
    print(len(list(d.glob("*.jpg"))), "strips,", sum(v["hash_matches_design_default"] for v in info.values()), "match design_default hashes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["runs", "strips"])
    ap.add_argument("--task", default="transfer_cube")
    ap.add_argument("--only")
    a = ap.parse_args()
    a.only = a.only or ORDER[a.task]
    if a.cmd == "strips":
        strips(a.task)
    else:
        run(a.task, a.only.split(","))


if __name__ == "__main__":
    main()
