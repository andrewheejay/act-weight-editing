"""Process pool. One simulator and one GPU policy per worker process."""
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from . import render
from .ledger import Ledger
from .seeds import check_seed, seed_set

_W = {}


def default_workers() -> int:
    v = os.environ.get("AWE_WORKERS")
    return int(v) if v else min(8, os.cpu_count() or 4)


def _init(task, renderer, n_frames, trace_dir=None):
    render.configure(renderer)
    from .env import FastAloha
    from .policy import Policy

    _W.update(env=FastAloha(task), policy=Policy(task), n_frames=n_frames, trace_dir=trace_dir)


def trace_name(seed, pose) -> str:
    if seed is not None:
        return f"{seed}.npz"
    if len(pose) == 7:
        return f"pose_{pose[0]:.4f}_{pose[1]:.4f}.npz"
    return "pose_" + "_".join(f"{v:.4f}" for v in pose[:2]) + "_" + "_".join(f"{v:.4f}" for v in pose[7:9]) + ".npz"


def _save_trace(d, arr, seed, pose):
    """Per-step arrays for later analysis. A cache, outside git, rebuilt by re-running the rollout."""
    import numpy as np

    keys = ("action", "reward", "qpos", "left", "right", "table", "cube", "tcpl", "tcpr", "leftboth")
    keys += ("peg", "pegR", "sock", "sockR", "pin", "pegsock", "pegtable", "socktable", "rightboth")
    out = {k: (arr[k].astype(np.float32) if arr[k].dtype.kind == "f" else arr[k].astype(np.int8)) for k in keys if k in arr}
    tmp = Path(d) / (trace_name(seed, pose) + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **out)
    os.replace(tmp, Path(d) / trace_name(seed, pose))


def _run(job):
    from . import rollout

    spec, seed, pose = job
    if seed is not None:
        check_seed(seed)
    _W["policy"].set_arm(spec)
    arr = rollout.run_episode(_W["env"], _W["policy"], seed=seed, pose=pose, n_frames=_W["n_frames"])
    if _W.get("trace_dir"):
        _save_trace(_W["trace_dir"], arr, seed, pose)
    return rollout.summarize(arr, spec["id"], seed, pose), arr.get("frames")


def _key(arm, seed, pose):
    return (arm, seed, None if pose is None else json.dumps([float(x) for x in pose]))


def run(task, jobs, out, *, budget, purpose, workers=None, renderer=None, frames_cb=None, n_frames=0, trace_dir=None):
    """Run rollouts. jobs is a list of (arm_spec, seed or None, pose or None). Appends to `out`, skips what is there.

    trace_dir keeps the per-step arrays of every rollout as npz files (cache only).

    Every rollout gets a ledger line. The renderer is the recorded choice unless stage 0 compares renderers.
    """
    for _, seed, _ in jobs:
        if seed is not None:
            check_seed(seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(_key(r["arm"], r.get("seed"), r.get("pose")))
    todo = [j for j in jobs if _key(j[0]["id"], j[1], j[2]) not in done]
    if not todo:
        return 0
    renderer = renderer or render.chosen(task)
    led = Ledger(task, budget, purpose)
    led.check_budget(len(todo))
    nw = min(workers or default_workers(), len(todo))
    render.configure(renderer)
    t0, n = time.time(), 0
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    chunk = max(1, min(8, len(todo) // (nw * 4)))
    with ctx.Pool(nw, initializer=_init, initargs=(task, renderer, n_frames, None if trace_dir is None else str(trace_dir))) as pool, open(out, "a") as fh:
        for rec, frames in pool.imap_unordered(_run, todo, chunksize=chunk):
            if frames is not None and frames_cb:
                frames_cb(rec, frames)
            # ledger first, so a crash can leave an extra ledger line but never a missing one
            seed = rec.get("seed")
            cand = rec["arm"] if seed is not None and seed_set(seed) == "validation" else None
            led.log(rec["arm"], seed, rec.get("pose"), cand)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            if n % 200 == 0:
                print(f"{n}/{len(todo)} rollouts, {n / (time.time() - t0):.1f}/s", flush=True)
    led.close()
    dt = time.time() - t0
    print(f"{n} rollouts in {dt:.1f}s ({n / dt:.1f}/s, {nw} workers, {renderer}) -> {out}", flush=True)
    return n
