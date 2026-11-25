"""Smoke test of the insertion atlas on a small frame subset (a temporary cache and output folder, no rollouts). Wiring only: numbers here are not results."""
import sys
sys.path.insert(0, ".")
from awe import atlas_ins as A, render
import awe.paths, awe.atlas_ins_report as R
from pathlib import Path
import os, tempfile
tmp = Path(tempfile.mkdtemp(prefix="atlas_smoke_"))
(tmp / "insertion").mkdir()
os.environ["AWE_CACHE_SMOKE"] = str(tmp)
real = A.cache_dir
def cd(task):
    return tmp / task
def fl_small():
    poses, out = _fl()
    keep = [k for k in out if (k[1] % 7 == 0 or k[1] < 3)]
    return poses, keep
_fl = A.frame_list
if __name__ == "__main__":
    import numpy as np
    (tmp / "insertion" / "traces").symlink_to(real("insertion") / "traces")
    A.cache_dir = cd
    A.frame_list = fl_small
    A.render.configure(A.render.chosen("insertion"))
    A.GRID_N  # unchanged; report cannot run on a subset, so only frames and effects are exercised
    A.cmd_frames("insertion")
    m = np.load(tmp / "insertion" / "atlas" / "meta.npz")
    print({k: m[k].shape for k in m}, "nan centroids", int(np.isnan(m["cen"]).any((1, 2)).sum()))
    feats = np.load(tmp / "insertion" / "atlas" / "feats.npy", mmap_mode="r")
    F = A.Fwd("insertion")
    meta = {k: m[k] for k in m}
    rec = A.check_fwd("insertion", F, meta, feats, n=40)
    for s in A.STEPS:
        e = [x for q, x in rec if q == s]
        print("step", s, "n", len(e), "max", max(e), "median", float(np.median(e)))
    print("tmp", tmp)
