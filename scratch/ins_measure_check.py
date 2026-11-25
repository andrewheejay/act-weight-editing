"""Recompute the new insertion measurements from stage 1 traces and compare with the stage 1 feature and depth files. No rollouts."""
import gzip, json, sys
import numpy as np
sys.path.insert(0, ".")
from awe import measure
from awe.paths import cache_dir, exp
S1 = exp("insertion") / "stage1"
rd = lambda p: [json.loads(l) for l in gzip.open(p, "rt")]
F = {r["seed"]: r for r in rd(S1 / "features_default.jsonl.gz")}
D = {r["seed"]: r for r in rd(S1 / "depth_default.jsonl.gz")}
R = {r["seed"]: r for r in rd(S1 / "runs_default.jsonl.gz")}
n, dm, dr, dd, dn = 0, 0., 0, 0, 0
for s, f in F.items():
    z = np.load(cache_dir("insertion") / "traces" / "default" / f"{s}.npz")
    arr = {k: (z[k].astype(bool) if k in ("left","right","table","pegtable","socktable","pin","pegsock","leftboth","rightboth") else z[k]) for k in z}
    arr.update(peg0=arr["peg"][0], sock0=arr["sock"][0], vis=[0]*4, visb=[0]*4)
    o = measure.insertion({k: (v.astype(float) if getattr(v, "dtype", None) == np.float32 else v) for k, v in arr.items()})
    n += 1
    if ("pre_tip_lat" in f) != ("pre_tip_lat" in o): dn += 1
    if "pre_tip_lat" in f: dm = max(dm, abs(f["pre_tip_lat"] - o["pre_tip_lat"]))
    if s in R: dr += o["strict_run"] != R[s]["strict_run"]
    if s in D:
        a, b = D[s]["g18"], o["deep_in_ax"]
        dd += (a is None) != (b is None) or (a is not None and abs(a - b) > 2e-5)
print("seeds", n, "met mismatch", dn, "max pre_tip_lat diff (m)", dm, "strict_run mismatch", dr, "of", len(R), "deep_in_ax mismatch", dd, "of", len(D))
