"""Effects on the smoke subset (a few variants), then the report on a SYNTHETIC raw.npz (random numbers, wiring only). Writes only under a temp folder."""
import sys, time
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
from awe import atlas_ins as A, atlas_ins_report as R, render
from awe.paths import exp
tmp = Path(sys.argv[1])
real = exp("insertion")
A.render.configure(A.render.chosen("insertion"))
e2 = tmp / "exp_insertion"
(e2 / "stage2" / "atlas").mkdir(parents=True, exist_ok=True)
if not (e2 / "stage1").exists():
    (e2 / "stage1").symlink_to(real / "stage1")
A.exp = R.exp = lambda t: e2
A.cache_dir = lambda t: tmp / t
fl0 = A.frame_list
A.frame_list = lambda: (fl0()[0], [k for k in fl0()[1] if (k[1] % 7 == 0 or k[1] < 3)])
vs = A.variants()
A.variants = lambda: [v for v in vs if v[0] in ("enc0.h3.x0", "enc2.h1.x2", "cross.h2.x0", "dself.h4.x2", "lat5.-2")]
t0 = time.time(); A.cmd_effects("insertion"); print("effects", time.time() - t0)
z = np.load(e2 / "stage2" / "atlas" / "raw.npz")
print({k: z[k].shape for k in z}); print(z["names"], "mean eff per variant", z["eff"].mean(1), "var_sum", z["var_sum"])
# synthetic full-size raw
rng = np.random.default_rng(0)
poses, fl = fl0()
N, V = len(fl), len(vs)
g = {k: np.array([x[j] for x in fl]) for j, k in enumerate(("grid", "index", "step"))}
np.savez_compressed(e2 / "stage2" / "atlas" / "raw.npz", names=np.array([v[0] for v in vs]), eff=rng.random((V, N)).astype(np.float16), grid=g["grid"], index=g["index"], step=g["step"],
    joint=rng.random((V, 8, 14)), timep=rng.random((V, 8, 10)), var_sum=rng.random((V + 1, 2)), attn=rng.random((8, 8, 302)), attn_query=rng.random((8, 8, 10, 302)),
    attn_cell=rng.random((N, 8, 2)), attn_cen=rng.random((N, 8, 2)), cen=rng.random((N, 2, 2)) * 400, on_cells=rng.integers(1, 9, (N, 2)), recheck=np.array([[0, .1], [100, .1], [200, .1], [300, .1]]))
t0 = time.time(); R.report("insertion"); print("report", time.time() - t0)
import os
for f in sorted((e2 / "stage2" / "atlas").iterdir()): print(f.name, f.stat().st_size)
