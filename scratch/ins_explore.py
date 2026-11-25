import gzip, json, sys
import numpy as np
sys.path.insert(0, ".")
from pathlib import Path
from awe.stage1_analysis import auc
S1 = Path("experiments/insertion/stage1")
def load(n):
    with gzip.open(S1/f"features_{n}.jsonl.gz","rt") as fh: return [json.loads(l) for l in fh]
F = load("default")
K = ["S","A","M","G2","G1","G0"]
def q(v): 
    v=np.array([x for x in v if x is not None and np.isfinite(x)],float)
    return None if len(v)==0 else [round(float(x),4) for x in np.percentile(v,[10,50,90])]
for key in ["pre_lat","pre_ax","pre20_lat","min_axial","lat_at_min_axial","ang_at_min_axial","axial_t","t_meet","t_pin","peg_z_max","sock_z_max","held_steps","meet_lat","pre_ang"]:
    print(key, {k: q([r.get(key) for r in F if r["kind"]==k]) for k in K})
