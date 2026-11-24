import sys; sys.path.insert(0,".")
from awe import render; render.configure("egl_mesa")
import numpy as np, pandas as pd
from awe.stage1_analysis import gripper_frames, load_trace, load_records, auc
at=gripper_frames("transfer_cube")
F=pd.read_pickle("/tmp/F.pkl").set_index("seed")
recs={r["seed"]:r for r in load_records("transfer_cube","default")}
rows=[]
for s,r in recs.items():
    z=load_trace("transfer_cube","default",r); t=r["t_touch"]
    if t<0 or t<1: continue
    tt=max(t-1,0)
    R,a,b=at(z["qpos"][tt+1]); off=z["cube"][tt]-0.5*(a+b); ax=(b-a)/np.linalg.norm(b-a)
    loc=R.T@off
    rows.append(dict(seed=s,lat=float(off@ax),fwd=float(loc[0]),lz=float(loc[2]),t=t, gz=z["tcpr"][tt][2]-z["cube"][tt][2]))
E=pd.DataFrame(rows).set_index("seed").join(F[["grasp_fail","lift_fail","benchmark_success"]])
for nm,m in (("ok",E.benchmark_success),("gf",E.grasp_fail),("lift",E.lift_fail)):
    d=E[m]; print(nm,len(d),"t",d.t.median(),"lat med %.4f sd %.4f |lat| med %.4f fwd med %.4f gz med %.4f"%(d.lat.median(),d.lat.std(),d.lat.abs().median(),d.fwd.median(),d.gz.median()))
g=E.grasp_fail; ok=E.benchmark_success
for k in ("lat","fwd","gz"):
    v=E[k].abs() if k=="lat" else E[k]
    print(k, "AUC gf vs ok (|lat| for lat)", round(auc(v[g|ok],g[g|ok]),3))
