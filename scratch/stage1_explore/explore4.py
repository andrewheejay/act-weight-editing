import numpy as np, pandas as pd, sys
sys.path.insert(0,".")
from awe.stage1_analysis import load_features, auc
F=pd.DataFrame(load_features("transfer_cube","default"))
F["ok"]=F.benchmark_success
rel=np.array(list(F.reach_rel)); F[["rx","ry","rz"]]=rel
mu=F.loc[F.ok,["rx","ry","rz"]].mean().values
F["re"]=np.linalg.norm(rel-mu,axis=1)
F["re_xy"]=np.hypot(F.rx-mu[0],F.ry-mu[1])
g=F.grasp_fail; ok=F.ok
for k in ["re","re_xy","lat","fwd"]:
    x=F[k]; print(k,"AUC gf vs ok",round(auc(x[ok|g],g[ok|g]),3))
print("abs lat AUC", auc(F.lat.abs()[ok|g], g[ok|g]))
F["cx"]=F.cube_x0-0.1; F["cy"]=F.cube_y0-0.5
# regress rel on position for ok only and all
A=np.c_[np.ones(ok.sum()),F.cx[ok],F.cy[ok]]
for c in ["rx","ry","rz"]:
    b,*_=np.linalg.lstsq(A,F.loc[ok,c].values,rcond=None); r=F.loc[ok,c].values-A@b
    print(c,"ok: intercept %.4f, slope_x %.3f, slope_y %.3f, resid sd %.4f"%(b[0],b[1],b[2],r.std()))
# magnitude by grasp_fail probability vs position (low y)
for nm,m in [("y<0.44",F.cube_y0<0.44),("y>=0.44",F.cube_y0>=0.44)]:
    d=F[m]; print(nm,len(d),"gf",d.grasp_fail.mean().round(3),"re median ok",d.re[d.ok].median().round(4),"re median gf",d.re[d.grasp_fail].median().round(4), "AUC in zone", round(auc(d.re[d.ok|d.grasp_fail],d.grasp_fail[d.ok|d.grasp_fail]),3))
# lat, fwd by group
for nm,m in [("ok",ok),("gf",g),("lift",F.lift_fail)]:
    d=F[m]; print(nm,len(d),"lat med %.4f iqr %.4f fwd med %.4f t_touch %g knock med %.4f"%(d.lat.median(), d.lat.quantile(.75)-d.lat.quantile(.25), d.fwd.median(), d.t_touch.median(), d.knock_m.median()))
F.to_pickle("/tmp/F.pkl")
