import numpy as np, pandas as pd
df=pd.read_pickle("/tmp/df.pkl")
T="experiments/transfer_cube/cache/traces/default/%d.npz"
df["grasp_fail"]=df.failure_class.isin(["0","1"])|((df.failure_class=="2")&(df.zx<=0.06))
df["ok"]=df.benchmark_success
def auc(x,y):
    x=np.asarray(x,float); y=np.asarray(y,bool); m=~np.isnan(x); x=x[m]; y=y[m]
    r=pd.Series(x).rank().values
    n1=y.sum(); n0=(~y).sum()
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
rows=[]
for s in df.index:
    z=np.load(T%s); cube=z["cube"]; tr=z["tcpr"]; L=z["left"].astype(bool);R=z["right"].astype(bool); a=z["action"]
    rel=tr-cube
    t=df.t_rclose[s]
    d={}
    d["rel99"]=rel[99]; 
    tt=df.t_touch[s]
    d["touch_gap"]=t-tt if tt>=0 else np.nan
    # cube displacement direction at touch+10 from start
    d["kx"],d["ky"]=(cube[min(tt+20,399),:2]-cube[30,:2]) if tt>=0 else (np.nan,np.nan)
    # distance at step 99, cube pos
    d["d99"]=np.linalg.norm(rel[99][:2])
    for k in ("rclose",):
        d["reldx"],d["reldy"],d["reldz"]=rel[t]
    d["t"]=t
    rows.append(pd.Series(d,name=s))
E=pd.DataFrame(rows)
for k in ("rel99",): 
    v=np.stack(E[k].values); E["rel99x"],E["rel99y"],E["rel99z"]=v.T
df=df.join(E.drop(columns=["rel99","t"]))
ok=df.ok; g=df.grasp_fail
mu=df.loc[ok,["reldx","reldy","reldz"]].mean()
sg=df.loc[ok,["reldx","reldy","reldz"]].std()
df["reach_err_xy"]=np.hypot(df.reldx-mu.reldx, df.reldy-mu.reldy)
df["reach_err_z"]=(df.reldz-mu.reldz).abs()
df["reach_err"]=np.sqrt(((df[["reldx","reldy","reldz"]]-mu)**2).sum(axis=1))
for k in ["reach_err_xy","reach_err_z","reach_err","d99","knock_m","touch_gap"]:
    x=df[k]
    print(k, "AUC grasp_fail vs ok", round(auc(x[ok|g],g[ok|g]),3), "median ok", round(x[ok].median(),4), "gf", round(x[g].median(),4), "n touch nan", x.isna().sum())
df.to_pickle("/tmp/df2.pkl")
