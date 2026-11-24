import numpy as np, pandas as pd
df=pd.read_pickle("/tmp/df2.pkl")
d2=df[(df.failure_class=="2")&(df.zx>0.06)]
print("class2 real lift:", len(d2), "dropped later:", (d2.tdrop>=0).sum(), "held to end:", ((d2.tdrop<0)).sum(), "final_z>0.1:", (d2.final_z>0.1).sum())
d4=df[df.failure_class=="4a"]
print("4a", len(d4), "dropped", (d4.tdrop>=0).sum())
print(d2.groupby(d2.tdrop>=0)[["t_realift","lift_steps","min_dl","t_lclose","t_left_touch","final_z","cube_x0","cube_y0"]].median().round(3))
for nm,m in [("low-lift fails (grasp) cls2", (df.failure_class=="2")&(df.zx<=0.06)), ("class1", df.failure_class=="1"), ("class0", df.failure_class=="0")]:
    d=df[m]; print(nm, len(d), "early touch (<t_rclose-20):", (d.touch_gap>20).sum(), "median touch_gap", d.touch_gap.median(), "knock median", d.knock_m.median().round(4))
ok=df[df.ok]
print("ok early touch share", (ok.touch_gap>20).mean().round(3), "ok knock median", ok.knock_m.median().round(4))
print("gf median t_rclose", df.t_rclose[df.grasp_fail].median())
