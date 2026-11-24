import numpy as np, pandas as pd
df=pd.read_pickle("/tmp/df2.pkl")
T="experiments/transfer_cube/cache/traces/default/%d.npz"
g=df.grasp_fail; ok=df.ok
steps=[40,60,70,80,90,95,99,105,110,120,130,140,150]
acc={"gf":[],"ok":[]}
for nm,ss in (("gf",df.index[g]),("ok",df.index[ok][:300])):
    for s in ss:
        z=np.load(T%s); acc[nm].append((z["tcpr"]-z["cube"])[steps])
for nm in acc:
    a=np.stack(acc[nm]); 
    print(nm, len(a)); print(pd.DataFrame(a.mean(0),index=steps,columns=list("xyz")).round(3).T.to_string())
    print("  std", pd.DataFrame(a.std(0),index=steps,columns=list("xyz")).round(3).T.to_string())
