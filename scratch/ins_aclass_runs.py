"""Class 4a (A) and strict (S) longest pin runs on the design default traces (needs the trace cache): PYTHONPATH=. python scratch/ins_aclass_runs.py"""
import numpy as np, glob, gzip, json
from awe.measure import longest_run
p='experiments/insertion/stage1/design_default.jsonl'
import os
rows=[json.loads(l) for l in (open(p) if os.path.exists(p) else gzip.open(p+'.gz','rt'))]
R={r['seed']:r for r in rows}
def runs(m):
    out=[];s=None
    for i,v in enumerate(m):
        if v and s is None: s=i
        if not v and s is not None: out.append((s,i)); s=None
    if s is not None: out.append((s,len(m)))
    return out
cls={'A':[],'S':[]}
for sd,r in R.items():
    if r['strict_success']: k='S'
    elif r['failure_class']=='4a': k='A'
    else: continue
    d=np.load(f'experiments/insertion/cache/traces/default/{sd}.npz')
    pin=d['pin'].astype(bool); 
    rr=runs(pin)
    s,e=max(rr,key=lambda x:x[1]-x[0])
    act=d['action']; dq=np.abs(np.diff(act,axis=0))
    v=lambda a,b,j: float(dq[a:b][:,j].max()) if b>a else np.nan
    cls[k].append(dict(seed=sd,t0=s,len=e-s,
        right_v=float(np.mean(dq[max(s,1):e-1][:,7:13])) if e-s>3 else np.nan,
        left_v=float(np.mean(dq[max(s,1):e-1][:,0:6])) if e-s>3 else np.nan,
        rg=float(act[s:e,13].min()), lg=float(act[s:e,6].min()),
        end_ax=float(np.abs((d['peg']-d['sock']))[min(e,399)].max()),
        end=e, pinlost=int(pin[-1]==0),
        rmove=float(np.linalg.norm(d['qpos'][min(e,400)][:14]-d['qpos'][s][:14])) ))
for k in cls:
    c=cls[k]; print(k,len(c))
    for key in ('t0','len','right_v','left_v','rg','lg','end','rmove'):
        x=np.array([q[key] for q in c]); print('  ',key,np.nanpercentile(x,[10,50,90]).round(4))
