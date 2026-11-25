import sys, json, gzip
import numpy as np
sys.path.insert(0, ".")
from pathlib import Path
from awe.stage1_analysis import auc
S1 = Path("experiments/insertion/stage1")
def load(n):
    with gzip.open(S1/f"features_{n}.jsonl.gz","rt") as fh: return [json.loads(l) for l in fh]
def spear(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return np.corrcoef(ra,rb)[0,1]
for n in ["default","grid_peg","grid_sock"]:
    F=load(n); G=[r for r in F if r["kind"] in "SAM" and "pre_tip_lat" in r]
    lat=np.array([r["pre_tip_lat"] for r in G]); kind=np.array([r["kind"] for r in G])
    px=np.array([r["peg_x0"] for r in G])-0.15; py=np.array([r["peg_y0"] for r in G])-0.5
    sx=np.array([r["sock_x0"] for r in G])+0.15; sy=np.array([r["sock_y0"] for r in G])-0.5
    print(n, len(G), "spearman lat vs: peg_dist", round(spear(lat,np.hypot(px*2,py)),3), "sock_dist", round(spear(lat,np.hypot(sx*2,sy)),3), "px",round(spear(lat,px),3),"py",round(spear(lat,py),3),"sx",round(spear(lat,sx),3),"sy",round(spear(lat,sy),3))
    M=kind=="M"
    print("  AUC M/S(+A)", round(auc(lat, M),3), " by kind medians cm", {k: round(100*float(np.median(lat[kind==k])),2) for k in "SAM" if (kind==k).any()})
    # M rate and lat by quantile of peg dist & sock dist
    for nm,v in (("peg_x",px),("peg_y",py),("sock_x",sx),("sock_y",sy)):
        ed=np.quantile(v,[0,.2,.4,.6,.8,1]); b=np.clip(np.searchsorted(ed,v)-1,0,4)
        print("   ",nm,"meet-M-share by fifth",[round(float(np.mean(M[b==i])),2) for i in range(5)],"median lat cm",[round(100*float(np.median(lat[b==i])),2) for i in range(5)])
print("=== AUC M vs S and A vs S, by run; AUC on other numbers")
for n in ["default","chunk50","chunk25","ens","grid_peg","grid_sock"]:
    F=load(n); G=[r for r in F if r["kind"] in "SAM" and "pre_tip_lat" in r]
    kind=np.array([r["kind"] for r in G]); out={}
    for key in ["pre_tip_lat","pre_lat","pre_ang","pre_tip_lat_20","post_min_tip_ax"]:
        v=np.array([r[key] for r in G])
        res={}
        for nm,pos,neg in (("M/S",["M"],["S"]),("A/S",["A"],["S"]),("M/SA",["M"],["S","A"])):
            m=np.isin(kind,pos+neg)
            if len(set(kind[m]))<2: continue
            res[nm]=round(float(auc(v[m], np.isin(kind[m],pos))),3)
        out[key]=res
    print(n, {k:int((kind==k).sum()) for k in "SAM"}, out)
