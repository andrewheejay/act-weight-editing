import sys, json, gzip
import numpy as np
sys.path.insert(0, ".")
from pathlib import Path
from awe.paths import cache_dir
from awe.pool import trace_name
from awe.stage1_analysis import auc
S1 = Path("experiments/insertion/stage1")
TR = cache_dir("insertion")/"traces"/"default"
def load(n):
    with gzip.open(S1/f"features_{n}.jsonl.gz","rt") as fh: return [json.loads(l) for l in fh]
F = load("default")
def tip_off(z):
    peg, pR, so, sR = [z[k].astype(float) for k in ("peg","pegR","sock","sockR")]
    ax = pR[:, :, 0]                      # peg long axis, world
    # leading end is the one closer to the socket centre
    e1, e2 = peg + 0.06*ax, peg - 0.06*ax
    d1 = np.linalg.norm(e1 - so, axis=1); d2 = np.linalg.norm(e2 - so, axis=1)
    tip = np.where((d1 < d2)[:, None], e1, e2)
    return np.einsum("tij,ti->tj", sR, tip - so)
if __name__ == "__main__":
    out = []
    for r in F:
        if r["kind"] not in "SAM" or r["t_meet"] is None or r["t_meet"] < 1: continue
        z = np.load(TR/trace_name(r["seed"], None))
        k = r["t_meet"] - 1
        d = tip_off(z)[k]
        out.append((r["kind"], r["seed"], d))
    kinds = np.array([o[0] for o in out]); D = np.array([o[2] for o in out])
    lat = np.abs(D[:, 1:]).max(1); rad = np.linalg.norm(D[:, 1:], axis=1)
    print("axial of tip in socket frame", np.percentile(np.abs(D[:,0]), [5,50,95]))
    for nm, v in (("cheb", lat), ("rad", rad)):
        print(nm, {k: np.round(np.percentile(v[kinds==k],[10,50,90])*100,2).tolist() for k in "SAM"},
              "AUC M/S", round(auc(v[(kinds=='M')|(kinds=='S')], (kinds[(kinds=='M')|(kinds=='S')]=='M')),3),
              "M/SA", round(auc(v[kinds!=''], kinds=='M'),3), "AM/S", round(auc(v, kinds!='S'),3))
    # signs
    for i,nm in ((1,"y"),(2,"z")):
        print(nm, {k: np.round([np.mean(D[kinds==k,i])*100, np.std(D[kinds==k,i])*100],2).tolist() for k in "SAM"})
    # fraction of M within tolerance 0.8 cm
    print("frac lat<0.8cm:", {k: round(float((lat[kinds==k]<0.008).mean()),3) for k in "SAM"})
