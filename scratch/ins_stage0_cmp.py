"""Insertion stage 0 comparisons: repeats, sham, renderers, stock. Recomputes every figure quoted in the notebook."""
import json, sys
from pathlib import Path
R=Path(__file__).resolve().parent.parent/"experiments/insertion/stage0"
def L(p): return {json.loads(l)["seed"]:json.loads(l) for l in open(R/p)}
K=("benchmark_success","strict_success","traj_hash")
h=L("harness_repro.jsonl"); a,b,s=(L(f"{n}.jsonl") for n in ("repeat_a","repeat_b","sham"))
for n,(x,y) in {"a-b":(a,b),"a-sham":(a,s),"a-harness500":(a,h)}.items():
    c=sorted(set(x)&set(y)); print(n,len(c),"seeds, differing",[i for i in c if any(x[i][k]!=y[i][k] for k in K)])
def pair(n,x,y):
    c=sorted(set(x)&set(y)); f=lambda k: sum(x[i][k]==y[i][k] for i in c)
    print(n,len(c),"bench agree %.1f%%"%(100*f("benchmark_success")/len(c)),"strict agree %.1f%%"%(100*f("strict_success")/len(c)),"max_reward agree %.1f%%"%(100*f("max_reward")/len(c)),"identical hash",f("traj_hash"),"bench rates %.1f %.1f"%tuple(100*sum(r["benchmark_success"] for r in q.values())/len(q) for q in (x,y)))
p1,p2=L("renderer_compare/gpu.jsonl"),L("renderer_compare/gpu_run2.jsonl")
pair("gpu vs gpu2",p1,p2); pair("gpu vs mesa",p1,h); pair("gpu2 vs mesa",p2,h)
import collections
for n,q in (("mesa",h),("gpu",p1)): print(n,sorted(collections.Counter(r["failure_class"] for r in q.values()).items()),"strict",sum(r["strict_success"] for r in q.values()))
