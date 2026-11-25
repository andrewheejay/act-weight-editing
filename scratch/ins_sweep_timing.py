"""Timing of meet and pin per sweep arm (hypotheses.md 2026-11-25): run from the repo root, PYTHONPATH=. python scratch/ins_sweep_timing.py"""
import gzip,json,numpy as np,collections
g=collections.defaultdict(list)
for f in ('sweep.jsonl.gz','sweep_half2.jsonl.gz'):
    for l in gzip.open('experiments/insertion/stage2/'+f,'rt'):
        r=json.loads(l); g[r['arm']].append(r)
def info(rows):
    m=[r['t_meet'] for r in rows if r['t_meet']>=0]; p=[r['t_pin'] for r in rows if r['t_pin']>=0]
    late=np.mean([r['t_pin']>375 for r in rows if r['t_pin']>=0]) if p else np.nan
    return dict(n=len(rows),meet=len(m),med_meet=float(np.median(m)),pin=len(p),med_pin=float(np.median(p)),late_pin_share=round(float(late),3),strict=round(float(np.mean([r['strict_success'] for r in rows])),3))
for a in ['original','cross.h0.x0','cross.h2.x0','enc3.h4.x2','enc0.h6.x2','enc3.h3.x0','enc2.h6.x0','enc2.h1.x0','enc1.h1.x2','enc3.h7.x2','cross.h4.x0','cross.h3.x0']:
    print(a,info(g[a]))
