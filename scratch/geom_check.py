import json
from awe import pool
from awe.arms import make_spec
if __name__=="__main__":
    seeds=list(range(20000,20024))
    pool.run("transfer_cube",[(make_spec("original"),s,None) for s in seeds],"scratch/geom_check.jsonl",budget="agent",purpose="check cmd_lat8 field")
    import gzip
    F={r["seed"]:r for r in map(json.loads,gzip.open("experiments/transfer_cube/stage1/features_default.jsonl.gz","rt"))}
    H={r["seed"]:r for r in map(json.loads,open("experiments/transfer_cube/stage1/design_default.jsonl"))}
    bad=0; md=0
    for l in open("scratch/geom_check.jsonl"):
        r=json.loads(l); f=F[r["seed"]]; h=H[r["seed"]]
        ok=all(r[k]==h[k] for k in h)
        d=abs(r["cmd_lat8"]-f["cmd_lat_h"]["0.08"]) if r["cmd_lat8"] is not None else (0 if f["cmd_lat_h"]["0.08"] is None else 9)
        z=abs(r["zmax20"]-f["zmax20"]); md=max(md,d,z)
        bad+= (not ok) or r["cmd_clean"]!=f["cmd_clean"] or r["grasp_fail"]!=f["grasp_fail"]
    print("bad",bad,"maxdiff",md)
