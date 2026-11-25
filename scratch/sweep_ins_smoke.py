"""8 rollouts through the pool (4 original, 4 of one edit, design seeds), compared with stage 1, then sweep_ins_report.stats on them. Ledger: agent, purpose smoke."""
import json, sys, gzip
sys.path.insert(0, ".")
if __name__ == "__main__":
    from awe import pool, sweep, sweep_ins_report as S
    from awe.arms import make_spec
    from awe.paths import exp
    sp = sweep.single_head_specs()
    arms = [make_spec("original"), sp["cross.h1.x0"]]
    out = exp("insertion") / "cache" / "sweep_smoke.jsonl"
    out.unlink(missing_ok=True)
    pool.run("insertion", [(a, s, None) for a in arms for s in range(20000, 20004)], out, budget="agent", purpose="smoke", workers=4)
    R = [json.loads(l) for l in out.read_text().splitlines()]
    F = {r["seed"]: r for r in map(json.loads, gzip.open(exp("insertion") / "stage1" / "features_default.jsonl.gz", "rt"))}
    for r in R:
        if r["arm"] == "original":
            f = F[r["seed"]]
            print(r["seed"], "hash same", r["traj_hash"] == f["traj_hash"], "tip", r.get("pre_tip_lat"), f.get("pre_tip_lat"), "strict_run", r["strict_run"], "deep", r.get("deep_in_ax"), "bytes", len(json.dumps(r)))
    base = {r["seed"]: r for r in R if r["arm"] == "original"}
    ref = S.refs("insertion")
    print(S.stats(base, [r for r in R if r["arm"] != "original"], ref))
    print(S.stats(base, list(base.values()), ref)["gain"])
