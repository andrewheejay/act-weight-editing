"""Wiring check of awe/baselines_ins.py: 3 recipes on 4 design seeds through the pool (12 rollouts, budget baselines, purpose smoke)."""
import json

from awe import baselines_ins as B, pool
from awe.arms import make_spec
from awe.paths import cache_dir

IDS = ("late_j8_p0.08", "gain_both_grip_1.2", "off_j9_m0.08")

if __name__ == "__main__":
    S = {a: r for fam in B.settings().values() for a, r in fam.items()}
    out = cache_dir("insertion") / "baseline_smoke.jsonl"
    pool.run("insertion", [(make_spec(i, post=S[i]), s, None) for i in IDS for s in range(20000, 20004)], out, budget="baselines", purpose="smoke", workers=4)
    for r in map(json.loads, open(out)):
        print(r["arm"], r["seed"], r["strict_success"], r["benchmark_success"], r["traj_hash"][:8], r["t_meet"])
