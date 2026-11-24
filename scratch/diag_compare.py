"""Pairwise agreement of benchmark success and of the trajectory hash between the numerics variants, the stock runs and the published file."""
import json, sys
sys.path.insert(0, ".")
from referee.common import stock_per_episode
def h(f): return {json.loads(l)["seed"]: json.loads(l) for l in open(f)}
S = range(1000, 1500)
stock = json.load(open("outputs/stock_mesa/eval_info.json"))
tab = {"stock_mesa(b8,tf32)": dict(zip(S, stock["per_task"][0]["metrics"]["successes"]))}
pub = {r["seed"]: r["success"] for r in stock_per_episode(json.load(open("experiments/transfer_cube/stage0/published_eval_info.json")))}
tab["published"] = pub
hz = {}
for n, f in (("harness(on1)", "experiments/transfer_cube/stage0/harness_repro.jsonl"), ("off1", "scratch/diag/off1.jsonl"), ("off8", "scratch/diag/off8.jsonl"), ("on8", "scratch/diag/on8.jsonl")):
    r = h(f); tab[n] = {s: r[s]["benchmark_success"] for s in S}; hz[n] = {s: r[s]["traj_hash"] for s in S}
ks = list(tab)
for i, a in enumerate(ks):
    print(f"{a:22s} success {sum(bool(tab[a][s]) for s in S)/5:.1f}%  agree:", {b: round(100*sum(bool(tab[a][s]) == bool(tab[b][s]) for s in S)/500, 1) for b in ks[i+1:]})
print("identical trajectory hashes:", {(a, b): sum(hz[a][s] == hz[b][s] for s in S) for i, a in enumerate(hz) for b in list(hz)[i+1:]})
