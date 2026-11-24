"""Stock lerobot-eval (b1) against the harness: success bits, max rewards, and the published file."""
import json, sys
sys.path.insert(0, ".")
from referee.common import stock_per_episode
S = range(1000, 1500); d = "experiments/transfer_cube/stage0/"
m = json.load(open(d + "stock_eval/eval_info.json"))["per_task"][0]["metrics"]
h = {r["seed"]: r for r in map(json.loads, open(d + "harness_repro.jsonl"))}
pub = {r["seed"]: r["success"] for r in stock_per_episode(json.load(open(d + "published_eval_info.json")))}
print("n harness", len(h), "success mismatches", sum(bool(m["successes"][s-1000]) != h[s]["benchmark_success"] for s in S))
print("max_reward mismatches", [s for s in S if m["max_rewards"][s-1000] != h[s]["max_reward"]])
print("stock vs published agree %", 100*sum(bool(m["successes"][s-1000]) == bool(pub[s]) for s in S)/500)
