"""Stands in for lerobot-eval. Writes an eval_info.json from the fake world.

The default layout is the one LeRobot 0.7.0 really writes, plain lists with no seeds. `old` is the
published hub file's layout, and `per_task` keeps per-episode rows with seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import world

TASKS = {"AlohaTransferCube-v0": "transfer_cube", "AlohaInsertion-v0": "insertion"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", default="lerobot07")
    a = ap.parse_args()
    task = TASKS[a.task]
    arm, manifest = "original", None
    meta = Path(a.policy) / "arm.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        arm, manifest = m["arm"], json.loads(Path(m["manifest"]).read_text())
    rows = []
    for i in range(a.n):
        r = world.rollout(task, arm, a.seed + i, manifest)
        rows.append({"episode_ix": i, "sum_reward": 0.0, "max_reward": r["max_reward"], "success": r["benchmark_success"], "seed": a.seed + i})
    agg = {"pc_success": 100.0 * sum(r["success"] for r in rows) / len(rows), "n_episodes": len(rows)}
    if a.layout == "old":
        info = {"per_episode": rows, "aggregated": agg}
    elif a.layout == "per_task":
        info = {"per_task": [{"task_group": "aloha", "task_id": 0, "metrics": {"successes": [r["success"] for r in rows], "per_episode": rows}}], "per_group": {}, "overall": agg}
    else:  # what lerobot_eval.eval_one returns in LeRobot 0.7.0
        metrics = {
            "sum_rewards": [r["sum_reward"] for r in rows],
            "max_rewards": [r["max_reward"] for r in rows],
            "successes": [r["success"] for r in rows],
            "video_paths": [],
            "predicted_video_paths": [],
        }
        n_ok = sum(1 for r in rows if r["success"])
        summary = {"n_episodes": len(rows), "n_success": n_ok, "pc_success": agg["pc_success"]}
        info = {"per_task": [{"task_group": "aloha", "task_id": 0, "metrics": metrics, **summary}], "per_group": {"aloha": summary}, "overall": summary}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_info.json").write_text(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
