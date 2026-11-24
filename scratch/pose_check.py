"""A pose rollout equals the seed rollout at that seed's pose. Uses 3 design seeds, 6 rollouts, all in the ledger."""
import sys, json
sys.path.insert(0, "/home/ubuntu/act-weight-editing")
from awe import pool
from awe.arms import make_spec
from awe.paths import ROOT
import gym_aloha.utils as u

if __name__ == "__main__":
    spec = make_spec("original")
    seeds = [20000, 20001, 20002]
    poses = [[float(x) for x in u.sample_box_pose(s)] for s in seeds]
    out1, out2 = ROOT / "scratch/pose_check_seed.jsonl", ROOT / "scratch/pose_check_pose.jsonl"
    pool.run("transfer_cube", [(spec, s, None) for s in seeds], out1, budget="agent", purpose="stage1 posecheck", trace_dir="/tmp/pc_seed", workers=3)
    pool.run("transfer_cube", [(spec, None, p) for p in poses], out2, budget="agent", purpose="stage1 posecheck", trace_dir="/tmp/pc_pose", workers=3)
    a = {r["seed"]: r for r in map(json.loads, open(out1))}
    b = {tuple(r["pose"]): r for r in map(json.loads, open(out2))}
    for s, p in zip(seeds, poses):
        print(s, a[s]["traj_hash"] == b[tuple(p)]["traj_hash"], a[s]["benchmark_success"], b[tuple(p)]["benchmark_success"])
