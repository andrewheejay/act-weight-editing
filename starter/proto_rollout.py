"""Prototype of the spec's rollout loop, run with a random-weight ACT (no hub access in the sandbox).

Checks the plumbing only: chunked execution with 4 decisions, per-step measurements, strict success,
determinism, and timing. Success numbers from random weights mean nothing.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastenv import FastAloha  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(0)

EPISODE_STEPS = 400
STRICT_WINDOW = 25


def make_policy():
    cfg = ACTConfig(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        pretrained_backbone_weights=None,
        device="cpu",
    )
    return ACTPolicy(cfg).eval()


def to_batch(image_hwc_uint8, agent_pos):
    img = torch.from_numpy(image_hwc_uint8.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    return {"observation.images.top": img, "observation.state": torch.from_numpy(agent_pos).float().unsqueeze(0)}


def rollout(policy, env, seed, n_action_steps=100, to_env_action=lambda a: a):
    """One episode. Never stops early, so strict success can be scored at the end."""
    agent_pos = env.reset(seed)
    cube0 = env.cube_pos()
    queue = []
    rewards, left, right, table, cube_z, actions = [], [], [], [], [], []
    n_decisions = 0
    for t in range(EPISODE_STEPS):
        if not queue:
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(to_batch(env.image(), agent_pos))[0, :n_action_steps]
            queue = list(to_env_action(chunk).numpy())
            n_decisions += 1
        a = queue.pop(0)
        agent_pos, r = env.step(a)
        l, rt, tb = env.touch_flags()
        rewards.append(int(r)); left.append(l); right.append(rt); table.append(tb)
        cube_z.append(float(env.cube_pos()[2])); actions.append(a)
    rewards = np.array(rewards); left = np.array(left); right = np.array(right); table = np.array(table)
    first = lambda m: int(np.argmax(m)) if m.any() else -1  # noqa: E731
    tail = slice(EPISODE_STEPS - STRICT_WINDOW, EPISODE_STEPS)
    return {
        "seed": seed,
        "cube_x0": float(cube0[0]), "cube_y0": float(cube0[1]),
        "max_stage": int(rewards.max()),
        "benchmark_success": bool(rewards.max() == 4),
        "t_success": first(rewards == 4),
        "strict_success": bool(left[tail].all() and not right[tail].any() and not table[tail].any()),
        "t_right_touch": first(right), "t_lift": first(rewards >= 2), "t_left_touch": first(left),
        "dual_hold_steps": int((left & right).sum()),
        "cube_z_max": float(np.max(cube_z)),
        "n_decisions": n_decisions,
        "actions": np.stack(actions),
    }


if __name__ == "__main__":
    policy = make_policy()
    env = FastAloha(lazy=True)
    # Random weights emit near-zero normalized actions; map them near the start pose so the arms move a bit.
    start = torch.tensor([0, -0.96, 1.16, 0, -0.3, 0, 1.0, 0, -0.96, 1.16, 0, -0.3, 0, 1.0])
    to_env = lambda chunk: start + 0.3 * chunk  # noqa: E731

    t0 = time.perf_counter()
    r1 = rollout(policy, env, 20000, to_env_action=to_env)
    t1 = time.perf_counter() - t0
    r2 = rollout(policy, env, 20000, to_env_action=to_env)
    print({k: v for k, v in r1.items() if k != "actions"})
    print(f"episode wall time on 1 CPU thread: {t1:.2f} s")
    print("same seed twice, identical actions:", bool(np.array_equal(r1["actions"], r2["actions"])),
          "| identical outcome fields:", all(r1[k] == r2[k] for k in r1 if k != "actions"))
    r3 = rollout(policy, env, 20000, n_action_steps=50, to_env_action=to_env)
    print("n_action_steps=50 ->", r3["n_decisions"], "decisions per episode")
