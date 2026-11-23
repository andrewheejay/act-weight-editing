"""Sandbox checks: lazy rendering is equivalent, reward wiring works on this MuJoCo, timing."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastenv import FastAloha  # noqa: E402

hold = np.array([0, -0.96, 1.16, 0, -0.3, 0, 1.0, 0, -0.96, 1.16, 0, -0.3, 0, 1.0], dtype=np.float32)


def scripted(t):
    a = hold.copy()
    a[7] = 0.4 * np.sin(t / 15.0)      # right waist
    a[8] = -0.96 + 0.5 * np.sin(t / 25.0)  # right shoulder
    a[1] = -0.96 + 0.3 * np.sin(t / 20.0)  # left shoulder
    a[13] = 0.5 + 0.5 * np.cos(t / 10.0)   # right gripper
    return a


def run(lazy, seed, steps=120, decision_every=40):
    env = FastAloha(lazy=lazy)
    env.reset(seed)
    qs, imgs, rewards = [], [], []
    for t in range(steps):
        if t % decision_every == 0:
            imgs.append(env.image().copy())
        _, r = env.step(scripted(t))
        qs.append(env.physics.data.qpos.copy())
        rewards.append(r)
    return np.stack(qs), np.stack(imgs), rewards


t0 = time.perf_counter()
q_lazy, i_lazy, _ = run(True, 11)
t_lazy = time.perf_counter() - t0
t0 = time.perf_counter()
q_stock, i_stock, _ = run(False, 11, steps=120)
t_stock = time.perf_counter() - t0
print("lazy vs stock: max |dqpos|", float(np.abs(q_lazy - q_stock).max()),
      "| max |dimage| at decision steps", int(np.abs(i_lazy.astype(int) - i_stock.astype(int)).max()))
print(f"120 steps: lazy {t_lazy:.2f} s (incl. env build) vs stock {t_stock:.2f} s")

# ---- Reward wiring on this MuJoCo version: put the cube where a finger is and read the reward.
env = FastAloha(lazy=True)
env.reset(5)
ph = env.physics
task = env.env._env.task
print("at rest on table -> reward", task.get_reward(ph), "| flags (left, right, table)", env.touch_flags())
for side, expect in (("right", 2), ("left", 4)):
    env.reset(5)
    finger_geom = f"vx300s_{side}/10_{side}_gripper_finger"
    gid = ph.model.name2id(finger_geom, "geom")
    target = ph.data.geom_xpos[gid].copy()
    with ph.reset_context():
        ph.named.data.qpos["red_box_joint"][:3] = target
    print(f"cube placed inside the {side} arm's {side} finger -> reward", task.get_reward(ph), f"(expected {expect})",
          "| flags (left, right, table)", env.touch_flags())

# ---- Per-episode cost of the fast path: 400 physics steps + 4 renders
env.reset(3)
t0 = time.perf_counter()
for t in range(400):
    if t % 100 == 0:
        env.image()
    env.step(hold)
    env.touch_flags()
    env.cube_pos()
    env.tcp("right")
t_ep = time.perf_counter() - t0
print(f"fast episode (400 steps, 4 renders, per-step measurements): {t_ep:.2f} s")
print("cube z at rest", round(float(env.cube_pos()[2]), 4), "| right TCP", np.round(env.tcp("right"), 3),
      "| left TCP", np.round(env.tcp("left"), 3))
