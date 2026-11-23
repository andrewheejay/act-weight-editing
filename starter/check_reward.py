"""Sandbox check: does the benchmark's reward wiring fire on this MuJoCo version?"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastenv import FastAloha

hold = np.array([0, -0.96, 1.16, 0, -0.3, 0, 1.0, 0, -0.96, 1.16, 0, -0.3, 0, 1.0], dtype=np.float32)
env = FastAloha(lazy=True)
ph = env.physics
task = env.env._env.task

env.reset(5)
for _ in range(50):
    env.step(hold)
print("settled on table -> reward", task.get_reward(ph), "| flags (left, right, table)", env.touch_flags(),
      "| cube", np.round(env.cube_pos(), 3))
raw = [(ph.model.id2name(c.geom1, "geom"), ph.model.id2name(c.geom2, "geom")) for c in ph.data.contact[: ph.data.ncon]]
print("raw contact pairs at rest:", sorted(set(raw)))

for side, expect in (("right", 2), ("left", 4)):
    env.reset(5)
    gid = ph.model.name2id(f"vx300s_{side}/10_{side}_gripper_finger", "geom")
    target = ph.data.geom_xpos[gid].copy()
    ph.data.qpos[-7:-4] = target
    ph.data.qvel[-6:] = 0
    ph.forward()
    raw = sorted({(ph.model.id2name(c.geom1, "geom"), ph.model.id2name(c.geom2, "geom"))
                  for c in ph.data.contact[: ph.data.ncon]})
    print(f"cube moved onto the {side} arm's {side} finger at {np.round(target, 3)} -> reward",
          task.get_reward(ph), f"(benchmark rule expects {expect})", "| flags", env.touch_flags())
    print("   raw pairs:", raw)
