"""Sandbox check: close a gripper on the cube (gravity off) and read the benchmark reward."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastenv import FastAloha

env = FastAloha(lazy=True)
ph = env.physics
task = env.env._env.task
for side, expect in (("right", 2), ("left", 4)):
    env.reset(5)
    g0 = ph.model.opt.gravity.copy()
    ph.model.opt.gravity[:] = 0
    a = np.array([0, -0.96, 1.16, 0, -0.3, 0, 1.0, 0, -0.96, 1.16, 0, -0.3, 0, 1.0], dtype=np.float32)
    for _ in range(25):            # open both grippers first
        env.step(a)
    ph.data.qpos[-7:-4] = env.tcp(side)   # cube between the fingers
    ph.data.qvel[-6:] = 0
    ph.forward()
    a[6 if side == "left" else 13] = 0.0  # close that gripper
    seen = []
    for t in range(40):
        _, r = env.step(a)
        seen.append(int(r))
    print(f"{side} gripper closed on the cube -> reward sequence tail {seen[-5:]}, max {max(seen)} (benchmark rule expects {expect})",
          "| flags (left, right, table)", env.touch_flags())
    ph.model.opt.gravity[:] = g0
