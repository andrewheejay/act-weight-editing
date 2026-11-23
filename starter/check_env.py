"""Sandbox check of gym-aloha facts used in the experiment spec. Not part of the repo."""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym
import mujoco
import dm_control
import gym_aloha  # noqa: F401  (registers the envs)
from gym_aloha.utils import sample_box_pose
from gym_aloha.constants import START_ARM_POSE

print("mujoco", mujoco.__version__, "| gymnasium", gym.__version__, "| MUJOCO_GL", os.environ["MUJOCO_GL"])

env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=400)
base = env.unwrapped
physics = base._env.physics

seed = 1000
obs, info = env.reset(seed=seed)
print("obs keys", list(obs.keys()), "| top image", obs["pixels"]["top"].shape, obs["pixels"]["top"].dtype,
      "| agent_pos", obs["agent_pos"].shape)
print("reset info", info)
print("sampled box pose  ", np.round(sample_box_pose(seed), 4))
print("physics box qpos  ", np.round(physics.named.data.qpos["red_box_joint"], 4))
print("box xpos via body ", np.round(physics.named.data.xpos["box"], 4))
print("model timestep", physics.model.opt.timestep, "| control dt", base._env.control_timestep(),
      "| substeps", round(base._env.control_timestep() / physics.model.opt.timestep))

geoms = [physics.model.id2name(i, "geom") for i in range(physics.model.ngeom)]
print("geoms of interest", [g for g in geoms if g and ("finger" in g or g in ("red_box", "table"))])
bodies = [physics.model.id2name(i, "body") for i in range(physics.model.nbody)]
print("bodies with gripper/finger", [b for b in bodies if b and ("gripper" in b or "finger" in b)])
print("n actuators", physics.model.nu, "| action space", env.action_space.shape)

# Hold-still action in the env's 14-dim action space (gripper normalized, 1 = open).
hold = np.array([0, -0.96, 1.16, 0, -0.3, 0, 1.0, 0, -0.96, 1.16, 0, -0.3, 0, 1.0], dtype=np.float32)

# Timing: full env.step (physics + 3 renders) vs single render vs physics only.
n = 40
t0 = time.perf_counter()
for _ in range(n):
    obs, reward, terminated, truncated, info = env.step(hold)
t_step = (time.perf_counter() - t0) / n
t0 = time.perf_counter()
for _ in range(n):
    physics.render(height=480, width=640, camera_id="top")
t_render = (time.perf_counter() - t0) / n
task = base._env.task
t0 = time.perf_counter()
for _ in range(n):
    task.before_step(hold, physics)
    physics.step(round(base._env.control_timestep() / physics.model.opt.timestep))
t_phys = (time.perf_counter() - t0) / n
print(f"per step: env.step {t_step*1e3:.1f} ms | one render {t_render*1e3:.1f} ms | physics only {t_phys*1e3:.2f} ms")
print(f"per 400-step episode: stock {t_step*400:.1f} s | physics + 4 renders {t_phys*400 + 4*t_render:.2f} s")
print("box rest z after settling", round(float(physics.named.data.xpos["box"][2]), 4))
print("step info keys", list(info.keys()), "| reward", reward, "| terminated", terminated, "| truncated", truncated)


def rollout(seed, steps=60):
    obs, _ = env.reset(seed=seed)
    imgs = [obs["pixels"]["top"].copy()]
    qs = [physics.data.qpos.copy()]
    for t in range(steps):
        a = hold.copy()
        a[7] = 0.3 * np.sin(t / 10.0)  # move the right waist a little
        obs, *_ = env.step(a)
        imgs.append(obs["pixels"]["top"].copy())
        qs.append(physics.data.qpos.copy())
    return np.stack(imgs), np.stack(qs)


i1, q1 = rollout(7)
i2, q2 = rollout(7)
print("determinism, same seed twice: max |dqpos|", float(np.abs(q1 - q2).max()),
      "| max |dimage|", int(np.abs(i1.astype(int) - i2.astype(int)).max()),
      "| frames that differ", int((np.abs(i1.astype(int) - i2.astype(int)).reshape(len(i1), -1).max(1) > 0).sum()), "of", len(i1))

# Does stepping continue to work after `terminated`? (needed for the strict success check)
obs, _ = env.reset(seed=3)
print("can step after reset without wrapper complaints:", True)
env.close()
