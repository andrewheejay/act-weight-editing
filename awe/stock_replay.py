"""Replays recorded actions through the unmodified stock gym-aloha environment, with its own three renders per step.

Runs in its own process, so nothing is patched. Used by the test that compares the fast harness with stock physics.
Usage: python -m awe.stock_replay ACTIONS.npz OUT.npz   with arrays "<seed>" of shape (400, 14)
"""
import sys

import numpy as np


def main(inp, out, task="transfer_cube"):
    import gymnasium as gym
    import gym_aloha  # noqa: F401

    from awe.paths import TASKS

    env = gym.make(f"gym_aloha/{TASKS[task]}", obs_type="pixels_agent_pos", render_mode="rgb_array").unwrapped
    data = np.load(inp)
    res = {}
    for key in data.files:
        env.reset(seed=int(key))
        traj = [env._env.physics.data.qpos.copy()]
        rew = []
        for a in data[key]:
            rew.append(int(env.step(np.asarray(a, dtype=np.float32))[1]))
            traj.append(env._env.physics.data.qpos.copy())
        res[key] = np.array(traj)
        res[f"{key}_reward"] = np.array(rew)
    np.savez(out, **res)


if __name__ == "__main__":
    from awe import render

    render.configure("egl_nvidia")  # only physics is compared, so the faster renderer is used here
    main(*sys.argv[1:4])
