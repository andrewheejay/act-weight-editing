"""Prototype of the fast ALOHA env wrapper described in the spec (sandbox-tested).

Idea: the stock task renders three 480x640 cameras on every step, but ACT with
n_action_steps=100 only looks at one camera on 4 steps per episode. Skip the rest.
Physics is untouched, so trajectories must be identical to the stock env.
"""
import collections
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import gym_aloha  # noqa: F401
from gym_aloha.tasks import sim as aloha_sim

LEFT_FINGERS = ("vx300s_left/10_left_gripper_finger", "vx300s_left/10_right_gripper_finger")
RIGHT_FINGERS = ("vx300s_right/10_left_gripper_finger", "vx300s_right/10_right_gripper_finger")

_stock_get_observation = aloha_sim.BimanualViperXTask.get_observation


def _lazy_get_observation(self, physics):
    obs = collections.OrderedDict()
    obs["qpos"] = self.get_qpos(physics)
    obs["qvel"] = self.get_qvel(physics)
    obs["env_state"] = self.get_env_state(physics)
    obs["images"] = {"top": None}  # rendered on demand by FastAloha.image()
    return obs


class FastAloha:
    """One ALOHA sim episode at a time, with rendering only when asked."""

    def __init__(self, task="AlohaTransferCube-v0", lazy=True):
        aloha_sim.BimanualViperXTask.get_observation = _lazy_get_observation if lazy else _stock_get_observation
        self.lazy = lazy
        self.env = gym.make(f"gym_aloha/{task}", obs_type="pixels_agent_pos", max_episode_steps=10_000).unwrapped
        self.physics = self.env._env.physics
        self._names = [self.physics.model.id2name(i, "geom") for i in range(self.physics.model.ngeom)]

    def reset(self, seed):
        if self.lazy:
            # AlohaEnv._format_raw_obs copies the image, so bypass it.
            gym.Env.reset(self.env, seed=seed)
            self.env._env.task.random.seed(seed)
            self.env._env.task._random = np.random.RandomState(seed)
            from gym_aloha.utils import sample_box_pose, sample_insertion_pose

            if self.env.task == "transfer_cube":
                aloha_sim.BOX_POSE[0] = sample_box_pose(seed)
            else:
                aloha_sim.BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))
            raw = self.env._env.reset().observation
            return raw["qpos"]
        obs, _ = self.env.reset(seed=seed)
        return obs["agent_pos"]

    def step(self, action):
        """Returns (agent_pos, reward). Never terminates early; the caller counts steps."""
        if self.lazy:
            _, reward, _, raw = self.env._env.step(action)
            return raw["qpos"], reward
        obs, reward, *_ = self.env.step(action)
        return obs["agent_pos"], reward

    def image(self):
        return self.physics.render(height=480, width=640, camera_id="top")

    # ---- measurements straight from physics -------------------------------------------------
    def contacts(self):
        d = self.physics.data
        pairs = set()
        for i in range(d.ncon):
            c = d.contact[i]
            pairs.add(frozenset((self._names[c.geom1], self._names[c.geom2])))
        return pairs

    def touch_flags(self, obj="red_box"):
        p = self.contacts()
        left = any(frozenset((obj, f)) in p for f in LEFT_FINGERS)
        right = any(frozenset((obj, f)) in p for f in RIGHT_FINGERS)
        table = frozenset((obj, "table")) in p
        return left, right, table

    def cube_pos(self):
        return self.physics.named.data.xpos["box"].copy()

    def tcp(self, side):
        x = self.physics.named.data.xpos
        return 0.5 * (x[f"vx300s_{side}/left_finger_link"] + x[f"vx300s_{side}/right_finger_link"])
