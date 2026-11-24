"""Fast ALOHA simulator. Same physics as gym-aloha, but it renders only when asked.

The patches are bound to the task INSTANCE, so a stock environment in the same process stays untouched.
One simulator per process, because gym-aloha passes the cube pose through a module-level global.
"""
import collections
import types

from . import policy as _policy  # noqa: F401  (LLVM import order, see awe/policy.py)

import gym_aloha  # noqa: F401
import gymnasium as gym
import numpy as np
from gym_aloha.tasks import sim as aloha_sim
from gym_aloha.utils import sample_box_pose, sample_insertion_pose

from .geom import ArmFK
from .paths import TASKS


def _lazy_get_observation(self, physics):
    obs = collections.OrderedDict()
    obs["qpos"] = self.get_qpos(physics)
    return obs


class FastAloha:
    def __init__(self, task: str):
        self.task = task
        self.env = gym.make(
            f"gym_aloha/{TASKS[task]}", obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=10_000
        ).unwrapped
        self.dm = self.env._env
        self.physics = self.dm.physics
        self.tk = self.dm.task
        m = self.physics.model
        self.gid = {m.id2name(i, "geom"): i for i in range(m.ngeom)}
        self.names = [m.id2name(i, "geom") for i in range(m.ngeom)]
        self.bid = {m.id2name(i, "body"): i for i in range(m.nbody)}
        g = self.gid
        self.LF = [g["vx300s_left/10_left_gripper_finger"], g["vx300s_left/10_right_gripper_finger"]]
        self.RF = [g["vx300s_right/10_left_gripper_finger"], g["vx300s_right/10_right_gripper_finger"]]
        self.table = g["table"]
        self.obj = [g["red_box"]] if task == "transfer_cube" else [g["red_peg"]]
        self.tcp_ids = {
            s: (self.bid[f"vx300s_{s}/left_finger_link"], self.bid[f"vx300s_{s}/right_finger_link"])
            for s in ("left", "right")
        }
        self.box_body = self.bid["box"] if task == "transfer_cube" else None
        self.arm_fk = ArmFK(self) if task == "transfer_cube" else None
        self._stock_reward = self.tk.get_reward
        self.last_reward = 0
        self.tk.get_observation = types.MethodType(_lazy_get_observation, self.tk)
        if task == "transfer_cube":
            self.tk.get_reward = types.MethodType(lambda t, physics: self._fast_reward(), self.tk)

    # ---- contacts ---------------------------------------------------------------------------
    def _contacts(self):
        d = self.physics.data
        n = d.ncon
        if n == 0:
            z = np.zeros(0, dtype=int)
            return z, z
        c = d.contact
        return np.array(c.geom1[:n]), np.array(c.geom2[:n])

    def _fast_reward(self):
        """Transfer-cube reward, identical to TransferCubeTask.get_reward, plus the any-finger flags."""
        g1, g2 = self._contacts()
        box = self.obj[0]
        first = g2[g1 == box]  # ordered pairs (box, other), as the stock code matches them
        tl = bool((first == self.LF[0]).any())
        tr = bool((first == self.RF[1]).any())
        tt = bool((first == self.table).any())
        r = 0
        if tr:
            r = 1
        if tr and not tt:
            r = 2
        if tl:
            r = 3
        if tl and not tt:
            r = 4
        other = np.concatenate([first, g1[g2 == box]])
        self.flags = (
            bool(np.isin(other, self.LF).any()),
            bool(np.isin(other, self.RF).any()),
            bool((other == self.table).any()),
            bool((other == self.LF[0]).any()) and bool((other == self.LF[1]).any()),
        )
        return r

    # ---- episode ----------------------------------------------------------------------------
    def reset(self, seed=None, pose=None):
        if (seed is None) == (pose is None):
            raise ValueError("give a seed or a pose")
        gym.Env.reset(self.env, seed=seed)
        if seed is not None:
            self.tk.random.seed(seed)
            self.tk._random = np.random.RandomState(seed)
            if self.task == "transfer_cube":
                pose = sample_box_pose(seed)
            else:
                pose = np.concatenate(sample_insertion_pose(seed))
        aloha_sim.BOX_POSE[0] = np.asarray(pose, dtype=float)
        raw = self.dm.reset().observation
        if self.task == "transfer_cube":
            self._fast_reward()
        return raw["qpos"]

    def step(self, action):
        _, reward, _, raw = self.dm.step(action)
        return raw["qpos"], reward

    def image(self):
        return self.physics.render(height=480, width=640, camera_id="top")

    # ---- state ------------------------------------------------------------------------------
    def qpos(self):
        return np.array(self.physics.data.qpos, dtype=np.float64)

    def cube_pos(self):
        return np.array(self.physics.data.xpos[self.box_body])

    def tcp(self, side):
        a, b = self.tcp_ids[side]
        x = self.physics.data.xpos
        return 0.5 * (np.array(x[a]) + np.array(x[b]))
