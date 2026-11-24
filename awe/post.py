"""Output-level rules for the baseline arms. They see the executed action and the step, nothing else (no image, no state, no cube).

A recipe is one JSON object. Joints are the 14 action dimensions: 0 to 6 the left arm (6 its gripper), 7 to 13 the right arm
(13 its gripper). Arm joints are radians, grippers are normalised (0 closed, 1 open).
  {"kind": "offset", "joint": j, "delta": d}            constant shift of one target for the whole episode
  {"kind": "gain", "g": g, "joints": [...]}             a' = m_t + g (a - m_t), m_t = freeze/mean_action.json at step t
  {"kind": "approach_offset", "joint": j, "delta": d}   the same shift, only until the right gripper first starts to close
The close rule is the one in awe/measure.py (first command below CLOSE after one above OPEN), read on the policy's own command.
"""
import json

import numpy as np

from .measure import CLOSE, OPEN
from .paths import exp

N_JOINTS, STEPS, RIGHT_GRIPPER = 14, 400, 13
KINDS = ("offset", "gain", "approach_offset")


def mean_path(task):
    return exp(task) / "freeze" / "mean_action.json"


def load_mean(task) -> np.ndarray:
    m = np.array(json.loads(mean_path(task).read_text())["mean_action"], dtype=np.float64)
    assert m.shape == (STEPS, N_JOINTS), m.shape
    return m


class Post:
    def __init__(self, recipe: dict, task: str):
        self.r, self.kind = recipe, recipe["kind"]
        if self.kind not in KINDS:
            raise ValueError(f"unknown output recipe {self.kind}")
        if self.kind == "gain":
            self.mean = load_mean(task)
            self.joints = [int(j) for j in recipe["joints"]]
            self.g = float(recipe["g"])
            assert self.joints and all(0 <= j < N_JOINTS for j in self.joints)
        else:
            self.joint, self.delta = int(recipe["joint"]), float(recipe["delta"])
            assert 0 <= self.joint < N_JOINTS
        self.opened = self.closed = False

    def __call__(self, t: int, a) -> np.ndarray:
        """The action to execute at step t. Never changes its input."""
        a = np.array(a, dtype=np.float64)
        assert a.shape == (N_JOINTS,), a.shape
        if self.kind == "gain":
            j = self.joints
            a[j] = self.mean[t, j] + self.g * (a[j] - self.mean[t, j])
        elif self.kind == "offset":
            a[self.joint] += self.delta
        else:
            raw = a[RIGHT_GRIPPER]
            if raw > OPEN:
                self.opened = True
            elif self.opened and raw < CLOSE:
                self.closed = True
            if not self.closed:
                a[self.joint] += self.delta
        return a.astype(np.float32)
