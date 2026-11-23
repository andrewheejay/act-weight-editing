"""Turn one episode's per-step arrays into measurements. Definitions follow SPEC.md, Measurements.

Scores and classes
  benchmark   reward 4 at any step (the stock reward, exactly what LeRobot's evaluator calls success)
  strict      25 consecutive steps with a left finger on the cube, no right finger on it, and the cube off the table
  class       0..3 is the highest stock reward reached. 4a / 4b is reward 4 without / with strict success
Event steps are 0-based control steps. A missing event is -1.
"""
import hashlib

import numpy as np

STRICT_WINDOW = 25
OPEN = 0.7  # a gripper command above this counts as open
CLOSE = 0.5  # closing means first below CLOSE after having been above OPEN


def longest_run(mask) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def first(mask) -> int:
    m = np.asarray(mask)
    return int(np.argmax(m)) if m.any() else -1


def failure_class(max_reward: int, strict: bool) -> str:
    return {0: "0", 1: "1", 2: "2", 3: "3"}.get(int(max_reward), "4b" if strict else "4a")


def traj_hash(qpos: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(qpos, dtype=np.float64).tobytes()).hexdigest()


def close_start(cmd, t0=0) -> int:
    """First step >= t0 with the command below CLOSE, after the command was above OPEN."""
    opened = False
    for t in range(t0, len(cmd)):
        if cmd[t] > OPEN:
            opened = True
        elif opened and cmd[t] < CLOSE:
            return t
    return -1


def strict_success(left, right, table) -> bool:
    ok = np.asarray(left) & ~np.asarray(right) & ~np.asarray(table)
    return longest_run(ok) >= STRICT_WINDOW


def vec(a) -> list:
    return [round(float(x), 5) for x in a]


def transfer_cube(arr) -> dict:
    """Behaviour numbers for the transfer-cube task. Vectors are in metres, (x, y, z)."""
    act, rew = arr["action"], arr["reward"]
    L, R, TB = arr["left"], arr["right"], arr["table"]
    cube = arr["cube"]
    start = arr["cube0"]
    out = {}
    t_touch = first(R)
    t_lift = first(rew >= 2)
    t_left = first(L)
    t_rclose = close_start(act[:, 13])
    t_lclose = close_start(act[:, 6], t0=max(t_lift, 0)) if t_lift >= 0 else -1
    t_release = -1
    if t_left >= 0:
        off = ~R[t_left:]
        for k in range(len(off) - 4):
            if off[k : k + 5].all():
                t_release = t_left + k
                break
    out.update(
        t_touch=t_touch, t_lift=t_lift, t_left_touch=t_left, t_release=t_release, t_rclose=t_rclose, t_lclose=t_lclose
    )
    if t_rclose >= 0:
        out["reach_rel"] = vec(arr["tcpr"][t_rclose] - cube[t_rclose])
    if t_lclose >= 0:
        out["meet_rel"] = vec(arr["tcpl"][t_lclose] - cube[t_lclose])
    out["cube_z_max"] = round(float(cube[:, 2].max()), 5)
    if t_left >= 0:
        out["exchange_z"] = round(float(cube[t_left, 2]), 5)
    out["dual_hold_steps"] = int((L & R).sum())
    end = t_lift if t_lift >= 0 else len(cube)
    slide = np.linalg.norm(cube[:end, :2] - start[:2], axis=1)
    out["knock_m"] = round(float(slide.max()) if len(slide) else 0.0, 5)
    jumps = [float(np.abs(act[t] - act[t - 1]).max()) for t in (100, 200, 300) if t < len(act)]
    out["chunk_jump"] = round(max(jumps), 5)
    out["cube_x0"], out["cube_y0"] = float(start[0]), float(start[1])
    out["vis_px"] = [int(v) for v in arr["vis"]]
    out["strict_both_fingers"] = bool(longest_run(arr["leftboth"] & ~R & ~TB) >= STRICT_WINDOW)
    return out
