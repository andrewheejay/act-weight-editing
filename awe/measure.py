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


def strict_run(arr) -> int:
    """Longest run of steps meeting the strict insertion condition (the strict score needs 25)."""
    return longest_run(arr["pin"] & arr["right"] & arr["left"] & ~arr["table"])


def strict_insertion(arr) -> bool:
    """DEVIATIONS.md, 2026-11-25: peg on the pin, right gripper on the peg, left gripper on the socket, no table contact, 25 steps."""
    return strict_run(arr) >= STRICT_WINDOW


def tip_in_socket(peg, pegR, sock, sockR):
    """The peg's leading end (the end nearer the socket centre) in the socket frame (m), per step: [axial, across y, across z].

    Same definition as experiments/insertion/stage1/analysis_ins.py tip_offset, which made the stage 1 features.
    """
    ax = pegR[:, :, 0]
    e1, e2 = peg + 0.06 * ax, peg - 0.06 * ax
    near1 = np.linalg.norm(e1 - sock, axis=1) < np.linalg.norm(e2 - sock, axis=1)
    return np.einsum("tij,ti->tj", sockR, np.where(near1[:, None], e1, e2) - sock)


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


# Centre distance along the socket axis (m): the peg tip enters at 0.12 and touches the pin at 0.10 (half-lengths 0.06, 0.06, 0.04).
# The interior half-gap is 0.018 and the peg's half-width 0.010, so a peg inside can sit at most 0.008 off axis. 0.012 allows slack.
# Used only for the extra variant strict_depth.
PEG_ENTER, PIN_REACH = 0.12, 0.10
INSERT_LATERAL = 0.012


def peg_in_socket(peg, pegR, sock, sockR):
    """Axial distance, lateral offset (m) and axis angle (degrees) of the peg in the socket frame, per step.

    Both long axes are the bodies' own x axes. The offsets are measured in the socket frame.
    """
    d = np.einsum("tij,ti->tj", sockR, peg - sock)  # socket frame
    cosang = np.abs(np.einsum("ti,ti->t", pegR[:, :, 0], sockR[:, :, 0]))
    ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    return np.abs(d[:, 0]), np.abs(d[:, 1:]).max(1), ang, d


def insertion(arr) -> dict:
    """Behaviour numbers for the insertion task. Vectors are in metres, (x, y, z)."""
    act, rew = arr["action"], arr["reward"]
    L, R = arr["left"], arr["right"]
    peg, sock = arr["peg"], arr["sock"]
    out = {}
    t_rtouch, t_ltouch = first(R), first(L)
    t_grasp = first(rew >= 2)  # both held off the table
    t_meet = first(rew >= 3)
    t_pin = first(arr["pin"])
    t_rclose, t_lclose = close_start(act[:, 13]), close_start(act[:, 6])
    out.update(t_rtouch=t_rtouch, t_ltouch=t_ltouch, t_grasp=t_grasp, t_meet=t_meet, t_pin=t_pin, t_rclose=t_rclose, t_lclose=t_lclose)
    if t_rclose >= 0:
        out["reach_rel_r"] = vec(arr["tcpr"][t_rclose] - peg[t_rclose])
    if t_lclose >= 0:
        out["reach_rel_l"] = vec(arr["tcpl"][t_lclose] - sock[t_lclose])
    out["peg_z_max"] = round(float(peg[:, 2].max()), 5)
    out["sock_z_max"] = round(float(sock[:, 2].max()), 5)
    held = R & L & ~arr["pegtable"] & ~arr["socktable"]
    out["held_steps"] = int(held.sum())
    ax, lat, ang, d = peg_in_socket(peg, arr["pegR"], sock, arr["sockR"])
    near = held & (ax < 0.2)
    if near.any():
        t = int(np.argmin(np.where(near, ax, 9.0)))
        out["min_axial"], out["axial_t"] = round(float(ax[t]), 5), t
        out["lat_at_min_axial"], out["ang_at_min_axial"] = round(float(lat[t]), 5), round(float(ang[t]), 3)
    if t_meet >= 0:
        out["meet_lat"], out["meet_ang"] = round(float(lat[t_meet]), 5), round(float(ang[t_meet]), 3)
        out["meet_d"] = vec(d[t_meet])
        tip = tip_in_socket(peg, arr["pegR"], sock, arr["sockR"])
        alat = np.abs(tip[:, 1:]).max(1)
        if t_meet >= 1:  # the tracking number of the diagnosis: one step before the first peg-socket contact off the table
            out["pre_tip_d"], out["pre_tip_lat"] = vec(tip[t_meet - 1]), round(float(alat[t_meet - 1]), 5)
        after = np.arange(len(tip)) >= t_meet
        ink = after & (alat < 0.018)  # the tip centre inside the 1.8 cm half-opening
        out["deep_in_ax"] = round(float(np.abs(tip[ink, 0]).min()), 5) if ink.any() else None
    out["strict_run"], out["pin_steps"] = strict_run(arr), int(arr["pin"].sum())
    inside = (ax < PIN_REACH + 0.005) & (lat < INSERT_LATERAL)
    out["strict_depth"] = bool(longest_run(inside & held) >= STRICT_WINDOW)
    out["strict_both_fingers"] = bool(longest_run(arr["pin"] & arr["leftboth"] & arr["rightboth"] & ~arr["table"]) >= STRICT_WINDOW)
    for name, b, b0 in (("peg", peg, arr["peg0"]), ("sock", sock, arr["sock0"])):
        off = (rew >= 2).argmax() if (rew >= 2).any() else len(b)
        slide = np.linalg.norm(b[:off, :2] - b0[:2], axis=1)
        out[f"{name}_knock_m"] = round(float(slide.max()) if len(slide) else 0.0, 5)
        out[f"{name}_x0"], out[f"{name}_y0"] = float(b0[0]), float(b0[1])
    jumps = [float(np.abs(act[t] - act[t - 1]).max()) for t in (100, 200, 300) if t < len(act)]
    out["chunk_jump"] = round(max(jumps), 5)
    out["vis_px"] = [int(v) for v in arr["vis"]]
    out["vis_blue"] = [int(v) for v in arr["visb"]]
    return out
