"""Commanded-gripper geometry for the per-seed record (transfer cube). Pure kinematics on a PRIVATE MjData, so the simulation
is never touched. The definitions are those of awe/stage1_analysis.py (zmax20, cmd_clean, cmd_lat_h at 0.08 m)."""
import numpy as np

SETTLE = 20
LIFT_Z = 0.06
HEIGHT = 0.08


class ArmFK:
    def __init__(self, env):
        import mujoco  # here, not at the top: LLVM import order, see awe/policy.py

        self.kin = mujoco.mj_kinematics
        self.m = env.physics.model.ptr
        self.d = mujoco.MjData(self.m)
        self.lf = env.bid["vx300s_right/left_finger_link"]
        self.rf = env.bid["vx300s_right/right_finger_link"]

    def fingers(self, qpos):
        self.d.qpos[:] = qpos
        self.kin(self.m, self.d)
        return np.array(self.d.xpos[self.lf]), np.array(self.d.xpos[self.rf])


def transfer_cube(arr, rec, fk) -> dict:
    cube, q, act = arr["cube"], arr["qpos"], arr["action"]
    zmax = float(cube[SETTLE:, 2].max())
    out = {"zmax20": round(zmax, 5), "grasp_fail": bool(not rec["benchmark_success"] and zmax <= LIFT_Z), "cmd_lat8": None, "cmd_clean": None}
    t, tt = rec["t_rclose"], rec["t_touch"]
    if t < 0:
        return out
    out["cmd_clean"] = bool(tt < 0 or tt >= 100 * (t // 100))
    c0 = cube[min(SETTLE, t)]
    for k in range(min(SETTLE, t), t + 1):
        qq = q[k + 1].copy()
        qq[8:14] = act[k][7:13]  # the right arm's 6 joint targets
        a, b = fk.fingers(qq)
        mid = 0.5 * (a + b)
        if mid[2] - c0[2] < HEIGHT:
            ax = (b - a) / np.linalg.norm(b - a)
            out["cmd_lat8"] = round(float((c0 - mid) @ ax), 5)
            break
    return out
