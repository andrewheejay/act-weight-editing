"""Aim versus contact deflection at the grasps. Standalone (not in awe/). No rollouts: kinematics of the saved states only.

python experiments/insertion/stage1/reach_ins.py reach    -> stage1/reach_default.jsonl.gz  (per episode, from the trace cache)
python experiments/insertion/stage1/reach_ins.py stats    -> stage1/ins_reach.json          (from committed files only)

Per arm (r = right arm and peg, l = left arm and socket):
  t_close   first close step (the gripper's command falls below 0.5 after having been above 0.7; recorded in the traces)
  t_free    the step before the first contact of that gripper with its object, or before the close if there was none, so nothing has been pushed
  touched_before_close   a finger touched the object before the close step
  meas_*    gripper centre (traced) minus object centre at that step
  cmd_*     the gripper centre the commanded joints would give (same step, other joints as in the state), minus the object centre
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from analysis_ins import load, load_trace  # noqa: E402
from awe.stage1_analysis import auc  # noqa: E402

S1 = Path(__file__).resolve().parent


def build_fk():
    from awe.env import FastAloha

    env = FastAloha("insertion")
    import mujoco

    m = env.physics.model.ptr
    d = mujoco.MjData(m)
    ids = {"r": [env.bid["vx300s_right/left_finger_link"], env.bid["vx300s_right/right_finger_link"]],
           "l": [env.bid["vx300s_left/left_finger_link"], env.bid["vx300s_left/right_finger_link"]]}
    return m, d, ids, mujoco


def tcp_of(m, d, ids, mujoco, qpos, arm):
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    a, b = ids[arm]
    return 0.5 * (np.array(d.xpos[a]) + np.array(d.xpos[b]))


ARM = {"r": ("tcpr", "peg", 8, 7, 13), "l": ("tcpl", "sock", 0, 0, 6)}   # trace key, object key, first joint in qpos, first in action, gripper in action


def cmd_reach():
    m, d, ids, mujoco = build_fk()
    rows, err = [], []
    for r in load("default"):
        z = load_trace("default", r)
        out = {"seed": r["seed"], "kind": r["kind"]}
        for arm in ("r", "l"):
            tkey, okey, q0, a0, _ = ARM[arm]
            tc = r["t_rclose"] if arm == "r" else r["t_lclose"]
            if tc is None or tc < 0:
                continue
            tc = int(tc)
            obj = z[okey].astype(float)
            q = z["qpos"].astype(float)
            contact = z["right"] if arm == "r" else z["left"]
            touched = np.nonzero(contact[:tc] > 0)[0]
            tfree = int(touched[0]) - 1 if len(touched) else tc - 1
            o = {"t_close": tc, "touched_before_close": bool(len(touched)), "t_free": tfree}
            for lab, t in (("close", tc), ("free", tfree)):
                if t < 0:
                    continue
                meas = z[tkey][t].astype(float)
                err.append(np.abs(tcp_of(m, d, ids, mujoco, q[t + 1], arm) - meas).max())
                qq = q[t + 1].copy()
                qq[q0: q0 + 6] = z["action"][t][a0: a0 + 6]
                cmd = tcp_of(m, d, ids, mujoco, qq, arm)
                o[f"meas_{lab}"] = [round(float(x), 5) for x in meas - obj[t]]
                o[f"cmd_{lab}"] = [round(float(x), 5) for x in cmd - obj[t]]
            out[arm] = o
        rows.append(out)
    print("max |FK of state - traced gripper centre| (m):", float(np.max(err)))
    with gzip.GzipFile(S1 / "reach_default.jsonl.gz", "wb", mtime=0) as fh:
        for r in rows:
            fh.write((json.dumps(r) + "\n").encode())


def cmd_stats():
    with gzip.open(S1 / "reach_default.jsonl.gz", "rt") as fh:
        R = [json.loads(l) for l in fh]
    F = load("default")
    from review1_ins import part_of

    part = {r["seed"]: part_of(r) for r in F}
    Sset = {r["seed"] for r in F if r["kind"] == "S"}
    met = {r["seed"] for r in F if r["kind"] in "SAM"}
    out = {}
    for arm, grp in (("r", "peg_not_lifted"), ("l", "socket_not_lifted")):
        o = {}
        for lab in ("close", "free"):
            for src in ("meas", "cmd"):
                key = f"{src}_{lab}"
                V = {r["seed"]: np.array(r[arm][key]) for r in R if arm in r and key in r[arm]}
                mu = np.mean([v for s, v in V.items() if s in Sset], 0)
                e = {s: float(np.linalg.norm(v - mu)) for s, v in V.items()}
                G = [e[s] for s in V if part[s] == grp]
                Sv = [e[s] for s in V if s in Sset]
                Mv = [e[s] for s in V if s in met]
                y = np.r_[np.ones(len(G), bool), np.zeros(len(Sv), bool)]
                y2 = np.r_[np.ones(len(G), bool), np.zeros(len(Mv), bool)]
                d = np.array([V[s] - mu for s in V if part[s] == grp])
                o[key] = {"n_part": len(G), "n_S": len(Sv), "median_dev_cm_part": round(100 * float(np.median(G)), 2), "median_dev_cm_S": round(100 * float(np.median(Sv)), 2),
                          "p90_S_cm": round(100 * float(np.percentile(Sv, 90)), 2), "share_part_above_S_p90": round(float(np.mean(np.array(G) > np.percentile(Sv, 90))), 3),
                          "auc_part_vs_S": round(float(auc(np.r_[G, Sv], y)), 4), "auc_part_vs_met": round(float(auc(np.r_[G, Mv], y2)), 4),
                          "mean_signed_dev_cm_xyz_part": [round(100 * float(x), 2) for x in d.mean(0)]}
        t = [r[arm]["touched_before_close"] for r in R if arm in r and part[r["seed"]] == grp]
        ts = [r[arm]["touched_before_close"] for r in R if arm in r and r["seed"] in Sset]
        o["touched_before_close_share_part_S"] = [round(float(np.mean(t)), 3), round(float(np.mean(ts)), 3)]
        o["median_t_close_part_S"] = [int(np.median([r[arm]["t_close"] for r in R if arm in r and part[r["seed"]] == grp])), int(np.median([r[arm]["t_close"] for r in R if arm in r and r["seed"] in Sset]))]
        out[f"{arm}_arm__{grp}"] = o
    (S1 / "ins_reach.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


HEIGHTS = (0.10, 0.08, 0.06, 0.05)


def cmd_matched():
    """Height-matched commanded lateral offset: at the first step (up to the close) where the commanded gripper is less than h above the object,
    the horizontal (x, y) offset of the commanded gripper from the object. Contact-free if no finger of that arm touched the object earlier."""
    m, d, ids, mujoco = build_fk()
    rows = []
    for r in load("default"):
        z = load_trace("default", r)
        out = {"seed": r["seed"], "kind": r["kind"]}
        for arm in ("r", "l"):
            tkey, okey, q0, a0, _ = ARM[arm]
            tc = r["t_rclose"] if arm == "r" else r["t_lclose"]
            if tc is None or tc < 0:
                continue
            tc = int(tc)
            obj = z[okey].astype(float)
            q = z["qpos"].astype(float)
            contact = z["right"] if arm == "r" else z["left"]
            first_c = int(np.argmax(contact > 0)) if (contact > 0).any() else 10**6
            o = {}
            for k in range(0, tc + 1):
                qq = q[k + 1].copy()
                qq[q0: q0 + 6] = z["action"][k][a0: a0 + 6]
                c = tcp_of(m, d, ids, mujoco, qq, arm) - obj[k]
                for h in HEIGHTS:
                    key = f"h{int(h * 100)}"
                    if key not in o and c[2] < h:
                        o[key] = [round(float(c[0]), 5), round(float(c[1]), 5), bool(first_c > k), k]
            out[arm] = o
        rows.append(out)
    with gzip.GzipFile(S1 / "reach_matched_default.jsonl.gz", "wb", mtime=0) as fh:
        for r in rows:
            fh.write((json.dumps(r) + "\n").encode())
    print(len(rows))


def cmd_matched_stats():
    with gzip.open(S1 / "reach_matched_default.jsonl.gz", "rt") as fh:
        R = [json.loads(l) for l in fh]
    F = load("default")
    from review1_ins import part_of

    part = {r["seed"]: part_of(r) for r in F}
    Sset = {r["seed"] for r in F if r["kind"] == "S"}
    out = {}
    for arm, grp in (("r", "peg_not_lifted"), ("l", "socket_not_lifted")):
        o = {}
        for h in HEIGHTS:
            key = f"h{int(h * 100)}"
            V = {r["seed"]: np.array(r[arm][key][:2]) for r in R if arm in r and key in r[arm]}
            clean = {r["seed"]: r[arm][key][2] for r in R if arm in r and key in r[arm]}
            mu = np.mean([v for s, v in V.items() if s in Sset], 0)
            e = {s: float(np.linalg.norm(v - mu)) for s, v in V.items()}
            for lab, ok in (("all", lambda s: True), ("contact_free", lambda s: clean[s])):
                G = [e[s] for s in V if part[s] == grp and ok(s)]
                Sv = [e[s] for s in V if s in Sset and ok(s)]
                if len(G) < 8 or len(Sv) < 8:
                    o[f"{key}_{lab}"] = {"n_part": len(G), "n_S": len(Sv)}
                    continue
                y = np.r_[np.ones(len(G), bool), np.zeros(len(Sv), bool)]
                dd = np.array([V[s] - mu for s in V if part[s] == grp and ok(s)])
                o[f"{key}_{lab}"] = {"n_part": len(G), "n_S": len(Sv), "median_dev_cm_part": round(100 * float(np.median(G)), 2), "median_dev_cm_S": round(100 * float(np.median(Sv)), 2),
                                     "auc_part_vs_S": round(float(auc(np.r_[G, Sv], y)), 4), "share_part_above_S_p90": round(float(np.mean(np.array(G) > np.percentile(Sv, 90))), 3),
                                     "mean_signed_xy_dev_cm_part": [round(100 * float(x), 2) for x in dd.mean(0)]}
        out[f"{arm}_arm__{grp}"] = o
    (S1 / "ins_reach_matched.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"reach": cmd_reach, "stats": cmd_stats, "matched": cmd_matched, "matched_stats": cmd_matched_stats}[sys.argv[1]]()
