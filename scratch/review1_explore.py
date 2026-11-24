"""Exploration for the Gate 1 review, round 1. Design seeds (and the grid with the same reference). No rollouts.

Per episode, at the first close t of the right gripper:
  aim      measured gripper centre minus the untouched cube position (the number in the diagnosis)
  cmd      FK of the COMMANDED right-arm joints at step t, minus the same cube position. In the clean subset the chunk was
           decided before any right-finger contact, so this cannot contain a reaction to contact or a physical deflection
  defl     measured minus commanded gripper centre at t (tracking lag, sag, and any push from contact)
  pre      measured gripper minus cube at fixed leads before the first contact and before the close
"""
import numpy as np
from awe import render
render.configure(render.chosen("transfer_cube"))
from awe.env import FastAloha
from awe.stage1_analysis import SETTLE, auc, load_features, load_records, load_trace
env = FastAloha("transfer_cube")
ph = env.physics
a, b = env.tcp_ids["right"]
def tcp(q):
    with ph.reset_context():
        ph.data.qpos[:] = q
    return 0.5 * (np.array(ph.data.xpos[a]) + np.array(ph.data.xpos[b]))

def table(name):
    recs = load_records("transfer_cube", name)
    feats = {(f["seed"], str(f["pose"])): f for f in load_features("transfer_cube", name)}
    out = []
    for r in recs:
        f = feats[(r.get("seed"), str(r.get("pose")))]
        t = r["t_rclose"]
        if t is None or t < 0:
            continue
        z = load_trace("transfer_cube", name, r)
        q, act, tr, cube = z["qpos"].astype(float), z["action"].astype(float), z["tcpr"].astype(float), z["cube"].astype(float)
        c0 = cube[min(SETTLE, t)]
        qq = q[t + 1].copy(); qq[8:14] = act[t][7:13]
        cmd = tcp(qq)
        tt = r["t_touch"]
        out.append(dict(ok=r["benchmark_success"], gf=f["grasp_fail"], cls=r["failure_class"], t=t, tt=tt,
                        clean=(tt is None or tt < 0 or tt >= 100 * (t // 100)), chunk=t // 100,
                        aim=tr[t] - c0, cmd=cmd - c0, defl=tr[t] - cmd, x0=r["cube_x0"], y0=r["cube_y0"],
                        touching=bool(tt is not None and 0 <= tt < t)))
    return out

def report(name, rows, ref=None):
    ok = np.array([r["ok"] for r in rows]); gf = np.array([r["gf"] for r in rows]); cl = np.array([r["clean"] for r in rows])
    res = {}
    print("=====", name, "rows", len(rows), "ok", ok.sum(), "gf", gf.sum(), "clean ok", (ok & cl).sum(), "clean gf", (gf & cl).sum())
    for key in ("aim", "cmd"):
        v = np.array([r[key] for r in rows])
        mu = v[ok & cl].mean(0) if ref is None else ref[key]
        res[key] = mu
        e = np.linalg.norm(v - mu, axis=1)
        sel = cl & (ok | gf)
        p90, p95 = np.percentile(e[ok & cl], [90, 95])
        print(key, "ref", np.round(mu, 4), "AUC clean", round(auc(e[sel], gf[sel]), 3), "med ok/gf cm", np.round(100 * np.median(e[ok & cl]), 2), np.round(100 * np.median(e[gf & cl]), 2),
              "p90/p95 ok cm", np.round(100 * p90, 2), np.round(100 * p95, 2), "gf above p90", int((e[gf & cl] > p90).sum()), "above p95", int((e[gf & cl] > p95).sum()), "of", int((gf & cl).sum()))
        d = v - mu
        for c, j in (("x", 0), ("y", 1), ("z", 2)):
            print("   comp", c, "mean gf", np.round(100 * d[gf & cl, j].mean(), 2), "abs-median ok/gf", np.round(100 * np.median(abs(d[ok & cl, j])), 2), np.round(100 * np.median(abs(d[gf & cl, j])), 2), "AUC |dev|", round(auc(abs(d[:, j])[sel], gf[sel]), 3), "signed AUC", round(auc(d[:, j][sel], gf[sel]), 3))
        exy = np.linalg.norm(d[:, :2], axis=1)
        print("   xy AUC", round(auc(exy[sel], gf[sel]), 3), "med ok/gf", np.round(100 * np.median(exy[ok & cl]), 2), np.round(100 * np.median(exy[gf & cl]), 2))
    df = np.array([r["defl"] for r in rows])
    n = np.linalg.norm(df, axis=1)
    print("defl |.| median cm ok/gf", np.round(100 * np.median(n[ok & cl]), 2), np.round(100 * np.median(n[gf & cl]), 2), " z median ok/gf", np.round(100 * np.median(df[ok & cl, 2]), 2), np.round(100 * np.median(df[gf & cl, 2]), 2), "z p90 ok/gf", np.round(100 * np.percentile(df[ok & cl, 2], 90), 2), np.round(100 * np.percentile(df[gf & cl, 2], 90), 2))
    gap = np.array([r["t"] - r["tt"] if r["touching"] else 0 for r in rows])
    print("gap touch to close, median ok/gf", np.median(gap[ok & cl]), np.median(gap[gf & cl]), " gf with touch>20 before close", int((gap[gf & cl] > 20).sum()))
    return res

D = table("default")
ref = report("default", D)
G = table("grid")
report("grid (design reference)", G, ref)
