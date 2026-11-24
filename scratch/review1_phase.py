"""Matched-phase commanded geometry. For each clean-subset episode, the commanded gripper (FK of the action) at the first step of
the active chunk where the commanded gripper centre is below h metres above the untouched cube. Lateral offset along the finger axis,
forward offset along the gripper x axis, yaw of the finger axis. Also the measured gripper at the same step (contact-free only
before the first contact)."""
import sys
import numpy as np
from awe import render
render.configure(render.chosen("transfer_cube"))
from awe.stage1_analysis import SETTLE, auc, gripper_frames, load_features, load_records, load_trace

task = "transfer_cube"
frame = gripper_frames(task)

def run(name, hs=(0.12, 0.10, 0.08, 0.06)):
    recs = load_records(task, name)
    feats = {(f["seed"], str(f["pose"])): f for f in load_features(task, name)}
    rows = []
    for r in recs:
        f = feats[(r.get("seed"), str(r.get("pose")))]
        t = r["t_rclose"]
        if t is None or t < 0:
            continue
        tt = r["t_touch"]
        if not (tt is None or tt < 0 or tt >= 100 * (t // 100)):
            continue
        z = load_trace(task, name, r)
        q, act, tr, cube = z["qpos"].astype(float), z["action"].astype(float), z["tcpr"].astype(float), z["cube"].astype(float)
        c0 = cube[min(SETTLE, t)]
        row = dict(ok=r["benchmark_success"], gf=f["grasp_fail"], x0=r["cube_x0"], y0=r["cube_y0"])
        path = []
        for k in range(SETTLE, t + 1):  # whole command history. In the clean subset nothing touches the cube before the close chunk starts, so it is all contact-free up to the first touch

            qq = q[k + 1].copy(); qq[8:14] = act[k][7:13]
            R, a, b = frame(qq)
            mid = 0.5 * (a + b)
            ax = (b - a) / np.linalg.norm(b - a)
            off = c0 - mid
            path.append((k, mid, off @ ax, (R.T @ off)[0], np.arctan2(ax[1], ax[0]), mid[2] - c0[2]))
        row["t"] = t
        for h in hs:
            # first commanded crossing below h on the way down within the active chunk
            idx = next((i for i, p in enumerate(path) if p[5] < h), None)  # first commanded step below h
            if idx is None:
                row[h] = None
                continue
            k, mid, lat, fwd, yaw, hh = path[idx]
            meas = tr[k] - c0
            row[h] = dict(lat=lat, fwd=fwd, yaw=yaw, t=k, dxy=mid[:2] - c0[:2], meas_dxy=meas[:2], contact_free=(tt is None or tt < 0 or k < tt))
        # at the close
        p = path[-1]
        row["close"] = dict(lat=p[2], fwd=p[3], yaw=p[4], t=p[0], dxy=p[1][:2] - c0[:2], meas_dxy=None, contact_free=True)  # commands only; decided before any touch in the clean subset
        rows.append(row)
    return rows

def summarize(name, rows, hs):
    print("=====", name, len(rows), "ok", sum(r["ok"] for r in rows), "gf", sum(r["gf"] for r in rows))
    for h in list(hs) + ["close"]:
        sub = [(r, r[h]) for r in rows if r.get(h)]
        ok = np.array([r["ok"] for r, _ in sub]); gf = np.array([r["gf"] for r, _ in sub])
        sel = ok | gf
        lat = np.array([s["lat"] for _, s in sub]); fwd = np.array([s["fwd"] for _, s in sub]); yaw = np.array([s["yaw"] for _, s in sub])
        mu = lat[ok].mean(); muf = fwd[ok].mean(); muy = yaw[ok].mean()
        n_miss = len(rows) - len(sub)
        e = abs(lat - mu); p90 = np.percentile(e[ok], 90); p95 = np.percentile(e[ok], 95)
        print(f"   h={h} signed lat dev mean cm ok {100*(lat-mu)[ok].mean():.2f} gf {100*(lat-mu)[gf].mean():.2f} (gf share positive {(lat-mu)[gf].mean() and (lat[gf]>mu).mean():.2f}) | p90/p95 ok {100*p90:.2f}/{100*p95:.2f} cm; gf above p90 {int((e[gf]>p90).sum())} p95 {int((e[gf]>p95).sum())} of {int(gf.sum())}; ok above p90 {int((e[ok]>p90).sum())}/{int(ok.sum())}")
        print(f"h={h}: n={len(sub)} (no crossing {n_miss}) | lat ref {100*mu:.2f}cm: |dev| med ok {100*np.median(abs(lat-mu)[ok]):.2f} gf {100*np.median(abs(lat-mu)[gf]):.2f} AUC {auc(abs(lat-mu)[sel], gf[sel]):.3f} | fwd |dev| AUC {auc(abs(fwd-muf)[sel], gf[sel]):.3f} (gf med dev {100*np.median(fwd[gf]-muf):.2f}) | yaw |dev| AUC {auc(abs(yaw-muy)[sel], gf[sel]):.3f} gf med |dev| deg {np.degrees(np.median(abs(yaw-muy)[gf])):.2f} ok {np.degrees(np.median(abs(yaw-muy)[ok])):.2f} | signed lat AUC {auc((lat-mu)[sel], gf[sel]):.3f}")

if __name__ == "__main__":
    hs = (0.12, 0.10, 0.08, 0.06)
    for name in sys.argv[1:] or ["default", "grid"]:
        rows = run(name, hs)
        summarize(name, rows, hs)
