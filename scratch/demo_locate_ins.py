"""Locate the peg (red) and the socket (blue) in the first frame of each insertion training episode.

Calibration: render the simulator's reset frame with one object on an 11 by 11 grid of its start box and the other at its box centre
(reset + camera only, no rollout). The objects sit 5 cm above the table at reset, as in the demos' first frame (checked below).
Check: for every demo, render the reset frame at the estimated poses and compare it with the demo's first frame.
"""
import json, sys
sys.path.insert(0, "/home/ubuntu/act-weight-editing")
from awe import render
render.configure("egl_mesa")
import numpy as np
from awe.env import FastAloha
from awe.stage1 import INS_BOX, INS_CENTRE

env = FastAloha("insertion")

def masks(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return (r - np.maximum(g, b)) > 60, (b - np.maximum(r, g)) > 60

def centroid(img):
    out = []
    for m in masks(img):
        v, u = np.nonzero(m)
        out += [u.mean(), v.mean(), len(u)] if len(u) else [np.nan, np.nan, 0]
    return out  # red u, v, n, blue u, v, n

def pose(peg, sock):
    return [peg[0], peg[1], 0.05, 1, 0, 0, 0, sock[0], sock[1], 0.05, 1, 0, 0, 0]

def feats(u, v):
    u, v = (u - 320) / 320, (v - 240) / 240
    return np.stack([np.ones_like(u), u, v, u * v, u * u, v * v, u**2 * v, u * v**2, u**3, v**3], 1)

coef, cal = {}, {}
for body, k in (("peg", 0), ("sock", 3)):
    (x0, x1), (y0, y1) = INS_BOX[body]
    P, C = [], []
    for x in np.linspace(x0, x1, 11):
        for y in np.linspace(y0, y1, 11):
            at = dict(INS_CENTRE); at[body] = (x, y)
            env.reset(pose=pose(at["peg"], at["sock"]))
            c = centroid(env.image())
            P.append([x, y]); C.append([c[k], c[k + 1], c[k + 2]])
    P, C = np.array(P), np.array(C)
    A = feats(C[:, 0], C[:, 1])
    coef[body], *_ = np.linalg.lstsq(A, P, rcond=None)
    res = A @ coef[body] - P
    cal[body] = {"residual_max_m": np.abs(res).max(0).tolist(), "residual_rms_m": res.std(0).tolist(), "pixels_range": [int(C[:, 2].min()), int(C[:, 2].max())]}
    print(body, cal[body])

first = np.load("experiments/insertion/cache/demo_first_frames.npz")["first"]
rows = []
for k, f in enumerate(first):
    c = centroid(f)
    peg = feats(np.array([c[0]]), np.array([c[1]])) @ coef["peg"]
    sock = feats(np.array([c[3]]), np.array([c[4]])) @ coef["sock"]
    p, s = peg[0], sock[0]
    # reconstruction check: the reset frame at the estimated poses against the demo frame
    env.reset(pose=pose([float(np.clip(p[0], 0.1, 0.2)), float(np.clip(p[1], 0.4, 0.6))],
                        [float(np.clip(s[0], -0.2, -0.1)), float(np.clip(s[1], 0.4, 0.6))]))
    img = env.image().astype(int)
    diff = np.abs(img - f.astype(int)).sum(2)
    rm, bm = masks(f)
    r2, b2 = masks(img.astype(np.uint8))
    rows.append({"episode": k, "peg_x": float(p[0]), "peg_y": float(p[1]), "sock_x": float(s[0]), "sock_y": float(s[1]),
                 "peg_px": int(c[2]), "sock_px": int(c[5]),
                 "mean_abs_diff": float(diff.mean()), "peg_iou": float((rm & r2).sum() / max(1, (rm | r2).sum())), "sock_iou": float((bm & b2).sum() / max(1, (bm | b2).sum()))})
R = rows
inb = lambda v, lo, hi, tol=0.002: lo - tol <= v <= hi + tol
print("peg range", min(r["peg_x"] for r in R), max(r["peg_x"] for r in R), min(r["peg_y"] for r in R), max(r["peg_y"] for r in R))
print("sock range", min(r["sock_x"] for r in R), max(r["sock_x"] for r in R), min(r["sock_y"] for r in R), max(r["sock_y"] for r in R))
print("in box peg", sum(inb(r["peg_x"], .1, .2) and inb(r["peg_y"], .4, .6) for r in R), "sock", sum(inb(r["sock_x"], -.2, -.1) and inb(r["sock_y"], .4, .6) for r in R))
print("mean abs diff", np.mean([r["mean_abs_diff"] for r in R]).round(3), "max", np.max([r["mean_abs_diff"] for r in R]).round(3))
print("iou peg min/mean", min(r["peg_iou"] for r in R), np.mean([r["peg_iou"] for r in R]), "sock", min(r["sock_iou"] for r in R), np.mean([r["sock_iou"] for r in R]))
json.dump({"calibration": cal, "coef": {k: v.tolist() for k, v in coef.items()}, "demos": R}, open("experiments/insertion/stage1/demo_objects.json", "w"), indent=1)
