"""Locate the cube in the first frame of each training episode.

Calibration: render the simulator's reset frame at an 11 by 11 grid of known cube positions (reset + camera only, no rollout).
"""
import json, sys
sys.path.insert(0, "/home/ubuntu/act-weight-editing")
from awe import render
render.configure("egl_mesa")
import numpy as np
from awe.env import FastAloha

def red_mask(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return (r - np.maximum(g, b)) > 60

def centroid(img):
    m = red_mask(img)
    v, u = np.nonzero(m)
    return (u.mean(), v.mean(), len(u)) if len(u) else (np.nan, np.nan, 0)

env = FastAloha("transfer_cube")
P, C = [], []
for x in np.linspace(0, 0.2, 11):
    for y in np.linspace(0.4, 0.6, 11):
        env.reset(pose=[x, y, 0.05, 1, 0, 0, 0])
        u, v, n = centroid(env.image())
        P.append([x, y]); C.append([u, v, n])
P, C = np.array(P), np.array(C)
print("pixels per cube", C[:, 2].min(), C[:, 2].max())
def feats(u, v):
    u, v = (u - 320) / 320, (v - 240) / 240
    return np.stack([np.ones_like(u), u, v, u * v, u * u, v * v, u**2 * v, u * v**2, u**3, v**3], 1)
A = feats(C[:, 0], C[:, 1])
coef, *_ = np.linalg.lstsq(A, P, rcond=None)
res = A @ coef - P
print("calibration residual, max abs (m):", np.abs(res).max(0), "rms", res.std(0))
first = np.load("experiments/transfer_cube/cache/demo_first_frames.npz")["first"]
D = np.array([centroid(f) for f in first])
xy = feats(D[:, 0], D[:, 1]) @ coef
print("demo cube px min/max", D[:, 2].min(), D[:, 2].max())
print("demo xy range", xy.min(0), xy.max(0))
out = {"calibration": {"grid": 11, "coef": coef.tolist(), "residual_max_m": np.abs(res).max(0).tolist(), "cube_pixels_range": [int(C[:, 2].min()), int(C[:, 2].max())]},
       "demos": [{"episode": k, "x": float(xy[k, 0]), "y": float(xy[k, 1]), "pixels": int(D[k, 2]), "u": float(D[k, 0]), "v": float(D[k, 1])} for k in range(len(xy))]}
json.dump(out, open("experiments/transfer_cube/stage1/demo_cubes.json", "w"), indent=1)
print("in square:", int(((xy[:, 0] > -0.002) & (xy[:, 0] < 0.202) & (xy[:, 1] > 0.398) & (xy[:, 1] < 0.602)).sum()), "of 50")
