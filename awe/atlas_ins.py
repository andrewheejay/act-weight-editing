"""Stage 2 open-loop head atlas for insertion. No rollouts, no gradients. Same method as awe/atlas.py, with two grids instead of one.

python -m awe.atlas_ins frames     re-render the decision frames of the stage 1 grid episodes, cache backbone features (cache/atlas/)
python -m awe.atlas_ins effects    run every variant on every frame, write stage2/atlas/raw.npz
python -m awe.atlas_ins report     stage2/atlas/heads.json, effect_maps.npz and the figures (awe/atlas_ins_report.py)

Frames: recorded states of the original's grid_peg and grid_sock episodes (1,681 poses each, DEVIATIONS.md 2026-11-25). Steps 0, 100 and
200 at every pose, step 300 at every second pose in x and y (21 x 21). Step 200 is at every pose because the target failure's approach is
mostly commanded by the chunk decided there (the median first peg-socket contact is at step 304 in M and 286 in S).
Variants, the effect-size definition and the normalized chunk units are those of awe/atlas.py.

FIXED BEFORE ANY INSERTION SWEEP RESULT EXISTED (2026-11-25, session 22), the insertion reading of the transfer-cube rules:
  failure map   primary-score (strict) failure, 0/1, per 41 x 41 grid, Gaussian-smoothed with sigma 1 cm (4 cells in x, 2 in y),
                normalised at the edges. One map per grid.
  overlap       per edit: Spearman correlation over the 1,681 poses of a grid of the effect size at STEP 0 with that grid's smoothed failure
                map, averaged over the two grids. Secondary numbers: steps 100 and 200, and maps of class 3 (the target), of benchmark
                failure, and of the grasp failures (classes 0 to 2).
  top 5         the 5 heads with the highest overlap of the mean of their x0 and x2 effect maps. For each the scale (x0 or x2) with the
                higher own overlap goes into the split-half recheck.
  top 10        the 10 single-head edits with the largest PRIMARY (strict) gain on seeds 20000 to 20499 (ties by benchmark gain, then arm id).
                Transfer cube used benchmark because it was primary there.
  kept          an edit is kept if the sign of its strict gain is the same, and not zero, in both halves of the design seeds.
  object signal 1 - (variance of the edited step-0 chunk over one grid's poses, summed over steps and joints) / (same, original), per grid:
                peg signal from grid_peg, socket signal from grid_sock.
"""
import argparse
import json

from . import render
from .atlas import KH, KW, TB, Fwd, start_layer, variants
from .paths import cache_dir, exp

STEPS = (0, 100, 200, 300)
GRIDS = ("peg", "sock")
FULL_STEPS = (0, 100, 200)
GRID_N = 41


def frame_list():
    from .stage1 import insertion_grid_poses

    poses = {g: insertion_grid_poses(g) for g in GRIDS}
    out = []
    for t in STEPS:
        for g in GRIDS:
            for i in range(GRID_N * GRID_N):
                if t in FULL_STEPS or (i // GRID_N % 2 == 0 and i % GRID_N % 2 == 0):
                    out.append((g, i, t))
    return poses, out


_R = {}


def _rinit(task, renderer):
    render.configure(renderer)
    from .env import FastAloha

    _R["env"] = FastAloha(task)
    _R["task"] = task


def _rframe(job):
    import numpy as np

    from .pool import trace_name

    g, i, t, pose = job
    env = _R["env"]
    z = np.load(cache_dir(_R["task"]) / "traces" / f"grid_{g}" / trace_name(None, pose))
    ph = env.physics
    with ph.reset_context():
        ph.data.qpos[:] = z["qpos"][t].astype(np.float64)
    img = env.image()
    st = env.tk.get_qpos(ph).astype(np.float32)
    r, gr, b = (img[..., k].astype(int) for k in range(3))
    red, blue = (r - np.maximum(gr, b)) > 60, (b - np.maximum(r, gr)) > 60
    cells, cen = [], []
    for m in (red, blue):
        cells.append(m[: KH * 32, : KW * 32].reshape(KH, 32, KW, 32).sum((1, 3)).astype(np.int16))
        ys, xs = np.nonzero(m)
        cen.append((float(xs.mean()), float(ys.mean())) if len(xs) else (np.nan, np.nan))
    return g, i, t, img, st, np.stack(cells), np.array(cen)


def cmd_frames(task):
    import multiprocessing as mp

    import numpy as np

    renderer = render.chosen(task)
    poses, fl = frame_list()
    d = cache_dir(task) / "atlas"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "meta.npz").exists():
        print("frames cached")
        return
    ctx = mp.get_context("spawn")
    N = len(fl)
    pos = {k: n for n, k in enumerate(fl)}
    jobs = [(g, i, t, poses[g][i]) for g, i, t in fl]
    with ctx.Pool(6, initializer=_rinit, initargs=(task, renderer)) as pool:
        import torch

        from .policy import Policy

        P = Policy(task)
        m = P.policy.model
        feats = np.lib.format.open_memmap(d / "feats.npy", mode="w+", dtype=np.float32, shape=(N, 512, KH, KW))
        state = np.zeros((N, 14), np.float32)
        raw = np.zeros((N, 14), np.float32)
        cells = np.zeros((N, 2, KH, KW), np.int16)
        cen = np.zeros((N, 2, 2), np.float32)
        n = 0
        for g, i, t, img, st, c, ce in pool.imap_unordered(_rframe, jobs, chunksize=8):
            k = pos[(g, i, t)]
            b = P._batch(img, st)
            with torch.inference_mode():
                feats[k] = m.backbone(b["observation.images.top"])["feature_map"][0].cpu().numpy()
                state[k] = b["observation.state"][0].cpu().numpy()
            raw[k], cells[k], cen[k] = st, c, ce
            n += 1
            if n % 500 == 0:
                print(n, "/", N, flush=True)
    feats.flush()
    np.savez_compressed(d / "meta.npz", state=state, raw=raw, cells=cells, cen=cen, grid=np.array([g for g, _, _ in fl]), index=np.array([i for _, i, _ in fl]),
                        step=np.array([t for _, _, t in fl]))
    print("frames done", N)


def check_fwd(task, F, meta, feats, n=80):
    """Recorded rollout actions against the base chunk from the re-rendered frame, on n evenly spread frames."""
    import numpy as np

    from .pool import trace_name

    t = F.t
    poses, fl = frame_list()
    out = []
    for k in np.linspace(0, len(fl) - 1, n).astype(int):
        x, pos = F.embed(t.tensor(np.array(feats[k][None])).cuda(), t.tensor(meta["state"][k][None]).cuda())
        a = F.P.post(F.dec(F.enc(x, pos), pos)[0]).cpu().numpy()
        g, i, st = fl[k]
        z = np.load(cache_dir(task) / "traces" / f"grid_{g}" / trace_name(None, poses[g][i]))
        out.append((st, float(np.abs(a - z["action"][st : st + 100]).max())))
    return out


def cmd_effects(task):
    import numpy as np

    from . import arms

    d = cache_dir(task) / "atlas"
    meta = {k: v for k, v in np.load(d / "meta.npz").items()}
    feats = np.load(d / "feats.npy", mmap_mode="r")
    F = Fwd(task)
    t, P = F.t, F.P
    vs = variants()
    N = len(feats)
    rec = check_fwd(task, F, meta, feats)
    print("recorded actions minus base chunk from re-rendered frames, max abs:", {s: round(max(e for q, e in rec if q == s), 5) for s in STEPS}, "median", round(float(np.median([e for _, e in rec])), 6), flush=True)
    nG = len(GRIDS) * len(STEPS)
    eff = np.zeros((len(vs), N), np.float32)
    joint = np.zeros((len(vs), nG, 14))
    timep = np.zeros((len(vs), nG, TB))
    cnt = np.zeros(nG)
    s1 = np.zeros((len(vs) + 1, 2, 100, 14))  # step-0 sums, per grid, for the variance over poses
    s2 = np.zeros((len(vs) + 1, 2, 100, 14))
    attn = np.zeros((nG, 8, 302))
    attn_query = np.zeros((nG, 8, 10, 302))
    attn_cell = np.zeros((N, 8, 2))  # attention mass on image cells with red (peg) and blue (socket) pixels
    attn_cen = np.zeros((N, 8, 2))
    yy, xx = np.mgrid[0:KH, 0:KW]
    with t.inference_mode():
        for gi, g in enumerate(GRIDS):
            for si, sn in enumerate(STEPS):
                k = gi * len(STEPS) + si
                ids = np.nonzero((meta["step"] == sn) & (meta["grid"] == g))[0]
                for a in range(0, len(ids), 48):
                    idx = ids[a : a + 48]
                    P.set_arm(arms.make_spec("original"))
                    f = t.tensor(np.array(feats[idx])).cuda()
                    st = t.tensor(meta["state"][idx]).cuda()
                    x0, pos = F.embed(f, st)
                    mem, outs = F.enc(x0, pos, keep=True)
                    base = F.dec(mem, pos)
                    W = F.cross_attn(mem, pos).float()  # B 8 100 302
                    wm = W.mean(2)
                    attn[k] += wm.sum(0).double().cpu().numpy()
                    attn_query[k] += W.reshape(len(idx), 8, 10, 10, 302).mean(3).sum(0).double().cpu().numpy()
                    im = wm[:, :, 2:].cpu().numpy()
                    for c in range(2):
                        on = (meta["cells"][idx, c] >= 20).reshape(len(idx), 1, -1)
                        attn_cell[idx, :, c] = (im * on).sum(2)
                    s = im.sum(2, keepdims=True)
                    attn_cen[idx, :, 0] = ((im / s) * xx.reshape(1, 1, -1)).sum(2)
                    attn_cen[idx, :, 1] = ((im / s) * yy.reshape(1, 1, -1)).sum(2)
                    if sn == 0:
                        s1[-1, gi] += base.sum(0).double().cpu().numpy()
                        s2[-1, gi] += (base.double() ** 2).sum(0).cpu().numpy()
                    cnt[k] += len(idx)
                    for v, (name, b, h, nm, op) in enumerate(vs):
                        P.set_arm(arms.make_spec(name, [op]))
                        sl = start_layer(b)
                        if sl == 9:
                            y = F.dec(mem, pos)
                        elif b == "latent":
                            x, pp = F.embed(f, st)
                            y = F.dec(F.enc(x, pp), pp)
                        else:
                            y = F.dec(F.enc(outs[sl], pos, start=sl), pos)
                        dl = (y - base).float()
                        eff[v, idx] = dl.pow(2).mean((1, 2)).sqrt().cpu().numpy()
                        ad = dl.abs()
                        joint[v, k] += ad.mean(1).sum(0).double().cpu().numpy()
                        timep[v, k] += ad.reshape(len(idx), TB, 10, 14).mean((2, 3)).sum(0).double().cpu().numpy()
                        if sn == 0:
                            s1[v, gi] += y.sum(0).double().cpu().numpy()
                            s2[v, gi] += (y.double() ** 2).sum(0).cpu().numpy()
                    P.set_arm(arms.make_spec("original"))
                    if (a // 48) % 10 == 0:
                        print(g, "step", sn, a, "/", len(ids), flush=True)
    joint /= cnt[None, :, None]
    timep /= cnt[None, :, None]
    n0 = GRID_N * GRID_N
    var = s2 / n0 - (s1 / n0) ** 2  # (variant, grid, step, joint)
    out = exp(task) / "stage2" / "atlas"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "raw.npz", names=np.array([v[0] for v in vs]), eff=eff.astype(np.float16), grid=meta["grid"], index=meta["index"], step=meta["step"],
                        joint=joint.astype(np.float32), timep=timep.astype(np.float32), var_sum=var.sum((2, 3)), attn=(attn / cnt[:, None, None]).astype(np.float32),
                        attn_query=(attn_query / cnt[:, None, None, None]).astype(np.float32), attn_cell=attn_cell.astype(np.float32), attn_cen=attn_cen.astype(np.float32),
                        cen=meta["cen"], on_cells=(meta["cells"] >= 20).sum((2, 3)), recheck=np.array([[s, e] for s, e in rec]))
    print("effects done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "effects", "report", "all"])
    ap.add_argument("--task", default="insertion")
    a = ap.parse_args()
    render.configure(render.chosen(a.task))
    if a.cmd in ("frames", "all"):
        cmd_frames(a.task)
    if a.cmd in ("effects", "all"):
        cmd_effects(a.task)
    if a.cmd in ("report", "all"):
        from . import atlas_ins_report

        atlas_ins_report.report(a.task)


if __name__ == "__main__":
    main()
