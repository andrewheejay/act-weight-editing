"""Stage 2 open-loop head atlas. No rollouts: recorded states of the 1,681 grid episodes (stage 1 traces, default policy) are
re-rendered and fed through the policy with edited weights. No gradients.

python -m awe.atlas frames     render the decision frames, cache backbone features (cache/atlas/)
python -m awe.atlas effects    run every variant on every frame, write stage2/atlas/raw.npz
python -m awe.atlas report     stage2/atlas/heads.json, effect_maps.npz and the figures

Frames: decision steps 0 and 100 at all 41 x 41 poses, steps 200 and 300 at every second pose in x and y (21 x 21).
Variants: 48 heads, each at scale 0 and 2 (scale_head) and sharpened 2 and softened 0.5 (sharpen_head); 32 latent axes at -2, +2.
Chunks are compared in the checkpoint's normalized action units (each joint divided by its training std), all 100 steps.

FIXED BEFORE ANY SWEEP RESULT WAS LOOKED AT (2026-11-24, session 7):
  effect size   rms over the 100 steps and 14 joints of (edited chunk - original chunk), per frame
  failure map   benchmark failure (0/1) of the 41 x 41 default-policy grid, smoothed with a Gaussian of sigma 1 cm (2 grid
                steps), normalised at the edges
  overlap       Spearman correlation over the 1,681 grid poses of the effect size at STEP 0 with the smoothed failure map,
                per edit. The step 100 score and the scores against grasp failures only are recorded as secondary numbers.
  top 5         the 5 heads with the highest overlap of the mean of their x0 and x2 effect maps. For each the scale
                (x0 or x2) whose own map has the higher overlap goes into the split-half recheck.
  top 10        by success gain: the 10 single-head edits with the largest benchmark gain on seeds 20000 to 20499
                (ties by strict gain, then by arm id).
  kept          an edit is kept if the sign of its benchmark gain is the same, and not zero, in both halves of the design
                seeds. The recheck uses seeds 20500 to 20999, with the original re-run there too.
  cube signal   1 - (variance of the edited step-0 chunk over the grid poses, summed over steps and joints) / (same, original).
                Positive: the head carries position signal that is lost without it.
"""
import argparse
import json
import os

from . import render
from .paths import cache_dir, exp

STEPS = (0, 100, 200, 300)
JOINTS = [f"{s}_{j}" for s in ("left", "right") for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]
SMOOTH_SIGMA_M = 0.01
KH, KW = 15, 20  # image feature cells, 32 px each
TB = 10  # time bins of 10 steps


def frame_list():
    from .stage1 import GRID_N, grid_poses

    poses = grid_poses()
    out = []
    for t in STEPS:
        for i, p in enumerate(poses):
            if t < 200 or (i // GRID_N % 2 == 0 and i % GRID_N % 2 == 0):
                out.append((i, t))
    return poses, out


# ------------------------------------------------------------------ rendering
_R = {}


def _rinit(task, renderer):
    render.configure(renderer)
    from .env import FastAloha

    _R["env"] = FastAloha(task)
    _R["task"] = task


def _rframe(job):
    import numpy as np

    from .pool import trace_name

    i, t, pose = job
    env = _R["env"]
    z = np.load(cache_dir(_R["task"]) / "traces" / "grid" / trace_name(None, pose))
    q = z["qpos"][t].astype(np.float64)
    ph = env.physics
    with ph.reset_context():
        ph.data.qpos[:] = q
    img = env.image()
    st = env.tk.get_qpos(ph).astype(np.float32)
    r, g, b = (img[..., k].astype(int) for k in range(3))
    red = (r - np.maximum(g, b)) > 60
    cells = red[: KH * 32, : KW * 32].reshape(KH, 32, KW, 32).sum((1, 3)).astype(np.int16)
    ys, xs = np.nonzero(red)
    cen = (float(xs.mean()), float(ys.mean())) if len(xs) else (np.nan, np.nan)
    return i, t, img, st, cells, cen


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
    jobs = [(i, t, poses[i]) for i, t in fl]
    with ctx.Pool(6, initializer=_rinit, initargs=(task, renderer)) as pool:
        from .policy import Policy
        import torch

        P = Policy(task)
        m = P.policy.model
        feats = np.lib.format.open_memmap(d / "feats.npy", mode="w+", dtype=np.float32, shape=(N, 512, KH, KW))
        state = np.zeros((N, 14), np.float32)
        raw = np.zeros((N, 14), np.float32)
        cells = np.zeros((N, KH, KW), np.int16)
        cen = np.zeros((N, 2), np.float32)
        n = 0
        for i, t, img, st, c, ce in pool.imap_unordered(_rframe, jobs, chunksize=8):
            k = pos[(i, t)]
            b = P._batch(img, st)
            with torch.inference_mode():
                im = b["observation.images.top"]
                feats[k] = m.backbone(im)["feature_map"][0].cpu().numpy()
                state[k] = b["observation.state"][0].cpu().numpy()
            raw[k], cells[k], cen[k] = st, c, ce
            n += 1
            if n % 500 == 0:
                print(n, "/", N, flush=True)
    feats.flush()
    np.savez_compressed(d / "meta.npz", state=state, raw=raw, cells=cells, cen=cen, index=np.array([i for i, _ in fl]), step=np.array([t for _, t in fl]))
    print("frames done", N)


# ------------------------------------------------------------------ forward pass from cached backbone features
class Fwd:
    """The LeRobot ACT forward pass, restarted at any layer. Identical to ACTPolicy.predict_action_chunk (scratch/atlas_smoke.py)."""

    def __init__(self, task):
        import torch

        from .policy import Policy

        self.t = torch
        self.P = Policy(task)
        self.m = self.P.policy.model

    def embed(self, feats, state, latent=None):
        t, m = self.t, self.m
        B = len(feats)
        z = t.zeros(B, 32, device="cuda") if latent is None else latent.expand(B, 32)
        tok = [m.encoder_latent_input_proj(z)[None], m.encoder_robot_state_input_proj(state)[None]]
        cam_pos = m.encoder_cam_feat_pos_embed(feats)
        tok.append(m.encoder_img_feat_input_proj(feats).flatten(2).permute(2, 0, 1))
        pos = t.cat([m.encoder_1d_feature_pos_embed.weight[:, None].expand(-1, 1, -1), cam_pos.flatten(2).permute(2, 0, 1)])
        return t.cat(tok), pos

    def enc(self, x, pos, start=0, keep=False):
        outs = [x]
        for layer in self.m.encoder.layers[start:]:
            x = layer(x, pos_embed=pos)
            outs.append(x)
        if self.m.encoder.norm is not None:
            x = self.m.encoder.norm(x)
        return (x, outs) if keep else x

    def dec(self, mem, pos):
        m, t = self.m, self.t
        q = t.zeros(m.config.chunk_size, mem.shape[1], 512, device="cuda")
        o = m.decoder(q, mem, encoder_pos_embed=pos, decoder_pos_embed=m.decoder_pos_embed.weight[:, None])
        return m.action_head(o.transpose(0, 1))

    def cross_attn(self, mem, pos):
        """Per-head attention of the cross-attention (B, 8, 100, 302), the original weights."""
        m, t = self.m, self.t
        lay = m.decoder.layers[0]
        qpos = m.decoder_pos_embed.weight[:, None]
        x = t.zeros(m.config.chunk_size, mem.shape[1], 512, device="cuda")
        q = k = lay.maybe_add_pos_embed(x, qpos)
        x = x + lay.self_attn(q, k, value=x)[0]
        x = lay.norm1(x)
        w = lay.multihead_attn(query=lay.maybe_add_pos_embed(x, qpos), key=lay.maybe_add_pos_embed(mem, pos), value=mem, need_weights=True, average_attn_weights=False)[1]
        return w


def variants():
    from . import edits
    from .sweep import SHORT, arm_id

    out = []
    for b in edits.BLOCKS:
        for h in range(edits.N_HEADS):
            for nm, op in (("x0", {"op": "scale_head", "block": b, "head": h, "scale": 0}), ("x2", {"op": "scale_head", "block": b, "head": h, "scale": 2}),
                           ("sharp2", {"op": "sharpen_head", "block": b, "head": h, "factor": 2}), ("soft0.5", {"op": "sharpen_head", "block": b, "head": h, "factor": 0.5})):
                out.append((f"{SHORT[b]}.h{h}.{nm}", b, h, nm, op))
    for k in range(32):
        for s in (-2, 2):
            v = [0.0] * 32
            v[k] = 1.0
            out.append((f"lat{k}.{s:+d}", "latent", k, f"{s:+d}", {"op": "latent_shift", "direction": v, "size": s}))
    return out


def start_layer(block):
    if block.startswith("model.encoder.layers."):
        return int(block.split(".")[3])
    return {"latent": 0}.get(block, 9)  # 9: decoder only


def check_fwd(task, F, meta, feats):
    """Recorded rollout actions against the base chunk from the re-rendered frame (60 frames). Stock-forward identity: scratch/atlas_smoke.py."""
    import numpy as np

    t = F.t
    poses, fl = frame_list()
    from .pool import trace_name

    rec_err = []
    idx = np.linspace(0, len(fl) - 1, 60).astype(int)
    for k in idx:
        x, pos = F.embed(t.tensor(feats[k][None]).cuda(), t.tensor(meta["state"][k][None]).cuda())
        a = F.P.post(F.dec(F.enc(x, pos), pos)[0]).cpu().numpy()
        i, st = fl[k]
        z = np.load(cache_dir(task) / "traces" / "grid" / trace_name(None, poses[i]))
        rec_err.append((st, float(np.abs(a - z["action"][st : st + 100]).max())))
    return rec_err


def cmd_effects(task):
    import numpy as np

    from . import arms

    d = cache_dir(task) / "atlas"
    meta = np.load(d / "meta.npz")
    feats = np.load(d / "feats.npy", mmap_mode="r")
    F = Fwd(task)
    t = F.t
    P = F.P
    vs = variants()
    N = len(feats)
    rec = check_fwd(task, F, meta, feats)
    print("recorded actions minus base chunk from re-rendered frames, max abs:", {s: round(max(e for q, e in rec if q == s), 5) for s in STEPS}, "median", round(float(np.median([e for _, e in rec])), 6), flush=True)
    eff = np.zeros((len(vs), N), np.float32)
    joint = np.zeros((len(vs), 4, 14))
    timep = np.zeros((len(vs), 4, TB))
    cnt = np.zeros(4)
    s1 = np.zeros((len(vs) + 1, 100, 14))  # step 0 sums for the variance across poses
    s2 = np.zeros((len(vs) + 1, 100, 14))
    attn = np.zeros((4, 8, 302))
    attn_cell = np.zeros((N, 8))  # attention mass on image cells holding red pixels, per frame and cross head
    attn_cen = np.zeros((N, 8, 2))  # attention-weighted image centroid, in cells
    attn_query = np.zeros((4, 8, 10, 302))  # by chunk-step decile
    yy, xx = np.mgrid[0:KH, 0:KW]
    si = {s: k for k, s in enumerate(STEPS)}
    with t.inference_mode():
        for sn in STEPS:
            ids = np.nonzero(meta["step"] == sn)[0]
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
                attn[si[sn]] += wm.sum(0).double().cpu().numpy()
                attn_query[si[sn]] += W.reshape(len(idx), 8, 10, 10, 302).mean(3).sum(0).double().cpu().numpy()
                im = wm[:, :, 2:].cpu().numpy()
                red = (meta["cells"][idx] >= 20).reshape(len(idx), 1, -1)
                attn_cell[idx] = (im * red).sum(2)
                s = im.sum(2, keepdims=True)
                attn_cen[idx, :, 0] = ((im / s) * xx.reshape(1, 1, -1)).sum(2)
                attn_cen[idx, :, 1] = ((im / s) * yy.reshape(1, 1, -1)).sum(2)
                if sn == 0:
                    s1[-1] += base.sum(0).double().cpu().numpy()
                    s2[-1] += (base.double() ** 2).sum(0).cpu().numpy()
                cnt[si[sn]] += len(idx)
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
                    joint[v, si[sn]] += ad.mean(1).sum(0).double().cpu().numpy()
                    timep[v, si[sn]] += ad.reshape(len(idx), TB, 10, 14).mean((2, 3)).sum(0).double().cpu().numpy()
                    if sn == 0:
                        s1[v] += y.sum(0).double().cpu().numpy()
                        s2[v] += (y.double() ** 2).sum(0).cpu().numpy()
                P.set_arm(arms.make_spec("original"))
                if (a // 48) % 5 == 0:
                    print("step", sn, a, "/", len(ids), flush=True)
    joint /= cnt[None, :, None]
    timep /= cnt[None, :, None]
    n0 = int(cnt[0])
    var = s2 / n0 - (s1 / n0) ** 2  # population variance over poses, per (step, joint)
    out = exp(task) / "stage2" / "atlas"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "raw.npz", names=np.array([v[0] for v in vs]), eff=eff.astype(np.float16), index=meta["index"], step=meta["step"], joint=joint.astype(np.float32), timep=timep.astype(np.float32),
                        var_sum=var.sum((1, 2)), var_joint=var.sum(1).astype(np.float32), attn=(attn / cnt[:, None, None]).astype(np.float32),
                        attn_query=(attn_query / cnt[:, None, None, None]).astype(np.float32), attn_cell=attn_cell.astype(np.float32),
                        attn_cen=attn_cen.astype(np.float32), cen=meta["cen"], red_cells=(meta["cells"] >= 20).sum((1, 2)),
                        recheck=np.array([[s, e] for s, e in rec]))
    print("effects done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "effects", "report", "all"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    render.configure(render.chosen(a.task))
    if a.cmd in ("frames", "all"):
        cmd_frames(a.task)
    if a.cmd in ("effects", "all"):
        cmd_effects(a.task)
    if a.cmd in ("report", "all"):
        from . import atlas_report

        atlas_report.report(a.task)


if __name__ == "__main__":
    main()
