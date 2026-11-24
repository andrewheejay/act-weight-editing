"""Closed-loop size of the numerics noise: the harness on seeds 1000..1499 with different inference numerics.
variant: off1 (TF32 off, batch 1, deterministic, today's harness otherwise), off8 (TF32 off, batch of 8 copies, cudnn.benchmark on, as stock does it), on8 (TF32 on, same)
Usage: python scratch/diag_numerics.py VARIANT OUT"""
import json, sys, os, multiprocessing as mp, time
sys.path.insert(0, ".")
from awe import render, pool
from awe.ledger import Ledger
from awe.arms import make_spec
V, OUT = sys.argv[1], sys.argv[2]
def init(variant):
    pool._init("transfer_cube", "egl_mesa", 0)
    import torch, numpy as np
    from awe import rollout
    from lerobot.envs.utils import preprocess_observation
    pol = pool._W["policy"]
    tf = variant.startswith("on")
    torch.backends.cuda.matmul.allow_tf32 = tf; torch.backends.cudnn.allow_tf32 = tf
    n = int(variant[-1])
    if n > 1:
        torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False
        @torch.inference_mode()
        def chunk(image, agent_pos):
            ims = np.repeat(np.ascontiguousarray(image)[None], n, 0); st = np.repeat(np.asarray(agent_pos)[None], n, 0)
            b = preprocess_observation({"pixels": {"top": ims}, "agent_pos": st}); b["task"] = [""] * n
            return pol.post(pol.policy.predict_action_chunk(pol.pre(b)))[0].cpu().numpy()
        pol.chunk = chunk
def work(seed):
    from awe import rollout
    pool._W["policy"].set_arm(make_spec())
    arr = rollout.run_episode(pool._W["env"], pool._W["policy"], seed=seed)
    return rollout.summarize(arr, "original", seed)
if __name__ == "__main__":
    render.configure("egl_mesa")
    led = Ledger("transfer_cube", "setup", f"stage0 numerics diagnostic {V}")
    ctx = mp.get_context("spawn")
    t = time.time()
    with ctx.Pool(8, initializer=init, initargs=(V,)) as p, open(OUT, "w") as fh:
        for r in p.imap_unordered(work, range(1000, 1500), chunksize=4):
            led.log("original", r["seed"]); fh.write(json.dumps(r) + "\n")
    print(V, time.time() - t)
