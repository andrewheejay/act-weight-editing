"""Why harness and stock disagree. Images at reset (stock vs harness env) and chunks at batch 1 vs batch 8 vs cudnn.benchmark.
Only resets, no rollout: no ledger line. Seeds 1000..1007 (reproduction set)."""
import sys
sys.path.insert(0, ".")
from awe import render
render.configure("egl_mesa")
from awe.policy import Policy  # LLVM import order
import numpy as np, torch, hashlib
from awe.env import FastAloha
import gymnasium as gym, gym_aloha  # noqa
from awe.paths import TASKS
pol = Policy("transfer_cube")
env = FastAloha("transfer_cube")
imgs, states = [], []
for s in range(1000, 1008):
    st = env.reset(seed=s); im = env.image().copy()
    imgs.append(im); states.append(np.array(st))
stock = gym.make(f"gym_aloha/{TASKS['transfer_cube']}", obs_type="pixels_agent_pos", render_mode="rgb_array").unwrapped
for i, s in enumerate(range(1000, 1008)):
    obs, _ = stock.reset(seed=s)
    d = np.abs(obs["pixels"]["top"].astype(int) - imgs[i].astype(int))
    print("seed", s, "image differs in", int((d > 0).sum()), "max", int(d.max()), "state equal", np.array_equal(np.asarray(obs["agent_pos"]), states[i]))
def chunk(ims, sts):
    from lerobot.envs.utils import preprocess_observation
    b = preprocess_observation({"pixels": {"top": np.stack(ims)}, "agent_pos": np.stack(sts)})
    b["task"] = [""] * len(ims)
    with torch.inference_mode():
        return pol.post(pol.policy.predict_action_chunk(pol.pre(b))).cpu().numpy()
one = np.concatenate([chunk([imgs[i]], [states[i]]) for i in range(8)])
eight = chunk(imgs, states)
torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False
oneb = np.concatenate([chunk([imgs[i]], [states[i]]) for i in range(8)])
eightb = chunk(imgs, states)
for name, x in (("batch8", eight), ("batch1+benchmark", oneb), ("batch8+benchmark", eightb)):
    print(name, "max |diff| vs batch1", float(np.abs(x - one).max()), "identical", np.array_equal(x, one))
print("--- tf32 off")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
a = np.concatenate([chunk([imgs[i]], [states[i]]) for i in range(8)])
a2 = np.concatenate([chunk([imgs[i]], [states[i]]) for i in range(8)])
print("repeat identical", np.array_equal(a, a2), "vs tf32 batch1", float(np.abs(a - one).max()))
b = chunk(imgs, states)
torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False
c = np.concatenate([chunk([imgs[i]], [states[i]]) for i in range(8)])
d = chunk(imgs, states)
for name, x in (("batch8", b), ("batch1+benchmark", c), ("batch8+benchmark", d)):
    print("tf32off", name, "max |diff| vs tf32off batch1", float(np.abs(x - a).max()))
