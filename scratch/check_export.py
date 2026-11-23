"""Export round trip: the original exports byte-identical weights, and an edited arm's policy gives the chunk the in-memory edit gives."""
import sys, json, hashlib
sys.path.insert(0, "/home/ubuntu/act-weight-editing")
from awe import render; render.configure("egl_mesa")
import numpy as np, torch
from safetensors.torch import load_file
from awe import export_checkpoint
from awe.policy import Policy
from awe.arms import make_spec
from awe.paths import checkpoint_dir
from awe.env import FastAloha
m = "/tmp/man.json"
ops = [{"op": "scale_head", "block": "model.encoder.layers.2.self_attn", "head": 5, "scale": 2}, {"op": "sharpen_head", "block": "model.decoder.layers.0.multihead_attn", "head": 1, "factor": 2}, {"op": "latent_shift", "direction": [1.0] + [0.0] * 31, "size": 1.5}]
json.dump({"task": "transfer_cube", "operating_point": {"n_action_steps": 100}, "arms": [{"id": "original", "role": "original"}, {"id": "e1", "role": "agent_edit", "ops": ops}]}, open(m, "w"))
export_checkpoint.export("transfer_cube", "original", m, "/tmp/exp_orig")
export_checkpoint.export("transfer_cube", "e1", m, "/tmp/exp_e1")
a, b = load_file(str(checkpoint_dir("transfer_cube") / "model.safetensors")), load_file("/tmp/exp_orig/model.safetensors")
print("orig tensors equal:", set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), "n", len(a))
e = load_file("/tmp/exp_e1/model.safetensors")
print("edited keys:", [k for k in a if not torch.equal(a[k], e[k])])
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
export_checkpoint.export("transfer_cube", "e1", m, "/tmp/exp_e1b")
print("export deterministic:", h("/tmp/exp_e1/model.safetensors") == h("/tmp/exp_e1b/model.safetensors"), "orig file hash same as checkpoint:", h("/tmp/exp_orig/model.safetensors") == h(checkpoint_dir("transfer_cube") / "model.safetensors"))
env = FastAloha("transfer_cube"); env.reset(seed=1000); img = env.image(); st = np.zeros(14)
mem = Policy("transfer_cube"); mem.set_arm(make_spec("e1", ops)); c1 = mem.chunk(img, st)
disk = Policy("transfer_cube", ckpt="/tmp/exp_e1"); c2 = disk.chunk(img, st)
base = Policy("transfer_cube"); c0 = base.chunk(img, st)
print("edit on disk equals edit in memory:", np.array_equal(c1, c2), "| edit changes the chunk:", float(np.abs(c1 - c0).max()))
