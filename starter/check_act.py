"""Sandbox check of LeRobot ACT facts used in the experiment spec (random weights, no hub access)."""
import importlib.util
import time

import torch
import torch.nn.functional as F

import lerobot
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

torch.manual_seed(0)
print("lerobot", lerobot.__version__, "| torch", torch.__version__)
print("migration module present:",
      importlib.util.find_spec("lerobot.processor.migrate_policy_normalization") is not None)

cfg = ACTConfig(
    input_features={
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    pretrained_backbone_weights=None,  # no downloads in the sandbox
    device="cpu",
)
policy = ACTPolicy(cfg).eval()
# Give every parameter a non-trivial value (xavier leaves biases at zero, which would hide effects).
with torch.no_grad():
    for n, p in policy.named_parameters():
        if p.dim() == 1 and "backbone" not in n:
            p.add_(0.05 * torch.randn_like(p))

sd = policy.state_dict()
n_total = sum(p.numel() for p in policy.parameters())
by = {}
for n, p in policy.named_parameters():
    top = n.split(".")[1]
    by[top] = by.get(top, 0) + p.numel()
print(f"total parameters {n_total:,}")
print("by module", {k: f"{v:,}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])})

attn_keys = [k for k in sd if k.endswith("out_proj.weight")]
print("attention blocks", len(attn_keys))
for k in attn_keys:
    print("  ", k, tuple(sd[k].shape))
print("decoder_pos_embed", tuple(sd["model.decoder_pos_embed.weight"].shape),
      "| action_head", tuple(sd["model.action_head.weight"].shape))

E, H = cfg.dim_model, cfg.n_heads
hd = E // H
print(f"dim {E} | heads {H} | head dim {hd} | weights touched per head {E*hd:,} = {100*E*hd/n_total:.4f}% of all")


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "observation.images.top": torch.rand(1, 3, 480, 640, generator=g),
        "observation.state": torch.randn(1, 14, generator=g),
    }


def chunk(pol, b):
    with torch.no_grad():
        return pol.predict_action_chunk(dict(b))  # (1, 100, 14)


b1, b2 = batch(1), batch(2)
base1, base2 = chunk(policy, b1), chunk(policy, b2)
print("chunk shape", tuple(base1.shape), "| repeat-call identical:", torch.equal(base1, chunk(policy, b1)))

# ---- 1. Head scaling on out_proj columns == scaling that head's output (unit test on MHA itself)
mha = torch.nn.MultiheadAttention(E, H).eval()
with torch.no_grad():
    mha.in_proj_bias.normal_(0, 0.1)
    mha.out_proj.bias.normal_(0, 0.1)
L, S = 7, 11
q, k, v = torch.randn(L, 1, E), torch.randn(S, 1, E), torch.randn(S, 1, E)


def manual_mha(m, q, k, v, scale=None):
    Wq, Wk, Wv = m.in_proj_weight.chunk(3)
    bq, bk, bv = m.in_proj_bias.chunk(3)
    Q = F.linear(q, Wq, bq).view(q.shape[0], H, hd).transpose(0, 1)  # (H, L, hd)
    K = F.linear(k, Wk, bk).view(k.shape[0], H, hd).transpose(0, 1)
    V = F.linear(v, Wv, bv).view(v.shape[0], H, hd).transpose(0, 1)
    A = torch.softmax(Q @ K.transpose(1, 2) / hd**0.5, dim=-1)
    O = A @ V  # (H, L, hd) per-head outputs
    if scale is not None:
        h, s = scale
        O[h] = O[h] * s
    cat = O.transpose(0, 1).reshape(q.shape[0], 1, E)
    return F.linear(cat, m.out_proj.weight, m.out_proj.bias)


with torch.no_grad():
    ref = mha(q, k, v)[0]
    print("manual MHA matches torch MHA:", float((manual_mha(mha, q, k, v) - ref).abs().max()))
    want = manual_mha(mha, q, k, v, scale=(3, 2.5))
    W0 = mha.out_proj.weight.clone()
    mha.out_proj.weight[:, 3 * hd:4 * hd] *= 2.5
    got = mha(q, k, v)[0]
    mha.out_proj.weight.copy_(W0)
    print("edit(out_proj columns of head 3 x2.5) vs scaling head 3 output: max abs diff",
          float((got - want).abs().max()))


def scale_head(pol, block, head, s):
    """block is a state-dict prefix such as 'model.encoder.layers.2.self_attn'."""
    W = pol.get_parameter(block + ".out_proj.weight")
    with torch.no_grad():
        W[:, head * hd:(head + 1) * hd] *= s


def restore(pol):
    pol.load_state_dict(sd)


sd = {k_: v_.clone() for k_, v_ in policy.state_dict().items()}

# ---- 2. Sham edit (x1) is bitwise identity; x0 on an encoder head changes the chunk
scale_head(policy, "model.encoder.layers.2.self_attn", 5, 1.0)
print("sham edit bitwise identical:", torch.equal(chunk(policy, b1), base1))
scale_head(policy, "model.encoder.layers.2.self_attn", 5, 0.0)
print("encoder L2H5 x0 changes chunk by (mean abs):", float((chunk(policy, b1) - base1).abs().mean()))
restore(policy)
scale_head(policy, "model.decoder.layers.0.multihead_attn", 1, 0.0)
print("decoder cross-attn H1 x0 changes chunk by (mean abs):", float((chunk(policy, b1) - base1).abs().mean()))
restore(policy)

# ---- 3. Decoder self-attention output is a constant (same for every action step and every observation)
captured = []
hook = policy.model.decoder.layers[0].self_attn.register_forward_hook(lambda m, i, o: captured.append(o[0].detach()))
chunk(policy, b1)
chunk(policy, b2)
hook.remove()
o1, o2 = captured
print("decoder self-attn out shape", tuple(o1.shape),
      "| spread across the 100 action steps:", float((o1 - o1[0:1]).abs().max()),
      "| difference between two observations:", float((o1 - o2).abs().max()),
      "| magnitude:", float(o1.abs().mean()))

# ---- 4. Reversing the decoder position table reverses the chunk in time (each action step is independent)
with torch.no_grad():
    policy.model.decoder_pos_embed.weight.copy_(sd["model.decoder_pos_embed.weight"].flip(0))
rev = chunk(policy, b1)
print("pos-table reversed -> chunk reversed: max abs diff", float((rev - base1.flip(1)).abs().max()))
restore(policy)

# ---- 5. VAE encoder is unused at inference
with torch.no_grad():
    for n, p in policy.named_parameters():
        if n.startswith("model.vae_encoder"):
            p.zero_()
print("zeroing every VAE-encoder weight changes the chunk by:", float((chunk(policy, b1) - base1).abs().max()))
restore(policy)

# ---- 6. Action-head bias edit == constant action offset
with torch.no_grad():
    policy.model.action_head.bias[3] += 0.2
d = chunk(policy, b1) - base1
print("action-head bias +0.2 on joint 3: offset on joint 3", float(d[..., 3].mean()),
      "| other joints", float(d[..., [0, 1, 2, 4]].abs().max()))
restore(policy)

# ---- 7. Encoder token count and where each token type sits
tok = []
h2 = policy.model.encoder.register_forward_hook(lambda m, i, o: tok.append(o.shape))
chunk(policy, b1)
h2.remove()
print("encoder tokens", tok[0][0], "= 1 latent + 1 state + image grid", tok[0][0] - 2, "(15 x 20)")

# ---- 8. Latent token is a constant at inference; a latent of z is the same as a bias edit
lat = policy.model.encoder_latent_input_proj
print("latent proj", tuple(lat.weight.shape), "| inference latent token = bias only")

# ---- 9. CPU timing of one decision (one forward pass)
for threads in (1, 2):
    torch.set_num_threads(threads)
    chunk(policy, b1)
    t0 = time.perf_counter()
    for _ in range(5):
        chunk(policy, b1)
    print(f"forward pass on CPU with {threads} thread(s): {(time.perf_counter() - t0) / 5 * 1e3:.0f} ms")

# ---- 10. Inference head inventory
inv = [k_[: -len(".out_proj.weight")] for k_ in attn_keys if "vae_encoder" not in k_]
print("inference attention blocks:", inv)
print("inference heads:", len(inv) * H, "| of which decoder self-attn (constant):", H)
