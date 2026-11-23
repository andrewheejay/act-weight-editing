"""Weight edits as operations on a state dict. Nothing here computes a gradient.

Every op is a closed-form change of checkpoint tensors, so an arm made of ops exports as a normal checkpoint.
"""
import torch

HEAD_DIM = 64
N_HEADS = 8
BLOCKS = [f"model.encoder.layers.{i}.self_attn" for i in range(4)] + [
    "model.decoder.layers.0.multihead_attn",
    "model.decoder.layers.0.self_attn",
]
LATENT = "model.encoder_latent_input_proj"


def _cols(head):
    if not 0 <= int(head) < N_HEADS:
        raise ValueError(f"head {head} out of range")
    return slice(int(head) * HEAD_DIM, (int(head) + 1) * HEAD_DIM)


def touched(op: dict) -> list[str]:
    k = op["op"]
    if k == "scale_head":
        return [op["block"] + ".out_proj.weight"]
    if k == "sharpen_head":
        return [op["block"] + ".in_proj_weight", op["block"] + ".in_proj_bias"]
    if k == "latent_shift":
        return [LATENT + ".bias"]
    raise ValueError(f"unknown op {k}")


def apply_op(sd: dict, op: dict) -> None:
    """Apply one op in place to the tensors of sd."""
    k = op["op"]
    with torch.no_grad():
        if k == "scale_head":
            assert op["block"] in BLOCKS, op["block"]
            sd[op["block"] + ".out_proj.weight"][:, _cols(op["head"])] *= float(op["scale"])
        elif k == "sharpen_head":
            assert op["block"] in BLOCKS, op["block"]
            c = _cols(op["head"])  # the first 512 rows of in_proj are the queries
            sd[op["block"] + ".in_proj_weight"][c] *= float(op["factor"])
            sd[op["block"] + ".in_proj_bias"][c] *= float(op["factor"])
        elif k == "latent_shift":
            # z = size * direction instead of 0 is exactly a bias change of the latent projection
            W = sd[LATENT + ".weight"]
            d = torch.as_tensor(op["direction"], dtype=W.dtype, device=W.device)
            assert d.shape == (32,), d.shape
            sd[LATENT + ".bias"] += float(op["size"]) * (W @ d)
        else:
            raise ValueError(f"unknown op {k}")


def apply_ops(sd: dict, ops: list) -> None:
    for op in ops:
        apply_op(sd, op)


def sham_ops() -> list:
    return [{"op": "scale_head", "block": b, "head": h, "scale": 1} for b in BLOCKS for h in range(N_HEADS)]
