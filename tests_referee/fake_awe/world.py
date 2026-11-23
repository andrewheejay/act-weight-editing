from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

P_ORIG = {"transfer_cube": 0.83, "insertion": 0.21}
TEST_BASE = 1_000_000_000


def h01(*parts) -> float:
    d = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64


def effect_of(arm: str, manifest: dict | None) -> dict:
    """fix is the chance a failing seed is fixed, brk the chance a passing seed is broken."""
    if arm in ("original", "sham"):
        return {"fix": 0.0, "brk": 0.0, "key": "none"}
    if manifest:
        for a in manifest.get("arms", []):
            if a["id"] == arm:
                s = a.get("sim", {})
                if a.get("role") in ("original", "original_default"):
                    return {"fix": 0.0, "brk": 0.0, "key": "none"}
                return {"fix": s.get("fix", 0.0), "brk": s.get("brk", 0.0), "key": arm}
    # arms used before a manifest exists, such as sweep arms, carry their effect in the name: NAME~fix~brk
    if "~" in arm:
        _, fix, brk = arm.split("~")
        return {"fix": float(fix), "brk": float(brk), "key": arm}
    return {"fix": 0.0, "brk": 0.0, "key": arm}


def check_seed(seed: int) -> None:
    if seed >= TEST_BASE:
        draw = os.environ.get("AWE_JUDGE_DRAW")
        if not draw:
            raise SystemExit(f"seed {seed} is a test seed and no referee draw file was given")
        b = json.loads(Path(draw).read_text())["block"]
        if not b["start"] <= seed <= b["end"]:
            raise SystemExit(f"seed {seed} is outside the drawn block")


def rollout(task: str, arm: str, seed: int, manifest: dict | None) -> dict:
    check_seed(seed)
    eff = effect_of(arm, manifest)
    ok = h01(task, seed, "base") < P_ORIG[task]
    if eff["key"] != "none":
        if not ok and h01(task, seed, eff["key"], "fix") < eff["fix"]:
            ok = True
        elif ok and h01(task, seed, eff["key"], "brk") < eff["brk"]:
            ok = False
    strict = ok and h01(task, seed, "strict") < 0.93
    cls = "4b" if strict else ("4a" if ok else ("1" if h01(task, seed, "cls") < 0.55 else "2"))
    return {
        "seed": seed,
        "arm": arm,
        "benchmark_success": ok,
        "strict_success": strict,
        "max_reward": 4 if ok else 2,
        "failure_class": cls,
        "traj_hash": hashlib.sha256(f"{task}|{seed}|{eff['key']}|{ok}|{strict}".encode()).hexdigest()[:24],
    }


def tensors(kind: str = "original", backbone: bool = False) -> dict:
    import numpy as np

    rng = np.random.default_rng(7)
    t = {
        "model.backbone.layer1.weight": rng.standard_normal((8, 8)).astype("float32"),
        "model.encoder.layers.0.self_attn.out_proj.weight": rng.standard_normal((16, 16)).astype("float32"),
        "model.encoder.layers.1.self_attn.out_proj.weight": rng.standard_normal((16, 16)).astype("float32"),
        "model.decoder.layers.0.multihead_attn.out_proj.weight": rng.standard_normal((16, 16)).astype("float32"),
        "model.action_head.weight": rng.standard_normal((14, 16)).astype("float32"),
        "model.vae_encoder.layers.0.weight": rng.standard_normal((16, 16)).astype("float32"),
    }
    if kind == "agent_edit":
        t["model.encoder.layers.0.self_attn.out_proj.weight"][:, 0:2] *= 2.0
        if backbone:
            t["model.backbone.layer1.weight"][0, 0] += 1.0
    elif kind == "autogreedy":
        t["model.encoder.layers.1.self_attn.out_proj.weight"][:, 2:4] *= 2.0
    return t
