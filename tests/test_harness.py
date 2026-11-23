import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from awe import edits, rollout  # noqa: E402
from awe.arms import make_spec  # noqa: E402
import util  # noqa: E402

SEEDS20 = list(range(1000, 1020))
ORIG = make_spec("original")
SHAM = make_spec("sham", ops=edits.sham_ops())


def test_fast_harness_matches_stock(tmp_path):
    """Joint positions of the fast harness equal the stock simulator's at every step, for 20 seeds.

    The harness plays its own policy actions. The stock environment then replays the same actions, with all three
    renders per step, and the full qpos trajectory must match exactly.
    """
    acts, mine = {}, {}
    for s in SEEDS20:
        arr, _ = util.run(s, ORIG)
        acts[str(s)] = arr["action"]
        mine[str(s)] = arr["qpos"]
    inp, out = tmp_path / "a.npz", tmp_path / "q.npz"
    np.savez(inp, **acts)
    subprocess.run([sys.executable, "-m", "awe.stock_replay", str(inp), str(out)], cwd=ROOT, check=True)
    ref = np.load(out)
    for k in acts:
        assert mine[k].shape == ref[k].shape
        assert np.array_equal(mine[k], ref[k]), f"seed {k} joint positions differ"


def _mha(W, bias, out_w, out_b, q, k, v, scale=None, head=None, hd=64, nh=8):
    """Multi-head attention written out, with one head's output scaled before the output projection."""
    Q = q @ W[: hd * nh].T + bias[: hd * nh]
    K = k @ W[hd * nh : 2 * hd * nh].T + bias[hd * nh : 2 * hd * nh]
    V = v @ W[2 * hd * nh :].T + bias[2 * hd * nh :]
    T, B, _ = Q.shape
    S = K.shape[0]
    sp = lambda x: x.view(x.shape[0], B, nh, hd).permute(1, 2, 0, 3)  # B, nh, L, hd
    Qh, Kh, Vh = sp(Q), sp(K), sp(V)
    att = torch.softmax(Qh @ Kh.transpose(-1, -2) / hd**0.5, dim=-1)
    O = (att @ Vh)  # B, nh, T, hd
    if head is not None:
        O = O.clone()
        O[:, head] *= scale
    O = O.permute(2, 0, 1, 3).reshape(T, B, nh * hd)
    return O @ out_w.T + out_b


def test_head_scaling_exact():
    """Editing the out_proj matrix gives the same output as scaling the head's output directly, to 1e-6."""
    from awe.policy import Policy

    pol = Policy("transfer_cube", device="cpu")
    torch.manual_seed(0)
    worst = 0.0
    for block in edits.BLOCKS:
        mod = pol.policy.get_submodule(block).double()
        base = {k: v.clone() for k, v in mod.state_dict().items()}
        W, b = base["in_proj_weight"], base["in_proj_bias"]
        ow, ob = base["out_proj.weight"], base["out_proj.bias"]
        L, S, B = 5, 9, 3
        q = torch.randn(L, B, 512, dtype=torch.float64)
        kv = torch.randn(S, B, 512, dtype=torch.float64)
        for head in (0, 3, 7):
            for s in (0.0, 0.5, 2.0, 4.0):
                sd = {k: v.clone() for k, v in base.items()}
                edits.apply_op({block + "." + k: v for k, v in sd.items()}, {"op": "scale_head", "block": block, "head": head, "scale": s})
                mod.load_state_dict(sd)
                got = mod(q, kv, kv, need_weights=False)[0]
                want = _mha(W, b, ow, ob, q, kv, kv, scale=s, head=head)
                worst = max(worst, float((got - want).abs().max()))
        mod.load_state_dict(base)
    assert worst < 1e-6, worst


def test_sham_edit_identical():
    """Scaling every head by 1 leaves the predicted chunk, the weights and a whole rollout bit for bit identical."""
    env, pol = util.runner()
    env.reset(seed=1000)
    image = env.image()
    state = np.zeros(14)
    pol.set_arm(ORIG)
    base = pol.chunk(image, state)
    sd0 = {k: v.clone() for k, v in pol.policy.state_dict().items()}
    pol.set_arm(SHAM)
    got = pol.chunk(image, state)
    for k, v in pol.policy.state_dict().items():
        assert torch.equal(v, sd0[k]), k
    assert np.array_equal(base, got)
    for s in (1000, 1005):
        _, a = util.run(s, ORIG)
        _, b = util.run(s, SHAM)
        for key in ("benchmark_success", "strict_success", "traj_hash"):
            assert a[key] == b[key], (s, key)


def test_repeat_run_identical():
    """The same seed gives the same actions and the same trajectory twice, including after other episodes."""
    for s in (1000, 1003):
        a1, r1 = util.run(s, ORIG)
        util.run(1001, ORIG)  # an episode in between must not leak state
        a2, r2 = util.run(s, ORIG)
        assert np.array_equal(a1["action"], a2["action"])
        assert r1["traj_hash"] == r2["traj_hash"]


def test_seed_lock(tmp_path, monkeypatch):
    """A seed of 1,000,000,000 or more raises unless the referee supplies a draw that covers it."""
    from awe.seeds import SeedLockError, check_seed

    monkeypatch.delenv("AWE_JUDGE_DRAW", raising=False)
    for s in (10**9, 10**9 + 7, 2 * 10**9):
        with pytest.raises(SeedLockError):
            check_seed(s)
    for s in (5, 999, 19999, 40000):  # outside every allowed set
        with pytest.raises(SeedLockError):
            check_seed(s)
    for s in (1000, 1499, 20000, 20999, 30000, 30499):
        assert check_seed(s) == s
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"block": {"start": 10**9 + 5, "end": 10**9 + 1004}}))
    monkeypatch.setenv("AWE_JUDGE_DRAW", str(draw))
    assert check_seed(10**9 + 5) == 10**9 + 5
    assert check_seed(10**9 + 1004) == 10**9 + 1004
    for s in (10**9 + 4, 10**9 + 1005, 10**9):
        with pytest.raises(SeedLockError):
            check_seed(s)
    monkeypatch.delenv("AWE_JUDGE_DRAW")
    # the command the referee calls refuses before it loads anything
    sf = tmp_path / "s.json"
    sf.write_text(json.dumps({"seeds": [10**9 + 5]}))
    out = tmp_path / "o.jsonl"
    env = {k: v for k, v in os.environ.items() if k != "AWE_JUDGE_DRAW"}
    r = subprocess.run(
        [sys.executable, "-m", "awe.run_arm", "--task", "transfer_cube", "--arm", "original", "--seeds-file", str(sf), "--out", str(out)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "SeedLockError" in r.stderr
    assert not out.exists()
