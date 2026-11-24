"""The output-level recipes of the baseline arms (awe/post.py)."""
import json

import numpy as np
import pytest

import util
from awe import post as P
from awe import arms
from awe.arms import make_spec


def test_offset_adds_one_joint_and_keeps_the_input():
    a = np.arange(14, dtype=np.float32) / 10
    keep = a.copy()
    out = P.Post({"kind": "offset", "joint": 8, "delta": 0.25}, "transfer_cube")(0, a)
    assert np.array_equal(a, keep) and out.dtype == np.float32
    want = a.astype(np.float64)
    want[8] += 0.25
    assert np.allclose(out, want.astype(np.float32), atol=0)


def test_gain_moves_only_its_joints_about_the_mean(monkeypatch):
    m = np.random.RandomState(0).normal(size=(400, 14))
    monkeypatch.setattr(P, "load_mean", lambda task: m)
    a = np.random.RandomState(1).normal(size=14).astype(np.float32)
    out = P.Post({"kind": "gain", "g": 1.5, "joints": [7, 9]}, "transfer_cube")(3, a)
    want = a.astype(np.float64)
    for j in (7, 9):
        want[j] = m[3, j] + 1.5 * (float(a[j]) - m[3, j])
    assert np.allclose(out, want.astype(np.float32), atol=1e-7)
    keep = [j for j in range(14) if j not in (7, 9)]
    assert np.array_equal(out[keep], a[keep])


def test_approach_offset_stops_at_the_first_close():
    p = P.Post({"kind": "approach_offset", "joint": 7, "delta": 0.5}, "transfer_cube")
    grip = [0.1, 0.9, 0.9, 0.6, 0.4, 0.9, 0.1]  # closed at the start, opened, closing at 0.4, then later values stay off
    got = []
    for t, g in enumerate(grip):
        a = np.zeros(14, dtype=np.float32)
        a[13] = g
        got.append(float(p(t, a)[7]))
    # on at the start (it has not opened yet), on until the first command under 0.5 after opening, off afterwards
    assert got == [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]


def test_zero_offset_is_the_original_rollout_and_a_real_offset_is_not():
    env, pol = util.runner()
    orig, o = util.run(1000, make_spec("original"))
    zero, z = util.run(1000, make_spec("off0", post={"kind": "offset", "joint": 7, "delta": 0.0}))
    moved, _ = util.run(1000, make_spec("off", post={"kind": "offset", "joint": 13, "delta": -0.2}))
    assert np.array_equal(orig["action"], zero["action"]) and np.array_equal(orig["qpos"], zero["qpos"])
    assert np.abs(orig["qpos"] - moved["qpos"]).max() > 0.001
    # the first step of the first chunk is the policy's own action plus the shift
    assert np.allclose(moved["action"][0] - orig["action"][0], np.eye(14)[13] * -0.2, atol=1e-6)


def test_manifest_arm_carries_the_recipe(tmp_path):
    r = {"kind": "offset", "joint": 7, "delta": 0.02}
    m = {"operating_point": {"n_action_steps": 100}, "arms": [{"id": "original", "role": "original"}, {"id": "base_offset", "role": "baseline", "family": "constant_offset", "post": r}]}
    assert arms.from_manifest(m, "base_offset")["post"] == r
    assert arms.from_manifest(m, "original")["post"] is None
    f = tmp_path / "m.json"
    f.write_text(json.dumps(m))
    assert arms.resolve("base_offset", f)["post"] == r


def test_export_refuses_output_recipes(tmp_path):
    from awe.export_checkpoint import export

    f = tmp_path / "m.json"
    f.write_text(json.dumps({"arms": [{"id": "b", "role": "baseline", "post": {"kind": "offset", "joint": 7, "delta": 0.02}}]}))
    with pytest.raises(SystemExit):
        export("transfer_cube", "b", f, tmp_path / "out")
