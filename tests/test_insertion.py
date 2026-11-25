"""The insertion task in the fast harness: physics and reward equal the stock simulator, and the two scores are consistent."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1000, 1001, 1002, 1003, 1004, 1005]


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if not (ROOT / "experiments/insertion/stage0/renderer.json").exists() or not (ROOT / "checkpoints/insertion").exists():
        pytest.skip("insertion checkpoint not converted here")
    d = tmp_path_factory.mktemp("ins")
    out = d / "fast.npz"
    subprocess.run([sys.executable, "tests/insertion_probe.py", str(out), *map(str, SEEDS)], cwd=ROOT, check=True)
    return d


def test_insertion_harness_matches_stock(probe):
    """The stock environment replays the harness's actions. qpos at every step and the reward at every step must be equal."""
    ref = probe / "stock.npz"
    subprocess.run([sys.executable, "-m", "awe.stock_replay", str(probe / "fast_act.npz"), str(ref), "insertion"], cwd=ROOT, check=True)
    fast, stock = np.load(probe / "fast.npz"), np.load(ref)
    for s in SEEDS:
        assert np.array_equal(fast[f"{s}_q"], stock[str(s)]), s
        assert np.array_equal(fast[f"{s}_r"], stock[f"{s}_reward"]), s


def test_insertion_strict_implies_benchmark(probe):
    recs = json.loads((probe / "fast.json").read_text())
    for s, r in recs.items():
        assert r["benchmark_success"] or not r["strict_success"], s


def test_strict_insertion_needs_25_steps_and_every_condition():
    from awe.measure import strict_insertion

    n = 60
    base = {k: np.ones(n, dtype=bool) for k in ("pin", "right", "left")}
    base["table"] = np.zeros(n, dtype=bool)
    assert strict_insertion(base)
    for k in ("pin", "right", "left"):
        a = {x: v.copy() for x, v in base.items()}
        a[k][20:40] = False  # longest stretch without the break is 20
        assert not strict_insertion(a), k
    a = {x: v.copy() for x, v in base.items()}
    a["table"][20:40] = True
    assert not strict_insertion(a)
    a = {x: v.copy() for x, v in base.items()}
    a["pin"][:35] = False  # 25 steps left
    assert strict_insertion(a)
