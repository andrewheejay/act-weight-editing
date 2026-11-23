"""Unit tests for the verdict rule, the exact test, and the seed draw."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from referee import beacon, stats
from referee.common import load_config

CFG = load_config()


def test_mcnemar_matches_scipy():
    scipy = pytest.importorskip("scipy.stats")
    for b, c in [(0, 0), (1, 0), (5, 0), (3, 3), (8, 2), (2, 8), (71, 8), (16, 2), (40, 39), (120, 85), (0, 17)]:
        want = 1.0 if b + c == 0 else scipy.binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
        assert stats.mcnemar_exact(b, c) == pytest.approx(want, rel=1e-9, abs=1e-15), (b, c)


def test_random_allowed():
    assert [stats.random_allowed(k) for k in (40, 39, 38, 19, 18, 0)] == [1, 1, 0, 0, -1, -1]


def _arms(n=1000, fail=170, fixed=70, broke=8, guard_broke=0):
    """Build paired outcome tables with exact counts."""
    orig = {s: s >= fail for s in range(n)}  # the first `fail` seeds fail
    edit = dict(orig)
    for s in range(fixed):
        edit[s] = True
    for s in range(fail, fail + broke):
        edit[s] = False
    g_orig = dict(orig)
    g_edit = dict(edit)
    for s in range(fail + broke, fail + broke + guard_broke):
        g_edit[s] = False
    return orig, edit, g_orig, g_edit


def _verdict(orig, edit, g_orig, g_edit, randoms, baselines, auto=(1.0, 6.2)):
    return stats.verdict(primary_orig=orig, primary_edit=edit, guard_orig=g_orig, guard_edit=g_edit, random_pairs=randoms, baseline_pairs=baselines, autogreedy_pair=auto, cfg=CFG, boot_seed=1)


def test_strong_win():
    v = _verdict(*_arms(), randoms=[(0.5, 6.2)] * 40, baselines={"deviation_gain": (2.0, 6.2)})
    assert v["ending"] == "strong_win"
    assert v["h1"]["fixed"] == 70 and v["h1"]["broke"] == 8
    assert v["h1"]["gain_points"] == pytest.approx(6.2)
    assert v["h1"]["share_removed"] == pytest.approx(62 / 170)
    lo, hi = v["h1"]["ci95"]["gain_points"]
    assert lo < 6.2 < hi and lo > 3 and hi < 9.5


def test_modest_win_when_under_a_quarter_of_failures():
    v = _verdict(*_arms(fixed=30, broke=5), randoms=[(0.1, 2.5)] * 40, baselines={"constant_offset": (1.0, 2.5)})
    assert v["h1"]["share_removed"] < 0.25 and v["h1"]["significant"]
    assert v["ending"] == "modest_win"


def test_partial_when_two_random_edits_match():
    randoms = [(7.0, 6.2), (6.2, 6.2)] + [(0.0, 6.2)] * 38
    v = _verdict(*_arms(), randoms=randoms, baselines={"constant_offset": (1.0, 6.2)})
    assert v["h2"]["n_gaining_as_much"] == 2 and not v["h2"]["pass"]
    assert v["ending"] == "partial"


def test_one_matching_random_edit_is_allowed():
    randoms = [(7.0, 6.2)] + [(0.0, 6.2)] * 39
    assert _verdict(*_arms(), randoms=randoms, baselines={})["ending"] == "strong_win"


def test_partial_when_a_baseline_ties():
    v = _verdict(*_arms(), randoms=[(0.0, 6.2)] * 40, baselines={"deviation_gain": (6.2, 6.2), "constant_offset": (1.0, 6.2)})
    assert v["h3"]["matching_baselines"] == ["deviation_gain"]
    assert v["ending"] == "partial"


def test_baseline_on_fewer_seeds_is_compared_like_for_like():
    # the ensembling arm ran on 200 seeds, where the edit itself gained 9.0 points
    v = _verdict(*_arms(), randoms=[(0.0, 6.2)] * 40, baselines={"temporal_ensembling": (7.0, 9.0)})
    assert v["ending"] == "strong_win"
    v = _verdict(*_arms(), randoms=[(0.0, 6.2)] * 40, baselines={"temporal_ensembling": (9.5, 9.0)})
    assert v["ending"] == "partial"


def test_partial_when_too_few_random_edits_exist():
    v = _verdict(*_arms(), randoms=[(0.0, 6.2)] * 10, baselines={})
    assert not v["h2"]["testable"] and v["ending"] == "partial"


def test_null_when_not_significant():
    v = _verdict(*_arms(fixed=12, broke=9), randoms=[(0.0, 0.3)] * 40, baselines={})
    assert not v["h1"]["significant"] and v["ending"] == "null"


def test_null_when_the_edit_hurts():
    v = _verdict(*_arms(fixed=5, broke=40), randoms=[(0.0, -3.5)] * 40, baselines={})
    assert v["h1"]["p_mcnemar"] < 0.05 and v["ending"] == "null"


def test_null_when_the_guard_fails():
    v = _verdict(*_arms(guard_broke=90), randoms=[(0.0, 6.2)] * 40, baselines={})
    assert v["h1"]["significant"] and not v["guard"]["ok"]
    assert v["guard"]["drop_points"] == pytest.approx(2.8)
    assert v["ending"] == "null"


def test_guard_allows_a_two_point_drop():
    v = _verdict(*_arms(guard_broke=82), randoms=[(0.0, 6.2)] * 40, baselines={})
    assert v["guard"]["drop_points"] == pytest.approx(2.0) and v["guard"]["ok"]
    assert v["ending"] == "strong_win"


def test_beacon_selfcheck_with_a_real_round():
    assert beacon.selfcheck() == []


def test_round_arithmetic():
    b = CFG["beacon"]
    g, p = b["genesis_time"], b["period"]
    assert beacon.time_of_round(1, g, p) == g
    assert beacon.first_round_at_or_after(g + 1, g, p) == 2
    assert beacon.first_round_at_or_after(g + 3, g, p) == 2
    assert beacon.first_round_at_or_after(g + 3.01, g, p) == 3
    r = beacon.target_round(g + 1000.5, b)
    assert beacon.time_of_round(r, g, p) >= g + 1000.5 + b["delay_seconds"] > beacon.time_of_round(r - 1, g, p)


def test_seed_block_is_stable_and_in_range():
    t = CFG["test_seeds"]
    a = beacon.seed_block(beacon.TEST_VECTOR["randomness"], "transfer_cube", "a" * 40, t)
    assert a == beacon.seed_block(beacon.TEST_VECTOR["randomness"], "transfer_cube", "a" * 40, t)
    assert a["start"] == 1_000_000_000 + int(a["digest"][:16], 16) % 1_000_000_000
    assert t["base"] <= a["start"] and a["end"] < 2**31 and a["count"] == 1000
    others = {beacon.seed_block(beacon.TEST_VECTOR["randomness"], task, c * 40, t)["start"] for task in ("transfer_cube", "insertion") for c in "ab"}
    assert len(others) == 4  # the task and the commit both change the draw


def test_verify_draw_accepts_a_fair_draw_and_rejects_a_forged_one(tmp_path):
    b, t = CFG["beacon"], CFG["test_seeds"]
    rnd = beacon.TEST_VECTOR["round"]
    pub = beacon.time_of_round(rnd, b["genesis_time"], b["period"]) - b["delay_seconds"]
    commit = "c" * 40
    blk = beacon.seed_block(beacon.TEST_VECTOR["randomness"], "transfer_cube", commit, t)
    com = {"body": {"freeze_commit": commit}, "witness": {"kind": "github-release", "repo": "x/y", "release_tag": "freeze-transfer_cube-1", "published_at": pub, "published_at_iso": "2023-08-23T15:11:09Z"}, "target_round": rnd}
    draw = {"task": "transfer_cube", "freeze_tag": "freeze-transfer_cube-1", "freeze_commit": commit, "beacon": {"round": rnd, "signature": beacon.TEST_VECTOR["signature"], "randomness": beacon.TEST_VECTOR["randomness"]}, "block": dict(blk)}
    (tmp_path / "commitment.json").write_text(json.dumps(com))
    (tmp_path / "draw.json").write_text(json.dumps(draw))
    run = lambda: subprocess.run([sys.executable, "-m", "referee.verify_draw", str(tmp_path)], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent))  # noqa: E731
    r = run()
    assert r.returncode == 0, r.stdout
    assert "FAIL" not in r.stdout

    draw["block"]["start"] += 1  # someone nudges the seeds
    (tmp_path / "draw.json").write_text(json.dumps(draw))
    r = run()
    assert r.returncode == 1 and "FAIL  seed block follows the formula" in r.stdout

    draw["block"] = dict(blk)
    com["witness"]["published_at"] = pub + 4  # the commitment was really published later than claimed
    (tmp_path / "draw.json").write_text(json.dumps(draw))
    (tmp_path / "commitment.json").write_text(json.dumps(com))
    r = run()
    assert r.returncode == 1 and "FAIL  round follows the rule" in r.stdout
