"""End-to-end tests of the referee on a simulated experiment.

A scripted agent with a fake harness plays the whole protocol, from an empty repo to DONE, while
the real referee code checks gates, rejects a bad freeze, draws seeds, runs the test and writes the
verdict. Other tests then break one rule at a time and check that the referee notices.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pytest

from conftest import PY, drive, git, status


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """Play the full two-task campaign once, with a review that asks for changes and a bad first freeze."""
    import conftest

    tmp = tmp_path_factory.mktemp("campaign")
    root = conftest.repo.__wrapped__(tmp)
    env = conftest.env.__wrapped__(tmp)
    env.update({"STUB_REVISE_FIRST": "g1", "STUB_BAD_FIRST_FREEZE": "1"})
    trail = drive(root, env, max_steps=150)
    return root, env, trail


def test_campaign_reaches_done(played):
    root, env, trail = played
    assert trail[-1][0] == "DONE", trail[-6:]
    seen = [(s, t, st, a, g, r) for s, t, st, a, g, r in trail]
    # the referee and the reviewer each took their turns, in order, for the first task
    tc = [x for x in seen if x[1] == "transfer_cube"]
    kinds = [(x[0], x[3] or x[4]) for x in tc if x[0] != "AGENT_TURN"]
    assert kinds[:4] == [("JUDGE_TURN", "confirm_g0"), ("REVIEW_TURN", "g1"), ("REVIEW_TURN", "g1"), ("JUDGE_TURN", "stamp_g1")]
    assert ("REVIEW_TURN", "g2.1") in kinds and ("REVIEW_TURN", "g2.2") in kinds
    assert kinds.count(("JUDGE_TURN", "freeze")) == 2 and kinds.count(("JUDGE_TURN", "test")) == 1
    assert any(x[1] == "insertion" and x[3] == "test" for x in seen)
    assert trail[-2][2] == 7  # publish prep comes last


def test_review_round_trip(played):
    root, _, _ = played
    r1 = json.loads((root / "reviews" / "transfer_cube-g1-r1.json").read_text())
    r2 = json.loads((root / "reviews" / "transfer_cube-g1-r2.json").read_text())
    assert r1["verdict"] == "revise" and r1["blocking_issues"]
    assert (root / "reviews" / "transfer_cube-g1-r1-response.md").exists()
    assert r2["verdict"] == "pass"
    stamp = json.loads((root / "reviews" / "transfer_cube-g1-passed.json").read_text())
    assert stamp["review_rounds"] == 2 and stamp["review_unresolved"] is False
    assert stamp["recomputed"]["primary_metric"] == "benchmark" and stamp["recomputed"]["n_action_steps"] == 100


def test_bad_freeze_was_rejected_then_accepted(played):
    root, _, _ = played
    fz = root / "experiments" / "transfer_cube" / "freeze"
    rej = (fz / "REJECTED-1.md").read_text()
    assert "edit.no_backbone" in rej and "freeze.edit_sha" not in rej
    assert (fz / "REQUEST-1.rejected.json").exists()
    acc = json.loads((fz / "ACCEPTED.json").read_text())
    assert acc["tag"] == "freeze-transfer_cube-1" and acc["genesis"] == "genesis-1"
    assert git(root, "rev-parse", "freeze-transfer_cube-1^{commit}") == acc["commit"]
    assert list(acc["edit"]["changed_keys"]) == ["model.encoder.layers.0.self_attn.out_proj.weight"]
    assert acc["edit"]["weights_changed"] == 32
    assert acc["validation"]["passes_validation_rule"] is True


def test_seeds_were_drawn_after_the_commitment_and_only_then_used(played):
    root, _, _ = played
    td = root / "experiments" / "transfer_cube" / "test"
    com, draw = json.loads((td / "commitment.json").read_text()), json.loads((td / "draw.json").read_text())
    assert com["body"]["freeze_commit"] == draw["freeze_commit"]
    assert draw["beacon"]["round"] == com["target_round"]
    assert 1_000_000_000 <= draw["block"]["start"] < 2_000_000_000 and draw["block"]["count"] == 1000
    # the order of events is in the git history: commitment, then draw, then results
    log = git(root, "log", "--format=%s", "--reverse").splitlines()
    i_com = log.index("referee: commitment published for freeze-transfer_cube-1")
    i_draw = log.index("referee: test seeds drawn for freeze-transfer_cube-1")
    i_ver = next(i for i, m in enumerate(log) if m.startswith("referee: verdict for freeze-transfer_cube-1"))
    assert i_com < i_draw < i_ver
    # the two tasks got different seed blocks
    other = json.loads((root / "experiments" / "insertion" / "test" / "draw.json").read_text())
    assert other["block"]["start"] != draw["block"]["start"]
    # no test seed ever appears in the agent's ledger
    led = subprocess.run([PY, "-m", "referee.recompute", "ledger"], cwd=str(root), capture_output=True, text=True, check=True)
    assert json.loads(led.stdout)["test_range_rollouts"] == 0


def test_verdicts(played):
    root, _, _ = played
    v = json.loads((root / "experiments" / "transfer_cube" / "test" / "verdict.json").read_text())
    assert v["ending"] == "strong_win", v["reasons"]
    assert v["dry_run"] is True
    assert v["h1"]["n"] == 1000 and v["h1"]["significant"] and v["h1"]["share_removed"] > 0.25
    assert v["h2"]["n_random"] == 40 and v["h2"]["pass"]
    assert set(v["h3"]["baseline_gains_points"]) == {"constant_offset", "deviation_gain", "shorter_chunk", "simplest_fix", "temporal_ensembling"}
    assert v["arms"]["original_ens"]["n"] == 200 and v["arms"]["random_00"]["n"] == 1000
    assert len(v["arms"]) == 53
    assert v["stock_crosscheck"]["ok"] and v["stock_crosscheck"]["arms"]["agent_edit"]["agreement"] == 1.0
    md = (root / "experiments" / "transfer_cube" / "test" / "VERDICT.md").read_text()
    assert "**Ending. Strong win**" in md and "DRY RUN" in md
    assert md.split("<!-- referee-verdict-begin -->")[1].split("<!-- referee-verdict-end -->")[0] in (root / "experiments" / "transfer_cube" / "report" / "report.md").read_text()

    w = json.loads((root / "experiments" / "insertion" / "test" / "verdict.json").read_text())
    assert w["ending"] == "partial", w["reasons"]  # a real but small gain that the simple baselines beat
    assert w["h1"]["significant"] and not w["h1"]["large"] and w["h3"]["matching_baselines"]


def test_frozen_task_ignores_later_changes_to_the_harness(played):
    root, env, _ = played
    (root / "awe" / "report.py").write_text("print('new plotting code')\n")
    st = status(root, env)
    assert st["state"] == "DONE"
    (root / "awe" / "report.py").unlink()


# ------------------------------------------------------------------ one rule broken at a time


@pytest.fixture
def at_stage(repo, env):
    def go(stage: int, task: str = "transfer_cube"):
        drive(repo, env, stop_at=lambda st: st["state"] == "AGENT_TURN" and st.get("task") == task and st.get("stage") == stage)
        return repo
    return go


def _commit(root, msg="agent work"):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", msg)


def test_gate0_fails_outside_the_reproduction_window(repo, env):
    drive(repo, env, max_steps=1)  # the scripted agent does stage 0
    f = repo / "experiments" / "transfer_cube" / "stage0" / "stock_eval" / "eval_info.json"
    info = json.loads(f.read_text())
    succ = info["per_task"][0]["metrics"]["successes"]  # LeRobot 0.7.0 keeps plain lists
    succ[:60] = [False] * 60
    f.write_text(json.dumps(info))
    st = status(repo, env)
    miss = " ".join(m for t in st["open_tasks"] for m in t["missing"])
    assert st["stage"] == 0 and "s0.stock.reproduces" in miss and "s0.harness.agreement" in miss


def test_gate0_replay_catches_a_fabricated_table(repo, env):
    drive(repo, env, max_steps=1)
    f = repo / "experiments" / "transfer_cube" / "stage0" / "harness_repro.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    for r in rows:
        r["traj_hash"] = "0" * 24  # outcomes kept, trajectories invented
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _commit(repo)
    assert status(repo, env)["action"] == "confirm_g0"
    drive(repo, env, max_steps=1)
    st = status(repo, env)
    assert st["state"] == "AGENT_TURN" and st["stage"] == 0 and st["open_tasks"][0]["id"] == "g0.deep"
    assert "audit.reproduction" in st["open_tasks"][0]["missing"][0]


def test_gate0_is_rechecked_when_the_harness_changes(at_stage, env):
    root = at_stage(2)
    assert status(root, env)["stage"] == 2
    (root / "awe" / "atlas.py").write_text("# new harness code\n")
    _commit(root)
    st = status(root, env)
    assert (st["state"], st["action"]) == ("JUDGE_TURN", "confirm_g0")
    drive(root, env, max_steps=1)
    assert status(root, env)["stage"] == 2


def test_gate1_rejects_a_wrong_primary_metric(at_stage, env):
    root = at_stage(2)
    # the agent may not change its mind after Gate 1, and the recomputed rule still disagrees
    d = root / "experiments" / "transfer_cube" / "stage1" / "decisions.json"
    dec = json.loads(d.read_text())
    dec["primary_metric"] = "strict"
    d.write_text(json.dumps(dec))
    st = status(root, env)
    assert st["stage"] == 1 and "s1.decisions.primary_metric" in " ".join(st["open_tasks"][0]["missing"])


def test_task_is_skipped_when_it_rarely_fails(repo, env, monkeypatch):
    drive(repo, env, stop_at=lambda st: st.get("stage") == 1)
    e = repo / "experiments" / "transfer_cube" / "stage1"
    import sys

    sys.path.insert(0, str(repo))
    from sim_agent import stage1  # noqa: F401

    drive(repo, env, max_steps=1)  # stage 1 files
    for name in ("design_default", "design_chunk50", "design_chunk25"):
        rows = [json.loads(x) for x in (e / f"{name}.jsonl").read_text().splitlines()]
        for r in rows[30:]:
            r["benchmark_success"] = r["strict_success"] = True
        (e / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    st = status(repo, env)
    assert "s1.decisions.proceed" in " ".join(m for t in st["open_tasks"] for m in t["missing"])
    dec = json.loads((e / "decisions.json").read_text())
    dec["proceed"] = False
    (e / "decisions.json").write_text(json.dumps(dec))
    _commit(repo)
    trail = drive(repo, env, stop_at=lambda st: st.get("task") == "insertion")
    assert trail[-1][1] == "insertion" and trail[-1][2] == 0
    assert "transfer_cube skipped" in " ".join(status(repo, env)["notes"])


def _freeze_ready(at_stage, env):
    root = at_stage(3)
    drive(root, env, max_steps=1)  # stages 3 and 4, then the request
    return root


def _open_missing(root, env) -> str:
    st = status(root, env)
    return " ".join(m for t in st.get("open_tasks", []) for m in t["missing"])


def test_ledger_rules(at_stage, env):
    root = _freeze_ready(at_stage, env)
    assert status(root, env)["state"] == "REVIEW_TURN"
    led = root / "ledger" / "extra.jsonl"

    led.write_text(json.dumps({"task": "transfer_cube", "seed": 40000, "arm": "x", "budget": "agent"}) + "\n")
    assert "ledger.legal_seeds" in _open_missing(root, env)

    led.write_text(json.dumps({"task": "transfer_cube", "seed": 1_500_000_000, "arm": "x", "budget": "agent"}) + "\n")
    assert "ledger.no_test_seeds" in _open_missing(root, env)

    led.write_text("".join(json.dumps({"task": "transfer_cube", "seed": 30000 + i, "arm": f"c{k}", "candidate": f"extra{k}", "budget": "agent"}) + "\n" for k in range(8) for i in range(3)))
    miss = _open_missing(root, env)
    assert "validation_checks.agent: 11 of 10" in miss
    assert "no hypothesis line names extra0" in miss  # and each of them needed a hypothesis first

    with gzip.open(root / "ledger" / "big.jsonl.gz", "wt") as f:
        for i in range(60000):
            f.write(json.dumps({"task": "transfer_cube", "seed": 20000 + i % 1000, "arm": "y", "budget": "agent"}) + "\n")
    led.write_text("")
    assert "budget.agent" in _open_missing(root, env)
    (root / "ledger" / "big.jsonl.gz").unlink()

    led.write_text(json.dumps({"task": "transfer_cube", "seed": None, "pose": [0.1, 0.5], "arm": "grid", "budget": "agent"}) + "\n")
    assert status(root, env)["state"] == "REVIEW_TURN"  # explicit poses are fine


def test_freeze_integrity_rules(at_stage, env):
    root = _freeze_ready(at_stage, env)

    tune = root / "experiments" / "transfer_cube" / "scripts" / "tune.py"
    tune.parent.mkdir()
    tune.write_text("import torch\nloss.backward()  # tune the head\n")
    miss = _open_missing(root, env)
    assert "s4.integrity.no_gradients" in miss and "experiments/transfer_cube/scripts/tune.py:2" in miss
    tune.write_text("import torch\nwith torch.no_grad():\n    pass  # .backward( is only mentioned in this comment\n")
    assert status(root, env)["state"] == "REVIEW_TURN"
    tune.unlink()

    p = root / "PROGRESS.md"
    keep = p.read_text()
    p.write_text("\n".join(keep.splitlines()[:-2]) + "\n")
    _commit(root, "rewrite the notebook")
    assert "s4.integrity.PROGRESS.md" in _open_missing(root, env)
    p.write_text(keep)
    _commit(root, "put it back")
    assert "s4.integrity.PROGRESS.md" in _open_missing(root, env)  # history remembers

    d = root / "experiments" / "transfer_cube" / "stage1" / "decisions.json"
    dec = json.loads(d.read_text())
    dec["tracking_number"] = "meet_error"
    d.write_text(json.dumps(dec))
    assert "s4.integrity.decisions_frozen" in _open_missing(root, env)


def test_freeze_rules_on_the_edit_and_the_manifest(at_stage, env):
    root = _freeze_ready(at_stage, env)
    fz = root / "experiments" / "transfer_cube" / "freeze"
    edit = json.loads((fz / "edit.json").read_text())

    bad = dict(edit, ops=[{"op": "scale_head", "block": "model.encoder.layers.0.self_attn", "head": h, "scale": 2} for h in range(7)])
    (fz / "edit.json").write_text(json.dumps(bad))
    assert "s3.edit.max_heads: 7 heads, limit 6" in _open_missing(root, env)

    bad = dict(edit, ops=[{"op": "scale_head", "block": "model.encoder.layers.0.self_attn", "head": 0, "scale": 2.7}])
    (fz / "edit.json").write_text(json.dumps(bad))
    assert "scale 2.7 is not one of" in _open_missing(root, env)

    bad = dict(edit, ops=[{"op": "other", "description": "fit a rank-one update"}], free_numbers=12)
    (fz / "edit.json").write_text(json.dumps(bad))
    assert "s3.edit.free_numbers: 12 free numbers, limit 3" in _open_missing(root, env)
    (fz / "edit.json").write_text(json.dumps(edit))

    m = json.loads((fz / "test_manifest.json").read_text())
    short = dict(m, arms=[a for a in m["arms"] if a["id"] not in ("random_39", "base_deviation_gain", "dose_x4")])
    (fz / "test_manifest.json").write_text(json.dumps(short))
    miss = _open_missing(root, env)
    assert "39 random arms, need 40" in miss and "missing families deviation_gain" in miss and "3 dose arms, need 4" in miss

    rf = json.loads((fz / "random_family.json").read_text())
    (fz / "random_family.json").write_text(json.dumps(dict(rf, n_available=39)))
    assert "random arms" not in _open_missing(root, env)  # all 39 that exist is acceptable


def test_an_arm_that_cannot_run_blocks_the_freeze_before_any_seed_is_drawn(at_stage, env):
    root = _freeze_ready(at_stage, env)
    fz = root / "experiments" / "transfer_cube" / "freeze"
    m = json.loads((fz / "test_manifest.json").read_text())
    next(a for a in m["arms"] if a["id"] == "random_07")["sim"]["crash"] = True
    (fz / "test_manifest.json").write_text(json.dumps(m))
    _commit(root)
    drive(root, env, stop_at=lambda st: st["state"] == "AGENT_TURN")
    rej = (fz / "REJECTED-1.md").read_text()
    assert "freeze.every_arm_runs" in rej and "random_07" in rej
    assert not (root / "experiments" / "transfer_cube" / "test").exists()
    assert "REJECTED-1.md" in status(root, env)["note"]


def test_test_seeds_are_locked_without_the_referees_draw(repo, env):
    seeds = repo / "s.json"
    seeds.write_text(json.dumps({"seeds": [1_234_567_890]}))
    e = dict(env)
    e.pop("AWE_JUDGE_DRAW", None)
    r = subprocess.run([PY, "-m", "awe.run_arm", "--task", "transfer_cube", "--arm", "original", "--seeds-file", str(seeds), "--out", str(repo / "o.jsonl")], cwd=str(repo), env=e, capture_output=True, text=True)
    assert r.returncode != 0 and "test seed" in r.stderr


def test_guard_restores_protected_files(repo, env):
    (repo / "referee" / "config.json").write_text(json.dumps({"budgets": {"agent": 10**9}}))
    (repo / "run" / "extra.sh").write_text("echo hi\n")
    (repo / "SPEC.md").write_text("# a kinder spec\n")
    _commit(repo, "quietly change the rules")
    # the loop runs the referee from its own untouched copy, so a broken copy in the repo cannot stop it
    kit = __import__("conftest").KIT
    r = subprocess.run([PY, "-m", "referee.guard", "--repo", str(repo)], cwd=str(kit), env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "referee/config.json" in r.stdout and "SPEC.md" in r.stdout and "run/extra.sh" in r.stdout
    r = subprocess.run([PY, "-m", "referee.status", "--repo", str(repo), "--json"], cwd=str(kit), env=env, capture_output=True, text=True)
    assert any("protected files differ" in w for w in json.loads(r.stdout)["warnings"])
    r = subprocess.run([PY, "-m", "referee.guard", "--repo", str(repo), "--heal"], cwd=str(kit), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((repo / "referee" / "config.json").read_text())["budgets"]["agent"] == 100000
    assert not (repo / "run" / "extra.sh").exists()
    assert status(repo, env)["warnings"] == []
    assert git(repo, "log", "-1", "--format=%s") == "referee: restored protected files to genesis-1"


def test_waiting_file_stops_everything(repo, env):
    (repo / "WAITING_FOR_ANDREW.md").write_text("# Waiting\n\nThe checkpoint will not convert.\n")
    st = status(repo, env)
    assert st["state"] == "WAITING_FOR_ANDREW" and "checkpoint will not convert" in st["note"]


def test_real_witness_is_required_outside_a_dry_run(at_stage, env):
    root = _freeze_ready(at_stage, env)
    drive(root, env, stop_at=lambda st: st.get("action") == "test")
    e = dict(env)
    e.pop("AWE_DRY_RUN")
    r = subprocess.run([PY, "-m", "referee.judge", "step"], cwd=str(root), env=e, capture_output=True, text=True)
    assert r.returncode != 0 and "only allowed in a dry run" in (r.stdout + r.stderr)
    assert not (root / "experiments" / "transfer_cube" / "test" / "draw.json").exists()


def test_a_usage_limit_never_skips_a_review(repo, env):
    env.update({"AWE_SCOPE": "main"})
    drive(repo, env, stop_at=lambda st: st["state"] == "REVIEW_TURN")
    step = [PY, "-m", "referee.review", "step"]

    # the reviewer is turned away by a usage limit. Nothing is counted and nothing is written
    for _ in range(5):
        r = subprocess.run(step, cwd=str(repo), env={**env, "STUB_REVIEW_LIMIT": "99"}, capture_output=True, text=True)
        assert r.returncode == 75, r.stdout + r.stderr
    assert "weekly limit" in (repo / ".awe" / "review-last-error.txt").read_text()
    assert not list((repo / "reviews").glob("transfer_cube-g1-r*.json")) and not list((repo / ".awe").glob("review-fail-*"))
    assert status(repo, env)["state"] == "REVIEW_TURN"
    # so does a claude binary that is missing or prints nothing
    r = subprocess.run(step, cwd=str(repo), env={**env, "AWE_CLAUDE_BIN": "/nonexistent/claude"}, capture_output=True, text=True)
    assert r.returncode == 75 and "FileNotFoundError" in r.stdout
    r = subprocess.run(step, cwd=str(repo), env={**env, "AWE_CLAUDE_BIN": "true"}, capture_output=True, text=True)
    assert r.returncode == 75 and "printed no result" in r.stdout

    # once it can run again the review happens as usual
    trail = drive(repo, env)
    assert trail[-1][0] == "DONE", trail[-5:]
    v = json.loads((repo / "experiments" / "transfer_cube" / "test" / "verdict.json").read_text())
    assert [(x["gate"], x["rounds"], x["not_reviewed"], x["open_issues"]) for x in v["reviews"]] == [("g1", 1, False, 0), ("g2.1", 1, False, 0)]
    md = (repo / "experiments" / "transfer_cube" / "test" / "VERDICT.md").read_text()
    assert "Gate 1 passed review in round 1. Freeze request 1 passed review in round 1" in md
    assert not any("review" in x for x in v["reasons"])


@pytest.mark.parametrize("junk", ["turns", "shape"])
def test_a_reviewer_that_never_answers_is_recorded_in_the_verdict(repo, env, junk):
    env.update({"AWE_SCOPE": "main"})
    drive(repo, env, stop_at=lambda st: st["state"] == "REVIEW_TURN")
    step = [PY, "-m", "referee.review", "step"]
    codes = [subprocess.run(step, cwd=str(repo), env={**env, "STUB_REVIEW_JUNK": junk}, capture_output=True, text=True).returncode for _ in range(3)]
    assert codes == [1, 1, 0]
    r1 = json.loads((repo / "reviews" / "transfer_cube-g1-r1.json").read_text())
    assert r1["verdict"] == "pass" and r1["reviewer_failed"] is True

    trail = drive(repo, env)
    assert trail[-1][0] == "DONE", trail[-5:]
    assert json.loads((repo / "reviews" / "transfer_cube-g1-passed.json").read_text())["review_not_independent"] is True
    v = json.loads((repo / "experiments" / "transfer_cube" / "test" / "verdict.json").read_text())
    assert v["reviews"][0] == {"gate": "g1", "rounds": 1, "last_verdict": "pass", "open_issues": 0, "not_reviewed": True}
    assert any(x.startswith("WARNING: Gate 1 was not independently reviewed") for x in v["reasons"])
    assert "Gate 1 was not independently reviewed" in (repo / "experiments" / "transfer_cube" / "test" / "VERDICT.md").read_text()


def test_the_reviewer_runs_on_opus_at_xhigh_unless_told_otherwise(repo, env):
    env.update({"AWE_SCOPE": "main"})
    drive(repo, env, stop_at=lambda st: st["state"] == "REVIEW_TURN")
    step = [PY, "-m", "referee.review", "step"]
    log = Path(env["STUB_STATE"]) / "stub-argv.jsonl"

    # a level Claude Code would ignore is refused before anything runs, and nothing is counted
    r = subprocess.run(step, cwd=str(repo), env={**env, "AWE_REVIEW_EFFORT": "ultra"}, capture_output=True, text=True)
    assert r.returncode == 2 and "AWE_REVIEW_EFFORT is 'ultra'" in r.stdout
    assert not log.exists() and not list((repo / "reviews").glob("transfer_cube-g1-r*.json"))

    # by default the reviewer is named Opus at xhigh, and a stray level in the environment never reaches it
    r = subprocess.run(step, cwd=str(repo), env={**env, "CLAUDE_CODE_EFFORT_LEVEL": "low"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    call = json.loads(log.read_text().splitlines()[-1])
    a = call["argv"]
    assert a[a.index("--effort") + 1] == "xhigh" and a[a.index("--model") + 1] == "opus" and call["effort_env"] is None
    assert json.loads((repo / "reviews" / "transfer_cube-g1-r1.json").read_text())["effort"] == "xhigh"
