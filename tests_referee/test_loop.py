"""Runs the real loop script against the scripted agent, from an empty repo to the end."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from pathlib import Path

from conftest import PY, git, status


def run_loop(repo: Path, env: dict, timeout: int = 420) -> subprocess.CompletedProcess:
    # no model, effort or turn flags unless a test sets them
    e = {"AWE_MODEL": "", "AWE_EFFORT": "", "AWE_MAX_TURNS": ""}
    e.update(env)
    e.update({"AWE_LIMIT_SLEEP": "1", "AWE_ERROR_SLEEP": "1", "AWE_JOB_POLL": "1", "HOME": str(Path(env["AWE_STATE_DIR"]).parent / "home")})
    Path(e["HOME"]).mkdir(exist_ok=True)
    return subprocess.run(["bash", str(repo / "run" / "loop.sh")], cwd=str(repo), env=e, capture_output=True, text=True, timeout=timeout)


def test_loop_runs_the_whole_protocol(repo, env):
    env.update({"AWE_SCOPE": "main", "STUB_FAIL_SESSIONS": "2", "STUB_REVISE_FIRST": "g1", "STUB_REVIEW_LIMIT": "2"})
    # the shipped defaults for the model and effort, and a stray effort level left on the box
    env.update({"AWE_MODEL": "opus", "AWE_EFFORT": "high", "CLAUDE_CODE_EFFORT_LEVEL": "low"})
    # a real remote, so every push in the loop and in the referee really happens
    env.pop("AWE_NO_PUSH")
    origin = repo.parent / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "origin", "main", "genesis-1")
    # someone quietly loosens the rules after genesis, and a long job is already running
    (repo / "SPEC.md").write_text("# a kinder spec\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "loosen the rules")
    subprocess.run([PY, str(repo / "run" / "jobs.py"), "start", "nap", "--", "sleep", "3"], cwd=str(repo), env=env, check=True, capture_output=True)

    r = run_loop(repo, env)
    log = (repo / "logs" / "loop.log").read_text()
    assert r.returncode == 0, log[-3000:] + r.stderr[-2000:]
    assert status(repo, env)["state"] == "DONE"

    # it healed the tampering before doing anything else, and waited for the job
    assert "placeholder" in (repo / "SPEC.md").read_text() or "kinder" not in (repo / "SPEC.md").read_text()
    assert "referee: restored protected files to genesis-1" in git(repo, "log", "--format=%s")
    assert "Waiting for jobs: nap" in log
    # it rode out two usage-limit failures as the agent and two more as the reviewer, and skipped no review
    assert log.count("Usage limit or overload") == 4
    assert all("reviewer_failed" not in f.read_text() for f in (repo / "reviews").glob("*-r*.json"))
    assert len(list((repo / "reviews").glob("transfer_cube-g1-r*.json"))) == 2
    # everything it did is on the remote, on the run's branch
    assert git(repo, "ls-remote", "origin", "refs/heads/main").split()[0] == git(repo, "rev-parse", "HEAD")
    assert git(repo, "ls-remote", "origin", "refs/heads/HEAD") == ""
    assert "Run finished" in log

    rows = (repo / "logs" / "sessions.tsv").read_text().splitlines()
    assert rows[0].split("\t") == ["time", "session", "model", "effort", "stage", "turns", "cost_estimate_usd", "result", "exit"] and len(rows) >= 11
    assert rows[1].split("\t")[7] == "error_during_execution" and rows[3].split("\t")[2] == "stub-model"
    gz = sorted((repo / "logs" / "agent").glob("*.jsonl.gz"))
    assert len(gz) == len(rows) - 1
    assert json.loads(gzip.open(gz[-1], "rt").readline())["subtype"] == "init"

    # effort follows the stage. xhigh for the diagnosis (1) and the edit design (3), high for the rest
    effort_by_stage: dict[str, set] = {}
    for x in rows[1:]:
        c = x.split("\t")
        effort_by_stage.setdefault(c[4], set()).add(c[3])
    assert effort_by_stage.get("1") == {"xhigh"} and effort_by_stage.get("3") == {"xhigh"}, effort_by_stage
    assert all(v == {"high"} for k, v in effort_by_stage.items() if k not in ("1", "3")), effort_by_stage
    # and those are the flags Claude Code really received, in order. Every review ran on Opus at xhigh
    calls = [json.loads(x) for x in (Path(env["STUB_STATE"]) / "stub-argv.jsonl").read_text().splitlines()]
    agent = [c["argv"] for c in calls if "-p" in c["argv"] and "--json-schema" not in c["argv"]]
    review = [c["argv"] for c in calls if "--json-schema" in c["argv"]]
    assert [a[a.index("--effort") + 1] for a in agent] == [x.split("\t")[3] for x in rows[1:]]
    assert all(a[a.index("--model") + 1] == "opus" for a in agent)
    assert len(review) >= 4 and all(a[a.index("--effort") + 1] == "xhigh" and a[a.index("--model") + 1] == "opus" for a in review)
    assert all(json.loads(f.read_text())["effort"] == "xhigh" for f in (repo / "reviews").glob("*-r*.json"))
    # the stray level on the box never reached a session, and the log says why
    assert all(c["effort_env"] is None for c in calls)
    assert "CLAUDE_CODE_EFFORT_LEVEL=low was set on this box. The loop cleared it" in log
    assert "Effort is high for most sessions, xhigh for stages 1 and 3, and xhigh for reviews." in log
    assert "starting (stage 1, model opus, effort xhigh)" in log and "starting (stage 0, model opus, effort high)" in log

    note = (repo / "logs" / "notify.log").read_text()
    for title in ("Run started", "Paused by a usage limit", "Run resumed", "Gate 0 passed", "Gate 1 passed", "Freeze accepted", "Commitment published", "Test seeds drawn", "Main arms done", "Verdict (transfer_cube): Strong win", "Run finished"):
        assert title in note, title

    # the loop ran from a read-only copy taken at the genesis tag, with absolute deny rules
    state = Path(env["AWE_STATE_DIR"])
    pristine = next(state.glob("genesis-1-*"))
    assert (pristine / "referee" / "judge.py").exists() and not os.access(pristine / "referee" / "judge.py", os.W_OK) or os.geteuid() == 0
    deny = json.loads((state / "claude-settings.json").read_text())["permissions"]["deny"]
    assert f"Edit(/{repo}/referee/**)" in deny and f"Edit(/{state}/**)" in deny and "Bash(git push *)" in deny


def test_loop_stops_when_the_agent_is_stuck(repo, env):
    env.update({"STUB_IDLE": "1"})
    r = run_loop(repo, env)
    assert r.returncode == 3, (repo / "logs" / "loop.log").read_text()[-2000:]
    w = (repo / "WAITING_FOR_ANDREW.md").read_text()
    assert "The last 4 sessions changed nothing" in w and "stage 0" in w
    assert "Run is waiting for you" in (repo / "logs" / "notify.log").read_text()
    assert len((repo / "logs" / "sessions.tsv").read_text().splitlines()) == 5
    # starting again without fixing anything stops at once, for the same reason
    r = run_loop(repo, env)
    assert r.returncode == 3 and "State WAITING_FOR_ANDREW" in (repo / "logs" / "loop.log").read_text()


def test_loop_stops_when_sessions_keep_crashing(repo, env):
    env.update({"STUB_CRASH": "1", "AWE_FAIL_SESSIONS": "3"})
    r = run_loop(repo, env)
    assert r.returncode == 3, (repo / "logs" / "loop.log").read_text()[-2000:]
    w = (repo / "WAITING_FOR_ANDREW.md").read_text()
    assert "3 Claude Code sessions in a row failed to run" in w and "TypeError" in w
    note = (repo / "logs" / "notify.log").read_text()
    assert "Sessions are failing" in note and "Run is waiting for you" in note and "usage limit" not in note.lower().replace("it was not a usage limit", "")
    rows = (repo / "logs" / "sessions.tsv").read_text().splitlines()
    assert len(rows) == 4 and all(x.split("\t")[7:] == ["none", "1"] for x in rows[1:])


def test_loop_stops_when_the_agent_leaves_the_branch(repo, env):
    env.update({"STUB_LEAVES_BRANCH": "1"})
    r = run_loop(repo, env)
    assert r.returncode == 3, (repo / "logs" / "loop.log").read_text()[-2000:]
    assert "The repo is on scratch, and the run lives on the branch main" in (repo / "WAITING_FOR_ANDREW.md").read_text()
    assert (Path(env["AWE_STATE_DIR"]) / "branch").read_text().strip() == "main"
    # deleting the note is not enough. The loop will not adopt another branch by accident
    (repo / "WAITING_FOR_ANDREW.md").unlink()
    r = run_loop(repo, env)
    assert r.returncode == 2 and "This run lives on the branch main, and the repo is on scratch" in r.stderr
    git(repo, "checkout", "-q", "--detach")
    r = run_loop(repo, env)
    assert r.returncode == 2 and "not on a branch" in r.stderr


def test_loop_refuses_an_effort_level_claude_code_would_ignore(repo, env):
    for name, bad in (("AWE_EFFORT_DESIGN", "xhgh"), ("AWE_REVIEW_EFFORT", "ultra"), ("AWE_EFFORT", "High")):
        r = run_loop(repo, {**env, name: bad})
        assert r.returncode == 2 and f"{name} is '{bad}'. Use low, medium, high, xhigh or max" in r.stderr, r.stderr
    # nothing started, so nothing was spent
    assert not (repo / "logs" / "sessions.tsv").exists() and not (Path(env["STUB_STATE"]) / "stub-argv.jsonl").exists()


def test_loop_refuses_to_start_before_setup(tmp_path, env):
    root = tmp_path / "bare"
    (root / "run").mkdir(parents=True)
    kit = __import__("conftest").KIT
    (root / "run" / "loop.sh").write_text((kit / "run" / "loop.sh").read_text())
    git(root, "init", "-q", "-b", "main")
    r = subprocess.run(["bash", str(root / "run" / "loop.sh")], cwd=str(root), env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "No genesis tag yet" in r.stderr
