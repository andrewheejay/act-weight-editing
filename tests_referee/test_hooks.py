"""Tests for the three Claude Code hooks, fed the same JSON that Claude Code sends them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import git

HOOKS = Path(__file__).resolve().parent.parent / ".claude" / "hooks"


def hook(name: str, payload: dict, root: Path, agent: bool = True) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWE_")}
    env["CLAUDE_PROJECT_DIR"] = str(root)
    if agent:
        env["AWE_AGENT"] = "1"
    r = subprocess.run([sys.executable, str(HOOKS / name)], input=json.dumps(payload), capture_output=True, text=True, env=env, cwd=str(root))
    return r.returncode, r.stdout.strip()


def decision(tool: str, inp: dict, root: Path, agent: bool = True, cwd: Path | None = None) -> str:
    rc, out = hook("protect.py", {"session_id": "s", "cwd": str(cwd or root), "hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": inp}, root, agent)
    assert rc == 0
    if not out:
        return "allow"
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"]


DENIED_EDITS = [
    "referee/stats.py",
    "referee/config.json",
    "SPEC.md",
    "PLAN.json",
    "CLAUDE.md",
    "run/loop.sh",
    ".claude/settings.json",
    ".claude/hooks/protect.py",
    "experiments/transfer_cube/test/verdict.json",
    "experiments/insertion/test/arms/original.jsonl",
    "experiments/transfer_cube/freeze/ACCEPTED.json",
    "reviews/transfer_cube-g1-r1.json",
    "reviews/transfer_cube-g1-passed.json",
    ".git/refs/tags/genesis-1",
    "awe/../referee/gates.py",
    "logs/sessions.tsv",
    "logs/agent/x.jsonl",
    "PAUSE",
]
ALLOWED_EDITS = [
    "awe/rollout.py",
    "tests/test_harness.py",
    "experiments/transfer_cube/diagnosis.md",
    "experiments/transfer_cube/freeze/edit.json",
    "experiments/transfer_cube/freeze/REQUEST.json",
    "experiments/transfer_cube/report/report.md",
    "reviews/transfer_cube-g1-r1-response.md",
    "PROGRESS.md",
    "DEVIATIONS.md",
    "WAITING_FOR_ANDREW.md",
    "ledger/stage0.jsonl",
    "publish/post.md",
    ".gitignore",
]
DENIED_BASH = [
    "git push origin main",
    "git push",
    "cd /tmp && git -C {root} push origin HEAD",
    "git -c user.name=x push",
    "git tag freeze-transfer_cube-1",
    "git tag -d genesis-1",
    "git tag -f genesis-1 HEAD",
    "gh release create v1 --notes hi",
    "gh api repos/x/y",
    "sed -i 's/0.05/0.5/' referee/config.json",
    "sed -i.bak s/a/b/ SPEC.md",
    "echo '{{}}' > referee/config.json",
    "python make.py >> run/loop.sh",
    "cat patch.txt | tee referee/gates.py",
    "rm -rf experiments/transfer_cube/test",
    "rm reviews/transfer_cube-g1-r1.json",
    "mv referee referee_old",
    "cp /tmp/gates.py referee/gates.py",
    "chmod +w referee/config.json",
    "truncate -s 0 PLAN.json",
    "git checkout HEAD~3 -- referee/",
    "git restore --source=HEAD~1 SPEC.md",
    "git rm -r run",
    "git commit --amend -m rewrite",
    "git rebase -i HEAD~4",
    "git clean -fdx",
    "git remote set-url origin https://example.com/x.git",
    "claude -p 'keep going'",
    "claude --bg 'another agent'",
    "sudo shutdown -h now",
    "python a.py\ngit push",
    "true; rm -f {root}/referee/stats.py",
    "FOO=1 sudo rm CLAUDE.md",
    "dd if=/dev/zero of=referee/config.json",
    "echo x > ~/.awe/act-weight-editing/config.env",
    "rm -rf logs",
    "rm -rf experiments",
    "rm -rf experiments/transfer_cube",
    "mv experiments experiments_old",
    "rm -rf .",
    "git rm -r experiments/insertion",
    "touch PAUSE",
    "truncate -s 0 logs/loop.log",
    "git switch -c scratch",
    "git checkout -b scratch",
    "git checkout genesis-1",
    "git checkout -q HEAD~1",
    "git -C {root} checkout --detach",
    "git checkout HEAD~1 awe/run_arm.py",
    "git reset --hard HEAD~1",
    "git reset --soft HEAD^",
    "git reset --hard origin/main",
    "git branch -D main",
    "git branch -m main trunk",
    "git symbolic-ref HEAD refs/heads/other",
]
ALLOWED_BASH = [
    "python -m referee.status",
    "python -m referee.status > /tmp/status.txt",
    "python -m referee.recompute ledger | head -40",
    "cat referee/RULES.md",
    "grep -rn seed referee/ | wc -l",
    "cp referee/CONTRACT.md /tmp/contract.md",
    "git add -A && git commit -m 'stage 0: reproduction'",
    "git commit -m 'fix; rm the old cache and push on'",
    "git status --porcelain",
    "git log --oneline -5",
    "git tag -l",
    "git tag --list 'genesis-*'",
    "git describe --tags",
    "git stash push -m wip",
    "git diff genesis-1 -- awe/",
    "git checkout -- awe/rollout.py",
    "python run/jobs.py start sweep -- python -m awe.sweep --task transfer_cube",
    "python -m awe.sweep --task transfer_cube > logs/sweep.txt 2>&1",
    "python -m pytest tests -q 2>&1 | tail -5",
    "rm -rf experiments/transfer_cube/cache",
    "rm experiments/transfer_cube/stage0/old.jsonl",
    "mkdir -p experiments/transfer_cube/stage0 && touch experiments/transfer_cube/stage0/x",
    "sed -n 1,40p referee/gates.py",
    "sed -i s/a/b/ awe/rollout.py",
    "uv pip install matplotlib",
    "claude --version",
    "lerobot-eval --policy.path=checkpoints/transfer_cube --seed=1000 --output_dir=experiments/transfer_cube/stage0/stock_eval",
    "echo done > /dev/null",
    "tail -n 50 logs/loop.log",
    "rm -rf experiments/transfer_cube/stage2/tmp",
    "mv experiments/transfer_cube/stage1/old.png experiments/transfer_cube/stage1/failure_map.png",
    "rm -rf awe/__pycache__",
    "python run/jobs.py tail sweep -n 20",
    "git checkout HEAD~1 -- awe/run_arm.py",
    "git checkout awe/run_arm.py",
    "git checkout .",
    "git reset",
    "git reset HEAD awe/run_arm.py",
    "git reset --hard",
    "git reset --hard HEAD",
    "git reset -q -- awe/run_arm.py",
    "git branch",
    "git branch scratch",
    "git revert --no-edit HEAD",
    "git show HEAD~1:awe/run_arm.py",
    "git symbolic-ref --short HEAD",
]


@pytest.mark.parametrize("path", DENIED_EDITS)
def test_protect_blocks_edits_to_protected_files(repo, path):
    assert decision("Edit", {"file_path": str(repo / path)}, repo) == "deny"
    assert decision("Write", {"file_path": path}, repo) == "deny"


@pytest.mark.parametrize("path", ALLOWED_EDITS)
def test_protect_allows_the_agents_own_files(repo, path):
    assert decision("Write", {"file_path": str(repo / path)}, repo) == "allow"


@pytest.mark.parametrize("cmd", DENIED_BASH)
def test_protect_blocks_bash(repo, cmd):
    assert decision("Bash", {"command": cmd.format(root=repo)}, repo) == "deny", cmd


@pytest.mark.parametrize("cmd", ALLOWED_BASH)
def test_protect_allows_ordinary_bash(repo, cmd):
    assert decision("Bash", {"command": cmd}, repo) == "allow", cmd


def test_protect_resolves_paths_from_the_shells_directory(repo):
    assert decision("Bash", {"command": "sed -i s/a/b/ config.json"}, repo, cwd=repo / "referee") == "deny"
    assert decision("Bash", {"command": "echo x > ../SPEC.md"}, repo, cwd=repo / "awe") == "deny"
    assert decision("Bash", {"command": "echo x > notes.txt"}, repo, cwd=repo / "awe") == "allow"


def test_hooks_leave_a_persons_own_session_alone(repo):
    assert decision("Edit", {"file_path": str(repo / "referee/stats.py")}, repo, agent=False) == "allow"
    assert decision("Bash", {"command": "git push origin main"}, repo, agent=False) == "allow"
    assert hook("stop_check.py", {"session_id": "x"}, repo, agent=False) == (0, "")
    assert hook("session_start.py", {"session_id": "x", "source": "startup"}, repo, agent=False) == (0, "")


def test_session_start_prints_status_and_notebook(repo):
    rc, out = hook("session_start.py", {"session_id": "abc", "source": "startup", "cwd": str(repo)}, repo)
    assert rc == 0
    assert "STATE AGENT_TURN" in out and "stage 0" in out
    assert "s0.env" in out and "Tail of PROGRESS.md" in out
    assert (repo / ".awe" / "sessions" / "abc.json").exists()
    rc, out = hook("session_start.py", {"session_id": "abc", "source": "compact", "cwd": str(repo)}, repo)
    assert "context was just compacted" in out


def test_stop_hook_wants_a_notebook_entry_and_a_commit(repo):
    hook("session_start.py", {"session_id": "s1", "source": "startup"}, repo)
    stop = lambda: hook("stop_check.py", {"session_id": "s1", "stop_hook_active": False}, repo)  # noqa: E731

    rc, out = stop()
    assert json.loads(out)["decision"] == "block" and "PROGRESS.md" in json.loads(out)["reason"]

    (repo / "awe" / "new.py").write_text("x = 1\n")
    with open(repo / "PROGRESS.md", "a") as f:
        f.write("\n## entry\n- did a thing\n")
    rc, out = stop()
    reason = json.loads(out)["reason"]
    assert "commit your changes" in reason and "awe/new.py" in reason and "append a dated entry" not in reason

    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "work")
    assert stop() == (0, "")


def test_stop_hook_lets_go_after_three_blocks(repo):
    hook("session_start.py", {"session_id": "s2", "source": "startup"}, repo)
    outs = [hook("stop_check.py", {"session_id": "s2"}, repo)[1] for _ in range(4)]
    assert all(json.loads(o)["decision"] == "block" for o in outs[:3])
    assert outs[3] == ""
