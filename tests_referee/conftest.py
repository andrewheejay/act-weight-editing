"""Test fixtures. Each test gets a throwaway git repo that holds the kit plus the fake harness."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
PY = sys.executable
sys.path.insert(0, str(KIT))
os.chmod(TESTS / "stub_claude.py", 0o755)  # archives do not always keep the executable bit


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.com", *args], check=True, capture_output=True, text=True)
    return r.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "act-weight-editing"
    root.mkdir()
    for name in ("referee", "run", ".claude"):
        shutil.copytree(KIT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("PLAN.json", "CLAUDE.md", "PROGRESS.md", "DEVIATIONS.md", ".gitignore", "README.md", "SPEC.md"):
        if (KIT / name).exists():
            shutil.copy(KIT / name, root / name)
        else:
            (root / name).write_text(f"# {name}\n\nplaceholder for tests\n")
    shutil.copytree(TESTS / "fake_awe", root / "awe", ignore=shutil.ignore_patterns("__pycache__"))
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "genesis")
    git(root, "tag", "genesis-1")
    return root


@pytest.fixture
def env(tmp_path: Path) -> dict:
    beacon_file = tmp_path / "beacon.json"
    beacon_file.write_text(json.dumps({"signature": "95a9f9f5b231b7714de1553105d8ffdf3dcda24cfdb1e689319bccf79a9c8ce430a91b811fbfaf763900bc998b5d686a"}))
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("AWE_", "STUB_")):
            e.pop(k)
    e.update({
        "AWE_DRY_RUN": "1",
        "AWE_WITNESS": "local",
        "AWE_BEACON_FILE": str(beacon_file),
        "AWE_NO_PUSH": "1",
        "AWE_WAYBACK": "0",
        "AWE_PYTHON": PY,
        "AWE_CLAUDE_BIN": str(TESTS / "stub_claude.py"),
        "AWE_STATE_DIR": str(tmp_path / "state"),
        "STUB_STATE": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return e


def status(root: Path, env: dict) -> dict:
    r = subprocess.run([PY, "-m", "referee.status", "--json"], cwd=str(root), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def drive(root: Path, env: dict, max_steps: int = 120, stop_at=None) -> list[tuple]:
    """Play the loop's part by hand. One agent, reviewer or referee step per pass."""
    trail = []
    for _ in range(max_steps):
        st = status(root, env)
        trail.append((st["state"], st.get("task"), st.get("stage"), st.get("action"), st.get("gate"), st.get("round")))
        if st["state"] in ("DONE", "WAITING_FOR_ANDREW", "SETUP_INCOMPLETE"):
            break
        if stop_at and stop_at(st):
            break
        if st["state"] == "AGENT_TURN":
            cmd = [PY, str(TESTS / "sim_agent.py"), str(root)]
        elif st["state"] == "REVIEW_TURN":
            cmd = [PY, "-m", "referee.review", "step"]
        else:
            cmd = [PY, "-m", "referee.judge", "step"]
        r = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True)
        assert r.returncode == 0, f"{cmd} failed\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return trail
