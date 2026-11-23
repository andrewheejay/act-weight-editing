"""Runs the real setup script against a local bare remote.

gh, claude, tmux and the virtual environment are stand-ins, so the test needs no network and no
accounts. Everything else is the real script, including the tagging, the pushes and the release
logic that makes the genesis tag public.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from conftest import git

FAKE_GH = r"""#!/usr/bin/env bash
# A stand-in for gh. A release is a file in $FAKE_GH_DIR.
set -u
d="$FAKE_GH_DIR"
mkdir -p "$d"
echo "$*" >> "$d/calls.log"
case "${1:-} ${2:-}" in
  "auth status") exit 0 ;;
  "repo view")
    if [[ -z "${3:-}" || "${3:-}" == --* ]]; then echo "${FAKE_GH_VISIBILITY:-PUBLIC}"; else echo "${FAKE_GH_WITNESS_VISIBILITY:-PUBLIC}"; fi
    exit 0 ;;
  "release view") [[ -f "$d/release-$3" ]]; exit $? ;;
  "release create")
    if [[ -f "$d/fail-next-create" ]]; then mv "$d/fail-next-create" "$d/failed-once"; echo "HTTP 502" >&2; exit 1; fi
    # --verify-tag means the tag has to be on the remote already
    git ls-remote --exit-code --tags origin "refs/tags/$3" >/dev/null || { echo "tag $3 is not on the remote" >&2; exit 1; }
    printf '%s\n' "$*" > "$d/release-$3"
    exit 0 ;;
esac
exit 0
"""

FAKE_CLAUDE = """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "2.1.330 (Claude Code)"; fi
exit 0
"""

FAKE_PYTHON = """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "Python 3.12.0 (stand-in)"; exit 0; fi
if [[ "$*" == *pytest* ]]; then echo "all passed (stand-in)"; fi
exit 0
"""


@pytest.fixture
def box(repo: Path, env: dict, tmp_path: Path):
    """A repo as it looks before setup has ever run: the kit, a remote, no genesis tag."""
    git(repo, "tag", "-d", "genesis-1")
    git(repo, "config", "user.name", "Test Person")
    git(repo, "config", "user.email", "test@example.com")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    git(repo, "remote", "add", "origin", str(origin))
    fake = tmp_path / "fakebin"
    fake.mkdir()
    for name, text in (("gh", FAKE_GH), ("claude", FAKE_CLAUDE), ("tmux", "#!/usr/bin/env bash\nexit 0\n")):
        (fake / name).write_text(text)
        (fake / name).chmod(0o755)
    venv = repo / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(FAKE_PYTHON)
    (venv / "python").chmod(0o755)
    (repo / ".venv" / ".awe-installed").write_text("")
    home = tmp_path / "home"
    home.mkdir()
    e = dict(env)
    e.update({"PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "HOME": str(home), "FAKE_GH_DIR": str(tmp_path / "gh")})
    return repo, e, tmp_path / "gh"


def setup(repo: Path, e: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(repo / "run" / "setup_box.sh"), *args], cwd=str(repo), env=e, capture_output=True, text=True, timeout=120)


def remote_refs(repo: Path) -> dict:
    return {line.split()[1]: line.split()[0] for line in git(repo, "ls-remote", "origin").splitlines() if line.strip()}


def test_setup_publishes_the_genesis_tag_once(box):
    repo, e, gh = box
    r = setup(repo, e, "--check")
    assert r.returncode == 0 and "All checks passed." in r.stdout and git(repo, "tag", "-l") == ""

    r = setup(repo, e)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "genesis-1 is tagged, pushed and published." in r.stdout
    refs = remote_refs(repo)
    assert refs["refs/heads/main"] == git(repo, "rev-parse", "HEAD") and "refs/tags/genesis-1" in refs
    note = (gh / "release-genesis-1").read_text()
    assert "--verify-tag" in note and git(repo, "rev-parse", "genesis-1^{commit}") in note
    assert (Path(e["AWE_STATE_DIR"]) / "config.env").read_text().startswith("# Settings for the run loop")
    assert git(repo, "status", "--porcelain") == ""  # everything, the lock file included, is in the genesis commit

    r = setup(repo, e)
    assert r.returncode == 0 and "genesis-1 already exists and is published. Nothing to do." in r.stdout
    assert (gh / "calls.log").read_text().count("release create") == 1 and git(repo, "tag", "-l") == "genesis-1"


def test_setup_finishes_a_publish_that_failed_half_way(box):
    repo, e, gh = box
    gh.mkdir()
    (gh / "fail-next-create").write_text("")
    r = setup(repo, e)
    assert r.returncode != 0 and git(repo, "tag", "-l") == "genesis-1" and not (gh / "release-genesis-1").exists()

    # running it again must not say that there is nothing to do
    r = setup(repo, e)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "genesis-1 exists on this box and was never published. Publishing it now." in r.stdout
    assert (gh / "release-genesis-1").exists() and "refs/tags/genesis-1" in remote_refs(repo)
    assert git(repo, "tag", "-l") == "genesis-1"


def test_regenesis_records_why_the_rules_changed(box):
    repo, e, gh = box
    assert setup(repo, e).returncode == 0
    with (repo / "referee" / "RULES.md").open("a") as f:
        f.write("\nThe Gate 0 window is one point wider.\n")
    r = setup(repo, e, "--regenesis")
    assert r.returncode == 1 and "Give a reason." in r.stdout and git(repo, "tag", "-l") == "genesis-1"

    r = setup(repo, e, "--regenesis", "the Gate 0 window was one point too tight")
    assert r.returncode == 0, r.stdout + r.stderr
    assert git(repo, "tag", "-l").split() == ["genesis-1", "genesis-2"]
    assert "the Gate 0 window was one point too tight" in (repo / "DEVIATIONS.md").read_text()
    assert "Rules changed by a person. Reason. the Gate 0 window was one point too tight" in (gh / "release-genesis-2").read_text()
    refs = remote_refs(repo)
    assert "refs/tags/genesis-2" in refs and refs["refs/heads/main"] == git(repo, "rev-parse", "HEAD")
    assert "one point wider" in git(repo, "show", "genesis-2:referee/RULES.md")


def test_setup_wants_a_public_place_for_the_commitment(box):
    repo, e, gh = box
    r = setup(repo, {**e, "FAKE_GH_VISIBILITY": "PRIVATE"})
    assert r.returncode == 1 and "Make it public, or create a small public repo and set AWE_WITNESS_REPO" in r.stdout
    assert git(repo, "tag", "-l") == ""
    # the first run wrote the config file, and that is where the witness repo is named
    cfg = Path(e["AWE_STATE_DIR"]) / "config.env"
    cfg.write_text(cfg.read_text().replace("AWE_WITNESS_REPO=\n", "AWE_WITNESS_REPO=someone/witness\n"))
    r = setup(repo, {**e, "FAKE_GH_VISIBILITY": "PRIVATE", "FAKE_GH_WITNESS_VISIBILITY": "PRIVATE"})
    assert r.returncode == 1 and "It has to be public" in r.stdout and git(repo, "tag", "-l") == ""
    r = setup(repo, {**e, "FAKE_GH_VISIBILITY": "PRIVATE"})
    assert r.returncode == 0 and "the witness repo someone/witness is public" in r.stdout and git(repo, "tag", "-l") == "genesis-1"


def test_setup_refuses_a_detached_commit(box):
    repo, e, _ = box
    git(repo, "checkout", "-q", "--detach")
    r = setup(repo, e)
    assert r.returncode == 1 and "detached commit" in r.stdout and git(repo, "tag", "-l") == ""
