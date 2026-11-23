"""Shared helpers for the referee. Standard library only, so hooks can import it with the system Python."""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass
class Check:
    """One yes or no finding. `soft` checks are shown and never block."""

    name: str
    ok: bool
    detail: str = ""
    soft: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def all_ok(checks: list[Check]) -> bool:
    return all(c.ok or c.soft for c in checks)


def failing(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.ok and not c.soft]


# ---------------------------------------------------------------- locations


def repo_root(arg: str | None = None) -> Path:
    """The experiment repo. Given explicitly, or by AWE_REPO, or found by walking up from the cwd."""
    if arg:
        return Path(arg).resolve()
    env = os.environ.get("AWE_REPO")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "PLAN.json").exists() and (p / "SPEC.md").exists():
            return p
    raise SystemExit("referee: no repo found. Run from inside the repo or pass --repo.")


def load_config() -> dict:
    """Config always comes from the referee's own folder, never from the working tree being judged."""
    return json.loads((HERE / "config.json").read_text())


def exp_dir(root: Path, task: str) -> Path:
    return root / "experiments" / task


def scratch(root: Path) -> Path:
    d = root / ".awe"
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------- files


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON ({e})") from e


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def find_jsonl(path: Path) -> Path | None:
    """Accept NAME.jsonl or NAME.jsonl.gz."""
    path = Path(path)
    if path.exists():
        return path
    gz = Path(str(path) + ".gz")
    return gz if gz.exists() else None


def read_jsonl(path: Path) -> list[dict]:
    p = find_jsonl(path)
    if p is None:
        raise FileNotFoundError(str(path))
    out = []
    with _open_text(p) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p} line {n} is not valid JSON ({e})") from e
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(paths: list[Path]) -> str:
    """Content hash over a set of files. A missing file hashes as 'missing'."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(p.encode())
        h.update(b"\0")
        q = find_jsonl(Path(p)) or Path(p)
        h.update(sha256_file(q).encode() if q.exists() and q.is_file() else b"missing")
    return h.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- git


def git(root: Path, *args: str, check: bool = False, env: dict | None = None) -> tuple[int, str]:
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=e)
    out = (r.stdout or "") + (("\n" + r.stderr) if r.returncode != 0 and r.stderr else "")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed\n{out}")
    return r.returncode, out.strip()


REFEREE_IDENT = ["-c", "user.name=awe-referee", "-c", "user.email=awe-referee@users.noreply.github.com"]


def referee_commit(root: Path, paths: list[str], message: str) -> bool:
    """Commit the given paths as the referee. Returns True if a commit was made."""
    existing = [p for p in paths if (root / p).exists()]
    if not existing:
        return False
    git(root, "add", "-A", "--", *existing, check=True)
    rc, _ = git(root, "diff", "--cached", "--quiet", "--", *existing)
    if rc == 0:
        return False
    subprocess.run(
        ["git", "-C", str(root), *REFEREE_IDENT, "commit", "-q", "-m", message, "--", *existing],
        check=True,
        capture_output=True,
        text=True,
    )
    return True


def current_genesis(root: Path) -> str | None:
    """The newest tag named genesis-N. The rules in force are the ones at that tag."""
    rc, out = git(root, "tag", "-l", "genesis-*")
    best, best_n = None, -1
    for t in out.split():
        try:
            n = int(t.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if n > best_n:
            best, best_n = t, n
    return best


def tag_exists(root: Path, tag: str) -> bool:
    rc, out = git(root, "tag", "-l", tag)
    return rc == 0 and tag in out.split()


def head_commit(root: Path) -> str:
    return git(root, "rev-parse", "HEAD", check=True)[1]


def tracked_dirty(root: Path) -> list[str]:
    rc, out = git(root, "status", "--porcelain", "--untracked-files=no")
    return [line for line in out.splitlines() if line.strip()]


# ---------------------------------------------------------------- protected paths


def is_protected(rel: str, cfg: dict) -> bool:
    """rel is a path relative to the repo root, with forward slashes."""
    rel = rel.lstrip("./")
    p = cfg["protected"]
    if rel in p["files"]:
        return True
    if any(rel == d.rstrip("/") or rel.startswith(d) for d in p["dirs"]):
        return True
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel, g.replace("/**", "/*")) for g in p["globs"])


def protected_pathspecs(cfg: dict) -> list[str]:
    """Paths compared against the genesis tag. Runtime outputs under experiments/ and reviews/ are excluded."""
    p = cfg["protected"]
    return [*p["files"], *[d for d in p["dirs"] if d != ".git/"]]


def changed_since_genesis(root: Path, cfg: dict) -> tuple[str | None, list[str]]:
    """Protected files that differ from the current genesis tag, in the commit history or the working tree."""
    tag = current_genesis(root)
    if tag is None:
        return None, []
    rc, out = git(root, "diff", "--name-only", tag, "--", *protected_pathspecs(cfg))
    changed = [x for x in out.splitlines() if x.strip()] if rc == 0 else [f"git diff failed: {out}"]
    rc, out = git(root, "ls-files", "--others", "--exclude-standard", "--", *protected_pathspecs(cfg))
    changed += [x for x in out.splitlines() if x.strip() and "__pycache__" not in x]
    return tag, sorted(set(changed))


def appended_only(root: Path, path: str) -> tuple[bool, int]:
    """True if no commit since genesis removed a line from the file."""
    tag = current_genesis(root)
    if tag is None:
        return True, 0
    rc, out = git(root, "log", "--numstat", "--format=", f"{tag}..HEAD", "--", path)
    removed = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[1].isdigit():
            removed += int(parts[1])
    return removed == 0, removed


# ---------------------------------------------------------------- per-episode records


def by_seed(records: list[dict], key: str) -> dict[int, bool]:
    out = {}
    for r in records:
        if r.get("seed") is None or key not in r:
            continue
        out[int(r["seed"])] = bool(r[key])
    return out


def in_range(seed: int, rng: list[int]) -> bool:
    return rng[0] <= seed <= rng[1]


def seeds_cover(records: list[dict], rng: list[int], need: int | None = None) -> tuple[bool, int]:
    have = {int(r["seed"]) for r in records if r.get("seed") is not None and in_range(int(r["seed"]), rng)}
    need = (rng[1] - rng[0] + 1) if need is None else need
    return len(have) >= need, len(have)


def rate(records: list[dict], key: str) -> float | None:
    vals = [bool(r[key]) for r in records if key in r]
    return sum(vals) / len(vals) if vals else None


def stock_per_episode(info: dict, start_seed: int | None = None) -> list[dict]:
    """Per-episode rows from LeRobot's eval_info.json.

    Three layouts are read.
    - The old one, a top-level `per_episode` list with seeds. The published hub file uses it.
    - `per_task[i].metrics.per_episode`, with seeds.
    - The one LeRobot 0.7.0 really writes. `per_task[0].metrics` holds plain lists (`successes`,
      `max_rewards`) and no seeds. Its `eval_policy` runs episode i on seed `start_seed + i`, in order,
      batch after batch, and keeps that order in the lists. So the caller must pass the start seed it
      gave `--seed`. This layout is read only for a single task, because with several tasks the same
      seeds would repeat. Without a start seed it yields nothing, which fails closed.
    """
    if isinstance(info.get("per_episode"), list):
        return info["per_episode"]
    tasks = [t for t in (info.get("per_task") or []) if isinstance(t, dict)]
    rows: list[dict] = []
    for t in tasks:
        rows.extend((t.get("metrics") or {}).get("per_episode") or [])
    if rows or start_seed is None or len(tasks) != 1:
        return rows
    m = tasks[0].get("metrics") or {}
    succ = m.get("successes") or []
    maxr = m.get("max_rewards") or []
    out = []
    for i, s in enumerate(succ):
        row = {"episode_ix": i, "seed": int(start_seed) + i, "success": bool(s)}
        if i < len(maxr):
            row["max_reward"] = maxr[i]
        out.append(row)
    return out
