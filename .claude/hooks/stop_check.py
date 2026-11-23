#!/usr/bin/env python3
"""Stop hook. A session may end only after a clean handoff.

That means the lab notebook has a new entry and the work is committed, so the next fresh session
can pick up from the files. The hook blocks at most three times per session and then lets go, so
it can never trap a session. Only active in sessions the run loop starts (AWE_AGENT=1).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MAX_BLOCKS = 3


def run(root: Path, *cmd: str, timeout: int = 120) -> str:
    try:
        return subprocess.run(list(cmd), cwd=str(root), capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def main() -> None:
    if os.environ.get("AWE_AGENT") != "1":
        sys.exit(0)
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get("AWE_REPO") or data.get("cwd") or ".").resolve()
    sid = str(data.get("session_id") or "unknown")
    sfile = root / ".awe" / "sessions" / f"{sid}.json"
    try:
        st = json.loads(sfile.read_text())
    except Exception:
        sys.exit(0)  # no baseline for this session, so nothing to compare against
    if st.get("blocks", 0) >= MAX_BLOCKS:
        sys.exit(0)

    state = run(root, sys.executable, "-m", "referee.status", "--state-only", timeout=180)
    if state in ("DONE", "WAITING_FOR_ANDREW"):
        problems = []
    else:
        problems = []
        prog = root / "PROGRESS.md"
        sha = hashlib.sha256(prog.read_bytes()).hexdigest() if prog.exists() else ""
        if sha == st.get("progress_sha"):
            problems.append("append a dated entry to PROGRESS.md with what you did, the numbers you measured, what you ruled out, and the next step")
    dirty = [x for x in run(root, "git", "status", "--porcelain", "--untracked-files=no").splitlines() if x.strip()]
    if dirty:
        problems.append("commit your changes (" + ", ".join(d[3:] for d in dirty[:6]) + ")")
    untracked = [x for x in run(root, "git", "ls-files", "--others", "--exclude-standard").splitlines() if x.strip()]
    if untracked:
        problems.append("add the new files to git, or to .gitignore if they are caches (" + ", ".join(untracked[:6]) + ")")

    if not problems:
        sys.exit(0)
    st["blocks"] = st.get("blocks", 0) + 1
    try:
        sfile.write_text(json.dumps(st))
    except Exception:
        pass
    print(json.dumps({"decision": "block", "reason": "Before this session ends: " + "; ".join(problems) + ". Then stop."}))
    sys.exit(0)


if __name__ == "__main__":
    main()
