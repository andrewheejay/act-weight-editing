#!/usr/bin/env python3
"""SessionStart hook. Prints the referee's status and the tail of the lab notebook into context.

It runs when a session starts, resumes, or comes back from a compaction, so the agent always
re-orients from the files and never from a stale memory. Only active in sessions the run loop
starts (AWE_AGENT=1).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    if os.environ.get("AWE_AGENT") != "1":
        sys.exit(0)
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get("AWE_REPO") or data.get("cwd") or ".").resolve()
    sid = str(data.get("session_id") or "unknown")
    source = str(data.get("source") or "startup")

    # remember what the notebook looked like when this session began, for the Stop hook
    sdir = root / ".awe" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    sfile = sdir / f"{sid}.json"
    prog = root / "PROGRESS.md"
    if not sfile.exists():
        sha = hashlib.sha256(prog.read_bytes()).hexdigest() if prog.exists() else ""
        sfile.write_text(json.dumps({"progress_sha": sha, "start": time.time(), "blocks": 0}))

    try:
        st = subprocess.run([sys.executable, "-m", "referee.status"], cwd=str(root), capture_output=True, text=True, timeout=180).stdout.strip()
    except Exception as e:
        st = f"(the status command failed: {e}. Run python -m referee.status yourself.)"

    tail = ""
    if prog.exists():
        lines = prog.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-70:])

    print("=== Referee status (computed from the files just now) ===")
    print(st)
    print()
    print("=== Tail of PROGRESS.md ===")
    print(tail or "(empty)")
    print()
    if source == "compact":
        print("Your context was just compacted. The status above is current. Reread CLAUDE.md if any rule is unclear.")
    print(f"Session id {sid}. The hard rules are in CLAUDE.md. Before stopping, append to PROGRESS.md and commit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
