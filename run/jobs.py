#!/usr/bin/env python3
"""A tiny job runner for long computations, so the agent never sits and waits on them.

    python run/jobs.py start NAME -- COMMAND ...     start a detached job, logging to jobs/NAME.log
    python run/jobs.py status [--json]               running, finished and dead jobs
    python run/jobs.py tail NAME [-n 40]             the end of a job's log
    python run/jobs.py wait [--timeout SECONDS]      block until nothing is running
    python run/jobs.py clear NAME                    forget a finished job

The agent starts a sweep here, writes its handoff note and ends the session. The loop then waits for
the job without spending any tokens, and starts the next session when it is done.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("AWE_REPO") or Path(__file__).resolve().parent.parent)
JOBS = ROOT / "jobs"


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _state(name: str) -> dict:
    meta = json.loads((JOBS / f"{name}.json").read_text())
    ex = JOBS / f"{name}.exit"
    if ex.exists():
        return {**meta, "name": name, "state": "done", "exit": ex.read_text().strip()}
    return {**meta, "name": name, "state": "running" if _alive(meta.get("pid")) else "dead"}


def _all() -> list[dict]:
    if not JOBS.is_dir():
        return []
    return [_state(p.stem) for p in sorted(JOBS.glob("*.json"))]


def start(name: str, cmd: list[str]) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
        print("job names use letters, digits, dot, dash and underscore")
        return 2
    if not cmd:
        print("give a command after --")
        return 2
    JOBS.mkdir(exist_ok=True)
    if (JOBS / f"{name}.json").exists():
        st = _state(name)
        if st["state"] == "running":
            print(f"job {name} is already running (pid {st['pid']})")
            return 1
        for suffix in (".exit", ".json"):
            (JOBS / f"{name}{suffix}").unlink(missing_ok=True)
    log = JOBS / f"{name}.log"
    exit_file = JOBS / f"{name}.exit"
    line = " ".join(shlex.quote(c) for c in cmd)
    wrapper = f"{line}; echo $? > {shlex.quote(str(exit_file))}"
    with open(log, "ab") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} start: {line}\n".encode())
        p = subprocess.Popen(["bash", "-c", wrapper], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (JOBS / f"{name}.json").write_text(json.dumps({"pid": p.pid, "cmd": line, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(f"started {name} (pid {p.pid}). Log: jobs/{name}.log")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "start":
        if "--" not in rest or not rest or rest[0] == "--":
            print("usage: jobs.py start NAME -- COMMAND ...")
            return 2
        i = rest.index("--")
        return start(rest[0], rest[i + 1 :])
    if cmd == "status":
        jobs = _all()
        if "--json" in rest:
            print(json.dumps(jobs))
        elif not jobs:
            print("no jobs")
        else:
            for j in jobs:
                extra = f"exit {j['exit']}" if j["state"] == "done" else f"pid {j.get('pid')}"
                print(f"{j['name']}: {j['state']} ({extra}), started {j.get('started')}, {j.get('cmd', '')[:100]}")
        return 0
    if cmd == "tail":
        if not rest:
            return 2
        n = int(rest[rest.index("-n") + 1]) if "-n" in rest else 40
        p = JOBS / f"{rest[0]}.log"
        if not p.exists():
            print(f"no log for {rest[0]}")
            return 1
        print("\n".join(p.read_text(errors="replace").splitlines()[-n:]))
        return 0
    if cmd == "wait":
        timeout = float(rest[rest.index("--timeout") + 1]) if "--timeout" in rest else None
        t0 = time.time()
        while any(j["state"] == "running" for j in _all()):
            if timeout is not None and time.time() - t0 > timeout:
                print("still running")
                return 1
            time.sleep(5)
        return 0
    if cmd == "clear":
        if not rest:
            return 2
        if (JOBS / f"{rest[0]}.json").exists() and _state(rest[0])["state"] == "running":
            print("that job is still running")
            return 1
        for suffix in (".json", ".exit"):
            (JOBS / f"{rest[0]}{suffix}").unlink(missing_ok=True)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
