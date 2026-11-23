#!/usr/bin/env python3
"""A stand-in for the claude command, used only by the tests.

As an agent session it runs one step of the scripted agent. As a reviewer (called with
--json-schema) it returns a structured verdict. It can be told to fail in the ways the real CLI
fails, so the tests can watch the loop ride out a usage limit, a crash and a reviewer that never
answers. The failure shapes were copied from Claude Code 2.1.330, which reports a failed session
as subtype success with is_error true and exit code 1.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    if "--version" in argv:
        print("2.1.330 (stub)")
        return 0
    prompt = argv[argv.index("-p") + 1] if "-p" in argv else ""

    state = Path(os.environ.get("STUB_STATE", "/tmp"))
    # every call is recorded, so the tests can check the flags the loop and the reviewer really pass
    shown = ["<prompt>" if i and argv[i - 1] == "-p" else a for i, a in enumerate(argv)]
    with open(state / "stub-argv.jsonl", "a") as f:
        f.write(json.dumps({"argv": shown, "effort_env": os.environ.get("CLAUDE_CODE_EFFORT_LEVEL")}) + "\n")
    if "--json-schema" in argv:
        seen = state / "stub-review-count"
        k = int(seen.read_text()) if seen.exists() else 0
        seen.write_text(str(k + 1))
        if k < int(os.environ.get("STUB_REVIEW_LIMIT", "0")):  # a usage limit turns the reviewer away, as the real CLI reports it
            print(json.dumps({"type": "result", "subtype": "success", "is_error": True, "num_turns": 1, "result": "You've hit your weekly limit · resets 3am", "structured_output": None, "terminal_reason": "api_error", "session_id": "stub-review"}))
            return 1
        junk = os.environ.get("STUB_REVIEW_JUNK")
        if junk == "turns":  # it ran out of turns
            print(json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True, "num_turns": 80, "structured_output": None, "session_id": "stub-review"}))
            return 1
        if junk == "shape":  # it finished and its answer does not fit the schema
            print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 9, "result": "looks fine to me", "structured_output": {"verdict": "maybe"}, "session_id": "stub-review"}))
            return 0
        revise = os.environ.get("STUB_REVISE_FIRST") == "g1" and "Review Gate 1" in prompt and "review round 1." in prompt
        out = {"verdict": "revise", "blocking_issues": ["diagnosis.md shows three hand-picked episodes as near-miss evidence. Show it across all failing design seeds."], "notes": ""} if revise else {"verdict": "pass", "blocking_issues": [], "notes": "nothing blocks"}
        print(json.dumps({"type": "result", "subtype": "success", "structured_output": out, "session_id": "stub-review", "modelUsage": {"stub-model": {}}}))
        return 0

    counter = state / "stub-fail-count"
    want = int(os.environ.get("STUB_FAIL_SESSIONS", "0"))
    n = int(counter.read_text()) if counter.exists() else 0
    print(json.dumps({"type": "system", "subtype": "init", "model": "stub-model", "session_id": "stub"}))
    if os.environ.get("STUB_CRASH") == "1":  # it dies before it does anything, and it is not a usage limit
        print("TypeError: cannot read properties of undefined", file=sys.stderr)
        return 1
    if os.environ.get("STUB_LEAVES_BRANCH") == "1":  # an agent that wanders off the run's branch
        import subprocess

        subprocess.run(["git", "checkout", "-q", "-b", "scratch"], check=True)
        print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "session_id": "stub", "result": "moved"}))
        return 0
    if n < want:
        counter.write_text(str(n + 1))
        print(json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True, "num_turns": 0, "session_id": "stub", "result": "You've hit your weekly limit. It resets soon."}))
        return 1
    if os.environ.get("STUB_IDLE") == "1":  # an agent that talks and changes nothing
        print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "session_id": "stub", "result": "thinking"}))
        return 0
    sys.path.insert(0, str(HERE))
    from sim_agent import step

    did = step(Path.cwd())
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 3, "total_cost_usd": 0.0, "session_id": "stub", "result": did}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
