"""Runs the fresh-context reviewer for a gate. The reviewer is a separate Claude Code session that
can read everything and change nothing. The loop calls this. The agent never does.

    python -m referee.review step
    python -m referee.review --task T --gate g1 --round 1

The review is advisory. Every blocking issue needs a written answer from the agent, and unresolved
issues are published with the result.

Two kinds of failure are kept apart.

- The session could not run at all (a usage limit, a sign-out, an API error, a missing binary).
  Nothing is counted. This command exits with code 75 and the loop waits and tries again, exactly
  as it does for an agent session. A usage limit therefore never skips a review.
- The session ran and still gave no usable verdict (it ran out of turns, or its answer did not fit
  the schema). After three of those the gate is recorded as not independently reviewed, so a
  broken reviewer never deadlocks the run, and the verdict says so.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import status as status_mod
from .common import HERE, head_commit, load_config, now_iso, read_json, referee_commit, repo_root, scratch, write_json

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise"]},
        "blocking_issues": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["verdict", "blocking_issues", "notes"],
    "additionalProperties": False,
}

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
TEMPFAIL = 75  # the reviewer could not run. Try again later, and count nothing
RAN_OUT = ("error_max_turns", "error_max_structured_output_retries")  # it ran and did not finish


def build_prompt(task: str, gate: str, rnd: int) -> str:
    base = gate.split(".")[0]
    text = (HERE / f"reviewer_{base}.md").read_text()
    prev = ""
    if rnd > 1:
        prev = (
            f"This is round {rnd}. Read reviews/{task}-{gate}-r{rnd - 1}.json, which is the earlier review, and "
            f"reviews/{task}-{gate}-r{rnd - 1}-response.md, which is the agent's answer. Keep only the issues that "
            "are still open after that answer. Do not raise new minor points."
        )
    return text.replace("{{TASK}}", task).replace("{{GATE}}", gate).replace("{{ROUND}}", str(rnd)).replace("{{PREVIOUS}}", prev)


def run_review(root: Path, cfg: dict, task: str, gate: str, rnd: int) -> int:
    out = root / "reviews" / f"{task}-{gate}-r{rnd}.json"
    if out.exists():
        print(f"review already exists: {out}")
        return 0
    fails = scratch(root) / f"review-fail-{task}-{gate}-r{rnd}"
    n_fail = int(fails.read_text()) if fails.exists() else 0

    cmd = [
        os.environ.get("AWE_CLAUDE_BIN", "claude"),
        "-p",
        build_prompt(task, gate, rnd),
        "--permission-mode", "dontAsk",
        "--allowedTools", "Read", "Grep", "Glob", "Bash(python -m referee.status*)", "Bash(python -m referee.recompute *)",
        "--disallowedTools", "Edit", "Write", "NotebookEdit",
        "--append-system-prompt-file", str(HERE / "reviewer_system.md"),
        "--output-format", "json",
        "--json-schema", json.dumps(SCHEMA),
        "--max-turns", os.environ.get("AWE_REVIEW_MAX_TURNS", "80"),
    ]
    # Named on purpose. Left to the account default, a box whose default model is Fable would bill
    # usage credits without asking, because print mode never asks.
    model = os.environ.get("AWE_REVIEW_MODEL") or os.environ.get("AWE_MODEL", "opus")
    if model:
        cmd += ["--model", model]
    effort = os.environ.get("AWE_REVIEW_EFFORT", "xhigh")
    if effort and effort not in EFFORT_LEVELS:
        # Claude Code would ignore the unknown level and run at the model's default. Nothing is counted
        print(f"AWE_REVIEW_EFFORT is {effort!r}. Use one of {', '.join(EFFORT_LEVELS)}, or leave it empty for the model's default. Claude Code would ignore an unknown level and quietly run at the default.")
        return 2
    if effort:
        cmd += ["--effort", effort]
    env = dict(os.environ)
    env.pop("AWE_AGENT", None)  # the agent's hooks do not apply to the reviewer
    env.pop("CLAUDE_CODE_EFFORT_LEVEL", None)  # it can outrank --effort
    result, raw, unavailable = None, "", None
    last_error = scratch(root) / "review-last-error.txt"
    try:
        r = subprocess.run(cmd, cwd=str(root), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3 * 3600)
        raw = r.stdout
        if "effort" in (r.stderr or "").lower():  # for example a cap that lowered the level
            print("Claude Code said something about effort. " + " ".join(r.stderr.split())[:300])
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            unavailable = f"claude exited with code {r.returncode} and printed no result. {(r.stderr or r.stdout)[-400:]}"
        else:
            so = data.get("structured_output")
            if isinstance(so, dict) and so.get("verdict") in ("pass", "revise"):
                result = {**so, "session_id": data.get("session_id"), "model_usage": sorted((data.get("modelUsage") or {}).keys()), "effort": effort or "default"}
            elif data.get("is_error") and data.get("subtype") not in RAN_OUT:
                unavailable = f"{data.get('result') or data.get('subtype') or 'the session ended with an error'} {(r.stderr or '')[-300:]}"
    except subprocess.TimeoutExpired as e:
        raw = f"{type(e).__name__}: {e}"
    except OSError as e:
        unavailable = f"{type(e).__name__}: {e}"

    if unavailable is not None:
        unavailable = " ".join(unavailable.split())[:600]
        last_error.write_text(unavailable + "\n")
        print(f"the reviewer could not run. Nothing was counted. {unavailable}")
        return TEMPFAIL
    last_error.unlink(missing_ok=True)

    if result is None:
        n_fail += 1
        fails.write_text(str(n_fail))
        (scratch(root) / f"review-raw-{task}-{gate}-r{rnd}.txt").write_text(raw[-20000:])
        print(f"the reviewer did not return a usable verdict (attempt {n_fail})")
        if n_fail < 3:
            return 1
        result = {"verdict": "pass", "blocking_issues": [], "notes": "The reviewer ran 3 times and never returned a usable verdict. This gate was not independently reviewed.", "reviewer_failed": True}

    if result["verdict"] == "revise" and not result["blocking_issues"]:
        result["verdict"] = "pass"  # nothing to answer means nothing blocks
    result.update({"task": task, "gate": gate, "round": rnd, "time": now_iso(), "reviewed_commit": head_commit(root)})
    write_json(out, result)
    referee_commit(root, [str(out.relative_to(root))], f"referee: review of {gate} for {task}, round {rnd}, {result['verdict']}")
    print(f"review written: {out} ({result['verdict']}, {len(result['blocking_issues'])} blocking issues)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="referee.review")
    ap.add_argument("command", nargs="?", choices=["step"])
    ap.add_argument("--repo")
    ap.add_argument("--task")
    ap.add_argument("--gate")
    ap.add_argument("--round", type=int)
    a = ap.parse_args(argv)
    root = repo_root(a.repo)
    cfg = load_config()
    task, gate, rnd = a.task, a.gate, a.round
    if not (task and gate and rnd):
        st = status_mod.compute(root, cfg)
        if st["state"] != "REVIEW_TURN":
            print(f"no review is due. State is {st['state']}")
            return 0
        task, gate, rnd = st["task"], st["gate"], st["round"]
    return run_review(root, cfg, task, gate, rnd)


if __name__ == "__main__":
    sys.exit(main())
