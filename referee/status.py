"""The single source of truth for where the run stands. Read-only.

    python -m referee.status            a short report for people and for the agent
    python -m referee.status --json     the same, for the loop

The state is computed from the files in the repo every time. Nothing is ticked off by hand, so a task
is done only when its checks pass. States:

    AGENT_TURN          the agent has open tasks
    REVIEW_TURN         a fresh reviewer session should read a gate package
    JUDGE_TURN          the referee has a step to run, with no LLM involved
    WAITING_FOR_ANDREW  something only a person may decide
    DONE                the protocol is finished
    SETUP_INCOMPLETE    the box has not been set up
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from . import gates
from . import ledger as ledger_mod
from .common import (
    all_ok,
    changed_since_genesis,
    current_genesis,
    exp_dir,
    failing,
    fingerprint,
    load_config,
    read_json,
    repo_root,
)

STAGE_NAMES = {
    0: "Setup and reproduction",
    1: "Baseline and diagnosis",
    2: "Head atlas and sweep",
    3: "Edit design",
    4: "Freeze package",
    5: "Report",
    6: "Stress checks",
    7: "Publish prep",
}


def load_plan(root: Path) -> dict:
    plan = read_json(root / "PLAN.json")
    if not plan:
        raise SystemExit("referee: PLAN.json is missing")
    return plan


def freeze_tag(task: str, n: int = 1) -> str:
    return f"freeze-{task}-{n}"


# ------------------------------------------------------------------ reviews


def review_state(root: Path, cfg: dict, task: str, gate: str) -> dict:
    """Where a gate's review stands. Reviews are advisory, and every blocking issue needs a written answer."""
    d = root / "reviews"
    n = 0
    while (d / f"{task}-{gate}-r{n + 1}.json").exists():
        n += 1
    if n == 0:
        return {"kind": "need_review", "round": 1}
    r = read_json(d / f"{task}-{gate}-r{n}.json") or {}
    if r.get("verdict") == "pass":
        return {"kind": "settled", "round": n, "unresolved": False}
    resp = d / f"{task}-{gate}-r{n}-response.md"
    if not resp.exists() or len(resp.read_text(errors="replace").split()) < 20:
        return {"kind": "need_response", "round": n, "issues": r.get("blocking_issues", []), "response_file": str(resp.relative_to(root))}
    if n >= cfg["review"]["max_rounds"]:
        return {"kind": "settled", "round": n, "unresolved": True}
    return {"kind": "need_review", "round": n + 1}


def reviews_summary(root: Path, task: str) -> list[dict]:
    """One line per reviewed gate of a task: rounds, how it ended, and whether the reviewer really ran."""
    d = root / "reviews"
    gates_seen: dict[str, list[tuple[int, dict]]] = {}
    for f in sorted(d.glob(f"{task}-g*-r*.json")) if d.is_dir() else []:
        m = re.fullmatch(re.escape(task) + r"-(g\d+(?:\.\d+)?)-r(\d+)\.json", f.name)
        if m:
            gates_seen.setdefault(m.group(1), []).append((int(m.group(2)), read_json(f) or {}))
    out = []
    for gate in sorted(gates_seen, key=lambda g: [int(x) for x in g[1:].split(".")]):
        rounds = sorted(gates_seen[gate], key=lambda t: t[0])
        last = rounds[-1][1]
        out.append({
            "gate": gate,
            "rounds": len(rounds),
            "last_verdict": last.get("verdict"),
            "open_issues": len(last.get("blocking_issues") or []) if last.get("verdict") == "revise" else 0,
            "not_reviewed": any(r.get("reviewer_failed") for _, r in rounds),
        })
    return out


# ------------------------------------------------------------------ jobs


def jobs_summary(root: Path) -> dict:
    d = root / "jobs"
    running, dead, done = [], [], []
    if d.is_dir():
        for meta in sorted(d.glob("*.json")):
            name = meta.stem
            info = read_json(meta) or {}
            exit_file = d / f"{name}.exit"
            if exit_file.exists():
                done.append({"name": name, "exit": exit_file.read_text().strip()})
                continue
            pid = info.get("pid")
            alive = False
            if isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False
            (running if alive else dead).append({"name": name, "pid": pid})
    return {"running": running, "dead": dead, "done": done}


# ------------------------------------------------------------------ per task


def _stage_tasks(plan: dict, stage: int) -> list[dict]:
    return [t for t in plan["per_task"] if t["stage"] == stage]


def _open(plan_tasks: list[dict], root, cfg, task) -> list[dict]:
    out = []
    for t in plan_tasks:
        checks = gates.run_task(t["id"], root, cfg, task)
        if not all_ok(checks):
            out.append(
                {
                    "id": t["id"],
                    "title": t["title"],
                    "detail": t.get("detail", "").replace("{T}", f"experiments/{task}"),
                    "missing": [f"{c.name}: {c.detail}".rstrip(": ") for c in failing(checks)],
                }
            )
    return out


def _agent(task, stage, open_tasks, note="") -> dict:
    return {"state": "AGENT_TURN", "task": task, "stage": stage, "open_tasks": open_tasks, "note": note}


def task_phase(root: Path, cfg: dict, plan: dict, task: str, is_main: bool) -> dict:
    e = exp_dir(root, task)
    accepted = (e / "freeze" / "ACCEPTED.json").exists()

    # ---- after the freeze, the earlier stages are locked in the tag and are not re-checked
    if accepted:
        if (e / "test" / "FAILED.json").exists():
            f = read_json(e / "test" / "FAILED.json") or {}
            return {"state": "WAITING_FOR_ANDREW", "task": task, "stage": 5, "note": "the test failed after the seeds were drawn: " + str(f.get("reason", "see test/FAILED.json"))}
        if not (e / "test" / "verdict.json").exists():
            return {"state": "JUDGE_TURN", "task": task, "stage": 5, "action": "test", "note": "the referee runs the held-out test"}
        o = _open(_stage_tasks(plan, 5), root, cfg, task)
        if o:
            return _agent(task, 5, o, "write the report from the referee's results. The verdict is final")
        if is_main:
            o = _open(_stage_tasks(plan, 6), root, cfg, task)
            if o:
                return _agent(task, 6, o)
        return {"state": "complete", "task": task}

    # ---- stage 0
    o = _open(_stage_tasks(plan, 0), root, cfg, task)
    if o:
        return _agent(task, 0, o)
    deep = read_json(root / "reviews" / f"{task}-g0-deep.json")
    fp_data = fingerprint(gates.stage0_files(root, task))
    fp_code = gates.code_fingerprint(root)
    if not deep or deep.get("fingerprint_data") != fp_data or deep.get("fingerprint_code") != fp_code:
        return {"state": "JUDGE_TURN", "task": task, "stage": 0, "action": "confirm_g0", "note": "the referee replays a sample of rollouts to confirm Gate 0"}
    if not deep.get("ok"):
        miss = [{"id": "g0.deep", "title": "Gate 0 replay check", "detail": "The referee re-ran the tests and a sample of rollouts and they did not match.", "missing": deep.get("failing", [])}]
        return _agent(task, 0, miss, "Gate 0 failed its replay check. Fix the harness, then the referee checks again")

    # ---- stage 1
    o = _open(_stage_tasks(plan, 1), root, cfg, task)
    if o:
        return _agent(task, 1, o)
    stamp = read_json(root / "reviews" / f"{task}-g1-passed.json")
    if not stamp:
        rs = review_state(root, cfg, task, "g1")
        if rs["kind"] == "need_review":
            return {"state": "REVIEW_TURN", "task": task, "stage": 1, "gate": "g1", "round": rs["round"]}
        if rs["kind"] == "need_response":
            miss = [{"id": "g1.review", "title": "Answer the Gate 1 review", "detail": f"Write {rs['response_file']}. Answer every blocking issue, and change the diagnosis where the reviewer is right.", "missing": rs["issues"]}]
            return _agent(task, 1, miss, "a reviewer in a fresh session raised blocking issues")
        return {"state": "JUDGE_TURN", "task": task, "stage": 1, "action": "stamp_g1", "note": "the referee records Gate 1"}
    dec = read_json(e / "stage1" / "decisions.json") or {}
    if dec.get("proceed") is False:
        return {"state": "skipped", "task": task, "note": "fewer than 5% of design episodes fail, so this task is not edited"}

    # ---- stage 2
    o = _open(_stage_tasks(plan, 2), root, cfg, task)
    if o:
        return _agent(task, 2, o)

    # ---- stages 3 and 4
    rejections = sorted((e / "freeze").glob("REJECTED-*.md")) if (e / "freeze").is_dir() else []
    if len(rejections) >= cfg["freeze"]["max_rejections"]:
        return {"state": "WAITING_FOR_ANDREW", "task": task, "stage": 4, "note": f"the freeze package was rejected {len(rejections)} times"}
    note = f"the last freeze request was rejected. Read {rejections[-1].relative_to(root)}" if rejections else ""
    o3 = _open(_stage_tasks(plan, 3), root, cfg, task)
    o4 = _open(_stage_tasks(plan, 4), root, cfg, task)
    if o3 or o4:  # design and packaging are worked together, so show both lists
        return _agent(task, 3 if o3 else 4, o3 + o4, note)
    gate = f"g2.{len(rejections) + 1}"
    rs = review_state(root, cfg, task, gate)
    if rs["kind"] == "need_review":
        return {"state": "REVIEW_TURN", "task": task, "stage": 4, "gate": gate, "round": rs["round"]}
    if rs["kind"] == "need_response":
        miss = [{"id": "g2.review", "title": "Answer the freeze review", "detail": f"Write {rs['response_file']}. Answer every blocking issue. If you change the package, say what changed.", "missing": rs["issues"]}]
        return _agent(task, 4, miss, "a reviewer in a fresh session raised blocking issues")
    return {"state": "JUDGE_TURN", "task": task, "stage": 4, "action": "freeze", "note": "the referee checks the package, tags it and publishes the commitment"}


# ------------------------------------------------------------------ whole run


def compute(root: Path, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    plan = load_plan(root)
    warnings: list[str] = []

    w = root / "WAITING_FOR_ANDREW.md"
    if w.exists():
        first = " ".join(w.read_text(errors="replace").split())[:300]
        return _finish(root, cfg, {"state": "WAITING_FOR_ANDREW", "note": first}, warnings)

    genesis = current_genesis(root)
    if genesis is None:
        return _finish(root, cfg, {"state": "SETUP_INCOMPLETE", "note": "no genesis tag. Run run/setup_box.sh"}, warnings)
    _, changed = changed_since_genesis(root, cfg)
    if changed:
        warnings.append("protected files differ from " + genesis + ": " + ", ".join(changed[:6]))

    scope = os.environ.get("AWE_SCOPE", "full")
    campaign = cfg["campaign"] if scope != "main" else cfg["campaign"][:1]
    main_task = None
    notes = []
    for task in campaign:
        ph = task_phase(root, cfg, plan, task, is_main=(main_task is None))
        if ph["state"] == "skipped":
            notes.append(f"{task} skipped: {ph['note']}")
            continue
        if main_task is None:
            main_task = task
        if ph["state"] != "complete":
            ph["main_task"] = main_task
            ph["notes"] = notes
            return _finish(root, cfg, ph, warnings)

    o = []
    for t in plan.get("global", []):
        checks = gates.run_task(t["id"], root, cfg, None)
        if not all_ok(checks):
            o.append({"id": t["id"], "title": t["title"], "detail": t.get("detail", ""), "missing": [f"{c.name}: {c.detail}" for c in failing(checks)]})
    if o:
        return _finish(root, cfg, {"state": "AGENT_TURN", "task": None, "stage": 7, "open_tasks": o, "note": "", "main_task": main_task, "notes": notes}, warnings)
    return _finish(root, cfg, {"state": "DONE", "note": "the protocol is finished. See publish/CHECKLIST.md", "main_task": main_task, "notes": notes}, warnings)


def _finish(root: Path, cfg: dict, st: dict, warnings: list[str]) -> dict:
    st["warnings"] = warnings
    st["genesis"] = current_genesis(root)
    st["jobs"] = jobs_summary(root)
    task = st.get("task") or st.get("main_task")
    if task:
        try:
            st["budgets"] = ledger_mod.budget_line(root, cfg, task)
        except Exception as e:
            st["budgets"] = f"ledger unreadable ({e})"
    sig = json.dumps(
        [st["state"], st.get("task"), st.get("stage"), st.get("action"), st.get("gate"), st.get("round"), sorted(m for t in st.get("open_tasks", []) for m in [t["id"], *t["missing"]])],
        sort_keys=True,
    )
    st["signature"] = hashlib.sha256(sig.encode()).hexdigest()[:16]
    return st


def render(st: dict) -> str:
    L = []
    head = f"STATE {st['state']}"
    if st.get("task"):
        head += f"   task {st['task']}"
    if st.get("stage") is not None:
        head += f"   stage {st['stage']} ({STAGE_NAMES.get(st['stage'], '')})"
    L.append(head)
    if st.get("action"):
        L.append(f"Referee step: {st['action']}")
    if st.get("gate"):
        L.append(f"Review: gate {st['gate']}, round {st.get('round')}")
    if st.get("note"):
        L.append(f"Note: {st['note']}")
    for n in st.get("notes", []):
        L.append(f"Note: {n}")
    if st.get("open_tasks"):
        L.append("")
        L.append("Open tasks, in order")
        for t in st["open_tasks"]:
            L.append(f"- {t['id']}  {t['title']}")
            if t.get("detail"):
                L.append(f"    {t['detail']}")
            for m in t["missing"][:8]:
                L.append(f"    not yet: {m}")
    if st.get("budgets"):
        L.append("")
        L.append(f"Budgets used: {st['budgets']}")
    j = st.get("jobs", {})
    if j.get("running"):
        L.append("Jobs running: " + ", ".join(x["name"] for x in j["running"]))
    if j.get("dead"):
        L.append("Jobs that died without an exit code: " + ", ".join(x["name"] for x in j["dead"]))
    if j.get("done"):
        L.append("Jobs finished: " + ", ".join(f"{x['name']} (exit {x['exit']})" for x in j["done"]))
    for w in st.get("warnings", []):
        L.append(f"WARNING: {w}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="referee.status")
    ap.add_argument("--repo")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--state-only", action="store_true")
    a = ap.parse_args(argv)
    root = repo_root(a.repo)
    st = compute(root)
    if a.state_only:
        print(st["state"])
    elif a.json:
        print(json.dumps(st, indent=2))
    else:
        print(render(st))
    return 0


if __name__ == "__main__":
    sys.exit(main())
