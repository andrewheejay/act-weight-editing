"""Reads the rollout ledger and checks it against the budgets and the seed rules.

One ledger line is one rollout. The harness writes them. Required keys are task, seed, arm and budget.
A rollout at an explicit pose (the Stage 1 grid, the Stage 6 stress checks) has seed null and a pose.
A rollout on a validation seed also names its candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .common import Check, in_range, read_json, read_jsonl, scratch, write_json


def ledger_files(root: Path) -> list[Path]:
    d = root / "ledger"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))


def _stamp(files: list[Path]) -> str:
    return "|".join(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files)


def summarize(root: Path, cfg: dict, use_cache: bool = True) -> dict:
    files = ledger_files(root)
    stamp = _stamp(files)
    cache = scratch(root) / "ledger-summary.json"
    if use_cache:
        c = read_json(cache)
        if c and c.get("stamp") == stamp:
            return c

    sets = cfg["seed_sets"]
    legal = [sets["reproduction"], sets["design"], sets["validation"]]
    test_base = cfg["test_seeds"]["base"]

    counts: dict[str, Counter] = defaultdict(Counter)  # task -> budget -> n
    illegal: list[str] = []
    test_range = 0
    bad_lines = 0
    unknown_budget = Counter()
    val_candidates: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))  # task -> budget -> ids
    total = 0
    known = set(cfg["budgets"]) | set(cfg["uncapped_budgets"])

    for f in files:
        try:
            rows = read_jsonl(f)
        except ValueError as e:
            illegal.append(str(e))
            continue
        for r in rows:
            total += 1
            task, budget, seed = r.get("task"), r.get("budget"), r.get("seed")
            if not task or not budget or "arm" not in r:
                bad_lines += 1
                continue
            if budget not in known:
                unknown_budget[budget] += 1
            counts[task][budget] += 1
            if seed is None:
                if "pose" not in r:
                    bad_lines += 1
                continue
            seed = int(seed)
            if seed >= test_base:
                test_range += 1
            elif not any(in_range(seed, rng) for rng in legal):
                if len(illegal) < 20:
                    illegal.append(f"{f.name}: seed {seed} is outside the reproduction, design and validation sets")
            if in_range(seed, sets["validation"]):
                cand = str(r.get("candidate") or r.get("arm"))
                if cand != "original":
                    val_candidates[task][budget].add(cand)

    out = {
        "stamp": stamp,
        "files": [p.name for p in files],
        "total": total,
        "counts": {t: dict(c) for t, c in counts.items()},
        "illegal": illegal,
        "test_range_rollouts": test_range,
        "bad_lines": bad_lines,
        "unknown_budget": dict(unknown_budget),
        "validation_candidates": {t: {b: sorted(s) for b, s in d.items()} for t, d in val_candidates.items()},
    }
    write_json(cache, out)
    return out


def checks(root: Path, cfg: dict, task: str) -> list[Check]:
    s = summarize(root, cfg)
    out = [
        Check("ledger.exists", s["total"] > 0, f"{s['total']} rollouts in {len(s['files'])} files"),
        Check("ledger.well_formed", s["bad_lines"] == 0, f"{s['bad_lines']} lines lack task, arm, budget, or a pose"),
        Check(
            "ledger.budget_names",
            not s["unknown_budget"],
            "unknown budget names " + ", ".join(s["unknown_budget"]) if s["unknown_budget"] else "",
        ),
        Check("ledger.legal_seeds", not s["illegal"], "; ".join(s["illegal"][:5])),
        Check(
            "ledger.no_test_seeds",
            s["test_range_rollouts"] == 0,
            f"{s['test_range_rollouts']} rollouts used seeds in the test range",
        ),
    ]
    used = s["counts"].get(task, {})
    for name, cap in cfg["budgets"].items():
        n = used.get(name, 0)
        out.append(Check(f"budget.{name}", n <= cap, f"{n:,} of {cap:,} rollouts"))
    cands = s["validation_candidates"].get(task, {})
    for name, cap in cfg["validation_checks"].items():
        ids = cands.get(name, [])
        out.append(Check(f"validation_checks.{name}", len(ids) <= cap, f"{len(ids)} of {cap} ({', '.join(ids[:12])})"))
    other = sorted(b for b in cands if b not in cfg["validation_checks"])
    out.append(
        Check(
            "validation_checks.budget",
            not other,
            "validation seeds were used under the budgets " + ", ".join(other) if other else "",
        )
    )
    return out


def budget_line(root: Path, cfg: dict, task: str) -> str:
    s = summarize(root, cfg)
    used = s["counts"].get(task, {})
    parts = [f"{b} {used.get(b, 0):,}/{cap:,}" for b, cap in cfg["budgets"].items()]
    cands = s["validation_candidates"].get(task, {})
    parts.append(f"validation checks {len(cands.get('agent', []))}/{cfg['validation_checks']['agent']}")
    return ", ".join(parts)
