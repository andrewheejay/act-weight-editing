"""Read-only helpers for recomputing numbers from the raw per-seed files.

    python -m referee.recompute rates FILE [FILE ...]
    python -m referee.recompute paired ORIGINAL.jsonl OTHER.jsonl [--metric benchmark|strict]
    python -m referee.recompute classes FILE
    python -m referee.recompute ledger
    python -m referee.recompute decisions --task T
    python -m referee.recompute validation --task T

The agent and the reviewer may both use this. It never writes anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import gates, ledger, stats
from .common import by_seed, load_config, rate, read_jsonl, repo_root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="referee.recompute")
    ap.add_argument("--repo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rates")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("paired")
    p.add_argument("original")
    p.add_argument("other")
    p.add_argument("--metric", choices=["benchmark", "strict"], default="benchmark")
    p.add_argument("--arm", help="keep only rows with this arm id in OTHER")
    p = sub.add_parser("classes")
    p.add_argument("file")
    sub.add_parser("ledger")
    p = sub.add_parser("decisions")
    p.add_argument("--task", required=True)
    p = sub.add_parser("validation")
    p.add_argument("--task", required=True)
    a = ap.parse_args(argv)
    cfg = load_config()

    if a.cmd == "rates":
        for f in a.files:
            rows = read_jsonl(Path(f))
            arms = Counter(str(r.get("arm")) for r in rows)
            b, s = rate(rows, "benchmark_success"), rate(rows, "strict_success")
            print(f"{f}: {len(rows)} rows, benchmark {None if b is None else round(100 * b, 2)}%, strict {None if s is None else round(100 * s, 2)}%, arms {dict(arms.most_common(5))}")
        return 0
    if a.cmd == "paired":
        key = f"{a.metric}_success"
        o = by_seed(read_jsonl(Path(a.original)), key)
        rows = read_jsonl(Path(a.other))
        if a.arm:
            rows = [r for r in rows if str(r.get("arm")) == a.arm]
        print(json.dumps(stats.paired(o, by_seed(rows, key)), indent=2))
        return 0
    if a.cmd == "classes":
        rows = read_jsonl(Path(a.file))
        print(json.dumps(dict(sorted(Counter(str(r.get("failure_class", "unknown")) for r in rows).items())), indent=2))
        return 0

    root = repo_root(a.repo)
    if a.cmd == "ledger":
        s = ledger.summarize(root, cfg, use_cache=False)
        s.pop("stamp", None)
        print(json.dumps(s, indent=2))
    elif a.cmd == "decisions":
        exp, checks = gates.expected_decisions(root, cfg, a.task)
        print(json.dumps({"recomputed": exp, "problems": [c.as_dict() for c in checks if not c.ok]}, indent=2))
    elif a.cmd == "validation":
        print(json.dumps(gates.validation_result(root, cfg, a.task), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
