"""The auto-greedy edit for insertion, SPEC.md Controls and Stage 4: the search of awe/autogreedy.py with one change, the score.

python -m awe.autogreedy_ins check     no rollouts: the pool, the pairs, the single-head starts on the search seeds
python -m awe.autogreedy_ins run       stage3/autogreedy.jsonl, then freeze/autogreedy.json (budget autogreedy, at most 9,900 of 10,000)
python -m awe.autogreedy_ins report    only rebuild freeze/autogreedy.json from the files already run

Rule. Everything is the rule of awe/autogreedy.py (pool of 6 kept single edits by the mean gain of the two halves, 450 search seeds 20500 to 20949,
all pairs of distinct heads, then single additions up to 4 heads, accepted only if the net gain grows), with the primary score of this task, strict
success, where that file has benchmark success. stage2/split_half.json of this task is already on strict. Nothing else differs: same seeds, pool size, tie
rules and caps. It runs only after stage3/choice.json exists, and choice.json does not read it.

What the author had seen when this file was written (2026-11-25, session 25). The search rule is older than every insertion rollout (awe/autogreedy.py,
2026-11-24). The author had seen the single-head sweep, the dose response, the group A table of 16 hand-written combinations (stage3/combo_summary.json,
seeds 20500 to 20999, which contain all 450 search seeds), and the design net of c06 (cross.h0.x0 with enc3.h3.x0) on 1,000 seeds. If both of its heads are
in the pool, the score of that pair is therefore known to the author and the search may return it, so this arm can be the agent's own candidate and is not an
independent no-LLM result by construction. The freeze states the relation from the files (as awe/freeze.py `disclose` does for transfer cube). No validation
result had been seen.
"""
import argparse

from . import autogreedy as A
from .paths import exp


def net_strict(base, rows):
    f = sum(rows[s]["strict_success"] and not base[s]["strict_success"] for s in A.SEEDS)
    b = sum(base[s]["strict_success"] and not rows[s]["strict_success"] for s in A.SEEDS)
    return f, b


A.net = net_strict  # the only change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "check"])
    ap.add_argument("--task", default="insertion")
    a = ap.parse_args()
    if a.cmd == "check":
        return A.check(a.task)
    assert (exp(a.task) / "stage3" / "choice.json").exists(), "the agent's choice comes first"
    cur, log, pl = A.search(a.task)
    A.write(a.task, cur, log, pl)
    import json

    p = exp(a.task) / "freeze" / "autogreedy.json"
    out = json.loads(p.read_text())
    out["built_by"] = "awe/autogreedy_ins.py (python -m awe.autogreedy_ins run)"
    out["rule"] = __doc__ + "\n\nThe search rule it reuses, awe/autogreedy.py:\n\n" + A.__doc__
    p.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
