"""The rollout ledger. One JSON line per rollout, in ledger/<name>.jsonl. Always on, except inside a referee step."""
import gzip
import json
import os
import time

from .paths import ROOT

CAPS = {"agent": 100_000, "baselines": 50_000, "autogreedy": 10_000}
BUDGETS = {"agent", "baselines", "autogreedy", "setup", "stress"}


def spent(task: str) -> dict:
    out = {}
    d = ROOT / "ledger"
    if not d.is_dir():
        return out
    for f in sorted(d.iterdir()):
        if not f.name.endswith((".jsonl", ".jsonl.gz")):
            continue
        with (gzip.open if f.name.endswith(".gz") else open)(f, "rt") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r.get("task") == task:
                        out[r["budget"]] = out.get(r["budget"], 0) + 1
    return out


class Ledger:
    def __init__(self, task, budget, purpose):
        if budget not in BUDGETS:
            raise ValueError(f"unknown budget {budget}")
        self.task, self.budget, self.purpose, self.fh = task, budget, purpose, None
        # CONTRACT.md: under the referee the harness writes nothing but --out. Only the referee sets this.
        if os.environ.get("AWE_REFEREE") != "1":
            d = ROOT / "ledger"
            d.mkdir(exist_ok=True)
            name = f"{task}-{budget}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
            self.fh = open(d / f"{name}.jsonl", "a")

    def check_budget(self, n_new: int):
        cap = CAPS.get(self.budget)
        if cap is not None and self.fh is not None:
            used = spent(self.task).get(self.budget, 0)
            if used + n_new > cap:
                raise RuntimeError(f"{self.budget} budget for {self.task}: {used} spent + {n_new} asked > cap {cap}")

    def log(self, arm, seed=None, pose=None, candidate=None):
        if self.fh is None:
            return
        row = {"task": self.task, "seed": seed, "arm": arm, "budget": self.budget, "purpose": self.purpose}
        if pose is not None:
            row["pose"] = [float(x) for x in pose]
        if candidate is not None:
            row["candidate"] = candidate
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
