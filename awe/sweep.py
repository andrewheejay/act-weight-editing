"""Stage 2 closed-loop sweep, SPEC.md Stage 2 step 4 and 5. Resumable, every rollout goes to the ledger.

python -m awe.sweep run          the original and all 96 single-head edits (x0 and x2) on seeds 20000 to 20499
python -m awe.sweep recheck      the edits listed in stage2/recheck_arms.json and the original on seeds 20500 to 20999
"""
import argparse
import json

from . import edits, pool
from .arms import make_spec
from .paths import exp

FIRST, SECOND = range(20000, 20500), range(20500, 21000)
SCALES = (0, 2)
SHORT = {b: f"enc{i}" for i, b in enumerate(edits.BLOCKS[:4])} | {edits.BLOCKS[4]: "cross", edits.BLOCKS[5]: "dself"}


def arm_id(block, head, scale):
    return f"{SHORT[block]}.h{head}.x{scale}"


def single_head_specs():
    return {arm_id(b, h, s): make_spec(arm_id(b, h, s), [{"op": "scale_head", "block": b, "head": h, "scale": s}])
            for b in edits.BLOCKS for h in range(edits.N_HEADS) for s in SCALES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "recheck"])
    ap.add_argument("--task", default="transfer_cube")
    a = ap.parse_args()
    d = exp(a.task) / "stage2"
    specs = single_head_specs()
    if a.cmd == "run":
        arms, seeds, out, purpose = list(specs.values()), FIRST, d / "sweep.jsonl", "sweep"
    else:
        ids = json.loads((d / "recheck_arms.json").read_text())["arms"]
        arms, seeds, out, purpose = [specs[i] for i in ids], SECOND, d / "sweep_half2.jsonl", "split_half"
    arms = [make_spec("original")] + arms
    pool.run(a.task, [(sp, s, None) for sp in arms for s in seeds], out, budget="agent", purpose=purpose)


if __name__ == "__main__":
    main()
