"""Runs insertion episodes in a process of its own (one simulator per process). Not a test file, test_insertion.py calls it.

Usage: python tests/insertion_probe.py OUT.npz SEED...   Each rollout gets a ledger line.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from awe import render  # noqa: E402

render.configure(render.chosen("insertion"))
import json  # noqa: E402

import numpy as np  # noqa: E402

from awe import rollout  # noqa: E402
from awe.arms import make_spec  # noqa: E402
from awe.env import FastAloha  # noqa: E402
from awe.ledger import Ledger  # noqa: E402
from awe.policy import Policy  # noqa: E402

out, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
pol, env = Policy("insertion"), FastAloha("insertion")
led = Ledger("insertion", "setup", "test suite")
acts, res, recs = {}, {}, {}
for s in seeds:
    led.log("original", s)
    arr = rollout.run_episode(env, pol, seed=s)
    acts[f"{s}"], res[f"{s}_q"], res[f"{s}_r"] = arr["action"], arr["qpos"], arr["reward"]
    recs[str(s)] = rollout.summarize(arr, "original", s)
led.close()
np.savez(out, **res)
np.savez(out.replace(".npz", "_act.npz"), **acts)
json.dump(recs, open(out.replace(".npz", ".json"), "w"))
