"""Run one arm on a list of seeds. The command the referee calls. See referee/CONTRACT.md."""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="awe.run_arm")
    ap.add_argument("--task", required=True, choices=["transfer_cube", "insertion"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--budget", default=None, help="ledger budget. Unused under the referee")
    ap.add_argument("--purpose", default="run_arm")
    a = ap.parse_args(argv)
    from .seeds import check_seed, seed_set

    with open(a.seeds_file) as f:
        seeds = [check_seed(s) for s in json.load(f)["seeds"]]
    budget = a.budget or ("setup" if all(seed_set(s) == "reproduction" for s in seeds) else "agent")
    from . import arms, pool

    spec = arms.resolve(a.arm, a.manifest)
    pool.run(a.task, [(spec, s, None) for s in seeds], a.out, budget=budget, purpose=a.purpose, workers=a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
