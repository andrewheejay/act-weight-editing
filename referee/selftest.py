"""Quick self-test of the referee. Run this on a new box before the first session.

    python -m referee.selftest            offline checks
    python -m referee.selftest --online   also fetch and verify a live beacon round
"""

from __future__ import annotations

import sys
import time

from . import beacon, stats
from .common import load_config


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cfg = load_config()
    problems = list(beacon.selfcheck())

    # exact McNemar against values worked out by hand
    if abs(stats.mcnemar_exact(0, 0) - 1.0) > 1e-12:
        problems.append("mcnemar(0,0) should be 1")
    if abs(stats.mcnemar_exact(5, 0) - 2 / 32) > 1e-12:
        problems.append("mcnemar(5,0) should be 1/16")
    if abs(stats.mcnemar_exact(3, 3) - 1.0) > 1e-12:
        problems.append("mcnemar(3,3) should be 1")
    if abs(stats.mcnemar_exact(8, 2) - 2 * (1 + 10 + 45) / 1024) > 1e-12:
        problems.append("mcnemar(8,2) should be 112/1024")
    for k, want in ((40, 1), (39, 1), (19, 0), (18, -1)):
        if stats.random_allowed(k) != want:
            problems.append(f"random_allowed({k}) should be {want}")

    blk = beacon.seed_block(beacon.TEST_VECTOR["randomness"], "transfer_cube", "0" * 40, cfg["test_seeds"])
    t = cfg["test_seeds"]
    if not (t["base"] <= blk["start"] < t["base"] + t["span"]) or blk["end"] >= 2**31:
        problems.append("seed block falls outside the allowed range")

    if "--online" in argv:
        b = cfg["beacon"]
        rnd = beacon.first_round_at_or_after(time.time() - 30, b["genesis_time"], b["period"])
        try:
            v = beacon.fetch_round(rnd, b)
            print(f"live beacon round {rnd} verified through {v['relays'][0]}")
        except Exception as e:
            problems.append(f"live beacon check failed ({e})")

    if problems:
        print("SELFTEST FAILED")
        for p in problems:
            print(" -", p)
        return 1
    print("referee selftest passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
