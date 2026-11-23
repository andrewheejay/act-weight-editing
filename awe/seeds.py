"""The seed lock. Rollouts may only use the pre-registered seed sets."""
import json
import os

TEST_BASE = 1_000_000_000
ALLOWED = ((1000, 1499), (20000, 20999), (30000, 30499))


class SeedLockError(RuntimeError):
    pass


def check_seed(seed: int) -> int:
    seed = int(seed)
    if seed >= TEST_BASE:
        draw = os.environ.get("AWE_JUDGE_DRAW")
        if not draw:
            raise SeedLockError(f"seed {seed} is a test seed and no referee draw is supplied")
        try:
            block = json.load(open(draw))["block"]
            lo, hi = int(block["start"]), int(block["end"])
        except Exception as e:
            raise SeedLockError(f"cannot read the referee draw file {draw}: {e}") from e
        if lo < TEST_BASE or not lo <= seed <= hi:
            raise SeedLockError(f"seed {seed} is outside the referee draw {lo} to {hi}")
        return seed
    if not any(a <= seed <= b for a, b in ALLOWED):
        raise SeedLockError(f"seed {seed} is not in the reproduction, design or validation sets")
    return seed


def seed_set(seed: int) -> str:
    for name, (a, b) in zip(("reproduction", "design", "validation"), ALLOWED):
        if a <= seed <= b:
            return name
    return "test" if seed >= TEST_BASE else "none"
