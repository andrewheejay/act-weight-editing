"""The pre-registered statistics and the verdict rule. Pure standard library, exact arithmetic where it matters.

Nothing here is tuned to data. The rules are written out in RULES.md and in SPEC.md.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b counts seeds the edit fixed (original failed, edit succeeded).
    c counts seeds the edit broke (original succeeded, edit failed).
    Under the null, each discordant seed is a fair coin, so this is an exact binomial test.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2**n)))


def paired(orig: dict[int, bool], new: dict[int, bool]) -> dict:
    """Paired comparison on the seeds both arms have."""
    seeds = sorted(set(orig) & set(new))
    n = len(seeds)
    b = sum(1 for s in seeds if not orig[s] and new[s])
    c = sum(1 for s in seeds if orig[s] and not new[s])
    f0 = sum(1 for s in seeds if not orig[s])
    out = {
        "n": n,
        "fixed": b,
        "broke": c,
        "orig_success": (n - f0) / n if n else None,
        "new_success": (n - f0 + b - c) / n if n else None,
        "gain_points": 100.0 * (b - c) / n if n else None,
        "orig_failures": f0,
        "share_removed": (b - c) / f0 if f0 else None,
        "p_mcnemar": mcnemar_exact(b, c),
    }
    return out


def bootstrap_ci(orig: dict[int, bool], new: dict[int, bool], n_boot: int, seed: int) -> dict:
    """95% percentile intervals for the gain in points and the share of failures removed."""
    seeds = sorted(set(orig) & set(new))
    n = len(seeds)
    if n == 0:
        return {"gain_points": None, "share_removed": None, "n_boot": 0}
    # Per-seed codes. fixed gives +1, broke gives -1, and fail marks an original failure.
    delta = [(1 if (not orig[s] and new[s]) else 0) - (1 if (orig[s] and not new[s]) else 0) for s in seeds]
    fail = [0 if orig[s] else 1 for s in seeds]
    gains, shares = [], []
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        d = np.asarray(delta)
        f = np.asarray(fail)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            ds, fs = int(d[idx].sum()), int(f[idx].sum())
            gains.append(100.0 * ds / n)
            if fs:
                shares.append(ds / fs)
    except ImportError:
        rng = random.Random(seed)
        for _ in range(n_boot):
            ds = fs = 0
            for _ in range(n):
                i = rng.randrange(n)
                ds += delta[i]
                fs += fail[i]
            gains.append(100.0 * ds / n)
            if fs:
                shares.append(ds / fs)

    def pct(xs):
        if not xs:
            return None
        xs = sorted(xs)
        lo = xs[int(math.floor(0.025 * (len(xs) - 1)))]
        hi = xs[int(math.ceil(0.975 * (len(xs) - 1)))]
        return [lo, hi]

    return {"gain_points": pct(gains), "share_removed": pct(shares), "n_boot": n_boot}


def random_allowed(k: int, luck: float = 0.05) -> int:
    """How many of k matched random edits may gain as much as the real edit.

    The real edit passes H2 if at most this many random edits match it. With k = 40 the answer is 1,
    and a random edit would pass by luck 2 times in 41. A negative answer means k is too small for
    the 5% rule, so H2 cannot be tested.
    """
    return int(math.floor(luck * (k + 1) + 1e-9)) - 1


def verdict(
    *,
    primary_orig: dict[int, bool],
    primary_edit: dict[int, bool],
    guard_orig: dict[int, bool],
    guard_edit: dict[int, bool],
    random_pairs: list[tuple[float, float]],
    baseline_pairs: dict[str, tuple[float, float]],
    autogreedy_pair: tuple[float, float] | None,
    cfg: dict,
    boot_seed: int = 0,
) -> dict:
    """Apply the pre-registered rule and name one of the four endings.

    Each comparison arm arrives as a pair. The first number is the arm's gain over the original, and
    the second is the edit's gain over the original on the same seeds. An arm "gains as much" when
    the first is at least the second. Most arms run on every test seed, so the second number is just
    the edit's gain. Arms that run on fewer seeds are still compared like for like.
    """
    v = cfg["verdict"]
    main = paired(primary_orig, primary_edit)
    ci = bootstrap_ci(primary_orig, primary_edit, v["bootstrap"], boot_seed)
    guard = paired(guard_orig, guard_edit)

    gain = main["gain_points"] or 0.0
    significant = bool(main["p_mcnemar"] < v["alpha"] and main["fixed"] > main["broke"])
    share = main["share_removed"]
    large = bool(share is not None and share >= v["large_share"])

    guard_drop = -(guard["gain_points"] or 0.0)  # positive means the guard score fell
    guard_ok = bool(guard_drop <= v["guard_drop_points"] + 1e-9)

    k = len(random_pairs)
    allowed = random_allowed(k, v["random_luck_rate"])
    n_match = sum(1 for g, ge in random_pairs if g >= ge - 1e-12)
    h2_testable = allowed >= 0
    h2_pass = bool(h2_testable and n_match <= allowed)

    matching = sorted(name for name, (g, ge) in baseline_pairs.items() if g >= ge - 1e-12)
    best = max(baseline_pairs.items(), key=lambda kv: kv[1][0] - kv[1][1]) if baseline_pairs else None
    best_baseline = (best[0], best[1][0], best[1][1]) if best else None
    h3_clear = not matching

    reasons = []
    if not significant:
        ending = "null"
        reasons.append("no significant gain on the test seeds")
    elif not guard_ok:
        ending = "null"
        reasons.append(f"the gain is significant but the guard score fell by {guard_drop:.1f} points")
    elif h2_pass and h3_clear:
        ending = "strong_win" if large else "modest_win"
        reasons.append("significant, beats the random edits, and no simple baseline matches it")
        reasons.append("removes at least 25% of failures" if large else "removes under 25% of failures")
    else:
        ending = "partial"
        if not h2_testable:
            reasons.append(f"only {k} matched random edits exist, too few to test H2")
        elif not h2_pass:
            reasons.append(f"{n_match} of {k} random edits gained as much, and at most {allowed} may")
        if not h3_clear:
            reasons.append("a simple baseline did as well (" + ", ".join(matching) + ")")

    return {
        "ending": ending,
        "reasons": reasons,
        "h1": {**main, "significant": significant, "large": large, "ci95": ci},
        "guard": {"drop_points": guard_drop, "ok": guard_ok, "limit_points": v["guard_drop_points"], **{
            "orig": guard["orig_success"], "edit": guard["new_success"]}},
        "h2": {
            "n_random": k,
            "n_gaining_as_much": n_match,
            "allowed": allowed if h2_testable else None,
            "testable": h2_testable,
            "pass": h2_pass,
            "random_gains_points": sorted((g for g, _ in random_pairs), reverse=True),
        },
        "h3": {
            "baseline_gains_points": {name: g for name, (g, _) in baseline_pairs.items()},
            "edit_gain_on_same_seeds": {name: ge for name, (_, ge) in baseline_pairs.items()},
            "best_baseline": list(best_baseline) if best_baseline else None,
            "matching_baselines": matching,
            "clear": h3_clear,
        },
        "autogreedy": {
            "gain_points": autogreedy_pair[0] if autogreedy_pair else None,
            "matches_edit": bool(autogreedy_pair and autogreedy_pair[0] >= autogreedy_pair[1] - 1e-12),
        },
    }


ENDING_TEXT = {
    "strong_win": "Strong win",
    "modest_win": "Modest win",
    "partial": "Partial",
    "null": "Null",
}
