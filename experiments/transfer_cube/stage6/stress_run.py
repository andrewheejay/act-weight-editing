"""Stage 6 stress checks (SPEC.md Stage 6 item 2). Reported, not judged. Explicit poses only, ledger budget `stress`, no seed.

python experiments/transfer_cube/stage6/stress_run.py run      -> stage6/stress_runs.jsonl (resumable)
python experiments/transfer_cube/stage6/stress_run.py report   -> stage6/stress.json, stress.md

Poses are fixed before any stress rollout (rng 20261125):
- band_3cm     1,000 cube positions uniform in the 3 cm band outside the square, that is inside x -0.03 to 0.23, y 0.37 to 0.63 and outside x 0 to 0.2, y 0.4 to 0.6, no rotation
- rotation_0, rotation_15, rotation_30   the same 500 positions uniform in the square, cube turned about the vertical by +/-15 or +/-30 degrees, sign random per position.
  rotation_0 is the control and is not asked for by SPEC.md
Arms: original and agent_edit, from freeze/test_manifest.json, at 100-step chunks. The stock reset drops the cube at z 0.05 as sample_box_pose does.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

T = ROOT / "experiments" / "transfer_cube"
OUT = T / "stage6"


def sets():
    rng = np.random.default_rng(20261125)
    S = {}
    band = []
    while len(band) < 1000:
        x, y = rng.uniform(-0.03, 0.23), rng.uniform(0.37, 0.63)
        if not (0 <= x <= 0.2 and 0.4 <= y <= 0.6):
            band.append([x, y, 0.05, 1.0, 0.0, 0.0, 0.0])
    S["band_3cm"] = band
    pos = [(rng.uniform(0, 0.2), rng.uniform(0.4, 0.6), 1.0 if rng.random() < 0.5 else -1.0) for _ in range(500)]
    for deg in (0, 15, 30):
        S[f"rotation_{deg}"] = [[x, y, 0.05, math.cos(math.radians(s * deg) / 2), 0.0, 0.0, math.sin(math.radians(s * deg) / 2)] for x, y, s in pos]
    return S


def rnd(p):
    return [float(v) for v in p]


def specs():
    from awe.arms import from_manifest, make_spec

    man = json.loads((T / "freeze" / "test_manifest.json").read_text())
    return {"original": make_spec("original"), "agent_edit": from_manifest(man, "agent_edit")}


def run():
    from awe import pool

    S, A = sets(), specs()
    jobs = [(a, None, rnd(p)) for k in S for p in S[k] for a in A.values()]
    pool.run("transfer_cube", jobs, OUT / "stress_runs.jsonl", budget="stress", purpose="stage6 stress")


def report():
    from referee.stats import bootstrap_ci, paired

    rows = {}
    for l in (OUT / "stress_runs.jsonl").read_text().splitlines():
        r = json.loads(l)
        rows.setdefault(r["arm"], {})[json.dumps(rnd(r["pose"]))] = r
    out, L = {"note": "explicit poses, rng 20261125, reported and not judged. Poses in stress_run.py"}, ["| set | n | original | edit | gain (points) | 95% interval | fixed | broke | p | strict original | strict edit |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, ps in sets().items():
        keys = [json.dumps(rnd(p)) for p in ps]
        o = {str(i): bool(rows["original"][x]["benchmark_success"]) for i, x in enumerate(keys) if x in rows["original"]}
        e = {str(i): bool(rows["agent_edit"][x]["benchmark_success"]) for i, x in enumerate(keys) if x in rows["agent_edit"]}
        os_, es = ({str(i): bool(rows[a][x]["strict_success"]) for i, x in enumerate(keys) if x in rows[a]} for a in ("original", "agent_edit"))
        p, c, st = paired(o, e), bootstrap_ci(o, e, 10_000, 0), paired(os_, es)
        out[k] = {"n": p["n"], "original_success": p["orig_success"], "agent_edit_success": p["new_success"], "gain_points": p["gain_points"], "gain_ci95": c["gain_points"],
                  "fixed": p["fixed"], "broke": p["broke"], "p_mcnemar": p["p_mcnemar"], "strict_original": st["orig_success"], "strict_agent_edit": st["new_success"], "strict_gain_points": st["gain_points"]}
        L.append(f"| {k} | {p['n']} | {100 * p['orig_success']:.1f}% | {100 * p['new_success']:.1f}% | {p['gain_points']:+.1f} | {c['gain_points'][0]:+.1f} to {c['gain_points'][1]:+.1f} | {p['fixed']} | {p['broke']} | {p['p_mcnemar']:.2g} | {100 * st['orig_success']:.1f}% | {100 * st['new_success']:.1f}% |")
    (OUT / "stress.json").write_text(json.dumps(out, indent=1) + "\n")
    (OUT / "stress.md").write_text("\n".join(["# Stress checks, transfer_cube", "", "Explicit poses, reported and not judged. Intervals are 95% bootstrap intervals, descriptive."] + [""] + L) + "\n")
    print((OUT / "stress.md").read_text())


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
