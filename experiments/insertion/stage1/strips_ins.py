"""Frame strips for insertion, SPEC.md Stage 1 step 6: 30 failures by kind (see analysis_ins.kind), proportional to their counts among
the strict failures of design_default, and 10 strict successes, drawn with rng 0. Run:  python experiments/insertion/stage1/strips_ins.py
The rollouts are repeats of design_default seeds (agent budget); every trajectory hash must match."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from awe import pool  # noqa: E402
from awe.arms import make_spec  # noqa: E402
from analysis_ins import S1, kind, records  # noqa: E402

KINDS = ["M", "G1", "G0", "A", "G2"]


def pick(n_fail=30, n_good=10):
    R = sorted(records("default"), key=lambda r: r["seed"])  # file order is completion order, so sort
    by = {k: [r for r in R if kind(r) == k] for k in KINDS + ["S"]}
    total = sum(len(by[k]) for k in KINDS)
    want = {k: max(1, round(n_fail * len(by[k]) / total)) for k in KINDS}
    while sum(want.values()) != n_fail:
        k = max(KINDS, key=lambda k: len(by[k]))
        want[k] += n_fail - sum(want.values())
    rng = np.random.default_rng(0)
    picked = []
    for k in KINDS:
        picked += [(k, by[k][i]["seed"]) for i in sorted(rng.choice(len(by[k]), want[k], replace=False))]
    picked += [("S", by["S"][i]["seed"]) for i in sorted(rng.choice(len(by["S"]), n_good, replace=False))]
    return picked, want


def main():
    d = S1 / "strips"
    d.mkdir(parents=True, exist_ok=True)
    picked, want = pick()
    kinds = {s: k for k, s in picked}
    ref = {r["seed"]: r["traj_hash"] for r in records("default")}
    info = {}

    def save(rec, frames):
        f = [Image.fromarray(x).resize((200, 150)) for x in frames]
        w = Image.new("RGB", (200 * len(f), 150))
        for i, x in enumerate(f):
            w.paste(x, (200 * i, 0))
        name = f"{kinds[rec['seed']]}_{rec['seed']}_class{rec['failure_class']}.jpg"
        w.save(d / name, quality=80)
        info[rec["seed"]] = {"kind": kinds[rec["seed"]], "file": name, "class": rec["failure_class"], "strict": rec["strict_success"],
                             "hash_matches_design_default": rec["traj_hash"] == ref[rec["seed"]]}

    out = S1 / "strips_runs.jsonl"
    spec = make_spec("original")
    pool.run("insertion", [(spec, s, None) for _, s in picked], out, budget="agent", purpose="stage1 strips", frames_cb=save, n_frames=9, workers=4)
    (S1 / "strips_index.json").write_text(json.dumps({str(k): v for k, v in sorted(info.items())}, indent=1))
    print("want", want, "strips", len(list(d.glob("*.jpg"))), "hash match", sum(v["hash_matches_design_default"] for v in info.values()), "of", len(info))


if __name__ == "__main__":
    main()
