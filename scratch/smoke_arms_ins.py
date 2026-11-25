"""Insertion: run every arm of freeze/test_manifest.json on a few design seeds and compare with the rows already in the repo.

python scratch/smoke_arms_ins.py run       rollouts (agent budget, purpose freeze_smoke), rows in experiments/insertion/stage3/smoke_arms.jsonl
python scratch/smoke_arms_ins.py compare   no rollouts: trajectory hashes against earlier rows, stage3/smoke_compare.json

Seeds 20500 to 20502 (design) for every arm, plus 20000 to 20002 for the two ensembling arms (the only earlier ensembling rows are on 20000 to 20199).
References: original (design_default), agent_edit (combo_AB r1), autogreedy (its pair in stage3/autogreedy.jsonl, and combo_A c06 for the same seeds),
the baselines (baselines_full, design_chunk50 for the chunk arm), original_ens (design_ens on 20000 to 20002). Dose, random and agent_edit_ens
have no earlier rows and are only run. Same idea as scratch/smoke_arms.py.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "insertion"
SEEDS = [20500, 20501, 20502]
ENS_SEEDS = [20000, 20001, 20002]
OUT = EXP / "stage3" / "smoke_arms.jsonl"
MANIFEST = EXP / "freeze" / "test_manifest.json"


def rows(p):
    p = Path(p)
    if not p.exists():
        p = Path(str(p) + ".gz")
    op = gzip.open if p.suffix == ".gz" else open
    return [json.loads(l) for l in op(p, "rt") if l.strip()]


def seeds_of(a):
    return SEEDS + (ENS_SEEDS if a["role"].startswith("ensembling") else [])


def run():
    from awe import arms, pool

    m = json.loads(MANIFEST.read_text())
    jobs = [(arms.resolve(a["id"], MANIFEST), s, None) for a in m["arms"] for s in seeds_of(a)]
    print(len(jobs), "jobs", flush=True)
    pool.run("insertion", jobs, OUT, budget="agent", purpose="freeze_smoke")


def ref_rows(path, arm, seeds):
    return {r["seed"]: r for r in rows(EXP / path) if r["seed"] in seeds and (arm is None or r["arm"] == arm)}


def compare():
    m = json.loads(MANIFEST.read_text())
    got = {}
    for r in rows(OUT):
        got.setdefault(r["arm"], {})[r["seed"]] = r
    out, bad, checked = {}, [], 0
    for a in m["arms"]:
        aid, want = a["id"], seeds_of(a)
        rec = got.get(aid, {})
        item = {"role": a["role"], "n_rows": len(rec), "wanted": len(want),
                "benchmark_success": [rec[s]["benchmark_success"] for s in want if s in rec], "strict_success": [rec[s]["strict_success"] for s in want if s in rec]}
        if set(rec) < set(want):
            bad.append(f"{aid}: {len(rec)} of {len(want)} rows")
        refs = []
        if a["role"] == "original":
            refs = [("stage1/design_default.jsonl", None, SEEDS)]
        elif a["role"] == "agent_edit":
            refs = [("stage3/combo_AB.jsonl", "r1", SEEDS)]
        elif a["role"] == "autogreedy":
            refs = [("stage3/autogreedy.jsonl", "ag_cross.h0.x0_enc3.h3.x0", SEEDS), ("stage3/combo_A.jsonl", "c06", SEEDS)]
        elif a["role"] == "baseline":
            refs = [("stage1/design_chunk50.jsonl", None, SEEDS)] if a["family"] == "shorter_chunk" else [("stage3/baselines_full.jsonl", a["setting"], SEEDS)]
        elif a["role"] == "ensembling_original":
            refs = [("stage1/design_ens.jsonl", None, ENS_SEEDS)]
        item["references"] = []
        for f, arm, seeds in refs:
            by = ref_rows(f, arm, seeds)
            hm = {str(s): (rec.get(s, {}).get("traj_hash") == by[s]["traj_hash"]) if s in by else None for s in seeds}
            item["references"].append({"file": f, "arm": arm, "hash_match": hm})
            checked += 1
            if not all(v is True for v in hm.values()):
                bad.append(f"{aid} vs {f} {arm}: {hm}")
        out[aid] = item
    res = {"seeds": SEEDS, "ens_seeds": ENS_SEEDS, "arms": out, "references_checked": checked, "problems": bad}
    (EXP / "stage3" / "smoke_compare.json").write_text(json.dumps(res, indent=1) + "\n")
    print(len(out), "arms;", checked, "references checked; problems:", bad)


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
