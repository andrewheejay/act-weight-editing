"""Run every arm of freeze/test_manifest.json on 3 design seeds and compare with the rows already in the repo.

python scratch/smoke_arms.py run       rollouts (agent budget), rows in experiments/transfer_cube/stage3/smoke_arms.jsonl
python scratch/smoke_arms.py compare   no rollouts: trajectory hashes against the rows of the earlier stages, stage3/smoke_compare.json

Seeds 20500 to 20502 (design). Arms whose recipe already ran on these seeds must give the same trajectory hash:
original (design_default), agent_edit (combo r4), autogreedy (the pair in stage3/autogreedy.jsonl), the baselines (baselines_full and
design_chunk50) and original_ens (design_ens). Dose, random and agent_edit_ens arms have no earlier rows on these seeds and are only run.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "transfer_cube"
SEEDS = [20500, 20501, 20502]
OUT = EXP / "stage3" / "smoke_arms.jsonl"
MANIFEST = EXP / "freeze" / "test_manifest.json"


def rows(p):
    p = Path(p)
    if not p.exists():
        p = Path(str(p) + ".gz")
    op = gzip.open if p.suffix == ".gz" else open
    return [json.loads(l) for l in op(p, "rt") if l.strip()]


def run():
    from awe import arms, pool

    m = json.loads(MANIFEST.read_text())
    jobs = [(arms.resolve(a["id"], MANIFEST), s, None) for a in m["arms"] for s in SEEDS]
    print(len(jobs), "jobs", flush=True)
    pool.run("transfer_cube", jobs, OUT, budget="agent", purpose="freeze_smoke")


# arm id of the manifest -> (file, arm id in that file, or None for the file's single arm)
REFS = {
    "original": ("stage1/design_default.jsonl", None),
    "agent_edit": ("stage3/combo.jsonl", "r4"),
    "autogreedy": ("stage3/autogreedy.jsonl", "ag_enc0.h1.x0_enc3.h4.x2"),
    "original_ens": ("stage1/design_ens.jsonl", None),
}


def compare():
    m = json.loads(MANIFEST.read_text())
    got = {}
    for r in rows(OUT):
        got.setdefault(r["arm"], {})[r["seed"]] = r
    refs = dict(REFS)
    for a in m["arms"]:
        if a["role"] == "baseline":
            refs[a["id"]] = ("stage1/design_chunk50.jsonl", None) if a["family"] == "shorter_chunk" else ("stage3/baselines_full.jsonl", a["setting"])
    out, bad = {}, []
    for a in m["arms"]:
        aid = a["id"]
        rec = got.get(aid, {})
        item = {"role": a["role"], "n_rows": len(rec), "benchmark_success": [rec[s]["benchmark_success"] for s in SEEDS if s in rec], "strict_success": [rec[s]["strict_success"] for s in SEEDS if s in rec]}
        if len(rec) != len(SEEDS):
            bad.append(f"{aid}: {len(rec)} of {len(SEEDS)} rows")
        if aid in refs:
            f, ref_arm = refs[aid]
            by = {r["seed"]: r for r in rows(EXP / f) if r["seed"] in SEEDS and (ref_arm is None or r["arm"] == ref_arm)}
            item["reference"] = [f, ref_arm]
            item["hash_match"] = {str(s): (rec.get(s, {}).get("traj_hash") == by[s]["traj_hash"]) if s in by else None for s in SEEDS}
            item["success_match"] = {str(s): (rec.get(s, {}).get("benchmark_success") == by[s]["benchmark_success"]) if s in by else None for s in SEEDS}
            if not all(v is True for v in item["hash_match"].values()):
                bad.append(f"{aid}: hashes {item['hash_match']}")
        out[aid] = item
    res = {"seeds": SEEDS, "arms": out, "problems": bad}
    (EXP / "stage3" / "smoke_compare.json").write_text(json.dumps(res, indent=1) + "\n")
    print(len(out), "arms;", "problems:", bad)


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
