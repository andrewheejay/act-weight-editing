"""Design-seed comparison of combination arms with the rival (gain_both_nogrip_0.9). Screening output only, no rollouts."""
import json
from pathlib import Path
from awe.paths import exp
from awe.sweep_report import find, rows_of, refs, stats
from awe.combo import ALL
from referee.stats import mcnemar_exact

d = exp("transfer_cube")
base = {r["seed"]: r for r in rows_of(find(d / "stage1", "design_default.jsonl"))}
reach, lat = refs("transfer_cube")
arms = {}
for r in rows_of(d / "stage3" / "combo.jsonl.gz"):
    arms.setdefault(r["arm"], {})[r["seed"]] = r
riv = {r["seed"]: r for r in rows_of(d / "stage3" / "baselines_full.jsonl.gz") if r["arm"] == "gain_both_nogrip_0.9"}
for p in ("baselines_screen.jsonl.gz",):
    for r in rows_of(d / "stage3" / p):
        if r["arm"] == "gain_both_nogrip_0.9":
            riv[r["seed"]] = r
print("rival rows", len(riv))
nf = sum(not r["benchmark_success"] for r in base.values())
out = {}
for cid in ["s1", "r4", "r5", "r1", "r2", "r6", "c06", "c03", "r3", "s2", "c07", "c05"]:
    rows = arms[cid]
    s = stats(base, list(rows.values()), reach, lat)
    common = sorted(set(rows) & set(riv))
    e_only = sum(rows[x]["benchmark_success"] and not riv[x]["benchmark_success"] for x in common)
    r_only = sum(riv[x]["benchmark_success"] and not rows[x]["benchmark_success"] for x in common)
    print(f"{cid:4} n {s['n']} gain {100*s['gain']:+.1f} share {(s['fixed']-s['broken'])/nf:.3f} strict {100*s['strict_gain']:+.1f} fixed {s['fixed']} broken {s['broken']} vs_rival +{e_only} -{r_only} p {mcnemar_exact(e_only, r_only):.3f} heads {len(ALL[cid])}")
    out[cid] = {"gain": s["gain"], "strict_gain": s["strict_gain"], "fixed": s["fixed"], "broken": s["broken"], "share_removed": round((s["fixed"]-s["broken"])/nf, 4), "rival_paired": [e_only, r_only]}
print("rival", riv and stats(base, list(riv.values()), reach, lat)["gain"])
Path("scratch/cand_vs_rival.json").write_text(json.dumps(out, indent=1))
