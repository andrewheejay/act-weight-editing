"""Stage 0 runs: reproduction seeds, repeat runs, the sham edit and the renderer comparison.

python -m awe.stage0 repro --renderer egl_mesa --out FILE       500 reproduction seeds, arm original
python -m awe.stage0 repeat --name repeat_a                     100 seeds at the recorded renderer, arm original
python -m awe.stage0 sham --name sham                           the same 100 seeds, every head scaled by 1
"""
import argparse

from . import edits, pool
from .arms import make_spec
from .paths import exp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repro", "repeat", "sham"])
    ap.add_argument("--task", default="transfer_cube")
    ap.add_argument("--renderer")
    ap.add_argument("--out")
    ap.add_argument("--name")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    s0 = exp(a.task) / "stage0"
    if a.cmd == "repro":
        seeds, spec, out, purpose = range(1000, 1500), make_spec(), a.out or s0 / "harness_repro.jsonl", "reproduction"
    else:
        ops = edits.sham_ops() if a.cmd == "sham" else []
        spec = make_spec("sham" if a.cmd == "sham" else "original", ops)
        seeds = range(1000, 1000 + a.n)
        out = s0 / f"{a.name or a.cmd}.jsonl"
        purpose = f"stage0 {a.name or a.cmd}"
    pool.run(a.task, [(spec, s, None) for s in seeds], out, budget="setup", purpose=purpose, workers=a.workers, renderer=a.renderer)
