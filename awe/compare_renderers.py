"""Renderer comparison on the reproduction seeds, against the published per-seed file and the stock runs."""
import json
import sys
from pathlib import Path

from .paths import exp


def load(p):
    return {json.loads(l)["seed"]: json.loads(l) for l in open(p)}


def published(task="transfer_cube"):
    d = json.loads((exp(task) / "stage0" / "published_eval_info.json").read_text())
    return {e["seed"]: (bool(e["success"]), float(e["max_reward"])) for e in d["per_episode"]}


if __name__ == "__main__":
    s0 = exp("transfer_cube") / "stage0"
    pub = published()
    for name, p in (("egl_nvidia", s0 / "renderer_compare/gpu.jsonl"), ("egl_mesa", s0 / "harness_repro.jsonl")):
        r = load(p)
        seeds = sorted(set(r) & set(pub))
        suc = {s: r[s]["benchmark_success"] for s in seeds}
        agree = sum(suc[s] == pub[s][0] for s in seeds) / len(seeds)
        cls = sum(r[s]["max_reward"] == pub[s][1] for s in seeds) / len(seeds)
        from collections import Counter
        print(name, "n", len(seeds), "success %.1f%%" % (100 * sum(suc.values()) / len(seeds)), "published success agreement %.1f%%" % (100 * agree), "max_reward agreement %.1f%%" % (100 * cls), dict(Counter(r[s]["failure_class"] for s in seeds)))
    print("published success %.1f%%" % (100 * sum(v[0] for v in pub.values()) / len(pub)), "classes by max reward", {k: sum(1 for v in pub.values() if v[1] == k) for k in (0, 1, 2, 3, 4)})
