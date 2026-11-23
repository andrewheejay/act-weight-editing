from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import world


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--workers")
    a = ap.parse_args()
    manifest = json.loads(Path(a.manifest).read_text()) if a.manifest else None
    if a.arm not in ("original", "sham") and "~" not in a.arm:
        if not manifest or not any(x["id"] == a.arm for x in manifest["arms"]):
            raise SystemExit(f"unknown arm {a.arm}")
        if any(x["id"] == a.arm and x.get("sim", {}).get("crash") for x in manifest["arms"]):
            raise SystemExit("this arm is set to crash")
    seeds = json.loads(Path(a.seeds_file).read_text())["seeds"]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines() if line.strip()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for s in seeds:
            if s not in done:
                f.write(json.dumps(world.rollout(a.task, a.arm, int(s), manifest)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
