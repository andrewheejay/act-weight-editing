from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import world


def main() -> int:
    from safetensors.numpy import save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    arm = next(x for x in manifest["arms"] if x["id"] == a.arm)
    kind = arm["role"] if arm["role"] in ("agent_edit", "autogreedy") else "original"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(world.tensors(kind, backbone=bool(arm.get("sim", {}).get("backbone"))), str(out / "model.safetensors"))
    (out / "arm.json").write_text(json.dumps({"arm": a.arm, "manifest": str(Path(a.manifest).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
