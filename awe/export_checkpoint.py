"""Write an arm's edited checkpoint as a folder lerobot-eval can load. See referee/CONTRACT.md.

Only weight-edit ops are exported (an output-level recipe cannot be written into a checkpoint and raises).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def export(task, arm_id, manifest_path, out):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from . import edits
    from .arms import resolve
    from .paths import checkpoint_dir

    spec = resolve(arm_id, manifest_path)
    if spec.get("post"):
        raise SystemExit(f"arm {arm_id!r} has an output-level recipe, which a checkpoint cannot carry")
    src, out = checkpoint_dir(task), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():  # config, processors, and the README are copied as they are
        if f.name not in ("model.safetensors", "config.json") and f.is_file():
            shutil.copyfile(f, out / f.name)
    with safe_open(str(src / "model.safetensors"), "pt") as h:
        meta = h.metadata()
    sd = load_file(str(src / "model.safetensors"))
    names = list(sd)
    for op in spec["ops"]:
        for k in edits.touched(op):
            sd[k] = sd[k].clone()
    edits.apply_ops(sd, spec["ops"])
    save_file({n: sd[n].contiguous() for n in names}, str(out / "model.safetensors"), metadata=meta)
    cfg = json.loads((src / "config.json").read_text())
    coeff = spec["temporal_ensemble_coeff"]
    cfg["temporal_ensemble_coeff"] = coeff
    cfg["n_action_steps"] = 1 if coeff is not None else int(spec["n_action_steps"])
    (out / "config.json").write_text(json.dumps(cfg, indent=4) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="awe.export_checkpoint")
    ap.add_argument("--task", required=True, choices=["transfer_cube", "insertion"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    export(a.task, a.arm, a.manifest, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
