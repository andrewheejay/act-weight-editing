"""Arm lookup. An arm is a recipe fully determined by files in the repo: weight ops plus an operating point."""
import json
from pathlib import Path


def make_spec(arm_id="original", ops=None, n_action_steps=100, temporal_ensemble_coeff=None, post=None):
    """post is an output-level recipe (awe/post.py), only for the baseline arms."""
    return {
        "id": arm_id,
        "ops": list(ops or []),
        "n_action_steps": int(n_action_steps),
        "temporal_ensemble_coeff": temporal_ensemble_coeff,
        "post": post,
    }


def from_manifest(manifest: dict, arm_id: str) -> dict:
    hits = [a for a in manifest["arms"] if a["id"] == arm_id]
    if len(hits) != 1:
        raise KeyError(f"arm {arm_id!r} appears {len(hits)} times in the manifest")
    a = hits[0]
    n = a.get("n_action_steps", manifest.get("operating_point", {}).get("n_action_steps", 100))
    return make_spec(arm_id, a.get("ops"), n, a.get("temporal_ensemble_coeff"), a.get("post"))


def resolve(arm_id: str, manifest_path=None) -> dict:
    if manifest_path:
        return from_manifest(json.loads(Path(manifest_path).read_text()), arm_id)
    if arm_id != "original":
        raise SystemExit(f"arm {arm_id!r} needs --manifest")
    return make_spec("original")
