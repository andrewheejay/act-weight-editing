"""A scripted stand-in for the agent, used only to test the referee and the loop.

Each call to step() looks at the referee's status and produces the files for the current stage,
using the fake harness in fake_awe. It follows the contract the way a careful agent would, and it
can also be told to misbehave so the tests can watch the referee catch it.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
WORDS = "The policy reaches short on cubes that start in the low strip of the square and the gripper closes beside the cube. "


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _jsonl(path: Path, rows: list[dict], gz: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r) + "\n" for r in rows)
    if gz:
        with gzip.open(str(path) + ".gz", "wt") as f:
            f.write(data)
    else:
        path.write_text(data)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=sim-agent", "-c", "user.email=sim@example.com", *args], check=True, capture_output=True)


def _ledger(root: Path, name: str, rows: list[dict], task: str, budget: str, purpose: str, candidate: str | None = None) -> None:
    out = []
    for r in rows:
        line = {"task": task, "seed": r["seed"], "arm": r["arm"], "budget": budget, "purpose": purpose}
        if candidate:
            line["candidate"] = candidate
        out.append(line)
    _jsonl(root / "ledger" / f"{name}.jsonl", out, gz=len(out) > 5000)


def stage0(root: Path, T: str, world) -> None:
    from safetensors.numpy import save_file

    E = root / "experiments" / T / "stage0"
    ck = root / "checkpoints" / T
    ck.mkdir(parents=True, exist_ok=True)
    save_file(world.tensors("original"), str(ck / "model.safetensors"))
    sys.path.insert(0, str(root))
    from referee.common import load_config, sha256_file

    cfg = load_config()
    _w(E / "checkpoint.json", json.dumps({"dir": f"checkpoints/{T}", "model_sha256": sha256_file(ck / "model.safetensors"), "source": cfg["tasks"][T]["hub_repo"]}))
    cmd = [sys.executable, "-m", "awe.fake_stock", "--policy={policy_path}", "--task={env_task}", "--n={n_episodes}", "--seed={seed}", "--out={output_dir}"]
    _w(E / "env.json", json.dumps({"lerobot": "0.7.0", "gym_aloha": "0.1.5", "mujoco": "3.8.1", "torch": "2.15.0", "python": "3.12", "renderer": "egl", "rollouts_per_second": 100.0, "stock_seconds_per_episode": 0.001, "stock_eval_cmd": cmd}))
    if not (root / "tests" / "test_harness.py").exists():
        names = ["test_fast_harness_matches_stock", "test_head_scaling_exact", "test_sham_edit_identical", "test_repeat_run_identical", "test_seed_lock"]
        _w(root / "tests" / "test_harness.py", "".join(f"def {n}():\n    assert True\n\n\n" for n in names))
    layout = "lerobot07" if T == "transfer_cube" else "per_task"
    subprocess.run([sys.executable, "-m", "awe.fake_stock", f"--policy={ck}", f"--task={cfg['tasks'][T]['env_task']}", "--n=500", "--seed=1000", f"--out={E / 'stock_eval'}", f"--layout={layout}"], cwd=str(root), check=True)
    info = json.loads((E / "stock_eval" / "eval_info.json").read_text())
    _w(E / "published_eval_info.json", json.dumps({"per_episode": [{"seed": 1000 + i, "success": world.rollout(T, "original", 1000 + i, None)["benchmark_success"] ^ (i % 25 == 0)} for i in range(500)], "aggregated": {}}))
    assert info
    rows = [world.rollout(T, "original", s, None) for s in range(1000, 1500)]
    _jsonl(E / "harness_repro.jsonl", rows)
    rep = [world.rollout(T, "original", s, None) for s in range(1000, 1100)]
    _jsonl(E / "repeat_a.jsonl", rep)
    _jsonl(E / "repeat_b.jsonl", rep)
    _jsonl(E / "sham.jsonl", [{**world.rollout(T, "sham", s, None)} for s in range(1000, 1100)])
    _w(E / "renderer.json", json.dumps({"chosen": "egl"}))
    _ledger(root, f"{T}-stage0", rows + rep + rep + rep, T, "setup", "reproduction")


def stage1(root: Path, T: str, world) -> None:
    E = root / "experiments" / T
    seeds = range(20000, 21000)
    runs = {"design_default": "original", "design_chunk50": "c50~0.03~0.01", "design_chunk25": "c25~0.02~0.02"}
    led = []
    for name, arm in runs.items():
        rows = [world.rollout(T, arm, s, None) for s in seeds]
        _jsonl(E / "stage1" / f"{name}.jsonl", rows)
        led += rows
    rows = [world.rollout(T, "ens~0.05~0.01", s, None) for s in range(20000, 20200)]
    _jsonl(E / "stage1" / "design_ens.jsonl", rows)
    _ledger(root, f"{T}-stage1", led + rows, T, "agent", "diagnosis")
    for n in ("failure_map.png", "reach_error.png"):
        (E / "stage1" / n).write_bytes(PNG)
    from referee import gates
    from referee.common import load_config

    exp, _ = gates.expected_decisions(root, load_config(), T)
    _w(E / "stage1" / "decisions.json", json.dumps({"primary_metric": exp["primary_metric"], "operating_point": {"n_action_steps": exp["n_action_steps"]}, "proceed": exp["proceed"], "target_failure_class": "1", "tracking_number": "reach_error"}))
    _w(E / "diagnosis.md", "# Diagnosis\n\n" + "".join(f"## {h}\n\n{WORDS * 4}\n\n" for h in ("Target failure", "Tracking number", "Near-miss evidence", "Primary metric", "Operating point")))


def stage2(root: Path, T: str, world) -> None:
    E = root / "experiments" / T / "stage2"
    _w(E / "atlas" / "heads.json", json.dumps({"heads": [{"head": i, "overlap": world.h01(T, i, "ov")} for i in range(48)]}))
    rows = [world.rollout(T, "original", s, None) for s in range(20000, 20500)]
    arms = [f"h{i}x{k}~{0.02 + 0.2 * world.h01(T, i, k, 'f'):.3f}~0.02" for i in range(48) for k in (0, 2)]
    led = []
    for arm in arms:
        r = [world.rollout(T, arm, s, None) for s in range(20000, 20500)]
        rows += r
        led += r
    _jsonl(E / "sweep.jsonl", rows, gz=True)
    half = []
    for arm in arms[:15]:
        half += [world.rollout(T, arm, s, None) for s in range(20500, 21000)]
    _w(E / "split_half.json", json.dumps({"rechecked": arms[:15], "kept": arms[:6]}))
    _ledger(root, f"{T}-stage2", led + half, T, "agent", "sweep")


def stage34(root: Path, T: str, world, bad_backbone: bool) -> None:
    E = root / "experiments" / T
    F = E / "freeze"
    strong = T == "transfer_cube"
    fix = 0.40 if strong else 0.02
    cands = {"c1": f"c1~{fix}~0.01", "c2": "c2~0.10~0.02", "c3": "c3~0.05~0.01"}
    _w(E / "hypotheses.md", "# Hypotheses\n\n" + "".join(f"- {c}. Turning up the heads that read the cube in the weak zone should shorten the reach error.\n" for c in cands))
    _w(E / "stage3" / "dose_response.json", json.dumps({"scales": [0.5, 1.5, 2, 3, 4], "reach_error": [1.2, 0.9, 0.7, 0.5, 0.45]}))
    vs = range(30000, 30500)
    orig = [world.rollout(T, "original", s, None) for s in vs]
    _jsonl(E / "stage3" / "validation" / "original.jsonl", orig)
    _ledger(root, f"{T}-val-original", orig, T, "agent", "validation", "original")
    for c, arm in cands.items():
        rows = [world.rollout(T, arm, s, None) for s in vs]
        _jsonl(E / "stage3" / "validation" / f"{c}.jsonl", rows)
        _ledger(root, f"{T}-val-{c}", rows, T, "agent", "validation", c)
    fams = ["constant_offset", "deviation_gain", "shorter_chunk", "simplest_fix"]
    for f in fams:
        rows = [world.rollout(T, f"{f}~0.10~0.02", s, None) for s in vs]
        _ledger(root, f"{T}-val-base-{f}", rows, T, "baselines", "validation", f"base_{f}")
    _w(F / "baselines.json", json.dumps({"baselines": [{"family": f, "setting": "best on design seeds"} for f in fams]}))
    _w(F / "autogreedy.json", json.dumps({"built_by": "awe/autogreedy.py", "ops": [{"op": "scale_head", "block": "model.encoder.layers.1.self_attn", "head": 1, "scale": 2}]}))
    _w(F / "random_family.json", json.dumps({"rule": "same count, same blocks, same scales, different heads", "rng_seed": 12345, "n_available": 4000}))
    _w(F / "analysis.py", "print('tables and figures')\n")
    dec = json.loads((E / "stage1" / "decisions.json").read_text())
    sys.path.insert(0, str(root))
    from referee.common import sha256_file

    arms = [
        {"id": "original", "role": "original"},
        {"id": "agent_edit", "role": "agent_edit", "sim": {"fix": fix, "brk": 0.01, "backbone": bad_backbone}},
        {"id": "autogreedy", "role": "autogreedy", "sim": {"fix": fix * 0.5, "brk": 0.01}},
    ]
    arms += [{"id": f"random_{i:02d}", "role": "random", "sim": {"fix": 0.06 * world.h01(T, "rnd", i), "brk": 0.02}} for i in range(40)]
    arms += [{"id": f"base_{f}", "role": "baseline", "family": f, "sim": {"fix": 0.10, "brk": 0.02}} for f in fams]
    arms += [{"id": f"dose_x{s}", "role": "dose", "sim": {"fix": fix * k, "brk": 0.01}} for s, k in (("0.5", 0.2), ("1.5", 0.6), ("3", 1.0), ("4", 0.8))]
    arms += [{"id": "original_ens", "role": "ensembling_original", "n_seeds": 200, "sim": {"fix": 0.05, "brk": 0.01}}, {"id": "agent_edit_ens", "role": "ensembling_edit", "n_seeds": 200, "sim": {"fix": fix, "brk": 0.01}}]
    manifest = {"task": T, "primary_metric": dec["primary_metric"], "operating_point": dec["operating_point"], "original_checkpoint_sha256": sha256_file(root / "checkpoints" / T / "model.safetensors"), "arms": arms}
    _w(F / "test_manifest.json", json.dumps(manifest, indent=1))
    tmp = root / ".awe" / "sim-export"
    subprocess.run([sys.executable, "-m", "awe.export_checkpoint", "--task", T, "--arm", "agent_edit", "--manifest", str(F / "test_manifest.json"), "--out", str(tmp)], cwd=str(root), check=True)
    _w(F / "edit.json", json.dumps({"task": T, "candidate": "c1", "family": "head_scaling", "ops": [{"op": "scale_head", "block": "model.encoder.layers.0.self_attn", "head": 0, "scale": 2}], "free_numbers": 0, "hypothesis": "The heads that read the cube in the weak zone under-react, so doubling one of them should shorten the reach error there.", "edited_checkpoint_sha256": sha256_file(tmp / "model.safetensors")}))
    _w(F / "REQUEST.json", json.dumps({"task": T, "note": "ready"}))


def stage5(root: Path, T: str) -> None:
    E = root / "experiments" / T
    v = (E / "test" / "VERDICT.md").read_text()
    _w(E / "report" / "report.md", "# Report\n\n" + v + "\n\n" + WORDS * 20)
    for n in ("before_after.png", "random_hist.png", "dose.png"):
        (E / "report" / n).write_bytes(PNG)


def step(root: Path) -> str:
    root = Path(root).resolve()
    for p in (str(root), str(Path(__file__).resolve().parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from awe import world
    from referee import status as status_mod
    from referee.common import load_config

    cfg = load_config()
    st = status_mod.compute(root, cfg)
    if st["state"] != "AGENT_TURN":
        return st["state"]
    T, stage = st.get("task"), st["stage"]
    ids = [t["id"] for t in st["open_tasks"]]
    did = f"stage {stage}"
    if any(i.endswith(".review") for i in ids):
        e = root / "experiments" / T / "freeze"
        gate = "g1" if "g1.review" in ids else f"g2.{len(list(e.glob('REJECTED-*.md'))) + 1}"
        rs = status_mod.review_state(root, cfg, T, gate)
        _w(root / rs["response_file"], "The reviewer asked for evidence across many seeds. " + WORDS * 2)
        did = f"answered the {gate} review"
    elif stage == 0:
        stage0(root, T, world)
    elif stage == 1:
        stage1(root, T, world)
    elif stage == 2:
        stage2(root, T, world)
    elif stage in (3, 4):
        rejected = list((root / "experiments" / T / "freeze").glob("REJECTED-*.md")) if (root / "experiments" / T / "freeze").is_dir() else []
        bad = os.environ.get("STUB_BAD_FIRST_FREEZE") == "1" and not rejected and T == "transfer_cube"
        stage34(root, T, world, bad_backbone=bad)
    elif stage == 5:
        stage5(root, T)
    elif stage == 6:
        _w(root / "experiments" / T / "stage6" / "stress.json", json.dumps({"band_3cm": {}, "rotation_15": {}, "rotation_30": {}}))
    elif stage == 7:
        _w(root / "publish" / "post.md", "# Post\n\n" + WORDS * 30)
        _w(root / "publish" / "atlas" / "index.html", "<html><body>" + "<p>head</p>" * 300 + "</body></html>")
        _w(root / "publish" / "CHECKLIST.md", "# Checklist\n\n" + "- Upload the edited checkpoint to the hub and attach the session logs.\n" * 5)
    with open(root / "PROGRESS.md", "a") as f:
        f.write(f"\n## sim entry\n\n- did {did} for {T}\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", f"agent: {did} for {T}")
    return f"did {did}"


if __name__ == "__main__":
    print(step(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
