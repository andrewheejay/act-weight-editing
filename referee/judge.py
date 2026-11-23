"""The referee's own steps. No LLM runs here.

    python -m referee.judge step          do whatever the status says the referee should do next
    python -m referee.judge step --task T --action confirm_g0|stamp_g1|freeze|test

confirm_g0   replay a random sample of rollouts and run the tests, to confirm Gate 0
stamp_g1     record Gate 1, which fixes the primary score and the operating point
freeze       check the freeze package, tag it, or reject it with reasons
test         publish the commitment, draw the test seeds from the public beacon, run every arm, and
             write the verdict

Every step is safe to run again after a crash. It picks up from the files that already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import beacon, gates, stats
from . import status as status_mod
from .common import (
    Check,
    all_ok,
    by_seed,
    current_genesis,
    exp_dir,
    failing,
    find_jsonl,
    fingerprint,
    git,
    head_commit,
    load_config,
    now_iso,
    read_json,
    read_jsonl,
    referee_commit,
    repo_root,
    scratch,
    sha256_file,
    stock_per_episode,
    tag_exists,
    tracked_dirty,
    write_json,
)

DRY = os.environ.get("AWE_DRY_RUN") == "1"


def log(msg: str) -> None:
    print(f"[referee {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def py() -> str:
    return os.environ.get("AWE_PYTHON") or sys.executable


_last_beat = 0.0


def heartbeat(root: Path) -> None:
    global _last_beat
    (scratch(root) / "heartbeat").write_text(now_iso())
    url = os.environ.get("AWE_HEALTHCHECK_URL")
    if url and time.time() - _last_beat > 300:
        _last_beat = time.time()
        try:
            urllib.request.urlopen(url, timeout=10).read()
        except Exception:
            pass


def notify(title: str, message: str) -> None:
    cmd = os.environ.get("AWE_NOTIFY_CMD")
    log(f"{title}. {message}")
    if cmd:
        try:
            subprocess.run(["bash", cmd, title, message], timeout=30, capture_output=True)
        except Exception:
            pass


def sh(cmd: list[str], cwd: Path | None = None, env: dict | None = None, timeout: float | None = None) -> tuple[int, str]:
    e = dict(os.environ)
    bindir = str(Path(py()).parent)
    e["PATH"] = bindir + os.pathsep + e.get("PATH", "")
    if env:
        e.update(env)
    try:
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=e, capture_output=True, text=True, timeout=timeout)
        return r.returncode, ((r.stdout or "") + (r.stderr or ""))[-4000:]
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout} s"
    except FileNotFoundError as ex:
        return 127, str(ex)


def run_env(root: Path, draw: Path | None = None) -> dict:
    e = {
        "AWE_CHECKPOINTS": str(root / "checkpoints"),
        "AWE_CACHE": str(scratch(root) / "cache"),
        "AWE_REFEREE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if draw:
        e["AWE_JUDGE_DRAW"] = str(draw)
    return e


def run_arm(root: Path, cwd: Path, task: str, arm: str, seeds: list[int], out: Path, manifest: Path | None = None, draw: Path | None = None, timeout: float = 6 * 3600) -> tuple[int, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    sf = out.with_suffix(".seeds.json")
    write_json(sf, {"seeds": seeds})
    cmd = [py(), "-m", "awe.run_arm", "--task", task, "--arm", arm, "--seeds-file", str(sf), "--out", str(out), "--workers", str(os.environ.get("AWE_WORKERS") or os.cpu_count() or 1)]
    if manifest:
        cmd += ["--manifest", str(manifest)]
    rc, tail = sh(cmd, cwd=cwd, env=run_env(root, draw), timeout=timeout)
    sf.unlink(missing_ok=True)
    return rc, tail


def arm_rows(out: Path, seeds: list[int]) -> dict[int, dict] | None:
    """Rows for exactly these seeds, or None if any is missing."""
    if find_jsonl(out) is None:
        return None
    try:
        rows = {int(r["seed"]): r for r in read_jsonl(out) if r.get("seed") is not None}
    except (ValueError, KeyError):
        return None
    return {s: rows[s] for s in seeds} if all(s in rows for s in seeds) else None


def make_worktree(root: Path, ref: str, name: str) -> Path:
    wt = scratch(root) / "worktrees" / name
    if wt.exists():
        git(root, "worktree", "remove", "--force", str(wt))
        shutil.rmtree(wt, ignore_errors=True)
    wt.parent.mkdir(parents=True, exist_ok=True)
    git(root, "worktree", "prune")
    git(root, "worktree", "add", "--detach", str(wt), ref, check=True)
    return wt


def drop_worktree(root: Path, wt: Path) -> None:
    git(root, "worktree", "remove", "--force", str(wt))
    shutil.rmtree(wt, ignore_errors=True)


def replay_audit(root: Path, cwd: Path, cfg: dict, task: str, reference: Path, rng: list[int], label: str, n: int) -> Check:
    """Re-run a random sample of seeds with the original policy and compare with a file the agent wrote."""
    try:
        ref = {int(r["seed"]): r for r in read_jsonl(reference) if r.get("seed") is not None}
    except (FileNotFoundError, ValueError) as e:
        return Check(f"audit.{label}", False, str(e))
    pool = [s for s in range(rng[0], rng[1] + 1) if s in ref]
    if len(pool) < n:
        return Check(f"audit.{label}", False, f"only {len(pool)} seeds available to audit")
    seeds = sorted(random.SystemRandom().sample(pool, n))
    out = scratch(root) / "audit" / f"{task}-{label}.jsonl"
    out.unlink(missing_ok=True)
    rc, tail = run_arm(root, cwd, task, "original", seeds, out, timeout=3600)
    rows = arm_rows(out, seeds)
    if rc != 0 or rows is None:
        return Check(f"audit.{label}", False, f"python -m awe.run_arm failed (exit {rc}). {tail[-600:]}")
    keys = ("benchmark_success", "strict_success", "traj_hash")
    diff = [s for s in seeds if any(rows[s].get(k) != ref[s].get(k) for k in keys if k in ref[s])]
    return Check(f"audit.{label}", not diff, f"replayed {n} random seeds from {reference.name}, {len(diff)} differ {diff[:6] if diff else ''}")


def run_pytest(root: Path, cwd: Path) -> Check:
    rc, tail = sh([py(), "-m", "pytest", "tests", "-q", "-x", "-p", "no:cacheprovider"], cwd=cwd, env=run_env(root), timeout=3600)
    return Check("tests.pass", rc == 0, "" if rc == 0 else tail[-800:])


def stock_eval(root: Path, cwd: Path, task: str, cfg: dict, policy_path: Path, seed: int, n: int, out_dir: Path, timeout: float, draw: Path | None = None) -> tuple[dict[int, bool] | None, str]:
    env = read_json(cwd / "experiments" / task / "stage0" / "env.json") or {}
    tpl = env.get("stock_eval_cmd")
    if not isinstance(tpl, list):
        return None, "no stock_eval_cmd in stage0/env.json"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(x).format(policy_path=str(policy_path), env_task=cfg["tasks"][task]["env_task"], n_episodes=n, seed=seed, output_dir=str(out_dir)) for x in tpl]
    rc, tail = sh(cmd, cwd=cwd, env=run_env(root, draw), timeout=timeout)
    info = read_json(out_dir / "eval_info.json")
    if rc != 0 or not info:
        return None, f"stock evaluator failed (exit {rc}). {tail[-600:]}"
    rows = stock_per_episode(info, start_seed=seed)
    return {int(r["seed"]): bool(r["success"]) for r in rows if r.get("seed") is not None}, ""


# ------------------------------------------------------------------ confirm_g0


def confirm_g0(root: Path, cfg: dict, task: str) -> int:
    e = exp_dir(root, task)
    out_path = root / "reviews" / f"{task}-g0-deep.json"
    prev = read_json(out_path) or {}
    fp_data = fingerprint(gates.stage0_files(root, task))
    fp_code = gates.code_fingerprint(root)
    checks: list[Check] = []

    ck = read_json(e / "stage0" / "checkpoint.json") or {}
    p = gates.checkpoint_path(root, ck)
    if p.exists():
        got = sha256_file(p)
        checks.append(Check("g0.checkpoint_sha", got == ck.get("model_sha256"), f"checkpoint hash {got[:12]}, recorded {str(ck.get('model_sha256'))[:12]}"))
    else:
        checks.append(Check("g0.checkpoint_sha", False, f"missing {p}"))

    checks.append(run_pytest(root, root))
    n = cfg["gates"]["audit_seeds"]
    checks.append(replay_audit(root, root, cfg, task, e / "stage0" / "harness_repro.jsonl", cfg["seed_sets"]["reproduction"], "reproduction", n))

    first_time = not (prev.get("ok") and prev.get("fingerprint_data") == fp_data and prev.get("stock_audit"))
    stock_note = prev.get("stock_audit")
    if first_time and os.environ.get("AWE_STOCK_CROSSCHECK", "1") != "0":
        rng = cfg["seed_sets"]["reproduction"]
        s0 = random.SystemRandom().randint(rng[0], rng[1] - 4)
        table, err = stock_eval(root, root, task, cfg, p.parent, s0, 5, scratch(root) / "audit" / f"{task}-stock", timeout=3600)
        if table is None:
            checks.append(Check("g0.stock_cmd_runs", False, err))
        else:
            mine = {}
            try:
                mine = gates._stock_table(root, cfg, task)[0] or {}
            except Exception:
                pass
            same = sum(1 for s, v in table.items() if mine.get(s) == v)
            stock_note = f"re-ran lerobot-eval on seeds {s0} to {s0 + 4}. {same} of {len(table)} match the Stage 0 run"
            # the Stage 0 file carries no seeds, so this re-run is what shows its seeds were read correctly
            checks.append(Check("g0.stock_cmd_runs", len(table) == 5 and same >= 4, stock_note))

    ok = all_ok(checks)
    write_json(out_path, {
        "ok": ok,
        "task": task,
        "time": now_iso(),
        "commit": head_commit(root),
        "fingerprint_data": fp_data,
        "fingerprint_code": fp_code,
        "stock_audit": stock_note,
        "checks": [c.as_dict() for c in checks],
        "failing": [f"{c.name}: {c.detail}" for c in failing(checks)],
    })
    referee_commit(root, [str(out_path.relative_to(root))], f"referee: Gate 0 replay check for {task} ({'pass' if ok else 'fail'})")
    notify(f"Gate 0 {'passed' if ok else 'failed'} ({task})", "; ".join(f"{c.name} {c.detail}" for c in checks)[:500])
    return 0


# ------------------------------------------------------------------ stamp_g1


def stamp_g1(root: Path, cfg: dict, task: str) -> int:
    e = exp_dir(root, task)
    rs = status_mod.review_state(root, cfg, task, "g1")
    if rs["kind"] != "settled":
        log("Gate 1 review is not settled yet")
        return 1
    exp, _ = gates.expected_decisions(root, cfg, task)
    out = root / "reviews" / f"{task}-g1-passed.json"
    write_json(out, {
        "task": task,
        "time": now_iso(),
        "commit": head_commit(root),
        "decisions_sha256": sha256_file(e / "stage1" / "decisions.json"),
        "diagnosis_sha256": sha256_file(e / "diagnosis.md"),
        "recomputed": exp,
        "review_rounds": rs["round"],
        "review_unresolved": rs["unresolved"],
        "review_not_independent": any(r["not_reviewed"] for r in status_mod.reviews_summary(root, task) if r["gate"] == "g1"),
    })
    referee_commit(root, [str(out.relative_to(root))], f"referee: Gate 1 recorded for {task}")
    d = read_json(e / "stage1" / "decisions.json") or {}
    notify(f"Gate 1 passed ({task})", f"primary score {d.get('primary_metric')}, operating point {(d.get('operating_point') or {}).get('n_action_steps')} steps, proceed {d.get('proceed')}")
    return 0


# ------------------------------------------------------------------ freeze


def tensor_diff(orig: Path, new: Path, cfg: dict) -> tuple[list[Check], dict]:
    try:
        import numpy as np
        from safetensors.numpy import load_file
    except ImportError as ex:
        return [Check("edit.tensor_diff", False, f"numpy and safetensors are needed for this check ({ex})")], {}
    a, b = load_file(str(orig)), load_file(str(new))
    lim = cfg["edit_limits"]
    structure = sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k].shape != b[k].shape)
    changed = {}
    for k in sorted(set(a) & set(b)):
        if a[k].shape == b[k].shape and not np.array_equal(a[k], b[k]):
            changed[k] = int((a[k] != b[k]).sum())
    unused = tuple(lim["unused_prefixes"])
    active_total = int(sum(v.size for k, v in a.items() if not k.startswith(unused)))
    forbidden = [k for k in changed if k.startswith(tuple(lim["forbidden_prefixes"]))]
    inert = tuple(lim["output_level_prefixes"]) + unused
    real = [k for k in changed if not k.startswith(inert)]
    info = {
        "changed_keys": changed,
        "weights_changed": int(sum(changed.values())),
        "inference_weights": active_total,
        "share_of_inference_weights": (sum(changed.values()) / active_total) if active_total else None,
    }
    checks = [
        Check("edit.same_architecture", not structure, "tensors added, removed or reshaped: " + ", ".join(structure[:6]) if structure else ""),
        Check("edit.changes_something", bool(changed), "the edited checkpoint is identical to the original" if not changed else f"{info['weights_changed']:,} weights changed in {len(changed)} tensors"),
        Check("edit.no_backbone", not forbidden, "the vision backbone was edited: " + ", ".join(forbidden[:4]) if forbidden else ""),
        Check("edit.not_output_level", bool(real) or not changed, "the edit only touches output-level or unused tensors, so it belongs with the simple baselines" if changed and not real else ""),
    ]
    return checks, info


def export_checkpoint(root: Path, cwd: Path, task: str, arm: str, manifest: Path, out: Path) -> tuple[Path | None, str]:
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    rc, tail = sh([py(), "-m", "awe.export_checkpoint", "--task", task, "--arm", arm, "--manifest", str(manifest), "--out", str(out)], cwd=cwd, env=run_env(root), timeout=1800)
    f = out / "model.safetensors"
    if rc != 0 or not f.exists():
        return None, f"python -m awe.export_checkpoint failed for {arm} (exit {rc}). {tail[-600:]}"
    return f, ""


def freeze(root: Path, cfg: dict, task: str) -> int:
    e = exp_dir(root, task)
    fz = e / "freeze"
    tag = status_mod.freeze_tag(task)
    k = len(list(fz.glob("REJECTED-*.md"))) + 1
    checks: list[Check] = []

    dirty = tracked_dirty(root)
    checks.append(Check("freeze.clean_tree", not dirty, "uncommitted changes: " + ", ".join(dirty[:6]) if dirty else ""))
    # a tag left behind by an interrupted run is fine, as long as it points at this very commit
    tag_ok = not tag_exists(root, tag) or git(root, "rev-parse", f"{tag}^{{commit}}")[1] == head_commit(root)
    checks.append(Check("freeze.tag_free", tag_ok, f"tag {tag} already exists on another commit"))
    plan = status_mod.load_plan(root)
    for t in plan["per_task"]:
        if t["stage"] in (0, 1, 2, 3, 4):
            checks.extend(c for c in gates.run_task(t["id"], root, cfg, task) if not c.ok and not c.soft)

    manifest = read_json(fz / "test_manifest.json") or {}
    ck = read_json(e / "stage0" / "checkpoint.json") or {}
    orig_file = gates.checkpoint_path(root, ck)
    info: dict = {}
    if all_ok(checks) and orig_file.exists():
        got = sha256_file(orig_file)
        checks.append(Check("freeze.original_checkpoint", got == ck.get("model_sha256") == manifest.get("original_checkpoint_sha256"), "the original checkpoint hash must match Stage 0 and the manifest"))
    elif not orig_file.exists():
        checks.append(Check("freeze.original_checkpoint", False, f"missing {orig_file}"))

    if all_ok(checks):
        commit = head_commit(root)
        wt = make_worktree(root, commit, f"freezecheck-{task}")
        try:
            wm = wt / "experiments" / task / "freeze" / "test_manifest.json"
            checks.append(run_pytest(root, wt))
            n = cfg["gates"]["audit_seeds"]
            checks.append(replay_audit(root, wt, cfg, task, e / "stage0" / "harness_repro.jsonl", cfg["seed_sets"]["reproduction"], "reproduction_at_freeze", n))
            checks.append(replay_audit(root, wt, cfg, task, e / "stage1" / "design_default.jsonl", cfg["seed_sets"]["design"], "design_at_freeze", n))

            edit = read_json(fz / "edit.json") or {}
            for arm in ("agent_edit", "autogreedy"):
                arm_id = next((a["id"] for a in manifest.get("arms", []) if a.get("role") == arm), None)
                f, err = export_checkpoint(root, wt, task, arm_id, wm, scratch(root) / "export" / f"{task}-{arm}")
                if f is None:
                    checks.append(Check(f"freeze.export.{arm}", False, err))
                    continue
                dchecks, dinfo = tensor_diff(orig_file, f, cfg)
                if arm == "agent_edit":
                    checks.extend(dchecks)
                    info = dinfo
                    sha = sha256_file(f)
                    info["edited_checkpoint_sha256"] = sha
                    checks.append(Check("freeze.edit_sha", sha == edit.get("edited_checkpoint_sha256"), f"exported checkpoint hash {sha[:12]}, edit.json says {str(edit.get('edited_checkpoint_sha256'))[:12]}"))
                else:
                    checks.extend(c for c in dchecks if c.name in ("edit.same_architecture", "edit.no_backbone") and not c.ok)

            # every arm must run from a clean checkout before any seed is drawn
            d0 = cfg["seed_sets"]["design"][0]
            smoke = list(range(d0, d0 + cfg["gates"]["smoke_seeds"]))
            bad = []
            for arm in manifest.get("arms", []):
                out = scratch(root) / "smoke" / task / f"{arm['id']}.jsonl"
                out.unlink(missing_ok=True)
                rc, tail = run_arm(root, wt, task, arm["id"], smoke, out, manifest=wm, timeout=3600)
                rows = arm_rows(out, smoke)
                if rc != 0 or rows is None or not all("benchmark_success" in r and "strict_success" in r for r in rows.values()):
                    bad.append(f"{arm['id']} (exit {rc}) {tail[-200:]}")
                heartbeat(root)
            shutil.rmtree(scratch(root) / "smoke" / task, ignore_errors=True)
            checks.append(Check("freeze.every_arm_runs", not bad, "; ".join(bad[:4])))
        finally:
            drop_worktree(root, wt)

    if all_ok(checks):
        commit = head_commit(root)
        if not tag_exists(root, tag):
            git(root, "tag", tag, commit, check=True)
        val = gates.validation_result(root, cfg, task)
        write_json(fz / "ACCEPTED.json", {
            "task": task,
            "tag": tag,
            "commit": commit,
            "tree": git(root, "rev-parse", f"{commit}^{{tree}}")[1],
            "genesis": current_genesis(root),
            "time": now_iso(),
            "edit": info,
            "validation": val,
            "checks_passed": len(checks),
        })
        referee_commit(root, [str((fz / "ACCEPTED.json").relative_to(root))], f"referee: freeze accepted for {task} at {commit[:12]}")
        notify(f"Freeze accepted ({task})", f"tag {tag} at {commit[:12]}. {info.get('weights_changed', 0):,} weights changed. Passed the validation rule: {val.get('passes_validation_rule') if val else 'unknown'}")
        return 0

    lines = [f"# Freeze request {k} for {task} was rejected", "", f"Time {now_iso()}, commit {head_commit(root)[:12]}.", "", "Fix each item below, then write a new freeze/REQUEST.json.", ""]
    lines += [f"- {c.name}. {c.detail}" for c in failing(checks)]
    (fz / f"REJECTED-{k}.md").write_text("\n".join(lines) + "\n")
    req = fz / "REQUEST.json"
    if req.exists():
        dst = fz / f"REQUEST-{k}.rejected.json"
        rc, _ = git(root, "mv", "-f", str(req.relative_to(root)), str(dst.relative_to(root)))
        if rc != 0:
            req.rename(dst)
    git(root, "add", "-A", "--", str(fz.relative_to(root)))
    subprocess.run(["git", "-C", str(root), "-c", "user.name=awe-referee", "-c", "user.email=awe-referee@users.noreply.github.com", "commit", "-q", "-m", f"referee: freeze request {k} rejected for {task}", "--", str(fz.relative_to(root))], capture_output=True)
    notify(f"Freeze rejected ({task})", "; ".join(c.name for c in failing(checks))[:400])
    return 0


# ------------------------------------------------------------------ test


def _push(root: Path, refs: list[str]) -> None:
    if os.environ.get("AWE_NO_PUSH") == "1":
        return
    rc, out = git(root, "push", "origin", *refs)
    if rc != 0:
        raise RuntimeError(f"git push failed. {out[-400:]}")


def _branch(root: Path) -> str:
    rc, out = git(root, "symbolic-ref", "--short", "-q", "HEAD")
    if rc != 0 or not out.strip():
        raise RuntimeError("the repo is not on a branch, so there is no branch to push")
    return out.strip()


def _gh(root: Path, args: list[str]) -> tuple[int, str]:
    return sh([os.environ.get("AWE_GH_BIN", "gh"), *args], cwd=root, timeout=120)


def _parse_time(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def publish_commitment(root: Path, cfg: dict, task: str, body: dict) -> dict:
    """Make the freeze public and get a timestamp from a third party. Returns the witness record."""
    tag = body["freeze_tag"]
    witness = os.environ.get("AWE_WITNESS", "github")
    if witness == "local":
        if not DRY:
            raise RuntimeError("a local witness is only allowed in a dry run (AWE_DRY_RUN=1)")
        return {"kind": "local-dry-run", "published_at": time.time(), "published_at_iso": now_iso(), "public": False}

    _push(root, [f"HEAD:refs/heads/{_branch(root)}", f"refs/tags/{tag}"])
    rc, out = _gh(root, ["repo", "view", "--json", "nameWithOwner,visibility"])
    if rc != 0:
        raise RuntimeError(f"gh repo view failed. Is gh logged in? {out[-300:]}")
    origin = json.loads(out)
    repo = os.environ.get("AWE_WITNESS_REPO") or origin["nameWithOwner"]
    same = repo == origin["nameWithOwner"]
    rel_tag = tag if same else f"{tag}-{body['freeze_commit'][:12]}"
    if not same:
        rc, out = _gh(root, ["repo", "view", repo, "--json", "nameWithOwner,visibility"])
        if rc != 0:
            raise RuntimeError(f"cannot see the witness repo {repo}. {out[-300:]}")
    vis = json.loads(out)["visibility"]
    if vis != "PUBLIC" and os.environ.get("AWE_ALLOW_PRIVATE_WITNESS") != "1":
        raise RuntimeError(f"the witness repo {repo} is {vis}. The commitment has to be public before the seeds are drawn. Make the repo public, or point AWE_WITNESS_REPO at a small public repo")

    view = ["release", "view", rel_tag, "--repo", repo, "--json", "publishedAt,url,tagName,isDraft,createdAt"]
    rc, out = _gh(root, view)
    if rc != 0:  # never create a second release for the same freeze
        notes = scratch(root) / f"commitment-{task}.json"
        notes.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        create = ["release", "create", rel_tag, "--repo", repo, "--title", f"Freeze commitment {tag}", "--notes-file", str(notes)]
        if same:
            create.append("--verify-tag")
        rc, out = _gh(root, create)
        if rc != 0:
            raise RuntimeError(f"gh release create failed. {out[-400:]}")
        rc, out = _gh(root, view)
        if rc != 0:
            raise RuntimeError(f"gh release view failed after creating it. {out[-300:]}")
    rel = json.loads(out)
    if rel.get("isDraft") or not rel.get("publishedAt"):
        raise RuntimeError("the commitment release is still a draft")
    rec = {
        "kind": "github-release",
        "repo": repo,
        "release_tag": rel_tag,
        "url": rel.get("url"),
        "published_at_iso": rel["publishedAt"],
        "published_at": _parse_time(rel["publishedAt"]),
        "public": vis == "PUBLIC",
        "wayback": None,
    }
    if rec["public"] and rec["url"] and os.environ.get("AWE_WAYBACK", "1") != "0":
        try:  # a second, independent timestamp. Best effort
            urllib.request.urlopen(urllib.request.Request("https://web.archive.org/save/" + rec["url"], headers={"User-Agent": "awe-referee/1.0"}), timeout=25).read(64)
            rec["wayback"] = "requested " + now_iso()
        except Exception as ex:
            rec["wayback"] = f"not saved ({type(ex).__name__})"
    return rec


def _tables(rows: dict[int, dict]) -> dict[str, dict[int, bool]]:
    return {
        "benchmark": {s: bool(r["benchmark_success"]) for s, r in rows.items()},
        "strict": {s: bool(r["strict_success"]) for s, r in rows.items()},
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def test(root: Path, cfg: dict, task: str) -> int:
    e = exp_dir(root, task)
    fz, td = e / "freeze", e / "test"
    td.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(Path(p).relative_to(root))  # noqa: E731
    acc = read_json(fz / "ACCEPTED.json")
    if not acc:
        log("no accepted freeze")
        return 1
    tag, commit = acc["tag"], acc["commit"]
    b = cfg["beacon"]

    # 1. commitment, public before any seed exists
    com = read_json(td / "commitment.json")
    if not com:
        body = {
            "scheme": "awe-commit-v1",
            "task": task,
            "freeze_tag": tag,
            "freeze_commit": commit,
            "tree": acc.get("tree"),
            "genesis": acc.get("genesis"),
            "edit_sha256": sha256_file(fz / "edit.json"),
            "manifest_sha256": sha256_file(fz / "test_manifest.json"),
            "edited_checkpoint_sha256": (acc.get("edit") or {}).get("edited_checkpoint_sha256"),
            "beacon": {"name": b["name"], "chain_hash": b["chain_hash"], "delay_seconds": b["delay_seconds"]},
            "rule": "The test seeds come from the first beacon round emitted at least delay_seconds after this commitment was published. start = base + int(sha256(domain|task|freeze_commit|randomness)[:8]) mod span, and the seeds are start to start + count - 1.",
            "test_seeds": cfg["test_seeds"],
        }
        wit = publish_commitment(root, cfg, task, body)
        com = {"body": body, "witness": wit, "target_round": beacon.target_round(wit["published_at"], b), "dry_run": DRY}
        write_json(td / "commitment.json", com)
        referee_commit(root, [rel(td / "commitment.json")], f"referee: commitment published for {tag}")
        _push(root, [f"HEAD:refs/heads/{_branch(root)}"])
        notify(f"Commitment published ({task})", f"{wit.get('url') or wit['kind']}. Seeds come from beacon round {com['target_round']}, in about {b['delay_seconds'] // 60} minutes")

    # 2. the draw
    draw = read_json(td / "draw.json")
    if not draw:
        rnd = com["target_round"]
        if os.environ.get("AWE_BEACON_FILE"):
            if not DRY:
                raise RuntimeError("a beacon file is only allowed in a dry run (AWE_DRY_RUN=1)")
            bv = read_json(Path(os.environ["AWE_BEACON_FILE"]))
            bv = {"round": rnd, "signature": bv["signature"], "randomness": beacon.randomness_of(bv["signature"]), "relays": ["file"], "bls_verified": False}
        else:
            bv = beacon.wait_for_round(rnd, b, log=log, heartbeat=lambda: heartbeat(root))
        round_time = beacon.time_of_round(rnd, b["genesis_time"], b["period"])
        if not DRY and round_time < com["witness"]["published_at"] + b["delay_seconds"]:
            raise RuntimeError("the beacon round is earlier than the rule allows. Refusing to draw")
        block = beacon.seed_block(bv["randomness"], task, commit, cfg["test_seeds"])
        draw = {"task": task, "freeze_tag": tag, "freeze_commit": commit, "published_at": com["witness"]["published_at"], "published_at_iso": com["witness"]["published_at_iso"], "beacon": {**bv, "round_time": round_time, "chain_hash": b["chain_hash"], "public_key": b["public_key"]}, "block": block, "drawn_at": now_iso(), "dry_run": DRY}
        write_json(td / "draw.json", draw)
        referee_commit(root, [rel(td / "draw.json")], f"referee: test seeds drawn for {tag}")
        _push(root, [f"HEAD:refs/heads/{_branch(root)}"])
        notify(f"Test seeds drawn ({task})", f"seeds {block['start']} to {block['end']} from beacon round {rnd}. Running the test now")

    seeds = list(range(draw["block"]["start"], draw["block"]["start"] + draw["block"]["count"]))

    # 3. every arm, from a clean checkout of the frozen commit
    wt = make_worktree(root, tag, f"test-{task}")
    try:
        wm = wt / "experiments" / task / "freeze" / "test_manifest.json"
        manifest = read_json(wm)
        order = {"original": 0, "agent_edit": 1, "original_default": 2, "agent_edit_default": 3}
        arms = sorted(manifest["arms"], key=lambda a: order.get(a["role"], 9))
        results: dict[str, dict[int, dict]] = {}
        headline_sent = (td / "headline.json").exists()
        for i, arm in enumerate(arms):
            want = seeds[: int(arm["n_seeds"])] if arm.get("n_seeds") else seeds
            out = td / "arms" / f"{arm['id']}.jsonl"
            rows = arm_rows(out, want)
            tries = 0
            while rows is None and tries < 2:
                tries += 1
                log(f"arm {i + 1} of {len(arms)}: {arm['id']} on {len(want)} seeds (attempt {tries})")
                rc, tail = run_arm(root, wt, task, arm["id"], want, out, manifest=wm, draw=td / "draw.json", timeout=24 * 3600)
                rows = arm_rows(out, want)
                heartbeat(root)
                if rows is None:
                    log(f"arm {arm['id']} incomplete (exit {rc}). {tail[-300:]}")
            if rows is None:
                write_json(td / "FAILED.json", {"arm": arm["id"], "reason": f"arm {arm['id']} did not finish after 2 attempts", "tail": tail[-1500:], "time": now_iso()})
                referee_commit(root, [rel(td)], f"referee: test failed for {tag}")
                notify(f"Test failed ({task})", f"arm {arm['id']} did not finish. This needs a person")
                return 1
            results[arm["id"]] = rows
            if not headline_sent and all(any(a["role"] == r and a["id"] in results for a in arms) for r in ("original", "agent_edit")):
                prim = manifest["primary_metric"]
                oid = next(a["id"] for a in arms if a["role"] == "original")
                eid = next(a["id"] for a in arms if a["role"] == "agent_edit")
                h = stats.paired(_tables(results[oid])[prim], _tables(results[eid])[prim])
                write_json(td / "headline.json", {"primary_metric": prim, **h, "note": "main arms only. The verdict follows when every arm has run"})
                referee_commit(root, [rel(td / "headline.json"), rel(td / "arms")], f"referee: main arms done for {tag}")
                notify(f"Main arms done ({task})", f"{_pct(h['orig_success'])} to {_pct(h['new_success'])}, {h['fixed']} fixed and {h['broke']} broke, p {h['p_mcnemar']:.2g}")
                headline_sent = True

        # 4. cross-check the two main arms with LeRobot's own evaluator
        cross = None
        if os.environ.get("AWE_STOCK_CROSSCHECK", "1") != "0":
            cross = read_json(td / "stock" / "crosscheck.json")
            if not cross:
                cross = stock_crosscheck(root, wt, cfg, task, manifest, wm, seeds, results, td)
                write_json(td / "stock" / "crosscheck.json", cross)
    finally:
        drop_worktree(root, wt)

    # 5. the verdict
    v = build_verdict(cfg, task, manifest, results, draw, com, acc, cross)
    v["reviews"] = status_mod.reviews_summary(root, task)
    for r in v["reviews"]:
        if r["not_reviewed"] or r["open_issues"]:
            v["reasons"].append("WARNING: " + _review_line(r) + ". The post must say so")
    write_json(td / "verdict.json", v)
    (td / "VERDICT.md").write_text(verdict_markdown(v))
    referee_commit(root, [rel(td)], f"referee: verdict for {tag} is {v['ending']}")
    try:
        _push(root, [f"HEAD:refs/heads/{_branch(root)}"])
    except RuntimeError as ex:
        log(str(ex))
    notify(f"Verdict ({task}): {stats.ENDING_TEXT[v['ending']]}", "; ".join(v["reasons"]))
    return 0


def stock_crosscheck(root, wt, cfg, task, manifest, wm, seeds, results, td) -> dict:
    env = read_json(wt / "experiments" / task / "stage0" / "env.json") or {}
    spe = float(env.get("stock_seconds_per_episode") or 60)
    budget = float(os.environ.get("AWE_STOCK_BUDGET_HOURS", "6")) * 3600
    n = max(min(len(seeds), 100), min(len(seeds), int(budget / (2 * spe))))
    out = {"n": n, "arms": {}, "note": ""}
    for role in ("original", "agent_edit"):
        arm_id = next(a["id"] for a in manifest["arms"] if a["role"] == role)
        f, err = export_checkpoint(root, wt, task, arm_id, wm, scratch(root) / "export" / f"{task}-test-{role}")
        if f is None:
            out["arms"][role] = {"error": err}
            continue
        table, err = stock_eval(root, wt, task, cfg, f.parent, seeds[0], n, td / "stock" / role, timeout=budget, draw=td / "draw.json")
        heartbeat(root)
        if table is None:
            out["arms"][role] = {"error": err}
            continue
        mine = _tables({s: results[arm_id][s] for s in seeds[:n]})["benchmark"]
        both = sorted(set(mine) & set(table))
        out["arms"][role] = {
            "episodes": len(both),
            "agreement": (sum(1 for s in both if mine[s] == table[s]) / len(both)) if both else None,
            "stock_success": (sum(table[s] for s in both) / len(both)) if both else None,
            "harness_success": (sum(mine[s] for s in both) / len(both)) if both else None,
        }
    ag = [a.get("agreement") for a in out["arms"].values()]
    need = cfg["verdict"]["stock_agreement_min"]
    out["ok"] = bool(ag) and all(x is not None and x >= need for x in ag)
    return out


def build_verdict(cfg, task, manifest, results, draw, com, acc, cross) -> dict:
    prim = manifest["primary_metric"]
    guard = "strict" if prim == "benchmark" else "benchmark"
    arms = manifest["arms"]
    tab = {a["id"]: _tables(results[a["id"]]) for a in arms}
    one = lambda role: next(a["id"] for a in arms if a["role"] == role)  # noqa: E731
    oid, eid = one("original"), one("agent_edit")

    def rel_gain(arm_id: str) -> tuple[float, float]:
        """An arm's gain, and the edit's gain on the same seeds."""
        s = set(tab[arm_id][prim])
        o = {k: v for k, v in tab[oid][prim].items() if k in s}
        ed = {k: v for k, v in tab[eid][prim].items() if k in s}
        return stats.paired(o, tab[arm_id][prim])["gain_points"], stats.paired(o, ed)["gain_points"]

    randoms, baselines = [], {}
    for a in arms:
        if a["role"] == "random":
            randoms.append(rel_gain(a["id"]))
        elif a["role"] in ("baseline", "ensembling_original"):
            name = a.get("family") or ("temporal_ensembling" if a["role"] == "ensembling_original" else a["id"])
            baselines[name] = rel_gain(a["id"])
    boot_seed = int(draw["beacon"]["randomness"][:16], 16)
    v = stats.verdict(
        primary_orig=tab[oid][prim],
        primary_edit=tab[eid][prim],
        guard_orig=tab[oid][guard],
        guard_edit=tab[eid][guard],
        random_pairs=randoms,
        baseline_pairs=baselines,
        autogreedy_pair=rel_gain(one("autogreedy")),
        cfg=cfg,
        boot_seed=boot_seed,
    )
    per_arm = {}
    for a in arms:
        t = tab[a["id"]]
        p = stats.paired({k: v2 for k, v2 in tab[oid][prim].items() if k in t[prim]}, t[prim])
        per_arm[a["id"]] = {"role": a["role"], "family": a.get("family"), "n": len(t[prim]), "benchmark_success": sum(t["benchmark"].values()) / len(t["benchmark"]), "strict_success": sum(t["strict"].values()) / len(t["strict"]), "gain_points": p["gain_points"], "fixed": p["fixed"], "broke": p["broke"], "p_mcnemar": p["p_mcnemar"]}
    classes = {}
    for label, arm_id in (("original", oid), ("agent_edit", eid)):
        c: dict[str, int] = {}
        for r in results[arm_id].values():
            k = str(r.get("failure_class", "unknown"))
            c[k] = c.get(k, 0) + 1
        classes[label] = dict(sorted(c.items()))
    v.update({
        "task": task,
        "primary_metric": prim,
        "guard_metric": guard,
        "operating_point": manifest.get("operating_point"),
        "freeze": {"tag": acc["tag"], "commit": acc["commit"], "genesis": acc.get("genesis"), "edit": acc.get("edit"), "validation": acc.get("validation")},
        "commitment": com["witness"],
        "draw": {"round": draw["beacon"]["round"], "round_time": draw["beacon"]["round_time"], "bls_verified": draw["beacon"]["bls_verified"], "start": draw["block"]["start"], "end": draw["block"]["end"], "count": draw["block"]["count"]},
        "arms": per_arm,
        "failure_classes": classes,
        "stock_crosscheck": cross,
        "dry_run": bool(draw.get("dry_run")),
        "time": now_iso(),
    })
    if cross is not None and not cross.get("ok"):
        v["reasons"].append("WARNING: the fast harness and LeRobot's evaluator agreed on fewer than 98% of cross-checked episodes, or the cross-check could not run. The post must say so")
    return v


def _review_line(r: dict) -> str:
    gate = r["gate"]
    name = "Gate 1" if gate == "g1" else f"Freeze request {gate.split('.')[-1]}" if gate.startswith("g2") else gate
    if r["not_reviewed"]:
        return f"{name} was not independently reviewed, because the reviewer never returned a usable verdict"
    if r["open_issues"]:
        return f"{name} ended with {r['open_issues']} open review issues after round {r['rounds']}"
    return f"{name} passed review in round {r['rounds']}"


def verdict_markdown(v: dict) -> str:
    h1, g, h2, h3 = v["h1"], v["guard"], v["h2"], v["h3"]
    ci = h1["ci95"]
    f = lambda x, d=1: "n/a" if x is None else f"{x:.{d}f}"  # noqa: E731
    ci_gain = f"{f(ci['gain_points'][0])} to {f(ci['gain_points'][1])}" if ci.get("gain_points") else "n/a"
    ci_share = f"{_pct(ci['share_removed'][0])} to {_pct(ci['share_removed'][1])}" if ci.get("share_removed") else "n/a"
    name = {"benchmark": "benchmark success", "strict": "strict handover"}
    d, w = v["draw"], v["commitment"]
    best = h3["best_baseline"]
    rows = [
        ("Test seeds", f"{d['count']:,} seeds, {d['start']} to {d['end']}, drawn from beacon round {d['round']} after the freeze was public"),
        ("Primary score", name[v["primary_metric"]]),
        ("Original policy", _pct(h1["orig_success"])),
        ("Edited policy", _pct(h1["new_success"])),
        ("Net gain", f"{f(h1['gain_points'])} points, 95% interval {ci_gain}"),
        ("Seeds fixed and broken", f"{h1['fixed']} fixed, {h1['broke']} broken"),
        ("Exact McNemar p", f"{h1['p_mcnemar']:.3g}"),
        ("Share of failures removed", f"{_pct(h1['share_removed'])}, 95% interval {ci_share}"),
        ("Guard score, " + name[v["guard_metric"]], f"{_pct(g['orig'])} to {_pct(g['edit'])}. A drop of up to {f(g['limit_points'])} points is allowed. {'Passed' if g['ok'] else 'FAILED'}"),
        ("H2, matched random edits", (f"{h2['n_gaining_as_much']} of {h2['n_random']} gained as much, and at most {h2['allowed']} may. {'Passed' if h2['pass'] else 'Not passed'}") if h2["testable"] else f"only {h2['n_random']} random edits exist, too few to test"),
        ("H3, best simple baseline", (f"{best[0]} at {f(best[1])} points, against {f(best[2])} for the edit on the same seeds. " if best else "") + ("No baseline matched the edit" if h3["clear"] else "Matched by " + ", ".join(h3["matching_baselines"]))),
        ("Auto-greedy edit, built with no LLM", f"{f(v['autogreedy']['gain_points'])} points. {'It matched the agent edit' if v['autogreedy']['matches_edit'] else 'It did not match the agent edit'}"),
    ]
    c = v.get("stock_crosscheck")
    if c and c.get("arms"):
        parts = []
        for role, a in c["arms"].items():
            parts.append(f"{role} {a['error'][:80]}" if "error" in a else f"{role} agreed on {_pct(a['agreement'])} of {a['episodes']} episodes")
        rows.append(("Cross-check with LeRobot's own evaluator", ". ".join(parts)))
    edit = (v["freeze"].get("edit") or {})
    if edit.get("weights_changed"):
        rows.append(("Size of the edit", f"{edit['weights_changed']:,} weights, {100 * (edit.get('share_of_inference_weights') or 0):.3f}% of the weights that run at inference"))
    if v.get("reviews"):
        rows.append(("Reviews by a separate session", ". ".join(_review_line(r) for r in v["reviews"])))
    rows.append(("Freeze", f"tag {v['freeze']['tag']}, commit {v['freeze']['commit'][:12]}, rules from {v['freeze'].get('genesis')}"))
    rows.append(("Commitment", f"{w.get('url') or w.get('kind')}, published {w.get('published_at_iso')}"))
    L = [gates.VERDICT_BEGIN, f"## Referee verdict for {v['task']}", ""]
    if v.get("dry_run"):
        L += ["**DRY RUN. These numbers are not a real test.**", ""]
    L += [f"**Ending. {stats.ENDING_TEXT[v['ending']]}**", "", "| Item | Value |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in rows]
    L += ["", "Why this ending."] + [f"- {r}" for r in v["reasons"]] + ["", gates.VERDICT_END, ""]
    return "\n".join(L)


# ------------------------------------------------------------------ entry


ACTIONS = {"confirm_g0": confirm_g0, "stamp_g1": stamp_g1, "freeze": freeze, "test": test}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="referee.judge")
    ap.add_argument("command", choices=["step"])
    ap.add_argument("--repo")
    ap.add_argument("--task")
    ap.add_argument("--action", choices=sorted(ACTIONS))
    a = ap.parse_args(argv)
    root = repo_root(a.repo)
    cfg = load_config()
    task, action = a.task, a.action
    if not (task and action):
        st = status_mod.compute(root, cfg)
        if st["state"] != "JUDGE_TURN":
            log(f"nothing for the referee to do. State is {st['state']}")
            return 0
        task, action = st["task"], st["action"]
    log(f"{action} for {task}")
    return ACTIONS[action](root, cfg, task)


if __name__ == "__main__":
    sys.exit(main())
