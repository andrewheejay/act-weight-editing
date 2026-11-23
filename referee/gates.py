"""Mechanical checks for every task in PLAN.json. Standard library only.

Each function returns a list of Check objects. A task is done when all of its hard checks pass.
The checks recompute the pre-registered rules from the raw per-seed files. They never trust a summary
the agent wrote. File layouts are described in CONTRACT.md.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import ledger as ledger_mod
from . import stats
from .common import (
    Check,
    appended_only,
    by_seed,
    changed_since_genesis,
    exp_dir,
    find_jsonl,
    in_range,
    rate,
    read_json,
    read_jsonl,
    seeds_cover,
    sha256_file,
    stock_per_episode,
)

REQUIRED_TESTS = [
    "test_fast_harness_matches_stock",
    "test_head_scaling_exact",
    "test_sham_edit_identical",
    "test_repeat_run_identical",
    "test_seed_lock",
]
ENV_KEYS = [
    "lerobot",
    "gym_aloha",
    "mujoco",
    "torch",
    "python",
    "renderer",
    "rollouts_per_second",
    "stock_seconds_per_episode",
    "stock_eval_cmd",
]
STOCK_PLACEHOLDERS = ["{policy_path}", "{env_task}", "{n_episodes}", "{seed}", "{output_dir}"]
DIAGNOSIS_HEADINGS = ["target failure", "tracking number", "near-miss evidence", "primary metric", "operating point"]
ARM_ROLES = [
    "original",
    "agent_edit",
    "original_default",
    "agent_edit_default",
    "autogreedy",
    "random",
    "baseline",
    "dose",
    "ensembling_original",
    "ensembling_edit",
]
ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
GRADIENT_PATTERNS = [
    r"\.backward\(",
    r"torch\.autograd",
    r"autograd\.grad",
    r"requires_grad_\(\s*(True)?\s*\)",
    r"requires_grad\s*=\s*True",
    r"torch\.optim\b",
    r"torch\.func\.(grad|jacrev|jacfwd|vjp|jvp)",
    r"\bjacobian\(",
]


def _load(path: Path, what: str) -> tuple[list[dict] | None, Check | None]:
    if find_jsonl(path) is None:
        return None, Check(what, False, f"missing {path}")
    try:
        return read_jsonl(path), None
    except ValueError as e:
        return None, Check(what, False, str(e))


def _json(path: Path, what: str) -> tuple[dict | None, Check | None]:
    try:
        d = read_json(path)
    except ValueError as e:
        return None, Check(what, False, str(e))
    if d is None:
        return None, Check(what, False, f"missing {path}")
    return d, None


def code_fingerprint(root: Path) -> str:
    """Hash of the harness code and its tests. A change here means Gate 0 gets re-confirmed."""
    h = hashlib.sha256()
    for top in ("awe", "tests"):
        d = root / top
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(root)).encode())
                h.update(sha256_file(p).encode())
    return h.hexdigest()


def stage0_files(root: Path, task: str) -> list[Path]:
    e = exp_dir(root, task) / "stage0"
    names = [
        "env.json",
        "checkpoint.json",
        "stock_eval/eval_info.json",
        "harness_repro.jsonl",
        "repeat_a.jsonl",
        "repeat_b.jsonl",
        "sham.jsonl",
        "renderer.json",
    ]
    return [e / n for n in names]


# ------------------------------------------------------------------ stage 0


def s0_env(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage0" / "env.json", "s0.env")
    if err:
        return [err]
    missing = [k for k in ENV_KEYS if k not in d]
    out = [Check("s0.env.keys", not missing, "missing keys " + ", ".join(missing) if missing else "")]
    cmd = d.get("stock_eval_cmd")
    if isinstance(cmd, list):
        joined = " ".join(str(x) for x in cmd)
        lack = [p for p in STOCK_PLACEHOLDERS if p not in joined]
        out.append(Check("s0.env.stock_eval_cmd", not lack, "missing placeholders " + ", ".join(lack) if lack else ""))
    elif "stock_eval_cmd" in d:
        out.append(Check("s0.env.stock_eval_cmd", False, "stock_eval_cmd must be a list of strings"))
    rps = d.get("rollouts_per_second")
    out.append(Check("s0.env.speed", isinstance(rps, (int, float)) and rps > 0, f"rollouts_per_second = {rps}"))
    return out


def checkpoint_path(root: Path, d: dict) -> Path:
    import os

    base = Path(os.environ.get("AWE_CHECKPOINTS", root / "checkpoints"))
    rel = Path(d.get("dir", ""))
    p = base.joinpath(*rel.parts[1:]) if rel.parts[:1] == ("checkpoints",) else (root / rel)
    return p / d.get("model_file", "model.safetensors")


def s0_checkpoint(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage0" / "checkpoint.json", "s0.checkpoint")
    if err:
        return [err]
    missing = [k for k in ("dir", "model_sha256", "source") if k not in d]
    if missing:
        return [Check("s0.checkpoint.keys", False, "missing keys " + ", ".join(missing))]
    want = cfg["tasks"][task]["hub_repo"]
    p = checkpoint_path(root, d)
    return [
        Check("s0.checkpoint.source", d["source"] == want, f"source is {d['source']}, expected {want}"),
        Check("s0.checkpoint.file", p.exists(), f"{p} {'found' if p.exists() else 'missing'}"),
        Check("s0.checkpoint.sha", bool(re.fullmatch(r"[0-9a-f]{64}", str(d["model_sha256"]))), "model_sha256 format"),
    ]


def s0_tests(root, cfg, task) -> list[Check]:
    text = ""
    d = root / "tests"
    if d.is_dir():
        for p in d.rglob("test_*.py"):
            text += p.read_text(errors="replace")
    missing = [t for t in REQUIRED_TESTS if not re.search(rf"def\s+{t}\b", text)]
    return [Check("s0.tests.present", not missing, "missing tests " + ", ".join(missing) if missing else "")]


def _stock_table(root, cfg, task) -> tuple[dict[int, bool] | None, Check | None]:
    """The Stage 0 stock run. It must be one lerobot-eval call with --seed set to the first reproduction seed."""
    d, err = _json(exp_dir(root, task) / "stage0" / "stock_eval" / "eval_info.json", "s0.stock")
    if err:
        return None, err
    rows = stock_per_episode(d, start_seed=cfg["seed_sets"]["reproduction"][0])
    table = {int(r["seed"]): bool(r["success"]) for r in rows if r.get("seed") is not None and "success" in r}
    return table, None


def s0_stock(root, cfg, task) -> list[Check]:
    table, err = _stock_table(root, cfg, task)
    if err:
        return [err]
    rng = cfg["seed_sets"]["reproduction"]
    have = {s: v for s, v in table.items() if in_range(s, rng)}
    need = rng[1] - rng[0] + 1
    out = [Check("s0.stock.coverage", len(have) >= need, f"{len(have)} of {need} reproduction seeds")]
    if have:
        r = sum(have.values()) / len(have)
        lo, hi = cfg["tasks"][task]["reproduction_window"]
        out.append(
            Check("s0.stock.reproduces", lo <= r <= hi, f"stock success {100 * r:.1f}%, window {100 * lo:.1f} to {100 * hi:.1f}%")
        )
    return out


def s0_harness(root, cfg, task) -> list[Check]:
    e = exp_dir(root, task) / "stage0"
    rows, err = _load(e / "harness_repro.jsonl", "s0.harness")
    if err:
        return [err]
    rng = cfg["seed_sets"]["reproduction"]
    ok, n = seeds_cover(rows, rng)
    out = [Check("s0.harness.coverage", ok, f"{n} of {rng[1] - rng[0] + 1} reproduction seeds")]
    table, err = _stock_table(root, cfg, task)
    if err:
        return out + [err]
    mine = by_seed(rows, "benchmark_success")
    both = sorted(set(mine) & set(table))
    if both:
        agree = sum(1 for s in both if mine[s] == table[s]) / len(both)
        need = cfg["gates"]["harness_agreement_min"]
        out.append(Check("s0.harness.agreement", agree >= need, f"agrees with lerobot-eval on {100 * agree:.1f}% of {len(both)} seeds"))
        if agree < 1.0:
            m = e / "mismatches.md"
            out.append(Check("s0.harness.mismatches_read", m.exists() and m.stat().st_size > 0, "write stage0/mismatches.md, one line per mismatched seed"))
    else:
        out.append(Check("s0.harness.agreement", False, "no seeds in common with the stock evaluation"))
    return out


def _same(a: list[dict], b: list[dict], keys=("benchmark_success", "strict_success", "traj_hash")) -> tuple[int, int, list[int]]:
    ia = {int(r["seed"]): r for r in a if r.get("seed") is not None}
    ib = {int(r["seed"]): r for r in b if r.get("seed") is not None}
    both = sorted(set(ia) & set(ib))
    diff = [s for s in both if any(ia[s].get(k) != ib[s].get(k) for k in keys)]
    return len(both), len(diff), diff[:8]


def s0_repeat(root, cfg, task) -> list[Check]:
    e = exp_dir(root, task) / "stage0"
    out = []
    loaded = {}
    for name in ("repeat_a", "repeat_b", "sham"):
        rows, err = _load(e / f"{name}.jsonl", f"s0.repeat.{name}")
        if err:
            out.append(err)
        else:
            loaded[name] = rows
    if len(loaded) < 3:
        return out
    need = cfg["gates"]["repeat_seeds"]
    has_hash = all("traj_hash" in r for rows in loaded.values() for r in rows)
    out.append(Check("s0.repeat.traj_hash", has_hash, "every record needs traj_hash, a hash of the full trajectory"))
    n, d, ex = _same(loaded["repeat_a"], loaded["repeat_b"])
    out.append(Check("s0.repeat.identical", n >= need and d == 0, f"{n} seeds compared, {d} differ {ex if d else ''}"))
    n, d, ex = _same(loaded["repeat_a"], loaded["sham"])
    out.append(Check("s0.repeat.sham_identical", n >= need and d == 0, f"{n} seeds compared against the sham edit, {d} differ {ex if d else ''}"))
    return out


def s0_renderer(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage0" / "renderer.json", "s0.renderer")
    if err:
        return [err]
    return [Check("s0.renderer.chosen", bool(d.get("chosen")), f"chosen renderer {d.get('chosen')}")]


def s0_published(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage0" / "published_eval_info.json", "s0.published")
    if err:
        return [err]
    rows = stock_per_episode(d)
    pub = {int(r["seed"]): bool(r["success"]) for r in rows if r.get("seed") is not None and "success" in r}
    out = [Check("s0.published.parsed", len(pub) > 0, f"{len(pub)} published episodes")]
    table, _ = _stock_table(root, cfg, task)
    if table and pub:
        both = set(table) & set(pub)
        if both:
            agree = sum(1 for s in both if table[s] == pub[s]) / len(both)
            out.append(Check("s0.published.agreement", True, f"our stock run matches the published file on {100 * agree:.1f}% of {len(both)} seeds", soft=True))
    return out


# ------------------------------------------------------------------ stage 1


def design_tables(root, cfg, task) -> tuple[dict[str, list[dict]], list[Check]]:
    e = exp_dir(root, task) / "stage1"
    out, tables = [], {}
    rng = cfg["seed_sets"]["design"]
    for name in ("design_default", "design_chunk50", "design_chunk25"):
        rows, err = _load(e / f"{name}.jsonl", f"s1.{name}")
        if err:
            out.append(err)
            continue
        ok, n = seeds_cover(rows, rng)
        out.append(Check(f"s1.{name}.coverage", ok, f"{n} of {rng[1] - rng[0] + 1} design seeds"))
        tables[name] = rows
    rows, err = _load(e / "design_ens.jsonl", "s1.design_ens")
    if err:
        out.append(err)
    else:
        need = cfg["gates"]["ensembling_min_seeds"]
        ok, n = seeds_cover(rows, rng, need)
        out.append(Check("s1.design_ens.coverage", ok, f"{n} design seeds with temporal ensembling, need {need}"))
        tables["design_ens"] = rows
    return tables, out


def s1_design_runs(root, cfg, task) -> list[Check]:
    return design_tables(root, cfg, task)[1]


def s1_figures(root, cfg, task) -> list[Check]:
    e = exp_dir(root, task) / "stage1"
    out = []
    for name in ("failure_map.png", "reach_error.png"):
        p = e / name
        out.append(Check(f"s1.figures.{name}", p.exists() and p.stat().st_size > 0, str(p)))
    strips = e / "strips"
    n = len(list(strips.glob("*"))) if strips.is_dir() else 0
    out.append(Check("s1.figures.strips", n >= 40, f"{n} frame strips, the spec asks for 30 failures and 10 successes", soft=True))
    return out


def expected_decisions(root, cfg, task) -> tuple[dict | None, list[Check]]:
    """Recompute the Gate 1 rules from the design runs."""
    tables, checks = design_tables(root, cfg, task)
    if not all(k in tables for k in ("design_default", "design_chunk50", "design_chunk25")):
        return None, checks
    g = cfg["gates"]
    bench = rate(tables["design_default"], "benchmark_success")
    strict = rate(tables["design_default"], "strict_success")
    if bench is None or strict is None:
        return None, checks + [Check("s1.decisions.scores", False, "design_default needs benchmark_success and strict_success")]
    primary = "strict" if (bench - strict) * 100 >= g["swap_gap_points"] else "benchmark"
    key = f"{primary}_success"
    base = rate(tables["design_default"], key)
    gains = {c: (rate(tables[f"design_chunk{c}"], key) - base) * 100 for c in (50, 25)}
    best = max(gains, key=lambda c: gains[c])
    op = best if gains[best] >= g["operating_point_margin_points"] else 100
    proceed = (1 - base) >= g["min_failure_rate"]
    exp = {
        "primary_metric": primary,
        "n_action_steps": op,
        "proceed": proceed,
        "benchmark_success": bench,
        "strict_success": strict,
        "failure_rate": 1 - base,
        "chunk_gains_points": gains,
    }
    return exp, checks


def s1_decisions(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage1" / "decisions.json", "s1.decisions")
    if err:
        return [err]
    exp, _ = expected_decisions(root, cfg, task)
    if exp is None:
        return [Check("s1.decisions.inputs", False, "the design runs are not complete yet")]
    out = [
        Check(
            "s1.decisions.primary_metric",
            d.get("primary_metric") == exp["primary_metric"],
            f"benchmark {100 * exp['benchmark_success']:.1f}%, strict {100 * exp['strict_success']:.1f}%, so the swap rule gives {exp['primary_metric']}. decisions.json says {d.get('primary_metric')}",
        ),
        Check(
            "s1.decisions.operating_point",
            (d.get("operating_point") or {}).get("n_action_steps") == exp["n_action_steps"],
            f"chunk gains in points {exp['chunk_gains_points']}, so the rule gives {exp['n_action_steps']} steps. decisions.json says {(d.get('operating_point') or {}).get('n_action_steps')}",
        ),
        Check(
            "s1.decisions.proceed",
            d.get("proceed") is exp["proceed"],
            f"failure rate {100 * exp['failure_rate']:.1f}% on the primary score, so proceed should be {exp['proceed']}",
        ),
    ]
    for k in ("target_failure_class", "tracking_number"):
        out.append(Check(f"s1.decisions.{k}", bool(d.get(k)), f"{k} = {d.get(k)}"))
    return out


def s1_diagnosis(root, cfg, task) -> list[Check]:
    p = exp_dir(root, task) / "diagnosis.md"
    if not p.exists():
        return [Check("s1.diagnosis", False, f"missing {p}")]
    text = p.read_text(errors="replace").lower()
    missing = [h for h in DIAGNOSIS_HEADINGS if h not in text]
    return [
        Check("s1.diagnosis.sections", not missing, "missing sections " + ", ".join(missing) if missing else ""),
        Check("s1.diagnosis.length", len(text.split()) >= 150, f"{len(text.split())} words"),
    ]


# ------------------------------------------------------------------ stage 2


def s2_atlas(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage2" / "atlas" / "heads.json", "s2.atlas")
    if err:
        return [err]
    n = len(d.get("heads", d)) if isinstance(d, (dict, list)) else 0
    return [Check("s2.atlas.heads", n >= 48, f"{n} heads described, need 48")]


def s2_sweep(root, cfg, task) -> list[Check]:
    rows, err = _load(exp_dir(root, task) / "stage2" / "sweep.jsonl", "s2.sweep")
    if err:
        return [err]
    rng = cfg["seed_sets"]["sweep_half"]
    g = cfg["gates"]
    per: dict[str, set] = {}
    for r in rows:
        if r.get("seed") is None or not in_range(int(r["seed"]), rng):
            continue
        per.setdefault(str(r.get("arm")), set()).add(int(r["seed"]))
    full = [a for a, s in per.items() if a != "original" and len(s) >= g["sweep_seeds"]]
    return [
        Check("s2.sweep.baseline", len(per.get("original", ())) >= g["sweep_seeds"], f"original on {len(per.get('original', ()))} of {g['sweep_seeds']} sweep seeds"),
        Check("s2.sweep.arms", len(full) >= g["sweep_arms"], f"{len(full)} single-head edits with all {g['sweep_seeds']} seeds, need {g['sweep_arms']}"),
    ]


def s2_split_half(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage2" / "split_half.json", "s2.split_half")
    if err:
        return [err]
    n = len(d.get("rechecked", []))
    return [
        Check("s2.split_half.rechecked", n >= cfg["gates"]["split_half_min"], f"{n} edits re-run on the second half of the design seeds"),
        Check("s2.split_half.kept", isinstance(d.get("kept"), list), "kept must list the edits whose effect held in both halves"),
    ]


# ------------------------------------------------------------------ stage 3


def s3_hypotheses(root, cfg, task) -> list[Check]:
    p = exp_dir(root, task) / "hypotheses.md"
    if not p.exists() or not p.read_text().strip():
        return [Check("s3.hypotheses", False, f"missing or empty {p}")]
    text = p.read_text(errors="replace")
    cands = ledger_mod.summarize(root, cfg)["validation_candidates"].get(task, {}).get("agent", [])
    missing = [c for c in cands if c not in text]
    return [Check("s3.hypotheses.cover_candidates", not missing, "no hypothesis line names " + ", ".join(missing) if missing else f"{len(cands)} validation candidates all have a line")]


def s3_dose(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage3" / "dose_response.json", "s3.dose")
    return [err] if err else [Check("s3.dose.present", True, "")]


def edit_checks(d: dict, cfg: dict, task: str, name: str = "edit") -> list[Check]:
    lim = cfg["edit_limits"]
    out = []
    missing = [k for k in ("task", "candidate", "ops", "hypothesis", "free_numbers", "edited_checkpoint_sha256") if k not in d]
    if missing:
        return [Check(f"{name}.keys", False, "missing keys " + ", ".join(missing))]
    out.append(Check(f"{name}.task", d["task"] == task, f"task is {d['task']}"))
    out.append(Check(f"{name}.candidate", bool(ID_RE.match(str(d["candidate"]))), f"candidate id {d['candidate']}"))
    out.append(Check(f"{name}.hypothesis", len(str(d["hypothesis"]).split()) >= 10, "the hypothesis needs at least a sentence"))
    out.append(Check(f"{name}.sha", bool(re.fullmatch(r"[0-9a-f]{64}", str(d["edited_checkpoint_sha256"]))), "edited_checkpoint_sha256 format"))
    ops = d["ops"] if isinstance(d["ops"], list) else []
    out.append(Check(f"{name}.ops", len(ops) > 0, f"{len(ops)} ops"))
    heads, bad, latent, free_form = set(), [], 0, 0
    for i, op in enumerate(ops):
        kind = op.get("op")
        if kind == "scale_head":
            heads.add((op.get("block"), op.get("head")))
            if op.get("scale") not in lim["scale_values"]:
                bad.append(f"op {i} scale {op.get('scale')} is not one of {lim['scale_values']}")
        elif kind == "sharpen_head":
            heads.add((op.get("block"), op.get("head")))
            if op.get("factor") not in lim["sharpen_values"]:
                bad.append(f"op {i} factor {op.get('factor')} is not one of {lim['sharpen_values']}")
        elif kind == "latent_shift":
            latent += 1
        elif kind in ("direction_edit", "other"):
            free_form += 1
            if not op.get("description"):
                bad.append(f"op {i} needs a description")
        else:
            bad.append(f"op {i} has unknown kind {kind}")
        if kind in ("scale_head", "sharpen_head"):
            blk = str(op.get("block", ""))
            if not isinstance(op.get("head"), int) or not 0 <= op["head"] <= 7:
                bad.append(f"op {i} head must be 0 to 7")
            if any(blk.startswith(p) for p in lim["forbidden_prefixes"]):
                bad.append(f"op {i} touches the vision backbone")
    out.append(Check(f"{name}.op_rules", not bad, "; ".join(bad[:6])))
    out.append(Check(f"{name}.max_heads", len(heads) <= lim["max_heads"], f"{len(heads)} heads, limit {lim['max_heads']}"))
    out.append(Check(f"{name}.one_latent_shift", latent <= 1, f"{latent} latent shifts"))
    fn = d["free_numbers"]
    if free_form:
        out.append(Check(f"{name}.free_numbers", isinstance(fn, int) and fn <= lim["max_free_numbers"], f"{fn} free numbers, limit {lim['max_free_numbers']}"))
    return out


def s3_edit(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "freeze" / "edit.json", "s3.edit")
    return [err] if err else edit_checks(d, cfg, task, "s3.edit")


def s3_baselines(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "freeze" / "baselines.json", "s3.baselines")
    if err:
        return [err]
    items = d.get("baselines", []) if isinstance(d, dict) else []
    fams = {str(b.get("family")) for b in items}
    lack = [f for f in cfg["baseline_families"] if f not in fams]
    return [Check("s3.baselines.families", not lack, "missing families " + ", ".join(lack) if lack else f"{len(items)} baselines frozen")]


def validation_result(root, cfg, task) -> dict | None:
    """The validation pass rule from Stage 3, recomputed. Informational. The stop rule lets a miss through."""
    e = exp_dir(root, task)
    edit = read_json(e / "freeze" / "edit.json") or {}
    dec = read_json(e / "stage1" / "decisions.json") or {}
    cand = edit.get("candidate")
    if not cand or find_jsonl(e / "stage3" / "validation" / f"{cand}.jsonl") is None:
        return None
    if find_jsonl(e / "stage3" / "validation" / "original.jsonl") is None:
        return None
    o = read_jsonl(e / "stage3" / "validation" / "original.jsonl")
    c = read_jsonl(e / "stage3" / "validation" / f"{cand}.jsonl")
    prim = dec.get("primary_metric", "benchmark")
    guard = "strict" if prim == "benchmark" else "benchmark"
    main = stats.paired(by_seed(o, f"{prim}_success"), by_seed(c, f"{prim}_success"))
    g = stats.paired(by_seed(o, f"{guard}_success"), by_seed(c, f"{guard}_success"))
    v = cfg["verdict"]
    passed = bool(
        main["share_removed"] is not None
        and main["share_removed"] >= v["large_share"]
        and main["p_mcnemar"] < v["alpha"]
        and -(g["gain_points"] or 0) <= v["guard_drop_points"]
    )
    return {"candidate": cand, "primary": main, "guard_drop_points": -(g["gain_points"] or 0), "passes_validation_rule": passed}


def s3_validation(root, cfg, task) -> list[Check]:
    e = exp_dir(root, task)
    rng = cfg["seed_sets"]["validation"]
    out = []
    rows, err = _load(e / "stage3" / "validation" / "original.jsonl", "s3.validation.original")
    if err:
        return [err]
    ok, n = seeds_cover(rows, rng)
    out.append(Check("s3.validation.original", ok, f"original on {n} of {rng[1] - rng[0] + 1} validation seeds"))
    edit = read_json(e / "freeze" / "edit.json") or {}
    cand = edit.get("candidate")
    if not cand:
        return out + [Check("s3.validation.candidate", False, "freeze/edit.json does not name a candidate yet")]
    rows, err = _load(e / "stage3" / "validation" / f"{cand}.jsonl", "s3.validation.candidate")
    if err:
        return out + [err]
    ok, n = seeds_cover(rows, rng)
    out.append(Check("s3.validation.candidate", ok, f"{cand} on {n} validation seeds"))
    res = validation_result(root, cfg, task)
    if res:
        m = res["primary"]
        out.append(
            Check(
                "s3.validation.rule",
                True,
                f"{cand} gain {m['gain_points']:.1f} points, share of failures removed {m['share_removed'] if m['share_removed'] is None else round(m['share_removed'], 3)}, p {m['p_mcnemar']:.4f}. Passes the validation rule: {res['passes_validation_rule']}",
                soft=True,
            )
        )
    return out


# ------------------------------------------------------------------ stage 4


def manifest_checks(m: dict, cfg: dict, task: str, root: Path) -> list[Check]:
    out = []
    arms = m.get("arms", []) if isinstance(m, dict) else []
    ids = [str(a.get("id")) for a in arms]
    out.append(Check("s4.manifest.task", m.get("task") == task, f"task is {m.get('task')}"))
    bad_ids = [i for i in ids if not ID_RE.match(i)]
    out.append(Check("s4.manifest.ids", len(set(ids)) == len(ids) and not bad_ids, "arm ids must be unique and file-safe"))
    bad_roles = sorted({str(a.get("role")) for a in arms} - set(ARM_ROLES))
    out.append(Check("s4.manifest.roles", not bad_roles, "unknown roles " + ", ".join(bad_roles) if bad_roles else ""))
    n = {r: sum(1 for a in arms if a.get("role") == r) for r in ARM_ROLES}

    dec = read_json(exp_dir(root, task) / "stage1" / "decisions.json") or {}
    op = (dec.get("operating_point") or {}).get("n_action_steps")
    out.append(Check("s4.manifest.primary_metric", m.get("primary_metric") == dec.get("primary_metric"), f"manifest {m.get('primary_metric')}, Gate 1 {dec.get('primary_metric')}"))
    out.append(Check("s4.manifest.operating_point", (m.get("operating_point") or {}).get("n_action_steps") == op, f"manifest {(m.get('operating_point') or {}).get('n_action_steps')}, Gate 1 {op}"))

    for r in ("original", "agent_edit", "autogreedy", "ensembling_original", "ensembling_edit"):
        out.append(Check(f"s4.manifest.{r}", n[r] == 1, f"{n[r]} arms with role {r}, need exactly 1"))
    extra = 1 if op not in (None, 100) else 0
    for r in ("original_default", "agent_edit_default"):
        out.append(Check(f"s4.manifest.{r}", n[r] == extra, f"{n[r]} arms with role {r}, need {extra}"))
    out.append(Check("s4.manifest.dose", n["dose"] == 4, f"{n['dose']} dose arms, need 4"))
    fams = {str(a.get("family")) for a in arms if a.get("role") == "baseline"}
    lack = [f for f in cfg["baseline_families"] if f not in fams]
    out.append(Check("s4.manifest.baselines", not lack, "baseline arms missing families " + ", ".join(lack) if lack else f"{n['baseline']} baseline arms"))

    rf = read_json(exp_dir(root, task) / "freeze" / "random_family.json") or {}
    target = cfg["verdict"]["random_target"]
    avail = rf.get("n_available")
    want = target if not isinstance(avail, int) else min(target, avail)
    out.append(Check("s4.manifest.random", n["random"] == want and want > 0, f"{n['random']} random arms, need {want}"))
    out.append(Check("s4.random_family", all(k in rf for k in ("rule", "rng_seed", "n_available")), "random_family.json needs rule, rng_seed and n_available"))
    out.append(Check("s4.manifest.checkpoint_sha", bool(re.fullmatch(r"[0-9a-f]{64}", str(m.get("original_checkpoint_sha256", "")))), "original_checkpoint_sha256 format"))
    return out


def s4_package(root, cfg, task) -> list[Check]:
    f = exp_dir(root, task) / "freeze"
    out = []
    for name in ("autogreedy.json", "random_family.json", "analysis.py"):
        out.append(Check(f"s4.package.{name}", (f / name).exists(), str(f / name)))
    ag = read_json(f / "autogreedy.json")
    if ag:
        out.append(Check("s4.autogreedy.by_script", bool(ag.get("built_by")), "autogreedy.json must name the script that built it in built_by"))
    m, err = _json(f / "test_manifest.json", "s4.manifest")
    if err:
        return out + [err]
    return out + manifest_checks(m, cfg, task, root)


def s4_ledger(root, cfg, task) -> list[Check]:
    return ledger_mod.checks(root, cfg, task)


def gradient_scan(root: Path) -> list[str]:
    hits = []
    for top in ("awe", "experiments"):
        d = root / top
        if not d.is_dir():
            continue
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                code = line.split("#", 1)[0]
                if any(re.search(pat, code) for pat in GRADIENT_PATTERNS):
                    hits.append(f"{p.relative_to(root)}:{n}")
    return hits


def s4_integrity(root, cfg, task) -> list[Check]:
    out = []
    tag, changed = changed_since_genesis(root, cfg)
    out.append(Check("s4.integrity.genesis", tag is not None, "no genesis tag. Run run/setup_box.sh"))
    out.append(Check("s4.integrity.protected", not changed, "protected files differ from " + str(tag) + ": " + ", ".join(changed[:8]) if changed else ""))
    for name in ("PROGRESS.md", "DEVIATIONS.md"):
        ok, removed = appended_only(root, name)
        out.append(Check(f"s4.integrity.{name}", ok, f"{removed} lines were removed from {name} since genesis. These logs are append-only"))
    hits = gradient_scan(root)
    out.append(Check("s4.integrity.no_gradients", not hits, "gradient or optimiser code at " + ", ".join(hits[:8]) if hits else ""))
    stamp = read_json(root / "reviews" / f"{task}-g1-passed.json")
    dec = exp_dir(root, task) / "stage1" / "decisions.json"
    if stamp and dec.exists():
        out.append(Check("s4.integrity.decisions_frozen", stamp.get("decisions_sha256") == sha256_file(dec), "stage1/decisions.json changed after Gate 1. The primary score and operating point are fixed at Gate 1"))
    return out


def s4_request(root, cfg, task) -> list[Check]:
    p = exp_dir(root, task) / "freeze" / "REQUEST.json"
    return [Check("s4.request", p.exists(), "write freeze/REQUEST.json when the package is ready" if not p.exists() else "")]


# ------------------------------------------------------------------ stages 5 to 7

VERDICT_BEGIN = "<!-- referee-verdict-begin -->"
VERDICT_END = "<!-- referee-verdict-end -->"


def s5_report(root, cfg, task) -> list[Check]:
    e = exp_dir(root, task)
    rep = e / "report" / "report.md"
    vmd = e / "test" / "VERDICT.md"
    if not vmd.exists():
        return [Check("s5.report.verdict", False, "the referee has not written test/VERDICT.md yet")]
    if not rep.exists():
        return [Check("s5.report", False, f"missing {rep}")]
    text, v = rep.read_text(errors="replace"), vmd.read_text()
    block = v[v.find(VERDICT_BEGIN) : v.find(VERDICT_END) + len(VERDICT_END)] if VERDICT_BEGIN in v else v
    figs = list((e / "report").glob("**/*.png"))
    return [
        Check("s5.report.verdict_verbatim", block.strip() in text, "report.md must contain the referee's verdict block from test/VERDICT.md, unchanged"),
        Check("s5.report.figures", len(figs) >= 3, f"{len(figs)} figures in report/"),
        Check("s5.report.length", len(text.split()) >= 300, f"{len(text.split())} words"),
    ]


def s6_stress(root, cfg, task) -> list[Check]:
    d, err = _json(exp_dir(root, task) / "stage6" / "stress.json", "s6.stress")
    if err:
        return [err]
    lack = [k for k in ("band_3cm", "rotation_15", "rotation_30") if k not in d]
    return [Check("s6.stress.keys", not lack, "missing " + ", ".join(lack) if lack else "")]


def s7_post(root, cfg, task=None) -> list[Check]:
    p = root / "publish" / "post.md"
    ok = p.exists() and len(p.read_text(errors="replace").split()) >= 400
    return [Check("s7.post", ok, "publish/post.md, a draft of at least 400 words")]


def s7_atlas(root, cfg, task=None) -> list[Check]:
    p = root / "publish" / "atlas" / "index.html"
    return [Check("s7.atlas", p.exists() and p.stat().st_size > 2000, "publish/atlas/index.html, the browsable head atlas")]


def s7_checklist(root, cfg, task=None) -> list[Check]:
    p = root / "publish" / "CHECKLIST.md"
    return [Check("s7.checklist", p.exists() and p.stat().st_size > 200, "publish/CHECKLIST.md, what Andrew does by hand to publish")]


TASK_CHECKS = {
    "s0.env": s0_env,
    "s0.checkpoint": s0_checkpoint,
    "s0.tests": s0_tests,
    "s0.stock": s0_stock,
    "s0.harness": s0_harness,
    "s0.repeat": s0_repeat,
    "s0.renderer": s0_renderer,
    "s0.published": s0_published,
    "s1.design_runs": s1_design_runs,
    "s1.figures": s1_figures,
    "s1.decisions": s1_decisions,
    "s1.diagnosis": s1_diagnosis,
    "s2.atlas": s2_atlas,
    "s2.sweep": s2_sweep,
    "s2.split_half": s2_split_half,
    "s3.hypotheses": s3_hypotheses,
    "s3.dose": s3_dose,
    "s3.validation": s3_validation,
    "s3.baselines": s3_baselines,
    "s3.edit": s3_edit,
    "s4.package": s4_package,
    "s4.ledger": s4_ledger,
    "s4.integrity": s4_integrity,
    "s4.request": s4_request,
    "s5.report": s5_report,
    "s6.stress": s6_stress,
    "s7.post": s7_post,
    "s7.atlas": s7_atlas,
    "s7.checklist": s7_checklist,
}


def run_task(task_id: str, root: Path, cfg: dict, task: str | None) -> list[Check]:
    fn = TASK_CHECKS[task_id]
    try:
        checks = fn(root, cfg, task)
    except Exception as e:  # a malformed file must never crash the status command
        checks = [Check(task_id, False, f"could not check ({type(e).__name__}: {e})")]
    prefix = str(root) + "/"
    for c in checks:  # show paths relative to the repo
        c.detail = c.detail.replace(prefix, "")
    return checks
