# The contract between the agent's code and the referee

The agent writes the harness in `awe/` and the experiment files under `experiments/<task>/`. The referee reads those files and calls two commands. This page is the full list of what it expects. `<T>` means `experiments/<task>`, and `<task>` is `transfer_cube` or `insertion`.

A file named `X.jsonl` may also be stored as `X.jsonl.gz`.

## Two commands the harness must provide

### `python -m awe.run_arm`

```
python -m awe.run_arm --task <task> --arm <arm id> --seeds-file <json> --out <jsonl>
                      [--manifest <test_manifest.json>] [--workers N]
```

- `--seeds-file` holds `{"seeds": [int, ...]}`.
- It appends one record per seed to `--out` and skips seeds already there, so an interrupted run can resume.
- The arm `original` must work with no manifest. Any other arm is looked up by id in the manifest, and its recipe must be fully determined by files in the repo.
- The same seed and arm must give the same record every time, on this box.
- Checkpoints are found under `$AWE_CHECKPOINTS` when it is set, otherwise under `./checkpoints`. Caches go under `$AWE_CACHE` when it is set.
- When `AWE_REFEREE=1` it writes nothing except `--out`. In particular it does not write to the ledger.
- It sets the renderer itself, from the choice recorded in `stage0/renderer.json`, before it imports MuJoCo. It must give the same result whatever `MUJOCO_GL` the caller has set.
- Seed lock. A seed at or above 1,000,000,000 is refused with an error unless `AWE_JUDGE_DRAW` points at a draw file and the seed lies inside that file's `block.start` to `block.end`.

### `python -m awe.export_checkpoint`

```
python -m awe.export_checkpoint --task <task> --arm <arm id> --manifest <json> --out <dir>
```

Writes a checkpoint folder that `lerobot-eval --policy.path=<dir>` can load, with the weights in `<dir>/model.safetensors`. The folder's config must carry the arm's operating point (`n_action_steps`). It must work for every arm whose recipe is a pure weight edit, and at least for the roles `original`, `agent_edit` and `autogreedy`.

## Per-seed records

One JSON object per line.

| Key | Type | Notes |
|---|---|---|
| `seed` | int or null | null only for rollouts at an explicit pose, which then carry `pose` |
| `arm` | string | `original` for the unedited policy |
| `benchmark_success` | bool | exactly what LeRobot's evaluator calls success |
| `strict_success` | bool | the strict score defined in SPEC.md, Measurements |
| `traj_hash` | string | a hash of the full joint trajectory. Required wherever runs are compared for identity |
| `max_reward`, `failure_class` | optional | used in reports |

Add any other measurements you like.

## Ledger

Files `ledger/*.jsonl`, one line per rollout the agent's code ran.

| Key | Notes |
|---|---|
| `task` | the task name |
| `seed` | int, or null with a `pose` key |
| `arm` | arm or edit id |
| `budget` | one of `agent`, `baselines`, `autogreedy`, `setup`, `stress` |
| `candidate` | required on validation seeds. Use `original` for the unedited policy |
| `purpose` | free text, such as `sweep` or `dose` |

Reproduction-seed rollouts use the budget `setup`. Stage 6 uses `stress`.

## Stage 0 files, in `<T>/stage0/`

| File | Content |
|---|---|
| `env.json` | keys `lerobot`, `gym_aloha`, `mujoco`, `torch`, `python`, `renderer`, `rollouts_per_second`, `stock_seconds_per_episode`, `stock_eval_cmd` |
| `checkpoint.json` | `dir` (such as `checkpoints/transfer_cube`), `model_sha256`, `source` (the hub repo id). Optional `model_file` |
| `stock_eval/eval_info.json` | the output of one `lerobot-eval` call with `--seed=1000 --eval.n_episodes=500`, saved as LeRobot wrote it. LeRobot 0.7.0 stores no seeds, so the referee reads episode i as seed 1000 + i, and Gate 0 checks that reading by re-running 5 seeds |
| `published_eval_info.json` | the per-episode file from the hub |
| `harness_repro.jsonl` | the fast harness on seeds 1000 to 1499, arm `original` |
| `mismatches.md` | one line per seed where the harness and the stock run disagree |
| `repeat_a.jsonl`, `repeat_b.jsonl`, `sham.jsonl` | the same 100 or more seeds, run twice, and once with every head scaled by 1 |
| `renderer.json` | `chosen`, plus whatever comparison supports it |

`stock_eval_cmd` is the exact command that worked, as a list of strings with the placeholders `{policy_path}`, `{env_task}`, `{n_episodes}`, `{seed}` and `{output_dir}`. The referee fills them in and reuses the command. If the command needs an environment variable, put it in the list, as in `["env", "MUJOCO_GL=osmesa", "lerobot-eval", ...]`. For example

```json
["lerobot-eval", "--policy.path={policy_path}", "--env.type=aloha", "--env.task={env_task}",
 "--eval.n_episodes={n_episodes}", "--eval.batch_size=50", "--policy.device=cuda",
 "--policy.use_amp=false", "--seed={seed}", "--output_dir={output_dir}"]
```

The five tests live in `tests/` and are named `test_fast_harness_matches_stock`, `test_head_scaling_exact`, `test_sham_edit_identical`, `test_repeat_run_identical` and `test_seed_lock`.

## Stage 1 files

| File | Content |
|---|---|
| `<T>/stage1/design_default.jsonl` | seeds 20000 to 20999 at 100-step chunks |
| `<T>/stage1/design_chunk50.jsonl`, `design_chunk25.jsonl` | the same seeds at 50 and 25 |
| `<T>/stage1/design_ens.jsonl` | at least 200 design seeds with temporal ensembling at 0.01 |
| `<T>/stage1/failure_map.png`, `reach_error.png`, `strips/` | figures |
| `<T>/stage1/decisions.json` | see below |
| `<T>/diagnosis.md` | sections named Target failure, Tracking number, Near-miss evidence, Primary metric, Operating point |

```json
{"primary_metric": "benchmark", "operating_point": {"n_action_steps": 100}, "proceed": true,
 "target_failure_class": "1", "tracking_number": "reach_error"}
```

## Stage 2 files, in `<T>/stage2/`

| File | Content |
|---|---|
| `atlas/heads.json` | one entry per head, 48 in all, either a list or `{"heads": [...]}` |
| `sweep.jsonl` | seeds 20000 to 20499 for the arm `original` and for 96 single-head edits, each with its own arm id |
| `split_half.json` | `rechecked` (at least 10 edit ids re-run on seeds 20500 to 20999) and `kept` |

## Stage 3 files

| File | Content |
|---|---|
| `<T>/hypotheses.md` | one line per candidate, containing the candidate id, written before it is evaluated |
| `<T>/stage3/dose_response.json` | the tracking number at each scale |
| `<T>/stage3/validation/original.jsonl` | the original on seeds 30000 to 30499 |
| `<T>/stage3/validation/<candidate>.jsonl` | one file per candidate checked |

## Freeze package, in `<T>/freeze/`

`edit.json`

```json
{"task": "transfer_cube", "candidate": "c07", "family": "head_scaling",
 "ops": [{"op": "scale_head", "block": "model.encoder.layers.2.self_attn", "head": 5, "scale": 2}],
 "free_numbers": 0, "hypothesis": "One paragraph on why this should fix the target failure.",
 "edited_checkpoint_sha256": "<sha256 of the model.safetensors that export_checkpoint writes>"}
```

Op kinds are `scale_head` (with `block`, `head`, `scale`), `sharpen_head` (with `block`, `head`, `factor`), `latent_shift` (with `direction` and `size`), and `direction_edit` or `other` (with a `description`, and `free_numbers` of 3 or fewer).

`baselines.json` is `{"baselines": [{"family": "constant_offset", ...}, ...]}` with at least the families `constant_offset`, `deviation_gain`, `shorter_chunk` and `simplest_fix`.

`autogreedy.json` holds the edit a script built, with `built_by` naming that script.

`random_family.json` holds `rule` (text), `rng_seed` (int) and `n_available` (how many distinct matched edits exist).

`test_manifest.json`

```json
{"task": "transfer_cube", "primary_metric": "benchmark", "operating_point": {"n_action_steps": 100},
 "original_checkpoint_sha256": "<sha256>",
 "arms": [
   {"id": "original", "role": "original"},
   {"id": "agent_edit", "role": "agent_edit"},
   {"id": "autogreedy", "role": "autogreedy"},
   {"id": "random_00", "role": "random"},
   {"id": "base_offset", "role": "baseline", "family": "constant_offset"},
   {"id": "dose_x0.5", "role": "dose"},
   {"id": "original_ens", "role": "ensembling_original", "n_seeds": 200},
   {"id": "agent_edit_ens", "role": "ensembling_edit", "n_seeds": 200}
 ]}
```

Roles are `original`, `agent_edit`, `original_default`, `agent_edit_default`, `autogreedy`, `random`, `baseline`, `dose`, `ensembling_original` and `ensembling_edit`. Arm ids use letters, digits, dot, dash and underscore. An arm may carry `n_seeds` to run on fewer test seeds, which SPEC.md allows for the slow ensembling arms on a CPU box. Each arm may carry any extra keys its recipe needs.

`analysis.py` builds the descriptive tables and figures and has already been run on validation data.

`REQUEST.json` asks for the freeze. Any JSON object will do, for example `{"task": "transfer_cube", "note": "ready"}`.

## What the referee writes

You never write these.

| File | Content |
|---|---|
| `reviews/<task>-g0-deep.json` | the latest Gate 0 replay check |
| `reviews/<task>-<gate>-r<n>.json` | a review |
| `reviews/<task>-g1-passed.json` | the Gate 1 record |
| `<T>/freeze/ACCEPTED.json` or `REJECTED-<n>.md` | the outcome of a freeze request |
| `<T>/test/commitment.json`, `draw.json` | the public commitment and the seed draw |
| `<T>/test/arms/<arm id>.jsonl` | per-seed results for every arm |
| `<T>/test/verdict.json`, `VERDICT.md` | the verdict |

You do write `reviews/<task>-<gate>-r<n>-response.md` when a review asks for one.

## After the verdict

| File | Content |
|---|---|
| `<T>/report/report.md` | contains the block between the two `referee-verdict` markers in `VERDICT.md`, unchanged. At least 300 words and 3 figures under `<T>/report/` |
| `<T>/stage6/stress.json` | main task only. Keys `band_3cm`, `rotation_15`, `rotation_30` |
| `publish/post.md`, `publish/atlas/index.html`, `publish/CHECKLIST.md` | the publishing package |
