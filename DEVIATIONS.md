# Deviations from SPEC.md

Every departure from the plan goes here, with a date and a reason. Append only.

The agent may log a fallback that the Risks table in SPEC.md already allows, or an operational change that does not alter what is tested. Anything that changes a hypothesis, a pass rule, a seed set, a budget, an edit family or the referee needs a person, who records it with `run/setup_box.sh --regenesis "reason"`.

## Entries

None yet.

### 2026-11-23, Stage 0, renderer (Risks table fallback)

GPU rendering (NVIDIA EGL) is not repeatable on this box. Fifty renders of one state gave 4 distinct images, and two runs of the 500 reproduction seeds gave identical trajectories on 137 seeds, with 2 benchmark and 7 strict outcomes flipping. The Risks table names the fallback, software rendering, which is what we use (Mesa llvmpipe through EGL). GPU rendering is slightly closer to the published per-seed file (95.0% and 94.6% agreement in two runs, against 93.8%), so step 8 of Stage 0 alone would have kept it. Repeatability is a gate and closeness is not, so software rendering wins. Evidence in experiments/transfer_cube/stage0/renderer.json and renderer_compare/.

### 2026-11-23, Stage 0, stock evaluation command (operational)

The stock run uses --eval.batch_size=8 with the default asynchronous environments, where SPEC.md shows 50. Software rendering of three cameras takes about 0.25 s per step, and 50 worker processes looked likely to exceed this box (30 GB of RAM, 8 vCPUs). I did not try it. The seeds, the checkpoint, the policy settings and the success rule are unchanged. Batch size changes only the batching of inference, which can flip a borderline seed. The spec's batch-50 command was also run under GPU rendering as a side reference (job stock_gpu).

### 2026-11-24, Stage 0, inference arithmetic and stock evaluation command (operational)

The first official stock run (batch 8, TF32 on, the settings logged in the entry above) gave 81.8% (409 of 500), inside the window, but agreed with the harness on 96.6% of seeds (17 mismatches), below the 98% gate. This entry was written after seeing that number. Diagnosis (scratch/diag_batch.py, scratch/diag_numerics.py, results in experiments/transfer_cube/stage0/numerics/): camera images and joint states at reset are byte-identical between the stock and harness environments, so the difference is in inference. With TF32 on, the first action chunk differs by about 1e-3 between batch sizes and between cudnn.benchmark on and off. With TF32 off it differs by about 1e-6. Running the harness itself with 5 configurations shows the same effect: two runs that differ only in inference numerics disagree on 1% to 4% of seeds.

Changes, applied to the harness and to the stock command alike:
- TF32 off everywhere (full float32). `lerobot-eval` gets `--tf32=false`, the harness sets it in awe/policy.py. The original evaluation command in SPEC.md uses LeRobot's default, which is on.
- The stock command uses `--eval.batch_size=1` (was 8, which was itself a logged change from 50), so the inference batch shape equals the harness's.
- The stock command sets LP_NUM_THREADS=8 (the harness uses 1 per process). Image and state hashes over 120 stock steps on seeds 1000, 1005 and 1010 are identical for 1, 4 and 8 threads (scratch/diag_lp_threads.py), so the pixels do not change.
The earlier stock run and the harness results with TF32 on stay in the repo under stage0/numerics/ (files named tf32on). Nothing about the seeds, the checkpoint, the gates or the success rule changed. All harness files in stage0 were re-made with TF32 off.

### 2026-11-24, correction to the entry above

- "5 configurations" is wrong. There are 4 harness configurations: TF32 on batch 1, TF32 off batch 1, TF32 off with 8 copies and cudnn.benchmark, TF32 on with 8 copies and cudnn.benchmark. Pairwise disagreement among them is 1.2% to 3.8% (agreement 96.2% to 98.8%).
- Images were compared at reset only, on seeds 1000 to 1007. Mid-episode images were not compared between stock and harness with policy actions. That the gap is inference rests on those reset images, the physics test (20 seeds) and the chunk differences. The closed-loop support is indirect: the harness emulation of the stock settings (on8) agrees with the stock run on 98.6% of seeds, against 96.6% for the batch-1 harness.
- The settings were changed after the 96.6% result was seen. The tf32-on versus tf32-off choice was made on the arithmetic evidence (1e-3 against 1e-6), but the agreement figures above were also visible when I chose it.

### 2026-11-24, Stage 1, what "holds the cube" means in strict handover (interpretation, logged after the Gate 1 review)

SPEC.md says strict handover needs the left gripper to hold the cube for 25 steps. The harness (awe/measure.py, since its first version of 2026-11-23, before any design rollout) takes "holds" as contact with any left gripper finger. Requiring contact with both left fingers gives 75.2% strict on the design seeds instead of 78.4%, a gap to benchmark of 6.4 points instead of 3.2. Both gaps are under the 10-point swap rule, so the primary metric (benchmark) and the guard (strict) do not change, and neither does the operating point. The strict rule in the code is unchanged. The stats hold both numbers (stage1_stats.json, strict_variant_both_left_fingers). The reviewer pointed out that this was written in the notebook and the diagnosis but not here.

### 2026-11-24, Stage 3, how the deviation-gain baseline is read (interpretation, written before any baseline rollout)

SPEC.md lists "six gains on the right arm, the left arm, or both" and does not say whether an arm includes its gripper joint. Both readings are tuned (36 settings instead of 18), which can only make this rival stronger. The six gains are 0.8, 0.9, 1.1, 1.2, 1.3 and 1.5, which SPEC.md does not fix either. The mean action is the original's mean executed action per step over the 1,000 design seeds (freeze/mean_action.json, from the stage 1 traces). The constant-offset levels (8 per joint on 10 joints) are mine too.

### 2026-11-24, Stage 3, two extra gains for the deviation-gain baseline (operational, makes the rival stronger)

SPEC.md lists "six gains" for the deviation gain. After screening the six (0.8, 0.9, 1.1, 1.2, 1.3, 1.5) the gain is sharply peaked and 0.9 wins on both arms (right and both, with and without the grippers, the +8.5 and +9.5 settings at 0.9, against about -12 points at 0.8 and -30 at 1.1 on the 200 screening seeds). Such a coarse grid may miss the real optimum of the rival, so for the two best joint sets (both arms without grippers, right arm without its gripper) I also tune g = 0.85 and g = 0.95 (4 settings, 200 screening seeds each, then up to 4 of the top 5 on the other 800, within the 50,000 baseline budget). No existing screening result is rerun or removed, and the rule "top 5 per family by screening gain, best all-1,000 gain frozen" is unchanged. This can only make the deviation-gain rival stronger, so it is a conservative change for the edit, but it departs from the fixed count of six gains and is therefore listed here. It was decided after seeing the group A and B combination results (the edit candidates), which the rival's extra settings do not touch. No baseline was rerun and no edit was changed by it.

### 2026-11-24, Stages 3 and 4, details SPEC.md leaves open (interpretations, none changes a hypothesis, a pass rule, a seed set or a budget)

- Choice among candidates that pass validation. SPEC.md says "pick one" and gives no ranking. The rule (pooled design plus validation net gain, and only candidates with at least 40 matched random edits) was committed before any validation rollout (awe/candidates.py) but after the design results, so it was not blind to them. It chose r4 (pooled net 152) over r6 (150). On validation alone r6 is ahead (+10.4 against +9.2 points). s1 (one head) could not be chosen under the rule.
- Auto-greedy search. SPEC.md says it starts from the six best single-head edits that survived the split-half check, up to 4 heads, and gives no seeds. The script searches on 450 design seeds (20500 to 20949): all 15 pairs, then greedy additions, at most 9,900 rollouts. It was rewritten after the author had seen the design results of the combinations and the validation results of the candidates, and before any rollout of it. The earlier version, never run, started from the top single edit only. Reason in the docstring of awe/autogreedy.py. The rival tunes on fewer seeds than the edit did.
- Dose arms ("the edit's heads at four other scales"). The edit turns one head up and one down, so one common scale makes no sense. The four arms use a strength dial k in 0.5, 1.5, 2, 3 that multiplies each scale's distance from 1, clipped at 0. Some resulting scales (0.75, 0.25, 2.5) are outside the list for edits. They are reported arms and are never candidates.
- Ensembling arms run on all 1,000 test seeds. SPEC.md allows 200 on a CPU-only box, and this box renders in software but has a GPU.
- A third zone in freeze/analysis.py (low corner) was defined after looking at the validation map. It is labelled post hoc there.

### 2026-11-24, Stage 4, the auto-greedy arm is not one the agent never saw (disclosure, after freeze review g2.1 round 1)

This corrects the auto-greedy item in the entry above and the docstring of awe/autogreedy.py, which name only c08 and, generically, the validation results. Numbers are in `relation_to_agent_candidates` of freeze/autogreedy.json, built from files by `python -m awe.freeze disclose`.

- The frozen auto-greedy edit (enc0.h1 x0, enc3.h4 x2) is agent candidate **r2**, op for op. The agent wrote r2 by hand in the refinement round. Its rows and the auto-greedy rows have the same trajectory hash on 450 of 450 search seeds.
- r2 on the 1,000 design seeds: 121 fixed, 22 broken, +9.9 points. These include the 450 search seeds (48 fixed, 8 broken). On the 500 validation seeds: 61 fixed, 23 broken, +7.6 points.
- When each was known. The design numbers (job combo_r, read in session 12) were known before either version of the script. The validation numbers (final files committed in 65513aa, 12:42:00) were known before the version that ran was committed (a58d08c, 12:44:18). The note that asked for a search from all six starts, with "all 15 pairs on 500 seeds" as its example, was committed in 26b2ef9 at 12:28:55, after r2's design numbers were known and while the validation job ran. The change to 450 seeds came with the rewrite.
- The first version (da0c746, never run) had to contain cross.h1.x0, so it could not have returned this pair. On the 450 seeds (its own rule used 500, which were not all run) its best addition ties the single edit at net +20, so a 450-seed run of that rule would have stopped at cross.h1.x0 alone. So the rewrite made this rival stronger.
- The agent had also run another of the 15 pairs on 1,000 design seeds, c07 (cross.h3 x2, enc0.h1 x0): 127 fixed, 53 broken, +7.4. The review did not name it.
- Departure. SPEC.md Stage 4 says the arm is built "so the agent never saw it", and the Controls table uses it as the "you do not need an LLM" control. That does not hold. The search code reads no candidate and no validation file, so the result follows from the sweep and the rule, but the author of the rule knew the result of the pair it returned. **The report must not describe this arm as an independent no-LLM result or as an edit the agent never saw.** The edit, the arm's ops, the manifest and every other arm are unchanged.

### 2026-11-25, Stage 5, side-by-side videos use design seeds (operational)

SPEC.md asks for side-by-side videos on the same seeds in the report. The test seeds may be run only by the referee, so the clips use design seeds (the same seed for the original and the edit in each clip), picked by a fixed rule with rng 0 from the earlier design results. 16 rollouts from the agent budget, purpose "report videos". They illustrate the edit and are not evidence, since the design seeds were used to choose it. Nothing in the freeze changed.

### 2026-11-25, insertion, what strict success and the failure classes mean (interpretation, written before any insertion rollout)

SPEC.md defines the strict score and the classes for transfer cube only and says of insertion "the measurements swap in the peg and the socket" (Stage 6). The definitions, fixed here before any rollout of the insertion policy on this box and before I have looked at any per-seed insertion result (the published success rate, 20.6%, is in SPEC.md):
- Benchmark success is what LeRobot's evaluator reports, stock reward 4 at any step (the peg touches the pin inside the socket).
- Strict insertion, the direct analogue of strict handover. Somewhere in the episode, for 25 steps in a row, the peg touches the pin, the right gripper holds the peg, the left gripper holds the socket, and neither the peg nor the socket touches the table. "Holds" means contact with any finger of that gripper, as for transfer cube (the insertion socket counts as its four walls, socket-1 to socket-4, as in the stock reward).
- Failure class 0 to 3 is the highest stock reward reached (0 nothing, 1 both grippers touch their object, 2 both objects held off the table, 3 peg touches socket off the table). 4a and 4b are reward 4 without and with strict insertion.
The gate rules (10-point swap, guard, operating point) apply unchanged to these two scores. Variants (both fingers, a depth-based version that tolerates contact flicker) may be logged beside them and never replace them.

### 2026-11-25, insertion Stage 0, no published per-seed file exists (operational)

SPEC.md Stage 0 step 4 asks for a seed-by-seed comparison with the published `eval_info.json` and a failure map redrawn from it. The insertion repo's file (lerobot/act_aloha_sim_insertion_human, revision 33259aa8) is byte for byte the transfer-cube file (SHA-256 566d1661..., 83.0% success, video paths under act_aloha_transfer). All 4 earlier commits that carry the file (04be2f0a, 1d723220, 2bd896d0, 33259aa8) hold the same bytes. SPEC.md's "Not checked yet" table already suspected this. So no per-seed insertion result was ever published there. The only published insertion number I know of is the model card's 20.6%.
- The file is stored where CONTRACT.md wants it (it is what the hub serves) and is never used as an insertion result. Any agreement the referee prints between our insertion run and this file is between two different tasks and means nothing.
- No published failure map is drawn for insertion. The failure map comes from our own runs.
- Consequently "agreement with the published per-seed file" cannot inform the insertion renderer choice.

### 2026-11-25, correction to the entry on the insertion published file

"All 4 earlier commits that carry the file" should read "the 4 commits that carry the file, the current one included". The initial commit 3dc6652c has no eval_info.json (404).

### 2026-11-25, insertion Stage 1, the 41 by 41 grid and the ensembling seeds (interpretation, written before any insertion design rollout)

SPEC.md Stage 1 step 3 asks for "an even grid of 41 by 41 cube positions". Insertion has two objects, so the start is a point in four dimensions. I run two grids of 41 by 41 poses each, 1,681 rollouts per grid: the peg on an even grid over its own start box (x 0.1 to 0.2, y 0.4 to 0.6, 10 by 20 cm) with the socket held at the centre of its box, and the socket on an even grid over its box (x -0.2 to -0.1, y 0.4 to 0.6) with the peg at the centre of its box. Both are conditional maps, not marginal ones. The 1,000 design seeds, where both move, are the marginal picture.
The temporal ensembling run uses the first 200 design seeds, 20000 to 20199, which is the minimum the plan allows ("at least 200"). At stock speed it costs about 28 minutes per 200 seeds. On transfer cube all 1,000 were run. If it matters for a later comparison I will extend it in the same file.

### 2026-11-25, insertion Stage 2, the two-grid atlas and the selection rules (interpretation, written before any insertion edit was run beyond a 4-seed smoke test, and before any sweep)

SPEC.md Stage 2 describes one grid of cube positions. Insertion has two objects and two 41 by 41 grids (entry of 2026-11-25, insertion Stage 1), so the rules fixed in the docstring of awe/atlas_ins.py are:
- The local failure rate is the strict (primary) failure, per grid, smoothed with a 1 cm Gaussian. The overlap score of an edit is the Spearman correlation of its step-0 effect map with that failure map, averaged over the two grids. Steps 100 and 200, and maps of class 3, benchmark failure and grasp failure, are secondary numbers and choose nothing.
- The "best 10 by success gain" are by the primary score (strict), ties by benchmark gain then arm id. "Kept" means the same non-zero sign of the strict gain in both halves.
- The atlas frames add step 200 at every pose (transfer cube had it at every second one), because the approach of the target failure is mostly commanded there.
The smoke test was 8 rollouts (the original and cross.h1.x0 on seeds 20000 to 20003, agent budget, purpose smoke). It showed 0 of 4 strict for both. No rule was changed after it.

### 2026-11-25, insertion Stages 3 and 4, scores and the auto-greedy arm (interpretation, written before any validation rollout and before any auto-greedy rollout)

- Scores. SPEC.md Stage 3 writes its validation rule with the failure rate and a guard score. For insertion the primary score is strict success and the guard is benchmark success (decisions.json, Gate 1), so the pass rule (25% of failures removed, exact McNemar p below 0.05, guard drop at most 2 points) is computed on strict with benchmark as guard. The same holds in awe/candidates_ins.py and awe/validate_ins.py, which are copies of the transfer-cube files with those scores.
- List and choice rules. awe/candidates_ins.py holds the list rule (the 8 best combinations by net strict gain on the 1,000 design seeds) and the choice rule. Its docstring says it was written before the group B and refinement results were read. That holds for the text. While checking that the code ran, I ran the `list` command once, and it printed c06's design net gain over 1,000 seeds (110, 37 broken) when its group B rows were done. The rule and N_LIST were not changed after that. The job was still running, so no other result was seen.
- Auto-greedy. awe/autogreedy_ins.py reuses the rule of awe/autogreedy.py (written 2026-11-24, before any insertion rollout) with the strict score in place of benchmark success. Nothing else changes. Its pool and its 450 search seeds (20500 to 20949) are inside seeds on which the author has already seen hand-written combinations, including c06 (cross.h0.x0 with enc3.h3.x0), whose two heads are in its pool. The search can return that pair, in which case the arm is the agent's own candidate and is not an independent result. The freeze states the relation from the files, as for transfer cube. It runs only after the agent's choice exists.

### 2026-11-25, insertion Stage 4, the auto-greedy arm and what the author knew (written before the auto-greedy run, after the validation results)

The search script awe/autogreedy_ins.py was committed in session 25, before any validation rollout, and is run unchanged. Since then I have read the validation results of the agent's candidates (stage3/choice.json, commit of session 26). The choice rule froze r1 (cross.h0.x0, enc3.h3.x0, enc0.h6.x2), with the stop rule. All three of its heads and scales are in the search's pool of six (cross.h0.x0, cross.h2.x0, enc0.h6.x2, enc3.h4.x2, enc3.h3.x0, enc2.h6.x0), and the docstring of awe/autogreedy_ins.py names only c06, so the search can return r1 exactly. I therefore state now, before the run: whatever it returns, the auto-greedy arm of this task is not an independent no-LLM result, because its pool contains the edit that the agent chose by hand and the author knew its validation result when the script was run. The relation to the agent's candidates will be computed from the files (as `relation_to_agent_candidates`, awe/freeze.py) and written in freeze/autogreedy.json. Its result cannot influence the edit, which is already in choice.json.

### 2026-11-25, insertion Stage 4, the dose arms and the two unused validation checks (interpretation, written before any dose, random or auto-greedy arm of the freeze was run)

- Dose arms. SPEC.md Stage 5 asks for the edit's heads at four other scales, and awe/freeze.py (transfer cube) used the dial k in (0.5, 1.5, 2, 3) with scale = max(0, 1 + k (s - 1)). The frozen insertion edit r1 has two heads at x0, and for those every k of 1 or more gives x0 again, so three of those four arms would differ from the edit only in the enc0.h6 scale. For insertion the dial is k in (0.5, -0.5, -1, -2), same formula, ids with m for the minus sign. This was decided and written in PROGRESS.md (session 25) before the validation results were read, for a c06-type edit, and is applied unchanged to r1. For r1 (cross.h0, enc3.h3, enc0.h6) the arms are k 0.5 (0.5, 0.5, 1.5), k -0.5 (1.5, 1.5, 0.5), k -1 (2, 2, 0) and k -2 (3, 3, 0). Negative k is the reverse of the edit. The arms and the edit are all different. It shows nothing between k 1 and k 2, and at k -1 and k -2 enc0.h6 is at x0 in both. No dose arm decides the ending. Only the referee's rules do.
- Validation checks. 8 of the 10 allowed checks were used and no candidate passed. The other 2 are left unused and the agent budget has 6,790 rollouts left. SPEC.md says the stop rule applies when the checks or the rollouts run out, and they did not run out. The reason for stopping here: the bar needs a net of 114 fixes of the 455 strict failures, the best of the 8 reached 41 (r2), the 22 combinations tried on design seeds peak at +11 points, and any further candidate would be built after reading these results. The list rule of awe/candidates_ins.py (8 candidates) was fixed before any validation rollout. I read the rule as a cap, not a quota. The choice of r1 is not affected by this.

### 2026-11-25, correction to the entry on the insertion dose arms and the validation checks

The title of that entry says it was written "before any dose, random or auto-greedy arm of the freeze was run". That is true for dose and random arms. It is not true for the auto-greedy search: job autogreedy_ins had been running since 15:42 UTC (1,400 of its 6,750 pair rollouts were done when the entry was written), and none of its results had been read. The two decisions in the entry (the dose dial, the unused validation checks) do not depend on the search.

### 2026-11-25, insertion Stage 4, the auto-greedy arm equals the agent's candidate c06 (written after the run. The pre-run entry of the same day, 'the auto-greedy arm and what the author knew', is above)

The search awe/autogreedy_ins.py ran after choice.json (8,550 rollouts, 450 search seeds 20500 to 20949, pool of six kept single edits). It returned cross.h0 x0 with enc3.h3 x0 (net +56 of 450: 70 fixed, 14 broken), the best of the 15 pairs. The third head did not raise the net (best addition also +56), so it stopped at two heads. This is, op for op, the agent's candidate c06. The trajectory hashes match on 450 of 450 search seeds. The frozen edit r1 is c06 plus enc0.h6 x2, so the edit and the auto-greedy arm differ by one head. Consequences, all stated in freeze/autogreedy.json (`relation_to_agent_candidates`) and in the freeze request:
- The arm is not an independent no-LLM result and the agent had seen it. The author's pre-run entry already said the result could not be independent. c06's design and validation numbers were known before the search was run (design +11.0 on 1,000 seeds, validation +7.0 on 500 seeds), the search is tuned on 450 of those design seeds, and the report must not present it as a rival the agent never saw.
- On the 500 validation seeds r1 and c06 split 34 (r1 only) to 30 (c06 only), exact McNemar p 0.71 (and r1 against r2 31 to 33, p 0.90). The candidates are not distinguishable from each other there.
- Nothing else changes: the edit, the manifest, the random family, the baselines and the dose arms were fixed before this result or do not depend on it.

### 2026-11-26, insertion Stage 5, side-by-side videos use design seeds (operational, same as transfer cube)

SPEC.md asks for side-by-side videos on the same seeds in the report. Only the referee may run test seeds, so the insertion clips use design seeds, as the transfer-cube clips did (entry of 2026-11-25). The script experiments/insertion/report/videos.py is the transfer-cube script with the insertion paths and the primary score (strict success): 2 fixed class-3 failures, 2 fixed other failures, 2 broken, 1 both succeed, 1 both fail, picked with rng 0 from stage1/design_default.jsonl against r1 in stage3/combo_AB.jsonl.gz. 16 rollouts from the agent budget, purpose "report videos" (agent 93,384 to 93,400). All 16 trajectory hashes equal the earlier rows. The clips illustrate the edit and are not evidence. Nothing in the freeze changed.
