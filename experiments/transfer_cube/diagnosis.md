# Diagnosis, transfer_cube, original policy

Checkpoint: `stage0/checkpoint.json` (hub revision and sha256). Numbers come from `python -m awe.stage1_analysis stats`, which writes `stage1/stage1_stats.json` from the committed per-seed files and `features_*.jsonl.gz`. Key names are in brackets. Rebuilding the features needs the trace cache, which is not in git. Data: design seeds 20000 to 20999 and the 41 by 41 grid of poses. No validation seed has been run.

The grid uses the same policy on other positions in the same square. It replicates across positions. It is not a new policy, and grid points lie millimetres from design seeds, so it is not a statistically independent sample.

## Target failure

The default policy succeeds on 816 of 1000 design seeds (81.6%, Wilson 95% interval 79.1% to 83.9%). Max-reward classes: 0: 23, 1: 62, 2: 99, 4a: 32, 4b: 784.

**The target is `grasp_fail`**: benchmark failure and the cube centre never above 0.06 m after step 20. It holds 126 of 184 failures (68%): 23 class 0, 62 class 1 and 41 of the 99 class 2. The other 58 are `lift_fail`.

Why not SPEC.md's classes as they stand.

- Class 2 only means the stock lift flag fired once. 41 class-2 episodes never exceed 0.053 m (median peak 0.028 m, resting height 0.02 m). I read these as bumps or tilts. That is an inference from the heights.
- Class 0 means the stock touch flag never fired. That flag watches one finger. In all 23 the other right finger touched the cube [class0_contact_by_any_right_finger].

The 0.06 m cut was chosen by me after seeing the distribution. Few episodes sit near it: the count is 125, 126 and 129 at 0.05, 0.06 and 0.08 [label_cut].

Is another group bigger or a clearer near miss?

- On SPEC's classes, class 2 (99) is bigger than class 1 (62). The target depends on my split of class 2.
- `lift_fail` is mixed. 24 of 58 first lift after step 250, against a success 95th percentile of 212, so many are failed first grasps that ran out of time [lift_fail_lateness]. 12 of the 34 with a left close are real handover near misses (cube within 10 cm in z, 6.2 cm off in xy, median). In the other 22 the z error is above 10 cm [lift_fail_meet].
- On the grid, grasp failures are 202 of 315 failures.

Whether grasp failures can be fixed by a weight edit is unknown.

## Tracking number

Reach error as SPEC.md defines it: right gripper centre minus cube centre at the first right close, minus the mean of that vector over successes. Length in 3D.

| | success | grasp_fail | lift_fail |
|---|---|---|---|
| median | 1.13 cm | 3.06 cm | 2.08 cm |

AUC grasp_fail against success is 0.945. 80% of grasp failures exceed the success 90th percentile (2.10 cm). lift_fail against success is 0.754, so the number is not specific to grasp failures.

**Some of 0.945 is the bump, not the aim.** 88% of grasp failures touch the cube more than 20 steps before closing (21% of successes), and the cube has then moved. I made three checks, after seeing 0.945.

| Check | Design | Grid |
|---|---|---|
| Reference from one half of the seeds, tested on the other | 0.930, 0.957 | 0.955 (design reference) |
| Aim error: same gripper position against the cube's untouched position | 0.890 | 0.901 |
| Aim error, only episodes whose active chunk was decided before any contact | 0.898 (80 grasp_fail, 734 success) | 0.923 (134, 1224) |

In the last row the bump could not have influenced the decision. The medians there are 3.2 cm (grasp_fail) and 1.1 cm (success) on the design seeds. So a real aim error exists in at least 80 of the 126, and the honest separation for aim is about 0.90. An edit that only reduces bumping would also lower reach error, so Stage 3 should report aim error beside it.

## Near-miss evidence

All numbers are over every grasp failure, not chosen episodes.

- Every one of the 126 touched the cube with a right finger, and the cube had moved at most 2.2 mm (median 0) by the step before that contact [pre_touch_cube_shift_mm].
- Size of the miss: the median grasp failure has a reach error of 3.06 cm, and 2.10 cm is the success 90th percentile, so the typical miss is about 1 cm past what still works. The cube is 4 cm wide.
- Failures sit next to successes: when a seed fails, the nearest seed fails 66% of the time, so 34% of failures have a succeeding nearest neighbour (8.7% of successes have a failing one) [neighbours].
- Many grasp failures retry or go through the handover motion with an empty gripper (`strips_notes.md`, all 40 strips looked at, low resolution).

**First-contact numbers, with a confound.** At first right-finger contact the lateral offset is 2.2 cm for grasp failures against 0.8 cm (AUC 0.861, grid 0.883), and the height above the cube 9.4 against 5.7 cm (AUC 0.898, grid 0.890). Successes first touch while closing (median 1 step before), failures on the way down (45 steps before). The two groups are compared at different phases. The height difference mostly says that, and is not evidence of aiming too high.

This shows a centimetre-scale miss. It does not show that a correct aim would have been followed by a grasp, nor any mechanism inside the network.

## Pattern against cube position

SPEC.md's diagnostic is reach error against start position. Over all episodes [error_vs_centre_distance]:

| distance from centre (cm) | 0-4 | 4-6 | 6-8 | 8-10 | 10+ |
|---|---|---|---|---|---|
| reach error, median, all (cm) | 1.16 | 1.23 | 1.15 | 1.20 | 1.58 |
| reach error, median, successes (cm) | 1.00 | 1.02 | 1.05 | 1.07 | 1.52 |
| failure rate | 21% | 21% | 20% | 23% | 8% |

- **Grows with distance from the centre.** Reach error does rise, only beyond 10 cm, where failures are rarest. Regressing aim on cube position over all episodes gives −0.042 in x and −0.069 in y, an under-reaction of 4% and 7%, or 0.4 and 0.7 cm at the edge. Present, and small beside a 3 cm miss.
- **Zone.** Failure rate is 37.9% for y below 0.44 against 13.4% above (grid 37.5% and 14.2%). But 106 of the 184 design failures lie outside that strip, and the grid map shows other patches (around x 0.08 y 0.48, x 0.05 y 0.515, x 0.12 at the top edge, and the corner at low x).

Best match is the zone pattern, with several zones. The distance pattern is weak, not refuted. SPEC suggests heads active in the zone for this pattern, and a gain on the action's deviation for the other. Which wins is H3's question and no baseline has been run. Failure rate also rises with distance to the nearest training cube (11%, 16%, 17%, 29% by quartile, AUC 0.62), weakly and confounded with zone.

## Primary metric

Benchmark success. Benchmark 81.6% minus strict 78.4% is 3.2 points, under the 10-point swap rule. Strict is the guard. "The left gripper holds the cube" is implemented as contact with either left finger. Requiring both gives 75.2%, a gap of 6.4 points and the same decision [strict_variant_both_left_fingers]. I watched 10 successes, all class 4b, so I have no evidence either way on whether the rule misses good 4a episodes, and I have not changed it.

## Operating point

100 steps. Chunk 50 scores 78.9% (−2.7 points, 96 fixed, 123 broken) and chunk 25 scores 72.9% (−8.7, 91 fixed, 178 broken). The rule needs +3. `proceed` is true (18.4% fail). Temporal ensembling at 0.01 scores 70.7% on 1000 seeds (−10.9, 95 fixed, 204 broken). It runs LeRobot's ensembler in my harness and has not been compared with the stock evaluator, so the size of that drop is unverified.

## Known weaknesses

- Cuts (0.06 m, step 20, 20 steps early touch) were set after seeing the design data.
- About 12 numbers were compared against `grasp_fail` in session 4, and the aim error in two forms in session 5, on the same seeds with no correction. Reach error is the tracking number because SPEC.md names it.
- The label `lift_fail` includes episodes that look like grasp failures in the strips (20674, 20961).
- 46 of the 126 grasp failures are outside the clean subset, so for those I cannot separate aim from bump.
