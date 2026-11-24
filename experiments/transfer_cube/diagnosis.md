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

AUC grasp_fail against success is 0.945. 80% of grasp failures exceed the success 90th percentile (2.10 cm). By class it is 10 of 23 (class 0), 57 of 62 (class 1) and 34 of 41 (the class 2 episodes inside grasp_fail) [reach_error.share_gf_above_p90_by_class]. So the number tracks the failure less well in class 0. lift_fail against success is 0.754, so it is not specific to grasp failures.

**What reach error measures, and what it does not.** An earlier version of this file said "a real aim error exists in at least 80 of the 126". The count had no support and it is withdrawn. The 80 is the size of the clean subset (below), not a count of aim failures. The reviewer asked for the number of clean-subset failures above the success p90 and p95, and for a contact-free quantity. Both are now computed [contact_free_geometry], on the clean subset: the chunk that is active at the first close was decided before any right-finger contact (80 of 126 grasp failures, 734 of 816 successes, and 134 and 1224 on the grid). The commands of that chunk and of the earlier ones are contact-free there, so the geometry of the COMMANDED gripper (forward kinematics of the joint targets) cannot contain a push or a reaction. The measured gripper can. Reference vectors are the means over the clean successes of the design seeds.

| quantity at the first close | AUC design / grid | grasp_fail median / success median | grasp_fail above success p90, design (8 expected if no difference) | above p95 |
|---|---|---|---|---|
| measured gripper minus untouched cube (aim) | 0.897 / 0.924 | 3.26 / 1.08 cm | 60 of 80 | 53 |
| commanded gripper minus untouched cube | 0.570 / 0.638 | 1.32 / 1.15 cm | 11 of 80 | 7 |

The measured gripper at the close is a median 3.35 cm away from where it was commanded to be, in the grasp failures, against 0.38 cm in the successes. Of that, 2.55 cm is upward (median z, against −0.14 cm in successes). One step before the first contact it was 0.55 cm (z 0.46) against 0.39 (z 0.15) [deflection_at_close, deflection_one_step_before_first_contact, n 69 grasp failures and 406 successes with a contact at or before the close]. The commanded gripper at the close is nearly the same in both groups. I therefore read the 0.897 aim AUC, like the 0.945, as mostly the arm being stopped or lifted by the cube after first contact. Contact starts a median 38 steps before the close in grasp failures and 4 in successes (65 of 80 more than 20 steps). On the grid the same holds (32 of 134 commanded above p90, 13.4 expected, against 111 of 134 measured). A natural split supports the reading: of the 80 clean grasp failures, 68 touched the cube before the close and 12 did not. The 68 have a measured aim error of 3.64 cm (median) and a 3.60 cm deflection, and 85% exceed the success p90. The 12 have 1.69 cm and 0.79 cm, and 17% do (2 of 12). On the grid the split is 124 and 10, with the same direction for the 124 (3.77 cm, 85%). For the 10 on the grid without contact the commanded error itself is 2.4 cm (median, deflection 0.5 cm), so some of those are real command misses. The sample is small and the two sets disagree on it, so I draw nothing from it beyond the main split [contact_free_geometry.*.grasp_fail_by_contact_before_close].

**A contact-free tracking number next to it.** The commanded gripper shows a smaller, real difference in the lateral direction (along the finger axis, gripper frame, offset of the untouched cube from the commanded gripper centre). I take it at the first step from step 20 on where the commanded gripper is less than h above the cube, so failures and successes are compared at the same height. Deviation from the clean-success mean, design seeds (grid in brackets):

| commanded height above the cube | success median | grasp_fail median | AUC | grasp_fail above success p90 (8 or 13.4 expected if like success) |
|---|---|---|---|---|
| 12 cm | 0.94 (0.88) cm | 1.76 (1.89) | 0.75 (0.81) | 33 of 80 (60 of 134) |
| 10 cm | 0.71 (0.66) | 1.48 (1.74) | 0.78 (0.82) | 32 (71) |
| 8 cm | 0.65 (0.60) | 1.51 (1.54) | 0.80 (0.81) | 34 (70) |
| 6 cm | 0.55 (0.52) | 1.22 (1.35) | 0.76 (0.80) | 34 of 76 (67 of 119) |
| the close step | 0.54 (0.53) | 0.97 (1.11) | 0.72 (0.74) | 30 (54) |

Grasp failures are also shifted on average to one side: mean signed deviation +0.3 to +0.9 cm across the heights, and 61 to 76% of them on the same side (signed AUC 0.63 to 0.69 on design, 0.59 to 0.68 on grid). Forward offset and yaw of the finger axis do not separate the groups (AUC about 0.52 to 0.61 and 0.42 to 0.56, `python -m scratch.review1_phase default grid`).

Selection, stated openly. These numbers were computed after the review and after seeing the failure labels, on the same seeds, with the grid as a replication that has the same policy (not a new one). Four heights, the close step, two sets of 3D numbers and the forward and yaw offsets were looked at. I keep 8 cm as the companion number because it is the lowest height that no episode misses, and it also has the highest design AUC of the four heights (0.80), so its choice is partly after the fact. The grid gives 0.81 for it. An earlier version of the lateral number, restricted to the active chunk, gave 0.87 at 10 cm but only for 225 of 814 episodes, and was dropped for the full-history version above, which loses no episode at 8 cm and above.

So the diagnosis is this. The tracking number stays reach error, since SPEC.md defines it, with the caveat that in the clean subset most of it is contact deflection, not aim. Stage 3 reports beside it `cmd_lat_h[0.08]`, the commanded lateral offset at 8 cm (design AUC 0.80, grid 0.81), which an edit can only move by changing where the policy aims. An edit that lowered reach error only by pushing less would not move this number.

## Near-miss evidence

Numbers over the grasp failures, not chosen episodes. What bears on a near miss, and what does not.

- Every one of the 126 grasp failures touched the cube with a right finger.
- The commands the policy gave at the close are close to the successes in size of error. In the clean subset the commanded gripper is 1.32 cm (median) from the success mean, against 1.15 cm for successes, and 11 of 80 are beyond the success p90 (8 expected). The planned final position is not far off for most failures, and the grasp still fails.
- The contact-free lateral miss before contact is larger. At 8 cm above the cube the median is 1.5 cm in the grasp failures against 0.65 cm in the successes, 0.9 cm more (0.8 cm at 12 and 10 cm, 0.7 at 6 cm, 0.4 at the close). 34 of 80 exceed the success p90 (1.59 cm), where 8 would by chance, so about 40% of the clean grasp failures lie outside the range that contains 90% of the successes. The other 60% overlap the successes on this number, so it does not explain every failure. The cube is 4 cm wide. No tolerance was measured, so I do not claim that a lateral offset of 1 cm is the cause.
- First contact is early and the arm is then held off. In the grasp failures contact starts a median 38 steps before the close (successes 4), and at the close the gripper is 3.35 cm (median) away from its commanded position (0.38 cm in successes). The reading "a finger lands on the cube during the descent" is an inference. I did not identify which finger or which face of the cube.
- First-contact geometry, measured one step before contact, has the same direction: |lateral offset| 2.2 cm against 0.8 cm (AUC 0.861, grid 0.883). The groups are compared at different phases, because over all episodes successes first touch a median 1 step before closing and failures on the way down (45 steps before), so I do not read the height difference (9.4 against 5.7 cm) as aiming too high.

What I removed after the review. "The typical miss is about 1 cm past what still works" compared a failure median with a success p90 of reach error, which is partly the bump, and no tolerance was measured. "34% of failures have a succeeding nearest neighbour" was evidence for the opposite of a near miss. Given a failure, the nearest seed also fails 65.8% of the time, against a base rate of 18.4% (8.7% after a success) [neighbours]. That is clustering and it is in the Pattern section.

The sanity check that the cube has not moved before contact (max 2.2 mm in grasp failures, median 0) is true almost by construction and is not used as near-miss evidence [pre_touch_cube_shift_mm].

This is a centimetre-scale miss in the command. It does not show that a corrected aim would be followed by a grasp, nor any mechanism inside the network.

## Pattern against cube position

SPEC.md's diagnostic is reach error against start position. Over all episodes [error_vs_centre_distance]:

| distance from centre (cm) | 0-4 | 4-6 | 6-8 | 8-10 | 10+ |
|---|---|---|---|---|---|
| reach error, median, all (cm) | 1.16 | 1.23 | 1.15 | 1.20 | 1.58 |
| reach error, median, successes (cm) | 1.00 | 1.02 | 1.05 | 1.07 | 1.52 |
| failure rate | 21% | 21% | 20% | 23% | 8% |

- **Grows with distance from the centre.** Reach error does rise, only beyond 10 cm, where failures are rarest. Regressing aim on cube position over all episodes gives −0.042 in x and −0.069 in y, an under-reaction of 4% and 7%, or 0.4 and 0.7 cm at the edge. Present, and small beside a 3 cm miss.
- **Centre pull.** Among the 816 successes, the mean gripper deviation per 4 cm cell points toward the centre of the square in all 4 corner cells (projection 0.6 to 1.9 cm, mean 1.3) and in 11 of 12 edge cells (mean 0.6 cm) [pattern.centre_pull_success_only]. In the corners the arrows in `reach_error.png` show the same. The pull is about 1 cm, and the corner cells fail 20% (bottom left), 19% (bottom right), 2% and 0% (top), so a gain pattern is at most part of the story.
- **Zone.** Failure rate is 37.9% for y below 0.44 against 13.4% above (grid 37.5% and 14.2%). But 106 of the 184 design failures lie outside that strip, and the grid map shows other patches (around x 0.08 y 0.48, x 0.05 y 0.515, x 0.12 at the top edge, and the corner at low x).

- **Clustering.** Given a failing seed, the nearest design seed fails 65.8% of the time against 18.4% overall, and 8.7% given a success. This is the zone reading, and it is not evidence for a near miss. Within the hot block (x 0.08 to 0.16, y 0.40 to 0.48) 77 of 159 seeds fail (48%), of which 64 are grasp failures, and the block holds 42% of all 184 failures [zones.block_x0.08_0.16_y0.40_0.48]. By 4 cm cells, the bottom row fails 20, 19, 68, 55 and 19% from x 0 to 0.2 m [zones.grid5x5_failrate].

Best match is the zone pattern, with several zones. The distance pattern is weak, not refuted. SPEC suggests heads active in the zone for this pattern, and a gain on the action's deviation for the other. Which wins is H3's question and no baseline has been run. Failure rate also rises with distance to the nearest training cube (11%, 16%, 17%, 29% by quartile, AUC 0.62), weakly and confounded with zone.

## Primary metric

Benchmark success. Benchmark 81.6% minus strict 78.4% is 3.2 points, under the 10-point swap rule. Strict is the guard. "The left gripper holds the cube" is implemented as contact with either left finger. Requiring both gives 75.2%, a gap of 6.4 points and the same decision [strict_variant_both_left_fingers]. I watched 10 successes, all class 4b, so I have no evidence either way on whether the rule misses good 4a episodes, and I have not changed it. The reading of "holds" is logged in DEVIATIONS.md (2026-11-24).

## Operating point

100 steps. Chunk 50 scores 78.9% (−2.7 points, 96 fixed, 123 broken) and chunk 25 scores 72.9% (−8.7, 91 fixed, 178 broken). The rule needs +3. `proceed` is true (18.4% fail). Temporal ensembling at 0.01 scores 70.7% on 1000 seeds (−10.9, 95 fixed, 204 broken). It runs LeRobot's ensembler in my harness and has not been compared with the stock evaluator, so the size of that drop is unverified.

## Known weaknesses

- Cuts (0.06 m, step 20, 20 steps early touch) were set after seeing the design data.
- About 12 numbers were compared against `grasp_fail` in session 4, the aim error in two forms in session 5, and the 3D commanded error, the lateral command at four heights, the close, forward offset and yaw in the review response, all on the same seeds with no correction. Reach error is the tracking number because SPEC.md names it. The grid repeats the numbers on other positions with the same policy, not a new one.
- The label `lift_fail` includes episodes that look like grasp failures in the strips (20674, 20961).
- 46 of the 126 grasp failures are outside the clean subset. For those I cannot separate aim from bump at all.
- The contact-free geometry uses the commanded joint targets, so anything upstream of the command (the perception of the cube) is included in it, and tracking lag before contact (0.4 to 0.6 cm) is not removed from the measured gripper.
- The contact direction and the finger that touches first are not identified. The "lands on the cube" reading is inferred from height, z deflection and timing only.
- The figure `failure_map.png` now has a third panel by SPEC class. The other two panels use the labels `grasp_fail` and `lift_fail` defined above.
