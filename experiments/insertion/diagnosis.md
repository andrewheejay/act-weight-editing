# Diagnosis, insertion, original policy

Checkpoint: `stage0/checkpoint.json` (hub revision 33259aa8). Numbers come from `python experiments/insertion/stage1/report_ins.py stats`, which writes `stage1/ins_pattern.json`, and from `analysis_ins.py stats <run>` (`ins_stats_<run>.json`), both from the committed per-seed files and `features_*.jsonl.gz`. Key names are in brackets. Rebuilding the features needs the trace cache, which is not in git. Data: design seeds 20000 to 20999 at the default setting, the same seeds at chunks of 50 and 25 and with ensembling (first 200 seeds), and two 41 by 41 grids (1,681 rollouts each, DEVIATIONS.md 2026-11-25). No validation seed has been run. Strict success is the primary score (below), so "failure" here means no strict success.

The grids use the same policy on other positions. They replicate across positions, they are not new policies, and grid points lie millimetres from each other. The chunk runs are the same design seeds.

## Target failure

| Group | What the episode reached (stock reward, highest) | Count of 1000 | Share of the 926 strict failures |
|---|---|---|---|
| S | strict success | 74 | |
| A | class 4a, the peg touched the pin, never strict | 114 | 12% |
| **M (class 3)** | the peg touched the socket off the table, the pin never | **405** | **44%** |
| G2, G1, G0 (classes 2, 1, 0) | no peg-socket contact off the table | 43, 205, 159 (407 together) | 44% |

The target is **class 3, the M group**: 405 of 1000 seeds (40.5%, Wilson 95% interval 37.5% to 43.6%). Benchmark success 18.8% (188, Wilson 16.5% to 21.3%) and strict success 7.4% (74, 5.9% to 9.2%).

Why M and not another group.
- By SPEC's classes it is the largest single class (405 against 205, 159, 114 and 43).
- The grasp classes 0, 1 and 2 together are as large as M (407 against 405) but they are mixtures of causes. Measured: in class 0 (n 159) a left finger touched the socket in 99% and a right finger the peg in 96%, the peg rose above 0.06 m in 59% and the socket in 25%. In class 1 (205) the peg rose above 0.06 in 13% and the socket in 76%. In class 2 (43) 47% and 93% (checked in session 19 notes, `features_default.jsonl.gz`, z > 0.06 is a rough cut). No one arm or step explains them, so an edit for one of them cannot be designed from these data alone.
- Class 4a is a smaller group (114) whose pin contact suggests near success. It is not the same failure as M. The peg-tip misalignment does not separate it from the strict successes (AUC 0.553, Tracking number below), so it is a hold-at-the-pin problem and needs a different fix. Mean `held_steps` medians: S 177.5, A 168, M 166 (not separating either).
- M and A together are the episodes that meet but do not finish (519). They are not merged because the tip number works only for M.

Whether an edit can fix M is unknown. Two reasons for doubt. The scatter (below) is not a constant bias, and the start-position dependence is weak, so there is no clean gain pattern to copy.

## Tracking number

`pre_tip_lat`: one control step before the first peg-socket contact off the table (t_meet - 1), the larger of the two absolute offsets, across the socket axis, of the peg's leading end (the end of the peg nearest the socket centre) from the socket's centre, in the socket frame, in cm. The peg can touch the pin only if its tip gets inside the opening, and the geometry leaves 0.8 cm on each side (half-gap 1.8 cm minus peg half-width 1.0 cm, constants in `awe/measure.py`). Defined only for episodes that meet (593 of 1000 in the design run). In all 593 no peg-socket contact is recorded before t_meet (`first_pegsock` < t_meet in 0).

| Run | AUC M vs S | AUC M vs S and A | AUC A vs S | median tip offset S / A / M (cm) | M within 0.8 cm |
|---|---|---|---|---|---|
| design default (n 74, 114, 405) | 0.843 (bootstrap 95% 0.786 to 0.896) | 0.803 | 0.553 | 0.80 / 0.94 / 2.05 | 4.4% |
| chunk 50 | 0.879 | 0.830 | 0.571 | 0.64 / 0.78 / 1.85 | 6.0% |
| chunk 25 | 0.909 | 0.840 | 0.563 | 0.63 / 0.71 / 1.83 | 11.3% |
| ensembling, 200 seeds | 0.919 | 0.879 | 0.540 | 0.68 / 0.80 / 2.19 | 6.6% |
| grid, peg moved | 0.800 | 0.731 | 0.631 | 0.68 / 1.18 / 1.96 | 3.5% |
| grid, socket moved | 0.658 | 0.607 | 0.568 | 1.34 / 1.51 / 1.59 | 6.8% |

Numbers recomputable with `ins_pattern.json` `tracking`. The last row is the weak spot. With only the socket moved the tip offset barely separates M from S, and the S median there is 1.34 cm, not 0.80. The grid is a conditional slice with the peg always at the centre of its box, so it may miss the configurations where M has large offsets. I did not test that. The tracking number is therefore confirmed on the runs where both objects vary and on the peg grid, and not on the socket grid.

How it was chosen. I computed several numbers on the same design seeds, all in the features file: the centre-based lateral offset (pre_lat, AUC M vs S 0.816), the angle (pre_ang, 0.539), the depth reached after contact (post_min_tip_ax, 0.772), the tip offset (this one, 0.843), and the radial version of the tip offset (0.835). The tip version was chosen on physical grounds (the tip is what has to enter, and the 0.8 cm half-width is a geometric limit) and it also has the highest AUC, so the choice is partly after the fact. It separates M from S, not all failures from S: it says nothing about the grasp classes (which have no meeting) and nothing about class 4a (AUC 0.553 against S).

## Near-miss evidence

Numbers over all of the 405 M episodes against all 74 strict successes.

- Scale. One step before contact, half of the strict successes are within 0.8 cm of the axis (median 0.80), the physical limit, against 4.4% of M (median 2.05, p10 to p90 1.03 to 3.29). 48.9% of M lie above the success p90 of 2.08 cm (about 10% of the S group is by definition above it).
- Direction. The signed offsets across the axis are similar in the two groups, about zero or a half cm in mean (y: S -0.57, M -0.49; z: S 0.13, M -0.09), but the scatter is wider in M (sd 1.80 and 1.78 against 0.95 and 0.97). So it is a wider spread around the target, not a common offset in one direction that an output-level constant bias could remove. (A constant bias in the average is small.) I did not measure whether an individual episode's error is predictable from its start, beyond the position tests below.
- Time course. Both groups start close to each other at 60 steps before contact (5.8 and 5.6 cm median tip offset) and 20 steps before contact are 1.6 and 1.9 cm. In the last 20 steps the successes close in to 0.80, while the M episodes stay at 1.8 to 2.0 (timecourse in `ins_pattern.json`, `time_course_default_tip_lat_cm_median`: S 1.56, 0.90, 0.80, 0.80 and M 1.87, 1.79, 1.95, 2.05 at 20, 10, 5 and 1 steps before contact). I take this as descriptive: M does not converge in the last 0.4 s. I cannot say from it whether the aim is wrong or the grasp already was.
- Depth. After contact, the tip gets at least 5 mm past the socket mouth in 40% of M and at least 15 mm past it in 25% (the mouth is 6.0 cm from the socket centre in tip coordinates, `post_min_tip_ax` is the deepest tip position). For S it is 100% and 81%, for A 100% and 79%. The median deepest tip position of M is 5.8 cm, that is, it stays at the mouth.
- Timing. M meets later than S (median step 304 against 286), and 20% of M meet in the last 50 steps of the 400-step episode (A 17.5%, S 0%). A late meet leaves less time. I did not test how many of the M would succeed with more time.
- Decision age. Meets that come after the step-300 decision (the latest replan) do not have a smaller tip offset in M (2.00 cm against 2.08 cm for meets before step 300). This does not suggest a stale open-loop plan as the cause of M. The chunk-50 result (below) is in the same direction, small.

What this does not show: that a better tip alignment would lead to an insertion (S includes only 50% of episodes within the tolerance, and S can succeed when the tip is 2 cm off one step before contact, because contact can push the peg in), or any mechanism in the network. Near miss here means the tip offset in M is about 2 cm where the successes have about 0.8 (the limit is 0.8), a factor of about 2.5.

## Pattern against start position

Spec table: grows with distance from the centre, one error everywhere, a zone, or no pattern. The data here: mostly scatter with a moderate zone structure, and no growth with distance for the tracking number.

- Distance from the box centre: the tip offset does not depend on it (Spearman 0.01 for the peg, -0.01 for the socket, on the design seeds). The SPEC reach-error test is a little different: the reach deviation of the right gripper at the first close rises with the peg's distance (median by distance fifth, all episodes, 1.64, 1.68, 1.81, 2.11, 2.42 cm, Spearman 0.14, and 0.26 among the 74 strict successes), and of the left gripper with the socket's (1.38 to 2.45, 0.28 and 0.34). The rise is about 1 cm from the centre to the edge. The same size of rise is not visible in the tracking number, and the reach number is a mixture of M, A and G episodes with very different values (median 1.0 cm S, 1.4 M, 2.3 to 3.0 cm in classes 0 and 1), so the gain pattern is at most a part of the picture.
- Position dependence that does exist: smaller peg x goes with a larger tip offset (Spearman -0.29 on design, -0.32 on the peg grid, median by fifth of peg x: 2.4, 2.1, 1.4, 1.4, 1.4 cm) and socket y (-0.11 design, -0.19 on the socket grid, median by fifth 1.9, 1.6, 1.7, 1.8, 1.5 cm design). Moderate and in different directions on the two objects.
- Linear model of the signed tip offset on the four start coordinates: R2 0.09 (y) and 0.05 (z) on design, 0.05 and 0.15 on the peg grid and 0.01 and 0.00 on the socket grid. So the start position explains little of the offset, and most of the scatter is within-position noise.
- Zones (M share of all rollouts in 5 by 5 cells, `ins_pattern.json` `cells`, also drawn in `failure_map.png`). Peg grid: 0% and 8% in the two low-x cells (peg x 0.10 to 0.12 and 0.12 to 0.14, y 0.44 to 0.48), up to 78% in the two middle-x cells at y 0.52 to 0.56 (x 0.14 to 0.18), and 17 to 49% in the top row (y 0.56 to 0.60). In the `failure_map.png` grid the band at peg y 0.46 to 0.50 is mostly the grasp classes, and M appears in most other places. Socket grid: up to 89% at x -0.16 to -0.14, y 0.52 to 0.56, and 5 to 6% in the corner cells with x -0.12 to -0.10 (y 0.56 to 0.60 and y 0.40 to 0.44). In the rightmost column (x -0.12 to -0.10) the M share by y from low to high is 6, 43, 25, 72 and 5%, so there is no single edge pattern.
- Clustering: given a seed in M, its nearest design seed (4-D, scaled by the box sizes) is in M 55% of the time, against a base rate 40.5% and 32% for seeds with a non-M neighbour. For the grasp classes (0 to 2) 62% and 26% (base rate 40.7%), so both are clustered, the grasp classes more.
- So the closest reading of the SPEC table is "error only inside one zone, with several zones, plus scatter without a pattern". The distance pattern is weak and the constant-bias reading is not supported. SPEC says heads active in the zone for the zone case and probably not editable for the last case. I note that the second half of the table is a real possibility here.

Demo overlay: the 50 training demos' first-frame object positions (from `demo_objects.json`, residual rms 0.3 mm, all inside both boxes) are drawn on the maps (`failure_map.png`). I did not quantify failure against distance to the nearest demo.

## Primary metric

Strict success (7.4%). The gap between benchmark success and strict success is 11.4 points (18.8% minus 7.4%), above the 10-point swap rule, so strict becomes primary and benchmark is the guard. `python -m referee.recompute decisions --task insertion` gives the same (strict, 100, proceed). The strict rule is the insertion definition in DEVIATIONS.md (2026-11-25, written before any insertion rollout), not SPEC's cube definition. Its variants on the design seeds are logged beside it: the depth-based version 62 of 1000 and the both-fingers version 22 of 1000, against 74 for the one-finger rule. All of them keep the gap above 10 points (12.6 and 16.6 points), so the decision does not depend on the reading of "holds".

## Operating point

100 steps. Chunk gains on the strict score: chunk 50 +2.8 points (102 against 74 of 1000, 76 fixed and 48 broken), chunk 25 -4.5 points (29). The rule needs +3 on the primary score, so 100 stays. This is close: on the benchmark score chunk 50 would gain exactly +3.0 (218 against 188, 132 fixed, 102 broken), which is the margin. The rule reads the primary score and I follow it as written. Ensembling at 0.01 on the first 200 design seeds (LeRobot's ensembler in my harness, not compared with the stock evaluator) scores 12 strict (6.0%) against 19 on the same 200 at default (9.5%), 6 fixed and 13 broken. Proceed is true: strict fails 92.6%.

## Known weaknesses

- The design seeds are used for the target choice, the tracking number, the thresholds and the classes' reading, with no correction for the number of looks. The grids are the same policy on other positions, and the peg-grid and socket-grid pictures differ, so a claim that holds on one and fails on the other should be treated as weak.
- The tracking number does not cover the 407 grasp failures (about 44% of all strict failures) and does not separate class 4a. If an edit changes the grasp stage, this number will not see it. The reach-error table in `ins_stats_default.json` separates the grasp classes from the rest only moderately (AUC 0.749 and 0.746, right and left) and those classes are mixtures of different causes. They are not target, but if a candidate changes the grasp classes, I should look at them.
- The tip-offset AUC on the socket grid is 0.66 (S median 1.34 cm). I do not know why the socket-moved grid behaves differently. A possible cause is that the peg is always at its box centre there, which I did not check.
- The strict class "S" is n = 74 on the design seeds. Medians and AUCs of S come from 74 episodes (and 12 in the ensembling run).
- A weight edit for M has no obvious mechanism in these numbers: the offset is scatter with a small position effect, and the time course suggests it is set in the last half second before contact. I did not separate arm tracking error, grasp pose and command error, and I did not check which of the chunks (at steps 200 or 300) commands the approach. A deeper decomposition would need the commanded trajectory, which I have not computed.
- "Ceiling": with strict 7.4%, a large win in the verdict rule needs a net fix of 25% of the strict failures of the test set, which is hard to reach by a head edit. I only record this, I do not change any rule.
- Frame strips are low resolution and no claim in this file uses them (`strips_notes.md`).
