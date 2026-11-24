# Diagnosis, transfer_cube, original policy (LeRobot ACT, revision in stage0/checkpoint.json)

Everything here is recomputed by `python -m awe.stage1_analysis stats` (writes `stage1/stage1_stats.json`), from `stage1/design_*.jsonl`, `stage1/grid_default.jsonl` and the trace cache. The Gate 1 decisions come from `python -m awe.stage1_analysis decisions` and match `python -m referee.recompute decisions --task transfer_cube`. All of it uses design seeds 20000 to 20999 and the 41 by 41 grid of explicit poses. No validation seed has been run.

## Target failure

The default policy (100-step chunks) succeeds on 816 of 1000 design seeds (81.6%, 95% Wilson interval 79.1% to 83.9%) on benchmark success. Strict handover is 78.4%, 3.2 points lower. Max-reward classes: 0: 23, 1: 62, 2: 99, 4a: 32, 4b: 784. Four of the class names in SPEC.md come straight from the stock reward. Two of them do not say exactly what they seem to say.

- Class 0 means the stock touch flag never fired. The flag looks at one finger geom of the right gripper (`10_right_gripper_finger`). In all 23 class-0 design episodes some finger of the right gripper did touch the cube at some step (`class0_contact_by_any_right_finger` in the stats). So "never touched" is a property of the flag, and the episodes are contact failures.
- Class 2 is decided by the stock lift flag, which fires when a right finger touches the cube while the cube is off the table. It also fires when the cube is bumped off the table without being held. I therefore split the 99 class-2 episodes by the highest cube centre height after step 20 (`zmax20`, skipping the reset drop). 41 never exceed 0.06 m, 58 do. Of the 58, 29 drop the cube after lifting and 29 hold it to the end without the left gripper completing the handover.

I label failures by the episode, not by the flag.

- `grasp_fail`: benchmark failure and `zmax20` at most 0.06 m. 126 episodes (68% of the 184 failures): all 23 class 0, all 62 class 1, 41 class 2.
- `lift_fail`: benchmark failure and `zmax20` above 0.06 m. 58 episodes (32%).

The split is not sensitive to the cut. The largest `zmax20` among grasp failures is 0.053 m and the smallest among the 58 lift failures is 0.062 m, so thresholds 0.05 and 0.08 give 125 and 129 grasp failures against 126 at 0.06. The success episodes are clearly lifted (minimum `zmax20` 0.277 m). The cut was chosen by me after seeing the distribution, but it sits in a gap.

The target failure is `grasp_fail`. It is the largest group, 126 of 184. The other candidates are smaller: lift failures 58, and the 32 episodes that are benchmark successes but fail the strict rule (class 4a). A grasp failure is the place where the policy, as I can measure it, goes wrong first. Whether it is fixable by a weight edit is not known yet; the reasons to think it might be are in the near-miss section.

Two things I can measure support "near miss" rather than "skill missing". First, every success reaches a cube-centre height of at least 0.277 m after step 20, while no grasp failure gets above 0.053 m. Second, the arm reaches the cube and touches it, but the first contact is wrong (next section). What I cannot show is that the policy would grasp with a different hand position, so "fixable" remains a hypothesis that Stages 2 and 3 test, and the freeze adds 40 random controls.

## Tracking number

Reach error, as SPEC.md defines it: the vector from the cube centre to the right gripper's centre at the step the right gripper starts to close, minus the mean of that vector over the successful design episodes (mean 0.0516, 0.0015, 0.0539 m). Number of interest: its 3D length.

- Median 1.13 cm in successes (n 816), 3.06 cm in grasp failures (n 126), 2.08 cm in lift failures. The 90th percentile among successes is 2.10 cm, and 80% of grasp failures lie above it. AUC for grasp failure against success is 0.945.
- Reference-vector check (the mean is computed from the same successes, so it is circular). Splitting the design seeds into even and odd, a reference from one half gives AUC 0.930 and 0.957 on the other half. On the 41 by 41 grid (1681 poses, independent of the design seeds; 1366 successes, 202 grasp failures) the design reference gives AUC 0.955. The separation is therefore not an artifact of one sample of seeds.
- What the circularity does not resolve: the number is measured at the first close, after contact in most grasp failures (88% of them touched the cube more than 20 steps before closing, against 21% of successes; the median gap is 45 steps against 1 step). A bump can move the cube and the gripper together. So reach error is a reliable label of a bad grasp, but by itself it does not say why the grasp is bad.
- It also does not separate only grasp failures. Lift failures have a moderately higher median than successes (2.1 cm), AUC 0.754 for lift failure against success.

Position dependence of the reach vector among successes (selection-conditioned, for orientation only): the y component has slope −0.106 per metre of cube y (residual SD 4.2 mm). The x component slopes −0.057 per metre of cube x (residual SD 9.7 mm) and the z component −0.046 per metre of cube y. I do not use this fit as the diagnosis, because it is conditioned on success.

## Near-miss evidence

Measured on all 126 grasp failures and 816 successes, not on hand-picked episodes. A few frame strips (30 failures and 10 successes, seeds in `strips_index.json`) were looked at by eye: I viewed four of them (one class 1, one class-2 not lifted, one class 0, one success). They showed the right gripper at the cube without a lift in the failures and a lift in the success, and they are not used as evidence beyond that. All 40 re-runs match the design-run trajectory hashes.

First right-finger contact, one step before contact, in the gripper frame (so the cube has not moved yet):

| | success | grasp failure | AUC (grasp failure vs success) |
|---|---|---|---|
| absolute lateral offset of the cube, median | 0.8 cm | 2.2 cm | 0.861 |
| gripper height above the cube, median | 5.7 cm | 9.4 cm | 0.898 |

The same numbers on the grid: AUC 0.883 (lateral) and 0.890 (height). The grasp failures also touch the cube long before they close (median 45 steps, 1 in successes), and the cube is slid 3.0 cm (median) before any lift, against 0.37 cm in successes (AUC 0.861 for knock). The two contact numbers are computed one step before the first contact, so they cannot be an effect of the bump on the cube: in all 126 grasp failures the cube moved by at most 2.2 mm (median 0.0, p95 0.07 mm) between step 20 and that step (`pre_touch_cube_shift_mm` in the stats). They are still taken from the same trajectory. These AUCs (0.86 to 0.90) are lower than the reach error AUC (0.945), and they are the more useful evidence because the quantity is fixed before the contact happens.

Wording: this shows that in grasp failures the gripper meets the cube 2 cm off laterally and 4 cm too high, and then pushes it. It does not show a mechanism inside the network, which is Stage 2's work.

## Pattern against the cube position (SPEC.md's diagnostic)

Failures are tied to cube y, not to distance from the centre. Design seeds: failure rate 37.9% for y below 0.44 (206 episodes) against 13.4% above. Grid: 37.5% against 14.2%. AUC for failure by low y is 0.67 (design) and 0.64 (grid). Distance from the centre of the square does not predict failure: AUC 0.43 (design) and 0.49 (grid), and the outermost quartile of distance fails least (8.4% design, 14.2% grid). Failure rate by centre-distance quartile, design: 20.8%, 20.4%, 24.0%, 8.4%; grid: 17.7%, 20.9%, 22.3%, 14.2%. Among successes, reach error rises a little with centre distance (Spearman 0.27) and falls a little with y (−0.17). So the "grows with distance from the centre" explanation is not supported by the failures, and the successes show only a weak trend. The low-y strip (and patches visible on the grid at x 0.07 to 0.09 near y 0.47, at x 0.04 to 0.06 near y 0.51, near x 0.11 to 0.13 near the top edge, and the corners) is the fault tied to position. Reading of the spec's table: "Error only inside one zone, a fault tied to position" for the low-y strip plus patches. Candidate fix: heads active in that zone. Output-level fixes (constant offset, deviation gain) cannot see position, which is why they are the honest rivals and not a substitute.

Other context, same stats file: the nearest design seed (by cube position) also fails in 65.8% of failing episodes, against 8.7% when the episode succeeds (base rate 18.4%). Failure rate by distance to the nearest training-demo cube position (quartiles of distance): 11.2%, 16.4%, 17.2%, 28.8%, AUC 0.62, weak but monotonic. The demo-cube positions (50 found in the first frames; `demo_cubes.json`) are laid over the map in `failure_map.png`.

## Primary metric

Benchmark success. The swap rule (SPEC.md Measurements, RULES.md Gate 1): strict becomes primary only if benchmark minus strict is 10 points or more at the default setting. Here it is 81.6% − 78.4% = 3.2 points, so benchmark stays primary and strict handover is the guard (the edit must not lower it by more than 2 points on the same seeds).

## Operating point

100-step chunks. Rule: 100 unless a chunk of 50 or 25 beats 100 by 3 points or more on the primary score on the design seeds, then the better of the two. Chunk 50: 78.9% (−2.7 points, 96 fixed, 123 broken). Chunk 25: 72.9% (−8.7 points, 91 fixed, 178 broken). Neither passes, so the operating point stays 100. Temporal ensembling at 0.01 (a rival in SPEC.md, not eligible for the operating point) was run on all 1000 design seeds through LeRobot's own ACTTemporalEnsembler in the harness: 70.7% benchmark, 62.5% strict, −10.9 points against default (95 fixed, 204 broken). That path has not been compared to the stock evaluator, so treat the −10.9 as a property of this implementation until the test arms say otherwise.

`proceed` is true: 18.4% of design episodes fail on benchmark (the rule for skipping the task is under 5%).

## What would change this diagnosis

- If a single number on the design seeds separated lift failures better than grasp failures, or if grasp failures were not the largest group on the grid (they are 202 of 315 grid failures), the target would be wrong.
- If the split-half and grid AUCs collapsed, the tracking number would be wrong. They did not (0.930, 0.957, 0.955).
- I have not shown that any head causes this. The evidence above is about behaviour, and Stage 2 is where the claim about heads starts.

## Known weaknesses of this diagnosis

- The reach error is measured after contact in 88% of grasp failures (caveat above). The contact-prior numbers are the cleaner evidence and are weaker.
- The labels and thresholds (0.06 m, `zmax20` after step 20, "early touch" of 20 steps) were chosen by me after looking at the design distributions, not pre-registered. They sit in gaps, but they were not blind. Several numbers were compared against `grasp_fail` in session 4 (about 12) without correction for multiplicity. The reach error is the spec's own definition, which is why it is the tracking number, not the winner of that comparison.
- The design seeds and the grid share one checkpoint and one policy. Positions differ, but the policy is the same one, so the replication is across positions, not across training runs.
- Only the 41 class-2 "not lifted" episodes were treated with a numeric rule; I did not inspect all 41 by frame.

