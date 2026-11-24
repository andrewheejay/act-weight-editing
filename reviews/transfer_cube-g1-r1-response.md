# Response to the Gate 1 review of transfer_cube, round 1

Both blocking issues are right. The diagnosis is changed (`experiments/transfer_cube/diagnosis.md`, sections Tracking number, Near-miss evidence, Pattern, Known weaknesses). The numbers asked for are computed by `python -m awe.stage1_analysis features default|grid` and `stats`, and they are in `stage1_stats.json` (`contact_free_geometry`, `reach_error.share_gf_above_p90_by_class`, `zones.block_x0.08_0.16_y0.40_0.48`, `pattern.centre_pull_success_only`). No rollout was run for this.

## Issue 1, "a real aim error in at least 80 of the 126"

Right, and the sentence is withdrawn. 80 was the size of the clean subset, not a count of aim failures. The clean subset only excludes the network reacting to contact. It does not exclude the arm being pushed. The requested checks, on the clean subset (80 grasp failures, 734 successes, grid 134 and 1224), with reference vectors from the clean successes:

- **Measured aim error at the first close** (gripper minus the untouched cube): 60 of 80 grasp failures exceed the success p90 (2.11 cm), 53 exceed p95 (2.52 cm). Grid 111 and 107 of 134. Medians 3.26 cm against 1.08 cm.
- **Commanded gripper** (FK of the commanded right-arm joints at the close, contact-free in this subset): 11 of 80 above the success p90 (8 expected if the groups did not differ), 7 above p95. AUC 0.570 (grid 0.638), medians 1.32 against 1.15 cm. Grid 32 of 134 above p90 (13.4 expected).
- **Measured minus commanded gripper at the close**: median 3.35 cm in grasp failures, 0.38 cm in successes. In grasp failures the z component is +2.55 cm (the reach_rel z of 0.076 to 0.087 against 0.054 that the review names). One step before the first contact it was 0.55 cm against 0.39 cm (n 69 grasp failures and 406 successes with a contact at or before the close). So the deflection arises after first contact.
- A split inside the 80: 68 grasp failures touched the cube before the close (measured aim median 3.64 cm, deflection 3.60, 85% above the success p90) and 12 did not (1.69 cm, 0.79, 2 of 12). On the grid the 124 with contact repeat this (3.77 cm, 85%); the 10 without contact do not (median commanded error 2.4 cm, so some of those are real command misses, a small sample).
- So the measured aim error at the close is mostly contact deflection. The 0.897 AUC and the 0.945 of reach error are not evidence of a wrong aim in the command, and the diagnosis no longer says so.
- **The contact-free quantity at a matched phase.** As the second option in the review I used the commanded lateral offset of the cube from the gripper, along the finger axis, at the first step where the commanded gripper is less than h above the cube. Same heights, so same phase, for both groups. AUC 0.72 to 0.80 on design (0.74 to 0.82 on grid) at 12, 10, 8, 6 cm and at the close. About 40% of grasp failures exceed the success p90 (expected 10%). At 8 cm, the medians are 1.51 cm (grasp_fail) against 0.65 cm (success), and 34 of 80 exceed the success p90 (1.59 cm). This is the near-miss evidence that remains.
- The diagnosis reports both and keeps reach error as the tracking number, because SPEC.md names it, with the caveat. Stage 3 also reports the commanded lateral offset at 8 cm beside it. Forward offset and finger-axis yaw do not separate the groups (AUC 0.52 to 0.61 and 0.42 to 0.56, `scratch/review1_phase.py`).
- Selection, as the diagnosis says: four heights, the close, two 3D numbers, the forward offset and the yaw were looked at on the same seeds after the labels were known. 8 cm is the companion number because it is the lowest height that no episode misses. It also has the highest design AUC of the four (0.80), so its choice is partly after the fact (grid 0.81). A first version limited to the active chunk gave 0.87 at 10 cm on 225 of 814 episodes and was replaced by one that loses none at 8 cm and above.

The sentence "the typical miss is about 1 cm past what still works" is also withdrawn. It compared a failure median with a success p90 of a number that is partly the bump, and no tolerance was measured. No tolerance is claimed now.

## Issue 2, "34% of failures have a succeeding nearest neighbour"

Right. The sentence was evidence for the opposite of what it was listed for. It is removed from the near-miss section. The same numbers, with the base rate beside them, are now in Pattern: given a failure, the nearest seed also fails 65.8% of the time, against 18.4% overall (8.7% given a success). That is clustering, and it supports the zone reading.

## The smaller observations

- Reach error per class, done: 10 of 23 (class 0), 57 of 62 (class 1) and 34 of 41 (class 2 in grasp_fail) exceed the success p90, which fits the review's hand counts. The diagnosis says the number works less well in class 0.
- Zone, done: x 0.08 to 0.16, y 0.40 to 0.48 holds 77 failures in 159 seeds (48%, 64 of them grasp failures), 42% of all failures. The bottom row by 4 cm cells fails 20, 19, 68, 55, 19%.
- Centre pull, done: the per-cell mean deviation of the successes points to the centre in all 4 corner cells (0.6 to 1.9 cm) and 11 of 12 edge cells. The diagnosis calls the gain pattern weak and present, not refuted.
- The pre-touch cube shift is labelled as a sanity check, not as near-miss evidence.
- `failure_map.png` has a third panel by SPEC class, as step 3 asks.
- The "holds" reading of the strict rule is logged in DEVIATIONS.md (2026-11-24). It was in the harness before any design rollout, and the both-fingers variant gives the same Gate 1 decisions.

## What I did not do

I did not identify which finger or which face of the cube is hit, or measure a grasp tolerance. The reading "a finger lands on the cube on the way down" is an inference from the z deflection and the early contact, and the diagnosis says so. The command-based number rests on the traces that are in the gitignored cache on this box, as the features already did.
