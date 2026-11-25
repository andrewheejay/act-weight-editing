# Response to the Gate 1 review of insertion, round 1

All three blocking issues are right. The diagnosis is changed and every new or corrected passage is marked [R1]. No rollout was run for this. The numbers come from `stage1/review1_ins.py` (`depth`, `runs`, `stats`, writing `ins_review1.json`) and `stage1/reach_ins.py` (`reach`, `stats`, `matched`, `matched_stats`, writing `ins_reach.json` and `ins_reach_matched.json`). The per-episode files they read are committed (`depth_*.jsonl.gz`, `runs_default.jsonl.gz`, `reach_default.jsonl.gz`, `reach_matched_default.jsonl.gz`), so the stats can be recomputed without the trace cache.

## Issue 1, the depth numbers do not measure entry

Right. The 40% and 25% are withdrawn. The shares again on the design seeds, with the lateral condition at the deepest step (tip centre less than 1.8 cm from the axis, the half-opening):

| Group | n | deepest tip past 5 mm, no lateral condition | in the opening | outside | past 15 mm, no lateral condition | in the opening | outside |
|---|---|---|---|---|---|---|---|
| S | 74 | 74 | 74 | 0 | 60 | 60 | 0 |
| A | 114 | 114 | 114 | 0 | 90 | 90 | 0 |
| M | 405 | 163 | 22 | 141 | 101 | 1 | 100 |

A stricter gate (all four corners of the leading face inside the opening, 2 mm slack) gives 21 and 1 for M. So 5% of M entered the opening by 5 mm, and 35% of M passed beside the socket. 77 of M have an ungated deepest tip below 3.8 cm and all 77 are outside, which is the impossibility you pointed to. The p10 of 2.36 cm is now in the text. The same picture holds at chunk 50 (44 in, 116 beside, of 415), on the peg grid (22 and 249 of 722) and on the socket grid (71 and 330 of 805).

The near-miss claim, restated: before contact 4.4% of M are within the 0.8 cm limit, 18% within 1.2 cm, 33% within 1.6 cm, 62% within 2.4 cm. The median M offset is 2.5 times the limit. M is a coarse miss and the diagnosis no longer calls it a near miss. Of the 405, 242 keep their deepest tip within 5 mm of the mouth plane, with the tip less than 0.8 cm off axis in 18, 0.8 to 1.8 cm in 119, and above 1.8 cm in 105. The other 163 are the ones in the table.

## Issue 2, the 407 grasp failures are not mixtures without a cause

Right, and the sentence is removed. The split by which object was not lifted (z above 0.06 m; 0.055 and 0.07 change the counts by at most 5):

| Part | n | class 0 / 1 / 2 | matching arm, median reach deviation (part, S) | AUC vs S | AUC vs all met | clustering (nearest seed in the part, against base rate) |
|---|---|---|---|---|---|---|
| peg not lifted only (right arm) | 214 | 39 / 152 / 23 | 3.27, 1.00 cm | 0.945 | 0.852 | 52.8% against 21.4% (2.5x) |
| socket not lifted only (left arm) | 120 | 94 / 23 / 3 | 4.17, 1.25 cm | 0.985 | 0.973 | 47.5% against 12% (4.0x) |
| neither lifted | 52 | 26 / 26 / 0 | | | | |
| both lifted | 21 | 0 / 4 / 17 | | | | |

Position patterns: right part by peg y fifth 0.11, 0.27, 0.30, 0.22, 0.19 and flat in peg x; left part by socket y 0.15, 0.21, 0.155, 0.035, 0.05 and by socket x 0.065, 0.09, 0.14, 0.195, 0.11. I checked whether the deviation is contact deflection, as the transfer-cube review taught. With the commanded gripper (joint targets, by kinematics, checked to 4e-8 m against the traced gripper centre) at the first step less than 10 cm above the object, with no contact yet, the lateral offset is 3.13 cm against 1.32 in S (AUC 0.899) for the right arm (209 of 209 untouched), and 2.88 against 1.48 (AUC 0.845) for the left arm (79 of 120 untouched). So the deviation is in the command, before any contact. I read it as an aim error, but a 3 cm lateral offset at 10 cm height could also come from a wider approach path, which I did not separate.

M by the same test: 81 of 405 meet at step 350 or later; 163 get the tip past 5 mm of the mouth plane, 141 of them beside the socket; the M clustering ratio is 1.36 (55% against 40.5%), lower than the grasp parts; M share by fifth: peg x 0.46, 0.38, 0.41, 0.42, 0.37, peg y 0.46, 0.31, 0.40, 0.51, 0.36, socket x 0.46, 0.43, 0.37, 0.40, 0.38, socket y 0.48, 0.42, 0.32, 0.47, 0.35.

**Target kept, for a reason that is not near-miss.** I considered the alternatives. The stage arithmetic is now in the diagnosis: 593 of 1000 episodes meet, 188 reach the pin, 74 are strict. A rescued grasp episode is worth about 0.11 strict (strict rate of the nearest 10 seeds that met, 0.106; 20 nearest 0.111), a rescued M episode that reaches the pin about 0.39 (74 of 188 pin contacts became strict), a rescued 4a episode 1.0. Upper bounds on strict with a whole stage fixed and later rates unchanged (an untested assumption): grasp 12.5%, M 23.3%, 4a 18.8%. M has the largest bound and is the largest group at that value (405 against 214 and 120 by arm, or 114 for 4a). Class 4a is the closer near miss by the data (median longest strict run 14 of 25 steps, 29 of 114 reach 20 steps; pin contact lost in 76, episode end in 25) and is reported beside M, but it is small and the tip number does not see it. The target field in decisions.json now says this. The grasp parts are the cleanest by signature and the least valuable per rescue on the primary score; if a candidate changes them, their counts are reported, split by arm.

## Issue 3, the pattern against start position

Right on both points.

- The sentence "most of the scatter is within-position noise" is removed. Policy and physics are deterministic, and the linear R2 shows only that the dependence is not linear.
- Peg y was missing and is now reported: Spearman of the tip offset with peg y +0.21 on design and +0.21 on the peg grid (socket y -0.11 and -0.19). Medians by fifth of peg y: 1.26, 1.33, 2.04, 2.03, 1.86 (design), 1.73, 1.49, 1.37, 2.15, 2.92 (peg grid).
- Local test, on the two 41 by 41 grids (signed tip offset of met episodes, neighbours 2.5 mm apart in x and 5 mm in y): correlation with the x neighbour 0.61 and 0.71 (signed y and z, peg grid), 0.60 and 0.65 (socket grid); with the y neighbour 0.24 to 0.44; 2 cm away 0.09 to 0.36 in x and about zero in y. Half the mean squared x-neighbour difference over the variance: 0.39, 0.29, 0.38, 0.34. Leave-one-out k-nearest-neighbour R2 (k 4, 8, 16) on the grids: 0.33, 0.27, 0.23 (y) and 0.45, 0.39, 0.35 (z) on the peg grid, 0.40, 0.40, 0.38 (y) and 0.47, 0.44, 0.42 (z) on the socket grid. On the design seeds, where the median nearest neighbour is 0.25 of a box half-width away, it is at most 0.17 for the signed offset and 0.24 to 0.29 for the absolute one.
- The conclusion is now limited to "a deterministic dependence with a length scale of roughly 1 to 2 cm, not linear". The text no longer says "probably not editable" or "no clean gain pattern to copy". I make no claim on whether an edit can use a structure at that scale, because a head's effect is global in the input.

## Smaller observations, also changed

- "It also has the highest AUC" holds on design only. The text now says that on the socket grid the rejected pre_lat is 0.778 against 0.658 for the tip number, and on chunk 25 the depth number is 0.9125 against 0.9086.
- pre_tip_lat_20 (0.548 on design, 0.612 chunk 50, 0.696 chunk 25, 0.841 ensembling, 0.418 and 0.467 on the grids) is listed as a number tried and dropped.
- The radial AUCs (0.835 M against S, 0.797 M against S and A, 0.553 A against S) are coded (`radial_tip_offset_auc_design`).
- Class 4a: the pin-contact run lengths are now reported (above), instead of calling it a hold-at-the-pin problem on assertion. I found that 36 of 74 (49%) of the S runs also end on the loss of pin contact, the first longest run, so the pin loss alone does not separate the groups; the length of the run does. The flicker you raise (23 of 114 A are strict_depth, 37 of 74 S) is consistent with this, and I did not resolve it further.
- "32% for seeds with a non-M neighbour" was the wrong way round and is corrected (the rate of an M neighbour among seeds not in M).
- The met count beside the tracking number is now a requirement in the text and in decisions.json.
- Decision age: the S and A numbers (0.87 to 0.60 and 1.14 to 0.73 cm) are now in the text next to M (2.08 to 2.00).

## What I did not do

I did not measure whether a tip aligned within the limit would lead to a strict success, I did not decompose the approach to the socket into command and tracking error (the commanded-aim check is for the two grasps only), and I did not test the assumption behind the stage arithmetic. The mechanism of the M miss before contact is still unknown. The grasp-part and 4a numbers rest on the traces in the gitignored cache as the features already did; the per-episode outputs are committed.
