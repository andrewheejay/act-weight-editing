# Hypotheses, transfer_cube

One line per idea, written before it is evaluated. Append only. Candidate ids are the arm ids in the files.

## 2026-11-24, dose response (stage 3 step 2), written before the run

The eight heads are picked by the rule in `awe/dose.py`, not by hand. Each has ids `<head>.x0.5`, `<head>.x1.5`, `<head>.x3`, `<head>.x4`. x0 and x2 are already in the sweep. The mechanisms are guesses. None has been tested.

- cross.h1 (cross.h1.x0.5, cross.h1.x1.5, cross.h1.x3, cross.h1.x4): grasp failures fall steadily as this cross-attention head is turned down, and rise when it is turned up. At x0 it halved grasp failures but broke the most seeds, so a partial setting (x0.5) may keep most of the gain and break less.
- enc0.h6 (enc0.h6.x0.5, enc0.h6.x1.5, enc0.h6.x3, enc0.h6.x4): same as cross.h1. Turning it down reduces grasp failures.
- enc3.h4 (enc3.h4.x0.5, enc3.h4.x1.5, enc3.h4.x3, enc3.h4.x4): turning it up reduces grasp failures, and x3 does more than x2.
- enc0.h1 (enc0.h1.x0.5, enc0.h1.x1.5, enc0.h1.x3, enc0.h1.x4): turning it down reduces grasp failures.
- cross.h3 (cross.h3.x0.5, cross.h3.x1.5, cross.h3.x3, cross.h3.x4): the x0 edit moves mostly the right gripper joint, so turning it up should change the gripper opening. x0 collapses, x2 gains. I expect an interior best scale.
- enc1.h7 (enc1.h7.x0.5, enc1.h7.x1.5, enc1.h7.x3, enc1.h7.x4): turning it down reduces grasp failures.
- enc2.h3 (enc2.h3.x0.5, enc2.h3.x1.5, enc2.h3.x3, enc2.h3.x4): turning it down reduces grasp failures.
- cross.h6 (cross.h6.x0.5, cross.h6.x1.5, cross.h6.x3, cross.h6.x4): turning it down reduces grasp failures but breaks nearly as many seeds, so I expect little net gain.

Expectation on the tracking number, stated now because the sweep already hints at it: median reach error does not follow the gain. In split_half.json, cross.h1.x0 cuts grasp failures from 63 to 25 of 500 (first half) while its median reach error goes up (1.28 to 1.42 cm). So reach error may fail the dial test on the heads that help most, and that would be reported as it is.

Corrections, same day, while the run was going and before any of its results were read:
- cross.h1.x0 did not break "the most seeds". It broke 88 of the 1,000 design seeds over both halves. cross.h5.x0 broke 135 and cross.h6.x0 broke 89.
- The head rule (positive in both halves, the 8 largest mean gains) and the seed range were written in this session, after the split-half table was seen. The seeds 20000 to 20299 are part of the half that selected these edits, so their x0 and x2 values are optimistic.

## 2026-11-24, combinations (stage 3 step 3), written before any combination was run

What was known when these were written: the dose report (stage3/dose_response.json, 300 design seeds 20000 to 20299). These seeds also chose the heads and the sweep scales, so every single-head number there is optimistic. Measured on those 300 seeds, gain in points (fixed, broken): enc3.h4.x3 +13.0 (44, 5), cross.h1.x0 +12.0 (53, 17), cross.h1.x0.5 +10.3 (38, 7), enc0.h6.x0 +9.3, cross.h3.x2 +8.7 (37, 11), enc0.h1.x0 +8.3 (33, 8), cross.h6.x0 +7.7, cross.h6.x0.5 +6.7, enc3.h4.x2 +6.0. The x0.5, x3 and x4 settings of these heads were not among the sweep's. The second half (20500 to 20999) of the sweep heads gave much smaller gains (for example cross.h1.x0 +5.0), so I expect every combination to gain less than its parts did on the 300 seeds. The common idea is that the heads reduce grasp failures through different routes, so that failures fixed by one are not the ones fixed by another, with mild scales where a strong setting broke seeds. Mechanism claims are guesses. The combinations run on seeds 20500 to 20999 first (the half that chose neither the dose scales nor the first sweep table). Their spec is `awe/combo.py`.

- s1 (enc3.h4.x3 alone): repeats the best single-head dose setting on the other half of the design seeds, to check whether its +13 holds where it was not chosen.
- s2 (cross.h1.x0.5 alone): same check for the half-strength cross-attention edit, which broke fewer seeds than x0 (7 of 300 against 17).
- c01 (enc3.h4.x3 + cross.h1.x0.5): the two strongest single settings, one encoder head turned up, one cross head turned down. If the gains come from different failures, they should add to roughly the sum, with fewer than both parts' broken seeds.
- c02 (enc3.h4.x3 + cross.h3.x2): enc3.h4 up plus the gripper-linked cross head up (cross.h3.x2 cut grasp failures 41 to 16 in the dose set).
- c03 (enc3.h4.x3 + enc0.h1.x0): enc3.h4 up plus turning off the first-layer head that lowered reach error (1.18 cm median, the only head at x0 with a clear reach improvement).
- c04 (enc3.h4.x3 + enc0.h6.x0): enc3.h4 up plus enc0.h6 off (+9.3, 39 fixed, 11 broken on the 300 seeds).
- c05 (cross.h1.x0.5 + cross.h3.x2): two cross heads, gripper-related and not, without the encoder head.
- c06 (cross.h1.x0.5 + enc0.h1.x0): cross head down, first-layer head off.
- c07 (cross.h3.x2 + enc0.h1.x0): gripper-linked cross head up, first-layer head off.
- c08 (enc3.h4.x3 + cross.h1.x0): full-strength version of c01. x0 broke more seeds than x0.5 alone, so this tests whether the pair benefits from the stronger setting or is hurt.
- c09 (enc3.h4.x3 + cross.h1.x0.5 + cross.h3.x2): three heads, all three directions of the best dose rows.
- c10 (enc3.h4.x3 + cross.h1.x0.5 + enc0.h1.x0): adds the reach-error head to c01.
- c11 (enc3.h4.x3 + cross.h3.x2 + enc0.h1.x0): as c02 with the reach-error head.
- c12 (enc3.h4.x3 + cross.h1.x0.5 + enc0.h6.x0): c01 plus enc0.h6 off.
- c13 (enc3.h4.x2 + cross.h1.x0.5 + cross.h3.x2 + enc0.h1.x0): four heads with the milder enc3.h4 setting (x2, which broke 8 of 300), in case x3 overshoots when stacked.
- c14 (enc3.h4.x3 + cross.h1.x0.5 + enc0.h1.x0 + enc0.h6.x0): four encoder/cross heads that each gained alone.
- c15 (enc3.h4.x2 + cross.h1.x0.5 + cross.h3.x1.5 + enc0.h1.x0 + enc1.h7.x0): five heads, all at mild settings, in case many small edits beat few big ones.
- c16 (enc3.h4.x3 + cross.h1.x0.5 + cross.h3.x2 + enc0.h1.x0 + enc0.h6.x0): the strong version with five heads. Expected to overshoot, kept as a check on whether effects saturate.
- c17 (enc3.h4.x3 + cross.h6.x0.5 + cross.h1.x0.5): c01 plus the third-largest single cross-head effect (cross.h6.x0.5 was +6.7 on the 300 seeds).

Shared expectation: a combination is interesting only if its gain is larger than its best part alone on the same seeds and it breaks no more seeds. If gains are sub-additive, the combinations will cluster around +6 to +10 points on the second half, since the single parts did. If the two-sided test on the 500 second-half seeds shows the target removal (25% of the 92 failures of that half, about 23 fewer failures) it is a candidate for validation.

Corrections to the combination section, same day, before any combination was run:
- The second half (20500 to 20999) has 88 original failures, not 92. 25% of them is 22, and 35% is 31 (30.8).
- "The half that chose neither the dose scales nor the first sweep table" overstates it. Seeds 20500 to 20999 did take part in choosing the eight heads, through the same-sign rule and the mean gain of both halves. They did not choose the scales x0.5, x1.5, x3 and x4. So group A is less selected, and it is not a clean holdout. Only the validation seeds are.
- "The only head at x0 with a clear reach improvement" is wrong. On the 300 seeds the x0 medians are enc0.h1 1.18, enc1.h7 1.20 and enc2.h3 1.21 cm against 1.30. I picked enc0.h1 because it also has the larger gain.
- "Cluster around +6 to +10 since the single parts did" is wrong. In the second half the single parts I know gained +4.0 to +5.2. I have no sound forecast for the combinations. If gains are fully sub-additive they would land near +5, and if they add, near +10.
- The 19 arms and their scales were written by hand after reading the 300-seed dose table. That is a selection, and the 19 comparisons on group A are screening.

## 2026-11-24, refinement round, written after the group A and group B results of the combinations, before any refinement rollout

Known when written: s1 (enc3.h4.x3) +10.1 points over all 1,000 design seeds (134 fixed, 33 broken), c06 (cross.h1.x0.5 + enc0.h1.x0) +9.6 (126, 30), c03 (enc3.h4.x3 + enc0.h1.x0) +9.1 (144, 53), s2 (cross.h1.x0.5) +8.2, c07 +7.4, c05 +7.5. All were chosen on the same design seeds, so these numbers are optimistic. Stacking a strong cross head (cross.h1.x0.5) on the strong encoder head (enc3.h4.x3) lowered the gain (c01 +4.8 on group A), and c16/c14, the larger stacks, hurt. My reading (a guess, untested): the helpful edits move the arm in the same direction and overshoot when stacked at full strength, so milder partners or heads that act on different episodes might add without overshooting. The six new combinations run on all 1,000 design seeds (20000 to 20999). No new candidate gets a validation check before that.

- r1 (enc3.h4.x3 + enc1.h7.x0): the best single encoder head up, plus an encoder layer 1 head off (it lowered median reach error at x0 and gained +5.0 in the dose set). Expected to help if enc1.h7 acts on episodes enc3.h4 does not fix.
- r2 (enc3.h4.x2 + enc0.h1.x0): a milder version of c03. If overshoot is what made c03 break more seeds than s1, the milder scale should break fewer at a similar gain.
- r3 (cross.h1.x0.5 + enc0.h1.x0 + enc1.h7.x0): c06 plus the enc1.h7 head off. The question is whether the third mild head adds.
- r4 (enc3.h4.x2 + cross.h1.x0.5): the c01 pair at the milder encoder scale (c01 used x3 and gained less than either part alone).
- r5 (enc3.h4.x3 + enc2.h3.x0.5): adds the mild head that gained +3.3 and broke 9 in the dose set.
- r6 (cross.h1.x0.5 + enc0.h1.x0 + enc3.h4.x1.5): c06 plus a small increase of enc3.h4.

Shared expectation: none of the six is expected to exceed s1 or c06 by more than 2 points, given that combination gains so far were sub-additive. If none beats the best single edit (s1), s1 (or c06) goes to validation.

Corrections to this section, same day: none yet.
