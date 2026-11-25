# Hypotheses, insertion

One line per idea, written before it is evaluated. Append only. Candidate ids are the arm ids in the files.

## 2026-11-25, dose response (stage 3 step 2), written before the run

Known when written: the sweep (stage2/sweep_summary.json, 500 seeds) and the split-half check (stage2/split_half.json, 1,000 seeds). The primary score is strict success (7.4% on the design seeds), so every edit here acts on a small base rate and a 2 to 3 point gain is about two standard errors on 500 seeds.

The largest single effect is cross.h0.x0: strict +8.0 and +6.4 points (guard +4.4 and +6.8). In the sweep table (both halves, 1,000 seeds) it raises the number of pin contacts from 188 to 244, the share of pins that become strict from 39% to 60%, and lowers the share of pin contacts that start after step 375 from 16.5% to 5.7%. Those pins arrive earlier (median pin step 338.5 to 332.0). I read it as a timing effect more than an alignment effect, and say so before testing: cross.h0.x0 does not lower the tip offset (1.62 cm against 1.62 in the first half). The other kept edits give +2 to +4 points with no clear timing signature.

The eight heads are picked by the rule in `awe/dose_ins.py`, not by hand. Each has ids `<head>.x0.5`, `<head>.x1.5`, `<head>.x3`, `<head>.x4`. x0 and x2 come from the sweep. The mechanisms are guesses. None has been tested.

- cross.h0 (cross.h0.x0.5, cross.h0.x1.5, cross.h0.x3, cross.h0.x4): turning this cross-attention head down raises strict success steadily, x0 best, x0.5 about half. Turning it up lowers it. Expected direction of the tracking number: small and unclear, since the gain looks like timing.
- cross.h2 (cross.h2.x0.5, cross.h2.x1.5, cross.h2.x3, cross.h2.x4): turning it down raises strict success, as x0 did in both halves (+2.6, +4.8). A partial setting (x0.5) keeps less.
- enc3.h4 (enc3.h4.x0.5, enc3.h4.x1.5, enc3.h4.x3, enc3.h4.x4): turning it up raises strict success, x2 already did (+3.2, +3.6), and x3 gives more or the same.
- enc0.h6 (enc0.h6.x0.5, enc0.h6.x1.5, enc0.h6.x3, enc0.h6.x4): turning it up raises strict success (x2 +2.4, +4.4).
- enc3.h3 (enc3.h3.x0.5, enc3.h3.x1.5, enc3.h3.x3, enc3.h3.x4): turning it down raises strict success (x0 +3.2, +3.2).
- enc2.h6 (enc2.h6.x0.5, enc2.h6.x1.5, enc2.h6.x3, enc2.h6.x4): turning it down raises strict success (x0 +2.6, +3.2).
- enc2.h1 (enc2.h1.x0.5, enc2.h1.x1.5, enc2.h1.x3, enc2.h1.x4): turning it down raises strict success (x0 +3.0, +2.4).
- enc1.h1 (enc1.h1.x0.5, enc1.h1.x1.5, enc1.h1.x3, enc1.h1.x4): turning it up raises strict success (x2 +2.6, +1.4), small.

Expectation on the tracking number (median tip offset before contact, met episodes): it should move with strict success for an edit that works through alignment. For cross.h0 I expect the gain to come from timing, so its tracking number may stay flat. If the tracking number does not move in step with the strict gain for the heads that help, that is reported as it is.

Corrections, same day, while the dose run was still waiting to start (the 7-minute delay) and before any of its results were read:
- "The other kept edits ... with no clear timing signature" is wrong. `scratch/ins_sweep_timing.py` (1,000 seeds per arm, both halves pooled) gives the share of pin contacts that start after step 375 (t_pin > 375): original 16.5%, cross.h0.x0 5.7%, and lower than the original for every other kept edit I checked (cross.h2.x0 13.3%, enc3.h4.x2 13.7%, enc0.h6.x2 13.9%, enc3.h3.x0 10.1%, enc2.h6.x0 11.6%, enc2.h1.x0 15.2%, enc1.h1.x2 12.4%, enc3.h7.x2 12.7%). So a smaller late share is common to all of them and cross.h0.x0 is the extreme. Median pin step 338.5 for the original and 332.0 for cross.h0.x0 (others 333 to 341). The timing reading is therefore about cross.h0 and weak for the rest, and the pin share of strict among pins (39.4% original, 59.8% cross.h0.x0) is a second reading: the gain may partly be better conversion of pins to 25-step strict runs, not only arrival. These are descriptive (the heads were chosen on the same seeds).
- "Late share" in this file is t_pin > 375. The dose report uses t_pin >= 375 (awe/dose_ins.py LATE). The small difference does not change any statement.

## 2026-11-25, session 24, head combinations (stage 3 step 3), written before any combination rollout

Known when written: the dose response (stage3/dose_response.json, seeds 20000 to 20299, 8 heads x 4 scales) and everything before it. Seeds 20500 to 20999 ("group A") are used for these combinations because the heads were chosen on 20000 to 20499 and partly checked on 20500 to 20999. Group A is therefore not untouched for the heads (the recheck used these same seeds for the same single heads), so any combination result on group A is optimistic for the heads but not for the scale mixtures. Group B (20000 to 20499) adds the rest for the best ones. Each combination is judged by strict success (primary, against 7.2% on group A), with the benchmark guard shown beside it.

What the dose response says (300 seeds, descriptive). cross.h0: strict gain falls with the scale in every step except x1 to x1.5 (x0 +8.0, x0.5 +5.0, x1 0, x1.5 +0.3, then -0.7, -4.3, -5.7 at x2, x3, x4), so not a strict dial. cross.h2 (x0 +4.3, x0.5 +5.0, x1.5 +2.3) and enc3.h3 (x0 and x0.5 both +3.7, x1.5 -1.0) have their best values at or below 0.5, with cross.h2 still +2.3 at x1.5 and negative from x2. enc3.h4, enc0.h6, enc1.h1 and enc2.h1 are positive at scales above 1 (enc3.h4 x2 +4.7, x3 +4.0; enc0.h6 x4 +5.7, x1.5 to x3 +3 to +4; enc1.h1 x4 +4.0, x3 +3.0). enc2.h6 is best at x0 (+3.3, +1.0 or less at every other scale), enc2.h1 at x0 and x0.5 (+3.7, +3.0) and x1.5 (+3.3). With 300 seeds these differences are about 2 standard errors at best, so they do not order scales, they only give the candidate scale per head. The tip median of the met episodes moves with no clear pattern (cross.h0 at x0 1.62 against 1.39 for x1, so it is not the mechanism).

Design rules for the list below. One head per block (block = encoder layer, or the single cross-attention layer). The reason is the matched random family of the protocol. A random edit replaces the head by one of 7 others in the same block at the same scale, so a 2-head combination has 49 distinct random edits (40 are needed for H2), and a pair in one block would be outside that rule. Heads: the ones with a positive strict gain in both halves of the split-half check (stage2/split_half.json). Scales: the value with the largest gain at or near the dose evidence, not tuned by search: cross.h0 at x0 (the dial's best end), x0.5 as a gentler version; enc3.h4 x2; enc0.h6 x2; enc2.h1 x0; enc2.h6 x0; enc1.h1 x2; enc3.h3 x0; cross.h2 x0 as an alternative to cross.h0.
Expectation: cross.h0 x0 alone (+7.2 pooled, sweep and recheck) should remain the largest single effect, and the question of the list is whether a second head adds a measurable part on top, or nothing (the first hypothesis is that the gains are largely the same episodes and therefore do not add up).

- c01 cross.h0 x0 with enc3.h4 x2: the two strongest single families (cross + encoder 3) together. Expected about as much as cross.h0 alone, or more if they fix different episodes.
- c02 cross.h0 x0 with enc0.h6 x2: second strongest scale family by the dose (enc0.h6 up) with the timing edit. Expected larger than cross.h0 alone by 1 to 2 points if the effects add, none if not.
- c03 cross.h0 x0 with enc2.h1 x0: both at scale 0, both lowering a head. Expected a small addition.
- c04 cross.h0 x0 with enc2.h6 x0: as c03, the other enc2 head that was kept. Expected a small addition.
- c05 cross.h0 x0 with enc1.h1 x2: as c02 with the enc1 head. Expected a small addition (enc1.h1 is the weakest of the kept heads).
- c06 cross.h0 x0 with enc3.h3 x0: replaces the enc3 partner by the other kept head of the same block. Expected similar to c01.
- c07 cross.h0 x0.5 with enc3.h4 x2: the gentle version of c01 (the dose shows x0.5 keeps about half of the gain). Expected less strict gain than c01, smaller guard loss.
- c08 cross.h0 x0 with enc3.h4 x2 and enc0.h6 x2: three families. Expected no more than c01 or c02 if the effects overlap, and possibly a guard loss.
- c09 cross.h0 x0 with enc3.h4 x2 and enc2.h1 x0: three families with the enc2 head.
- c10 cross.h0 x0 with enc0.h6 x2 and enc2.h1 x0: as c09 without the enc3 head.
- c11 cross.h0 x0 with enc3.h4 x2, enc0.h6 x2 and enc2.h1 x0: the four strongest families at the single-head scales. Expected overlap, so this is a test for diminishing returns rather than the best candidate.
- c12 cross.h0 x0.5 with enc3.h4 x2 and enc0.h6 x2: the gentle three-family version (c08 with cross at x0.5).
- c13 cross.h2 x0 with enc3.h4 x2: the same idea with the other cross head, to see whether cross.h2 is an alternative (same block as cross.h0, so never combined with it here).
- c14 cross.h2 x0 with enc0.h6 x2 and enc2.h1 x0: the variant with cross.h2.
- c15 enc3.h4 x2 with enc0.h6 x2 and enc2.h1 x0: no cross-attention head at all, to see if the encoder edits add up without it.
- c16 cross.h0 x0, enc3.h4 x2, enc0.h6 x2, enc2.h1 x0, enc1.h1 x2: five heads. Expected below or at c11 if the effects overlap, kept as the widest combination.

If the combinations come out like the single-head rows (about +7 for cross.h0-based, +3 for the others), the answer is that the main effect is one head and the others add little, and I will say so.
Ids are the arm ids in the result files (awe/combo_ins.py).

What the combinations are measured against, written before any combination ran. The simple baselines were tuned on the same design seeds (stage3/baselines_summary.json, 1,000 seeds, strict gains, guard gains): constant_offset off_j1_p0.04 +6.7 (guard +6.6), simplest_fix late_j2_p0.02 +5.5 (guard +4.7), deviation_gain gain_right_nogrip_0.8 +2.5 (guard -0.3), shorter chunk 50 +2.8 (guard +3.0). Their gains are also selected on the same 1,000 seeds, as the edits' will be, and so are optimistic in the same direction. cross.h0.x0 alone is at +7.2 (guard +5.6) on the 1,000 seeds (sweep and recheck rows), so the best single edit and the best baseline are about level, within the noise of the comparison. A win (H3 in RULES.md) needs the edit to beat every baseline on the test seeds, which is why the combinations try for more than about +8 points on the design seeds. The baselines are close in mechanism too: off_j1_p0.04 shifts the left shoulder by a constant and late_j2_p0.02 shifts the left elbow from step 200 on, and both lower the tip offset (median over the met episodes 1.38 cm, 581 met, and 1.29 cm, 587 met, against 1.70 cm for the original, 593 met of 1,000, in features_default), while cross.h0.x0 gives 1.69 cm (the sweep and recheck rows, met counts not re-counted here). Whether cross.h0 works by the same route as the offsets is open and will be said either way.
A 25% failures-removed pass needs about 230 net fixes out of about 926 strict failures per 1,000 seeds (+23 points), which no arm here comes near. So the validation pass rule of SPEC.md Stage 3 (25% and p < 0.05 and guard) is expected to fail for every candidate, and the stop rule (freeze the best, say it missed) is the likely path. Nothing will be tuned to it.

## 2026-11-25, session 25, group B and refinements (stage 3 step 3), written before any of these rollouts

Known when written: group A of c01 to c16 (stage3/combo_summary.json, seeds 20500 to 20999, 500 seeds). Nothing on group B for any combination, nothing on any refinement.
Group A, strict gain / guard gain in points: c06 +12.4 / +10.2 (78 fixed, 16 broken), c08 +8.6 / +7.4, c02 +8.4 / +8.8, c01 +7.6 / +8.8, c11 +7.6 / +8.2, c07 +7.6 / +6.4, the reference single cross.h0.x0 +6.4 / +6.8. The others are between +2.6 (c13) and +6.8.
c06 is the best of 16 arms on the seeds that picked it, so it is optimistic (winner's curse). With 94 discordant seeds the standard error of its gain is about 1.9 points.

Group B (seeds 20000 to 20499). By the plan of session 24 it goes to the four best arms by group A strict gain. Three arms tie for fourth at +7.6 (c01, c07, c11). The tie is broken by the guard gain, so c01 (+8.8) goes. These are the seeds on which the heads were first chosen (sweep) and the dose scales, so group B is optimistic for the heads and not independent of the choice of cross.h0 and enc3.h3.
- c06 (group B): expected strict gain +8 to +12, less than the group A value.
- c08 (group B): expected +6 to +9.
- c02 (group B): expected +6 to +9.
- c01 (group B): expected +5 to +8.

Refinements, all on 1,000 design seeds (group AB), built on c06 because it is the best. The idea of each is stated before it runs. One head per block as before (so the matched random family has at least 7 x 7 = 49 edits for two heads).
What group A says about what could add to it, descriptive only: on top of the cross.h0 x0 edit, adding enc0.h6 x2 gave +2.0 points on group A (c02 +8.4 against the single +6.4), adding enc3.h4 x2, enc2.h1 x0, enc2.h6 x0 or enc1.h1 x2 gave +1.2, +0.2, +0.4 and -0.6. c06 is the pair where the second head is enc3.h3 x0, which gave +6.0. The first-reading explanation is that c06 and the other arms rescue different episodes (an offline count on group A: 53 of 79 c08 rescues lie inside the 98 c06 rescues, the union of the two would be 124 rescues of 500).
- r1 (cross.h0 x0, enc3.h3 x0, enc0.h6 x2): adds the head that helped cross.h0 most. Expected +1 to +2 over c06, or nothing.
- r2 (cross.h0 x0, enc3.h3 x0, enc0.h6 x4): as r1 with the largest scale of the dose run, where enc0.h6 x4 was +5.7 alone (x2 was +3.0, x3 +3.0). Expected like r1.
- r3 (cross.h0 x0, enc3.h3 x0, enc2.h1 x0): a third head at scale 0, which added 0.2 on top of cross.h0. Expected nothing.
- r4 (cross.h0 x0, enc3.h3 x0.5): enc3.h3 at half strength. The dose gave the same strict gain (+3.7) and a much better guard (+5.0 against -0.7 for x0) with 300 seeds, so it may keep more benchmark episodes. Expected about c06 on strict, a higher guard.
- r5 (cross.h0 x0, enc3.h3 x0.5, enc0.h6 x2): r4 with the head of r1. Expected like r1 or better.
- r6 (cross.h0 x0, enc3.h3 x0, enc0.h6 x2, enc2.h1 x0): four heads. Expected no more than r1.

Rules that stay as they are: the choice of the final edit uses the design and the validation data in the way the candidates file will say, written before any validation rollout. Nothing here is tuned to the validation rule, which needs 25% of failures removed and is out of reach (the failure pool of the primary score is about 93%, so 25% means about +23 points).

Correction, same session, before any group B or refinement result was read (job stage3c_ins just started): the line above "53 of 79 c08 rescues lie inside the 98 c06 rescues, the union of the two would be 124 rescues of 500" is wrong. 79, 98, 53 and 124 are strict successes on group A (including the 36 the original already had), not rescues. Counted properly (rescue = strict success where the original was not strict, group A): c06 78, c08 58, c02 61, c01 62, the single 52. c06 and c08 share 35 rescues, their union is 101; with c02 shared 36, union 103; with c01 shared 40, union 100. So the arms rescue partly different episodes, but a smaller difference than the wrong numbers suggested, and a union is not a forecast for a combination.
