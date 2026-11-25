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
