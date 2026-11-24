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
