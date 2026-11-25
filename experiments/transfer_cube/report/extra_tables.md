# Extra descriptive tables (test data)

The edit against other arms on the same seeds. Positive means the edit did better.

| other arm | difference (points) | 95% interval | only the edit succeeds | only the other succeeds | p | strict difference (points) |
|---|---|---|---|---|---|---|
| base_deviation_gain | +2.3 | +0.3 to +4.3 | 63 | 40 | 0.03 | +0.4 |
| base_simplest_fix | +6.1 | +3.6 to +8.6 | 111 | 50 | 1.7e-06 | +6.9 |
| base_constant_offset | +6.3 | +3.6 to +9.0 | 125 | 62 | 4.8e-06 | +6.6 |
| base_shorter_chunk | +12.1 | +9.1 to +15.0 | 178 | 57 | 1.1e-15 | +15.9 |
| original_ens | +17.2 | +14.2 to +20.2 | 216 | 44 | 1.9e-28 | +21.2 |
| autogreedy | +0.0 | -1.9 to +1.9 | 47 | 47 | 1 | +0.0 |
| random_00 | -0.8 | -2.7 to +1.1 | 45 | 53 | 0.48 | +1.3 |
| dose_k0.5 | +3.0 | +0.9 to +5.1 | 72 | 42 | 0.0064 | +3.2 |
| dose_k1.5 | +2.7 | +0.6 to +4.8 | 73 | 46 | 0.017 | +2.9 |

Random edits, gain in points by heads:

| enc3 head x2 | cross.h0 x0.5 | cross.h2 x0.5 | cross.h3 x0.5 | cross.h4 x0.5 | cross.h5 x0.5 | cross.h6 x0.5 | cross.h7 x0.5 |
|---|---|---|---|---|---|---|---|
| enc3.h0 | -5.1 | not drawn | -13.8 | -1.4 | +3.1 | +2.8 | not drawn |
| enc3.h1 | -0.3 | +3.3 | -9.4 | +2.5 | +9.2 | not drawn | not drawn |
| enc3.h2 | -1.2 | -5.8 | -9.4 | +1.9 | +6.4 | +4.3 | +3.3 |
| enc3.h3 | not drawn | -1.1 | -13.4 | +0.4 | +6.4 | +5.3 | +1.9 |
| enc3.h5 | -0.4 | -17.8 | -13.2 | +2.7 | +3.8 | +6.7 | +1.4 |
| enc3.h6 | -4.8 | -10.0 | not drawn | -3.5 | not drawn | +1.2 | not drawn |
| enc3.h7 | not drawn | -1.8 | -9.3 | +0.2 | +7.2 | +2.2 | +1.6 |
