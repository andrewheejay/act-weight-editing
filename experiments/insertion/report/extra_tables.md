# Extra descriptive tables (test data)

The edit against other arms on the same 1,000 test seeds. Positive means the edit did better. Descriptive, after the test.

| other arm | strict difference (points) | 95% interval | only the edit | only the other | p | benchmark difference (points) | 95% interval |
|---|---|---|---|---|---|---|---|
| original | +8.1 | +5.5 to +10.7 | 129 | 48 | 9.3e-10 | +10.7 | +7.6 to +13.7 |
| autogreedy | +0.7 | -1.6 to +3.0 | 70 | 63 | 0.6 | -0.1 | -2.4 to +2.2 |
| base_constant_offset | +4.5 | +1.6 to +7.3 | 127 | 82 | 0.0023 | +4.0 | +0.7 to +7.3 |
| base_simplest_fix | +6.4 | +3.7 to +9.1 | 130 | 66 | 5.7e-06 | +6.4 | +3.1 to +9.5 |
| base_deviation_gain | +9.7 | +7.1 to +12.3 | 141 | 44 | 4.8e-13 | +10.5 | +7.2 to +13.6 |
| base_shorter_chunk | +11.0 | +8.4 to +13.7 | 151 | 41 | 5.3e-16 | +11.2 | +7.8 to +14.5 |
| dose_k0.5 | +4.9 | +2.5 to +7.3 | 105 | 56 | 0.00014 | +5.2 | +2.2 to +8.1 |
| original_ens | +12.1 | +9.6 to +14.7 | 153 | 32 | 4e-20 | +8.8 | +5.5 to +12.0 |
| agent_edit_ens | +8.2 | +5.5 to +10.9 | 139 | 57 | 4.3e-09 | +3.4 | +0.2 to +6.5 |

With ensembling, edit against original: strict +3.9 (+1.8 to +6.0, 79 against 40, p 0.00045), benchmark +5.4.

Timing (steps of 400; pin = first step the peg tip is in the socket, late = at or after step 375):

| arm | median first meet | median first pin | episodes with a pin | of those strict | late pin share |
|---|---|---|---|---|---|
| original | 300 | 338 | 197 | 112 | 0.112 |
| agent_edit | 289 | 330 | 304 | 193 | 0.033 |
| autogreedy | 289 | 331 | 305 | 186 | 0.046 |
| dose_k0.5 | 294 | 332 | 252 | 144 | 0.075 |
| dose_km0.5 | 317 | 340 | 134 | 66 | 0.149 |
| dose_km1 | 328 | 338 | 87 | 43 | 0.184 |
| dose_km2 | 343 | 371 | 45 | 12 | 0.400 |

Random edits by the cross-attention head turned off (strict gain, points):

| head | n | mean | min | max |
|---|---|---|---|---|
| cross.h1 | 7 | -3.20 | -5.6 | -2.2 |
| cross.h2 | 5 | -0.80 | -1.7 | -0.2 |
| cross.h3 | 7 | -5.20 | -7.0 | -3.3 |
| cross.h4 | 4 | -5.95 | -6.9 | -3.3 |
| cross.h5 | 5 | -4.50 | -6.9 | -3.3 |
| cross.h6 | 6 | -1.22 | -3.7 | +1.5 |
| cross.h7 | 6 | -2.95 | -5.6 | -2.1 |

Random edits by the encoder layer 3 head turned off:

| head | n | mean | min | max |
|---|---|---|---|---|
| enc3.h0 | 6 | -2.83 | -6.8 | +0.8 |
| enc3.h1 | 2 | -2.55 | -2.9 | -2.2 |
| enc3.h2 | 3 | -5.43 | -6.9 | -3.8 |
| enc3.h4 | 6 | -3.12 | -6.9 | -0.2 |
| enc3.h5 | 11 | -3.75 | -7.0 | +1.5 |
| enc3.h6 | 9 | -3.01 | -6.8 | -0.2 |
| enc3.h7 | 3 | -2.87 | -3.3 | -2.1 |
