# Descriptive tables (validation data)

Primary score: strict success. Paired by seed against the original. Intervals are 95% bootstrap intervals. The referee decides the verdict.

| arm | role | n | strict success | gain (points) | 95% interval | fixed | broke | p | share of failures removed | benchmark gain (points) |
|---|---|---|---|---|---|---|---|---|---|---|
| cand_r2 | other_candidate | 500 | 17.2% | +8.2 | +4.4 to +11.8 | 65 | 24 | 1.6e-05 | 9.0% | +10.0 |
| agent_edit | agent_edit | 500 | 16.8% | +7.8 | +4.2 to +11.6 | 63 | 24 | 3.5e-05 | 8.6% | +11.0 |
| cand_r5 | other_candidate | 500 | 16.4% | +7.4 | +4.0 to +11.0 | 60 | 23 | 6e-05 | 8.1% | +7.6 |
| cand_c06 | other_candidate | 500 | 16.0% | +7.0 | +3.2 to +10.8 | 64 | 29 | 0.00037 | 7.7% | +6.4 |
| cand_r6 | other_candidate | 500 | 14.6% | +5.6 | +2.0 to +9.2 | 58 | 30 | 0.0037 | 6.2% | +10.0 |
| base_constant_offset | baseline | 500 | 13.8% | +4.8 | +1.2 to +8.4 | 54 | 30 | 0.012 | 5.3% | +7.6 |
| cand_r3 | other_candidate | 500 | 13.6% | +4.6 | +1.2 to +8.0 | 49 | 26 | 0.011 | 5.1% | +8.2 |
| cand_c08 | other_candidate | 500 | 13.2% | +4.2 | +0.4 to +8.0 | 59 | 38 | 0.042 | 4.6% | +5.2 |
| cand_c02 | other_candidate | 500 | 12.4% | +3.4 | +0.2 to +6.8 | 45 | 28 | 0.06 | 3.7% | +7.2 |
| base_simplest_fix | baseline | 500 | 11.8% | +2.8 | +0.0 to +5.6 | 33 | 19 | 0.07 | 3.1% | +4.8 |
| base_deviation_gain | baseline | 500 | 7.8% | -1.2 | -4.0 to +1.4 | 21 | 27 | 0.47 | -1.3% | -4.0 |
| base_shorter_chunk | baseline | 500 | 6.8% | -2.2 | -5.4 to +1.0 | 27 | 38 | 0.21 | -2.4% | -0.6 |

## Behaviour numbers (parts: S strict, A class 4a, M class 3, grasp failures by what was not lifted; tip: median offset before contact over episodes that meet)

| arm | S | A | M | peg_not_lifted | socket_not_lifted | neither_lifted | both_lifted | met | pin | late pin share | tip (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 45 | 54 | 197 | 107 | 62 | 25 | 10 | 296 | 99 | 0.1212 | 1.634 |
| agent_edit | 84 | 70 | 159 | 109 | 39 | 32 | 7 | 313 | 154 | 0.0649 | 1.846 |
| base_constant_offset | 69 | 68 | 149 | 123 | 63 | 24 | 4 | 286 | 137 | 0.1095 | 1.391 |
| base_deviation_gain | 39 | 40 | 190 | 131 | 55 | 35 | 10 | 269 | 79 | 0.1266 | 1.629 |
| base_shorter_chunk | 34 | 62 | 229 | 74 | 56 | 25 | 20 | 325 | 96 | 0.2083 | 1.461 |
| base_simplest_fix | 59 | 64 | 172 | 107 | 61 | 26 | 11 | 295 | 123 | 0.1301 | 1.282 |
| cand_c06 | 80 | 51 | 166 | 122 | 43 | 32 | 6 | 297 | 131 | 0.0534 | 1.822 |
| cand_r2 | 86 | 63 | 170 | 103 | 43 | 29 | 6 | 319 | 149 | 0.0671 | 1.774 |
| cand_c08 | 66 | 59 | 180 | 116 | 47 | 25 | 7 | 305 | 125 | 0.104 | 1.602 |
| cand_r6 | 73 | 76 | 158 | 115 | 44 | 29 | 5 | 307 | 149 | 0.1074 | 1.813 |
| cand_r5 | 82 | 55 | 171 | 114 | 41 | 33 | 4 | 308 | 137 | 0.0876 | 1.856 |
| cand_r3 | 68 | 72 | 170 | 109 | 46 | 29 | 6 | 310 | 140 | 0.0786 | 1.8 |
| cand_c02 | 62 | 73 | 172 | 115 | 44 | 30 | 4 | 307 | 135 | 0.0593 | 1.75 |

Parts that grew in the edit: S (45 to 84), A (54 to 70), peg_not_lifted (107 to 109), neither_lifted (25 to 32)

Original failures fixed by kind: A 14 of 54, M 38 of 197, peg_not_lifted 7 of 107, socket_not_lifted 2 of 62, neither_lifted 2 of 25, both_lifted 0 of 10. Broken successes became: {'M': 10, 'peg_not_lifted': 3, 'A': 8, 'socket_not_lifted': 3}

## Zones

| zone | n | original | edit | fixed | broke |
|---|---|---|---|---|---|
| peg x below 0.143 (larger tip offset), inside | 216 | 9.3% | 13.0% | 19 | 11 |
| peg x below 0.143 (larger tip offset), outside | 284 | 8.8% | 19.7% | 44 | 13 |
| peg y above 0.483 (larger tip offset), inside | 306 | 5.9% | 12.1% | 31 | 12 |
| peg y above 0.483 (larger tip offset), outside | 194 | 13.9% | 24.2% | 32 | 12 |
