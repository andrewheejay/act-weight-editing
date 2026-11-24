# Descriptive tables (validation data)

Paired by seed against the original. Intervals are 95% bootstrap intervals. The referee decides the verdict.

| arm | role | n | success | gain (points) | 95% interval | fixed | broke | p | share of failures removed | strict gain (points) |
|---|---|---|---|---|---|---|---|---|---|---|
| cand_r6 | other_candidate | 500 | 92.6% | +10.4 | +7.0 to +14.0 | 68 | 16 | 8.5e-09 | 58.4% | +12.2 |
| agent_edit | agent_edit | 500 | 91.4% | +9.2 | +5.6 to +12.8 | 67 | 21 | 9.2e-07 | 51.7% | +9.4 |
| cand_r1 | other_candidate | 500 | 90.4% | +8.2 | +4.6 to +12.0 | 67 | 26 | 2.5e-05 | 46.1% | +7.8 |
| cand_r5 | other_candidate | 500 | 90.0% | +7.8 | +4.2 to +11.6 | 67 | 28 | 7.8e-05 | 43.8% | +8.6 |
| cand_c06 | other_candidate | 500 | 90.0% | +7.8 | +4.4 to +11.2 | 58 | 19 | 9.8e-06 | 43.8% | +9.8 |
| cand_s1 | other_candidate | 500 | 90.0% | +7.8 | +4.2 to +11.4 | 62 | 23 | 2.8e-05 | 43.8% | +8.8 |
| cand_r2 | other_candidate | 500 | 89.8% | +7.6 | +4.0 to +11.2 | 61 | 23 | 4.1e-05 | 42.7% | +8.6 |
| cand_c03 | other_candidate | 500 | 89.8% | +7.6 | +3.6 to +11.6 | 69 | 31 | 0.00018 | 42.7% | +8.2 |
| base_deviation_gain | baseline | 500 | 86.6% | +4.4 | +0.4 to +8.4 | 63 | 41 | 0.039 | 24.7% | +7.0 |
| base_simplest_fix | baseline | 500 | 85.2% | +3.0 | -0.4 to +6.6 | 48 | 33 | 0.12 | 16.9% | +2.8 |
| base_constant_offset | baseline | 500 | 84.6% | +2.4 | +0.0 to +4.8 | 25 | 13 | 0.073 | 13.5% | +1.8 |
| base_shorter_chunk | baseline | 500 | 79.2% | -3.0 | -7.2 to +1.2 | 52 | 67 | 0.2 | -16.9% | -3.0 |

## Behaviour numbers

| arm | grasp failures | median reach error (cm) | median lateral command deviation (cm) |
|---|---|---|---|
| original | 51 | 1.198 | 0.698 |
| agent_edit | 28 | 1.282 | 0.572 |
| base_constant_offset | 41 | 1.212 | 0.753 |
| base_deviation_gain | 45 | 1.449 | 0.552 |
| base_shorter_chunk | 53 | 1.298 | 0.762 |
| base_simplest_fix | 38 | 1.473 | 0.648 |
| cand_r5 | 28 | 1.251 | 0.567 |
| cand_r1 | 29 | 1.212 | 0.587 |
| cand_r2 | 28 | 1.177 | 0.66 |
| cand_r6 | 23 | 1.265 | 0.6 |
| cand_c06 | 26 | 1.23 | 0.568 |
| cand_c03 | 30 | 1.263 | 0.631 |
| cand_s1 | 31 | 1.216 | 0.579 |

## Failure classes, original against the edit

| class | original | edit |
|---|---|---|
| 0 | 6 | 14 |
| 1 | 33 | 6 |
| 2 | 50 | 23 |
| 4a | 20 | 19 |
| 4b | 391 | 438 |

Failure classes that grew: 0 (6 to 14)

Original failures fixed: grasp_fail 39 of 51, other_fail 28 of 38. Broken successes became: {'other_fail': 4, 'grasp_fail': 17}

## Zones

| zone | n | original | edit | fixed | broke |
|---|---|---|---|---|---|
| hot block x 0.08-0.16, y 0.40-0.48, inside | 83 | 54.2% | 88.0% | 31 | 3 |
| hot block x 0.08-0.16, y 0.40-0.48, outside | 417 | 87.8% | 92.1% | 36 | 18 |
| low strip y < 0.44, inside | 92 | 56.5% | 78.3% | 27 | 7 |
| low strip y < 0.44, outside | 408 | 88.0% | 94.4% | 40 | 14 |
| low corner x < 0.06, y < 0.47 (post hoc), inside | 50 | 86.0% | 64.0% | 0 | 11 |
| low corner x < 0.06, y < 0.47 (post hoc), outside | 450 | 81.8% | 94.4% | 67 | 10 |
