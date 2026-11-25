<!-- referee-verdict-begin -->
## Referee verdict for transfer_cube

**Ending. Strong win**

| Item | Value |
|---|---|
| Test seeds | 1,000 seeds, 1711781023 to 1711782022, drawn from beacon round 34241540 after the freeze was public |
| Primary score | benchmark success |
| Original policy | 81.7% |
| Edited policy | 90.1% |
| Net gain | 8.4 points, 95% interval 5.9 to 10.9 |
| Seeds fixed and broken | 131 fixed, 47 broken |
| Exact McNemar p | 2.29e-10 |
| Share of failures removed | 45.9%, 95% interval 34.8% to 55.8% |
| Guard score, strict handover | 78.1% to 87.3%. A drop of up to 2.0 points is allowed. Passed |
| H2, matched random edits | 1 of 40 gained as much, and at most 1 may. Passed |
| H3, best simple baseline | deviation_gain at 6.1 points, against 8.4 for the edit on the same seeds. No baseline matched the edit |
| Auto-greedy edit, built with no LLM | 8.4 points. It matched the agent edit |
| Cross-check with LeRobot's own evaluator | original agreed on 100.0% of 357 episodes. agent_edit agreed on 100.0% of 357 episodes |
| Size of the edit | 65,536 weights, 0.191% of the weights that run at inference |
| Reviews by a separate session | Gate 1 passed review in round 2. Freeze request 1 passed review in round 2 |
| Freeze | tag freeze-transfer_cube-1, commit dd626ffe5163, rules from genesis-1 |
| Commitment | https://github.com/andrewheejay/act-weight-editing/releases/tag/freeze-transfer_cube-1, published 2026-11-24T13:36:24Z |

Why this ending.
- significant, beats the random edits, and no simple baseline matches it
- removes at least 25% of failures

<!-- referee-verdict-end -->
