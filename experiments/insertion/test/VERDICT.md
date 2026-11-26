<!-- referee-verdict-begin -->
## Referee verdict for insertion

**Ending. Modest win**

| Item | Value |
|---|---|
| Test seeds | 1,000 seeds, 1642360013 to 1642361012, drawn from beacon round 34273862 after the freeze was public |
| Primary score | strict handover |
| Original policy | 11.2% |
| Edited policy | 19.3% |
| Net gain | 8.1 points, 95% interval 5.5 to 10.7 |
| Seeds fixed and broken | 129 fixed, 48 broken |
| Exact McNemar p | 9.3e-10 |
| Share of failures removed | 9.1%, 95% interval 6.3% to 11.9% |
| Guard score, benchmark success | 19.7% to 30.4%. A drop of up to 2.0 points is allowed. Passed |
| H2, matched random edits | 0 of 40 gained as much, and at most 1 may. Passed |
| H3, best simple baseline | constant_offset at 3.6 points, against 8.1 for the edit on the same seeds. No baseline matched the edit |
| Auto-greedy edit, built with no LLM | 7.4 points. It did not match the agent edit |
| Cross-check with LeRobot's own evaluator | original agreed on 100.0% of 255 episodes. agent_edit agreed on 100.0% of 255 episodes |
| Size of the edit | 98,304 weights, 0.287% of the weights that run at inference |
| Reviews by a separate session | Gate 1 passed review in round 2. Freeze request 1 passed review in round 1 |
| Freeze | tag freeze-insertion-1, commit e27fe8eae63b, rules from genesis-1 |
| Commitment | https://github.com/andrewheejay/act-weight-editing/releases/tag/freeze-insertion-1, published 2026-11-25T16:32:29Z |

Why this ending.
- significant, beats the random edits, and no simple baseline matches it
- removes under 25% of failures

<!-- referee-verdict-end -->
