# The referee's rules

These are the exact rules the code in this folder enforces. They restate SPEC.md in mechanical form, and they were fixed at the genesis tag, before any rollout. Where SPEC.md left a detail open, the choice made here is marked as a clarification.

## Who does what

| Role | What it is | What it may do |
|---|---|---|
| Agent | Claude Code sessions started by `run/loop.sh` | Everything in SPEC.md except the test |
| Reviewer | A separate Claude Code session with read-only tools | Read, recompute, and raise blocking issues |
| Referee | The Python code in this folder, run by the loop with no LLM | Check gates, tag the freeze, draw the seeds, run the test, write the verdict |
| Andrew | The person | Set the box up, publish at the end, and decide anything the rules do not cover |

## Seeds

| Set | Seeds | Who runs it |
|---|---|---|
| Reproduction | 1000 to 1499 | Agent |
| Design | 20000 to 20999 | Agent |
| Validation | 30000 to 30499 | Agent, for at most 10 candidates per task, plus one check per baseline family |
| Test | 1,000 consecutive seeds starting between 1,000,000,000 and 1,999,999,999 | Referee only |

The test seeds do not exist while the agent works. They are drawn after the freeze is public, as described below. The harness must refuse any seed at or above 1,000,000,000 unless the referee supplies the draw file.

Clarification. The spec's first draft reserved seeds 40000 and up for the test. With a mechanical referee those are retired and never run.

## Gate 0, the setup reproduces

All of these, recomputed from the per-seed files.

1. LeRobot's own evaluator on seeds 1000 to 1499 gives a success rate inside the task's window. The window is 78% to 88% for transfer cube. For insertion it is 15.6% to 25.6%, which is the published 20.6% plus or minus 5 points (a clarification, by analogy).
2. The fast harness agrees with that run on at least 98% of the 500 seeds, and every mismatch is listed in `mismatches.md`.
3. Two runs of the same 100 seeds, and a third with every head scaled by 1, are identical on both success scores and on the trajectory hash.
4. The five named tests in `tests/` pass.
5. The referee replays 20 random reproduction seeds itself with `python -m awe.run_arm`, and all 20 match the agent's file.
6. The stock evaluator command recorded in `env.json` runs. The referee runs it on 5 random reproduction seeds, and at least 4 must match the Stage 0 stock run.

How the stock file is read. LeRobot 0.7.0 saves plain lists with no seeds. Its evaluator runs episode i on seed `--seed` plus i, in order, so the referee reads the Stage 0 file as seeds 1000, 1001 and so on, and reads its own runs from the seed it passed. Check 6 is what confirms that reading on the box. Files that carry seeds, such as the published hub file, are read by their seeds.

Check 5 is repeated whenever the harness code changes, and again at the freeze. A mismatch sends the run back to Stage 0.

## Gate 1, the diagnosis is fixed

The referee recomputes three decisions from the 1,000 design seeds and `decisions.json` must match them.

1. Primary score. Strict handover becomes primary if benchmark success minus strict success is 10 points or more at the default setting. Otherwise benchmark success is primary. The other score is the guard.
2. Operating point. 100-step chunks, unless a chunk of 50 or 25 beats 100 by 3 points or more on the primary score. Then the better of those two.
3. Proceed. If fewer than 5% of design episodes fail on the primary score, this task is not edited and the next task in the campaign becomes the main experiment.

A reviewer then reads the diagnosis. When the review is settled, the referee records the hash of `decisions.json`. That file can never change afterwards.

## Reviews

A review happens at Gate 1 and at each freeze request. The reviewer returns `pass` or `revise` with a list of blocking issues. The agent must answer every blocking issue in writing. There are at most 2 rounds per gate. If issues remain after round 2 the run continues, the gate is marked as having unresolved review issues, and both sides are published. Two kinds of reviewer failure are kept apart. When the reviewer cannot run at all, for example because of a usage limit, a sign-out or an API error, nothing is counted and the loop waits and tries again, so a usage limit never skips a review. When the reviewer runs 3 times and never returns a usable verdict, the gate is recorded as not independently reviewed. The Gate 1 record and the verdict both say so, and the verdict lists every gate's review outcome.

## Budgets, per task

| Budget | Cap |
|---|---|
| Agent rollouts on design and validation seeds, sweep included | 100,000 |
| Simple baselines | 50,000 |
| Auto-greedy script, beyond the shared sweep | 10,000 |
| Candidates checked on validation seeds | 10 |
| Baseline families checked on validation seeds | 5 |

Counts come from the ledger. Rollouts under the `setup` and `stress` budgets are logged and not capped.

## The freeze

A freeze request is accepted only if all of this holds at one clean commit.

1. Every task check for stages 0 to 4 passes.
2. The tests pass from a clean checkout of that commit.
3. The referee replays 20 random reproduction seeds and 20 random design seeds from that checkout, and all match the agent's files.
4. The edit obeys the family limits. At most 6 heads, scales from 0, 0.5, 1.5, 2, 3, 4, sharpening factors from 0.5, 2, 4, at most one latent shift, and at most 3 free numbers for a direction edit or any other closed-form edit.
5. The exported edited checkpoint has the same tensors and shapes as the original, differs from it, leaves the vision backbone untouched, and changes something other than the action head and the decoder position table. Its hash matches `edit.json`.
6. The manifest lists one original arm, one agent edit arm, one auto-greedy arm, 40 matched random edits (or all that exist, if fewer), at least one baseline from each of the four families, 4 dose-response arms, and the two temporal ensembling arms. If the operating point is not 100 steps, it also lists the original and the edit at 100.
7. Every arm runs on 3 design seeds from the clean checkout.
8. The ledger is inside the budgets and holds no illegal seed.
9. No protected file differs from the genesis tag. `PROGRESS.md` and `DEVIATIONS.md` have only grown. No gradient or optimiser code is present. `decisions.json` is unchanged since Gate 1.

On success the referee tags the commit `freeze-<task>-1`. On failure it writes `REJECTED-<n>.md` with reasons. After 5 rejections the run stops for a person.

There is one confirmatory freeze per task. There is no second attempt and no redraw.

## The seed draw

1. The referee pushes the freeze tag and publishes a commitment as a GitHub release. The commitment holds the frozen commit hash, the hashes of `edit.json`, the manifest and the edited checkpoint, and this rule. GitHub's `published_at` time is the timestamp. The repo that holds the release must be public.
2. The seeds come from round R of the drand quicknet beacon, where R is the first round emitted at least 600 seconds after that timestamp. Round r is emitted at `1692803367 + 3 (r - 1)` seconds Unix time.
3. The beacon value is checked against the network's public key (BLS on BLS12-381, scheme `bls-unchained-g1-rfc9380`). Its randomness is the SHA-256 of the signature.
4. `start = 1000000000 + int(SHA256("awe-test-seeds-v1|<task>|<frozen commit>|<randomness hex>")[:8]) mod 1000000000`. The test seeds are `start` to `start + 999`.
5. The draw is committed and pushed before any test rollout runs.

Anyone can check all of this later with `python -m referee.verify_draw experiments/<task>/test --online`.

Why this is fair. The round's value cannot be known before it is emitted, so when the edit was frozen nobody could know which cube positions it would be tested on. The frozen commit is in the hash, so the seeds also could not be chosen by picking when to publish.

## The test

Every arm in the manifest runs once on the drawn seeds, from a clean checkout of the frozen tag. Arms that the manifest marks with `n_seeds` run on the first `n_seeds` of the block. An arm that fails twice stops the run for a person.

The two main arms are also run through LeRobot's unmodified evaluator on the same seeds, as many as fit in the time budget and at least 100. If the fast harness and that evaluator agree on fewer than 98% of those episodes for either arm, the verdict carries a warning that the post must repeat.

## The verdict

Let b be the number of test seeds the edit fixed and c the number it broke, on the primary score, at the operating point from Gate 1.

1. Significant gain. The exact two-sided McNemar p-value is below 0.05 and b is greater than c. The p-value is `min(1, 2 P(X <= min(b, c)))` for X binomial with b + c trials and probability one half.
2. Large. `(b - c)` divided by the number of seeds the original failed is at least 0.25.
3. Guard. The guard score of the edit is no more than 2 points below the original's.
4. H2. With k matched random edits, at most `floor(0.05 (k + 1)) - 1` of them may gain as much as the edit. With k = 40 that is 1. Below k = 19 the rule cannot be met and H2 counts as untested. (The generalisation to k below 40 is a clarification.)
5. H3. A simple baseline matches the edit if its gain is at least the edit's gain on the same seeds. The baselines are the four families and the original policy with temporal ensembling. (Using point gains for "ties or wins" is a clarification.)

| Ending | Rule |
|---|---|
| Strong win | Significant, guard holds, H2 passes, no baseline matches, and large |
| Modest win | The same, but not large |
| Partial | Significant and guard holds, but H2 fails or is untested, or a baseline matches |
| Null | Not significant, or the guard fails |

Clarification. A significant gain that fails the guard is a null, and the post says why.

The auto-greedy arm, the dose-response arms, the ensembled edit and the 100-step pair are reported beside the verdict and do not change the ending. If the auto-greedy edit gains as much as the agent's edit, the verdict says so in its own line.

Intervals are 95% percentile intervals from 10,000 bootstrap resamples of the test seeds, seeded from the beacon value.

## What stops the run for a person

- The agent writes `WAITING_FOR_ANDREW.md` because the plan cannot be followed or the referee looks wrong.
- 5 freeze rejections, a test arm that fails twice, or a referee step that fails 3 times in a row.
- 4 sessions in a row that change nothing.
- A protected file that cannot be restored, or a sign-in failure.

## Changing these rules

Only a person can. They edit the files, commit, and run `run/setup_box.sh --regenesis "reason"`. That writes the reason into `DEVIATIONS.md` and publishes a new genesis tag, so every rule change is dated and public.
