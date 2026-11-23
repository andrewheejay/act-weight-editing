# ACT weight editing, autonomous run

You are the agent in a pre-registered experiment. SPEC.md is the plan. You are running unattended inside a loop that starts a fresh session whenever the last one ends. Nobody is watching and nobody can answer questions. Each session starts with no memory, so the files in this repo are your memory.

## Hard rules

These hold in every stage. Breaking one voids the result.

1. Never compute a gradient, and never run an optimiser over more than 3 free numbers.
2. Run rollouts only on reproduction seeds (1000 to 1499), design seeds (20000 to 20999) and validation seeds (30000 to 30499), plus the explicit poses SPEC.md asks for. Never run a seed of 1,000,000,000 or higher. Those are test seeds and only the referee may use them.
3. Write every rollout to `ledger/`. Stay inside the budgets (agent 100,000, baselines 50,000, auto-greedy 10,000 per task).
4. Check at most 10 candidates on the validation seeds per task, ever. Never change a candidate after seeing its validation result.
5. Never edit a protected file. Those are `SPEC.md`, `PLAN.json`, `CLAUDE.md`, `README.md`, everything under `referee/`, `run/`, `.claude/`, `tests_referee/` and `starter/`, the referee's outputs under `experiments/*/test/`, `reviews/*.json`, and the loop's own logs under `logs/`. Your own job output may go in `logs/` under a new file name.
6. Never push, tag, create a release, call `gh`, or start another `claude` process. The loop pushes for you after every session. Stay on the branch you start on, and never move it back to an older commit. To undo a commit, add a new one with `git revert`.
7. Never run the held-out test or look for its seeds. The referee draws them from a public randomness beacon after your freeze is public, then runs every arm itself.
8. Every number you write down must be recomputable from files in this repo. `python -m referee.recompute` does the common ones.

## What done means

The run is finished when the protocol is finished, not when the edit wins. SPEC.md lists four endings and all four are worth publishing. A clean null beats a win that bent a rule. If nothing passes validation within the budget, the stop rule applies. Freeze the best candidate, say plainly that it missed, and let the test give the number.

## How state works

- `python -m referee.status` is the source of truth. It recomputes everything from the files. A hook prints it at the start of every session and after every compaction.
- `PLAN.json` lists the tasks. A task is done when its checks pass. You cannot tick one off.
- `referee/CONTRACT.md` gives the exact file names and formats the checks read. Follow it to the letter.
- `referee/RULES.md` gives the exact rules for gates, the freeze and the verdict.
- `PROGRESS.md` is the lab notebook. Append to it. Never rewrite or remove earlier entries.
- `DEVIATIONS.md` records every departure from SPEC.md, with a date and a reason. Append only.

## Session routine

1. Read the status at the top of your context, then the tail of `PROGRESS.md`. Reread the part of SPEC.md for the current stage.
2. Work the open tasks from the top. Prefer finishing one task to starting three.
3. Verify like the referee will. Rerun `python -m referee.status` and read the "not yet" lines.
4. Before you stop, append a dated entry to `PROGRESS.md` saying what you did, the numbers you measured, what you ruled out, and the next step. Then commit everything. A hook blocks the stop until both are done.

End the session when the status moves to a referee or review step, when a long job is running and nothing else is useful, when your context is more than about 60% full, or when you are blocked. Short focused sessions with good handoff notes work better than one long one.

## Long computations

Anything that takes more than about 8 minutes goes through the job runner.

```
python run/jobs.py start sweep -- python -m awe.sweep --task transfer_cube
python run/jobs.py status
python run/jobs.py tail sweep
```

Start the job, write the handoff note, commit, and end the session. The loop waits for the job without spending tokens and starts the next session when it finishes. Make every long job resumable, because boxes restart.

## Reviews

At Gate 1 and at the freeze, a reviewer in a fresh session reads your work and tries to refute it. If it raises blocking issues, the status names a response file. Answer every issue there. Where the reviewer is right, fix the work and say what changed. Where it is wrong, show the evidence. Reviews are advisory, and both sides are published.

## The freeze

When stages 3 and 4 pass, write `experiments/<task>/freeze/REQUEST.json` and commit. The referee then replays samples of your earlier rollouts from a clean checkout, runs every arm on a few design seeds, diffs the edited checkpoint against the original, and either tags the freeze or writes `REJECTED-n.md` with reasons. After a tag, nothing about that task's edit, baselines or analysis can change.

## After the verdict

`experiments/<task>/test/VERDICT.md` is written by the referee and is final. Your report copies its verdict block unchanged and explains it. Do not rerun, reinterpret or soften it.

## Departures from the plan

- A fallback already listed in the Risks table of SPEC.md. Apply it and log it in `DEVIATIONS.md`.
- An operational change that does not alter what is tested, such as a renderer, a pinned version, or fewer seeds for a slow arm where SPEC.md allows it. Apply it and log it.
- Anything that would change a hypothesis, a pass rule, a seed set, a budget, an edit family, or the referee. You may not do this. Write `WAITING_FOR_ANDREW.md` at the repo root, saying what is wrong and what you recommend, commit, and stop. Use this only when the plan truly cannot be followed, or when you believe the referee itself has a bug.

## Practical notes

- `starter/` has scripts that were checked in a sandbox with random weights only. Use them as a starting point for `awe/`, and verify everything against the real checkpoint.
- The simulator keeps the cube pose in a module-level global, so run one simulator per process, never per thread.
- Pick one renderer in Stage 0 and never switch. Per-seed outcomes can differ between renderers. The harness must set the renderer itself from `stage0/renderer.json`, because the referee calls it from its own environment.
- Keep bulky caches out of git with `.gitignore`. Keep per-seed result files in git, gzipped if large.
- Record wall time and the number of rollouts for each stage in `PROGRESS.md`. Readers will want them.

## Compact instructions

When compacting, keep the current task id, every measured number, the names of running jobs, open reviewer issues, and anything you ruled out. Drop tool output you have already written to files.
