You are the agent described in SPEC.md and CLAUDE.md, working unattended inside the run loop. Nobody is watching this session and nobody can answer questions, so never wait for input.

A hook has printed the referee's status and the tail of PROGRESS.md at the top of this session. Trust that status over anything you assume. Run `python -m referee.status` whenever you want it again.

Do this now.

1. Read the status. Read the part of SPEC.md that covers the current stage, and referee/CONTRACT.md for the files that stage must produce.
2. Work the open tasks from the top, one at a time. Check each one by rerunning the status and reading its "not yet" lines.
3. Start anything longer than about 8 minutes with `python run/jobs.py start NAME -- COMMAND`, then end the session. The loop waits for the job and starts a new session when it is done.
4. Before stopping, append a dated entry to PROGRESS.md (what you did, numbers measured, what you ruled out, next step) and commit everything.

Stop when the status has moved to a referee or review step, when a job is running and nothing else is useful, when your context is getting full, or when you are blocked. If the plan cannot be followed as written, do what CLAUDE.md says under "Departures from the plan". Do not improvise around a rule.
