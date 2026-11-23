# ACT weight editing

Can an AI agent make a small robot policy better by editing its weights directly, with no training? This repo runs that experiment end to end with no person in the loop, and it is built so a stranger can check the result was fair.

`SPEC.md` is the plan. This page is how to run it.

## How the run works

Three roles share the work, and only one of them is an LLM doing research.

| Role | What it is | What it does |
|---|---|---|
| Agent | Fresh Claude Code sessions, one after another | Reproduces the benchmark, diagnoses failures, maps the heads, designs the edit and its rivals, writes the report |
| Reviewer | A separate read-only Claude Code session, at xhigh effort | Tries to refute the diagnosis and the freeze package |
| Referee | Plain Python in `referee/`, no LLM | Checks every gate, tags the freeze, draws the test seeds, runs the test, writes the verdict |

`run/loop.sh` is the engine. On each pass it asks the referee where things stand and starts exactly one of those three.

Think of a relay race where every runner has amnesia. The baton is the repo. Each session reads the referee's status and the lab notebook, does one leg, writes down what it learned, and hands off.

## Why the result can be trusted

- **The rules came first.** Setup publishes a `genesis-1` tag and release that fix the spec, the plan and the referee before any rollout.
- **The test seeds did not exist while the agent worked.** After the edit is frozen and public, the referee draws 1,000 seeds from a public randomness beacon (drand) round that is emitted ten minutes later. Nobody could know them in advance.
- **The agent cannot touch the referee.** Hooks and deny rules block it, the loop runs from a read-only copy, and any protected file that changes is restored before the next session.
- **The headline is measured twice.** The referee's harness runs every arm, and LeRobot's own evaluator re-runs the two main arms on the same seeds.
- **Anyone can re-check.** `python -m referee.verify_draw experiments/transfer_cube/test --online` re-derives the seed draw from public data.

The exact rules are in `referee/RULES.md`.

## Before you start

- [ ] A Linux box. The spec suggests an AWS g6.2xlarge, and an 8-core CPU box also works.
- [ ] `git`, `gh`, `jq`, `curl`, `tmux`, `python3` and `uv` installed.
- [ ] Claude Code 2.1.259 or later, signed in with your Claude account. Run `claude`, then `/login`.
- [ ] `gh auth login` done, and `git config user.name` and `user.email` set.
- [ ] An empty **public** GitHub repo named `act-weight-editing`, cloned onto the box, with this kit unpacked inside it.
- [ ] The ntfy app on your phone, subscribed to a topic name that is hard to guess.

Want the work private until you publish? Keep this repo private, create a tiny public repo, and put its name in `AWE_WITNESS_REPO`. Only hashes go there.

## Start

```bash
bash run/setup_box.sh                         # checks the box, installs, publishes genesis-1
nano ~/.awe/act-weight-editing/config.env     # set AWE_NTFY_TOPIC, and anything else you like
tmux new -s awe ./run/loop.sh                 # start the run, then detach with Ctrl-b d
```

Setup takes a few minutes. It also runs the referee's own tests, which play a whole simulated campaign on your box, so you know the plumbing works before the real run starts.

## Watch

| Where | What you see |
|---|---|
| Your phone | A push at every gate, at the freeze, when the seeds are drawn, at the verdict, and whenever the run needs you |
| GitHub | `PROGRESS.md` grows after every session, and `logs/sessions.tsv` lists each one |
| The box | `python -m referee.status`, `tail -f logs/loop.log`, `tail -f logs/referee.log` |

## Pause, stop, resume

| You want to | Do this |
|---|---|
| Pause after the current step | `touch PAUSE`, and `rm PAUSE` to carry on |
| Stop now | `tmux kill-session -t awe` |
| Resume after any stop | `tmux new -s awe ./run/loop.sh`. It picks up from the files |
| Sit in and steer | `touch PAUSE`, then `bash run/interactive.sh`. That session also shows up in the Claude app |
| Run only the first task | Set `AWE_SCOPE=main` in the config |
| Change how hard sessions think | Set `AWE_EFFORT` (high, most sessions), `AWE_EFFORT_DESIGN` (xhigh, stages 1 and 3) or `AWE_REVIEW_EFFORT` (xhigh, every review) in the config. Each session's level is in `logs/sessions.tsv` |

While the run is going, change the repo only on the box. A commit made on GitHub puts the remote ahead, and then the loop's pushes fail until you pull. The loop tells you when 3 pushes in a row have failed.

## When it asks for you

The loop stops and notifies you when a rule says a person must decide. The reason is in `WAITING_FOR_ANDREW.md` or in the status.

| Reason | What to do |
|---|---|
| The agent says the plan cannot be followed | Read its note. Fix the cause or change the plan (see below). Delete the file and restart |
| Four sessions in a row changed nothing | Read the tail of `PROGRESS.md`. Often one hint in a new `PROGRESS.md` entry is enough |
| Many sessions in a row failed to run, and it was not a usage limit | Read the message and `logs/loop.log`. Fix the box, for example the network or a Claude Code update, delete the file and restart |
| The repo left the run's branch | `git checkout` the run's branch, merge anything worth keeping, delete the file and restart |
| The freeze was rejected five times | Read the `REJECTED-n.md` files. The referee or the agent has a bug |
| A test arm failed after the seeds were drawn | Read `test/FAILED.json`. Fix only infrastructure, never the edit, delete that file and restart. The same seeds are reused |
| A referee or review step failed several times in a row | Read `logs/referee.log`. If a check is wrong, see the next section |
| Claude Code is signed out | `claude`, `/login`, restart |

## Changing a rule

Only you can. Edit the file, commit, and run

```bash
bash run/setup_box.sh --regenesis "why the rule changed"
```

That writes the reason into `DEVIATIONS.md` and publishes `genesis-2`, so the change is dated and public. Expect to need this once or twice. The referee was tested against a simulated agent, and its first contact with the real harness may show a check that is too strict.

## Usage and cost

- **Claude usage.** The loop runs one agent at a time, and most wall time is spent waiting on simulations, which costs no tokens. If a usage limit hits, the loop sleeps and retries every 30 minutes until it clears. That holds for the reviewer too, so a usage limit never skips a review. Stage 1, Stage 3 and every review run at xhigh effort, which uses more of the weekly limit than high. If the limit keeps pausing the run, lower `AWE_EFFORT` first, since it only covers the routine sessions.
- **Fable.** The default model is `opus`. You can set `AWE_MODEL_DESIGN=fable` for the two thinking-heavy stages. In an unattended run, Fable requests beyond your plan's allowance are billed to usage credits without asking, so set a monthly spend limit first.
- **The box.** It bills for every hour it is on, including the hours Claude is thinking. Set `AWE_SHUTDOWN_ON_EXIT=1` to power it off when the run ends, and set an AWS budget alert.

## Before you publish

1. Run `python -m referee.verify_draw experiments/<task>/test --online` for each task.
2. Check that nothing protected changed. `git diff --stat genesis-1 freeze-transfer_cube-1 -- referee run SPEC.md PLAN.json CLAUDE.md` should print nothing, unless you ran a regenesis.
3. Follow `publish/CHECKLIST.md`, which the agent writes at the end.
4. Back up `logs/agent/` before you terminate the box. Those are the full session transcripts, with credentials masked.

## What this kit does not do

- It checks the rules, the numbers and the order of events. It cannot check that the science is wise. The reviewer helps, and its notes are published either way.
- The agent and the referee run as the same Linux user. Protection comes from hooks, deny rules, the read-only copy, automatic restore, and public diffs against the genesis tag. It does not come from operating-system isolation.
- GitHub's clock is the timestamp for the commitment. A reader has to trust that clock.
- The referee has been tested against a scripted agent with a fake harness, and never yet against the real one. One pass of the real loop was run against a real Claude Code 2.1.330 that was signed out, which exercised its flags, its settings file and its session-start hook. No signed-in session has run under the loop yet.

## Layout

```
SPEC.md  PLAN.json  CLAUDE.md      the plan, the task graph, the agent's standing rules
referee/                           gates, seed draw, test, verdict. RULES.md and CONTRACT.md live here
run/                               loop.sh, setup_box.sh, jobs.py, notify.sh, watchdog.sh, interactive.sh
.claude/                           hooks that guard the rules and enforce clean handoffs
starter/                           scripts from the spec's sandbox checks, as a head start for awe/
tests_referee/                     the referee's own tests.  python -m pytest tests_referee -q
PROGRESS.md  DEVIATIONS.md         the lab notebook and the list of departures, both append-only
awe/  tests/  experiments/  ledger/  publish/     written by the agent during the run
```
