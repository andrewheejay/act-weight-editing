# ACT Weight Editing Experiment Spec

> This file is the pre-registered plan, exported from the online spec on November 22, 2026. It is protected. The agent may not change it, and a person who does must publish a new genesis tag.

Nov 22, 2026 · @Andrew Lee

## At a glance

We will test whether an AI agent can make a small robot policy better by editing its weights directly, with no training, and whether that gain holds on 1,000 cube positions the agent never saw.

| Item | Plan |
| --- | --- |
| Question | Does weight-space programming work on a small imitation policy, and does the gain survive a large held-out test? |
| Model | ACT, using the public LeRobot checkpoint for the ALOHA transfer-cube task |
| Roles | Claude Code diagnoses, sweeps and edits. A scripted referee with no LLM checks the gates, draws the test seeds and runs the final test. Andrew sets the question and publishes. |
| What is new | A small policy that is not a VLA, 1,000 paired test seeds where earlier posts used 3 to 30 trials, matched random controls, and a setup anyone can rerun on one cheap box |
| What ships | A post, the code, the edited checkpoint, and the full agent logs |
| If no edit works | We publish the head map and the negative result anyway |
| Why it matters | ACT still ships in LeRobot and is widely used on low-cost arms, so a fix that needs no data and no training would help many people |

**Picture it like tuning a piano that is already built.** We do not rebuild the instrument. We find the few strings that are off, turn those pegs, and then check the result on songs we did not practise with.

**How to read this.** The first four sections, Autonomous run and the last one are for you. Everything in between is reference for Claude Code, written so it can run the stages without guessing.

## Key terms

These twelve words carry the whole spec, so they are defined once here.

| Term | Plain meaning |
| --- | --- |
| Weight edit | Changing numbers inside a trained model by hand, with no retraining |
| Attention head | One small sub-unit of a transformer. Each head routes one slice of information |
| Action chunk | The 100 future actions ACT predicts in one go, which is 2 seconds of motion |
| Rollout | One full simulated episode, from reset to the end |
| Seed | A number that fixes where the cube starts. Same seed, same cube position |
| Paired comparison | Running two versions of the policy on the exact same seeds, so only the edit differs |
| Dose response | Turning a dial in steps (×0.5, ×1.5, ×2, ×3) and checking the behaviour moves in step with it |
| Random control | The same kind of edit applied to randomly picked heads, to show the chosen ones are special |
| Freeze | Locking the edit in git before the final test, so the test cannot shape the edit |
| Budget | The cap on how many rollouts the agent may spend while designing the edit |
| Referee | Plain code with no LLM that checks the gates, runs the test and writes the verdict |
| Randomness beacon | A public service that publishes a fresh random number every few seconds. Nobody can know a number before it comes out |

## What we are testing

The edit counts as a win only if it beats the original policy on unseen seeds, beats random edits, and is compared honestly against simple fixes. These rules are fixed now, before any rollout, so the results cannot bend them.

| # | Claim | Test | Pass rule |
| --- | --- | --- | --- |
| H1 | The edit helps on unseen seeds | Original and edited policy on the same 1,000 test seeds | The gain is significant by an exact McNemar test at 0.05. It counts as large if it removes at least 25% of the original's failures |
| H2 | The chosen parts are special | 40 random edits matched on count, layer type and scale | At most 1 of the 40 gains as much as the real edit |
| H3 | It is more than an output tweak | The best simple baseline, tuned on the same design seeds | Reported side by side. If a baseline ties or wins, the headline says so |
| H4 | The recipe repeats | The same procedure and budget on the insertion checkpoint | Reported as a second, independent result |

**One guard on top.** The benchmark's own success rule is loose (see Measurements), so we also score a strict handover. The edit must not lower strict success by more than 2 points.

### Four possible endings

| Ending | When | What we post |
| --- | --- | --- |
| Strong win | The gain is significant and large, it beats the random edits, and no simple baseline matches it | "Editing N heads fixed X% of ACT's failures, with no training" |
| Modest win | The gain is significant but under 25% of failures, and it still beats the random edits and the baselines | "A small but real fix. Here is its size and its limits" |
| Partial | The gain is significant, but random edits or a simple baseline do as well | "It works, but it is not special. Here is what did as well" |
| Null | No significant gain within the budget | "We could not edit ACT. Here is the head map and why" |

All four are worth posting. Every ending also yields a map of what each ACT head does, which is useful to others whatever the headline is.

**Exact wording.** A script applies these rules, so each one is written out in mechanical form in `referee/RULES.md`, fixed before any rollout. One case the table leaves open is a significant gain that fails the guard. That counts as a null.

### My forecast, written before any run

These are rough guesses, recorded so we can check them later.

| Ending | My rough odds |
| --- | --- |
| Strong win | 1 in 5 |
| Modest win | 1 in 5 |
| Partial | 1 in 3 |
| Null | 1 in 4 |

The reason for the caution is simple. ACT is small, so each head probably does several jobs at once, and turning one up may fix one zone while disturbing another.

## What we already know

Three things are clear before any rollout. They come from LeRobot's code, from checks I ran in a sandbox, and from the published evaluation.

### ACT is small and easy to take apart

| Fact | Why it matters here |
| --- | --- |
| 48 attention heads run at inference. 32 sit in the encoder, 8 in decoder self-attention and 8 in decoder cross-attention | A full sweep covers 48 heads, where π₀.₅ has 288 |
| The 8 decoder self-attention heads output the same vector whatever the camera sees | They can only act as fixed offsets, so 40 heads carry information |
| ACT makes 4 decisions per episode, one every 100 steps, each from a single camera frame | Every failure traces back to one of four decisions |
| Each of the 100 actions in a chunk is computed independently of the other 99 | A time-shift edit equals plain post-processing, so it counts as a simple baseline |
| A third of the checkpoint never runs at inference (17.4M of 51.6M weights, the training-only VAE encoder) | "Weights touched" should be reported against the 34.2M that do run |
| Scaling one head means multiplying 64 columns of one 512 by 512 matrix | One head is 32,768 weights, or 0.06% of the model, and the result saves as a normal checkpoint |
| The whole start square covers about 78 by 77 pixels of the camera image, which is under 3 by 3 cells of the vision backbone's feature grid | Grasping needs centimetre precision from features about 8 cm wide, which fits the many near-miss grasps |

### The benchmark's success rule is loose

The simulator scores success the instant the left gripper touches the cube while the cube is off the table. The right gripper never has to let go. An edit could raise that score without improving the handover, so we add a strict handover score (see Measurements).

### The published failures are not random

The [model card](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human) reports 83.0% success over 500 episodes. Its [per-episode file](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/blob/main/eval_info.json) shows where the other 85 stopped.

| Where the episode stopped | Episodes | Share of failures |
| --- | --- | --- |
| The right gripper never touched the cube | 8 | 9% |
| It touched the cube and never lifted it | 41 | 48% |
| It lifted the cube and the left gripper never reached it | 36 | 42% |

*Chart in the online spec.* Published failures cluster in the low-y strip of the start square. 38 of 102 episodes fail below y = 0.44, against 47 of 398 above it.

- **Failures cluster.** When an episode fails, the episode with the nearest cube position fails 59% of the time. Chance would give 17%.
- **One zone is weak.** 45% of all failures start in the fifth of the square with the lowest y. That strip fails 37% of the time, and the rest fails 12%.
- **Most look like near misses.** Seed 1000 touched the cube for 69 steps and never lifted it. Seed 1491 held it in the air for roughly 150 steps and the handover never connected.

**What this means.** A consistent error in one zone is the kind of fault an edit can fix. It is also the kind a simple fix might handle, which is why H3 matters.

These numbers came through a web fetch of the published file, so Stage 0 re-derives them on the box.

## Setup

Everything runs on one Linux box with pinned versions, and no result is ever compared across machines.

| Piece | Choice |
| --- | --- |
| Policy | `lerobot/act_aloha_sim_transfer_cube_human`, 51.6M weights, the 80k-step checkpoint |
| Second policy, for H4 | `lerobot/act_aloha_sim_insertion_human`, published at 20.6% success |
| Simulator | gym-aloha 0.1.5 on MuJoCo, tasks AlohaTransferCube-v0 and AlohaInsertion-v0 |
| Episode | 400 steps at 50 Hz (8 seconds), one top camera at 480 by 640, 14 joint targets per step |
| Library | LeRobot 0.7.0 from PyPI on Python 3.12 |
| Inference setting | The checkpoint's own, which is 100-step chunks and no temporal ensembling |
| Box | AWS g6.2xlarge on Ubuntu 22.04. A CPU-only box also works (see Compute) |
| Renderer | One renderer for the whole project, chosen in Stage 0 |

**The only thing that changes between episodes is where the cube starts.** Each seed puts it at one point in a 20 cm square (x from 0 to 0.2 m, y from 0.4 to 0.6 m), always facing the same way. The policy and the physics have no randomness, so one seed gives one outcome.

### Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "lerobot[aloha,evaluation]==0.7.0" scipy matplotlib
export MUJOCO_GL=egl
uv pip freeze > requirements.lock
```

In my sandbox this resolved to torch 2.15.0, MuJoCo 3.8.1, dm-control 1.0.41 and gymnasium 1.4.0.

### Three traps I already hit

1. **The checkpoint is in an old format.** The hub repo holds no processor files, so current LeRobot needs it converted first. LeRobot ships a script for that.
2. **Rendering eats almost all the time.** The stock simulator draws three cameras on every step. ACT looks at one camera, four times per episode. Skipping the unused renders left the trajectories identical in my test and cut the simulator's share of an episode from 84 seconds to under half a second on a CPU.
3. **One simulator per process.** The cube position travels through a module-level global, so parallel rollouts need separate processes, never threads.

```bash
python -m lerobot.processor.migrate_policy_normalization \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
  --output-dir checkpoints/transfer_cube

lerobot-eval --policy.path=checkpoints/transfer_cube \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=500 --eval.batch_size=50 \
  --policy.device=cuda --policy.use_amp=false --seed=1000
```

The second command is the published evaluation, on the same 500 seeds (1000 to 1499).

## Rules of the game

The agent designs the edit with a fixed toolbox and a fixed budget, and it never sees the test seeds.

### Who does what

| Who | Does |
| --- | --- |
| Andrew | Sets the question, sets the box up once, publishes the result |
| Claude Code, the agent | Diagnoses failures, maps the heads, designs the edit, writes the simple baselines |
| A reviewer, in a separate read-only Claude Code session | Tries to refute the diagnosis and the freeze package |
| The referee, scripts with no LLM | Checks every gate, tags the freeze, draws the test seeds, runs the test, computes the verdict |
| Other scripts with no LLM | Run rollouts and build the auto-greedy edit |

### Seed sets

| Set | Seeds | Size | Who may run it | Used for |
| --- | --- | --- | --- | --- |
| Reproduction | 1000 to 1499 | 500 | Anyone | Checking our setup against the published run |
| Design | 20000 to 20999 | 1,000 | Agent | Diagnosis, head maps, sweeps |
| Validation | 30000 to 30499 | 500 | Agent, for at most 10 candidate checks | Choosing the final edit |
| Test | 1,000 in a row, starting between 1,000,000,000 and 1,999,999,999 | 1,000 | The referee, once, after the freeze | The reported result |
| Any later freeze | A fresh draw of 1,000 | 1,000 each | The referee | Reported beside the first result, never in place of it |

The test seeds do not exist while the agent works. After the freeze is public, the referee draws them from a public randomness beacon (see Autonomous run). The rollout code refuses any seed from 1,000,000,000 up unless the referee supplies the draw. Every rollout is written to a ledger with its seed, its arm and its outcome.

**An honest limit.** Test positions are new, but they come from the same 20 cm square. With 1,000 design points in that square, a test cube usually starts a few millimetres from a design cube. Held-out seeds guard against lucky picks, not against new conditions. Stage 6 adds positions outside the square for that.

### What the agent may and may not do

| Allowed | Not allowed |
| --- | --- |
| Reading the weights and running any maths on them | Computing gradients of any kind |
| Forward passes on frames recorded from design seeds | Running an optimiser over more than 3 free numbers |
| Rollouts on design and validation seeds, within budget | Touching a test seed |
| Edits from the five families below | Editing the vision backbone, in this first version |
| Looking at videos and measurements of design rollouts | Changing anything after the freeze |

### Budget

| Who | Cap |
| --- | --- |
| Agent | 100,000 rollouts across design and validation, sweep included |
| Simple baselines | 50,000 rollouts on design seeds |
| Auto-greedy script | 10,000 rollouts beyond the shared sweep |

Forward passes on recorded frames are logged but not capped. The final post reports what was actually spent.

### Edit families

| Family | What changes | Free numbers |
| --- | --- | --- |
| Head scaling | The output of up to 6 of the 48 heads, each at ×0, ×0.5, ×1.5, ×2, ×3 or ×4 | Which heads and which scales |
| Head sharpening | How sharply up to 6 heads focus, by scaling their query weights by 0.5, 2 or 4 | Which heads and which factors |
| Latent shift | The 32-number style input, fixed at zero today, moved along one direction | One direction and one size |
| Direction edit | One direction added to or removed from a layer's output weights, in closed form | At most 3 |
| Other | Any closed-form change inside the transformer, with its hypothesis written first | At most 3 |

Edits that only reshape the output count as simple baselines. That covers shifting, rescaling or mixing the action head's outputs and re-ordering the decoder's position table, because plain post-processing of the actions does the same thing. I checked the bias shift and the position table directly.

### What the agent must write down

- `diagnosis.md` before the sweep, naming the target failure, the behaviour number that tracks it, and the evidence that it is a near miss.
- One line in `hypotheses.md` before each candidate edit is evaluated.
- `edit.json` for the final edit and `baselines.json` for the simple fixes.
- Every Claude Code transcript, saved to `logs/agent/` with the model name.

### Four gates

| Gate | After | The referee checks |
| --- | --- | --- |
| G0 | Stage 0 | The setup reproduces the published number and our harness agrees with LeRobot's eval. It replays 20 random rollouts itself |
| G1 | Stage 1 | The primary metric and the operating point follow the rules, and a reviewer has tried to refute the diagnosis, all before any sweep |
| G2 | Stage 4 | The freeze package is complete, every arm runs from a clean checkout, and a reviewer has read it. Then it tags the freeze |
| G3 | Stage 5 | It publishes the commitment, draws the test seeds, and runs the test once |

### Changes to this plan

This spec goes into the repo as `SPEC.md` before Stage 1. Any later change is written in `DEVIATIONS.md` with a date and a reason, so readers can see what was planned and what was not.

The spec, the plan and the referee are tagged `genesis-1` and published before any rollout. The agent may log a fallback from the Risks table or an operational change. Only a person may change a rule, and doing so publishes a new genesis tag with the reason.

## Autonomous run

The whole protocol runs with no person in the loop. A scripted referee stands in for Andrew at every gate, and the test seeds are drawn from a public randomness beacon after the freeze is public.

**Picture a sealed envelope and a lottery draw on live TV.** The agent seals its edit in the envelope first. Only then are the numbers drawn, in public, by a machine nobody controls. Whatever is in the envelope could not have been fitted to the numbers.

### Three roles

| Role | What it is | What it may never do |
| --- | --- | --- |
| Agent | Fresh Claude Code sessions, one after another | Edit the rules or the referee, push or tag, run the test |
| Reviewer | A separate Claude Code session with read-only tools | Change any file |
| Referee | Python scripts with no LLM, in `referee/` | Use judgment. It only applies rules written before the run |

### One pass of the loop

`run/loop.sh` asks the referee where things stand, then starts exactly one of three things.

| The status says | What runs | Example |
| --- | --- | --- |
| Agent turn | One fresh Claude Code session works the open tasks, writes a handoff note and commits | Stage 2, run the sweep |
| Review turn | One reviewer session tries to refute a gate package | The diagnosis at Gate 1 |
| Referee turn | One referee step, with no LLM | Replay 20 rollouts, tag the freeze, run the test |

Every session starts with no memory. A hook prints the referee's status and the tail of `PROGRESS.md` into it, so it picks up from the files. Long sweeps run as background jobs, and the loop waits for them without spending tokens.

### How the test seeds are drawn

*Diagram in the online spec.* Five steps run from the freeze to the verdict, and the seed draw comes only after the public commitment. The numbered list below says the same thing.

1. The referee checks the freeze package from a clean checkout and tags the commit.
2. It publishes the commit hash as a GitHub release. GitHub's clock gives the time.
3. It waits for the first round of the drand quicknet beacon that is emitted at least 10 minutes after that time, and checks the round's signature against the network's public key.
4. It hashes the beacon value together with the task name and the commit hash. The result picks where a block of 1,000 seeds starts, somewhere between 1,000,000,000 and 1,999,999,999.
5. It pushes that draw, then runs every arm once from the frozen code and writes the verdict.

**Why this is fair.** Nobody can know a beacon value before it comes out, so nobody knew the test positions when the edit was frozen. The commit hash is inside the draw, so choosing when to publish gives no control either. Anyone can re-check all of it later with `python -m referee.verify_draw`.

### What keeps the agent honest

| Worry | Guard |
| --- | --- |
| It edits the rules or the referee | Hooks and deny rules block it. The loop runs from a read-only copy. Any protected file that changes is restored before the next session, and all of them can be compared with the public `genesis-1` tag |
| It marks its own work | Every gate is recomputed from the raw per-seed files, and the referee replays 20 random rollouts itself |
| It keeps trying until it wins | One freeze per task and no redraw. The run is done when the protocol is done |
| It tunes on the test | The test seeds do not exist until the freeze is public |
| It loses the thread | Status and notebook at the start of every session, and a hook that blocks a session from ending without a handoff note and a commit |
| It runs forever | Four sessions that change nothing stop the run. So do 400 sessions or 21 days |

The headline is also measured twice. LeRobot's unmodified evaluator re-runs the two main arms on the same test seeds, and the verdict carries a warning if the two disagree on more than 2% of episodes.

### Rules I had to pin down

A person can use judgment at a gate and a script cannot, so six details are now fixed in `referee/RULES.md`.

- A significant gain that fails the guard is a null, and the post says why.
- H2 with fewer than 40 matched random edits. The allowed count shrinks so the luck rate stays at or under 5%. With fewer than 19, H2 counts as untested and the ending cannot be a win.
- H3. A baseline matches the edit when its gain is at least the edit's gain on the same seeds.
- Gate 0 for insertion uses a window as wide as transfer cube's, 15.6% to 25.6% around the published 20.6%.
- The old test range from 40000 up is retired and never run.
- A usage limit never skips a review. The loop waits and tries again. If the reviewer runs 3 times and still returns no usable verdict, the gate is recorded as not independently reviewed, and the verdict says so.

### When the run stops for you

- The agent writes `WAITING_FOR_ANDREW.md` because the plan cannot be followed or the referee looks wrong.
- The freeze is rejected 5 times, a test arm fails twice, or a referee step fails 3 times in a row.
- Four sessions in a row change nothing.
- Claude Code is signed out on the box.
- Twenty-four sessions in a row fail to run, and the cause is not a usage limit.
- The repo is moved off the run's branch.

You get a phone notification each time. You also get one at every gate, at the freeze, at the seed draw and at the verdict.

### Limits of this setup

- The referee checks rules, numbers and the order of events. It cannot tell whether the science is wise. The reviewer helps with that, and it is another Claude session.
- The agent and the referee run as the same Linux user. The guards are hooks, a read-only copy, automatic restore and public diffs. They are not operating-system isolation.
- GitHub's clock is the timestamp for the commitment, so a reader has to trust that clock.
- The kit passed 186 tests against a scripted agent with a fake harness. One pass of the real loop also ran against a signed-out Claude Code, which exercised the flags, the settings file and the session-start hook. No signed-in session has run under the loop yet.

## Measurements

Every rollout runs all 400 steps and logs two success scores, one failure class and a small set of behaviour numbers, all read straight from the physics.

### Two success scores

| Score | Rule | Default role |
| --- | --- | --- |
| Benchmark success | The left gripper touches the cube while it is off the table, at any step. This is exactly what LeRobot's eval reports | Primary |
| Strict handover | Somewhere in the episode, for 25 steps in a row (half a second), the left gripper holds the cube, the right gripper is off it, and it is off the table | Guard |

**The roles can swap, by a rule fixed now.** If strict handover is 10 or more points below benchmark success on the design seeds, strict becomes primary and benchmark becomes the guard. That choice is made at Gate 1, before any sweep.

### Failure classes

| Class | The episode got as far as | Published count, of 500 |
| --- | --- | --- |
| 0 | Nothing. The right gripper never touched the cube | 8 |
| 1 | A touch, but no lift | 41 |
| 2 | A lift, but the left gripper never reached the cube | 36 |
| 3 | A left-gripper touch, but only with the cube back on the table | 0 |
| 4a | A touch in the air, but no completed handover | Unknown, the benchmark stops the episode here |
| 4b | A completed handover | Unknown |

### Behaviour numbers

| Number | Definition | What it catches |
| --- | --- | --- |
| Reach error | Right gripper centre minus cube centre, at the step the right gripper starts to close, relative to the average over successful episodes | Reaching short, long or sideways |
| Meet error | The same for the left gripper, at the step it starts to close after the lift | A handover that lines up badly |
| Timings | First step of touch, lift, left touch and release | Being early or late |
| Lift and exchange height | Highest cube position, and cube height at the first left touch | A carry that is too low |
| Dual-hold steps | Steps with both grippers on the cube | A release that comes too soon |
| Cube knock | How far the cube slides before it is lifted | Bumping the cube away |
| Chunk jump | The largest action jump at steps 100, 200 and 300 | A jerk when a new chunk starts |
| Cube visibility | Cube pixels visible in the camera frame at each of the four decision steps | A decision made while an arm hides the cube |

### The one diagnostic that decides the strategy

Plot reach error against where the cube started. The shape of that plot says which kind of fix can work.

| Pattern | Meaning | Likely fix |
| --- | --- | --- |
| Error grows with distance from the centre of the square | The policy under-reacts to cube position | Turn up the heads that carry cube position. The simple rival is a gain on the action's deviation from average |
| The same error everywhere | A constant bias | A constant action offset will probably win, and we say so |
| Error only inside one zone | A fault tied to position | Heads that are active in that zone. Output-level fixes cannot see position |
| No pattern | Noise, or a skill the policy lacks | Probably not editable. Switch target or stop |

The published map already hints at the third pattern, because the weak zone sits on one side only.

## The stages

Eight stages run in order, and four of them end at a gate that the referee checks. No person signs anything along the way.

*Diagram in the online spec.* Stages 0 to 7 run in order with gates G0 to G3 between them. The agent works on design and validation seeds in Stages 1 to 3, and the referee runs the test seeds once in Stage 5.

The agent only ever touches design and validation seeds. The test seeds are drawn after gate G2, and the referee runs them once.

| Stage | Goal | Seeds | Rollouts, about |
| --- | --- | --- | --- |
| 0. Setup | Reproduce the published number and prove the harness | Reproduction | 2,500 |
| 1. Diagnosis | Find where and why the policy fails | Design | 6,000 |
| 2. Head atlas | Learn what each of the 48 heads does | Design | 56,000 |
| 3. Edit design | Build the edit and its simple rivals | Design and validation | Up to 104,000 |
| 4. Freeze | Lock everything in git | None | 0 |
| 5. Test | Run every arm once on unseen seeds | Test | 55,000 |
| 6. Replicate | Repeat on the insertion checkpoint, then stress the edit | Its own sets | 225,000 |
| 7. Publish | Post, code, checkpoint, logs | None | 0 |

### Stage 0. Setup and reproduction

1. Build the box, install, and save the versions, the GPU driver and the renderer name.
2. Convert the checkpoint with the migration script.
3. Run `lerobot-eval` on seeds 1000 to 1499, as the model card did.
4. Download the published `eval_info.json` and compare it seed by seed. Redraw the failure map from it.
5. Build the fast harness. It skips unused renders, owns the action queue, never stops an episode early, and logs every measurement.
6. Run the harness on the same 500 seeds and compare with step 3, seed by seed.
7. Run 100 seeds twice, and run a sham edit (every head ×1). Both must give identical results.
8. Run the 500 seeds under GPU rendering and under software rendering. Keep the one closer to the published per-seed results.
9. Time it, and fill in the cost table.

**Exit gate G0.** Success on the reproduction seeds lands between 78% and 88%. The harness agrees with `lerobot-eval` on at least 98% of seeds, and every mismatch has been looked at. Repeat runs are identical.

### Stage 1. Baseline and diagnosis

1. Run the 1,000 design seeds at the default setting, with all measurements.
2. Run them again with chunks of 50 and 25 steps, and with temporal ensembling at 0.01.
3. Draw the failure map by class. Add an even grid of 41 by 41 cube positions for a clean picture.
4. Make the reach-error plot, and the meet-error plot for class 2.
5. Find the cube in the first frame of each of the 50 training episodes, and lay those positions over the failure map. Failures may sit where the demos never went.
6. Look at frame strips of 30 failures and 10 successes.
7. Write `diagnosis.md`.

**Operating point rule.** The edit is designed at 100-step chunks unless a shorter chunk beats it by 3 or more points on the design seeds. Results at 100 are reported either way. Temporal ensembling is too slow to sweep at, so it stays a rival.

**Exit gate G1.** The referee recomputes the primary metric and the operating point from the design runs, and a reviewer in a fresh session tries to refute `diagnosis.md`. That file names the target failure, the behaviour number that tracks it, the primary metric and the operating point. If fewer than 5% of design episodes fail, the insertion checkpoint becomes the main experiment.

### Stage 2. Head atlas and sweep

1. **Open-loop atlas, with no rollouts.** Feed recorded frames through the policy with each head at ×0 and ×2, each head's attention sharpened and softened, and each latent direction at −2 and +2. Record how the predicted chunk changes.
2. From that, draw four things per head. An effect map over the table, the joints and time span it moves, how much of the cube-position signal it carries (the drop in variation of the first chunk across cube positions when the head is off), and where the cross-attention heads look in the image.
3. Score each head by how well its effect map overlaps the failure map, using the rank correlation between its effect size and the local failure rate across the grid.
4. **Closed-loop sweep.** Run all 96 single-head edits on design seeds 20000 to 20499, paired with the baseline.
5. **Split-half check.** Re-run the best 10 by success gain and the best 5 by overlap on seeds 20500 to 20999. Keep only effects that point the same way in both halves.

At step 0 the arms always start in the same pose, so the cube is the only thing that differs between frames. That makes the first-chunk effect map a clean picture of which heads read the cube.

### Stage 3. Edit design

1. Write each idea as one line in `hypotheses.md`, then turn it into a candidate edit.
2. Check dose response on the candidate heads at ×0.5, ×1.5, ×2, ×3 and ×4. The target behaviour number should move in step.
3. Combine up to 6 heads.
4. Tune the simple baselines on the same design seeds (see Controls).
5. Check at most 10 candidates on the validation seeds.
6. Pick one and write `edit.json`, with the reasoning.

**A candidate passes validation** if it removes at least 25% of failures with an exact McNemar p below 0.05, and the guard score drops by no more than 2 points.

**Stop rule.** If the 10 checks or the 100,000 rollouts run out with no pass, freeze the best candidate anyway and run the test once. A null result then has a clean number behind it.

### Stage 4. Freeze

The freeze package holds seven things, tagged `freeze-transfer_cube-1` on a clean git tree.

1. `edit.json` and the edited checkpoint, with its SHA-256.
2. `autogreedy.json`, built now by a script from the same sweep, so the agent never saw it.
3. `baselines.json`, with the best setting of each simple baseline.
4. `random_family.json`, with the rule for matched random edits and the random seed for the 40 draws.
5. `test_manifest.json`, listing every arm.
6. `analysis.py`, already run on validation data to prove it works.
7. The ledger so far.

**Exit gate G2.** The referee checks the package from a clean checkout, tags it, pushes the tag, and publishes the commit hash as a GitHub release before any test seed exists. That proves the order of events.

### Stage 5. Test

The referee draws 1,000 test seeds from the beacon and runs every arm once, from a clean checkout of the frozen tag. It then writes the verdict, and the agent writes the report around it.

| Arm | Count |
| --- | --- |
| Original policy, at 100-step chunks and at the chosen operating point | 1 or 2 |
| Agent's edit, same settings | 1 or 2 |
| Auto-greedy edit | 1 |
| Matched random edits | 40 |
| Simple baselines, best of each family | 4 |
| The edit's heads at four other scales | 4 |
| Original and edit with temporal ensembling | 2 |

The report contains the headline table, the before and after maps, a table of what was fixed and what broke by failure class, the random-edit histogram, the dose-response curve, and side-by-side videos on the same seeds.

After the test, nothing in the freeze changes. The autonomous run makes one freeze per task and never redraws. If a person later orders a second freeze, it gets a fresh draw and the post reports both.

### Stage 6. Replicate and stress

1. **Insertion checkpoint.** Same procedure, same budget, its own freeze and test. The measurements swap in the peg and the socket.
2. **Stress checks on the transfer-cube edit.** Cube positions in a 3 cm band outside the square, and cubes rotated by 15 and 30 degrees. These are reported, not judged.
3. **Optional.** Two more agent runs from Stage 1 in fresh sessions, each with its own freeze and its own seed draw, started by a person. This shows how much the result depends on one lucky run.

### Stage 7. Publish

- The post, built around the headline table and the maps.
- The public repo, with the code, `SPEC.md`, `DEVIATIONS.md`, the ledger, and per-seed outcomes for every arm.
- The edited checkpoint on the Hugging Face hub.
- The full agent logs.
- The head atlas as a page people can browse.

**Three places to post from, not one.** The field is moving fast, and each of these stands alone.

| After | Post | Why it stands alone |
| --- | --- | --- |
| Stage 1 | Where a public ACT checkpoint fails, and whether the benchmark overstates its success | It is all measurement, with no edit to defend |
| Stage 2 | What each of ACT's 48 heads does | Useful to anyone who trains ACT |
| Stage 5 | The edit, its controls and its result | The headline |

## Controls and simple baselines

Each control answers one objection a robotics reader will raise.

| Objection | Control | Detail |
| --- | --- | --- |
| "Random heads would do the same" | 40 matched random edits | Same number of heads, same blocks, same scales, different heads |
| "A constant offset would do the same" | Constant offset | One joint target shifted for the whole episode. 80 settings screened, the best kept |
| "It is just a gain" | Deviation gain | a′ₜ = āₜ + g (aₜ − āₜ), where āₜ is the average baseline action at step t. Six gains on the right arm, the left arm, or both |
| "Just re-plan more often" | Shorter chunks | 50 and 25 steps in place of 100 |
| "Just use temporal ensembling" | Ensembling arms | The original and the edited policy, both with coefficient 0.01 |
| "The simplest fix would do" | The agent's own simplest fix | A rule that only post-processes actions, with at most 3 free numbers, aimed at the same failure |
| "You do not need an LLM for this" | Auto-greedy script | Greedy search over the sweep table with no reasoning, up to 4 heads |
| "The score is only a touch" | Strict handover score | Reported for every arm |
| "You tuned on the test set" | Freeze, beacon-drawn seeds, ledger | The test seeds do not exist until the freeze is public |
| "It is noise" | 1,000 paired seeds | Repeat runs must match and the sham edit must be identical |
| "It was one lucky run" | Every freeze is reported | Extra agent runs in Stage 6 show the spread |
| "The agent marked its own work" | A scripted referee | The gates, the seed draw, the test and the verdict are plain code the agent cannot edit |

### Why the output-level fixes are the real rivals

An output-level fix works on the actions alone. It applies one fixed rule to every episode and cannot look at where the cube is. A head edit changes how the policy reads the scene, so it can act more in one zone than in another.

So if the weak zone needs a correction the rest of the table does not, the edit should win. If one global rule is enough, a simple fix will tie it. That would be a finding too, and the post would say it plainly.

### Keeping the comparison fair

- The simple baselines are tuned on the same design seeds as the edit.
- Each family gets one check on the validation seeds, and its best setting is frozen with the edit.
- The constant offset is screened on 200 design seeds, and its best five settings are re-run on all 1,000.
- The auto-greedy script starts from the six best single-head edits that survived the split-half check.
- For an edit that is not head scaling, the agent writes the matching rule for random edits before the freeze. If fewer than 40 distinct matched edits exist, all of them are used.

## Statistics

One test is confirmatory. Everything else is reported with intervals and no verdict.

### The confirmatory test

| Item | Choice |
| --- | --- |
| Comparison | Original policy against the agent's edit, on the primary score |
| Data | The 1,000 test seeds, paired seed by seed |
| Test | Exact McNemar, two-sided, at 0.05. It compares the seeds the edit fixed with the seeds it broke |
| Effect size | Net gain in points and share of failures removed, each with a 95% interval from 10,000 bootstrap resamples of the seeds |
| Verdict | A significant gain means the edit helps. Removing 25% or more of failures makes it a strong win |

### What the test can detect

I simulated the rule at the published 17% failure rate, with edits that fix some failures and also break 1 to 2% of successes.

| True share of failures removed | Clears the 25% bar on 1,000 test seeds | Clears it on 500 validation seeds |
| --- | --- | --- |
| 20% | 10% of the time | 18% |
| 25% | 51% | 51% |
| 30% | 90% | 82% |
| 35% | 99% | 96% |
| None | 0% | 0% |

**So aim above the bar.** An edit that barely clears 25% on validation has a coin-flip chance on the test. Freeze one that shows 35% or more if you can. Significance alone is easy here. Any true gain of 2.6 points or more was detected in every simulated run.

### The other comparisons

| Question | How it is reported |
| --- | --- |
| H2, are the chosen parts special | The edit's rank among the 40 random edits. It passes if at most 1 random edit gains as much. A random edit would pass by luck 4.9% of the time |
| H3, simple rivals | Each baseline's gain with its 95% interval beside the edit's, plus the paired difference between the edit and the best baseline |
| Dose response | The target behaviour number at each scale. It counts as a dial only if it moves the same way at every step |
| Regressions | The table of failure class before and after. Any class that grows is named in the post |
| Zones | Success before and after, inside the weak zone and outside it |

### Rules that keep the numbers honest

- The Stage 2 sweep is screening. Its 96 comparisons are never reported as findings.
- Selection uses design and validation seeds only, so the test estimate is unbiased.
- The test runs once per freeze, and every freeze is reported, failed ones included.
- If repeat runs ever differ, that flip rate is reported as the noise floor and the test is read against it.

**Who computes the verdict.** The referee does, from the per-seed files, by the rules written out in `referee/RULES.md`. The agent's own analysis describes the result and never decides it.

## Compute, time and cost

The whole project is about 450,000 rollouts if every budget is spent. That is a few hours on the GPU box, so the slow part will be reading and thinking.

### What I measured

These come from a small sandbox with 2 slow CPU cores, software rendering, and ACT with random weights.

| Item | Time |
| --- | --- |
| One stock simulator step, which renders three cameras | 209 ms |
| The physics alone for one step | 0.22 ms |
| One camera render in software | 74 ms |
| One policy forward pass on one CPU thread | 393 ms |
| A stock episode, simulator only | 84 s |
| A fast-harness episode, policy included, on one CPU core | 2.1 s |

### What that implies

| Box | Rollouts per second | All 450,000 rollouts |
| --- | --- | --- |
| 8 CPU cores, no GPU | 4 at sandbox speed, likely more on AWS | About 30 hours at that speed |
| g6.2xlarge with the L4 GPU | 15 to 30, my estimate, not measured | 4 to 8 hours, under $10 at $0.98 an hour |

**So the GPU quota is not a blocker.** Stages 0 and 1 can start on any Linux box with 8 cores. The GPU mainly helps the temporal ensembling arms, which need a render and a forward pass on every step. On a CPU-only box, run those arms on 200 seeds. The other slow item there is the stock eval in Stage 0, which renders three cameras on every step, so allow an hour or two for it.

Trade-off to know. A GPU renders slightly different pixels from software rendering, so per-seed outcomes can differ between boxes. Pick one box and one renderer in Stage 0 and stay there.

### Calendar

Driven by hand, the work fits the seven evenings below. The autonomous loop runs the same stages back to back, and my guess is two to four days. The box bills for every hour it is on, including the hours Claude spends thinking. Four days on a g6.2xlarge is about $94 at $0.98 an hour, and a CPU-only box costs less per hour.

| Evening | Work |
| --- | --- |
| 1 | Stage 0 and Gate 0 |
| 2 | Stage 1 and Gate 1 |
| 3 | Stage 2 |
| 4 | Stage 3 |
| 5 | Stages 4 and 5 |
| 6 | Stage 6, which mostly runs by itself |
| 7 | Draft the post |

Log the agent's wall time and tool calls for each stage. The Shanghai and Hong Kong team reported about 1 hour 50 minutes and 193 tool calls for their first run, so readers will want ours next to it.

## Risks and fallbacks

Fourteen things could go wrong, and each has a sign to watch for and a fallback decided now.

| Risk | How we would notice | Fallback |
| --- | --- | --- |
| The setup does not reproduce 83% | Gate 0 lands outside 78 to 88% | Check the checkpoint conversion, then the renderer, then pin MuJoCo to an older 3.x release. Stage 1 waits until it passes |
| GPU rendering is not repeatable | Two identical runs differ | Switch to software rendering, which was repeatable in my test |
| A shorter chunk already fixes most failures | Stage 1, step 2 | Design the edit at that setting. The headline becomes "on top of re-planning" |
| Failures show no pattern in our setup | The map looks scattered and reach error is flat | Switch the target to the handover class, then to the insertion checkpoint. If both are flat, stop with a null |
| No head has a clean effect | Every scaled head hurts, or gains vanish in the split-half check | Try the latent and direction families. If those fail, stop with a null and publish the atlas |
| The edit fixes one zone and breaks another | The before and after table shows a class or a zone growing | Prefer heads whose effect map stays inside the weak zone. Report the trade if it remains |
| The loose score improves but the handover does not | The guard fails | Reject the candidate |
| The strict score turns out to be badly defined | Good episodes end in a way the rule misses | Fix the rule once at Gate 1, after watching 10 successful episodes, and log it as a deviation |
| The agent overfits the validation seeds | Validation gain is far above test gain | The cap of 10 checks limits it and the test is the judge. Report both numbers |
| Someone posts the same experiment first | A new entry on the weight-space programming follow-up list | Our controls and the 1,000-seed test still stand. Move fast on Stages 0 and 1 |
| Versions drift after we publish | Others get different per-seed outcomes | Ship the lock file and the per-seed outcomes for every arm |
| The unattended run stalls or goes in circles | Four sessions in a row change neither the status nor any work product | The loop stops and sends Andrew a notification with the reason |
| A referee check is wrong or too strict | The agent cannot pass a gate it believes it has met, and says so in a note | Andrew fixes the referee and publishes a new genesis tag with the reason. The agent may never patch it |
| A usage limit pauses the agent | Sessions end at once with a limit message | The loop sleeps and retries every 30 minutes until the limit clears. The same holds for the reviewer, so a limit never skips a review |

### Limits the post will state up front

- One scene, one cube, and start positions from one 20 cm square.
- Simulation only. On a real arm, an edited policy needs joint-limit and speed checks before it runs.
- One trained checkpoint per task, so we cannot say how this varies across training runs.
- One agent model, and one run unless the optional extra runs happen.
- We show what the edited parts do. Why the policy computes it that way is a separate question.
- This spec gives the agent far more structure than the earlier posts did. So the result says more about weight editing than about how much an agent can do alone.
- No person checked the diagnosis or the freeze before the test. A scripted referee and a second Claude session did, and their records are published.

## Repo layout and what gets published

The repo stays small, with one package, one folder per experiment and one ledger.

```text
act-weight-editing/
  SPEC.md                 this document
  PLAN.json               the task graph the referee checks
  CLAUDE.md               the agent's standing rules
  PROGRESS.md             the lab notebook, append-only
  DEVIATIONS.md           every change to the plan, dated
  requirements.lock
  referee/                gates, seed draw, test and verdict. No LLM. Protected
    RULES.md              the exact rules, in mechanical form
    CONTRACT.md           the files and commands the referee expects
  run/                    the loop, setup, job runner and notifications. Protected
  .claude/                hooks that guard the rules and enforce clean handoffs
  starter/                the sandbox scripts behind this spec, as a head start
  awe/
    fastenv.py            simulator wrapper that skips unused renders
    rollout.py            one episode, all measurements, strict score
    run_arm.py            the command the referee calls to run one arm
    export_checkpoint.py  the command that writes an edited checkpoint
    pool.py               process pool, ledger, test-seed lock
    edits.py              head scaling and the other edit families
    atlas.py              open-loop head maps
    sweep.py              closed-loop sweep and split-half check
    baselines.py          the simple rivals
    autogreedy.py         the edit built with no LLM
    stats.py              descriptive tables. The verdict is the referee's
    report.py             tables, maps, videos
  experiments/
    transfer_cube/        diagnosis.md, hypotheses.md, freeze/, test/, report/
    insertion/
  reviews/                reviewer verdicts and the agent's answers
  ledger/                 one line per rollout
  logs/                   one line per session. Full transcripts stay on the box
  tests/                  the five checks below
```

### The edit itself is four lines

```python
def scale_head(policy, block, head, s, head_dim=64):
    W = policy.get_parameter(block + ".out_proj.weight")  # 512 x 512
    with torch.no_grad():
        W[:, head * head_dim:(head + 1) * head_dim] *= s
```

| Block name | Heads | Role |
| --- | --- | --- |
| `model.encoder.layers.0.self_attn` to `layers.3.self_attn` | 32 | Mix the camera, joint and latent tokens |
| `model.decoder.layers.0.multihead_attn` | 8 | Cross-attention, where each action step reads the scene |
| `model.decoder.layers.0.self_attn` | 8 | Constant output. Scaling one acts as a fixed internal offset, so these are a useful reference |

Head sharpening is the same idea on the query weights.

```python
def sharpen_head(policy, block, head, s, head_dim=64):
    W = policy.get_parameter(block + ".in_proj_weight")  # the first 512 rows are queries
    b = policy.get_parameter(block + ".in_proj_bias")
    with torch.no_grad():
        W[head * head_dim:(head + 1) * head_dim] *= s
        b[head * head_dim:(head + 1) * head_dim] *= s
```

### Simulator names the harness needs

I confirmed each of these in gym-aloha 0.1.5.

| Need | Name |
| --- | --- |
| Physics handle | `env.unwrapped._env.physics`, a dm\_control object |
| Cube | Body `box`, geom `red_box`, joint `red_box_joint` |
| Table | Geom `table` |
| Left gripper fingers | Geoms `vx300s_left/10_left_gripper_finger` and `vx300s_left/10_right_gripper_finger` |
| Right gripper fingers | Geoms `vx300s_right/10_left_gripper_finger` and `vx300s_right/10_right_gripper_finger` |
| Gripper centre | Midpoint of bodies `left_finger_link` and `right_finger_link` under `vx300s_left/` or `vx300s_right/` |
| Benchmark reward | `task.get_reward(physics)`. It checks one finger per gripper, the left arm's left finger and the right arm's right finger |
| Where rendering happens | `gym_aloha.tasks.sim.BimanualViperXTask.get_observation`, which draws the `top`, `angle` and `front_close` cameras |
| Cube start position | `gym_aloha.utils.sample_box_pose(seed)`, handed over through the global `gym_aloha.tasks.sim.BOX_POSE` |
| Gripper commands | Action index 6 for the left and 13 for the right, where 0 is closed and 1 is open |
| Time step | 0.002 s of physics, 10 per control step |
| Resting cube height | 0.02 m at the cube centre, after a 3 cm drop at reset |

### Five tests that must pass before Stage 1

| Test | Passes when |
| --- | --- |
| Fast harness against stock simulator | Joint positions match at every step for 20 seeds |
| Head scaling is exact | The edited matrix gives the same output as scaling the head directly, to 1e-6 |
| Sham edit | Scaling by 1 leaves the outputs bit for bit identical |
| Repeat run | The same seed gives the same actions twice |
| Test lock | A seed of 1,000,000,000 or more raises an error unless the referee supplies its draw file |

### What gets published

| Item | Where |
| --- | --- |
| The post | Wherever Andrew chooses to publish |
| Code, `SPEC.md`, `DEVIATIONS.md` | The GitHub repo, public from the first commit so the freeze has a witness |
| Edited checkpoint | Hugging Face hub. The original is Apache-2.0, so sharing with credit is fine |
| Ledger and per-seed outcomes for every arm | The repo |
| Agent logs | The repo |
| Head atlas | A page in the repo that people can browse |
| The commitment, the seed draw and the script that re-checks them | The repo and its GitHub releases |

## Verified facts and open assumptions

Twenty-one claims in this spec were checked directly. Nine were not, and each has a stage where it gets settled.

### Checked

Sandbox versions were LeRobot 0.7.0, gym-aloha 0.1.5, MuJoCo 3.8.1 and torch 2.15.0 on CPU. The model checks used ACT with random weights, because the sandbox cannot reach the hub.

| Claim | How I checked | Result |
| --- | --- | --- |
| Size of the model | Built LeRobot's ACT class with the hub config | 51,613,582 weights, of which 17.4M belong to the VAE encoder |
| 48 inference heads | Read the code and listed the weight names | 6 blocks of 8 heads |
| Head scaling is an exact weight edit | Compared the edited matrix with scaling the head's output directly | Largest difference 6e-8 |
| A sham edit changes nothing | Scaled a head by 1 | Bit for bit identical |
| Decoder self-attention is constant | Captured its output for two different inputs | The same vector at all 100 steps and for both inputs |
| Action steps are independent | Reversed the decoder position table | The chunk came out reversed, to 1e-6 |
| The VAE encoder is unused at inference | Set all its weights to zero | Output unchanged |
| A bias shift equals a constant offset | Added 0.2 to one action-head bias | That joint moved by 0.2 and no other moved |
| One seed, one cube position | Compared the sampler with the physics state | They match |
| The same seed gives the same episode | Ran a scripted episode twice with software rendering | Identical states and identical images |
| The success rule fires on this MuJoCo | Closed each gripper on the cube with gravity off | Reward 2 for the right gripper, 4 for the left |
| Skipping renders changes nothing | Fast wrapper against the stock simulator for 120 steps | Identical joint positions and images |
| The full rollout loop runs | Random-weight ACT through the fast wrapper | 4 decisions, 2.1 s, identical on repeat |
| The hub checkpoint is in the old format | Listed the repo files and read the migration script | No processor files. The script's own usage example names this checkpoint |
| Checkpoint settings | Read `config.json` on the hub | 100-step chunks, no ensembling, 4 encoder layers, 1 decoder layer, 8 heads |
| Published failures by class and seed | Read `eval_info.json` through a web fetch and spot-checked 17 seeds | The counts reproduce both the published 83.0% and the published average reward of 3.546 |
| Head sharpening is an exact weight edit | Scaled one head's query weights by 3 | That head's attention scores tripled and no other head changed |
| The cube is fully visible at the start | Rendered the camera frame at the nine extreme start positions | 256 to 317 cube pixels in every frame, and about 78 by 77 pixels for the whole square |
| A beacon value can be checked offline | Verified a real drand quicknet round against the network's public key | Round 42 verifies. The same signature fails for round 43, and a tampered one is rejected |
| The referee and the loop run the whole protocol | Played a simulated two-task campaign with a scripted agent and a fake harness, then broke one rule at a time | 186 tests pass, including a rejected freeze, a review that asks for changes, a usage limit that hits the agent and the reviewer, a crash loop and a stall |
| The loop starts a real Claude Code session | Ran one pass of the real loop with Claude Code 2.1.330 signed out, so no model call could happen | Every flag parsed, opus resolved to claude-opus-5-5, the session-start hook ran and printed the referee's status, and the loop stopped with a sign-in notice |

### Not checked yet

| Open point | Why it is open | Settled in |
| --- | --- | --- |
| The real checkpoint has never been loaded here | The sandbox cannot reach the hub | Stage 0 |
| Our setup reproduces 83% | MuJoCo and LeRobot have both moved on since the published run | Gate 0 |
| GPU speed and GPU repeatability | The sandbox has no GPU | Stage 0 |
| How often the handover completes under the strict rule | The benchmark stops the episode before it could tell | Stage 1 |
| Whether shorter chunks or ensembling already help | I found no numbers for this checkpoint | Stage 1 |
| The failure breakdown for insertion | The fetch returned the transfer-cube numbers for that page, so I do not trust it | Stage 6 |
| Whether someone has already edited ACT this way | The follow-up list on the weight-space programming post showed no ACT or LeRobot entry on November 22. I did not search beyond it | Before posting |
| The referee against the real harness | It has only met a scripted agent, so a check may prove too strict or a file format may not fit | Stage 0 |
| Signed-in Claude Code sessions under the loop | The sandbox has no Claude sign-in. Auto mode, the hooks that guard tool calls and session ends, and the wording of a usage-limit message were checked against the documentation only | The first pass of the loop |

### Sources

- [Abdul Rahim Mirani, weight editing on π₀.₅](https://www.abdulrahimmirani.com/research/vla-weight-editing)
- [Takehi, Teoh, Zhang and Khattab, Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
- [Gong and co-authors, Robotics Weight Editing](https://cuzyoung.github.io/robotics-weight-editing/en/)
- [LeRobot ACT transfer-cube model card](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human), its [config](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/blob/main/config.json) and its [per-episode results](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/blob/main/eval_info.json)
- [LeRobot ACT insertion model card](https://huggingface.co/lerobot/act_aloha_sim_insertion_human)
- [LeRobot source](https://github.com/huggingface/lerobot) and [gym-aloha source](https://github.com/huggingface/gym-aloha), both read on November 22, 2026
- [AWS G6 prices](https://www.doit.com/compute/spot/us-east-1/g6.xlarge)
- [drand developer documentation](https://docs.drand.love/developer/) and the [quicknet chain information](https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971/info)
- Claude Code documentation on [non-interactive runs](https://code.claude.com/docs/en/headless), [permission modes](https://code.claude.com/docs/en/permission-modes), [auto mode](https://code.claude.com/docs/en/auto-mode-config) and [hooks](https://code.claude.com/docs/en/hooks-guide), read on November 22, 2026
- [What is the Ultra plan?](https://support.claude.com/en/articles/17728952) and [Claude Fable models on your plan](https://support.claude.com/en/articles/15424964-claude-fable-models-on-your-plan)
- [Anthropic, formalizing Fermat's Last Theorem](https://www.anthropic.com/research/formalizing-fermats-last-theorem), for how a long unattended run is held together

## Next actions

The run starts with three commands on any Linux box with 8 cores. After that it needs nobody until it finishes or asks for you.

- [ ] Create a public GitHub repo named `act-weight-editing`, clone it onto the box, and unpack the kit into it. The kit already holds this spec as `SPEC.md`.
- [ ] Start a box. Use the g6.2xlarge if the G quota is there, and an 8-core CPU box if it is not.
- [ ] Sign in on the box. Run `claude` then `/login`, and run `gh auth login`.
- [ ] Install the ntfy app on your phone and subscribe to a topic name that is hard to guess.
- [ ] Run the three commands below. The first checks the box and publishes the pre-registration.

```bash
bash run/setup_box.sh                         # checks the box, installs, publishes genesis-1
nano ~/.awe/act-weight-editing/config.env     # set AWE_NTFY_TOPIC
tmux new -s awe ./run/loop.sh                 # start the run, then detach with Ctrl-b d
```

There is no brief to paste any more. The agent's standing rules live in `CLAUDE.md`, and `README.md` covers pausing, resuming, and what to do when the run asks for you.

### Five choices I made that you can overrule

| Choice | My pick | The alternative |
| --- | --- | --- |
| Primary score | Benchmark success, with the swap rule | Strict handover from the start |
| Bar for a strong win | 25% of failures removed | A lower bar such as 15% makes a win likelier and less striking |
| Agent model | Opus in every session. Effort is xhigh for Stage 1, Stage 3 and every review, and high everywhere else. Model and effort are recorded per session | Fable for the diagnosis and design stages, once you have set a spend limit |
| Insertion replicate | Inside this project, as Stage 6 | Post the transfer-cube result first and replicate after |
| Who referees | The scripted referee with beacon-drawn seeds, which you picked | You approve each freeze by hand |
