# Publishing checklist for Andrew

What a person has to do by hand. The agent cannot push, tag, upload or call `gh` (CLAUDE.md rule 6), so none of this has been done. Written 2026-11-26, session 29. Tick the boxes in your own copy.

## 1. Check the record before anything goes out

- [ ] `python -m referee.status` says the protocol is finished (state DONE) once this checklist is committed.
- [ ] `python -m referee.verify_draw experiments/transfer_cube/test` passes.
- [ ] `python -m referee.verify_draw experiments/insertion/test` passes. The agent did not run either command. It treats anything about test seeds as the referee's.
- [ ] `python -m referee.recompute ledger` reports `illegal []`, `test_range_rollouts 0`, `bad_lines 0`. At the end of session 29 it gave 313,151 rollouts in all. Transfer cube: agent 95,073, baselines 46,800, auto-greedy 8,550, setup 6,030, stress 5,000. Insertion: agent 93,400, baselines 46,812, auto-greedy 8,550, setup 2,936.
- [ ] The GitHub releases `genesis-1`, `freeze-transfer_cube-1` (published 2026-11-24T13:36:24Z) and `freeze-insertion-1` (2026-11-25T16:32:29Z) exist and point at the commits in `experiments/<task>/freeze/ACCEPTED.json` (dd626ffe5163 and e27fe8eae63b).
- [ ] The repo is public, and the loop's last push contains this commit.

## 2. Edited checkpoints on the Hugging Face hub

The checkpoints are gitignored (`outputs/`). Rebuild them from the frozen manifests, or use the copies on the box, which match:

```
python -m awe.export_checkpoint --task transfer_cube --arm agent_edit --manifest experiments/transfer_cube/freeze/test_manifest.json --out outputs/edit_ckpt_transfer_cube
python -m awe.export_checkpoint --task insertion --arm agent_edit --manifest experiments/insertion/freeze/test_manifest.json --out outputs/edit_ckpt_insertion
sha256sum outputs/edit_ckpt_*/model.safetensors
```

- [ ] `model.safetensors` SHA-256 is `29e3749264d2ac227b8d65cb81729d6e1f19d3c1a7aceb053f660036afc3e6be` (transfer cube) and `5d00999bbcd500f66484be84150afe8f61adaf4f3ac776e61fc642cb838fa2da` (insertion). These are the values in `freeze/edit.json` and `freeze/ACCEPTED.json`. On 2026-11-26 the files in `outputs/` matched both.
- [ ] Upload each folder as its own hub repo, under names you choose.
- [ ] Replace the copied `README.md` in each folder with a model card before uploading. It should credit the original (`lerobot/act_aloha_sim_transfer_cube_human` and `lerobot/act_aloha_sim_insertion_human`, Apache-2.0) and keep the licence. It should list the edit (which heads, which scales, 65,536 or 98,304 weights changed) and link this repo and the verdict. Add the simulation-only warning: on a real arm the policy needs joint-limit and speed checks before it runs.
- [ ] Optional: add the original checkpoint SHA-256 to each card (`original_checkpoint_sha256` in `freeze/edit.json`).

## 3. Agent and loop logs

SPEC.md promises the full agent logs. The repo holds only `logs/sessions.tsv` (one line per session, 28 sessions before this one). The transcripts are gitignored and stay on the box.

- [ ] `logs/agent/` (about 23 MB, one `.jsonl.gz` transcript and one `.err` file per session). `logs/loop.log`, `logs/referee.log` and `logs/notify.log` are gitignored too.
- [ ] **Read before publishing.** The transcripts may hold things that should not be public: your email address (it is in the session context), local paths, account names, tokens printed by tools, or hub and GitHub credentials. Search them for `@`, `token`, `hf_`, `ghp_`, `key`, and redact what you find.
- [ ] Attach them where the post links them, for example as a release asset or a separate archive. If you redact anything, give the redaction rule.
- [ ] Reviewer sessions: the verdicts and the agent's answers are in `reviews/` already. Attach the reviewer transcripts if the box kept them.

## 4. The post

- [ ] Edit `publish/post.md`. It is a draft. Change `publish/post_body.md` and rebuild with `python publish/write_post.py`, so the two verdict blocks stay byte-identical to `experiments/*/test/VERDICT.md`. Do not reword the verdict blocks.
- [ ] Image links in the post are relative to `publish/`. If the post is hosted elsewhere, replace them with absolute GitHub URLs, or copy the images: `experiments/transfer_cube/report/analysis/before_after_map.png` and `experiments/insertion/report/fixed_broken_map.png`. More figures are in each `report/` folder.
- [ ] Keep the limits section near the top. SPEC.md says the post states the limits up front.
- [ ] Decide whether to split the material into the three posts SPEC.md suggests (Stage 1 measurement, Stage 2 atlas, Stage 5 result). The draft is one post, with a short section for each.
- [ ] Choose the venue.

## 5. The head atlas

- [ ] `publish/atlas/index.html` is one self-contained file (2.1 MB, images embedded). Host it, for example with GitHub Pages from `publish/`, or link the raw file through a viewer. Rebuild it with `python publish/build_atlas.py` from committed files only.

## 6. Things the protocol leaves to a person

- [ ] The optional extra agent runs (SPEC.md, Stage 6, item 3): two more runs from Stage 1 in fresh sessions, each with its own freeze and seed draw. Only a person may start them. They did not happen, and the post says "one run".
- [ ] If anything in the referee looks wrong to you, the fix is a new genesis tag with a reason (SPEC.md, Risks table). The agent has written no `WAITING_FOR_ANDREW.md`.

## 7. What the reader should be able to rerun

- `python experiments/<task>/freeze/analysis.py --mode test` rebuilds each report's tables and maps from `test/arms/*.jsonl`.
- `python experiments/<task>/report/extra_tables.py` rebuilds the descriptive tables (no rollouts).
- `python -m referee.recompute rates experiments/<task>/stage0/harness_repro.jsonl` gives the Gate 0 numbers (81.8% and 18.2%).
- `requirements.lock` pins the versions. Per-seed outcomes for every arm are in the repo, so others can diff against their own runs if versions drift.
