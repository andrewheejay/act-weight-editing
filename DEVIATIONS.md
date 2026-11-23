# Deviations from SPEC.md

Every departure from the plan goes here, with a date and a reason. Append only.

The agent may log a fallback that the Risks table in SPEC.md already allows, or an operational change that does not alter what is tested. Anything that changes a hypothesis, a pass rule, a seed set, a budget, an edit family or the referee needs a person, who records it with `run/setup_box.sh --regenesis "reason"`.

## Entries

None yet.

### 2026-11-23, Stage 0, renderer (Risks table fallback)

GPU rendering (NVIDIA EGL) is not repeatable on this box. Fifty renders of one state gave 4 distinct images, and two runs of the 500 reproduction seeds gave identical trajectories on 137 seeds, with 2 benchmark and 7 strict outcomes flipping. The Risks table names the fallback, software rendering, which is what we use (Mesa llvmpipe through EGL). GPU rendering is slightly closer to the published per-seed file (95.0% and 94.6% agreement in two runs, against 93.8%), so step 8 of Stage 0 alone would have kept it. Repeatability is a gate and closeness is not, so software rendering wins. Evidence in experiments/transfer_cube/stage0/renderer.json and renderer_compare/.

### 2026-11-23, Stage 0, stock evaluation command (operational)

The stock run uses --eval.batch_size=8 with the default asynchronous environments, where SPEC.md shows 50. Software rendering of three cameras takes about 0.25 s per step, and 50 worker processes looked likely to exceed this box (30 GB of RAM, 8 vCPUs). I did not try it. The seeds, the checkpoint, the policy settings and the success rule are unchanged. Batch size changes only the batching of inference, which can flip a borderline seed. The spec's batch-50 command was also run under GPU rendering as a side reference (job stock_gpu).
