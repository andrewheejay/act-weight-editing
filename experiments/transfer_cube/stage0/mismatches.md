# Harness against lerobot-eval, seeds 1000 to 1499

Final comparison (stock_eval/eval_info.json against harness_repro.jsonl): 0 of 500 seeds disagree on success, and 0 of 500 disagree on max reward. No seed needs an explanation. Recompute with `python scratch/check_stock_agreement.py`.

Both runs use software rendering, full float32 (TF32 off) and an inference batch of 1. These settings were chosen after the first comparison, which had 17 mismatches (96.6% agreement) with the stock run at batch 8 and TF32 on. So 100% is the agreement of the settings used here, not evidence that the harness matches every stock configuration. The change and the order of events are in DEVIATIONS.md (2026-11-24). The first comparison's 17 seeds and their files are in numerics/ (tf32on) and in PROGRESS.md.
