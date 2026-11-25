# Harness against lerobot-eval, seeds 1000 to 1499

Final comparison (stock_eval/eval_info.json against harness_repro.jsonl): 0 of 500 seeds disagree on success, and 0 of 500 disagree on max reward. No seed needs an explanation. Recompute with `python scratch/check_stock_agreement.py insertion`. That script's last line (30.4%) compares against the stored published file, which is the transfer-cube one (see published_note.md), so ignore it.

This was the first comparison for insertion. Both runs use the settings fixed on transfer cube (software rendering, TF32 off, inference batch 1), where the first try at batch 8 and TF32 on had 17 mismatches. Agreement at these settings is not evidence that the harness matches every stock configuration.
