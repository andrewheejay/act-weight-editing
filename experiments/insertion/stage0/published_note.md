# published_eval_info.json is not an insertion result

The hub's `eval_info.json` in lerobot/act_aloha_sim_insertion_human is byte for byte the transfer-cube file, 566d1661791f6e939efef28826a5cd7f674795d9c94575496d9ce252a11705ba, the same in every commit that has it. It says 83.0% for 500 episodes. The model card's insertion figure is 20.6%, and no per-seed insertion file exists to check it. Recompute with `sha256sum experiments/*/stage0/published_eval_info.json`. See DEVIATIONS.md (2026-11-25).
