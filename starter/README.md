# Starter scripts

These are the scripts that checked the facts in SPEC.md. They ran in a small sandbox with LeRobot 0.7.0, gym-aloha 0.1.5 and MuJoCo 3.8.1, on a CPU, with software rendering.

One limit matters. The sandbox could not reach the Hugging Face hub, so every model check here used ACT with random weights. Nothing in this folder has seen the real checkpoint. Treat the code as a head start for `awe/` and verify all of it in Stage 0.

| File | What it shows |
|---|---|
| `fastenv.py` | A wrapper that skips the renders ACT never looks at. It gave the same joint positions and the same decision-step images as the stock simulator |
| `proto_rollout.py` | One full episode with the policy owning its action queue, the strict handover score and the per-step measurements |
| `check_env.py` | Seeds map to cube positions, episodes repeat exactly, and timing |
| `check_act.py` | Parameter counts, exact head scaling, the constant decoder self-attention, and the unused VAE encoder |
| `check_fastenv.py` | The fast wrapper against the stock simulator |
| `check_reward.py`, `check_reward2.py` | The benchmark's reward stages fire on this MuJoCo version |
| `check_sharpen.py` | Head sharpening is an exact weight edit |

This folder is protected. Copy what you need into `awe/` and change it there.
