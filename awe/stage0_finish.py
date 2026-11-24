"""After the job stock_b1 finishes: copy its eval_info.json as LeRobot wrote it, list every mismatch, write env.json.

python -m awe.stage0_finish [--rps R]   R is the harness rollouts per second, measured with nothing else running
"""
import argparse
import json
import platform
import shutil
import subprocess
from importlib.metadata import version

from .paths import ROOT, exp
from .render import env_vars

STOCK_FLAGS = [
    "lerobot-eval", "--policy.path={policy_path}", "--env.type=aloha", "--env.task={env_task}",
    "--eval.n_episodes={n_episodes}", "--eval.batch_size=1", "--policy.device=cuda", "--policy.use_amp=false", "--tf32=false",
    "--seed={seed}", "--output_dir={output_dir}",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, required=True)
    a = ap.parse_args()
    s0 = exp("transfer_cube") / "stage0"
    src = ROOT / "outputs/stock_b1/eval_info.json"
    (s0 / "stock_eval").mkdir(exist_ok=True)
    shutil.copyfile(src, s0 / "stock_eval/eval_info.json")
    info = json.loads(src.read_text())
    m = info["per_task"][0]["metrics"]
    mine = {json.loads(l)["seed"]: json.loads(l) for l in open(s0 / "harness_repro.jsonl")}
    bad = [s for i, s in enumerate(range(1000, 1500)) if bool(m["successes"][i]) != mine[s]["benchmark_success"]]
    print(f"stock {100 * sum(m['successes']) / 500:.1f}%  harness {100 * sum(r['benchmark_success'] for r in mine.values()) / 500:.1f}%  mismatches {len(bad)}: {bad}")
    for s in bad:
        i = s - 1000
        print(f"  seed {s}: stock max_reward {m['max_rewards'][i]}, harness max_reward {mine[s]['max_reward']}, class {mine[s]['failure_class']}, x {mine[s]['cube_x0']:.3f} y {mine[s]['cube_y0']:.3f}")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    env = {
        "lerobot": version("lerobot"), "gym_aloha": version("gym-aloha"), "mujoco": version("mujoco"), "torch": version("torch"),
        "dm_control": version("dm-control"), "gymnasium": version("gymnasium"), "python": platform.python_version(), "gpu": gpu,
        "os": platform.platform(), "renderer": "egl_mesa (llvmpipe, Mesa 23.2.1 LLVM 15.0.7)",
        "rollouts_per_second": a.rps, "stock_seconds_per_episode": info["overall"]["eval_ep_s"],
        "stock_eval_cmd": ["env", *[f"{k}={v}" for k, v in {**env_vars("egl_mesa"), "LP_NUM_THREADS": "8"}.items()], *STOCK_FLAGS],
    }
    (s0 / "env.json").write_text(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
