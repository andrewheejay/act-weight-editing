"""Is llvmpipe's image the same whatever LP_NUM_THREADS is, and how fast is the stock env step? Argument: thread count."""
import sys, os, hashlib, time
sys.path.insert(0, ".")
from awe import render
v = render.env_vars("egl_mesa"); v["LP_NUM_THREADS"] = sys.argv[1]; os.environ.update(v)
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
from awe.policy import Policy  # LLVM import order
import gymnasium as gym, gym_aloha, numpy as np  # noqa
from awe.paths import TASKS
env = gym.make(f"gym_aloha/{TASKS['transfer_cube']}", obs_type="pixels_agent_pos", render_mode="rgb_array").unwrapped
h = hashlib.sha256(); t = time.time()
for s in (1000, 1005, 1010):
    obs, _ = env.reset(seed=s)
    a = np.zeros(14, np.float32)
    for i in range(40):
        obs, *_ = env.step(a); h.update(obs["pixels"]["top"].tobytes())
    h.update(np.asarray(obs["agent_pos"]).tobytes())
print("threads", sys.argv[1], "hash", h.hexdigest()[:16], f"{(time.time() - t) / 120 * 1000:.0f} ms per step (3 renders)")
