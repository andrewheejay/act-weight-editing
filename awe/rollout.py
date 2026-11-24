"""One episode. Runs all 400 steps, never stops early, and returns the per-step arrays and the per-seed record."""
import numpy as np

from . import geom, measure

EPISODE_STEPS = 400


def red_pixels(img) -> int:
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return int(((r - np.maximum(g, b)) > 60).sum())


def run_episode(env, policy, seed=None, pose=None, n_frames=0):
    """Returns the arrays of one episode. n_frames > 0 also keeps that many evenly spaced camera frames."""
    spec = policy.spec
    n = spec["n_action_steps"]
    ens = spec.get("temporal_ensemble_coeff") is not None
    policy.reset()
    agent_pos = env.reset(seed=seed, pose=pose)
    tc = env.task == "transfer_cube"
    rec = {k: [] for k in ("action", "reward", "qpos", "left", "right", "table", "cube", "tcpl", "tcpr", "leftboth")}
    rec["qpos"].append(env.qpos())
    cube0 = env.cube_pos() if tc else None
    frames = []
    queue, vis, n_dec = [], {}, 0
    for t in range(EPISODE_STEPS):
        need = ens or not queue
        img = env.image() if need else None
        if n_frames and (t % max(1, EPISODE_STEPS // n_frames) == 0):
            frames.append((img if img is not None else env.image()).copy())
        if need:
            if ens:
                a = policy.ensemble_action(img, agent_pos)
            else:
                queue = list(policy.chunk(img, agent_pos)[:n])
                a = queue.pop(0)
            n_dec += 1
            if t % 100 == 0 and tc:
                vis[t] = red_pixels(img)
        else:
            a = queue.pop(0)
        agent_pos, r = env.step(a)
        rec["action"].append(np.array(a, dtype=np.float32))
        rec["reward"].append(int(r))
        rec["qpos"].append(env.qpos())
        rec["tcpl"].append(env.tcp("left"))
        rec["tcpr"].append(env.tcp("right"))
        if tc:
            L, R, TB, LB = env.flags
            rec["left"].append(L)
            rec["right"].append(R)
            rec["table"].append(TB)
            rec["leftboth"].append(LB)
            rec["cube"].append(env.cube_pos())
    arr = {k: np.array(v) for k, v in rec.items() if len(v)}
    arr["vis"] = [vis.get(t, -1) for t in (0, 100, 200, 300)]
    arr["n_decisions"] = n_dec
    if tc:
        arr["cube0"] = cube0
        arr["fk"] = env.arm_fk
    if frames:
        arr["frames"] = np.stack(frames)
    return arr


def summarize(arr, arm_id, seed=None, pose=None) -> dict:
    """The per-seed record. Numbers are in metres, steps or unitless."""
    out = {"seed": None if seed is None else int(seed), "arm": arm_id}
    if pose is not None:
        out["pose"] = [float(x) for x in pose]
    rew = arr["reward"]
    out["max_reward"] = int(rew.max())
    out["benchmark_success"] = bool((rew == 4).any())
    tc = "left" in arr
    strict = measure.strict_success(arr["left"], arr["right"], arr["table"]) if tc else False
    out["strict_success"] = bool(strict)
    out["failure_class"] = measure.failure_class(out["max_reward"], strict)
    out["traj_hash"] = measure.traj_hash(arr["qpos"])
    out["n_decisions"] = int(arr["n_decisions"])
    if tc:
        out.update(measure.transfer_cube(arr))
        out.update(geom.transfer_cube(arr, out, arr["fk"]))
    return out
