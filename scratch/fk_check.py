"""Sanity of the FK used for the commanded gripper position: does FK(measured qpos) give the recorded tcp, and how far is FK(command) from it."""
import numpy as np
from awe import render
render.configure(render.chosen("transfer_cube"))
from awe.env import FastAloha
from awe.paths import cache_dir
env = FastAloha("transfer_cube")
m = env.physics.model
for i in range(m.njnt):
    print(i, m.id2name(i, "joint"), m.jnt_qposadr[i], m.jnt_type[i])
ph = env.physics
a, b = env.tcp_ids["right"]
def tcp(q):
    with ph.reset_context():
        ph.data.qpos[:] = q
    return 0.5 * (np.array(ph.data.xpos[a]) + np.array(ph.data.xpos[b]))
z = np.load(cache_dir("transfer_cube") / "traces" / "default" / "20000.npz")
q, act, tr = z["qpos"].astype(float), z["action"], z["tcpr"]
# recorded tcpr[t] is after step t, qpos[t+1]
e = [np.linalg.norm(tcp(q[t + 1]) - tr[t]) for t in range(0, 400, 40)]
print("FK(measured) vs recorded tcp, max", max(e))
def cmd(t):
    qq = q[t + 1].copy()
    qq[8:14] = act[t][7:13]
    return tcp(qq)
for t in range(0, 400, 40):
    print(t, np.round(cmd(t) - tr[t], 4), np.round(np.linalg.norm(cmd(t) - tr[t]), 4))
