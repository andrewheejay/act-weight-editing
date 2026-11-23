import sys
sys.path.insert(0, "/home/ubuntu/act-weight-editing")
from awe import render
render.configure(sys.argv[1])
import numpy as np, time
from awe.env import FastAloha
env = FastAloha("transfer_cube")
env.reset(seed=1000)
t = time.time(); ims = [env.image() for _ in range(50)]; dt = (time.time() - t) / 50
ref = ims[0].astype(int)
d = [np.abs(i.astype(int) - ref) for i in ims]
print(sys.argv[1], "ms/render", round(1e3 * dt, 2), "distinct images", len({i.tobytes() for i in ims}), "max px diff", max(x.max() for x in d), "max pixels differing", max((x.max(-1) > 0).sum() for x in d))
