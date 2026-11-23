"""Which layer is not repeatable? Hash image, chunk and qpos at each decision, across repeats and episode order."""
import hashlib, sys
sys.path.insert(0, "/home/ubuntu/act-weight-editing")
from awe import render
render.configure(sys.argv[1] if len(sys.argv) > 1 else "egl_nvidia")
import numpy as np
from awe.policy import Policy
from awe.env import FastAloha
from awe.ledger import Ledger
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:8]
pol = Policy("transfer_cube"); env = FastAloha("transfer_cube")
led = Ledger("transfer_cube", "setup", "determinism probe")
def ep(seed):
    led.log("original", seed)
    ap = env.reset(seed=seed); out = []
    for t in range(400):
        if t % 100 == 0:
            img = env.image(); img2 = env.image(); ch = pol.chunk(img, ap); ch2 = pol.chunk(img, ap)
            out.append(dict(t=t, img=h(img), img_again=h(img2), ch=h(ch), ch_again=h(ch2), state=h(env.qpos())))
        ap, r = env.step(ch[t % 100])
    return out, h(env.qpos())
for s in [1000, 1000, 1001, 1000, 1001]:
    o, q = ep(s)
    print(s, "final", q)
    for d in o: print("   ", d)
