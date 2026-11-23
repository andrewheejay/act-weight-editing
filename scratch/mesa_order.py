import sys
sys.path.insert(0, "/home/ubuntu/act-weight-editing")
from awe import render
render.configure("egl_mesa")
mode = sys.argv[1]
from awe.env import FastAloha
env = FastAloha("transfer_cube"); env.reset(seed=1000)
if mode == "render_first": print("pre", env.image().mean())
import torch
print("torch", torch.cuda.is_available())
if mode != "cpuonly": torch.zeros(3).cuda()
print("img", env.image().mean())
