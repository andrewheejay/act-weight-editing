"""Renderer selection. configure() must run before anything imports MuJoCo.

The choice comes from experiments/<task>/stage0/renderer.json. Whatever MUJOCO_GL the caller set is overwritten.
"""
import json
import os

from .paths import exp

VENDOR = {
    "egl_nvidia": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    "egl_mesa": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
}
DEFAULT = "egl_nvidia"


def chosen(task: str) -> str:
    f = exp(task) / "stage0" / "renderer.json"
    return json.loads(f.read_text())["chosen"] if f.exists() else DEFAULT


def env_vars(name: str) -> dict:
    d = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": VENDOR[name]}
    if name == "egl_mesa":
        d["LIBGL_ALWAYS_SOFTWARE"] = "1"
        d["LP_NUM_THREADS"] = "1"
    return d


def configure(name: str) -> None:
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    os.environ.pop("EGL_DEVICE_ID", None)
    os.environ.update(env_vars(name))
