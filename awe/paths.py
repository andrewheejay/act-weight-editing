"""Where things live. Imports nothing heavy."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = {"transfer_cube": "AlohaTransferCube-v0", "insertion": "AlohaInsertion-v0"}


def exp(task: str) -> Path:
    return ROOT / "experiments" / task


def checkpoint_dir(task: str) -> Path:
    base = os.environ.get("AWE_CHECKPOINTS")
    return (Path(base) if base else ROOT / "checkpoints") / task


def cache_dir(task: str) -> Path:
    base = os.environ.get("AWE_CACHE")
    return (Path(base) / task) if base else exp(task) / "cache"
