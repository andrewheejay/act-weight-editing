import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from awe import render  # noqa: E402

render.configure(render.chosen("transfer_cube"))  # before anything imports MuJoCo
