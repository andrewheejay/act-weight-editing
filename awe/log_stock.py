"""Ledger lines for stock lerobot-eval runs (not run by the harness). python -m awe.log_stock --seeds 1000 1007 --purpose ..."""
import argparse

from .ledger import Ledger

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="transfer_cube")
    ap.add_argument("--seeds", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    ap.add_argument("--arm", default="original")
    ap.add_argument("--purpose", required=True)
    a = ap.parse_args()
    led = Ledger(a.task, "setup", a.purpose)
    for s in range(a.seeds[0], a.seeds[1] + 1):
        led.log(a.arm, s)
    led.close()
