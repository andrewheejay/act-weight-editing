"""Lets anyone check that the test seeds were drawn fairly. For readers, after publication.

    pip install py_ecc
    python -m referee.verify_draw experiments/transfer_cube/test

It reads commitment.json and draw.json and re-derives everything from public information.

1. The beacon round is the first one emitted at least ten minutes after the commitment went public.
2. The beacon value carries a valid signature from the drand quicknet network.
3. The seed block follows from that value and the frozen commit by the stated formula.
4. With --online, the publication time matches what GitHub reports for the release.
5. Inside the repo, the freeze tag points at the committed hash.

A pass means nobody could have known the test seeds when the edit was frozen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import beacon
from .common import load_config, read_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="referee.verify_draw")
    ap.add_argument("test_dir")
    ap.add_argument("--online", action="store_true", help="also ask GitHub for the release time")
    a = ap.parse_args(argv)
    td = Path(a.test_dir)
    com, draw = read_json(td / "commitment.json"), read_json(td / "draw.json")
    if not com or not draw:
        print("commitment.json or draw.json is missing")
        return 2
    cfg = load_config()
    b, t = cfg["beacon"], cfg["test_seeds"]
    results: list[tuple[str, bool, str]] = []

    pub = com["witness"]["published_at"]
    rnd = beacon.target_round(pub, b)
    results.append(("round follows the rule", rnd == com["target_round"] == draw["beacon"]["round"], f"published {com['witness']['published_at_iso']}, first round at least {b['delay_seconds']} s later is {rnd}"))
    rt = beacon.time_of_round(rnd, b["genesis_time"], b["period"])
    results.append(("round is later than the commitment", rt >= pub + b["delay_seconds"], f"round time {datetime.fromtimestamp(rt, timezone.utc).isoformat()}"))

    sig = draw["beacon"]["signature"]
    results.append(("randomness is the hash of the signature", beacon.randomness_of(sig) == draw["beacon"]["randomness"], ""))
    if beacon.bls_available():
        results.append(("beacon signature is valid", beacon.verify_signature(rnd, sig, b["public_key"]), "checked against the quicknet public key"))
    else:
        results.append(("beacon signature is valid", False, "install py_ecc to check this"))

    blk = beacon.seed_block(draw["beacon"]["randomness"], draw["task"], draw["freeze_commit"], t)
    results.append(("seed block follows the formula", blk["start"] == draw["block"]["start"] and blk["count"] == draw["block"]["count"], f"seeds {blk['start']} to {blk['end']}"))
    results.append(("commitment names the same commit", com["body"]["freeze_commit"] == draw["freeze_commit"], draw["freeze_commit"][:12]))

    try:
        r = subprocess.run(["git", "rev-parse", f"{draw['freeze_tag']}^{{commit}}"], capture_output=True, text=True, cwd=str(td))
        if r.returncode == 0:
            results.append(("freeze tag points at that commit", r.stdout.strip() == draw["freeze_commit"], draw["freeze_tag"]))
    except FileNotFoundError:
        pass

    w = com["witness"]
    if a.online and w.get("kind") == "github-release":
        url = f"https://api.github.com/repos/{w['repo']}/releases/tags/{w['release_tag']}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "awe-verify/1.0", "Accept": "application/vnd.github+json"})
            rel = json.loads(urllib.request.urlopen(req, timeout=20).read().decode())
            ts = datetime.fromisoformat(rel["published_at"].replace("Z", "+00:00")).timestamp()
            results.append(("GitHub reports the same publication time", abs(ts - pub) < 1.0, rel["published_at"]))
            results.append(("release body holds the commitment", com["body"]["freeze_commit"] in (rel.get("body") or ""), ""))
        except Exception as e:
            results.append(("GitHub reports the same publication time", False, f"could not reach GitHub ({type(e).__name__})"))
    if w.get("kind") != "github-release":
        results.append(("commitment had a public witness", False, f"witness kind is {w.get('kind')}. This was a dry run"))

    ok = True
    for name, passed, detail in results:
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}")
    print("\nAll checks passed." if ok else "\nSome checks failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
