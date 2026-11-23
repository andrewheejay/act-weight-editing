"""Keeps the protected files identical to the genesis tag.

    python -m referee.guard --repo ROOT           list protected files that differ
    python -m referee.guard --repo ROOT --heal    restore them from the genesis tag and commit

The loop runs the heal step before every session, so a stray edit to the rules never survives.
A person who means to change a protected file runs run/setup_box.sh --regenesis "why" afterwards,
which records the change in public.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .common import REFEREE_IDENT, changed_since_genesis, git, load_config, protected_pathspecs, repo_root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="referee.guard")
    ap.add_argument("--repo")
    ap.add_argument("--heal", action="store_true")
    a = ap.parse_args(argv)
    root = repo_root(a.repo)
    cfg = load_config()
    tag, changed = changed_since_genesis(root, cfg)
    if tag is None:
        print("no genesis tag")
        return 2
    if not changed:
        return 0
    print("protected files differ from " + tag)
    for c in changed:
        print("  " + c)
    if not a.heal:
        return 1
    specs = protected_pathspecs(cfg)
    present = [s for s in specs if git(root, "cat-file", "-e", f"{tag}:{s.rstrip('/')}")[0] == 0]
    # 1. put back everything the genesis tag has
    git(root, "checkout", tag, "--", *present, check=True)
    # 2. drop tracked files the genesis tag never had
    _, added = git(root, "diff", "--name-only", "--diff-filter=A", tag, "--", *specs)
    for f in [x for x in added.splitlines() if x.strip()]:
        git(root, "rm", "-q", "-f", "--", f)
    # 3. drop untracked files inside protected folders (ignored files such as __pycache__ stay)
    for s in specs:
        if (root / s).is_dir():
            git(root, "clean", "-f", "-q", "--", s)
    # 4. record the repair
    _, staged = git(root, "diff", "--cached", "--name-only")
    files = [x for x in staged.splitlines() if x.strip()]
    if files:
        subprocess.run(["git", "-C", str(root), *REFEREE_IDENT, "commit", "-q", "-m", f"referee: restored protected files to {tag}", "--", *files], check=True, capture_output=True)
    _, still = changed_since_genesis(root, cfg)
    print("restored" if not still else "could not restore: " + ", ".join(still))
    return 0 if not still else 1


if __name__ == "__main__":
    sys.exit(main())
