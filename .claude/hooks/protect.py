#!/usr/bin/env python3
"""PreToolUse hook. Keeps the agent's hands off the referee, the rules and the publishing steps.

It only acts in sessions the run loop starts (AWE_AGENT=1), so a person working in this repo is not
affected. A blocked call returns a reason, and Claude sees that reason and can adjust.

This is the first of two layers. The second is that every protected file is compared with the
public genesis tag before each session and at the freeze, so a change that slips past here is
still caught and undone.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import sys
from pathlib import Path

HELP = " If a rule or the referee seems wrong, write WAITING_FOR_ANDREW.md and stop. See CLAUDE.md."
REWRITE = "Do not rewrite git history. Commits here are a public record. Add a new commit instead."
STAY = "Stay on the run's branch. To read an old version use `git show <commit>:<path>`. To bring a file back use `git checkout <commit> -- <path>`."


def is_path(arg: str, cwd: Path) -> bool:
    """True if the word names something in the working tree, so git reads it as a path and not as a commit."""
    p = Path(os.path.expanduser(arg))
    return (p if p.is_absolute() else cwd / p).exists()


def deny(reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}))
    sys.exit(0)


def load_protected(root: Path) -> dict:
    try:
        return json.loads((root / "referee" / "config.json").read_text())["protected"]
    except Exception:
        return {
            "files": ["SPEC.md", "PLAN.json", "CLAUDE.md", "README.md"],
            "dirs": ["referee/", "run/", ".claude/", "tests_referee/", "starter/", ".git/"],
            "globs": ["experiments/*/test/**", "experiments/*/freeze/ACCEPTED.json", "experiments/*/freeze/REJECTED-*.md", "reviews/*.json", "PAUSE", "logs/agent/**", "logs/sessions.tsv", "logs/loop.log", "logs/referee.log", "logs/notify.log"],
        }


def protected_reason(path_str: str, cwd: Path, root: Path, prot: dict) -> str | None:
    """Why this path may not be written, or None if it may."""
    if not path_str or path_str.startswith("-"):
        return None
    p = Path(os.path.expanduser(path_str))
    if not p.is_absolute():
        p = cwd / p
    try:
        p = Path(os.path.normpath(str(p)))
        home = Path.home()
        for outside in (home / ".awe", home / ".claude"):
            if p == outside or outside in p.parents:
                return f"{path_str} is outside the repo, in a folder the run depends on"
        rel = p.relative_to(root).as_posix()
    except ValueError:
        return None
    if rel in prot["files"]:
        return f"{rel} is a protected file"
    for d in prot["dirs"]:
        if rel == d.rstrip("/") or rel.startswith(d):
            return f"{rel} is inside the protected folder {d}"
    for g in prot["globs"]:
        if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel + "/x", g):
            return f"{rel} is written only by the referee"
    return None


def holds_protected(path_str: str, cwd: Path, root: Path, prot: dict) -> str | None:
    """Why removing or moving this folder is not allowed, because something protected lives inside it."""
    if not path_str or path_str.startswith("-"):
        return None
    p = Path(os.path.expanduser(path_str))
    if not p.is_absolute():
        p = cwd / p
    try:
        rel = Path(os.path.normpath(str(p))).relative_to(root).as_posix()
    except ValueError:
        return None
    if rel in ("", "."):
        return "that is the whole repo"
    segs = rel.split("/")
    for entry in [*prot["files"], *[d.rstrip("/") for d in prot["dirs"]], *prot["globs"]]:
        parts = entry.split("/")
        if len(parts) > len(segs) and all(fnmatch.fnmatch(a, b) for a, b in zip(segs, parts)):
            return f"{rel} holds protected files ({entry})"
    return None


SPLIT = re.compile(r"\|\||&&|[;\n|&]")
GIT_GLOBAL_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
MUTATORS_ALL = {"rm", "rmdir", "unlink", "shred", "truncate", "chmod", "chown", "chgrp", "touch", "patch", "mv"}
MUTATORS_LAST = {"cp", "rsync", "ln", "install", "scp"}


OPERATORS = {";", "&&", "||", "|", "&", "|&", ";;", "(", ")"}
REDIRECTS = {">", ">>", ">|", "&>", "&>>"}


def simple_commands(cmd: str) -> list[tuple[list[str], list[str]]]:
    """Split a shell command into simple commands. Returns (words, redirect targets) for each.

    Quotes are respected, so a semicolon inside a commit message does not start a new command.
    A line the tokenizer cannot read, such as part of a heredoc, falls back to a plain split.
    """
    out = []
    for line in cmd.splitlines():
        if not line.strip():
            continue
        try:
            lex = shlex.shlex(line, posix=True, punctuation_chars=True)
            lex.whitespace_split = True
            toks = list(lex)
        except ValueError:
            for part in SPLIT.split(line):
                words = part.split()
                targets = [m.group(1).strip("'\"") for m in re.finditer(r"(?<![<&\d])(?:\d?>>?|&>>?)\s*([^\s;&|<>]+)", part)]
                out.append((words, targets))
            continue
        words, targets = [], []
        i = 0
        while i < len(toks):
            t = toks[i]
            if t in OPERATORS:
                out.append((words, targets))
                words, targets = [], []
            elif t in REDIRECTS:
                if i + 1 < len(toks):
                    targets.append(toks[i + 1])
                    i += 1
                if words and words[-1].isdigit():
                    words.pop()  # the 2 in 2>file
            elif t in (">&", "<", "<<", "<<<", "<&"):
                i += 1  # skip the file descriptor or the input source
            else:
                words.append(t)
            i += 1
        out.append((words, targets))
    return out


def strip_prefix(tok: list[str]) -> list[str]:
    """Drop leading VAR=value words and wrappers such as sudo, env, nohup, time."""
    i = 0
    while i < len(tok):
        t = tok[i]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", t) or t in ("sudo", "env", "nohup", "time", "command", "exec", "nice", "stdbuf"):
            i += 1
        elif t.startswith("-") and i > 0 and tok[i - 1] in ("sudo", "nice", "stdbuf"):
            i += 1
        else:
            break
    return tok[i:]


def git_subcommand(tok: list[str]) -> tuple[str | None, list[str]]:
    i = 1
    while i < len(tok):
        t = tok[i]
        if t in GIT_GLOBAL_WITH_ARG:
            i += 2
        elif t.startswith("-"):
            i += 1
        else:
            return t, tok[i + 1 :]
    return None, []


def check_bash(cmd: str, cwd: Path, root: Path, prot: dict) -> None:
    for words, targets in simple_commands(cmd):
        for target in targets:
            if target in ("/dev/null", "/dev/stdout", "/dev/stderr"):
                continue
            why = protected_reason(target, cwd, root, prot)
            if why:
                deny(f"{why}, so output may not be redirected into it." + HELP)
        tok = strip_prefix(words)
        if not tok:
            continue
        prog = os.path.basename(tok[0])

        if prog == "git":
            sub, args = git_subcommand(tok)
            if sub == "push":
                deny("The loop pushes after every session. Do not push yourself.")
            if sub == "tag" and args and not any(a in ("-l", "--list", "--contains", "--points-at", "-n") for a in args):
                deny("Only the referee creates or removes tags.")
            if sub == "remote" and args and args[0] in ("add", "set-url", "remove", "rm", "rename"):
                deny("Do not change the git remotes.")
            if sub in ("rebase", "filter-branch", "filter-repo", "update-ref", "replace") or (sub == "commit" and "--amend" in args) or (sub == "reflog" and "expire" in args):
                deny(REWRITE)
            if sub == "reset":
                # moving the branch back drops commits that are already public. Unstaging and discarding are fine
                refs = [a for a in (args[: args.index("--")] if "--" in args else args) if not a.startswith("-")]
                if refs and refs[0] != "HEAD" and not is_path(refs[0], cwd):
                    deny(REWRITE + " To undo a commit use `git revert`.")
            if sub == "switch" or (sub == "checkout" and any(a in ("-b", "-B", "--detach", "--orphan") for a in args)):
                deny(STAY)
            if sub == "checkout" and "--" not in args and any(not is_path(a, cwd) for a in args if not a.startswith("-")):
                deny(STAY)
            if sub == "branch" and any(a in ("-m", "-M", "--move", "-d", "-D", "--delete", "-f", "--force") for a in args):
                deny("Do not rename, delete or move branches. The run lives on one branch.")
            if sub == "symbolic-ref" and len([a for a in args if not a.startswith("-")]) > 1:
                deny(STAY)
            if sub == "clean" and any(a.startswith("-") and not a.startswith("--") and ("x" in a or "X" in a) for a in args):
                deny("git clean -x would delete the checkpoints, the job logs and the run state.")
            if sub in ("checkout", "restore", "rm", "mv", "reset", "clean", "apply", "stash"):
                for a in args:
                    why = protected_reason(a, cwd, root, prot) or (holds_protected(a, cwd, root, prot) if sub in ("rm", "mv", "clean") else None)
                    if why:
                        deny(f"{why}. This git command would change it." + HELP)
            continue

        if prog == "gh":
            deny("The agent does not use gh. The referee handles GitHub.")
        if prog == "claude" and any(a in ("-p", "--print", "--bg", "--background", "remote-control", "agents", "--cloud", "--remote") for a in tok[1:]):
            deny("Do not start another Claude session from inside this one.")
        if prog in ("shutdown", "reboot", "halt", "poweroff"):
            deny("Do not power the box off.")

        in_place = (prog == "sed" and any(a == "--in-place" or (a.startswith("-") and not a.startswith("--") and "i" in a) for a in tok[1:])) or (prog == "perl" and any(a.startswith("-") and not a.startswith("--") and "i" in a for a in tok[1:]))
        if prog in MUTATORS_ALL or prog in MUTATORS_LAST or in_place or prog == "tee":
            args = [a for a in tok[1:] if not a.startswith("-")]
            for a in args[-1:] if prog in MUTATORS_LAST else args:
                why = protected_reason(a, cwd, root, prot)
                if not why and prog in ("rm", "rmdir", "mv", "shred", "unlink"):
                    why = holds_protected(a, cwd, root, prot)
                if why:
                    deny(f"{why}, so `{prog}` may not change it." + HELP)
        if prog == "dd":
            for a in tok[1:]:
                if a.startswith("of="):
                    why = protected_reason(a[3:], cwd, root, prot)
                    if why:
                        deny(f"{why}." + HELP)


def main() -> None:
    if os.environ.get("AWE_AGENT") != "1":
        sys.exit(0)
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get("AWE_REPO") or data.get("cwd") or ".").resolve()
    cwd = Path(data.get("cwd") or root)
    prot = load_protected(root)
    tool = data.get("tool_name", "")
    inp = data.get("tool_input") or {}

    if tool in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        why = protected_reason(str(inp.get("file_path") or inp.get("notebook_path") or ""), cwd, root, prot)
        if why:
            deny(f"{why}, and the agent may not change it." + HELP)
    elif tool == "Bash":
        check_bash(str(inp.get("command") or ""), cwd, root, prot)
    sys.exit(0)


if __name__ == "__main__":
    main()
