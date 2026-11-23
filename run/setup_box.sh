#!/usr/bin/env bash
# One-time setup of the box, and the only way to change the rules later.
#
#   run/setup_box.sh                      first run. Checks the box, installs, and creates genesis-1
#   run/setup_box.sh --check              only run the checks
#   run/setup_box.sh --regenesis "why"    after a person edits a protected file. Creates genesis-N
#
# The genesis tag is the pre-registration. It is pushed and published as a GitHub release, so the
# plan, the rules and the referee carry a public timestamp that is earlier than any rollout.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STATE_DIR="${AWE_STATE_DIR:-$HOME/.awe/$(basename "$ROOT")}"
mkdir -p "$STATE_DIR"
TEST_LOG="$STATE_DIR/setup-tests.log"
MODE="${1:-setup}"
REASON="${2:-}"
ok() { echo "  ok    $*"; }
bad() { echo "  FAIL  $*"; FAILED=1; }
FAILED=0

echo "Checking tools"
for t in git gh jq curl python3 tmux claude; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t"; else bad "$t is not installed"; fi
done
if command -v claude >/dev/null 2>&1; then
  V="$(claude --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  if [[ -n "$V" && "$(printf '%s\n' "2.1.259" "$V" | sort -V | head -1)" == "2.1.259" ]]; then ok "Claude Code $V"; else bad "Claude Code $V is too old. Run claude update (need 2.1.259 or later)"; fi
  if claude auth status >/dev/null 2>&1; then ok "Claude Code is signed in"; else bad "Claude Code is not signed in. Run claude, then /login"; fi
fi

echo "Checking the repo"
BR=""
if git rev-parse --git-dir >/dev/null 2>&1; then
  BR="$(git symbolic-ref --short -q HEAD || true)"
  if [[ -n "$BR" ]]; then ok "git repo on branch $BR"; else bad "the repo is on a detached commit. Check out a branch first"; fi
else
  bad "not a git repo. Run git init and add a GitHub remote"
fi
if git remote get-url origin >/dev/null 2>&1; then ok "origin is $(git remote get-url origin)"; else bad "no remote named origin"; fi
if [[ -n "$(git config user.email || true)" && -n "$(git config user.name || true)" ]]; then ok "git identity is $(git config user.name)"; else bad "git identity is not set. Run git config --global user.name and user.email"; fi
if gh auth status >/dev/null 2>&1; then ok "gh is signed in"; else bad "gh is not signed in. Run gh auth login"; fi
VIS="$(gh repo view --json visibility -q .visibility 2>/dev/null || echo unknown)"
# shellcheck disable=SC1091
[[ -f "$STATE_DIR/config.env" ]] && set -a && source "$STATE_DIR/config.env" && set +a
if [[ "$VIS" == "PUBLIC" ]]; then
  ok "the repo is public, so freeze commitments can be published here"
elif [[ -n "${AWE_WITNESS_REPO:-}" ]]; then
  WV="$(gh repo view "$AWE_WITNESS_REPO" --json visibility -q .visibility 2>/dev/null || echo unknown)"
  if [[ "$WV" == "PUBLIC" ]]; then ok "the repo is $VIS and the witness repo $AWE_WITNESS_REPO is public"; else bad "the witness repo $AWE_WITNESS_REPO is $WV. It has to be public"; fi
else
  bad "the repo is $VIS. Make it public, or create a small public repo and set AWE_WITNESS_REPO in $STATE_DIR/config.env"
fi

echo "Checking Python"
# .venv/.awe-installed marks an install that finished, so a half-built .venv is rebuilt on the next run
if [[ ! -f .venv/.awe-installed && "$MODE" != "--check" ]]; then
  echo "  creating .venv (this takes a few minutes)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --allow-existing --python 3.12 .venv
    uv pip install --python .venv/bin/python "lerobot[aloha,evaluation]==0.7.0" scipy matplotlib py_ecc pytest
    uv pip freeze --python .venv/bin/python > requirements.lock
  else
    python3 -m venv .venv
    .venv/bin/pip install -q "lerobot[aloha,evaluation]==0.7.0" scipy matplotlib py_ecc pytest
    .venv/bin/pip freeze > requirements.lock
  fi
  touch .venv/.awe-installed
fi
if [[ -x .venv/bin/python ]]; then
  ok "$(.venv/bin/python --version)"
  if .venv/bin/python -m referee.selftest --online; then ok "referee selftest, including a live beacon round"; else bad "referee selftest failed"; fi
  if .venv/bin/python -m pytest tests_referee -q -p no:cacheprovider >"$TEST_LOG" 2>&1; then ok "referee test suite ($(tail -1 "$TEST_LOG"))"; else bad "referee test suite failed. See $TEST_LOG"; fi
else
  bad ".venv is missing"
fi

if [[ ! -f "$STATE_DIR/config.env" ]]; then
  cp run/config.env.example "$STATE_DIR/config.env"
  echo "  wrote $STATE_DIR/config.env. Edit it to set your notification topic."
fi

if [[ "$FAILED" != "0" ]]; then
  echo
  echo "Fix the items marked FAIL, then run this again."
  exit 1
fi
[[ "$MODE" == "--check" ]] && { echo; echo "All checks passed."; exit 0; }

published() { gh release view "$1" >/dev/null 2>&1; }
publish() {  # $1 tag, $2 extra note. Pushes the branch and the tag, then publishes the release. Safe to run again
  local tag="$1" notes
  git push -q origin "refs/heads/$BR:refs/heads/$BR" "refs/tags/$tag"
  published "$tag" && return 0
  notes="Pre-registration for the ACT weight editing experiment. This tag fixes SPEC.md, the task plan, the rules and the referee code before any rollout. Commit $(git rev-parse "$tag^{commit}").${2:+ $2}"
  gh release create "$tag" --verify-tag --title "Pre-registration ($tag)" --notes "$notes" >/dev/null
}

LAST="$(git tag -l 'genesis-*' | sort -t- -k2 -n | tail -1)"
if [[ -n "$LAST" ]] && ! published "$LAST"; then
  # an earlier run tagged it and then failed to push or to publish. Finish that job before anything else
  echo
  echo "$LAST exists on this box and was never published. Publishing it now."
  publish "$LAST" ""
  echo "$LAST is pushed and published."
  [[ "$MODE" == "--regenesis" ]] && echo "No new genesis was made. Run this again if the rules changed after $LAST."
  echo "Start the run with:  tmux new -s awe ./run/loop.sh"
  exit 0
fi
if [[ "$MODE" == "--regenesis" ]]; then
  [[ -n "$REASON" ]] || { echo 'Give a reason.  run/setup_box.sh --regenesis "why the rules changed"'; exit 1; }
  [[ -n "$LAST" ]] || { echo "There is no genesis tag yet. Run setup without arguments first."; exit 1; }
  N=$(( ${LAST#genesis-} + 1 ))
  printf '\n## %s, rules changed by a person (genesis-%s)\n\n%s\n' "$(date -u +%Y-%m-%d)" "$N" "$REASON" >> DEVIATIONS.md
elif [[ -n "$LAST" ]]; then
  echo
  echo "$LAST already exists and is published. Nothing to do. Start the run with:  tmux new -s awe ./run/loop.sh"
  exit 0
else
  N=1
fi

git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "genesis-$N: spec, plan, rules and referee"
fi
TAG="genesis-$N"
git tag "$TAG"
publish "$TAG" "${REASON:+Rules changed by a person. Reason. $REASON}"
echo
echo "$TAG is tagged, pushed and published."
echo "Optional hardening. In the GitHub repo settings, add a tag ruleset for genesis-* and freeze-* that blocks deleting and moving those tags."
echo "Start the run with:  tmux new -s awe ./run/loop.sh"
