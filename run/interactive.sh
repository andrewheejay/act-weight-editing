#!/usr/bin/env bash
# Sit in on the run yourself, from the terminal or from your phone.
#
#   touch PAUSE                 let the loop finish its current step and wait
#   bash run/interactive.sh     an interactive agent session with the same guard rails as the loop
#   rm PAUSE                    hand the run back to the loop when you are done
#
# The session starts with Remote Control on, so it also shows up in the Claude app. It runs in auto
# mode with the loop's deny rules and hooks. Commit before you leave, because the loop expects a
# clean tree.
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${AWE_STATE_DIR:-$HOME/.awe/$(basename "$ROOT")}"
mkdir -p "$STATE_DIR"
# shellcheck disable=SC1091
[[ -f "$STATE_DIR/config.env" ]] && set -a && source "$STATE_DIR/config.env" && set +a
if [[ ! -f "$ROOT/PAUSE" ]] && pgrep -f "run/loop.sh" >/dev/null; then
  echo "The loop is running. Run 'touch PAUSE' first and wait for its current step to end." >&2
  exit 1
fi
sed "s|__ROOT__|/$ROOT|g; s|__STATE__|/$STATE_DIR|g" "$ROOT/run/claude-settings.template.json" > "$STATE_DIR/claude-settings.json"
[[ -x "$ROOT/.venv/bin/python" ]] && export PATH="$ROOT/.venv/bin:$PATH"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$ROOT"
AWE_AGENT=1 exec claude --remote-control "$(basename "$ROOT")" --permission-mode auto --settings "$STATE_DIR/claude-settings.json" "$@"
