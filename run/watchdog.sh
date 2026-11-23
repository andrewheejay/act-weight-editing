#!/usr/bin/env bash
# A dead-man's switch.  Run it from cron every 30 minutes.
#
#   */30 * * * * /home/ubuntu/act-weight-editing/run/watchdog.sh
#
# It notifies you once if the loop has stopped, or has gone quiet for too long, while the run is
# unfinished.  A watchdog on the box cannot see the box itself die.  For that, set
# AWE_HEALTHCHECK_URL to a ping URL from a service such as healthchecks.io.  The loop pings it on
# every pass, and that service alerts you when the pings stop.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="${AWE_STATE_DIR:-$HOME/.awe/$(basename "$root")}"
# shellcheck disable=SC1091
[[ -f "$state_dir/config.env" ]] && set -a && source "$state_dir/config.env" && set +a
export AWE_REPO="$root"
hours="${AWE_WATCHDOG_HOURS:-6}"
py="$root/.venv/bin/python"
[[ -x "$py" ]] || py="$(command -v python3)"
state="$(cd "$root" && "$py" -m referee.status --state-only 2>/dev/null)"
case "$state" in DONE | WAITING_FOR_ANDREW | SETUP_INCOMPLETE | "") exit 0 ;; esac
[[ -f "$root/PAUSE" ]] && exit 0

mkdir -p "$root/.awe"
flag="$root/.awe/watchdog-state"
last="$(cat "$flag" 2>/dev/null || echo ok)"
alert() {  # notify once per problem, not every 30 minutes
  [[ "$last" == "alerted" ]] || bash "$root/run/notify.sh" "$1" "$2"
  echo alerted > "$flag"
  exit 0
}
if ! pgrep -f "run/loop.sh" >/dev/null; then
  alert "The loop is not running" "State is $state. Start it again with tmux new -s awe ./run/loop.sh"
fi
beat="$root/.awe/heartbeat"
if [[ -f "$beat" ]] && (( $(date +%s) - $(stat -c %Y "$beat") > hours * 3600 )); then
  alert "The run has gone quiet" "No heartbeat for more than $hours hours. State is $state."
fi
echo ok > "$flag"
