#!/usr/bin/env bash
# The engine of the autonomous run.
#
#   tmux new -s awe ./run/loop.sh        start, or restart after a stop
#   touch PAUSE                          pause before the next step.  rm PAUSE to carry on
#   python -m referee.status             see where things stand
#
# Each pass asks the referee for the state of the run and then does exactly one thing.
#   AGENT_TURN   starts one fresh Claude Code session that works on the open tasks
#   REVIEW_TURN  starts one read-only reviewer session
#   JUDGE_TURN   runs a referee step with no LLM (replay checks, the freeze, the seed draw, the test)
# It stops at DONE or WAITING_FOR_ANDREW and sends a notification.
#
# The loop and the referee run from a read-only copy taken at the genesis tag, so nothing a session
# does inside the repo can change them while they are running.

set -u

if [[ -z "${AWE_PRISTINE:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  GEN="$(git -C "$ROOT" tag -l 'genesis-*' | sort -t- -k2 -n | tail -1)"
  if [[ -z "$GEN" ]]; then
    echo "No genesis tag yet. Run run/setup_box.sh first." >&2
    exit 2
  fi
  STATE_DIR="${AWE_STATE_DIR:-$HOME/.awe/$(basename "$ROOT")}"
  PRISTINE="$STATE_DIR/$GEN-$(git -C "$ROOT" rev-parse --short "$GEN")"
  if [[ ! -d "$PRISTINE/referee" ]]; then
    mkdir -p "$PRISTINE"
    git -C "$ROOT" archive "$GEN" run referee | tar -x -C "$PRISTINE"
    chmod -R a-w "$PRISTINE"
  fi
  export AWE_PRISTINE="$PRISTINE" AWE_REPO="$ROOT" AWE_STATE_DIR="$STATE_DIR" AWE_GENESIS="$GEN"
  exec bash "$PRISTINE/run/loop.sh" "$@"
fi

ROOT="$AWE_REPO"
STATE_DIR="$AWE_STATE_DIR"
# shellcheck disable=SC1091
[[ -f "$STATE_DIR/config.env" ]] && set -a && source "$STATE_DIR/config.env" && set +a

PY="${AWE_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
export AWE_PYTHON="$PY"
export PATH="$(dirname "$PY"):$PATH"
CLAUDE="${AWE_CLAUDE_BIN:-claude}"
export AWE_NOTIFY_CMD="$AWE_PRISTINE/run/notify.sh"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

MODEL="${AWE_MODEL-opus}"
EFFORT="${AWE_EFFORT-high}"                              # every session outside stages 1 and 3
EFFORT_DESIGN="${AWE_EFFORT_DESIGN-xhigh}"               # stage 1 (diagnosis) and stage 3 (edit design)
export AWE_REVIEW_EFFORT="${AWE_REVIEW_EFFORT-xhigh}"    # every reviewer session
export AWE_REVIEW_MODEL="${AWE_REVIEW_MODEL:-$MODEL}"
# Claude Code ignores an effort level it does not know and quietly runs at the model's default,
# so a typo here would silently undo the setting. Refuse it before anything starts.
for setting in "AWE_EFFORT=$EFFORT" "AWE_EFFORT_DESIGN=$EFFORT_DESIGN" "AWE_REVIEW_EFFORT=$AWE_REVIEW_EFFORT"; do
  case "${setting#*=}" in
    ""|low|medium|high|xhigh|max) ;;
    *)
      echo "${setting%%=*} is '${setting#*=}'. Use low, medium, high, xhigh or max, or leave it empty for the model's default. Claude Code would ignore an unknown level and quietly run at the default." >&2
      exit 2
      ;;
  esac
done
# An effort level in the environment can outrank the --effort flag, so the loop clears it.
EFFORT_NOTES=""
if [[ -n "${CLAUDE_CODE_EFFORT_LEVEL:-}" ]]; then
  EFFORT_NOTES="CLAUDE_CODE_EFFORT_LEVEL=$CLAUDE_CODE_EFFORT_LEVEL was set on this box. The loop cleared it, so each session runs at the level in config.env."
  unset CLAUDE_CODE_EFFORT_LEVEL
fi
for f in "$HOME/.claude/settings.json" "$ROOT/.claude/settings.local.json"; do
  if [[ -f "$f" ]] && grep -q '"maxEffortLevel"' "$f"; then
    EFFORT_NOTES="${EFFORT_NOTES:+$EFFORT_NOTES }$f sets maxEffortLevel, which can cap the levels in config.env."
  fi
done
MAX_TURNS="${AWE_MAX_TURNS-400}"
SESSION_HOURS="${AWE_SESSION_HOURS:-6}"
STALL_LIMIT="${AWE_STALL_SESSIONS:-4}"
MAX_SESSIONS="${AWE_MAX_SESSIONS:-400}"
MAX_DAYS="${AWE_MAX_DAYS:-21}"
LIMIT_SLEEP="${AWE_LIMIT_SLEEP:-1800}"
ERROR_SLEEP="${AWE_ERROR_SLEEP:-300}"
JOB_POLL="${AWE_JOB_POLL:-60}"
JOB_MAX_HOURS="${AWE_JOB_MAX_HOURS:-24}"
FAIL_LIMIT="${AWE_FAIL_SESSIONS:-24}"

# The run lives on the branch that is checked out when the loop starts.
RUN_BRANCH="$(git -C "$ROOT" symbolic-ref --short -q HEAD || true)"
if [[ -z "$RUN_BRANCH" ]]; then
  echo "The repo is not on a branch. Check out the run's branch and start again." >&2
  exit 2
fi
mkdir -p "$ROOT/logs/agent" "$ROOT/.awe" "$STATE_DIR"
if [[ -s "$STATE_DIR/branch" && "$(cat "$STATE_DIR/branch")" != "$RUN_BRANCH" ]]; then
  echo "This run lives on the branch $(cat "$STATE_DIR/branch"), and the repo is on $RUN_BRANCH. Check out the run's branch and start again. If you mean to move the run, delete $STATE_DIR/branch first." >&2
  exit 2
fi
echo "$RUN_BRANCH" > "$STATE_DIR/branch"

LOG="$ROOT/logs/loop.log"
START_EPOCH="$(date +%s)"

say() { echo "[loop $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
notify() { bash "$AWE_NOTIFY_CMD" "$1" "${2:-}" >/dev/null 2>&1 || true; }
ref() { ( cd "$AWE_PRISTINE" && "$PY" -m "referee.$1" --repo "$ROOT" "${@:2}" ); }
beat() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/.awe/heartbeat"
  [[ -n "${AWE_HEALTHCHECK_URL:-}" ]] && curl -fsS -m 10 "$AWE_HEALTHCHECK_URL" >/dev/null 2>&1
  return 0
}
PUSH_FAILS=0
push() {  # publishes the run's branch. A failed push never stops the run, and the freeze step insists on one
  [[ "${AWE_NO_PUSH:-}" == "1" ]] && return 0
  if git -C "$ROOT" push -q origin "refs/heads/$RUN_BRANCH:refs/heads/$RUN_BRANCH" >>"$LOG" 2>&1; then
    PUSH_FAILS=0
    return 0
  fi
  PUSH_FAILS=$(( PUSH_FAILS + 1 ))
  say "git push failed ($PUSH_FAILS in a row). Will try again on the next pass."
  if (( PUSH_FAILS == 3 || PUSH_FAILS % 25 == 0 )); then
    notify "Pushes are failing" "$PUSH_FAILS in a row. The run carries on and nothing new is public. $(tail -n 3 "$LOG" | tr '\n' ' ' | cut -c1-300)"
  fi
  return 1
}
work_hash() { git -C "$ROOT" rev-parse "HEAD:experiments" "HEAD:ledger" "HEAD:awe" "HEAD:tests" "HEAD:publish" "HEAD:reviews" 2>/dev/null | sha256sum | cut -c1-16; }
stop_run() {  # $1 exit code, $2 title, $3 message
  push || true
  notify "$2" "$3"
  say "$2. $3"
  if [[ "${AWE_SHUTDOWN_ON_EXIT:-}" == "1" ]]; then
    say "Shutting the box down in 5 minutes. Run 'sudo shutdown -c' to cancel."
    sudo -n shutdown -h +5 >>"$LOG" 2>&1 || true
  fi
  exit "$1"
}
wait_for_andrew() {  # $1 reason. The loop writes the file so the next start shows why it stopped.
  if [[ ! -f "$ROOT/WAITING_FOR_ANDREW.md" ]]; then
    printf '# Waiting for Andrew\n\nWritten by the loop on %s.\n\n%s\n\nFix the cause, delete this file, and start run/loop.sh again.\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$1" > "$ROOT/WAITING_FOR_ANDREW.md"
  fi
  stop_run 3 "Run is waiting for you" "$1"
}

settings_file() {  # deny rules with absolute paths, so they hold wherever the session's shell wanders
  local out="$STATE_DIR/claude-settings.json"
  sed "s|__ROOT__|/$ROOT|g; s|__STATE__|/$STATE_DIR|g" "$AWE_PRISTINE/run/claude-settings.template.json" > "$out"
  echo "$out"
}

wait_for_jobs() {
  local waited=0 running
  while :; do
    running="$( (cd "$ROOT" && "$PY" "$AWE_PRISTINE/run/jobs.py" status --json) 2>/dev/null | jq -r '[.[] | select(.state=="running") | .name] | join(", ")' 2>/dev/null)"
    [[ -z "$running" ]] && return 0
    (( waited == 0 )) && say "Waiting for jobs: $running"
    beat
    sleep "$JOB_POLL"
    waited=$(( waited + JOB_POLL ))
    if (( waited > JOB_MAX_HOURS * 3600 )); then
      say "Jobs still running after $JOB_MAX_HOURS hours: $running. Starting a session to look at them."
      return 0
    fi
    [[ -f "$ROOT/PAUSE" ]] && return 0
  done
}

SESSIONS=0      # sessions that really ran. The session cap counts these
ATTEMPTS=0      # every start, including the ones a usage limit turned away
FAILS=0         # failed starts in a row, of any kind
HARD_FAILS=0    # failed starts in a row that were not a usage limit
JUDGE_FAILS=0
REVIEW_FAILS=0
STALL=0

ran_ok() {
  if (( FAILS > 0 )); then
    say "Running again after $FAILS failed starts."
    notify "Run resumed" "Sessions are running again after $FAILS failed starts."
  fi
  FAILS=0
  HARD_FAILS=0
}

after_failure() {  # $1 Session or Reviewer, $2 the text the failed start left behind. Sleeps as fits the cause
  local what="$1" text="$2"
  FAILS=$(( FAILS + 1 ))
  if grep -qiE 'run /login|not logged in|log in again|authenticat|unauthorized|oauth|credit balance|invalid api key' <<<"$text"; then
    wait_for_andrew "Claude Code could not sign in on the box. Last message. $text"
  fi
  if grep -qiE 'usage limit|rate.?limit|limit reached|hit your|(weekly|session|opus|fable|hour) limit|resets |out of extra usage|too many requests|overloaded|at capacity' <<<"$text"; then
    (( FAILS == 1 || FAILS % 12 == 0 )) && notify "Paused by a usage limit" "The loop retries every $(( LIMIT_SLEEP / 60 )) minutes. $text"
    say "Usage limit or overload. Sleeping $LIMIT_SLEEP s. $text"
    local slept=0 nap=60
    (( LIMIT_SLEEP < nap )) && nap="$LIMIT_SLEEP"
    while (( slept < LIMIT_SLEEP )); do beat; sleep "$nap"; slept=$(( slept + nap )); done
    return 0
  fi
  HARD_FAILS=$(( HARD_FAILS + 1 ))
  (( HARD_FAILS == 3 )) && notify "Sessions are failing" "$HARD_FAILS in a row. $text"
  if (( HARD_FAILS >= FAIL_LIMIT )); then
    wait_for_andrew "$HARD_FAILS Claude Code sessions in a row failed to run, and it was not a usage limit. Last message. $text"
  fi
  say "$what failed. Sleeping $ERROR_SLEEP s. $text"
  sleep "$ERROR_SLEEP"
}

agent_session() {  # $1 stage
  local stage="$1" ts log model effort rc result subtype turns cost sid used text is_err reason outcome rl
  ts="$(date -u +%Y%m%dT%H%M%SZ)-$(printf '%03d' "$ATTEMPTS")"
  ATTEMPTS=$(( ATTEMPTS + 1 ))
  log="$ROOT/logs/agent/$ts.jsonl"
  model="$MODEL"
  effort="$EFFORT"
  # the two stages that decide the result, the diagnosis (1) and the edit design (3), get their own settings
  if [[ "$stage" == "1" || "$stage" == "3" ]]; then
    [[ -n "${AWE_MODEL_DESIGN:-}" ]] && model="$AWE_MODEL_DESIGN"
    effort="$EFFORT_DESIGN"
  fi
  local args=(-p "$(cat "$AWE_PRISTINE/run/AGENT_PROMPT.md")" --permission-mode auto --permission-prompts none --settings "$(settings_file)" --output-format stream-json --verbose)
  [[ -n "$model" ]] && args+=(--model "$model")
  [[ -n "$effort" ]] && args+=(--effort "$effort")
  [[ -n "$MAX_TURNS" ]] && args+=(--max-turns "$MAX_TURNS")
  say "Session $ts starting (stage $stage, model ${model:-default}, effort ${effort:-default})."
  # stdin is closed on purpose. Print mode otherwise waits for piped input, or reads keys typed into the tmux pane
  ( cd "$ROOT" && AWE_AGENT=1 AWE_SESSION_TS="$ts" timeout --signal=INT --kill-after=120 "${SESSION_HOURS}h" "$CLAUDE" "${args[@]}" ) < /dev/null > "$log" 2> "$ROOT/logs/agent/$ts.err"
  rc=$?
  result="$(jq -R -c 'fromjson? | select(type == "object" and .type == "result")' "$log" 2>/dev/null | tail -1)"
  [[ -n "$result" ]] || result='{}'
  subtype="$(echo "$result" | jq -r '.subtype // "none"' 2>/dev/null || echo none)"
  is_err="$(echo "$result" | jq -r '.is_error // false' 2>/dev/null || echo false)"
  reason="$(echo "$result" | jq -r '.terminal_reason // empty' 2>/dev/null || true)"
  turns="$(echo "$result" | jq -r '.num_turns // 0' 2>/dev/null || echo 0)"
  cost="$(echo "$result" | jq -r '.total_cost_usd // 0' 2>/dev/null || echo 0)"
  sid="$(echo "$result" | jq -r '.session_id // "unknown"' 2>/dev/null || echo unknown)"
  used="$(jq -R -r 'fromjson? | select(type == "object" and .type == "system" and .subtype == "init") | .model // "unknown"' "$log" 2>/dev/null | head -1)"
  [[ -n "$used" ]] || used="unknown"
  text="$( { echo "$result" | jq -r '.result // empty' 2>/dev/null; tail -c 600 "$ROOT/logs/agent/$ts.err"; } | tr '\n' ' ' | cut -c1-500)"
  # Claude Code reports a failed session as subtype success with is_error true, so name the outcome by its cause
  outcome="$subtype"
  [[ "$is_err" == "true" && "$subtype" == "success" ]] && outcome="error_${reason:-unknown}"
  # usage notes that Claude Code prints into the stream, when it prints them
  rl="$(jq -R -c 'fromjson? | select(type == "object" and .type == "rate_limit_event") | .rate_limit_info' "$log" 2>/dev/null | tail -1)"
  if [[ -n "$rl" && "$(jq -r '.status // "allowed"' <<<"$rl" 2>/dev/null)" != "allowed" ]]; then
    say "Usage note from Claude Code. $(cut -c1-300 <<<"$rl")"
    [[ "$(jq -r '.status // ""' <<<"$rl" 2>/dev/null)" == "rejected" ]] && text="$text (rate limit rejected)"
  fi
  # Claude Code warns on stderr when it runs at a lower level than asked, for example under a cap
  if grep -qi "effort" "$ROOT/logs/agent/$ts.err" 2>/dev/null; then
    say "Claude Code said something about effort. $(grep -i "effort" "$ROOT/logs/agent/$ts.err" | head -n 2 | tr '\n' ' ' | cut -c1-300)"
  fi
  [[ -f "$ROOT/logs/sessions.tsv" ]] || printf 'time\tsession\tmodel\teffort\tstage\tturns\tcost_estimate_usd\tresult\texit\n' > "$ROOT/logs/sessions.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$sid" "$used" "${effort:-default}" "$stage" "$turns" "$cost" "$outcome" "$rc" >> "$ROOT/logs/sessions.tsv"
  # every start is on the record, the failed ones too, and the tree stays clean for the next session
  (cd "$ROOT" && git add logs/sessions.tsv 2>/dev/null && git -c user.name=awe-referee -c user.email=awe-referee@users.noreply.github.com commit -q -m "referee: session log" -- logs/sessions.tsv >/dev/null 2>&1) || true
  bash "$AWE_PRISTINE/run/redact.sh" "$log" || true
  gzip -f "$log" 2>/dev/null || true
  say "Session $ts ended (exit $rc, $outcome, $turns turns)."

  # a session that ran out of turns or time still did a leg of the work
  if [[ ( "$rc" == "0" && "$is_err" != "true" ) || "$subtype" == "error_max_turns" || "$rc" == "124" || "$rc" == "137" ]]; then
    SESSIONS=$(( SESSIONS + 1 ))
    ran_ok
    return 0
  fi
  after_failure "Session" "$text"
  return 1
}

say "Loop started. Repo $ROOT, rules from $AWE_GENESIS, python $PY."
say "Effort is ${EFFORT:-the model default} for most sessions, ${EFFORT_DESIGN:-the model default} for stages 1 and 3, and ${AWE_REVIEW_EFFORT:-the model default} for reviews."
[[ -n "$EFFORT_NOTES" ]] && say "Effort note. $EFFORT_NOTES"
notify "Run started" "$(basename "$ROOT") on $(hostname)"

while :; do
  beat
  if [[ -f "$ROOT/PAUSE" ]]; then sleep 30; continue; fi
  if (( SESSIONS >= MAX_SESSIONS )); then wait_for_andrew "The run used all $MAX_SESSIONS sessions without finishing."; fi
  if (( $(date +%s) - START_EPOCH > MAX_DAYS * 86400 )); then wait_for_andrew "The run passed $MAX_DAYS days without finishing."; fi
  CUR_BRANCH="$(git -C "$ROOT" symbolic-ref --short -q HEAD || true)"
  if [[ "$CUR_BRANCH" != "$RUN_BRANCH" ]]; then
    wait_for_andrew "The repo is on ${CUR_BRANCH:-a detached commit}, and the run lives on the branch $RUN_BRANCH. Check out $RUN_BRANCH and bring over any work worth keeping."
  fi

  # the rules in force are the ones at the genesis tag
  if [[ "$(git -C "$ROOT" tag -l 'genesis-*' | sort -t- -k2 -n | tail -1)" != "$AWE_GENESIS" ]]; then
    say "A newer genesis tag exists. Restarting on the new rules."
    unset AWE_PRISTINE
    exec bash "$ROOT/run/loop.sh" "$@"
  fi
  if ! ref guard --heal >>"$LOG" 2>&1; then
    wait_for_andrew "Protected files differ from $AWE_GENESIS and could not be restored. See logs/loop.log."
  fi

  wait_for_jobs
  [[ -f "$ROOT/PAUSE" ]] && continue

  if ! ST="$(ref status --json 2>>"$LOG")"; then
    say "The status command failed. Sleeping."
    sleep "$ERROR_SLEEP"
    continue
  fi
  STATE="$(jq -r .state <<<"$ST")"
  SIG_BEFORE="$(jq -r .signature <<<"$ST")/$(work_hash)"
  say "State $STATE $(jq -r '[.task // "", (if .stage != null then "stage \(.stage)" else "" end), .action // "", .gate // ""] | map(select(. != "")) | join(" ")' <<<"$ST")"

  case "$STATE" in
    DONE)
      stop_run 0 "Run finished" "The protocol is complete. Read publish/CHECKLIST.md."
      ;;
    WAITING_FOR_ANDREW)
      stop_run 3 "Run is waiting for you" "$(jq -r '.note // ""' <<<"$ST")"
      ;;
    SETUP_INCOMPLETE)
      stop_run 2 "Setup is not complete" "$(jq -r '.note // ""' <<<"$ST")"
      ;;
    JUDGE_TURN)
      if ref judge step >>"$ROOT/logs/referee.log" 2>&1; then
        JUDGE_FAILS=0
      else
        JUDGE_FAILS=$(( JUDGE_FAILS + 1 ))
        say "Referee step failed ($JUDGE_FAILS in a row). See logs/referee.log."
        if (( JUDGE_FAILS >= 3 )); then wait_for_andrew "A referee step failed 3 times in a row. $(tail -n 5 "$ROOT/logs/referee.log" | tr '\n' ' ' | cut -c1-400)"; fi
        sleep "$ERROR_SLEEP"
      fi
      push || true
      continue
      ;;
    REVIEW_TURN)
      ref review step >>"$ROOT/logs/referee.log" 2>&1
      RC=$?
      if [[ "$RC" == "0" ]]; then
        REVIEW_FAILS=0
        ran_ok
      elif [[ "$RC" == "75" ]]; then
        # the reviewer could not run at all. Wait as for an agent session, and never skip the review
        after_failure "Reviewer" "$(tr '\n' ' ' < "$ROOT/.awe/review-last-error.txt" 2>/dev/null | cut -c1-500)"
      else
        REVIEW_FAILS=$(( REVIEW_FAILS + 1 ))
        say "The reviewer gave no usable verdict ($REVIEW_FAILS in a row). It will be tried again."
        if (( REVIEW_FAILS >= 6 )); then wait_for_andrew "The review step failed 6 times in a row. $(tail -n 5 "$ROOT/logs/referee.log" | tr '\n' ' ' | cut -c1-400)"; fi
        sleep 60
      fi
      push || true
      continue
      ;;
    AGENT_TURN)
      agent_session "$(jq -r '.stage // 0' <<<"$ST")" || continue
      push || true
      ;;
    *)
      say "Unknown state $STATE."
      sleep "$ERROR_SLEEP"
      continue
      ;;
  esac

  # a stall is several sessions in a row that change neither the status nor any work product
  if ST2="$(ref status --json 2>>"$LOG")"; then
    SIG_AFTER="$(jq -r .signature <<<"$ST2")/$(work_hash)"
    if [[ "$SIG_AFTER" == "$SIG_BEFORE" ]]; then STALL=$(( STALL + 1 )); else STALL=0; fi
    if (( STALL >= STALL_LIMIT )); then
      wait_for_andrew "The last $STALL sessions changed nothing. The agent is stuck at $(jq -r '[.task // "", "stage", (.stage|tostring)] | join(" ")' <<<"$ST2"). Read the tail of PROGRESS.md."
    fi
  fi
done
