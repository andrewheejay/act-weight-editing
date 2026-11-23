#!/usr/bin/env bash
# Sends one notification.  usage: notify.sh "Title" "Message"
# Set AWE_NTFY_TOPIC for a phone push through ntfy.sh, or AWE_SLACK_WEBHOOK for Slack, or both.
# Every notification is also written to logs/notify.log.
title="${1:-notice}"
msg="${2:-}"
root="${AWE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$root/logs" 2>/dev/null
printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$title" "$msg" >> "$root/logs/notify.log" 2>/dev/null
if [[ -n "${AWE_NTFY_TOPIC:-}" ]]; then
  curl -fsS -m 15 -H "Title: $title" -d "${msg:-$title}" "${AWE_NTFY_SERVER:-https://ntfy.sh}/$AWE_NTFY_TOPIC" >/dev/null 2>&1
fi
if [[ -n "${AWE_SLACK_WEBHOOK:-}" ]]; then
  payload="$(jq -n --arg t "*$title*
$msg" '{text: $t}')"
  curl -fsS -m 15 -X POST -H 'Content-type: application/json' --data "$payload" "$AWE_SLACK_WEBHOOK" >/dev/null 2>&1
fi
exit 0
