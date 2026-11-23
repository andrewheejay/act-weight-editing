#!/usr/bin/env bash
# Masks anything that looks like a credential in a session log, in place.  usage: redact.sh FILE
# The logs are published with the result, so this runs on every one of them before it is compressed.
f="${1:?file}"
[[ -f "$f" ]] || exit 0
sed -i -E \
  -e 's/gh[pousr]_[A-Za-z0-9]{20,}/[redacted-github-token]/g' \
  -e 's/github_pat_[A-Za-z0-9_]{20,}/[redacted-github-token]/g' \
  -e 's/hf_[A-Za-z0-9]{20,}/[redacted-hf-token]/g' \
  -e 's/sk-ant-[A-Za-z0-9_-]{20,}/[redacted-anthropic-key]/g' \
  -e 's/AKIA[0-9A-Z]{16}/[redacted-aws-key]/g' \
  -e 's/(aws_secret_access_key|AWS_SECRET_ACCESS_KEY)([^A-Za-z0-9\/+]{1,6})[A-Za-z0-9\/+]{30,}/\1\2[redacted-aws-secret]/g' \
  "$f"
