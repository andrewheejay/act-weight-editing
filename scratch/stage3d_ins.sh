#!/bin/sh
# Stage 3 step 5 for insertion: the original, the candidates of stage3/candidates.json (awe/candidates_ins.py list rule) and the 4 baseline families on
# the 500 validation seeds, then the summary and the choice rule (both fixed before any validation rollout). Resumable (finished files are skipped).
set -e
cd /home/ubuntu/act-weight-editing
sleep "${1:-0}"
python -m awe.validate_ins original
for c in $(python -c "import json;print(*json.load(open('experiments/insertion/stage3/candidates.json')))"); do
  python -m awe.validate_ins candidate "$c"
done
for f in constant_offset deviation_gain shorter_chunk simplest_fix; do
  python -m awe.validate_ins baseline "$f"
done
python -m awe.validate_ins report
python -m awe.candidates_ins choose
