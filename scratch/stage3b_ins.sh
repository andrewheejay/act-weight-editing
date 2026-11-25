#!/bin/sh
# Stage 3 step 3 for insertion, combinations on group A (16 ids x 500 seeds = 8,000 rollouts), resumable. Nothing here chooses an edit.
set -e
cd /home/ubuntu/act-weight-editing
D=experiments/insertion/stage3
sleep "${1:-0}"
if [ ! -e $D/combo_A.jsonl.gz ]; then
  python -m awe.combo_ins run A
  [ "$(wc -l < $D/combo_A.jsonl)" = 8000 ]
  gzip -n $D/combo_A.jsonl
fi
python -m awe.combo_ins report
