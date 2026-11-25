#!/bin/sh
# Stage 3 step 3 for insertion, after group A: group B for the four best by group A (2,000 rollouts), then the refinements on all 1,000 design seeds
# (6 ids x 1,000 = 6,000 rollouts). Resumable, the report at the end. Nothing here chooses an edit.
set -e
cd /home/ubuntu/act-weight-editing
D=experiments/insertion/stage3
sleep "${1:-0}"
if [ ! -e $D/combo_B.jsonl.gz ]; then
  python -m awe.combo_ins run B c06 c08 c02 c01
  [ "$(wc -l < $D/combo_B.jsonl)" = 2000 ]
  gzip -n $D/combo_B.jsonl
fi
if [ ! -e $D/combo_AB.jsonl.gz ]; then
  python -m awe.combo_ins run AB r1 r2 r3 r4 r5 r6
  [ "$(wc -l < $D/combo_AB.jsonl)" = 6000 ]
  gzip -n $D/combo_AB.jsonl
fi
python -m awe.combo_ins report
