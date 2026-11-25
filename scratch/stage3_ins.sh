#!/bin/sh
# Stage 3 for insertion, first part, resumable: dose response, then the output-level baselines. A failed step stops the chain.
# Result files are gzipped when complete (row counts checked first). Nothing here chooses an edit.
set -e
cd /home/ubuntu/act-weight-editing
D=experiments/insertion/stage3
sleep "${1:-0}"
if [ ! -e $D/dose.jsonl.gz ]; then
  python -m awe.dose_ins run
  [ "$(wc -l < $D/dose.jsonl)" = 9600 ]
  gzip -n $D/dose.jsonl
fi
python -m awe.dose_ins report
if [ ! -e $D/baselines_screen.jsonl.gz ]; then
  python -m awe.baselines_ins screen
  [ "$(wc -l < $D/baselines_screen.jsonl)" = 32800 ]
  gzip -n $D/baselines_screen.jsonl
fi
if [ ! -e $D/baselines_full.jsonl.gz ]; then
  python -m awe.baselines_ins full
  [ "$(wc -l < $D/baselines_full.jsonl)" = 12000 ]
  gzip -n $D/baselines_full.jsonl
fi
python -m awe.baselines_ins report
