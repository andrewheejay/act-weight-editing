#!/bin/sh
# Stage 2 for insertion, resumable. Order: sweep, (wait for the atlas job) selection, recheck, split-half. Everything is rule-driven
# (awe/atlas_ins.py docstring). A failed step stops the chain with a non-zero exit.
set -e
cd /home/ubuntu/act-weight-editing
S=experiments/insertion/stage2
sleep "${1:-0}"
if [ ! -e $S/sweep.jsonl.gz ]; then
  python -m awe.sweep run --task insertion
  n=$(wc -l < $S/sweep.jsonl)
  echo "sweep rows $n"
  [ "$n" = 48500 ]
  gzip -n $S/sweep.jsonl
fi
while [ ! -e jobs/atlas_ins.exit ]; do sleep 20; done
[ -e $S/atlas/heads.json ]
python -m awe.sweep_ins_report first
if [ ! -e $S/sweep_half2.jsonl.gz ]; then
  python -m awe.sweep recheck --task insertion
  gzip -n $S/sweep_half2.jsonl
fi
python -m awe.sweep_ins_report second
