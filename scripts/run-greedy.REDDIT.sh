#!/usr/bin/env bash
set -e

PREFIX=230927_greedy

for PROBLEM in 'max_cut' 'vertex_cover' ; do
    LLsub ./submit-greedy.sh -- --dataset=REDDIT-BINARY --train_fraction=0.9 \
      --problem_type $PROBLEM \
      --prefix $PREFIX \
      --greedy True
    LLsub ./submit-greedy.sh -- --dataset=REDDIT-MULTI-5K --train_fraction=0.96 \
      --problem_type $PROBLEM \
      --prefix $PREFIX \
      --greedy True
    LLsub ./submit-greedy.sh -- --dataset=REDDIT-MULTI-12K --train_fraction=0.98324 \
      --problem_type $PROBLEM \
      --prefix $PREFIX \
      --greedy True
done
