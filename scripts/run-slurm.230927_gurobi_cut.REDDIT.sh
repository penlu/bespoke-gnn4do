#!/usr/bin/env bash

set -e

export PREFIX=230927_gurobi_cut

for TIMEOUT in '0.1' '0.5' '1' '2' '4' '8' ; do
    LLsub ./submit-baseline.sh -- \
      --dataset=REDDIT-BINARY --train_fraction=0.9 --problem_type=max_cut \
      --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
    LLsub ./submit-baseline.sh -- \
      --dataset=REDDIT-MULTI-5K --train_fraction=0.96 --problem_type=max_cut \
      --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
    LLsub ./submit-baseline.sh -- \
      --dataset=REDDIT-MULTI-12K --train_fraction=0.98324 --problem_type=max_cut \
      --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
done
