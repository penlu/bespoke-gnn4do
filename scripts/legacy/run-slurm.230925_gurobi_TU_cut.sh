#!/usr/bin/env bash

set -e

export PREFIX=230925_gurobi_TU_cut

for DATASET in 'ENZYMES' 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 'COLLAB' 'REDDIT-MULTI-5K' 'REDDIT-MULTI-12K' 'REDDIT-BINARY'; do
    for TIMEOUT in 2 4 8 ; do
        echo $DATASET $TIMEOUT
        LLsub ./submit-baseline.sh -- \
          --dataset=$DATASET --problem_type=max_cut \
          --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
    done
done
