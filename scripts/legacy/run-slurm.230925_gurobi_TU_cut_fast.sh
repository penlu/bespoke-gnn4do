#!/usr/bin/env bash

set -e

export PREFIX=230925_gurobi_TU_cut_fast

for DATASET in 'ENZYMES' 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 'COLLAB' 'REDDIT-MULTI-5K' 'REDDIT-MULTI-12K' 'REDDIT-BINARY'; do
    for TIMEOUT in '0.1' '0.5' '1' ; do
        echo $DATASET $TIMEOUT
        LLsub ./submit-baseline.sh -- \
          --dataset=$DATASET --problem_type=max_cut \
          --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
    done
done
