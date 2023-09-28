#!/usr/bin/env bash

set -e

export PREFIX=230927_gurobi_cut

for TIMEOUT in '0.1' '0.5' '1' '2' '4' '8' ; do
    for DATASET in 'ErdosRenyi' 'BarabasiAlbert' 'PowerlawCluster' 'WattsStrogatz' ; do
        echo $DATASET $TIMEOUT
        LLsub ./submit-baseline.sh -- \
          --dataset=$DATASET --problem_type=max_cut --parallel=20 --infinite=True --gen_n 50 100 \
          --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
        LLsub ./submit-baseline.sh -- \
          --dataset=$DATASET --problem_type=max_cut --parallel=20 --infinite=True --gen_n 100 200 \
          --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
        LLsub ./submit-baseline.sh -- \
          --dataset=$DATASET --problem_type=max_cut --parallel=20 --infinite=True --gen_n 400 500 \
          --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
    done
    for DATASET in 'ENZYMES' 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 'COLLAB' 'REDDIT-MULTI-5K' 'REDDIT-MULTI-12K' 'REDDIT-BINARY'; do
        echo $DATASET $TIMEOUT
        LLsub ./submit-baseline.sh -- \
          --dataset=$DATASET --problem_type=max_cut \
          --prefix=$PREFIX --gurobi=True --gurobi_timeout=$TIMEOUT
    done
done
