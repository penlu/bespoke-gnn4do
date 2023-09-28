#!/usr/bin/env bash

set -e

export PREFIX=230925_gurobi_generated_cut_fast

for DATASET in 'ErdosRenyi' 'BarabasiAlbert' 'PowerlawCluster' 'WattsStrogatz' ; do
    for TIMEOUT in '0.1' '0.5' '1' ; do
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
done
