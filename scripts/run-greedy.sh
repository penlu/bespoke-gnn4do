#!/usr/bin/env bash
set -e

PREFIX=230927_greedy

for PROBLEM in 'max_cut' 'vertex_cover' ; do
    for DATASET in 'ErdosRenyi' 'BarabasiAlbert' 'PowerlawCluster' 'WattsStrogatz' ; do
        echo $PROBLEM $DATASET
        LLsub ./submit-greedy.sh -- --dataset $DATASET --infinite=True --parallel=20 --gen_n 50 100 \
          --problem_type $PROBLEM \
          --prefix $PREFIX \
          --greedy True
        LLsub ./submit-greedy.sh -- --dataset $DATASET --infinite=True --parallel=20 --gen_n 100 200 \
          --problem_type $PROBLEM \
          --prefix $PREFIX \
          --greedy True
        LLsub ./submit-greedy.sh -- --dataset $DATASET --infinite=True --parallel=20 --gen_n 400 500 \
          --problem_type $PROBLEM \
          --prefix $PREFIX \
          --greedy True
    done
    for DATASET in 'ENZYMES' 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 'COLLAB' 'REDDIT-BINARY' 'REDDIT-MULTI-5K' 'REDDIT-MULTI-12K' ; do
        echo $PROBLEM $DATASET
        LLsub ./submit-greedy.sh -- --dataset $DATASET --infinite=True --parallel=20 \
          --problem_type $PROBLEM \
          --prefix $PREFIX \
          --greedy True
    done
done
