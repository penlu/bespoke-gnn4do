#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Please invoke with exactly one parameter: problem_type!"
fi

if [[ $1 != 'max_cut' && $1 != 'vertex_cover' && $1 != 'max_clique' ]] ; then
    echo "Invalid problem_type $1"
    exit
fi

for TIMEOUT in 1 5 20 ; do
    for DATASET in RANDOM ENZYMES PROTEINS IMDB-BINARY MUTAG COLLAB ForcedRB; do
        echo $DATASET
        python baselines.py --dataset $DATASET \
          --problem_type $1 \
          --prefix 230924_gurobi_$1_${TIMEOUT}s_$DATASET \
          --gurobi True --gurobi_timeout $TIMEOUT
    done
done
