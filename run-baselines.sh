#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Please invoke with exactly one parameter: problem_type!"
fi

if [[ $1 != 'max_cut' && $1 != 'vertex_cover' ]] ; then
    echo "Invalid problem_type $1"
    exit
fi

for DATASET in RANDOM ENZYMES PROTEINS IMDB-BINARY MUTAG COLLAB ForcedRB; do
    echo $DATASET
    if [ $DATASET = 'RANDOM' ] ; then
        TYPE='RANDOM'
    elif [ $DATASET = 'ForcedRB' ] ; then
        TYPE='ForcedRB'
    else
        TYPE='TU'
    fi
    python baselines.py --dataset $TYPE \
      --problem_type $1 \
      --prefix 230916_gurobi_$1_20s_$DATASET --TUdataset_name $DATASET \
      --gurobi True --gurobi_timeout 20
done
