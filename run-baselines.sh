#!/usr/bin/env bash
set -e

for i in RANDOM ENZYMES PROTEINS IMDB-BINARY MUTAG COLLAB; do
    echo $i
    if [ $i = 'RANDOM' ] ; then
        TYPE='RANDOM'
    else
        TYPE='TU'
    fi
    python baselines.py --dataset $TYPE \
      --problem_type vertex_cover \
      --prefix 230902_gurobi_5s_$i --TUdataset_name $i \
      --gurobi True --gurobi_timeout 5
done
