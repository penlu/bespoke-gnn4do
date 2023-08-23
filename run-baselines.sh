#!/usr/bin/env bash
set -e

for i in RANDOM PROTEINS IMDB-BINARY MUTAG COLLAB; do
    echo $i
    if [ $i = 'RANDOM' ] ; then
        TYPE='RANDOM'
    else
        TYPE='TU'
    fi
    python baselines.py --dataset $TYPE \
      --prefix 230823_gurobi_1s_$i --TUdataset_name $i \
      --gurobi True --gurobi_timeout 1
done
