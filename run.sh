#!/usr/bin/env bash

set -e

run_job () {
    JOBARRAY=($1)
    MODEL=${JOBARRAY[0]}
    DATASET=${JOBARRAY[1]}
    SLOT=$2
    echo $MODEL $DATASET $((SLOT - 1))
    CUDA_VISIBLE_DEVICES=$((SLOT - 1)) python train.py \
        --stepwise=True --steps=50000 \
        --valid_freq=1000 --dropout=0 \
        --prefix=230924_test \
        --model_type=$MODEL --dataset=$DATASET \
        --problem_type=vertex_cover --vc_penalty=1
}
export -f run_job

for model in 'LiftMP' 'GIN' 'GAT' 'GCNN' 'GatedGCNN' 'NegationGAT' ; do
    for dataset in 'RANDOM' 'ForcedRB' 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
        echo $model $dataset
    done
done | parallel --ungroup -j1 run_job {} {%}
