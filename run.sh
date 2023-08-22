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
        --valid_freq=100 --dropout=0 \
        --positional_encoding=laplacian_eigenvector --pe_dimension=8 \
        --prefix=230822_test \
        --model_type=$MODEL --TUdataset_name=$DATASET --dataset=TU
}
export -f run_job

for model in 'GIN' 'GAT' 'GCNN' 'GatedGCNN' ; do
    for dataset in 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
        echo $model $dataset
    done
done | parallel --ungroup -j1 run_job {} {%}
