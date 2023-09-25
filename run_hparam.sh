#!/usr/bin/env bash

set -e

run_job () {
    JOBARRAY=($1)
    MODEL=${JOBARRAY[0]}
    DATASET=${JOBARRAY[1]}
    PENALTY=${JOBARRAY[2]}
    R=${JOBARRAY[3]}
    LIFT_LAYERS=${JOBARRAY[4]}
    SLOT=$2
    echo $MODEL $DATASET $((SLOT - 1))
    CUDA_VISIBLE_DEVICES=$((SLOT - 1)) python train.py \
        --stepwise=True --steps=50000 \
        --valid_freq=1000 --dropout=0 \
        --prefix=230924_hparam_TU \
        --model_type=$MODEL --dataset=$DATASET \
        --num_layers=$LIFT_LAYERS --rank=$R --vc_penalty=$PENALTY --problem_type=vertex_cover \
        --positional_encoding=laplacian_eigenvector --pe_dimension=$((R/2)) \
        --batch_size=16
}
export -f run_job

for model in 'LiftMP' 'Nikos' ; do
    for dataset in 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
        for penalty in '1' ; do
            for r in '4' '8' '16' ; do
                for lift_layers in '1' '4' '8' '12' ; do
                    echo $model $dataset $penalty $r $lift_layers
                done
            done
        done
    done
done | parallel --ungroup -j2 run_job {} {%}
