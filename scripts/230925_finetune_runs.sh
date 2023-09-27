#!/usr/bin/env bash

set -e

run_job () {
    JOBARRAY=($1)
    MODEL_FOLDER=${JOBARRAY[0]}
    DATASET=${JOBARRAY[1]}
    SLOT=$2
    echo $MODEL $TYPE $DATASET $((SLOT - 1))
    CUDA_VISIBLE_DEVICES=$((SLOT - 1)) python train.py \
        --stepwise=True --steps=20000 \
        --valid_freq=200 --prefix=230926_finetune_ER_runs \
        --finetune_from=$MODEL_FOLDER --dataset=$DATASET --problem_type=vertex_cover --dropout=0 --vc_penalty=1
}
export -f run_job

pretrained_models=("/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:c259ee991d44601a24495085d478929b226e94ca5ffc2a75ab739957d94da4b0/best_model.pt"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:9ede9cd33a4971c3246500914847936dbc9690bd4c05a450cdef0d6146b815f4/best_model.pt"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:631e9afcf3bd0330c4f9625392cc0b183926a43b2ba3d8c5d8fbaf91c779dc90/best_model.pt"
)

for model_folder in "${pretrained_models[@]}" ; do
    for dataset in 'RANDOM' 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
        echo $model_folder $dataset
    done
done | parallel --ungroup -j1 run_job {} {%}