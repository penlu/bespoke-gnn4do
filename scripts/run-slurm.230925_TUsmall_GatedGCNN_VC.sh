#!/usr/bin/env bash

set -e

export PREFIX=230925_TUsmall_GatedGCNN_VC

for MODEL in 'GatedGCNN' ; do
    for DATASET in 'ENZYMES' 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 'COLLAB' ; do
        for LIFT_LAYERS in '8' '16' ; do
            for R in '4' '8' '16' ; do
                echo $MODEL $DATASET $R $LIFT_LAYERS
                LLsub ./submit2.sh -- \
                  --stepwise=True --steps=100000 \
                  --valid_freq=1000 --dropout=0 \
                  --prefix=$PREFIX \
                  --model_type=$MODEL --dataset=$DATASET \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --vc_penalty=1 --problem_type=vertex_cover \
                  --batch_size=16 --pe_dimension=$((R/2))
            done
            LLsub ./submit2.sh -- \
              --stepwise=True --steps=100000 \
              --valid_freq=1000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --vc_penalty=1 --problem_type=vertex_cover \
              --batch_size=16 --pe_dimension=8
        done
    done
done
