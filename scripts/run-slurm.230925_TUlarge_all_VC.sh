#!/usr/bin/env bash

set -e

export PREFIX=230925_TUlarge_all_VC

for MODEL in 'LiftMP' 'GIN' 'GAT' 'GCNN' 'GatedGCNN' ; do
    for DATASET in 'REDDIT-MULTI-5K' 'REDDIT-MULTI-12K' 'REDDIT-BINARY'; do
        for LIFT_LAYERS in '8' '16' ; do
            for R in '4' '8' '16' ; do
                echo $MODEL $DATASET $R $LIFT_LAYERS
                LLsub ./submit.sh -- \
                  --stepwise=True --steps=100000 \
                  --valid_freq=1000 --dropout=0 \
                  --prefix=$PREFIX \
                  --model_type=$MODEL --dataset=$DATASET \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --vc_penalty=1 --problem_type=vertex_cover \
                  --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
            done
            echo $MODEL $DATASET 32 $LIFT_LAYERS
            LLsub ./submit.sh -- \
              --stepwise=True --steps=100000 \
              --valid_freq=1000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --vc_penalty=1 --problem_type=vertex_cover \
              --batch_size=16 --positional_encoding=random_walk --pe_dimension=8
        done
    done
done
