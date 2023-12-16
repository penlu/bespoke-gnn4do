#!/usr/bin/env bash

set -e

for MODEL in 'LiftMP' 'GAT' 'GIN' 'GCNN' 'GatedGCNN' ; do
    for DATASET in 'random-sat' ; do
        for LIFT_LAYERS in '8' '16' ; do
            for R in '4' '8' '16' '32' ; do
                echo $MODEL $DATASET $PENALTY $R $LIFT_LAYERS
                LLsub ./submit.sh -- \
                  --stepwise=True --steps=100000 \
                  --valid_freq=2000 --dropout=0 \
                  --prefix=231216_sat \
                  --model_type=$MODEL --dataset=$DATASET --infinite=True \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --penalty=1 --problem_type=sat \
                  --batch_size=32 --gen_n=100 --gen_k=200
            done
        done
    done
done
