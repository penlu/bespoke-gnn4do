#!/usr/bin/env bash

set -e

for MODEL in 'LiftMP' ; do
    for DATASET in 'random-sat' ; do
        for LIFT_LAYERS in '8' '16' ; do
            for R in '4' '8' '16' '32' ; do
                echo $MODEL $DATASET $PENALTY $R $LIFT_LAYERS
                LLsub ./submit.sh -- \
                  --stepwise=True --steps=100000 \
                  --valid_freq=2000 --dropout=0 \
                  --prefix=240115_sat_N100_K100 \
                  --model_type=$MODEL --dataset=$DATASET --infinite=True --parallel=20 \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --penalty=1 --problem_type=sat \
                  --batch_size=32 --gen_n=100 --gen_k=100
            done
        done
    done
done
