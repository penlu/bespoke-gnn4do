#!/usr/bin/env bash

set -e

for MODEL in 'LiftMP' ; do
    for DATASET in 'random-sat' ; do
        for LIFT_LAYERS in '16' ; do
            for R in '64' ; do
                for PENALTY in '0.01' '0.02' '0.03' '0.04' '0.05' '0.08' '0.12' '0.18' ; do
                    echo $MODEL $DATASET $PENALTY $R $LIFT_LAYERS
                    LLsub ./submit.sh -- \
                      --stepwise=True --steps=100000 \
                      --valid_freq=2000 --dropout=0 \
                      --prefix=240117_sat_penalty_N100_K400 \
                      --model_type=$MODEL --dataset=$DATASET --infinite=True --parallel=20 \
                      --num_layers=$LIFT_LAYERS --rank=$R \
                      --penalty=$PENALTY --problem_type=sat \
                      --positional_encoding=random_walk --pe_dimension=8 \
                      --batch_size=32 --gen_n=100 --gen_k=400
                done
            done
        done
    done
done
