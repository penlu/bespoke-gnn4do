#!/usr/bin/env bash

set -e

export PREFIX=230925_generated_preset_cut

for MODEL in 'GIN' 'GAT' 'GCNN' 'GatedGCNN' ; do
    for DATASET in 'ErdosRenyi' ; do
        for R in '4' '8' '16' '32' ; do
            for LIFT_LAYERS in '8' '16' ; do
                echo $MODEL $DATASET $R $LIFT_LAYERS
                LLsub ./submit.sh -- \
                  --stepwise=True --steps=20000 \
                  --valid_freq=1000 --dropout=0 \
                  --prefix=$PREFIX \
                  --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 50 100 \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --problem_type=max_cut \
                  --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
                LLsub ./submit.sh -- \
                  --stepwise=True --steps=20000 \
                  --valid_freq=1000 --dropout=0 \
                  --prefix=$PREFIX \
                  --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 100 200 \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --problem_type=max_cut \
                  --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
                LLsub ./submit.sh -- \
                  --stepwise=True --steps=20000 \
                  --valid_freq=1000 --dropout=0 \
                  --prefix=$PREFIX \
                  --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 400 500 \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --problem_type=max_cut \
                  --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
            done
        done
    done
done
