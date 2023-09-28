#!/usr/bin/env bash

set -e

export PREFIX=230925_generated_preset_VC

for MODEL in 'GIN' 'GAT' 'GCNN' 'GatedGCNN' ; do
    for DATASET in 'ErdosRenyi' ; do
        for LIFT_LAYERS in '8' '16' ; do
            echo $MODEL $DATASET $LIFT_LAYERS
            LLsub ./submit.sh -- \
              --stepwise=True --steps=20000 \
              --valid_freq=1000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 50 100 \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --vc_penalty=1 --problem_type=vertex_cover \
              --batch_size=16 --positional_encoding=random_walk --pe_dimension=8
            LLsub ./submit.sh -- \
              --stepwise=True --steps=20000 \
              --valid_freq=1000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 100 200 \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --vc_penalty=1 --problem_type=vertex_cover \
              --batch_size=16 --positional_encoding=random_walk --pe_dimension=8
            LLsub ./submit.sh -- \
              --stepwise=True --steps=20000 \
              --valid_freq=1000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 400 500 \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --vc_penalty=1 --problem_type=vertex_cover \
              --batch_size=16 --positional_encoding=random_walk --pe_dimension=8
        done
    done
done
