#!/usr/bin/env bash

set -e

export PREFIX=230925_TUsmall_VC_32

for MODEL in 'LiftMP' 'GAT' 'GIN' 'GCNN' ; do
    for DATASET in 'ENZYMES' 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 'COLLAB' ; do
        for LIFT_LAYERS in '8' '16' ; do
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
