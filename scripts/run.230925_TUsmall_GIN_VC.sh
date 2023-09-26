#!/usr/bin/env bash

set -e

export PREFIX=230925_TUsmall_GIN_VC

for MODEL in 'GIN' ; do
    for DATASET in 'ENZYMES' 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 'COLLAB' ; do
        for LIFT_LAYERS in '8' '16' ; do
            for R in '4' '8' '16' ; do
                echo --stepwise=True --steps=100000 \
                  --valid_freq=1000 --dropout=0 \
                  --prefix=$PREFIX \
                  --model_type=$MODEL --dataset=$DATASET \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --vc_penalty=1 --problem_type=vertex_cover \
                  --batch_size=16 --positional_encoding=laplacian_eigenvector --pe_dimension=$((R/2))
                echo --stepwise=True --steps=100000 \
                  --valid_freq=1000 --dropout=0 \
                  --prefix=$PREFIX \
                  --model_type=$MODEL --dataset=$DATASET \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --vc_penalty=1 --problem_type=vertex_cover \
                  --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
            done
            echo --stepwise=True --steps=100000 \
              --valid_freq=1000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --vc_penalty=1 --problem_type=vertex_cover \
              --batch_size=16 --positional_encoding=laplacian_eigenvector --pe_dimension=8
            echo --stepwise=True --steps=100000 \
              --valid_freq=1000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --vc_penalty=1 --problem_type=vertex_cover \
              --batch_size=16 --positional_encoding=random_walk --pe_dimension=16
        done
    done
done | parallel --ungroup -j2 'export CUDA_VISIBLE_DEVICES=$(({%} - 1)); eval python -u train.py {}'
#'export CUDA_VISIBLE_DEVICES=$(({%} - 1)); python -u train.py {}'
