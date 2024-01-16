#!/usr/bin/env bash

set -e

export PREFIX=231116_TORCHGEOPRESET_SAT

for MODEL in 'GIN' 'GAT' 'GCNN' 'GatedGCNN'; do
    for DATASET in 'random-sat' ; do
        for LIFT_LAYERS in '8' '16' ; do
            for R in '4' '8' '16' ; do
                echo --stepwise=True --steps=200000 \
                  --valid_freq=2000 --dropout=0 \
                  --prefix=$PREFIX \∫d
                  --model_type=$MODEL --dataset=$DATASET \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --infinite=True --problem_type=sat \
                  --batch_size=1
            done
            echo --stepwise=True --steps=200000 \
              --valid_freq=2000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --infinite=True --problem_type=sat \
              --batch_size=1
        done
    done
done | parallel --ungroup -j2 'export CUDA_VISIBLE_DEVICES=$(({%} - 1)); eval python -u train.py {}'
#'export CUDA_VISIBLE_DEVICES=$(({%} - 1)); python -u train.py {}'