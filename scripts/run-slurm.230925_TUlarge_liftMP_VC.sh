#!/usr/bin/env bash

set -e

export PREFIX=230925_TUlarge_liftMP_VC

for MODEL in 'LiftMP' ; do
    for DATASET in 'REDDIT-BINARY' 'REDDIT-MULTI-5K' 'REDDIT-MULTI-12K' ; do
        # these fractions are tuned to produce a val and test split of 100 each
        if [ $DATASET = 'REDDIT-BINARY' ] ; then
            FRACTION=0.9
        elif [ $DATASET = 'REDDIT-MULTI-5K' ] ; then
            FRACTION=0.96
        elif [ $DATASET = 'REDDIT-MULTI-12K' ] ; then
            FRACTION=0.98324
        fi

        for LIFT_LAYERS in '8' '16' ; do
            for R in '4' '8' '16' ; do
                echo $MODEL $DATASET $R $LIFT_LAYERS
                LLsub ./submit.sh -- \
                  --stepwise=True --steps=100000 \
                  --valid_freq=2000 --dropout=0 \
                  --prefix=$PREFIX \
                  --model_type=$MODEL --dataset=$DATASET --train_fraction=$FRACTION \
                  --num_layers=$LIFT_LAYERS --rank=$R \
                  --vc_penalty=1 --problem_type=vertex_cover \
                  --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
            done
            echo $MODEL $DATASET 32 $LIFT_LAYERS
            LLsub ./submit.sh -- \
              --stepwise=True --steps=100000 \
              --valid_freq=2000 --dropout=0 \
              --prefix=$PREFIX \
              --model_type=$MODEL --dataset=$DATASET --train_fraction=$FRACTION \
              --num_layers=$LIFT_LAYERS --rank=32 \
              --vc_penalty=1 --problem_type=vertex_cover \
              --batch_size=16 --positional_encoding=random_walk --pe_dimension=8
        done
    done
done
