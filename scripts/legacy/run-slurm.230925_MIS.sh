#!/usr/bin/env bash

set -e

for MODEL in 'LiftMP' ; do
    for DATASET in 'ErdosRenyi' 'BarabasiAlbert' 'PowerlawCluster' 'WattsStrogatz' ; do
        for PENALTY in '1' ; do
            for R in '1' '2' '4' '8' '16' '32' ; do
                for LIFT_LAYERS in '2' '16' ; do
                    echo $MODEL $DATASET $PENALTY $R $LIFT_LAYERS
                    LLsub ./submit.sh -- \
                      --stepwise=True --steps=10000 \
                      --valid_freq=1000 --dropout=0 \
                      --prefix=230925_MIS \
                      --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 50 100 \
                      --num_layers=$LIFT_LAYERS --rank=$R \
                      --vc_penalty=$PENALTY --problem_type=vertex_cover \
                      --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
                    LLsub ./submit.sh -- \
                      --stepwise=True --steps=10000 \
                      --valid_freq=1000 --dropout=0 \
                      --prefix=230925_MIS \
                      --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 100 200 \
                      --num_layers=$LIFT_LAYERS --rank=$R \
                      --vc_penalty=$PENALTY --problem_type=vertex_cover \
                      --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
                    LLsub ./submit.sh -- \
                      --stepwise=True --steps=10000 \
                      --valid_freq=1000 --dropout=0 \
                      --prefix=230925_MIS \
                      --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True --gen_n 400 500 \
                      --num_layers=$LIFT_LAYERS --rank=$R \
                      --vc_penalty=$PENALTY --problem_type=vertex_cover \
                      --batch_size=16 --positional_encoding=random_walk --pe_dimension=$((R/2))
                done
            done
        done
    done
done
