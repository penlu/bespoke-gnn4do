#!/usr/bin/env bash

set -e

run_job () {
    JOBARRAY=($1)
    MODEL=${JOBARRAY[0]}
    DATASET=${JOBARRAY[1]}
    PENALTY=${JOBARRAY[2]}
    R=${JOBARRAY[3]}
    LIFT_LAYERS=${JOBARRAY[4]}
    SLOT=$2
    if [ $DATASET = 'RANDOM' ] ; then
        TYPE='RANDOM'
    else
        TYPE='TU'
    fi
    smallrs=('2' '4' '8')
    if [[ ${smallrs[@]} =~ $R ]] ;then
        PE_DIM=$R
    else
        PE_DIM='8'
    fi
    echo $MODEL $TYPE $DATASET $((SLOT - 1))
    CUDA_VISIBLE_DEVICES=$((SLOT - 1)) python train.py \
        --stepwise=True --steps=50000 \
        --valid_freq=100 --dropout=0 \
        --prefix=230904_hparam_att1 \
        --model_type=$MODEL --TUdataset_name=$DATASET --dataset=$TYPE \
        --num_layers=$LIFT_LAYERS --rank=$R --vc_penalty=$PENALTY --problem_type=vertex_cover
}
export -f run_job

for model in 'LiftMP' ; do
    for dataset in 'RANDOM' ; do
        for penalty in '0' '1' '2' '3' '4' '5'; do
            for r in '2' '4' '8' '16' '32'; do
                for lift_layers in '1' '2' '3' '4' '6' '10'; do
                    echo $model $dataset $penalty $r $lift_layers
                done
            done
        done
    done
done | parallel --ungroup -j2 run_job {} {%}
