#!/usr/bin/env bash

set -e

run_job () {
    JOBARRAY=($1)
    MODEL=${JOBARRAY[0]}
    DATASET=${JOBARRAY[1]}
    PENALTY=${JOBARRAY[2]}
    R=${JOBARRAY[3]}
    LIFT_LAYERS=${JOBARRAY[4]}
    LLsub ./submit-hparam.sh -- $MODEL $DATASET $PENALTY $R $LIFT_LAYERS
}
export -f run_job

for model in 'LiftMP' ; do
    for dataset in 'RANDOM_inf' 'ForcedRB_inf' ; do
        for penalty in '1' ; do
            for r in '2' '4' '8' '16' '32' ; do
                for lift_layers in '1' '4' '8' '12' '16' '20' ; do
                    echo $model $dataset $penalty $r $lift_layers
                done
            done
        done
    done
done | xargs -I{} bash -c "run_job '{}'"
