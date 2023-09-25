#!/usr/bin/env bash

set -e

for model in 'LiftMP' ; do
    for dataset in 'RANDOM_inf' 'ForcedRB_inf' ; do
        for penalty in '1' ; do
            for r in '2' '4' '8' '16' '32' ; do
                for lift_layers in '1' '4' '8' '12' '16' '20' ; do
                    echo $model $dataset $penalty $r $lift_layers
                    LLsub ./submit-hparam.sh -- $model $dataset $penalty $r $lift_layers
                done
            done
        done
    done
done
