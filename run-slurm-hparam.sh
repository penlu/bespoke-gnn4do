#!/usr/bin/env bash

set -e

for model in 'LiftMP' 'Nikos' ; do
    for dataset in 'ErdosRenyi' 'BarabasiAlbert' 'PowerlawCluster' 'WattsStrogatz' ; do
        for penalty in '1' ; do
            for r in '4' '8' '16' ; do
                for lift_layers in '1' '4' '8' '12' ; do
                    echo $model $dataset $penalty $r $lift_layers
                    LLsub ./submit-hparam.sh -- $model $dataset $penalty $r $lift_layers
                done
            done
        done
    done
done
