#!/usr/bin/env bash

set -e

for model in 'LiftMP' ; do
    for dataset in 'ErdosRenyi' 'BarabasiAlbert' 'PowerlawCluster' 'WattsStrogatz' 'ForcedRB' ; do
        for penalty in '1' ; do
            for r in '1' '2' '4' '8' '16' '32' ; do
                for lift_layers in '2' '16' ; do
                    echo $model $dataset $penalty $r $lift_layers
                    LLsub ./submit-hparam.sh -- $model $dataset $penalty $r $lift_layers
                done
            done
        done
    done
done
