#!/usr/bin/env bash

set -e

for model in 'GIN' 'GAT' 'GCNN' 'GatedGCNN' ; do
    for dataset in 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
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
