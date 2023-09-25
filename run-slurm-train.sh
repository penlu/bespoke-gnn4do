#!/usr/bin/env bash

set -e

for model in 'LiftMP' 'GIN' 'GAT' 'GCNN' 'GatedGCNN' 'NegationGAT' ; do
    for dataset in 'RANDOM' 'ForcedRB' 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
        LLsub ./submit-train.sh -- $MODEL $DATASET
    done
done
