#!/usr/bin/env bash

set -e

run_job () {
    JOBARRAY=($1)
    MODEL=${JOBARRAY[0]}
    DATASET=${JOBARRAY[1]}
    echo $MODEL $DATASET
    LLsub ./submit-train.sh -- $MODEL $DATASET
    #./submit.sh $MODEL $DATASET
}
export -f run_job

for model in 'LiftMP' 'GIN' 'GAT' 'GCNN' 'GatedGCNN' 'NegationGAT' ; do
    for dataset in 'RANDOM' 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
        echo $model $dataset
    done
done | xargs -I{} bash -c "run_job '{}'"
