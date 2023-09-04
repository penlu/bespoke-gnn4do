#!/usr/bin/env bash

set -e

run_job () {
    DATASET=$1
    echo $DATASET
    LLsub ./submit-sdp.sh -- $DATASET
}
export -f run_job

for dataset in 'RANDOM' 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
    echo $dataset
done | xargs -I{} bash -c "run_job '{}'"
