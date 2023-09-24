#!/usr/bin/env bash

set -e

run_job () {
    DATASET=$1
    echo $DATASET
    for i in $(seq 0 125 5000) ; do
        LLsub ./submit-sdp-bigrandom.sh -- $DATASET $i $((i + 125))
    done
}
export -f run_job

for dataset in 'COLLAB' ; do
    echo $dataset
done | xargs -I{} bash -c "run_job '{}'"
