#!/usr/bin/env bash

set -e

run_job () {
    DATASET=$1
    echo $DATASET
    for i in $(seq 0 5 200) ; do
        #echo $DATASET $i $((i + 5))
	LLsub ./submit-sdp-bigrandom.sh -- $DATASET $i $((i + 5))
    done
}
export -f run_job

for dataset in 'RANDOM' ; do
    echo $dataset
done | xargs -I{} bash -c "run_job '{}'"
