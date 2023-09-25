#!/usr/bin/env bash

set -e

for dataset in 'RANDOM' 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'ForcedRB' ; do
    LLsub ./submit-sdp.sh -- $DATASET
done
