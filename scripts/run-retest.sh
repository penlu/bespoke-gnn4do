#!/usr/bin/env bash

set -e

for i in $(cat analysis_ipynb/models_for_retest2.txt) ; do
    echo $i
    LLsub ./submit-retest.sh -- training_runs/$i
done
