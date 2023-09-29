#!/usr/bin/env bash

set -e

for i in $(cat analysis_ipynb/models_for_test2.txt | cut -d "'" -f 2 | head -n 52) ; do
    echo $i
    LLsub ./submit-retest.sh -- $i
done
