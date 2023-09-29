#!/usr/bin/env bash

set -e

for i in $(cat analysis_ipynb/models_for_test.txt | cut -d "'" -f 2 | sort | uniq) ; do
    echo $i
    LLsub ./submit-retest.sh -- $i
done
