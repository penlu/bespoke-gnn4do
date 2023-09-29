#!/usr/bin/env bash

set -e

for i in $(cat analysis_ipynb/models_for_test.txt | grep -v "nan," | cut -d "'" -f 2 | sort | uniq) ; do
    if [ -d $i ] ; then
        echo $i
        LLsub ./submit-test.sh -- --model_folder="$i" --model_file=best_model.pt --test_prefix=time_and_score
    fi
done
