#!/usr/bin/env bash

set -e

for i in $(cat analysis_ipynb/models_for_retest2.txt) ; do
    echo $i
    LLsub ./submit-revalidate.sh -- $i

    #echo -u test.py --model_folder=training_runs/$i --model_file=best_model.pt --test_prefix=retest_best
    #if [ -f training_runs/$i/model_step20000.pt ] ; then
    #    echo -u test.py --model_folder=training_runs/$i --model_file=model_step20000.pt --test_prefix=retest_last
    #else
    #    echo -u test.py --model_folder=training_runs/$i --model_file=model_step100000.pt --test_prefix=retest_last
    #fi
done
