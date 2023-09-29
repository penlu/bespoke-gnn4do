#!/usr/bin/env bash

set -e

for i in training_runs/230925_generated_liftMP_VC/* ; do
    echo $i
    LLsub ./submit-kamis.sh -- --model_folder="$i" --model_file=model_step20000.pt --test_prefix=kamis_test
done
