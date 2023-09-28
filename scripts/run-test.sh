#!/usr/bin/env bash

set -e

for i in $(cat analysis_ipynb/models_for_test.txt | grep -v "nan," | cut -d "'" -f 2) ; do
    if [ -d $i ] ; then
        echo $i
    fi
done
