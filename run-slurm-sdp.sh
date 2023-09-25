#!/usr/bin/env bash

set -e

for DATASET in 'ErdosRenyi' 'BarabasiAlbert' 'PowerlawCluster' 'WattsStrogatz' 'ForcedRB' ; do
    echo $DATASET
    LLsub ./submit-sdp-inf.sh -- $DATASET
done
for DATASET in 'ENZYMES' 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 'COLLAB' ; do
    echo $DATASET
    LLsub ./submit-sdp.sh -- $DATASET
done
