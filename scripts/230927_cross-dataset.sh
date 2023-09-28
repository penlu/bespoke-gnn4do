#!/usr/bin/env bash

set -e

run_job () {
    JOBARRAY=($1)
    MODEL_FOLDER=${JOBARRAY[0]}
    DATASET=${JOBARRAY[1]}
    SLOT=$2
    echo $MODEL $TYPE $DATASET $((SLOT - 1))
    CUDA_VISIBLE_DEVICES=$((SLOT - 1)) python test.py \
        --prefix=${DATASET}_testonly  \
        --model_folder=$MODEL_FOLDER --dataset=$DATASET --model_file=best_model.pt --problem_type=vertex_cover
}
export -f run_job

pretrained_models=(
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:1861e3873b61c0138f0fade13ba831a6cb31a0e2d5717c68a20a0009aa3859fb/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:864d82fb8b5b3647c51f85219aee5730cb01098f64dc5fafe29934637b25f1f0/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:39dcb2727b0fbd50b59b960e311816caf1dbce32cc94096c070c192f530359a4/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:5d1b0989edbfea2e797072b8c2b12001f723339dfa96406758e6208905030c61/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:7eae26a6a2d778806fa220b51c7c589f35ef252023708a0e1b70f1d809fefa16/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:c259ee991d44601a24495085d478929b226e94ca5ffc2a75ab739957d94da4b0/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:9ede9cd33a4971c3246500914847936dbc9690bd4c05a450cdef0d6146b815f4/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:631e9afcf3bd0330c4f9625392cc0b183926a43b2ba3d8c5d8fbaf91c779dc90/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:44c7a07f4cfa3adea844271356ad4eb9f408bafc526faf912be9985288fd058e/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:0e1f84c7f350a2df384d2be3ccb214f0a94111255927a19fbe60c460c9117ebb/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:176039cd545a47b6cb218b27e31f75dd958f21f293303d95e97a3964c1ae558c/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:81ae831246b731c00408289a4fa29a8586fff742dbb3d6f2fea260065f5e034f/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:53efe2843e3a8f9199398958fe32799b669b22c19f054ab089fb55ebb1b7534e/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:2a0f6b1a8c8feada2d90a7777d71b5652e2a0407b78bf34545fc38f33bdfdaaf/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:83fbb622af5fbb000c97c17df538bd3bee6082a5f9786d9a4f91ef7ce1045b28/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:034bbc1b5d117369cee302d1e2f6b87ba5ec5d5b4bf3a95e7fc99586f96fb415/"
"/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/LiftMP_runs/paramhash:815075705c4d1fec7ff5e2364cd541bc9cf945ca345e6568e1d5da54e795c7c8/"
)

for model_folder in "${pretrained_models[@]}" ; do
    for dataset in 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'REDDIT-BINARY' ; do
        echo $model_folder $dataset
    done
done | parallel --ungroup -j1 run_job {} {%}