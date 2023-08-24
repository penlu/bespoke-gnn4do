set -e

run_job () {
    JOBARRAY=($1)
    MODEL_FOLDER=${JOBARRAY[0]}
    DATASET=${JOBARRAY[1]}
    SLOT=$2
    if [ $DATASET = 'RANDOM' ] ; then
        TYPE='RANDOM'
    else
        TYPE='TU'
    fi
    echo $MODEL $TYPE $DATASET $((SLOT - 1))
    CUDA_VISIBLE_DEVICES=$((SLOT - 1)) python train.py \
        --stepwise=True --steps=5000 \
        --valid_freq=100 --prefix=230824_finetuning_short \
        --finetune_from=$MODEL_FOLDER --TUdataset_name=$DATASET --dataset=$TYPE
}
export -f run_job

pretrained_models=('/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/230822_test3_paramhash:a1149c15d7c63b45244a4655b61dadc83d24e350617b4ee0a5f3a1c79af17122/model_step100000.pt'
'/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/230822_test3_paramhash:1ccf2bfdb6b03293853e27935e7ac9772a060317de5746b8cde8d88648df4f93/model_step100000.pt'
'/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/230822_test3_paramhash:c4f554809137abf2b1aa7816d5aa17ae767dfaa7bea2ea21092e56bd17f0a4f0/model_step100000.pt'
'/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/230822_test3_paramhash:dea4b1fe5c3a1a23f8f062c25d2b677d6b8bf979de4787480f7af2125241ded9/model_step100000.pt'
)

for model_folder in "${pretrained_models[@]}" ; do
    for dataset in 'RANDOM' 'ENZYMES' 'PROTEINS' 'IMDB-BINARY' 'MUTAG' 'COLLAB' ; do
        echo $model_folder $dataset
    done
done | parallel --ungroup -j1 run_job {} {%}