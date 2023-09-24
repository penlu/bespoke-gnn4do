#!/bin/bash
#SBATCH -o 230922_hparam.%j.out
#SBATCH --job-name="230922_hparam"
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

MODEL=$1
DATASET=$2
PENALTY=$3
R=$4
LIFT_LAYERS=$5

#if [ $DATASET = 'RANDOM' ] ; then
#    TYPE='RANDOM'
#elif [ $DATASET = 'ForcedRB' ] ; then
#    TYPE='ForcedRB'
#else
#    TYPE='TU'
#fi
echo "Job ID $SLURM_JOB_ID"
echo "model=$MODEL dataset=$DATASET type=$TYPE penalty=$PENALTY r=$R lift_layers=$LIFT_LAYERS"

python -u train.py \
    --stepwise=True --steps=100000 \
    --valid_freq=1000 --dropout=0 \
    --prefix=230922_inf_vertex_cover_hparam \
    --model_type=$MODEL --dataset=$DATASET --parallel=20 \
    --num_layers=$LIFT_LAYERS --rank=$R \
    --vc_penalty=$PENALTY --problem_type=vertex_cover
