#!/bin/bash
#SBATCH -o 230901_VC.%j.out
#SBATCH --job-name="230901_VC"
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:volta:1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

MODEL=$1
DATASET=$2

if [ $DATASET = 'RANDOM' ] ; then
    TYPE='RANDOM'
elif [ $DATASET = 'ForcedRB' ] ; then
    TYPE='ForcedRB'
else
    TYPE='TU'
fi
echo "Job ID $SLURM_JOB_ID"
echo "model=$MODEL dataset=$DATASET type=$TYPE"
python -u train.py \
  --problem_type=vertex_cover --vc_penalty=2 \
  --stepwise=True --steps=50000 \
  --valid_freq=100 --dropout=0 \
  --prefix=230901_VC \
  --model_type=$MODEL --TUdataset_name=$DATASET --dataset=$TYPE
