#!/bin/bash
#SBATCH -o 230920_clique_train.%j.out
#SBATCH --job-name="230920_clique_train"
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
  --problem_type=max_clique --vc_penalty=1 \
  --stepwise=True --steps=50000 \
  --valid_freq=1000 --dropout=0 \
  --prefix=230920_clique \
  --positional_encoding=laplacian_eigenvector --pe_dimension=8 \
  --model_type=$MODEL --TUdataset_name=$DATASET --dataset=$TYPE
