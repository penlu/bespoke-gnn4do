#!/bin/bash
#SBATCH -o 230924_train.%j.out
#SBATCH --job-name="230924_train"
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

echo "Job ID $SLURM_JOB_ID"
echo "model=$MODEL dataset=$DATASET"
python -u train.py \
  --problem_type=vertex_cover --vc_penalty=1 \
  --stepwise=True --steps=100000 \
  --valid_freq=1000 --dropout=0 \
  --prefix=230924_train \
  --positional_encoding=laplacian_eigenvector --pe_dimension=8 \
  --model_type=$MODEL --dataset=$DATASET
