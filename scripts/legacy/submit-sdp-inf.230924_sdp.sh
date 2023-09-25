#!/bin/bash
#SBATCH -o 230924_sdp.%j.out
#SBATCH --job-name="230924_sdp"
#SBATCH -c 48
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

DATASET=$1

echo "Job ID $SLURM_JOB_ID"
echo "dataset=$DATASET"

python baselines.py --dataset $DATASET --infinite=True --parallel=20 --RB_n 30 60 --RB_k 15 27 \
  --problem_type vertex_cover \
  --prefix 230924_sdp_$DATASET \
  --sdp=True
