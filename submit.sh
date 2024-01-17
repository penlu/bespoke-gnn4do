#!/bin/bash
#SBATCH -o 240117_sat.%j.out
#SBATCH --job-name="240117_sat"
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

ulimit -n 65536

echo "Job ID $SLURM_JOB_ID"
echo "flags $@"

python -u train.py "$@"
