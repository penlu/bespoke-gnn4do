#!/bin/bash
#SBATCH -o 230925_gurobi_VC.%j.out
#SBATCH --job-name="230925_gurobi_VC"
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

echo "Job ID $SLURM_JOB_ID"
echo "flags $@"

python -u baselines.py "$@"
