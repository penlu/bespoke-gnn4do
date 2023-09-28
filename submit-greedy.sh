#!/bin/bash
#SBATCH -o 230925_greedy.%j.out
#SBATCH --job-name="230925_greedy"
#SBATCH -c 1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
module load gurobi/gurobi-1000
source activate test3

echo "Job ID $SLURM_JOB_ID"
echo "flags $@"

python -u baselines.py "$@"
