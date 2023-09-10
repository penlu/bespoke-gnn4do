#!/bin/bash
#SBATCH -o 230910_clique_sdp.%j.out
#SBATCH --job-name="230910_clique_sdp"
#SBATCH -c 48
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

DATASET=$1

if [ $DATASET = 'RANDOM' ] ; then
    TYPE='RANDOM'
else
    TYPE='TU'
fi
echo "Job ID $SLURM_JOB_ID"
echo "dataset=$DATASET type=$TYPE"

python -u baselines.py --dataset $TYPE \
  --problem_type max_clique \
  --prefix 230910_clique_sdp_$DATASET --TUdataset_name $DATASET
