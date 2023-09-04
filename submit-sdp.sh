#!/bin/bash
#SBATCH -o 230901_VC_sdp.%j.out
#SBATCH --job-name="230901_VC_sdp"
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
  --problem_type vertex_cover \
  --prefix 230902_VC_sdp_$DATASET --TUdataset_name $DATASET
