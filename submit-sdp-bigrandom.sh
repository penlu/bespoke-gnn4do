#!/bin/bash
#SBATCH -o 230914_VC_sdp_COLLAB.%j.out
#SBATCH --job-name="230914_VC_sdp_COLLAB"
#SBATCH -c 8
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

DATASET=$1
START_IDX=$2
END_IDX=$3

if [ $DATASET = 'RANDOM' ] ; then
    TYPE='RANDOM'
elif [ $DATASET = 'ForcedRB' ] ; then
    TYPE='ForcedRB'
else
    TYPE='TU'
fi
echo "Job ID $SLURM_JOB_ID"
echo "dataset=$DATASET type=$TYPE start_idx=$START_IDX end_idx=$END_IDX"

python -u baselines.py --dataset $TYPE \
  --problem_type vertex_cover \
  --prefix 230914_VC_sdp_$DATASET \
  --TUdataset_name $DATASET \
  --start_index $START_IDX --end_index $END_IDX \
  --sdp=True
