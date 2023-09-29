#!/bin/bash
#SBATCH -o 230928_retest.%j.out
#SBATCH --job-name="230928_retest"
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

echo "Job ID $SLURM_JOB_ID"
DIRNAME=$1
echo $DIRNAME

python -u test.py --model_folder=$DIRNAME --model_file=best_model.pt --test_prefix=retest_best
if [ -f $DIRNAME/model_step20000.pt ] ; then
    python -u test.py --model_folder=$DIRNAME --model_file=model_step20000.pt --test_prefix=retest_last
else
    python -u test.py --model_folder=$DIRNAME --model_file=model_step100000.pt --test_prefix=retest_last
fi
