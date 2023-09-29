#!/bin/bash
#SBATCH -o 230928_revalidate.%j.out
#SBATCH --job-name="230928_revalidate"
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

python -u test.py --model_folder=$DIRNAME --model_file=best_model.pt --test_prefix=revalidate_best --use_val_set=True
if [ -f $DIRNAME/model_step20000.pt ] ; then
    python -u test.py --model_folder=$DIRNAME --model_file=model_step20000.pt --test_prefix=revalidate_last --use_val_set=True
else
    python -u test.py --model_folder=$DIRNAME --model_file=model_step100000.pt --test_prefix=revalidate_last --use_val_set=True
fi
